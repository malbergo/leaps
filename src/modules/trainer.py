import torch
from torch.func import vmap, jacrev, jacfwd
from torchvision.utils import make_grid
import numpy as np
import math

from src.helpers import get_arch
from src.modules import loss
from src.modules import prior
from src.modules import rhot
from src.modules.sampler import DiscreteJarzynskiIntegrator, ess_func
from src.helpers import get_arch, get_arch_F, repeat_integers, visualize_lattices, inverse_repeat_quadratic
import random
import sys
import lightning as L


from torch.utils.data import DataLoader, Dataset

import wandb


class DummyDataset(Dataset):
    """A dummy dataset that returns None because data is generated in training_step."""
    def __len__(self):
        return 1000000  # arbitrary large number to allow enough training steps

    def __getitem__(self, idx):
        return torch.tensor(0.0)  # not used

def repeat_integers_with_increase(start, max_val, n_anneal, n_anneal_increase=0):
    """
    Generates a list where each integer from start to max_val (inclusive) is repeated
    a number of times that starts at n_anneal and increases by n_anneal_increase for each subsequent integer.
    
    Args:
        start (int): Starting integer.
        max_val (int): Last integer (inclusive).
        n_anneal (int): Number of repetitions for the first integer.
        n_anneal_increase (int): Amount to increase the repetition count for each subsequent integer.
    
    Returns:
        list: List of integers with the desired repetition pattern.
    """
    result = []
    for i, k in enumerate(range(start, max_val + 1)):
        repeats = n_anneal + (i-1) * n_anneal_increase
        result.extend([k] * repeats)
    return result


class IsingLightningModule(L.LightningModule):
    def __init__(self,config):
        super().__init__()
        # self.save_hyperparameters(ignore=['interpolant', 'prior']) ### saves the relevant hyperparameters
        self.save_hyperparameters( ) ### saves the relevant hyperparameters
        
        self.config  = config
        self.loss_fn = loss.make_loss(config)
        if config.model_class == "ising":
            self.Energy  = rhot.make_ising(config)
        elif config.model_class == "potts":
            self.Energy  = rhot.make_potts(config)
        else:
            raise NotImplementedError()
        self.net     = get_arch(config)
        self.F_t_net = get_arch_F(config)
        self.buffer = []
        if hasattr(config, "use_buffer"):
            self.use_buffer = config.use_buffer
            self.buffer_cycle = config.buffer_cycle
            self.max_buffer_size = config.max_buffer_size
        else:
            self.use_buffer = False
            self.buffer_cycle = 1
            self.max_buffer_size = 1
        
        ## setup some training parameters:
        self.k_max   = int(1/config.delta_t)
        if hasattr(config, "anneal_quadratic"):
            if config.anneal_quadratic:
                self.ks  = inverse_repeat_quadratic(config.starting_k, self.k_max, config.n_anneal)
            else:
                self.ks  = repeat_integers(config.starting_k, self.k_max, config.n_anneal)
        else:
            self.ks  = repeat_integers(config.starting_k, self.k_max, config.n_anneal)
        
        if hasattr(config, "warm_up") and config.warm_up > 0:
            self.ks = [config.starting_k] * config.warm_up + self.ks
            print("Added warm_up of: ", config.warm_up)
    
        self.eps     = torch.tensor(self.k_max)


    def training_step(self, batch, batch_idx):

        
        if self.global_step >= (len(self.ks) - 1):
            k = self.ks[-1]
        else:
            k = self.ks[self.global_step]
            
        final_t = k*self.config.delta_t
        t_set = final_t*torch.rand(k - 1) # -1 because we append 1.0 at the end
        t_set = torch.cat((torch.tensor([0.0]), t_set, torch.tensor([final_t]))).requires_grad_(True)
        ts = torch.sort(t_set).values.to(self.device)
        
        log_ess_flag = (self.global_step + 1) % self.config.log_every_local  == 0
        
        if (not self.use_buffer) or (self.global_step % self.buffer_cycle == self.buffer_cycle-1) or len(self.buffer) < self.max_buffer_size:
            if self.config.model_class == "ising":
                jit = DiscreteJarzynskiIntegrator(self.Energy, 
                                              self.eps, 
                                              ts, 
                                              Qt_net=self.net, 
                                              transport = True, n_save = k,
                                              resample = self.config.resample,
                                              resample_thres = self.config.resample_thres,
                                              model_class = self.config.model_class, 
                                              n_mcmc_per_net = self.config.n_mcmc_per_net)
                sigma_vec = 2 * torch.randint(0,2,size=(self.config.bs_per_gpu, self.config.L, self.config.L)).float().to(self.device) - 1
            elif self.config.model_class == "potts":
                sigma_vec = torch.randint(0,self.config.n_cat,size=(self.config.bs_per_gpu, self.config.L, self.config.L)).to(self.device)
                jit = DiscreteJarzynskiIntegrator(self.Energy, 
                                                    self.eps, 
                                                    ts, 
                                                    Qt_net=self.net, 
                                                    transport = True, n_save = k,
                                                    resample = self.config.resample,
                                                    resample_thres = self.config.resample_thres,
                                                    model_class = self.config.model_class, 
                                                    n_mcmc_per_net = self.config.n_mcmc_per_net,
                                                    q=self.config.n_cat)
            sigmas, As = jit.rollout(sigma_vec) ### [bs, n_save = k, L, L]
            sigmas = sigmas.detach()
            As = As.detach()
            if self.use_buffer:
                if len(self.buffer)>=self.max_buffer_size:
                    self.buffer.pop(0)
                self.buffer.append((sigmas.cpu(),As.cpu(),k, ts))
        else:
            sigmas, As, k, ts = random.choice(self.buffer)
            sigmas = sigmas.to(self.device)
            As = As.to(self.device)
        
        sub_bs    = self.config.bs_for_loss_per_gpu
        walk_idxs = torch.randint(low=0, high=self.config.bs_per_gpu, size=(sub_bs,))#.to(device)
        time_idxs = torch.randint(low=0, high=k + 1,  size=(sub_bs,))#.to(device)
        sigma_vec = sigmas[time_idxs, walk_idxs] #.requires_grad_(True)
        time_vec = ts[time_idxs] #.requires_grad_(True)
        
        # compute Jarzinsky corrector
        in_rates, stay_rate  = self.net.get_in_rates(sigma_vec, time_vec)
        _, delta_H_neighbors = self.Energy.Ut(sigma_vec, time_vec)
        lh_ratios            = torch.exp(-delta_H_neighbors)
        jarz_corrector       = self.net.get_jarzinsky_corrector(sigma_vec, time_vec, lh_ratios)

        # compute derivative over potential
        dtUt_vec, _ = self.Energy.dtUt(sigma_vec,time_vec)

        # Compute derivative of free energy estimator
        dt_F_t_vec = self.F_t_net.gradt(time_vec)

        # compute PINN objective
        loss_vec = ((-dtUt_vec - jarz_corrector + dt_F_t_vec)**2)
        loss_val = loss_vec.mean()
        
        
        # logging things
        if log_ess_flag:
            self.log("train_loss", loss_val, prog_bar=True, logger=True, sync_dist=True)
            
            _ess = ess_func(As[-1].double())
            self.log("ess", _ess, logger=True, sync_dist=True)
            self.log("k_ess", k, logger=True, sync_dist=True)
            self.log("t_final", final_t, logger=True, sync_dist=True)
        
            grad   = torch.tensor([torch.nn.utils.clip_grad_norm_(self.net.parameters(), float('inf'))]).detach()
            self.log("grad norm",      grad)

        if (self.global_step + 1) % self.config.plot_every == 0:
            plot_bs = 16 # hardcode
            full_bs = sigmas.shape[1]
            # lattices = make_grid(list([*sigmas[-1, :bs].detach().cpu()]))

            indices = torch.randperm(full_bs)[:plot_bs]  # randomly select indices
            some_sigmas = sigmas[-1][indices]  # Shape: [n_plots, 1, L, L] for grayscale
            
            fig = visualize_lattices(some_sigmas.cpu().detach())

#             grid = make_grid(some_sigmas, nrow=8, padding=2, normalize=True)  # Shape: (1, H, W)
#             # Convert grid to NumPy format (H, W, C)
#             grid_np = grid.permute(1, 2, 0).cpu().numpy()
            # Log to WandB
            # self.logger.experiment.log({"Lattice Visualization": wandb.Image(grid_np)})
            self.logger.experiment.log({"Lattice Visualization": wandb.Image(fig)})

        
        return loss_val
    
    def configure_optimizers(self):
        
        base_lr = self.config.base_lr
        # opt   = torch.optim.Adam(self.net.parameters(), lr=base_lr)
        opt = torch.optim.Adam(list(self.F_t_net.parameters()) + list(self.net.parameters()), lr = base_lr)
        
        sched = torch.optim.lr_scheduler.StepLR(opt, 
                                                step_size = self.config.lr_sched_step_size, 
                                                gamma = self.config.lr_sched_gamma)

        
        return [opt], [{"scheduler": sched, "interval": "step"}]
    
    
    def train_dataloader(self):
        """Return a dummy dataloader to satisfy PyTorch Lightning"""
        return DataLoader(DummyDataset(), batch_size=1)
    



import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def visualize_lattices(phi, n_rows = 2, n_cols = 5, save = None):
    """
    Visualize a subset of the lattices in a grid.

    Parameters:
    - phi: A tensor of shape [bs, L, L] representing the batch of lattices.
    - n_rows: Number of rows in the subplot grid.
    - n_cols: Number of columns in the subplot grid.
    - title: The title of the plot.
    """
    bs, L, _ = phi.shape
    n_plots = min(n_rows * n_cols, bs)
    print("N ROW:", n_rows)
    print("N COL:", n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4, 2.5))

    indices = np.random.choice(bs, n_plots, replace=False)  # Randomly select lattice indices
    
    # indices = np.arange(15,25)
    
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        # cmap = sns.cubehelix_palette(start=-0.8, rot=0.4, gamma=1.0, dark=0.2, light=0.9, reverse=False, as_cmap=True)
        palette = sns.color_palette("tab10", n_colors=7)
        cmap = ListedColormap(palette)
        im = ax.imshow(phi[idx], cmap=cmap, origin='lower')
        ax.axis('off') 

    fig.subplots_adjust( wspace=0.1, hspace=-0.5)
    
    return fig