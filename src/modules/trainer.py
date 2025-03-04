import torch
from torch.func import vmap, jacrev, jacfwd
from torchvision.utils import make_grid
import numpy as np
import math

from src.helpers import get_arch
from src.modules import loss
from src.modules import prior
from src.modules import rhot
from src.modules.sampler import DiscreteJarzynskiIntegrator
from src.helpers import get_arch, get_arch_F, repeat_integers, visualize_lattices

import lightning as L


from torch.utils.data import DataLoader, Dataset

import wandb


class DummyDataset(Dataset):
    """A dummy dataset that returns None because data is generated in training_step."""
    def __len__(self):
        return 1000000  # arbitrary large number to allow enough training steps

    def __getitem__(self, idx):
        return torch.tensor(0.0)  # not used




class IsingLightningModule(L.LightningModule):
    def __init__(self,config):
        super().__init__()
        # self.save_hyperparameters(ignore=['interpolant', 'prior']) ### saves the relevant hyperparameters
        self.save_hyperparameters( ) ### saves the relevant hyperparameters
        
        self.config  = config
        self.loss_fn = loss.make_loss(config)
        self.Energy  = rhot.make_ising(config)
        self.net     = get_arch(config)
        self.F_t_net = get_arch_F(config)
        
        ## setup some training parameters:
        self.k_max   = int(1/config.delta_t)
        self.ks      = repeat_integers(config.starting_k, self.k_max, config.n_opt)
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
        jit = DiscreteJarzynskiIntegrator(self.Energy, self.eps, 
                                          ts, Qt_net=self.net, 
                                          transport = True, n_save = k,
                                          resample = self.config.resample,
                                          resample_thres = self.config.resample_thres)
        

        sigma_vec = 2 * torch.randint(0,2,size=(self.config.bs_per_gpu, self.config.L, self.config.L)).float().to(self.device) - 1
        sigmas, As = jit.rollout(sigma_vec) ### [bs, n_save = k, L, L]
        sigmas = sigmas.detach()
        As = As.detach()
        
        sub_bs    = 100
        walk_idxs = torch.randint(low=0, high=self.config.bs_per_gpu, size=(sub_bs,))#.to(device)
        time_idxs = torch.randint(low=0, high=k + 1,  size=(sub_bs,))#.to(device)
        sigma_vec = sigmas[time_idxs, walk_idxs].requires_grad_(True)
        time_vec = ts[time_idxs].requires_grad_(True)
        
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
        if (self.global_step + 1) % self.config.log_every_local  == 0:
            self.log("train_loss", loss_val, prog_bar=True, logger=True, sync_dist=True)
            
            _ess = jit.ess(As[-1].double())
            self.log("ess", _ess, logger=True, sync_dist=True)
            self.log("t_final", final_t, logger=True, sync_dist=True)
        

        if (self.global_step + 1) % self.config.plot_every == 0:
            plot_bs = 16 # hardcode
            full_bs = sigmas.shape[1]
            # lattices = make_grid(list([*sigmas[-1, :bs].detach().cpu()]))

            indices = torch.randperm(full_bs)[:plot_bs]  # randomly select indices
            some_sigmas = sigmas[-1][indices].unsqueeze(1)  # Shape: [n_plots, 1, L, L] for grayscale

            grid = make_grid(some_sigmas, nrow=8, padding=2, normalize=True)  # Shape: (1, H, W)

            # Convert grid to NumPy format (H, W, C)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()

            # Log to WandB
            self.logger.experiment.log({"Lattice Visualization": wandb.Image(grid_np)})

        
        return loss_val
    
    def configure_optimizers(self):
        
        base_lr = self.config.base_lr
        opt   = torch.optim.Adam(self.net.parameters(), lr=base_lr)
        
        sched = torch.optim.lr_scheduler.StepLR(opt, 
                                                step_size = self.config.lr_sched_step_size, 
                                                gamma = self.config.lr_sched_gamma)

        
        return [opt], [{"scheduler": sched, "interval": "step"}]
    
    
    def train_dataloader(self):
        """Return a dummy dataloader to satisfy PyTorch Lightning"""
        return DataLoader(DummyDataset(), batch_size=1)
    

