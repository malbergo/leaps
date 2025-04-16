import torch
import torch.nn as nn
from torch.func import vmap, jacfwd
from dataclasses import dataclass
from typing import Union

import numpy as np

from src.modules.rhot import Ising, Potts
from src.modules.resample import systematic_resample

def ess_func(At):
    return torch.mean(torch.exp(At))**2 / torch.mean(torch.exp(2*At))


@dataclass
class DiscreteJarzynskiIntegrator:
    dpath: Union[Ising,Potts]
    eps: torch.tensor
    ts: torch.tensor
    Qt_net: nn.Module = None,
    transport: bool = False
    n_save: int  = 10
    resample: bool = False
    resample_thres: float = 0.7
    turn_off_jarz: bool = False
    model_class: str = "ising"
    n_mcmc_per_net: int = 1
    compute_is_weights: int = True
    q: int = 2

    def __post_init__(self) -> None:
        """Initialize time grid and step size."""
        self.n_step = len(self.ts) - 1
        assert self.n_save > 0 ## will always save the first sample
        assert self.n_step % self.n_save == 0, f"{self.n_step} is not divisible by {self.n_save}"
        self.save_freq = self.n_step // self.n_save
        
        if self.model_class == 'ising':
            assert self.q == 2
        
        
    def ess(self, At):
        return ess_func(At)
        
    def _single_step_glauber(self, cfg, t, dt):
        """
        Perform a Galuber update on a single spin configuration (cfg).

        cfg : (L, L)
        J   : coupling constant
        beta: inverse temperature (1 / k_B T))

        Returns
        new_cfg:  (L, L)
        """
        
        J, mu, B, beta = self.dpath.setup_params(t)
    
        L = cfg.shape[0]
        
        row = np.random.randint(0, L, ())
        col = np.random.randint(0, L, ())
        spins = cfg[row, col]
    
        # neighbors
        up_row = (row - 1) % L
        down_row = (row + 1) % L
        left_col = (col - 1) % L
        right_col = (col + 1) % L

        # local neighbor interaction
        neighbor_sum = cfg[up_row, col] + cfg[down_row, col] + cfg[row, left_col] + cfg[row, right_col]
        local_energy = J * neighbor_sum + mu * B
        delta_E = 2.0 * spins * local_energy
    
        acc = torch.exp(-beta * delta_E)
        acc = torch.minimum(acc, torch.tensor(1)) 


        flip =  (torch.rand(1, device=cfg.device) < acc)
        # minus or plus 1 depending on if flip is true or flase respectively
        pm1 = torch.where(flip, torch.tensor(-1.0, device=cfg.device), torch.tensor(1.0, device=cfg.device))
        cfg[row, col] = pm1 * cfg[row, col] 

        return cfg
    
    
    def _single_step_glauber_potts(self, cfg, t, dt):
        """
        Perform a single Glauber update for the potts model.
        cfg : (L, L)
        J   : coupling constant
        beta: inverse temperature
        """
        L = cfg.shape[0]
        
        J, B, beta = self.dpath.setup_params(t)

        # Randomly select site
        row = np.random.randint(0, L, ())
        col = np.random.randint(0, L, ())
        current_state = cfg[row, col]

        # ompute neighbors 
        neighbors = torch.stack([
            cfg[(row - 1) % L, col],
            cfg[(row + 1) % L, col],
            cfg[row, (col - 1) % L],
            cfg[row, (col + 1) % L]
        ])

        matches_current = (neighbors == current_state).sum()

        # propose a flip uniformly from {0,...,q-1}
        proposed_state = torch.randint(0, self.q , (), device=cfg.device)

        # adjust proposal state to always differ from current without branching
        proposed_state = (current_state + 1 + proposed_state) % self.q

        matches_proposed = (neighbors == proposed_state).sum()

        delta_E = -J * (matches_proposed - matches_current)
        acc_prob = torch.exp(-beta * delta_E).clamp(max=1.0)

        flip = (torch.rand(()).to(cfg.device) < acc_prob).float()

        # update cfg 
        updated_cfg = cfg.clone()
        updated_cfg[row, col] = flip * proposed_state + (1 - flip) * current_state

        return updated_cfg

    def metropolis_step(self, cfgs, t, dt):
        """
        Perform a Galuber update on a batch of  spin configurations (cfgs).
        """
        if self.model_class == "ising":
            bs = cfgs.shape[0]
            return vmap(self._single_step_glauber, in_dims = (0, 0, 0),
                        out_dims=(0), randomness='different')(cfgs, t, dt)
        elif self.model_class == "potts":
            return vmap(self._single_step_glauber_potts, in_dims = (0, 0, 0),
                        out_dims=(0), randomness='different')(cfgs, t, dt)
        else:
            raise NotImplementedError()
        
    def sigma_step(self, sigma, t, dt):
        L = sigma.shape[-1]
        if self.transport:
            with torch.no_grad():
                sigma_new = self.Qt_net.sample_next_step(sigma, t, dt)
        else:
            sigma_new = sigma
            
        for _ in range(self.n_mcmc_per_net):
            sigma_new = self.metropolis_step(sigma_new,t,dt)

        return sigma_new, None
        
    def A_step(self, sigma, t, A, dt):
        L = sigma.shape[-1]
        if self.transport and not self.turn_off_jarz:   
            _, delta_H_neighbors = self.dpath.Ut(sigma, t)
            lh_ratios = torch.exp(-delta_H_neighbors)
            with torch.no_grad():
                jarz_corrector = self.Qt_net.get_jarzinsky_corrector(sigma, t, lh_ratios)
            return A - dt * self.dpath.dtUt(sigma,t)[0] - dt * jarz_corrector
        else:
            return A - dt * self.dpath.dtUt(sigma,t)[0]
    
    def rollout(self, sigma_init):
        """
        Performs the rollout
        """
        bs = sigma_init.shape[0]
        sigma_stack = torch.zeros((self.n_save +1, *sigma_init.shape)).to(sigma_init)
        As = torch.zeros((self.n_save +1, bs)).to(sigma_init.device)

        # initialize weights A to zero
        A = torch.zeros(bs).to(sigma_init.device)
        sigma = sigma_init
        
        sigma_stack[0] = sigma_init.detach()
        As[0] = A
        
        save_ind = 1
        resample_count = 0
        for i, t in enumerate(self.ts[:-1]):
            dt = (self.ts[i+1] - self.ts[i]).repeat(len(sigma)).to(sigma_init.device) ### assumes first element in self.ts is 0
            t = t.repeat(len(sigma)).to(sigma_init.device) #.requires_grad_(True)
            sigma = sigma.to(sigma_init)
            sigma_new, _ = self.sigma_step(sigma, t, dt)
            
            A = A.to(sigma_init.device)
            A_new = self.A_step(sigma, t, A, dt)
                
            t_new = self.ts[i+1].repeat(len(sigma)) #.requires_grad_(True)

            if self.ess(A_new.double()) < self.resample_thres and self.resample:
                # raise NotImplementedError()
                resample_count +=1
                print("i:", i, "resample cout:", resample_count)
                weights = torch.softmax(A_new, dim = 0)
                indices = systematic_resample(weights)
                sigma_new   = torch.clone(sigma_new[indices])
                A_new   = torch.zeros(len(weights)).to(sigma_init.device)
                
            if (i + 1) % self.save_freq == 0:            
                sigma_stack[save_ind] = torch.clone(sigma_new.detach())
                As[save_ind] = torch.clone(A_new.detach())
                save_ind += 1

            sigma = sigma.detach()
            del sigma
            sigma = sigma_new.detach()
            A = A_new.detach()
            del sigma_new, A_new

        sigma_stack[-1] = sigma.detach()
        As[-1] = A.detach()

        return sigma_stack, As