import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

import math
import numpy as np
from torch.func import vmap, jacfwd
from rotary_embedding_torch import RotaryEmbedding

class NewMaskedConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        idx = kernel_size // 2
        self.mask[..., idx, idx] = 0

    def forward(self, x):
        # Multiply the learnable weight by the (fixed) mask before the convolution
        return torch.nn.functional.conv2d(x, self.weight * self.mask, self.bias,
                         self.stride, self.padding, self.dilation, self.groups)


def unfold_with_circular_padding(x, kernel_size):
    """
    x:          [B, C, H, W]
    kernel_size: integer (K)
    pad_size:    integer or tuple (p)
    """
    pad_size = kernel_size//2
    # 1) Manually circular-pad the input tensor
    #    If you need p on each side (left/right, top/bottom), then for a 2D input:
    #    pad = (pad_left, pad_right, pad_top, pad_bottom).
    #    Suppose pad_size is the same on all sides just for an example:
    x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='circular')
    # shape = [B, C, H + 2*pad_size, W + 2*pad_size]
    
    # 2) Unfold on the padded input, but with `padding=0` inside `unfold`
    unfolded = F.unfold(x_padded, kernel_size=kernel_size, padding=0)
    return unfolded.reshape(x.shape[0],x.shape[1],kernel_size * kernel_size, x.shape[2], x.shape[3])


class SimpleMLP(nn.Module):
    def __init__(self, kernel_size, hidden_dim=32):
        super(SimpleMLP, self).__init__()
        output_size = kernel_size * kernel_size
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class StateDependentConv2D(torch.nn.Module):
    def __init__(self, 
                 kernel_size: int = 3, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 prev_out_channels: int = 3):
        
        super(StateDependentConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = torch.nn.Parameter(0.002*torch.rand(in_channels, kernel_size*kernel_size, prev_out_channels)/torch.sqrt(torch.tensor(kernel_size*kernel_size)))
        self.b1 = torch.nn.Parameter(0.002*torch.rand(in_channels, kernel_size*kernel_size))
        # self.C = torch.nn.Parameter(torch.rand(in_channels, kernel_size*kernel_size,kernel_size*kernel_size)/torch.sqrt(torch.tensor(kernel_size*kernel_size)))
        self.b2 = torch.nn.Parameter(0.002*torch.rand(in_channels, kernel_size*kernel_size))
        self.kernel_size = kernel_size
        self.mid_idx = (kernel_size * kernel_size)//2
        self.prev_out_channels = prev_out_channels
        
        
        #### michael adds time mlp
        
        self.mlp = SimpleMLP(kernel_size, hidden_dim=64)

    def forward(self, x, t, prev_output):
        """
        ASD
        """
        # Unfold x (get stack of neighborhood elements):
        x_unfold = unfold_with_circular_padding(x,self.kernel_size)

        # Compute state-dependent weights:
        A_prev_out = torch.einsum("bcjk,lic->blijk", prev_output, self.A)
        A_b_prev_out = torch.nn.functional.silu(A_prev_out + self.b1[None,:,:,None,None])
        #A_b_prev_out = A_prev_out + self.b1[None,:,:,None,None]
        # k_x = torch.einsum("bcjkl,cij->bcikl", A_b_prev_out, self.C) + self.b2[None,:,:,None,None]
        
        t_emb = self.mlp(t.view(-1,1)) #### new time_emb
        k_x = A_b_prev_out + self.b2[None,:,:,None,None] + t_emb[:, None, :, None, None] ### new time_emb

        # Make k_x ignore x itself:
        k_x[:,:,self.mid_idx] = 0.0

        # Multiply weights onto x:
        return (k_x * x_unfold).sum(dim=2)

class LocEquivConvNet(torch.nn.Module):
    def __init__(self, kernel_sizes: list,  num_channels = 4, in_channels = 1, invariant: bool = False):
        super(LocEquivConvNet, self).__init__()
        self.num_channels = num_channels
        self.invariant = invariant

        sizes = [in_channels]
        self.convs = []
        self.initial_conv = NewMaskedConv2d(
                            in_channels, num_channels, kernel_sizes[0], padding=kernel_sizes[0]//2,
                            stride=1, padding_mode='circular', bias = False)
        self.mid_act = torch.nn.SiLU()
        self.convs = []
        for idx in range(len(kernel_sizes)-1):
            self.convs.append(StateDependentConv2D(
                                kernel_size=kernel_sizes[idx+1], 
                                in_channels=num_channels, 
                                out_channels=num_channels, 
                                prev_out_channels=num_channels))
        self.convs = torch.nn.ModuleList(self.convs)
        
        self.linear = torch.nn.Parameter(torch.randn(num_channels) / torch.sqrt(torch.tensor(num_channels)))

    def forward(self, x, t):
        t = t[:,None,None,None]
        x = x.unsqueeze(1)
        x_t = (1+t)*x
        mid_x_pre_act = self.initial_conv(x_t)
        for idx in range(len(self.convs)):
            mid_x_post_act = self.mid_act(mid_x_pre_act)
            mid_x_pre_act = self.convs[idx](x, t, mid_x_post_act) ### michael add T here
        final_out = mid_x_pre_act
        if not self.invariant:
            out = (final_out * x_t)
        else:
            out = final_out
        return torch.sum(self.linear[None, :, None, None] * out, dim = 1) 
    
    
class ConvSpinRateMatrix(nn.Module):
    def __init__(self, 
                 kernel_sizes: list,  
                 num_channels = 4, 
                 in_channels = 1,
                 invariant   =False):
        """
        Args:
            head_dim_in: dimension of attention for input
            head_dim_in: dimension of attention for output
            num_heads: Number of projection heads
            k: Dimension of each input vector x_j (in R^k).
            emb_qk: flag whether to apply rotary positional embedding on queries and keys
        """
        super().__init__()
        self.network = LocEquivConvNet(
                                    kernel_sizes=kernel_sizes,
                                    num_channels=num_channels,
                                    in_channels=in_channels)
        

    
    def forward(self, sigma_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        # print("IVE BEEN MADEEEE!")
        return self.network(sigma_vec, time_vec)

    def get_out_rates(self, sigma_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        output = self.forward(sigma_vec, time_vec)
        output = torch.relu(output)
        stay_rate = -output.sum(dim=[1,2])
        return output, stay_rate

    def get_in_rates(self, sigma_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        output = self.forward(sigma_vec, time_vec)
        stay_rate = -torch.relu(output).sum(dim=[1,2])
        return torch.relu(-output), stay_rate

    def sample_next_step(self, x, time_vec, dt_vec):
        
        # [batch_size, seq_len]
        out_rates, _ = self.get_out_rates(x, time_vec)
        
        # Get Bernoulli probabilities for flipping:
        dist = dt_vec[:,None,None] * out_rates
        dist = torch.clip(dist,min=0,max=1)

        # Get flip mask:
        flip_mask = (torch.rand_like(dist) < dist)

        # Flip spins:
        x[flip_mask] = -x[flip_mask]
        
        return x

    def get_jarzinsky_corrector(self, x, time_vec, neighbor_lh_ratios):
        in_rates, stay_rate = self.get_in_rates(x, time_vec)
        weighted_in_rates = (in_rates * neighbor_lh_ratios).sum(dim=[1,2])
        return stay_rate + weighted_in_rates


def get_conv_model(config):

    kwargs = {
        'kernel_sizes': config.kernel_sizes,
        'num_channels': config.num_channels,
        'in_channels' : config.in_channels , 
    }
    return ConvSpinRateMatrix(**kwargs)


"""CODE FOR POTTS MODEL"""

class GridEmbedding(nn.Module):
    def __init__(self, n_cat, embedding_dim):
        super(GridEmbedding, self).__init__()
        # Create an embedding layer that maps each category (0, ..., n_cat-1) to a vector of size embedding_dim (i.e., C)
        self.embedding = nn.Embedding(n_cat, embedding_dim)

    def forward(self, x):
        # x is of shape (bs, L, L) and contains integers in [0, n_cat-1]
        # Apply embedding: output shape will be (bs, L, L, embedding_dim)
        x = self.embedding(x)
        # Rearrange the dimensions to (bs, embedding_dim, L, L)
        x = x.permute(0, 3, 1, 2)
        return x

class PottsLocEquivConvNet(torch.nn.Module):
    def __init__(self, n_cat: int, kernel_sizes: list,  num_channels = 4):
        super(PottsLocEquivConvNet, self).__init__()
        self.emb = GridEmbedding(n_cat, num_channels)
        in_channels = num_channels

        self.num_channels = num_channels

        sizes = [in_channels]
        self.convs = []
        self.initial_conv = NewMaskedConv2d(
                            num_channels, num_channels, kernel_sizes[0], padding=kernel_sizes[0]//2,
                            stride=1, padding_mode='circular', bias = False)
        self.mid_act = torch.nn.SiLU()
        self.convs = []
        for idx in range(len(kernel_sizes)-1):
            self.convs.append(StateDependentConv2D(
                                kernel_size=kernel_sizes[idx+1], 
                                in_channels=num_channels, 
                                out_channels=num_channels, 
                                prev_out_channels=num_channels))
        self.convs = torch.nn.ModuleList(self.convs)
        
        self.linear = torch.nn.Parameter(torch.randn(size=(n_cat,num_channels)) / torch.sqrt(torch.tensor(num_channels)))

    def forward(self, x, t):
        t = t[:,None,None,None]
        x_emb = self.emb(x)
        x_t = (1+t)*x_emb
        mid_x_pre_act = self.initial_conv(x_t)
        for idx in range(len(self.convs)):
            mid_x_post_act = self.mid_act(mid_x_pre_act)
            mid_x_pre_act = self.convs[idx](x_emb, t, mid_x_post_act) ### michael add T here
        final_out = mid_x_pre_act
        
        proj_out = torch.einsum("ck,bkhw->bchw", self.linear, final_out)
        offset = proj_out.gather(dim=1, index=x.unsqueeze(1))
        return proj_out - offset

class ConvPottsRateMatrix(nn.Module):
    def __init__(self, 
                 n_cat: int,
                 kernel_sizes: list,  
                 num_channels = 4):
        super().__init__()
        self.network = PottsLocEquivConvNet(n_cat=n_cat,
                                            kernel_sizes=kernel_sizes, 
                                            num_channels=num_channels)
    
    def forward(self, x_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        return self.network(x_vec, time_vec)

    def forward_set_x_to_zero(self, x_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        """Same as forward function but we set the values corresponding to the category in x to zero"""
        output = self.forward(x_vec, time_vec)
        
        # Set rates from x to itself to zero:
        index_tensor = x_vec.unsqueeze(1)  # shape: [4, 1, 8, 8]
        zero_src = torch.zeros_like(index_tensor, dtype=output.dtype, device=output.device)
        output.scatter_(dim=1, index=index_tensor, src=zero_src)
        return output

    def get_out_rates(self, x_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        output = self.forward_set_x_to_zero(x_vec, time_vec)
        output = torch.relu(output)
        
        # Get the rate of not updating anything (across all dimensions)
        # and update the according category element
        stay_rate = -output.sum(dim=[1, 2, 3])

        return output, stay_rate

    def get_in_rates(self, x_vec: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        output = self.forward_set_x_to_zero(x_vec, time_vec)
        stay_rate = -torch.relu(output).sum(dim=[1,2,3])
        return torch.relu(-output), stay_rate

    def sample_next_step(self, x, time_vec, dt_vec):
        
        # [batch_size, seq_len]
        out_rates, _ = self.get_out_rates(x, time_vec)

        # Compute flip mask (whether to update state or not):
        flip_probs = 1 - torch.exp(-dt_vec[:,None,None] * out_rates.sum(dim=[1]))
        flip_mask = (torch.rand_like(flip_probs) <= flip_probs)
        
        # Get probabilities probabilities for flipping:
        rates_perm = out_rates.permute(0, 2, 3, 1)
        normalizer = rates_perm.sum(dim=-1)
        probs_perm = rates_perm/torch.clip(normalizer,min=1e-7)[:,:,:,None]

        # To ensure numerical stability and avoid NaNs, we set the ones 
        # that are zero everywhere to be simple the uniform
        unnormalizable_mask = (probs_perm.sum(dim=-1)<1)
        unif_dummy = 1/rates_perm.shape[-1]
        probs_perm[unnormalizable_mask] = unif_dummy

        # Sample from categorical
        dist = torch.distributions.Categorical(probs=probs_perm)
        samples = dist.sample()
        
        # Flip spins:
        samples[~flip_mask] = x[~flip_mask]
        return samples

    def get_jarzinsky_corrector(self, x, time_vec, neighbor_lh_ratios):
        """
        x - tensor of shape (bs,L,L)
        time_vec - tensor of shape (bs,)
        neighbor_lh_ratios - tensor shape (bs, C, L, L) where C is the number of categories of the Potts model
        """
        in_rates, stay_rate = self.get_in_rates(x, time_vec)
        weighted_in_rates = (in_rates * neighbor_lh_ratios).sum(dim=[1,2,3])
        return stay_rate + weighted_in_rates

def get_potts_conv_model(config):

    kwargs = {
        'n_cat': config.n_cat,
        'kernel_sizes': config.kernel_sizes,
        'num_channels': config.num_channels,
    }
    return ConvPottsRateMatrix(**kwargs)