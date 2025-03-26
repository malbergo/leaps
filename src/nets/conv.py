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
        self.A = torch.nn.Parameter(0.005*torch.rand(in_channels, kernel_size*kernel_size, prev_out_channels)/torch.sqrt(torch.tensor(kernel_size*kernel_size)))
        self.b1 = torch.nn.Parameter(0.005*torch.rand(in_channels, kernel_size*kernel_size))
        # self.C = torch.nn.Parameter(torch.rand(in_channels, kernel_size*kernel_size,kernel_size*kernel_size)/torch.sqrt(torch.tensor(kernel_size*kernel_size)))
        self.b2 = torch.nn.Parameter(0.005*torch.rand(in_channels, kernel_size*kernel_size))
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