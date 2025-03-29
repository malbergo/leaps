import torch
import numpy


def absolute_mag(cfgs, L):
    """
    Compute the absolute magnetization |M| = | sum_i s_i | on a batch of configurations
    """
    return torch.mean(torch.abs(torch.sum(cfgs, dim= (1,2))) / (L*L))

def mags(cfgs, L):
    """
    Compute the magnetization M =  sum_i s_i for each configuration in a batch
    """
    return torch.sum(cfgs, dim= (1,2)) / (L*L)