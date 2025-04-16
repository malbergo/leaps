import torch

from src.nets.conv import ConvSpinRateMatrix
from src.modules.rhot import Ising



def loss_ising(nnet: ConvSpinRateMatrix, Energy: Ising, sigma_vec, time_vec):
    
    in_rates, stay_rate = nnet.get_in_rates(sigma_vec, time_vec)
    _, delta_H_neighbors = Energy.Ut(sigma_vec, time_vec)
    lh_ratios = torch.exp(-delta_H_neighbors)
    jarz_corrector = nnet.get_jarzinsky_corrector(sigma_vec, time_vec, lh_ratios)

    # dtUt
    dtUt_vec, _ = Energy.dtUt(sigma_vec,time_vec)
    
    # dtFt
    dt_F_t_vec = F_t_net.gradt(time_vec)
    
    # pinn
    loss_vec = ((-dtUt_vec - jarz_corrector + dt_F_t_vec)**2)
    loss_val = loss_vec.mean()
    return loss_val


def make_loss(config):
    if config.target == 'ising' or config.target == "potts":
        loss_fn = loss_ising
        
    return loss_fn


if __name__ == '__main__':
    
    ## checking proper imports
    print(Ising)
    print(ConvSpinRateMatrix)
    