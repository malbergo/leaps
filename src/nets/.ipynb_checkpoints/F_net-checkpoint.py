import torch
from torch.func import vmap, jacfwd

class FNetwork(torch.nn.Module):
    # nn that takes x in R^d and t in [0, 1] and outputs a scalar
    def __init__(self, hidden_sizes=(32, 32, 32), activation=torch.nn.SiLU):
        super(FNetwork, self).__init__()
        layers = []
        input_dim = 1  # Assuming input dimension is 1; adjust as needed
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        layers.append(torch.nn.Linear(input_dim, 1))  # Output layer
        self.net = torch.nn.Sequential(*layers)
        
    def ensure_size1(self, tensor):
        
        if tensor.dim() == 0:
            return tensor.unsqueeze(0)
        return tensor

    def _single_forward(self, t):
       
        t = self.ensure_size1(t)
        
        return self.net(t).squeeze()

    def forward(self, t):
        return vmap(self._single_forward, in_dims=(0), out_dims=(0))(t)
    
    def _single_gradt(self, t):
        return jacfwd(self._single_forward, 0)(t)
       
    def gradt(self, t):
        return vmap(self._single_gradt, in_dims=(0), out_dims=(0), randomness='different')(t)
    
    
def get_F_model(config):

    kwargs = {
        'hidden_sizes': config.F_hidden_sizes, 
    }
    return FNetwork(**kwargs)