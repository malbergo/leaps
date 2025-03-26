import torch
from torch.func import vmap, jacfwd


def get_interpolating_functions(config):
    
    if config.path == 'linear':
        def J_func(t):
            return (1 - t) * config.J_0 + t * config.J_1
        def mu_func(t):
            return (1 - t) * config.mu_0 + t * config.mu_1
        def B_func(t):
            return (1 - t) * config.B_0 + t * config.B_1
        def beta_func(t):
            return (1 - t) * config.beta_0 + t * config.beta_1
        return J_func, mu_func, B_func, beta_func
    else:
        raise NotImplementedError
    
    
def make_ising(config):
    J_func, mu_func, B_func, beta_func = get_interpolating_functions(config)
    Energy = Ising(config.L, J_func, mu_func, B_func, beta_func)
    return Energy

class Ising:
    def __init__(self, L, J_func, mu_func, B_func, beta_func):
        self.J_func    = J_func
        self.mu_func   = mu_func
        self.B_func    = B_func
        self.beta_func = beta_func
        # self.device = device
        self.L      = L
        self.N      = (L,L)
        self.d      = 2
        t1          = torch.tensor(1.0)#.to(device)
        t0          = torch.tensor(0.0)#.to(device)
        self.J_1    = J_func(t1)
        self.J_0    = J_func(t0)
        self.mu_1   = mu_func(t1)
        self.mu_0   = mu_func(t0)
        self.B_1    = B_func(t1)
        self.B_0    = B_func(t0)
        self.beta_1    = beta_func(t1)
        self.beta_0    = beta_func(t0)
        print("Js:"    , self.J_1, self.J_0)
        print("mus:"   , self.mu_1, self.mu_0)
        print("Bs:"    , self.B_1, self.B_0)
        print("betas:" , self.beta_1, self.beta_0)
        
    def setup_params(self, t):
        

        J    = self.J_func(t)
        mu   = self.mu_func(t)
        B    = self.B_func(t)
        beta = self.beta_func(t)
        return J, mu, B, beta
    
    
    def _single_Ut(self, cfg, t):
        """
        Compute the energy of a single 2D Ising configuration.

        cfg: Tensor of shape (L, L) with values in {-1, +1}
        
        J    : coupling constant   (float)
        mu   : magnetic moment     (float)
        B    : external field      (float) 
        beta : inverse temperature (float)

        Returns: energy.
        """
        
        J, mu, B, beta = self.setup_params(t)

        # periodic bcs, shift right on columns and down on rows
        s_left = torch.roll(cfg, shifts=1, dims=1)
        s_top = torch.roll(cfg, shifts=1, dims=0)
        s_right = torch.roll(cfg, shifts=-1, dims=1)
        s_down = torch.roll(cfg, shifts=-1, dims=0)

        interaction_per_node = -J * (cfg * (s_right + s_down + s_left + s_top))

        interaction_delta_to_neighbor =  -2 * interaction_per_node 
        
        interaction = interaction_per_node.sum()/2

        ext_field_per_node = -mu * B * cfg

        ext_field_delta_to_neighbor = -2 * ext_field_per_node
        
        ext_field = ext_field_per_node.sum()

        H = interaction + ext_field
        delta_H_to_neighbor = interaction_delta_to_neighbor + ext_field_delta_to_neighbor

        return beta*H, beta*delta_H_to_neighbor
    

    def Ut(self, cfgs, ts):
        """
        Compute the time-dependent energy of a batch of 2D Ising configurations 
        by vectorizing _single_Ut.
        """
        return vmap(self._single_Ut, in_dims = (0,0), out_dims=(0,0), randomness='different')(cfgs, ts)
    
    def _single_dtUt(self, cfg, t):
        """
        Compute the time-derivative of the energy of a single 2D Ising configuration
        using jacfwd.
        """
        return jacfwd(self._single_Ut, 1)(cfg,t)
    
    def dtUt(self, cfgs, ts):
        """
        Compute the time-derivative of the energy of a batch 2D Ising configurations
        using jacfwd.
        """
        return vmap(self._single_dtUt, in_dims=(0,0), out_dims =(0,0), randomness='different')(cfgs,ts)
    
    
    def _single_dxUt(self, cfg, t):
        """
        Compute the spatial-derivative of the energy of a single 2D Ising configuration
        using jacfwd.
        """
        return jacfwd(self._single_Ut, 0)(cfg,t)
        
    def dxUt(self, cfgs, ts):
        """
        Compute the spatial-derivative of the energy of a batch of single 2D Ising configurations
        using jacfwd.
        """
        return vmap(self._single_dxUt, in_dims=(0,0), out_dims=(0), randomness='different')(cfgs,ts)


    def sample(self, ts):
        
        bs = len(ts)
        
        J, mu, B, beta = self.setup_params(ts)
        
        if (J.any() != 0) or (B.any() != 0 and mu.any() != 0):
            raise ValueError("can only directly sample the non-interacting (J=0; B=0 or mu=0) theory.")
            
        cfgs = 2 * torch.randint(0, 2, (bs, self.L, self.L)) - 1
        
        return cfgs.float()

    
    
def _test_Ising():
    
    "------testing time-dependent Ising Distribution------"
    
    import time 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(10)
    d = 2
    L = 15
    lattice_shape = [L for _ in range(d)]

    J_0  = torch.tensor(0.0)#.to(device)
    J_1  = torch.tensor(0.4)#.to(device)

    mu_0 = torch.tensor(0.0)#.to(device)
    mu_1 = torch.tensor(0.0)#.to(device)

    B_0  = torch.tensor(0.0)#.to(device)
    B_1  = torch.tensor(0.0)#.to(device)

    beta_0  = torch.tensor(0.70)#.to(device)
    beta_1  = torch.tensor(0.70)#.to(device)


    def J_func(t):
        return (1 - t) * J_0 + t * J_1

    def mu_func(t):
        return (1 - t) * mu_0 + t * mu_1

    def B_func(t):
        return (1 - t) * B_0 + t * B_1

    def beta_func(t):
        return (1 - t) * beta_0 + t * beta_1
        
        
    Energy = Ising(L, J_func, mu_func, B_func, beta_func)
    
    bs = 10
    cfgs =  (2 * torch.randint(0, 2, (bs, L, L)) - 1).float()#.to(device)
    
    ## force func
    tic = time.perf_counter()
    ts = torch.rand(bs)#.to(device)
    vals = Energy.Ut(cfgs, ts)
    dtvals = Energy.dtUt(cfgs, ts)
    F = Energy.dxUt(cfgs, ts)
    toc = time.perf_counter()     
    print("[Ising] energy, time deriv, and force took:", toc - tic, " seconds")
    
    ## sampling func 
    ts = torch.zeros((bs,))#.to(device)
    cfgs = Energy.sample(ts=ts)
    

# Potts model:
def get_interpolating_functions_potts(config):
    
    if config.path == 'linear':
        def J_func(t):
            return (1 - t) * config.J_0 + t * config.J_1
        def B_func(t):
            return (1 - t) * config.B_0 + t * config.B_1
        def beta_func(t):
            return (1 - t) * config.beta_0 + t * config.beta_1
        return J_func, B_func, beta_func
    else:
        raise NotImplementedError
    
    
def make_potts(config):
    J_func, B_func, beta_func = get_interpolating_functions_potts(config)
    Energy = Potts(config.n_cat, config.L, J_func, B_func, beta_func)
    return Energy

class Potts:
    def __init__(self, n_cat, L, J_func, B_func, beta_func):
        self.n_cat = n_cat
        self.J_func    = J_func
        self.B_func    = B_func
        self.beta_func = beta_func
        self.L      = L
        self.N      = (L,L)
        self.d      = 2
        t1          = torch.tensor(1.0)#.to(device)
        t0          = torch.tensor(0.0)#.to(device)
        self.J_1    = J_func(t1)
        self.J_0    = J_func(t0)
        self.B_1    = B_func(t1)
        self.B_0    = B_func(t0)
        self.beta_1    = beta_func(t1)
        self.beta_0    = beta_func(t0)
        print("Js:"    , self.J_1, self.J_0)
        print("Bs:"    , self.B_1, self.B_0)
        print("betas:" , self.beta_1, self.beta_0)
        
    def setup_params(self, t):
        

        J    = self.J_func(t)
        B    = self.B_func(t)
        beta = self.beta_func(t)
        return J, B, beta
    
    
    def _single_Ut(self, cfg, t):
        """
        Compute the energy of a single 2D Ising configuration.

        cfg: Tensor of shape (L, L) with values in {0,1,...,self.n_cat-1}
        
        J    : coupling constant   (float)
        beta : inverse temperature (float)

        Returns: energy.
        """
        
        J, B, beta = self.setup_params(t)

        # periodic bcs, shift right on columns and down on rows
        s_left = torch.roll(cfg, shifts=1, dims=1)
        s_top = torch.roll(cfg, shifts=1, dims=0)
        s_right = torch.roll(cfg, shifts=-1, dims=1)
        s_down = torch.roll(cfg, shifts=-1, dims=0)

        # Count number of edges with same category
        equal_left = (cfg == s_left).int()
        equal_right = (cfg == s_right).int()
        equal_top = (cfg == s_top).int()
        equal_down = (cfg == s_down).int()
        interaction_per_node = (equal_left + equal_right + equal_top + equal_down)

        # Get energy from interaction of neighbors
        interaction_energy = - J * interaction_per_node.sum()/2

        # Get the delta of the energy when flipping the spin 
        #(Now we need one value for each possible value the spin could flip to)
        cat_vec = torch.arange(0,self.n_cat).to(cfg.device)
        cat_equal_left = (cat_vec[:,None,None] == s_left[None,:,:]).int()
        cat_equal_right = (cat_vec[:,None,None] == s_right[None,:,:]).int()
        cat_equal_top = (cat_vec[:,None,None] == s_top[None,:,:]).int()
        cat_equal_down = (cat_vec[:,None,None] == s_down[None,:,:]).int()
        cat_interaction = (cat_equal_left + cat_equal_right + cat_equal_top + cat_equal_down)
        interaction_delta_to_neighbor =  - J * (cat_interaction - interaction_per_node[None,:,:])

 
        # Ignore external field for now
        ext_field_per_node = 0
        ext_field_delta_to_neighbor = 0
        
        # Get total Hamiltonian
        H = interaction_energy + ext_field_per_node
        delta_H_to_neighbor = interaction_delta_to_neighbor + ext_field_delta_to_neighbor

        return beta*H, beta*delta_H_to_neighbor
    

    def Ut(self, cfgs, ts):
        """
        Compute the time-dependent energy of a batch of 2D Ising configurations 
        by vectorizing _single_Ut.
        """
        return vmap(self._single_Ut, in_dims = (0,0), out_dims=(0,0), randomness='different')(cfgs, ts)
    
    def _single_dtUt(self, cfg, t):
        """
        Compute the time-derivative of the energy of a single 2D Ising configuration
        using jacfwd.
        """
        return jacfwd(self._single_Ut, 1)(cfg,t)
    
    def dtUt(self, cfgs, ts):
        """
        Compute the time-derivative of the energy of a batch 2D Ising configurations
        using jacfwd.
        """
        return vmap(self._single_dtUt, in_dims=(0,0), out_dims =(0,0), randomness='different')(cfgs,ts)
    
def _test_Potts():
    
    "------testing time-dependent Ising Distribution------"
    
    import time 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(10)
    d = 2
    L = 15
    n_cat = 5
    lattice_shape = [L for _ in range(d)]

    J_0  = torch.tensor(0.0)#.to(device)
    J_1  = torch.tensor(0.4)#.to(device)

    mu_0 = torch.tensor(0.0)#.to(device)
    mu_1 = torch.tensor(0.0)#.to(device)

    B_0  = torch.tensor(0.0)#.to(device)
    B_1  = torch.tensor(0.0)#.to(device)

    beta_0  = torch.tensor(0.70)#.to(device)
    beta_1  = torch.tensor(0.70)#.to(device)


    def J_func(t):
        return (1 - t) * J_0 + t * J_1

    def B_func(t):
        return (1 - t) * B_0 + t * B_1

    def beta_func(t):
        return (1 - t) * beta_0 + t * beta_1
        
        
    Energy = Potts(n_cat, L, J_func, B_func, beta_func)
    
    bs = 10
    cfgs = torch.randint(0, n_cat, (bs, L, L))
    
    ## force func
    tic = time.perf_counter()
    ts = torch.rand(bs)#.to(device)
    vals = Energy.Ut(cfgs, ts)
    dtvals = Energy.dtUt(cfgs, ts)
    toc = time.perf_counter()     
    print("[Potts] energy, time deriv, and force took:", toc - tic, " seconds")
    
    
if __name__ == '__main__':
    _test_Ising()
    _test_Potts()
    
    
    
    