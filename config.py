import yaml
import os
import argparse
from yaml import SafeLoader

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# deterministic python code that sets some more configs
# as a function of other configs
def set_more_stuff(config):
    
    c = config
    c.wandb_name = str(c.target) + '-L' + str(c.L)

    return c


def get_config(yml_path, ckpt_fname = None):

    with open(yml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=SafeLoader)
    config = dict2namespace(config_dict)


    ###### CKPT and RESULTS PATH ########################

    if ckpt_fname is not None:
        # config.ckpt_fname = config.results_path + "ckpts/" + ckpt_fname
        config.ckpt_fname =  ckpt_fname
        print("will be resuming from", config.ckpt_fname)
    else:
        config.ckpt_fname = None
        print("wont be resuming from a ckpt")
    ######################


    config = set_more_stuff(config)
    assert config.target in ['ising', 'uco'], "possible tasks are now ising and unsupervised combinatorial optimization"
    return config