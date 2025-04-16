import os
import torch
import argparse

from config import get_config


from src.modules.trainer import IsingLightningModule

from src.helpers import get_next_dir_number, zip_and_log_code

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.utilities import CombinedLoader


def main():

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--yml_path', type=str,  default = 'ymls/ising.yml')
    parser.add_argument('--ckpt_fname', type = str, default = None)

    
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    
    conf = get_config(
        yml_path = args.yml_path, 
        ckpt_fname = args.ckpt_fname,
    )
    
    
    print("CHECKPOINT FNAME HERE:", conf.ckpt_fname)
    
    ### print out the configuration you're using
    for k in vars(conf):
        print(k,':', getattr(conf,k))
    
    
    
    #### setting up checkpoint directories and such,
    #### basically if starting from a checkpoint, don't overwrite the previous one.
    if conf.ckpt_fname is not None:
        # Split the full checkpoint path into components
        path_parts = conf.ckpt_fname.split('/')

        # Find the index of 'ckpts' to get the directory just before it, which contains the number
        ckpt_index = path_parts.index('ckpts')
        dir_name = path_parts[ckpt_index - 1]  # The directory name right before 'ckpts'
        print("HERES' THE DIRECTORY NAME:", dir_name)
        # Extract the base run number from the directory name (ignoring potential suffixes)
        base_run_number = dir_name.split('-')[-1]
        if base_run_number[-1].isalpha():
            base_run_number = base_run_number[:-1]  # Remove the letter to get the base number

        # Determine the base path for experiments
        base_path = '/'.join(path_parts[:ckpt_index - 1])

        # Get all directories that start with the base run number
        try:
            existing_dirs = [d for d in os.listdir(base_path) if d.startswith(dir_name.split(f'{str(conf.L)}-')[0] + f'{str(conf.L)}-' + base_run_number)]
            existing_suffixes = [d.replace(dir_name, '').replace(base_run_number, '') for d in existing_dirs if len(d) > len(dir_name)]

            # Determine the next available letter suffix
            if existing_suffixes:
                last_letter = sorted(existing_suffixes)[-1]  # Get the last suffix in alphabetical order
                if last_letter:
                    next_letter = chr(ord(last_letter[-1]) + 1)
                else:
                    next_letter = 'a'
            else:
                next_letter = 'a'
        except FileNotFoundError:
            next_letter = 'a'  # If the directory does not exist, start with 'a'

        # Update the wandb name to append the new letter suffix to indicate a restart
        conf.wandb_name = f"{conf.wandb_name}-{base_run_number}{next_letter}"

        # Update the results directory
        conf.results_dir = f"{base_path}/{conf.wandb_name}/"

        # Ensure the directory exists
        if not os.path.exists(conf.results_dir):
            os.makedirs(conf.results_dir, exist_ok=True)


    else:
        base_path = f"{conf.results_path}/{conf.target}/L{conf.L}/expts"
        dir_number = get_next_dir_number(base_path, conf.wandb_name)
        conf.wandb_name = f"{conf.wandb_name}-{dir_number}" ## make unique
        conf.results_dir = f"{base_path}/{conf.wandb_name}/"

        if not os.path.exists(conf.results_dir):
            os.makedirs(conf.results_dir, exist_ok=True)
            
######################################################
            
       
    # setup wandb_logger
    wandb_logger = WandbLogger(project=conf.wandb_project, entity="samplers", 
                               name=conf.wandb_name, config=conf,
                               save_dir = conf.results_dir)
    
    # save code as wandb artifact, func in helpers.py
    zip_and_log_code(conf.code_path, conf.results_dir, wandb_logger)
    
    # check to make sure directories exist
    try:
        with open(os.path.join(conf.results_dir, 'test_write.txt'), 'w') as test_file:
            test_file.write('Testing write permissions.')
        os.remove(os.path.join(conf.results_dir, 'test_write.txt'))  # Clean up after the test
    except IOError as e:
        print(f"Error writing to directory {conf.results_dir}: {e}")
        
        
        
        
    # setup model
    if conf.target == 'ising' or conf.target == 'potts':
        if conf.ckpt_fname is None:
            model = IsingLightningModule(conf)
        else:
            model = IsingLightningModule.load_from_checkpoint(conf.ckpt_fname)
    else:
        raise NotImplementedError("only set up for ising and potts target right now")
        
        
    ### allows to monitor learning rate in wandb
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    ### allows to ckpt model in specified frequency of train steps
    ckpt_callback = ModelCheckpoint(save_top_k=-1, 
                                    every_n_train_steps = conf.ckpt_every, 
                                    save_last = True,
                                    dirpath = (conf.results_dir + conf.ckpts_dir))
    trainer = L.Trainer(accelerator = conf.accelerator, 
                        strategy='ddp_find_unused_parameters_true',
                        devices = conf.num_gpus,
                        num_nodes = conf.num_nodes,
                        logger=wandb_logger,
                        log_every_n_steps=conf.log_every_local,
                        gradient_clip_algorithm = 'norm', 
                        gradient_clip_val=conf.grad_clip_norm,
                        callbacks = [lr_monitor, ckpt_callback],
                        default_root_dir=conf.results_dir,
                        max_epochs = 20000
                       )
    
    # trainer.fit(model, ckpt_path = conf.ckpt_fname)
    trainer.fit(model, ckpt_path = conf.ckpt_fname)
    
if __name__ == '__main__':
    main() 
