import torch
import zipfile
import numpy as np
import os
import wandb

from src.nets.conv import get_conv_model
from src.nets.attn import get_attn_model
from src.nets.F_net import get_F_model

def grab(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def get_arch(config):
    
    if config.arch == 'conv':
        return get_conv_model(config)
    elif config.arg == 'attn':
        return get_attn_model(config)

    
def get_arch_F(config):
    return get_F_model(config)




def visualize_lattices(phi, n_rows = 2, n_cols = 5, save = None):
    """
    Visualize a subset of the lattices in a grid.

    Parameters:
    - phi: A tensor of shape [bs, L, L] representing the batch of lattices.
    - n_rows: Number of rows in the subplot grid.
    - n_cols: Number of columns in the subplot grid.
    - title: The title of the plot.
    """
    bs, L, _ = phi.shape
    n_plots = min(n_rows * n_cols, bs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5))

    indices = np.random.choice(bs, n_plots, replace=False)  # Randomly select lattice indices
        
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        # cmap = sns.cubehelix_palette(start=-0.8, rot=0.4, gamma=1.0, dark=0.2, light=0.9, reverse=False, as_cmap=True)
        v = 0.5
        im = ax.imshow(phi[idx], cmap='viridis', origin='lower', vmin=-v, vmax=v)
        ax.axis('off') 

    fig.subplots_adjust( wspace=0.1, hspace=-0.5)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    # plt.tight_layout()
    plt.show()
    
    


    
# ----------------- directory setup and stuff ----------------
def get_next_dir_number(base_path, base_name):
    print("MAKING THIS DIRECTORY IF IT DOESN'T EXIST:", base_path)
    os.makedirs(base_path, exist_ok=True)  # Ensure the base directory exists
    current_max = 0
    for name in os.listdir(base_path):
        if name.startswith(base_name):
            parts = name.split('-')
            try:
                num = int(parts[-1])
                if num > current_max:
                    current_max = num
            except ValueError:
                continue
    return current_max + 1
    
    
    
# e.g. results_dir = '/path/to/results/directory'
def zip_and_log_code(base_code_path, results_dir, logger):
    

    zip_path = os.path.join(results_dir, 'code.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # recursively add dict
        def add_directory(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, base_code_path)
                    zipf.write(file_path, arcname)

        # add src and ymls dirs
        add_directory(os.path.join(base_code_path, 'src'))
        add_directory(os.path.join(base_code_path, 'ymls'))
        
        # add config file
        config_path = os.path.join(base_code_path, 'config.py')
        zipf.write(config_path, os.path.relpath(config_path, base_code_path))
        
    
    # Log the zip file as an artifact 
    artifact = wandb.Artifact('code', type='code')
    artifact.add_file(zip_path)
    logger.experiment.log_artifact(artifact)
        
    
    
    
#### trainer helper functions

def repeat_integers(start, end, n_opt):
    """
    Use this function to make a list of length n_opt that anneals how many sampling steps are taken
    """
    range_length = end - start + 1
    full_repeats, remainder = divmod(n_opt, range_length)
    
    result = []
    for i in range(start, end + 1):
        result.extend([i] * full_repeats)
    
    if remainder > 0:
        result.extend([end] * remainder)
    
    return result


def inverse_repeat_quadratic(k_min, k_max, total_steps):
    """
    Generate a vector of length total_steps with values from k_min to k_max,
    where the number of repetitions for each integer increases linearly.
    
    For the i-th integer (starting with i=0 for k_min), the "ideal" repetition is:
        r_i = 1 + i*d
    and we require that the total sum equals total_steps:
        total_steps = N + d*(N-1)*N/2,
    where N = k_max - k_min + 1.
    
    Due to rounding we adjust the final vector to have exactly total_steps elements.
    
    Args:
        k_min (int): Starting integer.
        k_max (int): Ending integer.
        total_steps (int): Desired length of the output vector.
    
    Returns:
        list: A list of length total_steps with integers from k_min to k_max, where
              the number of repetitions increases approximately linearly.
    """
    N = k_max - k_min + 1
    if total_steps < N:
        raise ValueError("total_steps must be at least as many as the number of distinct values")
    
    # If only one value is requested, return a constant vector.
    if N == 1:
        return [k_min] * total_steps

    # Compute the step increase d from the formula:
    # total_steps = N + d*(N-1)*N/2   =>   d = 2*(total_steps - N) / (N*(N-1))
    d = 2 * (total_steps - N) / (N * (N - 1))
    
    # Compute the "ideal" repetition counts (in continuous space)
    r = [1 + i * d for i in range(N)]
    
    # Compute cumulative boundaries (continuous)
    cum = [0]
    for rep in r:
        cum.append(cum[-1] + rep)
    
    # Normalize the cumulative sum so that the last value equals total_steps
    scale = total_steps / cum[-1]
    cum_scaled = [x * scale for x in cum]
    
    # Round the cumulative boundaries to get integer indices
    cum_int = [int(round(x)) for x in cum_scaled]
    # Ensure the first and last boundaries are exactly 0 and total_steps
    cum_int[0] = 0
    cum_int[-1] = total_steps

    # Build the final vector: for each integer value, determine its count
    result = []
    for i in range(N):
        count = cum_int[i+1] - cum_int[i]
        result.extend([k_min + i] * count)
    
    # Due to rounding issues, adjust the length if necessary.
    if len(result) < total_steps:
        result.extend([k_max] * (total_steps - len(result)))
    elif len(result) > total_steps:
        result = result[:total_steps]
    
    return result