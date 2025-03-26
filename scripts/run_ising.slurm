#!/bin/bash
#SBATCH --job-name=my_job            # You can override this at submission with --job-name=...
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=256G
#SBATCH --mail-type=END
#SBATCH --mail-user=albergo@harvard.edu

# Use %x for the job name and %j for the job ID in filenames:
#SBATCH -o /n/holylabs/LABS/albergo_lab/Lab/albergo/projects/discrete-diffusion/leaps/slurmlogs/wandb/outfile-wandb-%x-%j
#SBATCH -e /n/holylabs/LABS/albergo_lab/Lab/albergo/projects/discrete-diffusion/leaps/slurmlogs/wandb/errfile-wandb-%x-%j

# --- Environment setup ---
source /n/home08/albergo/.bashrc
module load cuda
mamba activate tensor
export PYTHONPATH="${PYTHONPATH}:/n/home08/albergo/"
export PYTHONPATH="${PYTHONPATH}:/n/home08/albergo/discrete-diffusion/leaps/src/"

# --- Use the Slurm job name as the dataset name ---
TARGET="${SLURM_JOB_NAME}"

echo "JOB NAME (DATASET)  = ${DATASET}"
echo "OUTPUT LOG FILE     = outfile-wandb-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo "ERROR LOG FILE      = errfile-wandb-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

# --- Run the actual command, referencing the dataset ---
srun --export=ALL python ../main.py --yml_path "../ymls/${TARGET}.yml"