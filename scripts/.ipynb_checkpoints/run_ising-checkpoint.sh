#!/bin/bash

# environment settings and activate mamba source /n/home08/albergo/.bashrc
module load cuda
mamba activate tensor
export PYTHONPATH="${PYTHONPATH}:/n/home08/albergo/"
export PYTHONPATH="${PYTHONPATH}:/n/home08/albergo/projects/discrete-diffusion/leaps/src/"

# define paths to save dirs and parameters
TARGET='ising'

CUDA_VISIBLE_DEVICES=1 python ../main.py --yml_path "../ymls/${TARGET}.yml"


