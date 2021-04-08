#!/bin/bash
#SBATCH --output=out_gauss.out
#SBATCH --error=err_gauss.err
#SBATCH --job-name=r_mod
#SBATCH -p gpup100
#SBATCH --gres=gpu:4
#SBATCH -t 02:00:00
#SBATCH --mail-user=bgalgan1@jhu.edu
#SBATCH --mail-type=ALL

# move to working directory
cd $SLURM_SUBMIT_DIR

export XDG_RUNTIME_DIR=/home-1/bgalgan1@jhu.edu

# move to working directory
cd $SLURM_SUBMIT_DIR

# load module 
ml anaconda/2019.03
ml cuda/10.1

# activate conda environment
conda activate /home-net/home-1/bgalgan1@jhu.edu/code/env/tf

# run script
which python
python /home-1/bgalgan1@jhu.edu/repos/ClustNet/code/load_model.py

