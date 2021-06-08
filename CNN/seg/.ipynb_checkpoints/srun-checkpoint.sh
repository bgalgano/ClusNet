#!/bin/bash
#SBATCH --job-name=seg
#SBATCH -o output_seg.out
#SBATCH -p gpup100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bgalgan1@jhu.edu

# move to working directory
cd $SLURM_SUBMIT_DIR
export XDG_RUNTIME_DIR=/home-1/bgalgan1@jhu.edu

# load module 
ml anaconda/2019.03
ml cuda/10.1

# activate conda environment
conda activate /home-net/home-1/bgalgan1@jhu.edu/code/tf-new

# run script
which python
python /home-1/bgalgan1@jhu.edu/repos/ClusNet/CNN/seg/seg.py
