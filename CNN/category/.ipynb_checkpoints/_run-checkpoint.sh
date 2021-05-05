#!/bin/bash
export XDG_RUNTIME_DIR=/home-1/bgalgan1@jhu.edu

# move to working directory
cd $SLURM_SUBMIT_DIR

# load module 
ml anaconda/2019.03
ml cuda/10.1

# activate conda environment
conda activate /home-net/home-1/bgalgan1@jhu.edu/code/tf-new

# run script
which python
python /home-1/bgalgan1@jhu.edu/repos/ClusNet/CNN/category/run.py
