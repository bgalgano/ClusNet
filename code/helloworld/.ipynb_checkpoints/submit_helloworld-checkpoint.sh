#!/bin/bash
#SBATCH -J hello_world
#SBATCH -o hello_world_shared_1.log
#SBATCH -p shared
#SBATCH -t 2:00:00
#SBATCH --mem 1G
# move to working directory
cd $SLURM_SUBMIT_DIR
# load module and activate conda environment
#module list
module load anaconda/2019.03
conda activate python3.8
# run script
which python
python /home-3/mntampa1@jhu.edu/helloworld/hello_world.py
