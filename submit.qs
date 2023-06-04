#!/bin/bash
#SBATCH -A mcb200107p
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH  -p GPU-shared 
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=00:05:00
# NOTE : I have been doing interactive runs, with 4 GPUs, instead of this 
module load anaconda3
source activate protcls
python train.py