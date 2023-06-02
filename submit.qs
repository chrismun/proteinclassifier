#!/bin/bash
#SBATCH -A mcb200107p
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH  -p GPU-shared 
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=00:05:00
module load anaconda3
source activate protcls
python chris-train.py