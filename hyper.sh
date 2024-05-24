#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=md1823
#SBATCH --output=dgbl_%j.out
#SBATCH --partition=gpgpuM


source /vol/bitbucket/md1823/dgbl/dgl-team-project/venv/bin/activate

export PYTHONUNBUFFERED=TRUE

srun python hyperparam.py
