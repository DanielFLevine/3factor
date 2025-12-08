#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=daniel.levine@yale.edu
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00

module load miniconda
module load CUDA/12.8
conda activate metati
cd /gpfs/radev/home/dfl32/project/3factor/metati

python main.py \
    --hidden-size 200 \
    --wandb-project 3factor \