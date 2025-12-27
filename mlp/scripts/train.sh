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
conda activate 3factor
cd /gpfs/radev/home/dfl32/project/3factor/mlp

python main.py \
    --output_dir /gpfs/radev/pi/dijk/dfl32/3factor/outputs \
    --hidden_size 512 \
    --num_episodes 100000 \
    --num_train_trials 20 \
    --num_test_trials 10 \
    --item_size 15 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --grad_clip 2.0 \
    --num_episodes_per_reset 1 \
    --item_range 4 9 \
    --burn_in_period 100 \
    --nonadj_loss_multiplier 1.0 \
    --arbitrary \
    --plot_symbolic_distance \
    --delay_steps 2 \
    --additional_items 10
