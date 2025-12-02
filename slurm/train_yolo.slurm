#!/bin/bash
#SBATCH --job-name=yolo_fold
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --array=0-9
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access

module purge
module load anaconda cuda/12.9  # YOUR MODULES
conda activate ml              # your YOLO venv

echo "Running fold $SLURM_ARRAY_TASK_ID"

python train_fold.py --fold $SLURM_ARRAY_TASK_ID
