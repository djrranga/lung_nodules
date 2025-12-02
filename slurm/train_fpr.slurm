#!/bin/bash
#SBATCH --job-name=fpr_train
#SBATCH --qos=gpu_access
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/fpr_train_%j.out
#SBATCH --error=logs/fpr_train_%j.err

# -----------------------------
# Environment Setup
# -----------------------------
echo "Loading modules..."
module purge
module load anaconda
module load cuda/12.4
module load cudnn/8.9.7

# Activate your venv
echo "Activating virtual environment..."
conda activate ml

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

# Show environment
echo "Python: $(which python)"
python -V
nvidia-smi

# -----------------------------
# Training Command
# -----------------------------
echo "Starting FPR training..."

conda run -n ml python train_fpr.py \
    --data_root /work/users/d/r/dranga/dataset_fpr \
    --epochs 20 \
    --batch_size 16 \
    --num_workers 0 \
    --device cuda \
    --val_frac 0.2

echo "Training completed."
