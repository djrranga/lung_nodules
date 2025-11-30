#!/bin/bash
#SBATCH --job-name=annotate
#SBATCH --output=logs/pre_%A_%a.out
#SBATCH --error=logs/pre_%A_%a.err
#SBATCH --array=0-1185          # <-- CHANGE based on wc -l ct_paths.txt
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load anaconda
conda activate ml

DATASET=/work/users/d/r/dranga/dataset
CSV=/work/users/d/r/dranga/dataset/annotations.csv

# Execute Python script for THIS array index
conda run -n ml python annotate.py \
    --dataset $DATASET \
    --csv $CSV \
    --index $SLURM_ARRAY_TASK_ID
