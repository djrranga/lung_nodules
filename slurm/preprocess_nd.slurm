#!/bin/bash
#SBATCH --job-name=preprocess_nd
#SBATCH --output=logs/pre_%A_%a.out
#SBATCH --error=logs/pre_%A_%a.err
#SBATCH --array=0-887          # <-- CHANGE based on wc -l ct_paths.txt
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load anaconda
conda activate ml

INPUT_DIR=/work/users/d/r/dranga/ln
OUTPUT_DIR=/work/users/d/r/dranga/dataset
FILELIST=ct_paths.txt

# Execute Python script for THIS array index
conda run -n ml python preprocess_nodule_detection_slices.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    --filelist $FILELIST \
    --index $SLURM_ARRAY_TASK_ID
