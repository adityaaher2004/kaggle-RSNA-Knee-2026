#!/bin/bash
# SPEC: S3 -- DICOM -> cache v1. Array over 32 shards of the series list.
#SBATCH -J oxa-prep
#SBATCH -p general
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-31
#SBATCH -o /N/scratch/aaher/RSNA/runs/slurm/%x-%A_%a.out
module load python/3.14.5
cd "$SLURM_SUBMIT_DIR"
/N/scratch/aaher/RSNA/venv/bin/python -m data.preprocessing --cfg configs/base.yaml --hpc configs/hpc.yaml \
  --shard "$SLURM_ARRAY_TASK_ID" --n-shards 32 --workers 32
