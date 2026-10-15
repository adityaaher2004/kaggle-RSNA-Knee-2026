#!/bin/bash
# SPEC: S9 -- one fold per array task. Usage: sbatch scripts/slurm/train_fold.sbatch configs/stage_b_reader.yaml SEED
#SBATCH -J oxa-train
#SBATCH -p h100-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH --array=0-4
#SBATCH -o /N/scratch/aaher/RSNA/runs/slurm/%x-%A_%a.out
module load python/3.14.5
cd "$SLURM_SUBMIT_DIR"
/N/scratch/aaher/RSNA/venv/bin/python -m training.train_stage_b --cfg "${1:-configs/stage_b_reader.yaml}" \
  --hpc configs/hpc.yaml --fold "$SLURM_ARRAY_TASK_ID" --seed "${2:-0}"
