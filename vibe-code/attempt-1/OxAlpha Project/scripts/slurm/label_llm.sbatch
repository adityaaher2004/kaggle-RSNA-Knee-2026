#!/bin/bash
# SPEC: S6.2 -- report labeling with a local LLM on 1x H100.
#SBATCH -J oxa-label
#SBATCH -p h100-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /N/scratch/aaher/RSNA/runs/slurm/%x-%j.out
export HF_HOME=/N/scratch/aaher/RSNA/hf_cache
cd "$SLURM_SUBMIT_DIR"
/N/scratch/aaher/RSNA/venv-llm/bin/python -m labels.llm_labeler --cfg configs/labels_llm.yaml --hpc configs/hpc.yaml "$@"
