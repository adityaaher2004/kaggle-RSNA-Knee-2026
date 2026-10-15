#!/bin/bash
# SPEC: S0.2 -- re-download the competition data after the 2026-10-04 scratch purge.
# Usage: scripts/download_data.sh csv     (minutes; unblocks EDA + labeling)
#        scripts/download_data.sh dicom   (~570 GB zip; run inside a Slurm CPU job or tmux)
set -euo pipefail
module load python/3.14.5   # venv python needs libpython3.14 from the module
RAW=/N/scratch/aaher/RSNA/full_data/rsna-knee-abnormality-detection
COMP=rsna-knee-abnormality-detection
KAGGLE=${KAGGLE:-/N/scratch/aaher/RSNA/venv/bin/kaggle}
mkdir -p "$RAW"

case "${1:-csv}" in
  csv)
    for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
      "$KAGGLE" competitions download "$COMP" -f "$f" -p "$RAW"
      [ -f "$RAW/$f.zip" ] && unzip -o -q "$RAW/$f.zip" -d "$RAW" && rm "$RAW/$f.zip"
    done
    ls -la "$RAW"/*.csv ;;
  dicom)
    "$KAGGLE" competitions download "$COMP" -p "$RAW"
    unzip -n -q "$RAW/$COMP.zip" -d "$RAW"
    find "$RAW/train_series" -name '*.dcm' | wc -l   # expect ~819,640 incl. test (S0.3)
    ;;
  *) echo "usage: $0 csv|dicom"; exit 1 ;;
esac
