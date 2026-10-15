#!/bin/bash
# SPEC: S13 -- build an offline wheel set for the Kaggle kernel (internet OFF).
# Wheels must match the KAGGLE image's Python, not HPC's 3.14. Simplest reliable path: run this inside a
# Kaggle notebook WITH internet once, then save /kaggle/working/wheels as the dataset `oxalpha-wheels`.
set -euo pipefail
OUT=${1:-/kaggle/working/wheels}
mkdir -p "$OUT"
pip download -d "$OUT" timm pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
ls "$OUT"
# In the kernel: pip install --no-index --find-links /kaggle/input/oxalpha-wheels timm pydicom pylibjpeg ...
