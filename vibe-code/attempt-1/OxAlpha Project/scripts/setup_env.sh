#!/bin/bash
# SPEC: S0.1 -- project venv on IU HPC. Inherits torch 2.14+cu130 from the python/3.14.5 module.
# The LLM labeler (vLLM) gets its own venv because vLLM pins its own torch build.
set -euo pipefail
module load python/3.14.5
ROOT=/N/scratch/aaher/RSNA
HERE="$(cd "$(dirname "$0")/.." && pwd)"

python3 -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/pip" install -U pip
"$ROOT/venv/bin/pip" install -r "$HERE/requirements.txt"
"$ROOT/venv/bin/python" -c "import torch, timm, pydicom; print('ok', torch.__version__, timm.__version__, pydicom.__version__)"

# Labeler env (run on a GPU node if wheels need CUDA at build time).
python3 -m venv "$ROOT/venv-llm"
"$ROOT/venv-llm/bin/pip" install -U pip
"$ROOT/venv-llm/bin/pip" install vllm pandas pyyaml
