#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}" CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec python -m pid._src.pixeldit_qdm.synthetic_pipeline "$@"
