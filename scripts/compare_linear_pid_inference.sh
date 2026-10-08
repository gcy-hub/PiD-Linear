#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLA_DISABLE_TENSOR_CACHE="${FLA_DISABLE_TENSOR_CACHE:-1}"
IFS=',' read -ra LINEAR_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-${#LINEAR_GPU_ARRAY[@]}}" \
  -m pid._src.linear_pid.compare_inference "$@"
