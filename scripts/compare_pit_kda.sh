#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -ra PIT_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#PIT_GPU_ARRAY[@]} == 1 )); then
  exec python -m scripts.compare_pit_kda "$@"
else
  exec torchrun --standalone --nproc_per_node="${#PIT_GPU_ARRAY[@]}" -m scripts.compare_pit_kda "$@"
fi
