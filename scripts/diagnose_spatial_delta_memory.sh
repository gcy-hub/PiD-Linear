#!/usr/bin/env bash
set -euo pipefail
PID_MEMORY_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PID_MEMORY_REPO"
export PYTHONPATH="$PID_MEMORY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
IFS=',' read -ra PID_MEMORY_GPUS <<< "$CUDA_VISIBLE_DEVICES"
exec torchrun --standalone --nproc_per_node="${#PID_MEMORY_GPUS[@]}" -m scripts.diagnose_spatial_delta_memory "$@"
