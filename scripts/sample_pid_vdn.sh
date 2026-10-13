#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
IFS=',' read -ra PID_VDN_GPUS <<< "$CUDA_VISIBLE_DEVICES"
exec torchrun --standalone --nproc_per_node="${#PID_VDN_GPUS[@]}" -m scripts.sample_pid_vdn "$@"
