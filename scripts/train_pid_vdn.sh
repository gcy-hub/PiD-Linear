#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
IFS=',' read -ra PID_VDN_GPUS <<< "$CUDA_VISIBLE_DEVICES"
PID_VDN_NPROC="${NPROC_PER_NODE:-${#PID_VDN_GPUS[@]}}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$PID_VDN_NPROC" -m scripts.train_pid_vdn "$@"
else
  : "${MASTER_ADDR:?Set MASTER_ADDR}" "${NODE_RANK:?Set NODE_RANK}"
  exec torchrun --nnodes="$NNODES" --nproc_per_node="$PID_VDN_NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29583}" -m scripts.train_pid_vdn "$@"
fi
