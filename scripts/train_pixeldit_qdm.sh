#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
count="${NPROC_PER_NODE:-${#GPUS[@]}}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$count" -m scripts.train_pixeldit_qdm "$@"
else
  exec torchrun --nnodes="$NNODES" --nproc_per_node="$count" --node_rank="${NODE_RANK:?}" \
    --master_addr="${MASTER_ADDR:?}" --master_port="${MASTER_PORT:-29587}" -m scripts.train_pixeldit_qdm "$@"
fi
