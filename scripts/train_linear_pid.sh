#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -ra LINEAR_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
LINEAR_NPROC="${NPROC_PER_NODE:-${#LINEAR_GPU_ARRAY[@]}}"
PRESET=4gpu
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == --preset ]]; then j=$((i+1)); PRESET="${!j}"; fi
  if [[ "${!i}" == --preset=* ]]; then LINEAR_ARG="${!i}"; PRESET="${LINEAR_ARG#*=}"; fi
done
if [[ "$PRESET" == 8gpu && "${NNODES:-1}" != 2 ]]; then
  echo '8gpu needs NNODES=2 and one launch per node, or use train_linear_pid_8gpu.slurm.' >&2
  exit 2
fi
if [[ "${NNODES:-1}" == 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$LINEAR_NPROC" -m scripts.train \
    --config=pid/_src/configs/linear_pid/config.py -- "$@"
else
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
  : "${NODE_RANK:?Set NODE_RANK separately on each node}"
  exec torchrun --nnodes="$NNODES" --nproc_per_node="$LINEAR_NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29571}" -m scripts.train \
    --config=pid/_src/configs/linear_pid/config.py -- "$@"
fi
