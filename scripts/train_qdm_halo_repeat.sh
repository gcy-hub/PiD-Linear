#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export CPU_THREADS="${CPU_THREADS:-1}"
IFS=',' read -ra QDM_GPUS <<< "$GPU_IDS"
QDM_WORLD=$(( ${NNODES:-1} * ${NPROC_PER_NODE:-${#QDM_GPUS[@]}} ))
QDM_BATCH="${PID_QDM_BATCH:-4}"
QDM_ACCUM="${PID_QDM_ACCUM:-1}"
QDM_EFFECTIVE=$((QDM_WORLD * QDM_BATCH * QDM_ACCUM))
exec bash "$REPO_ROOT/scripts/train_pid_vdn_repeat.sh" \
  --stage all --cycle-seconds 1200 \
  --output-root "${PID_QDM_RUN:-outputs/pid-qdm-halo2}" \
  --memory-mode fixed4 --local-mode halo --local-halo 2 \
  --batch-size "$QDM_BATCH" --grad-accum "$QDM_ACCUM" --effective-batch "$QDM_EFFECTIVE" \
  --workers 4 --threads "$CPU_THREADS" --swanlab-mode offline "$@"
