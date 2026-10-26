#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"
exec python -m scripts.repeat_pid_vdn \
  --stage all --cycle-seconds 1200 --grace-seconds 180 \
  --output-root "${PIXELDIT_PARTIAL_RUN:-outputs/pixeldit-t2i-qdm-halo2-full-2-3-4-5-node01}" \
  --backbone pixeldit_t2i --memory-mode fixed4 --local-mode halo --local-halo 2 --region-size 8 \
  --full-attention-layers 2,3,4,5 --no-remote-rope \
  --resolution 1024 --flow-shift 4 --shift-base-size 1024 \
  --a1-steps 500 --a2-steps 1000 --b-steps 10000 \
  --a1-batch-size 4 --a2-batch-size 8 --b-batch-size 8 --grad-accum 1 \
  --save-steps 50 --validation-steps 500 --workers 4 --threads "${CPU_THREADS:-1}" \
  --swanlab-mode offline --resume auto "$@"
