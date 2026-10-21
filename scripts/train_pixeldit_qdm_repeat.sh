#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# The existing controller renews its own PID as well as torchrun every cycle.
exec python -m scripts.repeat_pid_vdn --stage all --output-root outputs/pixeldit-t2i-qdm-halo2-rope \
  --backbone pixeldit_t2i --memory-mode fixed4 --local-mode halo --local-halo 2 --region-size 8 \
  --resolution 1024 --flow-shift 4 --shift-base-size 1024 --remote-rope \
  --grad-accum 1 --a1-batch-size 4 --a2-batch-size 8 --b-batch-size 8 "$@"
