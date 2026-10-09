#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLA_DISABLE_TENSOR_CACHE=1
export OMP_NUM_THREADS=${CPU_THREADS:-1}
export MKL_NUM_THREADS=${CPU_THREADS:-1}
if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
PYTHON_BIN=${PYTHON_BIN:-python}
WORKERS=${WORKERS:-1}
if (( WORKERS > 1 )); then
  exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
    "$SCRIPT_DIR/diagnose_head_capacity.py" --threads "${CPU_THREADS:-1}" "$@"
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/diagnose_head_capacity.py" --threads "${CPU_THREADS:-1}" "$@"
fi
