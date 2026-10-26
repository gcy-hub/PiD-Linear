#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
exec torchrun --standalone --nproc_per_node="${#gpu_array[@]}" -m pid._src.pixeldit_qdm.recovery_diagnostics "$@"
