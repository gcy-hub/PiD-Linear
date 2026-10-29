#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/flux3:$PROJECT_ROOT/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=True DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
exec python -m flux3_qdm.launcher "$@"
