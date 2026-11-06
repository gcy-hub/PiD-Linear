#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/flux2${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLUX2_WEIGHTS_ROOT="${FLUX2_WEIGHTS_ROOT:-$HOME/huggingface_ckpts/FLUX.2-klein-base-4B}"
export FLUX2_DATASET_ROOT="${FLUX2_DATASET_ROOT:-$HOME/dataset/CLEAR-FLUX2-klein-base-4B-10K}"
export FLUX2_CACHE_ROOT="${FLUX2_CACHE_ROOT:-$FLUX2_DATASET_ROOT/flux2_qdm_cache}"
exec python -m flux2_qdm.launcher --weights-root "$FLUX2_WEIGHTS_ROOT" \
  --subset-root "$FLUX2_DATASET_ROOT" --cache-root "$FLUX2_CACHE_ROOT" "$@"
