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
export FLUX2_CACHE_ROOT="${FLUX2_CACHE_ROOT:-$FLUX2_DATASET_ROOT/flux2_qdm_cache_1k}"
exec python -m flux2_qdm.repeat --stage all --cycle-seconds 1200 \
  --output-root "${FLUX2_QDM_RUN:-../experiment/flux2-sliding11-qdm-1k-node01}" \
  --local-attention sliding --sliding-kernel-size 11 --replace-image-blocks \
  --resolution 1024 --batch-size 4 --grad-accum 1 --a1-steps 200 --a2-steps 200 --b-steps 1000 \
  --save-steps 25 --validation-steps 200 --workers 4 --threads 1 --resume auto "$@"
