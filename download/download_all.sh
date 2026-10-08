#!/usr/bin/env bash
# Run from the repository root. Edit these variables or set them in the environment.
set -euo pipefail

WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"
DATASET_ROOT="${DATASET_ROOT:-./raw_data/MultiAspect-4K-1M}"
MODEL_WORKERS="${MODEL_WORKERS:-4}"
IMAGE_WORKERS="${IMAGE_WORKERS:-8}"
export HF_HUB_OFFLINE=0
export HF_ENDPOINT=https://huggingface.co

# PiD v1.5 FLUX, undistilled: initialization and the original-model baseline.
python download/download_models.py --models pid \
    --weights-root "$WEIGHTS_ROOT" --workers "$MODEL_WORKERS"

# Frozen FLUX VAE for image conditioning.
python download/download_models.py --models vae \
    --weights-root "$WEIGHTS_ROOT" --workers "$MODEL_WORKERS"

# Frozen Gemma encoder for captions (accept its HF license and log in first).
python download/download_models.py --models gemma \
    --weights-root "$WEIGHTS_ROOT" --workers "$MODEL_WORKERS"

# Z-Image-Turbo for prompt-only inference and generated validation conditions.
python download/download_models.py --models zimage \
    --weights-root "$WEIGHTS_ROOT" --workers "$MODEL_WORKERS"

# Dataset metadata, then all available image URLs using the supplied downloader.
python download/download_metadata.py --output-root "$DATASET_ROOT"
python download/download_images.py all \
    --json-dir "$DATASET_ROOT/data_jsons" --datas-dir "$DATASET_ROOT/datas" \
    --workers "$IMAGE_WORKERS" --max-per-json 0 \
    --timeout 60 --retries 3 --flush-every 200
