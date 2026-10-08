#!/usr/bin/env bash
# Run after download_all.sh, from the repository root, in the training environment.
set -euo pipefail

WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"
DATASET_ROOT="${DATASET_ROOT:-./raw_data/MultiAspect-4K-1M}"
GALLERY_ROOT="${GALLERY_ROOT:-./outputs/linear-pid/assets}"
GPU_IDS="${GPU_IDS:-0}"
INDEX_WORKERS="${INDEX_WORKERS:-8}"
TEXT_WORKERS="${TEXT_WORKERS:-2}"
CPU_THREADS="${CPU_THREADS:-1}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-8}"

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
IFS=',' read -ra PREPARE_GPUS <<< "$GPU_IDS"

# Build the full image index; decode failures during training skip the global batch.
python scripts/prepare_linear_pid_data.py \
    --dataset-root "$DATASET_ROOT" --output-root "$DATASET_ROOT/linear_pid_index" \
    --workers "$INDEX_WORKERS" --validation-size 1024 --seed 42 \
    --image-verification header --resume

# Cache all captions on the selected GPUs, preserving per-shard progress.
torchrun --standalone --nproc_per_node="${#PREPARE_GPUS[@]}" \
    -m scripts.prepare_linear_pid_text_cache \
    --dataset-root "$DATASET_ROOT" --output-root "$DATASET_ROOT/linear_pid_text_cache" \
    --weights-root "$WEIGHTS_ROOT" --batch-size "$TEXT_BATCH_SIZE" \
    --workers "$TEXT_WORKERS" --threads "$CPU_THREADS" --max-seconds 0 --resume

# Fixed real-image and Z-Image-Turbo conditions for the 2K/4K validation galleries.
torchrun --standalone --nproc_per_node="${#PREPARE_GPUS[@]}" \
    -m scripts.prepare_linear_pid_assets \
    --weights-root "$WEIGHTS_ROOT" --index-root "$DATASET_ROOT/linear_pid_index" \
    --output-root "$GALLERY_ROOT" \
    --real-count 64 --generated-count 32 --four-k-count 8 --seed 42
