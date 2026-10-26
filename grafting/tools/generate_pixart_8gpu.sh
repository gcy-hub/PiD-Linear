#!/usr/bin/env bash
# Direct single-server generation on eight GPUs; rerun unchanged to resume.
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA_ROOT="${PIXART_DATA_ROOT:-$HOME/dataset/PixArt-Sigma-Grafting-12K}"
exec python -u "$TOOLS/generate_pixart_dataset.py" \
  --model-dir "${PIXART_WEIGHTS:-$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS}" \
  --prompts "$DATA_ROOT/prompts_12k.jsonl" --output-root "$DATA_ROOT" \
  --num-samples 12000 --stage1-size 8000 \
  --height 2048 --width 2048 --steps 20 --cfg 4.5 --seed 42 \
  --batch-size 2 --workers "${PNG_WORKERS:-4}" --threads "$CPU_THREADS" "$@"
