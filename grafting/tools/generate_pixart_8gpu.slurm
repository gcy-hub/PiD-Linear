#!/usr/bin/env bash
#SBATCH --job-name=pixart-sigma-12k-generate
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project root}}"
cd "$REPO_ROOT"
source "${CONDA_ROOT:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate linear-pid
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
unset GPU_IDS GPU_DEVICE_ORDINAL
PIXART_DATA="${PIXART_DATA_ROOT:-$HOME/dataset/PixArt-Sigma-Grafting-12K}"
exec python -u grafting/tools/generate_pixart_slurm.py \
  --output-root "$PIXART_DATA" --nodes "$SLURM_JOB_NUM_NODES" \
  --gpus-per-node 4 --cpus-per-task "$SLURM_CPUS_PER_TASK" -- \
  --model-dir "${PIXART_WEIGHTS:-$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS}" \
  --prompts "$PIXART_DATA/prompts_12k.jsonl" \
  --num-samples 12000 --stage1-size 8000 \
  --height 2048 --width 2048 --steps 20 --cfg 4.5 --seed 42 \
  --batch-size 2 --workers 4 --threads "$CPU_THREADS" "$@"
