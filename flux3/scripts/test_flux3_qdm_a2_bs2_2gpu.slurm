#!/usr/bin/env bash
#SBATCH --job-name=flux3-qdm-a2-bs2-test
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/flux3-qdm-all28-a1a2-bs4-acc4/slurm/%x-%j.out
set -euo pipefail
cd "${FLUX3_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${FLUX3_CONDA_BASE:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate flux3-qdm
export PYTHONPATH="$PWD/flux3:$PWD/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export FLUX3_PARENT_RUN="${FLUX3_PARENT_RUN:-$PWD/outputs/flux3-qdm-all28-a1a2-bs4-acc4}"
export FLUX3_TEST_RUN="${FLUX3_TEST_RUN:-$FLUX3_PARENT_RUN/preflight/a2-bs2-2gpu-$SLURM_JOB_ID}"
export FLUX3_WEIGHTS_ROOT="${FLUX3_WEIGHTS_ROOT:-$HOME/huggingface_ckpts}"
export FLUX3_SUBSET_ROOT="${FLUX3_SUBSET_ROOT:-$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets}"
export FLUX3_CACHE_ROOT="${FLUX3_CACHE_ROOT:-$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=True DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset GPU_IDS
if [[ -z "${FLUX3_A1_CHECKPOINT:-}" ]]; then
  export FLUX3_A1_CHECKPOINT="$(python - <<'PY'
import json,os
from pathlib import Path
print(json.loads((Path(os.environ['FLUX3_PARENT_RUN'])/'a1/checkpoints/latest.json').read_text())['checkpoint'])
PY
)"
fi
printf 'A2 bs2 test job=%s node=%s parent=%s output=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$FLUX3_A1_CHECKPOINT" "$FLUX3_TEST_RUN"
exec srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
  torchrun --standalone --nproc_per_node=2 --module flux3_qdm.training \
  --stage a2 --steps 20 --init-from "$FLUX3_A1_CHECKPOINT" \
  --batch-size 2 --grad-accum 1 --region-size 8 --caption-dropout 0.1 \
  --save-steps 10 --validation-steps 0 --skip-validation --resume auto \
  --workers 4 --threads 1 --swanlab-mode disabled --log-steps 1 \
  --output-root "$FLUX3_TEST_RUN" --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT"
