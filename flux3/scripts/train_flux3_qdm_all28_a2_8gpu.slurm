#!/usr/bin/env bash
#SBATCH --job-name=flux3-qdm-all28-a2
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/flux3-qdm-all28-a1a2-bs4-acc4/slurm/%x-%j.out
set -euo pipefail
cd "${FLUX3_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${FLUX3_CONDA_BASE:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate flux3-qdm
export PYTHONPATH="$PWD/flux3:$PWD/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export FLUX3_QDM_RUN="${FLUX3_QDM_RUN:-$PWD/outputs/flux3-qdm-all28-a1a2-bs4-acc4}"
export FLUX3_WEIGHTS_ROOT="${FLUX3_WEIGHTS_ROOT:-$HOME/huggingface_ckpts}"
export FLUX3_SUBSET_ROOT="${FLUX3_SUBSET_ROOT:-$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets}"
export FLUX3_CACHE_ROOT="${FLUX3_CACHE_ROOT:-$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=True DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLUX3_SLURM_START_ATTEMPTS="${FLUX3_SLURM_START_ATTEMPTS:-3}"
export FLUX3_SLURM_RETRY_SECONDS="${FLUX3_SLURM_RETRY_SECONDS:-10}"
unset GPU_IDS
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
mkdir -p "$FLUX3_QDM_RUN/a2"
if [[ -z "${FLUX3_A1_CHECKPOINT:-}" ]]; then
  export FLUX3_A1_CHECKPOINT="$(python - <<'PYCHECKPOINT'
import json,os
from pathlib import Path
p=Path(os.environ['FLUX3_QDM_RUN'])/'a1/checkpoints/latest.json'
print(json.loads(p.read_text())['checkpoint'])
PYCHECKPOINT
)"
fi
printf 'A2 parent checkpoint: %s\n' "$FLUX3_A1_CHECKPOINT"
printf 'FLUX3 Slurm allocation job=%s nodes=%s master=%s port=%s startup_attempts=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$MASTER_ADDR" "$MASTER_PORT" "$FLUX3_SLURM_START_ATTEMPTS"
token="flux3-all28-a2:$SLURM_JOB_ID"
request_stop() {
  for stage in a2; do
    (set -o noclobber; printf '%s' "$token" > "$FLUX3_QDM_RUN/$stage/STOP") 2>/dev/null || true
  done
}
cleanup_stop() {
  for stage in a2; do
    if [[ -f "$FLUX3_QDM_RUN/$stage/STOP" ]] && [[ "$(cat "$FLUX3_QDM_RUN/$stage/STOP")" == "$token" ]]; then
      rm -f "$FLUX3_QDM_RUN/$stage/STOP"
    fi
  done
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
python -m flux3_qdm.slurm_stages --stages a2 --stage a2 --steps 4000 --init-from "$FLUX3_A1_CHECKPOINT" \
  --qdm-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --batch-size 2 --grad-accum 1 --region-size 8 --caption-dropout 0.1 \
  --save-steps 500 --validation-steps 500 --resume auto \
  --workers 4 --threads 1 --swanlab-mode offline \
  --output-root "$FLUX3_QDM_RUN" --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" &
child=$!
while true; do
  set +e
  wait "$child"
  code=$?
  set -e
  if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
