#!/usr/bin/env bash
#SBATCH --job-name=linear-pid-kda4
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-slurm/slurm-%j.out
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the PiD-Linear root}}"
source /home/ganchangyi/software/miniconda3/etc/profile.d/conda.sh
conda activate "${LINEAR_PID_ENV:-linear-pid}"
cd "$REPO_ROOT"
export CPU_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
# Respect the GPU allocation supplied to each srun task by Slurm.
unset GPU_IDS
export LINEAR_PID_OUTPUT_ROOT="${LINEAR_PID_OUTPUT_ROOT:-/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-slurm/job-${SLURM_JOB_ID}}"
mkdir -p "$LINEAR_PID_OUTPUT_ROOT"
if [[ -n "${LINEAR_PID_SOURCE_RUN_DIR:-}" ]]; then
  LINEAR_PID_RESUME_FROM="$(python scripts/prepare_linear_pid_resume.py \
    --source-run-dir "$LINEAR_PID_SOURCE_RUN_DIR" --output-root "$LINEAR_PID_OUTPUT_ROOT")"
fi
export LINEAR_PID_RESUME_FROM="${LINEAR_PID_RESUME_FROM:-}"
export LINEAR_PID_SAVE_STEPS="${LINEAR_PID_SAVE_STEPS:-5000}"
export LINEAR_PID_SAVE_EPOCHS="${LINEAR_PID_SAVE_EPOCHS:-1}"
export LINEAR_PID_VALIDATION_STEPS="${LINEAR_PID_VALIDATION_STEPS:-5000}"
export LINEAR_PID_VALIDATION_EPOCHS="${LINEAR_PID_VALIDATION_EPOCHS:-1}"
export NNODES=2 NPROC_PER_NODE=4
export LINEAR_PID_BATCH_SIZE="${LINEAR_PID_BATCH_SIZE:-4}"
export LINEAR_PID_GRAD_ACCUM="${LINEAR_PID_GRAD_ACCUM:-1}"
export LINEAR_PID_EFFECTIVE_BATCH="${LINEAR_PID_EFFECTIVE_BATCH:-$((NNODES * NPROC_PER_NODE * LINEAR_PID_BATCH_SIZE * LINEAR_PID_GRAD_ACCUM))}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export REPO_ROOT
# A shared stop file reaches every worker without terminating torchrun's agent.
export LINEAR_PID_STOP_FILE="${LINEAR_PID_STOP_FILE:-$LINEAR_PID_OUTPUT_ROOT/slurm-${SLURM_JOB_ID}.stop}"
mkdir -p "$(dirname "$LINEAR_PID_STOP_FILE")"
rm -f "$LINEAR_PID_STOP_FILE"
trap 'touch "$LINEAR_PID_STOP_FILE"' USR1
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_PROCID"
  LINEAR_PID_RESUME_ARGS=()
  if [[ -n "$LINEAR_PID_RESUME_FROM" ]]; then
    LINEAR_PID_RESUME_ARGS+=(--allow-batch-size-change)
  fi
  exec bash "$REPO_ROOT/scripts/train_linear_pid.sh" \
    --preset 8gpu --layers "${KDA_LAYERS:-4}" \
    --batch-size "$LINEAR_PID_BATCH_SIZE" --grad-accum "$LINEAR_PID_GRAD_ACCUM" \
    --effective-batch "$LINEAR_PID_EFFECTIVE_BATCH" \
    --workers "${LINEAR_PID_WORKERS:-4}" --threads "$CPU_THREADS" \
    --pit-chunk-size "${LINEAR_PID_PIT_CHUNK_SIZE:-2048}" --lambda-out 0 \
    --output-root "$LINEAR_PID_OUTPUT_ROOT" \
    --resume auto --resume-from "$LINEAR_PID_RESUME_FROM" \
    --save-steps "$LINEAR_PID_SAVE_STEPS" --save-epochs "$LINEAR_PID_SAVE_EPOCHS" \
    --validation-steps "$LINEAR_PID_VALIDATION_STEPS" --validation-epochs "$LINEAR_PID_VALIDATION_EPOCHS" \
    --max-seconds 516600 --log-steps 10 "${LINEAR_PID_RESUME_ARGS[@]}" "$@"
' bash "$@" &
LINEAR_SRUN_PID=$!
while true; do
  set +e
  wait "$LINEAR_SRUN_PID"
  LINEAR_EXIT_CODE=$?
  set -e
  if ! kill -0 "$LINEAR_SRUN_PID" 2>/dev/null; then
    exit "$LINEAR_EXIT_CODE"
  fi
  # wait can be interrupted by USR1; keep the allocation until workers save.
done
