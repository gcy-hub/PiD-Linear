#!/usr/bin/env bash
#SBATCH --job-name=pid-qdm-4k-bs2-acc4
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail

export PID_VDN_REPO_ROOT="${PID_VDN_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the PiD-QDM worktree}}"
source "${PID_VDN_CONDA_ROOT:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate "${PID_VDN_ENV:-linear-pid}"
cd "$PID_VDN_REPO_ROOT"
: "${PID_WEIGHTS_ROOT:?Set local model weights root}"
: "${PID_INDEX_ROOT:?Set full training index root}"
: "${PID_TEXT_CACHE_ROOT:?Set full Gemma text cache root}"
: "${PID_GALLERY_ROOT:?Set the selected eight 4K gallery assets root}"
export PID_QDM_RUN="${PID_QDM_RUN:-$PID_VDN_REPO_ROOT/outputs/pid-qdm-halo2-4k-full-2node-8gpu-bs2-acc4}"
mkdir -p "$PID_QDM_RUN" "$PID_QDM_RUN/slurm"
unset GPU_IDS
export NNODES="$SLURM_JOB_NUM_NODES" NPROC_PER_NODE=4
export PID_QDM_BATCH="${PID_QDM_BATCH:-2}" PID_QDM_ACCUM="${PID_QDM_ACCUM:-4}"
export PID_QDM_EFFECTIVE=$((NNODES * NPROC_PER_NODE * PID_QDM_BATCH * PID_QDM_ACCUM))
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT="${MASTER_PORT:-$((22000 + SLURM_JOB_ID % 15000))}"
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PID_VDN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

STOP_TOKEN="pid-qdm-4k-slurm:${SLURM_JOB_ID}"
request_stop() {
  for stage in a1 a2 b; do
    mkdir -p "$PID_QDM_RUN/$stage"
    (set -o noclobber; printf '%s' "$STOP_TOKEN" > "$PID_QDM_RUN/$stage/STOP") 2>/dev/null || true
  done
}
cleanup_stop() {
  for stage in a1 a2 b; do
    if [[ -f "$PID_QDM_RUN/$stage/STOP" ]] && [[ "$(cat "$PID_QDM_RUN/$stage/STOP")" == "$STOP_TOKEN" ]]; then
      rm -f "$PID_QDM_RUN/$stage/STOP"
    fi
  done
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --gpu-bind=none --cpus-per-task=16 --kill-on-bad-exit=1 bash -c '
    export NODE_RANK="$SLURM_PROCID"
    exec bash "$PID_VDN_REPO_ROOT/scripts/train_pid_vdn.sh" \
      --preset 8gpu --stage all --resume auto \
      --weights-root "$PID_WEIGHTS_ROOT" --index-root "$PID_INDEX_ROOT" \
      --text-cache-root "$PID_TEXT_CACHE_ROOT" --gallery-root "$PID_GALLERY_ROOT" \
      --training-data full --resolution 4096 --allow-upscale \
      --memory-mode fixed4 --local-mode halo --local-halo 2 --region-size 8 \
      --batch-size "$PID_QDM_BATCH" --grad-accum "$PID_QDM_ACCUM" --effective-batch "$PID_QDM_EFFECTIVE" \
      --attention-batch-chunk 1 \
      --validation-resolution 4096 \
      --validation-cases generated_010,generated_029,generated_003,generated_009,generated_019,generated_022,real_011,real_028 \
      --validation-steps 500 --workers 2 --threads "$CPU_THREADS" \
      --log-steps 1 --swanlab-mode offline --max-seconds 0 --output-root "$PID_QDM_RUN"
  ' &
SRUN_PID=$!
while true; do
  set +e
  wait "$SRUN_PID"
  EXIT_CODE=$?
  set -e
  if ! kill -0 "$SRUN_PID" 2>/dev/null; then
    exit "$EXIT_CODE"
  fi
done
