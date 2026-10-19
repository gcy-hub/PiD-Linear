#!/usr/bin/env bash
#SBATCH --job-name=qdm-halo2-4gpu
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/pid-qdm-halo2/slurm/%x-%j.out
set -euo pipefail
REPO_ROOT="${PID_QDM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source "${PID_QDM_CONDA_ROOT:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate "${PID_QDM_ENV:-linear-pid}"
export PID_QDM_RUN="${PID_QDM_RUN:-$REPO_ROOT/outputs/pid-qdm-halo2}"
export PID_QDM_LOCAL_MODE=halo
export PID_WEIGHTS_ROOT="${PID_WEIGHTS_ROOT:-$HOME/huggingface_ckpts}"
export PID_INDEX_ROOT="${PID_INDEX_ROOT:-$HOME/dataset/MultiAspect-4K-1M/linear_pid_index}"
export PID_VDN_SUBSET_ROOT="${PID_VDN_SUBSET_ROOT:-$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets}"
export PID_TEXT_CACHE_ROOT="${PID_TEXT_CACHE_ROOT:-$HOME/dataset/MultiAspect-4K-1M/linear_pid_text_cache}"
export PID_GALLERY_ROOT="${PID_GALLERY_ROOT:-$HOME/code/PiD-Linear/outputs/linear-pid/assets}"
export NNODES=1 NPROC_PER_NODE=4 CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset GPU_IDS
mkdir -p "$PID_QDM_RUN/slurm"
QDM_STOP_TOKEN="qdm-slurm:${SLURM_JOB_ID}"
request_stop() {
  # Stop all stages so a boundary signal cannot accidentally advance the pipeline.
  for stage in a1 a2 b; do
    mkdir -p "$PID_QDM_RUN/$stage"
    (set -o noclobber; printf '%s' "$QDM_STOP_TOKEN" > "$PID_QDM_RUN/$stage/STOP") 2>/dev/null || true
  done
}
cleanup_stop() {
  for stage in a1 a2 b; do
    if [[ -f "$PID_QDM_RUN/$stage/STOP" ]] && [[ "$(cat "$PID_QDM_RUN/$stage/STOP")" == "$QDM_STOP_TOKEN" ]]; then
      rm -f "$PID_QDM_RUN/$stage/STOP"
    fi
  done
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash "$REPO_ROOT/scripts/train_qdm_stages.sh" "$@" &
QDM_SRUN_PID=$!
while true; do
  set +e
  wait "$QDM_SRUN_PID"
  QDM_EXIT=$?
  set -e
  if ! kill -0 "$QDM_SRUN_PID" 2>/dev/null; then exit "$QDM_EXIT"; fi
done
