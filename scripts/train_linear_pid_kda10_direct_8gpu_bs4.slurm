#!/usr/bin/env bash
#SBATCH --job-name=linear-pid-kda10-direct-bs4
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4/slurm-%j.out
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the PiD-Linear root}}"
source /home/ganchangyi/software/miniconda3/etc/profile.d/conda.sh
conda activate "${LINEAR_PID_ENV:-linear-pid}"
cd "$REPO_ROOT"
# GPU visibility is supplied independently to each node by Slurm.
unset GPU_IDS
export NNODES=2 NPROC_PER_NODE=4
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLA_DISABLE_TENSOR_CACHE=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export REPO_ROOT
export LINEAR_PID_STAGE_CONFIG="$REPO_ROOT/pid/_src/configs/linear_pid/kda10_direct_8gpu_bs4.json"
export LINEAR_PID_STAGE_OUTPUT_ROOT="${LINEAR_PID_STAGE_OUTPUT_ROOT:-/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4}"
mkdir -p "$LINEAR_PID_STAGE_OUTPUT_ROOT"
export LINEAR_PID_STOP_FILE="$LINEAR_PID_STAGE_OUTPUT_ROOT/slurm-${SLURM_JOB_ID}.stop"
rm -f "$LINEAR_PID_STOP_FILE"
trap 'touch "$LINEAR_PID_STOP_FILE"' USR1
# This configuration has no parent run. Repeated submission resumes only its own checkpoints.
python scripts/launch_linear_pid_stage.py --stage-config "$LINEAR_PID_STAGE_CONFIG" \
  --preset 8gpu --output-root "$LINEAR_PID_STAGE_OUTPUT_ROOT" --prepare-only
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_PROCID"
  exec python "$REPO_ROOT/scripts/launch_linear_pid_stage.py" \
    --stage-config "$LINEAR_PID_STAGE_CONFIG" --preset 8gpu \
    --output-root "$LINEAR_PID_STAGE_OUTPUT_ROOT"
' &
LINEAR_SRUN_PID=$!
while true; do
  set +e
  wait "$LINEAR_SRUN_PID"
  LINEAR_EXIT_CODE=$?
  set -e
  if ! kill -0 "$LINEAR_SRUN_PID" 2>/dev/null; then
    exit "$LINEAR_EXIT_CODE"
  fi
done
