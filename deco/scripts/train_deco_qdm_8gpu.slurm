#!/usr/bin/env bash
#SBATCH --job-name=deco-qdm-a1a2b-bs32-acc2
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/deco-qdm/slurm/%x-%j.out
set -euo pipefail
cd "${DECO_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${DECO_CONDA_BASE:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate deco-qdm
export DECO_RUN="${DECO_RUN:-$HOME/code/PiD-Linear/outputs/deco-qdm}"
export PYTHONPATH="$PWD/deco${PYTHONPATH:+:$PYTHONPATH}"
export NNODES=2 NPROC_PER_NODE=4 CPU_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
mkdir -p "$DECO_RUN/repeat"
stop_token="deco-slurm:$SLURM_JOB_ID"
request_stop() { (set -o noclobber; printf '%s' "$stop_token" > "$DECO_RUN/repeat/STOP") 2>/dev/null || true; }
cleanup_stop() {
  if [[ -f "$DECO_RUN/repeat/STOP" ]] && [[ "$(cat "$DECO_RUN/repeat/STOP")" == "$stop_token" ]]; then
    rm -f "$DECO_RUN/repeat/STOP"
  fi
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_PROCID"
  export GPU_IDS="${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation missing}"
  exec bash deco/scripts/train_deco_qdm.sh "$@"
' bash --stage all --resume auto --allow-batch-change --allow-schedule-change --batch-size 32 --grad-accum 2 \
  --a1-steps 1000 --a1-save-every 500 --a1-validate-every 500 \
  --a2-steps 2000 --a2-save-every 1000 --a2-validate-every 1000 \
  --b-steps 20000 --b-save-every 2000 --b-validate-every 2000 \
  --validation-student-only --workers 4 --threads 1 \
  --max-seconds 516600 --swanlab-mode offline --output-root "$DECO_RUN" &
launcher_pid=$!
while true; do
  set +e
  wait "$launcher_pid"
  code=$?
  set -e
  if ! kill -0 "$launcher_pid" 2>/dev/null; then exit "$code"; fi
done
