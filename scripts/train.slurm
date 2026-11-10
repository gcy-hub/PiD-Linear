#!/usr/bin/env bash
#SBATCH --job-name=pid-adaptation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --signal=B:USR1@1800
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${RUN_OUTPUT:?Set RUN_OUTPUT to the experiment directory}"
export RUN_OUTPUT NNODES="$SLURM_JOB_NUM_NODES"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29583}"
unset GPU_IDS
TOKEN="slurm:${SLURM_JOB_ID}"
stop_run() {
  for stage in a1 a2 b; do
    mkdir -p "$RUN_OUTPUT/$stage"
    (set -o noclobber; printf '%s' "$TOKEN" > "$RUN_OUTPUT/$stage/STOP") 2>/dev/null || true
  done
}
cleanup() {
  for stage in a1 a2 b; do
    if [[ -f "$RUN_OUTPUT/$stage/STOP" ]] && [[ "$(cat "$RUN_OUTPUT/$stage/STOP")" == "$TOKEN" ]]; then
      rm "$RUN_OUTPUT/$stage/STOP"
    fi
  done
}
trap stop_run USR1 TERM INT
trap cleanup EXIT
srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_PROCID"
  exec bash scripts/train_adapt.sh "$@" --output-root "$RUN_OUTPUT"
' bash "$@" &
child=$!
while true; do
  code=0
  wait "$child" || code=$?
  kill -0 "$child" 2>/dev/null || exit "$code"
done
