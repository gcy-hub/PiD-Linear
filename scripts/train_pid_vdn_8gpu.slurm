#!/usr/bin/env bash
#SBATCH --job-name=pid-vdn-a2-8gpu
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/pid-vdn/slurm/%x-%j.out
set -euo pipefail

export PID_VDN_REPO_ROOT="${PID_VDN_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the PiD-VDN worktree}}"
PID_VDN_CONDA_ROOT="${PID_VDN_CONDA_ROOT:-$(dirname "$(dirname "${CONDA_EXE:?Activate Conda or set PID_VDN_CONDA_ROOT}" )")}"
source "$PID_VDN_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${PID_VDN_ENV:-linear-pid}"
cd "$PID_VDN_REPO_ROOT"
export PID_VDN_RUN="${PID_VDN_RUN:-$PID_VDN_REPO_ROOT/outputs/pid-vdn}"
mkdir -p "$PID_VDN_RUN/a2" "$PID_VDN_RUN/slurm"
# Slurm defines visibility independently on each node.
unset GPU_IDS
export NNODES="$SLURM_JOB_NUM_NODES" NPROC_PER_NODE=4
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export PYTHONPATH="$PID_VDN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Recover the stopped stage's resource locations; explicit environment values win.
mapfile -d '' -t PID_VDN_SAVED_ARGS < <(python - <<'PY'
import json, os, sys
from pathlib import Path
root = Path(os.environ['PID_VDN_RUN'])
saved = {}
for stage in ('a2', 'a1'):
    path = root / stage / 'config.json'
    if path.exists():
        saved = json.loads(path.read_text())
        break
for name, variable, default in (
    ('weights_root', 'PID_WEIGHTS_ROOT', 'weights'),
    ('index_root', 'PID_INDEX_ROOT', 'dataset/MultiAspect-4K-1M/linear_pid_index'),
    ('text_cache_root', 'PID_TEXT_CACHE_ROOT', ''),
    ('subset_root', 'PID_VDN_SUBSET_ROOT', 'dataset/MultiAspect-4K-1M/pid_vdn_subsets'),
    ('gallery_root', 'PID_GALLERY_ROOT', 'outputs/linear-pid/assets'),
    ('region_size', 'PID_VDN_REGION_SIZE', 8),
    ('seed', 'PID_VDN_SEED', 42),
    ('lora_rank', 'PID_VDN_LORA_RANK', 64),
    ('lora_alpha', 'PID_VDN_LORA_ALPHA', 64),
    ('pit_chunk_size', 'PID_VDN_PIT_CHUNK_SIZE', 2048),
):
    value = os.environ.get(variable, saved.get(name, default))
    sys.stdout.write('--' + name.replace('_', '-') + '\0' + str(value) + '\0')
PY
)
# Request a boundary save; leave user-created stop markers untouched.
PID_VDN_STOP_TOKEN="pid-vdn-slurm:${SLURM_JOB_ID}"
request_stop() {
  (set -o noclobber; printf '%s' "$PID_VDN_STOP_TOKEN" > "$PID_VDN_RUN/a2/STOP") 2>/dev/null || true
}
cleanup_stop() {
  if [[ -f "$PID_VDN_RUN/a2/STOP" ]] && [[ "$(cat "$PID_VDN_RUN/a2/STOP")" == "$PID_VDN_STOP_TOKEN" ]]; then
    rm -f "$PID_VDN_RUN/a2/STOP"
  fi
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_PROCID"
  exec bash "$PID_VDN_REPO_ROOT/scripts/train_pid_vdn.sh" \
    --preset 8gpu --stage a2 --resume auto \
    --batch-size 4 --grad-accum 1 --effective-batch 32 \
    --max-steps "${PID_VDN_STEPS:-2000}" \
    --save-steps "${PID_VDN_SAVE_STEPS:-500}" \
    --validation-steps "${PID_VDN_VALIDATION_STEPS:-1000}" \
    --allow-schedule-change --max-seconds 0 \
    --workers "${PID_VDN_WORKERS:-4}" --threads "$CPU_THREADS" \
    --swanlab-mode "${PID_VDN_SWANLAB_MODE:-offline}" \
    --output-root "$PID_VDN_RUN" "$@"
' bash "${PID_VDN_SAVED_ARGS[@]}" "$@" &
PID_VDN_SRUN_PID=$!
while true; do
  set +e
  wait "$PID_VDN_SRUN_PID"
  PID_VDN_EXIT_CODE=$?
  set -e
  if ! kill -0 "$PID_VDN_SRUN_PID" 2>/dev/null; then
    exit "$PID_VDN_EXIT_CODE"
  fi
done
