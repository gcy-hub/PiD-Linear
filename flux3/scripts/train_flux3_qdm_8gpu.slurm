#!/usr/bin/env bash
#SBATCH --job-name=flux3-qdm-a1-8gpu
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/flux3-qdm-a1-image-kd-cfg/slurm/%x-%j.out
set -euo pipefail
FLUX3_PROJECT_ROOT="${FLUX3_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$FLUX3_PROJECT_ROOT"
source "${FLUX3_CONDA_BASE:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate flux3-qdm
export PYTHONPATH="$FLUX3_PROJECT_ROOT/flux3:$FLUX3_PROJECT_ROOT/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export FLUX3_QDM_RUN="${FLUX3_QDM_RUN:-$FLUX3_PROJECT_ROOT/outputs/flux3-qdm-a1-image-kd-cfg}"
export FLUX3_WEIGHTS_ROOT="${FLUX3_WEIGHTS_ROOT:-$HOME/huggingface_ckpts}"
export FLUX3_SUBSET_ROOT="${FLUX3_SUBSET_ROOT:-$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets}"
export FLUX3_CACHE_ROOT="${FLUX3_CACHE_ROOT:-$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=True DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset GPU_IDS
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
mkdir -p "$FLUX3_QDM_RUN/a1"
FLUX3_STOP_TOKEN="flux3-slurm:$SLURM_JOB_ID"
request_stop() {
  (set -o noclobber; printf '%s' "$FLUX3_STOP_TOKEN" > "$FLUX3_QDM_RUN/a1/STOP") 2>/dev/null || true
}
cleanup_stop() {
  if [[ -f "$FLUX3_QDM_RUN/a1/STOP" ]] && [[ "$(cat "$FLUX3_QDM_RUN/a1/STOP")" == "$FLUX3_STOP_TOKEN" ]]; then
    rm -f "$FLUX3_QDM_RUN/a1/STOP"
  fi
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
wait_step() {
  local child_pid="$1" child_code
  while true; do
    set +e
    wait "$child_pid"
    child_code=$?
    set -e
    if ! kill -0 "$child_pid" 2>/dev/null; then return "$child_code"; fi
  done
}
common=(--stage a1 --batch-size 4 --grad-accum 1 --caption-dropout 0.1
        --region-size 8 --steps 500 --save-steps 50 --resume auto
        --workers 4 --threads 1 --swanlab-mode offline
        --output-root "$FLUX3_QDM_RUN" --weights-root "$FLUX3_WEIGHTS_ROOT"
        --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT")
# One torchrun agent per node; all eight ranks participate in A1 gradient averaging.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  exec torchrun --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --module flux3_qdm.training "$@"
' bash "${common[@]}" --max-seconds 486000 &
FLUX3_SRUN_PID=$!
wait_step "$FLUX3_SRUN_PID"
if [[ -f "$FLUX3_QDM_RUN/a1/STOP" ]]; then exit 0; fi
# Training ranks have exited. Validate on one node's four allocated GPUs.
# A1 finishes here; A2/B are never started by this script.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  exec python -m flux3_qdm.launcher "$@"
' bash "${common[@]}" --max-seconds 450000 &
FLUX3_SRUN_PID=$!
wait_step "$FLUX3_SRUN_PID"
