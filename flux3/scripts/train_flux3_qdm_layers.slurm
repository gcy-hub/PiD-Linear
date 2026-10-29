#!/usr/bin/env bash
#SBATCH --job-name=flux3-qdm-layers
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/flux3-qdm-layer-sweep-20261020/slurm/%x-%j.out
set -euo pipefail
cd "${FLUX3_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${FLUX3_CONDA_BASE:-$HOME/software/miniconda3}/etc/profile.d/conda.sh"
conda activate flux3-qdm
export PYTHONPATH="$PWD/flux3:$PWD/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=True DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export PYTORCH_ALLOC_CONF=expandable_segments:True
count="${QDM_LAYER_COUNT:?Set QDM_LAYER_COUNT to 4, 8, 14 or 20}"
layers="$(python - "$count" <<'PY'
import sys
n=int(sys.argv[1])
if n not in (4,8,14,20):raise ValueError('Expected 4, 8, 14 or 20 layers')
print(','.join(str(round(i*27/(n-1))) for i in range(n)))
PY
)"
run="${FLUX3_QDM_RUN:-$PWD/outputs/flux3-qdm-layer-sweep-20261020/layers-$count}"
mkdir -p "$run/a1"
token="flux3-layers:$SLURM_JOB_ID"
request_stop() { (set -o noclobber; printf '%s' "$token" > "$run/a1/STOP") 2>/dev/null || true; }
cleanup_stop() {
  if [[ -f "$run/a1/STOP" ]] && [[ "$(cat "$run/a1/STOP")" == "$token" ]]; then
    rm -f "$run/a1/STOP"
  fi
}
trap request_stop USR1 TERM INT
trap cleanup_stop EXIT
printf 'QDM layers=%s node=%s allocated_gpus=%s run=%s\n' "$layers" "$SLURMD_NODENAME" "${CUDA_VISIBLE_DEVICES:-unset}" "$run"
srun --ntasks=1 --kill-on-bad-exit=1 bash -c '
  export GPU_IDS="${CUDA_VISIBLE_DEVICES:?Slurm must assign the GPUs}"
  exec bash flux3/scripts/train_flux3_qdm.sh "$@"
' bash --stage a1 --qdm-layers "$layers" --region-size 8 \
  --batch-size 4 --grad-accum 1 --steps 500 --save-steps 50 \
  --caption-dropout 0.1 --workers 4 --threads "$CPU_THREADS" \
  --resume auto --swanlab-mode offline --max-seconds 486000 \
  --output-root "$run" &
child=$!
while true; do
  set +e
  wait "$child"
  code=$?
  set -e
  if ! kill -0 "$child" 2>/dev/null; then exit "$code"; fi
done
