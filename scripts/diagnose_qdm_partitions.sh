#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="${CPU_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PID_DIAGNOSTIC_PYTHON:-python}"
ROOT="${PID_PARTITION_OUTPUT:-outputs/research/qdm-partition-diagnosis}"
mkdir -p "$ROOT/logs"
IFS=, read -ra GPUS <<< "${GPU_IDS:-0,1,2,3}"
# Independent workers; no DDP, no modifications to existing training/generation jobs.
# One mode/model/case per command; completed outputs and teacher trajectories are reused.
COMMON=(--weights "${PID_WEIGHTS_ROOT:-weights}" --assets "${PID_GALLERY_ROOT:-outputs/linear-pid/assets}"
        --text-cache "${PID_TEXT_CACHE_ROOT:-dataset/MultiAspect-4K-1M/linear_pid_text_cache}"
        --output "$ROOT" --threads "${CPU_THREADS:-1}")
CASES=(generated_009_2048 generated_029_2048 real_028_2048 generated_010_2048)
if [[ -n "${PID_DIAGNOSTIC_CASES:-}" ]]; then read -ra CASES <<< "$PID_DIAGNOSTIC_CASES"; fi
worker() {
    local slot="$1" gpu="$2" model mode label checkpoint
    case "$slot" in
      0) model=pid; mode=mask; label=;;
      1) model=pixeldit; mode=mask; label=;;
      2) model=pid; mode=boundary; label=halo-a2;
         checkpoint="${PID_QDM_CHECKPOINT:?Set PID_QDM_CHECKPOINT}";;
      3) model=pixeldit; mode=boundary; label=halo-rope-a1;
         checkpoint="${PIXELDIT_QDM_CHECKPOINT:?Set PIXELDIT_QDM_CHECKPOINT}";;
      4) model=pid; mode=control; label=;;
      5) model=pixeldit; mode=control; label=;;
    esac
    local extra=()
    [[ "$mode" != boundary ]] || extra=(--checkpoint "$checkpoint" --label "$label")
    local items=("${CASES[@]}")
    [[ "$mode" != control ]] || items=(generated_009_2048)
    for item in "${items[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m pid._src.pixeldit_qdm.run_partition_diagnostics \
          --mode "$mode" --model "$model" --case "$item" "${COMMON[@]}" "${extra[@]}"
    done
}
IFS=, read -ra SLOTS <<< "${PID_DIAGNOSTIC_SLOTS:-0,1,2,3,4,5}"
pids=()
# Distribute the four logical queues over the selected GPUs; sequential if fewer than four.
for ((g=0; g<${#GPUS[@]}; g++)); do
    (
      for ((slot=g; slot<${#SLOTS[@]}; slot+=${#GPUS[@]})); do worker "${SLOTS[slot]}" "${GPUS[g]}"; done
    ) >>"$ROOT/logs/worker_slots_${PID_DIAGNOSTIC_SLOTS:-all}_gpu_${GPUS[g]}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
