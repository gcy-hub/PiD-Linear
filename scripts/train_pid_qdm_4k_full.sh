#!/usr/bin/env bash
# Four-GPU, full-index 4K A1 -> A2 -> B with renewable node01 processes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CPU_THREADS="${CPU_THREADS:-1}"
RUN_ROOT="${PID_QDM_RUN:-outputs/pid-qdm-halo2-4k-full}"
QDM_BATCH="${PID_QDM_BATCH:-2}"
QDM_ACCUM="${PID_QDM_ACCUM:-4}"
IFS=',' read -ra QDM_GPUS <<< "$GPU_IDS"
QDM_WORLD=$(( ${NNODES:-1} * ${NPROC_PER_NODE:-${#QDM_GPUS[@]}} ))
QDM_BATCH_ARGS=()
if (( QDM_ACCUM > 0 )); then
  QDM_BATCH_ARGS=(--grad-accum "$QDM_ACCUM" --effective-batch "$((QDM_WORLD * QDM_BATCH * QDM_ACCUM))" --allow-batch-change)
fi
exec bash scripts/train_pid_vdn_repeat.sh \
  --stage all --cycle-seconds "${PID_QDM_CYCLE_SECONDS:-600}" --grace-seconds "${PID_QDM_GRACE_SECONDS:-1100}" \
  --output-root "$RUN_ROOT" \
  --weights-root "${PID_WEIGHTS_ROOT:-weights}" \
  --index-root "${PID_INDEX_ROOT:-dataset/MultiAspect-4K-1M/linear_pid_index}" \
  --text-cache-root "${PID_TEXT_CACHE_ROOT:-dataset/MultiAspect-4K-1M/linear_pid_text_cache}" \
  --training-data full --resolution 4096 --allow-upscale \
  --memory-mode fixed4 --local-mode halo --local-halo 2 --region-size 8 \
  --batch-size "$QDM_BATCH" "${QDM_BATCH_ARGS[@]}" --workers 2 --threads "$CPU_THREADS" \
  --attention-batch-chunk "${PID_QDM_ATTN_BATCH_CHUNK:-1}" \
  --gallery-root "${PID_GALLERY_ROOT:-$RUN_ROOT/assets}" --validation-resolution 4096 \
  --validation-cases generated_010,generated_029,generated_003,generated_009,generated_019,generated_022,real_011,real_028 \
  --validation-steps 500 --log-steps 1 --swanlab-mode offline "$@"
