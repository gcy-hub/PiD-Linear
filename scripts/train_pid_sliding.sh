#!/usr/bin/env bash
# Same PiD/QDM training recipe; only the Local neighborhood changes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=',' read -ra gpu_list <<< "${GPU_IDS:-1,2,3}"
export GPU_IDS="${GPU_IDS:-1,2,3}"
effective_batch=$(( ${#gpu_list[@]} * ${NNODES:-1} ))
exec bash "$repo/scripts/train_pid_vdn_repeat.sh" \
  --stage a1 --preset 4gpu --memory-mode fixed4 \
  --local-mode sliding --sliding-kernel-size "${SLIDING_KERNEL_SIZE:-11}" \
  --sliding-backend natten --sliding-query-chunk 128 --no-remote-rope \
  --training-data full --resolution 3840 --resolution-policy native --native-max-resolution 3840 \
  --context-parallel-size 1 --linalg-backend cusolver \
  --batch-size 1 --grad-accum 1 --effective-batch "$effective_batch" \
  --max-steps 500 --save-steps 25 --validation-steps 250 --validation-resolution 4096 \
  --validation-cases generated_010,generated_029,generated_003,generated_009,generated_019,generated_022,real_011,real_028,generated_000,real_000 \
  --workers 4 --threads 1 --attention-batch-chunk 0 --pit-chunk-size 2048 \
  --resume auto --log-steps 1 --swanlab-mode offline \
  --output-root outputs/pid-qdm-sliding11-4k \
  --cycle-seconds 1200 --grace-seconds 180 "$@"
