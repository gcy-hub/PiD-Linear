#!/usr/bin/env bash
# All 28 joint blocks: image Local Softmax + Halo2, no QDM allocation.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export GPU_IDS="${GPU_IDS:-0,1,2,3}" CPU_THREADS="${CPU_THREADS:-1}"
exec bash "$REPO/flux3/scripts/train_flux3_qdm_repeat.sh" \
  --stage b --from-base --attention-mode local-only --training-subset-size 20000 --steps 500 \
  --qdm-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --region-size 8 --halo 2 --batch-size 2 --grad-accum 1 \
  --workers 4 --threads "$CPU_THREADS" \
  --save-steps 100 --validation-steps 100 --cycle-seconds 1200 \
  --output-root "${FLUX3_ALL_LOCAL_RUN:-outputs/flux3-local-all28-halo2-direct-b-node01}" \
  "$@"
