#!/usr/bin/env bash
# Only four low-error Joint Attention layers are replaced; all others stay native.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export GPU_IDS="${GPU_IDS:-0,1,2,3}" CPU_THREADS="${CPU_THREADS:-1}"
exec bash "$REPO/flux3/scripts/train_flux3_qdm_repeat.sh" \
  --stage a1 --qdm-layers 4,24,26,27 --region-size 8 --halo 2 \
  --batch-size 2 --grad-accum 1 --workers 4 --threads "$CPU_THREADS" \
  --save-steps 100 --validation-steps 100 --cycle-seconds 1200 \
  --output-root "${FLUX3_LOW4_RUN:-outputs/flux3-qdm-lowmse4-halo2-node01}" \
  "$@"
