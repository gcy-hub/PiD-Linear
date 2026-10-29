#!/usr/bin/env bash
# Nine image-attention layers use only QDM; text keeps native global attention.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export GPU_IDS="${GPU_IDS:-1,2,3}" CPU_THREADS="${CPU_THREADS:-1}"
exec bash "$REPO/flux3/scripts/train_flux3_qdm_repeat.sh" \
  --stage b --from-base --attention-mode memory-only --training-subset-size 20000 --steps 500 \
  --qdm-layers 2,4,7,8,9,24,25,26,27 --region-size 8 --halo 2 \
  --batch-size 2 --grad-accum 1 --workers 4 --threads "$CPU_THREADS" \
  --save-steps 100 --validation-steps 100 --cycle-seconds 1200 \
  --output-root "${FLUX3_LOW9_MEMORY_RUN:-outputs/flux3-memory-lowmse9-direct-b-node01}" \
  "$@"
