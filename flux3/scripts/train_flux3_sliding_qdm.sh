#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# Additional arguments override these defaults, including stage, paths and steps.
exec bash flux3/scripts/train_flux3_qdm_repeat.sh \
  --stage all --cycle-seconds 1200 \
  --output-root "${FLUX3_QDM_RUN:-outputs/flux3-sliding11-qdm-2k-node01}" \
  --local-attention sliding --sliding-kernel-size 11 --attention-mode hybrid --replace-image-blocks \
  --training-subset-size 20000 \
  --a1-steps 500 --a2-steps 1000 --b-steps 10000 \
  --batch-size 2 --grad-accum 1 --save-steps 500 --validation-steps 500 \
  --workers 4 --threads 1 --resume auto "$@"
