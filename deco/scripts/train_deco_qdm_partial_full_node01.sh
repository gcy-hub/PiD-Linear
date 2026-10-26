#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3}" CPU_THREADS="${CPU_THREADS:-1}"
exec bash deco/scripts/train_deco_qdm_repeat.sh \
  --stage all --cycle-seconds 1200 --grace-seconds 180 \
  --output-root "${DECO_QDM_PARTIAL_RUN:-outputs/deco-qdm-full-1-2-3-4-node01-bs32}" \
  --full-attention-layers 1,2,3,4 \
  --batch-size 32 --grad-accum 1 --workers 4 --threads "$CPU_THREADS" \
  --a1-steps 500 --a1-save-every 500 --a1-validate-every 500 \
  --a2-steps 1000 --a2-save-every 500 --a2-validate-every 500 \
  --b-steps 10000 --b-save-every 500 --b-validate-every 500 \
  --validation-student-only --swanlab-mode offline "$@"
