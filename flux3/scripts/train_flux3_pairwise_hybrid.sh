#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/flux3:$REPO/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}" MKL_NUM_THREADS="${CPU_THREADS:-1}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-1}"
cd "$REPO"
exec python -m flux3_qdm.pair_sweep "$@"
