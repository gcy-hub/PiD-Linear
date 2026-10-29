#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/flux3:$REPO/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
exec python -m flux3_qdm.repeat "$@"
