#!/usr/bin/env bash
# One server: resume full-data downloads/preparation, then A1 -> A2 -> B.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m scripts.run_pid_qdm_4k_pipeline "$@"
