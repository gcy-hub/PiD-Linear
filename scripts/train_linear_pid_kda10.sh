#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$REPO_ROOT/scripts/launch_linear_pid_stage.py" \
  --stage-config "$REPO_ROOT/pid/_src/configs/linear_pid/kda10.json" "$@"
