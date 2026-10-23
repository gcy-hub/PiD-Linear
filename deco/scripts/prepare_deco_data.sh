#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repo_root}/deco${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m deco_qdm.data "$@"
