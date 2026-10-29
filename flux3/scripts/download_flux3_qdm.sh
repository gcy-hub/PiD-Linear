#!/usr/bin/env bash
set -euo pipefail
FLUX3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FLUX3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Explicit official endpoint, scoped to this child process; preserve the user's proxy.
export HF_ENDPOINT=https://huggingface.co
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python -m flux3_qdm.resources "$@"
