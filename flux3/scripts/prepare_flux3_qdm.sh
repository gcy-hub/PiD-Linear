#!/usr/bin/env bash
set -euo pipefail
FLUX3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FLUX3_ROOT}:${FLUX3_ROOT}/DiffSynth-Studio${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec python -m flux3_qdm.prepare "$@"
