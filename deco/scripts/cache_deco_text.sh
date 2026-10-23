#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repo_root}/deco${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -ra visible_gpus <<< "${GPU_IDS}"
  logical_ids="0"
  for ((i=1; i<${#visible_gpus[@]}; i++)); do logical_ids+=",${i}"; done
  # argparse accepts a later explicit --gpu-ids override (logical visible IDs).
  exec python -m deco_qdm.cache_text --gpu-ids "${logical_ids}" "$@"
fi
exec python -m deco_qdm.cache_text "$@"
