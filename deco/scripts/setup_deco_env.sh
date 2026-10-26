#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
# Clone when the existing project environment is available. Keep all changes isolated.
if ! conda env list | awk '{print $1}' | grep -qx deco-qdm; then
  CONDA_NUMBER_CHANNEL_NOTICES=0 conda create -y --offline -n deco-qdm --clone "${SOURCE_ENV:-linear-pid}"
fi
conda activate deco-qdm
python -m pip install -r "$repo_root/deco/requirements.txt"
