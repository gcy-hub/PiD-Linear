#!/usr/bin/env bash
# Creates a separate environment; never modifies the existing pixel environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
LINEAR_ENV="${LINEAR_PID_ENV:-linear-pid}"
if ! conda run -n "$LINEAR_ENV" python -V >/dev/null 2>&1; then
  conda create -n "$LINEAR_ENV" python=3.12 pip -y
fi
conda activate "$LINEAR_ENV"
# Optional direct PyTorch downloads affect only that child, not the caller or Codex.
pip_install() {
  if [[ "${LINEAR_PID_DOWNLOAD_DIRECT:-0}" == 1 ]]; then
    env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
      python -m pip install "$@"
  else
    python -m pip install "$@"
  fi
}
pip_install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$REPO_ROOT" pytest ruff 'swanlab==0.10.1' 'scipy==1.15.2'
python -m pip install 'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@9f38d24980c46d46bd38614e743cdacd21906578'
cd "$REPO_ROOT"
PYTHONPATH=. python scripts/verify_linear_pid_env.py
