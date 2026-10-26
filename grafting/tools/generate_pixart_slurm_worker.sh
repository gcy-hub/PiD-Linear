#!/usr/bin/env bash
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# One Slurm task owns all four GPUs on its node. Children get individual masks.
exec python -u "$TOOLS/generate_pixart_slurm_node.py" "$@"
