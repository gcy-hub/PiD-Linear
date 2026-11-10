#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/flux2:$PWD/flux3"
export OMP_NUM_THREADS=4
PYTHON=/home/ganchangyi/software/miniconda3/envs/flux3-qdm/bin/python
OUTPUT="${1:-outputs/flux2-stock-kda-clear-node01-bs2}"
WEIGHTS=/home/ganchangyi/huggingface_ckpts/FLUX.2-klein-base-4B
DATA=/home/ganchangyi/dataset/CLEAR-FLUX2-klein-base-4B-10K
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/launcher.lock"
flock -n 9 || exit 0
rm -f "$OUTPUT/exit-code"
trap 'printf "%s\n" "$?" > "$OUTPUT/exit-code"' EXIT
resume=()
completed=0
if [[ -e "$OUTPUT/latest.pt" ]]; then
    completed=$("$PYTHON" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["step"])' "$OUTPUT/latest.pt")
    resume=(--resume "$OUTPUT/latest.pt")
fi
for boundary in 1 2 300 600 900 1000; do
    if (( boundary < completed )); then continue; fi
    if (( boundary > completed )); then
    CUDA_VISIBLE_DEVICES=1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=3 \
        -m flux_kda.train --data-mode cached --shard-optimizer --gradient-checkpointing --family flux2 --weights "$WEIGHTS" \
        --subset-root "$DATA" --cache-root "$DATA/flux2_diffsynth_sliding11_cache" \
        --output "$OUTPUT" --steps "$boundary" --batch-size 2 --grad-accum 1 \
        --save-every 100 "${resume[@]}"
    resume=(--resume "$OUTPUT/latest.pt")
    completed=$boundary
    fi
    if (( boundary <= 2 )); then continue; fi
    validation="$OUTPUT/validation/step-$boundary"
    if [[ -f "$validation/kda.json" ]]; then continue; fi
    mkdir -p "$validation"
    cp outputs/stock-kda-speed/flux2-2k/conditions.pt "$validation/conditions.pt"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m flux_kda.benchmark --family flux2 \
        --weights "$WEIGHTS" --checkpoint "$OUTPUT/latest.pt" --variant kda \
        --size 1024 --steps 20 --cfg 3.5 --seed 0 --output "$validation"
    resume=(--resume "$OUTPUT/latest.pt")
done
