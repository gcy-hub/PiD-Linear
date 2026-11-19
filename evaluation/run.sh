#!/usr/bin/env bash
# Run inside one eight-GPU allocation; srun is a global barrier between models.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
output=${1:?usage: bash evaluation/run.sh /shared/output}
py=${PYTHON:-python}
: "${WEIGHTS_ROOT:?Set WEIGHTS_ROOT to the shared weights directory}" "${PID_EXPORT:?Set PID_EXPORT to the b-7629 directory}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for stage in conditions pid pid-dit; do
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        srun --ntasks=8 --gpus-per-task=1 --kill-on-bad-exit=1 bash -c '
            export RANK=$SLURM_PROCID WORLD_SIZE=8 LOCAL_RANK=0
            exec "$@"' _ "$py" "$here/generate.py" "$stage" "$output"
    else
        "$py" -c 'import torch; assert torch.cuda.device_count()==8, "requires eight allocated, visible GPUs"'
        "$py" -m torch.distributed.run --standalone --nproc_per_node=8 "$here/generate.py" "$stage" "$output"
    fi
done
"$py" "$here/report.py" "$output" efficiency
