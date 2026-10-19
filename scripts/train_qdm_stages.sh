#!/usr/bin/env bash
# Foreground A1 -> A2 -> B; Slurm provides process lifetime and GPU allocation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${PID_QDM_RUN:?Set PID_QDM_RUN}"
: "${PID_QDM_LOCAL_MODE:?Set halo or alternating_shift}"
QDM_WORLD=$(( ${NNODES:-1} * ${NPROC_PER_NODE:-4} ))
common=(--preset 4gpu --memory-mode fixed4 --region-size "${PID_QDM_REGION_SIZE:-8}"
        --local-mode "$PID_QDM_LOCAL_MODE" --local-halo "${PID_QDM_HALO:-2}" --local-shift "${PID_QDM_SHIFT:-4}"
        --resume auto --allow-batch-change --max-seconds 0
        --workers "${PID_QDM_WORKERS:-4}" --threads "${CPU_THREADS:-1}" --log-steps 1
        --swanlab-mode "${PID_QDM_SWANLAB_MODE:-offline}" --output-root "$PID_QDM_RUN"
        --weights-root "${PID_WEIGHTS_ROOT:?}" --index-root "${PID_INDEX_ROOT:?}"
        --subset-root "${PID_VDN_SUBSET_ROOT:?}" --text-cache-root "${PID_TEXT_CACHE_ROOT:?}"
        --gallery-root "${PID_GALLERY_ROOT:?}")
# Stage-specific batch settings. No node01 timer or restart controller is used here.
for profile in a1:4:1:500:50:-1 a2:4:2:1000:100:-1 b:4:2:10000:2000:2000; do
  IFS=: read -r stage batch accum steps save validate <<< "$profile"
  if [[ -e "$PID_QDM_RUN/$stage/STOP" ]]; then
    echo "Stop marker for $stage; later stages will not start."
    exit 0
  fi
  finished="$(python - "$PID_QDM_RUN" "$stage" "$steps" "$save" "$validate" <<'CHECK'
import json,sys,os
from pathlib import Path
from scripts.repeat_pid_vdn import stage_finished
root=Path(sys.argv[1]);stage=sys.argv[2]
if stage_finished(root,stage):
    cfg=json.loads((root/stage/'config.json').read_text())
    from pid._src.vdn_pid.config import VDNConfig,verify_memory_mode,verify_local_layout
    target=VDNConfig(memory_mode='fixed4',local_mode=os.environ['PID_QDM_LOCAL_MODE'],local_halo=int(os.getenv('PID_QDM_HALO','2')),local_shift=int(os.getenv('PID_QDM_SHIFT','4')),region_size=int(os.getenv('PID_QDM_REGION_SIZE','8')))
    verify_memory_mode(target,cfg);verify_local_layout(target,cfg)
    settings=cfg['stage_settings']
    expected=dict(steps=int(sys.argv[3]),save=int(sys.argv[4]),validate=(2000 if stage=='b' else 0) if int(sys.argv[5])<0 else int(sys.argv[5]))
    if any(settings[k]!=v for k,v in expected.items()):
        raise RuntimeError(f'{stage} completed with a different training schedule')
    print('complete')
else:
    print('pending')
CHECK
)"
  if [[ "$finished" == complete && "${PID_QDM_DRY_RUN:-0}" != 1 ]]; then
    echo "Skip completed and validated $stage"
    continue
  fi
  effective=$(( QDM_WORLD * batch * accum ))
  command=(bash "$REPO_ROOT/scripts/train_pid_vdn.sh" "${common[@]}"
           --stage "$stage" --batch-size "$batch" --grad-accum "$accum" --effective-batch "$effective"
           --max-steps "$steps" --save-steps "$save" --validation-steps "$validate" "$@")
  echo "Starting $stage: per-GPU batch=$batch accum=$accum world=$QDM_WORLD effective=$effective steps=$steps"
  if [[ "${PID_QDM_DRY_RUN:-0}" == 1 ]]; then
    WORLD_SIZE="$QDM_WORLD" python -m scripts.train_pid_vdn "${common[@]}" \
      --stage "$stage" --batch-size "$batch" --grad-accum "$accum" --effective-batch "$effective" \
      --max-steps "$steps" --save-steps "$save" --validation-steps "$validate" --dry-run
    continue
  fi
  "${command[@]}"
  # A stopped update/gallery can exit successfully; never mistake that for completion.
  complete="$(python - "$PID_QDM_RUN" "$stage" <<'CHECK'
from pathlib import Path
import sys
from scripts.repeat_pid_vdn import stage_finished
print(int(stage_finished(Path(sys.argv[1]),sys.argv[2])))
CHECK
)"
  if [[ "$complete" != 1 ]]; then
    echo "$stage paused before validation completed; resubmit the same job to resume."
    exit 0
  fi
done
