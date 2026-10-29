"""Serial torchrun/evaluation orchestration; the launcher never initializes CUDA."""
from __future__ import annotations
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .repeat import checkpoint_complete, read_json, stage_finished

STAGES=('a1','a2','b')


def replace_options(argv, **values):
    """Replace valued options, including --flag=value forms, preserving others."""
    flags={'--'+key.replace('_','-'):value for key,value in values.items()}
    result=[]; index=0
    while index<len(argv):
        item=argv[index]
        flag=item.split('=',1)[0]
        if flag in flags:
            index+=1 if '=' in item else 2
        else:
            result.append(item); index+=1
    for flag,value in flags.items():
        if value is not None:
            result.extend([flag,str(value)])
    return result


def run_owned(command,env,on_output=None):
    """Wait for an owned direct child, forwarding termination without signalling peers."""
    interrupted={'signal':None,'at':None}
    output_options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors='replace', bufsize=1) if on_output else {}
    process=subprocess.Popen(command,env=env,start_new_session=True,**output_options)
    pump=None
    if on_output:
        def forward_output():
            with process.stdout as stream:
                for line in stream:
                    on_output(line)
        pump=threading.Thread(target=forward_output,daemon=True)
        pump.start()
    def forward(signum,frame):
        interrupted.update(signal=signum,at=time.monotonic())
        if process.poll() is None:
            process.terminate()
    previous={sig:signal.signal(sig,forward) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        while True:
            try:
                code=process.wait(timeout=.2)
                return 128+interrupted['signal'] if interrupted['signal'] else code
            except subprocess.TimeoutExpired:
                if interrupted['at'] is not None and time.monotonic()-interrupted['at']>=30:
                    process.kill()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for sig,handler in previous.items():
            signal.signal(sig,handler)
        if pump:
            pump.join(timeout=5)


def pending_checkpoint(root,stage):
    directory=root/stage
    marker=read_json(directory/'validation_pending.json') or read_json(directory/'training_finished.json')
    if not marker:
        return None
    checkpoint=Path(marker['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint=directory/checkpoint
    checkpoint=checkpoint.resolve()
    if not checkpoint_complete(checkpoint):
        raise RuntimeError(f'Validation refers to an incomplete checkpoint: {checkpoint}')
    if read_json(checkpoint/'complete.json').get('stage')!=stage:
        raise RuntimeError(f'Validation checkpoint belongs to another stage: {checkpoint}')
    return checkpoint


def run(args,argv,child=run_owned,clock=time.monotonic):
    root=Path(args.output_root).resolve()
    gpu_ids=os.getenv('GPU_IDS','0')
    ids=[s.strip() for s in gpu_ids.split(',')]
    if not ids or any(not s.isdigit() for s in ids) or len(set(ids))!=len(ids):
        raise ValueError('GPU_IDS must list distinct nonnegative GPU IDs')
    gpu_ids=','.join(ids)
    env={**os.environ,'GPU_IDS':gpu_ids,'CUDA_VISIBLE_DEVICES':gpu_ids}
    started=clock()
    def remaining():
        return max(0,math.ceil(args.max_seconds-(clock()-started))) if args.max_seconds else None
    stages=STAGES if args.stage=='all' else (args.stage,)
    for stage in stages:
        resumed=False
        while not stage_finished(root,stage):
            if (root/stage/'STOP').exists() or (root/'repeat/STOP').exists():
                return 75
            budget=remaining()
            if budget is not None and budget<=0:
                return 75
            checkpoint=pending_checkpoint(root,stage)
            if checkpoint is not None:
                step=int(read_json(checkpoint/'complete.json')['step'])
                command=[sys.executable,'-m','flux3_qdm.evaluate','--checkpoint',str(checkpoint),
                         '--stage',stage,'--stage-root',str(root/stage),
                         '--output-root',str(root/stage/'galleries'/f'step_{step:09d}'),
                         '--weights-root',args.weights_root,'--gpu-ids',gpu_ids,
                         '--workers',str(args.workers),'--threads',str(args.threads),'--student-only','--resolutions','2048','--resume',
                         '--max-seconds',str(budget or 1200)]
                code=child(command,env)
                if code:
                    return code
                if stage_finished(root,stage):
                    break
                # Evaluation must acknowledge a midpoint before any more training.
                if pending_checkpoint(root,stage) is not None:
                    return 75
                resumed=True
                continue
            options=dict(stage=stage,max_seconds=budget if budget is not None else 0)
            if resumed:
                options['resume']='auto'
            command=['torchrun','--standalone',f'--nproc_per_node={len(ids)}',
                     '--module','flux3_qdm.training',*replace_options(argv,**options)]
            code=child(command,env)
            if code:
                return code
            # This wait boundary ensures every distributed process has exited.
            if stage_finished(root,stage):
                break
            if pending_checkpoint(root,stage) is None:
                return 75
            resumed=True
    return 0


def main(argv=None):
    # Parsing imports torch through training, but performs no CUDA query/allocation.
    from .training import parse_args
    original=list(sys.argv[1:] if argv is None else argv)
    args=parse_args(original)
    return run(args,original)


if __name__=='__main__':
    raise SystemExit(main())
