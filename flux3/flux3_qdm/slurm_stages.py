"""Keep every training segment multi-node, including after midpoint validation."""
import argparse
from collections import deque
import os
from pathlib import Path
import sys
import time
from .launcher import pending_checkpoint, replace_options, run_owned
from .repeat import read_json,stage_finished
from .training import parse_args,stage_settings


PAYLOAD_STARTED='FLUX3_QDM_PAYLOAD_STARTED'


def run_srun_step(command,env):
    """Retry transport errors only when no remote payload has started."""
    attempts=int(env.get('FLUX3_SLURM_START_ATTEMPTS','3'))
    delay=float(env.get('FLUX3_SLURM_RETRY_SECONDS','10'))
    if attempts<1 or delay<0:raise ValueError('Invalid Slurm startup retry settings')
    for attempt in range(1,attempts+1):
        tail=deque(maxlen=128)
        started={'value':False}
        def output(line):
            tail.append(line)
            if PAYLOAD_STARTED in line:started['value']=True
            sys.stdout.write(line)
            sys.stdout.flush()
        code=run_owned(command,env,on_output=output)
        message=''.join(tail).lower()
        allocation_failure=('unable to confirm allocation' in message or 'unable to create step' in message)
        transport_failure=('socket timed out' in message or 'unable to contact slurm controller' in message)
        if (code==0 or code<0 or code>=128 or started['value'] or not allocation_failure
                or not transport_failure or attempt==attempts):
            return code
        print(f'Slurm startup communication failed before payload launch; retry {attempt+1}/{attempts} in {delay:g}s',flush=True)
        time.sleep(delay)
    raise AssertionError('Unreachable retry loop')


def run(args,argv,stages,child=run_srun_step):
    root=Path(args.output_root).resolve()
    nodes=int(os.environ.get('SLURM_JOB_NUM_NODES','0'))
    if nodes!=2:raise ValueError('This launcher requires a two-node Slurm allocation')
    master=os.environ['MASTER_ADDR'];port=os.environ['MASTER_PORT']
    env={**os.environ};env.pop('GPU_IDS',None)
    for stage in stages:
        while not stage_finished(root,stage):
            if (root/stage/'STOP').exists():return 75
            checkpoint=pending_checkpoint(root,stage)
            if checkpoint is not None:
                step=int(read_json(checkpoint/'complete.json')['step'])
                command=['srun','--nodes=1','--ntasks=1','--ntasks-per-node=1','--kill-on-bad-exit=1',
                    'bash','-c',f'printf "%s\\n" {PAYLOAD_STARTED}; exec python -m flux3_qdm.evaluate --gpu-ids "${{CUDA_VISIBLE_DEVICES:?}}" "$@"','bash',
                    '--checkpoint',str(checkpoint),'--stage',stage,'--stage-root',str(root/stage),
                    '--output-root',str(root/stage/'galleries'/f'step_{step:09d}'),
                    '--weights-root',args.weights_root,'--workers',str(args.workers),
                    '--threads',str(args.threads),'--student-only','--resolutions','2048','--resume','--max-seconds','486000']
                code=child(command,env)
                if code:return code
                if not stage_finished(root,stage) and pending_checkpoint(root,stage) is not None:
                    raise RuntimeError('Validation returned without acknowledging its checkpoint')
                continue
            settings=stage_settings(args,stage)
            options=replace_options(argv,stage=stage,steps=settings['steps'],max_steps=0,resume='auto',max_seconds=0)
            command=['srun','--nodes=2','--ntasks=2','--ntasks-per-node=1','--kill-on-bad-exit=1',
                'bash','-c',
                f'printf "%s\\n" {PAYLOAD_STARTED}; exec torchrun --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_PROCID" '
                '--master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --module flux3_qdm.training "$@"',
                'bash',*options]
            code=child(command,env)
            if code:return code
            if not stage_finished(root,stage) and pending_checkpoint(root,stage) is None:
                return 75
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stages',default='a1,a2')
    own,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    stages=own.stages.split(',')
    if stages not in (['a1','a2'],['a1'],['a2']):raise ValueError('Expected a1,a2, a1 or a2')
    return run(parse_args(rest),rest,stages)


if __name__=='__main__':raise SystemExit(main())
