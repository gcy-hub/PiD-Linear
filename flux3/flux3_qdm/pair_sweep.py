"""Fourteen independent two-joint-layer A1 -> A2 -> B diagnostic runs.

Each renewable controller runs one bounded stage cycle, then hands its lease to
a new process. Existing trainer/evaluator checkpoint and child-cleanup contracts
are reused. Parameters, optimizer state and parents never cross pair roots.
"""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

from . import repeat

REPO = repeat.REPO
PAIRS = tuple(dict(name=f'pair_{i:02d}_{i+1:02d}', layers=[i, i+1]) for i in range(0,28,2))


def next_task(root):
    root=Path(root)
    for pair in PAIRS:
        stage=repeat.select_stage(root/pair['name'],'all')
        if stage is not None:
            return pair,stage
    return None


def stage_request(config,pair,stage):
    argv=['--weights-root',config['weights_root'],'--subset-root',config['subset_root'],
          '--cache-root',config['cache_root'],'--attention-mode','hybrid',
          '--qdm-layers',','.join(map(str,pair['layers'])),'--region-size','8','--halo','2',
          '--training-subset-size','20000','--steps',str(config['steps']),
          '--warmup-steps',str(config['warmup_steps']),'--save-steps',str(config['steps']),
          '--validation-steps',str(config['steps'] if stage=='b' else 0),
          '--batch-size',str(config['batch_size']),'--grad-accum',str(config['grad_accum']),
          '--workers',str(config['workers']),'--threads',str(config['threads']),
          '--seed',str(config['seed']),'--swanlab-mode','offline']
    if stage!='b':argv.append('--skip-validation')
    # A1 has no parent. Normal A2/B loading selects A1/A2 in THIS pair root.
    return {**config,'training_args':argv}


def update_summary(root):
    rows=[]
    for pair in PAIRS:
        directory=root/pair['name']
        row=dict(pair=pair['name'],layers=pair['layers'])
        for stage in repeat.STAGES:
            checkpoint=repeat.latest_complete(directory/stage)
            row[stage+'_step']=repeat.read_json(checkpoint/'complete.json').get('step',0) if checkpoint else 0
            row[stage+'_complete']=repeat.stage_finished(directory,stage)
        row['b_images']=len(list((directory/'b/galleries').glob('**/*.png')))
        rows.append(row)
    repeat.write_json(root/'summary.json',dict(experiments=rows,completed_pairs=sum(r['b_complete'] for r in rows)))
    pending=root/f'.summary-{os.getpid()}.csv'
    with pending.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(pending,root/'summary.csv')


def seed_fingerprint(root,pair_root,config):
    path=Path(config['weights_root'])/'flux-3-image-klein-base/model.safetensors'
    stat=path.stat();signature=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    shared=repeat.read_json(root/'base_fingerprint.json')
    if shared.get('signature')==signature and shared.get('sha256'):
        repeat.write_json(pair_root/'base_fingerprint.json',shared)
    return signature


def spawn_controller(root,fd):
    log=(root/'repeat/controller.log').open('a')
    env={**os.environ,'PYTHONPATH':str(REPO/'flux3')+os.pathsep+str(REPO/'flux3/DiffSynth-Studio')+os.pathsep+os.getenv('PYTHONPATH','')}
    child=subprocess.Popen([sys.executable,'-m','flux3_qdm.pair_sweep','--output-root',str(root),
                            '--worker','--lease-fd',str(fd)],cwd=REPO,env=env,pass_fds=(fd,),
                           start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    log.close();return child.pid


def worker(root,fd):
    control=root/'repeat';interrupted={'value':False}
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:interrupted.update(value=True))
    with os.fdopen(fd,'a') as lease:
        stopped=lambda:interrupted['value'] or (control/'STOP').exists()
        task=next_task(root)
        if stopped() or task is None:
            update_summary(root)
            repeat.write_json(control/'state.json',dict(status='stopped' if stopped() else 'completed',controller_pid=os.getpid()))
            return
        pair,stage=task;pair_root=root/pair['name'];pair_root.mkdir(parents=True,exist_ok=True)
        plan=repeat.read_json(root/'plan.json');config=plan['config']
        progress=repeat.read_json(control/'progress.json');cycle=progress.get('cycle',0)+1
        signature=seed_fingerprint(root,pair_root,config)
        state=dict(status='running',pair=pair['name'],layers=pair['layers'],stage=stage,
                   cycle=cycle,controller_pid=os.getpid(),pair_root=str(pair_root))
        repeat.write_json(control/'state.json',state)
        link=control/f'.latest-{os.getpid()}'
        link.symlink_to(os.path.relpath(pair_root/'repeat/latest-training.log',control))
        os.replace(link,control/'latest-training.log')
        request=stage_request(config,pair,stage)
        try:
            result=repeat.run_cycle(pair_root,stage,request,cycle,interrupted=stopped)
        except Exception as exc:
            repeat.write_json(control/'state.json',dict(**{k:v for k,v in state.items() if k!='status'},status='failed',error=f'{type(exc).__name__}: {exc}'))
            raise
        fingerprint=repeat.read_json(pair_root/'base_fingerprint.json')
        if fingerprint.get('signature')==signature and fingerprint.get('sha256'):
            repeat.write_json(root/'base_fingerprint.json',fingerprint)
        update_summary(root)
        stalled=0 if result.get('progressed') else progress.get('stalled_cycles',0)+1
        failed=result.get('fatal',False) or (result['returncode'] not in (0,75) and not (result.get('forced') or result.get('killed')))
        if stopped() or result['reason']=='manual-stop':status='stopped'
        elif next_task(root) is None:status='completed'
        elif failed or stalled>=3:status='failed'
        else:status='restarting'
        repeat.write_json(control/'state.json',{**state,**result,'status':status})
        repeat.write_json(control/'progress.json',dict(cycle=cycle,stalled_cycles=stalled))
        repeat.event(root,status,pair=pair['name'],stage=stage,cycle=cycle)
        if status=='restarting':
            successor=spawn_controller(root,lease.fileno())
            repeat.event(root,'controller-handoff',successor_pid=successor,cycle=cycle)


def launch(root,config):
    control=root/'repeat';control.mkdir(parents=True,exist_ok=True)
    with (control/'controller.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This pairwise sweep is already running') from None
        plan=dict(schema=1,config=config,pairs=list(PAIRS),initialization='independent original base per pair; only same-pair stage parents')
        old=repeat.read_json(root/'plan.json')
        if old and old!=plan:raise ValueError('Sweep configuration changed; use the same settings or a new output directory')
        for pair in PAIRS:
            state=repeat.read_json(root/pair['name']/'repeat/state.json')
            if repeat.owned_process(state.get('training_pid',0),state.get('training_start_ticks')):
                raise RuntimeError(f"Previous child still alive in {pair['name']}")
        task=next_task(root)
        if task is None:print('All 14 pairs already completed');return
        stop=root/task[0]['name']/task[1]/'STOP'
        if stop.exists():
            if stop.read_text().startswith(repeat.OWN_STOP):stop.unlink()
            else:raise RuntimeError(f'Remove manual stage stop before resuming: {stop}')
        (control/'STOP').unlink(missing_ok=True)
        repeat.write_json(root/'plan.json',plan)
        repeat.write_json(control/'progress.json',dict(cycle=0,stalled_cycles=0))
        pid=spawn_controller(root,lease.fileno())
        print(f'Started independent 14-pair sweep; controller PID {pid}',flush=True)
        print(f'Training log: {control / "latest-training.log"}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='outputs/flux3-pairwise-hybrid-50')
    p.add_argument('--gpu-ids',default=os.getenv('GPU_IDS','1,2,3'))
    for name,default in [('batch-size',2),('grad-accum',1),('workers',4),('threads',1),('steps',50),('warmup-steps',5),('seed',42),('cycle-seconds',1200),('grace-seconds',180)]:
        p.add_argument('--'+name,type=int,default=default)
    dataset=Path.home()/'dataset/MultiAspect-4K-1M'
    p.add_argument('--weights-root',default=os.getenv('FLUX3_WEIGHTS_ROOT',str(Path.home()/'huggingface_ckpts')))
    p.add_argument('--subset-root',default=os.getenv('FLUX3_SUBSET_ROOT',str(dataset/'pid_vdn_subsets')))
    p.add_argument('--cache-root',default=os.getenv('FLUX3_CACHE_ROOT',str(dataset/'flux3_qdm_cache')))
    for name in ('status','stop','dry-run','worker'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--lease-fd',type=int)
    args=p.parse_args(argv);root=Path(args.output_root).resolve()
    if args.worker:worker(root,args.lease_fd);return
    if args.status:
        state=repeat.read_json(root/'repeat/state.json')
        if state.get('pair_root'):state['child']=repeat.read_json(Path(state['pair_root'])/'repeat/state.json')
        print(__import__('json').dumps(state,indent=2));return
    if args.stop:
        (root/'repeat').mkdir(parents=True,exist_ok=True);(root/'repeat/STOP').touch()
        print('Stop requested; save at an update/image boundary and stop the whole sweep');return
    ids=args.gpu_ids.split(',')
    if any(not x.isdigit() for x in ids) or len(set(ids))!=len(ids):p.error('GPU IDs must be distinct nonnegative integers')
    if min(args.batch_size,args.grad_accum,args.workers,args.threads,args.steps,args.cycle_seconds)<1 or not 0<=args.warmup_steps<args.steps:
        p.error('Positive sizes required; warmup must be nonnegative and shorter than each stage')
    if args.grace_seconds<0 or args.cycle_seconds+args.grace_seconds>=1700:p.error('Cycle plus grace must stay below1700seconds on node01')
    config={k:v for k,v in vars(args).items() if k not in ('output_root','worker','lease_fd','status','stop','dry_run')}
    for key in ('weights_root','subset_root','cache_root'):config[key]=str(Path(config[key]).expanduser().resolve())
    if args.dry_run:
        import json
        print(json.dumps([dict(pair=pair,stages={s:repeat.training_command(root/pair['name'],s,stage_request(config,pair,s)) for s in repeat.STAGES}) for pair in PAIRS],indent=2));return
    launch(root,config)


if __name__=='__main__':main()
