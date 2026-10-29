"""Four two-GPU queues -> strict 28-layer merge -> eight-GPU A1/A2/B."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import pair_sweep, repeat

JOINT_STEPS={'a1':500,'a2':1000,'b':10000}


def make_groups(gpu_ids):
    ids=[x.strip() for x in gpu_ids.split(',')]
    if len(ids)!=8 or len(set(ids))!=8 or any(not x.isdigit() for x in ids):
        raise ValueError('This pipeline requires exactly eight distinct GPU IDs on one server')
    return [dict(name=f'group_{i}',gpu_ids=','.join(ids[2*i:2*i+2]),pair_indices=list(range(i,14,4))) for i in range(4)]


def pair_config(config,group):
    return {**{k:config[k] for k in ('weights_root','subset_root','cache_root','batch_size','grad_accum',
              'workers','threads','seed','cycle_seconds','grace_seconds')},
            'gpu_ids':group['gpu_ids'],'pair_indices':group['pair_indices'],'steps':50,'warmup_steps':5}


def joint_request(config,stage,merged):
    argv=['--weights-root',config['weights_root'],'--subset-root',config['subset_root'],
          '--cache-root',config['cache_root'],'--attention-mode','hybrid','--qdm-layers',','.join(map(str,range(28))),
          '--region-size','8','--halo','2','--retain-lora',
          '--training-subset-size',str(50000 if stage=='b' else 20000),
          '--steps',str(JOINT_STEPS[stage]),'--warmup-steps',str(200 if stage=='b' else 50),
          '--save-steps','500','--validation-steps','500',
          '--batch-size',str(config['batch_size']),'--grad-accum',str(config['grad_accum']),
          '--workers',str(config['workers']),'--threads',str(config['threads']),
          '--seed',str(config['seed']),'--swanlab-mode','offline']
    if stage=='a1':argv.extend(['--init-from',str(merged)])
    return {**config,'training_args':argv}


def active(root):
    path=Path(root)/'repeat/controller.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def stop_children(root,groups):
    for group in groups:
        repeat.write_json(root/'pairs'/group['name']/'repeat/STOP',{})
    # Also reach orphaned training children if their controller was forcibly killed.
    roots=[root/'joint']
    for group in groups:
        state=repeat.read_json(root/'pairs'/group['name']/'repeat/state.json')
        if state.get('pair_root'):roots.append(Path(state['pair_root']))
    for directory in roots:
        for stage in repeat.STAGES:
            target=directory/stage/'STOP'
            if target.parent.exists() and not target.exists():target.write_text(repeat.OWN_STOP+'pipeline-stop')


def group_status(root,groups):
    rows=[]
    for group in groups:
        directory=root/'pairs'/group['name']
        pairs=pair_sweep.selected_pairs(group)
        rows.append({**group,'complete':pair_sweep.next_task(directory,pairs) is None,
                     'state':repeat.read_json(directory/'repeat/state.json'),
                     'summary':repeat.read_json(directory/'summary.json')})
    return rows


def ensure_groups(root,config,groups,explicit_resume=False):
    for group in groups:
        directory=root/'pairs'/group['name']
        if pair_sweep.next_task(directory,pair_sweep.selected_pairs(group)) is None:continue
        if active(directory):continue
        state=repeat.read_json(directory/'repeat/state.json')
        if not explicit_resume and state.get('status') in ('failed','stopped'):
            raise RuntimeError(f"{group['name']} {state['status']}: inspect {directory}/repeat/controller.log")
        pair_sweep.launch(directory,pair_config(config,group))


def spawn(root,fd):
    log=(root/'repeat/controller.log').open('a')
    env={**os.environ,'PYTHONPATH':str(repeat.REPO/'flux3')+os.pathsep+str(repeat.REPO/'flux3/DiffSynth-Studio')+os.pathsep+os.getenv('PYTHONPATH','')}
    child=subprocess.Popen([sys.executable,'-m','flux3_qdm.parallel_pipeline','--output-root',str(root),
                            '--worker','--lease-fd',str(fd)],cwd=repeat.REPO,env=env,pass_fds=(fd,),
                           start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    log.close();return child.pid


def worker(root,fd):
    plan=repeat.read_json(root/'plan.json');config=plan['config'];groups=plan['groups']
    interrupted={'value':False};signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:interrupted.update(value=True))
    stopping=lambda:interrupted['value'] or (root/'repeat/STOP').exists()
    with os.fdopen(fd,'a') as lease:
        try:
            started=time.monotonic()
            while True:
                if stopping():
                    stop_children(root,groups)
                    # Keep the lease while the group's own controllers save and exit.
                    while any(active(root/'pairs'/g['name']) for g in groups):time.sleep(1)
                    repeat.write_json(root/'repeat/state.json',dict(status='stopped',controller_pid=os.getpid()))
                    return
                rows=group_status(root,groups)
                if all(row['complete'] for row in rows):break
                ensure_groups(root,config,groups)
                repeat.write_json(root/'repeat/state.json',dict(status='running',phase='pairs',controller_pid=os.getpid(),groups=rows))
                if time.monotonic()-started>=config['cycle_seconds']:
                    pid=spawn(root,lease.fileno());repeat.event(root,'controller-handoff',successor_pid=pid);return
                time.sleep(5)
            # Wait for the finished pair controllers to release their GPU processes/leases.
            if any(active(root/'pairs'/g['name']) for g in groups):
                time.sleep(1);pid=spawn(root,lease.fileno());repeat.event(root,'controller-handoff',successor_pid=pid);return
            from .merge_pairs import completed_sources, merge_checkpoints
            merged=merge_checkpoints(completed_sources([root/'pairs'/g['name'] for g in groups]),root/'merged')
            joint=root/'joint';joint.mkdir(parents=True,exist_ok=True)
            stage=repeat.select_stage(joint,'all')
            if stage is None:
                repeat.write_json(root/'repeat/state.json',dict(status='completed',phase='joint',controller_pid=os.getpid()))
                return
            repeat.write_json(root/'repeat/state.json',dict(status='running',phase='joint',stage=stage,controller_pid=os.getpid(),merged=str(merged)))
            previous=repeat.read_json(root/'repeat/progress.json');cycle=previous.get('joint_cycle',0)+1
            result=repeat.run_cycle(joint,stage,joint_request(config,stage,merged),cycle,interrupted=stopping)
            stalled=0 if result.get('progressed') else previous.get('stalled_cycles',0)+1
            repeat.write_json(root/'repeat/progress.json',dict(joint_cycle=cycle,stalled_cycles=stalled))
            if stopping() or result['reason']=='manual-stop':status='stopped'
            elif repeat.select_stage(joint,'all') is None:status='completed'
            elif result.get('fatal') or (result['returncode'] not in (0,75) and not (result.get('forced') or result.get('killed'))) or stalled>=3:
                raise RuntimeError(f'Joint stage failed: {result}; stalled_cycles={stalled}')
            else:status='restarting'
            repeat.write_json(root/'repeat/state.json',dict(status=status,phase='joint',stage=stage,result=result,controller_pid=os.getpid()))
            if status=='restarting':
                pid=spawn(root,lease.fileno());repeat.event(root,'controller-handoff',successor_pid=pid)
        except Exception as exc:
            stop_children(root,groups)
            repeat.write_json(root/'repeat/state.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',controller_pid=os.getpid()))
            raise


def launch(root,config,preflight=True):
    control=root/'repeat';control.mkdir(parents=True,exist_ok=True)
    groups=make_groups(config['gpu_ids'])
    with (control/'controller.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Pipeline already running') from None
        plan=dict(schema=1,config=config,groups=groups,joint_steps=JOINT_STEPS)
        old=repeat.read_json(root/'plan.json')
        if old and old!=plan:raise ValueError('Configuration changed; restore settings or use a new output root')
        child=repeat.read_json(root/'joint/repeat/state.json')
        if repeat.owned_process(child.get('training_pid',0),child.get('training_start_ticks')):
            raise RuntimeError('Previous joint training child is still alive')
        if preflight:
            from .resources import resource_manifest
            from .data import CachedDataset
            resource_manifest(config['weights_root'])
            for size in (20000,50000):
                dataset=CachedDataset(config['subset_root'],config['cache_root'],size)
                del dataset
        repeat.write_json(root/'plan.json',plan)
        (control/'STOP').unlink(missing_ok=True)
        for stage in repeat.STAGES:
            path=root/'joint'/stage/'STOP'
            if path.exists() and path.read_text().startswith(repeat.OWN_STOP):path.unlink()
        try:
            ensure_groups(root,config,groups,explicit_resume=True)
            pid=spawn(root,lease.fileno())
        except Exception:
            stop_children(root,groups);raise
        print(f'Background four-queue / merged-model pipeline started: PID {pid}',flush=True)
        print(f'Status: {control / "state.json"}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='outputs/flux3-pairs-parallel-then-joint')
    p.add_argument('--gpu-ids',default=os.getenv('GPU_IDS','0,1,2,3,4,5,6,7'))
    for name,default in [('batch-size',2),('grad-accum',1),('workers',4),('threads',1),('seed',42),('cycle-seconds',1200),('grace-seconds',180)]:p.add_argument('--'+name,type=int,default=default)
    data=Path.home()/'dataset/MultiAspect-4K-1M'
    for flag,env,default in [('weights-root','FLUX3_WEIGHTS_ROOT',Path.home()/'huggingface_ckpts'),
                             ('subset-root','FLUX3_SUBSET_ROOT',data/'pid_vdn_subsets'),('cache-root','FLUX3_CACHE_ROOT',data/'flux3_qdm_cache')]:
        p.add_argument('--'+flag,default=os.getenv(env,str(default)))
    for flag in ('status','stop','dry-run','worker'):p.add_argument('--'+flag,action='store_true')
    p.add_argument('--lease-fd',type=int)
    args=p.parse_args(argv);root=Path(args.output_root).resolve()
    if args.worker:worker(root,args.lease_fd);return
    if args.status:
        state=repeat.read_json(root/'repeat/state.json');plan=repeat.read_json(root/'plan.json')
        if plan:state['groups']=group_status(root,plan['groups'])
        state['joint']=repeat.read_json(root/'joint/repeat/state.json')
        print(__import__('json').dumps(state,indent=2));return
    if args.stop:
        repeat.write_json(root/'repeat/STOP',{})
        plan=repeat.read_json(root/'plan.json')
        if plan:stop_children(root,plan['groups'])
        print('Stop requested for all four queues and joint training; save at safe boundaries');return
    try:groups=make_groups(args.gpu_ids)
    except ValueError as exc:p.error(str(exc))
    args.gpu_ids=','.join(g['gpu_ids'] for g in groups)
    if min(args.batch_size,args.grad_accum,args.workers,args.threads,args.cycle_seconds)<1 or args.grace_seconds<0 or args.cycle_seconds+args.grace_seconds>=1700:
        p.error('Positive sizes required; cycle plus grace must be below1700seconds')
    config={k:v for k,v in vars(args).items() if k not in ('output_root','status','stop','dry_run','worker','lease_fd')}
    for key in ('weights_root','subset_root','cache_root'):config[key]=str(Path(config[key]).expanduser().resolve())
    if args.dry_run:
        import json
        print(json.dumps(dict(groups=groups,joint_steps=JOINT_STEPS,
                              joint_commands={s:repeat.training_command(root/'joint',s,joint_request(config,s,root/'merged/CHECKPOINT')) for s in JOINT_STEPS}),indent=2));return
    launch(root,config)


if __name__=='__main__':main()
