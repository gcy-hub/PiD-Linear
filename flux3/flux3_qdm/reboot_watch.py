"""User-cron restart recovery for saved FLUX.3 pair sweeps and parallel pipelines."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from . import repeat


def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def plan_hash(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()


def cron_marker(root,hostname):
    return '# flux3-reboot:'+hashlib.sha256(f'{Path(root).resolve()}:{hostname}'.encode()).hexdigest()[:16]


def cron_text(current,root,hostname,command):
    marker=cron_marker(root,hostname)
    lines=[line for line in current.splitlines() if not line.endswith(marker)]
    if command:
        command=command.replace('%',r'\%')
        lines.extend([f'@reboot {command} {marker}',f'* * * * * {command} {marker}'])
    return '\n'.join(lines)+'\n'


def current_cron():
    result=subprocess.run(['crontab','-l'],capture_output=True,text=True,timeout=10)
    if result.returncode and 'no crontab' not in result.stderr.lower():raise RuntimeError(result.stderr.strip())
    return result.stdout


def lock_held(path):
    path=Path(path)
    if not path.parent.exists():return False
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def gpu_free_memory():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=15)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in result.stdout.splitlines() if row.strip()}


def saved_plan(root):
    plan=repeat.read_json(root/'plan.json')
    if plan:return plan
    request=repeat.read_json(root/'repeat/request.json')
    if not request:return {}
    return dict(kind='staged',config={k:v for k,v in request.items() if k not in ('cycle','stalled_cycles')})


def pending_roots(root,plan):
    if plan.get('kind')=='staged':
        stage=repeat.select_stage(root,plan['config']['stage'])
        return [(root,stage)] if stage is not None else []
    if 'groups' not in plan:
        return [(root/p['name'],repeat.select_stage(root/p['name'],'all')) for p in plan['pairs']
                if repeat.select_stage(root/p['name'],'all') is not None]
    rows=[]
    for group in plan['groups']:
        directory=root/'pairs'/group['name']
        for index in group['pair_indices']:
            pair=directory/f'pair_{2*index:02d}_{2*index+1:02d}'
            stage=repeat.select_stage(pair,'all')
            if stage is not None:rows.append((pair,stage))
    stage=repeat.select_stage(root/'joint','all')
    if stage is not None:rows.append((root/'joint',stage))
    return rows


def control_roots(root,plan):
    return [root]+([root/'pairs'/g['name'] for g in plan['groups']]+[root/'joint'] if 'groups' in plan else [])


def child_alive(directory):
    state=repeat.read_json(directory/'repeat/state.json')
    pid=state.get('training_pid',0)
    if not repeat.owned_process(pid,state.get('training_start_ticks')):return False
    try:return str(directory).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:return False


def launch_saved(root,kind,config):
    # Cron does not activate conda: retain the selected interpreter and executable search path.
    os.environ.update(GPU_IDS=config['gpu_ids'],CPU_THREADS=str(config.get('threads',1)),
                      OMP_NUM_THREADS=str(config.get('threads',1)),MKL_NUM_THREADS=str(config.get('threads',1)),
                      OPENBLAS_NUM_THREADS=str(config.get('threads',1)),TOKENIZERS_PARALLELISM='false')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.getenv('PATH','/usr/bin:/bin')
    if kind=='staged':
        repeat.launch(root,config)
        return
    if kind=='parallel':
        from .parallel_pipeline import launch
    else:
        from .pair_sweep import launch
    launch(root,config)


def check_once(root,config):
    root=Path(root)
    def result(status,**details):return dict(status=status,updated_unix=time.time(),boot_id=boot_id(),**details)
    if socket.gethostname()!=config['hostname']:return result('different-host')
    if not config.get('enabled',False):return result('disabled')
    if (root/'repeat/STOP').exists():return result('stopped')
    plan=saved_plan(root)
    if not plan:return result('waiting-for-plan')
    if config.get('plan_sha256') and config['plan_sha256']!=plan_hash(plan):return result('plan-changed')
    if lock_held(root/'repeat/controller.lock'):return result('running')
    state=repeat.read_json(root/'repeat/state.json')
    if state.get('status') in ('failed','stopped','paused'):
        return result(state['status'],reason=state.get('error') or state.get('reason'))
    pending=pending_roots(root,plan)
    if not pending:return result('completed')
    for directory,stage in pending:
        stop=directory/stage/'STOP'
        if stop.exists() and not stop.read_text().startswith(repeat.OWN_STOP):return result('stopped',stage_root=str(directory/stage))
    for directory in control_roots(root,plan)+[p for p,_ in pending]:
        if directory!=root and lock_held(directory/'repeat/controller.lock'):return result('workers-running')
        if child_alive(directory):return result('workers-running')
    for directory,stage in pending:
        if lock_held(directory/stage/'.training.lock'):return result('workers-running')
    try:free=gpu_free_memory()
    except (OSError,ValueError,subprocess.SubprocessError) as exc:return result('waiting-for-gpu-driver',reason=str(exc))
    ids=[int(x) for x in plan['config']['gpu_ids'].split(',')]
    if any(free.get(i,0)<config['min_free_mib'] for i in ids):
        return result('waiting-for-gpu-memory',free_mib={i:free.get(i,0) for i in ids})
    launch_saved(root,config['kind'],plan['config'])
    return result('resumed',gpu_ids=plan['config']['gpu_ids'])


def install(root,args):
    directory=root/'repeat/reboot-watch';directory.mkdir(parents=True,exist_ok=True)
    plan=saved_plan(root)
    if not plan:raise ValueError('Start the workload once so plan.json records its exact resume configuration')
    kind=plan.get('kind') or ('parallel' if 'groups' in plan else 'pairs')
    if not plan.get('config',{}).get('gpu_ids'):raise ValueError('Saved plan lacks GPU IDs')
    if args.min_free_mib<1:raise ValueError('--min-free-mib must be positive')
    config=dict(enabled=True,hostname=socket.gethostname(),kind=kind,min_free_mib=args.min_free_mib,
                gpu_ids=plan['config']['gpu_ids'],plan_sha256=plan_hash(plan),python=sys.executable,
                repo=str(repeat.REPO),installed_unix=time.time())
    current=current_cron()
    (directory/f'crontab.before-{time.time_ns()}.txt').write_text(current)
    repeat.write_json(directory/'config.json',config)
    (directory/'status.json').unlink(missing_ok=True)
    # Set PYTHONPATH before -m resolves the package in a fresh cron shell.
    prefix=os.pathsep.join([str(repeat.REPO/'flux3'),str(repeat.REPO/'flux3/DiffSynth-Studio')])
    command=f'cd {shlex.quote(str(repeat.REPO))} && PYTHONPATH={shlex.quote(prefix)} '+shlex.join(
        [sys.executable,'-m','flux3_qdm.reboot_watch','--check','--output-root',str(root)])
    command+=f' >> {shlex.quote(str(directory/"watch.log"))} 2>&1'
    subprocess.run(['crontab','-'],input=cron_text(current,root,config['hostname'],command),text=True,check=True,timeout=10)
    print(f'Reboot recovery installed: host={config["hostname"]}, kind={kind}, GPUs={config["gpu_ids"]}',flush=True)
    print(f'Status: {directory/"status.json"}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    for name in ('install','check','status','remove'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--min-free-mib',type=int,default=32768)
    args=p.parse_args(argv);root=args.output_root.resolve();directory=root/'repeat/reboot-watch'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'check.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if args.install:install(root,args);return
        config=repeat.read_json(directory/'config.json')
        if args.remove:
            config['enabled']=False;repeat.write_json(directory/'config.json',config)
            subprocess.run(['crontab','-'],input=cron_text(current_cron(),root,config.get('hostname',socket.gethostname()),None),text=True,check=True,timeout=10)
            print('Reboot recovery removed; training is unchanged');return
        if args.status:
            print(json.dumps(dict(config=config,status=repeat.read_json(directory/'status.json')),indent=2));return
        if not config:raise RuntimeError('Reboot recovery is not installed')
        old=repeat.read_json(directory/'status.json')
        if old.get('status')=='launch-failed' and old.get('boot_id')==boot_id() and old.get('failures',0)>=3:return
        try:result=check_once(root,config)
        except Exception as exc:
            attempts=old.get('failures',0)+1 if old.get('boot_id')==boot_id() else 1
            result=dict(status='launch-failed',reason=f'{type(exc).__name__}: {exc}',failures=attempts,boot_id=boot_id(),updated_unix=time.time())
        repeat.write_json(directory/'status.json',result)
        if result['status']!=old.get('status') or result['status'] in ('resumed','launch-failed'):print(json.dumps(result),flush=True)


if __name__=='__main__':main()
