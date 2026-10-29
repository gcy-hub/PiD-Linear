"""Detached renewable 20K cache preparation followed by A1 only; CPU controller."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from . import repeat

REPO = Path(__file__).resolve().parents[2]


def _environment(request=None):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO/'flux3'), str(REPO/'flux3/DiffSynth-Studio'), env.get('PYTHONPATH','')])
    env['DIFFSYNTH_SKIP_DOWNLOAD'] = 'True'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    if request:
        for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CPU_THREADS']:
            env[name] = str(request['threads'])
    return env


def cache_progress(cache_root):
    """Count only current component .pt files, without reading captions or tensors."""
    root = Path(cache_root)
    manifest = repeat.read_json(root/'subsets/20000.json')
    components = manifest.get('components',{})
    result = {kind: sum(1 for _ in (root/kind/value).glob('*/*.pt'))
              for kind,value in components.items() if kind in ('image','text')}
    return dict(text=result.get('text',0), image=result.get('image',0), expected_images=20000,
                expected_text=(len({r['caption'] for r in manifest['rows']})+1 if len(manifest.get('rows',[]))>=20000 else None),
                indexed_rows=len(manifest.get('rows',[])))


def _missing_cache_entries(request):
    """Missing entries are resumable; existing inconsistent completion metadata is fatal."""
    from .data import CacheStore, fingerprint
    root = Path(request['cache_root'])
    manifest = repeat.read_json(root/'subsets/20000.json')
    store = CacheStore(root,manifest['components'])
    missing = False
    entries = [('text',fingerprint(''),fingerprint(''))]
    for row in manifest['rows']:
        entries.extend([('image',fingerprint(row['id']),row['image_source']),
                        ('text',row['caption'],row['caption'])])
    for kind,key,source in entries:
        path = store.path(kind,key)
        if not path.is_file() or not path.with_suffix('.json').is_file():
            missing = True
        elif not store.valid(kind,key,source):
            raise RuntimeError(f'Existing cache completion metadata is stale/corrupt: {path}; repair preparation explicitly')
    return missing


def cache_ready(request):
    from .data import CachedDataset
    try:
        CachedDataset(request['subset_root'],request['cache_root'],20000)
        return True
    except FileNotFoundError as exc:
        if str(exc).startswith('Missing cache manifest '):
            return False
        raise
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith('Cache contains only a limited smoke subset;'):
            return False
        if message.startswith(('Missing or stale image cache for ', 'Missing or stale text cache for ',
                               'Missing, stale or partial text cache:')) and _missing_cache_entries(request):
            return False
        raise


def prepare_command(request):
    return [sys.executable,'-m','flux3_qdm.prepare','--subset-root',request['subset_root'],
            '--cache-root',request['cache_root'],'--weights-root',request['weights_root'],
            '--subset-size','20000','--phase','all','--gpu-ids',request['gpu_ids'],
            '--workers',str(request['workers']),'--threads',str(request['threads']),
            '--resume','--max-seconds',str(math.ceil(request['cycle_seconds']))]


def training_request(request):
    return dict(stage='a1',gpu_ids=request['gpu_ids'],cycle_seconds=request['cycle_seconds'],
                grace_seconds=request['grace_seconds'],restart_delay=request['restart_delay'],max_cycles=0,
                training_args=['--weights-root',request['weights_root'],'--subset-root',request['subset_root'],
                               '--cache-root',request['cache_root'],'--batch-size','1','--grad-accum','1',
                               '--workers',str(request['workers']),'--threads',str(request['threads'])])


def _state(root,phase,**values):
    repeat.write_json(root/'bootstrap/state.json',dict(phase=phase,controller_pid=os.getpid(),
                      controller_start_ticks=(repeat.process_identity(os.getpid()) or (None,None))[0],
                      time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**values))


def _event(root,event,**values):
    path = root/'bootstrap/events.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    value = dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),controller_pid=os.getpid(),**values)
    with path.open('a') as out:
        out.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def run_once(root,request,interrupted=None,started=None):
    """One bounded preparation child or an A1 handoff; renewal is the caller's job."""
    root = Path(root).resolve()
    if (root/'bootstrap/STOP').exists() or (root/'repeat/STOP').exists() or (interrupted and interrupted()):
        _state(root,'stopped')
        return 'stopped'
    if repeat.stage_finished(root,'a1'):
        _state(root,'completed',stage='a1')
        return 'completed'
    if cache_ready(request):
        pid = repeat.launch(root,training_request(request),env=_environment(request))
        _state(root,'training',stage='a1',training_controller_pid=pid,cache=cache_progress(request['cache_root']))
        _event(root,'a1-handoff',training_controller_pid=pid)
        return 'training'
    before = cache_progress(request['cache_root'])
    _state(root,'preparing',cycle=request['cycle']+1,cache=before)
    elapsed = time.monotonic()-(started or time.monotonic())
    # Total preparation-manager lifetime includes grace and cleanup. The child retains
    # its documented 1200-second internal budget; the owner may stop it earlier.
    remaining = max(.02,request['cycle_seconds']-elapsed-10)
    grace = min(request['grace_seconds'],max(.01,remaining/3))
    cycle_request = dict(gpu_ids=request['gpu_ids'],cycle_seconds=max(.01,remaining-grace),grace_seconds=grace)
    result = repeat.run_cycle(root,'_prepare',cycle_request,request['cycle']+1,
                              command=prepare_command(request),
                              interrupted=lambda: (interrupted and interrupted()) or (root/'bootstrap/STOP').exists())
    after = cache_progress(request['cache_root'])
    if result['reason']=='manual-stop' or (root/'bootstrap/STOP').exists() or (root/'repeat/STOP').exists():
        _state(root,'stopped',cache=after,**result)
        return 'stopped'
    if result.get('fatal') or (result['returncode'] not in (0,75) and not result.get('forced') and not result.get('killed')):
        raise RuntimeError(f'Preparation child failed: {result}; see repeat/latest-training.log')
    gained = after['image']+after['text'] > before['image']+before['text']
    ready_after = cache_ready(request)
    request['stalled_cycles'] = 0 if gained or ready_after else request['stalled_cycles']+1
    request['cycle'] += 1
    if request['stalled_cycles'] >= 3:
        raise RuntimeError('Preparation added no cache files for three consecutive cycles; refusing an endless restart loop')
    _state(root,'restarting',cycle=request['cycle'],stalled_cycles=request['stalled_cycles'],cache=after,**result)
    _event(root,'prepare-cycle-end',cycle=request['cycle'],cache=after,**result)
    return 'restarting'


def spawn_worker(root,fd,request=None):
    with (root/'bootstrap/controller.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-m','flux3_qdm.bootstrap','--worker','--output-root',str(root),
                                 '--lease-fd',str(fd)],cwd=REPO,pass_fds=(fd,),start_new_session=True,
                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                env=_environment(request))


def _repeat_active(root):
    control = root/'repeat'
    control.mkdir(parents=True,exist_ok=True)
    with (control/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    state = repeat.read_json(control/'state.json')
    return repeat.owned_process(state.get('training_pid',0),state.get('training_start_ticks'))


def launch(root,request):
    root = Path(root).resolve()
    (root/'bootstrap').mkdir(parents=True,exist_ok=True)
    with (root/'bootstrap.lock').open('a') as lease:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'A bootstrap controller is already running for {root}') from None
        if _repeat_active(root):
            raise RuntimeError('A repeat controller or its owned child is already running; refuse duplicate preparation/training')
        if repeat.stage_finished(root,'a1'):
            _state(root,'completed',stage='a1')
            print('A1 already completed; bootstrap stops here')
            return None
        (root/'bootstrap/STOP').unlink(missing_ok=True)
        (root/'repeat/STOP').unlink(missing_ok=True)
        request = {**request,'cycle':0,'stalled_cycles':0}
        repeat.write_json(root/'bootstrap/request.json',request)
        process = spawn_worker(root,lease.fileno(),request)
        _event(root,'bootstrap-start',worker_pid=process.pid)
        print(f'Preparation → A1 controller PID {process.pid}; log {root / "bootstrap/controller.log"}',flush=True)
        return process.pid


def stop(root):
    root = Path(root).resolve()
    for folder in ['bootstrap','repeat']:
        (root/folder).mkdir(parents=True,exist_ok=True)
        (root/folder/'STOP').touch()
    print('Stop requested for cache preparation and training; owned children stop before renewal',flush=True)


def status(root,cache_root=None):
    root = Path(root).resolve()
    request = repeat.read_json(root/'bootstrap/request.json')
    state = repeat.read_json(root/'bootstrap/state.json')
    state.setdefault('phase','not-started')
    state['controller_alive'] = repeat.owned_process(state.get('controller_pid',0),state.get('controller_start_ticks'))
    training = repeat.read_json(root/'repeat/state.json')
    if state.get('phase')=='training':
        state['training'] = training
        if repeat.stage_finished(root,'a1'):
            state['phase']='completed'
        elif training.get('status') in ('stopped','failed','completed'):
            state['phase']=training['status']
    state['cache'] = cache_progress(cache_root or request.get('cache_root',''))
    state['stop_requested'] = (root/'bootstrap/STOP').exists() or (root/'repeat/STOP').exists()
    return state


def worker(root,fd):
    root = Path(root).resolve()
    started = time.monotonic()
    interrupted = {'value':False}
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(sig,lambda *_: interrupted.update(value=True))
    with os.fdopen(fd,'a') as lease:
        request = repeat.read_json(root/'bootstrap/request.json')
        try:
            action = run_once(root,request,lambda:interrupted['value'],started)
            if action != 'restarting':
                return
            repeat.write_json(root/'bootstrap/request.json',request)
            deadline = min(started+request['cycle_seconds']-1,time.monotonic()+request['restart_delay'])
            while time.monotonic()<deadline:
                if interrupted['value'] or (root/'bootstrap/STOP').exists() or (root/'repeat/STOP').exists():
                    _state(root,'stopped')
                    return
                time.sleep(min(.1,max(0,deadline-time.monotonic())))
            successor = spawn_worker(root,lease.fileno(),request)
            _event(root,'bootstrap-handoff',successor_pid=successor.pid,cycle=request['cycle'])
        except Exception as exc:
            _state(root,'failed',error_type=type(exc).__name__,error=str(exc),cycle=request.get('cycle',0))
            _event(root,'failed',error_type=type(exc).__name__,error=str(exc))
            raise


def main():
    from .resources import DEFAULT_WEIGHTS_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    dataset = Path(os.getenv('FLUX3_DATASET_ROOT',str(Path.home()/'dataset/MultiAspect-4K-1M')))
    p.add_argument('--output-root',default=os.getenv('FLUX3_QDM_RUN','outputs/flux3-qdm'))
    p.add_argument('--weights-root',default=DEFAULT_WEIGHTS_ROOT)
    p.add_argument('--subset-root',default=os.getenv('FLUX3_SUBSET_ROOT',str(dataset/'pid_vdn_subsets')))
    p.add_argument('--cache-root',default=os.getenv('FLUX3_CACHE_ROOT',str(dataset/'flux3_qdm_cache')))
    p.add_argument('--gpu-ids',default=os.getenv('GPU_IDS','0'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--cycle-seconds',type=float,default=1200)
    p.add_argument('--grace-seconds',type=float,default=180)
    p.add_argument('--restart-delay',type=float,default=5)
    p.add_argument('--stop',action='store_true')
    p.add_argument('--status',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--lease-fd',type=int,help=argparse.SUPPRESS)
    args = p.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    if args.worker:
        worker(root,args.lease_fd)
        return
    if args.stop:
        stop(root)
        return
    if args.status:
        print(json.dumps(status(root,args.cache_root),indent=2))
        return
    ids = [s.strip() for s in args.gpu_ids.split(',')]
    if not ids or any(not s.isdigit() for s in ids) or len(set(ids))!=len(ids):
        p.error('--gpu-ids must contain distinct nonnegative IDs')
    if min(args.workers,args.threads)<1 or not 20<args.cycle_seconds<=1200 or not 0<args.grace_seconds<args.cycle_seconds-10 or args.restart_delay<0:
        p.error('Positive worker/thread limits and 20 < cycle-seconds <= 1200 with smaller grace are required')
    request = dict(gpu_ids=','.join(ids),workers=args.workers,threads=args.threads,
                   cycle_seconds=args.cycle_seconds,grace_seconds=args.grace_seconds,restart_delay=args.restart_delay,
                   **{key:str(Path(getattr(args,key)).expanduser().resolve()) for key in ['weights_root','subset_root','cache_root']})
    launch(root,request)


if __name__=='__main__':
    main()
