"""Wait for a committed gallery, stop its repeat controller, then launch Sliding A1.

This process renews itself every 20 minutes. It never kills arbitrary GPU users
and never submits Slurm. STOP cancels a pending handoff without stopping training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def validation_images_complete(source, stage, step):
    source = Path(source)
    config = read(source / stage / 'config.json')
    names = [name.strip() for name in config.get('validation_cases', '').split(',') if name.strip()]
    if not names or len(names) != len(set(names)):
        return False
    resolution = config.get('validation_resolution', 4096)
    root = source / stage / 'galleries' / f'step_{step:09d}'
    for name in names:
        stem = f'{name}_{resolution}'
        if not (root / (stem + '.png')).is_file():
            return False
        done = read(root / (stem + '.done.json'))
        if done.get('settings', {}).get('step') != step:
            return False
    return True


def completed_checkpoint(source, stage, step):
    if not validation_images_complete(source, stage, step):
        return None
    from pid._src.vdn_pid.checkpoint import latest_state, load_state
    latest = latest_state(Path(source) / stage)
    if latest is None:
        return None
    state = load_state(latest)
    progress = state['progress']
    if progress['stage_step'] >= step and not progress.get('validation_pending', False):
        return str(latest)
    return None


def worker(root, fd):
    control = root / 'handoff'
    request = read(control / 'request.json')
    source = Path(request['source_root'])
    started = time.monotonic()

    def state(phase, **kw):
        write(control / 'state.json', dict(phase=phase, pid=os.getpid(), time=time.time(), **kw))
        print(f'{time.strftime("%F %T")} {phase}', flush=True)

    with os.fdopen(fd, 'a') as lease:
        previous = read(control / 'state.json')
        if previous.get('phase') == 'launched':
            return
        stopping = previous.get('phase') == 'stopping-source'
        stop_started = previous.get('stop_requested_time', time.time())
        if not stopping:
            state('waiting-validation', source_root=str(source), step=request['step'])
        while True:
            if (control / 'STOP').exists():
                state('cancelled')
                return
            if not stopping:
                checkpoint = completed_checkpoint(source, request['source_stage'], request['step'])
                if checkpoint:
                    # The repeat controller saves and stops on its next update boundary.
                    (source / 'repeat' / 'STOP').touch()
                    stopping = True
                    stop_started = time.time()
                    state('stopping-source', completed_checkpoint=checkpoint, stop_requested_time=stop_started)
                else:
                    source_state = read(source / 'repeat' / 'state.json')
                    if source_state.get('status') in ('failed', 'stopped') or (source / 'repeat' / 'STOP').exists():
                        state('blocked', reason='Source stopped before committing the required validation; no training was launched')
                        return
            if stopping:
                from scripts.repeat_pid_vdn import process_identity
                source_state = read(source / 'repeat' / 'state.json')
                process = process_identity(source_state.get('training_pid', 0))
                alive = process and process[0] == source_state.get('training_start_ticks')
                if source_state.get('status') == 'stopped' and not alive:
                    log = (control / 'launch.log').open('a')
                    process = subprocess.Popen(['bash', request['command_file']], cwd=REPO,
                        start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    log.close()
                    state('launched', launcher_pid=process.pid, command_file=request['command_file'])
                    return
                if time.time() - stop_started > 600:
                    state('failed', reason='Source controller did not confirm a clean stop within 600 seconds; target not launched')
                    return
            if time.monotonic() - started >= 1200:
                spawn(root, lease.fileno())
                return
            time.sleep(10)


def spawn(root, fd):
    with (root / 'handoff' / 'controller.log').open('a') as log:
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
            '--target-root', str(root), '--lease-fd', str(fd)], cwd=REPO, pass_fds=(fd,),
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).pid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root')
    p.add_argument('--source-stage', default='b')
    p.add_argument('--step', type=int, default=250)
    p.add_argument('--target-root', required=True)
    p.add_argument('--command-file')
    p.add_argument('--status', action='store_true')
    p.add_argument('--stop', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    root = Path(args.target_root).resolve()
    control = root / 'handoff'
    control.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps(read(control / 'state.json'), ensure_ascii=False, indent=2))
    elif args.stop:
        (control / 'STOP').touch()
        print('Pending handoff cancelled; this does not stop an already launched training run')
    elif args.worker:
        try:
            worker(root, args.lease_fd)
        except Exception as error:
            write(control / 'state.json', dict(phase='failed', error=f'{type(error).__name__}: {error}', time=time.time()))
            raise
    else:
        if not args.source_root or not args.command_file or args.step < 1:
            p.error('Provide --source-root, --command-file and a positive --step')
        source = Path(args.source_root).resolve()
        command = Path(args.command_file).resolve()
        if source == root or not command.is_file():
            p.error('Source and target must differ and the command file must exist')
        with (control / 'controller.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(control / 'state.json').get('phase') == 'launched':
                raise RuntimeError('Handoff already launched; use target repeat status/stop commands')
            (control / 'STOP').unlink(missing_ok=True)
            write(control / 'request.json', dict(source_root=str(source), source_stage=args.source_stage,
                  step=args.step, command_file=str(command)))
            print(f'Waiting for {source}/{args.source_stage} step {args.step}; controller PID {spawn(root, lease.fileno())}')


if __name__ == '__main__':
    main()
