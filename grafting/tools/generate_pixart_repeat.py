#!/usr/bin/env python3
"""Background generation controller with renewable PID and per-image resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from generate_pixart_dataset import atomic_json
from generate_pixart_handoff import local_start_gate


def next_status(returncode, finished, manual, progressed, stalled):
    stalled = 0 if progressed else stalled + 1
    if manual:
        return 'stopped', stalled
    if returncode:
        return 'failed', stalled
    if finished:
        return 'completed', stalled
    if stalled >= 3:
        return 'failed', stalled
    return 'restarting', stalled


def retry_status(returncode, finished, manual, progressed, stalled, failures,
                 max_retries, error):
    status, stalled = next_status(returncode, finished, manual, progressed, stalled)
    failures = failures + 1 if returncode else 0
    # A fresh process resets the poisoned CUDA context. This is recovery,
    # not a fix for the underlying illegal-access kernel error.
    if (status == 'failed' and returncode and stalled < 3
            and failures <= max_retries
            and 'an illegal memory access' in error.lower()
            and 'out of memory' not in error.lower()):
        status = 'restarting'
    return status, stalled, failures


def spawn(root, fd):
    with (root / 'repeat/controller.log').open('a') as log:
        return subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--output-root', str(root), '--worker', '--lease-fd', str(fd)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(fd,))


def run_worker(root, fd):
    control = root / 'repeat'
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def stop(*_):
        (control / 'STOP').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with os.fdopen(fd, 'a') as lease:
        request = json.loads((control / 'request.json').read_text())
        if (control / 'STOP').exists():
            atomic_json(control / 'state.json', {'status': 'stopped'})
            return
        cycle = request.get('cycle', 0) + 1
        before = len(list((root / 'metadata').glob('*.json')))
        log_path = control / f'cycle_{cycle:06d}_{time.time_ns()}.log'
        latest = control / 'latest-generation.log'
        latest.unlink(missing_ok=True)
        latest.symlink_to(log_path.name)
        command = [sys.executable, '-u', str(Path(__file__).with_name('generate_pixart_dataset.py')),
                   '--output-root', str(root), *request['generator_args'],
                   '--max-seconds', str(max(1, request['cycle_seconds'] - 90))]
        env = dict(os.environ, **request['env'])
        with log_path.open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, env=env, start_new_session=True)
        start = time.monotonic()
        requested_at = None
        state = dict(status='running', cycle=cycle, controller_pid=os.getpid(),
                     generation_pid=process.pid, log=str(log_path), started_at=time.time())
        try:
            while process.poll() is None:
                now = time.monotonic()
                if requested_at is None and ((control / 'STOP').exists() or now-start >= request['cycle_seconds']):
                    process.send_signal(signal.SIGTERM)
                    requested_at = now
                if requested_at is not None and now-requested_at > request['grace_seconds']:
                    # Only this controller's generation session is terminated.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break
                state['complete'] = len(list((root / 'metadata').glob('*.json')))
                state['updated_at'] = time.time()
                atomic_json(control / 'state.json', state)
                time.sleep(2)
        except BaseException:
            stop()
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            raise
        after = len(list((root / 'metadata').glob('*.json')))
        status_path = root / 'status.json'
        summary = json.loads(status_path.read_text()) if status_path.exists() else {}
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            error = log.read().decode(errors='replace')
        status, stalled, failures = retry_status(
            process.returncode, summary.get('finished', False),
            (control / 'STOP').exists(), after > before, request.get('stalled', 0),
            request.get('cuda_failures', 0), request.get('max_cuda_retries', 3), error)
        state.update(status=status, complete=after, returncode=process.returncode,
                     stalled=stalled, cuda_failures=failures)
        atomic_json(control / 'state.json', state)
        if status == 'restarting':
            request.update(cycle=cycle, stalled=stalled, cuda_failures=failures)
            atomic_json(control / 'request.json', request)
            # New process/session/PID, then release this controller's copy of the lease.
            successor = spawn(root, lease.fileno())
            print(f'Handoff to controller {successor.pid}', flush=True)
        else:
            print(json.dumps(state), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--cycle-seconds', type=int, default=1200)
    p.add_argument('--grace-seconds', type=int, default=180)
    p.add_argument('--max-cuda-retries', type=int, default=3,
                   help='Bounded fresh-process retries for CUDA illegal memory access; OOM is not retried')
    p.add_argument('--status', action='store_true')
    p.add_argument('--stop', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--lease-fd', type=int, default=-1, help=argparse.SUPPRESS)
    args, remaining = p.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    root = args.output_root.resolve()
    control = root / 'repeat'
    control.mkdir(parents=True, exist_ok=True)
    if args.stop:
        (control / 'STOP').touch()
        print('Stop requested; current batch will finish, then automatic restart stops.')
        return
    if args.status:
        print((control / 'state.json').read_text() if (control / 'state.json').exists() else 'Not started')
        return
    if args.worker:
        run_worker(root, args.lease_fd)
        return
    if not remaining:
        p.error('Provide generation arguments after --')
    if args.max_cuda_retries < 0:
        p.error('max-cuda-retries must be nonnegative')
    if not 120 <= args.cycle_seconds <= 1400 or not 10 <= args.grace_seconds <= 240:
        p.error('Use cycle-seconds 120..1400 and grace-seconds 10..240 on node01')
    with local_start_gate(root), (control / 'controller.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This output directory already has a generation controller.')
        (control / 'STOP').unlink(missing_ok=True)
        env = {k: os.environ[k] for k in ['GPU_IDS', 'CPU_THREADS', 'OMP_NUM_THREADS',
               'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_LAUNCH_BLOCKING'] if k in os.environ}
        request = dict(generator_args=remaining, cycle_seconds=args.cycle_seconds,
                       grace_seconds=args.grace_seconds, env=env, cycle=0, stalled=0,
                       max_cuda_retries=args.max_cuda_retries, cuda_failures=0)
        atomic_json(control / 'request.json', request)
        process = spawn(root, lease.fileno())
        print(f'Background generation controller PID {process.pid}\nLogs: {control}/latest-generation.log')


if __name__ == '__main__':
    main()
