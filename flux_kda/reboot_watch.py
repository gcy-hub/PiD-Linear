"""Short cron checks to resume FLUX2 KDA training after node process cleanup."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'flux_kda/train_flux2_node01.sh'


def active_workers(root, launcher=LAUNCHER):
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if not any(x in argv for x in ('flux_kda.train', 'flux_kda.benchmark')):
                continue
            output = Path(argv[argv.index('--output') + 1])
            if not output.is_absolute():
                output = proc.joinpath('cwd').resolve() / output
            if output.resolve().is_relative_to(root):
                return True
        except (OSError, ValueError, UnicodeError, IndexError):
            continue
    pidfile = root / 'launcher.pid'
    if pidfile.exists():
        try:
            argv = Path('/proc', pidfile.read_text().strip(), 'cmdline').read_bytes().decode().split('\0')
            return any(Path(x).name == launcher.name for x in argv if x)
        except (OSError, UnicodeError):
            pass
    return False


def check(root, launcher=LAUNCHER):
    if (root / 'STOP').exists():
        return 'stopped'
    final_step = 10000 if launcher.name == 'train_flux3_node01.sh' else 1000
    if (root / f'validation/step-{final_step}/kda.json').exists():
        return 'completed'
    if active_workers(root, launcher):
        return 'running'
    with (root / 'launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'running'
    exitfile = root / 'exit-code'
    if exitfile.exists() and int(exitfile.read_text()) not in (0, 137, 143):
        return 'training-error'
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in result.stdout.splitlines()}
    min_free_mib = 40960 if launcher.name == 'train_flux3_node01.sh' else 40000
    if any(free.get(i, 0) < min_free_mib for i in (1, 2, 3)):
        return 'waiting-for-gpus'
    with (root / 'run.log').open('a') as log:
        child = subprocess.Popen(['bash', str(launcher), str(root)], cwd=REPO,
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
    (root / 'launcher.pid').write_text(str(child.pid) + '\n')
    return 'resumed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('flux2', 'flux3'), default='flux2')
    parser.add_argument('--host', default=socket.gethostname().split('.')[0])
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != args.host:
        return
    root = args.output.resolve()
    with (root / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = check(root, REPO / f'flux_kda/train_{args.family}_node01.sh')
        temporary = root / 'watch-status.partial.json'
        temporary.write_text(json.dumps(dict(status=state, time=time.time())) + '\n')
        temporary.replace(root / 'watch-status.json')


if __name__ == '__main__':
    main()
