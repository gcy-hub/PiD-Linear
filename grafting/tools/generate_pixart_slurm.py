#!/usr/bin/env python3
"""One shared coordinator and one independently sharded PixArt worker per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from generate_pixart_dataset import atomic_json, digest, write_manifests
from generate_pixart_handoff import claim_cluster_output


def slurm_command(root, generator_args, nodes, gpus_per_node, cpus_per_task):
    return ['srun', f'--nodes={nodes}', f'--ntasks={nodes}',
            '--ntasks-per-node=1', f'--gpus-per-node={gpus_per_node}',
            f'--cpus-per-task={cpus_per_task}', '--gpu-bind=none',
            '--kill-on-bad-exit=1', '--label', 'bash',
            str(Path(__file__).with_name('generate_pixart_slurm_worker.sh')),
            '--gpus-per-node', str(gpus_per_node), '--output-root', str(root), *generator_args]


def preflight_nodes(command, root, nodes):
    # Uses the same Slurm topology and worker device mapping as real generation.
    # This must finish before touching any node01 stop marker.
    command = list(command)
    command.insert(command.index('--output-root'), '--preflight')
    subprocess.run(command, check=True, timeout=240)
    reports = [json.loads((root / 'slurm/preflight' / os.environ['SLURM_JOB_ID'] /
                           f'node_{node}.json').read_text()) for node in range(nodes)]
    if any(r['status'] != 'passed' for r in reports):
        raise RuntimeError('Some allocated nodes did not pass GPU preflight')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--nodes', type=int, default=2)
    p.add_argument('--gpus-per-node', type=int, default=4)
    p.add_argument('--cpus-per-task', type=int, default=16)
    p.add_argument('--command-only', action='store_true')
    args, remaining = p.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    if not remaining:
        p.error('Provide generator arguments after --')
    root = args.output_root.resolve()
    command = slurm_command(root, remaining, args.nodes, args.gpus_per_node, args.cpus_per_task)
    if args.command_only:
        print(json.dumps(command))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run this coordinator inside the provided Slurm job')
    control = root / 'slurm'
    control.mkdir(parents=True, exist_ok=True)
    stop = [False]

    def request_stop(*_):
        stop[0] = True
        (root / '.stop-requested').touch()
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, request_stop)
    atomic_json(control / 'state.json', dict(status='checking-allocated-gpus',
                job_id=os.environ['SLURM_JOB_ID'], updated_at=time.time()))
    try:
        preflight_nodes(command, root, args.nodes)
        atomic_json(control / 'state.json', dict(status='waiting-for-node01',
                    job_id=os.environ['SLURM_JOB_ID'], updated_at=time.time()))
        with claim_cluster_output(root, os.environ['SLURM_JOB_ID'], interrupted=lambda: stop[0]):
            run_generation(root, control, remaining, command, stop)
    except BaseException as exc:
        atomic_json(control / 'state.json', dict(status='failed', job_id=os.environ['SLURM_JOB_ID'],
                    error=str(exc), updated_at=time.time()))
        raise


def run_generation(root, control, remaining, command, stop):
    # Reuse the same recipe and IDs as node01 before dispatching any remote GPU.
    prepare = [sys.executable, '-u', str(Path(__file__).with_name('generate_pixart_dataset.py')),
               '--output-root', str(root), *remaining, '--plan-only']
    subprocess.run(prepare, check=True)
    records = json.loads((root / 'records.json').read_text())
    recipe = json.loads((root / 'generation_config.json').read_text())
    with (root / '.generation.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another generator is already using this dataset directory.')
        if not stop[0]:
            (root / '.stop-requested').unlink(missing_ok=True)
        if json.loads((root / 'status.json').read_text())['finished']:
            atomic_json(control / 'state.json', {'status': 'completed', 'job_id': os.environ['SLURM_JOB_ID']})
            return
        if stop[0]:
            return
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        state = dict(status='running', job_id=os.environ['SLURM_JOB_ID'],
                     nodes=os.environ.get('SLURM_JOB_NODELIST'),
                     world_size=int(os.environ['SLURM_NTASKS']) * int(command[command.index('--gpus-per-node') + 1]),
                     started_at=time.time(), coordinator_pid=os.getpid(), srun_pid=process.pid)
        last_manifest = time.monotonic()
        while process.poll() is None:
            state['complete'] = len(list((root / 'metadata').glob('*.json')))
            state['total'] = len(records)
            state['updated_at'] = time.time()
            state['stop_requested'] = stop[0]
            atomic_json(control / 'state.json', state)
            if time.monotonic()-last_manifest >= 300:
                write_manifests(root, records, digest(recipe), recipe['width'], recipe['height'], recipe['stage1_size'])
                last_manifest = time.monotonic()
            time.sleep(2)
        complete = write_manifests(root, records, digest(recipe), recipe['width'], recipe['height'], recipe['stage1_size'])
        state.update(complete=complete, total=len(records), returncode=process.returncode,
                     status='failed' if process.returncode else 'completed' if complete == len(records) else 'paused',
                     finished_at=time.time())
        atomic_json(control / 'state.json', state)
        print(json.dumps(state), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
