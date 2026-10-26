#!/usr/bin/env python3
"""One Slurm task owns the node's GPUs and launches isolated global-rank workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from generate_pixart_dataset import atomic_json, model_identity


def worker_specs(node_rank, nodes, gpus_per_node, environment):
    devices = [v.strip() for v in environment.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    if not 0 <= node_rank < nodes:
        raise ValueError(f'Invalid node rank {node_rank}/{nodes}')
    if len(devices) != gpus_per_node or len(set(devices)) != gpus_per_node:
        raise ValueError(f'Expected {gpus_per_node} distinct allocated GPUs, got CUDA_VISIBLE_DEVICES={devices}')
    result = []
    for local_rank, device in enumerate(devices):
        env = dict(environment, CUDA_VISIBLE_DEVICES=device,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        # This legacy second mask can conflict with CUDA_VISIBLE_DEVICES.
        env.pop('GPU_DEVICE_ORDINAL', None)
        env.pop('GPU_IDS', None)
        result.append((node_rank * gpus_per_node + local_rank, nodes * gpus_per_node, env))
    return result


def gpu_preflight(specs, timeout=180):
    processes = []
    deadline = time.monotonic() + timeout
    try:
        for rank, world, env in specs:
            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('generate_pixart_gpu_probe.py'))],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            processes.append((rank, p))
        reports = []
        for rank, p in processes:
            out, err = p.communicate(timeout=max(.1, deadline - time.monotonic()))
            if p.returncode:
                raise RuntimeError(f'CUDA probe failed for global rank {rank}: {err[-4000:]}')
            report = json.loads(out.strip().splitlines()[-1])
            report['rank'] = rank
            reports.append(report)
        if len({r['uuid'] for r in reports}) != len(specs):
            raise RuntimeError('Multiple workers were assigned the same physical GPU')
        return reports
    finally:
        for _, p in processes:
            if p.poll() is None:
                p.kill()
            p.wait()


def run_workers(specs, root, generator_args, grace_seconds=180):
    processes = []
    stopping_at = None
    requested = False

    def request_stop(*_):
        nonlocal requested
        requested = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    try:
        for rank, world, env in specs:
            command = [sys.executable, '-u', str(Path(__file__).with_name('generate_pixart_dataset.py')),
                       '--output-root', str(root), *generator_args,
                       '--worker-rank', str(rank), '--world-size', str(world)]
            print(f'Starting global rank {rank}/{world} on {socket.gethostname()} '
                  f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}', flush=True)
            processes.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL))
        while any(p.poll() is None for p in processes):
            bad = any(p.poll() not in (None, 0) for p in processes)
            if (requested or bad or (root / '.stop-requested').exists()) and stopping_at is None:
                (root / '.stop-requested').touch()
                stopping_at = time.monotonic()
                for p in processes:
                    if p.poll() is None:
                        p.send_signal(signal.SIGTERM)
            if stopping_at is not None and time.monotonic() - stopping_at > grace_seconds:
                for p in processes:
                    if p.poll() is None:
                        p.kill()
            time.sleep(1)
        return 1 if any(p.returncode != 0 for p in processes) else 0
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus-per-node', type=int, default=4)
    p.add_argument('--preflight', action='store_true')
    args, remaining = p.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    specs = worker_specs(int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS']),
                         args.gpus_per_node, os.environ)
    if args.preflight:
        opts = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        opts.add_argument('--model-dir', type=Path, required=True)
        opts.add_argument('--prompts', type=Path, required=True)
        paths, _ = opts.parse_known_args(remaining)
        model_identity(paths.model_dir)
        if not paths.prompts.is_file() or not (args.output_root / 'generation_config.json').is_file():
            raise FileNotFoundError('Model, prompts and existing generation recipe must be visible on every node')
        report = dict(status='passed', host=socket.gethostname(),
                      job_id=os.environ['SLURM_JOB_ID'], reports=gpu_preflight(specs), updated_at=time.time())
        destination = args.output_root / 'slurm/preflight' / os.environ['SLURM_JOB_ID']
        atomic_json(destination / f'node_{os.environ["SLURM_PROCID"]}.json', report)
        print(json.dumps(report), flush=True)
        return
    raise SystemExit(run_workers(specs, args.output_root, remaining))


if __name__ == '__main__':
    main()
