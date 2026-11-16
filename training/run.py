"""Launch resident training workers and resume complete checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from omegaconf import OmegaConf
from attention_framework.config import config_from_args, validate_config
from .checkpoint import latest, atomic_json


def failure_kind(exc):
    import torch
    if 'illegal memory access' in str(exc).lower():
        return 'cuda_restart'
    if not isinstance(exc, torch.cuda.OutOfMemoryError):
        return 'error'
    # CUDA's OOM report lists other GPU processes separately from this process.
    others = re.findall(r'\bProcess (\d+) has [\d.]+ [KMGT]iB memory in use', str(exc))
    return 'gpu_busy' if any(int(pid) != os.getpid() for pid in others) else 'cuda_oom'


def main():
    if '--worker-config' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker-config', required=True)
        parser.add_argument('--resume')
        args = parser.parse_args()
        config = OmegaConf.load(args.worker_config)
        try:
            from .engine import segment
            segment(config, args.resume)
        except Exception as exc:
            atomic_json(Path(config.paths.output) / f"failure-rank-{os.environ.get('RANK', '0')}.json",
                        dict(kind=failure_kind(exc), error=str(exc)))
            raise
        return
    if '--config' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args = parser.parse_args()
        config = OmegaConf.load(args.config)
        validate_config(config)
    else:
        args, config = config_from_args()
        if args.print_config:
            print(OmegaConf.to_yaml(config, resolve=True))
            return
    import fcntl
    root = Path(config.paths.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resolved = root / 'resolved.yaml'
        if (root / 'batch-reduced.json').exists():
            config.training.batch_size = 1
            config.global_batch_size = config.runtime.gpus * config.training.accumulation
        while not (root / 'STOP').exists():
            OmegaConf.save(config, resolved)
            resume = latest(root) or (Path(config.checkpoint.resume) if config.checkpoint.resume else None)
            if resume:
                status = json.loads((resume / 'complete.json').read_text())
                if status['done'] and (not status['pending_validation'] or (resume / 'validated.json').exists()):
                    atomic_json(root / 'DONE.json', status)
                    return
            for failure in root.glob('failure-rank-*.json'):
                failure.unlink()
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc_per_node={config.runtime.gpus}', '-m', 'training.run', '--worker-config', str(resolved)]
            if resume:
                command += ['--resume', str(resume)]
            with (root / 'worker.log').open('a+', encoding='utf-8', errors='replace') as log:
                start = log.tell()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                log.seek(max(start, log.tell()-16384))
                tail = log.read()
            failures = [json.loads(p.read_text()) for p in root.glob('failure-rank-*.json')]
            if result.returncode:
                if any(f['kind'] == 'cuda_restart' for f in failures):
                    atomic_json(root / 'launcher-result.json', dict(state='interrupted',
                                                                  returncode=result.returncode, failures=failures))
                    raise SystemExit(75)
                if failures and all(f['kind'] == 'gpu_busy' for f in failures):
                    atomic_json(root / 'launcher-result.json', dict(state='waiting-for-gpu',
                                                                  returncode=result.returncode, failures=failures))
                    raise SystemExit(75)
                if any(f['kind'] == 'cuda_oom' for f in failures) and config.training.batch_size == 2:
                    atomic_json(root / 'batch-reduced.json', dict(reason='cuda_oom', previous=2, batch_size=1,
                                                                checkpoint=str(resume), failures=failures))
                    config.training.batch_size = 1
                    config.global_batch_size = config.runtime.gpus * config.training.accumulation
                    continue
                interrupted = not failures and (result.returncode in (-9, -15, 137, 143)
                    or bool(re.search(r'exitcode\s*:\s*-(9|15)\b|Signal (9|15)\b', tail)))
                atomic_json(root / 'launcher-result.json', dict(state='interrupted' if interrupted else 'error',
                                                               returncode=result.returncode, failures=failures))
                raise SystemExit(75 if interrupted else 1)
            if (root / 'STOP').exists():
                return
            current = latest(root)
            if current is None or current == resume:
                raise RuntimeError('Training exited without publishing progress')


if __name__ == '__main__':
    main()
