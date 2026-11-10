"""Launch complete training processes, then independent validation processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf
from attention_framework.config import config_from_args
from .checkpoint import latest


def main():
    if '--worker-config' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker-config', required=True)
        parser.add_argument('--resume')
        args = parser.parse_args()
        from .engine import segment
        segment(OmegaConf.load(args.worker_config), args.resume)
        return
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
        OmegaConf.save(config, resolved)
        resume = latest(root) or (Path(config.checkpoint.resume) if config.checkpoint.resume else None)
        while True:
            previous_step = -1
            if resume:
                status = json.loads((resume / 'complete.json').read_text())
                previous_step = status['step']
                if status['pending_validation'] and not (resume / 'validated.json').exists():
                    env = dict(os.environ)
                    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
                        env.pop(key, None)
                    subprocess.run([sys.executable, '-m', 'training.validate', '--config', str(resolved), '--checkpoint', str(resume)], env=env, check=True)
                if status['done']:
                    return
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc_per_node={config.runtime.gpus}', '-m', 'training.run', '--worker-config', str(resolved)]
            if resume:
                command += ['--resume', str(resume)]
            subprocess.run(command, check=True)
            resume = latest(root)
            if resume is None or json.loads((resume / 'complete.json').read_text())['step'] <= previous_step:
                raise RuntimeError('Training exited without publishing progress')


if __name__ == '__main__':
    main()
