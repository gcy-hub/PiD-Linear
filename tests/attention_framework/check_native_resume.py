"""Native four-rank training-only exact resume regression from a resolved config."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from training.checkpoint import latest, read


def equal(a, b, path=''):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f'{path}/{i}')
    else:
        assert a == b, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ends = []
    for label, every in [('continuous', 100), ('resumed', 1)]:
        cfg = OmegaConf.load(args.config)
        root = args.output.resolve() / label
        root.mkdir(parents=True, exist_ok=False)
        cfg.paths.output = str(root)
        cfg.checkpoint.every = every
        resolved = root / 'input.yaml'
        OmegaConf.save(cfg, resolved)
        resume = None
        while True:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=4', '-m', 'training.run', '--worker-config', str(resolved)]
            if resume:
                command += ['--resume', str(resume)]
            subprocess.run(command, cwd=ROOT, check=True)
            resume = latest(root)
            if json.loads((resume / 'complete.json').read_text())['done']:
                break
        ends.append(resume)
    for rank in range(4):
        a, b = [read(end, rank, 4) for end in ends]
        for key in ('weights', 'optimizer', 'scheduler', 'progress', 'rng'):
            equal(a[key], b[key], f'rank{rank}/{key}')
        print(f'rank {rank}: native weights/optimizer/scheduler/progress/RNG exactly equal', flush=True)
    (args.output / 'result.json').write_text(json.dumps(dict(status='passed', ranks=4,
        config=str(args.config), comparison='torch.equal; no tolerance')))


if __name__ == '__main__':
    main()
