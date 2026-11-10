"""Run training segments and standalone validation, with no live NCCL group while sampling."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from .train import parse_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    cli = parser.parse_args()
    config_path = Path(cli.config).resolve()
    config = json.loads(config_path.read_text())
    args = parse_args(['--config', str(config_path)])
    gpus = config.get('gpus', 8)
    if not isinstance(gpus, int) or gpus < 1:
        raise ValueError('gpus must be a positive integer; CUDA_VISIBLE_DEVICES selects physical devices')
    validation = config.get('validation', {})
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = root / 'latest.pt'
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(repo), str(repo/'flux2'), str(repo/'flux3'), env.get('PYTHONPATH','')])
    step = 0
    resume = str(Path(args.resume).resolve()) if args.resume else None
    if checkpoint.exists():
        resume = str(checkpoint)
    if resume:
        state = torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
        step = state['step']
        del state
    while True:
        # Finish interrupted validation before advancing to another checkpoint.
        if step and (step % args.save_every == 0 or step == args.steps):
            out = root / 'validation' / f'step-{step}'
            if not (out/'kda.json').exists():
                selected = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
                subprocess.run([sys.executable, '-u', '-m', 'flux_kda.benchmark',
                    '--family',args.family,'--weights',args.weights,'--resources',args.resources,
                    '--output',str(out),'--checkpoint',str(resume),'--variant','kda',
                    '--size',str(validation.get('size',4096)), '--steps',str(validation.get('steps',20)),
                    '--cfg',str(validation.get('cfg',3.5)), '--seed',str(validation.get('seed',0)),
                    '--prompt',validation.get('prompt','A red double-decker bus on a rainy London street'),
                    '--no-condition-cache'], cwd=repo, env=dict(env,CUDA_VISIBLE_DEVICES=selected), check=True)
        if step >= args.steps:
            break
        target = min((step//args.save_every+1)*args.save_every,args.steps)
        command = [sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpus}',
                   '-m','flux_kda.train','--config',str(config_path),'--output',str(root),'--steps',str(target)]
        if resume:
            command += ['--resume',str(resume)]
        subprocess.run(command,cwd=repo,env=env,check=True)
        state = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if state['step'] != target:
            raise RuntimeError(f'Training did not publish checkpoint {target}')
        del state
        step, resume = target, str(checkpoint)


if __name__ == '__main__':
    main()
