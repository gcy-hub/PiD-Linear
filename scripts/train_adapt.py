"""Launch PiD/PixelDiT adaptation from a JSON argument list."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mode', choices=('train', 'repeat'), default='train')
    parser.add_argument('--print-command', action='store_true')
    args, overrides = parser.parse_known_args(argv)
    options = json.loads(args.config.read_text()) if args.config else []
    if not isinstance(options, list) or not all(isinstance(v, str) for v in options):
        parser.error('Config must be a JSON list of trainer arguments')
    options += overrides
    visible = os.environ.get('GPU_IDS', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    count = os.environ.get('NPROC_PER_NODE', str(len(visible.split(','))))
    nodes = os.environ.get('NNODES', '1')
    if '--batch-size' in options and '--grad-accum' in options and '--effective-batch' not in options:
        def last_value(flag):
            positions = [i for i, value in enumerate(options) if value == flag]
            return int(options[positions[-1] + 1])
        effective = int(count) * int(nodes) * last_value('--batch-size') * last_value('--grad-accum')
        if effective > 0:
            options += ['--effective-batch', str(effective)]
    if args.mode == 'repeat':
        if nodes != '1':
            parser.error('Use train mode under Slurm for multi-node runs')
        result = [sys.executable, '-m', 'scripts.repeat_pid_vdn', *options]
    else:
        launch = ['--standalone'] if nodes == '1' else [
            '--nnodes', nodes, '--node_rank', os.environ['NODE_RANK'],
            '--master_addr', os.environ['MASTER_ADDR'], '--master_port', os.getenv('MASTER_PORT', '29583')]
        result = [sys.executable, '-m', 'torch.distributed.run', *launch,
                  '--nproc_per_node', count, '-m', 'pid._src.vdn_pid.training', *options]
    return args, result


def main():
    args, result = command()
    if args.print_command:
        import shlex
        print(shlex.join(result))
        return
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    if 'GPU_IDS' in env:
        env['CUDA_VISIBLE_DEVICES'] = env['GPU_IDS']
    env.setdefault('OMP_NUM_THREADS', env.get('CPU_THREADS', '1'))
    raise SystemExit(subprocess.call(result, cwd=REPO, env=env))


if __name__ == '__main__':
    main()
