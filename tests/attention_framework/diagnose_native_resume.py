"""Trace two B updates from one checkpoint, including an uninterrupted replay control."""
import argparse
import copy
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from training import checkpoint, engine
from training.optim import local_optimizer
from check_native_resume import equal
from deterministic_training import configure


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def worker(config, resume):
    configure()
    import models
    cfg = OmegaConf.load(config)
    state = checkpoint.read(resume, int(os.environ['RANK']), cfg.runtime.gpus)
    step = state['progress']['step']
    del state
    root = Path(cfg.paths.output) / 'trace'
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    build_model, build_optimizer = models.build_model, engine.build_optimizer
    counters = {}
    active = {}
    gradient_hooks = []

    def record(event, value):
        key = (step, event)
        index = counters.get(key, 0)
        counters[key] = index + 1
        torch.save(snapshot(value), root / f'{step:09d}-{event}-{index}-rank{rank}.pt')

    def model(*args, **kwargs):
        adapter = build_model(*args, **kwargs)
        prepare, forward = adapter.prepare_batch, adapter.forward

        def prepared(*a, **kw):
            if (step, 'begin') not in counters:
                record('begin', dict(weights={r.name: r.parameter for r in active['roles']},
                    optimizer=local_optimizer(active['optimizer']).state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    generator_rng=kw['generator'].get_state()))
            batch = prepare(*a, **kw)
            record('input', vars(batch))
            return batch

        def forwarded(*a, **kw):
            output = forward(*a, **kw)
            record('teacher' if kw.get('teacher') else 'student', vars(output))
            return output

        adapter.prepare_batch, adapter.forward = prepared, forwarded
        return adapter

    def optimizer(config, roles, stage):
        for hook in gradient_hooks:
            hook.remove()
        gradient_hooks.clear()
        for role in roles:
            gradient_hooks.append(role.parameter.register_hook(
                lambda grad, name=role.name: record('localgrad.' + name, grad)))
        opt, sched = build_optimizer(config, roles, stage)
        update = opt.step

        @wraps(update)
        def traced(*a, **kw):
            nonlocal step
            record('preupdate', dict(weights={r.name: r.parameter for r in roles},
                gradients={r.name: r.parameter.grad for r in roles},
                optimizer=local_optimizer(opt).state_dict(),
                groups=[{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups]))
            result = update(*a, **kw)
            record('postupdate', dict(weights={r.name: r.parameter for r in roles},
                                     optimizer=local_optimizer(opt).state_dict()))
            step += 1
            return result

        # Keep the scheduler's installed step wrapper in place.
        opt.step = traced
        active.update(optimizer=opt, roles=roles)
        return opt, sched

    models.build_model, engine.build_optimizer = model, optimizer
    engine.segment(cfg, resume)


def launch(config, resume, trace=False):
    cfg = OmegaConf.load(config)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={cfg.runtime.gpus}']
    if trace:
        command += [str(Path(__file__).resolve()), '--worker-config', str(config)]
    else:
        command += [str(Path(__file__).with_name('deterministic_training.py')), '--worker-config', str(config)]
    if resume:
        command += ['--resume', str(resume)]
    subprocess.run(command, cwd=ROOT, check=True)
    return checkpoint.latest(cfg.paths.output)


def compare(left, right):
    def order(path):
        step, event, index, rank = path.stem.split('-')
        phase = {'begin': 0, 'input': 1, 'teacher': 2, 'student': 3,
                 'preupdate': 1000000, 'postupdate': 1000001}.get(event, 500000)
        assert event in ('begin', 'input', 'teacher', 'student', 'preupdate', 'postupdate') or event.startswith('localgrad.')
        return int(step), phase + 3 * int(index), event, rank

    files = sorted(left.glob('*.pt'), key=order)
    assert files and {p.name for p in files} == {p.name for p in right.glob('*.pt')}
    differences = []
    for path in files:
        try:
            equal(torch.load(path, weights_only=False),
                  torch.load(right / path.name, weights_only=False), path.name)
        except AssertionError as error:
            differences.append(str(error))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config, args.resume)
        return
    cfg = OmegaConf.load(args.config)
    assert cfg.training.strategy == 'vdn' and cfg.training.vdn.b.steps == 2
    cfg.validation.every = 0
    args.output.mkdir(parents=True, exist_ok=False)
    cfg.paths.output = str(args.output.resolve() / 'seed')
    cfg.checkpoint.every = 100
    seed_config = args.output / 'seed.yaml'
    OmegaConf.save(cfg, seed_config)
    resume = None
    while True:
        resume = launch(seed_config, resume)
        progress = json.loads((resume / 'complete.json').read_text())
        if progress['stage'] == 'a2' and progress['stage_step'] == cfg.training.vdn.a2.steps:
            break
    for label, every in [('continuous', 100), ('replay', 100), ('resumed', 1)]:
        cfg.paths.output = str(args.output.resolve() / label)
        cfg.checkpoint.every = every
        resolved = args.output / f'{label}.yaml'
        OmegaConf.save(cfg, resolved)
        end = launch(resolved, resume, trace=True)
        if every == 1:
            end = launch(resolved, end, trace=True)
    result = {label: compare(args.output / 'continuous/trace', args.output / label / 'trace')
              for label in ('replay', 'resumed')}
    (args.output / 'diagnosis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
