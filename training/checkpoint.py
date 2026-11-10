"""Atomic publication of complete, fixed-world-size rank-local checkpoints."""
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .distributed import barrier
from .optim import local_optimizer


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def latest(root):
    path = Path(root) / 'latest.json'
    return Path(root) / json.loads(path.read_text())['directory'] if path.exists() else None


def weights(adapter):
    return {r.name: r.parameter.detach().cpu().clone() for r in adapter.role_parameters()}


def restore_weights(adapter, state):
    roles = {r.name: r.parameter for r in adapter.role_parameters()}
    if roles.keys() != state.keys():
        raise ValueError('Checkpoint role parameters differ from model')
    with torch.no_grad():
        for name, p in roles.items():
            p.copy_(state[name])


def save(root, adapter, optimizer, scheduler, generator, progress, config, rank, world):
    root = Path(root)
    directory = root / f"step-{progress['step']:09d}"
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(weights=weights(adapter), optimizer=local_optimizer(optimizer).state_dict(),
                 scheduler=scheduler.state_dict(), progress=progress, world_size=world,
                 config=OmegaConf.to_container(config, resolve=True),
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                          generator=generator.get_state()))
    temporary = directory / f'rank-{rank}.partial'
    torch.save(state, temporary)
    temporary.replace(directory / f'rank-{rank}.pt')
    barrier()
    if rank == 0:
        for i in range(world):
            if not (directory / f'rank-{i}.pt').is_file():
                raise FileNotFoundError(f'Missing rank shard {i}')
        previous = latest(root)
        atomic_json(directory / 'complete.json', dict(world_size=world, **progress))
        atomic_json(root / 'latest.json', dict(directory=directory.name))
        if previous and previous != directory:
            shutil.rmtree(previous)
    barrier()
    return directory


def read(directory, rank, world):
    directory = Path(directory)
    manifest = json.loads((directory / 'complete.json').read_text())
    if manifest['world_size'] != world:
        raise ValueError('Checkpoint requires unchanged world size')
    for i in range(world):
        if not (directory / f'rank-{i}.pt').is_file():
            raise FileNotFoundError(f'Missing rank shard {i}')
    return torch.load(directory / f'rank-{rank}.pt', map_location='cpu', weights_only=False)


def restore(state, adapter, optimizer, scheduler, generator):
    restore_weights(adapter, state['weights'])
    local = local_optimizer(optimizer)
    local.load_state_dict(state['optimizer'])
    if local is not optimizer:
        for outer, inner in zip(optimizer.param_groups, local.param_groups):
            outer.update({k: v for k, v in inner.items() if k != 'params'})
    scheduler.load_state_dict(state['scheduler'])
    rng = state['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state(rng['cuda'])
    generator.set_state(rng['generator'])
