"""Standalone native sampling after all training ranks have exited."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .checkpoint import read, restore_weights, atomic_json


def validate_resident(config, directory, adapter, encoder, rank, world, group, tracker=None):
    """Keep weights resident; coordinate long sampling through the CPU group."""
    import random
    import numpy as np
    import torch.distributed as dist
    from models import sample
    if (directory / 'validated.json').exists():
        return
    py_state, np_state = random.getstate(), np.random.get_state()
    device = next(adapter.parameters()).device
    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            sample(adapter, config, output_dir=Path(config.paths.output) / 'validation' / directory.name,
                   encoder=encoder, rank=rank, world_size=world)
        if device.type == 'cuda':
            print(dict(validation=directory.name, rank=rank,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30), flush=True)
        if world > 1:
            dist.barrier(group=group)
        if rank == 0:
            output = Path(config.paths.output) / 'validation' / directory.name
            if any(not (output / f'{i:02d}.{ext}').is_file()
                   for i in range(config.validation.count) for ext in ('png', 'json')):
                raise RuntimeError('Validation images are incomplete')
            if tracker is not None:
                step = json.loads((directory / 'complete.json').read_text())['step']
                tracker.log_validation(output, step, config.validation.count)
            atomic_json(directory / 'validated.json', dict(count=config.validation.count))
        if world > 1:
            dist.barrier(group=group)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)



def validate(config, directory):
    from models import build_model, sample
    from algorithms import build_attention
    if torch.distributed.is_initialized():
        raise RuntimeError('Validation must run outside the training process group')
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    manifest = json.loads((directory / 'complete.json').read_text())
    state = read(directory, 0, manifest['world_size'])
    model = build_model(config, device=device, algorithm_factory=build_attention)
    restore_weights(model, state['weights'])
    del state
    model.module.eval()
    output = Path(config.paths.output) / 'validation' / directory.name
    with torch.no_grad():
        sample(model, config, output_dir=output)
    atomic_json(directory / 'validated.json', dict(step=manifest['step']))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    validate(OmegaConf.load(args.config), Path(args.checkpoint))


if __name__ == '__main__':
    main()
