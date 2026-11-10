"""Standalone native sampling after all training ranks have exited."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .checkpoint import read, restore_weights, atomic_json


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
    output = directory / 'validation'
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
