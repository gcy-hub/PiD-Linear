"""Opt-in rank-zero SwanLab metrics without changing training RNG state."""
from contextlib import contextmanager
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf


@contextmanager
def _preserve_rng():
    python, numpy = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)


def start(config, rank=0):
    settings = config.get('logging')
    if rank != 0 or not settings or settings.get('backend', 'none') == 'none':
        return None
    if settings.backend != 'swanlab':
        raise ValueError('logging.backend must be none or swanlab')
    with _preserve_rng():
        import swanlab
        directory = Path(config.paths.output) / 'swanlog'
        directory.mkdir(parents=True, exist_ok=True)
        run = swanlab.init(project=settings.get('project', 'attention-framework'),
                           name=settings.get('name'), mode=settings.get('mode', 'offline'),
                           log_dir=str(directory), config=OmegaConf.to_container(config, resolve=True))
    return SwanLabTracker(swanlab, run)


class SwanLabTracker:
    def __init__(self, sdk, run):
        self.sdk, self.run = sdk, run

    def log_step(self, metrics):
        with _preserve_rng():
            values = {f'train/{key}': value for key, value in metrics.items()
                      if key not in ('step', 'stage', 'losses')}
            values['train/stage'] = self.sdk.Text(metrics['stage'])
            values.update({f'train/losses/{key}': value for key, value in metrics['losses'].items()})
            self.run.log(values, step=metrics['step'])

    def log_validation(self, directory, step, count):
        with _preserve_rng():
            images = [self.sdk.Image(str(Path(directory) / f'{index:02d}.png'), caption=f'{index:02d}')
                      for index in range(count)]
            self.run.log({'validation/images': images}, step=step)

    def finish(self, error=None):
        with _preserve_rng():
            self.run.finish(state='crashed' if error is not None else 'success',
                            error=str(error) if error is not None else None)
