"""Native model loading and attention adaptation without training policy."""
from .base import NativeModel, load_native


def build_model(config, *, device, algorithm_factory):
    if config.model.family not in ('pid', 'flux2', 'flux3'):
        raise ValueError(f'Unknown model family: {config.model.family}')
    return NativeModel(load_native(config, device), config, algorithm_factory).to(device)
