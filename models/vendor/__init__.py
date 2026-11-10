"""One native DiffSynth runtime, imported without download entrypoints."""
import importlib
import sys
import types
from pathlib import Path


def native_module(family):
    if family not in ('flux2', 'flux3'):
        raise ValueError(f'Unknown DiffSynth family: {family}')
    root = Path(__file__).parent / 'DiffSynth-Studio' / 'diffsynth'
    prefix = '_attention_framework_diffsynth'
    for suffix, directory in (('', root), ('.models', root / 'models'), ('.core', root / 'core'),
                              ('.diffusion', root / 'diffusion')):
        name = prefix + suffix
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package
    return importlib.import_module(prefix + '.models.' + family + '_dit')


def import_framework():
    """Expose the full offline runtime for native VAE/text pipeline imports."""
    import os
    root = Path(__file__).parent / 'DiffSynth-Studio'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] = 'torch'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'True')
    return root


def flow_scheduler(family):
    native_module(family)
    module = importlib.import_module('_attention_framework_diffsynth.diffusion.flow_match')
    return module.FlowMatchScheduler('FLUX.2' if family == 'flux2' else 'FLUX.3')
