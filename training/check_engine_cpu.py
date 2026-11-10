"""Exercise common-engine accumulation, phase transitions and exact resume on CPU."""
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from attention_framework.config import compose_config
from attention_framework.contracts import ModelOutput, RoleParameter, TrainingBatch
from .engine import segment
from .checkpoint import latest, read


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Parameter(torch.tensor(0.2))
        self.lora = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Parameter(torch.tensor(0.3))
        self.projection = nn.Parameter(torch.tensor(0.4))
    @property
    def module(self):
        return self
    def role_parameters(self):
        return [RoleParameter(k, p, r, 'early_image.0') for (k, p), r in zip(self.named_parameters(), ('branch', 'lora', 'local_gate', 'image_qkvo'))]
    def prepare_batch(self, batch, *, generator):
        x = torch.rand(2, generator=generator) + torch.rand(2)
        return TrainingBatch({'x': x}, x * .5, x, ('toy',))
    def forward(self, batch, *, teacher=False, capture=()):
        x = batch.inputs['x']
        y = x if teacher else x * (self.branch + self.lora + self.gate + self.projection)
        return ModelOutput(y, {k: y for k in capture})
    def layerwise(self, batch, callback):
        x = batch.inputs['x'].detach()
        callback(SimpleNamespace(teacher=x, student=x*self.branch))


class Loader:
    def __iter__(self):
        while True:
            yield None


def run(root, every, strategy):
    config = compose_config('configs/models/flux3.yaml', 'configs/algorithms/sliding_qdm.yaml',
                            f'configs/training/{strategy}.yaml', 'configs/data/default.yaml',
                            [f'paths.output={root}', 'runtime.dtype=float32', 'data.workers=0',
                             'training.accumulation=2', 'training.steps=3', f'checkpoint.every={every}',
                             'validation.every=0', 'training.vdn.a1.steps=2', 'training.vdn.a2.steps=2',
                             'training.vdn.b.steps=2'])
    resume = None
    stages = []
    while True:
        segment(config, resume)
        resume = latest(root)
        state = read(resume, 0, 1)
        stages.append(state['progress']['stage'])
        if state['progress']['done']:
            return state, stages


def main():
    modules = {name: ModuleType(name) for name in ('models', 'data', 'algorithms')}
    modules['models'].build_model = lambda *a, **k: Toy()
    modules['data'].build_dataloader = lambda *a, **k: Loader()
    modules['algorithms'].build_attention = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
        for strategy in ('vdn', 'clear'):
            continuous, stages = run(Path(tmp)/strategy/'continuous', 100, strategy)
            resumed, _ = run(Path(tmp)/strategy/'resumed', 1, strategy)
            for name, p in continuous['weights'].items():
                assert torch.equal(p, resumed['weights'][name]), (strategy, name)
            assert continuous['progress'] == resumed['progress']
            assert torch.equal(continuous['rng']['torch'], resumed['rng']['torch'])
            assert continuous['progress']['cursor'] == continuous['progress']['step']*2
            if strategy == 'vdn':
                assert stages == ['a1', 'a2', 'b']
    print('CPU engine accumulation, stage transitions and exact resume passed')


if __name__ == '__main__':
    main()
