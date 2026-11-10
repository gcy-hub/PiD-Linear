"""Checkpoint-to-sampler binding and recovery after interrupted validation."""
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from .check_cpu import Toy
from .checkpoint import save
from .validate import validate


def main():
    modules = {name: ModuleType(name) for name in ('models', 'algorithms')}
    modules['models'].build_model = lambda *a, **k: Toy()
    modules['algorithms'].build_attention = lambda *a, **k: None
    calls = []
    def sample(adapter, config, *, output_dir):
        assert adapter.p.item() == 7.
        assert not torch.is_grad_enabled()
        assert not torch.distributed.is_initialized()
        calls.append(output_dir)
        if len(calls) == 1:
            raise RuntimeError('interrupted sampling')
        output_dir.mkdir()
        (output_dir/'sample.png').write_bytes(b'cpu binding check')
    modules['models'].sample = sample
    with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', modules):
        toy = Toy()
        toy.p.data.fill_(7.)
        optimizer = torch.optim.AdamW([toy.p])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        config = OmegaConf.create(dict(runtime=dict(gpus=1)))
        progress = dict(step=1, stage='clear', stage_step=1, cursor=1, pending_validation=True, done=True)
        directory = save(tmp, toy, optimizer, scheduler, torch.Generator(), progress, config, 0, 1)
        try:
            validate(config, directory)
        except RuntimeError as error:
            assert str(error) == 'interrupted sampling'
        else:
            raise AssertionError('Sampling failure swallowed')
        assert not (directory/'validated.json').exists()
        validate(config, directory)
        assert (directory/'validated.json').exists()
        assert len(calls) == 2
    print('CPU validation restore/binding/interruption recovery passed')


if __name__ == '__main__':
    main()
