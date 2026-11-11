"""The diagnostic must retain immutable values and identify exact divergence paths."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import torch
from diagnose_native_resume import snapshot, compare, worker
from omegaconf import OmegaConf
import models
from training import engine, checkpoint


class DiagnosticTest(unittest.TestCase):
    def test_worker_records_two_accumulated_updates(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))

            def prepare_batch(self, batch, *, generator):
                return SimpleNamespace(inputs=torch.rand(2, generator=generator))

            def forward(self, batch, **kwargs):
                return SimpleNamespace(prediction=self.weight * batch.inputs)

        def optimizer(config, roles, stage):
            opt = torch.optim.AdamW([r.parameter for r in roles], lr=.01)
            return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.)

        def segment(config, resume):
            model = models.build_model()
            roles = [SimpleNamespace(name='weight', parameter=model.weight)]
            opt, sched = engine.build_optimizer(config, roles, 'b')
            generator = torch.Generator().manual_seed(1)
            for _ in range(2):
                opt.zero_grad()
                for _ in range(2):
                    batch = model.prepare_batch(None, generator=generator)
                    (model(batch).prediction.sum() / 2).backward()
                opt.step()
                sched.step()

        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / 'input.yaml'
            OmegaConf.save(OmegaConf.create(dict(paths=dict(output=directory),
                                                runtime=dict(gpus=1))), cfg)
            with patch.dict('os.environ', RANK='0'), \
                 patch.object(models, 'build_model', lambda: Toy()), \
                 patch.object(engine, 'build_optimizer', optimizer), \
                 patch.object(engine, 'segment', segment), \
                 patch.object(checkpoint, 'read', return_value={'progress': {'step': 4}}):
                worker(cfg, 'unused')
            traces = Path(directory) / 'trace'
            self.assertEqual(len(list(traces.glob('*-input-*-rank0.pt'))), 4)
            for step in (4, 5):
                begin = torch.load(traces / f'{step:09d}-begin-0-rank0.pt', weights_only=False)
                end = torch.load(traces / f'{step:09d}-postupdate-0-rank0.pt', weights_only=False)
                self.assertFalse(torch.equal(begin['weights']['weight'], end['weights']['weight']))

    def test_snapshot_and_comparison(self):
        tensor = torch.tensor([1.], requires_grad=True)
        value = {'optimizer': {'state': [tensor]}, 'unused': None}
        saved = snapshot(value)
        with torch.no_grad():
            tensor.add_(1)
        self.assertEqual(saved['optimizer']['state'][0].item(), 1.)
        self.assertFalse(saved['optimizer']['state'][0].requires_grad)
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ('left', 'right')]
            left.mkdir()
            right.mkdir()
            name = '000000005-begin-0-rank0.pt'
            torch.save(saved, left / name)
            torch.save(saved, right / name)
            self.assertEqual(compare(left, right), [])
            torch.save(snapshot(value), right / name)
            self.assertEqual(compare(left, right), [name + '/optimizer/state/0'])


if __name__ == '__main__':
    unittest.main()
