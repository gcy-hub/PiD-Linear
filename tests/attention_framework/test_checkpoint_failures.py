"""Persisted master updates and publication failure must preserve the last set."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from training.check_cpu import Toy
from training import checkpoint
from training.optim import MasterProdigy


class CheckpointFailureTest(TestCase):
    def test_failed_save_preserves_previous_and_master_next_step(self):
        model = Toy().to(dtype=torch.bfloat16)
        opt = MasterProdigy([model.p], lr=1.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        generator = torch.Generator().manual_seed(7)
        def update():
            opt.zero_grad()
            model.p.grad = torch.ones_like(model.p)
            opt.step()
            scheduler.step()
        for _ in range(3):
            update()
        progress = dict(step=1, stage='clear', stage_step=1, cursor=2, pending_validation=False, done=False)
        with tempfile.TemporaryDirectory() as temp:
            first = checkpoint.save(temp, model, opt, scheduler, generator, progress, OmegaConf.create({}), 0, 1)
            state = checkpoint.read(first, 0, 1)
            update()
            expected = opt.param_groups[0]['params'][0].detach().clone()
            checkpoint.restore(state, model, opt, scheduler, generator)
            update()
            torch.testing.assert_close(opt.param_groups[0]['params'][0], expected, rtol=0, atol=0)
            with patch('training.checkpoint.torch.save', side_effect=OSError('acceptance disk failure')):
                with self.assertRaisesRegex(OSError, 'acceptance disk failure'):
                    checkpoint.save(temp, model, opt, scheduler, generator, dict(progress, step=2), OmegaConf.create({}), 0, 1)
            self.assertEqual(checkpoint.latest(temp), first)
            self.assertTrue((first / 'complete.json').is_file())
            self.assertFalse((Path(temp) / 'step-000000002/complete.json').exists())
