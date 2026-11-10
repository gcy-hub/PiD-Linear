"""CPU regressions for mixed precision and native teacher parity."""
import copy
import unittest

import torch
from omegaconf import OmegaConf

from attention_framework.contracts import LayerSpec, TrainingBatch
from models.attention import original_projections
from models.base import NativeModel
from models.check_cpu import DenseOperator
from models.flux2 import Flux2Attention
from models.pid import PidAttention
from models.vendor import native_module


def config():
    return OmegaConf.create({'model': {'family': 'pid', 'layer_counts': {'patch': 2}},
                            'layer_specs': [dict(name='patch.0', group='patch', index=0,
                                                 ordinal=0, feature_tap='pre_o')],
                            'runtime': {'gradient_checkpointing': False},
                            'training': {'vdn': {'lora': {'rank': 2, 'alpha': 2}}}})


class ModelRepairs(unittest.TestCase):
    def test_pid_autocast_padding_and_gradient(self):
        from pid._src.networks.pixeldit_official import MMDiTJointAttention
        torch.manual_seed(11)
        site = PidAttention(MMDiTJointAttention(16, 2), DenseOperator(),
                            LayerSpec('patch.0', 'patch', 0, 0, 'pre_o'), config())
        x, y = torch.randn(1, 6, 16), torch.randn(1, 3, 16)
        mask = torch.tensor([[True, True, False]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            image, text = site(x, y, torch.ones(6, 4, dtype=torch.complex64),
                               grid_size=(2, 3), text_valid_mask=mask)
        self.assertEqual(text.dtype, torch.bfloat16)
        self.assertEqual(torch.count_nonzero(text[:, 2:]), 0)
        (image.float().square().mean() + text.float().square().mean()).backward()
        self.assertGreater(site.orig.proj_x.train_weight.grad.abs().sum(), 0)

    def test_flux2_bf16_fused_teacher_and_isolation(self):
        torch.manual_seed(12)
        orig = native_module('flux2').Flux2ParallelSelfAttention(query_dim=16, heads=2, dim_head=8)
        native = copy.deepcopy(orig)
        site = Flux2Attention(orig, DenseOperator(),
                              LayerSpec('single.0', 'single', 0, 0, 'post_o'), config())
        site.grid = (2, 3)
        x = torch.randn(1, 8, 16)
        for training in (True, False):
            native.train(training)
            site.train(training)
            site.teacher, site.capture = True, True
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16), original_projections(site):
                expected, before = native(x), site(x)
                torch.testing.assert_close(before, expected, rtol=0, atol=0)
                self.assertEqual(site.feature.shape, (1, 6, 16))
                pairs = []
                def callback(pair):
                    pair.student.float().square().mean().backward()
                    pairs.append(pair.layer.name)
                site.callback = callback
                torch.testing.assert_close(site(x), before, rtol=0, atol=0)
                self.assertEqual(pairs, ['single.0'])
                site.callback = None
                for role in site.role_parameters():
                    role.parameter.add_(.1)
                torch.testing.assert_close(site(x), before, rtol=0, atol=0)

    def test_pid_partial_replacement_and_native_share_valid_text(self):
        from pid._src.networks.pid_net import PidNet
        torch.manual_seed(13)
        base = PidNet(hidden_size=16, num_groups=2, pixel_hidden_size=8,
                      pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=2,
                      pixel_depth=1, patch_size=2, txt_embed_dim=12, txt_max_length=8,
                      lq_hidden_dim=16, lq_num_res_blocks=0, sr_scale=2)
        # Exercise attention through nonzero native modulation and output weights.
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(0, .1)
        cfg = config()
        native_cfg = copy.deepcopy(cfg)
        native_cfg.layer_specs = []
        native = NativeModel(copy.deepcopy(base), native_cfg, lambda *a, **kw: DenseOperator())
        adapted = NativeModel(base, cfg, lambda *a, **kw: DenseOperator())
        batch = TrainingBatch(dict(x=torch.randn(2, 3, 4, 6), t=torch.ones(2),
                                   y=torch.randn(2, 4, 12),
                                   text_valid_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
                                   _grids=((2, 3), (2, 3))), torch.zeros(2, 3, 4, 6),
                              torch.ones(2), ('a', 'b'))
        for training in (True, False):
            native.train(training)
            adapted.train(training)
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                teacher = adapted(batch, teacher=True).prediction
                expected = torch.cat([native.base(x=batch.inputs['x'][i:i+1],
                                                  t=batch.inputs['t'][i:i+1],
                                                  y=batch.inputs['y'][i:i+1, mask.bool()])
                                      for i, mask in enumerate(batch.inputs['text_valid_mask'])])
                torch.testing.assert_close(teacher, expected, rtol=.02, atol=.02)
                torch.testing.assert_close(native(batch).prediction, expected, rtol=0, atol=0)
                changed = copy.deepcopy(batch)
                changed.inputs['y'][~batch.inputs['text_valid_mask'].bool()] += 100
                torch.testing.assert_close(native(changed).prediction, native(batch).prediction, rtol=0, atol=0)
                torch.testing.assert_close(adapted(batch, teacher=True).prediction, teacher, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
