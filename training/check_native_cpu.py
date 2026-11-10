"""CPU regressions for native PiD training precision and empty conditioning."""
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from attention_framework.config import compose_config
from attention_framework.contracts import EncodedBatch
from models.base import NativeModel
from models.check_cpu import DenseOperator
from .engine import segment
from .checkpoint import latest, read


def config(root, dtype='bfloat16', strategy='vdn'):
    return compose_config('configs/models/pid.yaml', 'configs/algorithms/sliding_qdm.yaml',
                          f'configs/training/{strategy}.yaml', 'configs/data/default.yaml',
                          [f'paths.output={root}', f'runtime.dtype={dtype}',
                           'model.layer_counts.patch=1', 'training.accumulation=2',
                           'training.vdn.a1.steps=1', 'training.vdn.a2.steps=1',
                           'training.vdn.b.steps=1', 'validation.every=0',
                           'training.steps=1', 'data.caption_dropout=1.0'])


def encoded(family='pid', empty_length=2):
    text = dict(embeddings=torch.randn(2, 3, 16), mask=torch.ones(2, 3, dtype=torch.bool),
                empty_embeddings=torch.full((2, empty_length, 16), 2.),
                empty_mask=torch.ones(2, empty_length, dtype=torch.bool))
    if family != 'pid':
        text.update(text_ids=torch.zeros(2, 3, 4), empty_text_ids=torch.ones(2, empty_length, 4))
    image = dict(pixels=torch.randn(2, 3, 8, 8), condition=torch.randn(2, 4, 2, 2)) if family == 'pid' else dict(latents=torch.randn(2, 4, 8), image_ids=torch.zeros(2, 4, 4))
    return EncodedBatch(('a', 'b'), family, image, text, ((2, 2), (2, 2)))


def dropout():
    for family in ('pid', 'flux2', 'flux3'):
        for length in (2, 5):
            model = NativeModel.__new__(NativeModel)
            nn.Module.__init__(model)
            model.base = nn.Linear(1, 1)
            model.base.time_guidance_embed = SimpleNamespace(guidance_embedder=None)
            model.family = family
            model.config = SimpleNamespace(data=SimpleNamespace(caption_dropout=0.5, image_dropout=0.))
            batch = encoded(family, length)
            # Mixed dropout, with the other random draws kept at a fixed value.
            def rand(n, **kwargs):
                return torch.tensor([0., 1.])
            with patch('torch.rand', rand):
                train = model.prepare_batch(batch, generator=torch.Generator().manual_seed(4))
            if family == 'flux3':
                packed = train.inputs['packed']
                assert packed.mode_indices['text'] == [(0, length), (length, length+3)]
                torch.testing.assert_close(packed.data['text'][:length], batch.text['empty_embeddings'][0])
                torch.testing.assert_close(packed.ids['text'][:length], batch.text['empty_text_ids'][0])
            else:
                key = 'y' if family == 'pid' else 'encoder_hidden_states'
                value = train.inputs[key]
                torch.testing.assert_close(value[0, :length], batch.text['empty_embeddings'][0])
                torch.testing.assert_close(value[1, :3], batch.text['embeddings'][1])
                mask = train.inputs['text_valid_mask' if family == 'pid' else '_text_mask']
                assert mask.sum(1).tolist() == [length, 3]
                if family == 'flux2':
                    torch.testing.assert_close(train.inputs['txt_ids'][0, :length], batch.text['empty_text_ids'][0])
    print('PASS: mixed empty-caption selection for all families, shorter/longer empty text, masks and IDs')


def main():
    from pid._src.networks.pid_net import PidNet
    torch.set_num_threads(1)
    dropout()
    for dtype, strategy in [('float32', 'vdn'), ('bfloat16', 'vdn'), ('float16', 'vdn'), ('bfloat16', 'clear')]:
        with tempfile.TemporaryDirectory() as root:
            cfg = config(root, dtype, strategy)
            built = []
            def build(*args, **kwargs):
                base = PidNet(hidden_size=32, num_groups=4, pixel_hidden_size=16,
                              pixel_num_groups=2, patch_depth=1, pixel_depth=1,
                              num_text_blocks=1, patch_size=4, txt_embed_dim=16, txt_max_length=8,
                              lq_in_channels=0, lq_latent_channels=4, lq_hidden_dim=16,
                              lq_num_res_blocks=1, sr_scale=1, latent_spatial_down_factor=4)
                nn.init.normal_(base.final_layer.linear.weight, std=.02)
                base.to(dtype=getattr(torch, dtype))
                model = NativeModel(base, cfg, lambda *a, **kw: DenseOperator())
                model.sites[0].operator.branch.data.fill_(.2)
                built.append(model)
                return model
            class Loader:
                def __iter__(self):
                    while True:
                        yield encoded()
            with patch('models.build_model', build), patch('data.build_dataloader', lambda *a, **kw: Loader()):
                resume = None
                for stage in (('a1', 'a2', 'b') if strategy == 'vdn' else ('clear',)):
                    segment(cfg, resume)
                    resume = latest(root)
                    state = read(resume, 0, 1)
                    assert state['progress']['stage'] == stage
                    grads = [r.parameter.grad for r in built[-1].role_parameters() if r.parameter.grad is not None]
                    assert grads and all(torch.isfinite(g).all() for g in grads)
                    if dtype != 'float16' or stage == 'a1':
                        assert any(g.count_nonzero() for g in grads)
                    assert all(torch.isfinite(v).all() for v in state['weights'].values())
            print(f'PASS: real small PiD {dtype} engine {strategy}, accumulation, finite gradients, checkpoint transitions')


if __name__ == '__main__':
    main()
