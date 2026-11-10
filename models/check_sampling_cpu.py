"""CPU sampling checks without loading pretrained weights or encoders."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import tempfile
import torch
from torch import nn
from PIL import Image
from omegaconf import OmegaConf
from attention_framework.contracts import EncodedBatch, ModelOutput
from .sampling import _sample_batch, sample
from .vendor import flow_scheduler


class Adapter(nn.Module):
    def __init__(self, family, flow_map=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.family = family
        self.base = SimpleNamespace(flow_map=flow_map, time_guidance_embed=SimpleNamespace(guidance_embedder=object()))
        self.calls = []

    def forward(self, batch):
        assert not torch.is_grad_enabled()
        self.calls.append(batch)
        x = batch.inputs.get('hidden_states', batch.inputs.get('x'))
        if self.family == 'flux3':
            packed = batch.inputs['packed']
            assert not packed.timesteps['text'].any()
            x = packed.data['image']
        return ModelOutput(torch.full_like(x, 0.25), {})


class VAE:
    def __init__(self, channels):
        self.bn = SimpleNamespace(running_mean=torch.arange(channels).float(), running_var=torch.full((channels,), 4.0))
        self.config = SimpleNamespace(batch_norm_eps=1e-5)
        self.input = None

    def decode(self, value, **kwargs):
        self.input = value.clone()
        if kwargs:
            return (value.repeat_interleave(8, -2).repeat_interleave(8, -1),)
        return value.repeat_interleave(16, -2).repeat_interleave(16, -1)


def batch_for(family):
    text = dict(embeddings=torch.ones(1, 3, 8), text_ids=torch.zeros(1, 3, 4), mask=torch.tensor([[1, 1, 0]]))
    image = dict(latents=torch.zeros(1, 6, 12 if family == 'flux2' else 3), image_ids=torch.zeros(1, 6, 4))
    if family == 'pid':
        image = dict(pixels=torch.zeros(1, 3, 32, 48), condition=torch.full((1, 16, 1, 2), 0.75))
    return EncodedBatch(('sample',), family, image, text, ((2, 3),))


def main():
    from diffusers import Flux2KleinPipeline
    for family in ('flux2', 'flux3'):
        for flow_map in ((False, True) if family == 'flux3' else (False,)):
            adapter, batch = Adapter(family, flow_map), batch_for(family)
            vae = VAE(batch.image['latents'].shape[-1])
            rgb = _sample_batch(adapter, batch, vae=vae, steps=3, seed=11)
            assert rgb.shape == (1, 3, 32, 48) and len(adapter.calls) == 3
            assert torch.all((rgb >= -1) & (rgb <= 1))
            expected = torch.randn(batch.image['latents'].shape, generator=torch.Generator().manual_seed(11)) - 0.25
            expected = expected.transpose(1, 2).reshape(1, -1, 2, 3)
            if family == 'flux2':
                expected = Flux2KleinPipeline._unpatchify_latents(expected * (4+1e-5)**0.5 + vae.bn.running_mean[None, :, None, None])
                assert adapter.calls[0].inputs['guidance'].item() == 4
            else:
                scheduler = flow_scheduler(family)
                scheduler.set_timesteps(3)
                if flow_map:
                    assert adapter.calls[0].inputs['timesteps_ref'].item() == scheduler.sigmas[1].item()
                    assert adapter.calls[-1].inputs['timesteps_ref'].item() == 0
                packed = adapter.calls[0].inputs['packed']
                assert packed.data['text'].shape[0] == 2
                assert packed.mode_indices['image_cond'] == [(0, 0)]
            torch.testing.assert_close(vae.input, expected)
            again = _sample_batch(adapter, batch, vae=vae, steps=3, seed=11)
            torch.testing.assert_close(rgb, again, atol=0, rtol=0)
    adapter, batch = Adapter('pid'), batch_for('pid')
    def native_pid(predict, text, mask, null_text, null_mask, condition, **kwargs):
        assert kwargs == dict(height=32, width=48, seed=11, steps=3, cfg=1.0, sigma=0.0)
        torch.testing.assert_close(condition, batch.image['condition'])
        x = torch.zeros(1, 3, 32, 48)
        output = predict(x, torch.ones(1), text, lq_latent=condition, degrade_sigma=torch.zeros(1), text_valid_mask=mask)
        assert adapter.calls[-1].inputs['_grids'] == ((2, 3),)
        return output
    with patch('pid._src.common.runtime.sample', native_pid):
        assert _sample_batch(adapter, batch, vae=None, steps=3, seed=11).shape == (1, 3, 32, 48)
    try:
        _sample_batch(adapter, batch, vae=None, steps=0, seed=11)
    except ValueError:
        pass
    else:
        raise AssertionError('zero steps accepted')
    # Exercise the public data lifecycle and PNG contract with data's concrete API shape.
    batch = batch_for('flux3')
    adapter = Adapter('flux3').train()
    calls = []
    encoder = SimpleNamespace(vae=VAE(3), encode_image=lambda pixels, grids: batch.image,
                              encode_text=lambda captions: batch.text)
    modules = {name: ModuleType(name) for name in ('data.dataset', 'data.preprocess', 'data.encoders')}
    modules['data.dataset'].ImageDataset = lambda *args: SimpleNamespace(samples=[SimpleNamespace(sample_id='sample', image_path='x', caption='caption')], sizes=[(48, 32)])
    modules['data.preprocess'].load_pixels = lambda *args: torch.zeros(3, 32, 48)
    def build_encoder(config, **kwargs):
        calls.append(kwargs)
        return encoder
    modules['data.encoders'].build_encoder = build_encoder
    cfg = OmegaConf.create(dict(data=dict(manifest='manifest', buckets='official', min_size=16),
                                paths=dict(dataset='dataset'), validation=dict(seed=11, steps=3)))
    with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', modules):
        paths = sample(adapter, cfg, output_dir=Path(root)/'validation')
        assert len(paths) == 1 and paths[0].is_file() and adapter.training
        with Image.open(paths[0]) as image:
            assert image.mode == 'RGB' and image.size == (48, 32)
        assert calls[0]['image'] and calls[0]['text']
        with patch('models.sampling._sample_batch', side_effect=RuntimeError('decoder failed')):
            try:
                sample(adapter, cfg, output_dir=Path(root)/'failed')
            except RuntimeError as error:
                assert str(error) == 'decoder failed' and adapter.training
            else:
                raise AssertionError('sampling failure swallowed')
        assert not (Path(root)/'failed/sample.png').exists()
    print('PASS: native FLUX schedules, FLUX2 BN/unpatchify/decode, FLUX3 packing/flow-map, deterministic noise, PiD clean condition/RGB delegation, public PNG lifecycle and failure propagation')


if __name__ == '__main__':
    main()
