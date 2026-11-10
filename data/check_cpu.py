"""Bounded CPU checks for manifests, native batch plumbing, caches and downloads."""
import itertools
import json
from pathlib import Path
import tempfile
import types
import sys
import numpy as np
from unittest.mock import patch
import torch
from PIL import Image
from omegaconf import OmegaConf
from attention_framework.config import compose_config
from .dataset import ImageDataset
from .sampler import BucketSampler
from .cache import CacheStore, identity
from .prepare import prepare
from .encoders import build_encoder, NativeEncoder
from .preprocess import load_pixels
from . import build_dataloader
from download.framework import download_resources


class FakeEncoder:
    def encode_image(self, pixels, grids):
        return {'pixels': pixels, 'condition': pixels[:, :, ::4, ::4]}

    def encode_text(self, captions):
        return {'embeddings': torch.ones(len(captions), 3, 2), 'mask': torch.ones(len(captions), 3, dtype=torch.bool)}


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        Image.new('RGB', (64, 32), 'red').save(root / 'image.png')
        manifest = root / 'manifest.jsonl'
        manifest.write_text('\n'.join(json.dumps(dict(id=i, image_path='image.png', caption='')) for i in range(8)))
        cfg = compose_config('configs/models/pid.yaml', 'configs/algorithms/sliding_qdm.yaml', 'configs/training/vdn.yaml', 'configs/data/default.yaml',
            [f'data.manifest={manifest}', 'data.buckets=[[32,16]]', 'data.min_size=16', f'cache.root={root / "cache"}', 'training.batch_size=2'])
        dataset = ImageDataset(manifest, cfg.data.buckets, 16)
        assert len(dataset) == 8 and dataset.sizes[0] == (32, 16)
        small = root / 'small.csv'
        small.write_text('id,image_path,caption\n1,image.png,hello\n')
        assert len(ImageDataset(small, [[32, 16]], 16)) == 1
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (64, 32)).save(root / 'rotated.jpg', exif=exif)
        rotated = root / 'rotated.jsonl'
        rotated.write_text(json.dumps(dict(image_path='rotated.jpg', caption='rotated')))
        oriented = ImageDataset(rotated, [[16, 32]], 16)
        assert (oriented[0].width, oriented[0].height) == (32, 64)
        assert load_pixels(oriented[0].image_path, (16, 32)).shape == (3, 32, 16)
        indexed = root / 'indexed'
        indexed.mkdir()
        lines = [json.dumps(dict(id=i, image_path='../image.png', caption='indexed')) + '\n' for i in range(3)]
        (indexed / 'manifest.jsonl').write_text(''.join(lines))
        (indexed / 'index.json').write_text('{}')
        np.save(indexed / 'offsets.npy', np.cumsum([0] + [len(line) for line in lines[:-1]]))
        np.save(indexed / 'train_1_1.npy', np.array([2, 0]))
        assert len(ImageDataset(indexed, [[32, 16]], 16)) == 2
        # Disabled native components import/load no TE or VAE, for every family.
        for family in ('pid', 'flux2', 'flux3'):
            cfg.model.family = family
            native = build_encoder(cfg, device=torch.device('cpu'), image=False, text=False)
            assert not hasattr(native, 'vae') and not hasattr(native, 'text')
        cfg.model.family = 'pid'
        # Check the extracted native FLUX2 packing/BN arithmetic without weights.
        native = NativeEncoder.__new__(NativeEncoder)
        native.family, native.device, native.dtype, native.image_enabled = 'flux2', torch.device('cpu'), torch.float32, True
        native.vae = types.SimpleNamespace(
            encode=lambda pixels: types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: torch.full((1, 4, 2, 2), 5.0))),
            bn=types.SimpleNamespace(running_mean=torch.ones(4), running_var=torch.full((4,), 4.0)),
            config=types.SimpleNamespace(batch_norm_eps=0.0))
        fake_diffusers = types.SimpleNamespace(Flux2KleinPipeline=types.SimpleNamespace(_patchify_latents=lambda z: z))
        with patch.dict(sys.modules, {'diffusers': fake_diffusers}):
            native_image = native.encode_image(torch.zeros(1, 3, 32, 32), ((2, 2),))
        assert native_image['latents'].shape == (1, 4, 4) and torch.all(native_image['latents'] == 2)

        sampler = lambda rank, cursor=0: iter(BucketSampler(dataset.bucket_indices, 2, rank, 2, 2, 42, cursor))
        a, b = sampler(0), sampler(1)
        draws = [next(a), next(b), next(a), next(b)]
        assert len(set(sum(draws, []))) == 8
        resumed = sampler(0, 1)
        assert next(resumed) == draws[2]
        calls = []
        def factory(config, *, device, image, text):
            calls.append((image, text))
            return FakeEncoder()
        with patch('data.prepare.build_encoder', factory):
            result = prepare(cfg, devices=['cpu'], batch_size=2)
            assert result['completed'] == 16
            assert prepare(cfg)['skipped'] == 16
        for image, text in itertools.product((False, True), repeat=2):
            cfg.cache.image, cfg.cache.text = image, text
            with patch('data.build_encoder', factory):
                loader = build_dataloader(cfg, rank=0, world_size=1)
                batch = next(iter(loader))
            assert calls[-1] == (not image, not text)
            assert batch.image['pixels'].shape == (2, 3, 16, 32)
            assert batch.image['condition'].shape == (2, 3, 4, 8)
            assert batch.text['embeddings'].shape == (2, 3, 2)
            batch.image['condition'].float().requires_grad_().sum().backward()
        store = CacheStore(cfg.cache.root)
        sample = dataset[0]
        meta = identity(cfg, sample, (32, 16), 'text')
        bad = dict(meta, caption='changed')
        try:
            store.read('text', sample.sample_id, bad)
            raise AssertionError('stale cache accepted')
        except ValueError:
            pass
        assert not list(Path(cfg.cache.root).rglob('*.partial'))
        before = store.read('text', sample.sample_id, meta)
        with patch('data.cache.torch.save', side_effect=OSError('simulated full disk')):
            try:
                store.write('text', sample.sample_id, meta, {'embeddings': torch.zeros(1)})
                raise AssertionError('failed write accepted')
            except OSError:
                pass
        assert torch.equal(store.read('text', sample.sample_id, meta)['embeddings'], before['embeddings'])
        assert not list(Path(cfg.cache.root).rglob('*.partial'))
        cfg.data.caption_dropout = 0.1
        with patch('data.prepare.build_encoder', factory):
            assert prepare(cfg, phase='text')['completed'] == 1
        with patch('data.build_encoder', factory):
            batch = next(iter(build_dataloader(cfg, rank=0, world_size=1)))
        assert 'empty_embeddings' in batch.text

        # A moved dataset root preserves identity when source metadata is retained.
        import shutil
        moved = root / 'moved'
        moved.mkdir()
        shutil.copy2(root / 'image.png', moved / 'image.png')
        shutil.copy2(manifest, moved / 'manifest.jsonl')
        moved_sample = ImageDataset(moved / 'manifest.jsonl', [[32, 16]], 16)[0]
        assert identity(cfg, sample, (32, 16), 'image') == identity(cfg, moved_sample, (32, 16), 'image')
        download_cfg = root / 'model.yaml'
        OmegaConf.save(OmegaConf.create({'model': {'resources': {'vae': str(root / 'vae'), 'text_encoder': str(root)},
            'downloads': {'vae': {'repo_id': 'example/model', 'revision': 'fixed', 'allow_patterns': ['vae/*']}}}}), download_cfg)
        with patch('download.framework.snapshot_download', return_value='done') as fake:
            assert download_resources(download_cfg, components=['vae', 'vae']) == {'vae': 'done'}
            assert fake.call_count == 1 and fake.call_args.kwargs['revision'] == 'fixed'
            assert download_resources(download_cfg, components=['text_encoder']) == {'text_encoder': str(root)}
            assert fake.call_count == 1
        try:
            download_resources(download_cfg, components=['unknown'])
            raise AssertionError('unknown component accepted')
        except ValueError:
            pass
    print('PASS: manifest counts, buckets, distributed cursor, four cache modes, stale metadata, atomic files, root reuse, fake downloads')


if __name__ == '__main__':
    main()
