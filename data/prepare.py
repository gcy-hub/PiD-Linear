"""Independent resumable image/text cache preparation using the online encoders."""
import argparse
import json
from pathlib import Path
import torch
from attention_framework.config import config_parser, compose_config
from .dataset import ImageDataset
from .cache import CacheStore, identity, component_identity, EMPTY_TEXT
from .encoders import build_encoder
from .preprocess import load_pixels
from . import token_grid


def prepare(config, *, devices=('cpu',), batch_size=1, phase='all'):
    if batch_size < 1 or not devices:
        raise ValueError('Need a positive batch size and at least one device')
    if not config.cache.root:
        raise ValueError('cache.root is required for preparation')
    dataset = ImageDataset(config.data.manifest or config.paths.dataset, config.data.buckets, config.data.min_size)
    store = CacheStore(config.cache.root)
    failures = []
    completed = skipped = 0
    if phase not in ('all', 'image', 'text'):
        raise ValueError(f'Unknown cache phase: {phase}')
    for kind in (('image', 'text') if phase == 'all' else (phase,)):
        resources = component_identity(config, kind)
        for device_index, device in enumerate(devices):
            pending = []
            for index in range(device_index, len(dataset), len(devices)):
                sample, size = dataset[index], dataset.sizes[index]
                meta = identity(config, sample, size, kind, resources=resources)
                if store.valid(kind, sample.sample_id, meta):
                    skipped += 1
                else:
                    pending.append((sample, size, meta))
            if kind == 'text' and device_index == 0 and config.data.caption_dropout > 0:
                meta = identity(config, EMPTY_TEXT, (0, 0), 'text', resources=resources)
                if store.valid('text', EMPTY_TEXT.sample_id, meta):
                    skipped += 1
                else:
                    pending.append((EMPTY_TEXT, (0, 0), meta))
            if not pending:
                continue
            encoder = build_encoder(config, device=torch.device(device), image=kind == 'image', text=kind == 'text')
            groups = {}
            for task in pending:
                key = tuple(task[1]) if kind == 'image' else 'text'
                groups.setdefault(key, []).append(task)
            for tasks in groups.values():
                native_batch = 1 if kind == 'text' and config.model.family == 'flux3' else batch_size
                for offset in range(0, len(tasks), native_batch):
                    batch = tasks[offset:offset + native_batch]
                    try:
                        if kind == 'image':
                            pixels = torch.stack([load_pixels(sample.image_path, size) for sample, size, _ in batch])
                            tensors = encoder.encode_image(pixels, tuple(token_grid(config, size) for _, size, _ in batch))
                            if config.model.family == 'pid':
                                tensors.pop('pixels')
                        else:
                            tensors = encoder.encode_text([sample.caption for sample, _, _ in batch])
                        for i, (sample, _, meta) in enumerate(batch):
                            store.write(kind, sample.sample_id, meta, {k: v[i:i + 1] for k, v in tensors.items()})
                            completed += 1
                    except Exception as error:
                        failures.extend(dict(sample=sample.sample_id, kind=kind, error=str(error)) for sample, _, _ in batch)
            del encoder
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
    path = Path(config.cache.root) / 'failures.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} samples failed; see {path}')
    return dict(samples=len(dataset), completed=completed, skipped=skipped)


def main(argv=None):
    parser = config_parser()
    parser.add_argument('--phase', choices=('image', 'text', 'all'), default='all')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--cache-batch-size', type=int, default=1)
    args = parser.parse_args(argv)
    config = compose_config(args.model_config, args.algorithm_config, args.training_config, args.data_config, args.overrides)
    print(json.dumps(prepare(config, devices=args.devices, batch_size=args.cache_batch_size, phase=args.phase)))


if __name__ == '__main__':
    main()
