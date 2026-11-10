"""Bucket-compatible native batches; encoders run only in the training process."""
import torch
from torch.utils.data import DataLoader
from attention_framework.contracts import EncodedBatch
from .dataset import ImageDataset
from .sampler import BucketSampler
from .preprocess import load_pixels
from .cache import CacheStore, identity, component_identity, EMPTY_TEXT
from .encoders import build_encoder


def token_grid(config, size):
    divisor = 16
    return size[1] // divisor, size[0] // divisor


def stack_text(items):
    result = {}
    length = max(item['embeddings'].shape[1] for item in items)
    for key in items[0]:
        values = []
        for item in items:
            tensor = item[key]
            if tensor.ndim >= 2 and tensor.shape[1] != length:
                padding = torch.zeros((1, length - tensor.shape[1], *tensor.shape[2:]), device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor, padding), dim=1)
            values.append(tensor)
        result[key] = torch.cat(values)
    if 'mask' not in result:
        result['mask'] = torch.stack([torch.arange(length, device=result['embeddings'].device) < item['embeddings'].shape[1] for item in items])
    return result


class BatchCollator:
    def __init__(self, config, dataset, encoder):
        self.config, self.dataset, self.encoder = config, dataset, encoder
        self.store = CacheStore(config.cache.root) if config.cache.image or config.cache.text else None
        self.indices = {s.sample_id: i for i, s in enumerate(dataset.samples)}
        self.metadata = {}
        for kind in ('image', 'text'):
            if not config.cache[kind]:
                continue
            resources = component_identity(config, kind)
            for sample, size in zip(dataset.samples, dataset.sizes):
                self.metadata[kind, sample.sample_id] = identity(config, sample, size, kind, resources=resources)
        self.empty_text = None
        if config.data.caption_dropout > 0:
            if config.cache.text:
                meta = identity(config, EMPTY_TEXT, (0, 0), 'text')
                self.empty_text = self.store.read('text', EMPTY_TEXT.sample_id, meta)
            else:
                self.empty_text = encoder.encode_text([''])
            if 'mask' not in self.empty_text:
                self.empty_text['mask'] = torch.ones(self.empty_text['embeddings'].shape[:2], dtype=torch.bool, device=self.empty_text['embeddings'].device)

    def __call__(self, samples):
        images, texts, grids = [], [], []
        for sample in samples:
            size = self.dataset.sizes[self.indices[sample.sample_id]]
            grid = token_grid(self.config, size)
            grids.append(grid)
            if self.config.cache.image:
                image = self.store.read('image', sample.sample_id, self.metadata['image', sample.sample_id])
                if self.config.model.family == 'pid':
                    image['pixels'] = load_pixels(sample.image_path, size)[None]
            else:
                image = self.encoder.encode_image(load_pixels(sample.image_path, size)[None], (grid,))
            if self.config.cache.text:
                text = self.store.read('text', sample.sample_id, self.metadata['text', sample.sample_id])
            else:
                text = self.encoder.encode_text([sample.caption])
            images.append(image)
            texts.append(text)
        text = stack_text(texts)
        if self.empty_text is not None:
            text.update({'empty_' + k: v.expand(len(samples), *v.shape[1:]) for k, v in self.empty_text.items()})
        return EncodedBatch(tuple(s.sample_id for s in samples), self.config.model.family,
                            {k: torch.cat([item[k] for item in images]) for k in images[0]},
                            text, tuple(grids))


def build_dataloader(config, *, rank, world_size, cursor=0):
    dataset = ImageDataset(config.data.manifest or config.paths.dataset, config.data.buckets, config.data.min_size)
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    encoder = build_encoder(config, device=device, image=not config.cache.image, text=not config.cache.text)
    sampler = BucketSampler(dataset.bucket_indices, config.training.batch_size, rank, world_size,
                            config.training.accumulation, config.training.seed, cursor)
    # Collation runs native encoding in this process, never in DataLoader workers.
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=BatchCollator(config, dataset, encoder))
