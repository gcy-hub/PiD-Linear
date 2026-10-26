"""Official PiD native-size mixed buckets over the existing raw-image index."""

import fcntl
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from pid._src.datasets.augmentors.multi_resolution_aspect_ratio import (
    MULTI_RES_GRID_BY_MAX_RESOLUTION, _classify_aspect_ratio,
)
from pid._src.linear_pid.data import RawImageDataset, GlobalBatchSampler, atomic_json


def native_target(width, height, max_resolution=3840):
    ar = _classify_aspect_ratio(width, height)
    for level, w, h in MULTI_RES_GRID_BY_MAX_RESOLUTION[str(max_resolution)][ar]:
        if width >= w and height >= h:
            return dict(size=(w, h), level=level, aspect=ar, key=f'L{level}_{ar}')
    return None


def prepare_native_buckets(index_root, max_resolution=3840):
    root = Path(index_root)
    source = json.loads((root / 'index.json').read_text())
    signature = dict(schema=1, source_fingerprint=source['fingerprint'], max_resolution=max_resolution)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output = root / f'native_buckets_{max_resolution}'
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'index.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['fingerprint'] == fingerprint and all((output / f).exists() for f in previous['files']):
                return output, previous
        rows = defaultdict(list)
        rejected = 0
        with (root / 'manifest.jsonl').open() as stream:
            for row, line in enumerate(tqdm(stream, total=source['train_size'] + source['validation_size'], desc='Native PiD buckets')):
                record = json.loads(line)
                if record['split'] != 'train':
                    continue
                target = native_target(record['width'], record['height'], max_resolution)
                if target is None:
                    rejected += 1
                else:
                    # Preserve the official composite level + aspect bucket.
                    rows[target['key']].append(row)
        files = []
        for key in sorted(rows):
            filename = f"train_{key.replace(',', '_')}.npy"
            with (output / (filename + '.tmp')).open('wb') as stream:
                np.save(stream, np.asarray(rows[key], dtype=np.int64))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(output / (filename + '.tmp'), output / filename)
            files.append(filename)
        if not rows:
            raise ValueError('No images fit the official native-size 2K minimum')
        manifest = dict(**signature, fingerprint=fingerprint, files=files,
                        bucket_counts={k: len(rows[k]) for k in sorted(rows)},
                        train_size=sum(map(len, rows.values())), rejected_too_small=rejected)
        atomic_json(path, manifest)
        return output, manifest


class NativeResolutionDataset(RawImageDataset):
    def __init__(self, index_root, max_resolution=3840, text_cache_root=None):
        super().__init__(index_root, text_cache_root=text_cache_root)
        self.max_resolution = max_resolution
        _, buckets = prepare_native_buckets(index_root, max_resolution)
        self.metadata = {**self.metadata, 'train_size': buckets['train_size'],
                         'native_buckets': buckets, 'fingerprint': buckets['fingerprint'],
                         'source_fingerprint': self.metadata.get('source_fingerprint', self.metadata['fingerprint'])}

    def __getitem__(self, row):
        record = self.record(row)
        cached = {}
        if self.text_cache:
            embs, mask = self.text_cache.read(record['id'], record['caption'])
            cached = dict(text_embs=embs, text_mask=mask)
        try:
            expected = native_target(record['width'], record['height'], self.max_resolution)
            if expected is None:
                raise ValueError('Image does not fit the native-size grid')
            tw, th = expected['size']
            with Image.open(record['path']) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                # Do not silently change the sampler's shape if a file changed.
                if image.size != (record['width'], record['height']):
                    raise ValueError('Image dimensions changed after indexing')
                factor = min(image.width / tw, image.height / th)
                if factor > 1:
                    image = image.resize((max(tw, math.ceil(image.width / factor)),
                                          max(th, math.ceil(image.height / factor))), Image.Resampling.LANCZOS)
                left, top = round((image.width - tw) / 2), round((image.height - th) / 2)
                array = np.array(image.crop((left, top, left + tw, top + th)), dtype=np.uint8, copy=True)
            return dict(image=torch.from_numpy(array).permute(2, 0, 1), caption=record['caption'],
                        id=record['id'], row=int(row), **cached)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            return dict(error=f"{record['id']}: {exc}", row=int(row))


class NativeBatchSampler(GlobalBatchSampler):
    def __init__(self, index_root, batch_size, accum, rank=0, world_size=1, seed=42, start=0,
                 effective_batch=0, max_resolution=3840):
        # Retain the existing stateless cursor and DP sharding implementation.
        super().__init__(index_root, batch_size, accum, rank, world_size, seed, start, effective_batch)
        root, manifest = prepare_native_buckets(index_root, max_resolution)
        self.buckets = [np.load(root / f, mmap_mode='r') for f in manifest['files']]
        counts = np.asarray([len(b) for b in self.buckets], dtype=float)
        self.underfull_buckets = {k: v for k, v in manifest['bucket_counts'].items()
                                  if v < self.global_batch_size // self.accum}
        self.probabilities = counts / counts.sum()

    def global_batch(self, cursor):
        # Official mixed-resolution loading buckets each microbatch, not the
        # entire accumulated update. This also retains small valid buckets.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, cursor]))
        count = self.global_batch_size // self.accum
        chunks = []
        for _ in range(self.accum):
            bucket = self.buckets[rng.choice(len(self.buckets), p=self.probabilities)]
            # As with an infinite resampled image stream, a rare shape bucket
            # may cycle across epochs while filling one global microbatch.
            # Surface this explicitly in run metadata; never drop valid images.
            chunks.append(bucket[rng.choice(len(bucket), count, replace=len(bucket) < count)])
        return np.concatenate(chunks)
