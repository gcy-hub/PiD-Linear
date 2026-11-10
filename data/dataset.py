"""Manifest counts come from completed image/caption records, never preset sizes."""
import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
from attention_framework.contracts import Sample
from .buckets import choose_bucket


def records(path):
    path = Path(path).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    if path.is_dir() and (root / 'index.json').exists():
        offsets = np.load(root / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        seen = set()
        source = root / 'manifest.jsonl'
        with source.open('rb') as stream:
            for selection in sorted(root.glob('train_*.npy')):
                for row_id in np.load(selection, mmap_mode='r', allow_pickle=False):
                    row_id = int(row_id)
                    if row_id in seen or not 0 <= row_id < len(offsets):
                        raise ValueError(f'Duplicate or invalid selected row: {row_id}')
                    seen.add(row_id)
                    stream.seek(int(offsets[row_id]))
                    yield source, row_id, json.loads(stream.readline())
        return
    if path.is_dir():
        manifest = path / 'manifest.jsonl'
        sources = [manifest] if manifest.exists() else sorted((path / 'data_jsons').glob('*.json'))
    else:
        sources = [path]
    if not sources:
        raise FileNotFoundError(f'No manifest or metadata in {path}')
    for source in sources:
        with source.open() as stream:
            if source.suffix == '.csv':
                rows = list(csv.DictReader(stream))
            elif source.suffix == '.jsonl':
                rows = [json.loads(line) for line in stream if line.strip()]
            else:
                rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError(f'Expected record list: {source}')
        for index, row in enumerate(rows):
            yield source, index, row


class ImageDataset:
    def __init__(self, manifest, buckets='official', min_size=256):
        self.samples, self.sizes = [], []
        self.bucket_indices = defaultdict(list)
        self.rejected = defaultdict(int)
        seen = set()
        for source, index, row in records(manifest):
            if not row.get('image_path'):
                self.rejected['missing_image'] += 1
                continue
            image_path = Path(row['image_path'])
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            image_path = image_path.resolve()
            caption = row.get('caption', row.get('prompt', row.get('text')))
            if not isinstance(caption, str):
                raise ValueError(f'Missing caption: {source}:{index}')
            with Image.open(image_path) as image:
                width, height = image.size
                if image.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
            size = choose_bucket(width, height, buckets)
            if min(width, height) < min_size or size is None:
                self.rejected['too_small'] += 1
                continue
            # Downloader IDs are local to each metadata shard.
            key = f'{source.name}:{row.get("id", index)}'
            if key in seen:
                raise ValueError(f'Duplicate sample ID: {key}')
            seen.add(key)
            self.bucket_indices[size].append(len(self.samples))
            self.samples.append(Sample(key, str(image_path), caption, width, height))
            self.sizes.append(size)
        if not self.samples:
            raise ValueError('No images fit the configured buckets/min_size')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
