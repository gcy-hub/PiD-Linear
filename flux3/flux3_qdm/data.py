"""Standalone indexed input and atomic FLUX.3 native conditioning cache."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# Locally copied PiD 2K bucket sizes (width, height); no PiD runtime dependency.
BUCKETS_2K = {'1,1': (2048, 2048), '4,3': (2304, 1728), '3,4': (1728, 2304),
              '16,9': (2688, 1536), '9,16': (1536, 2688),
              '3,2': (2688, 1792), '2,3': (1792, 2688)}
CACHE_SCHEMA = 1
TEXT_PREPROCESS = {'schema': 1, 'native': 'Flux3Unit_PromptEmbedder', 'layers': [4,8,12,16,20,24,28,32],
                   'max_length': 4096, 'pad_multiple': 80, 'channels': 20480, 'dtype': 'bfloat16'}
IMAGE_PREPROCESS = {'schema': 1, 'resize': 'PIL-LANCZOS-center-crop-exif-transpose',
                    'buckets': BUCKETS_2K, 'native': 'Flux2VAE.encode-BN', 'stride': 16,
                    'channels': 128, 'dtype': 'bfloat16', 'normalization': 'bf16 * (2/255) - 1'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def resolve_subset(subset_root, subset_size):
    root = Path(subset_root).expanduser().resolve()
    if not (root / 'index.json').is_file():
        root = root / str(subset_size)
    if not (root / 'index.json').is_file():
        raise FileNotFoundError(f'Missing subset index: {root / "index.json"}')
    index = json.loads((root / 'index.json').read_text())
    if int(index['train_size']) != int(subset_size):
        raise ValueError(f'Subset has {index["train_size"]} rows, requested {subset_size}')
    return root


def load_records(subset_root, subset_size):
    """Read selected row IDs through mmap offsets, without scanning the million-row JSONL."""
    root = resolve_subset(subset_root, subset_size)
    offsets = np.load(root / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    records = []
    seen_rows, seen_ids = set(), set()
    with (root / 'manifest.jsonl').open('rb', buffering=8192) as source:
        for file in sorted(root.glob('train_*.npy')):
            bucket = file.stem.removeprefix('train_').replace('_', ',')
            selected = np.load(file, mmap_mode='r', allow_pickle=False)
            for row_id in selected:
                row_id = int(row_id)
                if row_id in seen_rows:
                    raise ValueError(f'Duplicate selected row {row_id}')
                if not 0 <= row_id < len(offsets):
                    raise ValueError(f'Invalid selected row {row_id}')
                source.seek(int(offsets[row_id]))
                record = json.loads(source.readline())
                record['id'] = str(record['id'])
                if record['id'] in seen_ids:
                    raise ValueError(f'Duplicate image ID {record["id"]}')
                if record.get('bucket', bucket).replace('_', ',') != bucket:
                    raise ValueError(f'Selected bucket mismatch for {record["id"]}')
                record['bucket'] = bucket
                record['row_id'] = row_id
                record['image_path'] = record.get('image_path', record.get('path'))
                if not record['image_path'] or not isinstance(record.get('caption'), str):
                    raise ValueError(f'Missing image_path/caption for row {row_id}')
                if not Path(record['image_path']).is_absolute():
                    index = json.loads((root / 'index.json').read_text())
                    record['image_path'] = str(Path(index.get('dataset_root', root)) / record['image_path'])
                if bucket not in BUCKETS_2K:
                    raise ValueError(f'Unknown PiD 2K bucket {bucket}')
                seen_rows.add(row_id)
                seen_ids.add(record['id'])
                records.append(record)
    if len(records) != int(subset_size):
        raise ValueError(f'Expected {subset_size} selected rows, found {len(records)}')
    return records


def source_signature(subset_root, subset_size):
    root = resolve_subset(subset_root, subset_size)
    files = [root / 'index.json', *sorted(root.glob('train_*.npy'))]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in ['manifest.jsonl', 'offsets.npy']:
        p = (root / name).resolve()
        stat = p.stat()
        hashes[name] = dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return fingerprint(hashes)


def record_fingerprint(record):
    path = Path(record['image_path']).expanduser().resolve()
    stat = path.stat()
    return fingerprint(dict(id=record['id'], path=str(path), size=stat.st_size,
                            mtime_ns=stat.st_mtime_ns, bucket=record['bucket']))


def prompt_length(real_length):
    return min(math.ceil(real_length / 80) * 80, 4096)


def prompt_ids(length):
    ids = torch.zeros(length, 4, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, dtype=torch.float32)
    return ids


def image_ids(height, width):
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    zeros = torch.zeros_like(rows)
    ids = torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(-1, 4).float()
    ids[:, 1:3] *= .5
    return ids


def prepare_image(path, size):
    """Deterministic EXIF-corrected RGB, Lanczos cover resize and center crop."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = size
        scale = max(width / image.width, height / image.height)
        resized = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
        if image.size != resized:
            image = image.resize(resized, Image.Resampling.LANCZOS)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))


class CacheStore:
    def __init__(self, root, components):
        self.root = Path(root).expanduser().resolve()
        self.components = components

    def path(self, kind, key):
        # Every ID/caption is hashed: no slash traversal or caption text in file paths.
        return self.root / kind / self.components[kind] / key[:2] / f'{key}.pt'

    @staticmethod
    def keys(record):
        return fingerprint(record['id']), fingerprint(record['caption'])

    def valid(self, kind, key, source):
        path = self.path(kind, key)
        try:
            meta = json.loads(path.with_suffix('.json').read_text())
            stat = path.stat()
            return (meta['schema'] == CACHE_SCHEMA and meta['component'] == self.components[kind]
                    and meta['source'] == source and stat.st_size == meta['size']
                    and stat.st_mtime_ns == meta['mtime_ns'])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def write(self, kind, key, source, payload):
        import fcntl
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Shared directory lock prevents concurrent tensor/metadata replacement interleaving.
        with (path.parent / '.write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fd, tmp = tempfile.mkstemp(prefix=f'.{key}.', suffix='.partial', dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as f:
                    torch.save({k: v.detach().cpu().contiguous() if torch.is_tensor(v) else v
                                for k, v in payload.items()}, f)
                    f.flush()
                    os.fsync(f.fileno())
                digest = hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
                os.replace(tmp, path)
                stat = path.stat()
                atomic_json(path.with_suffix('.json'), dict(schema=CACHE_SCHEMA, component=self.components[kind],
                            source=source, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest))
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def read(self, kind, key, source):
        path = self.path(kind, key)
        if not self.valid(kind, key, source):
            raise RuntimeError(f'Missing, stale or partial {kind} cache: {path}; rerun preparation with --resume')
        data = path.read_bytes()
        meta = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(data).hexdigest() != meta['sha256']:
            # Leave bytes for inspection, but invalidate completion metadata so --resume repairs it.
            path.with_suffix('.json').unlink(missing_ok=True)
            raise RuntimeError(f'Corrupt {kind} cache: {path}; rerun preparation with --resume')
        try:
            return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        except Exception as exc:
            raise RuntimeError(f'Unreadable {kind} cache: {path}; rerun preparation with --resume') from exc

    def initialize_subset(self, subset_root, subset_size, rows, resources=None):
        signature = source_signature(subset_root, subset_size)
        manifest = dict(schema=CACHE_SCHEMA, subset_size=int(subset_size), source_signature=signature,
                        components=self.components, text_preprocess=TEXT_PREPROCESS, image_preprocess=IMAGE_PREPROCESS,
                        rows=[dict(id=r['id'], image_source=record_fingerprint(r), caption=fingerprint(r['caption'])) for r in rows],
                        resources=resources)
        manifest['fingerprint'] = fingerprint(manifest)
        atomic_json(self.root / 'subsets' / f'{subset_size}.json', manifest)
        return manifest


class CachedDataset(Dataset):
    def __init__(self, subset_root, cache_root, subset_size, limit=0):
        self.rows = load_records(subset_root, subset_size)
        if limit:
            self.rows = self.rows[:limit]
        self.records = self.rows
        path = Path(cache_root).expanduser().resolve() / 'subsets' / f'{subset_size}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing cache manifest {path}; run FLUX.3 preparation first')
        manifest = json.loads(path.read_text())
        if manifest.get('fingerprint') != fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'}):
            raise RuntimeError('Cache manifest fingerprint is corrupt; rerun preparation')
        if manifest.get('schema') != CACHE_SCHEMA or manifest.get('source_signature') != source_signature(subset_root, subset_size):
            raise RuntimeError('Cache source fingerprint is stale; rerun preparation with --resume')
        if manifest.get('text_preprocess') != json.loads(json.dumps(TEXT_PREPROCESS)) or manifest.get('image_preprocess') != json.loads(json.dumps(IMAGE_PREPROCESS)):
            raise RuntimeError('Cache preprocessing fingerprint is stale; rerun preparation')
        resources = manifest.get('resources')
        if resources:
            from .resources import resource_manifest
            current = resource_manifest(resources['weights_root'])
            if current['fingerprints'] != resources['fingerprints']:
                raise RuntimeError('Cache model fingerprint is stale; rerun preparation')
        self.store = CacheStore(cache_root, manifest['components'])
        self.fingerprint = (fingerprint(dict(manifest=manifest['fingerprint'], selected_ids=[r['id'] for r in self.rows]))
                            if limit else manifest['fingerprint'])
        self.bucket_indices = defaultdict(list)
        self.image_sources = []
        if len(manifest['rows']) < len(self.rows):
            raise RuntimeError('Cache contains only a limited smoke subset; finish full preparation or set limit explicitly')
        for i, (row, saved) in enumerate(zip(self.rows, manifest['rows'][:len(self.rows)], strict=True)):
            source = record_fingerprint(row)
            if saved != dict(id=row['id'], image_source=source, caption=fingerprint(row['caption'])):
                raise RuntimeError(f'Stale source fingerprint for {row["id"]}; rerun preparation')
            self.image_sources.append(source)
            self.bucket_indices[row['bucket']].append(i)
            ik, tk = self.store.keys(row)
            for kind, key, sig in [('image', ik, source), ('text', tk, fingerprint(row['caption']))]:
                if not self.store.valid(kind, key, sig):
                    raise RuntimeError(f'Missing or stale {kind} cache for {row["id"]}; rerun preparation with --resume')
        self.bucket_indices = dict(self.bucket_indices)
        self.grouped_indices = self.bucket_indices
        empty = fingerprint('')
        self.empty_prompt = self.store.read('text', empty, empty)
        self._validate_text(self.empty_prompt)

    @staticmethod
    def _validate_text(payload):
        tokens, ids = payload['prompt_tokens'], payload['prompt_ids']
        length = tokens.shape[0]
        if (tokens.ndim != 2 or tokens.shape[1] != 20480 or tokens.dtype != torch.bfloat16
                or not (length == 4096 or (0 < length <= 4096 and length % 80 == 0))
                or ids.shape != (length, 4) or ids.dtype != torch.float32):
            raise RuntimeError('Invalid native FLUX.3 prompt cache tensor shape/dtype')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ik, tk = self.store.keys(row)
        image = self.store.read('image', ik, self.image_sources[index])
        text = self.store.read('text', tk, fingerprint(row['caption']))
        self._validate_text(text)
        latent, ids = image['image_latents'], image['image_ids']
        grid = tuple(image['grid'])
        if (latent.shape != (grid[0] * grid[1], 128) or latent.dtype != torch.bfloat16
                or ids.shape != (latent.shape[0], 4) or ids.dtype != torch.float32):
            raise RuntimeError(f'Invalid native FLUX.3 image cache shape/dtype for {row["id"]}')
        return dict(id=row['id'], image_latents=latent, image_ids=ids,
                    prompt_tokens=text['prompt_tokens'], prompt_ids=text['prompt_ids'], grid=grid)


def collate_samples(batch):
    """Keep variable-length native text and image tokens unpacked for the trainer."""
    return list(batch)
