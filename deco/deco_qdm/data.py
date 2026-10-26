"""Validated, seekable BLIP tar indices and cached conditioning datasets.

Images stay inside their original uncompressed tar files. Every read opens its
own descriptor, so forked/persistent DataLoader workers never share a cursor.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

DATA_ROOT = Path.home() / 'dataset/BLIP-3o/BLIP3o-60k'
INDEX_VERSION = 1
TEXT_PREPROCESSING = {'version': 1, 'tokenizer': 'Qwen2Tokenizer', 'model': 'Qwen3Model',
                      'max_length': 128, 'padding_side': 'right', 'padding': 'max_length',
                      'truncation': True, 'chat_template': False, 'embed_dim': 2048,
                      'dtype': 'bfloat16', 'model_mode': 'eval'}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _prepare_shard(args):
    path, index_root, threads = args
    torch.set_num_threads(threads)
    path, index_root = Path(path), Path(index_root)
    name = path.stem
    meta_path = index_root / f'{name}.manifest.json'
    records_path = index_root / f'{name}.jsonl'
    signature = source_signature(path)
    if meta_path.exists() and records_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get('version') == INDEX_VERSION and meta.get('source') == signature
                and meta.get('records_sha256') == file_sha256(records_path)):
            return meta
    members, duplicates = {}, set()
    # Offsets are meaningful only for uncompressed tar; r: deliberately refuses gzip.
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in ('.jpg', '.jpeg', '.png', '.txt'):
                continue
            stem = str(Path(member.name).with_suffix(''))
            kind = 'caption' if suffix == '.txt' else 'image'
            entry = members.setdefault(stem, {})
            if kind in entry:
                duplicates.add(stem)
            entry[kind] = {'offset': member.offset_data, 'size': member.size, 'member': member.name}
    fd, temp = tempfile.mkstemp(prefix=f'.{name}.', dir=index_root)
    invalid, count = [], 0
    try:
        with os.fdopen(fd, 'w') as output, open(path, 'rb') as source:
            for stem, pair in tqdm(sorted(members.items()), desc=f'Validate {name}', unit='pairs', leave=False):
                record_id = f'{name}:{stem}'
                try:
                    if stem in duplicates:
                        raise ValueError('duplicate image or caption member')
                    if 'image' not in pair or 'caption' not in pair:
                        raise ValueError('missing image/caption pair')
                    source.seek(pair['caption']['offset'])
                    caption = source.read(pair['caption']['size']).decode('utf-8').strip()
                    if not caption:
                        raise ValueError('empty caption')
                    source.seek(pair['image']['offset'])
                    payload = source.read(pair['image']['size'])
                    if len(payload) != pair['image']['size']:
                        raise ValueError('truncated image payload')
                    with Image.open(io.BytesIO(payload)) as image:
                        image.convert('RGB').load()
                    row = {'id': record_id, 'tar_path': signature['path'],
                           'offset': pair['image']['offset'], 'size': pair['image']['size'],
                           'caption': caption}
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                except Exception as error:
                    invalid.append({'id': record_id, 'reason': f'{type(error).__name__}: {error}'})
            output.flush()
            os.fsync(output.fileno())
        if source_signature(path) != signature:
            raise RuntimeError(f'Archive changed during indexing: {path}')
        os.replace(temp, records_path)
        atomic_json(index_root / f'{name}.invalid.json', invalid)
        meta = {'version': INDEX_VERSION, 'shard': name, 'source': signature,
                'records_file': records_path.name, 'records_sha256': file_sha256(records_path),
                'valid': count, 'invalid': len(invalid)}
        atomic_json(meta_path, meta)
        return meta
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def prepare_index(dataset_root=DATA_ROOT, index_root=None, workers=4, threads=1):
    dataset_root = Path(dataset_root).expanduser()
    index_root = Path(index_root or dataset_root / 'deco_qdm_cache/index').expanduser()
    index_root.mkdir(parents=True, exist_ok=True)
    shards = sorted(dataset_root.glob('*.tar'))
    if not shards:
        raise FileNotFoundError(f'No completed *.tar archives in {dataset_root}')
    if workers < 1 or threads < 1:
        raise ValueError('workers and threads must be positive')
    tasks = [(str(path), str(index_root), threads) for path in shards]
    results = []
    if workers == 1:
        results = [_prepare_shard(task) for task in tqdm(tasks, desc='Validate/index shards')]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_prepare_shard, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Validate/index shards'):
                results.append(future.result())
    results.sort(key=lambda shard: shard['shard'])
    manifest = {'version': INDEX_VERSION, 'shards': results,
                'incomplete_shards': sorted(p.name for p in dataset_root.glob('*.tar.incomplete')),
                'valid': sum(row['valid'] for row in results),
                'invalid': sum(row['invalid'] for row in results)}
    manifest['fingerprint'] = canonical_digest(manifest)
    atomic_json(index_root / 'manifest.json', manifest)
    return manifest


def load_index(index_root):
    index_root = Path(index_root).expanduser()
    manifest = json.loads((index_root / 'manifest.json').read_text())
    unsigned = {key: value for key, value in manifest.items() if key != 'fingerprint'}
    if manifest.get('version') != INDEX_VERSION or canonical_digest(unsigned) != manifest.get('fingerprint'):
        raise ValueError('Invalid index manifest fingerprint; rerun data preparation')
    rows = []
    for shard in manifest['shards']:
        path = index_root / shard['records_file']
        if file_sha256(path) != shard['records_sha256']:
            raise ValueError(f'Index records changed: {path}')
        if source_signature(shard['source']['path']) != shard['source']:
            raise ValueError(f'Archive changed; rerun data preparation: {shard["source"]["path"]}')
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if len(rows) != manifest['valid'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Index count or IDs are inconsistent')
    return rows, manifest


def text_cache_path(root, record_id):
    key = hashlib.sha256(record_id.encode()).hexdigest()
    return Path(root) / 'records' / key[:2] / f'{key}.pt'


def load_text(path):
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (128, 2048) or tensor.dtype != torch.bfloat16:
        raise ValueError(f'Invalid cached text tensor: {path}')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Nonfinite cached text tensor: {path}')
    return tensor


def image_transform(resolution=512):
    # Match torchvision's native default interpolation and antialias behavior.
    return transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution),
                               transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])


class TarImageTextDataset(Dataset):
    def __init__(self, index_root, text_cache_root, resolution=512, allow_partial_data=False):
        self.records, manifest = load_index(index_root)
        if manifest.get('incomplete_shards') and not allow_partial_data:
            raise ValueError('Incomplete archives remain: ' + ', '.join(manifest['incomplete_shards'])
                             + '; finish downloads and rerun preparation, or explicitly allow partial data')
        self.fingerprint = manifest['fingerprint']
        self.ids = [row['id'] for row in self.records]
        self.text_cache_root = Path(text_cache_root).expanduser()
        cache = json.loads((self.text_cache_root / 'manifest.json').read_text())
        if cache.get('status') != 'ready' or cache.get('index_fingerprint') != self.fingerprint:
            raise ValueError('Text cache is incomplete or belongs to a different index')
        if cache.get('preprocessing') != TEXT_PREPROCESSING or not cache.get('model_fingerprint') or not cache.get('model_source'):
            raise ValueError('Text cache model/preprocessing provenance is missing or incompatible')
        if cache.get('count') != len(self.records):
            raise ValueError('Text cache count does not match index')
        self.model_source = cache['model_source']
        self.model_fingerprint = cache['model_fingerprint']
        self.text_fingerprint = canonical_digest({key: cache[key] for key in
                                                 ('index_fingerprint', 'model_fingerprint', 'preprocessing')})
        self.empty_text = load_text(self.text_cache_root / 'empty.pt')
        self.transform = image_transform(resolution)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        try:
            with open(row['tar_path'], 'rb') as stream:
                stream.seek(row['offset'])
                payload = stream.read(row['size'])
            if len(payload) != row['size']:
                raise ValueError('Truncated tar image')
            with Image.open(io.BytesIO(payload)) as image:
                tensor = self.transform(image.convert('RGB'))
            text = load_text(text_cache_path(self.text_cache_root, row['id']))
            return {'image': tensor, 'text': text, 'id': row['id']}
        except Exception as error:
            return {'error': f'{type(error).__name__}: {error}', 'id': row['id']}


def collate_samples(samples):
    errors = [{'id': sample['id'], 'error': sample['error']} for sample in samples if 'error' in sample]
    if errors or not samples:
        return {'valid': False, 'errors': errors, 'id': [sample['id'] for sample in samples],
                'image': None, 'text': None}
    return {'valid': True, 'errors': [], 'id': [sample['id'] for sample in samples],
            'image': torch.stack([sample['image'] for sample in samples]),
            'text': torch.stack([sample['text'] for sample in samples])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--index-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare_index(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
