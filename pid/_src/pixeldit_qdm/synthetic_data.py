"""Resumable native PixelDiT images with the exact original A1 caption schedule."""
import hashlib
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pid._src.linear_pid.data import BUCKETS, GlobalBatchSampler, RawImageDataset, atomic_json, sha256_file


def image_seed(seed, uid):
    return int.from_bytes(hashlib.sha256(f'{seed}:pixeldit-synthetic:{uid}'.encode()).digest()[:8], 'little') % (2**63-1)


def read_records(index):
    with (Path(index)/'manifest.jsonl').open() as stream:
        return [json.loads(line) for line in stream]


def _prepare_dataset(source_index, data_root, settings):
    source_index, root = Path(source_index).resolve(), Path(data_root).resolve()
    source = RawImageDataset(source_index)
    sampler = GlobalBatchSampler(source_index, settings['effective_batch'], 1, seed=settings['seed'])
    original = np.stack([sampler.global_batch(i) for i in range(settings['steps'])])
    rows, inverse = np.unique(original, return_inverse=True)
    validation = np.load(source_index/'validation.npy')
    if np.intersect1d(rows, validation).size:
        raise ValueError('Synthetic training captions overlap source validation')
    plan = inverse.reshape(original.shape)
    records = []
    for ordinal, row in enumerate(rows):
        record = source.record(int(row))
        relative = f'images/{ordinal:06d}.png'
        records.append(dict(id=record['id'], caption=record['caption'], source_row=int(row),
                            path=relative, image_relative_path=relative,
                            width=settings['width'], height=settings['height'], bucket='1,1',
                            group=f'pixeldit-synthetic:{record["id"]}',
                            seed=image_seed(settings['seed'], record['id'])))
    if source._stream is not None:
        source._stream.close()
    identity = dict(schema=1, source_fingerprint=source.metadata['fingerprint'], settings=settings,
                    records=records, batch_plan=plan.tolist())
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata = dict(schema=1, fingerprint=fingerprint, settings=settings,
                    source_index_fingerprint=source.metadata['fingerprint'],
                    source_fingerprint=source.metadata.get('source_fingerprint', source.metadata['fingerprint']),
                    dataset_root='..', train_size=len(records), validation_size=0,
                    exposures=int(plan.size), synthetic=True, synthetic_complete=False,
                    global_batch_plan='batch_plan.npy', source_index=str(source_index),
                    bucket_counts={key:len(records) if key=='1,1' else 0 for key in BUCKETS})
    root.mkdir(parents=True, exist_ok=True)
    index = root/'index'
    index.mkdir(exist_ok=True)
    marker = root/'generation.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('fingerprint') != fingerprint:
            raise ValueError('Synthetic dataset settings or source changed; use a new data directory')
        if not np.array_equal(np.load(index/'batch_plan.npy'),plan) or read_records(index)!=records:
            raise ValueError('Prepared synthetic sample plan/manifest changed')
        return previous
    offsets = []
    with (index/'manifest.jsonl.tmp').open('wb') as stream:
        for record in records:
            offsets.append(stream.tell())
            stream.write((json.dumps(record, ensure_ascii=False)+'\n').encode())
    os.replace(index/'manifest.jsonl.tmp', index/'manifest.jsonl')
    for name, array in [('offsets', np.array(offsets, dtype=np.uint64)), ('source_rows',rows),
                        ('batch_plan',plan), ('validation',np.array([], dtype=np.int64))]:
        with (index/(name+'.tmp')).open('wb') as stream:
            np.save(stream,array)
        os.replace(index/(name+'.tmp'),index/(name+'.npy'))
    for bucket in BUCKETS:
        np.save(index/f'train_{bucket.replace(",", "_")}.npy',
                np.arange(len(records),dtype=np.int64) if bucket=='1,1' else np.array([],dtype=np.int64))
    atomic_json(marker,metadata)
    atomic_json(index/'index.pending.json',metadata)
    return metadata


def prepare_dataset(source_index, data_root, settings):
    root=Path(data_root)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _prepare_dataset(source_index,root,settings)


def generated_complete(root, record, fingerprint, verify=False):
    path = Path(root)/record['image_relative_path']
    try:
        done = json.loads(path.with_suffix('.done.json').read_text())
        if any(done[key]!=value for key,value in dict(fingerprint=fingerprint, id=record['id'], seed=record['seed']).items()):
            raise ValueError(f'Generated sample identity changed: {path}')
        if path.stat().st_size != done['bytes']:
            return False
        if verify:
            if sha256_file(path)!=done['sha256']:
                return False
            with Image.open(path) as image:
                image.load()
                if image.size!=(record['width'],record['height']) or image.mode!='RGB':
                    return False
        return True
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def save_generated(root, record, image, fingerprint):
    path=Path(root)/record['image_relative_path']
    path.parent.mkdir(parents=True,exist_ok=True)
    if image.size!=(record['width'],record['height']):
        raise ValueError('Generated image dimensions differ from the prepared dataset')
    temporary=path.with_suffix('.png.incomplete')
    with temporary.open('wb') as stream:
        image.convert('RGB').save(stream,format='PNG')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    atomic_json(path.with_suffix('.done.json'),dict(fingerprint=fingerprint,id=record['id'],seed=record['seed'],
                bytes=path.stat().st_size,sha256=sha256_file(path)))


def publish_dataset(data_root, workers=4):
    root=Path(data_root)
    meta=json.loads((root/'generation.json').read_text())
    records=read_records(root/'index')
    def check(record):
        complete=generated_complete(root,record,meta['fingerprint'],verify=True)
        if not complete:
            (root/record['image_relative_path']).with_suffix('.done.json').unlink(missing_ok=True)
        return complete
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified=list(tqdm(pool.map(check,records),
                           total=len(records),desc='Verify synthetic images',disable=len(records)<100))
    if not all(verified):
        (root/'index/index.json').unlink(missing_ok=True)
        return False
    atomic_json(root/'index/index.json',{**meta,'synthetic_complete':True})
    return True


class PlannedBatchSampler(GlobalBatchSampler):
    """Keep original prompt draws, including repeats, without sampling a new subset."""
    def __init__(self,index_root,*args,**kwargs):
        super().__init__(index_root,*args,**kwargs)
        self.plan=np.load(Path(index_root)/'batch_plan.npy',mmap_mode='r')
        if self.plan.ndim!=2 or self.plan.shape[1]!=self.global_batch_size:
            raise ValueError('Synthetic batch plan does not match the effective batch')

    def global_batch(self,cursor):
        if not 0<=cursor<len(self.plan):
            raise ValueError('Synthetic sample plan budget exhausted')
        return self.plan[cursor]

    def __iter__(self):
        for cursor in range(self.start,len(self.plan)):
            rows=self.global_batch(cursor).reshape(self.accum,-1)
            first=sum(self.rank_batch_sizes[:self.rank]);last=first+self.rank_batch_sizes[self.rank]
            for micro in range(self.accum):
                yield rows[micro,first:last].tolist()
