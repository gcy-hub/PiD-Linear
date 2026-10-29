"""CPU-only checkpoint/base identity validation before inference assets load."""
from __future__ import annotations
import hashlib
import fcntl
from pathlib import Path
from .repeat import read_json,write_json


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    path=Path(path).resolve();stat=path.stat()
    return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def validate_checkpoint_identity(checkpoint,base_path,config,state=None):
    """Use a stat-verified SHA cache or hash bytes; never query or initialize CUDA."""
    checkpoint=Path(checkpoint).resolve()
    if state is None:
        import torch
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False,mmap=True)
    saved=state.get('config',{})
    from .model import normalize_qdm_layers
    layout=normalize_qdm_layers(config.get('qdm_layers'))
    if normalize_qdm_layers(saved.get('qdm_layers'))!=layout:
        raise ValueError('Checkpoint QDM layout does not match inference architecture')
    for key,default in [('attention_mode','hybrid'),('memory_text_write',False),('halo',0),('region_size',8),('lora_rank',64),('lora_alpha',64)]:
        if saved.get(key,default)!=config.get(key,default):
            raise ValueError(f'Checkpoint {key} does not match inference architecture')
    metadata=state.get('metadata',{})
    if metadata.get('memory_text_write',False)!=saved.get('memory_text_write',False):
        raise ValueError('Checkpoint memory_text_write metadata is incompatible')
    if normalize_qdm_layers(metadata.get('qdm_layers'))!=layout:
        raise ValueError('Checkpoint QDM layout metadata is incompatible')
    expected=metadata.get('base_fingerprint')
    if not expected:
        raise ValueError('Checkpoint lacks the required base fingerprint')
    architecture=metadata.get('architecture')
    dimensions=metadata.get('native_dimensions')
    if architecture!='flux3-qdm-single28-fixed4' or dimensions!=[3072,24,128]:
        raise ValueError('Checkpoint native architecture metadata is incompatible')
    cache_path=checkpoint.parents[2]/'base_fingerprint.json'
    # Concurrent evaluation workers share one fallback hash, rather than reread
    # the full base checkpoint independently on every GPU.
    with cache_path.with_suffix('.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        signature=file_signature(base_path)
        cache=read_json(cache_path)
        if cache.get('signature')==signature and isinstance(cache.get('sha256'),str):
            actual=cache['sha256']
        else:
            actual=sha256_file(base_path)
            if file_signature(base_path)!=signature:
                raise RuntimeError('Base weights changed during fingerprint validation')
            write_json(cache_path,dict(signature=signature,sha256=actual))
    if actual!=expected:
        raise ValueError('Checkpoint base fingerprint does not match local base weights')
    source=Path(__file__).resolve().parents[1]/'DiffSynth-Studio/diffsynth/models/flux3_dit.py'
    native_source=sha256_file(source)
    expected_source=metadata.get('native_source_sha256') or metadata.get('native_source_hash')
    if expected_source and native_source!=expected_source:
        raise ValueError('Checkpoint native source fingerprint does not match local source')
    return dict(base_fingerprint=actual,architecture=architecture,native_dimensions=dimensions,
                native_source_sha256=native_source)
