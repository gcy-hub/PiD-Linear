"""Component caches with direct boundary metadata comparison and atomic payloads."""
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
import torch
from omegaconf import OmegaConf
from attention_framework.contracts import Sample

EMPTY_TEXT = Sample("__empty__", "", "", 0, 0)


def resource_identity(value):
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.pth')) if path.is_dir() else [path]
    return dict(name=path.name, files=[dict(name=str(p.relative_to(path)) if path.is_dir() else p.name,
                                         size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files])


def component_identity(config, kind):
    resources = OmegaConf.to_container(config.model.resources, resolve=True)
    names = ('vae',) if kind == 'image' else ('text_encoder', 'tokenizer')
    # Absolute roots are deployment details, not resource identities.
    return {name: resource_identity(resources[name]) for name in names}


def identity(config, sample, size, kind, *, resources=None):
    model = component_identity(config, kind) if resources is None else resources
    source = Path(sample.image_path).stat() if kind == 'image' else None
    return dict(family=config.model.family, resources=model, dtype=config.runtime.dtype,
                sample=sample.sample_id, caption=sample.caption if kind == 'text' else None,
                source={'size': source.st_size, 'mtime_ns': source.st_mtime_ns} if source else None,
                size=list(size) if kind == 'image' else None,
                preprocess='exif-rgb-lanczos-center-native-v1' if kind == 'image' else 'native-text-v1')


class CacheStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, kind, key):
        return self.root / kind / (quote(key, safe='') + '.pt')

    def read(self, kind, key, metadata):
        path = self.path(kind, key)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['metadata'] != metadata:
            raise ValueError(f'Stale {kind} cache: {key}')
        return payload['tensors']

    def valid(self, kind, key, metadata):
        try:
            self.read(kind, key, metadata)
            return True
        except (OSError, ValueError, EOFError, RuntimeError):
            return False

    def write(self, kind, key, metadata, tensors):
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='.', suffix='.partial', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(dict(metadata=metadata, tensors={k: v.detach().cpu().contiguous() for k, v in tensors.items()}), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
