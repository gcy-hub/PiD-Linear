import hashlib
from pathlib import Path


def resource_paths(root):
    root=Path(root).resolve()
    path=root/'transformer/diffusion_pytorch_model.safetensors'
    if not path.is_file():raise FileNotFoundError(path)
    return dict(dit=str(root),weights_file=str(path))


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(16*1024**2),b''):h.update(part)
    return h.hexdigest()
