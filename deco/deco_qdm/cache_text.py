"""Cache official DeCo Qwen conditioning, independently sharded across GPUs."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from .data import (DATA_ROOT, TEXT_PREPROCESSING, atomic_json, canonical_digest,
                   file_sha256, load_index, load_text, text_cache_path)


def model_fingerprint(model_path):
    """Content fingerprint covers tokenizer, config and weights (not cache timestamps)."""
    root = Path(model_path).expanduser().resolve()
    patterns = ('*.json', '*.safetensors', '*.bin', '*.model', '*.txt', '*.tiktoken', '*.py')
    files = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    if not files:
        raise FileNotFoundError(f'No model/tokenizer files in {root}')
    return canonical_digest({path.name: file_sha256(path) for path in tqdm(files, desc='Fingerprint Qwen files')})


def atomic_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(tensor.detach().cpu().contiguous(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class QwenTextEncoder:
    """The reference Qwen3 encoder without compile or CPU offload.

    Raw prompts, right-padding to 128, last hidden state, truncate/zero-pad to
    2048 exactly as DeCo/src/models/conditioner/qwen3_text_encoder.py.
    """
    def __init__(self, model_path, device):
        from transformers import Qwen2Tokenizer, Qwen3Model
        model_path = Path(model_path).expanduser()
        self.device = torch.device(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            str(model_path), max_length=128, padding_side='right', local_files_only=True)
        self.model = Qwen3Model.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True)
        self.model.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, prompts):
        tokenized = self.tokenizer(prompts, truncation=True, max_length=128,
                                   padding='max_length', return_tensors='pt')
        result = self.model(input_ids=tokenized.input_ids.to(self.device),
                            attention_mask=tokenized.attention_mask.to(self.device))[0]
        if result.shape[-1] < 2048:
            result = torch.nn.functional.pad(result, (0, 2048 - result.shape[-1]))
        return result[:, :, :2048].to(dtype=torch.bfloat16)


class TextEncoder(QwenTextEncoder):
    def __init__(self, weights_root_or_model_path, device='cuda', max_length=128):
        if max_length != 128:
            raise ValueError('DeCo conditioning requires max_length=128')
        path = Path(weights_root_or_model_path).expanduser()
        if (path / 'Qwen3-1.7B').is_dir():
            path = path / 'Qwen3-1.7B'
        super().__init__(path, device)


def cache_records(records, root, encoder, batch_size, position=0):
    """Also exposed for lightweight tests with a deterministic stub encoder."""
    root = Path(root)
    pending = []
    for row in tqdm(records, desc='Check resumable text cache', position=position, leave=False):
        path = Path(row.get('cache_path') or text_cache_path(root, row['id']))
        try:
            load_text(path)
        except Exception:
            pending.append((row['caption'], path))
    for start in tqdm(range(0, len(pending), batch_size), desc='Encode text batches',
                      position=position, total=(len(pending) + batch_size - 1) // batch_size):
        batch = pending[start:start + batch_size]
        values = encoder.encode([caption for caption, _ in batch])
        if values.shape != (len(batch), 128, 2048) or values.dtype != torch.bfloat16:
            raise ValueError(f'Unexpected Qwen conditioning output: {values.shape}, {values.dtype}')
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite Qwen conditioning output')
        for (_, path), value in zip(batch, values):
            atomic_tensor(path, value)
    return len(pending)


def _worker(rank, devices, config):
    torch.set_num_threads(config['threads'])
    device = devices[rank]
    if device != 'cpu':
        torch.cuda.set_device(int(device))
        device = f'cuda:{device}'
    rows, _ = load_index(config['index_root'])
    assigned = rows[rank::len(devices)]
    if rank == 0:
        from .prompts import PROMPTS
        assigned += [{'id': '__empty__', 'caption': '',
                      'cache_path': str(Path(config['text_cache_root']) / 'empty.pt')}]
        assigned += [{'id': f'validation:{item["id"]}', 'caption': item['prompt'],
                      'cache_path': str(Path(config['text_cache_root']) / 'validation' / f'{item["id"]}.pt')}
                     for item in PROMPTS]
    # Skip model allocation entirely when the assigned records are already valid.
    needs_encoder = False
    for row in assigned:
        try:
            load_text(row.get('cache_path') or text_cache_path(config['text_cache_root'], row['id']))
        except Exception:
            needs_encoder = True
            break
    if needs_encoder:
        encoder = QwenTextEncoder(config['model_path'], device)
        cache_records(assigned, config['text_cache_root'], encoder, config['batch_size'], rank)


def build_cache(index_root, text_cache_root, model_path, gpu_ids='0', batch_size=8, workers=None, threads=1):
    index_root = Path(index_root).expanduser()
    text_cache_root = Path(text_cache_root).expanduser()
    model_path = Path(model_path).expanduser()
    if batch_size < 1 or threads < 1:
        raise ValueError('batch-size and threads must be positive')
    devices = [item.strip() for item in gpu_ids.split(',') if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('--gpu-ids must contain distinct visible CUDA device IDs (or cpu)')
    if devices != ['cpu']:
        if any(not item.isdecimal() for item in devices):
            raise ValueError('--gpu-ids must be nonnegative visible CUDA IDs, or cpu alone')
        devices = [str(int(item)) for item in devices]
        if len(set(devices)) != len(devices):
            raise ValueError('--gpu-ids contains duplicate CUDA device IDs')
        visible_count = torch.cuda.device_count()
        if any(int(item) >= visible_count for item in devices):
            raise ValueError(f'--gpu-ids refers outside {visible_count} visible CUDA devices; '
                             'IDs are logical indices after CUDA_VISIBLE_DEVICES masking')
    if workers is not None:
        if workers < 1:
            raise ValueError('workers must be positive')
        devices = devices[:workers]
    rows, index = load_index(index_root)
    if not rows:
        raise ValueError('Cannot cache an empty index')
    root = Path(text_cache_root)
    manifest_path = root / 'manifest.json'
    from .prompts import PROMPTS
    expected = {'version': 1, 'index_fingerprint': index['fingerprint'],
                'model_source': str(Path(model_path).resolve()),
                'model_fingerprint': model_fingerprint(model_path),
                'preprocessing': TEXT_PREPROCESSING, 'count': len(rows),
                'validation_prompts': PROMPTS}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f'Text cache provenance changed ({key}); use a fresh --text-cache-root')
    atomic_json(manifest_path, dict(expected, status='building'))
    config = {'index_root': str(index_root), 'text_cache_root': str(root),
              'model_path': str(model_path), 'batch_size': batch_size, 'threads': threads}
    if len(devices) == 1:
        _worker(0, devices, config)
    else:
        mp.spawn(_worker, args=(devices, config), nprocs=len(devices), join=True)
    from .prompts import PROMPTS
    # Verify every output before publishing readiness; workers may be interrupted safely.
    for row in tqdm(rows, desc='Verify complete text cache'):
        load_text(text_cache_path(root, row['id']))
    load_text(root / 'empty.pt')
    for item in PROMPTS:
        load_text(root / 'validation' / f'{item["id"]}.pt')
    expected['validation_prompts'] = PROMPTS
    atomic_json(manifest_path, dict(expected, status='ready'))
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-root', type=Path, default=DATA_ROOT / 'deco_qdm_cache/index')
    parser.add_argument('--text-cache-root', type=Path, default=DATA_ROOT / 'deco_qdm_cache/text')
    parser.add_argument('--model-path', type=Path, default=Path.home() / 'huggingface_ckpts/DeCo/Qwen3-1.7B')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated CUDA IDs visible in this process, or cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, help='Maximum independent GPU workers; default one per specified GPU')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_cache(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
