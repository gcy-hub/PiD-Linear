"""Prepare native Qwen3-VL conditioning / FLUX.3 VAE latents in separate phases."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import (BUCKETS_2K, IMAGE_PREPROCESS, TEXT_PREPROCESS, CacheStore, CachedDataset,
                   fingerprint, image_ids, load_records, prepare_image, record_fingerprint)
from .resources import DEFAULT_WEIGHTS_ROOT, resource_manifest, resource_paths

DEFAULT_DATASET_ROOT = Path(os.environ.get('FLUX3_DATASET_ROOT', str(Path.home() / 'dataset' / 'MultiAspect-4K-1M')))
DEFAULT_SUBSET_ROOT = os.environ.get('FLUX3_SUBSET_ROOT', str(DEFAULT_DATASET_ROOT / 'pid_vdn_subsets'))
DEFAULT_CACHE_ROOT = os.environ.get('FLUX3_CACHE_ROOT', str(DEFAULT_DATASET_ROOT / 'flux3_qdm_cache'))


def component_fingerprints(resources):
    return {'text': fingerprint(dict(model=resources['fingerprints']['text'], preprocess=TEXT_PREPROCESS)),
            'image': fingerprint(dict(model=resources['fingerprints']['image'], preprocess=IMAGE_PREPROCESS))}


class ImageInputs(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = prepare_image(record['image_path'], BUCKETS_2K[record['bucket']])
        return index, np.array(image, dtype=np.uint8, copy=True)


def one_item(batch):
    return batch[0]


def _load_phase(phase, weights_root, device):
    # Only these imports need the vendored native DiffSynth runtime.
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux3_image import Flux3ImagePipeline
    paths = resource_paths(weights_root)
    phase_paths = paths['text_encoder'] if phase == 'text' else paths['vae']
    config = ModelConfig(path=phase_paths, computation_device=device,
                         computation_dtype=torch.bfloat16, preparing_device=device,
                         preparing_dtype=torch.bfloat16)
    pipe = Flux3ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device, model_configs=[config],
        processor_config=ModelConfig(path=paths['processor']) if phase == 'text' else None)
    model = pipe.text_encoder if phase == 'text' else pipe.vae
    if model is None:
        raise RuntimeError(f'Native DiffSynth could not identify the official {phase} checkpoint')
    model.eval().requires_grad_(False)
    if pipe.dit is not None or pipe.vram_management_enabled:
        raise RuntimeError('Preparation requires only the selected encoder, resident on its assigned GPU')
    return pipe


def _worker(rank, args, phase, gpu_ids, components, started):
    # Each phase worker owns one device and disjoint samples; no DDP or CPU offload.
    gpu_id = gpu_ids[rank]
    device = f'cuda:{gpu_id}'
    torch.set_num_threads(args['threads'])
    torch.cuda.set_device(gpu_id)
    records = load_records(args['subset_root'], args['subset_size'])
    if args['limit']:
        records = records[:args['limit']]
    store = CacheStore(args['cache_root'], components)
    def expired():
        return bool(args['max_seconds'] and time.monotonic() - started >= args['max_seconds'])
    if phase == 'text':
        # Deduplicate before sharding so duplicate captions are never written concurrently.
        captions = list(dict.fromkeys([''] + [r['caption'] for r in records]))
        tasks = [(fingerprint(caption), caption) for caption in captions][rank::len(gpu_ids)]
        pending = [(key, caption) for key, caption in tasks
                   if not args['resume'] or not store.valid('text', key, key)]
    else:
        tasks = records[rank::len(gpu_ids)]
        pending = [r for r in tasks if not args['resume'] or
                   not store.valid('image', store.keys(r)[0], record_fingerprint(r))]
    skipped = len(tasks) - len(pending)
    if not pending or expired():
        print(json.dumps(dict(phase=phase, gpu=gpu_id, completed=0, skipped=skipped,
                              remaining=len(pending), deadline=expired())), flush=True)
        return
    pipe = _load_phase(phase, args['weights_root'], device)
    done = 0
    encode_started = time.monotonic()
    bar = tqdm(total=len(tasks), initial=skipped, desc=f'{phase} GPU {gpu_id}', position=rank,
               unit='caption' if phase == 'text' else 'image', mininterval=2)
    with torch.inference_mode():
        if phase == 'text':
            from diffsynth.pipelines.flux3_image import Flux3Unit_PromptEmbedder
            unit = Flux3Unit_PromptEmbedder()
            for key, caption in pending:
                if expired():
                    break
                payload = unit.process(pipe, caption)
                payload['prompt_tokens'] = payload['prompt_tokens'].to(torch.bfloat16)
                payload['prompt_ids'] = payload['prompt_ids'].float()
                CachedDataset._validate_text(payload)
                store.write('text', key, key, payload)
                done += 1
                bar.update()
        else:
            kwargs = dict(batch_size=1, num_workers=args['workers'], collate_fn=one_item,
                          pin_memory=False, persistent_workers=False)
            if args['workers']:
                kwargs.update(multiprocessing_context='spawn', prefetch_factor=1)
            loader = DataLoader(ImageInputs(pending), **kwargs)
            for index, array in loader:
                if expired():
                    break
                row = pending[index]
                before = record_fingerprint(row)
                # This matches BasePipeline.preprocess_image's native BF16 arithmetic order.
                image = torch.from_numpy(array).to(device=device, dtype=torch.bfloat16)
                image = (image * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
                latent = pipe.vae.encode(image)
                height, width = latent.shape[-2:]
                target_width, target_height = BUCKETS_2K[row['bucket']]
                if latent.shape != (1, 128, target_height // 16, target_width // 16):
                    raise RuntimeError(f'Unexpected native VAE latent shape {tuple(latent.shape)}')
                if record_fingerprint(row) != before:
                    raise RuntimeError(f'Source changed while encoding {row["image_path"]}; retry preparation')
                payload = dict(image_latents=latent[0].permute(1, 2, 0).reshape(-1, 128).to(torch.bfloat16),
                               image_ids=image_ids(height, width), grid=(height, width))
                store.write('image', store.keys(row)[0], before, payload)
                done += 1
                bar.update()
    bar.close()
    seconds = time.monotonic() - encode_started
    print(json.dumps(dict(phase=phase, gpu=gpu_id, completed=done, skipped=skipped,
                          remaining=len(pending) - done, encode_seconds=round(seconds, 2),
                          items_per_second=round(done / max(seconds, .001), 4), deadline=expired())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset-root', default=DEFAULT_SUBSET_ROOT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--weights-root', default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument('--subset-size', type=int, choices=[20000, 50000], default=20000)
    parser.add_argument('--phase', choices=['text', 'image', 'all'], default='all')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated logical CUDA IDs in the current allocation')
    parser.add_argument('--workers', type=int, default=4, help='CPU image-loader processes per GPU')
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads per encoder process')
    parser.add_argument('--resume', action='store_true', help='Skip complete cache files with matching fingerprints')
    parser.add_argument('--max-seconds', type=float, default=1200, help='Graceful total runtime budget; 0 disables')
    parser.add_argument('--limit', type=int, default=0, help='Smoke only: first N rows, using a separate --cache-root')
    parser.add_argument('--check-only', action='store_true', help='Validate complete selected cache without loading models')
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or args.max_seconds < 0 or args.limit < 0:
        parser.error('workers/limit/max-seconds must be nonnegative; threads must be positive')
    if args.check_only:
        dataset = CachedDataset(args.subset_root, args.cache_root, args.subset_size, limit=args.limit)
        print(json.dumps(dict(status='complete', rows=len(dataset), fingerprint=dataset.fingerprint)))
        return
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('--gpu-ids requires one or more distinct logical device IDs')
    if not torch.cuda.is_available() or any(i < 0 or i >= torch.cuda.device_count() for i in gpu_ids):
        parser.error('Requested GPU IDs are not available in this process CUDA visibility')
    resources = resource_manifest(args.weights_root)
    components = component_fingerprints(resources)
    rows = load_records(args.subset_root, args.subset_size)
    if args.limit:
        rows = rows[:args.limit]
    store = CacheStore(args.cache_root, components)
    store.initialize_subset(args.subset_root, args.subset_size, rows,
                            resources=dict(weights_root=str(Path(args.weights_root).resolve()),
                                           fingerprints=resources['fingerprints'], revision=resources['revision']))
    started = time.monotonic()
    for phase in (['text', 'image'] if args.phase == 'all' else [args.phase]):
        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
            break
        # New processes at every boundary release TE before loading VAE (and vice versa).
        torch.multiprocessing.spawn(_worker, args=(vars(args), phase, gpu_ids, components, started),
                                    nprocs=len(gpu_ids), join=True)
    print(json.dumps(dict(status='phase_finished_or_runtime_budget', phase=args.phase, rows=len(rows),
                          elapsed_seconds=round(time.monotonic() - started, 2), resume=True,
                          cache_root=str(Path(args.cache_root).resolve()))), flush=True)


if __name__ == '__main__':
    main()
