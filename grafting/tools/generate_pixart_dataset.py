#!/usr/bin/env python3
"""Generate resumable PixArt-Sigma image/caption pairs, one pipeline per GPU."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_prompts(path):
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == '.json':
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError('JSON must contain a list of prompts or objects.')
    else:
        rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    result = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            row = {'prompt': row}
        prompt = row.get('prompt', row.get('caption', row.get('text', '')))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Empty/invalid prompt at record {i}.')
        result.append({'id': str(row.get('id', i)), 'prompt': prompt.strip()})
    if not result:
        raise ValueError('Prompt file is empty.')
    return result


def make_records(prompts, seed):
    return [dict(row, index=i, seed=seed + i, image_path=f'images/{i:08d}.png')
            for i, row in enumerate(prompts)]


def rank_records(records, rank, world):
    return records[rank::world]


def ensure_recipe(root, recipe):
    path = root / 'generation_config.json'
    if path.exists():
        if json.loads(path.read_text()) != recipe:
            raise ValueError('Output directory has a different prompt/model/sampling recipe. Use a new output directory.')
    else:
        atomic_json(path, recipe)


def completed(root, record, recipe_hash, width, height):
    from PIL import Image
    path = root / 'metadata' / f'{record["index"]:08d}.json'
    try:
        metadata = json.loads(path.read_text())
        if metadata['recipe_hash'] != recipe_hash or any(metadata[k] != record[k] for k in record):
            return False
        image_path = root / record['image_path']
        stat = image_path.stat()
        if (metadata.get('width'), metadata.get('height')) != (width, height):
            return False
        # Freshly saved PNGs have a completion stamp. Repeated restarts should not
        # fully decode thousands of unchanged 2K images before loading the model.
        if metadata.get('image_size_bytes') == stat.st_size and metadata.get('image_mtime_ns') == stat.st_mtime_ns:
            return True
        with Image.open(image_path) as image:
            if image.size != (width, height):
                return False
            image.load()
        return True
    except (OSError, ValueError, KeyError):
        return False


def save_sample(root, record, image, recipe_hash, extra):
    path = root / record['image_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    image.save(tmp, format='PNG')
    os.replace(tmp, path)
    stat = path.stat()
    # Metadata is the completion marker, published only after the image.
    atomic_json(root / 'metadata' / f'{record["index"]:08d}.json',
                dict(record, recipe_hash=recipe_hash, width=image.width, height=image.height,
                     image_size_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns, **extra))


def write_manifests(root, records, recipe_hash, width, height, stage1_size):
    valid = [r for r in records if completed(root, r, recipe_hash, width, height)]
    for name, subset in [('manifest.jsonl', valid), ('stage1_8k.jsonl', valid[:stage1_size])]:
        # Fixed subset membership; an incomplete earlier image never substitutes a later image.
        if name.startswith('stage1'):
            subset = [r for r in valid if r['index'] < stage1_size]
        path = root / name
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
        os.replace(tmp, path)
    atomic_json(root / 'status.json', {'complete': len(valid), 'total': len(records),
                                      'finished': len(valid) == len(records), 'updated_at': time.time()})
    return len(valid)


def model_identity(root):
    paths = ['model_index.json', 'transformer/config.json', 'text_encoder/config.json',
             'vae/config.json', 'scheduler/scheduler_config.json']
    missing = [p for p in paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete local model: {missing}; run download_pixart_models.py first.')
    if not list((root / 'transformer').glob('*.safetensors')):
        raise FileNotFoundError('Missing transformer safetensors; model download is incomplete.')
    for component in ['text_encoder', 'vae']:
        if not list((root / component).glob('*.safetensors')):
            raise FileNotFoundError(f'Missing {component} safetensors; model download is incomplete.')
    for path in root.rglob('*.safetensors.index.json'):
        shards = set(json.loads(path.read_text())['weight_map'].values())
        if any(not (path.parent / shard).is_file() for shard in shards):
            raise FileNotFoundError(f'Incomplete weight shards listed in {path}')
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths} | {
        'weights': {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in root.rglob('*.safetensors')}}


def resume_model_identity(model_root, output_root):
    current = model_identity(model_root)
    recipe_path = output_root / 'generation_config.json'
    if not recipe_path.exists():
        return current
    saved = json.loads(recipe_path.read_text())['model_identity']
    if current == saved:
        return saved
    if ({k: v for k, v in current.items() if k != 'weights'} !=
            {k: v for k, v in saved.items() if k != 'weights'}):
        raise ValueError('Model configuration differs from this dataset; cannot resume.')
    proof_path = output_root / 'model_files_sha256.json'
    if not proof_path.exists():
        raise ValueError('Weight timestamps changed. Export the dataset with model SHA256 references before migration.')
    proof = json.loads(proof_path.read_text())
    if proof.get('model_identity_hash') != digest(saved) or set(proof['weights']) != set(saved['weights']):
        raise ValueError('Model SHA256 references do not match the original generation recipe.')
    if set(current['weights']) != set(saved['weights']):
        raise ValueError('Model weight file list differs from the original recipe.')
    cache_path = model_root / '.pixart_verified_sha256.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for name, original in saved['weights'].items():
        path = model_root / name
        stat = path.stat()
        expected = proof['weights'][name]
        if stat.st_size != original[0] or stat.st_size != expected['size']:
            raise ValueError(f'Model weight size / SHA256 mismatch: {name}')
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        entry = cache.get(name, {})
        # Some shared filesystems coarsen timestamp updates. Do not reuse a
        # digest for a file modified within the last two seconds.
        if entry.get('stamp') != stamp or time.time() - max(stat.st_mtime, stat.st_ctime) < 2:
            print(f'Verifying relocated model SHA256: {name}', flush=True)
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    h.update(chunk)
            entry = {'stamp': stamp, 'sha256': h.hexdigest()}
        if entry.get('sha256') != expected['sha256']:
            raise ValueError(f'Model SHA256 mismatch: {name}')
        cache[name] = entry
    atomic_json(cache_path, cache)
    # Preserve the historical recipe hash in every completed sample. We only
    # accept changed timestamps after verifying the exact original weight bytes.
    return saved


def worker(args):
    import torch
    from diffusers import DPMSolverMultistepScheduler, PixArtSigmaPipeline
    from tqdm import tqdm
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.output_root
    recipe = json.loads((root / 'generation_config.json').read_text())
    records = json.loads((root / 'records.json').read_text())
    recipe_hash = digest(recipe)
    records = rank_records(records, args.worker_rank, args.world_size)
    pending = [r for r in records if not completed(root, r, recipe_hash, args.width, args.height)]
    if not pending:
        return
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    torch.cuda.set_device(0)
    pipe = PixArtSigmaPipeline.from_pretrained(str(args.model_dir), torch_dtype=dtype,
                                               local_files_only=True, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, solver_order=2)
    pipe.to('cuda')
    pipe.vae.enable_slicing()
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    progress = tqdm(total=len(records), initial=len(records)-len(pending),
                    desc=f'GPU rank {args.worker_rank}', position=args.worker_rank)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(pending), args.batch_size):
            if stop or (root / '.stop-requested').exists() or (
                    args.max_seconds and time.monotonic() - started >= args.max_seconds):
                break
            batch = pending[offset:offset + args.batch_size]
            generators = [torch.Generator(device='cpu').manual_seed(r['seed']) for r in batch]
            torch.cuda.synchronize()
            begin = time.monotonic()
            images = pipe(prompt=[r['prompt'] for r in batch], negative_prompt=args.negative_prompt,
                          height=args.height, width=args.width, num_inference_steps=args.steps,
                          guidance_scale=args.cfg, generator=generators, max_sequence_length=300,
                          use_resolution_binning=False, clean_caption=args.clean_caption).images
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            futures = [pool.submit(save_sample, root, record, image, recipe_hash,
                                   {'batch_seconds': elapsed, 'batch_size': len(batch),
                                    'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING', '0'),
                                    'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
                       for record, image in zip(batch, images)]
            for future in futures:
                future.result()
            progress.update(len(batch))
            progress.set_postfix(batch_s=f'{elapsed:.1f}', img_s=f'{elapsed/len(batch):.1f}')
    progress.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('weights/PixArt-Sigma-XL-2-2K-MS'))
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/pixart-sigma-synthetic-12k'))
    parser.add_argument('--gpu-ids', default=os.environ.get('GPU_IDS', '0'))
    parser.add_argument('--batch-size', type=int, default=1, help='Images per GPU batch')
    parser.add_argument('--workers', type=int, default=4, help='PNG writer threads per GPU')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('CPU_THREADS', 1)))
    parser.add_argument('--num-samples', type=int, default=12000, help='0 uses all prompts; otherwise first N')
    parser.add_argument('--stage1-size', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--height', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--clean-caption', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vae-tiling', action='store_true', help='Explicit optional tiled decode; recorded in recipe')
    parser.add_argument('--max-seconds', type=int, default=0, help='Stop after a completed batch; 0 has no limit')
    parser.add_argument('--plan-only', action='store_true', help='Prepare recipe and records without loading a GPU')
    parser.add_argument('--worker-rank', type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument('--world-size', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.threads, args.steps, args.stage1_size) < 1:
        parser.error('batch/workers/threads/steps/stage1-size must be positive')
    if args.num_samples < 0 or args.max_seconds < 0 or min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
        parser.error('Dimensions must be positive multiples of 16; sample/time limits must be nonnegative')
    if args.worker_rank >= 0:
        worker(args)
        return
    if args.prompts is None:
        parser.error('--prompts is required (.txt, .jsonl or .json)')
    gpus = [g.strip() for g in args.gpu_ids.split(',') if g.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error('Provide distinct GPU IDs')
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another generator is already using this output directory.')
        prompts = read_prompts(args.prompts)
        if args.num_samples:
            prompts = prompts[:args.num_samples]
        records = make_records(prompts, args.seed)
        recipe = {k: getattr(args, k) for k in ['seed', 'width', 'height', 'steps', 'cfg', 'negative_prompt',
                                               'dtype', 'clean_caption', 'vae_tiling', 'stage1_size']}
        recipe.update(model_identity=resume_model_identity(args.model_dir, args.output_root), records_hash=digest(records),
                      count=len(records), scheduler='DPMSolverMultistepScheduler/order2', max_sequence_length=300,
                      schema=1)
        ensure_recipe(args.output_root, recipe)
        atomic_json(args.output_root / 'records.json', records)
        count = write_manifests(args.output_root, records, digest(recipe), args.width, args.height, args.stage1_size)
        print(f'{count}/{len(records)} complete; GPUs={gpus}; per-GPU batch={args.batch_size}', flush=True)
        if args.plan_only or count == len(records):
            return
        (args.output_root / '.stop-requested').unlink(missing_ok=True)
        processes = []
        stopping = False

        def request_stop(*_):
            nonlocal stopping
            stopping = True
            (args.output_root / '.stop-requested').touch()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            for rank, gpu in enumerate(gpus):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(args.threads),
                           MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                processes.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    *sys.argv[1:], '--worker-rank', str(rank), '--world-size', str(len(gpus))],
                    env=env, pass_fds=(lock.fileno(),)))
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    request_stop()
                time.sleep(1)
        finally:
            # On an unexpected parent exception, ask workers to save their current batch and exit.
            if any(p.poll() is None for p in processes):
                request_stop()
                for p in processes:
                    p.wait()
            count = write_manifests(args.output_root, records, digest(recipe), args.width, args.height, args.stage1_size)
        print(f'Saved {count}/{len(records)} images. Repeat the same command to resume.', flush=True)
        if any(p.returncode != 0 for p in processes):
            raise SystemExit('A GPU worker failed. See its traceback; batch size was not changed.')


if __name__ == '__main__':
    main()
