"""Resume FLUX.2-klein-base-4B teacher images from prompts.json and run.json.

Each selected GPU runs one offline native DiffSynth pipeline. Existing valid
image/metadata pairs are preserved. ``completion.json`` and the returned
``complete`` flag indicate whether the first ``target_count`` prompts are ready
for training; ``max_new_samples`` can stop generation before that point.
"""
import argparse
import json
import multiprocessing
import os
import tempfile
from functools import partial
from multiprocessing.connection import wait
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _valid_pair(root, row, config):
    image_path = root / 'images' / (row['id'] + '.png')
    try:
        metadata = json.loads(image_path.with_suffix('.json').read_text(encoding='utf-8'))
        if not isinstance(metadata, dict) or any(metadata.get(key) != row[key]
                                                  for key in ('id', 'source_id', 'prompt', 'seed')):
            return False
        with Image.open(image_path) as image:
            if image.format != 'PNG' or image.size != (config['width'], config['height']):
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def _save_sample(root, image, row, run):
    image_path = root / 'images' / (row['id'] + '.png')
    fd, temporary = tempfile.mkstemp(prefix='.' + row['id'], suffix='.png', dir=image_path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            image.convert('RGB').save(stream, format='PNG')
        os.replace(temporary, image_path)
        _atomic_json(image_path.with_suffix('.json'), dict(row, generation=run['config']))
    finally:
        Path(temporary).unlink(missing_ok=True)


def _weight_files(directory):
    indexes = sorted(directory.glob('*.safetensors.index.json'))
    if indexes:
        index = json.loads(indexes[0].read_text(encoding='utf-8'))
        paths = [directory / name for name in sorted(set(index['weight_map'].values()))]
    else:
        paths = sorted(directory.glob('*.safetensors'))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f'Missing local safetensors weights in {directory}')
    return [str(path) for path in paths]


def _load_pipeline(model_dir):
    import torch

    from models.vendor import import_framework

    import_framework()
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    model_dir = Path(model_dir)
    tokenizer = model_dir / 'tokenizer'
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)
    vram = dict(offload_device='cpu', offload_dtype=torch.bfloat16,
                onload_device='cpu', onload_dtype=torch.bfloat16,
                preparing_device='cuda:0', preparing_dtype=torch.bfloat16,
                computation_device='cuda:0', computation_dtype=torch.bfloat16)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cuda:0',
        model_configs=[ModelConfig(path=_weight_files(model_dir / name), skip_download=True, **vram)
                       for name in ('transformer', 'vae', 'text_encoder')],
        tokenizer_config=ModelConfig(path=str(tokenizer), skip_download=True),
    )
    pipe.eval()
    pipe.requires_grad_(False)
    return pipe


def _generate_samples(pipe, root, rows, run, rank):
    config = run['config']
    for row in tqdm(rows, desc=f'FLUX2 worker {rank}', unit='image', position=2 * rank):
        image = pipe(
            prompt=row['prompt'], seed=row['seed'], height=config['height'], width=config['width'],
            num_inference_steps=config['steps'], cfg_scale=config['cfg_scale'],
            negative_prompt=config.get('negative_prompt', ''), rand_device='cpu',
            progress_bar_cmd=partial(tqdm, desc=f'Worker {rank} denoising', position=2 * rank + 1, leave=False),
        )
        if image.size != (config['width'], config['height']):
            raise ValueError(f"Unexpected output size for {row['id']}: {image.size}")
        _save_sample(root, image, row, run)


def _worker(model_dir, root, rows, run, gpu, cpu_threads, rank):
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(cpu_threads), TOKENIZERS_PARALLELISM='false')
    import torch

    torch.set_num_threads(cpu_threads)
    pipe = _load_pipeline(model_dir)
    with torch.inference_mode():
        _generate_samples(pipe, root, rows, run, rank)


def _run_workers(model_dir, root, rows, run, gpus, cpu_threads):
    context = multiprocessing.get_context('spawn')
    processes = []
    try:
        for rank, gpu in enumerate(gpus):
            process = context.Process(target=_worker, name=f'flux2-gpu-{gpu}',
                                      args=(model_dir, root, rows[rank::len(gpus)], run, gpu, cpu_threads, rank))
            process.start()
            processes.append(process)
        pending = {process.sentinel: process for process in processes}
        while pending:
            for sentinel in wait(pending):
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise RuntimeError(f'{process.name} exited with code {process.exitcode}; rerun to resume')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


def complete_dataset(model_dir: str | Path, data_root: str | Path, *, gpu_ids: str,
                     workers: int = 1, cpu_threads: int = 4, target_count: int = 10000,
                     max_new_samples: int | None = None) -> dict:
    """Repair missing pairs; report ``complete=True`` only when the target is ready.

    ``workers`` is capped by the number of supplied GPU IDs and pending images.
    Generation uses the saved run's resolution, steps and CFG with native BF16
    full softmax attention and per-record CPU noise seeds. Historical model paths
    and provenance do not need to match the new server's local model directory.
    """
    gpus = [value.strip() for value in gpu_ids.split(',')]
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('gpu_ids must list distinct nonnegative integer GPU IDs')
    if min(workers, cpu_threads, target_count) < 1 or (max_new_samples is not None and max_new_samples < 0):
        raise ValueError('workers, cpu_threads and target_count must be positive; max_new_samples must be nonnegative')
    root = Path(data_root).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    rows = json.loads((root / 'prompts.json').read_text(encoding='utf-8'))
    run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) < target_count:
        raise ValueError(f'prompts.json must contain at least {target_count} records')
    rows = rows[:target_count]
    if any(not isinstance(row.get('id'), str) or not row['id'] or 'source_id' not in row
           or not isinstance(row.get('prompt'), str) or not row['prompt'].strip()
           or not isinstance(row.get('seed'), int) for row in rows):
        raise ValueError('Every prompt record needs id, source_id, prompt and integer seed')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Prompt IDs must be unique')
    config = run['config']
    if any(not isinstance(config.get(key), int) or config[key] < 16 or config[key] % 16
           for key in ('height', 'width')) or not isinstance(config.get('steps'), int) or config['steps'] < 1:
        raise ValueError('Run dimensions must be positive multiples of 16 and steps must be positive')
    if config.get('dtype') != 'bfloat16' or config.get('rand_device') != 'cpu' or config.get('attention') != 'torch-sdpa':
        raise ValueError('Expected native BF16, CPU noise and torch-sdpa teacher generation settings')
    if not isinstance(config.get('cfg_scale'), (int, float)):
        raise ValueError('run.json must specify cfg_scale')

    (root / 'images').mkdir(exist_ok=True)
    missing = [row for row in tqdm(rows, desc='Checking FLUX2 pairs', unit='image')
               if not _valid_pair(root, row, config)]
    initial_count = target_count - len(missing)
    pending = missing if max_new_samples is None else missing[:max_new_samples]
    gpus = gpus[:min(workers, len(pending))]
    result = dict(complete_count=initial_count, target_count=target_count, generated_count=0,
                  complete=initial_count == target_count)
    _atomic_json(root / 'completion.json', result)
    print(f'FLUX2: {initial_count}/{target_count} ready; generating {len(pending)} on {len(gpus)} GPU(s)', flush=True)
    if pending:
        try:
            _run_workers(model_dir, root, pending, run, gpus, cpu_threads)
        finally:
            count = sum(_valid_pair(root, row, config) for row in rows)
            result = dict(complete_count=count, target_count=target_count,
                          generated_count=count - initial_count, complete=count == target_count)
            _atomic_json(root / 'completion.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--gpu-ids', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--target-count', type=int, default=10000)
    parser.add_argument('--max-new-samples', type=int)
    args = parser.parse_args(argv)
    print(json.dumps(complete_dataset(**vars(args)), ensure_ascii=False))


if __name__ == '__main__':
    main()
