"""Complete the FLUX2 image dataset and run three-stage VDN with online encoders."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
from attention_framework.config import CONFIG_ROOT, compose_config


def parse_gpu_ids(value):
    devices = value.split(',')
    if any(not device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError('GPU IDs must be a comma-separated list of distinct nonnegative integers')
    return devices


def write_manifests(data_root, output, *, target_count=10000):
    data_root, output = Path(data_root), Path(output)
    prompts = json.loads((data_root/'prompts.json').read_text())[:target_count]
    if len(prompts) != target_count:
        raise ValueError(f'Need {target_count} original prompts, found {len(prompts)}')
    generation = json.loads((data_root/'run.json').read_text())['config']
    rows = []
    for row in prompts:
        image = data_root/'images'/f"{row['id']}.png"
        metadata = image.with_suffix('.json')
        if not image.is_file() or not metadata.is_file():
            raise ValueError(f'Dataset is incomplete: {row["id"]}')
        saved = json.loads(metadata.read_text())
        if any(saved.get(key) != row[key] for key in ('id', 'prompt', 'seed')):
            raise ValueError(f'Dataset is incomplete: mismatched metadata for {row["id"]}')
        rows.append(dict(id=row['id'], image_path=str(image.relative_to(data_root)),
                         caption=row['prompt'], width=generation['width'], height=generation['height']))
    output.mkdir(parents=True, exist_ok=True)
    manifest = data_root/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    validation = output/'validation.jsonl'
    validation.write_text(''.join(json.dumps(dict(row, image_path=str(data_root/row['image_path'])),
                                           ensure_ascii=False)+'\n' for row in rows[:10]))
    return manifest, validation


def build_training_config(model_dir, data_root, output, gpu_ids, *, batch_size=16, accumulation=1):
    model_dir, data_root, output = (Path(p).expanduser().resolve() for p in (model_dir, data_root, output))
    native = json.loads((model_dir/'transformer/config.json').read_text())
    options = {
        'model.layer_counts.double': native['num_layers'],
        'model.layer_counts.single': native['num_single_layers'],
        'model.resources.weights': str(model_dir/'transformer'),
        'model.resources.text_encoder': str(model_dir/'text_encoder'),
        'model.resources.tokenizer': str(model_dir/'tokenizer'),
        'model.resources.vae': str(model_dir/'vae'),
        'paths.weights': str(model_dir),
        'paths.dataset': str(data_root),
        'paths.output': str(output/'run'),
        'data.manifest': str(data_root/'manifest.jsonl'),
        'data.buckets': [[1024, 1024]],
        'validation.manifest': str(output/'validation.jsonl'),
        'runtime.gpus': len(gpu_ids),
        'training.batch_size': batch_size,
        'training.accumulation': accumulation,
    }
    overrides = [f'{key}={json.dumps(value)}' for key, value in options.items()]
    return compose_config(str(CONFIG_ROOT/'models/flux2.yaml'), str(CONFIG_ROOT/'algorithms/sliding_qdm.yaml'),
                          str(CONFIG_ROOT/'experiments/flux2_sliding11_wavelet_10k.yaml'),
                          str(CONFIG_ROOT/'data/default.yaml'), overrides)


def run(args):
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    workers = args.workers if args.workers is not None else len(gpu_ids)
    if not 1 <= workers <= len(gpu_ids) or args.cpu_threads < 1:
        raise ValueError('Workers must fit the selected GPUs and CPU threads must be positive')
    model_dir, data_root, output = (Path(p).expanduser().resolve() for p in
                                   (args.model_dir, args.data_root, args.output))
    for name in ('transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'):
        if not (model_dir/name).is_dir():
            raise FileNotFoundError(f'Missing FLUX.2-klein-base-4B resource: {model_dir/name}')
    config = build_training_config(model_dir, data_root, output, gpu_ids,
                                   batch_size=args.batch_size, accumulation=args.accumulation)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads),
               PYTORCH_ALLOC_CONF=os.environ.get('PYTORCH_ALLOC_CONF', 'expandable_segments:True'))
    print('Completing the original FLUX2 prompt list to 10,000 paired samples', flush=True)
    subprocess.run([sys.executable, '-m', 'data.complete_flux2', '--model-dir', str(model_dir),
                    '--data-root', str(data_root), '--gpu-ids', args.gpu_ids, '--workers', str(workers),
                    '--cpu-threads', str(args.cpu_threads), '--target-count', '10000'], env=env, check=True)
    write_manifests(data_root, output)
    config_path = output/'config.yaml'
    OmegaConf.save(config, config_path)
    print('Starting Sliding11 + QDM: A1 500 -> A2 1,000 -> B 50,000', flush=True)
    subprocess.run([sys.executable, '-m', 'training.run', '--config', str(config_path)],
                   env=dict(env, CUDA_VISIBLE_DEVICES=','.join(gpu_ids)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-root', required=True, help='Extracted FLUX2-10K directory with prompts.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-ids', required=True, help='Physical GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--workers', type=int, help='Parallel generation workers, default one per selected GPU')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size per GPU in all stages')
    parser.add_argument('--accumulation', type=int, default=1)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
