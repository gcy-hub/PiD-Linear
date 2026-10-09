"""A0: spatial error and held-out latent predictability, without changing PiD."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint, student_weights
from pid._src.linear_pid.compare_inference import ASSET_KEYS, validate_asset, file_identity
from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.runtime import build_net, load_original
from pid._src.linear_pid.spatial_diagnosis import (
    completed_case, concentration, publish_case, rank_correlation, ridge_scores,
    selected_mass, split_prompt_cases, tile_pool, token_errors,
)


def read_asset(path):
    asset = torch.load(path, map_location='cpu', weights_only=True)
    validate_asset(asset)
    return asset


def image_pixels(path, asset, device):
    with Image.open(path) as image:
        if image.size != (asset['width'], asset['height']):
            raise ValueError(f'Image dimensions differ from the asset: {path}')
        pixels = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device).div_(127.5).sub_(1)


def checked_image(directory, path, asset, digest, model):
    if not directory:
        return None
    image = Path(directory) / (path.stem + '.png')
    metadata = json.loads(image.with_suffix('.json').read_text())
    settings = metadata['settings']
    if (settings.get('input_sha256', settings.get('asset_sha256')) != digest
            or metadata['seed'] != asset['seed'] or metadata['caption'] != asset['caption']
            or settings.get('model') != model):
        raise ValueError(f'Reference image is not the paired {model} output: {image}')
    return image


def tile_features(tokens, grid, tile, projection=None):
    h, w = grid
    channels = tokens.shape[-1]
    values = tokens.view(1, h, w, channels).permute(0, 3, 1, 2)
    values = F.avg_pool2d(values.float(), tile, stride=tile, ceil_mode=True,
                          count_include_pad=False)[0].permute(1, 2, 0)
    if projection is not None:
        values = values @ projection.float()
    return values.detach().float().cpu().numpy()


def attention_call(module, inputs, kwargs, grid, valid_mask):
    # Cross-probes must bypass observation hooks; Module.__call__ would recurse
    # between the paired teacher/student hooks. Kernels and weights are identical.
    if getattr(module, 'is_linear_attention', False):
        return module.forward(*inputs[:4], grid_size=grid, text_valid_mask=valid_mask)
    mask = inputs[4] if len(inputs) > 4 else kwargs.get('attn_mask')
    return module.forward(*inputs[:4], attn_mask=mask)


class SpatialProbe:
    """Use hooks so official forward semantics and residuals remain unchanged."""
    def __init__(self, teacher, student, layers, grid, valid_mask, tile, projection):
        self.teacher, self.student, self.layers = teacher, student, layers
        self.grid, self.valid_mask, self.tile = grid, valid_mask, tile
        self.projection, self.handles = projection, []
        self.teacher_outputs, self.maps, self.features = {}, {}, {}

    def errors(self, name, student, teacher):
        absolute, relative = token_errors(student, teacher)
        for suffix, value in [('absolute', absolute), ('relative', relative)]:
            self.maps[name + '_' + suffix] = value.detach().float().cpu().numpy().reshape(self.grid)

    def teacher_hook(self, layer):
        def hook(module, inputs, kwargs, output):
            other = attention_call(self.student.patch_blocks[layer].attn, inputs, kwargs,
                                   self.grid, self.valid_mask)
            self.errors(f'layer{layer}_teacher_input', other[0], output[0])
            self.teacher_outputs[layer] = output[0].detach()
        return hook

    def student_hook(self, layer):
        def hook(module, inputs, kwargs, output):
            other = attention_call(self.teacher.patch_blocks[layer].attn, inputs, kwargs,
                                   self.grid, self.valid_mask)
            self.errors(f'layer{layer}_student_input', output[0], other[0])
            self.errors(f'layer{layer}_own_input', output[0], self.teacher_outputs.pop(layer))
        return hook

    def lq_hook(self, module, inputs, output):
        for layer in self.layers:
            index = module._get_output_index(layer)
            self.features[f'layer{layer}_adapter'] = tile_features(
                output[index], self.grid, self.tile, self.projection)

    def __enter__(self):
        self.handles.append(self.teacher.lq_proj.register_forward_hook(self.lq_hook))
        for layer in self.layers:
            self.handles.append(self.teacher.patch_blocks[layer].attn.register_forward_hook(
                self.teacher_hook(layer), with_kwargs=True))
            self.handles.append(self.student.patch_blocks[layer].attn.register_forward_hook(
                self.student_hook(layer), with_kwargs=True))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.teacher_outputs.clear()


def predict(net, noisy, timestep, asset):
    return net(noisy, torch.full((1,), timestep, device=noisy.device, dtype=torch.float32),
               asset['caption_embs'], lq_latent=asset['latent'],
               degrade_sigma=torch.zeros(1, device=noisy.device),
               text_valid_mask=asset['caption_mask'])


def pixel_patch_error(student, reference, patch=16):
    values = (student.float() - reference.float()).square().mean(1, keepdim=True)
    return F.avg_pool2d(values, patch, stride=patch)[0, 0].detach().cpu().numpy()


def edge_map(image, patch=16):
    gray = image.float().mean(1, keepdim=True)
    gradient = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1))
    gradient += F.pad((gray[:, :, 1:] - gray[:, :, :-1]).abs(), (0, 0, 0, 1))
    return F.avg_pool2d(gradient, patch, stride=patch)[0, 0].cpu().numpy()


def save_heatmap(maps, layers, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(layers), 4, figsize=(13, 3 * len(layers)), squeeze=False)
    for row, layer in enumerate(layers):
        keys = [f'layer{layer}_teacher_input_absolute', f'layer{layer}_student_input_absolute',
                f'layer{layer}_own_input_absolute', 'velocity_mse']
        for axis, key in zip(axes[row], keys):
            value = np.log1p(maps[key])
            im = axis.imshow(value, cmap='magma', vmin=np.quantile(value, .01),
                             vmax=max(float(np.quantile(value, .99)), 1e-8), interpolation='nearest')
            axis.set_title(key + '\nlog1p; individual color scale', fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(im, ax=axis, shrink=.7)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def record_metrics(maps, layers, tile, fraction):
    metrics = []
    for layer in layers:
        for context in ['teacher_input', 'student_input', 'own_input']:
            name = f'layer{layer}_{context}'
            absolute = maps[name + '_absolute']
            pooled = tile_pool(absolute, tile)
            result = dict(layer=layer, context=context, mean_absolute=float(absolute.mean()),
                          mean_relative=float(maps[name + '_relative'].mean()),
                          token_concentration=concentration(absolute, fraction),
                          tile_concentration=concentration(pooled, fraction),
                          velocity_spearman=rank_correlation(pooled, tile_pool(maps['velocity_mse'], tile)))
            for key in ['generation_mse', 'reference_edge']:
                if key in maps:
                    result[key + '_spearman'] = rank_correlation(pooled, tile_pool(maps[key], tile))
            metrics.append(result)
    return metrics


def router_comparison(records, output, train_prompts, tile, fraction, penalty):
    train, heldout = split_prompt_cases(records, train_prompts)
    layers = records[0]['layers']
    results = []
    for layer in layers:
        # Keep resolution and timestep strata: no train/test split of neighboring tiles.
        for resolution in sorted({r['resolution'] for r in records}):
            for timestep in sorted({r['timestep'] for r in records}):
                source = [r for r in train if r['resolution'] == resolution and r['timestep'] == timestep]
                target = [r for r in heldout if r['resolution'] == resolution and r['timestep'] == timestep]
                if not source or not target:
                    continue
                def arrays(record):
                    data = np.load(record['directory'] / 'maps.npz')
                    label = tile_pool(data[f'layer{layer}_student_input_absolute'], tile).ravel()
                    adapter = data[f'layer{layer}_adapter'].reshape(len(label), -1)
                    latent = data['latent_tiles'].reshape(len(label), -1)
                    h, w = data[f'layer{layer}_adapter'].shape[:2]
                    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
                    coords = np.column_stack([yy.ravel(), xx.ravel()])
                    return label, dict(adapter=np.column_stack([adapter, coords]),
                                       latent=np.column_stack([latent, coords]), position=coords), data
                training = [arrays(r) for r in source]
                labels = np.concatenate([np.log1p(v[0]) for v in training])
                for record in target:
                    error, features, data = arrays(record)
                    row = dict(case=record['id'], prompt_id=record['prompt_id'], resolution=resolution,
                               timestep=timestep, layer=layer, oracle_mass=concentration(error, fraction)['top_mass'],
                               uniform_expected_mass=concentration(error, fraction)['uniform_expected_mass'])
                    random = np.random.default_rng(42)
                    row['random_mass'] = float(np.mean([selected_mass(error, random.random(len(error)), fraction)
                                                       for _ in range(32)]))
                    for name in ['position', 'latent', 'adapter']:
                        x = np.concatenate([v[1][name] for v in training])
                        score = ridge_scores(x, labels, features[name], penalty)
                        row[name + '_mass'] = selected_mass(error, score, fraction)
                        row[name + '_spearman'] = rank_correlation(error, score)
                    if 'reference_edge' in data:
                        edge = tile_pool(data['reference_edge'], tile).ravel()
                        row['edge_mass'] = selected_mass(error, edge, fraction)
                    results.append(row)
    atomic_json(output / 'router_evaluation.json', dict(train_prompt_ids=sorted({r['prompt_id'] for r in train}),
                heldout_prompt_ids=sorted({r['prompt_id'] for r in heldout}), results=results,
                target='Absolute same-student-input teacher/student attention error, log1p for ridge fitting',
                note='Exploratory ridge probe; no local correction or image-quality recovery is demonstrated'))
    if results:
        columns = sorted(set().union(*(r.keys() for r in results)))
        with (output / 'router_evaluation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
    return results


def summarize(output, expected, args):
    records, rows = [], []
    for path in sorted((output / 'cases').glob('*/complete.json')):
        record = json.loads(path.read_text())
        settings = record['settings']
        if not completed_case(path.parent, settings):
            continue
        records.append(dict(id=settings['case'], prompt_id=settings['prompt_id'],
                            resolution=settings['resolution'], timestep=settings['timestep'],
                            layers=settings['layers'], directory=path.parent))
        for metric in record['metrics']['spatial']:
            row = dict(case=settings['case'], resolution=settings['resolution'], timestep=settings['timestep'],
                       layer=metric['layer'], context=metric['context'],
                       mean_absolute=metric['mean_absolute'], mean_relative=metric['mean_relative'],
                       top_tile_mass=metric['tile_concentration']['top_mass'],
                       uniform_expected_mass=metric['tile_concentration']['uniform_expected_mass'],
                       tile_gini=metric['tile_concentration']['gini'],
                       velocity_spearman=metric['velocity_spearman'])
            for key in ['generation_mse_spearman', 'reference_edge_spearman']:
                row[key] = metric.get(key)
            rows.append(row)
    if rows:
        with (output / 'spatial_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    router = router_comparison(records, output, args.train_prompts, args.tile_size,
                               args.top_fraction, args.ridge_penalty) if records else []
    atomic_json(output / 'summary.json', dict(completed_cases=len(records), expected_cases=expected,
                complete=len(records) == expected, spatial=rows, router=router,
                limits=['Five prompts are exploratory, not a statistical image-quality benchmark.',
                        'Attention errors mix mechanism, learned weights, padding behavior and feature-basis differences.',
                        'Teacher outputs are references, not ground truth.',
                        'Oracle error selection does not prove local attention can repair the selected regions.',
                        'This probe does not modify inference or measure a speedup.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--asset-dir', required=True)
    parser.add_argument('--asset-glob', default='prompt_*_[24]0??.pt',
                        help='Default covers prompt_*_2048.pt and prompt_*_4096.pt')
    parser.add_argument('--output-dir', default='./outputs/research/spatial-error-diagnosis')
    parser.add_argument('--weights-root')
    parser.add_argument('--reference-dir', help='Optional paired original PiD PNG/JSON directory')
    parser.add_argument('--student-images-dir', help='Optional paired trained-student PNG/JSON directory')
    parser.add_argument('--layers', default='0,4,8,12,7', help='KDA layers plus retained Full Attention layer 7 control')
    parser.add_argument('--timesteps', default='100,500,900')
    parser.add_argument('--tile-size', type=int, default=8)
    parser.add_argument('--feature-dim', type=int, default=32)
    parser.add_argument('--top-fraction', type=float, default=.2)
    parser.add_argument('--train-prompts', type=int, default=3)
    parser.add_argument('--ridge-penalty', type=float, default=1.)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--swanlab-mode', choices=['offline', 'cloud', 'disabled'], default='offline')
    args = parser.parse_args()
    if min(args.threads, args.tile_size, args.feature_dim, args.train_prompts) < 1 or args.workers < 0:
        parser.error('Threads, tile/features and train prompts must be positive; workers nonnegative')
    if not 0 < args.top_fraction <= 1 or args.ridge_penalty <= 0:
        parser.error('Selection fraction must be in (0,1] and ridge penalty positive')
    layers = sorted({int(v) for v in args.layers.split(',')})
    timesteps = sorted({int(v) for v in args.timesteps.split(',')})
    if not layers or min(layers) < 0 or max(layers) >= 14 or not timesteps or min(timesteps) < 0 or max(timesteps) > 1000:
        parser.error('Invalid layer indices or timesteps')
    torch.set_num_threads(args.threads)
    rank, local = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('gloo')
    paths = sorted(Path(args.asset_dir).glob(args.asset_glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f'No input assets match {args.asset_glob}')
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f'A0: {len(paths)} assets x {len(timesteps)} timesteps; loading checkpoint', flush=True)
    payload = load_checkpoint(checkpoint)
    config = LinearPiDConfig.from_dict(payload['config'])
    if payload['config'].get('pit_kda', False):
        raise ValueError('A0 requires an original-PiT student checkpoint')
    if args.weights_root:
        config.weights_root = args.weights_root
    teacher_digest = sha256_file(config.teacher_path)
    if teacher_digest != payload['metadata']['teacher_sha256']:
        raise ValueError('Official teacher differs from the student initialization')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # All settings affecting measurements are in the resume signature; workers may change.
    signature = dict(schema=1, checkpoint=file_identity(Path(checkpoint) / 'state.pt'
                     if Path(checkpoint).is_dir() else checkpoint), teacher_sha256=teacher_digest,
                     layers=layers, student_kda_layers=config.layers, timesteps=timesteps,
                     tile_size=args.tile_size, feature_dim=args.feature_dim, seed=42,
                     top_fraction=args.top_fraction, precision='bf16', source='teacher-generated pseudo-reference'
                     if args.reference_dir else 'fixed Gaussian pixels; off trajectory',
                     assets=[dict(path=str(p.resolve()), sha256=sha256_file(p)) for p in paths])
    run_signature = output / 'experiment.json'
    if run_signature.exists() and json.loads(run_signature.read_text()) != signature:
        raise ValueError('Experiment settings changed; choose a different output directory')
    if rank == 0:
        atomic_json(run_signature, signature)
    if world > 1:
        torch.distributed.barrier()
    torch.manual_seed(42)
    print('Building frozen official teacher and student (PiT unchanged)', flush=True)
    teacher = build_net().to(dtype=torch.bfloat16)
    load_original(teacher, config.teacher_path)
    student = build_net(config.layers, local_mixing=config.local_mixing).to(dtype=torch.bfloat16)
    student.load_state_dict(student_weights(payload), strict=True)
    del payload
    teacher = teacher.to(device).eval().requires_grad_(False)
    student = student.to(device).eval().requires_grad_(False)
    print('Both models are resident on GPU; starting spatial probes', flush=True)
    teacher.pit_chunk_size = student.pit_chunk_size = 2048
    projection = torch.randn(teacher.hidden_size, args.feature_dim,
                              generator=torch.Generator().manual_seed(42)).to(device) / args.feature_dim**.5
    run = None
    if args.swanlab_mode != 'disabled':
        import swanlab
        run = swanlab.init(project='Linear-PiD-Research', experiment_name=f'Spatial-Error-Diagnosis-rank{rank}',
                           mode=args.swanlab_mode, logdir=str(output / 'swanlog'), config=signature)
    assigned = paths[rank::world]
    try:
        pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers else None
        loaded = pool.map(read_asset, assigned) if pool else map(read_asset, assigned)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for path, asset in tqdm(zip(assigned, loaded), total=len(assigned), desc=f'Spatial A0 rank {rank}'):
                digest = sha256_file(path)
                prompt_id = asset.get('source_caption_id', path.stem.rsplit('_', 1)[0])
                reference_path = checked_image(args.reference_dir, path, asset, digest, 'original')
                student_path = checked_image(args.student_images_dir, path, asset, digest, 'trained')
                for key in ASSET_KEYS:
                    asset[key] = asset[key].to(device)
                grid = asset['height'] // 16, asset['width'] // 16
                reference = image_pixels(reference_path, asset, device) if reference_path else None
                noise = torch.randn(1, 3, asset['height'], asset['width'], device=device,
                                    generator=torch.Generator(device=device).manual_seed(asset['seed']))
                shared_maps = {}
                if reference is not None:
                    shared_maps['reference_edge'] = edge_map(reference)
                if student_path and reference is not None:
                    generated = image_pixels(student_path, asset, device)
                    shared_maps['generation_mse'] = pixel_patch_error(generated, reference)
                    del generated
                latent = F.interpolate(asset['latent'].float(), size=grid, mode='bilinear', align_corners=False)
                shared_maps['latent_tiles'] = tile_features(latent.flatten(2).transpose(1, 2), grid, args.tile_size)
                for timestep in timesteps:
                    case = f'{path.stem}_t{timestep:04d}'
                    destination = output / 'cases' / case
                    settings = dict(signature=signature, case=case, prompt_id=prompt_id,
                                    resolution=path.stem.rsplit('_', 1)[-1], timestep=timestep, layers=layers,
                                    reference=file_identity(reference_path) if reference_path else None,
                                    paired_student=file_identity(student_path) if student_path else None)
                    if completed_case(destination, settings):
                        if not (destination / 'heatmaps.png').exists():
                            with np.load(destination / 'maps.npz') as saved:
                                save_heatmap(saved, layers, destination / 'heatmaps.png')
                        continue
                    noisy = noise if reference is None else reference * (1 - timestep / 1000.) + noise * (timestep / 1000.)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    with SpatialProbe(teacher, student, layers, grid, asset['caption_mask'],
                                      args.tile_size, projection) as probe:
                        prediction_teacher = predict(teacher, noisy, timestep, asset)
                        prediction_student = predict(student, noisy, timestep, asset)
                    maps = dict(probe.maps, **probe.features, **shared_maps)
                    maps['velocity_mse'] = pixel_patch_error(prediction_student, prediction_teacher)
                    if not all(np.isfinite(v).all() for v in maps.values()):
                        raise FloatingPointError(f'Nonfinite diagnostic tensors: {case}')
                    torch.cuda.synchronize(device)
                    metrics = dict(spatial=record_metrics(maps, layers, args.tile_size, args.top_fraction),
                                   probe_seconds=time.perf_counter() - start,
                                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                                   input_protocol=signature['source'], caption=asset['caption'], seed=asset['seed'])
                    publish_case(destination, settings, maps, metrics)
                    save_heatmap(maps, layers, destination / 'heatmaps.png')
                    print(f'{case}: probe={metrics["probe_seconds"]:.2f}s, '
                          f'peak={metrics["peak_allocated_gib"]:.2f}GiB', flush=True)
                    if run:
                        import swanlab
                        swanlab.log({'diagnosis/velocity_mse': float(maps['velocity_mse'].mean()),
                                     'diagnosis/probe_seconds': metrics['probe_seconds'],
                                     'diagnosis/peak_gib': metrics['peak_allocated_gib'],
                                     'diagnosis/heatmap': swanlab.Image(str(destination / 'heatmaps.png'))})
                    del prediction_teacher, prediction_student, noisy, maps, probe
                del reference, noise, asset, latent, shared_maps
        if pool:
            pool.shutdown()
        if world > 1:
            torch.distributed.barrier()
        if rank == 0:
            summarize(output, len(paths) * len(timesteps), args)
    finally:
        if run:
            import swanlab
            swanlab.finish()
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
