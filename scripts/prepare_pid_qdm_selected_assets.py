"""Build selected 4K gallery conditions, reusing fixed captions and text encodings.

Generated conditions are fresh native 1024px Z-Image latents, never resized 512px
latents. Real conditions are VAE encodings of the same held-out source images.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from pid._src.linear_pid.data import RawImageDataset, atomic_json, resize_crop, sha256_file
from pid._src.linear_pid.runtime import Conditioning
from pid._src.linear_pid.training import init_distributed
from scripts.prepare_linear_pid_assets import prepare_generated_latents


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-assets', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--index-root', required=True)
    p.add_argument('--weights-root', required=True)
    p.add_argument('--text-cache-root', required=True)
    p.add_argument('--cases', required=True, help='Comma-separated kind_ordinal identifiers')
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    rank, world, device = init_distributed()
    torch.set_num_threads(args.threads)
    source, output = Path(args.source_assets), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / 'assets.json').read_text())
    source_index = json.loads((Path(args.index_root) / 'index.json').read_text())
    if meta['data_fingerprint'] != source_index['fingerprint']:
        raise ValueError('Source gallery and training index differ')
    names = [x.strip() for x in args.cases.split(',') if x.strip()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate cases')
    lookup = {f"{c['kind']}_{c['ordinal']:03d}": c for c in meta['cases'] if str(c['resolution']) == '2048'}
    reader = RawImageDataset(args.index_root)
    # Resolve real sample records strictly within the held-out split.
    validation = {r['path']: r for r in (reader.record(int(i)) for i in np.load(Path(args.index_root) / 'validation.npy'))}
    cases = []
    for name in names:
        original = lookup[name]
        cached = torch.load(source / original['file'], weights_only=True, map_location='cpu')
        case = dict(kind=original['kind'], ordinal=original['ordinal'], resolution='4096',
                    seed=original['seed'], caption=cached['caption'], source_file=original['file'],
                    file=name + '_4096.pt')
        if case['kind'] == 'real':
            case['record'] = validation[cached['source_image']]
        cases.append(case)
    build = dict(schema=1, data_fingerprint=source_index['fingerprint'], resolution=4096,
                 source_assets=str(source.resolve()), cases=names,
                 source_sha256={name: lookup[name]['sha256'] for name in names})
    build_path = output / 'selected-build.json'
    if build_path.exists() and json.loads(build_path.read_text()) != build:
        raise ValueError('Selected gallery inputs changed; choose another directory')
    if rank == 0:
        atomic_json(build_path, build)
    # This helper resumes per latent and per final asset, and uses all selected GPUs.
    prepare_generated_latents(cases, output, args.weights_root, rank, world, device)
    pending = [c for c in cases if not (output / c['file']).exists()]
    conditioning = Conditioning(args.weights_root, device, text_cache_root=args.text_cache_root) if pending else None
    with torch.no_grad():
        for case in tqdm(pending[rank::world], desc=f'4K conditions rank {rank}'):
            cached = torch.load(source / case['source_file'], weights_only=True, map_location='cpu')
            if case['kind'] == 'real':
                record = case['record']
                with Image.open(record['path']) as im:
                    im = resize_crop(ImageOps.exif_transpose(im).convert('RGB'), record['bucket'],
                                     resolution=4096, allow_upscale=True)
                    pixels = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
                latent, _ = conditioning.encode_image(pixels)
                height, width = pixels.shape[-2:]
                del pixels
            else:
                latent = torch.load(output / (Path(case['file']).stem + '.latent.pt'), weights_only=True)
                height, width = latent.shape[-2] * 32, latent.shape[-1] * 32
            asset = {**cached, 'latent': latent.cpu(), 'height': height, 'width': width}
            destination = output / case['file']
            temporary = destination.with_suffix('.tmp')
            torch.save(asset, temporary)
            temporary.replace(destination)
            del asset, cached, latent
    del conditioning
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        published = [{k: c[k] for k in ('kind', 'ordinal', 'resolution', 'seed', 'file')} |
                     dict(sha256=sha256_file(output / c['file'])) for c in cases]
        atomic_json(output / 'assets.json', dict(schema=1, data_fingerprint=source_index['fingerprint'], cases=published))
        print(f'Published {len(published)} selected native 4K conditions: {output}', flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
