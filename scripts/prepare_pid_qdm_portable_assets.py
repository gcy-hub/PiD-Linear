"""Prepare only the eight fixed 4K gallery inputs from portable IDs and prompts."""

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


def select_records(reader, cases):
    """Resolve real conditions exclusively from the validation split."""
    validation = {r['id']: r for r in
                  (reader.record(int(i)) for i in np.load(reader.root/'validation.npy'))}
    for case in cases:
        if case['kind'] != 'real':
            continue
        record = validation.get(case['sample_id'])
        if record is None:
            raise ValueError(f'Required image is not held out: {case["sample_id"]}')
        if record['group'] != case['source_url'] or record['caption'] != case['caption']:
            raise ValueError(f'Fixed validation identity differs: {case["sample_id"]}')
        case['record'] = record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('weights-root', 'index-root', 'output-root', 'case-spec'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args(argv)
    rank, world, device = init_distributed()
    torch.set_num_threads(a.threads)
    a.output_root.mkdir(parents=True, exist_ok=True)
    cases = json.loads(a.case_spec.read_text())['cases']
    reader = RawImageDataset(a.index_root)
    select_records(reader, cases)
    for case in cases:
        case['resolution'] = '4096'
        case['file'] = f'{case["kind"]}_{case["ordinal"]:03d}_4096.pt'
    expected = dict(schema=1, data_fingerprint=reader.metadata['fingerprint'],
                    case_spec_sha256=sha256_file(a.case_spec))
    build = a.output_root/'build.json'
    if build.exists() and json.loads(build.read_text()) != expected:
        raise ValueError('Gallery inputs changed; use a new output-root')
    if rank == 0:
        atomic_json(build, expected)
    published = a.output_root/'assets.json'
    if published.exists():
        meta = json.loads(published.read_text())
        if all((a.output_root/c['file']).is_file() and sha256_file(a.output_root/c['file'])==c['sha256']
               for c in meta['cases']):
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        raise ValueError('Published gallery files are corrupt')
    # Native 1024px Z-Image latents condition 4K PiD outputs. The helper frees
    # Z-Image before Gemma and the shared FLUX VAE are loaded.
    prepare_generated_latents(cases, a.output_root, a.weights_root, rank, world, device)
    pending = [c for c in cases[rank::world] if not (a.output_root/c['file']).exists()]
    conditioning = Conditioning(a.weights_root, device) if pending else None
    with torch.no_grad():
        for case in tqdm(pending, desc=f'Fixed 4K inputs rank {rank}'):
            embs, mask = conditioning.encode_text([case['caption']])
            if case['kind']=='real':
                record = case['record']
                with Image.open(record['path']) as im:
                    im = resize_crop(ImageOps.exif_transpose(im).convert('RGB'), record['bucket'],
                                     resolution='4096', allow_upscale=True)
                    pixels = torch.from_numpy(np.array(im,copy=True)).permute(2,0,1)[None].to(device).float()/127.5-1
                latent, _ = conditioning.encode_image(pixels)
                height, width = pixels.shape[-2:]
                del pixels
            else:
                latent = torch.load(a.output_root/(Path(case['file']).stem+'.latent.pt'),
                                    weights_only=True, map_location='cpu')
                height, width = latent.shape[-2]*32, latent.shape[-1]*32
            value = dict(latent=latent.cpu(), caption_embs=embs.cpu(), caption_mask=mask.cpu(),
                         null_embs=conditioning.null_embs.cpu(), null_mask=conditioning.null_mask.cpu(),
                         height=height, width=width, seed=case['seed'], caption=case['caption'],
                         source_image=case.get('record',{}).get('path'))
            destination = a.output_root/case['file']
            temporary = destination.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(destination)
            del value, latent, embs, mask
    del conditioning
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        rows = [{k:c[k] for k in ('kind','ordinal','resolution','seed','file')} |
                dict(sha256=sha256_file(a.output_root/c['file'])) for c in cases]
        atomic_json(published, {**expected,'cases':rows})
        print(f'Prepared exactly {len(rows)} fixed 4K validation inputs.', flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    main()
