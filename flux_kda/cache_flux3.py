"""Prepare resumable 1K FLUX3 latents and reuse the existing text cache."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--resources', default='weights')
    args = parser.parse_args()
    args.data = args.data.resolve()
    root = args.data / 'cache-flux3-1024'
    source = args.data / 'cache-flux3'
    with (args.data / 'metadata.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2000
    if args.rank is None:
        if (root / 'latent-COMPLETE.json').exists():
            return
        (root / 'latent').mkdir(parents=True, exist_ok=True)
        if not (root / 'text').exists():
            (root / 'text').symlink_to(source / 'text', target_is_directory=True)
        for name in ('text-manifest.json', 'text-COMPLETE.json'):
            shutil.copyfile(source / name, root / name)
        manifest = json.loads((source / 'latent-manifest.json').read_text())
        manifest['fingerprint'] = dict(source=manifest['fingerprint'], resolution=1024)
        manifest['spec']['resolution'] = 1024
        (root / 'latent-manifest.json').write_text(json.dumps(manifest))
        children = [subprocess.Popen([sys.executable, '-m', 'flux_kda.cache_flux3', '--data', str(args.data), '--rank', str(rank), '--resources', args.resources],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank+1))) for rank in range(3)]
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(f'Cache workers failed: {codes}')
        from .data import ClearFlux3Dataset
        temporary = root / 'latent-COMPLETE.partial.json'
        temporary.write_text(json.dumps(dict(fingerprint=manifest['fingerprint'], count=len(rows))))
        temporary.replace(root / 'latent-COMPLETE.json')
        dataset = ClearFlux3Dataset(args.data, root, 2000)
        assert dataset[0]['image_latents'].shape == (4096, 128)
        print('1K cache complete: 2000 images', flush=True)
        return
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    from flux2_qdm.diffsynth_model import import_framework
    import_framework()
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux3_image import Flux3ImagePipeline
    from flux3_qdm.resources import resource_paths
    torch.set_num_threads(4)
    pipe = Flux3ImagePipeline.from_pretrained(device='cuda', torch_dtype=torch.bfloat16,
        model_configs=[ModelConfig(path=resource_paths(args.resources)['vae'])], processor_config=None)
    with torch.inference_mode():
        for index in range(args.rank, len(rows), 3):
            row = rows[index]
            path = root / 'latent' / f"{Path(row['image']).stem}_latent_code.safetensors"
            if path.exists():
                continue
            with Image.open(args.data / row['image']) as image:
                array = np.array(ImageOps.exif_transpose(image).convert('RGB').resize((1024,1024), Image.Resampling.LANCZOS))
            image = torch.from_numpy(array).to(device='cuda', dtype=torch.bfloat16)
            image = (image * (2/255) - 1).permute(2,0,1).unsqueeze(0)
            latent = pipe.vae.encode(image)
            assert latent.shape == (1,128,64,64) and torch.isfinite(latent).all()
            latent = latent[0].permute(1,2,0).reshape(-1,128).cpu().contiguous()
            temporary = path.with_suffix('.partial')
            save_file({'latents': latent}, str(temporary))
            temporary.replace(path)
            if index % 99 == args.rank:
                print(f'cache rank {args.rank}: image {index+1}/2000', flush=True)


if __name__ == '__main__':
    main()
