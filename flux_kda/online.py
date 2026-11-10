"""Read an exported image/caption manifest and encode batches without disk caches."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pid._src.datasets.buckets import MULTI_RES_GRID_BY_MAX_RESOLUTION, _classify_aspect_ratio
from flux3_qdm.data import prepare_image, image_ids, fingerprint


class OnlineDataset:
    def __init__(self, root, max_resolution=4096):
        root = Path(root).expanduser().resolve()
        manifest = root if root.is_file() else root / 'manifest.jsonl'
        self.rows, self.bucket_indices = [], defaultdict(list)
        self.rejected = 0
        grid = MULTI_RES_GRID_BY_MAX_RESOLUTION[str(max_resolution)]
        with manifest.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                path = Path(row['image_path'])
                if not path.is_absolute():
                    path = manifest.parent / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                if not isinstance(row['caption'], str):
                    raise ValueError(f'Missing caption: {path}')
                # Dimensions are read after EXIF orientation, as in the PiD image loader.
                with Image.open(path) as image:
                    width, height = image.size
                    if image.getexif().get(274) in (5, 6, 7, 8):
                        width, height = height, width
                aspect = _classify_aspect_ratio(width, height)
                target = next(((level, w, h) for level, w, h in grid[aspect]
                               if width >= w and height >= h), None)
                if target is None:
                    self.rejected += 1
                    continue
                level, w, h = target
                key = f'L{level}_{aspect}'
                self.bucket_indices[key].append(len(self.rows))
                self.rows.append(dict(id=str(row.get('id', path)), image_path=str(path),
                                      caption=row['caption'], size=(w, h)))
        if not self.rows:
            raise ValueError('No images fit the official PiD minimum 2K buckets')
        # Reuse the existing dataset identity convention for checkpoint resume checks.
        self.fingerprint = fingerprint(dict(rows=self.rows, max_resolution=max_resolution))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class OnlineEncoder:
    def __init__(self, resources, device):
        from flux2_qdm.diffsynth_model import import_framework
        import_framework()
        from flux3_qdm.prepare import _load_phase
        from diffsynth.pipelines.flux3_image import Flux3Unit_PromptEmbedder
        self.text = _load_phase('text', resources, device)
        self.image = _load_phase('image', resources, device)
        self.prompt = Flux3Unit_PromptEmbedder()
        self.device = device

    @torch.no_grad()
    def __call__(self, rows):
        samples = []
        for row in rows:
            pixels = torch.from_numpy(np.array(prepare_image(row['image_path'], row['size']), copy=True))
            pixels = pixels.to(device=self.device, dtype=torch.bfloat16)
            pixels = (pixels * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
            latent = self.image.vae.encode(pixels)[0]
            del pixels
            h, w = latent.shape[-2:]
            text = self.prompt.process(self.text, row['caption'])
            samples.append(dict(id=row['id'], image_latents=latent.permute(1, 2, 0).reshape(-1, 128),
                                image_ids=image_ids(h, w), grid=(h, w),
                                prompt_tokens=text['prompt_tokens'], prompt_ids=text['prompt_ids']))
        return samples
