"""Read the existing CLEAR FLUX3 safetensors cache as native training samples."""
import csv
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from flux3_qdm.data import image_ids, prompt_ids


class ClearFlux3Dataset:
    def __init__(self, subset_root, cache_root, subset_size):
        self.root = Path(cache_root)
        with (Path(subset_root) / 'metadata.csv').open() as stream:
            self.rows = list(csv.DictReader(stream))
        if len(self.rows) != subset_size:
            raise ValueError(f'Expected {subset_size} rows, found {len(self.rows)}')
        manifests = {}
        for kind, suffix in [('latent', 'latent_code'), ('text', 'prompt_embed')]:
            manifest = json.loads((self.root / f'{kind}-manifest.json').read_text())
            complete = json.loads((self.root / f'{kind}-COMPLETE.json').read_text())
            if complete['count'] != subset_size or complete['fingerprint'] != manifest['fingerprint']:
                raise ValueError(f'Incomplete {kind} cache')
            if manifest['spec']['model']['model_type'] != 'flux3':
                raise ValueError('Expected FLUX3 cache')
            if kind == 'latent':
                self.grid_size = manifest['spec']['resolution'] // 16
            manifests[kind] = manifest['fingerprint']
            for row in self.rows:
                path = self.root / kind / f"{Path(row['image']).stem}_{suffix}.safetensors"
                if not path.is_file():
                    raise FileNotFoundError(path)
        self.fingerprint = manifests
        self.bucket_indices = {'1,1': list(range(len(self.rows)))}
        self.ids = image_ids(self.grid_size, self.grid_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        key = Path(self.rows[index]['image']).stem
        latent = load_file(str(self.root / 'latent' / f'{key}_latent_code.safetensors'))['latents']
        text = load_file(str(self.root / 'text' / f'{key}_prompt_embed.safetensors'))['prompt_embeds']
        if latent.shape != (self.grid_size ** 2, 128) or text.ndim != 2 or text.shape[1] != 20480:
            raise ValueError(f'Invalid FLUX3 cache shapes for {key}')
        return dict(id=key, image_latents=latent.to(torch.bfloat16), image_ids=self.ids,
                    prompt_tokens=text.to(torch.bfloat16), prompt_ids=prompt_ids(len(text)), grid=(self.grid_size, self.grid_size))
