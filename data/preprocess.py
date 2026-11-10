"""Deterministic native image preparation shared by online and cached encoding."""
import numpy as np
import torch
from PIL import Image, ImageOps


def prepare_image(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        w, h = size
        scale = max(w / image.width, h / image.height)
        image = image.resize((max(w, round(image.width * scale)), max(h, round(image.height * scale))), Image.Resampling.LANCZOS)
        x, y = (image.width - w) // 2, (image.height - h) // 2
        return image.crop((x, y, x + w, y + h))


def load_pixels(path, size):
    # Keep native BF16 arithmetic order, shared with FLUX.3's pipeline.
    array = np.array(prepare_image(path, size), copy=True)
    return torch.from_numpy(array).to(torch.bfloat16).mul(2 / 255).sub(1).permute(2, 0, 1)
