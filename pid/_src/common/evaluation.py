"""Shared validation images and conditioning assets."""
import json
from pathlib import Path
import torch
from PIL import Image

def tensor_image(tensor):
    value = tensor.detach()[0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(value)


def save_png(image, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    image.save(temp, format="PNG")
    temp.replace(destination)


def save_views(image, destination):
    destination = Path(destination)
    save_png(image, destination)
    size = min(512, image.width, image.height)
    for label, u, v in [("top_left", 0.0, 0.0), ("center", 0.5, 0.5), ("bottom_right", 1.0, 1.0)]:
        left, top = round((image.width - size) * u), round((image.height - size) * v)
        save_png(
            image.crop((left, top, left + size, top + size)), destination.with_name(f"{destination.stem}_{label}.png")
        )


def asset_cases(root, suite="quick", include_4k=False):
    root = Path(root)
    metadata = json.loads((root / "assets.json").read_text())
    cases = metadata["cases"]
    if suite == "quick":
        chosen = [c for c in cases if c["resolution"] == "2048" and c["ordinal"] < 8]
    else:
        chosen = [c for c in cases if c["resolution"] == "2048" or include_4k]
    return chosen, metadata


def _load_asset(root, case, device):
    asset = torch.load(Path(root) / case["file"], map_location="cpu", weights_only=True)
    for key in ["latent", "caption_embs", "caption_mask", "null_embs", "null_mask"]:
        asset[key] = asset[key].to(device)
    return asset
