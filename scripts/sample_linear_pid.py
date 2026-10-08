"""Standalone student sampling from a local image or normalized FLUX latent."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint, student_weights
from pid._src.linear_pid.data import aspect_bucket, atomic_json, resize_crop
from pid._src.linear_pid.environment import check_runtime
from pid._src.linear_pid.evaluation import save_views, tensor_image
from pid._src.linear_pid.runtime import Conditioning, build_net, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--latent", help="Normalized FLUX latent tensor or gallery .pt asset")
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-root", default="/home/ganchangyi/huggingface_ckpts")
    parser.add_argument("--weights", choices=["ema", "raw"], default="raw")
    parser.add_argument("--resolution", choices=["2048", "4096"], default="2048")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 2 or args.cfg < 1 or args.shift <= 0:
        parser.error("steps >= 2, cfg >= 1 and shift > 0 are required")
    check_runtime()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    payload = load_checkpoint(resolve_checkpoint(args.checkpoint))
    config = LinearPiDConfig.from_dict(payload["config"])
    net = build_net(config.layers, config.local_mixing, pit_kda=config.pit_kda,
                    pit_kda_heads=config.pit_kda_heads, pit_kda_compressed=config.pit_kda_compressed)
    net.load_state_dict(student_weights(payload, args.weights), strict=True)
    net = net.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    del payload
    conditioning = Conditioning(args.weights_root, device)
    if args.latent:
        latent = torch.load(args.latent, weights_only=True, map_location="cpu")
        if isinstance(latent, dict):
            latent = latent["latent"]
        if latent.ndim != 4 or latent.shape[:2] != (1, 16):
            raise ValueError("Expected normalized FLUX latent [1,16,H,W]")
        latent = latent.to(device)
        h, w = latent.shape[-2] * 32, latent.shape[-1] * 32
    else:
        import numpy as np

        with Image.open(args.image) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = resize_crop(image, aspect_bucket(*image.size), args.resolution, allow_upscale=True)
            array = np.array(image, copy=True)
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
        latent, _ = conditioning.encode_image(pixels)
        h, w = pixels.shape[-2:]
        del pixels
    embs, mask = conditioning.encode_text([args.caption])
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = sample(
            net,
            embs,
            mask,
            conditioning.null_embs,
            conditioning.null_mask,
            latent,
            height=h,
            width=w,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            shift=args.shift,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    output = Path(args.output)
    save_views(tensor_image(result), output)
    atomic_json(output.with_suffix(".json"), vars(args) | {"height": h, "width": w, "decode_seconds": elapsed})
    print(json.dumps({"output": str(output), "decode_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
