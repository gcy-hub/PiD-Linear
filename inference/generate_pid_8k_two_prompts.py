#!/usr/bin/env python3
"""Generate two independent 8192x8192 images with exported or original PiD.

Each job performs one complete global 8K decode on one GPU. No tiling or
stitching is used. Local imports resolve from this script’s repository.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(os.environ.get("PID_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from omegaconf import OmegaConf, open_dict

PROMPTS = (
    "A square, photorealistic travel photograph of an ornate Russian cathedral inspired by "
    "Saint Basil’s Cathedral on Moscow’s Red Square, seen from across a broad cobblestone "
    "plaza at pedestrian eye level. The cathedral sits slightly right of center, with coral-red "
    "brick walls, intricate white trim, arched windows, layered green roofs, and a cluster of "
    "colorful onion domes. One large dome has turquoise-and-ivory spiral ribs, a smaller one has "
    "cobalt-blue-and-white spirals, and the others are covered with red, green, and gold diamond "
    "patterns. A tall red tent-roof spire rises above the domes, capped by a gilded finial and a "
    "small flag. A separate slender red-brick tower stands near the left edge, with distant "
    "green-roofed towers on the right. Small pedestrians in casual jackets cross the gray "
    "cobblestones along the bottom edge, giving the buildings a monumental sense of scale. A broad "
    "expanse of pale blue sky with thin clouds and a few distant birds frames the upper spires. "
    "Bright, slightly hazy daylight, soft shadows, muted pastel colors, low contrast, crisp "
    "architectural details, candid documentary photography.",
    "A square, cinematic chest-up portrait of a young East Asian woman seated indoors, looking "
    "straight into the camera with a quiet, serious expression and slightly parted lips. She has "
    "shoulder-length, softly tousled chestnut-brown waves, wispy bangs grazing her eyebrows, and "
    "warm brown eyes. Her makeup features softly smoked reddish-brown eyeliner, peach blush, and "
    "glossy brick-red lipstick. She wears a loose cocoa-brown ribbed knit top, with one arm "
    "extending toward the lower-left edge of the frame. A narrow shaft of warm late-afternoon "
    "sunlight cuts diagonally across the viewer-right side of her face, catching one eye and the "
    "top of her cheek before falling across her neck and collarbone. The rest of her face stays in "
    "soft shadow. A tall, softly blurred multi-pane window sits in the upper-left background of a "
    "dim room. Warm amber highlights, deep brown shadows, subtle film grain, natural skin texture, "
    "sharp focus on the eyes, and shallow depth of field. One continuous, edge-to-edge photograph.",
)


def save_tensor(value, path):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def config_for(repo, weights_root, model="export"):
    config = OmegaConf.load(repo / "config.yaml" if model == "export"
                            else REPO_ROOT / "configs/default.yaml")
    root = Path(weights_root).expanduser()
    with open_dict(config):
        config.model.family = "pid"
        config.model.architecture = "native"
        config.model.resources.export = str(repo) if model == "export" else None
        config.runtime.dtype = "bfloat16"
        config.runtime.gradient_checkpointing = False
        config.model.resources.weights = str(root / "PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth")
        config.model.resources.text_encoder = str(root / "gemma-2-2b-it")
        config.model.resources.tokenizer = None
        config.model.resources.vae = str(root / "PiD/checkpoints/ae.safetensors")
    return config


def encode_text(config, prompt, device, output):
    from data.encoders import build_encoder
    from pid._src.common.text_encoder import NEGATIVE_PROMPT
    encoder = build_encoder(config, device=device, image=False, text=True)
    with torch.inference_mode():
        positive = encoder.encode_text([prompt])
        negative = encoder.encode_text([NEGATIVE_PROMPT])
    torch.cuda.synchronize(device)
    save_tensor({"positive": {k: v.cpu() for k, v in positive.items()},
                 "negative": {k: v.cpu() for k, v in negative.items()},
                 "prompt": prompt}, output / "text.pt")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def make_latent(weights_root, prompt, seed, device, output):
    from pid._src.inference.pipeline_registry import PIPELINE_REGISTRY, extract_latent
    spec = PIPELINE_REGISTRY["zimage-turbo"]
    model_path = Path(weights_root).expanduser() / "Z-Image-Turbo"
    module, name = spec.pipeline_class.rsplit(".", 1)
    pipeline = getattr(importlib.import_module(module), name).from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipeline.enable_model_cpu_offload(gpu_id=device.index)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        raw = pipeline(prompt=prompt, height=2048, width=2048,
                       num_inference_steps=spec.default_num_inference_steps,
                       guidance_scale=spec.default_guidance_scale, generator=generator,
                       output_type="latent", **spec.extra_generate_kwargs)
        latent = extract_latent(pipeline, raw, spec, 2048, 2048)
    torch.cuda.synchronize(device)
    assert tuple(latent.shape) == (1, 16, 256, 256), tuple(latent.shape)
    save_tensor({"latent": latent.cpu(), "prompt": prompt, "seed": seed}, output / "latent.pt")
    del pipeline, raw, latent
    gc.collect()
    torch.cuda.empty_cache()


def repo_root():
    return Path(os.environ.get("PID_REPO_ROOT", REPO_ROOT)).resolve()


def run_job(repo, weights_root, output_root, job_index, gpu, seed, steps, cfg, shift, model="export"):
    output = output_root / f"image-{job_index + 1:02d}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log = output / "generation.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--job", str(job_index),
               "--model", model, "--weights-root", str(weights_root),
               "--output", str(output), "--seed", str(seed), "--steps", str(steps),
               "--cfg", str(cfg), "--shift", str(shift)]
    if repo is not None:
        command += ["--repo", str(repo)]
    with log.open("w") as handle:
        subprocess.run(command, cwd=repo_root(), env=environment, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    print(f"image {job_index + 1}: GPU {gpu}, {time.perf_counter() - started:.2f}s", flush=True)


def generate_job(args):
    repo = args.repo
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = config_for(repo, args.weights_root, args.model)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    prompt = PROMPTS[args.job]
    generation_started = time.perf_counter()
    encode_text(config, prompt, device, output)
    make_latent(args.weights_root, prompt, args.seed, device, output)

    decode_started = time.perf_counter()
    net, fused = load_decoder(config, device, args.model)
    from pid._src.common.runtime import sample
    from pid._src.vdn_pid.attention import execution_mode

    text = torch.load(output / "text.pt", map_location=device, weights_only=True)
    latent = torch.load(output / "latent.pt", map_location=device, weights_only=True)["latent"].to(torch.bfloat16)
    with torch.inference_mode(), (execution_mode(net, inference=True)
                                  if args.model == "export" else nullcontext()):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        generated = sample(
            net, text["positive"]["embeddings"], text["positive"]["mask"],
            text["negative"]["embeddings"], text["negative"]["mask"], latent,
            height=8192, width=8192, seed=args.seed, steps=args.steps,
            cfg=args.cfg, shift=args.shift, sigma=0.0,
        )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    pixels = ((generated[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
    image_path = output / "pid-8192.png"
    Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy()).save(image_path, compress_level=1)
    finished = time.perf_counter()
    with Image.open(image_path) as image:
        assert image.size == (8192, 8192) and image.mode == "RGB"
    (output / "result.json").write_text(json.dumps({
        "prompt": prompt, "seed": args.seed, "width": 8192, "height": 8192,
        "steps": args.steps, "cfg": args.cfg, "shift": args.shift,
        "model": args.model, "repo": str(repo) if repo else None,
        "generation_seconds": finished - generation_started,
        "decode_seconds": finished - decode_started,
        "sampling_seconds": seconds, "fused_lora_projections": fused,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "global_decode": True,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"job {args.job + 1}: generation {finished - generation_started:.2f}s, "
          f"sampling {seconds:.2f}s", flush=True)


def load_decoder(config, device, model):
    if model == "original":
        from models.base import load_native
        return load_native(config, device).eval().requires_grad_(False), 0
    from algorithms import build_attention
    from models import build_model
    from models.attention import fuse_lora_projections
    net = build_model(config, device=device, algorithm_factory=build_attention).base
    net.eval().requires_grad_(False)
    return net, fuse_lora_projections(net)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("export", "original"), default="export")
    parser.add_argument("--repo", type=Path, help="Export directory containing config.yaml and model.safetensors")
    parser.add_argument("--weights-root", type=Path, default=Path.home() / "huggingface_ckpts")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/pid-{model}-8k-two-prompts")
    parser.add_argument("--gpus", default="0,1", help="Two physical GPU IDs, one per prompt")
    parser.add_argument("--seed", type=int, default=1045)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--job", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.model == "export" and args.repo is None:
        parser.error("--repo is required for --model export")
    if args.repo is not None:
        args.repo = args.repo.expanduser().resolve()
    if args.output is None:
        args.output = Path(f"./outputs/pid-{args.model}-8k-two-prompts")
    args.weights_root = args.weights_root.expanduser().resolve()
    if args.job is not None:
        generate_job(args)
        return
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2:
        raise ValueError("--gpus must contain exactly two IDs, one for each prompt")
    if gpus[0] == gpus[1]:
        print("Both jobs use the same GPU; they will run sequentially.", flush=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    job_args = [(args.repo, args.weights_root, output, i, gpus[i], args.seed + i,
                 args.steps, args.cfg, args.shift, args.model) for i in (0, 1)]
    if gpus[0] == gpus[1]:
        for values in job_args:
            run_job(*values)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run_job, *values) for values in job_args]
            for job in jobs:
                job.result()
    (output / "run.json").write_text(json.dumps({
        "model": args.model, "repo": str(args.repo) if args.repo else None,
        "prompts": list(PROMPTS), "gpus": gpus, "seed": args.seed,
        "steps": args.steps, "cfg": args.cfg, "shift": args.shift,
        "width": 8192, "height": 8192, "global_decode": True,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"completed two global 8192 images in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
