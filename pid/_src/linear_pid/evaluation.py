"""Fixed latent galleries and synchronized PiD inference measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.checkpoint import export_student, load_checkpoint, resolve_checkpoint, student_weights
from pid._src.linear_pid.data import atomic_json
from pid._src.linear_pid.runtime import build_net, evaluation_weights, load_original, sample


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


def _sample_asset(net, asset, steps, cfg, shift):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = sample(
        net,
        asset["caption_embs"],
        asset["caption_mask"],
        asset["null_embs"],
        asset["null_mask"],
        asset["latent"],
        height=asset["height"],
        width=asset["width"],
        seed=asset["seed"],
        steps=steps,
        cfg=cfg,
        shift=shift,
    )
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def benchmark_step(net, asset, repeats=10, warmup=3):
    device = asset["latent"].device
    noisy = torch.randn(
        1,
        3,
        asset["height"],
        asset["width"],
        device=device,
        generator=torch.Generator(device=device).manual_seed(asset["seed"]),
    )
    t, sigma = torch.full((1,), 500.0, device=device), torch.zeros(1, device=device)
    times = []
    # The live training model retains positional caches. Keep them ordinary
    # no-grad tensors so the next training forward can save them for backward.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for i in range(warmup + repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            net(
                noisy,
                t,
                asset["caption_embs"],
                lq_latent=asset["latent"],
                degrade_sigma=sigma,
                text_valid_mask=asset["caption_mask"],
            )
            end.record()
            end.synchronize()
            if i >= warmup:
                times.append(start.elapsed_time(end))
    return {
        "network_median_ms": float(np.median(times)),
        "network_mean_ms": float(np.mean(times)),
        "repeats": repeats,
        "warmup": warmup,
    }


def render_cases(
    net,
    teacher,
    config,
    output_dir,
    cases,
    teacher_hash,
    rank=0,
    world=1,
    tracker=None,
    step=0,
    steps=25,
    cfg=5.0,
    shift=6.0,
    benchmark=False,
    student_identity=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = next(net.parameters()).device
    measurements = []
    measured_resolutions = set()
    progress = tqdm(cases[rank::world], desc=f"Gallery rank {rank}")
    for case in progress:
        identity = f"{case['kind']}_{case['ordinal']:03d}_{case['resolution']}"
        done = output / f"{identity}.done.json"
        settings = {
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "seed": case["seed"],
            "asset_sha256": case["sha256"],
            "teacher_sha256": teacher_hash,
            "precision": "bf16",
            "gpu": torch.cuda.get_device_name(device),
        }
        reference_settings = dict(settings)
        settings["student_identity"] = student_identity
        if done.exists():
            previous = json.loads(done.read_text())
            if previous["settings"] != settings:
                raise ValueError(f"Gallery settings changed in {output}; choose a fresh output directory")
            if not (output / f"{identity}.png").exists():
                raise ValueError(f"Gallery completion marker has no image: {done}")
            measurements.append(previous)
            continue
        asset = _load_asset(config.gallery_root, case, device)
        reference_key = hashlib.sha256(json.dumps(reference_settings, sort_keys=True).encode()).hexdigest()[:24]
        reference_dir = Path(config.output_root) / "references" / teacher_hash[:16]
        reference_path = reference_dir / f"{identity}_{reference_key}.png"
        reference_meta = reference_path.with_suffix(".json")
        reference_ms = None
        if teacher is not None and not (reference_path.exists() and reference_meta.exists()):
            if case["resolution"] not in measured_resolutions:
                benchmark_step(teacher, asset, repeats=1, warmup=1)
            reference, reference_ms = _sample_asset(teacher, asset, steps, cfg, shift)
            save_views(tensor_image(reference), reference_path)
            atomic_json(reference_meta, {"settings": reference_settings, "decode_ms": reference_ms})
            del reference
        elif reference_meta.exists():
            reference_ms = json.loads(reference_meta.read_text())["decode_ms"]
        # Exclude initial kernel compilation from reported student decode time.
        if case["resolution"] not in measured_resolutions:
            benchmark_step(net, asset, repeats=1, warmup=1)
        image, decode_ms = _sample_asset(net, asset, steps, cfg, shift)
        image = tensor_image(image)
        save_views(image, output / f"{identity}.png")
        if reference_path.exists():
            with Image.open(reference_path) as reference:
                preview = Image.new("RGB", (1024, 512))
                preview.paste(ImageOps.pad(reference, (512, 512)), (0, 0))
                preview.paste(ImageOps.pad(image, (512, 512)), (512, 0))
                save_png(preview, output / f"{identity}_teacher_left_student_right.png")
        result = {
            "case": identity,
            "settings": settings,
            "caption": asset["caption"],
            "height": asset["height"],
            "width": asset["width"],
            "student_decode_ms": decode_ms,
            "teacher_decode_ms": reference_ms,
            "reference": str(reference_path) if reference_path.exists() else None,
            "lq_preview": case.get("preview"),
            "source_image": asset.get("source_image"),
        }
        if benchmark and case["resolution"] not in measured_resolutions:
            result["student_benchmark"] = benchmark_step(net, asset)
            if teacher is not None:
                result["teacher_benchmark"] = benchmark_step(teacher, asset)
        measured_resolutions.add(case["resolution"])
        atomic_json(done, result)
        measurements.append(result)
        if tracker is not None and rank == 0:
            tracker.log(
                {f"gallery/{identity}": tracker.Image(str(output / f"{identity}.png"), caption=asset["caption"])},
                step=step,
            )
        del asset
    atomic_json(output / f"measurements_rank_{rank}.json", measurements)
    return measurements


def training_gallery(net, teacher, config, step, full, rank, world, tracker, training_metadata):
    teacher_hash = training_metadata["teacher_sha256"]
    cases, metadata = asset_cases(config.gallery_root, "full" if full else "quick")
    index = json.loads((Path(config.index_root) / "index.json").read_text())
    if metadata["data_fingerprint"] != index["fingerprint"]:
        raise ValueError("Gallery belongs to a different validation split")
    with evaluation_weights(net), torch.no_grad():
        render_cases(
            net,
            teacher,
            config,
            Path(config.run_dir) / "galleries" / f"step_{step:09d}_raw",
            cases,
            teacher_hash,
            rank,
            world,
            tracker,
            step,
            student_identity={
                "run": config.run_dir,
                "step": step,
                "weights": "raw",
                "seed": config.seed,
                "layers": config.layers,
                "parent_checkpoint": training_metadata["parent_checkpoint"],
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", choices=["quick", "qualitative"], default="qualitative")
    parser.add_argument("--weights", choices=["ema", "raw"], default="raw")
    parser.add_argument("--weights-root")
    parser.add_argument("--gallery-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--export")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    if args.export:
        export_student(checkpoint, args.export, args.weights)
        print(f"Student exported: {args.export}")
        return
    from pid._src.linear_pid.training import init_distributed

    rank, world, device = init_distributed()
    payload = load_checkpoint(checkpoint)
    config = LinearPiDConfig.from_dict(payload["config"])
    if args.weights_root:
        config.weights_root = args.weights_root
    if args.gallery_root:
        config.gallery_root = args.gallery_root
    net = build_net(config.layers, config.local_mixing)
    net.load_state_dict(student_weights(payload, args.weights), strict=True)
    net = net.to(device=device, dtype=torch.bfloat16).eval()
    teacher_hash = payload["metadata"]["teacher_sha256"]
    teacher = None
    if not args.no_reference:
        from pid._src.linear_pid.data import sha256_file

        if sha256_file(config.teacher_path) != teacher_hash:
            raise ValueError("Reference teacher differs from training teacher")
        teacher = build_net()
        load_original(teacher, config.teacher_path)
        teacher = teacher.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    cases, _ = asset_cases(
        config.gallery_root, "quick" if args.suite == "quick" else "full", include_4k=args.suite == "qualitative"
    )
    output = args.output_dir or str(
        Path(config.run_dir)
        / "evaluation"
        / f"step_{payload['progress']['total_step']:09d}_{args.weights}_{args.suite}"
    )
    source_file = checkpoint / "state.pt" if checkpoint.is_dir() else checkpoint
    source_stat = source_file.stat()
    student_identity = {
        "checkpoint": str(checkpoint.resolve()),
        "bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "weights": args.weights,
    }
    del payload
    with torch.inference_mode():
        render_cases(
            net,
            teacher,
            config,
            output,
            cases,
            teacher_hash,
            rank,
            world,
            steps=args.steps,
            cfg=args.cfg,
            shift=args.shift,
            benchmark=not args.no_benchmark,
            student_identity=student_identity,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
