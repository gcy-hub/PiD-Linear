"""Paired PiD inference benchmarks using identical cached conditions on each GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from tqdm import tqdm

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.attention import convert_attention
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint, student_weights
from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.evaluation import asset_cases, benchmark_step, save_png, tensor_image
from pid._src.linear_pid.runtime import build_net, load_original, sample

GALLERY_ROOT = "/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/assets"
OUTPUT_ROOT = "/home/ganchangyi/code/PiD-Linear/outputs/inference_comparison"
ASSET_KEYS = ("latent", "caption_embs", "caption_mask", "null_embs", "null_mask")


def timing_summary(milliseconds):
    if not milliseconds or any(value <= 0 for value in milliseconds):
        raise ValueError("Timing measurements must be nonempty and positive")
    return {"mean_ms": statistics.mean(milliseconds), "median_ms": statistics.median(milliseconds),
            "min_ms": min(milliseconds), "max_ms": max(milliseconds),
            "std_ms": statistics.pstdev(milliseconds), "runs_ms": milliseconds}


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def choose_cases(args):
    if args.prompt:
        if args.assets:
            raise ValueError("Use --prompt for a custom input or --asset for prepared inputs")
        return [{"id": "custom", "path": str(Path(args.output_dir) / "conditions/custom.pt")}]
    if args.image or args.latent:
        raise ValueError("--image/--latent requires --prompt")
    if args.assets:
        paths = [Path(p).resolve() for p in args.assets]
        if len(set(paths)) != len(paths) or len({p.stem for p in paths}) != len(paths):
            raise ValueError("Asset paths and their filenames must be unique")
        return [{"id": p.stem, "path": str(p)} for p in paths]
    cases, _ = asset_cases(args.gallery_root, "full", include_4k=True)
    selected = [c for c in cases if c["resolution"] == args.resolution
                and (args.kind == "all" or c["kind"] == args.kind)]
    if args.case:
        selected = [c for c in selected if Path(c["file"]).stem in args.case]
        if set(args.case) != {Path(c["file"]).stem for c in selected}:
            raise ValueError("Some --case IDs do not match the chosen resolution/kind")
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No matching prepared inputs")
    return [{"id": Path(c["file"]).stem, "path": str(Path(args.gallery_root) / c["file"])}
            for c in selected]


def validate_asset(asset):
    missing = set((*ASSET_KEYS, "height", "width", "caption", "seed")) - asset.keys()
    if missing:
        raise ValueError(f"Input asset is missing {sorted(missing)}")
    h, w = asset["height"], asset["width"]
    if min(h, w) <= 0 or h % 32 or w % 32:
        raise ValueError("Output height and width must be positive multiples of 32")
    if tuple(asset["latent"].shape) != (1, 16, h // 32, w // 32):
        raise ValueError("Expected batch-one FLUX latent with shape [1,16,height/32,width/32]")
    for prefix in ["caption", "null"]:
        emb, mask = asset[prefix + "_embs"], asset[prefix + "_mask"]
        if emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[-1] != 2304 or mask.shape != emb.shape[:2]:
            raise ValueError(f"Invalid {prefix} embeddings or padding mask")


def prepare_custom(args, device):
    """Prepare once, save CPU conditions, and destroy encoders before benchmarking."""
    destination = Path(args.output_dir) / "conditions/custom.pt"
    specification = {"prompt": args.prompt, "image": file_identity(args.image) if args.image else None,
                     "latent": file_identity(args.latent) if args.latent else None,
                     "height": args.height, "width": args.width, "seed": args.seed,
                     "weights_root": str(Path(args.weights_root).resolve())}
    meta = destination.with_suffix(".json")
    if destination.exists() and meta.exists():
        if json.loads(meta.read_text())["inputs"] != specification:
            raise ValueError("Custom inputs changed; choose a new --output-dir")
        return
    start = time.perf_counter()
    if args.latent:
        latent = torch.load(args.latent, map_location="cpu", weights_only=True)
        if isinstance(latent, dict):
            latent = latent["latent"]
    elif args.image:
        import numpy as np
        from pid._src.degradation import simple_downsample_image
        from pid._src.tokenizers.flux_vae import FluxVAE

        with Image.open(args.image) as source:
            image = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"),
                                 (args.width, args.height), method=Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None]
        pixels = pixels.to(device).float().div_(127.5).sub_(1)
        vae = FluxVAE(vae_pth=str(Path(args.weights_root) / "PiD/checkpoints/ae.safetensors"),
                      dtype=torch.bfloat16, device=str(device), is_amp=False)
        with torch.no_grad():
            latent = vae.encode(simple_downsample_image(pixels, 4.0)).cpu()
        del vae, pixels
    else:
        from diffusers import ZImagePipeline

        pipeline = ZImagePipeline.from_pretrained(Path(args.weights_root) / "Z-Image-Turbo",
                                                  torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        with torch.no_grad():
            latent = pipeline(args.prompt, height=args.height // 4, width=args.width // 4,
                              num_inference_steps=9, guidance_scale=0.0, output_type="latent",
                              generator=torch.Generator(device=device).manual_seed(args.seed)).images
        if latent.ndim == 5:
            latent = latent.squeeze(2)
        latent = latent.cpu()
        del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    from pid._src.linear_pid.text_encoder import GemmaTextEncoder, NEGATIVE_PROMPT

    encoder = GemmaTextEncoder(args.weights_root, device)
    with torch.no_grad():
        embs, mask = encoder.encode_text([args.prompt])
        null_embs, null_mask = encoder.encode_text([NEGATIVE_PROMPT])
    asset = dict(latent=latent, caption_embs=embs.cpu(), caption_mask=mask.cpu(),
                 null_embs=null_embs.cpu(), null_mask=null_mask.cpu(), caption=args.prompt,
                 height=args.height, width=args.width, seed=args.seed)
    validate_asset(asset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(asset, temp)
    temp.replace(destination)
    atomic_json(meta, {"inputs": specification, "preparation_seconds": time.perf_counter() - start})
    del encoder, embs, mask, null_embs, null_mask, asset, latent
    gc.collect()
    torch.cuda.empty_cache()


def load_asset(case):
    asset = torch.load(case["path"], map_location="cpu", weights_only=True)
    validate_asset(asset)
    return asset


def assign_cases(cases, rank, world, gpu_ids, mapping=None):
    """Keep completed pairs on their original physical GPU when resuming."""
    if mapping is None:
        return cases[rank::world]
    if not isinstance(mapping, dict):
        raise ValueError("Case GPU map must be a JSON object")
    active = set(gpu_ids[:world])
    if any(case["id"] not in mapping or str(mapping[case["id"]]) not in active for case in cases):
        raise ValueError("Every selected case must map to an active GPU")
    return [case for case in cases if str(mapping[case["id"]]) == gpu_ids[rank]]


def iter_assets(cases, workers):
    if not workers:
        for case in cases:
            yield case, load_asset(case)
        return
    # Bound prefetched conditions to workers+1 cases.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending, remaining = [], iter(cases)
        for _ in range(workers + 1):
            case = next(remaining, None)
            if case is not None:
                pending.append((case, pool.submit(load_asset, case)))
        while pending:
            case, future = pending.pop(0)
            yield case, future.result()
            following = next(remaining, None)
            if following is not None:
                pending.append((following, pool.submit(load_asset, following)))


def completed_result(path, settings):
    path = Path(path)
    if not path.exists():
        return False
    previous = json.loads(path.read_text())
    if previous["settings"] != settings:
        raise ValueError(f"Benchmark settings/hardware changed: {path}; use a new --output-dir")
    if not path.with_suffix(".png").exists():
        return False
    return True


def measure(net, asset, args, device):
    network = benchmark_step(net, asset, repeats=args.network_repeats, warmup=args.network_warmup)
    kwargs = dict(height=asset["height"], width=asset["width"], seed=asset["seed"],
                  steps=args.steps, cfg=args.cfg, shift=args.shift)
    inputs = (net, asset["caption_embs"], asset["caption_mask"],
              asset["null_embs"], asset["null_mask"], asset["latent"])
    for _ in range(args.sample_warmup):
        image = sample(*inputs, **kwargs)
        del image
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    profile = None
    if getattr(args, "profile_sections", False):
        from pid._src.linear_pid.profiling import InferenceSectionTimer
        profile = InferenceSectionTimer(net)
    with profile if profile is not None else nullcontext():
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            image = sample(*inputs, **kwargs)
            torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000)
            preview = tensor_image(image)  # CPU conversion is outside the measured interval.
            del image
    result = {"network_forward": network, "pid_sampling": timing_summary(elapsed),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    if profile is not None:
        result["section_profile"] = profile.summary()
    return result, preview


def write_summary(output, cases, baseline):
    comparisons = []
    for case in cases:
        records = {name: json.loads((output / name / (case["id"] + ".json")).read_text())
                   for name in [baseline, "trained"]}
        before, after = records[baseline], records["trained"]
        if before["settings"]["hardware"] != after["settings"]["hardware"]:
            raise ValueError("A paired comparison must use the same GPU")
        original_ms = before["pid_sampling"]["median_ms"]
        trained_ms = after["pid_sampling"]["median_ms"]
        comparisons.append(dict(case=case["id"], prompt=after["caption"],
                                height=after["height"], width=after["width"], seed=after["seed"],
                                baseline=baseline, baseline_sampling_seconds=original_ms / 1000,
                                trained_sampling_seconds=trained_ms / 1000,
                                sampling_speedup=original_ms / trained_ms,
                                baseline_forward_ms=before["network_forward"]["network_median_ms"],
                                trained_forward_ms=after["network_forward"]["network_median_ms"],
                                forward_speedup=before["network_forward"]["network_median_ms"]
                                / after["network_forward"]["network_median_ms"],
                                forward_timing_scope="median conditional forward at t=500 before sampling warmup",
                                baseline_peak_gib=before["peak_allocated_gib"],
                                trained_peak_gib=after["peak_allocated_gib"]))
        if "section_profile" in before and "section_profile" in after:
            # Match the whole-forward denominator to the section measurements:
            # these include both CFG branches, all timesteps and full sampling warmup.
            a = before["section_profile"]["sections"]["network"]["mean_ms_per_forward"]
            b = after["section_profile"]["sections"]["network"]["mean_ms_per_forward"]
            comparisons[-1].update(baseline_forward_ms=a, trained_forward_ms=b, forward_speedup=a / b,
                                   forward_timing_scope="mean CUDA-event network forward during profiled sampling")
            for section in ["mmdit", "pit", "other", "mmdit_attention", "pit_attention"]:
                a = before["section_profile"]["sections"][section]["mean_ms_per_forward"]
                b = after["section_profile"]["sections"][section]["mean_ms_per_forward"]
                comparisons[-1].update({f"baseline_{section}_ms": a, f"trained_{section}_ms": b,
                                        f"{section}_speedup": a / b})
    atomic_json(output / "summary.json", {"comparisons": comparisons,
                "total_sampling_speedup": sum(r["baseline_sampling_seconds"] for r in comparisons)
                / sum(r["trained_sampling_seconds"] for r in comparisons),
                "timing_scope": "PiD sampling only; conditions, model loading, PNG I/O excluded"})
    temp = output / "summary.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    temp.replace(output / "summary.csv")
    for row in comparisons:
        print(f"{row['case']}: baseline={row['baseline_sampling_seconds']:.3f}s "
              f"trained={row['trained_sampling_seconds']:.3f}s "
              f"speedup={row['sampling_speedup']:.3f}x", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Complete checkpoint, run directory, or student export")
    parser.add_argument("--baseline", choices=["original", "untrained-kda"], default="original")
    parser.add_argument("--weights-root", default="/home/ganchangyi/huggingface_ckpts")
    parser.add_argument("--gallery-root", default=GALLERY_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_ROOT)
    parser.add_argument("--asset", dest="assets", action="append")
    parser.add_argument("--case-gpu-map", help="JSON case ID -> physical GPU ID; preserves paired hardware on resume")
    parser.add_argument("--case", action="append", help="Prepared case ID, e.g. generated_000_2048")
    parser.add_argument("--resolution", choices=["2048", "4096"], default="2048")
    parser.add_argument("--kind", choices=["generated", "real", "all"], default="generated")
    parser.add_argument("--limit", type=int, default=1, help="0 means all matching cases")
    parser.add_argument("--prompt", help="Custom prompt; without --image/--latent, create a fixed Z-Image latent")
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument("--image", help="Image condition, resized/cropped before frozen FLUX VAE encoding")
    condition.add_argument("--latent", help="FLUX-compatible [1,16,H/32,W/32] tensor or dict with latent")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42, help="Used only for custom inputs")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-warmup", type=int, default=1,
                        help="Full sampling warmups; use 0 for qualitative generation rather than warmed benchmarks")
    parser.add_argument("--network-repeats", type=int, default=10)
    parser.add_argument("--network-warmup", type=int, default=3)
    parser.add_argument("--profile-sections", action="store_true", help="CUDA-event MMDiT/PiT and attention timings during measured sampling")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2, help="Bounded CPU condition prefetch threads per GPU")
    parser.add_argument("--dryrun", action="store_true", help="Validate and print inputs without loading GPU models")
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats, args.network_repeats, args.threads, args.height, args.width) <= 0:
        parser.error("Steps, repeats, threads and image dimensions must be positive")
    if args.network_warmup < 1 or min(args.sample_warmup, args.workers, args.limit) < 0:
        parser.error("Network warmup must be at least 1; sampling warmup, workers and limit must be nonnegative")
    if args.height % 32 or args.width % 32 or args.cfg < 1 or args.shift <= 0:
        parser.error("Dimensions must divide 32, cfg >= 1 and shift > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = resolve_checkpoint(args.checkpoint)
    source_file = checkpoint / "state.pt" if checkpoint.is_dir() else checkpoint
    payload = load_checkpoint(checkpoint)
    config = LinearPiDConfig.from_dict(payload["config"])
    config.weights_root = args.weights_root
    cases = choose_cases(args)
    if args.dryrun:
        if not args.prompt:
            for case in cases:
                validate_asset(load_asset(case))
        print(json.dumps({"checkpoint": str(checkpoint), "baseline": args.baseline,
                          "layers": config.layers, "cases": cases, "steps": args.steps,
                          "cfg": args.cfg, "shift": args.shift, "repeats": args.repeats}, indent=2))
        return
    from pid._src.linear_pid.training import broadcast_object, init_distributed

    rank, world, device = init_distributed()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        # Resolve the same checkpoint even if a training retention sweep advances "auto".
        pinned = broadcast_object(file_identity(source_file) if rank == 0 else None, rank)
        if file_identity(source_file) != pinned:
            raise ValueError("Ranks resolved different checkpoints; pass an explicit complete checkpoint")
        original_hash = broadcast_object(sha256_file(config.teacher_path) if rank == 0 else None, rank)
        if payload["metadata"]["teacher_sha256"] != original_hash:
            raise ValueError("Original PiD weights differ from the student's initialization")
        if args.prompt:
            if rank == 0:
                prepare_custom(args, device)
            if world > 1:
                dist.barrier()
        properties = torch.cuda.get_device_properties(device)
        hardware = {"gpu": properties.name, "uuid": str(getattr(properties, "uuid", "unknown")),
                    "total_memory": properties.total_memory, "torch": torch.__version__,
                    "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[device.index]
                    if os.environ.get("CUDA_VISIBLE_DEVICES") else str(device.index)}
        base_settings = {"checkpoint": pinned, "original_sha256": original_hash, "layers": config.layers,
                         "local_mixing": config.local_mixing, "steps": args.steps, "cfg": args.cfg,
                         "shift": args.shift, "repeats": args.repeats, "sample_warmup": args.sample_warmup,
                         "network_repeats": args.network_repeats, "network_warmup": args.network_warmup,
                         "precision": "bf16", "batch_size": 1, "pit_chunk_size": config.pit_chunk_size or 2048,
                         "threads": args.threads,
                         "hardware": hardware, "initialization_seed": config.seed,
                         "fla_disable_tensor_cache": os.environ.get("FLA_DISABLE_TENSOR_CACHE", "0")}
        if args.profile_sections:
            base_settings["profile_sections"] = True
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        mapping = json.loads(Path(args.case_gpu_map).read_text()) if args.case_gpu_map else None
        assigned = assign_cases(cases, rank, world, gpu_ids, mapping)
        case_settings = {c["id"]: dict(base_settings, input_sha256=sha256_file(c["path"])) for c in assigned}
        for label in [args.baseline, "trained"]:
            pending = [c for c in assigned if not completed_result(output / label / (c["id"] + ".json"),
                       dict(case_settings[c["id"]], model=label))]
            if not pending:
                continue
            torch.manual_seed(config.seed)
            net = build_net()
            if label == "trained":
                convert_attention(net, config.layers, local_mixing=config.local_mixing)
                net.load_state_dict(student_weights(payload, "raw"), strict=True)
            else:
                load_original(net, config.teacher_path)
                if label == "untrained-kda":
                    convert_attention(net, config.layers, local_mixing=config.local_mixing)
            net.activation_checkpointing = False
            net.pit_chunk_size = config.pit_chunk_size or 2048
            net = net.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            with torch.inference_mode():
                for case, asset in tqdm(iter_assets(pending, args.workers), total=len(pending),
                                        desc=f"{label} GPU {rank}"):
                    for key in ASSET_KEYS:
                        asset[key] = asset[key].to(device)
                    result, image = measure(net, asset, args, device)
                    result.update(settings=dict(case_settings[case["id"]], model=label),
                                  caption=asset["caption"], height=asset["height"], width=asset["width"],
                                  seed=asset["seed"])
                    destination = output / label / (case["id"] + ".json")
                    save_png(image, destination.with_suffix(".png"))
                    atomic_json(destination, result)
                    del asset, image
            del net
            gc.collect()
            torch.cuda.empty_cache()
        if world > 1:
            dist.barrier()
        if rank == 0:
            write_summary(output, cases, args.baseline)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
