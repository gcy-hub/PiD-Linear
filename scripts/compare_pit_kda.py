"""Compare a trained PiD-KDA with its PiT-KDA conversion, keeping compression by default.

Prepared assets fix Gemma embeddings, latents and pixel-noise seeds. Each pair
runs sequentially on the same GPU, with only one model resident. torchrun can
distribute different asset pairs across GPUs. Completed cases are resumable.
Only the new KDA attention is random and untrained; this is a speed experiment.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint, student_weights
from pid._src.linear_pid.compare_inference import ASSET_KEYS, validate_asset
from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.evaluation import benchmark_step, save_png, tensor_image
from pid._src.linear_pid.pit_attention import convert_pit_attention
from pid._src.linear_pid.profiling import InferenceSectionTimer
from pid._src.linear_pid.runtime import build_net, sample


def summarize(output):
    pairs = []
    for path in sorted(output.glob("original-pit/*.json")):
        other = output / "kda-pit" / path.name
        if not other.exists():
            continue
        original, modified = json.loads(path.read_text()), json.loads(other.read_text())
        pairs.append({"case": path.stem, "original": original, "modified": modified,
                      "sampling_speedup": original["sampling_seconds"] / modified["sampling_seconds"],
                      "pit_speedup": original["profile"]["sections"]["pit"]["total_ms"]
                      / modified["profile"]["sections"]["pit"]["total_ms"]})
    atomic_json(output / "summary.json", {"pairs": pairs, "note": "New PiT KDA is randomly initialized, without fine-tuning"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset", action="append", required=True)
    parser.add_argument("--output-dir", default="./outputs/pit-kda-comparison")
    parser.add_argument("--compression", choices=("keep", "remove"), default="keep")
    parser.add_argument("--heads", type=int, default=None, help="Default: 16 with compression, 64 without")
    parser.add_argument("--conversion-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5)
    parser.add_argument("--shift", type=float, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--network-repeats", type=int, default=5)
    args = parser.parse_args()
    args.heads = args.heads if args.heads is not None else (16 if args.compression == "keep" else 64)
    dim = 1152 if args.compression == "keep" else 4096
    if min(args.steps, args.threads, args.warmup, args.network_repeats, args.heads) < 1:
        parser.error("Steps, threads, warmup, repeats and heads must be positive")
    if dim % args.heads or dim // args.heads > 256 or (dim // args.heads) % 4:
        parser.error(f"Heads must divide {dim} with head_dim divisible by 4 and <=256")
    if args.cfg < 1 or args.shift <= 0:
        parser.error("CFG must be >=1 and shift must be positive")
    if len({Path(path).stem for path in args.asset}) != len(args.asset):
        parser.error("Each asset must have a unique filename stem")
    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    checkpoint = resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    config = LinearPiDConfig.from_dict(payload["config"])
    if config.pit_kda:
        raise ValueError("Supply the pre-conversion checkpoint with original PiT")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(device)
    for asset_path in tqdm(args.asset[rank::world], desc=f"Inference pairs GPU {rank}"):
        asset_path = Path(asset_path)
        asset = torch.load(asset_path, map_location="cpu", weights_only=True)
        validate_asset(asset)
        for key in ASSET_KEYS:
            asset[key] = asset[key].to(device)
        for variant in ("original-pit", "kda-pit"):
            destination = output / variant / asset_path.stem
            settings = {"checkpoint": str(checkpoint.resolve()), "variant": variant,
                        "asset_sha256": sha256_file(asset_path), "layers": config.layers,
                        "heads": args.heads, "conversion_seed": args.conversion_seed,
                        "compression": args.compression,
                        "steps": args.steps, "cfg": args.cfg, "shift": args.shift,
                        "warmup": args.warmup, "network_repeats": args.network_repeats,
                        "threads": args.threads,
                        "gpu": properties.name, "gpu_uuid": str(properties.uuid), "precision": "bf16"}
            if destination.with_suffix(".json").exists() and destination.with_suffix(".png").exists():
                previous = json.loads(destination.with_suffix(".json").read_text())
                previous["settings"].setdefault("threads", 1)
                previous["settings"].setdefault("compression", "remove")
                if previous["settings"] != settings:
                    raise ValueError("Existing settings differ; choose a new output directory")
                continue
            torch.manual_seed(config.seed)
            net = build_net(config.layers, local_mixing=config.local_mixing)
            net.load_state_dict(student_weights(payload), strict=True)
            if variant == "kda-pit":
                torch.manual_seed(args.conversion_seed)
                convert_pit_attention(net, heads=args.heads, compressed=args.compression == "keep")
            net.activation_checkpointing = False
            net = net.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            print(f"{variant}: {asset['width']}x{asset['height']}, "
                  f"parameters={sum(p.numel() for p in net.parameters()):,}", flush=True)
            with torch.inference_mode():
                network = benchmark_step(net, asset, repeats=args.network_repeats, warmup=args.warmup)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                with InferenceSectionTimer(net) as timer:
                    start = time.perf_counter()
                    image = sample(net, asset["caption_embs"], asset["caption_mask"], asset["null_embs"],
                                   asset["null_mask"], asset["latent"], height=asset["height"],
                                   width=asset["width"], seed=asset["seed"], steps=args.steps,
                                   cfg=args.cfg, shift=args.shift)
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                if not torch.isfinite(image).all():
                    raise FloatingPointError("Nonfinite generated pixels")
                result = {"settings": settings, "caption": asset["caption"], "seed": asset["seed"],
                          "height": asset["height"], "width": asset["width"],
                          "sampling_seconds": elapsed, "network_forward": network,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                          "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                          "parameters": sum(p.numel() for p in net.parameters()), "profile": timer.summary(),
                          "pit_attention_dim": net.pixel_attn_hidden_size,
                          "pit_kda_untrained": variant == "kda-pit"}
                save_png(tensor_image(image), destination.with_suffix(".png"))
                atomic_json(destination.with_suffix(".json"), result)
                print(f"{variant} finished: sampling={elapsed:.3f}s, "
                      f"PiT={result['profile']['sections']['pit']['total_ms']/1000:.3f}s, "
                      f"peak={result['peak_allocated_gib']:.2f}GiB", flush=True)
                del image
            del net
            gc.collect()
            torch.cuda.empty_cache()
        del asset
        if world == 1:
            summarize(output)
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo")
        dist.barrier()
        if int(os.environ.get("RANK", 0)) == 0:
            summarize(output)
        dist.destroy_process_group()
    else:
        summarize(output)


if __name__ == "__main__":
    main()
