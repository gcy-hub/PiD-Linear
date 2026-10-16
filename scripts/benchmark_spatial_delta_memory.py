"""Original/Hybrid full-network and 25-step decoder benchmark on fixed assets."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.evaluation import _load_asset, asset_cases, save_views, tensor_image
from pid._src.linear_pid.runtime import build_net, load_original, sample
from pid._src.vdn_pid.attention import execution_mode
from pid._src.vdn_pid.benchmark import implementation_fingerprint
from pid._src.vdn_pid.checkpoint import load_delta, load_state, verify_base
from pid._src.vdn_pid.config import VDNConfig
from pid._src.vdn_pid.evaluation import ComponentTimers
from pid._src.vdn_pid.spatial_memory import replace_memory
from pid._src.vdn_pid.training import build_student


def select_square_cases(root, cases, resolutions):
    selected = []
    for resolution in resolutions:
        for case in cases:
            if case["resolution"] != resolution:
                continue
            asset = torch.load(Path(root) / case["file"], map_location="cpu", weights_only=True, mmap=True)
            if asset["height"] == asset["width"] == int(resolution):
                selected.append(case)
                break
        else:
            raise ValueError(f"No fixed square {resolution}x{resolution} asset; prepare gallery conditions first")
    return selected


def benchmark_identity(args):
    return {"model_seed": args.model_seed, "generation_requested": not args.skip_generation}


def main():
    defaults = VDNConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-root", default=defaults.weights_root)
    p.add_argument("--gallery-root", default=defaults.gallery_root)
    p.add_argument("--checkpoint")
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--resolutions", nargs="+", default=["2048", "4096"])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--output-root", default="outputs/pid-vdn/network-benchmark")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--models",
        nargs="+",
        default=["hybrid", "fixed4", "dense4", "original"],
        choices=["original", "hybrid", "fixed1", "fixed4", "dense4"],
    )
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    config = VDNConfig(weights_root=args.weights_root, gallery_root=args.gallery_root, region_size=args.region_size)
    payload = load_state(args.checkpoint) if args.checkpoint else None
    if payload:
        config = VDNConfig(**payload["config"])
        config.weights_root = args.weights_root
        config.gallery_root = args.gallery_root
        verify_base(config, payload)
    cases, _ = asset_cases(args.gallery_root, "full", True)
    selected = select_square_cases(args.gallery_root, cases, args.resolutions)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"base_sha256": sha256_file(config.teacher_path), "implementation_sha256": implementation_fingerprint()}
    identity.update(benchmark_identity(args))
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        identity["checkpoint_sha256"] = sha256_file(
            checkpoint_path / "state.pt" if checkpoint_path.is_dir() else checkpoint_path
        )
    measurements = []
    for label in args.models:
        torch.manual_seed(args.model_seed)
        if label == "original":
            net = build_net()
            load_original(net, config.teacher_path)
            net = net.to(device="cuda", dtype=torch.bfloat16)
        else:
            net = build_student(
                config, payload["stage"] if payload else "a2", "cuda", load_base=not payload or "model" not in payload
            )
            if payload:
                if "model" in payload:
                    net.load_state_dict(payload["model"], strict=True)
                else:
                    load_delta(net, payload["delta"])
            if label.startswith("fixed") or label.startswith("dense"):
                replace_memory(net, "dense" if label.startswith("dense") else "fixed", int(label[-1]))
        net.eval().requires_grad_(False)
        for case in selected:
            destination = root / label / case["resolution"]
            destination.mkdir(parents=True, exist_ok=True)
            settings = {
                "model": label,
                "resolution": case["resolution"],
                "height": int(case["resolution"]),
                "width": int(case["resolution"]),
                "region_size": config.region_size,
                "checkpoint": str(args.checkpoint),
                "trained": bool(payload),
                "memory_prototype_untrained": label not in ("original", "hybrid"),
                "prototype_policy": "zero initial image banks, no scan/decay, self writes included; text via unchanged local branch",
                "steps": 25,
                "cfg": 5.0,
                "shift": 6.0,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "asset_sha256": case["sha256"],
                "seed": case["seed"],
                **identity,
            }
            marker = destination / "measurement.json"
            if marker.exists():
                old = json.loads(marker.read_text())
                if old["settings"] != settings:
                    raise ValueError("Benchmark identity changed; use a fresh output root")
                measurements.append(old)
                continue
            asset = _load_asset(args.gallery_root, case, "cuda")
            x = torch.randn(
                1,
                3,
                asset["height"],
                asset["width"],
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(case["seed"]),
            )
            t = torch.ones(1, device="cuda") * 500
            sigma = torch.zeros(1, device="cuda")
            times = []
            first_ms = None
            captured = []
            layer = net.patch_blocks[8].attn

            def capture(module, inputs, kwargs):
                if not captured:
                    captured.append((inputs, kwargs))

            handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
            with (
                torch.no_grad(),
                execution_mode(net, inference=True),
                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            ):
                for i in range(args.warmup + args.repeats):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = net(
                        x,
                        t,
                        asset["caption_embs"],
                        lq_latent=asset["latent"],
                        degrade_sigma=sigma,
                        text_valid_mask=asset["caption_mask"],
                    )
                    end.record()
                    end.synchronize()
                    if first_ms is None:
                        first_ms = start.elapsed_time(end)
                    if i >= args.warmup:
                        times.append(start.elapsed_time(end))
                    del out
                handle.remove()
                layer_times = []
                for i in range(23):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = layer(*captured[0][0], **captured[0][1])
                    end.record()
                    end.synchronize()
                    if i >= 3:
                        layer_times.append(start.elapsed_time(end))
                    del output
                captured.clear()
                timers = ComponentTimers(net)
                try:
                    net(
                        x,
                        t,
                        asset["caption_embs"],
                        lq_latent=asset["latent"],
                        degrade_sigma=sigma,
                        text_valid_mask=asset["caption_mask"],
                    )
                    parts = timers.totals()
                finally:
                    timers.close()
                result = {
                    "settings": settings,
                    "network_median_ms": float(np.median(times)),
                    "network_std_ms": float(np.std(times)),
                    "first_iteration_ms": first_ms,
                    "layer8_attention_median_ms": float(np.median(layer_times)),
                    "layer8_attention_std_ms": float(np.std(layer_times)),
                    **parts,
                }
                if not args.skip_generation:
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    timers = ComponentTimers(net)
                    try:
                        image = sample(
                            net,
                            asset["caption_embs"],
                            asset["caption_mask"],
                            asset["null_embs"],
                            asset["null_mask"],
                            asset["latent"],
                            height=asset["height"],
                            width=asset["width"],
                            seed=asset["seed"],
                            steps=25,
                            cfg=5.0,
                            shift=6.0,
                        )
                        torch.cuda.synchronize()
                        result.update(
                            decode_seconds=time.perf_counter() - start,
                            decode_parts=timers.totals(),
                            peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                        )
                    finally:
                        timers.close()
                    save_views(tensor_image(image), destination / "generated.png")
                    del image
                atomic_json(marker, result)
                measurements.append(result)
                print(json.dumps(result), flush=True)
            del asset, x, t, sigma
        del net
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(root / "summary.json", measurements)


if __name__ == "__main__":
    main()
