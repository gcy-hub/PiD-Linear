"""Resumable multi-GPU fixed-gallery inference without a resident teacher."""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from pid._src.linear_pid.data import atomic_json
from pid._src.linear_pid.evaluation import _load_asset, asset_cases, save_views, tensor_image
from pid._src.linear_pid.runtime import sample
from pid._src.linear_pid.training import init_distributed

from .attention import JointVDN, execution_mode
from .checkpoint import delta_state, load_delta, load_state, verify_base
from .config import VDNConfig, local_layout, select_validation_cases
from .training import build_student


class ComponentTimers:
    def __init__(self, net):
        self.events = {"mmdit": [], "pit": []}
        self.handles = []
        for group, blocks in [("mmdit", net.patch_blocks), ("pit", net.pixel_blocks)]:
            for block in blocks:
                pending = []

                def pre(m, args, pending=pending):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    pending.append(e)

                def post(m, args, out, pending=pending, group=group):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    self.events[group].append((pending.pop(), e))

                self.handles.extend([block.register_forward_pre_hook(pre), block.register_forward_hook(post)])

    def totals(self):
        torch.cuda.synchronize()
        return {k + "_ms": sum(a.elapsed_time(b) for a, b in v) for k, v in self.events.items()}

    def close(self):
        for h in self.handles:
            h.remove()


def limit_by_resolution(cases, limit):
    if not limit:
        return cases
    counts = {}
    chosen = []
    for case in cases:
        resolution = case["resolution"]
        if counts.get(resolution, 0) >= limit:
            continue
        counts[resolution] = counts.get(resolution, 0) + 1
        chosen.append(case)
    return chosen


def render_gallery(
    net,
    config,
    output,
    rank=0,
    world=1,
    tracker=None,
    step=0,
    suite="qualitative",
    include_4k=False,
    limit=0,
    model_identity=None,
    limit_per_resolution=0,
    diagnostics=False,
    diagnostic_layers=None,
    should_stop=None,
    completion=None,
):
    include_4k = include_4k or getattr(config, "validation_resolution", 2048) == 4096
    cases, metadata = asset_cases(config.gallery_root, "quick" if suite == "quick" else "full", include_4k)
    cases = select_validation_cases(config, cases)
    source = json.loads((Path(config.index_root) / "index.json").read_text())
    if metadata["data_fingerprint"] != source.get("source_fingerprint", source["fingerprint"]):
        raise ValueError("Gallery/source index mismatch")
    if limit:
        cases = cases[:limit]
    cases = limit_by_resolution(cases, limit_per_resolution)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    measurements = []
    complete = True
    if model_identity is None:
        import hashlib

        digest = hashlib.sha256()
        for name, tensor in sorted(delta_state(net).items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        model_identity = digest.hexdigest()
    settings = {
        "model_identity": model_identity,
        "steps": 25,
        "cfg": 5.0,
        "shift": 6.0,
        "region_size": config.region_size,
        "step": step,
        "vda_rope": False,
        "softmax_rope": True,
        "scan_order": sorted({m.scan_order for m in net.modules() if isinstance(m, JointVDN)}),
        "diagnostics": diagnostics,
        "diagnostic_layers": diagnostic_layers,
    }
    if local_layout(config)["mode"]!="window":
        settings["local_attention"]=local_layout(config)
    if getattr(config, "memory_mode", "vda") != "vda":
        settings["memory_mode"] = config.memory_mode
    try:
        with execution_mode(net, inference=True), torch.no_grad():
            for case in tqdm(cases[rank::world], desc=f"VDN gallery rank {rank}"):
                # No collectives here: ranks may own different numbers of cases.
                # Finish a whole image before honoring the deadline/stop signal.
                if should_stop is not None and should_stop():
                    complete = False
                    break
                identity = f"{case['kind']}_{case['ordinal']:03d}_{case['resolution']}"
                path = output / (identity + ".png")
                done = output / (identity + ".done.json")
                current = {**settings, "asset_sha256": case["sha256"], "seed": case["seed"]}
                if done.exists():
                    previous = json.loads(done.read_text())
                    if previous["settings"] != current or not path.exists():
                        raise ValueError(f"Gallery identity changed: {done}")
                    measurements.append(previous)
                    continue
                asset = _load_asset(config.gallery_root, case, device)
                # Warm the exact network shape before measuring the 25-step sampler.
                noisy = torch.randn(1, 3, asset["height"], asset["width"], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    net(
                        noisy,
                        torch.ones(1, device=device) * 500,
                        asset["caption_embs"],
                        lq_latent=asset["latent"],
                        degrade_sigma=torch.zeros(1, device=device),
                        text_valid_mask=asset["caption_mask"],
                    )
                del noisy
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                timers = ComponentTimers(net)
                capture = None
                if diagnostics:
                    from .benchmark import SpatialCapture

                    capture = SpatialCapture(net, layers=diagnostic_layers, trained=step > 0)
                start = time.perf_counter()
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
                    elapsed = time.perf_counter() - start
                    parts = timers.totals()
                finally:
                    timers.close()
                    if capture:
                        capture.close()
                result = {
                    "settings": current,
                    "includes_diagnostic_overhead": diagnostics,
                    "caption": asset["caption"],
                    "height": asset["height"],
                    "width": asset["width"],
                    "decode_seconds": elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    **parts,
                }
                save_views(tensor_image(image), path)
                if capture:
                    atomic_json(
                        output / (identity + ".spatial.json"), {"settings": current, "records": capture.records}
                    )
                atomic_json(done, result)
                measurements.append(result)
                if tracker is not None and rank == 0:
                    tracker.log({f"gallery/{identity}": tracker.Image(str(path), caption=asset["caption"])}, step=step)
                del asset, image
    finally:
        net.train(was_training)
    atomic_json(output / f"measurements_rank_{rank}.json", measurements)
    if completion is not None:
        completion["complete"] = complete
    return measurements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights-root", default=VDNConfig().weights_root)
    p.add_argument("--index-root", default=VDNConfig().index_root)
    p.add_argument("--gallery-root", default=VDNConfig().gallery_root)
    p.add_argument("--output-dir", default="outputs/pid-vdn/inference")
    p.add_argument("--suite", choices=["quick", "qualitative"], default="qualitative")
    p.add_argument("--include-4k", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--limit-per-resolution", type=int, default=0)
    p.add_argument("--export")
    p.add_argument("--export-only", action="store_true")
    p.add_argument(
        "--scan-order",
        choices=["row", "column"],
        default="row",
        help="Column order is an inference-only spatial diagnostic.",
    )
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--diagnostic-layers", nargs="+", type=int)
    args = p.parse_args()
    if args.export_only and not args.export:
        p.error("--export-only requires --export")
    payload = load_state(args.checkpoint)
    config = VDNConfig(**payload["config"])
    config.weights_root = args.weights_root
    config.index_root = args.index_root
    config.gallery_root = args.gallery_root
    if "model" not in payload:
        verify_base(config, payload)
    rank, world, device = init_distributed()
    try:
        net = build_student(config, payload["stage"], device, load_base="model" not in payload)
        if "model" in payload:
            net.load_state_dict(payload["model"], strict=True)
        else:
            load_delta(net, payload["delta"])
        net.requires_grad_(False)
        for module in net.modules():
            if isinstance(module, JointVDN):
                module.scan_order = args.scan_order
        if args.export and rank == 0:
            # Fully self-contained student network weights; no teacher model needed at inference.
            export = Path(args.export)
            export.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "pid_vdn",
                    "schema": 2,
                    "config": config.to_dict(),
                    "stage": payload["stage"],
                    "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "metadata": payload["metadata"],
                    "progress": payload["progress"],
                },
                export,
            )
        if args.export_only:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return
        render_gallery(
            net,
            config,
            args.output_dir,
            rank,
            world,
            step=payload["progress"]["total_step"],
            suite=args.suite,
            include_4k=args.include_4k,
            limit=args.limit,
            limit_per_resolution=args.limit_per_resolution,
            diagnostics=args.diagnostics,
            diagnostic_layers=args.diagnostic_layers,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
