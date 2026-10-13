"""Pre-training speed/memory sweep and controlled scan-order diagnostics."""

import argparse
import gc
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.networks.pixeldit_official import MMDiTJointAttention

from .attention import JointVDN
from .config import VDNConfig


def summarize(values):
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def implementation_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def neighbor_retention(alpha, gh, gw):
    indices = torch.arange(gh * gw, device=alpha.device).reshape(gh, gw)
    logs = alpha.clamp_min(1e-30).log()
    prefix = torch.cat((torch.zeros_like(logs[:1]), logs.cumsum(0)))
    result = {}
    for name, source, target in [
        ("horizontal", indices[:, :-1], indices[:, 1:]),
        ("vertical", indices[:-1, :], indices[1:, :]),
    ]:
        if not source.numel():
            result[name + "_alpha_retention"] = None
            continue
        source, target = source.flatten(), target.flatten()
        forward = torch.exp(prefix[target + 1] - prefix[source + 1]).mean()
        reverse = torch.exp(prefix[target] - prefix[source]).mean()
        result[name + "_alpha_retention"] = float((forward + reverse) * 0.5)
    return result


class DirectionProbe:
    """Change region order only; same convolution/QKV. No full Jacobian allocation."""

    def __init__(self, module, trained=False):
        self.module, self.trained = module, trained

    @torch.no_grad()
    def __call__(self, x, y, grid, text_mask=None):
        module = self.module
        qkv, tqkv = module.raw_qkv(x, y)
        # Preserve the observed projections, then measure a small perturbation
        # without BF16 rounding in features or the output finite difference.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._probe(
                x.float(), y.float(), tuple(t.float() for t in qkv), tuple(t.float() for t in tqkv), grid, text_mask
            )

    def _probe(self, x, y, qkv, tqkv, grid, text_mask):
        module = self.module
        linear = module.linear_attention
        args = (x, y, qkv, tqkv, grid, text_mask)
        row = linear(*args, order="row")
        column = linear(*args, order="column")
        error = (row.float() - column.float()).square().sum() / row.float().square().sum().clamp_min(1e-30)
        region = module.region_size
        h, w = grid
        gh, gw = (h + region - 1) // region, (w + region - 1) // region
        alpha = None

        # Restore row order and perturb V after spatial convolution, holding Q,K,beta,alpha fixed.
        features = linear.features(qkv, grid)
        baseline = linear(*args, order="row")
        alpha = linear.last_alpha
        epsilon = 1e-3
        value = features[2].clone()
        value.reshape(x.shape[0], h, w, *value.shape[2:])[:, : min(region, h), : min(region, w)] += epsilon
        original_features = linear.features
        linear.features = lambda raw, g: (features[0], features[1], value)
        try:
            changed = linear(*args, order="row")
        finally:
            linear.features = original_features
        response = (changed - baseline).float().reshape(x.shape[0], h, w, -1)
        horizontal = response[:, : min(region, h), region : min(2 * region, w)]
        vertical = response[:, region : min(2 * region, h), : min(region, w)]
        return {
            "trained": self.trained,
            "probe_precision": "fp32-on-observed-qkv",
            "row_column_relative_mse": float(error),
            "horizontal_response": float(horizontal.square().mean().sqrt() / epsilon) if horizontal.numel() else 0.0,
            "vertical_response": float(vertical.square().mean().sqrt() / epsilon) if vertical.numel() else 0.0,
            "alpha_mean": float(alpha.mean()),
            **neighbor_retention(alpha, gh, gw),
            "region_grid": [gh, gw],
            "note": "Untrained diagnostics are structural only; alpha retention is not full transition retention.",
        }


class SpatialCapture:
    """Probe unchanged per-layer features at early/middle/late CFG forwards.

    Registered after warmup. Probes call the VDA branch directly, so they do not
    recursively trigger joint-attention hooks or change the generated trajectory.
    """

    def __init__(self, net, layers=None, calls=(0, 1, 24, 25, 48, 49), trained=True):
        self.handles = []
        self.records = []
        selected = set(range(len(net.patch_blocks))) if layers is None else set(layers)
        for index, block in enumerate(net.patch_blocks):
            if index not in selected:
                continue
            count = [0]

            def capture(module, args, kwargs, index=index, count=count):
                call = count[0]
                count[0] += 1
                if call not in calls:
                    return
                measurement = DirectionProbe(module, trained)(
                    args[0], args[1], kwargs["grid_size"], kwargs.get("text_valid_mask")
                )
                self.records.append(
                    {
                        "layer": index,
                        "network_call": call,
                        "cfg_branch": "null" if call % 2 == 0 else "conditional",
                        **measurement,
                    }
                )

            self.handles.append(block.attn.register_forward_pre_hook(capture, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()


class Parts:
    """CUDA event timers around existing operations, removed after one profiled forward."""

    def __init__(self, module):
        self.module = module
        self.records = []
        self.restores = []
        self.stack = []

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def measured(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            parent = self.stack[-1] if self.stack else None
            start.record()
            self.stack.append(label)
            try:
                return original(*args, **kwargs)
            finally:
                self.stack.pop()
                end.record()
                self.records.append((label, parent, start, end))

        setattr(obj, name, measured)
        self.restores.append((obj, name, original))

    def __enter__(self):
        from . import attention
        from .vendor.delta_rule import VdnDelta

        self.wrap(self.module, "raw_qkv", "qkv_projection")
        self.wrap(self.module, "local", "local_softmax_and_output")
        self.wrap(self.module.linear_attention, "features", "dwconv_and_feature_transform")
        self.wrap(attention, "frame_statistics", "region_and_text_statistics")
        self.wrap(VdnDelta, "factor_apply", "matrix_solve")
        self.wrap(attention, "_run_scans_inference", "bidirectional_scan_including_solve")
        self.wrap(attention, "gather_linear_state", "boundary_gather")
        self.wrap(self.module.linear_attention.to_out_linear, "forward", "vda_output_projection")
        return self

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)

    def totals(self):
        torch.cuda.synchronize()
        totals = {}
        nested_solve = 0.0
        for label, parent, start, end in self.records:
            elapsed = start.elapsed_time(end)
            totals[label] = totals.get(label, 0.0) + elapsed
            if label == "matrix_solve" and parent == "bidirectional_scan_including_solve":
                nested_solve += elapsed
        if "bidirectional_scan_including_solve" in totals:
            totals["bidirectional_scan_without_solve"] = totals["bidirectional_scan_including_solve"] - nested_solve
        return totals


def timed(call, module, x, y, repeats, warmup, backward):
    samples = []
    first_ms = None
    module.zero_grad(set_to_none=True)
    x.grad = y.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_reserved = torch.cuda.memory_allocated()
    for i in range(warmup + repeats):
        module.zero_grad(set_to_none=True)
        x.grad = y.grad = None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.set_grad_enabled(backward), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            outputs = call()
            loss = sum(o.float().square().mean() for o in outputs) if backward else None
        if backward:
            loss.backward()
        end.record()
        end.synchronize()
        if first_ms is None:
            first_ms = start.elapsed_time(end)
        if i >= warmup:
            samples.append(start.elapsed_time(end))
        del outputs, loss
    return {
        **summarize(samples),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_increment_gib": (torch.cuda.max_memory_allocated() - start_reserved) / 2**30,
        "repeats": repeats,
        "warmup": warmup,
        "first_iteration_ms": first_ms,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-root", default=VDNConfig().weights_root)
    p.add_argument("--resolutions", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--regions", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--backward-repeats", type=int, default=5)
    p.add_argument("--output-root", default="outputs/pid-vdn/benchmark")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--components", nargs="+", default=["full", "local", "vda", "hybrid"])
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    if not 0 <= args.layer < 14:
        raise ValueError("layer index out of range")
    if not all(x in ("full", "local", "vda", "hybrid") for x in args.components):
        raise ValueError("Unknown component")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    from .config import BASE_RELATIVE

    base = Path(args.weights_root) / BASE_RELATIVE
    identity = {"base_sha256": sha256_file(base), "implementation_sha256": implementation_fingerprint()}
    state = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
    prefix = f"net.patch_blocks.{args.layer}.attn."
    original_weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    results = []
    for resolution in tqdm(args.resolutions, desc="Resolutions"):
        grid = (resolution // 16, resolution // 16)
        for region in args.regions:
            original = MMDiTJointAttention(1536, 24).to(dtype=torch.bfloat16)
            original.load_state_dict(original_weights, strict=True)
            original.requires_grad_(False)
            torch.manual_seed(42)
            module = JointVDN(original, region).cuda()
            x = torch.randn(1, grid[0] * grid[1], 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            y = torch.randn(1, 300, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            px = torch.polar(
                torch.ones(grid[0] * grid[1], 32, device="cuda"), torch.randn(grid[0] * grid[1], 32, device="cuda")
            )
            py = torch.polar(torch.ones(300, 32, device="cuda"), torch.randn(300, 32, device="cuda"))
            for component in args.components:
                for backward in (False, True):
                    if backward and not args.backward_repeats:
                        continue
                    key = f"{resolution}_r{region}_{component}_{'backward' if backward else 'inference'}"
                    done = root / (key + ".json")
                    settings = {
                        "resolution": resolution,
                        "region_size": region,
                        "component": component,
                        "backward": backward,
                        "layer": args.layer,
                        "warmup": args.warmup,
                        "repeats": args.backward_repeats if backward else args.repeats,
                        "gpu": torch.cuda.get_device_name(),
                        "precision": "bf16-autocast/fp32-vda",
                        "torch": torch.__version__,
                        **identity,
                    }
                    if done.exists():
                        old = json.loads(done.read_text())
                        if old["settings"] != settings:
                            raise ValueError("Benchmark settings changed; use a fresh directory")
                        results.append(old)
                        continue
                    module.inference_mode = not backward
                    module.component = "hybrid" if component == "full" else component
                    module.teacher_mode = component == "full"
                    call = partial(module, x, y, px, py, grid_size=grid)
                    start = time.perf_counter()
                    try:
                        result = timed(call, module, x, y, settings["repeats"], args.warmup, backward)
                        result["cold_and_total_seconds"] = time.perf_counter() - start
                        if not backward and component == "hybrid":
                            with (
                                torch.no_grad(),
                                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                                Parts(module) as parts,
                            ):
                                call()
                            result["parts_ms"] = parts.totals()
                        status = "ok"
                    except torch.cuda.OutOfMemoryError:
                        result = {"error": "CUDA OOM"}
                        status = "oom"
                        module.zero_grad(set_to_none=True)
                        x.grad = y.grad = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    entry = {"settings": settings, "status": status, **result}
                    atomic_json(done, entry)
                    results.append(entry)
                    print(json.dumps(entry), flush=True)
                    del call
            if args.diagnostics:
                module.teacher_mode = False
                module.inference_mode = True
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    probe = DirectionProbe(module)(x, y, grid)
                atomic_json(root / f"direction_{resolution}_r{region}.json", probe)
            del module, original, x, y, px, py
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(root / "summary.json", results)


if __name__ == "__main__":
    main()
