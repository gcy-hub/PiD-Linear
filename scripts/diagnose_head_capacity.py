"""Frozen-layer projected-QK capacity diagnostic; this does not alter training architecture."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from pid._src.linear_pid.attention import reference_kda


class HeadProjection(nn.Module):
    """Per-head maps after original 64-D RoPE, before L2 normalization.

    Positive row-normalized squared K weights mix log-decay consistently.
    This diagonal-gate approximation is a confound, not a change of basis proof.
    """
    def __init__(self, heads, key_dim, seed=42, frozen=False):
        super().__init__()
        if key_dim not in (32, 48, 64, 80, 96):
            raise ValueError("Expected key dimension 32/48/64/80/96")
        generator = torch.Generator().manual_seed(seed)
        if key_dim < 64:
            # Random orthogonal projection, trainable rather than truncation-only.
            basis = torch.linalg.qr(torch.randn(64, 64, generator=generator)).Q[:, :key_dim].T
        else:
            basis = torch.eye(64)
            if key_dim > 64:
                extra = F.normalize(torch.randn(key_dim - 64, 64, generator=generator), dim=-1) * .05
                basis = torch.cat([basis, extra])
        self.q_map = nn.Parameter(basis.repeat(heads, 1, 1), requires_grad=not frozen)
        self.k_map = nn.Parameter(basis.repeat(heads, 1, 1), requires_grad=not frozen)
        self.frozen = frozen

    def forward(self, q, k, log_decay):
        if self.frozen:
            return q, k, log_decay
        qp = torch.einsum("bthd,hkd->bthk", q, self.q_map.to(q.dtype))
        kp = torch.einsum("bthd,hkd->bthk", k, self.k_map.to(k.dtype))
        positive = self.k_map.float().square()
        positive = positive / positive.sum(-1, keepdim=True).clamp_min(1e-12)
        gp = torch.einsum("bthd,hkd->bthk", log_decay.float(), positive)
        return qp.contiguous(), kp.contiguous(), gp.contiguous()


def grouped_kda(q, k, v, g, beta, dims, adapters, backend="fla"):
    """Stateless grouped calls: each group has K dimension, V remains 64."""
    outputs = []
    indices = []
    for dim in sorted(set(dims)):
        heads = [h for h, d in enumerate(dims) if d == dim]
        qq, kk, gg = adapters[dim](q[:, :, heads], k[:, :, heads], g[:, :, heads])
        vv, bb = v[:, :, heads].contiguous(), beta[:, :, heads].contiguous()
        if backend == "reference":
            value = reference_kda(F.normalize(qq.float(), dim=-1), F.normalize(kk.float(), dim=-1), vv.float(), gg, bb.float())
        else:
            from fla.ops.kda import chunk_kda
            value, _ = chunk_kda(qq, kk, vv, gg, bb,
                                 scale=dim ** -.5, use_qk_l2norm_in_kernel=True,
                                 initial_state=None, output_final_state=False)
        outputs.append(value)
        indices.extend(heads)
    inverse = [indices.index(h) for h in range(len(dims))]
    return torch.cat(outputs, dim=2)[:, :, inverse]


def allocate_budget(scores, key_dims, budget):
    """Exact multiple-choice knapsack using training-only per-head errors."""
    best = {0: (0., [])}
    for row in scores:
        updated = {}
        for used, (error, allocation) in best.items():
            for dim, value in zip(key_dims, row):
                total = used + dim
                if total <= budget:
                    candidate = (error + float(value), allocation + [dim])
                    if total not in updated or candidate[0] < updated[total][0]:
                        updated[total] = candidate
        best = updated
    if budget not in best:
        raise ValueError(f"No allocation achieves exact budget {budget}")
    return best[budget][1]


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(temporary, path)


def save_case(path, payload, fingerprint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".complete.json"), {"fingerprint": fingerprint, "bytes": path.stat().st_size})


def valid_case(path, fingerprint):
    path = Path(path)
    try:
        marker = json.loads(path.with_suffix(".complete.json").read_text())
        return marker["fingerprint"] == fingerprint and marker["bytes"] == path.stat().st_size
    except (OSError, KeyError, ValueError):
        return False


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalized(value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-8)


def errors(actual, target, image_start):
    actual, target = actual[:, image_start:].float(), target[:, image_start:].float()
    axes = (0, 1, 3)
    return {
        "normalized_mse": (normalized(actual) - normalized(target)).square().mean(axes).tolist(),
        "relative_mse": ((actual - target).square().mean(axes) / target.square().mean(axes).clamp_min(1e-12)).tolist(),
        "cosine": F.cosine_similarity(actual, target, dim=-1).mean((0, 1)).tolist(),
    }


@torch.no_grad()
def prepare_record(student, teacher, hidden, backend="fla"):
    """Pack exact student inputs; teacher sees the same normalized x/y hidden."""
    from pid._src.networks.pixeldit_official import apply_rotary_emb
    x, y = hidden["x"], hidden["y"]
    b, nx, _ = x.shape
    if b != 1:
        raise ValueError("Probe currently records one independent sequence per case")
    ny, heads, dim = y.shape[1], student.num_heads, student.head_dim
    valid_y = hidden["text_valid_mask"][:, :ny].bool()
    valid = torch.cat([valid_y, torch.ones(b, nx, dtype=torch.bool, device=x.device)], 1)
    qx, kx, vx = student._project(x, student.qkv_x, student.image_conv if student.local_mixing else None, hidden["grid_size"])
    qy, ky, vy = student._project(y, student.qkv_y, student.text_conv if student.local_mixing else None, valid=valid_y)
    qx, kx = apply_rotary_emb(qx, kx, hidden["pos_img"])
    if hidden["pos_txt"] is not None:
        qy, ky = apply_rotary_emb(qy, ky, hidden["pos_txt"])
    q, k, v = [torch.cat(parts, 1)[valid].unsqueeze(0).contiguous() for parts in [(qy, qx), (ky, kx), (vy, vx)]]
    decays, betas = [], []
    for value, gates in [(y, student.gates_y), (x, student.gates_x)]:
        raw = gates.forget(value).view(b, value.shape[1], heads, dim)
        decays.append(-gates.A_log.float().exp()[None, None, :, None] *
                      F.softplus(raw.float() + gates.dt_bias.float().view(heads, dim)))
        betas.append(gates.beta(value).sigmoid())
    g = torch.cat(decays, 1)[valid].unsqueeze(0).contiguous()
    beta = torch.cat(betas, 1)[valid].unsqueeze(0).contiguous()
    if backend == "reference":
        baseline = reference_kda(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v.float(), g, beta.float())
    else:
        from fla.ops.kda import chunk_kda
        baseline, _ = chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True,
                                initial_state=None, output_final_state=False)
    teacher_qkv = []
    for value, stream, pos in [(y, "y", hidden["pos_txt"]), (x, "x", hidden["pos_img"])]:
        tq, tk, tv = getattr(teacher, "qkv_" + stream)(value).view(b, value.shape[1], 3, heads, dim).unbind(2)
        tq, tk = getattr(teacher, "q_norm_" + stream)(tq), getattr(teacher, "k_norm_" + stream)(tk)
        if pos is not None:
            tq, tk = apply_rotary_emb(tq, tk, pos)
        teacher_qkv.append((tq, tk, tv))
    tq, tk, tv = [torch.cat([teacher_qkv[0][i], teacher_qkv[1][i]], 1)[valid].unsqueeze(0).transpose(1, 2).contiguous() for i in range(3)]
    # Full, bidirectional attention; drop padded text exactly as a boolean key mask does.
    softmax = F.scaled_dot_product_attention(tq, tk, tv, dropout_p=0.).transpose(1, 2).contiguous()
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta,
            "softmax": softmax, "kda": baseline, "image_start": int(valid_y.sum())}


class CapturedLayer(Exception):
    pass


@torch.no_grad()
def capture_cases(args, cases, fingerprint, rank, world):
    from pid._src.configs.linear_pid.config import LinearPiDConfig
    from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint, student_weights
    from pid._src.linear_pid.runtime import build_net
    from pid._src.networks.pixeldit_official import MMDiTJointAttention
    pending = [(case, path) for i, (case, path) in enumerate(cases)
               if i % world == rank and not valid_case(path, fingerprint)]
    if not pending:
        return
    payload = load_checkpoint(resolve_checkpoint(args.checkpoint))
    config = LinearPiDConfig.from_dict(payload["config"])
    net = build_net(config.layers, config.local_mixing)
    net.load_state_dict(student_weights(payload, args.weights), strict=True)
    del payload
    net = net.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    net.pit_chunk_size = args.pit_chunk_size
    student = net.patch_blocks[args.layer].attn
    if not getattr(student, "is_linear_attention", False) or student.head_dim != 64:
        raise ValueError("Representative layer must be trained KDA with 64-D values")
    teacher = MMDiTJointAttention(student.dim, student.num_heads)
    weights = torch.load(args.teacher, map_location="cpu", weights_only=True, mmap=True)
    prefix = f"net.patch_blocks.{args.layer}.attn."
    teacher.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    del weights
    teacher = teacher.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    captured = {}
    def hook(module, positional, keywords):
        captured.update({"x": positional[0], "y": positional[1], "pos_img": positional[2],
                         "pos_txt": positional[3] if len(positional) > 3 else None,
                         "grid_size": keywords["grid_size"], "text_valid_mask": keywords["text_valid_mask"]})
        raise CapturedLayer()
    handle = student.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for case, destination in tqdm(pending, desc=f"capture rank {rank}"):
            asset_path = Path(args.assets) / f"prompt_{case['prompt']:03d}_{case['resolution']}.pt"
            asset = torch.load(asset_path, weights_only=True, map_location=args.device)
            seed = int(asset["seed"])
            height, width = int(asset["height"]), int(asset["width"])
            noise = torch.randn(1, 3, height, width, device=args.device,
                                generator=torch.Generator(device=args.device).manual_seed(seed))
            protocol = "seeded_gaussian_off_trajectory"
            reference_sha = None
            if args.reference_dir:
                import numpy as np
                from PIL import Image
                reference = Path(args.reference_dir) / asset_path.with_suffix(".png").name
                metadata = json.loads(reference.with_suffix(".json").read_text())
                if metadata["seed"] != seed or metadata["settings"]["input_sha256"] != sha256(asset_path):
                    raise ValueError(f"Pseudo-reference does not match fixed asset: {reference}")
                with Image.open(reference) as image:
                    if image.size != (width, height):
                        raise ValueError("Pseudo-reference size mismatch")
                    pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).unsqueeze(0).to(args.device).float() / 127.5 - 1
                fraction = case["timestep"] / 1000
                noise = (1 - fraction) * pixels + fraction * noise
                del pixels
                protocol = "teacher_generated_pseudo_reference_interpolation_not_GT_or_sampler_trajectory"
                reference_sha = sha256(reference)
            captured.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                try:
                    net(noise, torch.tensor([case["timestep"]], device=args.device, dtype=torch.float32),
                        asset["caption_embs"], lq_latent=asset["latent"],
                        degrade_sigma=torch.zeros(1, device=args.device), text_valid_mask=asset["caption_mask"])
                except CapturedLayer:
                    pass
                if not captured:
                    raise RuntimeError("Layer hook did not capture normalized student hidden")
                record = prepare_record(student, teacher, captured)
            record.update({"case": case, "seed": seed, "protocol": protocol,
                           "reference_sha256": reference_sha, "asset_sha256": sha256(asset_path)})
            save_case(destination, {k: v.cpu() if torch.is_tensor(v) else v for k, v in record.items()}, fingerprint)
            captured.clear()
            del record, asset, noise
    finally:
        handle.remove()
        del net, teacher
        torch.cuda.empty_cache()


def load_record(path, device):
    record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in record.items()}


def evaluate_record(record, dims, adapters):
    return grouped_kda(*(record[k] for k in ("q", "k", "v", "g", "beta")), dims, adapters)


def fit_configs(args, cases, fingerprint, rank, world, tracker):
    training = [(case, path) for case, path in cases if case["split"] == "train"]
    heads = args.heads
    for i, dim in enumerate(args.key_dims):
        if i % world != rank:
            continue
        path = Path(args.output) / "adapters" / f"uniform_{dim}.pt"
        if valid_case(path, fingerprint):
            saved = torch.load(path, weights_only=True, map_location="cpu")
            if saved["step"] == args.fit_steps:
                continue
        adapter = HeadProjection(heads, dim, seed=args.seed).to(args.device)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
        step, history = 0, []
        if valid_case(path, fingerprint):
            saved = torch.load(path, weights_only=True, map_location="cpu")
            adapter.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            step, history = saved["step"], saved["history"]
        for iteration in tqdm(range(step, args.fit_steps), desc=f"fit K={dim} rank {rank}"):
            _, case_path = training[iteration % len(training)]
            record = load_record(case_path, args.device)
            optimizer.zero_grad(set_to_none=True)
            out = evaluate_record(record, [dim] * heads, {dim: adapter})
            start = record["image_start"]
            target = record[args.target]
            loss = (normalized(out[:, start:]) - normalized(target[:, start:])).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite calibration loss K={dim} step={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            value = float(loss.detach())
            history.append(value)
            tracker.log({f"fit/K{dim}/normalized_mse": value, f"fit/K{dim}/step": iteration + 1})
            # Save every optimizer step: normal restarts do not repeat completed work.
            save_case(path, {"model": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                             "optimizer": optimizer.state_dict(), "step": iteration + 1,
                             "history": history, "dim": dim, "target": args.target}, fingerprint)
            del record, out, loss, target
        del adapter, optimizer
        torch.cuda.empty_cache()


def load_uniform(args, dim, fingerprint):
    path = Path(args.output) / "adapters" / f"uniform_{dim}.pt"
    if not valid_case(path, fingerprint):
        raise RuntimeError(f"Missing/mismatched adapter: {path}")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["step"] != args.fit_steps:
        raise RuntimeError(f"Incomplete calibration K={dim}: {saved['step']}/{args.fit_steps}")
    return saved


def adapters_for_allocation(args, allocation, fingerprint, frozen=False):
    if frozen:
        return {64: HeadProjection(args.heads, 64, seed=args.seed, frozen=True).to(args.device)}
    result = {}
    for dim in sorted(set(allocation)):
        indices = [h for h, d in enumerate(allocation) if d == dim]
        saved = load_uniform(args, dim, fingerprint)
        module = HeadProjection(len(indices), dim, seed=args.seed).to(args.device)
        module.load_state_dict({k: v[indices] for k, v in saved["model"].items()})
        module.requires_grad_(False)
        result[dim] = module
    return result


@torch.no_grad()
def training_scores(args, cases, fingerprint):
    scores_path = Path(args.output) / "training_scores.json"
    if scores_path.exists():
        saved = json.loads(scores_path.read_text())
        if saved.get("fingerprint") == fingerprint:
            return torch.tensor(saved["scores"])
    rows = []
    for dim in tqdm(args.key_dims, desc="allocation selection on training prompts"):
        adapter = adapters_for_allocation(args, [dim] * args.heads, fingerprint)
        values = []
        for case, path in cases:
            if case["split"] != "train":
                continue
            record = load_record(path, args.device)
            out = evaluate_record(record, [dim] * args.heads, adapter)
            values.append(torch.tensor(errors(out, record[args.target], record["image_start"])["normalized_mse"]))
            del record, out
        rows.append(torch.stack(values).mean(0))
        del adapter
    scores = torch.stack(rows, dim=1)
    atomic_json(scores_path, {"fingerprint": fingerprint, "scores": scores.tolist(),
                             "head_ids": list(range(args.heads)), "key_dims": args.key_dims,
                             "selection_split": "train"})
    return scores


def configuration_counts(args, allocation, frozen=False):
    total = sum(allocation)
    return {"key_dimensions": allocation, "key_capacity": total, "value_dim": 64,
            "state_elements": total * 64, "state_bytes_fp32": total * 64 * 4,
            "adapter_parameters": 0 if frozen else total * 64 * 2,
            "trainable_adapter_parameters": 0 if frozen else total * 64 * 2,
            "estimated_native_qk_projection_parameters": 2 * args.hidden_size * total,
            "estimated_native_qkv_projection_parameters": args.hidden_size * (2 * total + args.heads * 64),
            "count_scope": "adapter actual; native counts hypothetical single-stream projections, exclude biases, RoPE/gates/local mixing/output"}


@torch.no_grad()
def latency(record, allocation, adapters, repeats, warmup):
    for _ in range(warmup):
        evaluate_record(record, allocation, adapters)
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        evaluate_record(record, allocation, adapters)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "min_ms": min(samples), "runs_ms": samples, "warmup": warmup,
            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - resident_bytes,
            "scope": "resident-input projected Q/K + positive decay mapping + stateless grouped KDA; excludes feature capture, disk IO, output gates/projection and PiT"}


@torch.no_grad()
def evaluate_configs(args, cases, fingerprint, rank, world, configurations, tracker):
    for index, (name, allocation) in enumerate(configurations.items()):
        if index % world != rank:
            continue
        frozen = name == "identity_64"
        adapters = adapters_for_allocation(args, allocation, fingerprint, frozen=frozen)
        for case, case_path in tqdm(cases, desc=f"evaluate {name} rank {rank}"):
            output = Path(args.output) / "evaluation" / name / (case_path.stem + ".json")
            if output.exists():
                saved = json.loads(output.read_text())
                if saved.get("fingerprint") == fingerprint:
                    continue
            record = load_record(case_path, args.device)
            value = evaluate_record(record, allocation, adapters)
            metric = {target: errors(value, record[target], record["image_start"]) for target in ("softmax", "kda")}
            timing = latency(record, allocation, adapters, args.latency_repeats, args.latency_warmup)
            identity_difference = float((value.float() - record["kda"].float()).abs().max()) if frozen else None
            atomic_json(output, {"fingerprint": fingerprint, "configuration": name, "case": case,
                                 "metrics": metric, "latency": timing,
                                 "identity_max_abs_error": identity_difference,
                                 "counts": configuration_counts(args, allocation, frozen),
                                 "protocol": record["protocol"], "seed": record["seed"]})
            tracker.log({f"eval/{name}/{case['split']}/softmax_nmse": statistics.mean(metric["softmax"]["normalized_mse"]),
                         f"latency/{name}/ms": timing["median_ms"]})
            del record, value
        del adapters
        torch.cuda.empty_cache()


def curves_png(output, summaries, dims, heads, target, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 6, figsize=(18, 11), sharex=True)
    for head, ax in enumerate(axes.flat):
        if head >= heads:
            ax.set_visible(False)
            continue
        values = [summaries[f"uniform_{d}"]["heldout"][target]["normalized_mse"][head] for d in dims]
        ax.plot(dims, values, "o-", color="#166b77", linewidth=1.5, markersize=4, label="calibrated Q/K")
        frozen = summaries["identity_64"]["heldout"][target]["normalized_mse"][head]
        ax.scatter([64], [frozen], color="black", marker="x", s=30, label="frozen identity64", zorder=5)
        ax.set_title(f"Actual head {head:02d}", fontsize=10)
        ax.set_xticks(dims)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25)
        if head % 6 == 0:
            ax.set_ylabel("Held-out normalized MSE", fontsize=9)
        if head >= 18:
            ax.set_xlabel("Key dimension (V=64)", fontsize=9)
    label = "Trained-KDA retention" if target == "kda" else "Original-softmax reference; head/value coordinates unverified"
    figure.suptitle(f"Layer {layer} — {label}\nPost-RoPE learned Q/K, positive decay approximation; artificial pseudo-reference inputs", fontsize=13)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, .035, 1, .94))
    figure.savefig(output, dpi=180)
    figure.savefig(Path(output).with_suffix(".pdf"))
    plt.close(figure)


def aggregate(args, cases, fingerprint, configurations):
    rows, summary, missing = [], {}, []
    for name in configurations:
        results = []
        for case, case_path in cases:
            path = Path(args.output) / "evaluation" / name / (case_path.stem + ".json")
            try:
                value = json.loads(path.read_text())
                if value["fingerprint"] != fingerprint:
                    raise ValueError("mismatched fingerprint")
                results.append(value)
                for head in range(args.heads):
                    row = {"configuration": name, **case, "head": head,
                           "key_dim": value["counts"]["key_dimensions"][head],
                           "latency_ms": value["latency"]["median_ms"]}
                    row.update({f"{target}_{metric}": scores[head]
                                for target, metrics in value["metrics"].items()
                                for metric, scores in metrics.items()})
                    rows.append(row)
            except (OSError, ValueError, KeyError):
                missing.append(str(path))
        if not results:
            continue
        item = {"counts": results[0]["counts"]}
        for split in ("train", "heldout"):
            chosen = [v for v in results if v["case"]["split"] == split]
            if chosen:
                item[split] = {target: {metric: torch.tensor([v["metrics"][target][metric] for v in chosen]).mean(0).tolist()
                                       for metric in ("normalized_mse", "relative_mse", "cosine")}
                               for target in ("softmax", "kda")}
                item[split]["cases"] = len(chosen)
                item[split]["softmax_mean_normalized_mse"] = statistics.mean(item[split]["softmax"]["normalized_mse"])
        item["latency_by_resolution"] = {str(resolution): statistics.median([v["latency"]["median_ms"] for v in results if v["case"]["resolution"] == resolution])
                                         for resolution in args.resolutions if any(v["case"]["resolution"] == resolution for v in results)}
        if name == "identity_64":
            item["identity_max_abs_error"] = max(v["identity_max_abs_error"] for v in results)
        summary[name] = item
    payload = {"fingerprint": fingerprint, "status": "complete" if not missing else "partial",
               "expected_case_configuration_pairs": len(cases) * len(configurations),
               "completed_case_configuration_pairs": len(rows) // args.heads,
               "missing": missing, "configurations": summary,
               "interpretation": "Projected-QK feature calibration only; teacher-generated pseudo-reference is not GT or a sampler trajectory; no end-to-end quality/architecture claim."}
    atomic_json(Path(args.output) / "results.json", payload)
    if rows:
        with (Path(args.output) / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not missing:
        for target in ("softmax", "kda"):
            curves_png(Path(args.output) / f"head_curves_{target}.png", summary, args.key_dims, args.heads, target, args.layer)
        atomic_json(Path(args.output) / "complete.json", {"fingerprint": fingerprint, "case_configuration_pairs": len(cases) * len(configurations)})
    else:
        (Path(args.output) / "complete.json").unlink(missing_ok=True)
    print(json.dumps({"status": payload["status"], "pairs": payload["completed_case_configuration_pairs"],
                      "expected": payload["expected_case_configuration_pairs"], "results": str(Path(args.output) / "results.json")}), flush=True)


def metadata(args, cases):
    def file_info(path):
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "state.pt"
        stat = path.stat()
        return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    sources = {}
    for case, _ in cases:
        stem = f"prompt_{case['prompt']:03d}_{case['resolution']}"
        if stem in sources:
            continue
        path = Path(args.assets) / (stem + ".pt")
        source = {"asset": file_info(path), "asset_sha256": sha256(path)}
        if args.reference_dir:
            image = Path(args.reference_dir) / (stem + ".png")
            source.update({"reference_sha256": sha256(image), "reference_metadata": json.loads(image.with_suffix(".json").read_text())})
        sources[stem] = source
    return {"schema": 1, "code_sha256": sha256(__file__), "checkpoint": file_info(args.checkpoint),
            "teacher": file_info(args.teacher), "sources": sources,
            "layer": args.layer, "heads": args.heads, "key_dims": args.key_dims,
            "resolutions": args.resolutions, "timesteps": args.timesteps,
            "train_prompts": args.train_prompts, "heldout_prompts": args.heldout_prompts,
            "fit_steps": args.fit_steps, "lr": args.lr, "grad_clip": args.grad_clip,
            "seed": args.seed, "weights": args.weights, "target": args.target,
            "latency_repeats": args.latency_repeats, "latency_warmup": args.latency_warmup,
            "pit_chunk_size": args.pit_chunk_size,
            "protocol": "teacher_generated_pseudo_reference_interpolation" if args.reference_dir else "seeded_gaussian_off_trajectory",
            "features": "packed valid-text then image, original 64-D RoPE; Q/K projections; row-normalized squared K maps mix nonpositive decay; fixed V,beta; initial_state=None",
            "loss": f"image-token per-head RMS-normalized MSE against {args.target} on same student hidden; frozen values; no output/value adaptation",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "fla_disable_tensor_cache": os.environ.get("FLA_DISABLE_TENSOR_CACHE")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--output", default="outputs/research/head-capacity-sensitivity")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--key-dims", type=int, nargs="+", default=[32, 48, 64, 80, 96])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 500, 900])
    parser.add_argument("--train-prompts", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--heldout-prompts", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--fit-steps", type=int, default=36)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--target", choices=["softmax", "kda"], default="kda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["raw", "ema"], default="raw")
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=2)
    parser.add_argument("--pit-chunk-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--swanlab-mode", choices=["offline", "disabled", "online"], default="offline")
    parser.add_argument("--phase", choices=["all", "capture", "fit", "evaluate", "aggregate"], default="all")
    args = parser.parse_args()
    if set(args.train_prompts) & set(args.heldout_prompts):
        parser.error("Train/held-out prompt IDs must be disjoint across all resolutions/timesteps")
    if not args.train_prompts or not args.heldout_prompts or args.fit_steps < 1:
        parser.error("Both prompt splits and at least one calibration step are required")
    if sorted(set(args.key_dims)) != [32, 48, 64, 80, 96]:
        parser.error("This research protocol requires all five key dimensions")
    args.key_dims = sorted(set(args.key_dims))
    if args.threads < 1 or args.latency_repeats < 1 or any(t < 0 or t > 1000 for t in args.timesteps):
        parser.error("Invalid thread/repeat count or timestep")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(args.device)
    args.heads, args.hidden_size = 24, 1536
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo")
    def barrier():
        if world > 1:
            torch.distributed.barrier()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for split, prompts in [("train", args.train_prompts), ("heldout", args.heldout_prompts)]:
        for prompt in prompts:
            for resolution in args.resolutions:
                for timestep in args.timesteps:
                    case = {"split": split, "prompt": prompt, "resolution": resolution, "timestep": timestep}
                    cases.append((case, output / "records" / f"p{prompt:03d}_r{resolution}_t{timestep:04d}.pt"))
    details = metadata(args, cases)
    fingerprint = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Resume metadata changed. Use a new --output directory; no results are overwritten.")
    if rank == 0:
        atomic_json(manifest_path, {"fingerprint": fingerprint, "metadata": details})
    import swanlab
    tracker = swanlab.init(project="PiD-Head-Capacity-Sensitivity", name=f"layer{args.layer}-rank{rank}",
                           config=details, mode=args.swanlab_mode, log_dir=str(output / "swanlog" / f"rank{rank}"))
    try:
        if args.phase in ("all", "capture"):
            capture_cases(args, cases, fingerprint, rank, world)
        barrier()
        if args.phase in ("all", "fit", "evaluate"):
            absent = [str(path) for _, path in cases if not valid_case(path, fingerprint)]
            if absent:
                raise RuntimeError(f"Capture incomplete: {len(absent)} cases, first {absent[0]}")
        if args.phase in ("all", "fit"):
            fit_configs(args, cases, fingerprint, rank, world, tracker)
        barrier()
        allocation_path = output / "allocation.json"
        if args.phase in ("all", "evaluate"):
            if rank == 0:
                scores = training_scores(args, cases, fingerprint)
                unrestricted = allocate_budget(scores, args.key_dims, 64 * args.heads)
                allocation = allocate_budget(scores, args.key_dims, 48 * args.heads)
                random = torch.tensor(allocation)[torch.randperm(args.heads, generator=torch.Generator().manual_seed(args.seed + 1))].tolist()
                atomic_json(allocation_path, {"fingerprint": fingerprint, "selected": allocation, "random": random,
                                             "unrestricted": unrestricted,
                                             "budget": 48 * args.heads, "unrestricted_budget": 64 * args.heads,
                                             "selection_split": "train",
                                             "constraint": "primary compression comparison against uniform48 at75% of original key capacity; full64-budget unrestricted optimum retained separately",
                                             "random_control": "same dimension histogram and exact capacity; shuffled head assignment"})
            barrier()
        configurations = {"identity_64": [64] * args.heads, **{f"uniform_{d}": [d] * args.heads for d in args.key_dims}}
        if allocation_path.exists():
            allocation = json.loads(allocation_path.read_text())
            if allocation["fingerprint"] != fingerprint:
                raise ValueError("Allocation fingerprint mismatch")
            configurations.update({"unrestricted_64_budget": allocation["unrestricted"],
                                   "guided_48_budget": allocation["selected"], "random_48_budget": allocation["random"]})
        elif args.phase == "aggregate":
            # Partial aggregation must still expect all agreed configurations.
            configurations.update({name: [] for name in ("unrestricted_64_budget", "guided_48_budget", "random_48_budget")})
        if args.phase in ("all", "evaluate"):
            evaluate_configs(args, cases, fingerprint, rank, world, configurations, tracker)
        barrier()
        if rank == 0 and args.phase in ("all", "evaluate", "aggregate"):
            aggregate(args, cases, fingerprint, configurations)
    finally:
        tracker.finish()
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
