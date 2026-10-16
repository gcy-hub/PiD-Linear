"""Sampled collision proxies on the actual NoPE VDA features; no N-by-N matrix."""

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def partial_correlation(x, y, controls):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    X = np.stack([np.ones(len(x)), *[rankdata(c) for c in controls]], 1)
    a, b = rankdata(x), rankdata(y)
    a = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    b = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else None


@torch.no_grad()
def collision_metrics(
    k, v, beta, error, teacher_energy, texture, grid, *, seed=42, anchors=512, candidates=64, tau=0.8
):
    k, v, beta = k[0].float(), v[0].float(), beta[0].float()
    n, heads, d = k.shape
    rng = torch.Generator(device=k.device).manual_seed(seed)
    ix = torch.randperm(n, device=k.device, generator=rng)[: min(anchors, n)]
    other = torch.randint(n, (len(ix), candidates), device=k.device, generator=rng)
    # Stratify sampling: half global, half local spatial offsets. Self pairs excluded.
    nearby = candidates // 2
    dy = torch.randint(-8, 9, (len(ix), nearby), device=k.device, generator=rng)
    dx = torch.randint(-8, 9, (len(ix), nearby), device=k.device, generator=rng)
    other[:, :nearby] = (ix[:, None] // grid[1] + dy).clamp(0, grid[0] - 1) * grid[1] + (
        ix[:, None] % grid[1] + dx
    ).clamp(0, grid[1] - 1)
    sim = (k[ix, None] * k[other]).sum(-1).clamp(-1, 1)
    va, vb = v[ix, None], v[other]
    disagreement = (va - vb).square().sum(-1) / (va.square().sum(-1) + vb.square().sum(-1)).clamp_min(1e-12)
    valid = (ix[:, None] != other)[..., None].expand_as(sim)
    conflict = (sim - tau).clamp_min(0) * disagreement * valid
    count = valid.sum((1, 2)).clamp_min(1)
    score = conflict.sum((1, 2)) / count
    weighted = (conflict * beta[other]).sum((1, 2)) / (beta[other] * valid).sum((1, 2)).clamp_min(1e-12)
    high = valid & (sim > tau)
    # Use a fixed subsample for the effective-dimension estimate and identify it as such.
    kk = k[torch.randperm(n, device=k.device, generator=rng)[: min(n, 4096)]].transpose(0, 1)
    with torch.autocast(device_type=k.device.type, enabled=False):
        A = kk.transpose(-1, -2) @ kk
        centered_k = kk - kk.mean(-2, keepdim=True)
        centered_A = centered_k.transpose(-1, -2) @ centered_k
    centered_effective = centered_A.diagonal(dim1=-2, dim2=-1).sum(-1).square() / centered_A.square().sum(
        (-1, -2)
    ).clamp_min(1e-12)
    mean_energy = kk.mean(-2).square().sum(-1) / kk.square().sum(-1).mean(-1).clamp_min(1e-12)
    centered = k - kk.mean(-2)[None]
    centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centered_sim = (centered[ix, None] * centered[other]).sum(-1).clamp(-1, 1)
    centered_score = ((centered_sim - tau).clamp_min(0) * disagreement * valid).sum((1, 2)) / count
    effective = A.diagonal(dim1=-2, dim2=-1).sum(-1).square() / A.square().sum((-1, -2)).clamp_min(1e-12)
    same_region = (ix[:, None] // grid[1] // 8 == other // grid[1] // 8) & (
        ix[:, None] % grid[1] // 8 == other % grid[1] // 8
    )
    distance = (
        ((ix[:, None] // grid[1] - other // grid[1]).float() / grid[0]).square()
        + ((ix[:, None] % grid[1] - other % grid[1]).float() / grid[1]).square()
    ).sqrt()
    arrays = {
        "indices": ix.cpu().numpy(),
        "conflict": score.cpu().numpy(),
        "beta_weighted_conflict": weighted.cpu().numpy(),
        "error": error[ix].float().cpu().numpy(),
        "teacher_energy": teacher_energy[ix].float().cpu().numpy(),
        "texture": texture[ix].float().cpu().numpy(),
        "value_energy": v[ix].square().mean((1, 2)).cpu().numpy(),
        "effective_dimensions": effective.cpu().numpy(),
        "centered_effective_dimensions": centered_effective.cpu().numpy(),
        "key_mean_energy_fraction": mean_energy.cpu().numpy(),
        "sampled_keys_bf16": k[ix].to(torch.bfloat16).view(torch.uint16).cpu().numpy(),
        "sampled_values_bf16": v[ix].to(torch.bfloat16).view(torch.uint16).cpu().numpy(),
        "sampled_beta": beta[ix].cpu().numpy(),
        "centered_conflict": centered_score.cpu().numpy(),
    }
    result = {
        "tau": tau,
        "sampled_anchors": len(ix),
        "candidates_per_anchor": candidates,
        "high_key_similarity_fraction": float(high.sum() / valid.sum()),
        "value_disagreement_high_key": float(disagreement[high].mean()) if high.any() else None,
        "effective_key_dimension_mean": float(effective.mean()),
        "nominal_key_dimension": d,
        "centered_effective_key_dimension_mean": float(centered_effective.mean()),
        "key_mean_energy_fraction": float(mean_energy.mean()),
        "centered_high_key_fraction": float(((centered_sim > tau) & valid).sum() / valid.sum()),
        "centered_conflict_error_spearman": correlation(arrays["centered_conflict"], arrays["error"]),
        "conflict_error_spearman": correlation(arrays["conflict"], arrays["error"]),
        "weighted_conflict_error_spearman": correlation(arrays["beta_weighted_conflict"], arrays["error"]),
        "conflict_error_partial_spearman": partial_correlation(
            arrays["conflict"], arrays["error"], [arrays["teacher_energy"], arrays["texture"], arrays["value_energy"]]
        ),
        "mean_attention_mse": float(error.mean()),
        "mean_conflict": float(score.mean()),
    }
    for label, group in [
        ("same_region", same_region),
        ("cross_region", ~same_region),
        ("near", distance < 0.1),
        ("far", distance > 0.4),
    ]:
        use = group[..., None] & valid
        result[label + "_high_key_fraction"] = float((high & use).sum() / use.sum()) if use.any() else None
        result[label + "_conflict"] = float(conflict[use].mean()) if use.any() else None
    for name in ["face", "text"]:
        # Filled from explicit ROI masks in the runner, not inferred from caption tags.
        result[name + "_roi_error"] = None
    return result, arrays
