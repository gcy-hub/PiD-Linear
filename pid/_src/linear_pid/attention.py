"""PiD dual-stream KDA; the reference recurrence is only for small correctness tests."""

from __future__ import annotations

import json
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from pid._src.networks.pixeldit_official import apply_rotary_emb

FLA_COMMIT = "9f38d24980c46d46bd38614e743cdacd21906578"
LAYER_PRESETS = {
    0: [],
    4: [1, 4, 8, 12],
    6: [1, 2, 4, 6, 8, 12],
    8: [0, 1, 2, 4, 6, 8, 10, 12],
    10: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12],
}


def resolve_layers(layers, depth=14):
    if isinstance(layers, (str, int)) and str(layers).isdigit():
        count = int(layers)
        if count not in LAYER_PRESETS:
            raise ValueError(f"No preset for {count} layers; use comma-separated zero-based indices")
        layers = LAYER_PRESETS[count]
    elif isinstance(layers, str):
        layers = (
            json.loads(layers) if layers.strip().startswith("[") else [int(i) for i in layers.split(",") if i.strip()]
        )
    layers = list(layers)
    if any(not isinstance(i, Integral) or isinstance(i, bool) for i in layers):
        raise ValueError(f"Layer indices must be integers: {layers}")
    if len(set(layers)) != len(layers) or any(i < 0 or i >= depth for i in layers):
        raise ValueError(f"Invalid KDA layer indices for depth {depth}: {layers}")
    return sorted(int(i) for i in layers)


def reference_kda(q, k, v, log_decay, beta):
    """Differentiable, non-fused KDA, [B,T,H,D]; never used as a training fallback."""
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    for t in range(q.shape[1]):
        state = state * log_decay[:, t].exp().unsqueeze(-1)
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[:, :, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state) / math.sqrt(q.shape[-1]))
    return torch.stack(outputs, dim=1)


class Gates(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        d = dim // heads
        self.forget = nn.Sequential(nn.Linear(dim, d, bias=False), nn.Linear(d, dim, bias=False))
        self.beta = nn.Linear(dim, heads, bias=False)
        self.output = nn.Sequential(nn.Linear(dim, d, bias=False), nn.Linear(d, dim))
        self.A_log = nn.Parameter(torch.empty(heads, dtype=torch.float32).uniform_(1, 16).log())
        dt = torch.empty(dim, dtype=torch.float32).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.norm_weight = nn.Parameter(torch.ones(d))

    def normalize(self, output, hidden):
        out = output.float()
        out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-5)
        out = out * self.norm_weight.float()
        gate = self.output(hidden).view_as(output).float().sigmoid()
        return (out * gate).to(output.dtype).flatten(-2)


class JointKDA(nn.Module):
    is_linear_attention = True

    def __init__(self, dim=1536, num_heads=24, local_mixing=True, backend="fla"):
        super().__init__()
        if dim % num_heads or backend not in {"fla", "reference"}:
            raise ValueError("Invalid head dimensions or KDA backend")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.backend = backend
        self.qkv_x, self.qkv_y = nn.Linear(dim, 3 * dim, bias=False), nn.Linear(dim, 3 * dim, bias=False)
        self.proj_x, self.proj_y = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.gates_x, self.gates_y = Gates(dim, num_heads), Gates(dim, num_heads)
        self.local_mixing = local_mixing
        if local_mixing:
            self.image_conv = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False) for _ in range(3))
            self.text_conv = nn.ModuleList(nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False) for _ in range(3))
            for conv in [*self.image_conv, *self.text_conv]:
                nn.init.zeros_(conv.weight)
        for projection in [self.qkv_x, self.qkv_y, self.proj_x, self.proj_y]:
            nn.init.trunc_normal_(projection.weight, std=0.02)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        self.last_stats = {}
        self.collect_stats = False

    def set_context_parallel_group(self, group):
        if group is not None and group.size() > 1:
            raise ValueError("Linear-PiD v1 supports DDP, not context parallelism")

    def _project(self, hidden, projection, convs, grid=None, valid=None):
        hidden = hidden if valid is None else hidden * valid.unsqueeze(-1)
        pieces = projection(hidden).chunk(3, dim=-1)
        outputs = []
        for i, value in enumerate(pieces):
            if convs is not None:
                if grid is not None:
                    h, w = grid
                    spatial = value.transpose(1, 2).reshape(value.shape[0], self.dim, h, w)
                    value = value + convs[i](spatial).flatten(2).transpose(1, 2)
                else:
                    value = value + convs[i](value.transpose(1, 2)).transpose(1, 2)
            if valid is not None:
                value = value * valid.unsqueeze(-1)
            outputs.append(value.reshape(*value.shape[:2], self.num_heads, self.head_dim))
        return outputs

    def forward(self, x, y, pos_img, pos_txt=None, *, grid_size=None, text_valid_mask=None):
        b, nx, _ = x.shape
        ny = y.shape[1]
        if grid_size is None or grid_size[0] * grid_size[1] != nx:
            raise ValueError("KDA requires the explicit image patch grid, including non-square grids")
        valid_y = (
            torch.ones((b, ny), dtype=torch.bool, device=x.device)
            if text_valid_mask is None
            else text_valid_mask[:, :ny].bool()
        )
        if valid_y.shape != (b, ny):
            raise ValueError("Text validity mask must match the text stream")
        qx, kx, vx = self._project(x, self.qkv_x, self.image_conv if self.local_mixing else None, grid_size)
        qy, ky, vy = self._project(y, self.qkv_y, self.text_conv if self.local_mixing else None, valid=valid_y)
        qx, kx = apply_rotary_emb(qx, kx, pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, pos_txt)
        valid = torch.cat([valid_y, torch.ones((b, nx), dtype=torch.bool, device=x.device)], 1)
        q, k, v = [torch.cat(parts, 1) for parts in [(qy, qx), (ky, kx), (vy, vx)]]
        g = torch.cat([self.gates_y.forget(y), self.gates_x.forget(x)], 1).view(
            b, ny + nx, self.num_heads, self.head_dim
        )
        # Text/image have separate gate parameters; activate before calling the fused operator.
        decay = torch.cat(
            [
                -self.gates_y.A_log.float().exp()[None, None, :, None]
                * F.softplus(g[:, :ny].float() + self.gates_y.dt_bias.float().view(self.num_heads, self.head_dim)),
                -self.gates_x.A_log.float().exp()[None, None, :, None]
                * F.softplus(g[:, ny:].float() + self.gates_x.dt_bias.float().view(self.num_heads, self.head_dim)),
            ],
            1,
        )
        beta = torch.cat([self.gates_y.beta(y), self.gates_x.beta(x)], 1).sigmoid()
        lengths = valid.sum(1, dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        if self.backend == "reference":
            q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            out = v.new_zeros(v.shape)
            for i in range(b):
                keep = valid[i]
                out[i, keep] = reference_kda(
                    q[i : i + 1, keep],
                    k[i : i + 1, keep],
                    v[i : i + 1, keep].float(),
                    decay[i : i + 1, keep],
                    beta[i : i + 1, keep],
                )[0].to(v.dtype)
        else:
            if x.device.type != "cuda":
                raise RuntimeError("Production KDA requires CUDA; use backend=reference only for tiny tests")
            try:
                from fla.ops.kda import chunk_kda
            except ImportError as exc:
                raise RuntimeError(f"Install FLA commit {FLA_COMMIT}; old FLA releases have no KDA") from exc
            packed = [value[valid].unsqueeze(0).contiguous() for value in (q, k, v, decay, beta)]
            values, _ = chunk_kda(
                *packed,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                initial_state=None,
                output_final_state=False,
            )
            out = values.new_zeros(v.shape)
            out[valid] = values[0]
        ox, oy = out[:, ny:], out[:, :ny]
        ox = self.proj_x(self.gates_x.normalize(ox, x))
        oy = self.proj_y(self.gates_y.normalize(oy, y)) * valid_y.unsqueeze(-1)
        if self.collect_stats:
            self.last_stats = {
                "output_rms": ox.detach().float().square().mean().sqrt().item(),
                "log_decay_mean": decay.detach()[valid].mean().item(),
                "beta_mean": beta.detach()[valid].float().mean().item(),
            }
        return ox, oy


def convert_attention(net, layers, *, local_mixing=True, backend="fla"):
    layers = resolve_layers(layers, net.patch_depth)
    for i in layers:
        old = net.patch_blocks[i].attn
        if isinstance(old, JointKDA):
            continue
        replacement = JointKDA(old.dim, old.num_heads, local_mixing, backend)
        parameter = next(old.parameters())
        net.patch_blocks[i].attn = replacement.to(device=parameter.device, dtype=parameter.dtype)
    net.kda_layers = layers
    return net
