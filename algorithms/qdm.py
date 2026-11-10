"""PiD/FLUX3 fixed-four memory; FP32 statistics and Cholesky solve, no scan.
Original gates retain their Apache-2.0 attribution in vendor/.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.gates import OutputGate
from .vendor.rms_norm import RMSNorm

def fixed_routes(grid, banks=4, device=None):
    side = math.isqrt(banks)
    if side * side != banks:
        raise ValueError("Fixed routes require square bank count")
    h, w = grid
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    labels = (y * side // h) * side + x * side // w
    return F.one_hot(labels.flatten(), banks).float()


def joint_memory(q, k, v, beta, write, read=None, mask=None, sparse=False, chunk_size=4096, shared=None):
    """Exact S=B(I+A)^-1, including each query's own write; all solves FP32.

    q/k/v [batch,tokens,heads,dim], beta [batch,tokens,heads], routes
    [batch,tokens,banks]. Sparse tied routes must be identical across batch.
    Optional shared=(K,V,beta) contains valid text tokens, written once into
    every bank. Their statistics are computed once and reused by all banks.
    """
    read = write if read is None else read
    b, n, heads, dk = k.shape
    banks, dv = write.shape[-1], v.shape[-1]
    if sparse and read is not write:
        raise ValueError("Sparse routes must be tied")
    if sparse and b > 1 and not torch.equal(write, write[:1].expand_as(write)):
        raise ValueError("Sparse routes must be identical across batch")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        if mask is not None:
            beta = beta * mask[:, :, None]
        out = torch.zeros(b, n, heads, dv, device=q.device)
        states = []
        eye = torch.eye(dk, device=q.device)
        shared_A = torch.zeros(b, heads, dk, dk, device=q.device)
        shared_B = torch.zeros(b, heads, dv, dk, device=q.device)
        if shared is not None:
            sk, sv, sb = (t.float() for t in shared)
            if sk.ndim != 4 or sk.shape[0] != b or sk.shape[2:] != (heads, dk):
                raise ValueError('Shared text K has incompatible batch/head/key dimensions')
            if sv.shape != (*sk.shape[:3], dv) or sb.shape != sk.shape[:3]:
                raise ValueError('Shared text V/beta shapes do not match K')
            for start in range(0, sk.shape[1], chunk_size):
                kk = sk[:, start:start+chunk_size].transpose(1, 2)
                vv = sv[:, start:start+chunk_size].transpose(1, 2)
                weight = sb[:, start:start+chunk_size].transpose(1, 2)[..., None]
                weighted = kk * weight
                shared_A = shared_A + kk.transpose(-1, -2) @ weighted
                shared_B = shared_B + vv.transpose(-1, -2) @ weighted
        for bank in range(banks):
            indices = torch.where(write[0, :, bank] > 0)[0] if sparse else torch.arange(n, device=q.device)
            A, B = shared_A, shared_B
            for start in range(0, len(indices), chunk_size):
                ix = indices[start : start + chunk_size]
                kk, vv = k[:, ix].transpose(1, 2), v[:, ix].transpose(1, 2)
                weight = (beta[:, ix] * write[:, ix, bank, None]).transpose(1, 2)
                weighted = kk * weight[..., None]
                A = A + kk.transpose(-1, -2) @ weighted
                B = B + vv.transpose(-1, -2) @ weighted
            chol = torch.linalg.cholesky(eye + A)
            state = torch.cholesky_solve(B.transpose(-1, -2), chol).transpose(-1, -2)
            states.append(state)
            for start in range(0, len(indices), chunk_size):
                ix = indices[start : start + chunk_size]
                result = torch.einsum("bhvk,bnhk->bnhv", state, q[:, ix]) * read[:, ix, bank, None, None]
                if torch.is_grad_enabled():
                    out = out.index_add(1, ix, result)
                else:
                    out.index_add_(1, ix, result)
        if mask is not None:
            out = out * mask[:, :, None, None]
        return out, torch.stack(states, dim=1)


def pack_regions(x, grid, size):
    """[B,H*W,...] -> padded [B,regions,size**2,...] with exact key validity."""
    h, w = map(int, grid)
    if size < 1 or h * w != x.shape[1]:
        raise ValueError(f"Invalid grid/window: {grid}/{size}")
    b, tail = x.shape[0], x.shape[2:]
    gh, gw = (h + size - 1) // size, (w + size - 1) // size
    image = x.flatten(2).transpose(1, 2).reshape(b, -1, h, w)
    image = F.pad(image, (0, gw * size - w, 0, gh * size - h))
    out = image.reshape(b, -1, gh, size, gw, size).permute(0, 2, 4, 3, 5, 1)
    out = out.reshape(b, gh * gw, size * size, *tail)
    rows = torch.arange(gh * size, device=x.device).view(gh, size)
    cols = torch.arange(gw * size, device=x.device).view(gw, size)
    valid = ((rows[:, None, :, None] < h) & (cols[None, :, None, :] < w)).reshape(gh * gw, size * size)
    return out, valid


def halo_indices(grid, size, halo, device):
    """Original-grid K/V addresses; edge replicas are masked, never written."""
    h, w = map(int, grid)
    gh, gw = (h + size - 1) // size, (w + size - 1) // size
    row, col = torch.meshgrid(torch.arange(gh, device=device) * size,
                              torch.arange(gw, device=device) * size, indexing="ij")
    dy, dx = torch.meshgrid(torch.arange(-halo, size + halo, device=device),
                            torch.arange(-halo, size + halo, device=device), indexing="ij")
    yy = row.flatten()[:, None] + dy.flatten()
    xx = col.flatten()[:, None] + dx.flatten()
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    return yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1), valid


def unpack_regions(packed, grid, size):
    h, w = map(int, grid)
    gh, gw = (h + size - 1) // size, (w + size - 1) // size
    b, tail = packed.shape[0], packed.shape[3:]
    x = packed.reshape(b, gh, gw, size, size, -1).permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(b, -1, gh * size, gw * size)[:, :, :h, :w]
    return x.flatten(2).transpose(1, 2).reshape(b, h * w, *tail)


class QuadrantMemory(nn.Module):
    def __init__(self, hidden, heads, chunk_size=4096):
        super().__init__()
        self.heads, self.d = heads, hidden // heads
        self.chunk_size = chunk_size
        self.beta_proj = nn.Linear(hidden, heads, bias=False)
        self.short_conv_k = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.short_conv_v = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        nn.init.normal_(self.short_conv_k.weight, std=1 / 5)
        nn.init.normal_(self.short_conv_v.weight, std=1 / 5)
        self.norm = RMSNorm(self.d)
        self.output_gate = OutputGate(hidden, heads, self.d, bottleneck=self.d, init="random")
        self.to_out_linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, qkv, grid, text=None):
        h, w = grid
        n, c = x.shape
        # Parameters and their optimizer state stay FP32. Feature projections and
        # the large independent output matrix use the native BF16 compute dtype;
        # only normalized feature accumulation and the exact solve promote FP32.
        compute_dtype = (
            torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else qkv[0].dtype
        )
        mixed_precision = compute_dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(
            device_type=x.device.type, dtype=compute_dtype if mixed_precision else None, enabled=mixed_precision
        ):
            raw_q, raw_k, raw_v = (t.to(compute_dtype) for t in qkv)
            q = F.silu(raw_q)
            q = F.normalize(q.float(), dim=-1).to(compute_dtype)
            feats = []
            for raw, conv in zip((raw_k, raw_v), (self.short_conv_k, self.short_conv_v)):
                z = raw.reshape(n, c).T.reshape(1, c, h, w)
                z = conv(z).flatten(2).transpose(1, 2).reshape(1, n, self.heads, self.d)
                feats.append(F.silu(z))
            k, v = feats
            k = F.normalize(k.float(), dim=-1).to(compute_dtype)
            beta = torch.sigmoid(self.beta_proj(x.to(compute_dtype)))[None]
            routes = fixed_routes(grid, 4, x.device)[None]
            shared = None
            if text is not None:
                text_x, text_k, text_v = text
                # Packed bounds already remove text padding. Text has no 2D
                # convolution and uses unrotated native K/V, like image memory.
                text_k = F.normalize(F.silu(text_k.to(compute_dtype)).float(), dim=-1).to(compute_dtype)
                text_v = F.silu(text_v.to(compute_dtype))
                text_beta = torch.sigmoid(self.beta_proj(text_x.to(compute_dtype)))[None]
                shared = (text_k[None], text_v[None], text_beta)
            out, _ = joint_memory(q[None], k, v, beta, routes, sparse=True, shared=shared, chunk_size=self.chunk_size)
            # RMSNorm accumulates its second moment in FP32 internally, while
            # returning the native compute dtype before gating and projection.
            out = self.norm(out[0].to(compute_dtype)) * self.output_gate(x.to(compute_dtype))
            return self.to_out_linear(out.reshape(n, c))
