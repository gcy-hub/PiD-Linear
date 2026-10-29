"""Packed FLUX3 local joint Softmax plus stateless fixed-four quadrant delta memory.

The exact FP32 normal-equation solver is adapted from PiD spatial_memory.py.
No recurrent state, raster scan, decay, text writes, or cross-sample memory exists.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .lora import base_linear
from .vendor.gates import OutputGate
from .vendor.native import native_module
from .vendor.rms_norm import RMSNorm

native = native_module()


@dataclass(frozen=True)
class SampleLayout:
    start: int
    end: int
    image_start: int
    image_end: int
    text_start: int
    text_end: int
    grid: tuple[int, int]


@dataclass(frozen=True)
class AttentionInput:
    """A1 callback input: detached native modulated tokens, RoPE, immutable layouts.

    callback(block, info, target) may immediately call block.student_attention(info)
    inside torch.enable_grad(), compare with detached projected native target, and
    backward. The teacher continues with its native attention and FFN afterwards.
    """

    x_mod: torch.Tensor
    freqs: torch.Tensor
    samples: tuple[SampleLayout, ...]


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
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.d = heads, hidden // heads
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
            out, _ = joint_memory(q[None], k, v, beta, routes, sparse=True, shared=shared)
            # RMSNorm accumulates its second moment in FP32 internally, while
            # returning the native compute dtype before gating and projection.
            out = self.norm(out[0].to(compute_dtype)) * self.output_gate(x.to(compute_dtype))
            return self.to_out_linear(out.reshape(n, c))


class QDMBlock(nn.Module):
    """Own exactly one frozen native Block; replace only its attention computation."""

    def __init__(self, orig, region_size=8, region_chunk=16, halo=0, attention_mode="hybrid", memory_text_write=False):
        super().__init__()
        if region_size < 1 or region_chunk < 1 or halo < 0:
            raise ValueError("Window/chunk must be positive")
        self.orig = orig
        self.halo = halo
        self.region_size, self.region_chunk = region_size, region_chunk
        hidden = orig.q_proj.weight.shape[0]
        self.heads, self.d = orig.heads, hidden // orig.heads
        if attention_mode not in ("hybrid", "local-only", "memory-only"):
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        self.attention_mode = attention_mode
        if memory_text_write and attention_mode == "local-only":
            raise ValueError('memory_text_write requires a QDM branch')
        self.memory_text_write = bool(memory_text_write)
        self.remote = QuadrantMemory(hidden, self.heads) if attention_mode != "local-only" else None
        self.softmax_gate = OutputGate(hidden, self.heads, init_value=0.99)
        self.teacher_mode = False
        self.bypass_softmax_gate = False
        self.remote_enabled = self.remote is not None

    def raw_qkv(self, x, teacher=False):
        project = base_linear if teacher else lambda layer, z: layer(z)
        return tuple(
            project(layer, x).reshape(1, x.shape[0], self.heads, self.d).transpose(1, 2)
            for layer in (self.orig.q_proj, self.orig.k_proj, self.orig.v_proj)
        )

    def project(self, x, freqs, teacher=False):
        q, k, v = self.raw_qkv(x, teacher)
        q, k = self.orig.norm(q, k, freqs)
        return q, k, v.to(q.dtype)

    def local_attention(self, x, q, k, v, samples):
        """Return unprojected [tokens,heads,dim]; pad keys are always masked."""
        pieces = []
        for s in samples:
            if s.image_start != s.start or s.image_end != s.text_start or s.text_end != s.end:
                raise ValueError("T2I packed order must be image then text")
            iq, ik, iv = [z[:, :, s.image_start : s.image_end].transpose(1, 2) for z in (q, k, v)]
            qp, valid = pack_regions(iq, s.grid, self.region_size)
            if self.halo:
                key_indices, key_valid = halo_indices(s.grid, self.region_size, self.halo, x.device)
            else:
                kp, key_valid = pack_regions(ik, s.grid, self.region_size)
                vp, _ = pack_regions(iv, s.grid, self.region_size)
            tk, tv = k[:, :, s.text_start : s.text_end], v[:, :, s.text_start : s.text_end]
            nt = s.text_end - s.text_start
            windows = []
            regions, sz = qp.shape[1:3]
            for start in range(0, regions, self.region_chunk):
                stop = min(regions, start + self.region_chunk)
                count = stop - start
                qq = qp[:, start:stop].permute(0, 1, 3, 2, 4).reshape(count, self.heads, sz, self.d)
                if self.halo:
                    # Gather only this chunk, after native RoPE on original coordinates.
                    index = key_indices[start:stop]
                    kv_size = index.shape[1]
                    kk = ik[:, index].permute(0, 1, 3, 2, 4).reshape(count, self.heads, kv_size, self.d)
                    vv = iv[:, index].permute(0, 1, 3, 2, 4).reshape_as(kk)
                else:
                    kk = kp[:, start:stop].permute(0, 1, 3, 2, 4).reshape(count, self.heads, sz, self.d)
                    vv = vp[:, start:stop].permute(0, 1, 3, 2, 4).reshape_as(kk)
                kk = torch.cat((kk, tk.expand(count, -1, -1, -1)), dim=2)
                vv = torch.cat((vv, tv.expand(count, -1, -1, -1)), dim=2)
                mask = torch.cat((key_valid[start:stop], torch.ones(count, nt, device=x.device, dtype=torch.bool)), dim=1)[
                    :, None, None, :
                ]
                if not self.halo and s.grid[0] % self.region_size == 0 and s.grid[1] % self.region_size == 0:
                    mask = None
                out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, dropout_p=0.0)
                windows.append(out.permute(0, 2, 1, 3)[None])
            image = unpack_regions(torch.cat(windows, dim=1), s.grid, self.region_size)[0]
            text = F.scaled_dot_product_attention(
                q[:, :, s.text_start : s.text_end], k[:, :, s.start : s.end], v[:, :, s.start : s.end], dropout_p=0.0
            )[0].transpose(0, 1)
            pieces.extend((image, text))
        out = torch.cat(pieces, dim=0)
        if not self.bypass_softmax_gate:
            with torch.autocast(device_type=x.device.type, enabled=False):
                gate = self.softmax_gate(x.float())
            out = out * gate.to(out.dtype)
        return out

    def memory_attention(self, x, raw, sample):
        a, b = sample.image_start, sample.image_end
        qkv = tuple(t[0, :, a:b].transpose(0, 1) for t in raw)
        if not self.memory_text_write:
            return self.remote(x[a:b], qkv, sample.grid)
        ta, tb = sample.text_start, sample.text_end
        text = (x[ta:tb], raw[1][0, :, ta:tb].transpose(0, 1), raw[2][0, :, ta:tb].transpose(0, 1))
        return self.remote(x[a:b], qkv, sample.grid, text=text)

    def student_attention(self, info):
        x, freqs, samples = info.x_mod, info.freqs, info.samples
        raw = self.raw_qkv(x)
        q, k = self.orig.norm(raw[0], raw[1], freqs)
        if self.attention_mode == "memory-only":
            # Image queries use QDM exclusively. Text queries retain the native
            # packed global path, so text conditioning can still reach later
            # blocks. No image window partition, Halo gather or local SDPA runs.
            chunks = []
            for s in samples:
                a, b = s.image_start, s.image_end
                image = self.memory_attention(x, raw, s)
                text = F.scaled_dot_product_attention(
                    q[:, :, s.text_start:s.text_end], k[:, :, s.start:s.end],
                    raw[2][:, :, s.start:s.end].to(q.dtype), dropout_p=0.
                )[0].transpose(0, 1)
                if not self.bypass_softmax_gate:
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        gate = self.softmax_gate(x[s.text_start:s.text_end].float())
                    text = text * gate.to(text.dtype)
                text = self.orig.attn_out(text.reshape(s.text_end-s.text_start, -1))
                chunks.extend((image.to(text.dtype), text))
            return torch.cat(chunks, dim=0)
        local = self.local_attention(x, q, k, raw[2].to(q.dtype), samples)
        result = self.orig.attn_out(local.reshape(x.shape[0], -1))
        if self.remote_enabled:
            chunks = []
            for s in samples:
                a, b = s.image_start, s.image_end
                image = self.memory_attention(x, raw, s)
                chunks.extend(
                    (image, torch.zeros(s.text_end - s.text_start, x.shape[-1], device=x.device, dtype=image.dtype))
                )
            result = result + torch.cat(chunks, dim=0).to(result.dtype)
        return result

    def forward(self, x, freqs, bounds, mod, *, samples=None, alignment_callback=None, teacher_mode=None):
        teacher = self.teacher_mode if teacher_mode is None else teacher_mode
        if samples is None:
            raise ValueError("QDM requires immutable explicit sample/grid metadata")
        shift, scale, gate = mod
        if teacher:
            with torch.no_grad():
                x_mod = native.layernorm_modulation(x, scale, shift)
                q, k, v = self.project(x_mod, freqs, teacher=True)
                target = base_linear(self.orig.attn_out, native.packed_attention(q, k, v, bounds))
            if alignment_callback is not None:
                alignment_callback(
                    self, AttentionInput(x_mod.detach(), freqs.detach(), tuple(samples)), target.detach()
                )
            with torch.no_grad():
                gate_in, up = self.orig.mlp_in(x_mod).chunk(2, dim=-1)
                return x + gate * (target + self.orig.mlp_out(F.silu(gate_in) * up))
        x_mod = native.layernorm_modulation(x, scale, shift)
        attn = self.student_attention(AttentionInput(x_mod, freqs, tuple(samples)))
        gate_in, up = self.orig.mlp_in(x_mod).chunk(2, dim=-1)
        return x + gate * (attn + self.orig.mlp_out(F.silu(gate_in) * up))
