"""Uncompressed PiT: one flattened P*P*pixel_dim token per patch, followed by KDA.

The patch sequence length stays H/P * W/P. No learned spatial compression or
expansion remains; the output is reshaped back to pixels before the residual.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pid._src.linear_pid.attention import FLA_COMMIT, Gates, reference_kda
from pid._src.networks.pixeldit_official import PiTBlock, apply_rotary_emb


class PatchKDA(nn.Module):
    is_linear_attention = True

    def __init__(self, dim, num_heads=64, *, backend="fla"):
        super().__init__()
        if dim % num_heads or dim // num_heads > 256 or backend not in {"fla", "reference"}:
            raise ValueError("PiT KDA requires divisible dimensions, head_dim <=256 and a supported backend")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.backend = backend
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.gates = Gates(dim, num_heads)
        for projection in (self.qkv, self.proj):
            nn.init.trunc_normal_(projection.weight, std=0.02)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def set_context_parallel_group(self, group):
        if group is not None and group.size() > 1:
            raise ValueError("Uncompressed PiT KDA supports DDP, not context parallelism")

    def forward(self, x, pos, mask=None):
        if mask is not None:
            raise ValueError("PiT KDA expects a complete image patch grid, without a padding mask")
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = apply_rotary_emb(q, k, pos)
        forget = self.gates.forget(x).reshape(b, n, self.num_heads, self.head_dim)
        decay = -self.gates.A_log.float().exp()[None, None, :, None] * F.softplus(
            forget.float() + self.gates.dt_bias.float().reshape(self.num_heads, self.head_dim)
        )
        beta = self.gates.beta(x).sigmoid()
        if self.backend == "reference":
            out = reference_kda(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
                                v.float(), decay, beta.float()).to(v.dtype)
        else:
            if x.device.type != "cuda":
                raise RuntimeError("Production PiT KDA requires CUDA; reference is only for small tests")
            try:
                from fla.ops.kda import chunk_kda
            except ImportError as exc:
                raise RuntimeError(f"Install FLA commit {FLA_COMMIT}") from exc
            out, _ = chunk_kda(q.contiguous(), k.contiguous(), v.contiguous(), decay.contiguous(),
                               beta.contiguous(), use_qk_l2norm_in_kernel=True,
                               initial_state=None, output_final_state=False)
        return self.proj(self.gates.normalize(out, x))


class UncompressedPiTBlock(PiTBlock):
    """Keep the pretrained pixel AdaLN/FFN; replace only the attention branch."""

    def __init__(self, source, *, heads=64, backend="fla"):
        nn.Module.__init__(self)
        self.pixel_dim, self.context_dim = source.pixel_dim, source.context_dim
        self.patch_size = source.patch_size
        self.attn_dim = self.patch_size**2 * self.pixel_dim
        self.num_heads = heads
        self.rope_mode = source.rope_mode
        self.rope_ref_grid_h, self.rope_ref_grid_w = source.rope_ref_grid_h, source.rope_ref_grid_w
        self.norm1, self.norm2, self.mlp = source.norm1, source.norm2, source.mlp
        self.adaLN_modulation = source.adaLN_modulation
        self.attn = PatchKDA(self.attn_dim, heads, backend=backend)
        self._pos_cache, self._cp_group = {}, None

    def _compress_pixels(self, pixels):
        return pixels.flatten(1)

    def _expand_pixels(self, attention):
        return attention.reshape(-1, self.patch_size**2, self.pixel_dim)


def convert_pit_attention(net, *, heads=64, backend="fla"):
    """Remove the old attention and both Linear maps after strict source loading."""
    for index, old in enumerate(net.pixel_blocks):
        if isinstance(old, UncompressedPiTBlock):
            if old.num_heads != heads or old.attn.backend != backend:
                raise ValueError("Existing PiT KDA architecture differs")
            continue
        parameter = next(old.parameters())
        replacement = UncompressedPiTBlock(old, heads=heads, backend=backend)
        net.pixel_blocks[index] = replacement.to(device=parameter.device, dtype=parameter.dtype)
    net.pixel_attn_hidden_size = net.patch_size**2 * net.pixel_hidden_size
    net.pixel_num_groups = heads
    net.pit_kda = True
    net.pit_kda_heads = heads
    return net
