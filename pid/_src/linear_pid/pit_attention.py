"""PiT KDA with pretrained compression, or the optional uncompressed variant."""

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
        if (num_heads <= 0 or dim % num_heads or dim // num_heads > 256 or (dim // num_heads) % 4
                or backend not in {"fla", "reference"}):
            raise ValueError("PiT KDA requires head_dim divisible by 4 and <=256, and a supported backend")
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
            raise ValueError("PiT KDA supports DDP, not context parallelism")

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


class CompressedPiTBlock(PiTBlock):
    """Replace Full Attention only; retain the learned compression and expansion."""

    def __init__(self, source, *, heads=None, backend="fla"):
        nn.Module.__init__(self)
        self.pixel_dim, self.context_dim = source.pixel_dim, source.context_dim
        self.patch_size, self.attn_dim = source.patch_size, source.attn_dim
        self.num_heads = source.num_heads if heads is None else heads
        self.rope_mode = source.rope_mode
        self.rope_ref_grid_h, self.rope_ref_grid_w = source.rope_ref_grid_h, source.rope_ref_grid_w
        self.norm1, self.norm2, self.mlp = source.norm1, source.norm2, source.mlp
        self.adaLN_modulation = source.adaLN_modulation
        self.compress_to_attn = source.compress_to_attn
        self.expand_from_attn = source.expand_from_attn
        self.attn = PatchKDA(self.attn_dim, self.num_heads, backend=backend)
        self._pos_cache, self._cp_group = {}, None


def convert_pit_attention(net, *, heads=None, backend="fla", compressed=True):
    """Convert after strict source loading; never reinitialize inherited Linear maps."""
    for index, old in enumerate(net.pixel_blocks):
        requested_heads = heads if heads is not None else (old.num_heads if compressed else 64)
        if isinstance(old, (UncompressedPiTBlock, CompressedPiTBlock)):
            if (old.num_heads != requested_heads or old.attn.backend != backend
                    or isinstance(old, CompressedPiTBlock) != compressed):
                raise ValueError("Existing PiT KDA architecture differs")
            continue
        parameter = next(old.parameters())
        block_type = CompressedPiTBlock if compressed else UncompressedPiTBlock
        replacement = block_type(old, heads=requested_heads, backend=backend)
        net.pixel_blocks[index] = replacement.to(device=parameter.device, dtype=parameter.dtype)
    net.pixel_attn_hidden_size = net.pixel_blocks[0].attn_dim
    net.pixel_num_groups = net.pixel_blocks[0].num_heads
    net.pit_kda = True
    net.pit_kda_heads = net.pixel_num_groups
    net.pit_kda_compressed = compressed
    return net
