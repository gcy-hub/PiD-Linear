"""Clipped 2D sliding image attention with jointly normalized text.

Each axis uses offsets [-floor(kernel/2), floor((kernel-1)/2)]. Odd kernels
are centered; even kernels include one extra position above/left.

CUDA uses PyTorch FlexAttention, including its backward. No unfold, copied
per-query K/V neighborhoods, or dense N*N CUDA mask is materialized. CPU SDPA
is a small-input correctness reference. Text queries retain global visibility.
"""
from functools import lru_cache

import torch
from torch.nn import functional as F

# Configure before entering non-reentrant checkpoint: setting this inside its
# forward is restored on exit, before the lazy backward compiler runs.
torch._dynamo.config.specialize_int = True
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 4096)
torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 8192)


def validate_local_attention(saved, current):
    if saved.get('replace_image_blocks', False) != current.get('replace_image_blocks', False):
        raise ValueError('Checkpoint early image block replacement scope mismatch')
    old, new = saved.get('local_attention', 'halo'), current.get('local_attention', 'halo')
    if old != new:
        raise ValueError('Checkpoint local attention geometry mismatch')
    if new == 'sliding' and saved.get('sliding_kernel_size', 11) != current.get('sliding_kernel_size', 11):
        raise ValueError('Checkpoint sliding kernel size mismatch')


def _predicate(height, width, text, kernel):
    image = height*width
    if kernel % 2:
        radius = kernel//2
        def centered(b, head, query, key):
            local = ((query//width-key//width).abs() <= radius) & ((query%width-key%width).abs() <= radius)
            return (query >= image) | (key >= image) | local
        return centered
    # Preserve the original centered predicate for odd windows.
    left, right = kernel//2, (kernel-1)//2
    def allowed(b, head, query, key):
        dy, dx = key//width-query//width, key%width-query%width
        # Keep captured scalar bounds nonnegative: torch 2.10 Flex lowering
        # cannot realize a negated symbolic closure scalar as a typed buffer.
        local = (dy + left >= 0) & (dy <= right) & (dx + left >= 0) & (dx <= right)
        # FlexAttention separately masks against the actual Q/K tensor lengths.
        return (query >= image) | (key >= image) | local
    return allowed


@lru_cache(maxsize=24)
def _block_mask(height, width, text, kernel, device):
    from torch.nn.attention.flex_attention import create_block_mask
    length = height*width+text
    return create_block_mask(_predicate(height, width, text, kernel), B=None, H=None,
                             Q_LEN=length, KV_LEN=length, device=device, _compile=True)


@lru_cache(maxsize=1)
def _compiled_attention():
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    # Text lengths and 2K aspect buckets vary. Static specialization hits the
    # default recompilation limit and silently falls back to dense eager math.
    # Fullgraph rejects such fallback instead of risking an N*N allocation.
    # On torch 2.10, symbolic integer geometry can yield incorrect dK/dV when
    # reusing backward across layouts. This must remain active during lazy
    # backward compilation as well as forward, so it cannot be a forward-only
    # config context. Tensor sequence lengths below remain dynamic.
    def attend(q, k, v, block_mask, kernel_options):
        current = BlockMask(
            seq_lengths=(q.shape[-2], k.shape[-2]),
            kv_num_blocks=block_mask.kv_num_blocks, kv_indices=block_mask.kv_indices,
            full_kv_num_blocks=block_mask.full_kv_num_blocks, full_kv_indices=block_mask.full_kv_indices,
            q_num_blocks=block_mask.q_num_blocks, q_indices=block_mask.q_indices,
            full_q_num_blocks=block_mask.full_q_num_blocks, full_q_indices=block_mask.full_q_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE, mask_mod=block_mask.mask_mod)
        return flex_attention(q, k, v, block_mask=current, kernel_options=kernel_options)
    return torch.compile(attend, dynamic=True, fullgraph=True)


def sliding_joint_attention(q, k, v, grid, text_length, kernel_size=11):
    """[1,heads,H*W+text,dim], image-first; invoke separately for each sample."""
    height, width = map(int, grid)
    if min(height, width, kernel_size) < 1 or text_length < 0:
        raise ValueError('Sliding needs a valid explicit grid and positive kernel')
    length = height*width+text_length
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[2] != length:
        raise ValueError('Sliding Q/K/V must be one image-first sample with matching dimensions')
    if q.is_cuda:
        # A40/Ampere shared memory cannot fit the default 128x128 tile at D128.
        # The decoding heuristic assumes BLOCK_M >= sequence length for short
        # inputs, conflicting with our Ampere tile64. Use the attention kernel
        # consistently for both small correctness checks and training inputs.
        options = {'BACKEND': 'TRITON'}
        if torch.cuda.get_device_capability(q.device)[0] < 9:
            options.update(BLOCK_M=64, BLOCK_N=64, num_stages=2,
                           bwd_BLOCK_M1=32, bwd_BLOCK_N1=64, bwd_BLOCK_M2=64, bwd_BLOCK_N2=32)
        return _compiled_attention()(q, k, v,
            block_mask=_block_mask(height, width, text_length, kernel_size, str(q.device)), kernel_options=options)
    indices = torch.arange(length, device=q.device)
    mask = _predicate(height, width, text_length, kernel_size)(None, None, indices[:, None], indices[None, :])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
