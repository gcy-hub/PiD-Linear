"""Centered 2-D Sliding Softmax + text, preserving the pretrained projections.

NATTEN uses shifted, full-size neighborhoods at image edges. We replace just
those edge queries with exact clipped neighborhoods; no padded key is visible.
Only bounded query chunks are gathered, never an image-wide unfold or N x N mask.
"""
from functools import lru_cache

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pid._src.networks.pixeldit_official import apply_rotary_emb


@lru_cache(maxsize=32)
def edge_queries(grid, kernel, device):
    # Cached topology may be first created during standalone inference and later
    # reused by training. Its indices must remain ordinary autograd-safe tensors.
    with torch.inference_mode(False):
        h, w = grid
        r = kernel // 2
        positions = torch.arange(h*w, device=device)
        y, x = positions // w, positions % w
        return positions[(y < r) | (y >= h-r) | (x < r) | (x >= w-r)]


def _queries(q, k, v, tk, tv, indices, grid, kernel, text_valid, chunk):
    h, w = grid
    radius = kernel // 2
    delta = torch.arange(-radius, radius+1, device=q.device)
    dy, dx = torch.meshgrid(delta, delta, indexing='ij')
    pieces = []
    for first in range(0, len(indices), chunk):
        ids = indices[first:first+chunk]
        yy = ids[:, None] // w + dy.flatten()
        xx = ids[:, None] % w + dx.flatten()
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        keys = yy.clamp(0, h-1)*w + xx.clamp(0, w-1)

        def attend(q, k, v, tk, tv, ids, keys, valid, text_valid):
            b, _, heads, d = q.shape
            count = len(ids)
            qq = q[:, ids].reshape(b*count, heads, 1, d)
            kk = k[:, keys].permute(0, 1, 3, 2, 4)
            vv = v[:, keys].permute(0, 1, 3, 2, 4)
            tt = tk.transpose(1, 2)[:, None].expand(b, count, heads, -1, d)
            vt = tv.transpose(1, 2)[:, None].expand_as(tt)
            kk = torch.cat((kk, tt), dim=3).reshape(b*count, heads, -1, d)
            vv = torch.cat((vv, vt), dim=3).reshape_as(kk)
            mask = torch.cat((valid[None].expand(b, -1, -1),
                              text_valid[:, None].expand(b, count, -1)), dim=-1)
            out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask.reshape(b*count, 1, 1, -1))
            return out.reshape(b, count, heads, d)

        tensors = (q, k, v, tk, tv, ids, keys, valid, text_valid)
        if q.is_cuda and torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            out = checkpoint(attend, *tensors, use_reentrant=False)
        else:
            out = attend(*tensors)
        pieces.append(out)
    return torch.cat(pieces, dim=1)


def sliding_attention(q, k, v, tk, tv, grid, kernel=11, *, text_valid=None,
                      backend='natten', query_chunk=128):
    h, w = map(int, grid)
    if kernel < 1 or kernel % 2 != 1 or query_chunk < 1 or h*w != q.shape[1]:
        raise ValueError('Sliding attention needs an odd kernel, positive chunk and explicit matching grid')
    if backend not in ('natten', 'reference'):
        raise ValueError('Sliding backend must be natten or reference')
    if text_valid is None:
        text_valid = torch.ones(q.shape[0], tk.shape[1], device=q.device, dtype=torch.bool)
    if tuple(text_valid.shape) != tuple(tk.shape[:2]):
        raise ValueError('Text validity must have shape [batch, text_tokens]')
    if backend == 'reference' or min(h, w) < kernel:
        ids = torch.arange(h*w, device=q.device)
        return _queries(q, k, v, tk, tv, ids, (h, w), kernel, text_valid, query_chunk)
    if not q.is_cuda:
        raise ValueError('NATTEN backend requires CUDA; use reference for CPU checks')
    try:
        from natten import na2d
    except ImportError as error:
        raise RuntimeError('Install the matching NATTEN CUDA wheel; see docs/pid-sliding-window.md') from error
    outputs = []
    # Packing removes text padding from the joint denominator. Different samples
    # can have different valid lengths; no state or context is shared across them.
    edges = edge_queries((h, w), kernel, str(q.device))
    for index in range(q.shape[0]):
        valid = text_valid[index]
        keys = tk[index:index+1, valid].contiguous()
        values = tv[index:index+1, valid].contiguous()
        spatial = [t[index:index+1].reshape(1, h, w, *t.shape[2:]).contiguous() for t in (q, k, v)]
        out = na2d(*spatial, kernel_size=(kernel, kernel), stride=(1, 1), dilation=(1, 1),
                   is_causal=False, additional_keys=keys if keys.shape[1] else None,
                   additional_values=values if keys.shape[1] else None, backend='cutlass-fna')
        out = out.reshape(1, h*w, *q.shape[2:])
        if len(edges):
            border = _queries(q[index:index+1], k[index:index+1], v[index:index+1], keys, values,
                              edges, (h, w), kernel, torch.ones(1, keys.shape[1], device=q.device,
                              dtype=torch.bool), query_chunk)
            out = out.index_copy(1, edges, border)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def local_attention(module, x, y, qkv, tqkv, pos_img, pos_txt, grid, attn_mask=None):
    orig = module.orig
    b, n, c = x.shape
    heads, d = orig.num_heads, orig.head_dim
    q, k, v = qkv
    tq, tk, tv = tqkv
    q, k = apply_rotary_emb(orig.q_norm_x(q), orig.k_norm_x(k), freqs_cis=pos_img)
    tq, tk = orig.q_norm_y(tq), orig.k_norm_y(tk)
    if pos_txt is not None:
        tq, tk = apply_rotary_emb(tq, tk, freqs_cis=pos_txt)
    # Match existing Full/Local padding behavior: the existing attention mask,
    # when provided, governs Local/text. Do not repurpose QDM's separate mask.
    text_valid = None if attn_mask is None else attn_mask.reshape(b, -1)[:, :y.shape[1]].bool()
    image = sliding_attention(q, k, v, tk, tv, grid, module.sliding_kernel_size,
                              text_valid=text_valid, backend=module.sliding_backend,
                              query_chunk=module.sliding_query_chunk)
    text = F.scaled_dot_product_attention(tq.transpose(1, 2), torch.cat((tk, k), 1).transpose(1, 2),
                torch.cat((tv, v), 1).transpose(1, 2), attn_mask=attn_mask, dropout_p=0.).transpose(1, 2)
    if not module.bypass_softmax_gate:
        image = image * module.softmax_gate_x(x.reshape(-1, c)).reshape(b, n, heads, 1)
        text = text * module.softmax_gate_y(y.reshape(-1, c)).reshape(b, y.shape[1], heads, 1)
    return orig.proj_drop_x(orig.proj_x(image.reshape(b, n, c))), orig.proj_drop_y(orig.proj_y(text.reshape(b, -1, c)))
