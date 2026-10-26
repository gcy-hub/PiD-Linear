"""QDM CP: split image queries, gather image K/V, sum quadrant statistics.

Text remains replicated. Shared parameters are averaged across the full DDP
world; CP copies never count as additional training samples.
"""

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as distributed_nn
from torch.nn import functional as F

from pid._src.networks.pixeldit_official import apply_rotary_emb
from pid._src.utils.context_parallel import cat_outputs_cp_with_grad


def topology(size):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if size < 1 or world % size:
        raise ValueError('World size must be divisible by context parallel size')
    if size == 1:
        return None, rank, world
    group = None
    for first in range(0, world, size):
        candidate = dist.new_group(list(range(first, first + size)))
        if first <= rank < first + size:
            group = candidate
    return group, rank // size, world // size


def _rope_slice(pos, first, last):
    return tuple(p[first:last] for p in pos) if isinstance(pos, tuple) else pos[first:last]


def local_forward(module, x, y, qkv, tqkv, pos_img, pos_txt, grid, full_k, full_v, first, mask):
    from .local_windows import region_indices

    original = module.orig
    b, n, c = x.shape
    heads, d = original.num_heads, original.head_dim
    q, _, _ = qkv
    tq, tk, tv = tqkv
    q, _ = apply_rotary_emb(original.q_norm_x(q), original.q_norm_x(q),
                             freqs_cis=_rope_slice(pos_img, first, first + n))
    _, k = apply_rotary_emb(original.k_norm_x(full_k), original.k_norm_x(full_k), freqs_cis=pos_img)
    tq, tk = original.q_norm_y(tq), original.k_norm_y(tk)
    if pos_txt is not None:
        tq, tk = apply_rotary_emb(tq, tk, freqs_cis=pos_txt)
    qi, qvalid, ki, kvalid = region_indices(grid, module.region_size, module.local_halo, 0, x.device)
    qvalid = qvalid & (qi >= first) & (qi < first + n)
    selected = qvalid.any(-1)
    qi, qvalid, ki, kvalid = qi[selected], qvalid[selected], ki[selected], kvalid[selected]
    text_length = y.shape[1]
    text_valid = (torch.ones(b, text_length, device=x.device, dtype=torch.bool) if mask is None
                  else mask.reshape(b, -1)[:, :text_length].bool())
    pieces = []
    for start in range(0, len(qi), module.region_chunk):
        stop = min(start + module.region_chunk, len(qi))
        count = stop - start
        local_q = (qi[start:stop] - first).clamp(0, n - 1)
        qq = q[:, local_q].permute(0, 1, 3, 2, 4).reshape(b * count, heads, -1, d)
        kk = k[:, ki[start:stop]].permute(0, 1, 3, 2, 4)
        vv = full_v[:, ki[start:stop]].permute(0, 1, 3, 2, 4)
        tk_window = tk.transpose(1, 2)[:, None].expand(b, count, heads, text_length, d)
        tv_window = tv.transpose(1, 2)[:, None].expand_as(tk_window)
        kk = torch.cat((tk_window, kk), dim=3).reshape(b * count, heads, -1, d)
        vv = torch.cat((tv_window, vv), dim=3).reshape_as(kk)
        allowed = torch.cat((text_valid[:, None].expand(b, count, text_length),
                             kvalid[None, start:stop].expand(b, count, -1)), dim=-1)
        output = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=allowed.reshape(b * count, 1, 1, -1))
        pieces.append(output.reshape(b, count, heads, -1, d).permute(0, 1, 3, 2, 4))
    order = qi[qvalid] - first
    image = torch.cat(pieces, dim=1).reshape(b, -1, heads, d)[:, qvalid.flatten()]
    image = image[:, torch.argsort(order)]
    text = F.scaled_dot_product_attention(tq.transpose(1, 2), torch.cat((tk, k), dim=1).transpose(1, 2),
                                         torch.cat((tv, full_v), dim=1).transpose(1, 2), attn_mask=mask).transpose(1, 2)
    if not module.bypass_softmax_gate:
        image = image * module.softmax_gate_x(x.reshape(-1, c)).reshape(b, n, heads, 1)
        text = text * module.softmax_gate_y(y.reshape(-1, c)).reshape(b, text_length, heads, 1)
    return original.proj_drop_x(original.proj_x(image.reshape(b, n, c))), original.proj_drop_y(
        original.proj_y(text.reshape(b, text_length, c)))


def memory_forward(memory, x, local_q, full_k, full_v, grid, first, group):
    from .spatial_memory import fixed_routes

    if memory.mode != 'fixed' or memory.banks != 4:
        raise ValueError('CP supports only fixed four-quadrant QDM')
    base = memory.base
    b, n, c = x.shape
    h, w = grid
    # Match native split-Q/gather-K,V CP. Full-grid convolution preserves the
    # exact 5x5 neighborhoods at CP seams; query work and statistics are local.
    values = []
    for raw, conv in zip((full_k, full_v), (base.short_conv_k, base.short_conv_v)):
        z = conv(raw.reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w))
        z = F.silu(z.flatten(2).transpose(1, 2).reshape_as(raw))
        values.append(z[:, first:first + n])
    k, v = values
    q = F.normalize(F.silu(local_q).float(), dim=-1)
    k = F.normalize(k.float(), dim=-1)
    beta = torch.sigmoid(base.beta_proj(x)).float()
    routes = fixed_routes(grid, 4, x.device)[first:first + n]
    indices = [torch.where(routes[:, bank] > 0)[0] for bank in range(4)]
    with torch.autocast(x.device.type, enabled=False):
        # One batched collective per statistic keeps forward/backward collective
        # order identical even when a rank has no queries in a given quadrant.
        matrices_a, matrices_b = [], []
        for ix in indices:
            kk = k[:, ix].transpose(1, 2)
            vv = v[:, ix].float().transpose(1, 2)
            weighted = kk * beta[:, ix].transpose(1, 2)[..., None]
            matrices_a.append(kk.mT @ weighted)
            matrices_b.append(vv.mT @ weighted)
        a = distributed_nn.all_reduce(torch.stack(matrices_a, dim=1), group=group)
        rhs = distributed_nn.all_reduce(torch.stack(matrices_b, dim=1), group=group)
        chol = torch.linalg.cholesky(torch.eye(base.d, device=x.device) + a)
        state = torch.cholesky_solve(rhs.mT, chol).mT
        outputs = [torch.einsum('bhvk,bnhk->bnhv', state[:, bank], q[:, ix])
                   for bank, ix in enumerate(indices)]
        order = torch.cat(indices)
        out = torch.cat(outputs, dim=1)[:, torch.argsort(order)]
    out = out.to(local_q.dtype)
    out = base.norm(out) * base.output_gate(x.reshape(-1, c)).reshape_as(out)
    return base.to_out_linear(out.reshape(b, n, c))


def student_forward(module, x, y, pos_img, pos_txt, grid, attn_mask):
    if module.local_mode != 'halo' or module.remote_rope or module.window_offset:
        raise ValueError('QDM CP requires unshifted Halo and NoPE remote memory')
    group = module._cp_group
    qkv, tqkv = module.raw_qkv(x, y)
    first = group.rank() * x.shape[1]
    if grid[0] * grid[1] != x.shape[1] * group.size():
        raise ValueError('CP image grid differs from the global token count')
    k = cat_outputs_cp_with_grad(qkv[1].contiguous(), 1, group)
    v = cat_outputs_cp_with_grad(qkv[2].contiguous(), 1, group)
    image, text = local_forward(module, x, y, qkv, tqkv, pos_img, pos_txt, grid, k, v, first, attn_mask)
    if module.component == 'hybrid':
        image = image + memory_forward(module.linear_attention, x, qkv[0], k, v, grid, first, group)
    elif module.component != 'local':
        raise ValueError('Unsupported CP component')
    return image, text
