"""Spatial local Softmax + exact bidirectional VDA; shared pretrained projections."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from pid._src.networks.pixeldit_official import apply_rotary_emb

from .scan import scan_bidirectional
from .vendor.delta_rule import VdnDelta
from .vendor.gates import OutputGate
from .vendor.layers import FrameKDAAlpha
from .vendor.rms_norm import RMSNorm
from .vendor.scan import _run_scans_inference, frame_statistics, gather_linear_state


def pack_regions(x, grid, size):
    """[B,H*W,...] -> [B,regions,size**2,...], row-major regions and pixels."""
    h, w = map(int, grid)
    if size < 1 or h * w != x.shape[1]:
        raise ValueError(f"Invalid region size/grid: {size}, {grid}, tokens={x.shape[1]}")
    b, tail = x.shape[0], x.shape[2:]
    gh, gw = (h + size - 1) // size, (w + size - 1) // size
    image = x.flatten(2).transpose(1, 2).reshape(b, -1, h, w)
    image = F.pad(image, (0, gw * size - w, 0, gh * size - h))
    packed = image.reshape(b, -1, gh, size, gw, size).permute(0, 2, 4, 3, 5, 1)
    packed = packed.reshape(b, gh * gw, size * size, *tail)
    rows = torch.arange(gh * size, device=x.device).view(gh, size)
    cols = torch.arange(gw * size, device=x.device).view(gw, size)
    valid = ((rows[:, None, :, None] < h) & (cols[None, :, None, :] < w)).reshape(gh * gw, size * size)
    return packed, valid


def unpack_regions(packed, grid, size):
    h, w = map(int, grid)
    gh, gw = (h + size - 1) // size, (w + size - 1) // size
    b, tail = packed.shape[0], packed.shape[3:]
    x = packed.reshape(b, gh, gw, size, size, -1).permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(b, -1, gh * size, gw * size)[:, :, :h, :w]
    return x.flatten(2).transpose(1, 2).reshape(b, h * w, *tail)


class SpatialVDA(nn.Module):
    def __init__(self, dim, heads, region_size):
        super().__init__()
        self.heads, self.d, self.region_size = heads, dim // heads, region_size
        self.alpha = FrameKDAAlpha(dim, heads, self.d)
        self.beta_proj = nn.Linear(dim, heads, bias=False)
        self.short_conv_k = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.short_conv_v = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        nn.init.normal_(self.short_conv_k.weight, std=1 / 5)
        nn.init.normal_(self.short_conv_v.weight, std=1 / 5)
        self.norm = RMSNorm(self.d)
        self.output_gate = OutputGate(dim, heads, self.d, bottleneck=self.d, init="random")
        self.to_out_linear = nn.Linear(dim, dim, bias=False)
        self.last_alpha = None

    def features(self, qkv, grid):
        h, w = grid
        b = qkv[0].shape[0]
        features = [F.silu(qkv[0])]
        for raw, conv in zip(qkv[1:], (self.short_conv_k, self.short_conv_v)):
            z = raw.reshape(b, h * w, -1).transpose(1, 2).reshape(b, -1, h, w)
            z = conv(z).flatten(2).transpose(1, 2).reshape_as(raw)
            features.append(F.silu(z))
        q, k, v = features
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
        return q, k, v

    def forward(self, x, y, qkv, text_qkv, grid, text_mask, inference=False, order="row"):
        if inference and torch.is_grad_enabled():
            raise RuntimeError("VDA inference path requires torch.no_grad()")
        b, n, c = x.shape
        rsize = self.region_size
        heads, d = self.heads, self.d
        q, k, v = self.features(qkv, grid)
        qp, valid = pack_regions(q, grid, rsize)
        kp, _ = pack_regions(k, grid, rsize)
        vp, _ = pack_regions(v, grid, rsize)
        xp, _ = pack_regions(x, grid, rsize)
        regions, s = qp.shape[1:3]
        permutation = torch.arange(regions, device=x.device)
        if order == "column":
            gh, gw = (grid[0] + rsize - 1) // rsize, (grid[1] + rsize - 1) // rsize
            permutation = permutation.reshape(gh, gw).T.flatten()
        elif order != "row":
            raise ValueError(order)
        qp, kp, vp, xp = [p[:, permutation] for p in (qp, kp, vp, xp)]
        valid = valid[permutation]
        beta = torch.sigmoid(self.beta_proj(xp)) * valid[None, :, :, None]

        def stats_shape(p):
            return p.permute(1, 0, 3, 2, 4).reshape(regions, b * heads, s, d)

        A, B = frame_statistics(
            stats_shape(kp), stats_shape(vp), beta.permute(1, 0, 3, 2).reshape(regions, b * heads, s)
        )
        mean = (xp.float() * valid[None, :, :, None]).sum(2) / valid.sum(1)[None, :, None]
        alpha = (
            self.alpha(mean.reshape(b * regions, c))
            .reshape(b, regions, heads, d)
            .permute(1, 0, 2, 3)
            .reshape(regions, b * heads, d)
        )
        self.last_alpha = alpha.detach()
        tk = F.normalize(F.silu(text_qkv[1]).float(), dim=-1).to(k.dtype)
        tv = F.silu(text_qkv[2])
        tmask = torch.ones(y.shape[:2], device=y.device, dtype=torch.bool) if text_mask is None else text_mask.bool()
        tb = torch.sigmoid(self.beta_proj(y)) * tmask[:, :, None]
        At, Bt = frame_statistics(tk.permute(0, 2, 1, 3), tv.permute(0, 2, 1, 3), tb.permute(0, 2, 1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, seed = VdnDelta().factor_apply(torch.ones(b, heads, d, device=x.device), At, Bt)
            seed = seed.reshape(b * heads, d, d) * 0.5
            scan = _run_scans_inference if inference else scan_bidirectional
            prefix, suffix = scan(VdnDelta(), alpha, A, B, text_state=seed)
            state = gather_linear_state(prefix, suffix, alpha, [(i, i) for i in range(regions)], text_state=seed)
        state = state.reshape(regions, b, heads, d, d).permute(1, 0, 2, 3, 4).to(qp.dtype)
        out = torch.einsum("brhvk,brshk->brshv", state, qp)
        inverse = torch.argsort(permutation)
        out = unpack_regions(out[:, inverse], grid, rsize)
        out = self.norm(out) * self.output_gate(x.reshape(-1, c)).reshape(b, n, heads, d)
        return self.to_out_linear(out.reshape(b, n, c))


class JointVDN(nn.Module):
    """Owns one original module; teacher bypasses every new parameter and LoRA."""

    is_linear_attention = True
    is_vdn_attention = True

    def __init__(self, original, region_size=8, region_chunk=16):
        super().__init__()
        self.orig = original
        self.region_size, self.region_chunk = region_size, region_chunk
        dim, heads = original.dim, original.num_heads
        self.linear_attention = SpatialVDA(dim, heads, region_size)
        self.softmax_gate_x = OutputGate(dim, heads, init_value=0.99)
        self.softmax_gate_y = OutputGate(dim, heads, init_value=0.99)
        self.teacher_mode = False
        self.inference_mode = False
        self.component = "hybrid"
        self.bypass_softmax_gate = False
        self.alignment_callback = None
        self.scan_order = "row"
        self.local_mode = "window"
        self.local_halo = 0
        self.window_offset = 0

    def set_context_parallel_group(self, group):
        if group is not None:
            raise ValueError("PiD-VDN uses data parallelism; region sharding is not implemented")

    def raw_qkv(self, x, y):
        b, n, c = x.shape
        h = self.orig.num_heads
        d = c // h

        def project(z, layer):
            return layer(z).reshape(b, z.shape[1], 3, h, d).unbind(2)

        return project(x, self.orig.qkv_x), project(y, self.orig.qkv_y)

    def local(self, x, y, qkv, tqkv, pos_img, pos_txt, grid, attn_mask=None):
        if self.local_mode != "window":
            from .local_windows import local_attention
            return local_attention(self,x,y,qkv,tqkv,pos_img,pos_txt,grid,attn_mask)
        orig = self.orig
        b, n, c = x.shape
        heads, d = orig.num_heads, orig.head_dim
        q, k, v = qkv
        tq, tk, tv = tqkv
        q, k = apply_rotary_emb(orig.q_norm_x(q), orig.k_norm_x(k), freqs_cis=pos_img)
        tq, tk = orig.q_norm_y(tq), orig.k_norm_y(tk)
        if pos_txt is not None:
            tq, tk = apply_rotary_emb(tq, tk, freqs_cis=pos_txt)
        qp, valid = pack_regions(q, grid, self.region_size)
        kp, _ = pack_regions(k, grid, self.region_size)
        vp, _ = pack_regions(v, grid, self.region_size)
        regions, s = qp.shape[1:3]
        text_len = y.shape[1]
        pieces = []
        for start in range(0, regions, self.region_chunk):
            stop = min(regions, start + self.region_chunk)
            count = stop - start
            qq = qp[:, start:stop].permute(0, 1, 3, 2, 4).reshape(b * count, heads, s, d)
            kk = kp[:, start:stop].permute(0, 1, 3, 2, 4)
            vv = vp[:, start:stop].permute(0, 1, 3, 2, 4)
            textk = tk.transpose(1, 2)[:, None].expand(b, count, heads, text_len, d)
            textv = tv.transpose(1, 2)[:, None].expand(b, count, heads, text_len, d)
            kk = torch.cat((textk, kk), dim=3).reshape(b * count, heads, text_len + s, d)
            vv = torch.cat((textv, vv), dim=3).reshape_as(kk)
            text_valid = torch.ones(b, text_len, device=x.device, dtype=torch.bool)
            if attn_mask is not None:
                text_valid = attn_mask.reshape(b, -1)[:, :text_len].bool()
            mask = torch.cat(
                (text_valid[:, None].expand(b, count, text_len), valid[None, start:stop].expand(b, count, s)), dim=-1
            ).reshape(b * count, 1, 1, text_len + s)
            if attn_mask is None and grid[0] % self.region_size == 0 and grid[1] % self.region_size == 0:
                mask = None
            out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, dropout_p=0.0)
            pieces.append(out.reshape(b, count, heads, s, d).permute(0, 1, 3, 2, 4))
        image = unpack_regions(torch.cat(pieces, dim=1), grid, self.region_size)
        jointk = torch.cat((tk, k), dim=1).transpose(1, 2)
        jointv = torch.cat((tv, v), dim=1).transpose(1, 2)
        text = F.scaled_dot_product_attention(
            tq.transpose(1, 2), jointk, jointv, attn_mask=attn_mask, dropout_p=0.0
        ).transpose(1, 2)
        if not self.bypass_softmax_gate:
            image = image * self.softmax_gate_x(x.reshape(-1, c)).reshape(b, n, heads, 1)
            text = text * self.softmax_gate_y(y.reshape(-1, c)).reshape(b, text_len, heads, 1)
        return orig.proj_drop_x(orig.proj_x(image.reshape(b, n, c))), orig.proj_drop_y(
            orig.proj_y(text.reshape(b, text_len, c))
        )

    def student_forward(self, x, y, pos_img, pos_txt=None, *, grid_size, text_valid_mask=None, attn_mask=None):
        qkv, tqkv = self.raw_qkv(x, y)
        if self.component == "vda":
            return self.linear_attention(
                x, y, qkv, tqkv, grid_size, text_valid_mask, self.inference_mode, order=self.scan_order
            ), torch.zeros_like(y)
        out_x, out_y = self.local(x, y, qkv, tqkv, pos_img, pos_txt, grid_size, attn_mask)
        if self.component == "hybrid":
            out_x = out_x + self.linear_attention(
                x, y, qkv, tqkv, grid_size, text_valid_mask, self.inference_mode, order=self.scan_order
            )
        return out_x, out_y

    def forward(self, x, y, pos_img, pos_txt=None, *, grid_size, text_valid_mask=None, attn_mask=None):
        if self.teacher_mode:
            output = self.orig(x, y, pos_img, pos_txt, attn_mask)
            if self.alignment_callback is not None:
                self.alignment_callback(
                    self,
                    (x, y, pos_img, pos_txt),
                    dict(grid_size=grid_size, text_valid_mask=text_valid_mask, attn_mask=attn_mask),
                    output,
                )
            return output
        return self.student_forward(
            x, y, pos_img, pos_txt, grid_size=grid_size, text_valid_mask=text_valid_mask, attn_mask=attn_mask
        )


def convert_attention(net, region_size=8, local_mode="window", local_halo=0, local_shift=4, local_chunk=16):
    for index,block in enumerate(net.patch_blocks):
        if isinstance(block.attn, JointVDN):
            raise ValueError("Already converted; reconstruct checkpoint architecture instead")
        block.attn = JointVDN(block.attn, region_size, region_chunk=local_chunk)
        block.attn.local_mode=local_mode
        block.attn.local_halo=local_halo if local_mode=="halo" else 0
        block.attn.window_offset=local_shift if local_mode=="alternating_shift" and index%2 else 0
    return net


@contextmanager
def execution_mode(net, teacher=False, inference=False):
    modules = [m for m in net.modules() if isinstance(m, JointVDN)]
    adapters = [m for m in net.modules() if hasattr(m, "adapters_enabled")]
    previous = [(m, m.teacher_mode, m.inference_mode) for m in modules]
    previous_adapters = [(m, m.adapters_enabled) for m in adapters]
    for m in modules:
        m.teacher_mode, m.inference_mode = teacher, inference
    for m in adapters:
        m.adapters_enabled = not teacher
    try:
        yield
    finally:
        for m, t, i in previous:
            m.teacher_mode, m.inference_mode = t, i
        for m, e in previous_adapters:
            m.adapters_enabled = e
