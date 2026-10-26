"""DeCo's immutable teacher and image-only local Softmax + quadrant delta memory.

Only the minimal official DeCo modules are vendored; there are no FLA, Lightning,
conditioner or external-checkout imports. New parameters remain FP32 while the
pretrained backbone is BF16. CUDA callers should use BF16 autocast.
"""
from contextlib import contextmanager
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vendor.dit_t2i_DeCo import PixNerDiT, Attention, modulate
from .vendor.rope import apply_rotary_emb
from .vendor.rmsnorm import RMSNorm
from .vendor.gates import OutputGate


def joint_memory(q, k, v, beta, grid):
    """Exact independent four-quadrant ridge states, with FP32 accumulation/solve.

    Q/K/V are [batch, image_tokens, heads, channels], beta [batch,tokens,heads].
    S = B(I+A)^-1; A=sum beta*k*k^T, B=sum beta*v*k^T. Queries read
    the state of their fixed spatial quadrant, including their own writes.
    """
    b, n, heads, dk = k.shape
    h, w = map(int, grid)
    if h * w != n or min(h, w) < 1:
        raise ValueError('Image token count does not match explicit grid')
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        yy, xx = torch.meshgrid(torch.arange(h, device=q.device),
                                torch.arange(w, device=q.device), indexing='ij')
        routes = ((yy * 2 // h) * 2 + xx * 2 // w).flatten()
        out = torch.zeros(b, n, heads, v.shape[-1], device=q.device, dtype=torch.float32)
        eye = torch.eye(dk, device=q.device, dtype=torch.float32)
        for bank in range(4):
            indices = torch.where(routes == bank)[0]
            kk, vv = k[:, indices].transpose(1, 2), v[:, indices].transpose(1, 2)
            weighted_k = kk * beta[:, indices].transpose(1, 2)[..., None]
            a = kk.transpose(-1, -2) @ weighted_k
            bb = vv.transpose(-1, -2) @ weighted_k
            chol = torch.linalg.cholesky(eye + a)
            # Solve L z = B^T, then L^T state^T = z. This is the same SPD
            # system as cholesky_solve, but avoids its batched MAGMA potrs path,
            # which raised a CUDA assertion during the real A1 backward run.
            rhs = torch.linalg.solve_triangular(chol, bb.transpose(-1, -2), upper=False)
            state = torch.linalg.solve_triangular(chol.transpose(-1, -2), rhs, upper=True).transpose(-1, -2)
            value = torch.einsum('bhvk,bnhk->bnhv', state, q[:, indices])
            out = out.index_copy(1, indices, value)
        return out


class LoRALinear(nn.Module):
    """Independent rank-r adapters on each Q/K/V segment, zero delta initially."""
    def __init__(self, base, rank=64, alpha=64, segments=1):
        super().__init__()
        if rank <= 0 or alpha <= 0 or base.out_features % segments:
            raise ValueError('Invalid LoRA rank, alpha or segments')
        self.base, self.rank, self.alpha, self.segments = base, rank, alpha, segments
        self.adapters_enabled = True
        self.A = nn.ParameterList([nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=torch.float32)) for _ in range(segments)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(
            base.out_features // segments, rank, device=base.weight.device, dtype=torch.float32)) for _ in range(segments)])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        output = self.base(x)
        if not self.adapters_enabled:
            return output
        # Cast the delta explicitly: FP32 master parameters stay intact under BF16.
        delta = [F.linear(F.linear(x.to(a.dtype), a), b) for a, b in zip(self.A, self.B)]
        return output + torch.cat(delta, -1).to(output.dtype) * (self.alpha / self.rank)


class QuadrantDeltaMemory(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.d = heads, dim // heads
        self.beta_proj = nn.Linear(dim, heads, bias=False)
        self.short_conv_k = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.short_conv_v = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        nn.init.normal_(self.short_conv_k.weight, std=1 / 5)
        nn.init.normal_(self.short_conv_v.weight, std=1 / 5)
        self.norm = RMSNorm(self.d)
        self.output_gate = OutputGate(dim, heads, self.d, bottleneck=64, init='random')
        self.to_out_linear = nn.Linear(dim, dim, bias=False)

    def features(self, qkv, grid):
        h, w = grid
        b = qkv[0].shape[0]
        features = [F.silu(qkv[0].float())]
        for raw, conv in zip(qkv[1:], (self.short_conv_k, self.short_conv_v)):
            image = raw.float().reshape(b, h * w, -1).transpose(1, 2).reshape(b, -1, h, w)
            features.append(F.silu(conv(image).flatten(2).transpose(1, 2).reshape_as(raw)))
        q, k, v = features
        return F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v

    def forward(self, x, qkv, grid):
        q, k, v = self.features(qkv, grid)
        beta = torch.sigmoid(self.beta_proj(x.float()))
        out = joint_memory(q, k, v, beta, grid)
        b, n, c = x.shape
        out = self.norm(out) * self.output_gate(x.float().reshape(-1, c)).reshape(b, n, self.heads, self.d)
        return self.to_out_linear(out.reshape(b, n, c))


def region_indices(grid, size, halo, device):
    h, w = map(int, grid)
    if min(h, w, size) < 1 or halo < 0:
        raise ValueError('Invalid grid, region size or halo')
    row, col = torch.meshgrid(torch.arange((h + size - 1) // size, device=device) * size,
                              torch.arange((w + size - 1) // size, device=device) * size, indexing='ij')
    def indices(border):
        delta = torch.arange(-border, size + border, device=device)
        dy, dx = torch.meshgrid(delta, delta, indexing='ij')
        yy, xx = row.flatten()[:, None] + dy.flatten(), col.flatten()[:, None] + dx.flatten()
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        return yy.clamp(0, h-1) * w + xx.clamp(0, w-1), valid
    return (*indices(0), *indices(halo))


class HybridAttention(nn.Module):
    """Image-only attention: shared raw QKV, local joint SDPA and remote QDM."""
    def __init__(self, orig, region_size=8, halo=2):
        super().__init__()
        self.orig, self.region_size, self.halo = orig, region_size, halo
        self.remote = QuadrantDeltaMemory(orig.dim, orig.num_heads).float()
        self.softmax_gate = OutputGate(orig.dim, orig.num_heads, init_value=.99).float()
        self.teacher, self.alignment_callback, self.grid = False, None, None
        self.region_chunk = 16

    def raw_features(self, x, y):
        b, n, c = x.shape
        qkv = self.orig.qkv_x(x).reshape(b, n, 3, self.orig.num_heads, self.orig.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.orig.kv_y(y).reshape(b, -1, 2, self.orig.num_heads, self.orig.head_dim).permute(2, 0, 3, 1, 4)
        return qkv, kv

    def softmax_features(self, x, y, pos, raw=None):
        qkv, kv = self.raw_features(x, y) if raw is None else raw
        q, k, v = qkv.unbind(0)
        q, k = apply_rotary_emb(self.orig.q_norm(q.contiguous()), self.orig.k_norm(k.contiguous()), pos)
        ky = self.orig.k_norm(kv[0].contiguous())
        return q, k, v, ky, kv[1]

    def local_forward(self, x, y, pos, grid, bypass_gate=False, raw=None):
        b, n, c = x.shape
        if grid[0] * grid[1] != n:
            raise ValueError('Image token count does not match explicit grid')
        heads, d = self.orig.num_heads, self.orig.head_dim
        q, k, v, ky, vy = self.softmax_features(x, y, pos, raw)
        qi, qvalid, ki, kvalid = region_indices(grid, self.region_size, self.halo, x.device)
        regions, queries = qi.shape
        length, keys = y.shape[1], ki.shape[1]
        pieces = []
        for start in range(0, regions, self.region_chunk):
            stop = min(regions, start + self.region_chunk)
            count = stop - start
            qq = q[:, :, qi[start:stop]].permute(0, 2, 1, 3, 4).reshape(b * count, heads, queries, d)
            kk = k[:, :, ki[start:stop]].permute(0, 2, 1, 3, 4)
            vv = v[:, :, ki[start:stop]].permute(0, 2, 1, 3, 4)
            textk = ky[:, None].expand(b, count, heads, length, d)
            textv = vy[:, None].expand(b, count, heads, length, d)
            kk = torch.cat((kk, textk), 3).reshape(b * count, heads, keys + length, d)
            vv = torch.cat((vv, textv), 3).reshape_as(kk)
            valid = torch.cat((kvalid[start:stop], torch.ones(count, length, device=x.device, dtype=torch.bool)), -1)
            mask = valid[None, :, None, None].expand(b, count, 1, 1, keys + length).reshape(b * count, 1, 1, keys + length)
            out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, dropout_p=0.)
            pieces.append(out.reshape(b, count, heads, queries, d).permute(0, 1, 3, 2, 4))
        packed = torch.cat(pieces, 1).reshape(b, regions * queries, heads, d)
        image = packed[:, qvalid.flatten()][:, torch.argsort(qi[qvalid])]
        if not bypass_gate:
            image = image * self.softmax_gate(x.float().reshape(-1, c)).reshape(b, n, heads, 1).to(image.dtype)
        return self.orig.proj_drop(self.orig.proj(image.reshape(b, n, c)))

    def teacher_forward(self, x, y, pos):
        adapters = [m for m in self.orig.modules() if isinstance(m, LoRALinear)]
        previous = [m.adapters_enabled for m in adapters]
        try:
            for m in adapters:
                m.adapters_enabled = False
            return self.orig(x, y, pos)
        finally:
            for m, enabled in zip(adapters, previous):
                m.adapters_enabled = enabled

    def student_forward(self, x, y, pos, grid):
        raw = self.raw_features(x, y)
        local = self.local_forward(x, y, pos, grid, raw=raw)
        # QDM consumes RAW image Q/K/V, never RoPE or text state.
        remote = self.remote(x, tuple(z.transpose(1, 2) for z in raw[0].unbind(0)), grid)
        return local + remote.to(local.dtype)

    def forward(self, x, y, pos):
        if self.grid is None:
            raise RuntimeError('Hybrid attention requires the explicit image grid')
        if self.teacher:
            with torch.no_grad():
                target = self.teacher_forward(x, y, pos)
            if self.alignment_callback is not None:
                with torch.enable_grad():
                    self.alignment_callback(self, x.detach(), y.detach(), pos, target.detach(), self.grid)
            return target
        return self.student_forward(x, y, pos, self.grid)


class DeCoQDM(PixNerDiT):
    def __init__(self, *args, checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpointing = checkpointing
        self._teacher = False
        self._alignment_callback = None

    def hybrid_modules(self):
        return [(name, m) for name, m in self.named_modules() if isinstance(m, HybridAttention)]

    @contextmanager
    def execution(self, teacher=False, alignment_callback=None):
        previous = self._teacher, self._alignment_callback
        self._teacher, self._alignment_callback = teacher, alignment_callback
        try:
            yield self
        finally:
            self._teacher, self._alignment_callback = previous

    def _run_block(self, block, s, y, condition, pos, grid, teacher, callback):
        m = block.attn
        if not isinstance(m, HybridAttention):
            # Native retained layers use their original path in both modes.
            # Keep autograd enabled for student gradients through frozen layers.
            return block(s, y, condition, pos)
        previous = m.teacher, m.alignment_callback, m.grid
        m.teacher, m.alignment_callback, m.grid = teacher, callback, grid
        try:
            return block(s, y, condition, pos)
        finally:
            m.teacher, m.alignment_callback, m.grid = previous

    def forward(self, x, t, y):
        b, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('Image dimensions must be divisible by patch size')
        if y.shape[1] != self.txt_max_length:
            raise ValueError(f'DeCo requires exactly {self.txt_max_length} text tokens')
        grid = height // self.patch_size, width // self.patch_size
        dtype = self.s_embedder.proj.weight.dtype
        # The official autocast path keeps raw pixels and timesteps in FP32.
        # Casting t before sinusoidal embedding would round the timestep itself.
        # Explicit casts are only a fallback for a BF16 backbone without autocast.
        if not torch.is_autocast_enabled(x.device.type):
            x, t, y = x.to(dtype), t.to(dtype), y.to(dtype)
        # Same operations/order as the official forward; rectangular grid is explicit.
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        pos = self.fetch_pos(*grid, x.device)
        t = self.t_embedder(t.view(-1)).view(b, -1, self.hidden_size)
        y = self.y_embedder(y).view(b, -1, self.hidden_size) + self.y_pos_embedding.to(y.dtype)
        condition = F.silu(t)
        for block in self.text_refine_blocks:
            y = block(y, condition)
        s = self.s_embedder(x)
        teacher, callback = self._teacher, self._alignment_callback
        for block in self.blocks:
            if self.checkpointing and not teacher and torch.is_grad_enabled():
                # Capture execution state and this exact block; contexts may end before
                # backward recomputes. Checkpointing must never switch to the teacher.
                def run(a, bb, cc, pp, block=block, grid=grid, teacher=teacher, callback=callback):
                    return self._run_block(block, a, bb, cc, pp, grid, teacher, callback)
                s = checkpoint(run, s, y, condition, pos, use_reentrant=False)
            else:
                s = self._run_block(block, s, y, condition, pos, grid, teacher, callback)
        s = F.silu(t + s)
        batch, length, _ = s.shape
        x = x.reshape(batch * length, self.in_channels, self.patch_size ** 2).transpose(1, 2)
        x = self.x_embedder(x)
        x = self.dec_net(x, s.reshape(batch * length, self.hidden_size))
        x = x.transpose(1, 2).reshape(batch, length, -1)
        return F.fold(x.transpose(1, 2).contiguous(), (height, width), kernel_size=self.patch_size, stride=self.patch_size)


def configure_stage(model, stage, rank=64, alpha=64):
    if stage not in {'a1', 'a2', 'b'}:
        raise ValueError(f'Unknown training stage: {stage}')
    model.requires_grad_(False)
    for _, m in model.hybrid_modules():
        m.remote.requires_grad_(True)
        m.softmax_gate.requires_grad_(stage == 'b')
        if stage == 'b':
            for name, segments in [('qkv_x', 3), ('kv_y', 2), ('proj', 1)]:
                old = getattr(m.orig, name)
                if isinstance(old, LoRALinear) and (old.rank != rank or old.alpha != alpha):
                    # Reconfiguring an untrained model for CPU/test rank is explicit;
                    # never silently discard a trained delta.
                    if any(torch.count_nonzero(p) for p in old.B):
                        raise ValueError('Cannot change rank/alpha of a nonzero LoRA')
                    old = old.base
                if not isinstance(old, LoRALinear):
                    old = LoRALinear(old, rank, alpha, segments)
                    setattr(m.orig, name, old)
                old.A.requires_grad_(True)
                old.B.requires_grad_(True)
    model.stage = stage
    return model


def optimizer_groups(model, stage, lr=1e-4):
    small, lora = set(), set()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora.update(id(p) for seq in (m.A, m.B) for p in seq)
        if isinstance(m, OutputGate) and m.down is not None:
            small.update(id(p) for p in m.up.parameters())
        elif isinstance(m, (RMSNorm, nn.Conv2d)):
            small.update(id(p) for p in m.parameters())
    groups = {}
    for p in model.parameters():
        if p.requires_grad:
            key = 'lora' if id(p) in lora else 'small' if id(p) in small else 'big'
            groups.setdefault(key, []).append(p)
    scales = {'lora': 1., 'big': .4 if stage == 'b' else 1., 'small': 2. if stage == 'b' else 5.}
    return [dict(params=params, lr=lr * scales[name], lr_scale=scales[name],
                 group_name=name, weight_decay=0., eps=2e-9 if name == 'small' else 1e-8)
            for name, params in groups.items()]


def _delta_names(model):
    names = set()
    for name, m in model.hybrid_modules():
        for part in ('remote', 'softmax_gate'):
            names.update(f'{name}.{part}.{key}' for key in getattr(m, part).state_dict())
        for part in ('qkv_x', 'kv_y', 'proj'):
            layer = getattr(m.orig, part)
            if isinstance(layer, LoRALinear):
                names.update(f'{name}.orig.{part}.{letter}.{i}' for letter in ('A', 'B')
                             for i in range(layer.segments))
    return names


def delta_state_dict(model):
    names = _delta_names(model)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def load_delta_state_dict(model, state, strict=True):
    expected, found = _delta_names(model), set(state)
    if strict and expected != found:
        raise RuntimeError(f'Delta state keys mismatch; missing={sorted(expected-found)}, unexpected={sorted(found-expected)}')
    current = model.state_dict()
    for name in expected & found:
        if current[name].shape != state[name].shape:
            raise RuntimeError(f'Delta shape mismatch: {name}: {state[name].shape} != {current[name].shape}')
    model.load_state_dict({name: state[name] for name in expected & found}, strict=False)
    return model


def normalize_full_attention_layers(value, count=16):
    if value is None or value == '':
        return []
    if isinstance(value, str):
        value = [int(part.strip()) for part in value.split(',')]
    layers = list(value)
    if (any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < count for i in layers)
            or len(layers) != len(set(layers))):
        raise ValueError(f'Full Attention indices must be unique integers in [0, {count-1}]')
    return sorted(layers)


def build_model(weights_root, stage='a1', region_size=8, halo=2, device='cpu', checkpointing=False, tiny=False,
                full_attention_layers=None):
    kwargs = dict(in_channels=3, patch_size=16, num_groups=24, hidden_size=1536,
                  txt_embed_dim=2048, txt_max_length=128, num_text_blocks=4,
                  decoder_hidden_size=32, num_encoder_blocks=16, num_decoder_blocks=3)
    if tiny:
        kwargs.update(patch_size=2, num_groups=2, hidden_size=32, txt_embed_dim=16,
                      txt_max_length=8, num_text_blocks=1, decoder_hidden_size=8,
                      num_encoder_blocks=2, num_decoder_blocks=1)
    retained = normalize_full_attention_layers(full_attention_layers, kwargs['num_encoder_blocks'])
    # Construct the original backbone in BF16 directly: native initialization should
    # not transiently allocate a second FP32 copy of the multi-billion-param model.
    old_dtype = torch.get_default_dtype()
    try:
        if not tiny:
            torch.set_default_dtype(torch.bfloat16)
        model = DeCoQDM(**kwargs, checkpointing=checkpointing)
    finally:
        torch.set_default_dtype(old_dtype)
    if not tiny:
        path = Path(weights_root) / 't2i_DeCo.ckpt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload)
        prefix = 'ema_denoiser.'
        official = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not official:
            raise RuntimeError(f'Checkpoint has no {prefix} weights: {path}')
        model.load_state_dict(official, strict=True)
        del payload, state, official
    model.requires_grad_(False)
    model.full_attention_layers = retained
    model.region_size, model.halo = region_size, halo
    for index, block in enumerate(model.blocks):
        if index not in retained:
            block.attn = HybridAttention(block.attn, region_size=region_size, halo=halo)
    model.to(device=device)
    configure_stage(model, stage)
    return model
