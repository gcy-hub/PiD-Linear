"""External patch-attention adaptation; PiD pixel blocks remain native."""
import torch
from attention_framework.contracts import AttentionInput, AttentionLayout
from .attention import AttentionSite, install_projection


class PidAttention(AttentionSite):
    is_linear_attention = True

    def __init__(self, original, operator, layer, config):
        super().__init__(original, operator, layer)
        for name in ('qkv_x', 'proj_x'):
            install_projection(original, name, config)

    def forward(self, x, y, pos_img, pos_txt=None, attn_mask=None, *, grid_size=None, text_valid_mask=None):
        from pid._src.networks.pixeldit_official import apply_rotary_emb
        a = self.orig
        if a._cp_group is not None:
            raise ValueError('PiD framework replacement does not support context parallelism')
        if attn_mask is not None:
            raise ValueError('PiD replacement requires a text_valid_mask rather than an attention mask')
        # PiD supplies padded embeddings; remove padding before the operator boundary.
        valid = torch.ones(y.shape[:2], device=y.device, dtype=torch.bool) if text_valid_mask is None else text_valid_mask[:, :y.shape[1]].bool()
        outputs_x, outputs_y, features = [], [], []
        for b in range(x.shape[0]):
            xx, yy = x[b:b+1], y[b:b+1, valid[b]]
            txtpos = pos_txt
            if txtpos is not None:
                txtpos = tuple(p[valid[b]] for p in txtpos) if isinstance(txtpos, tuple) else txtpos[valid[b]]
            layout = AttentionLayout(((*(grid_size or self.grid), yy.shape[1]),), text_first=True)
            def make_inputs(ix, iy):
                def qkv(value, projection):
                    return projection(value).reshape(1, value.shape[1], 3, a.num_heads, a.head_dim).unbind(2)
                q, k, v = qkv(ix, a.qkv_x)
                tq, tk, tv = qkv(iy, a.qkv_y)
                q, k = a.q_norm_x(q), a.k_norm_x(k)
                tq, tk = a.q_norm_y(tq), a.k_norm_y(tk)
                qp, kp = apply_rotary_emb(q, k, freqs_cis=pos_img)
                tqp, tkp = (tq, tk) if txtpos is None else apply_rotary_emb(tq, tk, freqs_cis=txtpos)
                cat = lambda t, i: torch.cat((t, i), 1)
                return AttentionInput(cat(iy, ix), cat(tq, q), cat(tk, k), cat(tv, v), cat(tqp, qp), cat(tkp, kp), layout, self.layer)
            self.student_inputs = lambda: make_inputs(xx.detach(), yy.detach())
            t = yy.shape[1]
            def project(pre):
                return torch.cat((a.proj_drop_y(a.proj_y(pre[:, :t])), a.proj_drop_x(a.proj_x(pre[:, t:]))), 1)
            post = self.attend(make_inputs(xx, yy), project)
            outputs_x.append(post[:, t:])
            text = y[b:b+1].new_zeros(y[b:b+1].shape)
            text[:, valid[b]] = post[:, :t]
            outputs_y.append(text)
            if self.capture:
                features.append(self.feature)
        if self.capture:
            self.feature = torch.cat(features)
        return torch.cat(outputs_x), torch.cat(outputs_y)

    def set_context_parallel_group(self, group):
        if group is not None:
            raise ValueError('PiD framework replacement does not support context parallelism')
        self.orig.set_context_parallel_group(group)


def layer_groups(base):
    return {'patch': base.patch_blocks}


def install(base, specs, config, factory):
    for spec in specs:
        block = base.patch_blocks[spec.index]
        original = block.attn
        op = factory(config, hidden_size=original.dim, num_heads=original.num_heads, layer=spec)
        block.attn = PidAttention(original, op, spec, config)
