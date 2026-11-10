"""FLUX.2 native double/single attention adapters; fused FFN stays native."""
import torch
from torch import nn
from attention_framework.contracts import AttentionInput, AttentionLayout
from .attention import AttentionSite, install_projection
from .vendor import native_module


class Flux2Attention(AttentionSite):
    def __init__(self, original, operator, layer, config):
        super().__init__(original, operator, layer)
        self.single = layer.group == 'single'
        h = original.inner_dim
        if self.single:
            install_projection(original, 'to_qkv_mlp_proj', config, rows=3*h)
            install_projection(original, 'to_out', config, columns=h)
        else:
            for name in ('to_q', 'to_k', 'to_v'):
                install_projection(original, name, config)
            install_projection(original.to_out, '0', config)

    def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None,
                attention_mask=None, kv_cache=None, **kwargs):
        if attention_mask is not None or kv_cache is not None:
            raise ValueError('FLUX2 replacement expects unpadded text-to-image tokens without KV cache')
        a = self.orig
        t = hidden_states.shape[1] - self.grid[0]*self.grid[1] if self.single else encoder_hidden_states.shape[1]
        layout = AttentionLayout(((*self.grid, t),), text_first=True)
        def make_inputs(x, text):
            if self.single:
                fused = a.to_qkv_mlp_proj(x)
                qkv = fused[..., :3*a.inner_dim]
                q, k, v = [z.unflatten(-1, (a.heads, -1)) for z in qkv.chunk(3, -1)]
            else:
                q, k, v = [p(x).unflatten(-1, (a.heads, -1)) for p in (a.to_q, a.to_k, a.to_v)]
            q, k = a.norm_q(q), a.norm_k(k)
            hidden = x
            if not self.single:
                tq, tk, tv = [p(text).unflatten(-1, (a.heads, -1)) for p in (a.add_q_proj, a.add_k_proj, a.add_v_proj)]
                q, k, v = [torch.cat(pair, 1) for pair in ((a.norm_added_q(tq), q), (a.norm_added_k(tk), k), (tv, v))]
                hidden = torch.cat((text, x), 1)
            qp, kp = q, k
            if image_rotary_emb is not None:
                rotate = native_module('flux2').apply_rotary_emb
                qp, kp = [rotate(z, image_rotary_emb, sequence_dim=1) for z in (q, k)]
            return AttentionInput(hidden, q, k, v, qp, kp, layout, self.layer)
        self.student_inputs = lambda: make_inputs(hidden_states.detach(), None if encoder_hidden_states is None else encoder_hidden_states.detach())
        attention_pre = None
        if self.single:
            def project(pre):
                nonlocal attention_pre
                if attention_pre is None:
                    attention_pre = pre
                return a.to_out.attention_output(pre)
        else:
            def project(pre):
                return torch.cat((a.to_add_out(pre[:, :t]), a.to_out[1](a.to_out[0](pre[:, t:]))), 1)
        post = self.attend(make_inputs(hidden_states, encoder_hidden_states), project)
        if self.single:
            fused = a.to_qkv_mlp_proj(hidden_states)
            mlp = a.mlp_act_fn(fused[..., 3*a.inner_dim:])
            if self.teacher:
                # Preserve the native fused GEMM and its single BF16 rounding for the teacher trunk.
                return a.to_out(torch.cat((attention_pre, mlp), -1))
            # O's attention columns were applied once above; retain frozen FFN columns/bias.
            return post + torch.nn.functional.linear(mlp, a.to_out.base.weight[:, a.inner_dim:], a.to_out.bias)
        return post[:, t:], post[:, :t]


def layer_groups(base):
    return {'double': base.transformer_blocks, 'single': base.single_transformer_blocks}


def install(base, specs, config, factory):
    groups = layer_groups(base)
    for spec in specs:
        block = groups[spec.group][spec.index]
        orig = block.attn
        operator = factory(config, hidden_size=orig.query_dim, num_heads=orig.heads, layer=spec)
        block.attn = Flux2Attention(orig, operator, spec, config)
