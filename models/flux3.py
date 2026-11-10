"""FLUX.3 image/joint replacement preserving native modulation and FFN."""
import torch
from torch.nn import functional as F
from attention_framework.contracts import AttentionInput, AttentionLayout
from .attention import AttentionSite, install_projection
from .vendor import native_module


class Flux3Block(AttentionSite):
    def __init__(self, original, operator, layer, config):
        super().__init__(original, operator, layer)
        for name in ('q_proj', 'k_proj', 'v_proj', 'attn_out'):
            install_projection(original, name, config)

    def forward(self, x, freqs, bounds, mod):
        native = native_module('flux3')
        a = self.orig
        shift, scale, gate = mod
        z = native.layernorm_modulation(x, scale, shift)
        samples = self.samples
        if self.layer.group == 'early_image':
            samples = tuple((h, w, 0) for h, w, t in samples)
        layout = AttentionLayout(samples, text_first=False)
        def make_inputs(hidden):
            split = lambda p: p(hidden).reshape(1, hidden.shape[0], a.heads, -1).transpose(1, 2)
            q0, k0, v = [split(p) for p in (a.q_proj, a.k_proj, a.v_proj)]
            qp, kp = a.norm(q0, k0, freqs)
            q = (q0.float() * (q0.float().square().mean(-1, keepdim=True)+1e-6).rsqrt() * a.norm.query_norm.scale.float()).to(q0.dtype)
            k = (k0.float() * (k0.float().square().mean(-1, keepdim=True)+1e-6).rsqrt() * a.norm.key_norm.scale.float()).to(k0.dtype)
            return AttentionInput(hidden[None], q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2).to(qp.dtype),
                                  qp.transpose(1, 2), kp.transpose(1, 2), layout, self.layer)
        self.student_inputs = lambda: make_inputs(z.detach())
        attn = self.attend(make_inputs(z), a.attn_out)[0]
        gate_in, up = a.mlp_in(z).chunk(2, -1)
        return x + gate * (attn + a.mlp_out(F.silu(gate_in) * up))


def layer_groups(base):
    return {'early_image': base.content_mode_blocks['image'], 'joint': base.single_blocks}


def install(base, specs, config, factory):
    groups = layer_groups(base)
    for spec in specs:
        original = groups[spec.group][spec.index]
        op = factory(config, hidden_size=original.q_proj.in_features, num_heads=original.heads, layer=spec)
        groups[spec.group][spec.index] = Flux3Block(original, op, spec, config)
