"""Native FLUX3 attention block with CLEAR projections and image KDA."""
import torch
from torch.nn import functional as F
from flux3_qdm.attention import QDMBlock, native
from flux2_qdm.diffsynth_model import AttentionProjection
from flux2_qdm.model import disable_lora
from .attention import ImageKDA


class Flux3KDABlock(QDMBlock):
    def __init__(self, orig, index, kernel):
        super().__init__(orig, attention_mode='local-only', local_attention='sliding', sliding_kernel_size=kernel)
        del self.softmax_gate
        self.bypass_softmax_gate = True
        self.collect_feature = index % 4 == 0
        self.layer_index = index
        self.kda = ImageKDA(orig.q_proj.out_features, orig.heads, index)
        for name in ('q_proj', 'k_proj', 'v_proj', 'attn_out'):
            setattr(orig, name, AttentionProjection(getattr(orig, name)))

    def forward(self, x, freqs, bounds, mod, *, samples=None, teacher_mode=False,
                alignment_callback=None, return_attention_feature=False):
        shift, scale, gate = mod
        z = native.layernorm_modulation(x, scale, shift)
        from contextlib import nullcontext
        with disable_lora(self.orig) if teacher_mode else nullcontext():
            raw = self.raw_qkv(z)
            q, k = self.orig.norm(raw[0], raw[1], freqs)
            v = raw[2].to(q.dtype)
            if teacher_mode:
                feature = native.packed_attention(q, k, v, bounds)
            else:
                local = self.local_attention(z, q, k, v, samples).flatten(1)
                chunks = []
                for s in samples:
                    a, b = s.image_start, s.image_end
                    image = self.kda(z[None, a:b], s.grid)[0]
                    chunks.extend((image, image.new_zeros(s.text_end-s.text_start, image.shape[-1])))
                feature = local + torch.cat(chunks)
            attn = self.orig.attn_out(feature)
        gate_in, up = self.orig.mlp_in(z).chunk(2, dim=-1)
        result = x + gate * (attn + self.orig.mlp_out(F.silu(gate_in) * up))
        return (result, feature) if return_attention_feature else result
