"""Standard FLA KDA with a four-layer spatial scan direction cycle."""
import torch
from torch import nn
from torch.func import functional_call


def scan(x, grid, direction, inverse=False):
    """Permute [B,H*W,...], preserving explicit rectangular grid coordinates."""
    h, w = grid
    if x.shape[1] != h * w or direction not in range(4):
        raise ValueError('Invalid image grid or scan direction')
    if inverse and direction % 2:
        x = x.flip(1)
    if direction >= 2:
        rows, cols = (w, h) if inverse else (h, w)
        x = x.reshape(x.shape[0], rows, cols, *x.shape[2:]).transpose(1, 2).flatten(1, 2)
    if not inverse and direction % 2:
        x = x.flip(1)
    return x.contiguous()


class ImageKDA(nn.Module):
    def __init__(self, hidden, heads, layer_index, expand_v=1.0):
        from fla.layers.kda import KimiDeltaAttention
        super().__init__()
        self.direction = layer_index % 4
        self.layer = KimiDeltaAttention(hidden_size=hidden, num_heads=heads,
                                        head_dim=hidden // heads, layer_idx=layer_index, expand_v=expand_v)

    def inference_weights(self, dtype=torch.bfloat16):
        for name, param in self.layer.named_parameters():
            if name not in ('A_log', 'dt_bias'):
                param.data = param.data.to(dtype)

    def forward(self, x, grid):
        ordered = scan(x, grid, self.direction)
        kwargs = dict(past_key_values=None, use_cache=False)
        if self.layer.q_proj.weight.dtype != x.dtype:
            # Preserve CLEAR FP32 masters while executing the stock BF16 layer.
            weights = {name: p if name in ('A_log', 'dt_bias') else p.to(x.dtype)
                       for name, p in self.layer.named_parameters()}
            out = functional_call(self.layer, weights, (ordered,), kwargs)[0]
        else:
            out = self.layer(ordered, **kwargs)[0]
        return scan(out, grid, self.direction, inverse=True)


from attention_framework.contracts import AttentionResult, RoleParameter
from .local import ClearAttention, sample_slices


class CrossScanKDA(nn.Module):
    def __init__(self, config, hidden_size, num_heads, layer):
        super().__init__()
        self.layer_spec = layer
        self.local = ClearAttention(window_shape='square', kernel_size=config.algorithm.local.window_size)
        self.kda = ImageKDA(hidden_size, num_heads, layer.ordinal, config.algorithm.kda.expand_v)

    def forward(self, inputs):
        local = self.local(inputs.q_pos.transpose(1, 2), inputs.k_pos.transpose(1, 2),
                           inputs.v.transpose(1, 2), inputs.layout).transpose(1, 2).flatten(2)
        pieces = []
        for grid, whole, image, text in sample_slices(inputs.layout):
            out = self.kda(inputs.hidden[:, image], grid)
            zeros = out.new_zeros(out.shape[0], text.stop-text.start, out.shape[-1])
            pieces.extend((zeros, out) if inputs.layout.text_first else (out, zeros))
        return AttentionResult(local + torch.cat(pieces, 1))

    def role_parameters(self):
        return [RoleParameter(n, p, 'branch', self.layer_spec.name) for n, p in self.named_parameters()]
