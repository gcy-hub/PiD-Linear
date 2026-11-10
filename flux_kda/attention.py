"""Standard FLA KDA with a four-layer spatial scan direction cycle."""
import torch
from torch import nn
from torch.func import functional_call
from fla.layers.kda import KimiDeltaAttention


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
    def __init__(self, hidden, heads, layer_index):
        super().__init__()
        self.direction = layer_index % 4
        self.layer = KimiDeltaAttention(hidden_size=hidden, num_heads=heads,
                                        head_dim=hidden // heads, layer_idx=layer_index)

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
