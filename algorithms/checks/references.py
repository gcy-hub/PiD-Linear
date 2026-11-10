"""Small independent numeric oracles; never used by production operators.

CLEAR visibility/pooling follows Huage001/CLEAR ddca131 (Apache-2.0).
QDM uses the fixed-four ridge equation from the pre-extraction implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DenseClear(nn.Module):
    def __init__(self, window_size=16, down_factor=1, window_shape='circle', kernel_size=11):
        super().__init__()
        self.radius, self.down = window_size, down_factor
        self.shape, self.kernel = window_shape, kernel_size
        self.register_parameter('spatial_weight', nn.Parameter(torch.ones(1,1,1,down_factor,1,down_factor,1) / down_factor**2) if down_factor > 1 else None)

    def forward(self, q, k, v, layout):
        pieces = []
        offset = 0
        for h, w, t in layout.samples:
            n = h*w+t
            image = t if layout.text_first else 0
            kk, vv = k[:,:,offset:offset+n], v[:,:,offset:offset+n]
            extra = 0
            if self.down > 1:
                pooled = []
                for z in (kk, vv):
                    cells = []
                    for y in range(0, h, self.down):
                        for x in range(0, w, self.down):
                            ids = [image+(y+dy)*w+x+dx for dy in range(self.down) for dx in range(self.down)]
                            cells.append((z[:,:,ids] * self.spatial_weight.flatten()[None,None,:,None]).sum(2))
                    pooled.append(torch.stack(cells, 2))
                kk, vv = torch.cat((kk,pooled[0]),2), torch.cat((vv,pooled[1]),2)
                extra = (h//self.down)*(w//self.down)
            mask = torch.zeros(n, n+extra, dtype=torch.bool, device=q.device)
            for i in range(n):
                qi = i-image
                for j in range(n+extra):
                    kj = j-image
                    if not 0 <= qi < h*w:
                        allowed = j < n
                    elif j >= n or not 0 <= kj < h*w:
                        allowed = True
                    else:
                        dy, dx = abs(qi//w-kj//w), abs(qi%w-kj%w)
                        allowed = dy*dy+dx*dx < self.radius**2 if self.shape == 'circle' else max(dy,dx) <= self.kernel//2
                    mask[i,j] = allowed
            scores = q[:,:,offset:offset+n] @ kk.transpose(-1,-2) / q.shape[-1]**.5
            pieces.append(scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vv)
            offset += n
        return torch.cat(pieces, 2)


def dense_quadrant_memory(module, x, qkv, grid):
    """Evaluate each quadrant separately with a dense solve, without chunking."""
    h, w = grid
    n, c = x.shape
    heads, d = module.heads, module.d
    q = F.normalize(F.silu(qkv[0]), dim=-1)
    features = []
    for raw, conv in zip(qkv[1:], (module.short_conv_k, module.short_conv_v)):
        z = F.conv2d(raw.reshape(n,c).T.reshape(1,c,h,w), conv.weight, padding=2, groups=c)
        features.append(F.silu(z.flatten(2).transpose(1,2).reshape(n,heads,d)))
    k, v = F.normalize(features[0], dim=-1), features[1]
    beta = F.linear(x, module.beta_proj.weight).sigmoid()
    outputs = []
    for i in range(n):
        ids = [j for j in range(n) if (j//w*2//h, j%w*2//w) == (i//w*2//h, i%w*2//w)]
        kh, vh = k[ids].transpose(0,1), v[ids].transpose(0,1)
        weighted = kh * beta[ids].T[:,:,None]
        a = torch.eye(d, device=x.device) + kh.transpose(-1,-2) @ weighted
        b = vh.transpose(-1,-2) @ weighted
        outputs.append((b @ torch.linalg.solve(a, q[i,:,:,None])).squeeze(-1))
    out = torch.stack(outputs)
    return module.to_out_linear((module.norm(out) * module.output_gate(x)).reshape(n,c))


class StockKDAReference(nn.Module):
    """Stock FLA with explicit rectangular index order, independent of scan()."""
    def __init__(self, hidden, heads, layer_index):
        super().__init__()
        from fla.layers.kda import KimiDeltaAttention
        self.direction = layer_index % 4
        self.layer = KimiDeltaAttention(hidden_size=hidden, num_heads=heads,
                                       head_dim=hidden//heads, layer_idx=layer_index)

    def forward(self, x, grid):
        h, w = grid
        ids = list(range(h*w)) if self.direction < 2 else [y*w+z for z in range(w) for y in range(h)]
        if self.direction % 2:
            ids.reverse()
        order = torch.tensor(ids, device=x.device)
        from torch.func import functional_call
        weights = {n: p if n in ('A_log','dt_bias') else p.to(x.dtype) for n,p in self.layer.named_parameters()}
        out = functional_call(self.layer, weights, (x[:,order],), dict(past_key_values=None,use_cache=False))[0]
        return out[:,order.argsort()]
