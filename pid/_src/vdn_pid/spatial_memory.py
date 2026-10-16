"""Global multi-bank VDA research prototype; requires its own recovery training.

Image banks start at zero, have no raster scan/decay, and include their own writes.
Text still reaches every image query through the unchanged local Softmax branch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def fixed_routes(grid, banks, device):
    side = math.isqrt(banks)
    if side * side != banks:
        raise ValueError("Fixed spatial routing requires a square number of banks")
    h, w = grid
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    labels = (y * side // h) * side + x * side // w
    return F.one_hot(labels.flatten(), banks).float()


def joint_memory(q, k, v, beta, write, read=None, mask=None, sparse=False, chunk_size=4096):
    """Q/K/V [B,N,H,D], beta [B,N,H], routes [B,N,banks]; FP32 exact normal equations.

    Sparse mode requires identical one-hot spatial routes across the batch and tied
    read/write routes. It gathers each partition once instead of dense zero multiplies.
    No initial state or persistent cache exists; states cannot leak across calls.
    """
    read = write if read is None else read
    b, n, heads, dk = k.shape
    banks, dv = write.shape[-1], v.shape[-1]
    if sparse and read is not write:
        raise ValueError("Sparse routing requires tied read/write routes")
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        if mask is not None:
            beta = beta * mask[:, :, None]
        out = torch.zeros(b, n, heads, dv, device=q.device)
        states = []
        eye = torch.eye(dk, device=q.device)
        for bank in range(banks):
            indices = torch.where(write[0, :, bank] > 0)[0] if sparse else torch.arange(n, device=q.device)
            A = torch.zeros(b, heads, dk, dk, device=q.device)
            B = torch.zeros(b, heads, dv, dk, device=q.device)
            for start in range(0, len(indices), chunk_size):
                ix = indices[start : start + chunk_size]
                kk, vv = k[:, ix].transpose(1, 2), v[:, ix].transpose(1, 2)
                weight = (beta[:, ix] * write[:, ix, bank, None]).transpose(1, 2)
                weighted = kk * weight[..., None]
                A = A + kk.transpose(-1, -2) @ weighted
                B = B + vv.transpose(-1, -2) @ weighted
            chol = torch.linalg.cholesky(eye + A)
            state = torch.cholesky_solve(B.transpose(-1, -2), chol).transpose(-1, -2)
            states.append(state)
            for start in range(0, len(indices), chunk_size):
                ix = indices[start : start + chunk_size]
                result = torch.einsum("bhvk,bnhk->bnhv", state, q[:, ix])
                result = result * read[:, ix, bank, None, None]
                if torch.is_grad_enabled():
                    out = out.index_add(1, ix, result)
                else:
                    # Avoid copying the entire image output for every token chunk.
                    out.index_add_(1, ix, result)
        if mask is not None:
            out = out * mask[:, :, None, None]
        return out, torch.stack(states, dim=1)


class SpatialDeltaMemory(nn.Module):
    """Keep trained VDA feature maps/output calibration; replace only its state mechanism.

    fixed4 is a spatial partition baseline. dense4 has random untrained content+position
    write/read routers and measures their real overhead, not conflict-aware quality.
    """

    def __init__(self, base, mode="fixed", banks=4):
        super().__init__()
        self.base, self.mode, self.banks = base, mode, banks
        dim = base.heads * base.d
        if mode == "dense":
            self.write_router = nn.Linear(dim + 4, banks, bias=True)
            self.read_router = nn.Linear(dim + 4, banks, bias=True)
            for router in (self.write_router, self.read_router):
                nn.init.normal_(router.weight, std=0.001)
                nn.init.zeros_(router.bias)
        elif mode != "fixed":
            raise ValueError(mode)

    def forward(self, x, y, qkv, text_qkv, grid, text_mask, inference=False, order="row"):
        q, k, v = self.base.features(qkv, grid)
        beta = torch.sigmoid(self.base.beta_proj(x))
        b, n, c = x.shape
        if self.mode == "fixed":
            routes = fixed_routes(grid, self.banks, x.device)[None].expand(b, -1, -1)
            out, _ = joint_memory(q, k, v, beta, routes, sparse=True)
        else:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, grid[0], device=x.device),
                torch.linspace(-1, 1, grid[1], device=x.device),
                indexing="ij",
            )
            pos = torch.stack((xx, yy, torch.sin(math.pi * xx), torch.sin(math.pi * yy)), -1)
            pos = pos.reshape(1, n, 4).expand(b, -1, -1).to(x.dtype)
            write = self.write_router(torch.cat((x, pos), -1)).float().softmax(-1)
            read = self.read_router(torch.cat((q.reshape(b, n, c), pos), -1)).float().softmax(-1)
            out, _ = joint_memory(q, k, v, beta, write, read)
        out = out.to(q.dtype)
        out = self.base.norm(out) * self.base.output_gate(x.reshape(-1, c)).reshape_as(out)
        return self.base.to_out_linear(out.reshape(b, n, c))


def replace_memory(net, mode="fixed", banks=4):
    from .attention import JointVDN

    for block in net.patch_blocks:
        if not isinstance(block.attn, JointVDN):
            raise TypeError("Construct the VDN student before replacing its memory")
        old = block.attn.linear_attention
        block.attn.linear_attention = SpatialDeltaMemory(old, mode, banks).to(next(old.parameters()).device)
    return net
