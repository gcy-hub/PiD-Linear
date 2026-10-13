"""Zero-delta, independently switchable Q/K/V adapters; structural LR classes."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import JointVDN
from .vendor.gates import OutputGate
from .vendor.layers import FrameKDAAlpha
from .vendor.rms_norm import RMSNorm


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=64, segments=1):
        super().__init__()
        self.base = base
        self.adapters_enabled = True
        self.rank, self.alpha, self.segments = rank, alpha, segments
        if rank <= 0 or alpha <= 0 or base.out_features % segments:
            raise ValueError("Invalid LoRA rank/alpha/segments")
        self.A = nn.ParameterList(
            [
                nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
                for _ in range(segments)
            ]
        )
        self.B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(base.out_features // segments, rank, device=base.weight.device, dtype=torch.float32)
                )
                for _ in range(segments)
            ]
        )
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        y = self.base(x)
        if not self.adapters_enabled:
            return y
        # Explicit casts also support inference outside autocast.
        delta = [F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) for a, b in zip(self.A, self.B)]
        return y + torch.cat(delta, dim=-1) * (self.alpha / self.rank)


def configure_stage(net, stage, rank=64, alpha=64):
    net.requires_grad_(False)
    for block in net.patch_blocks:
        m = block.attn
        if not isinstance(m, JointVDN):
            raise ValueError("Expected PiD-VDN model")
        m.linear_attention.requires_grad_(True)
        m.softmax_gate_x.requires_grad_(stage == "b")
        m.softmax_gate_y.requires_grad_(stage == "b")
        if stage == "b":
            for name, segments in [("qkv_x", 3), ("qkv_y", 3), ("proj_x", 1), ("proj_y", 1)]:
                old = getattr(m.orig, name)
                if not isinstance(old, LoRALinear):
                    old = LoRALinear(old, rank, alpha, segments)
                    setattr(m.orig, name, old)
                old.A.requires_grad_(True)
                old.B.requires_grad_(True)
    if stage == "b":
        last = net.patch_blocks[-1].attn
        last.orig.qkv_y.A[0].requires_grad_(False)
        last.orig.qkv_y.B[0].requires_grad_(False)
        last.orig.proj_y.requires_grad_(False)
        last.softmax_gate_y.requires_grad_(False)
    elif stage not in {"a1", "a2"}:
        raise ValueError(stage)
    return net


def optimizer_groups(net, stage):
    small = set()
    lora = set()
    for m in net.modules():
        if isinstance(m, LoRALinear):
            lora.update(id(p) for seq in (m.A, m.B) for p in seq)
        if isinstance(m, FrameKDAAlpha):
            small.update(id(p) for p in [m.A_log, m.dt_bias, *m.up.parameters()])
        elif isinstance(m, OutputGate) and m.down is not None:
            small.update(id(p) for p in m.up.parameters())
        elif isinstance(m, (RMSNorm, nn.Conv2d)):
            small.update(id(p) for p in m.parameters())
    groups = {}
    for p in net.parameters():
        if not p.requires_grad:
            continue
        key = "lora" if id(p) in lora else "small" if id(p) in small else "big"
        groups.setdefault(key, []).append(p)
    scales = {"lora": 1.0, "big": 0.4 if stage == "b" else 1.0, "small": 2.0 if stage == "b" else 5.0}
    return [
        dict(
            params=params,
            lr=1e-4 * scales[key],
            lr_scale=scales[key],
            group_name=key,
            weight_decay=0.0,
            eps=2e-9 if key == "small" else 1e-8,
        )
        for key, params in groups.items()
    ]
