"""FP32 adapters over immutable native BF16 projections."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=64):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank, self.alpha = int(rank), float(alpha)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.adapters_enabled = True
        self.base.requires_grad_(False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if self.adapters_enabled:
            with torch.autocast(device_type=x.device.type, enabled=False):
                delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
            out = out + (delta * (self.alpha / self.rank)).to(out.dtype)
        return out


def base_linear(module, x):
    return module.base(x) if isinstance(module, LoRALinear) else module(x)
