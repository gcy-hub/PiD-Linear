"""Square sliding local attention with an independently projected QDM branch."""
import torch
from torch import nn
from attention_framework.contracts import AttentionResult, RoleParameter
from .local import ClearAttention, sample_slices
from .qdm import QuadrantMemory
from .vendor.gates import OutputGate


class SlidingQDM(nn.Module):
    def __init__(self, config, hidden_size, num_heads, layer):
        super().__init__()
        self.layer_spec = layer
        self.heads = num_heads
        cfg = config.algorithm
        self.local = ClearAttention(window_shape='square', kernel_size=cfg.local.window_size)
        self.remote = QuadrantMemory(hidden_size, num_heads, cfg.memory.chunk_size)
        self.softmax_gate = OutputGate(hidden_size, num_heads, init_value=.99)
        self.region_size, self.halo = cfg.memory.region_size, cfg.memory.halo

    def local_attention(self, inputs):
        return self.local(inputs.q_pos.transpose(1, 2), inputs.k_pos.transpose(1, 2),
                          inputs.v.transpose(1, 2), inputs.layout).transpose(1, 2)

    def forward(self, inputs):
        local = self.local_attention(inputs)
        with torch.autocast(device_type=inputs.hidden.device.type, enabled=False):
            gate = self.softmax_gate(inputs.hidden.float()).reshape(*inputs.hidden.shape[:2], self.heads, 1)
        local = (local * gate.to(local.dtype)).flatten(2)
        pieces = []
        for grid, whole, image, text in sample_slices(inputs.layout):
            memory = torch.stack([self.remote(inputs.hidden[b, image],
                (inputs.q_raw[b, image], inputs.k_raw[b, image], inputs.v[b, image]), grid)
                for b in range(inputs.hidden.shape[0])])
            zeros = memory.new_zeros(memory.shape[0], text.stop-text.start, memory.shape[-1])
            pieces.extend((zeros, memory) if inputs.layout.text_first else (memory, zeros))
        return AttentionResult(local, torch.cat(pieces, 1))

    def role_parameters(self):
        return [RoleParameter(n, p, 'local_gate' if n.startswith('softmax_gate.') else 'branch', self.layer_spec.name)
                for n, p in self.named_parameters()]
