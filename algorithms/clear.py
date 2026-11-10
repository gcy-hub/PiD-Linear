"""CLEAR circle attention and optional learned image K/V pooling."""
from torch import nn
from attention_framework.contracts import AttentionResult, RoleParameter
from .local import ClearAttention


class CLEAR(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.layer_spec = layer
        cfg = config.algorithm['clear']
        self.local = ClearAttention(cfg.radius, cfg.down_factor, cfg.train_grid)
        self.proportional_attention = cfg.proportional_attention

    def forward(self, inputs):
        out = self.local(inputs.q_pos.transpose(1, 2), inputs.k_pos.transpose(1, 2),
                         inputs.v.transpose(1, 2), inputs.layout, self.proportional_attention)
        return AttentionResult(out.transpose(1, 2).flatten(2))

    def role_parameters(self):
        return [RoleParameter(n, p, 'pooling', self.layer_spec.name) for n, p in self.named_parameters()]
