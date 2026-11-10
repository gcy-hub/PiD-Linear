"""Native projection isolation and the shared attention execution boundary."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from attention_framework.contracts import AttentionResult, LayerPair, RoleParameter


class AttentionProjection(nn.Module):
    """Train attention slices while retaining immutable teacher and FFN weights."""
    def __init__(self, base, rank, alpha, *, rows=None, columns=None):
        super().__init__()
        if rows is not None and columns is not None:
            raise ValueError('Only one projection axis may be partitioned')
        if rank < 1:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        self.rows, self.columns = rows or base.out_features, columns or base.in_features
        self.in_features, self.out_features = base.in_features, base.out_features
        self.adapters_enabled = True
        self.scale = alpha / rank
        base.requires_grad_(False)
        weight = base.weight[:self.rows, :self.columns]
        self.train_weight = nn.Parameter(weight.detach().float().clone())
        self.lora_A = nn.Parameter(torch.empty(rank, self.columns, device=weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.rows, rank, device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def bias(self):
        return self.base.bias

    @property
    def weight(self):
        if not self.adapters_enabled:
            return self.base.weight
        weight = self.train_weight.to(self.base.weight.dtype)
        if self.rows < self.out_features:
            return torch.cat((weight, self.base.weight[self.rows:]), 0)
        if self.columns < self.in_features:
            return torch.cat((weight, self.base.weight[:, self.columns:]), 1)
        return weight

    def lora(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale).to(x.dtype)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.adapters_enabled:
            delta = self.lora(x[..., :self.columns])
            if self.rows < self.out_features:
                delta = F.pad(delta, (0, self.out_features-self.rows))
            out = out + delta.to(out.dtype)
        return out

    def attention_output(self, x):
        # Fused single-stream O: attention contribution excludes FFN and shared bias.
        out = F.linear(x, self.weight[:, :self.columns], None)
        return out + self.lora(x).to(out.dtype) if self.adapters_enabled else out


@contextmanager
def original_projections(module):
    projections = [m for m in module.modules() if isinstance(m, AttentionProjection)]
    states = [m.adapters_enabled for m in projections]
    try:
        for m in projections:
            m.adapters_enabled = False
        yield
    finally:
        for m, enabled in zip(projections, states):
            m.adapters_enabled = enabled


def image_tokens(tensor, layout):
    chunks, offset = [], 0
    for h, w, t in layout.samples:
        start = offset + (t if layout.text_first else 0)
        chunks.append(tensor[:, start:start + h*w])
        offset += h*w+t
    return torch.cat(chunks, 1)


def dense_attention(inputs):
    chunks, offset = [], 0
    for h, w, t in inputs.layout.samples:
        end = offset + h*w+t
        q, k, v = [z[:, offset:end].transpose(1, 2) for z in (inputs.q_pos, inputs.k_pos, inputs.v)]
        chunks.append(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        offset = end
    return AttentionResult(torch.cat(chunks, 1))


class AttentionSite(nn.Module):
    def __init__(self, original, operator, layer):
        super().__init__()
        self.orig, self.operator, self.layer = original, operator, layer
        self.teacher = False
        self.capture = False
        self.feature = None
        self.callback = None
        self.grid = None

    def attend(self, inputs, project):
        result = dense_attention(inputs) if self.teacher else self.operator(inputs)
        post = project(result.pre_o)
        if result.post_o_residual is not None:
            post = post + result.post_o_residual
        if self.capture:
            self.feature = image_tokens(result.pre_o if self.layer.feature_tap == 'pre_o' else post, inputs.layout)
        if self.callback is not None:
            detached = type(inputs)(**{name: value.detach() if isinstance(value, torch.Tensor) else value
                                      for name, value in vars(inputs).items()})
            # The native trunk remains under no_grad; each student graph is independent.
            with torch.enable_grad():
                self.teacher = False
                projections = [m for m in self.orig.modules() if isinstance(m, AttentionProjection)]
                for p in projections:
                    p.adapters_enabled = True
                try:
                    # Recompute QKV as well, so image-QKV LoRA belongs to this graph.
                    student_inputs = self.student_inputs()
                    student_result = self.operator(student_inputs)
                    student = project(student_result.pre_o)
                    if student_result.post_o_residual is not None:
                        student = student + student_result.post_o_residual
                    self.callback(LayerPair(self.layer, detached, image_tokens(post.detach(), inputs.layout),
                                            image_tokens(student, inputs.layout)))
                finally:
                    self.teacher = True
                    for p in projections:
                        p.adapters_enabled = False
        return post

    def role_parameters(self):
        roles = []
        for name, module in self.orig.named_modules():
            if isinstance(module, AttentionProjection):
                for key, role in (('train_weight', 'image_qkvo'), ('lora_A', 'lora'), ('lora_B', 'lora')):
                    roles.append(RoleParameter(f'orig.{name}.{key}', getattr(module, key), role, self.layer.name))
        for item in self.operator.role_parameters():
            roles.append(RoleParameter('operator.' + item.name, item.parameter, item.role, self.layer.name))
        return roles


def install_projection(module, name, config, **kwargs):
    base = getattr(module, name)
    lora = config.training.vdn.lora
    setattr(module, name, AttentionProjection(base, lora.rank, lora.alpha, **kwargs))
