"""VDN phase roles and detached layerwise distillation."""
import torch

ROLES = {'a1': {'branch', 'pooling'}, 'a2': {'branch', 'pooling'},
         'b': {'branch', 'pooling', 'local_gate', 'lora'}}


def backward_layerwise(adapter, batch, accumulation):
    losses = []
    def backward(pair):
        teacher = pair.teacher.detach().float()
        loss = (pair.student.float() - teacher).square().sum() / teacher.square().sum().clamp_min(1e-30)
        (loss / accumulation).backward()
        losses.append(loss.detach())
    adapter.layerwise(batch, backward)
    if not losses:
        raise ValueError('A1 requires selected attention layers')
    return torch.stack(losses).mean()
