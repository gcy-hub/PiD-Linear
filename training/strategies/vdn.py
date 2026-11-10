"""VDN phase roles and detached layerwise distillation."""
import torch
import torch.distributed as dist

ROLES = {'a1': {'branch', 'pooling'}, 'a2': {'branch', 'pooling'},
         'b': {'branch', 'pooling', 'local_gate', 'lora'}}


def teacher_energy(adapter, batches):
    energies = {}
    def collect(pair):
        name = pair.layer.name
        energy = pair.teacher.detach().float().square().sum()
        energies[name] = energies.get(name, 0) + energy
    for batch in batches:
        adapter.layerwise(batch, collect)
    for name in sorted(energies):
        if dist.is_initialized():
            dist.all_reduce(energies[name])
        energies[name].clamp_min_(1e-30)
    return energies


def backward_layerwise(adapter, batch, energies):
    losses = []
    def backward(pair):
        teacher = pair.teacher.detach().float()
        numerator = (pair.student.float() - teacher).square().sum()
        loss = numerator / energies[pair.layer.name]
        # The engine averages gradients after all immediate backwards.
        scale = dist.get_world_size() if dist.is_initialized() else 1
        (loss * scale).backward()
        losses.append(loss.detach())
    adapter.layerwise(batch, backward)
    if not losses:
        raise ValueError('A1 requires selected attention layers')
    return torch.stack(losses).sum()
