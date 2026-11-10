"""Prodigy with FP32 master parameters for each local ZeRO optimizer shard."""
import torch
from prodigyopt import Prodigy


class MasterProdigy(Prodigy):
    def __init__(self, params, **kwargs):
        groups = list(params)
        if groups and not isinstance(groups[0], dict):
            groups = [dict(params=groups)]
        self.compute_parameters = []
        master_groups = []
        for group in groups:
            compute = list(group['params'])
            self.compute_parameters.extend(compute)
            master_groups.append({**group, 'params': [torch.nn.Parameter(p.detach().float().clone()) for p in compute]})
        super().__init__(master_groups, **kwargs)
        self._record_masters()

    def _record_masters(self):
        for group in self.param_groups:
            for master in group['params']:
                self.state[master]['master'] = master.detach()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        with torch.no_grad():
            masters = [p for group in self.param_groups for p in group['params']]
            for compute, master in zip(self.compute_parameters, masters):
                master.copy_(self.state[master].get('master', compute))
        self._record_masters()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for p in self.compute_parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('MasterProdigy expects externally computed gradients')
        masters = [p for group in self.param_groups for p in group['params']]
        try:
            for compute, master in zip(self.compute_parameters, masters):
                master.grad = None if compute.grad is None else compute.grad.float()
            result = super().step()
            for compute, master in zip(self.compute_parameters, masters):
                compute.copy_(master)
            return result
        finally:
            for master in masters:
                master.grad = None


def local_optimizer(optimizer):
    return getattr(optimizer, 'optim', optimizer)


def move_state(optimizer, device):
    for state in local_optimizer(optimizer).state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'master':
                state[key] = value.to(device)


def build_optimizer(config, roles, stage):
    import math
    shard = config.runtime.optimizer_sharding and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    groups = []
    for role in sorted({r.role for r in roles}):
        groups.append(dict(params=[r.parameter for r in roles if r.role == role], role=role))
    options = dict(lr=config.training.lr, weight_decay=config.training.weight_decay)
    if config.training.optimizer == 'prodigy':
        cls = MasterProdigy
        options.update(use_bias_correction=True, safeguard_warmup=True,
                       fsdp_in_use=shard)
    elif config.training.optimizer == 'adamw':
        cls = torch.optim.AdamW
    else:
        raise ValueError('Unknown optimizer')
    if shard:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=cls, **options)
    else:
        optimizer = cls(groups, **options)
    phase = config.training.vdn[stage] if stage != 'clear' else None
    steps = phase.steps if phase else config.training.steps
    warmup = phase.warmup if phase else 0
    def factor(step):
        if config.training.scheduler == 'constant':
            return 1.
        if config.training.scheduler != 'cosine':
            raise ValueError('Unknown scheduler')
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * min(1., (step-warmup)/max(1, steps-warmup))))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
