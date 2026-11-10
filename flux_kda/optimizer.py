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
