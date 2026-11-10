"""CPU checks for phase roles, losses, resumable updates and atomic publication."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from omegaconf import OmegaConf
from attention_framework.contracts import RoleParameter, ModelOutput
from . import checkpoint
from .engine import select_roles
from .optim import MasterProdigy
from .strategies import clear, vdn


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(1.))
        self.gate = nn.Parameter(torch.tensor(2.))
        self.base = nn.Parameter(torch.tensor(3.))
    @property
    def module(self):
        return self
    def role_parameters(self):
        return [RoleParameter('p', self.p, 'branch', 'layer'), RoleParameter('gate', self.gate, 'local_gate', 'layer')]
    def layerwise(self, batch, callback):
        callback(SimpleNamespace(teacher=self.base.detach(), student=self.p * self.base.detach()))
        assert self.p.grad is not None
        assert self.base.grad is None


def main():
    toy = Toy()
    assert [r.name for r in select_roles(toy, 'a1')] == ['p']
    toy.p.data.fill_(2.)
    vdn.backward_layerwise(toy, None, 2)
    assert torch.allclose(toy.p.grad, torch.tensor(1.))
    assert {r.name for r in select_roles(toy, 'b')} == {'p', 'gate'}
    s = ModelOutput(torch.tensor([2.]), {'layer': torch.tensor([3.], requires_grad=True)})
    t = ModelOutput(torch.tensor([1.]), {'layer': torch.tensor([1.])})
    loss = clear.objective(s, t, torch.tensor([0.]), ('layer',), OmegaConf.create(dict(fm=1., prediction=.5, feature=.5)))
    assert loss.item() == 6.5
    loss.backward()
    assert s.features['layer'].grad.item() == 2.
    try:
        clear.objective(s, t, torch.tensor([0.]), ('missing',), None)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing tap accepted')
    optimizer = torch.optim.AdamW([toy.p], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    generator = torch.Generator().manual_seed(7)
    config = OmegaConf.create(dict(model='toy'))
    with tempfile.TemporaryDirectory() as tmp:
        progress = dict(step=1, stage='a1', stage_step=1, cursor=2, pending_validation=True, done=False)
        directory = checkpoint.save(tmp, toy, optimizer, scheduler, generator, progress, config, 0, 1)
        expected = torch.rand(2, generator=generator)
        toy.p.data.zero_()
        state = checkpoint.read(directory, 0, 1)
        checkpoint.restore(state, toy, optimizer, scheduler, generator)
        assert toy.p.item() == 2.
        assert torch.equal(expected, torch.rand(2, generator=generator))
        # An unpublished next shard cannot replace the last complete set.
        incomplete = Path(tmp) / 'step-000000002'
        incomplete.mkdir()
        torch.save({}, incomplete / 'rank-0.partial')
        assert checkpoint.latest(tmp) == directory
        (directory / 'rank-0.pt').unlink()
        try:
            checkpoint.read(directory, 0, 1)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing shard accepted')
    p = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt = MasterProdigy([p], lr=1.)
    for _ in range(3):
        p.grad = torch.ones_like(p)
        opt.step()
    master = opt.param_groups[0]['params'][0].detach().clone()
    assert not torch.equal(master, p.float())
    opt.zero_grad()
    assert p.grad is None
    state = opt.state_dict()
    other = MasterProdigy([p], lr=1.)
    other.load_state_dict(state)
    assert torch.equal(master, other.param_groups[0]['params'][0])
    print('CPU strategy/checkpoint/FP32 master checks passed')


if __name__ == '__main__':
    main()
