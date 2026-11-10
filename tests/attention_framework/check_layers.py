"""All 21 strategies on real native attention classes at small dimensions (CUDA)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from algorithms import build_attention
from attention_framework.config import COMBINATIONS, compose_config
from attention_framework.contracts import LayerSpec, ModelOutput
from models.attention import original_projections
from models.vendor import native_module
from training.engine import select_roles
from training.strategies import clear, vdn


def fixture(family, group, cfg):
    spec = LayerSpec(f'{group}.1', group, 1, 5, cfg.feature_tap)
    op = build_attention(cfg, hidden_size=128, num_heads=2, layer=spec).cuda()
    x, text = torch.randn(1, 6, 128, device='cuda'), torch.randn(1, 2, 128, device='cuda')
    if family == 'pid':
        from pid._src.networks.pixeldit_official import MMDiTJointAttention
        from models.pid import PidAttention
        site = PidAttention(MMDiTJointAttention(128, 2).cuda(), op, spec, cfg)
        pos = torch.ones(6, 32, dtype=torch.complex64, device='cuda')
        call = lambda: site(x, text, pos, grid_size=(2, 3))
    elif family == 'flux2':
        from models.flux2 import Flux2Attention
        native = native_module('flux2')
        single = group == 'single'
        cls = native.Flux2ParallelSelfAttention if single else native.Flux2Attention
        orig = cls(query_dim=128, heads=2, dim_head=64, **({} if single else {'added_kv_proj_dim': 128})).cuda()
        site = Flux2Attention(orig, op, spec, cfg)
        site.grid, site.text_length = (2, 3), 2
        z = torch.cat((text, x), 1)
        call = (lambda: site(z)) if single else (lambda: site(x, encoder_hidden_states=text))
    else:
        from models.flux3 import Flux3Block
        site = Flux3Block(native_module('flux3').Block(128, 2, 2).cuda(), op, spec, cfg)
        site.samples = ((2, 3, 2),)
        z = torch.cat((x, text), 1)[0]
        freqs = torch.eye(2, device='cuda').expand(1, 1, 8, 32, 2, 2).clone()
        mod = tuple(torch.randn(8, 128, device='cuda') for _ in range(3))
        if group == 'early_image':
            z, freqs, mod = z[:6], freqs[:, :, :6], tuple(m[:6] for m in mod)
        call = lambda: site(z, freqs, [(0, len(z))], mod)
    native_call = call
    def mixed_call():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return native_call()
    return site, mixed_call


class Adapter(nn.Module):
    def __init__(self, site):
        super().__init__()
        self.site = site
    @property
    def module(self):
        return self
    def role_parameters(self):
        return self.site.role_parameters()


def main():
    assert torch.cuda.is_available(), 'CUDA required'
    for family, algorithm, strategy in COMBINATIONS:
        cfg = compose_config(f'configs/models/{family}.yaml', f'configs/algorithms/{algorithm}.yaml',
                             f'configs/training/{strategy}.yaml', 'configs/data/default.yaml',
                             ['algorithm.local.window_size=3', 'algorithm.memory.region_size=2',
                              'algorithm.memory.halo=1', 'algorithm.clear.radius=2'])
        groups = {'pid': ('patch',), 'flux2': ('double', 'single'), 'flux3': ('early_image', 'joint')}[family]
        for group in groups:
            site, call = fixture(family, group, cfg)
            adapter = Adapter(site)
            if algorithm == 'cross_scan_kda':
                assert site.operator.kda.direction == 1
            site.teacher = True
            with torch.no_grad(), original_projections(site):
                before = call()
            frozen = {n: p.clone() for n, p in site.orig.named_parameters() if '.base.' in n}
            for stage in (('a1', 'a2', 'b') if strategy == 'vdn' else ('clear',)):
                roles = select_roles(adapter, stage)
                expected = clear.ROLES if stage == 'clear' else vdn.ROLES[stage]
                assert {r.role for r in roles} <= set(expected)
                adapter.zero_grad(set_to_none=True)
                if stage == 'a1':
                    seen = []
                    def callback(pair):
                        assert not pair.teacher.requires_grad
                        ((pair.student.float()-pair.teacher.float()).square().sum() /
                         pair.teacher.float().square().sum().clamp_min(1e-8)).backward()
                        seen.append(pair.layer.name)
                    site.callback = callback
                    with torch.no_grad(), original_projections(site):
                        call()
                    assert seen == [site.layer.name]
                    site.callback = None
                else:
                    site.teacher, site.capture = False, True
                    value = call()
                    values = value if isinstance(value, tuple) else (value,)
                    loss = sum(v.float().square().mean() for v in values)
                    if stage == 'clear':
                        assert site.feature is not None and site.feature.requires_grad
                        # Backpropagate feature KD alone to detect a missing memory tap.
                        site.feature.float().square().mean().backward(retain_graph=True)
                        branch = [r.parameter.grad for r in roles if r.role == 'branch']
                        if branch:
                            assert any(g is not None and g.abs().sum() > 0 for g in branch)
                    loss.backward()
                assert any(r.parameter.grad is not None for r in roles)
                assert all(torch.isfinite(r.parameter.grad).all() for r in roles if r.parameter.grad is not None)
                with torch.no_grad():
                    for role in roles:
                        if role.parameter.grad is not None:
                            role.parameter.add_(role.parameter.grad, alpha=-1e-4)
                site.teacher = True
                with torch.no_grad(), original_projections(site):
                    after = call()
                for a, b in zip(before if isinstance(before, tuple) else (before,), after if isinstance(after, tuple) else (after,)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for n, p in site.orig.named_parameters():
                    if n in frozen:
                        torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
            del site, adapter, call
        print(f'{family}/{algorithm}/{strategy}: native layers, roles, teacher, taps, gradients passed', flush=True)


if __name__ == '__main__':
    main()
