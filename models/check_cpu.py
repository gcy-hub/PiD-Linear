"""CPU checks for native projection order, isolation and independent A1 graphs.

Run with CUDA_VISIBLE_DEVICES='' python -m models.check_cpu.
"""
import copy
import torch
from torch import nn
from omegaconf import OmegaConf
from attention_framework.contracts import LayerSpec, RoleParameter, AttentionResult
from .attention import dense_attention, original_projections, AttentionProjection
from .flux2 import Flux2Attention
from .flux3 import Flux3Block
from .pid import PidAttention
from .vendor import native_module


class DenseOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        result = dense_attention(inputs)
        return AttentionResult(result.pre_o + self.branch)

    def role_parameters(self):
        return [RoleParameter('branch', self.branch, 'branch', '')]


def main():
    torch.manual_seed(7)
    cfg = OmegaConf.create({'training': {'vdn': {'lora': {'rank': 2, 'alpha': 2}}}})
    f2, f3 = native_module('flux2'), native_module('flux3')
    cases = []
    for single in (False, True):
        cls = f2.Flux2ParallelSelfAttention if single else f2.Flux2Attention
        orig = cls(query_dim=16, heads=2, dim_head=8, **({} if single else {'added_kv_proj_dim': 16}))
        # The native constructors accept separate image and text projections.
        x, text = torch.randn(1, 8 if single else 6, 16), torch.randn(1, 2, 16)
        kwargs = {} if single else {'encoder_hidden_states': text}
        baseline = copy.deepcopy(orig)(x, **kwargs)
        spec = LayerSpec('single.0' if single else 'double.0', 'single' if single else 'double', 0, 8, 'pre_o')
        site = Flux2Attention(orig, DenseOperator(), spec, cfg)
        site.grid, site.text_length = (2, 3), 2
        cases.append((site, lambda s, x=x, kwargs=kwargs: s(x, **kwargs), baseline))
    orig = f3.Block(16, 2, 2)
    x = torch.randn(8, 16)
    # Identity rotations with native per-head width 8.
    freqs = torch.eye(2).expand(1, 1, 8, 4, 2, 2).clone()
    mod = tuple(torch.randn(8, 16) for _ in range(3))
    baseline = copy.deepcopy(orig)(x, freqs, [(0, 8)], mod)
    site = Flux3Block(orig, DenseOperator(), LayerSpec('joint.4', 'joint', 4, 9, 'post_o'), cfg)
    site.samples = ((2, 3, 2),)
    cases.append((site, lambda s: s(x, freqs, [(0, 8)], mod), baseline))
    from pid._src.networks.pixeldit_official import MMDiTJointAttention
    orig = MMDiTJointAttention(16, 2)
    px, py = torch.randn(1, 6, 16), torch.randn(1, 2, 16)
    pos = torch.ones(6, 4, dtype=torch.complex64)
    baseline = copy.deepcopy(orig)(px, py, pos)
    site = PidAttention(orig, DenseOperator(), LayerSpec('patch.0', 'patch', 0, 0, 'pre_o'), cfg)
    cases.append((site, lambda s: s(px, py, pos, grid_size=(2, 3)), baseline))
    for site, call, baseline in cases:
        actual = call(site)
        for got, want in zip(actual if isinstance(actual, tuple) else (actual,), baseline if isinstance(baseline, tuple) else (baseline,)):
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
        site.teacher = True
        with torch.no_grad(), original_projections(site):
            teacher = call(site)
        for role in site.role_parameters():
            with torch.no_grad():
                role.parameter.add_(0.1)
        with torch.no_grad(), original_projections(site):
            after = call(site)
        for got, want in zip(after if isinstance(after, tuple) else (after,), teacher if isinstance(teacher, tuple) else (teacher,)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        pairs = []
        def callback(pair):
            pairs.append(pair.layer.name)
            (pair.student-pair.teacher).square().mean().backward()
        site.callback = callback
        with torch.no_grad(), original_projections(site):
            call(site)
        assert pairs and site.operator.branch.grad is not None
        roles = site.role_parameters()
        assert len({id(r.parameter) for r in roles}) == len(roles)
    projection = AttentionProjection(nn.Linear(7, 12), 2, 2, rows=6)
    before = projection.base.weight[6:].clone()
    projection(torch.randn(3, 7)).sum().backward()
    assert projection.train_weight.shape == (6, 7) and projection.base.weight.grad is None
    with torch.no_grad():
        projection.train_weight.add_(1)
        projection.lora_B.add_(1)
    torch.testing.assert_close(projection.weight[6:], before, atol=0, rtol=0)
    print('PASS: four native layer types; native parity, teacher isolation, A1 backward, roles, fused FFN slices')


def lifecycle():
    import json
    import tempfile
    from pathlib import Path
    from safetensors.torch import save_file
    from attention_framework.contracts import EncodedBatch
    from . import build_model
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/default.yaml')
    cfg.model.family = 'flux2'
    cfg.model.layer_counts = {'double': 1, 'single': 1}
    cfg.runtime.dtype = 'float32'
    cfg.layer_specs = [dict(name='single.0', group='single', index=0, ordinal=1, feature_tap='post_o')]
    kwargs = dict(in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=8,
                  num_attention_heads=2, joint_attention_dim=12, axes_dims_rope=[2, 2, 2, 2], guidance_embeds=False)
    base = native_module('flux2').Flux2DiT(**kwargs)
    with tempfile.TemporaryDirectory() as root:
        path = Path(root)
        save_file(base.state_dict(), str(path / 'model.safetensors'))
        (path / 'config.json').write_text(json.dumps(kwargs))
        cfg.model.resources.weights = root
        factory = lambda *a, **kw: DenseOperator()
        model = build_model(cfg, device=torch.device('cpu'), algorithm_factory=factory)
        batch = EncodedBatch(('a', 'b'), 'flux2',
                             dict(latents=torch.randn(2, 6, 8), image_ids=torch.zeros(2, 6, 4)),
                             dict(embeddings=torch.randn(2, 3, 12), text_ids=torch.zeros(2, 3, 4),
                                  mask=torch.tensor([[1, 1, 0], [1, 1, 1]])), ((2, 3), (2, 3)))
        train = model.prepare_batch(batch, generator=torch.Generator().manual_seed(42))
        model.requires_grad_(False)
        for role in model.role_parameters():
            role.parameter.requires_grad_(True)
        for checkpoint in (False, True):
            cfg.runtime.gradient_checkpointing = checkpoint
            out = model(train, capture=('single.0',))
            model(train, teacher=True, capture=('single.0',))
            assert out.prediction.shape == train.target.shape
            assert out.features['single.0'].shape == (2, 6, 16)
            (out.prediction.square().mean() + out.features['single.0'].square().mean()).backward()
        pairs = []
        def callback(pair):
            pair.student.square().mean().backward()
            pairs.append(pair.layer.name)
        model.layerwise(train, callback)
        assert pairs == ['single.0', 'single.0']
        assert [s.layer.ordinal for s in model.sites] == [1]
        cfg.model.layer_counts.single = 2
        try:
            build_model(cfg, device=torch.device('cpu'), algorithm_factory=factory)
        except ValueError as exc:
            assert 'Native layer counts' in str(exc)
        else:
            raise AssertionError('incorrect architecture accepted')
    print('PASS: offline strict load, partial layer ordinal, variable text lengths, batch preparation, feature/A1 gradients, checkpoint teacher bypass, count rejection')


if __name__ == '__main__':
    main()
    lifecycle()
