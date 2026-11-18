"""Numerical checks for the patch-side PiD inference kernels."""
import pytest
import torch

from pid._src.networks.pixeldit_official import RMSNorm, apply_rotary_emb
from pid._src.vdn_pid.sliding import sliding_attention
from pid._src.vdn_pid.spatial_memory import fixed_routes, joint_memory


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel')
@pytest.mark.parametrize('dim', [32, 64, 128])
@torch.inference_mode()
def test_fused_sliding_and_norm(dim):
    from pid._src.vdn_pid.sliding_points import norm_rope
    torch.manual_seed(1045)
    h, w = 13, 35
    def inputs(n):
        return torch.randn(2, n, 3, 2, dim, device='cuda', dtype=torch.bfloat16).unbind(2)
    q, k, v = inputs(h*w)
    tq, tk, tv = inputs(7)
    mask = torch.tensor([[1, 1, 0, 1, 0, 0, 0], [0]*7], device='cuda').bool()
    args = q, k, v, tk, tv, (h, w)
    expected = sliding_attention(*args, backend='reference', text_valid=mask)
    actual = sliding_attention(*args, backend='flex', text_valid=mask)
    torch.testing.assert_close(actual, expected, rtol=.015, atol=.008)
    norm = RMSNorm(dim).to('cuda', torch.bfloat16)
    norm.weight.uniform_(.5, 1.5)
    angle = torch.randn(h*w, dim//2, device='cuda')
    for a, b, pos in [(q, k, (angle.cos(), angle.sin())), (tq, tk, None)]:
        expected = norm(a), norm(b)
        if pos is not None:
            expected = apply_rotary_emb(*expected, freqs_cis=pos)
        actual = norm_rope(a, b, norm, norm, pos)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=.016, rtol=.008)


@torch.inference_mode()
def test_fixed4_writeback():
    torch.manual_seed(1045)
    q, k, v = torch.randn(3, 2, 35, 2, 8)
    routes = fixed_routes((5, 7), 4, 'cpu').expand(2, -1, -1)
    args = q, k, v, torch.rand(2, 35, 2), routes
    torch.testing.assert_close(joint_memory(*args, sparse=True, chunk_size=4),
                               joint_memory(*args, chunk_size=4))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA kernel'))])
def test_sliding_gradients(device):
    torch.manual_seed(1045)
    xs = [torch.randn(1, n, 2, 64, device=device, requires_grad=True) for n in (35, 35, 35, 3, 3)]
    def run(backend):
        y = sliding_attention(*xs, (5, 7), kernel=3, backend=backend)
        return y, torch.autograd.grad(y.sum(), xs)
    torch.testing.assert_close(run('flex'), run('reference'), atol=2e-5, rtol=2e-5)
