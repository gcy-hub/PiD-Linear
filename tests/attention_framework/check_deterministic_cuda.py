"""Exact repeated CUDA backward with the native BF16 attention backends."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from deterministic_training import configure
from algorithms.local import ClearAttention
from algorithms.qdm import joint_memory, fixed_routes
from attention_framework.contracts import AttentionLayout


def repeat(label, forward, inputs):
    output = forward()
    grad = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs, grad)
    assert all(torch.isfinite(g).all() for g in expected), label
    for _ in range(4):
        actual = torch.autograd.grad(forward(), inputs, grad)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual)), label
    print(label + ': five identical backwards', flush=True)


def main():
    configure()
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 16, 1024, 64, device='cuda', dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        repeat('BF16 FlashAttention', lambda: scaled_dot_product_attention(q, k, v), (q, k, v))
    layout = AttentionLayout(((32, 32, 0),), text_first=False)
    local = ClearAttention(window_size=11).cuda()
    repeat('BF16 compiled local FlexAttention', lambda: local(q, k, v, layout), (q, k, v))
    q, k, v = [torch.randn(1, 64, 2, 16, device='cuda', dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    beta = torch.rand(1, 64, 2, device='cuda', requires_grad=True)
    routes = fixed_routes((8, 8), device='cuda')[None]
    repeat('FP32 QDM sparse solve/index_add',
           lambda: joint_memory(q, k, v, beta, routes, sparse=True, chunk_size=17)[0],
           (q, k, v, beta))


if __name__ == '__main__':
    main()
