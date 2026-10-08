"""Check pinned KDA availability and optional small CUDA forward/backward."""

import argparse

import torch

from pid._src.linear_pid.attention import reference_kda
from pid._src.linear_pid.environment import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    for package, version in check_runtime().items():
        print(f"{package}: {version}")
    from fla.ops.kda import chunk_kda

    if args.skip_cuda:
        print("Imports passed; CUDA verification explicitly skipped")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    print("GPU:", torch.cuda.get_device_name())
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    g = torch.full_like(q, -0.01, requires_grad=True)
    beta = torch.full((1, 32, 2), 0.5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual, _ = chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    expected = reference_kda(
        torch.nn.functional.normalize(q.float(), dim=-1),
        torch.nn.functional.normalize(k.float(), dim=-1),
        v.float(),
        g.float(),
        beta.float(),
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.05)
    upstream = torch.randn_like(actual)
    leaves = [q, k, v, g, beta]
    got = torch.autograd.grad((actual * upstream).sum(), leaves, retain_graph=True)
    want = torch.autograd.grad((expected * upstream.float()).sum(), leaves)
    for a, b in zip(got, want):
        torch.testing.assert_close(a.float(), b.float(), atol=0.05, rtol=0.1)
        assert torch.isfinite(a).all()
    print("PASS: fused KDA outputs/gradients match the reference recurrence; CUDA/Triton compiled")


if __name__ == "__main__":
    main()
