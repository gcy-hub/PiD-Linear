"""Check BF16 Prodigy parameters, gradients, states and FP32 checkpoint restore."""
import copy
import torch
from flux2_qdm.clear_strategy import make_optimizer


def update(parameter, optimizer):
    optimizer.zero_grad(set_to_none=True)
    parameter.square().float().mean().backward()
    assert parameter.grad.dtype == parameter.dtype
    optimizer.step()
    assert torch.isfinite(parameter).all()
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.dtype == parameter.dtype
                assert torch.isfinite(value).all()


def main():
    torch.manual_seed(42)
    original = torch.nn.Parameter(torch.randn(64) * 1e-4)
    optimizer, scheduler = make_optimizer([original])
    update(original, optimizer)
    state = copy.deepcopy(optimizer.state_dict())
    bf16 = torch.nn.Parameter(original.detach().to(torch.bfloat16))
    resumed, _ = make_optimizer([bf16])
    resumed.load_state_dict(state)
    before = bf16.detach().clone()
    update(bf16, resumed)
    restored = torch.nn.Parameter(bf16.detach().clone())
    other, _ = make_optimizer([restored])
    other.load_state_dict(copy.deepcopy(resumed.state_dict()))
    for _ in range(4):
        update(bf16, resumed)
        update(restored, other)
        torch.testing.assert_close(bf16, restored, rtol=0, atol=0)
    assert not torch.equal(before, bf16), 'BF16 parameters did not update'
    print('BF16 Prodigy update, gradient/state dtypes, FP32 conversion and exact resume passed')


if __name__ == '__main__':
    main()
