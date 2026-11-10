import copy

import pytest
import torch

from pid._src.networks.pixeldit_official import PiTBlock


@pytest.mark.parametrize("chunk_size", [1, 5, 100])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_pit_chunking_preserves_full_grid_attention_outputs_and_gradients(chunk_size, checkpointing):
    torch.manual_seed(731)
    original = PiTBlock(4, 16, 2, 2, attn_hidden_size=16).train()
    chunked = copy.deepcopy(original)
    # Two non-square images; chunk_size=5 also crosses an image boundary.
    x = torch.randn(24, 4, 4, requires_grad=True)
    cond = torch.randn(24, 16, requires_grad=True)
    chunk_x = x.detach().clone().requires_grad_(True)
    chunk_cond = cond.detach().clone().requires_grad_(True)
    expected = original(x, cond, 6, 8, 2)
    actual = chunked.forward_chunked(
        chunk_x, chunk_cond, 6, 8, 2, chunk_size=chunk_size, checkpointing=checkpointing
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(chunk_x.grad, x.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(chunk_cond.grad, cond.grad, rtol=2e-5, atol=2e-6)
    for (name, parameter), (other_name, other) in zip(original.named_parameters(), chunked.named_parameters()):
        assert name == other_name and parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(other.grad, parameter.grad, rtol=3e-5, atol=5e-6, msg=name)
    assert tuple(original.state_dict()) == tuple(chunked.state_dict())
