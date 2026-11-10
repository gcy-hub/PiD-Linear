import contextlib

import torch

from pid._src.common import runtime


class RecordingStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, y, *, lq_latent, degrade_sigma, text_valid_mask):
        self.calls.append((y.clone(), text_valid_mask.clone(), t.clone()))
        return torch.zeros_like(x)


def test_standalone_multistep_cfg_masks_and_seed(monkeypatch):
    monkeypatch.setattr(runtime.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    model = RecordingStudent()
    condition, null = torch.ones(1, 5, 4), torch.zeros(1, 5, 4)
    conditional_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    null_mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    latent = torch.randn(1, 16, 1, 1)
    args = (model, condition, conditional_mask, null, null_mask, latent)
    first = runtime.sample(*args, height=8, width=12, steps=4, cfg=5, seed=42)
    assert len(model.calls) > 2
    for y, mask, timestep in model.calls:
        torch.testing.assert_close(mask, conditional_mask if y.any() else null_mask)
        assert timestep.dtype == torch.float32
    second = runtime.sample(*args, height=8, width=12, steps=4, cfg=5, seed=42)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (1, 3, 8, 12)


def test_gallery_keeps_live_weights_and_restores_training_mode(monkeypatch):
    model = torch.nn.Linear(3, 2).train()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def forbid_transfer(*args, **kwargs):
        raise AssertionError("Gallery must not copy model weights to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_transfer)
    with runtime.evaluation_weights(model):
        assert not model.training
        with torch.no_grad():
            model(torch.randn(1, 3))
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
