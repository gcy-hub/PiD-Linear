from types import SimpleNamespace

import pytest
import torch

from scripts.prepare_pid_assets import prepare_generated_latents


def test_generated_latents_can_precede_index_and_resume_with_matching_inputs(tmp_path, monkeypatch):
    import diffusers

    calls = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, caption, **kwargs):
            calls.append((caption, kwargs["height"], kwargs["generator"].initial_seed()))
            return SimpleNamespace(images=torch.ones(1, 16, 2, 2))

    monkeypatch.setattr(diffusers, "ZImagePipeline", Pipeline)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cases = [
        {"kind": "generated", "ordinal": 0, "resolution": resolution, "seed": 1042, "caption": "A fox"}
        for resolution in ("2048", "4096")
    ]
    prepare_generated_latents(cases, tmp_path, tmp_path / "weights", 0, 1, "cpu")
    assert calls == [("A fox", 512, 1042), ("A fox", 1024, 1042)]
    assert len(list(tmp_path.glob("*.latent.pt"))) == 2
    # The final asset stage can include real cases and extra output metadata.
    final_cases = [{"kind": "real"}] + [case | {"file": "unused.pt"} for case in cases]
    prepare_generated_latents(final_cases, tmp_path, tmp_path / "weights", 0, 1, "cpu")
    assert len(calls) == 2
    changed = [case | {"seed": 9} for case in cases]
    with pytest.raises(ValueError, match="inputs changed"):
        prepare_generated_latents(changed, tmp_path, tmp_path / "weights", 0, 1, "cpu")
