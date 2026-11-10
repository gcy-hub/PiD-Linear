"""Shared checkpoint state helpers."""
import json
import random
from pathlib import Path
import numpy as np
import torch

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def complete_checkpoints(run_dir):
    root = Path(run_dir) / "checkpoints"
    results = []
    for path in root.glob("step_*"):
        if not path.is_dir() or path.name.endswith(".incomplete"):
            continue
        try:
            metadata = json.loads((path / "complete.json").read_text())
            if (path / "state.pt").stat().st_size == metadata["bytes"]:
                results.append(path)
        except (OSError, ValueError, KeyError):
            pass
    return sorted(results)
