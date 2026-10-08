"""Atomic completed-checkpoint publication and exact same-stage training state."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from pid._src.linear_pid.data import atomic_json


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


def resolve_checkpoint(value, run_dir=None):
    if value in {"", "none"}:
        return None
    if value == "auto":
        completed = complete_checkpoints(run_dir)
        return completed[-1] if completed else None
    path = Path(value)
    if path.is_dir() and (path / "checkpoints").exists():
        completed = resolve_checkpoint("auto", path)
        if completed is None:
            raise FileNotFoundError(f"No complete checkpoints in {path}")
        return completed
    if path.is_dir():
        if path not in complete_checkpoints(path.parent.parent):
            raise ValueError(f"Incomplete checkpoint: {path}")
        return path
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        resolve_checkpoint(str(path))
    payload = torch.load(
        path / "state.pt" if path.is_dir() else path, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("schema") != 1:
        raise ValueError("Not a Linear-PiD v1 checkpoint/export")
    return payload


def validate_resume(payload, config, fingerprint):
    saved = payload["config"]
    for key in [
        "layers",
        "local_mixing",
        "seed",
        "lr_new",
        "lr_backbone",
        "weight_decay",
        "warmup",
        "lambda_out",
        "grad_clip",
        "fla_commit",
    ]:
        if saved[key] != config.to_dict()[key]:
            raise ValueError(
                f"Resume {key} mismatch: checkpoint={saved[key]} requested={config.to_dict()[key]}; use --init-from for a new stage"
            )
    if payload["metadata"]["data_fingerprint"] != fingerprint:
        raise ValueError("Dataset index changed; exact continuation requires the same index")


def save_checkpoint(run_dir, net, optimizer, scheduler, config, progress, metadata, keep=3):
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    rng = capture_rng()
    states = [None] * world
    if distributed:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    root = Path(run_dir) / "checkpoints"
    path = root / f"step_{progress['total_step']:09d}_batch_{progress['cursor']:09d}"
    error = None
    if rank == 0:
        try:
            _publish_checkpoint(
                root, path, run_dir, net, optimizer, scheduler, config, progress, metadata, states, world, keep
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if distributed:
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error:
        raise RuntimeError(f"Checkpoint publication failed: {error}")
    return path


def _publish_checkpoint(
    root, path, run_dir, net, optimizer, scheduler, config, progress, metadata, states, world, keep
):
    root.mkdir(parents=True, exist_ok=True)
    if not (path / "complete.json").exists():
        temporary = path.with_name(path.name + ".incomplete")
        temporary.mkdir(exist_ok=True)
        payload = {
            "schema": 1,
            "model": cpu_tree(net.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "progress": dict(progress),
            "metadata": metadata,
            "rng": states,
            "world_size": world,
        }
        with (temporary / "state.pt").open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(
            temporary / "complete.json",
            {
                "bytes": (temporary / "state.pt").stat().st_size,
                "step": progress["total_step"],
                "cursor": progress["cursor"],
            },
        )
        if path.exists():
            shutil.rmtree(path)
        os.replace(temporary, path)
    atomic_json(root / "latest.json", {"checkpoint": str(path)})
    checkpoints = complete_checkpoints(run_dir)
    for old in checkpoints[:-keep] if keep > 0 else []:
        if not (old / "KEEP").exists():
            shutil.rmtree(old)


def student_weights(payload, weights="raw"):
    """Raw by default; allow explicit EMA selection only for legacy files."""
    if weights == "raw":
        return payload["model"]
    if weights == "ema":
        if payload.get("ema") is not None:
            return payload["ema"]
        if payload.get("export_weights") == "ema":
            return payload["model"]
        raise ValueError("Checkpoint has no EMA weights; select --weights raw / --init-weights raw")
    raise ValueError(f"Unknown student weights: {weights}")


def export_student(checkpoint, destination, weights="raw"):
    payload = load_checkpoint(checkpoint)
    result = {
        "schema": 1,
        "model": student_weights(payload, weights),
        "config": payload["config"],
        "metadata": payload["metadata"],
        "progress": payload["progress"],
        "export_weights": weights,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    torch.save(result, temp)
    os.replace(temp, destination)
