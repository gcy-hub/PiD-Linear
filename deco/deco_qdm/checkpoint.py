"""Atomic, delta-only snapshots with rank-specific RNG and consumed data cursors."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

SCHEMA = 1
ARCHITECTURE = "native-deco-qdm-halo2"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


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
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value.get("cuda") is not None and torch.cuda.is_available():
        # Only the rank-local GPU is resident; reading every visible device
        # would create unnecessary CUDA contexts on all other ranks' devices.
        torch.cuda.set_rng_state(value["cuda"])


def fingerprint_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_checkpoints(stage_dir):
    paths = []
    for path in (Path(stage_dir) / "checkpoints").glob("step-*"):
        if not path.is_dir() or path.name.endswith(".incomplete"):
            continue
        marker = read_json(path / "complete.json")
        state = path / "state.pt"
        if marker and state.exists() and state.stat().st_size == marker.get("bytes"):
            paths.append(path)
    return sorted(paths, key=lambda path: (read_json(path / "complete.json")["step"], path.name))


def latest_state(stage_dir):
    paths = complete_checkpoints(stage_dir)
    return paths[-1] if paths else None


def load_state(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    state_path = directory / "state.pt"
    marker = read_json(directory / "complete.json")
    if not marker or not state_path.exists() or marker.get("bytes") != state_path.stat().st_size:
        raise ValueError(f"Incomplete checkpoint: {directory}")
    value = torch.load(state_path, map_location="cpu", weights_only=False)
    if value.get("schema") != SCHEMA or value.get("architecture") != ARCHITECTURE:
        raise ValueError("Not a native DeCo-QDM training checkpoint")
    if value["progress"]["stage_step"] != marker.get("step"):
        raise ValueError("Checkpoint step and complete marker disagree")
    return value


def save_state(stage_dir, model, optimizer, scheduler, *, config, stage, progress, metadata,
               milestone=False, keep=3):
    from .model import delta_state_dict
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    rng = [None] * world
    if distributed:
        dist.all_gather_object(rng, capture_rng())
    else:
        rng[0] = capture_rng()
    generation = [time.time_ns() if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(generation, src=0)
    directory = Path(stage_dir) / "checkpoints"
    path = directory / f"step-{progress['stage_step']:09d}-cursor-{progress['cursor']:012d}-{generation[0]}"
    error = [None]
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".incomplete")
            temporary.mkdir()
            payload = dict(schema=SCHEMA, architecture=ARCHITECTURE,
                           delta=cpu_tree(delta_state_dict(model)), optimizer=cpu_tree(optimizer.state_dict()),
                           scheduler=scheduler.state_dict(), config=dict(config), stage=stage,
                           progress=dict(progress), metadata=dict(metadata), rng=rng, world_size=world)
            with (temporary / "state.pt").open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(temporary / "complete.json", dict(step=progress["stage_step"],
                        cursor=progress["cursor"], bytes=(temporary / "state.pt").stat().st_size))
            if milestone:
                (temporary / "KEEP").touch()
            os.replace(temporary, path)
            atomic_json(directory / "latest.json", {"checkpoint": str(path.resolve())})
            checkpoints = complete_checkpoints(stage_dir)
            for old in checkpoints[:-keep] if keep > 0 else []:
                if not (old / "KEEP").exists():
                    shutil.rmtree(old)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f"Checkpoint publication failed: {error[0]}")
    return path
