"""Atomic VDN delta + complete training state; one immutable external base."""

import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist

from pid._src.linear_pid.checkpoint import capture_rng, complete_checkpoints, cpu_tree
from pid._src.linear_pid.data import atomic_json, sha256_file


def delta_state(net):
    from .attention import JointVDN
    prefixes = tuple(f'patch_blocks.{i}.attn.' for i, block in enumerate(net.patch_blocks)
                     if isinstance(block.attn, JointVDN))
    return {
        k: v
        for k, v in net.state_dict().items()
        if ".orig." not in k or ".A." in k or ".B." in k
        if k.startswith(prefixes)
    }


def load_delta(net, delta):
    expected = set(delta_state(net))
    if expected != set(delta):
        raise ValueError(f"VDN delta keys differ: missing={expected - set(delta)}, unexpected={set(delta) - expected}")
    net.load_state_dict(delta, strict=False)


def load_parent_delta(net, delta):
    # A1/A2 parents contain every new branch tensor and no LoRA tensors.
    expected = {k for k in delta_state(net) if ".orig." not in k}
    if expected != set(delta):
        raise ValueError(
            f"Parent delta keys differ: missing={expected - set(delta)}, unexpected={set(delta) - expected}"
        )
    net.load_state_dict(delta, strict=False)


def verify_base(config, payload):
    if sha256_file(config.teacher_path) != payload["metadata"]["base_sha256"]:
        raise ValueError("Original base fingerprint differs from the checkpoint")


def latest_state(root):
    states = complete_checkpoints(root)
    return states[-1] if states else None


def load_state(path):
    path = Path(path)
    if path.is_dir():
        if path not in complete_checkpoints(path.parent.parent):
            raise ValueError(f"Incomplete checkpoint: {path}")
        path = path / "state.pt"
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if value.get("architecture") not in ("pid_vdn", "pixeldit_qdm") or value.get("schema") != 2:
        raise ValueError("Not a PiD-VDN checkpoint")
    return value


def save_state(root, net, optimizer, scheduler, config, stage, progress, metadata, keep=3, milestone=False):
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    rng = [None] * world
    if distributed:
        dist.all_gather_object(rng, capture_rng())
    else:
        rng[0] = capture_rng()
    root = Path(root)
    directory = root / "checkpoints"
    path = directory / f"step_{progress['stage_step']:09d}_batch_{progress['cursor']:09d}"
    generation = str(time.time_ns()) if rank == 0 else None
    if distributed:
        values = [generation]
        dist.broadcast_object_list(values, src=0)
        generation = values[0]
    path = path.with_name(path.name + "_gen_" + generation)
    error = None
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".incomplete")
            tmp.mkdir(exist_ok=True)
            payload = dict(
                schema=2,
                architecture="pixeldit_qdm" if config.backbone=="pixeldit_t2i" else "pid_vdn",
                delta=cpu_tree(delta_state(net)),
                optimizer=cpu_tree(optimizer.state_dict()),
                scheduler=scheduler.state_dict(),
                config=config.to_dict(),
                stage=stage,
                progress=progress,
                metadata=metadata,
                rng=rng,
                world_size=world,
            )
            with (tmp / "state.pt").open("wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            atomic_json(
                tmp / "complete.json", {"bytes": (tmp / "state.pt").stat().st_size, "step": progress["stage_step"]}
            )
            if milestone:
                (tmp / "KEEP").touch()
            os.replace(tmp, path)
            atomic_json(directory / "latest.json", {"checkpoint": str(path)})
            for old in complete_checkpoints(root)[:-keep] if keep > 0 else []:
                if not (old / "KEEP").exists():
                    shutil.rmtree(old)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if distributed:
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error:
        raise RuntimeError(f"Checkpoint publication failed: {error}")
    return path
