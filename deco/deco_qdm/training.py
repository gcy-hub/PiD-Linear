"""Native 512px DeCo distillation: layer alignment, prediction KD, then KD + LoRA."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .checkpoint import (ARCHITECTURE, atomic_json, fingerprint_file, latest_state, load_state,
                         read_json, restore_rng, save_state)

STAGES = ("a1", "a2", "b")
STAGE_STEPS = {"a1": 500, "a2": 1000, "b": 10000}
WARMUP = {"a1": 50, "a2": 50, "b": 200}
SAVE_EVERY = {"a1": 50, "a2": 100, "b": 2000}


def distributed():
    return dist.is_available() and dist.is_initialized()


def broadcast(value, rank=0):
    values = [value]
    if distributed():
        dist.broadcast_object_list(values, src=rank)
    return values[0]


def consensus(value, device):
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def init_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1 and not distributed():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return (dist.get_rank() if distributed() else 0), world, device


class GlobalBatchSampler(Sampler):
    """Recreate global epoch permutations; prefetch never changes the saved cursor.

    ``cursor`` counts *consumed global microbatches*. The trainer advances it after
    reading each batch, including a collective invalid-batch skip. Every rank draws
    a disjoint slice of exactly the same permutation stream.
    """
    def __init__(self, size, batch_size, rank=0, world_size=1, seed=42, cursor=0):
        if size < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid dataset size, batch size or distributed rank")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed, self.cursor = rank, world_size, seed, cursor

    def __iter__(self):
        cursor = self.cursor
        cached_epoch, permutation = None, None
        while True:
            start = cursor * self.batch_size * self.world_size + self.rank * self.batch_size
            batch = []
            for ordinal in range(start, start + self.batch_size):
                epoch, offset = divmod(ordinal, self.size)
                if epoch != cached_epoch:
                    generator = torch.Generator().manual_seed(self.seed + epoch)
                    permutation = torch.randperm(self.size, generator=generator).tolist()
                    cached_epoch = epoch
                batch.append(permutation[offset])
            yield batch
            cursor += 1

    def state_dict(self):
        return {"cursor": self.cursor, "size": self.size, "batch_size": self.batch_size,
                "world_size": self.world_size, "seed": self.seed}

    def load_state_dict(self, state, allow_batch_change=False):
        for key in ("size", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler {key} differs on resume")
        previous = state["batch_size"] * state["world_size"]
        current = self.batch_size * self.world_size
        if previous != current or state["world_size"] != self.world_size or state["batch_size"] != self.batch_size:
            if not allow_batch_change:
                raise ValueError("Global microbatch/world size differs; explicitly pass --allow-batch-change")
            # Keep the consumed sample position. A new batch grid may replay up to
            # current-1 examples; record this explicit approximation in run metadata.
            self.cursor = state["cursor"] * previous // current
        else:
            self.cursor = state["cursor"]


def lr_factor(step, total, warmup, minimum=0.05):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def sample_seed(seed, stage, ordinal):
    return int.from_bytes(hashlib.sha256(f"deco-fm-v1:{seed}:{stage}:{ordinal}".encode()).digest()[:8], "little") % (2**63 - 1)


def training_inputs(batch, empty_text, device, *, seed, stage, cursor, rank=0, world_size=1):
    """Native FM inputs shared verbatim by teacher and student.

    x_t=t*x+(1-t)*noise, with the official shift-4 timestep transform.
    Randomness is keyed by consumed global sample ordinal, independent of workers.
    """
    images = batch["image"].to(device, non_blocking=True).float()
    text = batch["text"].to(device, non_blocking=True)
    size = len(images)
    noise, timesteps, captions = [], [], []
    for i in range(size):
        ordinal = cursor * world_size * size + rank * size + i
        generator = torch.Generator(device=device).manual_seed(sample_seed(seed, stage, ordinal))
        ni = torch.randn(images[i].shape, generator=generator, device=device)
        ti = torch.randn((), generator=generator, device=device).sigmoid()
        ti = ti / (ti + (1 - ti) * 4)
        drop = torch.rand((), generator=generator, device=device) < 0.1
        captions.append(empty_text.to(device=device, dtype=text.dtype) if drop.item() else text[i])
        noise.append(ni)
        timesteps.append(ti)
    noise, timesteps = torch.stack(noise), torch.stack(timesteps)
    t = timesteps.reshape(-1, 1, 1, 1)
    return (t * images + (1 - t) * noise, timesteps, torch.stack(captions)), images - noise


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda", cache_enabled=False)


class AlignCallback:
    """Backward one attention layer immediately while the teacher keeps its outputs."""
    def __init__(self, scale=1.0):
        self.scale, self.metrics, self.failed = scale, {}, False
        self.elapsed = 0.0

    def __call__(self, module, x, y, pos, target, grid):
        started = time.perf_counter()
        with torch.enable_grad(), autocast(x.device):
            prediction = module.student_forward(x.detach(), y.detach(), pos, grid)
            # The callback receives the projected image attention branch output.
            targets = target if isinstance(target, (tuple, list)) else (target,)
            predictions = prediction if isinstance(prediction, (tuple, list)) else (prediction,)
            squared = sum((s.float() - t.detach().float()).square().flatten(1).sum(1)
                          for s, t in zip(predictions, targets))
            energy = sum(t.detach().float().square().flatten(1).sum(1) for t in targets).clamp_min(1e-12)
            # Average per-example relative losses, so the same effective batch
            # has the same objective on one, four or eight ranks.
            loss = (squared / energy).mean()
            finite = bool(torch.isfinite(loss).item())
            self.failed |= not finite
            if finite:
                (loss * self.scale).backward()
            name = getattr(module, "_deco_training_name", str(len(self.metrics)))
            self.metrics[f"alignment/{name}/relative_mse"] = loss.detach()
            for label, output, reference in zip(("image", "text"), predictions, targets):
                self.metrics[f"alignment/{name}/{label}_mse"] = (output.float() - reference.float()).square().mean().detach()
        self.elapsed += time.perf_counter() - started


def synchronize_a1(model):
    if distributed():
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(dist.get_world_size())


def initial_broadcast(model):
    if distributed():
        # Frozen base is independently loaded from a verified immutable file.
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def finite_gradients(model):
    checks = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
              if parameter.requires_grad and parameter.grad is not None]
    return bool(torch.stack(checks).all().item()) if checks else True


def clip_gradients(model, stage):
    if stage == "a1":
        # These are independent layer objectives; clipping the combined norm would
        # introduce unintended coupling between all sixteen alignment losses.
        norms = [torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad], 0.1)
                 for _, module in model.hybrid_modules()]
        return float(torch.stack([n.float() for n in norms]).norm().item()) if norms else 0.0
    return float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1).item())


def base_fingerprint(model, args):
    if args.tiny:
        return "tiny-random-seeded-v1"
    fingerprint = getattr(model, "base_fingerprint", None)
    if fingerprint:
        return fingerprint
    root = Path(args.weights_root)
    # Native DeCo loader exposes its exact source if available.
    path = getattr(model, "base_path", None) or root / "t2i_DeCo.ckpt"
    if Path(path).is_file():
        return fingerprint_file(path)
    candidates = list(root.rglob("*.pth")) + list(root.rglob("*.pt")) + list(root.rglob("*.safetensors"))
    candidates = [p for p in candidates if "deco" in str(p).lower() and "qwen" not in str(p).lower()]
    if len(candidates) != 1:
        raise ValueError("Model must expose base_fingerprint/base_path, or weights-root must identify one DeCo checkpoint")
    return fingerprint_file(candidates[0])


class Tracking:
    def __init__(self, args, stage_dir, stage, run_id, rank):
        self.run = None
        self.rank, self.stage_dir = rank, stage_dir
        if rank == 0 and args.swanlab_mode != "disabled":
            import swanlab
            self.run = swanlab.init(project="DeCo-QDM", name=f"{Path(args.output_root).name}-{stage}",
                                    id=run_id, resume="allow", mode="online" if args.swanlab_mode == "cloud" else args.swanlab_mode,
                                    log_dir=str(stage_dir / "swanlog"), config=vars(args))

    def log(self, values, step):
        if self.rank == 0:
            values = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in values.items()}
            with (self.stage_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step, **values), allow_nan=False) + "\n")
            if self.run:
                self.run.log(values, step=step)

    def log_images(self, paths, step):
        if self.rank == 0 and self.run:
            import swanlab
            images = [swanlab.Image(str(path), caption=path.stem) for path in paths if path.exists()]
            if images:
                self.run.log({"validation/comparisons": images}, step=step)

    def finish(self):
        if self.run:
            self.run.finish()


def stage_finished(stage_dir):
    value = read_json(Path(stage_dir) / "finished.json")
    if not value:
        return False
    checkpoint = Path(value["checkpoint"])
    marker = read_json(checkpoint / "complete.json")
    state = checkpoint / "state.pt"
    if not marker or not state.exists() or state.stat().st_size != marker.get("bytes"):
        raise ValueError(f"Finished stage refers to an incomplete checkpoint: {checkpoint}")
    if value.get("step") != marker.get("step"):
        raise ValueError("Finished marker and checkpoint step disagree")
    return True


def validate(model, args, stage, step, stage_dir, device, rank, stop=None):
    from .inference import validate_checkpoint
    if distributed():
        dist.barrier()
    error = None
    result = {}
    # Validation must not perturb a resumed run's next training input/RNG stream.
    from .checkpoint import capture_rng
    rng = capture_rng()
    was_training = model.training
    try:
        model.eval()
        configuration = dict(vars(args))
        configuration["_stop_requested"] = lambda: bool(stop and stop["requested"])
        result = validate_checkpoint(model=model, stage=stage, step=step, config=configuration,
                                     output_dir=stage_dir / "validation" / f"step-{step:09d}")
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        model.train(was_training)
        restore_rng(rng)
    errors = [error]
    if distributed():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"Validation incomplete; resume will retry: {errors}")
    if distributed():
        dist.barrier()
    result = result or {}
    result["validation/complete"] = consensus(result.get("validation/complete", True), device)
    if distributed():
        count = torch.tensor(result.get("validation/images", 0), device=device, dtype=torch.long)
        dist.all_reduce(count)
        result["validation/images"] = int(count.item())
    return result


def _lock_stage(stage_dir, rank):
    lock = None
    error = None
    if rank == 0:
        lock = (stage_dir / "training.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            error = f"A trainer already owns {stage_dir}"
    error = broadcast(error)
    if error:
        raise RuntimeError(error)
    return lock


def validation_due(args, stage, step, total):
    interval = getattr(args, f"{stage}_validate_every")
    return step == total or (interval > 0 and step % interval == 0)


def run_stage(args, stage, rank, world, device, stop, parent=None):
    from .data import TarImageTextDataset, collate_samples
    from .model import build_model, load_delta_state_dict, optimizer_groups
    stage_dir = Path(args.output_root).resolve() / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    lock = _lock_stage(stage_dir, rank)
    started = time.monotonic()
    total = getattr(args, f"{stage}_steps")
    checkpoint = latest_state(stage_dir) if args.resume == "auto" else None
    if args.resume not in ("auto", "none"):
        checkpoint = Path(args.resume)
    if args.resume == "none" and latest_state(stage_dir):
        raise ValueError("Existing checkpoints: select --resume auto or a new output root")
    payload = load_state(checkpoint) if checkpoint else None
    if payload and payload["stage"] != stage:
        raise ValueError("Resume stage mismatch; use --init-from for a parent stage")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = TarImageTextDataset(args.index_root, args.text_cache_root,
                                 resolution=args.resolution,
                                 allow_partial_data=args.allow_partial_data)
    sampler = GlobalBatchSampler(len(dataset), args.batch_size, rank, world, args.seed)
    model = build_model(args.weights_root, stage=stage, region_size=args.region_size, halo=args.halo,
                        device=device, checkpointing=args.activation_checkpointing, tiny=args.tiny,
                        full_attention_layers=args.full_attention_layers)
    model.train()
    for name, module in model.hybrid_modules():
        module._deco_training_name = name
    metadata = dict(base_fingerprint=broadcast(base_fingerprint(model, args) if rank == 0 else None),
                    data_fingerprint=dataset.fingerprint,
                    text_fingerprint=getattr(dataset, "text_fingerprint", None),
                    architecture=dict(name=ARCHITECTURE, region_size=args.region_size, halo=args.halo, tiny=args.tiny),
                    stage_data=dict(seed=args.seed, resolution=args.resolution,
                                    batch_size=args.batch_size, accumulation=args.accumulation, world_size=world,
                                    effective_batch=args.batch_size * args.accumulation * world),
                    schedule=dict(total=total, warmup=min(WARMUP[stage], total), base_lr=args.learning_rate,
                                  minimum_lr=args.minimum_lr),
                    parent_checkpoint=str(parent) if parent else args.init_from or None,
                    swanlab_run_id=broadcast(uuid.uuid4().hex if rank == 0 else None))
    # Preserve the historical metadata shape for all-Hybrid checkpoints/jobs.
    if model.full_attention_layers:
        metadata['architecture']['full_attention_layers'] = model.full_attention_layers
    model.base_fingerprint = metadata["base_fingerprint"]
    progress = dict(stage_step=0, cumulative_step=0, cursor=0, skipped_batches=0, validations=[])
    if payload:
        for key in ("base_fingerprint", "data_fingerprint", "text_fingerprint", "architecture"):
            if payload["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        if payload["metadata"]["stage_data"] != metadata["stage_data"]:
            old = payload["metadata"]["stage_data"]
            if old["seed"] != args.seed or old["resolution"] != metadata["stage_data"]["resolution"]:
                raise ValueError("Resume seed or resolution mismatch")
            if not args.allow_batch_change:
                raise ValueError("Resume batch/world/accumulation differs; use --allow-batch-change explicitly")
        if payload["metadata"]["schedule"] != metadata["schedule"]:
            if not args.allow_schedule_change:
                raise ValueError("Resume schedule differs; use --allow-schedule-change explicitly")
            if total < payload["progress"]["stage_step"]:
                raise ValueError("New total precedes the saved optimizer step")
            metadata["previous_schedule"] = payload["metadata"]["schedule"]
        metadata["swanlab_run_id"] = payload["metadata"]["swanlab_run_id"]
        metadata["parent_checkpoint"] = payload["metadata"].get("parent_checkpoint")
        load_delta_state_dict(model, payload["delta"])
        progress.update(payload["progress"])
        sampler.load_state_dict(payload["progress"]["sampler"], args.allow_batch_change)
        progress["cursor"] = sampler.cursor
        if payload["metadata"]["stage_data"] != metadata["stage_data"]:
            old_sampler = payload["progress"]["sampler"]
            metadata["batch_change"] = dict(previous=payload["metadata"]["stage_data"],
                current=metadata["stage_data"],
                replayed_samples=old_sampler["cursor"] * old_sampler["batch_size"] * old_sampler["world_size"]
                                 - sampler.cursor * args.batch_size * world)
    elif parent or args.init_from:
        parent_path = parent or args.init_from
        inherited = load_state(parent_path)
        expected_parent = {"a2": "a1", "b": "a2"}.get(stage)
        if inherited["stage"] != expected_parent:
            raise ValueError(f"Stage {stage} requires {expected_parent} parent")
        for key in ("base_fingerprint", "architecture", "data_fingerprint", "text_fingerprint"):
            if inherited["metadata"][key] != metadata[key]:
                raise ValueError(f"Parent {key} mismatch")
        load_delta_state_dict(model, inherited["delta"], strict=stage != "b")
        progress["cumulative_step"] = inherited["progress"]["cumulative_step"]
    elif stage != "a1":
        raise ValueError(f"Stage {stage} requires --init-from or a completed preceding stage")
    initial_broadcast(model)
    groups = optimizer_groups(model, stage, args.learning_rate)
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=0, eps=1e-8)
    warmup = min(WARMUP[stage], total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(
        step, total, warmup, args.minimum_lr / args.learning_rate))
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if "previous_schedule" in metadata:
            factor = lr_factor(scheduler.last_epoch, total, warmup, args.minimum_lr / args.learning_rate)
            scheduler._last_lr = [base * factor for base in scheduler.base_lrs]
            for group, lr in zip(optimizer.param_groups, scheduler._last_lr):
                group["lr"] = lr
        old_world = payload["world_size"]
        if world != old_world and not args.allow_batch_change:
            raise ValueError("Rank-specific RNG resume requires unchanged world size")
    tracker = Tracking(args, stage_dir, stage, metadata["swanlab_run_id"], rank)
    if rank == 0:
        atomic_json(stage_dir / "config.json", dict(vars(args), stage=stage))
        atomic_json(stage_dir / "run.json", metadata)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False, find_unused_parameters=False) if distributed() and stage != "a1" else model
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_samples, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed + rank),
                        **({"prefetch_factor": 2} if args.workers else {}))
    iterator = iter(loader)
    if payload:
        # Setup and DataLoader worker initialization must not consume the restored
        # stream. Per-sample input noise has its own stateless generators.
        restore_rng(payload["rng"][rank % payload["world_size"]])
    optimizer.zero_grad(set_to_none=True)
    accumulated, values = 0, {}
    last_checkpoint = checkpoint
    bar = tqdm(total=total, initial=progress["stage_step"], disable=rank != 0, desc=f"DeCo {stage.upper()}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot(milestone=False):
        progress["sampler"] = sampler.state_dict()
        return save_state(stage_dir, model, optimizer, scheduler, config=vars(args), stage=stage,
                          progress=progress, metadata=metadata, milestone=milestone)

    def run_validation():
        nonlocal last_checkpoint
        step = progress["stage_step"]
        if step in progress["validations"]:
            return True
        # Publish first, so a killed/failed gallery is pending on resume.
        progress["pending_validation"] = step
        last_checkpoint = snapshot()
        result = validate(model, args, stage, step, stage_dir, device, rank, stop=stop)
        if not result.get("validation/complete", True):
            last_checkpoint = snapshot()
            return False
        progress["pending_validation"] = None
        progress["validations"].append(step)
        last_checkpoint = snapshot(milestone=True)
        numeric = {(key if key.startswith("validation/") else f"validation/{key}"): value for key, value in result.items()
                   if isinstance(value, (int, float)) and math.isfinite(value)}
        tracker.log(numeric, progress["cumulative_step"])
        examples = sorted((stage_dir / "validation" / f"step-{step:09d}").glob("*.png" if args.validation_student_only else "*_comparison.png"))[:2]
        tracker.log_images(examples, progress["cumulative_step"])
        return True

    try:
        # A final update checkpoint can precede a killed validation. Retry it
        # before finishing, even when there are no optimizer steps remaining.
        if progress["stage_step"] > 0 and (validation_due(args, stage, progress["stage_step"], total) or
                                           progress.get("pending_validation") is not None):
            if not run_validation():
                return False, last_checkpoint
        while progress["stage_step"] < total:
            batch_started = time.perf_counter()
            batch = next(iterator)
            cursor = sampler.cursor
            sampler.cursor += 1
            progress["cursor"] = sampler.cursor
            if not consensus(batch.get("valid", True), device):
                progress["skipped_batches"] += 1
                errors = [dict(rank=rank, **entry) for entry in batch.get("errors", [])]
                if distributed():
                    all_errors = [None] * world
                    dist.all_gather_object(all_errors, errors)
                    errors = [entry for rank_errors in all_errors for entry in rank_errors]
                if rank == 0:
                    with (stage_dir / "invalid_batches.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(cursor=cursor, errors=errors)) + "\n")
                # Stop an all-invalid stream without manufacturing updates.
                if progress["skipped_batches"] > args.max_invalid_batches:
                    raise RuntimeError("Too many invalid global batches; repair the data/cache")
                continue
            empty_text = dataset.empty_text
            if args.tiny:
                # Native caches remain validated [128,2048] tensors; tiny models
                # use their leading token/channel block only in CPU test runs.
                batch["text"] = batch["text"][:, :8, :16].float()
                empty_text = empty_text[:8, :16].float()
            inputs, fm_target = training_inputs(batch, empty_text, device, seed=args.seed,
                                               stage=stage, cursor=cursor, rank=rank, world_size=world)
            teacher_started = time.perf_counter()
            callback = AlignCallback(1 / args.accumulation) if stage == "a1" else None
            with torch.no_grad(), autocast(device), model.execution(teacher=True, alignment_callback=callback):
                teacher = model(*inputs)
            teacher_time = time.perf_counter() - teacher_started
            teacher_finite = torch.isfinite(teacher).all().item()
            if not consensus(teacher_finite and not (callback and callback.failed), device):
                raise FloatingPointError("Nonfinite teacher prediction or A1 alignment on a rank")
            local = {"loss/fm_teacher": (teacher.float() - fm_target).square().mean().detach(),
                     "time/teacher_seconds": teacher_time}
            if stage == "a1":
                local.update(callback.metrics)
                local["loss/alignment"] = torch.stack([value for key, value in callback.metrics.items() if key.endswith("/relative_mse")]).mean() if callback.metrics else 0.0
                local["time/alignment_seconds"] = callback.elapsed
            else:
                synchronize = accumulated + 1 == args.accumulation
                sync = contextlib.nullcontext() if synchronize or not distributed() else wrapped.no_sync()
                student_started = time.perf_counter()
                # Keep the student context alive through checkpoint recomputation
                # in backward; switching it earlier silently invokes the teacher.
                with sync, model.execution(teacher=False), autocast(device):
                    student = wrapped(*inputs)
                    loss = (student.float() - teacher.detach().float()).square().mean()
                    if not consensus(torch.isfinite(loss).item(), device):
                        raise FloatingPointError("Nonfinite prediction KD on a rank")
                    (loss / args.accumulation).backward()
                local.update({"loss/kd": loss.detach(),
                              "loss/fm_student": (student.detach().float() - fm_target).square().mean(),
                              "time/student_seconds": time.perf_counter() - student_started})
            accumulated += 1
            local["time/batch_seconds"] = time.perf_counter() - batch_started
            for key, value in local.items():
                values[key] = values.get(key, 0.0) + (float(value.item()) if isinstance(value, torch.Tensor) else value) / args.accumulation
            if accumulated != args.accumulation:
                continue
            if stage == "a1":
                synchronize_a1(model)
            if not consensus(finite_gradients(model), device):
                raise FloatingPointError("Nonfinite gradients on a rank; refusing optimizer update")
            values["gradient/norm"] = clip_gradients(model, stage)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["stage_step"] += 1
            progress["cumulative_step"] += 1
            accumulated = 0
            bar.update(1)
            if distributed():
                keys = sorted(values)
                reduced = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
                dist.all_reduce(reduced)
                values = {key: float(value) / world for key, value in zip(keys, reduced.tolist())}
            values.update({"progress/stage_step": progress["stage_step"], "progress/cumulative_step": progress["cumulative_step"],
                           "progress/cursor": sampler.cursor, "progress/skipped_global_batches": progress["skipped_batches"],
                           "progress/effective_batch": args.batch_size * world * args.accumulation,
                           "time/elapsed_seconds": time.monotonic() - started,
                           "memory/peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0})
            for index, group in enumerate(optimizer.param_groups):
                values[f"lr/{group.get('group_name', index)}"] = group["lr"]
            if progress["stage_step"] % args.log_every == 0 or progress["stage_step"] == 1:
                tracker.log(values, progress["cumulative_step"])
                if rank == 0:
                    bar.set_postfix(loss=f"{values.get('loss/kd', values.get('loss/alignment', 0)):.4g}")
            values = {}
            should_validate = validation_due(args, stage, progress["stage_step"], total)
            if should_validate:
                if not run_validation():
                    return False, last_checkpoint
            elif getattr(args, f"{stage}_save_every") > 0 and progress["stage_step"] % getattr(args, f"{stage}_save_every") == 0:
                last_checkpoint = snapshot()
            should_stop = (stop["requested"] or (stage_dir / "STOP").exists() or
                           (Path(args.output_root) / "repeat/STOP").exists() or
                           (args.max_seconds > 0 and time.monotonic() - args.started_at >= args.max_seconds))
            if not consensus(not should_stop, device) and progress["stage_step"] < total:
                last_checkpoint = snapshot()
                return False, last_checkpoint
        # Finished means final validation succeeded and its state was published.
        if total not in progress["validations"]:
            if not run_validation():
                return False, last_checkpoint
        if rank == 0:
            atomic_json(stage_dir / "finished.json", dict(stage=stage, step=total, checkpoint=str(last_checkpoint),
                        validation=str(stage_dir / "validation" / f"step-{total:09d}")))
        return True, last_checkpoint
    finally:
        bar.close()
        tracker.finish()
        del iterator, loader
        if distributed():
            # Normal-path cleanup is coordinated by main. Avoid barriers here:
            # on an exception other ranks may be inside a failed collective.
            pass
        del wrapped, model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if lock:
            lock.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    p.add_argument("--weights-root", default=os.path.expanduser("~/huggingface_ckpts/DeCo"))
    p.add_argument("--index-root", default=os.path.expanduser("~/dataset/BLIP-3o/BLIP3o-60k/deco_qdm_cache/index"))
    p.add_argument("--text-cache-root", default=os.path.expanduser("~/dataset/BLIP-3o/BLIP3o-60k/deco_qdm_cache/text"))
    p.add_argument("--output-root", default="outputs/deco-qdm")
    p.add_argument("--resume", default="auto", help="auto, none, or a complete checkpoint directory")
    p.add_argument("--init-from", default="", help="preceding stage checkpoint")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", "--grad-accum", dest="accumulation", type=int, default=1)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--halo", type=int, default=2)
    p.add_argument("--full-attention-layers", default="", help="Comma-separated zero-based native image Attention layers to retain frozen; default replaces all")
    for stage in STAGES:
        p.add_argument(f"--{stage}-steps", type=int, default=STAGE_STEPS[stage])
        p.add_argument(f"--{stage}-save-every", type=int, default=SAVE_EVERY[stage])
        p.add_argument(f"--{stage}-validate-every", type=int, default=2000 if stage == "b" else 0)
    p.add_argument("--validation-student-only", action="store_true", help="Generate the ten fixed student images without teacher comparisons")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--minimum-lr", type=float, default=5e-6)
    p.add_argument("--max-seconds", type=float, default=0, help="save/exit at the next update boundary; 0 means unlimited")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-invalid-batches", type=int, default=1000)
    p.add_argument("--allow-batch-change", action="store_true")
    p.add_argument("--allow-schedule-change", action="store_true")
    p.add_argument("--allow-partial-data", action="store_true")
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--swanlab-mode", choices=("offline", "online", "cloud", "disabled"), default="offline")
    p.add_argument("--tiny", action="store_true", help="CPU unit-test model; never loads native weights")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.batch_size, args.accumulation, args.threads, args.log_every, args.a1_steps, args.a2_steps, args.b_steps) < 1:
        raise ValueError("Batches, accumulation, threads, logging and stage steps must be positive")
    if args.workers < 0 or args.max_seconds < 0 or args.max_invalid_batches < 0:
        raise ValueError("Workers and time/invalid limits cannot be negative")
    if args.resolution < 2 or args.resolution % (2 if args.tiny else 16):
        raise ValueError("Resolution must be positive and divisible by the model patch size")
    if not args.tiny and args.resolution != 512:
        raise ValueError("The native training baseline requires --resolution 512")
    if args.halo != 2:
        raise ValueError("This baseline requires local halo=2")
    if not 0 < args.minimum_lr <= args.learning_rate:
        raise ValueError("Expected 0 < minimum-lr <= learning-rate")
    if any(getattr(args, f"{stage}_{kind}_every") < 0 for stage in STAGES for kind in ("save", "validate")):
        raise ValueError("Save and validation intervals must be nonnegative")
    args.started_at = time.monotonic()
    torch.set_num_threads(args.threads)
    rank, world, device = init_distributed()
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    parent = None
    try:
        stages = STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            stage_dir = Path(args.output_root).resolve() / stage
            if stage_finished(stage_dir):
                parent = Path(read_json(stage_dir / "finished.json")["checkpoint"])
                continue
            if parent is None and stage != "a1" and not args.init_from:
                previous = {"a2": "a1", "b": "a2"}[stage]
                marker = read_json(Path(args.output_root) / previous / "finished.json")
                if marker:
                    parent = Path(marker["checkpoint"])
            finished, parent = run_stage(args, stage, rank, world, device, stop, parent=parent)
            if not finished:
                break
            if distributed():
                dist.barrier()
            if stop["requested"] or (args.max_seconds > 0 and time.monotonic() - args.started_at >= args.max_seconds):
                break
    finally:
        if distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
