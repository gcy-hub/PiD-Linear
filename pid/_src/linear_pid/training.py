"""DDP Attention conversion, independent of PiD's DMD few-step distillation trainer."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import logging
import os
import random
import signal
import time
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from pid._src.configs.linear_pid.config import LinearPiDConfig
from pid._src.linear_pid.attention import FLA_COMMIT, convert_attention, resolve_layers
from pid._src.linear_pid.pit_attention import convert_pit_attention
from pid._src.linear_pid.checkpoint import (
    complete_checkpoints,
    load_checkpoint,
    resolve_checkpoint,
    restore_rng,
    save_checkpoint,
    student_weights,
    validate_resume,
)
from pid._src.linear_pid.data import GlobalBatchSampler, RawImageDataset, atomic_json, collate_raw, sha256_file
from pid._src.linear_pid.environment import check_runtime
from pid._src.linear_pid.runtime import Conditioning, build_net, freeze_student, load_original, optimizer_groups
from pid._src.models.latent_noising import LatentNoiser, LatentNoisingConfig
from pid._src.networks.flow_matching import FlowMatchingTrainer

LOG = logging.getLogger("linear-pid")


def init_distributed():
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Linear-PiD training requires CUDA")
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return (dist.get_rank() if dist.is_initialized() else 0), world, torch.device("cuda", local)


def consensus(value, device):
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def broadcast_object(value, rank):
    if dist.is_initialized():
        objects = [value if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]
    return value


def lock_training_run(run_dir, rank):
    """Rank zero holds the run lock until exit, including orphaned-launcher cases."""
    lock, error = None, None
    if rank == 0:
        lock = (Path(run_dir) / ".train.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # DataLoader forks must not keep the lock alive after rank zero dies.
            os.register_at_fork(after_in_child=lock.close)
        except BlockingIOError:
            lock.close()
            lock = None
            error = f"Another trainer already owns {run_dir}"
    error = broadcast_object(error, rank)
    if error:
        raise RuntimeError(error)
    return lock


def build_training_teacher(config, device):
    """Only positive output supervision needs a resident teacher network."""
    if config.lambda_out == 0:
        return None
    teacher = build_net()
    load_original(teacher, config.teacher_path)
    return teacher.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def recovery_losses(prediction, x0, noise, reference=None, lambda_out=0.0):
    """Raw PiD velocity targets remain noise-x0 with or without distillation."""
    fm_loss = (prediction.float() - (noise - x0)).square().mean()
    if lambda_out == 0:
        return fm_loss, fm_loss, fm_loss.new_zeros(())
    if reference is None:
        raise ValueError("Positive --lambda-out requires a teacher prediction")
    kd_loss = (prediction.float() - reference.float()).square().mean()
    return fm_loss + lambda_out * kd_loss, fm_loss, kd_loss


def batch_loss_scale(local_batch, world, accum, effective_batch):
    """Undo DDP's equal-rank average when ranks receive unequal image counts."""
    return local_batch * world * accum / effective_batch


def resolve_training_resume(config, run_dir):
    """Continue the destination first; otherwise fork the full source state once."""
    checkpoint = resolve_checkpoint(config.resume, run_dir)
    if checkpoint:
        return str(checkpoint), False
    if complete_checkpoints(run_dir):
        raise ValueError("Destination already contains checkpoints; resume it or choose a fresh --output-root")
    if config.resume_from:
        checkpoint = resolve_checkpoint(config.resume_from)
        if checkpoint is None:
            raise ValueError("--resume-from requires a complete source checkpoint")
        if Path(checkpoint).resolve().parent.parent == Path(run_dir).resolve():
            raise ValueError("--resume-from requires a different destination; use --resume for this run")
        return str(checkpoint), True
    return "", False


def resume_identity(payload, checkpoint_path, forked):
    if forked:
        return checkpoint_path, uuid.uuid4().hex
    if payload:
        return payload["metadata"]["parent_checkpoint"], payload["metadata"]["swanlab_run_id"]
    return "", uuid.uuid4().hex


def resume_batch_transitions(payload, config, effective_batch, forked, checkpoint_path=None):
    """Record an explicitly allowed batch change without resetting training state."""
    history = list(payload["metadata"].get("batch_transitions", [])) if payload else []
    if payload and payload["metadata"]["effective_batch"] != effective_batch:
        if not config.allow_batch_size_change:
            raise ValueError("Resuming requires the same effective batch unless "
                             "--allow-batch-size-change is supplied")
        previous = payload["metadata"]["effective_batch"]
        history.append({"from": previous, "to": effective_batch,
                        "total_step": payload["progress"]["total_step"],
                        "source_checkpoint": checkpoint_path or (config.resume_from if forked else config.resume)})
        LOG.warning("Continuation changes effective batch %d -> %d; "
                    "optimizer/scheduler retained, subsequent sample groups change", previous, effective_batch)
    return history


def interval_due(step, samples_seen, previous_samples, train_size, steps, epochs):
    """Trigger once when an update hits a step boundary or crosses an epoch boundary."""
    step_due = bool(steps and step > 0 and step % steps == 0)
    epoch_due = bool(
        epochs and samples_seen // (train_size * epochs) > previous_samples // (train_size * epochs)
    )
    return step_due or epoch_due


def maintenance_due(config, step, samples_seen, previous_samples, train_size, skipped, device):
    """All ranks save/evaluate at successful optimizer-update boundaries."""
    validation_due = not skipped and interval_due(
        step, samples_seen, previous_samples, train_size, config.validation_steps, config.validation_epochs
    )
    validation_due = not consensus(not validation_due, device)
    save_due = validation_due or (not skipped and interval_due(
        step, samples_seen, previous_samples, train_size, config.save_steps, config.save_epochs
    ))
    # Publish the evaluated weights before rendering, so interrupted galleries can resume.
    return not consensus(not save_due, device), validation_due


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pid/_src/configs/linear_pid/config.py")
    parser.add_argument("--preset", choices=["4gpu", "8gpu"], default="4gpu")
    parser.add_argument(
        "--layers", "--kda-layers", default="4", help="Preset count or comma-separated zero-based indices"
    )
    for name in [
        "weights-root",
        "index-root",
        "text-cache-root",
        "output-root",
        "gallery-root",
        "resume",
        "resume-from",
        "init-from",
        "init-weights",
        "swanlab-mode",
    ]:
        parser.add_argument("--" + name, default=None)
    for name in [
        "batch-size",
        "grad-accum",
        "effective-batch",
        "workers",
        "threads",
        "pit-chunk-size",
        "pit-kda-heads",
        "seed",
        "warmup",
        "save-steps",
        "save-epochs",
        "keep-checkpoints",
        "max-steps",
        "max-seconds",
        "log-steps",
        "validation-steps",
        "validation-epochs",
    ]:
        parser.add_argument("--" + name, type=int, default=None)
    for name in ["lr-new", "lr-backbone", "weight-decay", "lambda-out", "grad-clip"]:
        parser.add_argument(
            "--" + name,
            type=float,
            default=None,
            help="Teacher output supervision weight; 0 skips teacher loading and forward (default)"
            if name == "lambda-out"
            else None,
        )
    parser.add_argument("--no-local-mixing", action="store_true")
    parser.add_argument("--pit-kda", action="store_true", help="Replace PiT Full Attention with KDA, retaining compression")
    parser.add_argument("--pit-kda-uncompressed", action="store_true", help="Use the earlier 4096-dimensional PiT variant")
    parser.add_argument("--allow-batch-size-change", action="store_true",
                        help="Allow and record an effective batch change while restoring the full training state")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument(
        "--online-text", action="store_true", help="Encode captions with a GPU-resident Gemma instead of the disk cache"
    )
    parser.add_argument(
        "--no-offload-text-encoder", action="store_true", help="Compatibility flag; models always stay on GPU"
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="Validate configuration/local files without CUDA or training"
    )
    args = parser.parse_args([arg for arg in (argv if argv is not None else os.sys.argv[1:]) if arg != "--"])
    config = LinearPiDConfig(
        preset=args.preset, layers=resolve_layers(args.layers), grad_accum=1 if args.preset == "8gpu" else 2
    )
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key) and key not in {"layers", "preset"}:
            setattr(config, key, value)
    # Keep the already measured small-batch path; bound PiT memory for large batches.
    if args.pit_chunk_size is None and config.batch_size >= 4:
        config.pit_chunk_size = 2048
    config.local_mixing = not args.no_local_mixing
    config.activation_checkpointing = not args.no_activation_checkpointing
    config.pit_kda_compressed = not args.pit_kda_uncompressed
    if args.pit_kda_uncompressed and not config.pit_kda:
        parser.error("--pit-kda-uncompressed requires --pit-kda")
    if args.pit_kda_heads is None and args.pit_kda_uncompressed:
        config.pit_kda_heads = 64
    if args.online_text:
        config.text_cache_root = ""
    if config.init_weights not in {"ema", "raw"} or config.swanlab_mode not in {"offline", "cloud", "disabled"}:
        parser.error("init-weights must be ema/raw and swanlab-mode offline/cloud/disabled")
    positive = ["batch_size", "grad_accum", "threads", "keep_checkpoints", "log_steps"]
    if any(getattr(config, key) <= 0 for key in positive) or config.workers < 0 or config.warmup < 0:
        parser.error("Invalid batch, thread, checkpoint, logging, worker, or warmup value")
    if (
        min(
            config.max_steps,
            config.max_seconds,
            config.validation_steps,
            config.validation_epochs,
            config.lambda_out,
            config.weight_decay,
            config.pit_chunk_size,
            config.effective_batch,
            config.save_steps,
            config.save_epochs,
        )
        < 0
    ):
        parser.error("Limits, gallery intervals, distillation weight and weight decay must be nonnegative")
    if min(config.lr_new, config.lr_backbone, config.grad_clip) <= 0:
        parser.error("Learning rates and clip norm must be positive")
    pit_dim = 1152 if config.pit_kda_compressed else 4096
    if config.pit_kda and (config.pit_kda_heads <= 0 or pit_dim % config.pit_kda_heads
                           or pit_dim // config.pit_kda_heads > 256 or (pit_dim // config.pit_kda_heads) % 4):
        parser.error(f"PiT KDA heads must divide {pit_dim} with head_dim divisible by 4 and <=256")
    if config.init_from and config.resume not in {"auto", "none", ""}:
        parser.error("--init-from cannot be combined with an explicit --resume")
    if config.resume_from and (config.init_from or config.resume not in {"auto", "none", ""}):
        parser.error("--resume-from cannot be combined with --init-from or an explicit --resume")
    return config, args.dryrun


def preflight(config):
    paths = [
        config.teacher_path,
        str(Path(config.weights_root) / "PiD/checkpoints/ae.safetensors"),
        str(Path(config.index_root) / "index.json"),
    ]
    if config.text_cache_root:
        from pid._src.linear_pid.text_cache import TextCacheReader

        TextCacheReader(config.text_cache_root)
    else:
        paths.append(str(Path(config.weights_root) / "gemma-2-2b-it/config.json"))
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if (config.validation_steps or config.validation_epochs) and not (Path(config.gallery_root) / "assets.json").exists():
        raise FileNotFoundError(
            "Prepare the fixed gallery first with scripts/prepare_linear_pid_assets.py "
            "(or set --validation-steps 0 --validation-epochs 0)"
        )
    print(json.dumps(config.to_dict(), indent=2))


def main(argv=None):
    job_start = time.monotonic()
    config, dryrun = parse_args(argv)
    preflight(config)
    if dryrun:
        return
    versions = check_runtime()
    rank, world, device = init_distributed()
    torch.set_num_threads(config.threads)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = lock_training_run(run_dir, rank)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / f"rank_{rank}.log")],
    )
    dataset = RawImageDataset(config.index_root, text_cache_root=config.text_cache_root)
    fingerprint = dataset.metadata["fingerprint"]
    text_fingerprint = dataset.text_cache.fingerprint if dataset.text_cache else None
    resume_result, resume_error = None, None
    if rank == 0:
        try:
            resume_result = resolve_training_resume(config, run_dir)
        except Exception as exc:
            resume_error = f"{type(exc).__name__}: {exc}"
    resume_result, resume_error = broadcast_object((resume_result, resume_error), rank)
    if resume_error:
        raise RuntimeError(resume_error)
    checkpoint_path, forked = resume_result
    if config.init_from and checkpoint_path:
        raise ValueError("Destination stage already has checkpoints; use --resume none with --init-from")
    payload = load_checkpoint(checkpoint_path) if checkpoint_path else None
    parent = load_checkpoint(resolve_checkpoint(config.init_from)) if config.init_from else None
    if payload:
        validate_resume(payload, config, fingerprint)
        if forked:
            LOG.info("Forking complete training state from %s into %s", checkpoint_path, run_dir)
        saved_text_fingerprint = payload["metadata"].get("text_cache_fingerprint")
        if saved_text_fingerprint and saved_text_fingerprint != text_fingerprint:
            raise ValueError("Text cache changed since this checkpoint")
    net = build_net()
    if payload or parent:
        source = payload or parent
        previous_layers = source["config"]["layers"]
        if not set(previous_layers).issubset(config.layers):
            raise ValueError("A new stage must retain all existing KDA layers")
        if source["config"]["local_mixing"] != config.local_mixing:
            raise ValueError("Cannot change the local mixing architecture while reusing a student")
        convert_attention(net, previous_layers, local_mixing=config.local_mixing)
        if source["config"].get("pit_kda", False):
            if (not config.pit_kda or source["config"].get("pit_kda_heads", 64) != config.pit_kda_heads
                    or source["config"].get("pit_kda_compressed", False) != config.pit_kda_compressed):
                raise ValueError("A new stage must retain the existing PiT KDA architecture")
            convert_pit_attention(net, heads=config.pit_kda_heads, compressed=config.pit_kda_compressed)
        weights = student_weights(source, "raw" if payload else config.init_weights)
        net.load_state_dict(weights, strict=True)
        convert_attention(net, config.layers, local_mixing=config.local_mixing)
    else:
        load_original(net, config.teacher_path)
        convert_attention(net, config.layers, local_mixing=config.local_mixing)
    if config.pit_kda:
        convert_pit_attention(net, heads=config.pit_kda_heads, compressed=config.pit_kda_compressed)
    freeze_student(net)
    net.activation_checkpointing = config.activation_checkpointing
    net.pit_chunk_size = config.pit_chunk_size
    net = net.to(device)
    # DDP synchronizes randomly initialized KDA parameters before the first update.
    model = (
        DistributedDataParallel(
            net,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
        if world > 1
        else net
    )
    optimizer = torch.optim.AdamW(optimizer_groups(net, config), betas=(0.9, 0.999), eps=1e-8, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, config.warmup)))
    progress = {
        "stage_step": 0, "total_step": 0, "cursor": 0, "samples_seen": 0,
        "skipped_batches": 0, "validation_pending": False,
    }
    if parent:
        progress.update(parent["progress"])
        progress["stage_step"] = 0
        progress["validation_pending"] = False
    if payload:
        progress.update(payload["progress"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        LOG.info(
            "Restored stage=%d total=%d cursor=%d optimizer_states=%d scheduler_step=%d",
            progress["stage_step"],
            progress["total_step"],
            progress["cursor"],
            len(optimizer.state),
            scheduler.last_epoch,
        )
    effective_batch = config.effective_batch or config.batch_size * config.grad_accum * world
    batch_transitions = resume_batch_transitions(payload, config, effective_batch, forked, checkpoint_path)
    teacher_hash = broadcast_object(sha256_file(config.teacher_path) if rank == 0 else None, rank)
    if payload and payload["metadata"]["teacher_sha256"] != teacher_hash:
        raise ValueError("Original PiD initialization/reference checkpoint changed since checkpoint")
    parent_checkpoint, swanlab_run_id = broadcast_object(
        resume_identity(payload, checkpoint_path, forked) if rank == 0 else None, rank
    )
    metadata = {
        "teacher_sha256": teacher_hash,
        "teacher_path": config.teacher_path,
        "data_fingerprint": fingerprint,
        "text_cache_fingerprint": text_fingerprint,
        "fla_commit": FLA_COMMIT,
        "parent_checkpoint": parent_checkpoint if payload else config.init_from,
        "effective_batch": effective_batch,
        "batch_transitions": batch_transitions,
        "ema_enabled": False,
        "model_offload": False,
        "teacher_enabled": config.lambda_out > 0,
        "training_objective": "fm+output_mse" if config.lambda_out > 0 else "fm",
        "swanlab_run_id": swanlab_run_id,
    }
    if parent and parent["metadata"]["teacher_sha256"] != teacher_hash:
        raise ValueError("A new stage must keep the original PiD initialization/reference checkpoint")
    if parent and parent["metadata"]["data_fingerprint"] != fingerprint:
        raise ValueError("A new stage must retain the same prepared dataset index")
    metadata["runtime_versions"] = versions
    teacher = build_training_teacher(config, device)
    conditioning = Conditioning(config.weights_root, device, text_cache_root=config.text_cache_root)
    LOG.info(
        "Models stay on GPU; EMA disabled; teacher %s; Gemma %s",
        "enabled" if teacher is not None else "disabled (FM loss only)",
        "replaced by disk embeddings" if config.text_cache_root else "resident on GPU",
    )
    LOG.info("PiT local chunk size=%d; CUDA_VISIBLE_DEVICES=%s", config.pit_chunk_size, os.getenv("CUDA_VISIBLE_DEVICES", "all"))
    if payload and payload["world_size"] == world:
        restore_rng(payload["rng"][rank])
    elif payload:
        continued_seed = config.seed + rank + 1_000_003 * progress["total_step"]
        random.seed(continued_seed)
        np.random.seed(continued_seed % 2**32)
        torch.manual_seed(continued_seed)
        LOG.info("Changed world size: optimizer/data progress restored; per-rank RNG reseeded")
    if payload or parent:
        del source, weights
    del payload, parent
    gc.collect()
    sampler = GlobalBatchSampler(
        config.index_root, config.batch_size, config.grad_accum, rank, world, config.seed, progress["cursor"],
        effective_batch=effective_batch,
    )
    LOG.info("Per-rank micro-batches=%s; accumulation=%d; effective batch=%d",
             sampler.rank_batch_sizes, config.grad_accum, effective_batch)
    for bucket, count in zip(sampler.buckets, sampler.probabilities):
        if count > 0 and len(bucket) < effective_batch:
            raise ValueError("A populated bucket is smaller than effective batch")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.workers,
        collate_fn=collate_raw,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        **({"prefetch_factor": 2} if config.workers else {}),
    )
    # Isolate DataLoader base-seed creation from model RNG across continuation.
    loader.generator = torch.Generator().manual_seed(config.seed)
    iterator = iter(loader)
    tracker = None
    if rank == 0 and config.swanlab_mode != "disabled":
        import swanlab

        tracker = swanlab
        swanlab.init(
            project="Linear-PiD",
            experiment_name=run_dir.name,
            config=config.to_dict(),
            mode="online" if config.swanlab_mode == "cloud" else config.swanlab_mode,
            log_dir=str(run_dir / "swanlog"),
            id=metadata["swanlab_run_id"],
            resume="allow",
        )
    if rank == 0:
        atomic_json(run_dir / "config.json", config.to_dict())
        atomic_json(run_dir / "metadata.json", metadata)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, request_stop)
    fm = FlowMatchingTrainer(timescale=1000.0, t_sampler_type="logit_normal", prediction_type="velocity")
    noiser = LatentNoiser(
        LatentNoisingConfig(enabled=True, backbone="flow_matching", add_sigma_min=0.0, add_sigma_max=0.8)
    )
    start = job_start
    torch.cuda.reset_peak_memory_stats(device)
    bar = tqdm(
        initial=progress["stage_step"], total=config.max_steps or None, desc="Optimizer updates", disable=rank != 0
    )

    def gallery():
        if not (config.validation_steps or config.validation_epochs):
            progress["validation_pending"] = False
            return
        from pid._src.linear_pid.evaluation import training_gallery

        # Rendering may be interrupted/resumed; it must not advance training RNG.
        with torch.random.fork_rng(devices=[device.index]):
            training_gallery(net, teacher, config, progress["total_step"], True, rank, world, tracker, metadata)
        if dist.is_initialized():
            dist.barrier()
        progress["validation_pending"] = False

    def interrupted():
        external = os.environ.get("LINEAR_PID_STOP_FILE")
        return (
            stop["requested"]
            or (run_dir / "STOP").exists()
            or (bool(external) and Path(external).exists())
            or (config.max_seconds and time.monotonic() - start >= config.max_seconds)
        )

    def requested_stop():
        return interrupted() or (config.max_steps and progress["stage_step"] >= config.max_steps)

    net.train()
    if progress["validation_pending"] and consensus(not interrupted(), device):
        gallery()
    while True:
        if not consensus(not requested_stop(), device):
            save_checkpoint(run_dir, net, optimizer, scheduler, config, progress, metadata, config.keep_checkpoints)
            break
        # Keep DDP bucket gradient views across accumulation windows. Clearing
        # them to None makes no_sync allocate a second full set of FP32 grads.
        optimizer.zero_grad(set_to_none=False)
        previous_samples = progress["samples_seen"]
        skipped, fm_value, kd_value, teacher_ms, student_ms = False, 0.0, 0.0, 0.0, 0.0
        h, w = 0, 0
        tick = time.monotonic()
        for micro in range(config.grad_accum):
            batch = next(iterator)
            valid = consensus(not batch["errors"], device)
            if not valid or skipped:
                if batch["errors"]:
                    with (run_dir / f"rejected_rank_{rank}.jsonl").open("a") as stream:
                        stream.write(json.dumps({"cursor": progress["cursor"], "errors": batch["errors"]}) + "\n")
                skipped = True
                continue
            x0 = batch["image"].to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
            with torch.no_grad():
                latent, _ = conditioning.encode_image(x0)
                latent, sigma = noiser(latent)
                captions = ["" if torch.rand((), device=device).item() < 0.1 else c for c in batch["caption"]]
                emb, mask = conditioning.encode_text(
                    captions, cached_embs=batch.get("text_embs"), cached_mask=batch.get("text_mask")
                )
                keep = (torch.rand(x0.shape[0], device=device) >= 0.1).view(-1, 1, 1, 1)
                latent = latent * keep
                t = fm.sample_t(x0.shape[0], device=device)
                h, w = x0.shape[-2:]
                shift = 6.0 * (math_sqrt_area(h, w) / 2048.0) ** 0.5
                t = shift * t / (1 + (shift - 1) * t)
                noisy, noise, _ = fm.add_noise(x0, t)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            reference = None
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    reference = teacher(noisy, t * 1000.0, emb, lq_latent=latent, degrade_sigma=sigma)
            events[1].record()
            sync = model.no_sync() if world > 1 and micro < config.grad_accum - 1 else contextlib.nullcontext()
            with sync:
                events[2].record()
                # Checkpointed blocks discard their intermediates; the autocast
                # weight cache would retain all BF16 student weights until PiT.
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    prediction = model(
                        noisy, t * 1000.0, emb, lq_latent=latent, degrade_sigma=sigma, text_valid_mask=mask
                    )
                # PiD passes -raw_output to FlowMatchingTrainer (target x0-noise).
                # Supervise raw network velocity with noise-x0, as in the released model.
                loss, fm_loss, kd_loss = recovery_losses(prediction, x0, noise, reference, config.lambda_out)
                if not consensus(torch.isfinite(loss).item(), device):
                    raise FloatingPointError(
                        "Non-finite loss on at least one rank; restart from the last complete checkpoint"
                    )
                loss_scale = batch_loss_scale(x0.shape[0], world, config.grad_accum, effective_batch)
                (loss * loss_scale / config.grad_accum).backward()
                events[3].record()
            events[3].synchronize()
            if teacher is not None:
                teacher_ms += events[0].elapsed_time(events[1])
            student_ms += events[2].elapsed_time(events[3])
            fm_value += fm_loss.detach().item() * loss_scale / config.grad_accum
            kd_value += kd_loss.detach().item() * loss_scale / config.grad_accum
            del reference, prediction, loss, fm_loss, kd_loss, x0, noisy, noise, latent, emb
        progress["cursor"] += 1
        norm = (
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            if not skipped
            else torch.tensor(0.0, device=device)
        )
        if skipped or not consensus(torch.isfinite(norm).item(), device):
            skipped = True
            progress["skipped_batches"] += 1
            optimizer.zero_grad(set_to_none=False)
        else:
            optimizer.step()
            scheduler.step()
            progress["stage_step"] += 1
            progress["total_step"] += 1
            progress["samples_seen"] += effective_batch
            bar.update()
        # Include the actual fused AdamW GPU update, rather than just its launch.
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - tick
        if progress["cursor"] % config.log_steps == 0:
            values = torch.tensor([fm_value, kd_value], device=device)
            if world > 1:
                dist.all_reduce(values)
                values /= world
            peaks = torch.tensor([elapsed, torch.cuda.max_memory_allocated(device) / 2**30], device=device)
            timing = torch.tensor([teacher_ms, student_ms], device=device)
            if world > 1:
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
                dist.all_reduce(timing)
                timing /= world
            elapsed, peak_memory = peaks.tolist()
            teacher_ms, student_ms = timing.tolist()
            info = {
                "train/fm_loss": values[0].item(),
                "train/output_mse": values[1].item(),
                "train/grad_norm": norm.item(),
                "train/samples_seen": progress["samples_seen"],
                "train/sample_passes": progress["samples_seen"] / dataset.metadata["train_size"],
                "train/epoch": progress["samples_seen"] / dataset.metadata["train_size"],
                "train/images_per_second": effective_batch / elapsed if not skipped else 0.0,
                "train/step_seconds": elapsed,
                "train/teacher_ms": teacher_ms,
                "train/student_forward_backward_ms": student_ms,
                "train/peak_memory_gib": peak_memory,
                "efficiency/updates_6d_excluding_galleries_io": 516600 / elapsed if not skipped else 0.0,
                "train/skipped_batches": progress["skipped_batches"],
                "train/height": h,
                "train/width": w,
                "stage/step": progress["stage_step"],
            }
            for i, group in enumerate(optimizer.param_groups):
                info[f"lr/{group['group_name']}_{i}"] = group["lr"]
            for i in config.layers:
                info.update({f"kda/{i}/{k}": v for k, v in net.patch_blocks[i].attn.last_stats.items()})
            if rank == 0:
                with (run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": progress["total_step"], **info}) + "\n")
                LOG.info(
                    "step=%d FM=%.5f KD=%.5f %.2fs %.2fGiB",
                    progress["stage_step"],
                    *values.tolist(),
                    elapsed,
                    info["train/peak_memory_gib"],
                )
                if tracker:
                    tracker.log(info, step=progress["total_step"])
            bar.set_postfix(fm=round(values[0].item(), 4), kd=round(values[1].item(), 4))
        for i in config.layers:
            net.patch_blocks[i].attn.collect_stats = (progress["cursor"] + 1) % config.log_steps == 0
        should_stop = requested_stop()
        should_stop = not consensus(not should_stop, device)
        save_due, validation_due = maintenance_due(
            config, progress["stage_step"], progress["samples_seen"], previous_samples,
            dataset.metadata["train_size"], skipped, device,
        )
        if validation_due:
            progress["validation_pending"] = True
        if save_due or should_stop:
            saved = save_checkpoint(
                run_dir, net, optimizer, scheduler, config, progress, metadata, config.keep_checkpoints
            )
            LOG.info("Complete checkpoint: %s", saved)
        if validation_due and consensus(not interrupted(), device):
            gallery()
        if should_stop:
            break
    bar.close()
    if tracker:
        tracker.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    if run_lock is not None:
        run_lock.close()


def math_sqrt_area(h, w):
    return (h * w) ** 0.5
