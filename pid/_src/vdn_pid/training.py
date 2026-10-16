"""A1 layer hooks, A2 prediction KD, B KD+LoRA; a single immutable base on GPU."""

import argparse
import contextlib
import gc
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from pid._src.linear_pid.checkpoint import restore_rng
from pid._src.linear_pid.data import GlobalBatchSampler, RawImageDataset, atomic_json, collate_raw, sha256_file
from pid._src.linear_pid.runtime import Conditioning, build_net, load_original
from pid._src.linear_pid.training import broadcast_object, consensus, init_distributed, lock_training_run
from pid._src.models.latent_noising import LatentNoiser, LatentNoisingConfig
from pid._src.networks.flow_matching import FlowMatchingTrainer

from .attention import convert_attention, execution_mode
from .checkpoint import latest_state, load_delta, load_parent_delta, load_state, save_state
from .config import STAGES, VDNConfig, resume_metadata
from .lora import configure_stage, optimizer_groups


class AlignCallback:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.metrics = {}
        self.student_ms = 0.0

    def __call__(self, module, args, kwargs, target):
        timing = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if args[0].is_cuda else None
        if timing:
            timing[0].record()
        args = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args)
        with (
            torch.enable_grad(),
            torch.autocast(
                device_type=args[0].device.type, dtype=torch.bfloat16, enabled=args[0].is_cuda, cache_enabled=False
            ),
        ):
            prediction = module.student_forward(*args, **kwargs)
            tx, ty = [t.detach().float() for t in target]
            sx, sy = [s.float() for s in prediction]
            image_error = (sx - tx).square().sum()
            text_error = (sy - ty).square().sum()
            denominator = (tx.square().sum() + ty.square().sum()).clamp_min(1e-30)
            loss = (image_error + text_error) / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite A1 alignment")
            (loss * self.scale).backward()
            if timing:
                timing[1].record()
                timing[1].synchronize()
                self.student_ms = timing[0].elapsed_time(timing[1])
            self.metrics = {
                "relative_mse": loss.detach().item(),
                "image_mse": image_error.detach().item() / tx.numel(),
                "text_mse": text_error.detach().item() / ty.numel(),
            }


def build_student(config, stage, device, load_base=True):
    net = build_net()
    if load_base:
        load_original(net, config.teacher_path)
    net.requires_grad_(False).to(dtype=torch.bfloat16)
    convert_attention(net, config.region_size)
    configure_stage(net, stage, config.lora_rank, config.lora_alpha)
    net.activation_checkpointing = config.activation_checkpointing
    net.pit_chunk_size = config.pit_chunk_size
    return net.to(device)


def lr_factor(step, total, warmup, extension=None):
    if extension:
        start, end = extension["start_step"], extension["end_step"]
        fraction = max(0.0, min(1.0, (step - start) / max(1, end - start)))
        return 0.05 + (extension["start_factor"] - 0.05) * 0.5 * (1 + math.cos(math.pi * fraction))
    if step < warmup:
        return (step + 1) / warmup
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * fraction))


def synchronize_a1(net):
    if not dist.is_initialized():
        return
    world = dist.get_world_size()
    for p in net.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad.div_(world)


def sample_seed(seed, stage, cursor, row):
    value = f"{seed}:{stage}:{cursor}:{row}:pid-vdn-inputs-v1".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def training_inputs(batch, conditioning, fm, noiser, device, seed=42, stage="a1", cursor=0):
    x0 = batch["image"].to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
    with torch.no_grad():
        latent, _ = conditioning.encode_image(x0)  # FLUX encoder uses posterior mean, deterministic.
        captions = []
        latent_parts = []
        sigmas = []
        times = []
        noisy_parts = []
        targets = []
        h, w = x0.shape[-2:]
        shift = 6 * ((h * w) ** 0.5 / 2048) ** 0.5
        for i, row in enumerate(batch["row"]):
            # Key randomness by global row/update, independent of rank, microbatch and world size.
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(sample_seed(seed, stage, cursor, row))
                li, sigma = noiser(latent[i : i + 1])
                captions.append("" if torch.rand((), device=device).item() < 0.1 else batch["caption"][i])
                li = li * (torch.rand((), device=device) >= 0.1)
                t = fm.sample_t(1, device=device)
                t = shift * t / (1 + (shift - 1) * t)
                noisy, noise, _ = fm.add_noise(x0[i : i + 1], t)
                latent_parts.append(li)
                sigmas.append(sigma)
                times.append(t)
                noisy_parts.append(noisy)
                targets.append(noise - x0[i : i + 1])
        emb, mask = conditioning.encode_text(
            captions, cached_embs=batch.get("text_embs"), cached_mask=batch.get("text_mask")
        )
    return (
        (torch.cat(noisy_parts), torch.cat(times) * 1000.0, emb),
        dict(lq_latent=torch.cat(latent_parts), degrade_sigma=torch.cat(sigmas), text_valid_mask=mask),
        torch.cat(targets),
    )


def run_stage(config, stage, rank, world, device, parent=None):
    settings = config.stage_settings(stage, world)
    stage_dir = Path(config.output_root) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_training_run(stage_dir, rank)
    subset = Path(config.subset_root) / str(50000 if stage == "b" else 20000)
    dataset = RawImageDataset(subset, text_cache_root=config.text_cache_root or None)
    checkpoint = latest_state(stage_dir) if config.resume == "auto" else None
    if config.resume not in ("auto", "none", ""):
        checkpoint = Path(config.resume)
    if config.resume in ("none", "") and latest_state(stage_dir):
        raise ValueError("Existing checkpoints: use --resume auto or a new output directory")
    checkpoint = broadcast_object(str(checkpoint) if checkpoint else None, rank)
    payload = load_state(checkpoint) if checkpoint else None
    seed = config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_hash = broadcast_object(sha256_file(config.teacher_path) if rank == 0 else None, rank)
    metadata = {
        "base_sha256": base_hash,
        "base_relative_path": str(Path(config.teacher_path).relative_to(config.weights_root)),
        "source_commit": "e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b",
        "subset_fingerprint": dataset.metadata["fingerprint"],
        "data_fingerprint": dataset.metadata.get("source_fingerprint", dataset.metadata["fingerprint"]),
        "text_cache_fingerprint": dataset.text_cache.fingerprint if dataset.text_cache else None,
        "settings": settings,
        "parent_checkpoint": str(parent) if parent else None,
        "swanlab_run_id": broadcast_object(uuid.uuid4().hex if rank == 0 else None, rank),
    }
    if payload:
        if payload["stage"] != stage:
            raise ValueError("Resume stage mismatch; use --init-from for stage transition")
        for key in ("region_size", "seed", "lora_rank", "lora_alpha"):
            if payload["config"][key] != getattr(config, key):
                raise ValueError(f"Resume {key} mismatch")
        for key in ("base_sha256", "subset_fingerprint", "text_cache_fingerprint"):
            if payload["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume {key} mismatch")
        metadata = resume_metadata(
            payload["metadata"],
            settings,
            payload["progress"],
            config.allow_batch_change,
            config.allow_schedule_change,
            payload["scheduler"],
        )
    net = build_student(config, stage, device)
    progress = dict(
        stage_step=0, total_step=0, cursor=0, samples_seen=0, seen_rows=[], skipped_batches=0, validation_pending=False
    )
    if parent and not payload:
        previous = load_state(parent)
        if previous["metadata"]["base_sha256"] != base_hash or previous["config"]["region_size"] != config.region_size:
            raise ValueError("Parent base/region configuration mismatch")
        if previous["config"]["seed"] != config.seed:
            raise ValueError("Parent data seed mismatch")
        for key in ("data_fingerprint", "text_cache_fingerprint"):
            if previous["metadata"][key] != metadata[key]:
                raise ValueError(f"Parent {key} mismatch")
        if previous["stage"] != {"a2": "a1", "b": "a2"}.get(stage):
            raise ValueError("Expected previous-stage checkpoint")
        # B adds zero adapters, so an A2 delta deliberately has no LoRA tensors.
        load_parent_delta(net, previous["delta"])
        progress["total_step"] = previous["progress"]["total_step"]
        del previous
    if payload:
        load_delta(net, payload["delta"])
        progress.update(payload["progress"])
    elif stage != "a1" and not parent:
        raise ValueError(f"{stage} requires --init-from or completed previous stage")
    if world > 1 and stage == "a1":
        for name, p in net.named_parameters():
            if ".attn." in name and ".orig." not in name:
                dist.broadcast(p.data, src=0)
    model = (
        DDP(net, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
        if world > 1 and stage != "a1"
        else net
    )
    optimizer = torch.optim.AdamW(optimizer_groups(net, stage), betas=(0.9, 0.999), fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, settings["steps"], settings["warmup"], metadata.get("lr_extension"))
    )
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    conditioning = Conditioning(config.weights_root, device, text_cache_root=config.text_cache_root or None)
    if payload:
        restore_rng(payload["rng"][rank % payload["world_size"]])
        # Future conditioning/noise draws use sample_seed, including when ranks change.
    del payload
    gc.collect()
    seen = set(progress["seen_rows"])
    sampler = GlobalBatchSampler(
        subset, settings["batch"], settings["accum"], rank, world, config.seed, progress["cursor"]
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.workers,
        collate_fn=collate_raw,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        generator=torch.Generator().manual_seed(config.seed),
        **({"prefetch_factor": 2} if config.workers else {}),
    )
    iterator = iter(loader)
    tracker = None
    if rank == 0 and config.swanlab_mode != "disabled":
        import swanlab

        tracker = swanlab
        swanlab.init(
            project="PiD-VDN",
            experiment_name=stage,
            config={**config.to_dict(), **settings},
            mode="online" if config.swanlab_mode == "cloud" else config.swanlab_mode,
            log_dir=str(stage_dir / "swanlog"),
            id=metadata["swanlab_run_id"],
            resume="allow",
        )
    if rank == 0:
        atomic_json(stage_dir / "config.json", {**config.to_dict(), "stage_settings": settings})
        print(
            json.dumps(
                {
                    "stage": stage,
                    "batch": settings,
                    "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
                    "frozen_parameters": sum(p.numel() for p in net.parameters() if not p.requires_grad),
                    "resume": checkpoint,
                }
            ),
            flush=True,
        )
    interrupted = {"value": False}

    def stop(sig, frame):
        interrupted["value"] = True

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    start = time.monotonic()

    def stopping():
        return (
            interrupted["value"]
            or (stage_dir / "STOP").exists()
            or (config.max_seconds and time.monotonic() - start >= config.max_seconds)
        )

    fm = FlowMatchingTrainer(timescale=1000.0, t_sampler_type="logit_normal", prediction_type="velocity")
    noiser = LatentNoiser(
        LatentNoisingConfig(enabled=True, backbone="flow_matching", add_sigma_min=0.0, add_sigma_max=0.8)
    )
    callbacks = []
    if stage == "a1":
        for block in net.patch_blocks:
            callback = AlignCallback(scale=1 / settings["accum"])
            block.attn.alignment_callback = callback
            callbacks.append(callback)

    def save(milestone=False):
        progress["seen_rows"] = sorted(seen)
        result = save_state(
            stage_dir, net, optimizer, scheduler, config, stage, progress, metadata, config.keep_checkpoints, milestone
        )
        if rank == 0:
            print(f"Complete checkpoint: {result}", flush=True)
        return result

    def gallery():
        if config.validation_steps == 0:
            progress["validation_pending"] = False
            return
        from .evaluation import render_gallery

        completion = {}
        with torch.random.fork_rng(devices=[device.index]):
            render_gallery(
                net,
                config,
                stage_dir / "galleries" / f"step_{progress['stage_step']:09d}",
                rank,
                world,
                tracker,
                progress["total_step"],
                should_stop=stopping,
                completion=completion,
            )
        # Exactly one collective after each rank finishes or pauses its own cases.
        progress["validation_pending"] = not consensus(completion["complete"], device)
        save(milestone=True)

    bar = tqdm(total=settings["steps"], initial=progress["stage_step"], desc=f"{stage} updates", disable=rank != 0)
    net.train()
    torch.cuda.reset_peak_memory_stats(device)
    final_path = None
    try:
        if progress["validation_pending"] and consensus(not stopping(), device):
            gallery()
        while progress["stage_step"] < settings["steps"]:
            if not consensus(not stopping(), device):
                final_path = save()
                break
            optimizer.zero_grad(set_to_none=False)
            tick = time.monotonic()
            skipped = False
            kd = fm_s = fm_t = teacher_ms = student_ms = 0.0
            for micro in range(settings["accum"]):
                batch = next(iterator)
                if not consensus(not batch["errors"], device) or skipped:
                    if batch["errors"]:
                        with (stage_dir / f"rejected_rank_{rank}.jsonl").open("a") as f:
                            f.write(json.dumps({"cursor": progress["cursor"], "errors": batch["errors"]}) + "\n")
                    skipped = True
                    continue
                args, kwargs, fm_target = training_inputs(
                    batch, conditioning, fm, noiser, device, config.seed, stage, progress["cursor"]
                )
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                events[0].record()
                if stage == "a1":
                    with (
                        execution_mode(net, teacher=True),
                        torch.no_grad(),
                        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                    ):
                        teacher = net(*args, **kwargs)
                    events[1].record()
                    events[2].record()
                    events[3].record()
                    kd += sum(c.metrics["relative_mse"] for c in callbacks) / len(callbacks) / settings["accum"]
                    fm_t += torch.nn.functional.mse_loss(teacher.float(), fm_target).item() / settings["accum"]
                else:
                    with (
                        execution_mode(net, teacher=True),
                        torch.no_grad(),
                        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                    ):
                        teacher = net(*args, **kwargs).detach()
                    events[1].record()
                    events[2].record()
                    sync = model.no_sync() if world > 1 and micro < settings["accum"] - 1 else contextlib.nullcontext()
                    with sync, execution_mode(net), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        student = model(*args, **kwargs)
                        loss = torch.nn.functional.mse_loss(student.float(), teacher.float())
                        if not consensus(torch.isfinite(loss).item(), device):
                            raise FloatingPointError("Nonfinite KD loss")
                        (loss / settings["accum"]).backward()
                    events[3].record()
                    kd += loss.detach().item() / settings["accum"]
                    fm_s += torch.nn.functional.mse_loss(student.detach().float(), fm_target).item() / settings["accum"]
                    fm_t += torch.nn.functional.mse_loss(teacher.float(), fm_target).item() / settings["accum"]
                    del student, loss
                events[3].synchronize()
                alignment_ms = sum(c.student_ms for c in callbacks) if stage == "a1" else 0.0
                teacher_ms += events[0].elapsed_time(events[1]) - alignment_ms
                student_ms += alignment_ms + events[2].elapsed_time(events[3])
                del args, kwargs, fm_target, teacher, batch
            global_rows = sampler.global_batch(progress["cursor"])
            progress["cursor"] += 1
            if not skipped:
                if stage == "a1":
                    synchronize_a1(net)
                if stage == "a1":
                    norms = [torch.nn.utils.clip_grad_norm_(block.attn.parameters(), 0.1) for block in net.patch_blocks]
                    norm = torch.stack(norms).max()
                else:
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if not consensus(torch.isfinite(norm).item(), device):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                scheduler.step()
                progress["stage_step"] += 1
                progress["total_step"] += 1
                progress["samples_seen"] += settings["effective"]
                seen.update(map(int, global_rows))
                bar.update()
            else:
                progress["skipped_batches"] += 1
                optimizer.zero_grad(set_to_none=False)
                norm = torch.tensor(0.0, device=device)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            if progress["cursor"] % config.log_steps == 0:
                values = torch.tensor([kd, fm_s, fm_t, teacher_ms, student_ms], device=device)
                peaks = torch.tensor([elapsed, torch.cuda.max_memory_allocated(device) / 2**30], device=device)
                if world > 1:
                    dist.all_reduce(values)
                    values /= world
                    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
                metrics = {
                    "train/kd_loss": values[0].item(),
                    "train/fm_student": values[1].item() if stage != "a1" else None,
                    "train/fm_teacher": values[2].item(),
                    "efficiency/teacher_ms": values[3].item(),
                    "efficiency/student_forward_backward_ms": values[4].item(),
                    "efficiency/seconds_per_update": peaks[0].item(),
                    "efficiency/peak_memory_gib": peaks[1].item(),
                    "efficiency/images_per_second": settings["effective"] / peaks[0].item() if not skipped else 0.0,
                    "stage/step": progress["stage_step"],
                    "stage/unique_samples": len(seen),
                    "stage/samples_seen": progress["samples_seen"],
                    "stage/equivalent_epochs": progress["samples_seen"] / dataset.metadata["train_size"],
                    "train/grad_norm": norm.item(),
                }
                for g in optimizer.param_groups:
                    metrics["lr/" + g["group_name"]] = g["lr"]
                for i, callback in enumerate(callbacks):
                    vm = torch.tensor(list(callback.metrics.values()), device=device)
                    if world > 1:
                        dist.all_reduce(vm)
                        vm /= world
                    metrics.update({f"alignment/layer_{i}/{k}": v for k, v in zip(callback.metrics, vm.tolist())})
                if rank == 0:
                    with (stage_dir / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"step": progress["total_step"], **metrics}) + "\n")
                    print(
                        f"{stage} step={progress['stage_step']} KD={values[0].item():.6f} {peaks[0].item():.2f}s/update peak={peaks[1].item():.2f}GiB",
                        flush=True,
                    )
                    if tracker:
                        tracker.log({k: v for k, v in metrics.items() if v is not None}, step=progress["total_step"])
            finished = progress["stage_step"] >= settings["steps"]
            valid_due = not skipped and (
                finished or (settings["validate"] and progress["stage_step"] % settings["validate"] == 0)
            )
            progress["validation_pending"] = bool(valid_due and config.validation_steps != 0)
            stop_all = not consensus(not stopping(), device)
            save_due = not skipped and progress["stage_step"] % settings["save"] == 0
            if save_due or finished or stop_all or valid_due:
                final_path = save(milestone=bool(valid_due or finished))
            if progress["validation_pending"] and not stop_all:
                gallery()
            if stop_all:
                break
        completed = progress["stage_step"] >= settings["steps"] and not progress["validation_pending"]
        if completed:
            final_path = save(milestone=True)
            if rank == 0:
                atomic_json(
                    stage_dir / "finished.json", {"checkpoint": str(final_path), "total_step": progress["total_step"]}
                )
        return final_path or latest_state(stage_dir), completed
    finally:
        for block in net.patch_blocks:
            block.attn.alignment_callback = None
        bar.close()
        if tracker:
            tracker.finish()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if lock:
            lock.close()
        del model, conditioning, iterator, loader
        net = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = VDNConfig()
    for name in (
        "weights_root",
        "index_root",
        "text_cache_root",
        "subset_root",
        "gallery_root",
        "output_root",
        "stage",
        "preset",
        "resume",
        "init_from",
        "swanlab_mode",
    ):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name))
    for name in (
        "region_size",
        "seed",
        "workers",
        "threads",
        "batch_size",
        "grad_accum",
        "effective_batch",
        "max_steps",
        "max_seconds",
        "save_steps",
        "validation_steps",
        "log_steps",
        "keep_checkpoints",
        "lora_rank",
        "lora_alpha",
        "pit_chunk_size",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--allow-batch-change", action="store_true")
    parser.add_argument("--allow-schedule-change", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    dry = args.pop("dry_run")
    args["activation_checkpointing"] = not args.pop("no_activation_checkpointing")
    config = VDNConfig(**args)
    if config.stage not in ("all", *STAGES):
        parser.error("stage must be a1,a2,b,all")
    if config.region_size < 1 or config.workers < 0 or config.threads < 1 or config.log_steps < 1:
        parser.error("Invalid size/workers/threads/log interval")
    if config.max_steps < 0 or config.save_steps < 0 or config.validation_steps < -1 or config.lora_rank < 1:
        parser.error("Invalid steps/LoRA")
    return config, dry


def main(argv=None):
    config, dry = parse_args(argv)
    stages = list(STAGES) if config.stage == "all" else [config.stage]
    world = int(os.getenv("WORLD_SIZE", 1))
    if dry:
        print(
            json.dumps(
                {"config": config.to_dict(), "stages": {s: config.stage_settings(s, world) for s in stages}}, indent=2
            )
        )
        return
    if config.stage == "all" and config.max_steps:
        raise ValueError("--max-steps only applies to one selected stage")
    for path in (
        config.teacher_path,
        Path(config.index_root) / "index.json",
        Path(config.subset_root) / "subsets.json",
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if config.validation_steps != 0 and not (Path(config.gallery_root) / "assets.json").exists():
        raise FileNotFoundError("Prepare fixed gallery assets, or explicitly use --validation-steps 0 for a short test")
    rank, world, device = init_distributed()
    torch.set_num_threads(config.threads)
    parent = config.init_from or None
    try:
        for stage in stages:
            if not parent and stage != "a1":
                parent = latest_state(Path(config.output_root) / {"a2": "a1", "b": "a2"}[stage])
            parent, completed = run_stage(config, stage, rank, world, device, parent)
            if not completed:
                break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
