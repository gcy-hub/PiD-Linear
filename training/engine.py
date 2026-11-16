"""Resident distributed training with recoverable checkpoints and validation."""
from datetime import timedelta
from pathlib import Path
import time
import sys
from contextlib import nullcontext
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import checkpoint, tracking
from .distributed import setup, average_gradients
from .optim import build_optimizer, move_state
from .strategies import vdn, clear, wavelet


def phases(config):
    if config.training.strategy != 'vdn':
        return ['clear']
    start = config.training.vdn.get('start_stage') or ('b' if config.model.resources.get('export') else 'a1')
    return ['a1', 'a2', 'b'][['a1', 'a2', 'b'].index(start):]


def phase_steps(config, stage):
    return config.training.steps if stage == 'clear' else config.training.vdn[stage].steps


def event_due(config, progress, event):
    """Phase overrides use stage_step; absent/null values retain global steps."""
    every = (config.training.vdn[progress['stage']].get(f'{event}_every')
             if config.training.strategy == 'vdn' else None)
    step = progress['stage_step'] if every is not None else progress['step']
    if every is None:
        every = config[event].every
    return bool(every and step > 0 and step % every == 0)


def select_roles(adapter, stage):
    adapter.module.requires_grad_(False)
    selected = [r for r in adapter.role_parameters() if r.role in (clear.ROLES if stage == 'clear' else vdn.ROLES[stage])]
    if not selected:
        raise ValueError(f'No trainable parameters for {stage}')
    for r in selected:
        r.parameter.requires_grad_(True)
    return selected


def segment(config, resume=None):
    from models import build_model
    from algorithms import build_attention
    from data import build_dataloader
    device, rank, world = setup(config.runtime.gpus)
    tracker = None
    try:
        random.seed(config.training.seed + rank)
        np.random.seed(config.training.seed + rank)
        torch.manual_seed(config.training.seed)
        generator = torch.Generator(device=device).manual_seed(config.training.seed + rank)
        adapter = build_model(config, device=device, algorithm_factory=build_attention)
        torch.manual_seed(config.training.seed + rank)
        control_group = dist.new_group(backend='gloo', timeout=timedelta(hours=2)) if world > 1 else None
        progress = dict(step=0, stage=phases(config)[0], stage_step=0, cursor=0, pending_validation=False, done=False)
        directory = Path(resume) if resume else None
        state = checkpoint.read(resume, rank, world) if resume else None
        if state:
            progress = state['progress'].copy()
            # Only operational settings may differ on resume.
            for key in ('model', 'replacement', 'algorithm', 'training', 'data', 'cache'):
                from omegaconf import OmegaConf
                previous = state['config'][key]
                current = OmegaConf.to_container(config[key], resolve=True)
                if key == 'training':
                    # Missing phase intervals in older checkpoints mean global scheduling.
                    for training in (previous, current):
                        for name in ('a1', 'a2', 'b'):
                            for interval in ('checkpoint_every', 'validation_every'):
                                if training['vdn'][name].get(interval) is None:
                                    training['vdn'][name].pop(interval, None)
                if key == 'training' and (Path(config.paths.output) / 'batch-reduced.json').exists():
                    if previous['batch_size'] == 2 and current['batch_size'] == 1:
                        previous = dict(previous, batch_size=1)
                if previous != current:
                    raise ValueError(f'Checkpoint {key} configuration differs')
        stage = progress['stage']
        roles = select_roles(adapter, stage)
        optimizer, scheduler = build_optimizer(config, roles, stage)
        parameters = [r.parameter for r in roles]
        student = adapter.module
        if stage != 'a1' and world > 1:
            student = DDP(student, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=True)
        loader = build_dataloader(config, rank=rank, world_size=world, cursor=progress['cursor'])
        loader.generator = torch.Generator().manual_seed(config.training.seed)
        encoder = loader.collate_fn.encoder
        if not encoder.image_enabled or not encoder.text_enabled:
            encoder = None
        loader = iter(loader)
        if state:
            checkpoint.restore(state, adapter, optimizer, scheduler, generator)
            del state
        tracker = tracking.start(config, rank)
        last_saved = time.monotonic()
        if rank == 0:
            print(dict(stage=stage, parameters={role: sum(r.parameter.numel() for r in roles if r.role == role) for role in sorted({r.role for r in roles})}, feature_layers=list(config.feature_layers), feature_tap=config.feature_tap), flush=True)
        while True:
            if progress['pending_validation']:
                from .validate import validate_resident
                optimizer.zero_grad(set_to_none=True)
                move_state(optimizer, 'cpu')
                validate_resident(config, directory, adapter, encoder,
                                  rank, world, control_group, tracker=tracker)
                if not config.runtime.optimizer_cpu_offload:
                    move_state(optimizer, device)
                progress['pending_validation'] = False
                last_saved = time.monotonic()
            if progress['stage_step'] == phase_steps(config, stage):
                index = phases(config).index(stage) + 1
                if index == len(phases(config)):
                    return
                del student, optimizer, scheduler
                stage = phases(config)[index]
                progress.update(stage=stage, stage_step=0)
                roles = select_roles(adapter, stage)
                parameters = [r.parameter for r in roles]
                optimizer, scheduler = build_optimizer(config, roles, stage)
                student = adapter.module
                if world > 1:
                    student = DDP(student, device_ids=[device.index] if device.type == 'cuda' else None,
                                  find_unused_parameters=True)
            optimizer.zero_grad(set_to_none=True)
            batches = []
            layerwise_losses = []
            metrics = {}
            if stage == 'a1':
                batches = [adapter.prepare_batch(next(loader), generator=generator)
                           for _ in range(config.training.accumulation)]
                with torch.autocast(device.type, dtype=getattr(torch, config.runtime.dtype),
                                    enabled=config.runtime.dtype != 'float32'):
                    energies = vdn.teacher_energy(adapter, batches)
            for micro in range(config.training.accumulation):
                batch = batches[micro] if stage == 'a1' else adapter.prepare_batch(next(loader), generator=generator)
                progress['cursor'] += 1
                sync = student.no_sync() if isinstance(student, DDP) and micro+1 < config.training.accumulation else nullcontext()
                with sync:
                    with torch.autocast(device.type, dtype=getattr(torch, config.runtime.dtype),
                                        enabled=config.runtime.dtype != 'float32'):
                        if stage == 'a1':
                            loss = vdn.backward_layerwise(adapter, batch, energies)
                            layerwise_losses.append(loss)
                        else:
                            capture = tuple(config.feature_layers) if stage == 'clear' else ()
                            with torch.no_grad():
                                teacher = adapter.forward(batch, teacher=True, capture=capture)
                            output = student(batch, capture=capture)
                            terms = {}
                            if stage == 'clear':
                                loss = clear.objective(output, teacher, batch.target, capture, config.training.loss, metrics=terms)
                            else:
                                objective = config.training.vdn[stage].objective
                                if objective not in ('teacher', 'fm'):
                                    raise ValueError(f'Unknown VDN objective: {objective}')
                                target = teacher.prediction if objective == 'teacher' else batch.target
                                loss = torch.nn.functional.mse_loss(output.prediction.float(), target.float())
                                terms['prediction_mse' if objective == 'teacher' else 'fm'] = loss.detach()
                            if config.training.wavelet.weight and (stage == 'clear' or objective == 'teacher'):
                                auxiliary = wavelet.prediction_loss(output.prediction, teacher.prediction, batch,
                                                                    config.model.family, config.training.wavelet)
                                weighted = config.training.wavelet.weight * auxiliary
                                loss = loss + weighted
                                terms.update(wavelet=auxiliary.detach(), wavelet_weighted=weighted.detach())
                            terms['total'] = loss.detach()
                            for key, value in terms.items():
                                metrics[key] = metrics.get(key, 0) + value / config.training.accumulation
                    if stage != 'a1':
                        (loss/config.training.accumulation).backward()
            if stage == 'a1':
                loss = torch.stack(layerwise_losses).sum()
                if dist.is_initialized():
                    dist.all_reduce(loss)
                average_gradients(parameters)
                for layer in sorted({r.layer for r in roles}):
                    torch.nn.utils.clip_grad_norm_([r.parameter for r in roles if r.layer == layer], .1)
            else:
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if config.runtime.optimizer_cpu_offload:
                move_state(optimizer, device)
            optimizer.step()
            scheduler.step()
            if config.runtime.optimizer_cpu_offload:
                move_state(optimizer, 'cpu')
            progress['step'] += 1
            progress['stage_step'] += 1
            step = progress['step']
            boundary = progress['stage_step'] == phase_steps(config, stage)
            progress['done'] = boundary and stage == phases(config)[-1]
            progress['pending_validation'] = event_due(config, progress, 'validation')
            if rank == 0:
                record = dict(step=step, stage=stage, stage_step=progress['stage_step'],
                              loss=float(loss.detach()), losses={k: float(v) for k, v in metrics.items()},
                              lr=optimizer.param_groups[0]['lr'],
                              peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0)
                print(record, flush=True)
                if tracker is not None:
                    tracker.log_step(record)
            stop = (Path(config.paths.output) / 'STOP').exists()
            flags = torch.tensor([int(stop), int(boundary or progress['pending_validation'] or stop
                                   or event_due(config, progress, 'checkpoint')
                                   or (config.checkpoint.seconds and time.monotonic()-last_saved >= config.checkpoint.seconds))],
                                 device=device)
            if world > 1:
                dist.broadcast(flags, src=0)
            if flags[1].item():
                directory = checkpoint.save(config.paths.output, adapter, optimizer, scheduler, generator, progress, config, rank, world)
                last_saved = time.monotonic()
            batches.clear()
            if stage != 'a1':
                del output, teacher
                target = auxiliary = weighted = None
            del batch, loss
            if flags[0].item():
                return
    finally:
        try:
            if tracker is not None:
                tracker.finish(sys.exc_info()[1])
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
