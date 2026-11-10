"""One training segment, ending at a save, validation or phase boundary."""
from contextlib import nullcontext
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import checkpoint
from .distributed import setup, average_gradients
from .optim import build_optimizer, move_state
from .strategies import vdn, clear


def phases(config):
    return ['a1', 'a2', 'b'] if config.training.strategy == 'vdn' else ['clear']


def phase_steps(config, stage):
    return config.training.steps if stage == 'clear' else config.training.vdn[stage].steps


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
    try:
        random.seed(config.training.seed + rank)
        np.random.seed(config.training.seed + rank)
        torch.manual_seed(config.training.seed + rank)
        generator = torch.Generator(device=device).manual_seed(config.training.seed + rank)
        adapter = build_model(config, device=device, algorithm_factory=build_attention)
        progress = dict(step=0, stage=phases(config)[0], stage_step=0, cursor=0, pending_validation=False, done=False)
        state = checkpoint.read(resume, rank, world) if resume else None
        if state:
            progress = state['progress'].copy()
            # Only operational settings may differ on resume.
            for key in ('model', 'replacement', 'algorithm', 'training', 'data', 'cache'):
                from omegaconf import OmegaConf
                if state['config'][key] != OmegaConf.to_container(config[key], resolve=True):
                    raise ValueError(f'Checkpoint {key} configuration differs')
        stage = progress['stage']
        roles = select_roles(adapter, stage)
        optimizer, scheduler = build_optimizer(config, roles, stage)
        if state:
            checkpoint.restore(state, adapter, optimizer, scheduler, generator)
            del state
        if progress['stage_step'] == phase_steps(config, stage):
            index = phases(config).index(stage) + 1
            if index == len(phases(config)):
                return
            stage = phases(config)[index]
            progress.update(stage=stage, stage_step=0)
            roles = select_roles(adapter, stage)
            optimizer, scheduler = build_optimizer(config, roles, stage)
        parameters = [r.parameter for r in roles]
        student = adapter.module
        if stage != 'a1' and world > 1:
            student = DDP(student, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=True)
        loader = build_dataloader(config, rank=rank, world_size=world, cursor=progress['cursor'])
        loader.generator = torch.Generator().manual_seed(config.training.seed)
        loader = iter(loader)
        if rank == 0:
            print(dict(stage=stage, parameters={role: sum(r.parameter.numel() for r in roles if r.role == role) for role in sorted({r.role for r in roles})}, feature_layers=list(config.feature_layers), feature_tap=config.feature_tap), flush=True)
        while True:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(config.training.accumulation):
                batch = adapter.prepare_batch(next(loader), generator=generator)
                progress['cursor'] += 1
                sync = student.no_sync() if isinstance(student, DDP) and micro+1 < config.training.accumulation else nullcontext()
                with sync:
                    if stage == 'a1':
                        loss = vdn.backward_layerwise(adapter, batch, config.training.accumulation)
                    else:
                        capture = tuple(config.feature_layers) if stage == 'clear' else ()
                        with torch.no_grad():
                            teacher = adapter.forward(batch, teacher=True, capture=capture)
                        output = student(batch, capture=capture)
                        if stage == 'clear':
                            loss = clear.objective(output, teacher, batch.target, capture, config.training.loss)
                        else:
                            objective = config.training.vdn[stage].objective
                            if objective not in ('teacher', 'fm'):
                                raise ValueError(f'Unknown VDN objective: {objective}')
                            target = teacher.prediction if objective == 'teacher' else batch.target
                            loss = torch.nn.functional.mse_loss(output.prediction.float(), target.float())
                        (loss/config.training.accumulation).backward()
            if stage == 'a1':
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
            progress['pending_validation'] = bool(config.validation.every and step % config.validation.every == 0)
            if rank == 0:
                print(dict(step=step, stage=stage, loss=float(loss.detach())), flush=True)
            if boundary or progress['pending_validation'] or (config.checkpoint.every and step % config.checkpoint.every == 0):
                checkpoint.save(config.paths.output, adapter, optimizer, scheduler, generator, progress, config, rank, world)
                return
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
