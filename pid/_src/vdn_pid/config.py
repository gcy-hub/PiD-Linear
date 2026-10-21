"""Portable PiD-VDN architecture and stage presets."""

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = {
    "a1": dict(steps=500, batch=1, effective=8, save=50, warmup=50),
    "a2": dict(steps=1000, batch=2, effective=16, save=100, warmup=50),
    "b": dict(steps=10000, batch=4, effective=32, save=2000, warmup=200),
}
BASE_RELATIVE = "PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth"


def resume_metadata(
    metadata, settings, progress, allow_batch_change=False, allow_schedule_change=False, scheduler_state=None
):
    """Keep update-based scheduling; record explicit changes to future batch draws."""
    previous = metadata["settings"]
    if previous["warmup"] != settings["warmup"]:
        raise ValueError("Resume stage settings warmup mismatch")
    changed = {key: settings[key] for key in ("steps", "save", "validate") if previous[key] != settings[key]}
    if changed and not allow_schedule_change:
        raise ValueError(f"Resume stage settings {next(iter(changed))} mismatch; use --allow-schedule-change")
    schedule_history = list(metadata.get("schedule_changes", []))
    extension = metadata.get("lr_extension")
    if changed:
        if settings["steps"] <= progress["stage_step"]:
            raise ValueError("New total steps must exceed the restored stage step")
        schedule_history.append(
            dict(
                stage_step=progress["stage_step"],
                cursor=progress["cursor"],
                previous={key: previous[key] for key in changed},
                settings=changed,
            )
        )
        if "steps" in changed:
            if scheduler_state is None:
                raise ValueError("Schedule extension requires restored scheduler state")
            factors = [lr / base for lr, base in zip(scheduler_state["_last_lr"], scheduler_state["base_lrs"])]
            if not factors or not all(math.isclose(f, factors[0], rel_tol=1e-9) for f in factors):
                raise ValueError("Restored LR groups do not share a schedule factor")
            extension = dict(start_step=progress["stage_step"], start_factor=factors[0], end_step=settings["steps"])
    history = list(metadata.get("batch_changes", []))
    if previous["effective"] != settings["effective"]:
        if not allow_batch_change:
            raise ValueError("Resume stage settings effective mismatch; use --allow-batch-change")
        history.append(
            dict(
                stage_step=progress["stage_step"],
                cursor=progress["cursor"],
                samples_seen=progress["samples_seen"],
                previous_effective=previous["effective"],
                effective=settings["effective"],
            )
        )
    return {
        **metadata,
        "settings": settings,
        "batch_changes": history,
        "schedule_changes": schedule_history,
        "lr_extension": extension,
    }


@dataclass
class VDNConfig:
    weights_root: str = field(default_factory=lambda: os.getenv("PID_WEIGHTS_ROOT", "weights"))
    index_root: str = field(
        default_factory=lambda: os.getenv("PID_INDEX_ROOT", "dataset/MultiAspect-4K-1M/linear_pid_index")
    )
    text_cache_root: str = field(default_factory=lambda: os.getenv("PID_TEXT_CACHE_ROOT", ""))
    subset_root: str = field(
        default_factory=lambda: os.getenv("PID_VDN_SUBSET_ROOT", "dataset/MultiAspect-4K-1M/pid_vdn_subsets")
    )
    gallery_root: str = field(default_factory=lambda: os.getenv("PID_GALLERY_ROOT", "outputs/linear-pid/assets"))
    output_root: str = "outputs/pid-vdn"
    backbone: str = "pid"
    teacher_checkpoint: str = ""
    a1_batch_size: int = 4
    a2_batch_size: int = 8
    b_batch_size: int = 8
    a1_steps: int = 500
    a2_steps: int = 1000
    b_steps: int = 10000
    resolution: str = "2048"
    flow_shift: float = 6.0
    shift_base_size: int = 2048
    region_size: int = 8
    local_mode: str = "window"
    local_halo: int = 0
    local_shift: int = 4
    local_chunk: int = 16
    memory_mode: str = "vda"
    remote_rope: bool = False
    stage: str = "all"
    preset: str = "4gpu"
    seed: int = 42
    workers: int = 4
    threads: int = 1
    batch_size: int = 0
    grad_accum: int = 0
    effective_batch: int = 0
    allow_batch_change: bool = False
    allow_schedule_change: bool = False
    max_steps: int = 0
    max_seconds: int = 0
    save_steps: int = 0
    validation_steps: int = -1
    log_steps: int = 10
    keep_checkpoints: int = 3
    resume: str = "auto"
    init_from: str = ""
    swanlab_mode: str = "offline"
    lora_rank: int = 64
    lora_alpha: int = 64
    pit_chunk_size: int = 2048
    activation_checkpointing: bool = True

    def to_dict(self):
        return asdict(self)

    @property
    def teacher_path(self):
        if self.teacher_checkpoint:
            return str(Path(self.teacher_checkpoint).expanduser().resolve())
        if self.backbone == "pixeldit_t2i":
            from pid._src.pixeldit_qdm.runtime import checkpoint_path
            return str(checkpoint_path(Path(self.weights_root)/"PixelDiT-T2I"))
        return str(Path(self.weights_root) / BASE_RELATIVE)

    def stage_settings(self, stage, world):
        p = STAGES[stage].copy()
        if self.backbone == "pixeldit_t2i":
            p["steps"] = getattr(self, stage+"_steps")
            p["batch"] = self.batch_size or getattr(self, stage+"_batch_size")
            p["effective"] = self.effective_batch or world*p["batch"]*(self.grad_accum or 1)
        p["steps"] = self.max_steps or p["steps"]
        p["batch"] = self.batch_size or p["batch"]
        p["effective"] = self.effective_batch or p["effective"]
        if not self.grad_accum and p["effective"] % (world * p["batch"]):
            raise ValueError(
                "Effective batch must divide world_size * batch_size; set explicit batch/accum/effective values"
            )
        p["accum"] = self.grad_accum or p["effective"] // (world * p["batch"])
        if p["accum"] < 1 or p["accum"] * p["batch"] * world != p["effective"]:
            raise ValueError("batch_size * grad_accum * world_size must equal effective_batch")
        p["save"] = self.save_steps or p["save"]
        p["validate"] = (2000 if stage == "b" else 0) if self.validation_steps < 0 else self.validation_steps
        return p


def verify_memory_mode(config, saved_config):
    """Legacy checkpoints use VDA; never silently resume another state mechanism."""
    if config.memory_mode not in ("vda", "fixed4"):
        raise ValueError(f"Unknown memory_mode: {config.memory_mode}")
    saved_mode = saved_config.get("memory_mode", "vda")
    if saved_mode != config.memory_mode:
        raise ValueError(f"Checkpoint memory_mode={saved_mode} differs from requested {config.memory_mode}")
    if saved_config.get("remote_rope", False) != config.remote_rope:
        raise ValueError("Checkpoint remote_rope differs; use a new experiment directory")


def local_layout(config):
    def get(key,default):return config.get(key,default) if isinstance(config,dict) else getattr(config,key,default)
    mode=get('local_mode','window')
    return dict(schema=1,mode=mode,region_size=get('region_size',8),
        halo=get('local_halo',0) if mode=='halo' else 0,
        shift=get('local_shift',4) if mode=='alternating_shift' else 0)


def verify_local_layout(config,saved_config):
    if local_layout(config)!=local_layout(saved_config):
        raise ValueError('Checkpoint local attention layout differs; use a new experiment directory')
