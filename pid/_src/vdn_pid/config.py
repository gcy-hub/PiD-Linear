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
    region_size: int = 8
    memory_mode: str = "vda"
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
        return str(Path(self.weights_root) / BASE_RELATIVE)

    def stage_settings(self, stage, world):
        p = STAGES[stage].copy()
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
