"""Portable PiD-VDN architecture and stage presets."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = {
    "a1": dict(steps=500, batch=1, effective=8, save=50, warmup=50),
    "a2": dict(steps=1000, batch=2, effective=16, save=100, warmup=50),
    "b": dict(steps=10000, batch=4, effective=32, save=2000, warmup=200),
}
BASE_RELATIVE = "PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth"


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
    stage: str = "all"
    preset: str = "4gpu"
    seed: int = 42
    workers: int = 4
    threads: int = 1
    batch_size: int = 0
    grad_accum: int = 0
    effective_batch: int = 0
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
