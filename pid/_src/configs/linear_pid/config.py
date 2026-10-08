"""Independent Linear-PiD configuration, dispatched through scripts.train."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from pid._src.linear_pid.attention import FLA_COMMIT, resolve_layers


@dataclass
class LinearPiDConfig:
    weights_root: str = "/home/ganchangyi/huggingface_ckpts"
    index_root: str = "/home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_index"
    text_cache_root: str = "/home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache"
    output_root: str = "/home/ganchangyi/code/PiD-Linear/outputs/linear-pid"
    preset: str = "4gpu"
    layers: list[int] = field(default_factory=lambda: resolve_layers(4))
    batch_size: int = 1
    grad_accum: int = 2
    effective_batch: int = 0
    workers: int = 4
    threads: int = 1
    seed: int = 42
    lr_new: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-3
    warmup: int = 500
    lambda_out: float = 0.0
    grad_clip: float = 1.0
    local_mixing: bool = True
    activation_checkpointing: bool = True
    pit_chunk_size: int = 0
    pit_kda: bool = False
    pit_kda_heads: int = 16
    pit_kda_compressed: bool = True
    save_steps: int = 5000
    save_epochs: int = 1
    keep_checkpoints: int = 3
    max_steps: int = 0
    max_seconds: int = 0
    log_steps: int = 10
    validation_steps: int = 5000
    validation_epochs: int = 1
    gallery_root: str = "/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/assets"
    swanlab_mode: str = "offline"
    resume: str = "auto"
    resume_from: str = ""
    allow_batch_size_change: bool = False
    init_from: str = ""
    init_weights: str = "raw"
    fla_commit: str = FLA_COMMIT

    @property
    def kda_layers(self):
        return self.layers

    @kda_layers.setter
    def kda_layers(self, value):
        self.layers = resolve_layers(value)

    @property
    def teacher_path(self):
        return str(
            Path(self.weights_root)
            / "PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth"
        )

    @property
    def run_dir(self):
        label = "-".join(map(str, self.layers)) or "full"
        suffix = f"_pit-kda-h{self.pit_kda_heads}" if self.pit_kda else ""
        if self.pit_kda and self.pit_kda_compressed:
            suffix = f"_pit-kda-compressed-h{self.pit_kda_heads}"
        return str(Path(self.output_root) / f"kda_{label}{suffix}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        # Earlier checkpoints carried CPU EMA/offload settings. They remain
        # readable, while the current training path keeps all models on GPU.
        legacy = {
            "ema_decay", "offload_text_encoder", "save_seconds",
            "validation_seconds", "quick_every", "full_every",
        }
        values = {key: value for key, value in values.items() if key not in legacy}
        # The first PiT KDA checkpoints used the uncompressed architecture.
        if values.get("pit_kda") and "pit_kda_compressed" not in values:
            values["pit_kda_compressed"] = False
        return cls(**values)


def make_config():
    return LinearPiDConfig()
