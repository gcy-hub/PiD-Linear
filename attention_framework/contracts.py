"""Shared tensor and lifecycle contracts; no model or strategy implementation."""
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from torch import Generator, Tensor, device, nn
from omegaconf import DictConfig

Family = Literal['pid', 'flux2', 'flux3']
AlgorithmName = Literal['cross_scan_kda', 'sliding_qdm', 'qdm_halo', 'clear']
FeatureTap = Literal['pre_o', 'post_o']
ParameterRole = Literal['branch', 'local_gate', 'image_qkvo', 'lora', 'pooling']


@dataclass(frozen=True)
class AttentionLayout:
    # Each batch row has this packed sequence; padding lies after length.
    samples: tuple[tuple[int, int, int], ...]  # (image height, width, text length)
    text_first: bool = True

    def __post_init__(self):
        if not self.samples or any(h < 1 or w < 1 or t < 0 for h, w, t in self.samples):
            raise ValueError('Invalid attention layout')

    @property
    def length(self) -> int:
        return sum(h * w + t for h, w, t in self.samples)


@dataclass(frozen=True)
class LayerSpec:
    name: str  # e.g. joint.4
    group: str
    index: int
    ordinal: int  # over ALL native image-attention layers, before selection
    feature_tap: FeatureTap


@dataclass
class AttentionInput:
    hidden: Tensor  # normalized input [B,L,C], KDA consumes image slices
    q_raw: Tensor  # all QKV [B,L,H,D]; raw means after native QK norm, before RoPE
    k_raw: Tensor
    v: Tensor
    q_pos: Tensor
    k_pos: Tensor
    layout: AttentionLayout
    layer: LayerSpec


@dataclass
class AttentionResult:
    pre_o: Tensor  # [B,L,H*D], native O applied exactly once by adapter
    post_o_residual: Tensor | None = None  # [B,L,C_out], QDM memory already projected


class AttentionOperator(Protocol):
    def __call__(self, inputs: AttentionInput) -> AttentionResult: ...


@dataclass
class RoleParameter:
    name: str
    parameter: nn.Parameter
    role: ParameterRole
    layer: str
    # Fused FFN slices must be separate trainable slices, never a full fused parameter.


@dataclass
class Sample:
    sample_id: str
    image_path: str  # absolute, resolved relative to manifest
    caption: str
    width: int
    height: int


@dataclass
class EncodedBatch:
    sample_ids: tuple[str, ...]
    family: Family
    image: dict[str, Tensor]  # native clean image/condition encodings; PiD also full RGB
    text: dict[str, Tensor]  # native embeddings, masks, IDs
    grids: tuple[tuple[int, int], ...]


@dataclass
class TrainingBatch:
    inputs: dict[str, Tensor]  # native forward kwargs, runtime noise/degradation included
    target: Tensor  # PiD full RGB; FLUX native flow target
    timesteps: Tensor
    sample_ids: tuple[str, ...]


@dataclass
class ModelOutput:
    prediction: Tensor
    features: dict[str, Tensor]  # LayerSpec.name -> image query features only


@dataclass
class LayerPair:
    layer: LayerSpec
    inputs: AttentionInput  # detached native teacher layer input
    teacher: Tensor  # detached post-O image query attention output, no FFN/residual
    student: Tensor  # independent graph, same slice; backward before callback returns


class ModelAdapter(Protocol):
    module: nn.Module
    layers: tuple[LayerSpec, ...]  # selected layers, stable native order

    def role_parameters(self) -> list[RoleParameter]: ...
    def prepare_batch(self, batch: EncodedBatch, *, generator: Generator) -> TrainingBatch: ...
    def forward(self, batch: TrainingBatch, *, teacher: bool = False,
                capture: tuple[str, ...] = ()) -> ModelOutput: ...
    def layerwise(self, batch: TrainingBatch, callback: Callable[[LayerPair], None]) -> None: ...


class Encoder(Protocol):
    def encode_image(self, pixels: Tensor, grids: tuple[tuple[int, int], ...]) -> dict[str, Tensor]: ...
    def encode_text(self, captions: list[str]) -> dict[str, Tensor]: ...


class AlgorithmFactory(Protocol):
    def __call__(self, config: DictConfig, *, hidden_size: int, num_heads: int,
                 layer: LayerSpec) -> nn.Module: ...


class ModelFactory(Protocol):
    def __call__(self, config: DictConfig, *, device: device,
                 algorithm_factory: AlgorithmFactory) -> ModelAdapter: ...
