# Shared framework contract

## Attention boundary (A ↔ M)

Concrete declarations are in `attention_framework/contracts.py`:

```python
AttentionLayout(samples: tuple[tuple[int, int, int], ...], text_first: bool = True)
LayerSpec(name: str, group: str, index: int, ordinal: int, feature_tap: FeatureTap)
AttentionInput(hidden: Tensor, q_raw: Tensor, k_raw: Tensor, v: Tensor,
               q_pos: Tensor, k_pos: Tensor, layout: AttentionLayout, layer: LayerSpec)
AttentionResult(pre_o: Tensor, post_o_residual: Tensor | None = None)
```

`hidden` is normalized block input `[B,L,C]`. QKV are `[B,L,H,D]`; adapters transpose at native/operator boundaries. Raw QK have native QK normalization applied but no positional transform. Positioned QK have native RoPE applied. Local/CLEAR consumes positioned QK; QDM consumes raw QK/V; KDA consumes normalized image hidden states. V has no positional transform. Each algorithm returns flattened `pre_o [B,L,H*D]` in native token order. M applies the native attention O exactly once, then adds optional `post_o_residual [B,L,C_out]`, then retains the native FFN, residual and modulation sequence. Thus KDA is `O(local + KDA)`; QDM is `O(local) + memory`, whose branch owns its output projection. Text queries keep the local/global text semantics and receive no KDA/QDM image-memory residual.

Each layout entry `(height, width, text_length)` is one sample in a packed sequence. `text_first` orders text/image **within each sample**; sample offsets are cumulative `h*w+t`. Every batch row has that same packed layout; ordinary batches with varying grids/lengths must be bucketed or processed in separate calls. `L == layout.length`; padding is removed by M before operator invocation, so no padded query or key participates. Images use row-major raster order. Attention never crosses sample boundaries; image queries see same-sample text and their spatial neighborhood, text queries see all same-sample original tokens. CLEAR pooled image K/V are image-query-only. This matches the existing CLEAR layout; A can import the shared type directly when extracting its implementation.

M enumerates all native image attention layers before selecting replacements: PiD `patch.0..13` (pixel blocks untouched); FLUX2 `double.0..7`, then `single.0..47`; FLUX3 `early_image.0..4`, then `joint.0..27`. Counts are architecture configuration and must match the loaded model. The ordinal is cumulative across these groups, including unselected layers. FLUX3 `joint.4` has ordinal 9; its KDA direction is `9 % 4 == 1` even if it is the only selected layer. Text/reference-only blocks are excluded. M constructs `LayerSpec` from resolved `cfg.layer_specs` and checks counts against the actual architecture at load time.

A exports from `algorithms/__init__.py`:

```python
def build_attention(config: DictConfig, *, hidden_size: int, num_heads: int,
                    layer: LayerSpec) -> nn.Module: ...
# module.forward(self, inputs: AttentionInput) -> AttentionResult
```

`config` is the **full resolved config**, not its algorithm subsection. `AlgorithmFactory` and `AttentionOperator` protocols declare these call boundaries. Operators own branch/local-gate/pooling modules, no model resources, optimizers or strategy imports. KDA direction comes from `layer.ordinal`; no state persists between calls. M includes A's explicit parameter roles in its role table; A should expose `role_parameters() -> list[RoleParameter]` on each built module, with names local to the operator and `layer=layer.name`.

## Model, roles and training boundary (M ↔ T)

M exports from `models/__init__.py`:

```python
def build_model(config: DictConfig, *, device: torch.device,
                algorithm_factory: AlgorithmFactory) -> ModelAdapter: ...
```

`ModelFactory`, `ModelAdapter`, `RoleParameter`, `ModelOutput` and `LayerPair` are actual shared declarations. Adapter operations:

```python
module: nn.Module
layers: tuple[LayerSpec, ...]
def role_parameters(self) -> list[RoleParameter]: ...
def prepare_batch(self, batch: EncodedBatch, *, generator: torch.Generator) -> TrainingBatch: ...
def forward(self, batch: TrainingBatch, *, teacher: bool = False,
            capture: tuple[str, ...] = ()) -> ModelOutput: ...
def layerwise(self, batch: TrainingBatch, callback: Callable[[LayerPair], None]) -> None: ...
```

The adapter's `module` owns every model/algorithm trainable parameter. Its `forward` implements the same signature as adapter forward so T can wrap **module** with DDP for student end-to-end training. Teacher forwards bypass DDP; no model object is built per step. `ModelOutput.prediction` and `TrainingBatch.target` share native shape. `features` maps requested stable layer names to image-query-only tensors at their configured tap. KDA/CLEAR tap pre-O; QDM taps post-O including its memory. Teacher uses the identical tap, native frozen projection and no student branch. `capture=()` avoids feature collection.

`layerwise` runs the teacher trunk under no-grad. For each selected layer, it computes an independent student attention graph from detached teacher inputs and invokes the callback with `LayerPair(layer, inputs, teacher, student)`. Both output tensors are image-query post-O attention outputs, including the student's remote branch, excluding FFN/residual. Callback performs backward immediately before returning; the teacher continues using its own output. T owns relative-MSE reduction, accumulation, manual gradient averaging and clipping. This path does not pass through full-model DDP.

A `RoleParameter(name, parameter, role, layer)` has exactly one role: `branch`, `local_gate`, `image_qkvo`, `lora` or `pooling`. Names are unique fully-qualified names in M's table; the same Parameter appears once. Frozen backbone parameters are absent. Roles describe capabilities, not current trainability; T freezes all parameters then enables roles per strategy/phase. KDA internal gates belong to `branch`; only existing independent softmax gates use `local_gate`. A1/A2 select branch (and any algorithm-required pooling), B adds local_gate and lora; CLEAR selects branch/local_gate/image_qkvo/pooling, with LoRA disabled. T owns the definitive per-phase role groups and optimizer settings.

M retains immutable base projections for teacher. Direct image projection weights and LoRA are isolated student parameters; switching teacher must bypass both without copying student updates into base weights. FLUX2 fused QKV rows and O columns use existing slice-aware projection/LoRA helpers; the fused FFN is never part of a trainable full fused parameter. Model construction must not choose strategy-specific `requires_grad` or optimizer policy. M can create needed projection helpers based on config; T selects active roles.

## Data handshake (D ↔ M ↔ T)

Shared values:

```python
Sample(sample_id: str, image_path: str, caption: str, width: int, height: int)
EncodedBatch(sample_ids: tuple[str, ...], family: Family,
             image: dict[str, Tensor], text: dict[str, Tensor], grids: tuple[tuple[int, int], ...])
TrainingBatch(inputs: dict[str, Tensor], target: Tensor, timesteps: Tensor,
              sample_ids: tuple[str, ...])
```

D exports `build_encoder(config: DictConfig, *, device: torch.device, image: bool, text: bool) -> Encoder` from `data/encoders.py`. Flags mean **load this component**; false components are absent and their encode operation is not called. Both false loads neither TE nor VAE. `Encoder.encode_image(pixels: Tensor, grids: tuple[tuple[int,int],...]) -> dict[str,Tensor]` and `encode_text(captions: list[str]) -> dict[str,Tensor]` are shared protocol methods. Online and cache preparation call the same functions under `torch.no_grad()`; returned tensors remain usable in student autograd.

Image dictionaries: PiD `pixels` (full RGB target) and `condition` (clean native downsampled VAE condition); FLUX2/FLUX3 `latents` (native packed clean tokens) and `image_ids` (native positional IDs). Text dictionaries: `embeddings`, plus native `text_ids`, `mask`, `pooled` when the family requires them. M consumes native fields without forcing the three families into one latent format. `grids` are attention token grids, not RGB dimensions. D and M must preserve the installed family's native embedding shapes and scaling. PiD full RGB stays available even with cached conditions. Noise, degradation, dropout and timesteps are generated by M's `prepare_batch`, using the supplied generator, after reading clean encodings.

D exports from `data/__init__.py`:

```python
def build_dataloader(config: DictConfig, *, rank: int, world_size: int,
                     cursor: int = 0) -> torch.utils.data.DataLoader: ...
```

It yields `EncodedBatch` with cache/online resolution performed in the training process (no CUDA encoders inside workers). Batches are bucket-compatible; each rank's deterministic cursor counts consumed **local microbatches**. T checkpoints that cursor with RNG and phase; D's sampler resumes it. `Sample.image_path` resolves relative to the manifest directory; configured manifest/cache/output paths resolve relative to launch cwd. Cache metadata compares family/resource/tokenizer identity, preprocessing, bucket and sample identity at read boundaries. No training randomness is cached. This contract defines no historical-cache compatibility.

## Configuration and entry points

`compose_config(model_config: str, algorithm_config: str, training_config: str, data_config: str, overrides: list[str] | None = None) -> DictConfig` composes shared defaults → model → algorithm → training → data → dotlist, resolves interpolations, validates and resolves runtime paths. Layer maps replace family-specific defaults before the merge. OmegaConf struct mode rejects unknown keys. Access CLEAR settings with `cfg.algorithm["clear"]` because `clear` is also a DictConfig method. New fields require explicit declarations in `configs/default.yaml`.

`config_parser() -> argparse.ArgumentParser` and `config_from_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, DictConfig]` are exported in `attention_framework.config`. T's `training.run` reuses these: `--print-config` prints `OmegaConf.to_yaml(cfg, resolve=True)` and exits before loading models/devices/data. The standalone configuration entry is:

```bash
python -m attention_framework.config --print-config \
  --model-config configs/models/pid.yaml \
  --algorithm-config configs/algorithms/sliding_qdm.yaml \
  --training-config configs/training/vdn.yaml \
  --data-config configs/data/default.yaml \
  training.batch_size=2 runtime.gpus=4 training.accumulation=2
```

All 12 model/algorithm pairs and 21 strategy combinations are represented by `COMBINATIONS`; CLEAR+VDN is rejected. Derived fields are `global_batch_size`, `selected_layers`, `layer_specs`, `feature_tap`, `feature_layers`; callers never override these. `runtime.gpus` is requested world size; T verifies it matches the launched distributed world size. Per-GPU batch × world size × accumulation gives global batch. Model resource values may be local paths or resource identities; model loading is offline and must never implicitly download. Null resources in representative defaults require explicit local resources for real training.

Config resource YAMLs ship through the `configs` package; package discovery includes `attention_framework`, models, algorithms, data, training and download, preserving `pid`. Family files are representative native architectures, not weight downloads. Real resource paths are supplied by the local configuration.
