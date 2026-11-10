"""Native FLUX3 loading, stateless packed execution, and stage/delta APIs.

The base model is allocated on meta and strict assigned from the native checkpoint.
The 28 joint blocks and optionally five early image blocks own QDM additions.
Early text/reference blocks, modulation, FFN and final layer retain native math.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import QDMBlock, SampleLayout
from .lora import LoRALinear
from .vendor.native import native_module

native = native_module()


def normalize_qdm_layers(layers, depth=28):
    """Canonical zero-based joint-layer layout; None retains the original all-layer default."""
    if layers is None:
        return tuple(range(depth))
    if isinstance(layers, str):
        layers = [] if not layers.strip() else [int(x.strip()) for x in layers.split(',')]
    layers = tuple(layers)
    if any(type(i) is not int or i < 0 or i >= depth for i in layers):
        raise ValueError(f'QDM layer indices must lie in [0, {depth})')
    if len(set(layers)) != len(layers):
        raise ValueError('Duplicate QDM layer indices')
    return tuple(sorted(layers))


def iter_qdm_blocks(model):
    return (m for m in model.modules() if isinstance(m, QDMBlock))


def _stage(stage):
    stage = str(stage).upper()
    if stage not in ("A1", "A2", "B"):
        raise ValueError(f"Unknown stage {stage!r}")
    return stage


def _delta_named_parameters(model):
    """Membership is architectural, independent of current requires_grad flags."""
    ids = set()
    for block in iter_qdm_blocks(model):
        if block.remote is not None:
            ids.update(id(p) for p in block.remote.parameters())
        ids.update(id(p) for p in block.softmax_gate.parameters())
    for m in model.modules():
        if isinstance(m, LoRALinear):
            ids.update((id(m.lora_A), id(m.lora_B)))
    return ((name, p) for name, p in model.named_parameters() if id(p) in ids)


def configure_stage(model, stage, rank=64, alpha=64, retain_lora=False):
    stage = _stage(stage)
    if stage == "B" or retain_lora:
        for block in iter_qdm_blocks(model):
            for name in ("q_proj", "k_proj", "v_proj", "attn_out"):
                layer = getattr(block.orig, name)
                if isinstance(layer, LoRALinear):
                    if layer.rank != rank or layer.alpha != alpha:
                        raise ValueError("Existing LoRA rank/alpha mismatch")
                else:
                    setattr(block.orig, name, LoRALinear(layer, rank, alpha))
    model.requires_grad_(False)
    for block in iter_qdm_blocks(model):
        if block.remote is not None:
            block.remote.requires_grad_(True)
        block.softmax_gate.requires_grad_(stage == "B")
        for m in block.orig.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad_(stage == "B")
                m.lora_B.requires_grad_(stage == "B")
    model.stage = stage
    return model


def optimizer_groups(model, stage):
    """Approved AdamW groups: structural small tensors, big matrices, and LoRA."""
    stage = _stage(stage)
    small_ids = set()
    for block in iter_qdm_blocks(model):
        remote = block.remote
        if remote is None:
            continue
        small_ids.update(
            (
                id(remote.short_conv_k.weight),
                id(remote.short_conv_v.weight),
                id(remote.norm.weight),
                id(remote.output_gate.up.weight),
                id(remote.output_gate.up.bias),
            )
        )
    buckets = {"big": [], "small": [], "lora": []}
    for name, p in _delta_named_parameters(model):
        if not p.requires_grad:
            continue
        group = "lora" if "lora_" in name else "small" if id(p) in small_ids else "big"
        buckets[group].append(p)
    multipliers = {"big": 0.4 if stage == "B" else 1.0, "small": 2.0 if stage == "B" else 5.0, "lora": 1.0}
    return [
        dict(
            params=params,
            group_name=name,
            lr=1e-4 * multipliers[name],
            weight_decay=0.0,
            eps=2e-9 if name == "small" else 1e-8,
        )
        for name, params in buckets.items()
        if params
    ]


def delta_state_dict(model):
    # Serialization copies these views to CPU synchronously before the next update.
    # Avoid duplicating the entire delta on GPU just to serialize a checkpoint.
    return {name: p.detach() for name, p in _delta_named_parameters(model)}


def load_delta(model, state, strict=True):
    expected = dict(_delta_named_parameters(model))
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if strict and (missing or unexpected):
        raise RuntimeError(f"Delta keys mismatch: missing={missing}, unexpected={unexpected}")
    for key in set(expected) & set(state):
        if expected[key].shape != state[key].shape:
            raise RuntimeError(f"Delta shape mismatch for {key}: {state[key].shape} vs {expected[key].shape}")
    with torch.no_grad():
        for key in set(expected) & set(state):
            expected[key].copy_(state[key])
    return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)


@contextmanager
def execution_mode(model, teacher=False):
    """Select next forward mode. Checkpoint closures capture it, never reread it."""
    modules = [model] + list(iter_qdm_blocks(model))
    previous = [(m, getattr(m, "teacher_mode", False)) for m in modules]
    for m in modules:
        m.teacher_mode = bool(teacher)
    try:
        yield model
    finally:
        for m, value in previous:
            m.teacher_mode = value


def _layouts(packed):
    indices = packed.mode_indices
    count = packed.num_samples
    if any(b > a for a, b in indices["image_cond"]):
        raise ValueError("FLUX3-QDM currently supports T2I only; real image_cond tokens are rejected")
    grids = getattr(packed, "image_grids", None)
    if grids is None:
        raise ValueError("Packed.image_grids must explicitly provide (height,width) for every sample")
    if len(grids) != count:
        raise ValueError("image_grids sample count mismatch")
    offset = 0
    layouts = []
    for i, grid in enumerate(grids):
        h, w = map(int, grid)
        a, b = indices["image"][i]
        ta, tb = indices["text"][i]
        ni, nt = b - a, tb - ta
        if min(h, w) < 1 or ni != h * w:
            raise ValueError(f"Grid {grid} does not match sample {i} image tokens {ni}")
        ids = packed.ids["image"][a:b]
        # The native image IDs use (0,row*0.5,col*0.5,0); validate exact row-major
        # Cartesian order without inferring dimensions from token count.
        row, col = ids[:, 1].reshape(h, w), ids[:, 2].reshape(h, w)
        if not torch.equal(row, row[:, :1].expand_as(row)) or not torch.equal(col, col[:1].expand_as(col)):
            raise ValueError(f"Sample {i} IDs must have row-major image grid order")
        if h > 1 and not bool(torch.all(row[1:, 0] > row[:-1, 0])):
            raise ValueError("Image rows must ascend")
        if w > 1 and not bool(torch.all(col[0, 1:] > col[0, :-1])):
            raise ValueError("Image columns must ascend")
        layouts.append(
            SampleLayout(offset, offset + ni + nt, offset, offset + ni, offset + ni, offset + ni + nt, (h, w))
        )
        offset += ni + nt
    return tuple(layouts)


class Flux3QDM(nn.Module):
    def __init__(self, base, region_size=8, qdm_layers=None, halo=0, attention_mode="hybrid", memory_text_write=False, local_attention="halo", sliding_kernel_size=11, replace_image_blocks=False):
        super().__init__()
        if base.flow_map:
            raise ValueError("The base FLUX3 T2I checkpoint must not use flow_map")
        self.base = base
        self.teacher_mode = False
        self.attention_mode = attention_mode
        self.memory_text_write = bool(memory_text_write)
        self.replace_image_blocks = bool(replace_image_blocks)
        self.qdm_layers = normalize_qdm_layers(qdm_layers, len(base.single_blocks))
        selected = [(base.single_blocks, i, i) for i in self.qdm_layers]
        if self.replace_image_blocks:
            selected = [(base.content_mode_blocks['image'], i, f'image_{i}')
                        for i in range(len(base.content_mode_blocks['image']))] + selected
        for blocks, i, label in selected:
            block = blocks[i]
            if isinstance(block, QDMBlock):
                raise ValueError("Native blocks already wrapped")
            wrapped = QDMBlock(block, region_size=region_size, halo=halo, attention_mode=attention_mode, memory_text_write=memory_text_write,
                               local_attention=local_attention, sliding_kernel_size=sliding_kernel_size)
            wrapped.layer_index = label
            # New trainable parameters remain FP32; preserve native base dtype.
            if wrapped.remote is not None:
                wrapped.remote.to(device=block.q_proj.weight.device)
            wrapped.softmax_gate.to(device=block.q_proj.weight.device)
            blocks[i] = wrapped
        self.base.requires_grad_(False)

    @property
    def single_blocks(self):
        return self.base.single_blocks

    @property
    def qdm_blocks(self):
        return tuple(iter_qdm_blocks(self))

    def forward(self, packed, use_gradient_checkpointing=False, alignment_callback=None, collect_features=False):
        features = []
        samples = _layouts(packed)
        # Capture the execution mode once. Nonreentrant checkpoint recomputation is
        # correct even if a teacher/student context has since exited or changed.
        teacher = bool(self.teacher_mode)
        if alignment_callback is not None and not teacher:
            raise ValueError("Alignment callbacks require teacher execution mode")
        base = self.base
        data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for m in native.MODES:
            if data[m].shape[0] == 0:
                data[m] = data[m].new_zeros(1, data[m].shape[1])
                ids[m] = ids[m].new_zeros(1, 4)
                timesteps[m] = timesteps[m].new_zeros(1)

        def mod_key(m):
            return "txt" if m == "text" else m

        # Frozen embeddings/modulations need no graph for offline-cache inputs.
        with torch.no_grad():
            vec = {m: base.time_in(native.timestep_embedding(timesteps[m])) for m in native.MODES}
            early = {m: base.early_stream_modulations[mod_key(m)](vec[m]) for m in native.MODES}
            single = {m: base.single_stream_modulations[mod_key(m)](vec[m]) for m in native.MODES}
            x = {m: (base.txt_in if m == "text" else base.emb_in[m])(data[m]) for m in native.MODES}
            freqs = {m: native.rope_embed(ids[m][None]) for m in native.MODES}

        def call_block(block, z, f, bounds, mod, layouts):
            # Each closure owns its block/bounds/layouts and the forward's mode;
            # later teacher calls cannot switch student checkpoint recomputation.
            def run(z, f, shift, scale, gate):
                if not isinstance(block, QDMBlock):
                    return block(z, f, bounds, (shift, scale, gate))
                options = dict(return_attention_feature=True) if collect_features and getattr(block, 'collect_feature', False) else {}
                return block(z, f, bounds, (shift, scale, gate), samples=layouts,
                             alignment_callback=alignment_callback, teacher_mode=teacher, **options)
            if use_gradient_checkpointing and not teacher and torch.is_grad_enabled():
                result = checkpoint(run, z, f, *mod, use_reentrant=False)
            else:
                result = run(z, f, *mod)
            if isinstance(result, tuple):
                result, feature = result
                features.append(feature)
            return result

        # Early image tokens are packed without text. Their QDM and local
        # visibility use the same explicit grids, but no fabricated text tokens.
        image_samples = tuple(SampleLayout(a, b, a, b, b, b, s.grid)
                              for (a, b), s in zip(packed.mode_indices['image'], samples, strict=True))
        for d in range(len(base.txt_mode_blocks)):
            block = base.content_mode_blocks['image'][d]
            if isinstance(block, QDMBlock):
                x['image'] = call_block(block, x['image'], freqs['image'], packed.mode_indices['image'], early['image'], image_samples)
            else:
                with torch.no_grad():
                    x['image'] = block(x['image'], freqs['image'], packed.mode_indices['image'], early['image'])
            with torch.no_grad():
                x['image_cond'] = base.content_mode_blocks['image_cond'][d](
                    x['image_cond'], freqs['image_cond'], packed.mode_indices['image_cond'], early['image_cond'])
                x["text"] = base.txt_mode_blocks[d](
                    x["text"], freqs["text"], packed.mode_indices["text"], early["text"]
                )
        chunks, id_chunks, mods, bounds, placement, offset = [], [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in native.MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        freqs_sm = native.rope_embed(torch.cat(id_chunks)[None])
        mods = tuple(torch.cat(c) for c in mods)
        for block in base.single_blocks:
            h = call_block(block, h, freqs_sm, bounds, mods, samples)
        image = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                image[a:b] = h[start : start + b - a]
        # Native final projection stays frozen but propagates student image gradients.
        result = base.final_layer["image"](image, vec["image"])
        return (result, features) if collect_features else result


def build_model(weights_path, stage, device, region_size=8, lora_rank=64, lora_alpha=64, qdm_layers=None, halo=0, attention_mode="hybrid", memory_text_write=False, retain_lora=False, local_attention="halo", sliding_kernel_size=11, replace_image_blocks=False):
    from safetensors.torch import load_file

    path = Path(weights_path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    with torch.device("meta"):
        base = native.Flux3DiT()
    state = load_file(str(path), device="cpu")
    if any(t.dtype != torch.bfloat16 for t in state.values()):
        raise ValueError("Native FLUX3 base checkpoint must contain BF16 parameters")
    base.load_state_dict(state, strict=True, assign=True)
    del state
    base.to(device=device)
    model = Flux3QDM(base, region_size, qdm_layers=qdm_layers, halo=halo, attention_mode=attention_mode, memory_text_write=memory_text_write,
                     local_attention=local_attention, sliding_kernel_size=sliding_kernel_size, replace_image_blocks=replace_image_blocks)
    return configure_stage(model, stage, rank=lora_rank, alpha=lora_alpha, retain_lora=retain_lora)
