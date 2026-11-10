"""CLEAR projection adaptation for native FLUX2/FLUX3 with spatial KDA."""
from contextlib import contextmanager
from pathlib import Path
import torch
from .attention import ImageKDA


def build_model(family, weights, device, kernel=11):
    if family == 'flux2':
        from flux2_qdm.diffsynth_model import build_model as build_flux2
        model = build_flux2(weights, 'b', device, kernel_size=kernel, training_strategy='clear')
        for block in model.qdm_blocks:
            block.kda = ImageKDA(block.orig.inner_dim, block.orig.heads, block.layer_index).to(device)
    elif family == 'flux3':
        from safetensors.torch import load_file
        from flux3_qdm.model import Flux3QDM, native
        from .flux3 import Flux3KDABlock
        with torch.device('meta'):
            base = native.Flux3DiT()
        path = Path(weights)
        base.load_state_dict(load_file(str(path / 'model.safetensors' if path.is_dir() else path)), strict=True, assign=True)
        base.to(device=device, dtype=torch.bfloat16)
        model = Flux3QDM(base, attention_mode='local-only', local_attention='sliding',
                         sliding_kernel_size=kernel, replace_image_blocks=True)
        index = 0
        for blocks in (base.content_mode_blocks['image'], base.single_blocks):
            for i, wrapped in enumerate(blocks):
                blocks[i] = Flux3KDABlock(wrapped.orig, index, kernel).to(device=device)
                index += 1
    else:
        raise ValueError(f'Unknown model family: {family}')
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if '.kda.' in name or name.endswith('train_weight'):
            param.requires_grad_(True)
    return model


@contextmanager
def teacher_mode(model):
    from flux2_qdm.model import disable_lora
    previous = model.teacher_mode
    model.teacher_mode = True
    try:
        with disable_lora(model):
            yield
    finally:
        model.teacher_mode = previous


def delta_state_dict(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters()
            if '.kda.' in name or name.endswith('train_weight')}


def load_delta(model, state):
    expected = {n: p for n, p in model.named_parameters() if '.kda.' in n or n.endswith('train_weight')}
    if expected.keys() != state.keys():
        raise ValueError('KDA checkpoint parameter keys differ from model')
    with torch.no_grad():
        for name, p in expected.items():
            if p.shape != state[name].shape:
                raise ValueError(f'Checkpoint shape mismatch: {name}')
            p.copy_(state[name])
