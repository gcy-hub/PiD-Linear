"""Inference loader shared by global, tiled and checkpoint benchmarks."""
def load_decoder(config, device, model, *, checkpoint=None):
    if model == 'original':
        if checkpoint is not None:
            raise ValueError('original PiD does not take student weights')
        from models.base import load_native
        return load_native(config, device).eval().requires_grad_(False), 0

    from algorithms import build_attention
    from models import build_model
    from models.attention import fuse_lora_projections

    if model not in ('export', 'pid-linear') or (model == 'pid-linear') != (checkpoint is not None):
        raise ValueError('use export, or pid-linear with a complete training checkpoint')
    adapter = build_model(config, device=device, algorithm_factory=build_attention)
    if checkpoint is not None:
        from pathlib import Path
        import json
        import torch
        from training.checkpoint import restore_weights

        if len(adapter.sites) != 2 or any(s.teacher for s in adapter.sites):
            raise ValueError('load the PiT student architecture from its training runtime')
        checkpoint = Path(checkpoint)
        manifest = json.loads((checkpoint / 'complete.json').read_text())
        state = torch.load(checkpoint / 'rank-0.pt', map_location='cpu', weights_only=False)
        if state['progress']['step'] != manifest['step']:
            raise ValueError('checkpoint step does not match manifest')
        restore_weights(adapter, state['weights'])
    net = adapter.base
    net.eval().requires_grad_(False)
    return net, fuse_lora_projections(net)
