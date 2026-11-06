"""Shared DiT compute contract for training and both sampling entry points."""

DIT_COMPUTE = dict(schema=1, autocast_dtype='bfloat16', autocast_cache=False,
                   frozen_weights='bfloat16', trainable_weights='float32')


def dit_autocast():
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)
