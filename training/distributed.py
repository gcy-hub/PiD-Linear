"""Rank setup and A1 synchronization without full-model DDP."""
import os
import torch
import torch.distributed as dist


def setup(world_size):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != world_size:
        raise ValueError(f'Configured world size {world_size} differs from launched {world}')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, int(os.environ.get('RANK', '0')), world


def average_gradients(parameters):
    if not dist.is_initialized():
        return
    for p in parameters:
        present = torch.tensor(int(p.grad is not None), device=p.device)
        dist.all_reduce(present)
        if present.item():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad.div_(dist.get_world_size())


def barrier():
    if dist.is_initialized():
        dist.barrier()
