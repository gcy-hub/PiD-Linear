"""Two-rank Gloo check of A1 averaging and sharded optimizer updates/resume."""
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from .distributed import setup, average_gradients
from .check_cpu import Toy
from .optim import build_optimizer
from .checkpoint import save, read, restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    device, rank, world = setup(2)
    try:
        model = Toy()
        model.requires_grad_(False)
        model.p.requires_grad_(True)
        (model.p * (rank+1)).backward()
        average_gradients([model.p])
        assert model.p.grad.item() == 1.5
        linear = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        with linear.no_sync():
            (linear(torch.tensor([[float(rank+1)]])).sum()/2).backward()
        (linear(torch.tensor([[float(rank+1)]])).sum()/2).backward()
        assert linear.module.weight.grad.item() == 1.5
        model.gate.requires_grad_(True)
        config = OmegaConf.create(dict(training=dict(optimizer='adamw', lr=.01, weight_decay=0., scheduler='constant', steps=3), runtime=dict(optimizer_sharding=True)))
        roles = model.role_parameters()
        optimizer, scheduler = build_optimizer(config, roles, 'clear')
        for _ in range(3):
            optimizer.zero_grad()
            (model.p + model.gate).backward()
            average_gradients([model.p, model.gate])
            optimizer.step()
            scheduler.step()
        generator = torch.Generator().manual_seed(rank)
        progress = dict(step=3, stage='clear', stage_step=3, cursor=6, pending_validation=False, done=True)
        directory = save(args.output, model, optimizer, scheduler, generator, progress, config, rank, world)
        before = [p.detach().clone() for p in (model.p, model.gate)]
        state = read(directory, rank, world)
        restore(state, model, optimizer, scheduler, generator)
        assert all(torch.equal(a, b) for a, b in zip(before, (model.p, model.gate)))
        if rank == 0:
            print('Two-rank CPU A1 averaging and DDP and sharded AdamW checkpoint passed')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
