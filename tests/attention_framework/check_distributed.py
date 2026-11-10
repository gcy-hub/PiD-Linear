"""Four-rank collective/checkpoint checks; real family DDP runs in *_cycle.

--cpu --world-size 2 provides a bounded Gloo harness check, not GPU acceptance.
"""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from training.distributed import setup, average_gradients
from training.check_cpu import Toy
from attention_framework.contracts import RoleParameter
from training.optim import build_optimizer
from training.checkpoint import save, read, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--world-size', type=int, default=4)
    args = parser.parse_args()
    device, rank, world = setup(args.world_size)
    try:
        assert args.cpu or device.type == 'cuda', 'GPU acceptance requires CUDA'
        expected = (world + 1) / 2
        toy = Toy().to(device)
        toy.requires_grad_(False)
        toy.p.requires_grad_(True)
        (toy.p * (rank + 1)).backward()
        average_gradients([toy.p])
        assert toy.p.grad.item() == expected
        # DDP accumulation must equal a single-process global-batch gradient.
        linear = torch.nn.Linear(1, 1, bias=False).to(device)
        ddp = DDP(linear, device_ids=[device.index] if device.type == 'cuda' else None)
        for micro in range(2):
            from contextlib import nullcontext
            with ddp.no_sync() if micro == 0 else nullcontext():
                (ddp(torch.tensor([[float(rank + 1)]], device=device)).sum() / 2).backward()
        assert linear.weight.grad.item() == expected
        toy.gate.requires_grad_(True)
        toy.extra = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), device=device)) for _ in range(world)])
        roles = toy.role_parameters() + [RoleParameter(f'extra.{i}', p, 'branch', 'layer')
                                         for i, p in enumerate(toy.extra)]
        toy.role_parameters = lambda: roles
        parameters = [r.parameter for r in roles]
        cfg = OmegaConf.create(dict(training=dict(optimizer='adamw' if args.cpu else 'prodigy', lr=.01, weight_decay=0., scheduler='constant', steps=4),
                                    runtime=dict(optimizer_sharding=True)))
        optimizer, scheduler = build_optimizer(cfg, toy.role_parameters(), 'clear')
        generator = torch.Generator(device=device).manual_seed(42 + rank)
        def update():
            optimizer.zero_grad()
            ((toy.p + toy.gate + sum(toy.extra)) * torch.rand((), device=device, generator=generator)).backward()
            average_gradients(parameters)
            optimizer.step()
            scheduler.step()
        update()
        root = Path(os.environ['ACCEPTANCE_REPORT_DIR'])
        progress = dict(step=1, stage='clear', stage_step=1, cursor=2, pending_validation=True, done=False)
        directory = save(root, toy, optimizer, scheduler, generator, progress, cfg, rank, world)
        state = read(directory, rank, world)
        update()
        expected_weights = [p.detach().clone() for p in parameters]
        restore(state, toy, optimizer, scheduler, generator)
        update()
        for got, want in zip(parameters, expected_weights):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        # All ranks must reject an incomplete rank set, including an absent peer.
        dist.barrier()
        missing = directory / f'rank-{world-1}.pt'
        backup = missing.with_suffix('.hidden')
        if rank == 0:
            missing.rename(backup)
        dist.barrier()
        try:
            read(directory, rank, world)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing peer shard accepted')
        dist.barrier()
        if rank == 0:
            backup.rename(missing)
        dist.barrier()
        print(f'rank {rank}: A1 average, DDP accumulation, sharded optimizer/RNG next-step and missing peer passed')
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized()
    if rank == 0:
        # Delivered checks exercise failed publication, FP32 master restore and
        # interrupted standalone validation without replacing real cycle tests.
        import subprocess
        for module in ('training.check_cpu', 'training.check_validate_cpu'):
            subprocess.run([sys.executable, '-m', module], check=True,
                           env={k: v for k, v in dict(os.environ, CUDA_VISIBLE_DEVICES='').items()
                                if k not in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')})


if __name__ == '__main__':
    main()
