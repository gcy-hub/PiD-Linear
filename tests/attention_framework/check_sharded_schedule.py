"""Exact next-update restore with ZeRO and changing learning rates (torchrun)."""
import copy
import random
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from training.check_cpu import Toy
from training.distributed import setup, average_gradients
from training.optim import build_optimizer, local_optimizer
from training import checkpoint


def main():
    device, rank, world = setup(2)
    try:
        for schedule, warmup in [('constant', 0), ('cosine', 0), ('cosine', 4)]:
            model = Toy().to(device)
            roles = model.role_parameters()
            cfg = OmegaConf.create(dict(training=dict(optimizer='adamw', lr=.01,
                weight_decay=0., scheduler=schedule, vdn=dict(a2=dict(steps=8, warmup=warmup))),
                runtime=dict(optimizer_sharding=True)))
            optimizer, scheduler = build_optimizer(cfg, roles, 'a2')
            generator = torch.Generator(device=device).manual_seed(rank)
            def update():
                optimizer.zero_grad()
                (model.p.square() + model.gate.square()).backward()
                average_gradients([r.parameter for r in roles])
                optimizer.step()
                scheduler.step()
            update()
            state = copy.deepcopy(dict(weights=checkpoint.weights(model),
                optimizer=local_optimizer(optimizer).state_dict(), scheduler=scheduler.state_dict(),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(), cuda=None, generator=generator.get_state())))
            expected_lr = scheduler.get_last_lr()
            update()
            expected = checkpoint.weights(model)
            optimizer, scheduler = build_optimizer(cfg, roles, 'a2')
            checkpoint.restore(state, model, optimizer, scheduler, generator)
            assert [g['lr'] for g in optimizer.param_groups] == expected_lr, (schedule, warmup)
            update()
            for name, value in checkpoint.weights(model).items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            print(f'rank {rank}: exact sharded {schedule} warmup={warmup} resume passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
