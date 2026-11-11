"""CPU storage policy and two-rank FP32-master exact-resume checks."""
import copy
import tempfile
from unittest.mock import patch
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from attention_framework.contracts import RoleParameter
from .optim import build_optimizer, local_optimizer
from .check_cpu import Toy
from .checkpoint import save, read, restore


def config():
    return OmegaConf.create(dict(model=dict(family='flux3'), algorithm=dict(name='cross_scan_kda'),
        runtime=dict(dtype='bfloat16', optimizer_sharding=True),
        training=dict(optimizer='prodigy', lr=1., weight_decay=0., scheduler='constant', steps=4,
                      fp32_updates=True)))


def policy():
    from models.attention import AttentionProjection, original_projections
    projection = AttentionProjection(torch.nn.Linear(4, 4, dtype=torch.bfloat16), 2, 2)
    teacher_weight = projection.base.weight.detach().clone()
    build_optimizer(config(), [RoleParameter('train_weight', projection.train_weight, 'image_qkvo', 'layer')], 'clear')
    assert projection.weight.data_ptr() == projection.train_weight.data_ptr()
    assert projection.lora_A.dtype == projection.lora_B.dtype == torch.float32
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    with original_projections(projection), torch.no_grad():
        teacher = projection(x)
    projection(x).float().sum().backward()
    assert projection.train_weight.grad is not None and projection.base.weight.grad is None
    assert torch.equal(projection.base.weight, teacher_weight)
    assert torch.equal(teacher, projection.base(x))
    for optimizer, family, algorithm, dtype, expected in (
        ('prodigy', 'flux3', 'cross_scan_kda', 'bfloat16', torch.bfloat16),
        ('adamw', 'flux3', 'cross_scan_kda', 'bfloat16', torch.float32),
        ('prodigy', 'flux2', 'cross_scan_kda', 'bfloat16', torch.float32),
        ('prodigy', 'flux3', 'sliding_qdm', 'bfloat16', torch.float32),
        ('prodigy', 'flux3', 'cross_scan_kda', 'float32', torch.float32),
    ):
        cfg = config()
        cfg.training.optimizer, cfg.model.family = optimizer, family
        cfg.algorithm.name, cfg.runtime.dtype = algorithm, dtype
        model = Toy()
        decay = torch.nn.Parameter(torch.ones(2))
        roles = model.role_parameters() + [RoleParameter('kda.A_log', decay, 'branch', 'layer')]
        opt, _ = build_optimizer(cfg, roles, 'clear')
        assert model.p.dtype == model.gate.dtype == expected
        assert decay.dtype == expected
        assert model.base.dtype == torch.float32
        assert cfg.training.fp32_updates
        if optimizer == 'prodigy':
            assert all(p.dtype == torch.float32 for g in opt.param_groups for p in g['params'])


def distributed():
    dist.init_process_group('gloo')
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        cfg = config()
        model = Toy()
        opt, scheduler = build_optimizer(cfg, model.role_parameters(), 'clear')
        local = local_optimizer(opt)
        owned = [i for i, p in enumerate((model.p, model.gate))
                 if any(p is q for q in local.compute_parameters)]
        ownership = [None] * world
        dist.all_gather_object(ownership, owned)
        assert sorted(i for shard in ownership for i in shard) == [0, 1]
        assert all(len(shard) == 1 for shard in ownership)
        masters = [p for g in local.param_groups for p in g['params']]
        assert all(p.dtype == torch.float32 for p in masters)
        assert sum(p.numel() for p in masters) == 1

        def step():
            opt.zero_grad(set_to_none=True)
            (model.p + model.gate).backward()
            # Prodigy hardcodes CUDA for its two-scalar collective; use Gloo storage here.
            with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                opt.step()
            scheduler.step()

        for _ in range(3):
            step()
        assert any(not torch.equal(p, q.float()) for p, q in zip(masters, local.compute_parameters))
        assert all(s['exp_avg'].dtype == s['exp_avg_sq'].dtype == torch.float32
                   for s in local.state.values())
        roots = [tempfile.mkdtemp() if rank == 0 else None]
        dist.broadcast_object_list(roots)
        generator = torch.Generator().manual_seed(rank)
        progress = dict(step=3, stage='clear', stage_step=3, cursor=6, pending_validation=False, done=False)
        path = save(roots[0], model, opt, scheduler, generator, progress, cfg, rank, world)
        step()
        expected_weights = [p.detach().clone() for p in (model.p, model.gate)]
        expected = copy.deepcopy(local.state_dict())
        restored = Toy()
        opt2, sched2 = build_optimizer(cfg, restored.role_parameters(), 'clear')
        restore(read(path, rank, world), restored, opt2, sched2, generator)
        model, opt, scheduler = restored, opt2, sched2
        step()
        assert all(torch.equal(a, b) for a, b in zip(expected_weights, (model.p, model.gate)))
        actual = local_optimizer(opt).state_dict()
        assert expected['param_groups'] == actual['param_groups']
        for key, state in expected['state'].items():
            for name, value in state.items():
                other = actual['state'][key][name]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other, name
        dist.barrier()
        if rank == 0:
            import shutil
            shutil.rmtree(roots[0])
            print('PASS: BF16 compute, unique local FP32 masters/moments, tiny updates and exact next-step resume')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    import os
    if 'RANK' in os.environ:
        distributed()
    else:
        policy()
        print('PASS: scoped compute storage, FP32 AdamW and frozen teacher storage')
