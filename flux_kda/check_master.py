"""Check sharded master updates, legacy restore, and exact FP32 resume."""
import copy
import os
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from flux2_qdm.clear_strategy import make_optimizer
from .optimizer import MasterProdigy
from .checkpoint import save_optimizer, load_optimizer


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(42)
    params = [torch.nn.Parameter((torch.randn(n, device='cuda') * .02).bfloat16()) for n in (17,29,43,61,89,113)]
    legacy, _ = make_optimizer(params, shard=True)
    for p in params: p.grad = torch.ones_like(p)
    legacy.step()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_optimizer(legacy, root, 0, dist.get_rank())
        optimizer, _ = make_optimizer(params, shard=True, optimizer_class=MasterProdigy)
        load_optimizer(optimizer, root, 0, dist.get_rank())
        local = optimizer.optim
        masters = [p for g in local.param_groups for p in g['params']]
        for p,m in zip(local.compute_parameters, masters):
            assert m.dtype == torch.float32
            torch.testing.assert_close(m, p.float(), rtol=0, atol=0)
        reference = [torch.nn.Parameter(m.detach().clone()) for m in masters]
        # The local baseline participates in the same global-statistics reduction.
        from prodigyopt import Prodigy
        baseline = Prodigy(reference)
        baseline.load_state_dict(copy.deepcopy(local.state_dict()))
        initial = [m.detach().clone() for m in masters]
        for step in range(32):
            for p in params: p.grad = torch.ones_like(p) * (1 + step / 32)
            for p,r in zip(local.compute_parameters,reference): r.grad = p.grad.float()
            optimizer.step(); baseline.step()
            for p,m,r in zip(local.compute_parameters, masters, reference):
                torch.testing.assert_close(m, r, rtol=0, atol=0)
                torch.testing.assert_close(p, m.bfloat16(), rtol=0, atol=0)
            if step == 4:
                save_optimizer(optimizer, root, 5, dist.get_rank())
                optimizer, _ = make_optimizer(params, shard=True, optimizer_class=MasterProdigy)
                load_optimizer(optimizer, root, 5, dist.get_rank())
                local = optimizer.optim
                masters = [p for g in local.param_groups for p in g['params']]
        assert any(not torch.equal(a,b) for a,b in zip(initial,masters))
        assert all(v.dtype == torch.float32 for st in local.state.values() for v in st.values() if torch.is_tensor(v))
    if dist.get_rank()==0: print('Three-rank FP32 master update parity, legacy BF16 conversion and exact save/restore passed',flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
