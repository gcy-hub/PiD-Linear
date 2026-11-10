"""Check sharded Prodigy updates and checkpoint restore against unsharded Prodigy."""
import os
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import shutil
from .checkpoint import save_optimizer, load_optimizer, move_optimizer_state
from flux2_qdm.clear_strategy import make_optimizer


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(n, device='cuda')) for n in (17, 29, 43, 61, 89, 113)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizer, _ = make_optimizer(params, shard=True)
    baseline, _ = make_optimizer(reference)
    for step in range(4):
        for p, r in zip(params, reference):
            p.grad = p.detach().square() + .3
            r.grad = r.detach().square() + .3
        move_optimizer_state(optimizer, 'cuda')
        optimizer.step(); baseline.step()
        move_optimizer_state(optimizer, 'cpu')
        assert all(not value.is_cuda for state in optimizer.optim.state.values()
                   for value in state.values() if torch.is_tensor(value))
        for p, r in zip(params, reference):
            torch.testing.assert_close(p, r, rtol=1e-6, atol=1e-7)
        if step == 1:
            path = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(path)
            root = Path(path[0])
            save_optimizer(optimizer, root, step, dist.get_rank())
            optimizer, _ = make_optimizer(params, shard=True)
            load_optimizer(optimizer, root, step, dist.get_rank())
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(root)
    if dist.get_rank() == 0:
        print('Three-rank Prodigy parity and checkpoint restore passed', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
