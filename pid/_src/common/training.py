"""Shared distributed training setup and run ownership."""
import os
import fcntl
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist

def init_distributed():
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("PiD training requires CUDA")
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return (dist.get_rank() if dist.is_initialized() else 0), world, torch.device("cuda", local)


def consensus(value, device):
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def broadcast_object(value, rank):
    if dist.is_initialized():
        objects = [value if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]
    return value


def lock_training_run(run_dir, rank):
    """Rank zero holds the run lock until exit, including orphaned-launcher cases."""
    lock, error = None, None
    if rank == 0:
        lock = (Path(run_dir) / ".train.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # DataLoader forks must not keep the lock alive after rank zero dies.
            os.register_at_fork(after_in_child=lock.close)
        except BlockingIOError:
            lock.close()
            lock = None
            error = f"Another trainer already owns {run_dir}"
    error = broadcast_object(error, rank)
    if error:
        raise RuntimeError(error)
    return lock
