"""Compare A1 ratio-of-sums gradients across rows, accumulation and Gloo ranks."""
import tempfile
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .strategies.vdn import teacher_energy, backward_layerwise
from .distributed import average_gradients


class Pairs:
    def __init__(self, rows):
        self.p = torch.nn.Parameter(torch.tensor(.7))
        self.rows = rows

    def layerwise(self, batch, callback):
        for layer, factor in (('a', 1.), ('b', 3.)):
            for values in batch.split(1) if self.rows else (batch,):
                teacher = values * factor
                student = self.p * values + 1
                callback(SimpleNamespace(layer=SimpleNamespace(name=layer),
                                         teacher=teacher, student=student))


def reference(values):
    p = torch.tensor(.7, requires_grad=True)
    loss = sum(((p*values + 1 - factor*values).square().sum() /
                (factor*values).square().sum()) for factor in (1., 3.))
    loss.backward()
    return p.grad


def check(values, expected, rows, accumulation):
    adapter = Pairs(rows)
    batches = values.chunk(accumulation)
    energies = teacher_energy(adapter, batches)
    assert all(not value.requires_grad for value in energies.values())
    for batch in batches:
        backward_layerwise(adapter, batch, energies)
    average_gradients([adapter.p])
    torch.testing.assert_close(adapter.p.grad, expected, rtol=1e-6, atol=1e-7)


def worker(rank, rendezvous, values):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        for rows in (False, True):
            for accumulation in (1, 2):
                check(values.chunk(2)[rank], reference(values), rows, accumulation)
    finally:
        dist.destroy_process_group()


def main():
    values = torch.tensor([[1., 2.], [4., 8.], [2., 3.], [11., 17.]])
    for rows in (False, True):
        for accumulation in (1, 2, 4):
            check(values, reference(values), rows, accumulation)
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root + '/gloo', values), nprocs=2, join=True)
    print('PASS: nonuniform teacher energies, two layers, row/batched callbacks, accumulation 1/2/4, two Gloo ranks match global ratio-of-sums')


if __name__ == '__main__':
    main()
