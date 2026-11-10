"""A1 must normalize the sums across sample callbacks for each selected layer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
import torch
from training.strategies.vdn import backward_layerwise, teacher_energy


def check(device):
    p = torch.nn.Parameter(torch.tensor(1., device=device))
    class Adapter:
        def layerwise(self, batch, callback):
            for source, target in ((2., 1.), (1., 3.)):
                callback(SimpleNamespace(layer=SimpleNamespace(name='patch.0'),
                    student=p * source, teacher=torch.tensor(target, device=device)))
    expected = torch.nn.Parameter(p.detach().clone())
    numerator = (expected * 2 - 1).square() + (expected - 3).square()
    (numerator / 10).backward()
    adapter = Adapter()
    backward_layerwise(adapter, None, teacher_energy(adapter, [None]))
    torch.testing.assert_close(p.grad, expected.grad, rtol=1e-6, atol=1e-6)
    print('A1 callback ratio-of-sums gradient matches concatenated global reference')


if __name__ == '__main__':
    check(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
