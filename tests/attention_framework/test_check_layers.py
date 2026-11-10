"""Exercise the layer acceptance gradient checks without CUDA fixtures."""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import torch
from torch import nn

import check_layers
from attention_framework.contracts import RoleParameter


class Site(nn.Module):
    def __init__(self, disconnected):
        super().__init__()
        self.orig = nn.Linear(1, 1)
        self.branch = nn.Parameter(torch.tensor(2.))
        self.teacher = True
        self.disconnected = disconnected
        self.clean_grads = []

    def role_parameters(self):
        return [RoleParameter('branch', self.branch, 'branch', 'patch.1')]

    def forward(self):
        if self.teacher:
            return self.orig(torch.ones(1, 1))
        self.clean_grads.append(self.branch.grad is None)
        shared = self.branch.square()
        unrelated = torch.ones((), requires_grad=True)
        self.feature = unrelated if self.disconnected == 'feature' else shared
        return unrelated if self.disconnected == 'output' else shared.square()


class LayerGradientChecks(unittest.TestCase):
    def test_independent_graphs_and_gradient_assertions(self):
        for disconnected in (None, 'feature', 'output'):
            with self.subTest(disconnected=disconnected):
                site = Site(disconnected)
                with patch.object(check_layers.torch.cuda, 'is_available', return_value=True), \
                     patch.object(check_layers, 'COMBINATIONS', [('pid', 'clear', 'clear')]), \
                     patch.object(check_layers, 'compose_config', return_value=None), \
                     patch.object(check_layers, 'fixture', return_value=(site, site)), \
                     redirect_stdout(StringIO()):
                    if disconnected is None:
                        check_layers.main()
                        self.assertEqual(site.clean_grads, [True, True])
                    else:
                        with self.assertRaises(AssertionError):
                            check_layers.main()


if __name__ == '__main__':
    unittest.main()
