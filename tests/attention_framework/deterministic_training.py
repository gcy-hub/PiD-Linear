"""Strict reproducibility controls for exact training continuation checks."""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def configure():
    # Set before training creates CUDA BLAS handles, including in each new rank.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    configure()
    runpy.run_module('training.run', run_name='__main__')
