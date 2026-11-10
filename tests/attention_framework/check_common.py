"""Native small-layer matrix plus CUDA operators and delivered numerical references."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for script in ('check_algorithms.py', 'check_layers.py', 'check_a1_ratio.py'):
    subprocess.run([sys.executable, str(Path(__file__).with_name(script))], cwd=ROOT, check=True)
for module in ('algorithms.checks.cpu', 'models.check_cpu'):
    subprocess.run([sys.executable, '-m', module], cwd=ROOT, check=True,
                   env=dict(__import__('os').environ, CUDA_VISIBLE_DEVICES=''))
