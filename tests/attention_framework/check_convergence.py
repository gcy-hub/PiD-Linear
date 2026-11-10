"""Check installed entrypoints and protected paths against the integration baseline."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = '418cf2f1651db1443f877a81e3fe7514a969da6c'
subprocess.run(['git', 'diff', '--exit-code', BASE, '--', 'pid/'], cwd=ROOT, check=True)
deleted = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=D', BASE], cwd=ROOT, text=True)
for name in deleted.splitlines():
    assert not any(part in ('weights', 'logs', 'experiments') for part in Path(name).parts), name
    # Exact deleted paths must have no remaining tracked callers.
    result = subprocess.run(['git', 'grep', '-n', '-F', name, '--', '*.py', '*.sh'], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 1, result.stdout
for module in ('training.run', 'data.prepare', 'training.validate', 'attention_framework.config', 'download.framework'):
    subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
from importlib.metadata import distribution
dist = distribution('pid')
assert dist is not None
print('Protected pid paths, deleted callers, package discovery and integrated CLI help passed')
