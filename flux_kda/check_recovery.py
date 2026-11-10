"""Exercise cron decisions and validation recovery without launching GPU work."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from . import reboot_watch as watch


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(watch, 'active_workers', return_value=True), patch.object(watch.subprocess, 'Popen') as launch:
            assert watch.check(root) == 'running'
            launch.assert_not_called()
        with patch.object(watch, 'active_workers', return_value=False), patch.object(watch.subprocess, 'run') as gpu, patch.object(watch.subprocess, 'Popen') as launch:
            (root / 'STOP').touch()
            assert watch.check(root) == 'stopped'
            (root / 'STOP').unlink()
            (root / 'exit-code').write_text('1')
            assert watch.check(root) == 'training-error'
            (root / 'exit-code').unlink()
            gpu.return_value = SimpleNamespace(stdout='1, 2000\n2, 46000\n3, 46000\n')
            assert watch.check(root) == 'waiting-for-gpus'
            launch.assert_not_called()
            gpu.return_value.stdout = '1, 46000\n2, 46000\n3, 46000\n'
            launch.return_value.pid = 123
            assert watch.check(root) == 'resumed'
            assert (root / 'launcher.pid').read_text().strip() == '123'
            launch.assert_called_once()
        fake = root / 'python'
        fake.write_text('''#!/usr/bin/env python3
import sys,json
from pathlib import Path
args=sys.argv[1:]
if args[0]=='-c': print(300)
elif args[:2]==['-m','flux_kda.cache_flux3']: pass
else:
 output=Path(args[args.index('--output')+1]); step=int(args[args.index('--steps')+1])
 with Path(__file__).with_name('calls').open('a') as f: f.write(json.dumps(args)+'\\n')
 if 'flux_kda.benchmark' in args: (output/'kda.json').write_text('{}')
''')
        fake.chmod(0o755)
        script = watch.LAUNCHER.read_text().replace('cd "$(dirname "$0")/.."', f'cd "{watch.REPO}"')
        script = script.replace('PYTHON="${PYTHON:-python}"', f'PYTHON="{fake}"')
        runner = root / 'run.sh'
        runner.write_text(script)
        (root / 'latest.pt').touch()
        subprocess.run(['bash', str(runner), str(root)], check=True)
        calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
        assert 'flux_kda.benchmark' in calls[0] and calls[0][-1].endswith('step-300')
        training = [int(args[args.index('--steps') + 1]) for args in calls if 'flux_kda.train' in args]
        assert training == [600, 900, 1000], training
        assert watch.check(root) == 'completed'
        flux3_launcher = watch.REPO / 'flux_kda/train_flux3_node01.sh'
        with patch.object(watch, 'active_workers', return_value=False), patch.object(watch.subprocess, 'run') as gpu:
            gpu.return_value = SimpleNamespace(stdout='1, 2000\n2, 46000\n3, 46000\n')
            assert watch.check(root, flux3_launcher) == 'waiting-for-gpus'
        fake.write_text(fake.read_text().replace('print(300)', 'print(1000)'))
        (root / 'calls').unlink()
        script = flux3_launcher.read_text().replace('cd "$(dirname "$0")/.."', f'cd "{watch.REPO}"')
        script = script.replace('PYTHON="${PYTHON:-python}"', f'PYTHON="{fake}"')
        runner.write_text(script)
        subprocess.run(['bash', str(runner), str(root)], check=True)
        calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
        training = [int(args[args.index('--steps') + 1]) for args in calls if 'flux_kda.train' in args]
        validation = [int(args[args.index('--output') + 1].rsplit('-', 1)[1]) for args in calls if 'flux_kda.benchmark' in args]
        assert training == list(range(1200, 10001, 200)), training
        assert validation == training, validation
        assert watch.check(root, flux3_launcher) == 'completed'
    print('Cron decisions and checkpoint-before-validation recovery passed')


if __name__ == '__main__':
    main()
