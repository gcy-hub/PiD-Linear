"""Generate controller argv bindings from local native model/data resource configs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from accept_attention_framework import SCENARIOS, validate_bindings


def generate(resources):
    spec = json.loads(resources.read_text())
    if set(spec['models']) != {'pid', 'flux2', 'flux3'}:
        raise ValueError('models must supply pid, flux2 and flux3 local YAML paths')
    for path in [*spec['models'].values(), spec['data']]:
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise ValueError(f'Config must be an existing absolute path: {path}')
    bindings = {}
    for name in SCENARIOS:
        if name in ('config_data', 'pid', 'flux2', 'flux3') or name.endswith('_cycle'):
            argv = [sys.executable, str(ROOT / 'tests/attention_framework/check_integrated.py'),
                    '--resources', str(resources.resolve()), '--scenario', name]
        elif name == 'common':
            argv = [sys.executable, str(ROOT / 'tests/attention_framework/check_common.py')]
        elif name == 'distributed':
            argv = [str(ROOT / 'tests/attention_framework/check_distributed.py')]
        elif name == 'convergence':
            argv = [sys.executable, str(ROOT / 'tests/attention_framework/check_convergence.py')]
        else:
            # Scheduler state is owned by the existing workflow, never recreated here.
            argv = spec.get('handoff_argv', [sys.executable,
                str(ROOT / 'tests/attention_framework/check_handoff.py'), '--resources', str(resources.resolve())])
        bindings[name] = dict(argv=argv, config=dict(resource_config=str(resources.resolve()),
                             overrides=spec.get('overrides', [])), resources=spec)
    validate_bindings(bindings)
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bindings = generate(args.resources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bindings, indent=2) + '\n')


if __name__ == '__main__':
    main()
