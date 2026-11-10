"""Reuse bounded handoff tests and verify exported durable execution records."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def models(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'model' and isinstance(item, str):
                yield item
            yield from models(item)
    elif isinstance(value, list):
        for item in value:
            yield from models(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.resources.read_text())
    subprocess.run([sys.executable, str(ROOT / 'plans/attention-framework/check_handoff.py')], check=True)
    subprocess.run(['node', str(ROOT / 'plans/attention-framework/checks/workflow_contract.cjs')], cwd=ROOT, check=True)
    records = spec.get('durable_records', [])
    if not records:
        print('Blocked: export actual durable run records to durable_records; workflow source is not execution evidence')
        raise SystemExit(77)
    if any(not Path(path).is_file() for path in records):
        print('Blocked: durable run-record file missing')
        raise SystemExit(77)
    actual = {m for path in records for m in models(json.loads(Path(path).read_text()))}
    required = {'zzz/gpt-6-astra:high', 'zzz/gpt-6.1-sol:high'}
    assert required <= actual, f'Missing specified models in durable records: {required - actual}'
    assert all(m.startswith('zzz/') for m in actual), f'Unexpected provider in records: {actual}'
    print('Bounded handoff and actual durable model records passed; remote publication remains a final integration action')


if __name__ == '__main__':
    main()
