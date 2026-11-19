"""Freeze the DPG-Bench subset; the CSV repeats each prompt once per proposition."""
import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'data/dpg1000.jsonl'
MODELS = ('pid', 'pid-dit')


def extract(path):
    items = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key, text = row['item_id'], row['text'].strip()
            if not text or items.setdefault(key, text) != text:
                raise ValueError(f'Inconsistent prompt: {key}')
    # Use one source ID per distinct prompt.
    by_text = {items[k]: k for k in sorted(items)}
    keys = random.Random(1045).sample(sorted(by_text.values()), 1000)
    return [dict(id=i, source_id=k, prompt=items[k], seed=1045+i) for i,k in enumerate(keys)]


def manifest():
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    if [r['id'] for r in rows] != list(range(1000)) or len({r['prompt'] for r in rows}) != 1000:
        raise ValueError('Expected 1000 unique ordered DPG prompts')
    return rows


def pairs(rank, world=8):
    if not 0 <= rank < world:
        raise ValueError('Invalid worker rank')
    return [(2*i, 2*i+1) for i in range(rank, 500, world)]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    args = p.parse_args()
    rows = extract(args.csv)
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
