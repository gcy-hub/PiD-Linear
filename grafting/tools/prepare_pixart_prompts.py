#!/usr/bin/env python3
"""Build a reproducible COCO caption pool and synthetic-training prompt subset."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--pool-size', type=int, default=30000)
    p.add_argument('--num-samples', type=int, default=12000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source-url', default='https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/coco/coco2014/annotations/captions_val2014.json')
    args = p.parse_args()
    if not 0 < args.num_samples <= args.pool_size:
        p.error('Require 0 < num-samples <= pool-size')
    raw = args.annotations.read_bytes()
    data = json.loads(raw)
    by_image = {}
    for row in data['annotations']:
        caption = ' '.join(row['caption'].split())
        if caption:
            by_image.setdefault(row['image_id'], []).append((row['id'], caption))
    rng = random.Random(args.seed)
    images = sorted(by_image)
    rng.shuffle(images)
    pool, seen = [], set()
    for image_id in images:
        captions = sorted(by_image[image_id])
        rng.shuffle(captions)
        for annotation_id, caption in captions:
            normalized = caption.casefold()
            if normalized not in seen:
                seen.add(normalized)
                pool.append({'id': f'coco2014-val-{image_id}-caption-{annotation_id}',
                             'image_id': image_id, 'annotation_id': annotation_id, 'prompt': caption})
                break
        if len(pool) == args.pool_size:
            break
    if len(pool) < args.pool_size:
        raise ValueError('Not enough distinct images/captions for the requested pool')
    selected = rng.sample(pool, args.num_samples)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in [('prompt_pool_30k.jsonl', pool), ('prompts_12k.jsonl', selected)]:
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        path = args.output_root / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f'Refusing to replace a different prompt selection: {path}')
        path.write_text(content)
    (args.output_root / 'prompt_provenance.json').write_text(json.dumps({
        'source_url': args.source_url, 'annotations_sha256': hashlib.sha256(raw).hexdigest(),
        'source_images': len(data['images']), 'source_captions': len(data['annotations']),
        'pool_size': len(pool), 'selected': len(selected), 'seed': args.seed,
        'selection': 'one unique caption per image; seeded 30K pool then seeded 12K sample',
        'paper_relation': 'Same synthetic-data construction and 8K/12K scale; author prompt IDs were not released. COCO source is our reproduction choice.'
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(selected)} prompts from a {len(pool)}-prompt pool.')


if __name__ == '__main__':
    main()
