"""Research-only spatial mismatch probes; never imported by training."""

import math
import os
from pathlib import Path

import numpy as np
import torch

from pid._src.linear_pid.data import atomic_json, sha256_file


def token_errors(student, reference, epsilon=1e-6):
    squared = (student.float() - reference.float()).square().sum(-1)
    return squared, squared / (reference.float().square().sum(-1) + epsilon)


def tile_pool(values, tile):
    """Average spatial tiles, retaining partial tiles without zero-padding bias."""
    values = np.asarray(values)
    if tile < 1 or values.ndim < 2:
        raise ValueError('Expected spatial array and positive tile size')
    h, w = values.shape[:2]
    pooled = np.empty((math.ceil(h / tile), math.ceil(w / tile), *values.shape[2:]), dtype=np.float64)
    for row in range(pooled.shape[0]):
        for col in range(pooled.shape[1]):
            pooled[row, col] = values[row * tile:(row + 1) * tile,
                                      col * tile:(col + 1) * tile].mean(axis=(0, 1))
    return pooled


def concentration(error, fraction=.2):
    values = np.asarray(error, dtype=np.float64).ravel()
    if not 0 < fraction <= 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative errors and selection fraction in (0,1]')
    count = math.ceil(values.size * fraction)
    ordered, total = np.sort(values), values.sum()
    mass = ordered[-count:].sum() / total if total else 0.
    gini = ((2 * np.arange(1, values.size + 1) - values.size - 1) @ ordered
            / (values.size * total)) if total else 0.
    return dict(top_mass=float(mass), uniform_expected_mass=count / values.size,
                gini=float(gini), zero_error=not bool(total), selected_count=count)


def rank_correlation(a, b):
    def ranks(values):
        _, inverse, counts = np.unique(np.asarray(values).ravel(), return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        return (starts + (counts - 1) / 2.)[inverse]
    ra, rb = ranks(a), ranks(b)
    if len(ra) != len(rb):
        raise ValueError('Correlation maps must have matching sizes')
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.
    return float(np.corrcoef(ra, rb)[0, 1])


def split_prompt_cases(cases, train_prompts):
    ids = sorted({case['prompt_id'] for case in cases})
    if not 0 < train_prompts < len(ids):
        raise ValueError('Need nonempty train and held-out prompt sets')
    training = set(ids[:train_prompts])
    return ([c for c in cases if c['prompt_id'] in training],
            [c for c in cases if c['prompt_id'] not in training])


def ridge_scores(train_x, train_y, test_x, penalty=1.):
    x, test = np.asarray(train_x, dtype=np.float64), np.asarray(test_x, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    x = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    test = np.column_stack([(test - mean) / scale, np.ones(len(test))])
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[-1, -1] = 0.
    return test @ np.linalg.solve(x.T @ x + regularizer, x.T @ train_y)


def selected_mass(error, scores, fraction=.2):
    error, scores = np.asarray(error).ravel(), np.asarray(scores).ravel()
    count = math.ceil(len(error) * fraction)
    # Stable tie order: identical scores cannot silently become an oracle.
    selected = np.argsort(scores, kind='stable')[-count:]
    return float(error[selected].sum() / error.sum()) if error.sum() else 0.


def publish_case(directory, settings, maps, metrics=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'maps.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **maps)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / 'maps.npz')
    atomic_json(directory / 'complete.json', dict(settings=settings, metrics=metrics or {},
                bytes=(directory / 'maps.npz').stat().st_size,
                maps_sha256=sha256_file(directory / 'maps.npz')))


def completed_case(directory, settings):
    import json
    directory = Path(directory)
    if not (directory / 'complete.json').exists():
        return False
    record = json.loads((directory / 'complete.json').read_text())
    if record['settings'] != settings:
        raise ValueError(f'Existing case settings differ: {directory}; select a new output directory')
    path = directory / 'maps.npz'
    return (path.exists() and path.stat().st_size == record['bytes']
            and sha256_file(path) == record['maps_sha256'])
