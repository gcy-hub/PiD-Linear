"""Official PiD buckets, with explicit small buckets for smoke runs."""
import math
from pid._src.datasets.buckets import MULTI_RES_GRID_BY_MAX_RESOLUTION, _classify_aspect_ratio


def choose_bucket(width, height, buckets='official'):
    if buckets == 'official':
        entries = MULTI_RES_GRID_BY_MAX_RESOLUTION['4096'][_classify_aspect_ratio(width, height)]
        return next(((w, h) for _, w, h in entries if width >= w and height >= h), None)
    choices = [tuple(map(int, size)) for size in buckets]
    if not choices or any(min(size) <= 0 or any(v % 16 for v in size) for size in choices):
        raise ValueError('Buckets must be positive [width, height] multiples of 16')
    fitting = [(w, h) for w, h in choices if width >= w and height >= h]
    return min(fitting, key=lambda s: abs(math.log((width / height) / (s[0] / s[1])))) if fitting else None
