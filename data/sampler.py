"""Deterministic bucket draws, resumed by consumed local microbatch cursor."""
import numpy as np
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    def __init__(self, buckets, batch_size, rank, world_size, accumulation=1, seed=42, cursor=0):
        self.groups = [v for v in buckets.values() if v]
        if not self.groups or min(batch_size, world_size, accumulation) < 1 or not 0 <= rank < world_size or cursor < 0:
            raise ValueError('Invalid sampler sizes, rank or cursor')
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.accumulation, self.seed, self.cursor = accumulation, seed, cursor

    def __iter__(self):
        cursor = self.cursor
        counts = np.array([len(v) for v in self.groups], dtype=np.float64)
        effective = self.batch_size * self.world_size * self.accumulation
        while True:
            step, micro = divmod(cursor, self.accumulation)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            group = self.groups[rng.choice(len(self.groups), p=counts / counts.sum())]
            draw = rng.choice(group, size=effective, replace=len(group) < effective)
            start = (micro * self.world_size + self.rank) * self.batch_size
            yield draw[start:start + self.batch_size].tolist()
            cursor += 1
