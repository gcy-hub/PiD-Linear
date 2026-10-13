"""Nested subsets referencing the existing mmap manifest; no image copies."""

import fcntl
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pid._src.linear_pid.data import BUCKETS, RawImageDataset, atomic_json


def prepare_subsets(index_root, output_root, small=20000, large=50000, seed=42, workers=4):
    source = Path(index_root).resolve()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source_meta = json.loads((source / "index.json").read_text())
    settings = dict(source_fingerprint=source_meta["fingerprint"], small=small, large=large, seed=seed)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings_path = root / "settings.json"
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError("Partial subset preparation settings changed; choose another directory")
        atomic_json(settings_path, settings)
        marker = root / "subsets.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["settings"] != settings:
                raise ValueError("Subset settings changed; choose another output directory")
            if all(
                (root / str(n) / file).exists() for n in (small, large) for file in ("index.json", "sample_ids.jsonl")
            ):
                return old
        buckets = [np.load(source / f"train_{b.replace(',', '_')}.npy", mmap_mode="r") for b in BUCKETS]
        counts = np.array([len(b) for b in buckets])
        if not 0 < small <= large <= counts.sum():
            raise ValueError("Subset sizes exceed training data")
        if len(np.intersect1d(np.concatenate(buckets), np.load(source / "validation.npy"))):
            raise ValueError("Source training/validation rows overlap")

        def quotas(n):
            raw = counts * n / counts.sum()
            q = np.floor(raw).astype(int)
            for i in np.argsort(-(raw - q))[: n - q.sum()]:
                q[i] += 1
            return q

        qsmall, qlarge = quotas(small), quotas(large)
        chosen = {}
        for bucket_index, (bucket, rows, amount) in enumerate(zip(BUCKETS, buckets, qlarge)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, bucket_index]))
            shard = root / f"choice_{bucket.replace(',', '_')}.npy"
            if shard.exists():
                selected = np.load(shard)
            else:
                selected = rng.permutation(rows)[:amount]
                with shard.with_suffix(".tmp").open("wb") as f:
                    np.save(f, selected)
                os.replace(shard.with_suffix(".tmp"), shard)
            if shard.exists() and len(selected) != amount:
                raise ValueError("Corrupt subset choice")
            chosen[bucket] = selected
        # Seed each bucket independently so resumed preparation never changes choices.
        # Persisted choices are the authoritative selection once created.
        fingerprints = {}
        for size, q in ((small, qsmall), (large, qlarge)):
            destination = root / str(size)
            destination.mkdir(exist_ok=True)
            selected = []
            for bucket, amount in zip(BUCKETS, q):
                rows = chosen[bucket][:amount]
                selected.extend(rows.tolist())
                tmp = destination / f"train_{bucket.replace(',', '_')}.tmp"
                with tmp.open("wb") as f:
                    np.save(f, rows)
                os.replace(tmp, tmp.with_suffix(".npy"))
            for name in ("manifest.jsonl", "offsets.npy", "validation.npy"):
                target = destination / name
                if not target.exists():
                    target.symlink_to(source / name)
            fingerprint = hashlib.sha256(
                (source_meta["fingerprint"] + json.dumps(settings) + json.dumps(sorted(selected))).encode()
            ).hexdigest()
            fingerprints[str(size)] = fingerprint
            atomic_json(
                destination / "index.json",
                {
                    **source_meta,
                    "fingerprint": fingerprint,
                    "source_fingerprint": source_meta["fingerprint"],
                    "train_size": size,
                    "subset": True,
                },
            )
        local = threading.local()

        def describe(row):
            if not hasattr(local, "reader"):
                local.reader = RawImageDataset(source)
            record = local.reader.record(int(row))
            caption = record["caption"].lower()
            tags = {
                key: int(bool(re.search(pattern, caption)))
                for key, pattern in {
                    "person": r"\b(person|people|man|woman|girl|boy|face|portrait)\b",
                    "text": r"\b(text|letter|word|sign|logo|writing|poster)\b",
                    "texture": r"\b(texture|fabric|fur|hair|wood|brick|pattern)\b",
                }.items()
            }
            return int(row), record["id"], tags

        counts_labels = {k: 0 for k in ("person", "text", "texture")}
        counts_small = counts_labels.copy()
        small_rows = set(np.concatenate([chosen[b][:n] for b, n in zip(BUCKETS, qsmall)]).tolist())
        ids = {small: [], large: []}
        if workers < 1:
            raise ValueError("workers must be positive")
        with ThreadPoolExecutor(workers) as pool:
            for row, sample_id, tags in tqdm(
                pool.map(describe, np.concatenate(list(chosen.values()))), total=large, desc="Subset caption coverage"
            ):
                for k, v in tags.items():
                    counts_labels[k] += v
                    if row in small_rows:
                        counts_small[k] += v
                entry = {"row": row, "id": sample_id}
                ids[large].append(entry)
                if row in small_rows:
                    ids[small].append(entry)
        for size, records in ids.items():
            tmp = root / str(size) / "sample_ids.jsonl.incomplete"
            with tmp.open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, tmp.with_suffix(""))
        result = {
            "settings": settings,
            "fingerprints": fingerprints,
            "caption_keyword_counts_50k": counts_labels,
            "caption_keyword_counts_20k": counts_small,
            "bucket_counts_50k": dict(zip(BUCKETS, qlarge.tolist())),
        }
        atomic_json(marker, result)
        return result
