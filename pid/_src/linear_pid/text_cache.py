"""Durable BF16 caption shards and bounded, per-worker mmap readers.

The cache is independent of image validation: ids are metadata stem/record index,
exactly as in the image manifest. Full 300-position embeddings preserve the
released Full Attention behavior, including Gemma's contextual padding outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.text_encoder import PROMPT_PREFIX, TEXT_DIM, TEXT_LENGTH

CACHE_SCHEMA = 1
ARRAY_NAMES = ("embeddings.npy", "masks.npy", "rows.npy", "caption_hashes.npy")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def caption_digest(caption):
    return np.frombuffer(hashlib.sha256(caption.encode()).digest(), dtype=np.uint8)


def source_inventory(dataset_root):
    sources = sorted((Path(dataset_root) / "data_jsons").glob("*.json"))
    if not sources:
        raise FileNotFoundError(f"No metadata JSONs in {Path(dataset_root) / 'data_jsons'}")
    return {p.stem: {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources}


def encoding_spec(weights_root):
    """Fingerprint the actual local HF checkpoint and all tokenizer inputs."""
    import transformers

    root = Path(weights_root) / "gemma-2-2b-it"
    index = root / "model.safetensors.index.json"
    if index.exists():
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (root / "model.safetensors").exists():
        weights = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No Hugging Face safetensors checkpoint in {root}")
    names = weights + [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    files = {name: sha256_file(root / name) for name in names if (root / name).exists()}
    return {
        "encoder": "gemma-2-2b-it",
        "files_sha256": files,
        "prefix": PROMPT_PREFIX,
        "padding_side": "right",
        "max_length_extra": 298,
        "selected_positions": "[0] + last 299",
        "use_cache": False,
        "shape": [TEXT_LENGTH, TEXT_DIM],
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def read_prompt_source(path):
    path = Path(path)
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")
    rows, captions = [], []
    for row, record in enumerate(records):
        caption = record.get("en_caption", "") if isinstance(record, dict) else ""
        if isinstance(caption, str) and caption.strip():
            rows.append(row)
            captions.append(caption.strip())
    return {
        "stem": path.stem,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(records),
        "rows": rows,
        "captions": captions,
    }


def prefetch_sources(paths, workers):
    """Bound metadata RAM to workers+1 files rather than a million captions."""
    if workers == 0:
        for path in paths:
            yield read_prompt_source(path)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending, iterator = [], iter(paths)
        for _ in range(workers + 1):
            path = next(iterator, None)
            if path is not None:
                pending.append(pool.submit(read_prompt_source, path))
        while pending:
            result = pending.pop(0).result()
            path = next(iterator, None)
            if path is not None:
                pending.append(pool.submit(read_prompt_source, path))
            yield result


def _durable_flush(path, arrays):
    for name, array in arrays.items():
        array.flush()
        with (path / name).open("rb") as stream:
            os.fsync(stream.fileno())


def completed_shard(path, fingerprint=None):
    path = Path(path)
    try:
        metadata = json.loads((path / "complete.json").read_text())
        if fingerprint is not None and metadata["fingerprint"] != fingerprint:
            raise ValueError(f"Text cache inputs changed in {path}")
        if any((path / name).stat().st_size != size for name, size in metadata["file_bytes"].items()):
            raise ValueError(f"Text cache shard is truncated in {path}")
        return metadata
    except FileNotFoundError:
        return None


class CaptionShardWriter:
    def __init__(self, root, source, spec_fingerprint, shape=(TEXT_LENGTH, TEXT_DIM)):
        self.path = Path(root) / "shards" / source["stem"]
        self.path.mkdir(parents=True, exist_ok=True)
        count = len(source["captions"])
        self.metadata = {
            "schema": CACHE_SCHEMA,
            "stem": source["stem"],
            "source_sha256": source["source_sha256"],
            "record_count": source["record_count"],
            "count": count,
            "spec_fingerprint": spec_fingerprint,
            "shape": list(shape),
        }
        self.fingerprint = digest_json(self.metadata)
        self.complete = completed_shard(self.path, self.fingerprint)
        self.arrays = {}
        self.cursor = count if self.complete else 0
        if self.complete:
            return
        progress_path = self.path / "progress.json"
        resume = progress_path.exists()
        if resume:
            progress = json.loads(progress_path.read_text())
            if progress["fingerprint"] != self.fingerprint:
                raise ValueError(f"Text cache inputs changed in {self.path}")
            self.cursor = progress["cursor"]
            if not 0 <= self.cursor <= count:
                raise ValueError(f"Invalid text cache cursor in {self.path}")
        definitions = {
            "embeddings.npy": (np.uint16, (count, *shape)),
            "masks.npy": (np.bool_, (count, shape[0])),
            "rows.npy": (np.int32, (source["record_count"],)),
            "caption_hashes.npy": (np.uint8, (count, 32)),
        }
        for name, (dtype, array_shape) in definitions.items():
            if resume:
                array = np.load(self.path / name, mmap_mode="r+")
                if array.shape != array_shape or array.dtype != dtype:
                    raise ValueError(f"Invalid cached array {self.path / name}")
            else:
                array = np.lib.format.open_memmap(self.path / name, mode="w+", dtype=dtype, shape=array_shape)
            self.arrays[name] = array
        if not resume:
            self.arrays["rows.npy"][:] = -1
            self.arrays["rows.npy"][source["rows"]] = np.arange(count, dtype=np.int32)
            for i, caption in enumerate(source["captions"]):
                self.arrays["caption_hashes.npy"][i] = caption_digest(caption)
            self.commit()

    def write(self, embeddings, masks):
        embeddings = embeddings.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        masks = masks.detach().to(device="cpu", dtype=torch.bool).contiguous()
        count = len(embeddings)
        end = self.cursor + count
        if list(embeddings.shape[1:]) != self.metadata["shape"] or masks.shape != embeddings.shape[:2]:
            raise ValueError("Unexpected Gemma embedding/mask shape")
        if end > self.metadata["count"] or not torch.isfinite(embeddings).all():
            raise ValueError("Invalid/non-finite caption cache output")
        self.arrays["embeddings.npy"][self.cursor : end] = embeddings.view(torch.uint16).numpy()
        self.arrays["masks.npy"][self.cursor : end] = masks.numpy()
        self.cursor = end

    def commit(self):
        _durable_flush(self.path, self.arrays)
        atomic_json(self.path / "progress.json", {"fingerprint": self.fingerprint, "cursor": self.cursor})

    def finish(self):
        if self.complete:
            return self.complete
        if self.cursor != self.metadata["count"]:
            raise ValueError("Cannot publish an incomplete caption shard")
        self.commit()
        result = {
            **self.metadata,
            "fingerprint": self.fingerprint,
            "file_bytes": {name: (self.path / name).stat().st_size for name in ARRAY_NAMES},
        }
        atomic_json(self.path / "complete.json", result)
        self.complete = result
        return result


def publish_cache(root):
    root = Path(root)
    build = json.loads((root / "build.json").read_text())
    shards = {}
    for stem in build["sources"]:
        metadata = completed_shard(root / "shards" / stem)
        if metadata is None:
            raise ValueError(f"Unfinished text cache shard: {stem}")
        if metadata["spec_fingerprint"] != build["spec_fingerprint"]:
            raise ValueError(f"Wrong Gemma encoding version in shard {stem}")
        shards[stem] = metadata
    special_hash = sha256_file(root / "special.pt")
    metadata = {
        "schema": CACHE_SCHEMA,
        "build_fingerprint": build["fingerprint"],
        "spec": build["spec"],
        "dataset_root": build["dataset_root"],
        "shards": shards,
        "count": sum(s["count"] for s in shards.values()),
        "special_sha256": special_hash,
    }
    metadata["fingerprint"] = digest_json(metadata)
    atomic_json(root / "cache.json", metadata)
    return metadata


class TextCacheReader:
    def __init__(self, root, max_open_shards=4):
        self.root = Path(root)
        path = self.root / "cache.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Complete Gemma cache missing: {path}; run prepare_linear_pid_text_cache.sh --resume"
            )
        self.metadata = json.loads(path.read_text())
        if self.metadata["schema"] != CACHE_SCHEMA:
            raise ValueError("Unsupported caption cache schema")
        self.fingerprint = self.metadata["fingerprint"]
        content = {key: value for key, value in self.metadata.items() if key != "fingerprint"}
        if digest_json(content) != self.fingerprint:
            raise ValueError("Text cache manifest fingerprint mismatch")
        if sha256_file(self.root / "special.pt") != self.metadata["special_sha256"]:
            raise ValueError("Corrupt empty/CFG text cache")
        self.special = torch.load(self.root / "special.pt", map_location="cpu", weights_only=True)
        self.max_open_shards = max_open_shards
        self._pid, self._arrays = None, OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pid"], state["_arrays"] = None, OrderedDict()
        return state

    def read(self, uid, caption):
        if self._pid != os.getpid():
            self._arrays = OrderedDict()
            self._pid = os.getpid()
        stem, row = uid.rsplit("/", 1)
        if stem not in self._arrays:
            if stem not in self.metadata["shards"]:
                raise KeyError(f"Missing text cache for {uid}")
            path = self.root / "shards" / stem
            metadata = self.metadata["shards"][stem]
            if completed_shard(path, metadata["fingerprint"]) is None:
                raise ValueError(f"Missing completed text cache shard {stem}")
            self._arrays[stem] = {name: np.load(path / name, mmap_mode="r") for name in ARRAY_NAMES}
            if len(self._arrays) > self.max_open_shards:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(stem)
        arrays = self._arrays[stem]
        row = int(row)
        if not 0 <= row < len(arrays["rows.npy"]):
            raise KeyError(f"Missing text cache record {uid}")
        offset = int(arrays["rows.npy"][row])
        if offset < 0 or not np.array_equal(arrays["caption_hashes.npy"][offset], caption_digest(caption)):
            raise ValueError(f"Text cache caption mismatch for {uid}")
        embs = torch.from_numpy(np.array(arrays["embeddings.npy"][offset], copy=True)).view(torch.bfloat16)
        mask = torch.from_numpy(np.array(arrays["masks.npy"][offset], copy=True))
        return embs, mask
