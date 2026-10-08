"""Resumable validation of raw MultiAspect data and stateless global batch sampling."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO

BUCKETS = list(IMAGE_RES_SIZE_INFO["2048"])


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aspect_bucket(width, height):
    return min(
        BUCKETS,
        key=lambda key: abs(math.log(width / height) - math.log(float(key.split(",")[0]) / float(key.split(",")[1]))),
    )


def _source_signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@lru_cache(maxsize=4096)
def _canonical_parent(path):
    return Path(path).resolve()


def _image_path(source, relative):
    path = source.parent / relative
    path = _canonical_parent(str(path.parent)) / path.name
    return path.resolve() if path.is_symlink() else path


def _header_orientation(image):
    # Pillow's PNG getexif() fully decodes images when EXIF wasn't encountered
    # before IDAT. Seek over compressed chunks to find optional trailing EXIF.
    if image.format == "PNG" and "exif" not in image.info:
        stream = image.fp
        position = stream.tell()
        try:
            stream.seek(8)
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    break
                size, kind = struct.unpack(">I4s", header)
                if kind == b"eXIf":
                    image.info["exif"] = b"Exif\x00\x00" + stream.read(size)
                    break
                if kind == b"IEND":
                    break
                stream.seek(size + 4, os.SEEK_CUR)
        finally:
            stream.seek(position)
    return Image.Image.getexif(image).get(274)


def _prepare_shard(args):
    source, output, resume = args[:3]
    image_verification = args[3] if len(args) > 3 else "full"
    source, output = Path(source), Path(output)
    destination = output / f"{source.stem}.jsonl"
    done = destination.with_suffix(".done.json")
    signature = _source_signature(source)
    if resume and done.exists() and destination.exists():
        result = json.loads(done.read_text())
        verification_matches = image_verification == "header" or result.get("image_verification", "full") == "full"
        if verification_matches and result["source"] == signature and result["bytes"] == destination.stat().st_size:
            return result
    records = json.loads(source.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {source}")
    errors_path = output / f"{source.stem}.rejected.jsonl"
    temp, errors_temp = destination.with_suffix(".tmp"), errors_path.with_suffix(".tmp")
    counts = Counter()
    with temp.open("w") as good, errors_temp.open("w") as bad:
        for index, record in enumerate(records):
            uid = f"{source.stem}/{index:06d}"
            try:
                if not isinstance(record, dict):
                    raise ValueError("invalid_record")
                caption = record.get("en_caption", "")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError("empty_english_caption")
                image_path = _image_path(source, record["image_path"])
                with Image.open(image_path) as image:
                    if image_verification == "full":
                        image.load()
                    width, height = image.size
                    orientation = image.getexif().get(274) if image_verification == "full" else _header_orientation(image)
                    if orientation in {5, 6, 7, 8}:
                        width, height = height, width
                    if image_verification == "full":
                        if image.mode != "RGB":
                            image.convert("RGB").load()
                bucket = aspect_bucket(width, height)
                tw, th = IMAGE_RES_SIZE_INFO["2048"][bucket]
                if min(width / tw, height / th) <= 0.9:
                    raise ValueError("too_small_for_2k_bucket")
                good.write(
                    json.dumps(
                        {
                            "id": uid,
                            "path": str(image_path),
                            "caption": caption.strip(),
                            "width": width,
                            "height": height,
                            "bucket": bucket,
                            "group": str(record.get("image_url") or image_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts["accepted"] += 1
            except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                counts[reason] += 1
                bad.write(json.dumps({"id": uid, "reason": reason, "detail": str(exc)}) + "\n")
        for stream in (good, bad):
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(temp, destination)
    os.replace(errors_temp, errors_path)
    result = {
        "source": signature,
        "bytes": destination.stat().st_size,
        "counts": dict(counts),
        "image_verification": image_verification,
    }
    atomic_json(done, result)
    return result


def _prepare_index(
    dataset_root, output_root, workers=16, validation_size=1024, seed=42, resume=False, max_json_files=None,
    image_verification="full",
):
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    sources = sorted((root / "data_jsons").glob("*.json"))
    if max_json_files:
        sources = sources[:max_json_files]
    if not sources:
        raise FileNotFoundError(f"No metadata JSONs in {root / 'data_jsons'}")
    build = {
        "root": str(root),
        "seed": seed,
        "validation_size": validation_size,
        "sources": {str(p): _source_signature(p) for p in sources},
        "schema": 1,
    }
    if image_verification != "full":
        build["image_verification"] = image_verification
    signature = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    if resume and (output / "index.json").exists():
        previous = json.loads((output / "index.json").read_text())
        if previous["build_signature"] == signature and all((output / p).exists() for p in previous["files"]):
            print(
                f"Complete index already exists: {previous['train_size']} train / {previous['validation_size']} validation"
            )
            return previous
    totals = Counter()
    fully_decoded_size = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_prepare_shard, (str(p), str(shard_dir), resume, image_verification)) for p in sources]
        progress = tqdm(as_completed(futures), total=len(futures), desc=f"Metadata shards ({image_verification})")
        for future in progress:
            result = future.result()
            totals.update(result["counts"])
            if result.get("image_verification", "full") == "full":
                fully_decoded_size += result["counts"].get("accepted", 0)
            progress.set_postfix(accepted=totals["accepted"], rejected=sum(totals.values()) - totals["accepted"])
    # Split duplicate URL groups together; merge needs only group hashes, not image/caption payloads.
    groups = {}
    for source in sources:
        with (shard_dir / f"{source.stem}.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                group = hashlib.sha256(record["group"].encode()).hexdigest()
                if group not in groups:
                    score = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
                    groups[group] = (record["bucket"], score)
    by_bucket = defaultdict(list)
    for group, (bucket, score) in groups.items():
        by_bucket[bucket].append((score, group))
    selected = set()
    group_count = sum(map(len, by_bucket.values()))
    quotas = {bucket: validation_size * len(items) / max(1, group_count) for bucket, items in by_bucket.items()}
    budgets = {bucket: int(value) for bucket, value in quotas.items()}
    remainder = max(0, validation_size - sum(budgets.values()))
    for bucket in sorted(quotas, key=lambda k: quotas[k] - budgets[k], reverse=True)[:remainder]:
        budgets[bucket] += 1
    for bucket, items in by_bucket.items():
        # Leave at least one group for training in each represented bucket.
        limit = min(budgets[bucket], max(0, len(items) - 1))
        selected.update(group for _, group in sorted(items)[:limit])
    offsets, train, validation = [], defaultdict(list), []
    destination = output / "manifest.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as merged:
        for source in sources:
            with (shard_dir / f"{source.stem}.jsonl").open() as stream:
                for line in stream:
                    record = json.loads(line)
                    group = hashlib.sha256(record["group"].encode()).hexdigest()
                    record["split"] = "validation" if group in selected else "train"
                    row = len(offsets)
                    offsets.append(merged.tell())
                    (validation if record["split"] == "validation" else train[record["bucket"]]).append(row)
                    merged.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
        merged.flush()
        os.fsync(merged.fileno())
    os.replace(temporary, destination)
    files = ["manifest.jsonl", "offsets.npy", "validation.npy"]
    arrays = {"offsets.npy": offsets, "validation.npy": validation}
    for bucket in BUCKETS:
        name = f"train_{bucket.replace(',', '_')}.npy"
        arrays[name] = train[bucket]
        files.append(name)
    for name, values in arrays.items():
        with (output / (name + ".tmp")).open("wb") as stream:
            np.save(stream, np.asarray(values, dtype=np.int64))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(output / (name + ".tmp"), output / name)
    metadata = {
        "schema": 1,
        "build_signature": signature,
        "fingerprint": sha256_file(destination),
        "train_size": sum(map(len, train.values())),
        "validation_size": len(validation),
        "counts": dict(totals),
        "bucket_counts": {k: len(train[k]) for k in BUCKETS},
        "files": files,
        "dataset_root": str(root),
        "seed": seed,
        "image_verification": image_verification,
        "fully_decoded_size": fully_decoded_size,
    }
    if not metadata["train_size"]:
        raise ValueError("No valid training images; inspect shard rejection logs")
    atomic_json(output / "index.json", metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def resize_crop(image, bucket, resolution="2048", allow_upscale=False):
    """Same adaptive cover resize and 0.9 near-miss rescue as PiD ResizeScale."""
    tw, th = IMAGE_RES_SIZE_INFO[str(resolution)][bucket]
    width, height = image.size
    factor = min(width / tw, height / th)
    if factor <= 0.9 and not allow_upscale:
        raise ValueError("Image became too small after indexing")
    if factor > 1 or factor < 1:
        if factor > 1 or allow_upscale or factor > 0.9:
            image = image.resize(
                (max(tw, math.ceil(width / factor)), max(th, math.ceil(height / factor))), Image.Resampling.LANCZOS
            )
    left, top = round((image.width - tw) / 2), round((image.height - th) / 2)
    return image.crop((left, top, left + tw, top + th))


class RawImageDataset(Dataset):
    def __init__(self, index_root, resolution="2048", text_cache_root=None):
        self.root, self.resolution = Path(index_root), str(resolution)
        self.metadata = json.loads((self.root / "index.json").read_text())
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self._stream = None
        self._pid = None
        if text_cache_root:
            from pid._src.linear_pid.text_cache import TextCacheReader

            self.text_cache = TextCacheReader(text_cache_root)
        else:
            self.text_cache = None

    def __len__(self):
        return len(self.offsets)

    def record(self, row):
        if self._pid != os.getpid():
            if self._stream is not None:
                self._stream.close()
            self._stream = (self.root / "manifest.jsonl").open("rb")
            self._pid = os.getpid()
        self._stream.seek(int(self.offsets[row]))
        return json.loads(self._stream.readline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"], state["_pid"] = None, None
        return state

    def __getitem__(self, row):
        record = self.record(row)
        cached = {}
        if self.text_cache:
            embs, mask = self.text_cache.read(record["id"], record["caption"])
            cached = {"text_embs": embs, "text_mask": mask}
        try:
            with Image.open(record["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = resize_crop(image, record["bucket"], self.resolution)
                array = np.array(image, dtype=np.uint8, copy=True)
            # uint8 remains compact in worker/prefetch memory; normalization occurs on GPU.
            return {
                "image": torch.from_numpy(array).permute(2, 0, 1),
                "caption": record["caption"],
                "id": record["id"],
                "row": int(row),
                **cached,
            }
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            return {"error": f"{record['id']}: {exc}", "row": int(row)}


def collate_raw(records):
    errors = [record["error"] for record in records if "error" in record]
    if errors:
        return {"errors": errors}
    batch = {
        "image": torch.stack([r["image"] for r in records]),
        "caption": [r["caption"] for r in records],
        "id": [r["id"] for r in records],
        "row": [r["row"] for r in records],
        "errors": [],
    }
    if "text_embs" in records[0]:
        batch["text_embs"] = torch.stack([r["text_embs"] for r in records])
        batch["text_mask"] = torch.stack([r["text_mask"] for r in records])
    return batch


class GlobalBatchSampler(Sampler):
    """Prefetch cannot advance saved progress. Each global update is independently reproducible.

    Bucket selection follows sample counts; updates sample without replacement internally,
    and with replacement between updates. The curriculum is step-based, not epoch-based.
    """

    def __init__(self, index_root, batch_size, accum, rank=0, world_size=1, seed=42, start=0, effective_batch=0):
        self.buckets = [np.load(Path(index_root) / f"train_{k.replace(',', '_')}.npy", mmap_mode="r") for k in BUCKETS]
        counts = np.asarray([len(b) for b in self.buckets], dtype=float)
        self.probabilities = counts / counts.sum()
        self.batch_size, self.accum, self.rank, self.world_size = batch_size, accum, rank, world_size
        self.seed, self.start = seed, start
        self.effective_batch = effective_batch or batch_size * accum * world_size
        if self.effective_batch % accum or not world_size <= self.effective_batch // accum <= batch_size * world_size:
            raise ValueError("Effective batch must divide accumulation and give every rank 1..batch_size images")
        quotient, remainder = divmod(self.effective_batch // accum, world_size)
        self.rank_batch_sizes = [quotient + int(i < remainder) for i in range(world_size)]

    @property
    def global_batch_size(self):
        return self.effective_batch

    def global_batch(self, cursor):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, cursor]))
        bucket = self.buckets[rng.choice(len(self.buckets), p=self.probabilities)]
        if len(bucket) < self.global_batch_size:
            raise ValueError(
                "Selected bucket has fewer samples than effective batch; lower batch or filter that bucket"
            )
        return bucket[rng.choice(len(bucket), self.global_batch_size, replace=False)]

    def __iter__(self):
        cursor = self.start
        while True:
            rows = self.global_batch(cursor).reshape(self.accum, -1)
            first = sum(self.rank_batch_sizes[: self.rank])
            last = first + self.rank_batch_sizes[self.rank]
            for micro in range(self.accum):
                yield rows[micro, first:last].tolist()
            cursor += 1


def prepare_index(
    dataset_root, output_root, workers=16, validation_size=1024, seed=42, resume=False, max_json_files=None,
    image_verification="full",
):
    if workers < 1 or validation_size < 0:
        raise ValueError("workers must be positive and validation-size nonnegative")
    if image_verification not in {"full", "header"}:
        raise ValueError("image_verification must be full or header")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another index preparation is running in {output}") from exc
        return _prepare_index(
            dataset_root, output_root, workers, validation_size, seed, resume, max_json_files, image_verification
        )
