import json
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from pid._src.common.data import (
    BUCKETS, GlobalBatchSampler, RawImageDataset, _header_orientation, _image_path,
    _prepare_shard, collate_raw, prepare_index,
)


def test_header_png_with_trailing_exif_never_decodes_pixels(tmp_path, monkeypatch):
    path = tmp_path / "orientation.png"
    Image.new("RGB", (32, 48)).save(path)
    exif = Image.Exif()
    exif[274] = 6
    payload = exif.tobytes()[6:]
    kind = b"eXIf"
    chunk = struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    original = path.read_bytes()
    path.write_bytes(original[:-12] + chunk + original[-12:])
    with Image.open(path) as image:
        def forbidden(*args, **kwargs):
            raise AssertionError("Header indexing must not decode PNG pixels")
        monkeypatch.setattr(image, "load", forbidden)
        before = image.fp.tell()
        assert _header_orientation(image) == 6
        assert image.fp.tell() == before


def test_cached_parent_resolution_keeps_image_symlinks_canonical(tmp_path):
    folder = tmp_path / "actual"
    folder.mkdir()
    image = folder / "image.jpg"
    image.write_bytes(b"data")
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    (folder / "link.jpg").symlink_to(image)
    source = tmp_path / "metadata" / "0001.json"
    assert _image_path(source, "../alias/image.jpg") == image.resolve()
    assert _image_path(source, "../alias/link.jpg") == image.resolve()


def test_header_index_orientation_and_full_verification_resume(tmp_path):
    root, output = tmp_path / "dataset", tmp_path / "shards"
    root.mkdir()
    output.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    image_path = root / "good.jpg"
    Image.new("RGB", (2048, 3072), (10, 20, 30)).save(image_path, exif=exif)
    broken = root / "truncated.jpg"
    payload = image_path.read_bytes()
    broken.write_bytes(payload[:len(payload) // 2])
    source = root / "0001.json"
    source.write_text(json.dumps([
        {"image_path": path.name, "en_caption": "An oriented photograph"}
        for path in (image_path, broken)
    ]))
    header = _prepare_shard((str(source), str(output), True, "header"))
    assert header["counts"]["accepted"] == 2
    manifest = output / "0001.jsonl"
    record = json.loads(manifest.read_text().splitlines()[0])
    assert (record["width"], record["height"]) == (3072, 2048)
    # A header-only shard must never satisfy a subsequent full-decode request.
    full = _prepare_shard((str(source), str(output), True, "full"))
    assert full["image_verification"] == "full"
    assert full["counts"]["accepted"] == 1
    before = manifest.stat().st_mtime_ns
    # Full verification is sufficient when resuming a header-mode build.
    assert _prepare_shard((str(source), str(output), True, "header")) == full
    assert manifest.stat().st_mtime_ns == before


def test_raw_index_resume_split_and_decode_failure(tmp_path):
    root, output = tmp_path / "dataset", tmp_path / "index"
    (root / "data_jsons").mkdir(parents=True)
    (root / "datas").mkdir()
    Image.new("RGB", (2048, 2048), (10, 20, 30)).save(root / "datas" / "good.jpg")
    (root / "datas" / "broken.jpg").write_bytes(b"broken jpeg")
    for shard in ["0001", "0002"]:
        records = [
            {
                "id": i,
                "image_path": "../datas/good.jpg",
                "en_caption": f"Caption {i}",
                "image_url": f"https://example.test/{i}",
            }
            for i in range(12)
        ]
        records += [
            {"id": 12, "image_path": "../datas/broken.jpg", "en_caption": "broken"},
            {"id": 13, "image_path": "../datas/good.jpg", "en_caption": ""},
        ]
        (root / "data_jsons" / f"{shard}.json").write_text(json.dumps(records))
    metadata = prepare_index(root, output, workers=2, validation_size=3, resume=True)
    assert metadata["counts"]["accepted"] == 24
    assert sum(metadata["counts"].values()) == 28
    assert metadata["validation_size"] == 6  # Duplicate URL groups stay together.
    dataset = RawImageDataset(output)
    splits = {}
    for i in range(len(dataset)):
        record = dataset.record(i)
        splits.setdefault(record["group"], set()).add(record["split"])
    assert all(len(value) == 1 for value in splits.values())
    shard = output / "shards" / "0001.jsonl"
    before = shard.stat().st_mtime_ns
    # Simulate interrupted merge; finished source shards must not decode again.
    (output / "index.json").unlink()
    again = prepare_index(root, output, workers=2, validation_size=3, resume=True)
    assert again["fingerprint"] == metadata["fingerprint"]
    assert shard.stat().st_mtime_ns == before
    sample = dataset[0]
    assert sample["image"].shape == (3, 2048, 2048)
    assert sample["image"].dtype == __import__("torch").uint8
    (root / "datas" / "good.jpg").write_bytes(b"new runtime corruption")
    assert collate_raw([dataset[0]])["errors"]


def test_global_sampler_no_duplicates_resume_and_world_size_change(tmp_path):
    for key in BUCKETS:
        np.save(tmp_path / f"train_{key.replace(',', '_')}.npy", np.arange(100, dtype=np.int64))
    samplers = [GlobalBatchSampler(tmp_path, 1, 2, rank=i, world_size=4, seed=42, start=12) for i in range(4)]
    batches = [iter(s) for s in samplers]
    consumed = []
    for _ in range(2):
        consumed.extend(next(iterator)[0] for iterator in batches)
    assert len(set(consumed)) == 8
    assert consumed == samplers[0].global_batch(12).tolist()
    eight = GlobalBatchSampler(tmp_path, 1, 1, rank=0, world_size=8, seed=42, start=12)
    assert eight.global_batch(12).tolist() == consumed
    resumed = GlobalBatchSampler(tmp_path, 1, 2, rank=0, world_size=4, seed=42, start=13)
    assert next(batches[0]) == next(iter(resumed))


def test_global_sampler_preserves_eight_images_with_three_unequal_ranks(tmp_path):
    for key in BUCKETS:
        np.save(tmp_path / f"train_{key.replace(',', '_')}.npy", np.arange(100, dtype=np.int64))
    for accum, ceiling, sizes in [(1, 3, [3, 3, 2]), (2, 2, [2, 1, 1])]:
        samplers = [GlobalBatchSampler(tmp_path, ceiling, accum, rank=i, world_size=3,
                                      start=86, effective_batch=8) for i in range(3)]
        iterators = [iter(sampler) for sampler in samplers]
        rows = []
        for _ in range(accum):
            for iterator, size in zip(iterators, sizes):
                local = next(iterator)
                assert len(local) == size
                rows.extend(local)
        assert len(set(rows)) == 8
        original = GlobalBatchSampler(tmp_path, 4, 1, world_size=2, start=86)
        assert rows == original.global_batch(86).tolist()
        resumed = GlobalBatchSampler(tmp_path, ceiling, accum, rank=0, world_size=3,
                                     start=87, effective_batch=8)
        assert next(iterators[0]) == next(iter(resumed))
    with pytest.raises(ValueError, match="Effective batch"):
        GlobalBatchSampler(tmp_path, 2, 1, world_size=3, effective_batch=8)
