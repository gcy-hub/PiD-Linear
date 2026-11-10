import json
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from pid._src.common.data import RawImageDataset, atomic_json, collate_raw
from pid._src.common.text_cache import CaptionShardWriter, TextCacheReader, publish_cache, read_prompt_source


def fixture_source(tmp_path):
    path = tmp_path / "0001.json"
    path.write_text(
        json.dumps(
            [
                {"en_caption": " first caption "},
                {"en_caption": ""},
                {"en_caption": "second caption"},
                {"en_caption": None},
            ]
        )
    )
    return read_prompt_source(path)


def publish_fixture(root):
    atomic_json(
        root / "build.json",
        {
            "sources": {"0001": {}},
            "spec_fingerprint": "spec",
            "spec": {"shape": [3, 4]},
            "fingerprint": "build",
            "dataset_root": "fixture",
        },
    )
    special = {
        "empty_embs": torch.full((1, 3, 4), -1, dtype=torch.bfloat16),
        "null_embs": torch.full((1, 3, 4), -2, dtype=torch.bfloat16),
        "empty_mask": torch.tensor([[1, 0, 0]], dtype=torch.bool),
        "null_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }
    torch.save(special, root / "special.pt")
    return publish_cache(root)


def test_text_cache_partial_resume_bf16_bits_and_caption_identity(tmp_path):
    source = fixture_source(tmp_path)
    root = tmp_path / "cache"
    assert source["rows"] == [0, 2]
    writer = CaptionShardWriter(root, source, "spec", shape=(3, 4))
    outputs = torch.arange(24).reshape(2, 3, 4).to(torch.bfloat16)
    masks = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    writer.write(outputs[:1], masks[:1])
    writer.commit()
    # A kill after writing but before publishing progress must re-encode row 1.
    writer.write(torch.zeros_like(outputs[1:]), masks[1:])
    writer.arrays["embeddings.npy"].flush()
    resumed = CaptionShardWriter(root, source, "spec", shape=(3, 4))
    assert resumed.cursor == 1
    with pytest.raises(ValueError, match="incomplete"):
        resumed.finish()
    resumed.write(outputs[1:], masks[1:])
    complete = resumed.finish()
    metadata = publish_fixture(root)
    assert metadata["count"] == complete["count"] == 2
    reader = TextCacheReader(root)
    for i, row in enumerate([0, 2]):
        embs, mask = reader.read(f"0001/{row:06d}", source["captions"][i])
        torch.testing.assert_close(embs, outputs[i], rtol=0, atol=0)
        torch.testing.assert_close(mask, masks[i])
        assert embs.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="caption mismatch"):
        reader.read("0001/000000", "changed caption")
    with pytest.raises(ValueError, match="caption mismatch"):
        reader.read("0001/000001", "missing caption")
    rebuilt = pickle.loads(pickle.dumps(reader))
    assert not rebuilt._arrays
    torch.testing.assert_close(rebuilt.read("0001/000002", "second caption")[0], outputs[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="inputs changed"):
        CaptionShardWriter(root, source, "other-model", shape=(3, 4))


def test_reader_rejects_incomplete_and_truncated_shards(tmp_path):
    root = tmp_path / "cache"
    writer = CaptionShardWriter(root, fixture_source(tmp_path), "spec", shape=(3, 4))
    writer.write(torch.zeros(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(FileNotFoundError, match="Complete Gemma cache missing"):
        TextCacheReader(root)
    with pytest.raises(ValueError, match="Unfinished"):
        publish_fixture(root)
    writer.finish()
    publish_fixture(root)
    (writer.path / "embeddings.npy").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="truncated"):
        TextCacheReader(root).read("0001/000000", "first caption")


def test_empty_metadata_shard_can_complete(tmp_path):
    path = tmp_path / "0001.json"
    path.write_text('[{"en_caption": ""}]')
    root = tmp_path / "cache"
    writer = CaptionShardWriter(root, read_prompt_source(path), "spec", shape=(3, 4))
    assert writer.finish()["count"] == 0
    assert publish_fixture(root)["count"] == 0


def test_cached_conditioning_never_constructs_gemma_and_preserves_dropout(tmp_path, monkeypatch):
    from pid._src.common import runtime
    from pid._src.tokenizers import flux_vae

    root = tmp_path / "cache"
    writer = CaptionShardWriter(root, fixture_source(tmp_path), "spec", shape=(3, 4))
    expected = torch.arange(24).reshape(2, 3, 4).to(torch.bfloat16)
    mask = torch.ones(2, 3, dtype=torch.bool)
    writer.write(expected, mask)
    writer.finish()
    publish_fixture(root)

    def forbid_gemma(*args, **kwargs):
        raise AssertionError("Cached training must never instantiate Gemma")

    monkeypatch.setattr(runtime, "GemmaTextEncoder", forbid_gemma)
    monkeypatch.setattr(flux_vae, "FluxVAE", lambda **kwargs: object())
    conditions = runtime.Conditioning("unused", "cpu", text_cache_root=root)
    assert conditions.text is conditions.tokenizer is conditions.encoder is None
    actual, actual_mask = conditions.encode_text(["first caption", ""], cached_embs=expected, cached_mask=mask)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1:2], conditions.empty_embs, rtol=0, atol=0)
    torch.testing.assert_close(actual_mask[1:2], conditions.empty_mask)
    assert not torch.equal(conditions.empty_embs, conditions.null_embs)
    with pytest.raises(ValueError, match="data loader"):
        conditions.encode_text(["new prompt"])


def test_raw_loader_prefetches_cached_embeddings(tmp_path):
    root = tmp_path / "cache"
    writer = CaptionShardWriter(root, fixture_source(tmp_path), "spec", shape=(3, 4))
    outputs = torch.arange(24).reshape(2, 3, 4).to(torch.bfloat16)
    masks = torch.ones(2, 3, dtype=torch.bool)
    writer.write(outputs, masks)
    writer.finish()
    publish_fixture(root)
    index = tmp_path / "index"
    index.mkdir()
    image = tmp_path / "image.png"
    Image.new("RGB", (2048, 2048)).save(image)
    record = {"id": "0001/000002", "caption": "second caption", "path": str(image), "bucket": "1,1"}
    (index / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    np.save(index / "offsets.npy", np.asarray([0], dtype=np.int64))
    atomic_json(index / "index.json", {"fingerprint": "fixture"})
    dataset = RawImageDataset(index, text_cache_root=root)
    batch = collate_raw([dataset[0]])
    torch.testing.assert_close(batch["text_embs"][0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(batch["text_mask"][0], masks[1])
    assert batch["text_embs"].dtype == torch.bfloat16
