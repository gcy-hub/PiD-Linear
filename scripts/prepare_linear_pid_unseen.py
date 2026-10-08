"""Prepare paired 2K/4K conditions from training images unseen by a fresh-stage student.

Replay the checkpoint's stateless sampler, including effective-batch transitions.
Unseen refers to this student's fine-tuning, not the released PiD's pretraining.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm

from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO
from pid._src.linear_pid.checkpoint import load_checkpoint, resolve_checkpoint
from pid._src.linear_pid.compare_inference import validate_asset
from pid._src.linear_pid.data import GlobalBatchSampler, RawImageDataset, atomic_json, resize_crop, sha256_file
from pid._src.linear_pid.evaluation import save_png, tensor_image
from pid._src.linear_pid.runtime import Conditioning
from pid._src.linear_pid.training import broadcast_object, init_distributed


def sampling_intervals(payload):
    progress, metadata = payload["progress"], payload["metadata"]
    stop = progress["cursor"]
    if not (stop == progress["stage_step"] == progress["total_step"]):
        raise ValueError("Only a fresh-stage student with cursor == stage_step == total_step is supported")
    if payload["config"].get("init_from"):
        raise ValueError("Inherited-stage students require replaying the parent training history too")
    transitions = metadata.get("batch_transitions", [])
    current = transitions[0]["from"] if transitions else metadata["effective_batch"]
    start, intervals = 0, []
    for transition in transitions:
        end = transition["total_step"]
        if transition["from"] != current or not start <= end <= stop:
            raise ValueError("Inconsistent effective-batch transition history")
        intervals.append((start, end, current))
        start, current = end, transition["to"]
    intervals.append((start, stop, current))
    if current != metadata["effective_batch"] or sum((b - a) * n for a, b, n in intervals) != progress["samples_seen"]:
        raise ValueError("Sampler replay does not account for the checkpoint's samples_seen")
    return intervals


def select_records(payload, index_root, count, seed, workers):
    dataset = RawImageDataset(index_root)
    if dataset.metadata["fingerprint"] != payload["metadata"]["data_fingerprint"]:
        raise ValueError("Training data index differs from the checkpoint")
    intervals = sampling_intervals(payload)
    seen = np.zeros(len(dataset), dtype=bool)
    for start, stop, size in intervals:
        sampler = GlobalBatchSampler(index_root, size, 1, seed=payload["config"]["seed"], effective_batch=size)
        for cursor in tqdm(range(start, stop), desc=f"Replay batch {size}"):
            seen[sampler.global_batch(cursor)] = True
    groups, paths = set(), set()
    for row in tqdm(np.flatnonzero(seen), desc="Exclude used source images"):
        record = dataset.record(int(row))
        groups.add(record["group"])
        paths.add(record["path"])
    rng = np.random.default_rng(seed)
    selected = []
    # Alternate square, landscape and portrait compositions.
    bucket_order = ["1,1", "4,3", "3,4", "16,9", "9,16", "3,2", "2,3"]

    def check(record):
        with Image.open(record["path"]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            width, height = IMAGE_RES_SIZE_INFO["4096"][record["bucket"]]
            if source.width < width or source.height < height:
                raise ValueError("Source does not support native 4K")
            resize_crop(source, record["bucket"], "4096").load()
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(selected) < count:
            before = len(selected)
            for bucket in bucket_order:
                rows = np.load(Path(index_root) / f"train_{bucket.replace(',', '_')}.npy", mmap_mode="r")
                candidates = []
                for row in rng.choice(rows, min(len(rows), 2000), replace=False):
                    if seen[row]:
                        continue
                    record = dataset.record(int(row))
                    width, height = IMAGE_RES_SIZE_INFO["4096"][bucket]
                    if (record["group"] in groups or record["path"] in paths
                            or record["width"] < width or record["height"] < height):
                        continue
                    candidates.append(record | {"manifest_row": int(row)})
                    if len(candidates) == workers:
                        break
                # Decode a bounded number of candidates concurrently.
                futures = [pool.submit(check, record) for record in candidates]
                for future in futures:
                    try:
                        record = future.result()
                    except (OSError, ValueError) as exc:
                        print(f"Rejected inference input: {exc}", flush=True)
                        continue
                    selected.append(record)
                    groups.add(record["group"])
                    paths.add(record["path"])
                    break
                if len(selected) == count:
                    break
            if len(selected) == before:
                raise ValueError("Not enough decodable, native-4K, unseen training images")
    return selected, {"sampling_intervals": intervals, "unique_rows_excluded": int(seen.sum()),
                      "source_group_exclusion": True, "scope": "student fine-tuning only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if min(args.count, args.workers, args.threads) < 1:
        parser.error("count, workers and threads must be positive")
    torch.set_num_threads(args.threads)
    checkpoint = resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    signature = {"checkpoint": str(checkpoint), "checkpoint_progress": payload["progress"],
                 "data_fingerprint": payload["metadata"]["data_fingerprint"], "count": args.count, "seed": args.seed}
    rank, world, device = (0, 1, None) if args.select_only else init_distributed()
    try:
        selection = None
        if rank == 0:
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
                if selection["signature"] != signature:
                    raise ValueError("Selection settings changed; use a different output directory")
            else:
                records, audit = select_records(payload, payload["config"]["index_root"], args.count, args.seed, args.workers)
                selection = {"signature": signature, "audit": audit, "records": records}
                atomic_json(selection_path, selection)
            print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        if args.select_only:
            return
        selection = broadcast_object(selection, rank)
        config = payload["config"]
        conditioning = Conditioning(config["weights_root"], device, text_cache_root=config["text_cache_root"])
        if conditioning.cache.fingerprint != payload["metadata"]["text_cache_fingerprint"]:
            raise ValueError("Text cache differs from the student's training conditions")
        tasks = [(i, record, resolution) for resolution in ["2048", "4096"]
                 for i, record in enumerate(selection["records"])]
        for i, record, resolution in tqdm(tasks[rank::world], desc=f"Conditions GPU {rank}"):
            destination = output / f"unseen_{i:03d}_{resolution}.pt"
            if destination.exists():
                validate_asset(torch.load(destination, map_location="cpu", weights_only=True))
                continue
            with Image.open(record["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = resize_crop(image, record["bucket"], resolution)
                array = np.array(image, copy=True)
            pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
            latent, lq = conditioning.encode_image(pixels)
            embs, mask = conditioning.cache.read(record["id"], record["caption"])
            asset = {"latent": latent.cpu(), "caption_embs": embs.unsqueeze(0), "caption_mask": mask.unsqueeze(0),
                     "null_embs": conditioning.null_embs.cpu(), "null_mask": conditioning.null_mask.cpu(),
                     "height": pixels.shape[-2], "width": pixels.shape[-1], "seed": args.seed + i,
                     "caption": record["caption"], "source_image": record["path"], "source_id": record["id"]}
            validate_asset(asset)
            save_png(tensor_image(lq), destination.with_suffix(".lq.png"))
            preview = image.copy()
            preview.thumbnail((768, 768))
            save_png(preview, destination.with_suffix(".source.png"))
            temporary = destination.with_suffix(".tmp")
            torch.save(asset, temporary)
            temporary.replace(destination)
            atomic_json(destination.with_suffix(".json"), {"source_id": record["id"], "resolution": resolution,
                        "height": asset["height"], "width": asset["width"], "seed": asset["seed"],
                        "asset_sha256": sha256_file(destination)})
            del pixels, latent, lq, asset
        if world > 1:
            torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
