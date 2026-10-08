"""Encode all metadata captions on local GPUs with durable, resumable BF16 shards."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from pid._src.linear_pid.data import atomic_json
from pid._src.linear_pid.text_cache import (
    CACHE_SCHEMA,
    CaptionShardWriter,
    digest_json,
    encoding_spec,
    prefetch_sources,
    publish_cache,
    source_inventory,
)
from pid._src.linear_pid.text_encoder import NEGATIVE_PROMPT, GemmaTextEncoder


def prepare_build(args):
    root = Path(args.output_root)
    sources = source_inventory(args.dataset_root)
    # Full weight hashes are checked at every launch; never reuse a cache made
    # with different weights, tokenizer rules or caption metadata.
    spec = encoding_spec(args.weights_root)
    build = {
        "schema": CACHE_SCHEMA,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "sources": sources,
        "spec": spec,
        "spec_fingerprint": digest_json(spec),
    }
    build["fingerprint"] = digest_json(build)
    path = root / "build.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != build:
            raise ValueError("Text cache inputs changed; select a fresh output-root")
        if not args.resume:
            raise ValueError("Cache already exists; pass --resume to continue")
    else:
        atomic_json(path, build)
    return build


def run(args, rank, world, device, stopped):
    root = Path(args.output_root)
    job_start = time.monotonic()
    build, error = None, None
    if rank == 0:
        try:
            build = prepare_build(args)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if world > 1:
        objects = [build, error]
        dist.broadcast_object_list(objects, src=0)
        build, error = objects
    if error:
        raise RuntimeError(error)
    if (root / "cache.json").exists():
        if rank == 0:
            print(f"Complete text cache already exists: {root}", flush=True)
        return
    encoder = GemmaTextEncoder(args.weights_root, device)
    if rank == 0 and not (root / "special.pt").exists():
        embeddings, masks = encoder.encode_text(["", NEGATIVE_PROMPT])
        special = {
            "empty_embs": embeddings[0:1].cpu(),
            "empty_mask": masks[0:1].cpu(),
            "null_embs": embeddings[1:2].cpu(),
            "null_mask": masks[1:2].cpu(),
            "spec_fingerprint": build["spec_fingerprint"],
        }
        temp = root / "special.pt.tmp"
        with temp.open("wb") as stream:
            torch.save(special, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(root / "special.pt")
    sources = [Path(args.dataset_root) / "data_jsons" / s["name"] for s in build["sources"].values()][rank::world]
    status_path = root / f"rank_{rank}.status.json"
    samples, newly_encoded, completed = 0, 0, 0
    already_complete = sum((root / "shards" / path.stem / "complete.json").exists() for path in sources)
    bar = tqdm(
        total=len(sources), initial=already_complete, desc=f"Gemma shards rank {rank}", position=rank, mininterval=10
    )
    writer = None
    last_commit, last_status = job_start, 0.0

    def status(state, stem=""):
        atomic_json(
            status_path,
            {
                "pid": os.getpid(),
                "rank": rank,
                "world_size": world,
                "state": state,
                "shard": stem,
                "shard_cursor": writer.cursor if writer else 0,
                "samples_seen": samples,
                "newly_encoded": newly_encoded,
                "completed_shards": completed,
                "assigned_shards": len(sources),
                "elapsed_seconds": round(time.monotonic() - job_start, 2),
                "updated_unix": time.time(),
                "batch_size": args.batch_size,
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            },
        )

    try:
        for source in prefetch_sources(sources, args.workers):
            if (
                stopped[0]
                or (root / "STOP").exists()
                or (args.max_seconds and time.monotonic() - job_start >= args.max_seconds)
            ):
                break
            expected = build["sources"][source["stem"]]
            current = (Path(args.dataset_root) / "data_jsons" / expected["name"]).stat()
            if current.st_size != expected["bytes"] or current.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"Metadata changed while encoding {source['stem']}")
            writer = CaptionShardWriter(root, source, build["spec_fingerprint"])
            samples += writer.cursor
            committed_cursor = writer.cursor
            while writer.cursor < len(source["captions"]):
                if (
                    stopped[0]
                    or (root / "STOP").exists()
                    or (args.max_seconds and time.monotonic() - job_start >= args.max_seconds)
                ):
                    stopped[0] = True
                    break
                start = writer.cursor
                batch = source["captions"][start : start + args.batch_size]
                embeddings, masks = encoder.encode_text(batch)
                writer.write(embeddings, masks)
                del embeddings, masks
                samples += len(batch)
                newly_encoded += len(batch)
                now = time.monotonic()
                if writer.cursor - committed_cursor >= args.save_rows or now - last_commit >= args.save_seconds:
                    writer.commit()
                    committed_cursor, last_commit = writer.cursor, now
                if now - last_status >= 10:
                    status("encoding", source["stem"])
                    last_status = now
                    bar.set_postfix(shard=source["stem"], row=writer.cursor, new=newly_encoded)
            if writer.cursor == len(source["captions"]):
                was_complete = writer.complete is not None
                writer.finish()
                completed += 1
                if not was_complete:
                    bar.update(1)
            else:
                writer.commit()
                break
            status("encoding", source["stem"])
            writer = None
        status("paused" if stopped[0] or completed != len(sources) else "rank_complete")
    except Exception:
        # Previous committed progress stays valid even if OOM/serialization fails.
        status("failed")
        raise
    finally:
        bar.close()
    complete = torch.tensor(int(completed == len(sources)), dtype=torch.int32)
    if world > 1:
        dist.all_reduce(complete, op=dist.ReduceOp.MIN)
    if complete.item() and rank == 0:
        metadata = publish_cache(root)
        print(f"Complete: {metadata['count']:,} captions / {len(metadata['shards'])} shards in {root}", flush=True)
    elif rank == 0:
        print("Progress saved; rerun the same --resume command to continue.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/home/ganchangyi/dataset/MultiAspect-4K-1M")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--weights-root", default="/home/ganchangyi/huggingface_ckpts")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2, help="Bounded metadata prefetch threads per GPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-rows", type=int, default=64)
    parser.add_argument("--save-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=1680, help="Save and exit after 28 minutes; 0 disables")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(args.batch_size, args.threads, args.save_rows, args.save_seconds) <= 0
        or min(args.workers, args.max_seconds) < 0
    ):
        parser.error("Invalid batch/thread/checkpoint/worker/time value")
    if args.output_root is None:
        args.output_root = str(Path(args.dataset_root) / "linear_pid_text_cache")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rank, world, local = (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(local)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    # One launcher owns the entire output. Other ranks are covered by rank 0's lock.
    lock = (root / ".encode.lock").open("a") if rank == 0 else None
    try:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if world > 1:
            dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        run(args, rank, world, torch.device("cuda", local), stopped)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
