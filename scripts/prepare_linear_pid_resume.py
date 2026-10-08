#!/usr/bin/env python3
"""Pin a complete training checkpoint for an independent Slurm continuation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path


def completed(run_dir):
    results = []
    for path in (Path(run_dir) / "checkpoints").glob("step_*"):
        if not path.is_dir() or path.name.endswith(".incomplete"):
            continue
        try:
            info = json.loads((path / "complete.json").read_text())
            if (path / "state.pt").stat().st_size == info["bytes"]:
                results.append(path)
        except (OSError, ValueError, KeyError):
            pass
    return sorted(results)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot_resume(source_run_dir, output_root):
    source, output = Path(source_run_dir).resolve(), Path(output_root).resolve()
    if source == output or source in output.parents:
        raise ValueError("The independent output root must be outside the active source run")
    seed = output / "initial_state"
    seed.mkdir(parents=True, exist_ok=True)
    with (seed / ".fork.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = completed(seed)
        if existing:
            return existing[-1]
        # An open source descriptor remains readable if retention removes its
        # directory while copying. Never modify or pin the active run's files.
        stream = None
        for _ in range(3):
            sources = completed(source)
            if not sources:
                raise FileNotFoundError(f"No complete checkpoints in {source}")
            checkpoint = sources[-1]
            try:
                info = json.loads((checkpoint / "complete.json").read_text())
                stream = (checkpoint / "state.pt").open("rb")
                if os.fstat(stream.fileno()).st_size != info["bytes"]:
                    stream.close()
                    stream = None
                    continue
                break
            except FileNotFoundError:
                continue
        if stream is None:
            raise RuntimeError("Source checkpoints changed while opening; retry preparation")
        destination = seed / "checkpoints" / checkpoint.name
        temporary = destination.with_name(destination.name + ".incomplete")
        temporary.mkdir(parents=True, exist_ok=True)
        print(f"Copying complete step {info['step']} to {destination}", file=sys.stderr, flush=True)
        with stream, (temporary / "state.pt").open("wb") as target:
            shutil.copyfileobj(stream, target, length=32 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if (temporary / "state.pt").stat().st_size != info["bytes"]:
            raise IOError("Incomplete checkpoint copy; no completion marker published")
        write_json(temporary / "source.json", {
            "source_run_dir": str(source), "source_checkpoint": str(checkpoint),
            "copied_unix": time.time(), "step": info["step"], "bytes": info["bytes"],
        })
        write_json(temporary / "complete.json", info)
        os.replace(temporary, destination)
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(snapshot_resume(args.source_run_dir, args.output_root))


if __name__ == "__main__":
    main()
