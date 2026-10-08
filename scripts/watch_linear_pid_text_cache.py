"""A short cron check that resumes caption encoding after node01 process cleanup.

Every invocation exits quickly. The system cron daemon launches the next check,
so a killed user-space supervisor cannot strand this resumable workload.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def process_identity(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if tail[0] == "Z" else tail[19]
    except (OSError, IndexError):
        return None


def cron_marker(root):
    import hashlib

    return "# linear-pid-text-cache:" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def update_cron(root, line=None):
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode and "no crontab" not in result.stderr.lower():
        raise RuntimeError(result.stderr.strip())
    marker = cron_marker(root)
    lines = [entry for entry in result.stdout.splitlines() if not entry.endswith(marker)]
    if line:
        lines.append(line + " " + marker)
    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)


def saved_progress(root):
    rows, shards = 0, 0
    for path in (root / "shards").glob("*/progress.json"):
        try:
            rows += json.loads(path.read_text())["cursor"]
            shards += int((path.parent / "complete.json").exists())
        except (OSError, ValueError, KeyError):
            pass
    return {"saved_rows": rows, "completed_shards": shards}


def gpu_capacity(ids, minimum):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines()}
    selected = [int(value) for value in ids.split(",")]
    available = {str(gpu): memory[gpu] for gpu in selected}
    return all(memory[gpu] >= minimum for gpu in selected), available


def check_once(root, config):
    status_path = root / "watch.status.json"
    if (root / "cache.json").exists():
        metadata = json.loads((root / "cache.json").read_text())
        write_json(status_path, {"state": "complete", "count": metadata["count"], "updated_unix": time.time()})
        update_cron(root)
        print(f"Cache complete: {metadata['count']:,} captions; removed this cache's cron entry.", flush=True)
        return
    if (root / "STOP").exists():
        write_json(status_path, {"state": "stopped", "updated_unix": time.time(), **saved_progress(root)})
        return
    if socket.gethostname() != config["hostname"]:
        print(f"Cache is assigned to {config['hostname']}; skipping this host.", flush=True)
        return
    previous_path = root / "run.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous and process_identity(previous["pid"]) == previous["process_identity"]:
        write_json(
            status_path,
            {
                "state": "running",
                "pid": previous["pid"],
                "started_unix": previous["started_unix"],
                "updated_unix": time.time(),
                **saved_progress(root),
            },
        )
        return
    # A killed torchrun parent may leave its ranks alive. Rank zero owns this
    # lock until encoding ends; do not create a second launcher in that interval.
    with (root / ".encode.lock").open("a") as encode_lock:
        try:
            fcntl.flock(encode_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write_json(status_path, {"state": "workers_running", "updated_unix": time.time(), **saved_progress(root)})
            return
    ready, free = gpu_capacity(config["gpu_ids"], config["min_free_mib"])
    if not ready:
        write_json(
            status_path,
            {"state": "waiting_for_gpu_memory", "free_mib": free, "updated_unix": time.time(), **saved_progress(root)},
        )
        print(f"Waiting for free GPU memory: {free}", flush=True)
        return
    repo = Path(config["repo"])
    env = os.environ.copy()
    env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    env.update(GPU_IDS=config["gpu_ids"], CPU_THREADS=str(config["threads"]), PYTHONUNBUFFERED="1")
    command = [
        "bash",
        str(repo / "scripts/prepare_linear_pid_text_cache.sh"),
        "--dataset-root",
        config["dataset_root"],
        "--output-root",
        str(root),
        "--weights-root",
        config["weights_root"],
        "--batch-size",
        str(config["batch_size"]),
        "--workers",
        str(config["workers"]),
        "--threads",
        str(config["threads"]),
        "--max-seconds",
        str(config["max_seconds"]),
        "--resume",
    ]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    label = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logs / f"encode_{label}.log"
    with log.open("a") as stream:
        child = subprocess.Popen(
            command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
    launch = {
        "pid": child.pid,
        "process_identity": process_identity(child.pid),
        "started_unix": time.time(),
        "log": str(log),
        "command": command,
        "launch_count": previous.get("launch_count", 0) + 1 if previous else 1,
    }
    write_json(previous_path, launch)
    write_json(status_path, {"state": "starting", "updated_unix": time.time(), **launch, **saved_progress(root)})
    print(f"Started/resumed PID {child.pid}, GPUs {config['gpu_ids']}, log {log}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/home/ganchangyi/dataset/MultiAspect-4K-1M")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--weights-root", default="/home/ganchangyi/huggingface_ckpts")
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "0,1,2,3"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=1680)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", action="store_true", help="Install a two-minute cron check and start now")
    group.add_argument("--uninstall", action="store_true", help="Remove only this cache's cron entry")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root or Path(args.dataset_root) / "linear_pid_text_cache").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.uninstall:
        update_cron(root)
        print("Removed this cache's cron entry. Existing encoding reaches its next saved exit.")
        return
    if args.status:
        path = root / "watch.status.json"
        print(path.read_text() if path.exists() else "No watch status yet.")
        return
    with (root / ".watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        config_path = root / "watch.config.json"
        if args.install:
            if min(args.batch_size, args.threads, args.max_seconds, args.min_free_mib) <= 0 or args.workers < 0:
                parser.error("Invalid batch/thread/time/memory/worker value")
            ids = args.gpu_ids.split(",")
            if any(not value.isdecimal() for value in ids) or len(set(ids)) != len(ids):
                parser.error("gpu-ids must be unique numeric GPU indices")
            config = {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "weights_root": args.weights_root,
                "gpu_ids": args.gpu_ids,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "threads": args.threads,
                "max_seconds": args.max_seconds,
                "min_free_mib": args.min_free_mib,
                "repo": str(Path(__file__).resolve().parents[1]),
                "python": sys.executable,
                "hostname": socket.gethostname(),
            }
            write_json(config_path, config)
            log = root / "logs"
            log.mkdir(exist_ok=True)
            command = shlex.join([sys.executable, str(Path(__file__).resolve()), "--output-root", str(root)])
            update_cron(root, f"*/2 * * * * {command} >> {shlex.quote(str(log / 'watch.log'))} 2>&1")
        elif config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            parser.error("Install this cache's watcher first with --install")
        check_once(root, config)


if __name__ == "__main__":
    main()
