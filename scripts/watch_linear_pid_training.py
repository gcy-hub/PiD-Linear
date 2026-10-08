"""Resume node01 training with short cron checks that survive process cleanup."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__:
    from .watch_linear_pid_text_cache import gpu_capacity, process_identity, write_json
else:
    from watch_linear_pid_text_cache import gpu_capacity, process_identity, write_json


def cron_marker(root):
    return "# linear-pid-training:" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def update_cron(root, line=None):
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode and "no crontab" not in result.stderr.lower():
        raise RuntimeError(result.stderr.strip())
    marker = cron_marker(root)
    entries = [entry for entry in result.stdout.splitlines() if not entry.endswith(marker)]
    if line:
        entries.append(line + " " + marker)
    subprocess.run(["crontab", "-"], input="\n".join(entries) + "\n", text=True, check=True)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def saved_progress(root):
    for directory in sorted((root / "checkpoints").glob("step_*"), reverse=True):
        if directory.name.endswith(".incomplete"):
            continue
        marker = read_json(directory / "complete.json")
        if marker:
            try:
                if (directory / "state.pt").stat().st_size == marker["bytes"]:
                    return {"checkpoint": str(directory), "saved_step": marker["step"]}
            except (OSError, KeyError):
                pass
    return {"checkpoint": None, "saved_step": 0}


def locked(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def active_training_pids(output_root):
    """Also recognize a manually started torchrun during its bootstrap window."""
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if (path / "comm").read_text().strip() == "pt_data_worker":
                continue
            argv = [value.decode() for value in (path / "cmdline").read_bytes().split(b"\0") if value]
            if "scripts.train" not in argv or "--output-root" not in argv:
                continue
            root = argv[len(argv) - 1 - argv[::-1].index("--output-root") + 1]
            if Path(root).resolve() == Path(output_root).resolve() and process_identity(int(path.name)):
                found.append(int(path.name))
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            pass
    return found


def check_once(root, config):
    def status(state, **values):
        write_json(root / "training_watch.status.json", {
            "state": state, "updated_unix": time.time(), **saved_progress(root), **values,
        })

    if socket.gethostname() != config["hostname"]:
        return
    if not config.get("enabled", True) or (root / "STOP").exists():
        status("stopped")
        return
    previous = read_json(root / "training_watch.run.json")
    if previous and previous.get("process_identity") and process_identity(previous["pid"]) == previous["process_identity"]:
        status("running", **previous)
        return
    manual = active_training_pids(root.parent)
    if manual:
        status("running_external", pids=manual)
        return
    # The launcher can be killed before its ranks. Rank zero's training lock
    # prevents a second launcher from consuming the same data/checkpoint.
    if locked(root / ".train.lock") or locked(root / ".training_launch.lock"):
        status("workers_running")
        return
    if previous:
        result = read_json(Path(previous["result"]))
        if result and result["returncode"] not in {0, -9, -15, 137, 143} and previous.get("install_id") == config["install_id"]:
            status("training_error", **previous, returncode=result["returncode"])
            return
    ready, free = gpu_capacity(config["gpu_ids"], config["min_free_mib"])
    if not ready:
        status("waiting_for_gpu_memory", free_mib=free)
        return
    logs = root / "watch_logs"
    logs.mkdir(exist_ok=True)
    label = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    log, result = logs / f"train_{label}.log", logs / f"train_{label}.exit.json"
    env = os.environ.copy()
    env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    env.update(
        GPU_IDS=config["gpu_ids"], CPU_THREADS=str(config["threads"]),
        OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]),
        OPENBLAS_NUM_THREADS=str(config["threads"]), PYTHONUNBUFFERED="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    # Cron starts with a fresh environment. Retain this diagnostic/workaround
    # across restarts without retaining CUDA_LAUNCH_BLOCKING's slowdown.
    if config.get("disable_fla_tensor_cache", False):
        env["FLA_DISABLE_TENSOR_CACHE"] = "1"
    command = [config["python"], str(Path(__file__).resolve()), "--run-dir", str(root), "--run-child", str(result)]
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=config["repo"], env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    launch = {
        "pid": child.pid, "process_identity": process_identity(child.pid), "started_unix": time.time(),
        "log": str(log), "result": str(result), "install_id": config["install_id"],
        "launch_count": previous.get("launch_count", 0) + 1 if previous else 1,
    }
    write_json(root / "training_watch.run.json", launch)
    status("starting", **launch)
    print(f"Resuming training: PID {child.pid}, GPUs {config['gpu_ids']}, log {log}", flush=True)


def run_child(root, result_path):
    config = json.loads((root / "training_watch.config.json").read_text())
    # Also protect the launcher bootstrap, before rank zero acquires its lock.
    with (root / ".training_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STOP").exists() or locked(root / ".train.lock") or not config["enabled"]:
            returncode = 0
        else:
            returncode = subprocess.call(config["command"], cwd=config["repo"])
        write_json(Path(result_path), {"returncode": returncode, "ended_unix": time.time()})
    return returncode


def install_config(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pid._src.linear_pid.attention import resolve_layers

    layers = resolve_layers(args.layers)
    root = Path(args.output_root).resolve() / ("kda_" + ("-".join(map(str, layers)) or "full"))
    root.mkdir(parents=True, exist_ok=True)
    ids = args.gpu_ids.split(",")
    if any(not value.isdecimal() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("GPU IDs must be unique numeric indices")
    if min(args.batch_size, args.grad_accum, args.threads, args.max_seconds, args.min_free_mib) <= 0 or args.workers < 0:
        raise ValueError("Invalid batch/accum/thread/time/memory/worker value")
    saved = read_json(root / "config.json")
    metadata = read_json(root / "metadata.json")
    effective = args.effective_batch or len(ids) * args.batch_size * args.grad_accum
    if effective % args.grad_accum or not len(ids) <= effective // args.grad_accum <= len(ids) * args.batch_size:
        raise ValueError("Effective batch must divide accumulation and give every GPU 1..batch_size images")
    allow_batch_change = bool(getattr(args, "allow_batch_size_change", False)
                              or (saved or {}).get("allow_batch_size_change", False))
    if metadata and metadata["effective_batch"] != effective and not allow_batch_change:
        raise ValueError(f"Checkpoint effective batch is {metadata['effective_batch']}; requested {effective}; "
                         "use --allow-batch-size-change to continue in the same directory")
    if saved is None:
        from pid._src.configs.linear_pid.config import LinearPiDConfig
        saved = LinearPiDConfig(layers=layers).to_dict()
    # Preserve the run's original spelling across /home and /fs1 symlink
    # aliases; gallery identities include this string, although locks use realpath.
    output_root = saved.get("output_root", args.output_root)
    if Path(output_root).resolve() != Path(args.output_root).resolve():
        output_root = args.output_root
    saved.update(
        preset="4gpu", layers=layers, output_root=output_root,
        batch_size=args.batch_size, grad_accum=args.grad_accum, workers=args.workers,
        effective_batch=args.effective_batch,
        threads=args.threads, max_seconds=args.max_seconds, resume="auto", init_from="",
        allow_batch_size_change=allow_batch_change,
    )
    repo = Path(__file__).resolve().parents[1]
    stage_config = getattr(args, "stage_config", None)
    if stage_config:
        from scripts.launch_linear_pid_stage import read_stage_config, stage_command

        source, values = read_stage_config(stage_config, "4gpu", {
            "output_root": output_root, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.effective_batch, "workers": args.workers, "threads": args.threads,
            "max_seconds": args.max_seconds,
        })
        if values["layers"] != layers:
            raise ValueError("Watcher --layers must match the stage JSON layout")
        source_override = getattr(args, "source_run_dir", None)
        if source is None and source_override:
            raise ValueError("Original-PiD initialization cannot override a parent run")
        source = source_override or source
        # Validate the parent/destination selection without copying it here.
        stage_command(values, "4gpu", source)
        command = [sys.executable, str(repo / "scripts/launch_linear_pid_stage.py"),
                   "--stage-config", str(Path(stage_config).resolve()), "--preset", "4gpu",
                   "--output-root", output_root,
                   "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
                   "--effective-batch", str(args.effective_batch), "--workers", str(args.workers),
                   "--threads", str(args.threads), "--max-seconds", str(args.max_seconds)]
        if source:
            command.extend(["--source-run-dir", source])
    else:
        command = ["bash", str(repo / "scripts/train_linear_pid.sh")]
        for key, value in saved.items():
            if key in {"fla_commit", "init_from", "local_mixing", "activation_checkpointing"}:
                continue
            if key == "allow_batch_size_change":
                if value:
                    command.append("--allow-batch-size-change")
                continue
            if key == "layers":
                value = ",".join(map(str, value)) or "0"
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if not saved["local_mixing"]:
            command.append("--no-local-mixing")
        if not saved["activation_checkpointing"]:
            command.append("--no-activation-checkpointing")
    return root, {
        "enabled": True, "install_id": str(time.time_ns()), "hostname": socket.gethostname(),
        "gpu_ids": args.gpu_ids, "threads": args.threads, "min_free_mib": args.min_free_mib,
        "disable_fla_tensor_cache": bool(getattr(args, "disable_fla_tensor_cache", False)),
        "repo": str(repo), "python": sys.executable, "command": command,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--output-root", default="/home/ganchangyi/code/PiD-Linear/outputs/linear-pid")
    parser.add_argument("--layers", default="4")
    parser.add_argument("--stage-config", help="Use JSON original-PiD/parent initialization and later resumes")
    parser.add_argument("--source-run-dir", help="Override the stage JSON's parent run")
    parser.add_argument("--gpu-ids", default=os.getenv("GPU_IDS", "0,1,2,3"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=38000)
    parser.add_argument("--disable-fla-tensor-cache", action="store_true",
                        help="Keep FLA's tensor index cache disabled on every cron restart")
    parser.add_argument("--allow-batch-size-change", action="store_true",
                        help="Retain model/optimizer/history in this run when changing the effective batch")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", action="store_true")
    group.add_argument("--uninstall", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--run-child", metavar="RESULT_PATH")
    args = parser.parse_args()
    if args.install:
        root, config = install_config(args)
    elif args.run_dir:
        root = Path(args.run_dir).resolve()
        config = read_json(root / "training_watch.config.json")
    else:
        parser.error("Use --install, or supply --run-dir")
    if args.run_child:
        sys.exit(run_child(root, args.run_child))
    if args.status:
        print(json.dumps({**(read_json(root / "training_watch.status.json") or {}), **saved_progress(root)}, indent=2))
        return
    if config is None:
        parser.error("Install the training watcher first")
    with (root / ".training_watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if args.uninstall:
            config["enabled"] = False
            write_json(root / "training_watch.config.json", config)
            update_cron(root)
            print("Removed this run's cron check; active training continues until its configured stop.")
            return
        if args.install:
            write_json(root / "training_watch.config.json", config)
            logs = root / "watch_logs"
            logs.mkdir(exist_ok=True)
            command = shlex.join([sys.executable, str(Path(__file__).resolve()), "--run-dir", str(root)])
            update_cron(root, f"* * * * * {command} >> {shlex.quote(str(logs / 'watch.log'))} 2>&1")
        check_once(root, config)


if __name__ == "__main__":
    main()
