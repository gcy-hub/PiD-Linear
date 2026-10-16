"""Eight-GPU fixed4 A1 -> A2 -> B, advancing only after completed validation."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .config import VDNConfig

PROFILES = {
    "a1": dict(batch=2, accum=1, effective=16, steps=1000, save=500, validate=500),
    "a2": dict(batch=4, accum=1, effective=32, steps=4000, save=2000, validate=2000),
    "b": dict(batch=8, accum=2, effective=128, steps=20000, save=2000, validate=2000),
}
REPO = Path(__file__).resolve().parents[3]


def latest_complete(stage_dir):
    choices = []
    for path in (Path(stage_dir) / "checkpoints").glob("step_*"):
        try:
            marker = json.loads((path / "complete.json").read_text())
            if path.name.endswith(".incomplete") or (path / "state.pt").stat().st_size != marker["bytes"]:
                continue
            choices.append((marker["step"], path.name, path))
        except (OSError, ValueError, KeyError):
            continue
    return max(choices)[2] if choices else None


def finished_checkpoint(root, stage):
    directory = root / stage
    marker = directory / "finished.json"
    if not marker.exists():
        return None
    declared = Path(json.loads(marker.read_text())["checkpoint"])
    # A whole output directory may have moved to a different server.
    if not declared.exists():
        declared = directory / "checkpoints" / declared.name
    if not (declared / "complete.json").exists() or not (declared / "state.pt").exists():
        raise ValueError(f"Incomplete finished checkpoint: {declared}")
    latest = latest_complete(directory)
    if latest is None:
        raise ValueError(f"No complete checkpoint for finished {stage}")
    step = json.loads((latest / "complete.json").read_text())["step"]
    config = json.loads((directory / "config.json").read_text())
    expected = PROFILES[stage]
    saved = config.get("stage_settings", {})
    if (
        step != expected["steps"]
        or config.get("memory_mode", "vda") != "fixed4"
        or any(saved.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"{stage} finished with another architecture/training profile; choose a new output root")
    return latest


def run_pipeline(output_root, common_args, driver=None, world=8, dry_run=False):
    if world != 8:
        raise ValueError("This preset requires 8 GPUs total (one 8-GPU node or two 4-GPU nodes)")
    root = Path(output_root).resolve()
    driver = Path(driver or REPO / "scripts/train_pid_vdn.sh")
    parent = None
    stopping = [False]
    active = [None]
    owned_stop = [None]
    token = f"pid-vdn-pipeline:{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"

    def request_stop(sig, frame):
        stopping[0] = True
        if active[0] is not None:
            path = root / active[0] / "STOP"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x") as stream:
                    stream.write(token)
                owned_stop[0] = path
            except FileExistsError:
                pass

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for number, (stage, profile) in enumerate(PROFILES.items()):
            if stopping[0]:
                return 0
            complete = finished_checkpoint(root, stage)
            if complete is not None:
                print(f"Skip completed {stage}: {complete}", flush=True)
                parent = complete
                continue
            command = [
                "bash",
                str(driver),
                *map(str, common_args),
                "--stage",
                stage,
                "--memory-mode",
                "fixed4",
                "--preset",
                "8gpu",
                "--resume",
                "auto",
                "--output-root",
                str(root),
                "--batch-size",
                str(profile["batch"]),
                "--grad-accum",
                str(profile["accum"]),
                "--effective-batch",
                str(profile["effective"]),
                "--max-steps",
                str(profile["steps"]),
                "--save-steps",
                str(profile["save"]),
                "--validation-steps",
                str(profile["validate"]),
            ]
            if parent is not None:
                command += ["--init-from", str(parent)]
            print(json.dumps({"stage": stage, "settings": profile, "command": command}), flush=True)
            if dry_run:
                parent = root / stage / "checkpoints" / "<latest-complete-after-validation>"
                continue
            root.mkdir(parents=True, exist_ok=True)
            active[0] = stage
            env = {**os.environ, "MASTER_PORT": str(int(os.getenv("MASTER_PORT", "29583")) + number)}
            with subprocess.Popen(command, cwd=REPO, env=env) as process:
                code = process.wait()
            active[0] = None
            if code:
                print(f"{stage} failed (exit {code}); later stages were not started", flush=True)
                return code
            if stopping[0]:
                return 0
            parent = finished_checkpoint(root, stage)
            if parent is None:
                print(
                    f"{stage} paused before training/validation completed; rerun the same command to resume", flush=True
                )
                return 0
        return 0
    finally:
        path = owned_stop[0]
        if path and path.exists() and path.read_text() == token:
            path.unlink()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    defaults = VDNConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/pid-vdn-fixed4-8gpu")
    for name in ("weights_root", "index_root", "text_cache_root", "subset_root", "gallery_root", "swanlab_mode"):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=int(os.getenv("CPU_THREADS", "1")))
    parser.add_argument("--region-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    root, dry = args.pop("output_root"), args.pop("dry_run")
    common = [item for key, value in args.items() for item in ("--" + key.replace("_", "-"), str(value))]
    count = int(
        os.getenv("NPROC_PER_NODE")
        or len(os.getenv("GPU_IDS", os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")).split(","))
    )
    world = count * int(os.getenv("NNODES", "1"))
    raise SystemExit(run_pipeline(root, common, world=world, dry_run=dry))


if __name__ == "__main__":
    main()
