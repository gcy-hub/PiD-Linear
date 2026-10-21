"""Detached, renewable controller for time-limited, resumable PiD-VDN training."""

import argparse
import ctypes
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pid._src.vdn_pid.config import VDNConfig

REPO = Path(__file__).resolve().parents[1]
STAGES = ("a1", "a2", "b")
OWN_STOP = "pid-vdn-repeat:"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def process_identity(pid):
    """Linux PID plus creation ticks: never signal an unrelated reused PID."""
    try:
        path = Path(f"/proc/{pid}")
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19]), int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def enable_subreaper():
    # Adopt orphan ranks even if torchrun exits before the first /proc snapshot.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")


def remember_children(parent, known, excluded=None, include_parent=True):
    records = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if entry.name.isdigit():
                identity = process_identity(int(entry.name))
                if identity:
                    records[int(entry.name)] = identity
    excluded = excluded or {}
    blocked = {pid for pid, ticks in excluded.items() if pid in records and records[pid][0] == ticks}
    descendants = {parent}
    while True:
        added = {pid for pid, (_, ppid) in records.items() if ppid in descendants and pid not in blocked} - descendants
        if not added:
            break
        descendants.update(added)
    for pid in descendants:
        if pid in records and (include_parent or pid != parent):
            known[pid] = records[pid][0]


def kill_remaining(known, excluded):
    # Torchrun ranks can start their own sessions, so killpg alone is insufficient.
    # Freeze, then rescan adopted children until all owned processes are stopped.
    deadline = time.monotonic() + 2
    while True:
        remember_children(os.getpid(), known, excluded, include_parent=False)
        stopped = True
        for pid, ticks in list(known.items()):
            identity = process_identity(pid)
            if identity and identity[0] == ticks:
                try:
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
                    if fields[0] not in ("T", "t"):
                        stopped = False
                    os.kill(pid, signal.SIGSTOP)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        if stopped or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    for pid, ticks in reversed(list(known.items())):
        identity = process_identity(pid)
        if identity and identity[0] == ticks:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def reap_adopted(known, training_pid):
    remaining = {pid: ticks for pid, ticks in known.items() if pid != training_pid}
    deadline = time.monotonic() + 2
    while remaining:
        for pid, ticks in list(remaining.items()):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            identity = process_identity(pid)
            if identity is None or identity[0] != ticks:
                remaining.pop(pid)
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Owned training children did not exit: {list(remaining)}")
        time.sleep(0.01)


def stage_finished(root, stage):
    value = read_json(root / stage / "finished.json")
    if not value:
        return False
    checkpoint = Path(value["checkpoint"])
    if not (checkpoint / "complete.json").exists() or not (checkpoint / "state.pt").exists():
        raise RuntimeError(f"Finished marker refers to an incomplete checkpoint: {checkpoint}")
    return True


def select_stage(root, requested):
    stages = STAGES if requested == "all" else (requested,)
    return next((stage for stage in stages if not stage_finished(root, stage)), None)


def training_command(root, stage, request):
    # A stopped stage's config supplies paths/architecture even in a fresh terminal.
    saved = read_json(root / stage / "config.json")
    if not saved:
        previous = {"a2": "a1", "b": "a2"}.get(stage)
        inherited = read_json(root / previous / "config.json") if previous else {}
        common = (
            "weights_root",
            "backbone", "teacher_checkpoint",
            "a1_batch_size", "a2_batch_size", "b_batch_size", "a1_steps", "a2_steps", "b_steps",
            "resolution", "flow_shift", "shift_base_size",
            "index_root",
            "text_cache_root",
            "subset_root",
            "gallery_root",
            "region_size",
            "local_mode",
            "local_halo",
            "local_shift",
            "local_chunk",
            "memory_mode",
            "remote_rope",
            "seed",
            "workers",
            "threads",
            "preset",
            "swanlab_mode",
            "pit_chunk_size",
            "lora_rank",
            "lora_alpha",
            "activation_checkpointing",
        )
        saved = {key: inherited[key] for key in common if key in inherited}
    arguments = []
    for name in VDNConfig.__dataclass_fields__:
        if name not in saved or name in ("stage", "output_root", "resume", "max_seconds"):
            continue
        value = saved[name]
        if name == "activation_checkpointing":
            if not value:
                arguments.append("--no-activation-checkpointing")
        elif name == "remote_rope":
            arguments.append("--remote-rope" if value else "--no-remote-rope")
        elif name in ("allow_batch_change", "allow_schedule_change"):
            if value:
                arguments.append("--" + name.replace("_", "-"))
        else:
            arguments.extend(("--" + name.replace("_", "-"), str(value)))
    return [
        "bash",
        str(request.get("driver", REPO / "scripts/train_pid_vdn.sh")),
        *arguments,
        *request["training_args"],
        "--stage",
        stage,
        "--output-root",
        str(root),
        "--resume",
        "auto",
        "--max-seconds",
        str(math.ceil(request["cycle_seconds"])),
    ]


def progress_marker(stage_dir):
    latest = read_json(stage_dir / "checkpoints/latest.json")
    checkpoint = Path(latest["checkpoint"]) if latest else None
    saved = read_json(checkpoint / "complete.json") if checkpoint else {}
    # Publishing another generation of the same step is not training progress.
    return saved.get("step", -1), len(list((stage_dir / "galleries").glob("**/*.done.json")))


def event(root, kind, **values):
    value = dict(event=kind, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), controller_pid=os.getpid(), **values)
    with (root / "repeat/events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_cycle(root, stage, request, cycle, command=None, interrupted=None):
    root = Path(root)
    control = root / "repeat"
    control.mkdir(parents=True, exist_ok=True)
    stage_dir = root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    stop = stage_dir / "STOP"
    if stop.exists():
        return dict(reason="manual-stop", returncode=0, forced=False)
    token = f"{OWN_STOP}{os.getpid()}:{time.time_ns()}"
    child_log = control / f"cycle_{cycle:06d}_{stage}_{time.time_ns()}.log"
    latest = control / "latest-training.log"
    temporary = latest.with_name(f".latest-{os.getpid()}")
    temporary.symlink_to(child_log.name)
    os.replace(temporary, latest)
    before = progress_marker(stage_dir)
    enable_subreaper()
    excluded = {}
    remember_children(os.getpid(), excluded, include_parent=False)
    known = {}
    forced, reason, requested_at = False, "", None
    started = time.monotonic()
    with child_log.open("a") as output:
        process = subprocess.Popen(
            command or training_command(root, stage, request),
            cwd=REPO,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        identity = process_identity(process.pid)
        write_json(
            control / "state.json",
            dict(
                status="running",
                stage=stage,
                cycle=cycle,
                controller_pid=os.getpid(),
                training_pid=process.pid,
                training_start_ticks=identity[0] if identity else None,
                log=str(child_log),
            ),
        )
        event(root, "cycle-start", stage=stage, cycle=cycle, training_pid=process.pid, log=str(child_log))
        try:
            while process.poll() is None:
                remember_children(os.getpid(), known, excluded, include_parent=False)
                now = time.monotonic()
                manual = stop.exists() and stop.read_text() != token
                user_stop = (control / "STOP").exists() or (interrupted and interrupted())
                if requested_at is None and (manual or user_stop or now - started >= request["cycle_seconds"]):
                    reason = "manual-stop" if manual or user_stop else "deadline"
                    requested_at = now
                    try:
                        with stop.open("x") as stream:
                            stream.write(token)
                    except FileExistsError:
                        pass
                    event(root, "save-request", stage=stage, cycle=cycle, reason=reason)
                if requested_at is not None and now - requested_at >= request["grace_seconds"]:
                    forced = True
                    event(root, "hard-deadline", stage=stage, cycle=cycle)
                    kill_remaining(known, excluded)
                    break
                time.sleep(min(1.0, max(0.02, request["cycle_seconds"] / 10)))
            code = process.wait(timeout=10)
        finally:
            kill_remaining(known, excluded)
            if process.poll() is None:
                process.kill()
                process.wait()
            reap_adopted(known, process.pid)
            if stop.exists() and stop.read_text() == token:
                stop.unlink()
    text = child_log.read_text(errors="replace")[-20000:]
    fatal = any(marker in text for marker in ("CUDA out of memory", "torch.OutOfMemoryError"))
    killed = not fatal and (
        code in (-9, -15, 137, 143)
        or any(
            marker in text for marker in ("Signal 9 (SIGKILL)", "Signal 15 (SIGTERM)", "exitcode: -9", "exitcode: -15")
        )
    )
    result = dict(
        reason=reason or ("external-kill" if killed else "exit"),
        returncode=code,
        forced=forced,
        killed=killed,
        fatal=fatal,
        progressed=progress_marker(stage_dir) != before,
    )
    event(root, "cycle-end", stage=stage, cycle=cycle, **result)
    return result


def spawn_worker(root, fd, env=None):
    with (root / "repeat/controller.log").open("a") as output:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.repeat_pid_vdn",
                "--worker",
                "--output-root",
                str(root),
                "--lease-fd",
                str(fd),
            ],
            cwd=REPO,
            pass_fds=(fd,),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )


def launch(root, request, env=None):
    root = Path(root).resolve()
    control = root / "repeat"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "controller.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"An automatic controller is already running for {root}") from None
        previous = read_json(control / "state.json")
        identity = process_identity(previous.get("training_pid", 0))
        if identity and identity[0] == previous.get("training_start_ticks"):
            raise RuntimeError("Previous training is still alive; do not launch a second copy")
        stage = select_stage(root, request["stage"])
        if stage is None:
            print("Requested stage(s) already completed", flush=True)
            return None
        stop = root / stage / "STOP"
        if stop.exists():
            if stop.read_text().startswith(OWN_STOP):
                stop.unlink()
            else:
                raise RuntimeError(f"Remove the manual stop marker before starting: {stop}")
        (control / "STOP").unlink(missing_ok=True)
        request = {**request, "cycle": 0, "stalled_cycles": 0}
        write_json(control / "request.json", request)
        process = spawn_worker(root, lease.fileno(), env)
        print(f"Started automatic {request['stage']} resume; controller PID {process.pid}", flush=True)
        print(f"Training log: {control / 'latest-training.log'}", flush=True)
        print(f"Controller log: {control / 'controller.log'}", flush=True)
        return process.pid


def worker(root, fd):
    root = Path(root).resolve()
    control = root / "repeat"
    interrupted = {"value": False}
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: interrupted.update(value=True))
    with os.fdopen(fd, "a") as lease:
        request = read_json(control / "request.json")
        if (control / "STOP").exists():
            write_json(control / "state.json", dict(status="stopped", controller_pid=os.getpid()))
            return
        stage = select_stage(root, request["stage"])
        if stage is None:
            write_json(control / "state.json", dict(status="completed", controller_pid=os.getpid()))
            return
        cycle = request["cycle"] + 1
        result = run_cycle(root, stage, request, cycle, interrupted=lambda: interrupted["value"])
        stopped = result["reason"] == "manual-stop" or interrupted["value"] or (control / "STOP").exists()
        completed = select_stage(root, request["stage"]) is None
        errors = result.get("fatal", False) or (
            result["returncode"] != 0 and not (result["forced"] or result["killed"])
        )
        stalled = 0 if result.get("progressed") else request["stalled_cycles"] + 1
        limited = request["max_cycles"] and cycle >= request["max_cycles"]
        status = (
            "stopped"
            if stopped
            else "failed"
            if errors
            else "completed"
            if completed
            else "failed"
            if stalled >= 3
            else "paused"
            if limited
            else "restarting"
        )
        write_json(
            control / "state.json", dict(status=status, stage=stage, cycle=cycle, controller_pid=os.getpid(), **result)
        )
        event(root, status, stage=stage, cycle=cycle)
        if status != "restarting":
            return
        request.update(cycle=cycle, stalled_cycles=stalled)
        write_json(control / "request.json", request)
        end = time.monotonic() + request["restart_delay"]
        while time.monotonic() < end:
            if interrupted["value"] or (control / "STOP").exists():
                write_json(control / "state.json", dict(status="stopped", stage=stage, cycle=cycle))
                return
            time.sleep(min(0.1, end - time.monotonic()))
        # Passing the open lock description keeps ownership across controller renewal.
        # No exec(): exec preserves PID age and would not reset a process-age limit.
        successor = spawn_worker(root, lease.fileno())
        event(root, "controller-handoff", successor_pid=successor.pid, cycle=cycle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=(*STAGES, "all"), default="a2")
    p.add_argument("--output-root", default=os.getenv("PID_VDN_RUN", "outputs/pid-vdn"))
    p.add_argument("--cycle-seconds", type=float, default=1200)
    p.add_argument("--grace-seconds", type=float, default=180)
    p.add_argument("--restart-delay", type=float, default=5)
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--stop", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args, training_args = p.parse_known_args()
    root = Path(args.output_root).resolve()
    if args.worker:
        worker(root, args.lease_fd)
        return
    if args.stop:
        (root / "repeat").mkdir(parents=True, exist_ok=True)
        (root / "repeat/STOP").touch()
        print("Stop requested; save at an update/image boundary, then stop restarting")
        return
    if args.status:
        print(json.dumps(read_json(root / "repeat/state.json"), indent=2, ensure_ascii=False))
        return
    if args.cycle_seconds <= 0 or args.grace_seconds <= 0 or args.restart_delay < 0 or args.max_cycles < 0:
        p.error("Invalid cycle/grace/restart limits")
    launch(
        root,
        dict(
            stage=args.stage,
            cycle_seconds=args.cycle_seconds,
            grace_seconds=args.grace_seconds,
            restart_delay=args.restart_delay,
            max_cycles=args.max_cycles,
            training_args=training_args,
        ),
    )


if __name__ == "__main__":
    main()
