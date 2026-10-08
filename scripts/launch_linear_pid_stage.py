#!/usr/bin/env python3
"""Initialize KDA from original PiD or a parent stage, then resume its own checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_linear_pid_resume import completed, snapshot_resume


def read_stage_config(path, preset, overrides=None):
    spec = json.loads(Path(path).read_text())
    values = dict(spec["training"])
    values.update(spec["presets"][preset])
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    layers = values["layers"]
    if (not layers or any(type(i) is not int or not 0 <= i < 14 for i in layers)
            or layers != sorted(set(layers))):
        raise ValueError("Stage layers must be unique, sorted, zero-based MMDiT indices")
    if any(key in values for key in ("resume", "resume_from", "init_from", "allow_batch_size_change", "init_weights")):
        raise ValueError("The stage launcher owns initialization/resume; do not put these modes in the JSON")
    initialization = spec.get("initialization", "parent_checkpoint")
    if initialization == "original_pid":
        if spec.get("source_run_dir"):
            raise ValueError("Original-PiD initialization must not specify a parent run")
        source = None
    elif initialization == "parent_checkpoint":
        source = spec.get("source_run_dir")
        if not source:
            raise ValueError("Parent-checkpoint initialization requires source_run_dir")
    else:
        raise ValueError(f"Unknown initialization mode: {initialization!r}")
    return source, values


def stage_command(values, preset, source_run_dir, *, prepare=False):
    run_dir = Path(values["output_root"]) / ("kda_" + "-".join(map(str, values["layers"])))
    source = Path(source_run_dir) if source_run_dir else None
    if source is not None and (run_dir.resolve() == source.resolve() or source.resolve() in run_dir.resolve().parents
            or run_dir.resolve() in source.resolve().parents):
        raise ValueError("Expansion output must be separate from the active source run")
    own = completed(run_dir)
    if own:
        mode, initialization = "resume", ["--resume", "auto"]
        checkpoint = own[-1]
    elif source is None:
        mode, checkpoint = "initialize_original", None
        initialization = ["--resume", "none"]
    else:
        seeds = completed(Path(values["output_root"]) / "initial_state")
        candidates = seeds or completed(source)
        if not candidates:
            raise FileNotFoundError(f"No complete parent checkpoints in {source}")
        checkpoint = candidates[-1]
        if prepare:
            checkpoint = snapshot_resume(source, values["output_root"])
        mode = "expand"
        initialization = ["--resume", "none", "--init-from", str(checkpoint), "--init-weights", "raw"]
    command = ["bash", str(REPO / "scripts/train_linear_pid.sh"), "--preset", preset]
    for key, value in values.items():
        if key in {"local_mixing", "activation_checkpointing"}:
            if type(value) is not bool:
                raise ValueError(f"Expected a boolean for {key}")
            if not value:
                command.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (dict, bool)):
            raise ValueError(f"Unsupported stage value: {key}={value!r}")
        if key == "layers":
            value = ",".join(map(str, value))
        command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(initialization)
    return command, {"mode": mode, "checkpoint": str(checkpoint) if checkpoint else None,
                     "run_dir": str(run_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-config", default=str(REPO / "pid/_src/configs/linear_pid/kda10.json"))
    parser.add_argument("--preset", choices=("4gpu", "8gpu"), default="4gpu")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--gpu-ids", help="Override GPU_IDS for a direct launch; Slurm uses its allocation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dryrun", action="store_true", help="Print the full plan without copying or starting training")
    mode.add_argument("--prepare-only", action="store_true", help="Pin the initial parent checkpoint without starting training")
    for name in ("batch-size", "grad-accum", "effective-batch", "workers", "threads", "warmup",
                 "pit-chunk-size", "max-steps", "max-seconds", "save-steps", "save-epochs",
                 "validation-steps", "validation-epochs", "log-steps"):
        parser.add_argument("--" + name, type=int)
    for name in ("lr-new", "lr-backbone"):
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args(argv)
    reserved = {"stage_config", "preset", "source_run_dir", "gpu_ids", "dryrun", "prepare_only"}
    source, values = read_stage_config(args.stage_config, args.preset,
                                      {key: value for key, value in vars(args).items() if key not in reserved})
    if source is None and args.source_run_dir:
        parser.error("Original-PiD initialization cannot override a parent run")
    source = args.source_run_dir or source
    # Check every generated argument before potentially copying a large parent.
    command, plan = stage_command(values, args.preset, source)
    from pid._src.linear_pid.training import parse_args, preflight
    config, _ = parse_args(command[2:])
    preflight(config)
    if args.dryrun:
        print(json.dumps(plan, indent=2))
        print(shlex.join(command))
        return
    command, plan = stage_command(values, args.preset, source, prepare=True)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    env = os.environ.copy()
    if args.gpu_ids:
        env["GPU_IDS"] = args.gpu_ids
    env.setdefault("CPU_THREADS", str(values["threads"]))
    env.setdefault("MKL_NUM_THREADS", str(values["threads"]))
    env.setdefault("OPENBLAS_NUM_THREADS", str(values["threads"]))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    print(json.dumps(plan, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    os.chdir(REPO)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
