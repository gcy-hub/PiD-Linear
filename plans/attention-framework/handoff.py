"""Project handoff; scheduling state and idempotency belong to cron and durable."""
import datetime as dt
import json
import os
import re
from pathlib import Path
import subprocess
import sys


def durable(config, *args):
    env = os.environ.copy()
    env.pop("DSA_SESSION", None)
    result = subprocess.run([config["durable"], *args, "--json"], env=env,
                            cwd=config["repo"], capture_output=True, text=True)
    if result.returncode not in (0, 75):
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def terminal(record):
    return record["state"] in ("sealed", "pruned", "rejected")


def failed(record):
    if not terminal(record):
        return False
    return record["state"] == "rejected" or record.get("status", record.get("pruned", {}).get("status")) not in ("done", "ok")


def handoff(config, now, cron_running=False, invoke=durable):
    implementation = invoke(config, "describe", "--key", config["implementation_request"])
    if failed(implementation):
        return "stopped: implementation failed"
    # Re-submit existing exact content to wake crash recovery, never create implementation here.
    if implementation["state"] in ("pending", "running", "asking"):
        invoke(config, "run", "--request", config["implementation_request"], "--spec", config["implementation_spec"])
    if now < dt.datetime.fromisoformat(config["due"]):
        return "before due"
    acceptance = invoke(config, "describe", "--key", config["acceptance_request"])
    if terminal(acceptance):
        return "stopped: acceptance " + acceptance["state"]
    if cron_running and acceptance["state"] == "absent":
        return "cron owns handoff"
    invoke(config, "run", "--request", config["acceptance_request"], "--spec", config["acceptance_spec"])
    return "handed off: " + config["acceptance_request"]


def cron_active(config):
    lock = Path(config["repo"]) / ".agents/cron/run.lock"
    if not lock.exists():
        return False
    try:
        os.kill(json.loads(lock.read_text())["pid"], 0)
        return True
    except (ProcessLookupError, KeyError, ValueError):
        return False


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    # Cron's own task invokes --handoff while holding its project run lock.
    owned = "--handoff" not in sys.argv and cron_active(config)
    if not owned and "--handoff" not in sys.argv:
        registry = json.loads(Path(config["registry"]).read_text())
        task = next(t for t in registry["tasks"] if t["id"] == config["task_id"])
        if task.get("lastStatus") == "error":
            runs = Path(config["repo"]) / ".agents/cron/runs"
            matching = [json.loads(p.read_text()) for p in runs.glob("*.json")]
            matching = [r for r in matching if r["taskId"] == config["task_id"]]
            latest = max(matching, key=lambda r: r["startedAt"], default={})
            if re.search(r"unknown.model|model.unavailable|model.not.found|tool.*(unknown|unavailable|reject)|unknown.tool", latest.get("error", ""), re.I):
                print("stopped: explicit model/tool rejection; inspect agents-cron runs " + config["task_id"])
                return
    print(handoff(config, dt.datetime.now(dt.timezone.utc), owned))


if __name__ == "__main__":
    main()
