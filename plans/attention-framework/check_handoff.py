"""Bounded project handoff checks; no scheduler, provider or GPU execution."""
import datetime as dt
from handoff import handoff

config = dict(due="2026-11-11T03:30:00+08:00", implementation_request="implementation",
              acceptance_request="acceptance", implementation_spec="implementation.json",
              acceptance_spec="acceptance.json")
due = dt.datetime.fromisoformat(config["due"])
records = {"implementation": {"state": "absent"}, "acceptance": {"state": "absent"}}
launches = []
calls = []


def invoke(config, verb, *args):
    calls.append((verb, args))
    key = args[1]
    if verb == "describe":
        return records[key]
    if records[key]["state"] == "absent":
        launches.append(key)
    records[key] = {"state": "running", "wid": "saved-workflow"}
    return records[key]


assert handoff(config, due - dt.timedelta(seconds=1), invoke=invoke) == "before due"
assert not launches
assert handoff(config, due, cron_running=True, invoke=invoke) == "cron owns handoff"
assert not launches
assert handoff(config, due, invoke=invoke).startswith("handed off")
assert launches == ["acceptance"]
# The submission persisted but the caller was interrupted before retaining its reply.
assert handoff(config, due, invoke=invoke).startswith("handed off")
assert launches == ["acceptance"]
records["acceptance"] = {"state": "sealed", "status": "done"}
count = len([c for c in calls if c[0] == "run"])
assert handoff(config, due, invoke=invoke) == "stopped: acceptance sealed"
assert len([c for c in calls if c[0] == "run"]) == count
records["implementation"] = {"state": "sealed", "status": "failed"}
assert handoff(config, due, invoke=invoke) == "stopped: implementation failed"
assert launches == ["acceptance"]
print("PASS: before due, cron ownership, due handoff, interrupted reply/no duplicate, completed/no new run, explicit failure")
