import json
import socket
from types import SimpleNamespace

from scripts import watch_pid_text_cache as watch


def config():
    return {
        "hostname": socket.gethostname(),
        "gpu_ids": "0,1,2,3",
        "min_free_mib": 12000,
        "repo": "/repository",
        "python": "/conda/bin/python",
        "threads": 1,
        "dataset_root": "/dataset",
        "weights_root": "/weights",
        "batch_size": 8,
        "workers": 2,
        "max_seconds": 1680,
    }


def test_watcher_resumes_killed_launcher_and_never_duplicates_live_process(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(watch, "gpu_capacity", lambda *args: (True, {"0": 20000}))
    monkeypatch.setattr(watch, "process_identity", lambda pid: "new" if pid == 123 else None)

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(watch.subprocess, "Popen", launch)
    watch.write_json(tmp_path / "run.json", {"pid": 99, "process_identity": "dead", "launch_count": 1})
    watch.check_once(tmp_path, config())
    assert len(calls) == 1
    assert calls[0][1]["env"]["GPU_IDS"] == "0,1,2,3"
    assert calls[0][1]["start_new_session"]
    assert "--resume" in calls[0][0]
    watch.check_once(tmp_path, config())
    assert len(calls) == 1
    assert json.loads((tmp_path / "watch.status.json").read_text())["state"] == "running"


def test_watcher_waits_for_memory_and_removes_only_its_own_cron_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "gpu_capacity", lambda *args: (False, {"0": 500}))
    watch.check_once(tmp_path, config())
    assert json.loads((tmp_path / "watch.status.json").read_text())["state"] == "waiting_for_gpu_memory"
    assert not (tmp_path / "run.json").exists()
    own = f"*/2 * * * * own-command {watch.cron_marker(tmp_path)}"
    other = "0 * * * * unrelated-command # unrelated"
    invocations = []

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=other + "\n" + own + "\n", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", run)
    watch.update_cron(tmp_path)
    assert invocations[1][1]["input"] == other + "\n"
    removed = []
    monkeypatch.setattr(watch, "update_cron", lambda root: removed.append(root))
    (tmp_path / "cache.json").write_text('{"count": 20}')
    watch.check_once(tmp_path, config())
    assert removed == [tmp_path]
    assert json.loads((tmp_path / "watch.status.json").read_text())["state"] == "complete"
