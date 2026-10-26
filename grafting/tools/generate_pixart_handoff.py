"""Shared-filesystem handoff from the node01 generator to a scheduled job."""
from contextlib import contextmanager
import fcntl
import json
import os
import time
import uuid

from generate_pixart_dataset import atomic_json


@contextmanager
def local_start_gate(root):
    control = root / 'slurm'
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'takeover.lock').open('a') as gate:
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The Slurm job is taking over or generating; local restart is disabled.') from None
        if (control / 'takeover.json').exists():
            raise RuntimeError('A Slurm takeover marker exists; inspect its job before restarting locally.')
        yield


def wait_for_lock(stream, deadline, interrupted, stop_file=None):
    while True:
        if stop_file is not None and not stop_file.exists():
            stop_file.touch()
        if interrupted and interrupted():
            raise RuntimeError('Cluster handoff interrupted')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Local generator has not released its lock; cluster workers were not started.')
            time.sleep(.05)


def wait_for_local_stop(root, deadline, interrupted):
    # Some network-filesystem lock configurations are node-local. Require
    # the local controller's explicit acknowledgement as well as file locks.
    state_path = root / 'repeat/state.json'
    while state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('status') not in ('running', 'restarting'):
            return
        if interrupted and interrupted():
            raise RuntimeError('Cluster handoff interrupted')
        (root / '.stop-requested').touch()
        if time.monotonic() >= deadline:
            raise TimeoutError('node01 has not confirmed that its batch is saved and generation stopped.')
        time.sleep(.1)


@contextmanager
def claim_cluster_output(root, job_id, timeout=300, interrupted=None):
    control = root / 'slurm'
    local = root / 'repeat'
    control.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    marker = control / 'takeover.json'
    token = uuid.uuid4().hex
    with (control / 'takeover.lock').open('a') as gate:
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Slurm coordinator already owns this dataset.') from None
        try:
            atomic_json(marker, dict(job_id=str(job_id), token=token, coordinator_pid=os.getpid(),
                                     requested_at=time.time()))
            # The controller stops renewing; workers finish their current batch.
            (local / 'STOP').touch()
            (root / '.stop-requested').touch()
            deadline = time.monotonic() + timeout
            with (local / 'controller.lock').open('a') as lease:
                wait_for_lock(lease, deadline, interrupted, root / '.stop-requested')
                # An orphaned worker may still hold the output lease even after
                # its controller died. Check the writer lock as well.
                with (root / '.generation.lock').open('a') as writer:
                    wait_for_lock(writer, deadline, interrupted, root / '.stop-requested')
                wait_for_local_stop(root, deadline, interrupted)
                # Keep the takeover gate and local-controller lease for the
                # entire cluster run. The coordinator now prepares and acquires
                # its own output lease before dispatching any GPU workers.
                yield
        finally:
            if marker.exists() and json.loads(marker.read_text()).get('token') == token:
                marker.unlink()
