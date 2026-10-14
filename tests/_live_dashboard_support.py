"""Plain helpers for the live dashboard tests: not a test module, loaded by path.

``FakeClock`` gives deterministic aware-UTC timestamps; ``VolumeLikeStore`` models Modal
Volume visibility (writes become visible to readers only after the writer commits and
the reader refreshes) and asserts the publication ordering on every write and commit;
``publish_task`` writes a hand-built task record for reader-side scenarios.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from live_dashboard import publication as pub
from live_dashboard.storage import DirectoryStore

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
GOLDEN_ATTEMPT = f"{EPOCH:%Y%m%dT%H%M%SZ}-00000000"
WORKER = pub.WorkerInfo(platform="test", cpu_count=1, gpu_requested=None, gpu_name=None)


class FakeClock:
    """Monotonic, thread-safe clock: each call returns the current time, then steps."""

    def __init__(self, start: datetime = EPOCH, step: float = 1.0) -> None:
        self.now = start
        self.step = timedelta(seconds=step)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            current = self.now
            self.now = current + self.step
            return current

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += timedelta(seconds=seconds)


def _writable_copy(source: Path, target: Path) -> None:
    """Copy a tree without its permission bits (sources may be read-only goldens)."""
    shutil.copytree(source, target, copy_function=shutil.copyfile)
    for directory in [target, *(path for path in target.rglob("*") if path.is_dir())]:
        directory.chmod(0o755)


def _mirror(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    _writable_copy(source, target)


def clock_after(moment: datetime, hours: float = 1.0) -> FakeClock:
    """A clock starting well after ``moment``, so later attempts sort after earlier ones."""
    return FakeClock(start=moment + timedelta(hours=hours))


class VolumeLikeStore(DirectoryStore):
    """Writer view of a Volume: a private working directory published by ``commit()``.

    Writing a task record asserts that its payload was already committed with the
    recorded size and digest; every commit asserts that each committed task record has
    a committed run manifest. ``reader()`` returns an independent view that changes only
    when that view is refreshed (like ``Volume.reload``).
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root / "working", on_commit=self._publish)
        self.root.mkdir(parents=True, exist_ok=True)
        self.committed = root / "committed"
        self.committed.mkdir(parents=True, exist_ok=True)
        self.commits = 0

    def write_new(self, path: str, data: bytes) -> None:
        parts = path.split("/")
        if len(parts) == 5 and parts[0] == pub.RUNS_DIR and parts[2] == "tasks":
            manifest = self.committed / pub.run_manifest_path(parts[1])
            assert manifest.is_file(), f"{path} written before run.json was committed"
            payload = json.loads(data)["payload"]
            if payload is not None:
                committed = self.committed / payload["path"]
                assert committed.is_file(), f"{path} written before its payload was committed"
                assert committed.stat().st_size == payload["bytes"]
                assert pub.sha256_hex(committed.read_bytes()) == payload["sha256"]
        super().write_new(path, data)

    def _publish(self) -> None:
        _mirror(self.root, self.committed)
        for record in self.committed.glob("runs/*/tasks/*/*.json"):
            run_dir = record.parents[2]
            assert (run_dir / "run.json").is_file(), f"{record} committed without run.json"
            payload = json.loads(record.read_bytes())["payload"]
            if payload is not None:
                assert (self.committed / payload["path"]).is_file()
        self.commits += 1

    def reader(self, root: Path) -> DirectoryStore:
        root.mkdir(parents=True, exist_ok=True)
        return DirectoryStore(root, on_refresh=lambda: _mirror(self.committed, root))


def make_read_only(root: Path) -> None:
    """Freeze a shared golden tree so a test that forgets to copy it fails loudly."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def copy_store(source: Path, target: Path) -> DirectoryStore:
    """A writable copy of a store tree (golden trees are read-only and never mutated)."""
    _writable_copy(source, target)
    return DirectoryStore(target)


def task_record_files(root: Path, run_id: str, task_id: str) -> list[Path]:
    return sorted((root / pub.task_dir(run_id, task_id)).glob("*.json"))


def rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Load a JSON record, let ``mutate`` edit it in place, and write it back."""
    body = json.loads(path.read_bytes())
    mutate(body)
    path.write_text(json.dumps(body, indent=2, sort_keys=True))


def only_run_id(root: Path) -> str:
    (run_id,) = sorted(path.name for path in (root / pub.RUNS_DIR).iterdir())
    return run_id


def load_manifest(store: DirectoryStore, run_id: str) -> pub.RunManifest:
    return pub.decode_run_manifest(store.read_bytes(pub.run_manifest_path(run_id)), run_id=run_id)


def publish_task(
    store: DirectoryStore,
    manifest: pub.RunManifest,
    task_index: int,
    *,
    attempt_id: str,
    payload: bytes | None,
    accepted: int,
    started_at: datetime = EPOCH,
    seconds: float = 2.0,
    accounting: pub.Accounting | None = None,
) -> pub.TaskRecord:
    """Write ``payload`` (if any) and a matching task record, as an attempt would."""
    run_id = manifest.run_id
    task_id = pub.task_id_for(task_index)
    reference = None
    if payload is not None:
        path = pub.payload_path(run_id, task_id, attempt_id)
        store.write_new(path, payload)
        store.commit()
        reference = pub.PayloadRef(
            path=path, sha256=pub.sha256_hex(payload), size=len(payload), corpus_schema_version=5
        )
    record = pub.TaskRecord(
        run_id=run_id,
        task_id=task_id,
        task_index=task_index,
        seed=manifest.request.task_seed(task_index),
        target_worlds=manifest.worlds_per_task,
        accepted_worlds=accepted,
        world_ids=tuple(pub.world_id_for(run_id, task_id, row) for row in range(accepted)),
        payload=reference,
        accounting=accounting or pub.Accounting(accepted, 0, 0, None, None),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=seconds),
        generation_seconds=seconds,
        attempt_id=attempt_id,
        worker=WORKER,
    )
    store.write_new(pub.task_record_path(run_id, task_id, attempt_id), record.encode())
    store.commit()
    return record
