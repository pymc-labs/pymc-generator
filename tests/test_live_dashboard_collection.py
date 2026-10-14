"""Collector: what the dashboard shows for any state of the publication store.

Stores are written with the contract's own records (``publish_task`` commits payloads
before records) and the real corpora of the session's golden run; damaged or
out-of-protocol states are produced by editing the files a writer left behind. Every
collector reads a frozen clock, so results never depend on how often it reads the time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from live_dashboard import publication as pub
from live_dashboard.collection import Collector
from live_dashboard.extract import extract_payload
from live_dashboard.snapshot import CollectionSnapshot, RunSnapshot
from live_dashboard.storage import DirectoryStore


def _load(name: str) -> ModuleType:
    """Load a plain helper module of ``tests/`` by path, once per session."""
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")

EPOCH = support.EPOCH
GENERATOR = pub.GeneratorInfo(
    preset="small",
    config={"n_cells": 2, "draws_per_cell": 1},
    pymc_generator_version="0.0.2",
    corpus_schema_version=5,
    code_sha256="ab" * 32,
    code_revision="0123abcd",
    environment={"python": "3.13.9", "numpy": "2.4.6"},
)


@pytest.fixture(autouse=True)
def _dashboard_environment(monkeypatch):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def payloads(live_golden) -> tuple[bytes, ...]:
    """The golden run's payload bytes in task order: three distinct two-world corpora."""
    store = DirectoryStore(live_golden.root)
    run_id = live_golden.request.run_id
    result = tuple(
        store.read_bytes(pub.payload_path(run_id, pub.task_id_for(index), live_golden.attempt_id))
        for index in range(live_golden.request.n_tasks)
    )
    assert len(set(result)) == len(result) == 3
    return result


# -- helpers --------------------------------------------------------------------------


def _attempt_id(n: int) -> str:
    """Deterministic attempt ids whose sort order is ``n``'s."""
    return f"20260101T{n:06d}Z-{n:08x}"


def _clock(seconds: float = 0.0) -> Any:
    """A clock frozen at ``EPOCH + seconds`` (tests move it with ``advance``)."""
    return support.FakeClock(start=EPOCH + timedelta(seconds=seconds), step=0)


def _manifest(
    store: DirectoryStore,
    *,
    label: str = "demo",
    worlds: int = 4,
    worlds_per_task: int = 2,
    created_at: datetime = EPOCH,
) -> pub.RunManifest:
    request = pub.RunRequest(label, "small", worlds, worlds_per_task, 1)
    manifest = pub.RunManifest(
        run_id=request.run_id, created_at=created_at, request=request, generator=GENERATOR
    )
    store.write_new(pub.run_manifest_path(manifest.run_id), manifest.encode())
    store.commit()
    return manifest


def _attempt(
    store: DirectoryStore,
    manifest: pub.RunManifest,
    n: int,
    state: pub.AttemptState,
    *,
    started: float = 0.0,
    deadline: float | None = None,
    commit: bool = True,
) -> None:
    """Write attempt ``n``; times are seconds after ``EPOCH``."""
    started_at = EPOCH + timedelta(seconds=started)
    record = pub.AttemptRecord(
        run_id=manifest.run_id,
        attempt_id=_attempt_id(n),
        state=state,
        started_at=started_at,
        updated_at=started_at,
        deadline_at=None if deadline is None else EPOCH + timedelta(seconds=deadline),
        current_task_index=None,
        published_task_ids=(),
        reused_task_ids=(),
        error=pub.ErrorInfo("RuntimeError", "worker crashed") if state == "failed" else None,
        worker=support.WORKER,
    )
    path = pub.attempt_record_path(manifest.run_id, record.attempt_id)
    store.write_replace(path, record.encode())
    if commit:
        store.commit()


def _publish(
    store: DirectoryStore,
    manifest: pub.RunManifest,
    index: int,
    n: int,
    payload: bytes | None = None,
    *,
    accepted: int | None = None,
    started_at: datetime = EPOCH,
) -> pub.TaskRecord:
    """Publish task ``index`` as attempt ``n`` (a zero-world record without payload)."""
    if accepted is None:
        accepted = 0 if payload is None else 2
    return support.publish_task(
        store,
        manifest,
        index,
        attempt_id=_attempt_id(n),
        payload=payload,
        accepted=accepted,
        started_at=started_at,
    )


def _record_file(root: Path, manifest: pub.RunManifest, index: int, n: int) -> Path:
    return root / pub.task_record_path(manifest.run_id, pub.task_id_for(index), _attempt_id(n))


def _edit(data: bytes, **changes: object) -> bytes:
    body = json.loads(data)
    body.update(changes)
    return json.dumps(body).encode()


def _flipped(data: bytes) -> bytes:
    middle = len(data) // 2
    return data[:middle] + bytes([data[middle] ^ 0xFF]) + data[middle + 1 :]


def _collector(store: DirectoryStore, clock: Callable[[], datetime] | None = None) -> Collector:
    collector = Collector(store, clock=clock or _clock())
    collector.refresh()
    return collector


def _run(collector: Collector) -> RunSnapshot:
    (run,) = collector.snapshot.runs
    return run


def _run_of(snapshot: CollectionSnapshot, manifest: pub.RunManifest) -> RunSnapshot:
    return snapshot.run_index[manifest.run_id]


def _statuses(collector: Collector) -> list[str]:
    return [task.status for task in _run(collector).tasks]


def _spy(calls: Counter[str], failing: bytes | None = None) -> Callable[[bytes], object]:
    """An extractor counting calls per digest; ``failing`` bytes raise an OSError."""

    def extractor(data: bytes):
        calls[pub.sha256_hex(data)] += 1
        if data == failing:
            raise OSError("no space left for the scratch file")
        return extract_payload(data)

    return extractor


class _ReadSpyStore(DirectoryStore):
    """Records every path whose bytes were read."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads: list[str] = []

    def read_bytes(self, path: str) -> bytes:
        self.reads.append(path)
        return super().read_bytes(path)


class _FaultyStore(DirectoryStore):
    """Raises an unexpected error for any size or file listing under ``fail_on``."""

    def __init__(self, root: Path, fail_on: str) -> None:
        super().__init__(root)
        self.fail_on = fail_on

    def _check(self, path: str) -> None:
        if self.fail_on in path:
            raise RuntimeError(f"disk fault under {self.fail_on}")

    def size(self, path: str) -> int | None:
        self._check(path)
        return super().size(path)

    def list_files(self, prefix: str) -> list[str]:
        self._check(prefix)
        return super().list_files(prefix)


# -- store-level behaviour --------------------------------------------------------------


def test_a_new_collector_has_not_refreshed_and_an_empty_store_has_no_runs(tmp_path):
    collector = Collector(DirectoryStore(tmp_path), clock=_clock())
    initial = collector.snapshot
    assert (initial.sequence, initial.refreshed_at, initial.last_error, initial.runs) == (
        0,
        None,
        None,
        (),
    )
    collector.refresh()
    snapshot = collector.snapshot
    assert (snapshot.sequence, snapshot.refreshed_at, snapshot.last_error) == (1, EPOCH, None)
    assert (snapshot.runs, snapshot.issues) == ((), ())


def test_the_golden_run_reads_back_as_the_writer_published_it(live_golden, payloads):
    store = DirectoryStore(live_golden.root)  # read-only: the collector writes nothing
    snapshot = _collector(store).snapshot
    assert (snapshot.last_error, snapshot.issues) == (None, ())
    (run,) = snapshot.runs
    request = live_golden.request
    assert (run.run_id, run.label, run.preset, run.seed) == (request.run_id, "golden", "small", 1)
    assert (run.state, run.issues, run.published_worlds, run.target_worlds) == (
        "complete",
        (),
        6,
        6,
    )
    assert [(attempt.attempt_id, attempt.status) for attempt in run.attempts] == [
        (live_golden.attempt_id, "finished")
    ]
    for task in run.tasks:
        record = pub.decode_task_record(
            store.read_bytes(pub.task_record_path(run.run_id, task.task_id, task.attempt_id)),
            manifest=support.load_manifest(store, run.run_id),
            task_id=task.task_id,
            attempt_id=live_golden.attempt_id,
        )
        accounting = record.accounting
        assert (task.status, task.duplicate_records, task.detail) == ("published", 0, None)
        assert (task.started_at, task.finished_at, task.generation_seconds) == (
            record.started_at,
            record.finished_at,
            record.generation_seconds,
        )
        assert (task.candidates_evaluated, task.rejected_candidates, task.draw_failures) == (
            accounting.candidates_evaluated,
            accounting.rejected_candidates,
            accounting.draw_failures,
        )
    # Precondition: rejections exist, so swapped accounting fields would show.
    assert any(task.rejected_candidates for task in run.tasks)
    for index, payload in enumerate(payloads):
        metrics = extract_payload(payload)
        task_id = pub.task_id_for(index)
        worlds = [world for world in run.worlds if world.task_id == task_id]
        assert [world.world_id for world in worlds] == [
            pub.world_id_for(run.run_id, task_id, row) for row in range(2)
        ]
        assert [world.channel_share for world in worlds] == list(metrics.channel)


@pytest.mark.parametrize(
    ("method", "error"),
    [("refresh", OSError("volume reload failed")), ("list_dirs", RuntimeError("listing broke"))],
)
def test_store_failures_set_last_error_and_keep_the_previous_runs(tmp_path, method, error):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    _publish(store, manifest, 0, 1)
    clock = _clock()
    collector = _collector(store, clock)
    before = collector.snapshot
    clock.advance(60)

    def fail(*args: object) -> None:
        raise error

    setattr(store, method, fail)
    collector.refresh()
    failed = collector.snapshot
    assert failed.last_error == f"{type(error).__name__}: {error}"
    assert (failed.sequence, failed.refreshed_at) == (2, EPOCH)
    assert (failed.runs, failed.issues) == (before.runs, before.issues)
    delattr(store, method)
    collector.refresh()
    recovered = collector.snapshot
    assert (recovered.sequence, recovered.refreshed_at, recovered.last_error) == (
        3,
        EPOCH + timedelta(seconds=60),
        None,
    )
    assert recovered.runs == before.runs


def test_run_directories_without_a_manifest_are_silent_and_bad_names_reported(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    unpublished = pub.RunRequest("later", "small", 4, 2, 1).run_id
    store.write_new(pub.task_record_path(unpublished, "task-00000", _attempt_id(1)), b"{}")
    store.write_new("runs/Not_A_Run/run.json", manifest.encode())
    snapshot = _collector(store).snapshot
    assert [run.run_id for run in snapshot.runs] == [manifest.run_id]
    assert snapshot.issues == ("runs/Not_A_Run: not a run id (ignored)",)


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (lambda data: _edit(data, contract_version="1.7", future_field={"x": 1}), None),
        (lambda data: _edit(data, contract_version="2.0"), "is not supported"),
        (lambda data: _edit(data, contract_version="one"), "invalid contract_version"),
        (lambda data: _edit(data, target_worlds=99), "target_worlds"),
        (lambda data: data[:40], "not a UTF-8 JSON document"),
    ],
    ids=["newer-minor", "newer-major", "bad-version", "inconsistent", "partial"],
)
def test_manifest_versions_and_damage(tmp_path, damage, problem):
    store = DirectoryStore(tmp_path)
    other = _manifest(store, label="other")
    manifest = _manifest(store)
    path = tmp_path / pub.run_manifest_path(manifest.run_id)
    path.write_bytes(damage(path.read_bytes()))
    snapshot = _collector(store).snapshot
    if problem is None:
        assert {run.run_id for run in snapshot.runs} == {manifest.run_id, other.run_id}
        assert snapshot.issues == ()
    else:
        assert [run.run_id for run in snapshot.runs] == [other.run_id]
        (issue,) = snapshot.issues
        assert issue.startswith(f"runs/{manifest.run_id}/run.json: ")
        assert problem in issue


def test_runs_are_listed_newest_first(tmp_path):
    store = DirectoryStore(tmp_path)
    old = _manifest(store, label="zzz-old", created_at=EPOCH)
    new = _manifest(store, label="aaa-new", created_at=EPOCH + timedelta(hours=1))
    middle = _manifest(store, label="mmm-mid", created_at=EPOCH + timedelta(minutes=1))
    runs = _collector(store).snapshot.runs
    assert [run.run_id for run in runs] == [new.run_id, middle.run_id, old.run_id]


def test_a_volume_reader_sees_each_commit_after_exactly_one_refresh(tmp_path, payloads):
    writer = support.VolumeLikeStore(tmp_path / "volume")
    manifest = _manifest(writer)
    collector = Collector(writer.reader(tmp_path / "reader"), clock=_clock())
    assert collector.snapshot.runs == ()
    collector.refresh()
    assert _statuses(collector) == ["pending", "pending"]

    _attempt(writer, manifest, 1, "running", commit=False)
    collector.refresh()
    assert _run(collector).attempts == ()  # written, not committed
    writer.commit()
    assert _run(collector).attempts == ()  # committed, not yet refreshed
    collector.refresh()
    assert [attempt.attempt_id for attempt in _run(collector).attempts] == [_attempt_id(1)]

    _publish(writer, manifest, 1, 1, payloads[0])
    assert _statuses(collector) == ["pending", "pending"]
    collector.refresh()
    assert _statuses(collector) == ["pending", "published"]
    assert _run(collector).published_worlds == 2


# -- task records -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("damage", "status"),
    [
        (lambda data: _edit(data, contract_version="1.9", future_field=[1]), "zero"),
        (lambda data: _edit(data, contract_version="2.0"), "unsupported"),
        (lambda data: _edit(data, contract_version="v1"), "invalid"),
        (lambda data: _edit(data, accepted_worlds=1), "invalid"),
        (lambda data: data[: len(data) // 2], "unreadable"),
    ],
    ids=["newer-minor", "newer-major", "bad-version", "inconsistent", "partial"],
)
def test_task_record_versions_and_damage(tmp_path, damage, status):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    _publish(store, manifest, 0, 1)
    path = _record_file(tmp_path, manifest, 0, 1)
    path.write_bytes(damage(path.read_bytes()))
    task = _run(_collector(store)).tasks[0]
    assert task.status == status
    if status == "zero":
        assert (task.attempt_id, task.detail) == (_attempt_id(1), None)
    else:
        assert task.attempt_id is None and task.started_at is None
        assert task.detail.startswith(f"{_attempt_id(1)}.json: ")


@pytest.mark.parametrize(
    ("damage", "visible"),
    [
        (lambda data: _edit(data, contract_version="1.9", future_field=[1]), True),
        (lambda data: _edit(data, contract_version="2.0"), False),
        (lambda data: _edit(data, state="paused"), False),
        (lambda data: data[: len(data) // 2], False),
    ],
    ids=["newer-minor", "newer-major", "unknown-state", "partial"],
)
def test_attempt_record_versions_and_damage(tmp_path, damage, visible):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    _attempt(store, manifest, 1, "running")
    path = tmp_path / pub.attempt_record_path(manifest.run_id, _attempt_id(1))
    path.write_bytes(damage(path.read_bytes()))
    run = _run(_collector(store))
    if visible:
        assert [attempt.status for attempt in run.attempts] == ["running"]
        assert (run.state, run.issues) == ("running", ())
    else:
        assert (run.attempts, run.state) == ((), "waiting")
        (issue,) = run.issues
        assert issue.startswith(f"attempts/{_attempt_id(1)}.json: ")


def test_a_partial_record_is_unreadable_until_it_is_complete(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=2)
    _attempt(store, manifest, 1, "running")
    _publish(store, manifest, 0, 1, payloads[0])
    path = _record_file(tmp_path, manifest, 0, 1)
    complete = path.read_bytes()
    path.write_bytes(complete[: len(complete) // 3])
    collector = _collector(store)
    assert _statuses(collector) == ["unreadable"]
    assert (_run(collector).state, _run(collector).worlds) == ("running", ())
    path.write_bytes(complete)
    collector.refresh()
    assert _statuses(collector) == ["published"]
    assert (_run(collector).state, len(_run(collector).worlds)) == ("complete", 2)


def test_the_strongest_problem_decides_and_ignored_siblings_are_counted(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=6)
    damages = {
        "unsupported": lambda data: _edit(data, contract_version="2.0"),
        "invalid": lambda data: _edit(data, accepted_worlds=1),
        "unreadable": lambda data: data[:10],
    }
    layout = {
        0: ["unreadable", "invalid", "unsupported"],
        1: ["unreadable", "invalid"],
        2: [None, "unreadable", "invalid"],  # a valid record with problem siblings
    }
    for index, kinds in layout.items():
        for n, kind in enumerate(kinds, start=1):
            _publish(store, manifest, index, n)
            if kind is not None:
                path = _record_file(tmp_path, manifest, index, n)
                path.write_bytes(damages[kind](path.read_bytes()))
    tasks = _run(_collector(store)).tasks
    assert [task.status for task in tasks] == ["unsupported", "invalid", "zero"]
    assert tasks[0].detail.startswith(f"{_attempt_id(3)}.json: contract_version 2.0")
    assert tasks[0].detail.endswith("problem records: 1 unsupported, 1 invalid, 1 unreadable")
    assert tasks[1].detail.startswith(f"{_attempt_id(2)}.json: ")
    assert tasks[1].detail.endswith("problem records: 1 invalid, 1 unreadable")
    assert tasks[2].attempt_id == _attempt_id(1)
    assert tasks[2].detail == "ignored problem records: 1 invalid, 1 unreadable"


def test_oversized_records_are_invalid_without_being_read(tmp_path):
    store = _ReadSpyStore(tmp_path)
    manifest = _manifest(store)
    _publish(store, manifest, 0, 1)
    path = _record_file(tmp_path, manifest, 0, 1)
    record = path.read_bytes()
    path.write_bytes(record[:-2] + b" " * pub.MAX_RECORD_BYTES + record[-2:])
    task = _run(_collector(store)).tasks[0]
    assert task.status == "invalid"
    assert f"exceeds {pub.MAX_RECORD_BYTES} bytes" in task.detail
    assert pub.task_record_path(manifest.run_id, "task-00000", _attempt_id(1)) not in store.reads


def test_names_and_paths_outside_the_layout_are_never_used(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    # task 0: a valid record under a name that is not its attempt id, and a stray file
    _publish(store, manifest, 0, 1)
    path = _record_file(tmp_path, manifest, 0, 1)
    path.rename(path.with_name("latest.json"))
    path.with_name("notes.txt").write_text("hello")
    # task 1: a record naming another attempt's payload
    _publish(store, manifest, 1, 1, payloads[0])
    original = _record_file(tmp_path, manifest, 1, 1)
    redirected = _edit(original.read_bytes(), attempt_id=_attempt_id(2))
    original.unlink()
    _record_file(tmp_path, manifest, 1, 2).write_bytes(redirected)
    tasks_dir = tmp_path / pub.tasks_dir(manifest.run_id)
    (tasks_dir / "task-00099").mkdir()
    (tasks_dir / "misc").mkdir()
    run = _run(_collector(store))
    assert [task.status for task in run.tasks] == ["invalid", "invalid"]
    assert run.tasks[0].detail.startswith("latest.json: invalid attempt_id 'latest'")
    assert run.tasks[0].detail.endswith("problem records: 2 invalid")
    redirected_to = pub.payload_path(manifest.run_id, "task-00001", _attempt_id(2))
    assert f"path must be '{redirected_to}'" in run.tasks[1].detail
    assert run.worlds == ()
    assert run.issues == (
        "tasks/misc: not a task of this run (ignored)",
        "tasks/task-00099: not a task of this run (ignored)",
    )


def test_staging_files_are_invisible(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    task_dir = tmp_path / pub.task_dir(manifest.run_id, "task-00000")
    task_dir.mkdir(parents=True)
    (task_dir / f".{_attempt_id(1)}.json.0123456789ab.partial").write_bytes(b'{"contract"')
    attempts_dir = tmp_path / pub.attempts_dir(manifest.run_id)
    attempts_dir.mkdir()
    (attempts_dir / f".{_attempt_id(1)}.json.0123456789ab.partial").write_bytes(b"{")
    (tmp_path / pub.RUNS_DIR / ".staging").mkdir()
    snapshot = _collector(store).snapshot
    run = _run_of(snapshot, manifest)
    assert [task.status for task in run.tasks] == ["pending", "pending"]
    assert (run.attempts, run.issues, snapshot.issues) == ((), (), ())
    assert run.state == "waiting"


# -- deduplication, conflicts and payloads ---------------------------------------------------


def test_duplicates_count_once_and_the_smallest_attempt_is_authoritative(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=2)
    later = EPOCH + timedelta(hours=1)
    _publish(store, manifest, 0, 5, payloads[0], started_at=later)  # arrives first
    _publish(store, manifest, 0, 3, payloads[0])
    run = _run(_collector(store))
    (task,) = run.tasks
    assert (task.status, task.duplicate_records) == ("published", 1)
    assert (task.attempt_id, task.started_at) == (_attempt_id(3), EPOCH)
    assert run.published_worlds == 2
    assert [world.world_id for world in run.worlds] == [
        f"{manifest.run_id}/task-00000/w0",
        f"{manifest.run_id}/task-00000/w1",
    ]


def test_conflicting_records_are_excluded_also_for_a_fresh_collector(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 1, 1)
    first = _collector(store)
    assert _statuses(first) == ["published", "zero"]
    _publish(store, manifest, 0, 2, payloads[1])
    _publish(store, manifest, 1, 2, payloads[0])
    first.refresh()
    fresh = _collector(store)
    for collector in (first, fresh):
        run = _run(collector)
        assert [task.status for task in run.tasks] == ["conflict", "conflict"]
        assert (run.worlds, run.impacts, run.published_worlds) == ((), (), 0)
        detail = run.tasks[0].detail
        for n, payload in ((1, payloads[0]), (2, payloads[1])):
            assert f"{_attempt_id(n)} (2 worlds, sha256 {pub.sha256_hex(payload)})" in detail
        assert f"{_attempt_id(1)} (0 worlds, no payload)" in run.tasks[1].detail
    assert fresh.snapshot.runs == first.snapshot.runs


def test_payload_verification_sequence_on_one_collector(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=2)
    good = payloads[0]
    record = _publish(store, manifest, 0, 1, good)
    payload = tmp_path / record.payload.path
    calls: Counter[str] = Counter()
    collector = Collector(store, clock=_clock(), extractor=_spy(calls))

    payload.unlink()
    collector.refresh()
    assert _statuses(collector) == ["unavailable"]
    payload.write_bytes(_flipped(good))  # same length, other digest
    collector.refresh()
    assert _statuses(collector) == ["unavailable"]
    payload.write_bytes(payloads[1])  # a valid corpus, but not the recorded one
    collector.refresh()
    assert _statuses(collector) == ["unavailable"]
    assert _run(collector).tasks[0].attempt_id == _attempt_id(1)  # timing still shown
    payload.write_bytes(good)
    collector.refresh()
    assert _statuses(collector) == ["published"]
    assert calls == Counter({record.payload.sha256: 1})  # only verified bytes are extracted

    # Documented limitation: once verified, an in-place change that keeps the size is
    # trusted by this collector until it restarts; a new collector checks the digest.
    payload.write_bytes(_flipped(good))
    collector.refresh()
    assert _statuses(collector) == ["published"]
    assert _statuses(_collector(store)) == ["unavailable"]
    # A size change drops the cached verification.
    payload.write_bytes(good + b"\0")
    collector.refresh()
    assert _statuses(collector) == ["unavailable"]


def test_a_missing_payload_falls_back_to_a_duplicate_record(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=2)
    first = _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 0, 2, payloads[0])
    (tmp_path / first.payload.path).unlink()
    (task,) = _run(_collector(store)).tasks
    assert (task.status, task.attempt_id, task.duplicate_records) == (
        "published",
        _attempt_id(1),
        1,
    )


def test_extraction_failures_stay_with_their_task(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=8)
    not_a_corpus = b"bytes that are not a corpus archive"
    _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 1, 1, not_a_corpus)  # deterministic content error
    _publish(store, manifest, 2, 1, payloads[1])  # the extractor fails for another reason
    _publish(store, manifest, 3, 1, payloads[2], accepted=1)  # holds 2 rows, records 1
    calls: Counter[str] = Counter()
    collector = Collector(store, clock=_clock(), extractor=_spy(calls, failing=payloads[1]))
    collector.refresh()
    collector.refresh()
    snapshot = collector.snapshot
    assert snapshot.last_error is None
    run = _run(collector)
    assert [task.status for task in run.tasks] == ["published", "invalid", "invalid", "invalid"]
    assert run.tasks[1].detail.startswith("payload cannot be used: payload is not a readable")
    assert run.tasks[2].detail == (
        "payload extraction failed: OSError: no space left for the scratch file"
    )
    assert run.tasks[3].detail == "payload holds 2 worlds, the record accepts 1"
    assert all(task.has_timing for task in run.tasks)
    assert {world.task_id for world in run.worlds} == {"task-00000"}
    assert run.state == "complete"
    # Content errors are cached by digest; the other failure is retried each refresh.
    assert calls == Counter(
        {
            pub.sha256_hex(payloads[0]): 1,
            pub.sha256_hex(not_a_corpus): 1,
            pub.sha256_hex(payloads[1]): 2,
            pub.sha256_hex(payloads[2]): 1,
        }
    )


def test_each_payload_is_extracted_once_and_identities_attached_per_task(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=6)
    _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 0, 2, payloads[0])
    _publish(store, manifest, 1, 1, payloads[0])
    _publish(store, manifest, 2, 1, payloads[1])
    calls: Counter[str] = Counter()
    collector = Collector(store, clock=_clock(), extractor=_spy(calls))
    for _ in range(3):
        collector.refresh()
    assert calls == Counter({pub.sha256_hex(payloads[0]): 1, pub.sha256_hex(payloads[1]): 1})
    run = _run(collector)
    assert _statuses(collector) == ["published"] * 3
    by_task = {}
    for world in run.worlds:
        by_task.setdefault(world.task_id, []).append(world)
    first, second = by_task["task-00000"], by_task["task-00001"]
    assert [world.world_id for world in second] == [
        f"{manifest.run_id}/task-00001/w0",
        f"{manifest.run_id}/task-00001/w1",
    ]
    assert [world.channel_share for world in second] == [world.channel_share for world in first]


def test_a_shortfall_task_publishes_its_accepted_worlds(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=4, worlds_per_task=4)
    _publish(store, manifest, 0, 1, payloads[0], accepted=2)
    run = _run(_collector(store))
    (task,) = run.tasks
    assert (task.status, task.accepted_worlds, task.target_worlds) == ("published", 2, 4)
    assert (run.published_worlds, run.target_worlds, run.state) == (2, 4, "complete")


def test_out_of_order_arrival_is_shown_in_task_and_row_order(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=6)
    _attempt(store, manifest, 1, "running", deadline=3600.0)
    collector = Collector(store, clock=_clock(60))

    _publish(store, manifest, 2, 1, payloads[2])
    record = _record_file(tmp_path, manifest, 2, 1)
    record.rename(tmp_path / "aside.json")  # only the payload has arrived so far
    collector.refresh()
    assert _statuses(collector) == ["pending", "pending", "pending"]
    assert _run(collector).state == "running"
    (tmp_path / "aside.json").rename(record)
    collector.refresh()
    assert _statuses(collector) == ["pending", "pending", "published"]
    _publish(store, manifest, 0, 1, payloads[0])
    collector.refresh()
    assert _statuses(collector) == ["published", "pending", "published"]
    _publish(store, manifest, 1, 1, payloads[1])
    collector.refresh()

    run = _run(collector)
    assert (run.state, run.published_worlds) == ("complete", 6)
    assert [world.world_id for world in run.worlds] == [
        f"{manifest.run_id}/task-{index:05d}/w{row}" for index in range(3) for row in range(2)
    ]
    for index, payload in enumerate(payloads):
        metrics = extract_payload(payload)
        task_id = pub.task_id_for(index)
        worlds = [world for world in run.worlds if world.task_id == task_id]
        assert [(w.channel_share, w.control_share, w.baseline_share) for w in worlds] == list(
            zip(metrics.channel, metrics.control, metrics.baseline, strict=True)
        )
        saturation = metrics.saturation_forms or (None,) * metrics.n_worlds
        assert [
            (w.adstock_forms, w.saturation_forms, w.active_channels, w.active_controls)
            for w in worlds
        ] == list(
            zip(
                metrics.adstock_forms,
                saturation,
                metrics.active_channels,
                metrics.active_controls,
                strict=True,
            )
        )
        assert {(w.channel_slots, w.control_slots) for w in worlds} == {
            (metrics.channel_slots, metrics.control_slots)
        }
        impacts = [unit for unit in run.impacts if unit.task_id == task_id]
        assert [(unit.world_id, unit.channel, unit.share) for unit in impacts] == [
            (f"{manifest.run_id}/{task_id}/w{row}", label, share)
            for row, label, share in zip(
                metrics.impact_rows, metrics.impact_labels, metrics.impact_shares, strict=True
            )
        ]
    assert [unit.task_id for unit in run.impacts] == sorted(unit.task_id for unit in run.impacts)


def test_idle_refreshes_and_independent_collectors_agree(tmp_path, payloads):
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store, worlds=8)
    _attempt(store, manifest, 1, "failed")
    _attempt(store, manifest, 2, "running", started=60.0, deadline=3600.0)
    _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 1, 1, payloads[1])
    _publish(store, manifest, 1, 2, payloads[2])
    _publish(store, manifest, 2, 1, b"not a corpus")
    (tmp_path / pub.tasks_dir(manifest.run_id) / "stray").mkdir()
    clock = _clock()
    first = _collector(store, clock)
    before = first.snapshot
    clock.advance(30)
    first.refresh()
    after = first.snapshot
    assert (after.sequence, after.refreshed_at) == (2, EPOCH + timedelta(seconds=30))
    assert (after.runs, after.issues) == (before.runs, before.issues)
    second = _collector(store, clock)
    assert second.snapshot.runs == after.runs
    run = _run(second)
    assert [task.status for task in run.tasks] == ["published", "conflict", "invalid", "pending"]
    assert [(a.status, a.error) for a in run.attempts] == [
        ("running", None),
        ("failed", "RuntimeError: worker crashed"),
    ]


ATTEMPT_MATRIX = {
    # name: (recorded tasks, attempts (n, state, started, deadline), now, state, statuses)
    "no-attempt-nothing-recorded": (0, [], 0, "waiting", []),
    "no-attempt-all-recorded": (2, [], 0, "complete", []),
    "running-before-deadline": (1, [(1, "running", 0, 600)], 300, "running", ["running"]),
    "running-at-deadline": (1, [(1, "running", 0, 600)], 600, "running", ["running"]),
    "running-past-deadline": (1, [(1, "running", 0, 600)], 601, "lost", ["lost"]),
    "running-without-deadline": (1, [(1, "running", 0, None)], 10**6, "running", ["running"]),
    "superseded-by-finished": (
        1,
        [(1, "running", 0, None), (2, "finished", 100, None)],
        200,
        "stopped",
        ["finished", "superseded"],
    ),
    "newest-failed": (
        1,
        [(1, "finished", 0, None), (2, "failed", 100, None)],
        200,
        "failed",
        ["failed", "finished"],
    ),
    "newest-lost-older-superseded": (
        0,
        [(1, "running", 0, None), (2, "running", 100, 200)],
        300,
        "lost",
        ["lost", "superseded"],
    ),
    "same-start-ordered-by-attempt-id": (
        0,
        [(2, "running", 0, None), (1, "running", 0, None)],
        10,
        "running",
        ["running", "superseded"],
    ),
    "complete-despite-failure": (2, [(1, "failed", 0, None)], 10, "complete", ["failed"]),
}


@pytest.mark.parametrize("case", sorted(ATTEMPT_MATRIX))
def test_run_states_and_attempt_statuses(tmp_path, case):
    recorded, attempts, now, state, statuses = ATTEMPT_MATRIX[case]
    store = DirectoryStore(tmp_path)
    manifest = _manifest(store)
    for index in range(recorded):
        _publish(store, manifest, index, 9)
    for n, attempt_state, started, deadline in attempts:
        _attempt(store, manifest, n, attempt_state, started=started, deadline=deadline)
    run = _run(_collector(store, _clock(now)))
    assert run.state == state
    assert [attempt.status for attempt in run.attempts] == statuses
    newest_first = sorted(attempts, key=lambda spec: (spec[2], spec[0]), reverse=True)
    assert [attempt.attempt_id for attempt in run.attempts] == [
        _attempt_id(spec[0]) for spec in newest_first
    ]


@pytest.mark.parametrize("scope", ["task", "run"])
def test_unexpected_errors_stay_with_their_task_or_run(tmp_path, payloads, scope):
    store = DirectoryStore(tmp_path)
    other = _manifest(store, label="other")
    manifest = _manifest(store)
    _publish(store, manifest, 0, 1, payloads[0])
    _publish(store, manifest, 1, 1, payloads[1])
    _publish(store, other, 0, 1)
    if scope == "task":
        fail_on = f"{pub.task_dir(manifest.run_id, 'task-00001')}/"
    else:
        fail_on = pub.attempts_dir(manifest.run_id)
    snapshot = _collector(_FaultyStore(tmp_path, fail_on)).snapshot
    assert snapshot.last_error is None
    expected_issue = f"disk fault under {fail_on}"
    if scope == "task":
        run = _run_of(snapshot, manifest)
        assert [task.status for task in run.tasks] == ["published", "invalid"]
        assert run.tasks[1].detail == f"unexpected error: RuntimeError: {expected_issue}"
        assert len(run.worlds) == 2
        assert snapshot.issues == ()
    else:
        assert manifest.run_id not in snapshot.run_index
        assert snapshot.issues == (f"runs/{manifest.run_id}: RuntimeError: {expected_issue}",)
    assert _run_of(snapshot, other).tasks[0].status == "zero"
