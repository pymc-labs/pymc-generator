"""Generation writer: layout, accounting, resume, self-healing, refusals and deadlines.

The writer runs the real generator (``small`` preset): byte-identical regeneration,
exact accounting and zero-world exhaustion only exist in real output. Later writer calls
start from a copy of the session's golden run (``live_golden``), never from the golden
tree itself, with a clock that starts after the golden attempt.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import platform
import re
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pymc_generator
from live_dashboard import publication as pub
from live_dashboard.collection import Collector
from live_dashboard.extract import extract_payload
from live_dashboard.generation import (
    PRESETS,
    RunConflictError,
    generator_info,
    run_generation,
    run_generation_or_refuse,
)
from live_dashboard.storage import DirectoryStore
from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus
from pymc_generator.sampler import SCMPrior


def _load(name: str) -> ModuleType:
    """Load the plain helper module ``tests/<name>.py`` once (``tests`` is not a package)."""
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")

TASK_IDS = ["task-00000", "task-00001", "task-00002"]
CONFLICT_FIELDS = {"config_sha256", "code_sha256", "environment"}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


def _forbidden_generate(self, *args, **kwargs):
    raise AssertionError("the writer generated a task it should not have generated")


@pytest.fixture
def no_generation(monkeypatch):
    monkeypatch.setattr(DataGenerator, "generate", _forbidden_generate)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` (staging names included) with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _golden_copy(golden: Any, tmp_path: Path) -> DirectoryStore:
    return support.copy_store(golden.root, tmp_path / "store")


def _resume(store: DirectoryStore, golden: Any, hours: float = 1.0, **kwargs: Any) -> dict:
    return run_generation(
        store,
        golden.request,
        worker=golden.worker,
        clock=support.clock_after(golden.end_time, hours),
        **kwargs,
    )


def _task_record(
    store: DirectoryStore, manifest: pub.RunManifest, task_id: str, attempt_id: str
) -> pub.TaskRecord:
    data = store.read_bytes(pub.task_record_path(manifest.run_id, task_id, attempt_id))
    return pub.decode_task_record(data, manifest=manifest, task_id=task_id, attempt_id=attempt_id)


def _attempt_record(
    store: DirectoryStore, manifest: pub.RunManifest, attempt_id: str
) -> pub.AttemptRecord:
    data = store.read_bytes(pub.attempt_record_path(manifest.run_id, attempt_id))
    return pub.decode_attempt_record(data, manifest=manifest, attempt_id=attempt_id)


def _corpus(data: bytes, directory: Path) -> dict[str, Any]:
    path = directory / f"{pub.sha256_hex(data)}.npz"
    path.write_bytes(data)
    return load_corpus(path)


def _new_files(before: dict[str, bytes], after: dict[str, bytes]) -> set[str]:
    """Paths added between two trees; fails if any existing file changed or vanished."""
    assert {path: after.get(path) for path in before} == before
    return set(after) - set(before)


def _unfillable(worlds_per_task: int) -> SCMPrior:
    # No channel series varies this much: every candidate fails the realism gate.
    return dataclasses.replace(PRESETS["small"](worlds_per_task), treatment_cv_floor=50.0)


def _other_cv_floor(worlds_per_task: int) -> SCMPrior:
    return dataclasses.replace(PRESETS["small"](worlds_per_task), treatment_cv_floor=0.6)


# -- golden run ------------------------------------------------------------------------


def test_a_run_publishes_exactly_the_contract_layout(live_golden, tmp_path):
    run_id, attempt_id = live_golden.request.run_id, live_golden.attempt_id
    expected = {pub.run_manifest_path(run_id), pub.attempt_record_path(run_id, attempt_id)}
    for task_id in TASK_IDS:
        expected.add(pub.task_record_path(run_id, task_id, attempt_id))
        expected.add(pub.payload_path(run_id, task_id, attempt_id))
    assert set(_tree(live_golden.root)) == expected

    store = DirectoryStore(live_golden.root)
    manifest = support.load_manifest(store, run_id)
    assert (manifest.request, manifest.created_at) == (live_golden.request, support.EPOCH)
    attempt = _attempt_record(store, manifest, attempt_id)
    assert (attempt.state, attempt.error, attempt.current_task_index) == ("finished", None, None)
    assert (attempt.published_task_ids, attempt.reused_task_ids) == (tuple(TASK_IDS), ())
    for task_id in TASK_IDS:
        record = _task_record(store, manifest, task_id, attempt_id)
        assert (record.status, record.worker) == ("complete", live_golden.worker)
        assert record.started_at <= record.finished_at
        data = store.read_bytes(record.payload.path)
        assert (len(data), pub.sha256_hex(data)) == (record.payload.size, record.payload.sha256)
        corpus = _corpus(data, tmp_path)
        assert DataGenerator.validate_corpus(corpus) == []
        assert len(corpus["cell_id"]) == record.accepted_worlds == 2


def test_accounting_reports_the_generator_diagnostics_of_each_task(live_golden, tmp_path):
    store = DirectoryStore(live_golden.root)
    manifest = support.load_manifest(store, live_golden.request.run_id)
    records = [
        _task_record(store, manifest, task_id, live_golden.attempt_id) for task_id in TASK_IDS
    ]
    # Preconditions: rejections and inactive channel slots occur, so nothing below is vacuous.
    assert sum(record.accounting.rejected_candidates for record in records) > 0
    assert any(
        (
            _corpus(store.read_bytes(record.payload.path), tmp_path)["treatment_active_mask"] == 0
        ).any()
        for record in records
    )
    # One generator call per task, each with the task's own seed.
    assert set(live_golden.diagnostics) == {record.seed for record in records}
    for record in records:
        diagnostics = live_golden.diagnostics[record.seed]
        evaluated = diagnostics["n_draws_evaluated"]
        rejected = round(diagnostics["rejection_rate"] * evaluated)
        assert record.accounting == pub.Accounting(
            candidates_evaluated=evaluated,
            rejected_candidates=rejected,
            draw_failures=diagnostics["n_draw_failures"],
            surplus_discarded=None,
            generation_failure=None,
        )
        assert evaluated == record.accepted_worlds + rejected
        assert record.accepted_worlds == diagnostics["n_tasks"]
        assert record.generation_seconds == diagnostics["timing"]["elapsed_s"]


def test_each_task_publishes_the_independent_batch_of_its_seed(live_golden, tmp_path):
    store = DirectoryStore(live_golden.root)
    manifest = support.load_manifest(store, live_golden.request.run_id)
    record = _task_record(store, manifest, "task-00002", live_golden.attempt_id)
    batch = DataGenerator(PRESETS["small"](2)).generate(
        n_tasks=2, seed=live_golden.request.seed + 2
    )
    save_corpus(batch, tmp_path / "batch.npz")
    assert (tmp_path / "batch.npz").read_bytes() == store.read_bytes(record.payload.path)


# -- resume and self-healing -----------------------------------------------------------


def test_a_rerun_reuses_every_published_task_without_generating(
    live_golden, tmp_path, no_generation
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    before = _tree(store.root)
    result = _resume(store, live_golden)
    assert result == {
        "status": "finished",
        "run_id": run_id,
        "attempt_id": result["attempt_id"],
        "published": [],
        "reused": TASK_IDS,
    }
    assert result["attempt_id"] > live_golden.attempt_id
    added = _new_files(before, _tree(store.root))
    assert added == {pub.attempt_record_path(run_id, result["attempt_id"])}
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert (attempt.state, attempt.published_task_ids, attempt.reused_task_ids) == (
        "finished",
        (),
        tuple(TASK_IDS),
    )


@pytest.mark.parametrize(
    "lose",
    [
        pytest.param(lambda path: path.unlink(), id="record-never-written"),
        pytest.param(lambda path: path.write_bytes(path.read_bytes()[:100]), id="record-partial"),
    ],
)
def test_a_crash_before_the_record_is_resumed_with_identical_content(live_golden, tmp_path, lose):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    manifest = support.load_manifest(store, run_id)
    lost = _task_record(store, manifest, "task-00001", live_golden.attempt_id)
    lose(store.root / pub.task_record_path(run_id, "task-00001", live_golden.attempt_id))
    orphan = store.read_bytes(lost.payload.path)

    result = _resume(store, live_golden)
    assert (result["published"], result["reused"]) == (["task-00001"], ["task-00000", "task-00002"])
    record = _task_record(store, manifest, "task-00001", result["attempt_id"])
    assert record.payload.path == pub.payload_path(run_id, "task-00001", result["attempt_id"])
    # Same digest as the orphan: readers see one scientific content, never a conflict.
    assert record.payload.sha256 == pub.sha256_hex(orphan)
    assert (record.content_key, record.world_ids) == (lost.content_key, lost.world_ids)
    assert store.read_bytes(lost.payload.path) == orphan


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda path: path.unlink(), id="missing"),
        pytest.param(
            lambda path: path.write_bytes(
                bytes([path.read_bytes()[0] ^ 1]) + path.read_bytes()[1:]
            ),
            id="corrupted-same-size",
        ),
    ],
)
def test_a_task_without_a_verifying_payload_is_regenerated_and_deduplicated(
    live_golden, tmp_path, damage
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    manifest = support.load_manifest(store, run_id)
    damaged = _task_record(store, manifest, "task-00000", live_golden.attempt_id)
    damage(store.root / damaged.payload.path)

    result = _resume(store, live_golden)
    assert (result["published"], result["reused"]) == (["task-00000"], TASK_IDS[1:])
    healed = _task_record(store, manifest, "task-00000", result["attempt_id"])
    assert healed.content_key == damaged.content_key

    collector = Collector(store, clock=support.clock_after(live_golden.end_time, hours=2))
    collector.refresh()
    assert collector.snapshot.last_error is None
    run = collector.snapshot.run_index[run_id]
    assert (run.tasks[0].status, run.tasks[0].duplicate_records) == ("published", 1)
    assert run.published_worlds == 6


# -- generation outcomes ---------------------------------------------------------------


def test_cells_that_cannot_fill_publish_a_zero_world_record(tmp_path, monkeypatch):
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("exhausted", "small", 2, 2, 1)
    presets = {"small": _unfillable}
    result = run_generation(
        store, request, worker=support.WORKER, clock=support.FakeClock(), presets=presets
    )
    assert (result["status"], result["published"]) == ("finished", ["task-00000"])

    manifest = support.load_manifest(store, request.run_id)
    assert manifest.generator.config["treatment_cv_floor"] == 50.0
    record = _task_record(store, manifest, "task-00000", result["attempt_id"])
    assert (record.status, record.shortfall_worlds, record.payload) == ("zero", 2, None)
    accounting = record.accounting
    counts = (
        accounting.candidates_evaluated,
        accounting.rejected_candidates,
        accounting.draw_failures,
        accounting.surplus_discarded,
    )
    assert counts == (None, None, None, None)
    assert accounting.generation_failure.type == "RuntimeError"
    assert re.search(r"tasks accepted after \d+ rounds", accounting.generation_failure.message)
    assert record.generation_seconds > 0
    assert store.list_dirs(f"{pub.RUNS_DIR}/{request.run_id}/payloads") == []

    # A zero-world record is a completed task: a rerun reuses it.
    with monkeypatch.context() as patch:
        patch.setattr(DataGenerator, "generate", _forbidden_generate)
        rerun = run_generation(
            store,
            request,
            worker=support.WORKER,
            clock=support.clock_after(support.EPOCH),
            presets=presets,
        )
    assert (rerun["published"], rerun["reused"]) == ([], ["task-00000"])


def test_a_clock_stepping_backwards_still_publishes_ordered_timing(tmp_path):
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("backwards", "small", 2, 2, 1)
    result = run_generation(
        store,
        request,
        worker=support.WORKER,
        clock=support.FakeClock(step=-1.0),
        presets={"small": _unfillable},
    )
    record = _task_record(
        store, support.load_manifest(store, request.run_id), "task-00000", result["attempt_id"]
    )
    assert record.finished_at == record.started_at


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unrelated"),
        RecursionError("cell 0: only 0/1 tasks accepted after 8 rounds"),
        TypeError("unexpected keyword"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_other_generation_failures_fail_the_attempt_without_a_record(
    live_golden, tmp_path, monkeypatch, error
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.task_record_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    before = _tree(store.root)

    def generate(self, *args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(DataGenerator, "generate", generate)
        with pytest.raises(type(error)) as raised:
            _resume(store, live_golden)
    assert raised.value is error
    (added,) = _new_files(before, _tree(store.root))
    attempt_id = Path(added).stem
    assert added == pub.attempt_record_path(run_id, attempt_id)
    attempt = _attempt_record(store, support.load_manifest(store, run_id), attempt_id)
    assert (attempt.state, attempt.error) == ("failed", pub.ErrorInfo.from_exception(error))
    assert (attempt.current_task_index, attempt.reused_task_ids) == (1, ("task-00000",))
    assert attempt.published_task_ids == ()

    later = _resume(store, live_golden, hours=2)
    assert (later["status"], later["published"]) == ("finished", ["task-00001"])


def test_a_task_record_readers_would_reject_is_never_written(live_golden, tmp_path, monkeypatch):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.task_record_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    generate = DataGenerator.generate

    def miscounting(self, *args, **kwargs):
        corpus = generate(self, *args, **kwargs)
        corpus["diagnostics"]["n_draws_evaluated"] = 0  # fewer candidates than accepted worlds
        return corpus

    monkeypatch.setattr(DataGenerator, "generate", miscounting)
    with pytest.raises(pub.RecordError):
        _resume(store, live_golden)
    assert support.task_record_files(store.root, run_id, "task-00001") == []
    manifest = support.load_manifest(store, run_id)
    (attempt_name,) = set(store.list_files(pub.attempts_dir(run_id))) - {
        f"{live_golden.attempt_id}.json"
    }
    attempt = _attempt_record(store, manifest, attempt_name.removesuffix(".json"))
    assert (attempt.state, attempt.error.type) == ("failed", "RecordError")


def test_an_attempt_record_readers_would_reject_is_never_written(tmp_path, no_generation):
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("bad-worker", "small", 2, 2, 1)
    worker = pub.WorkerInfo(platform="test", cpu_count=0, gpu_requested=None, gpu_name=None)
    with pytest.raises(pub.RecordError):
        run_generation(store, request, worker=worker, clock=support.FakeClock())
    assert set(_tree(tmp_path)) == {pub.run_manifest_path(request.run_id)}


def test_an_empty_gpu_name_is_recorded_as_none(live_golden, tmp_path, no_generation):
    store = _golden_copy(live_golden, tmp_path)
    worker = dataclasses.replace(live_golden.worker, gpu_requested="A100", gpu_name="")
    result = run_generation(
        store,
        live_golden.request,
        worker=worker,
        clock=support.clock_after(live_golden.end_time),
    )
    manifest = support.load_manifest(store, live_golden.request.run_id)
    attempt = _attempt_record(store, manifest, result["attempt_id"])
    assert (attempt.worker.gpu_requested, attempt.worker.gpu_name) == ("A100", None)


# -- refusals --------------------------------------------------------------------------


def _set_stored_environment(body: dict[str, Any]) -> None:
    body["generator"]["environment"]["numpy"] = "0.0.0"


@pytest.mark.parametrize(
    ("field", "mutate", "presets"),
    [
        pytest.param("config_sha256", None, {"small": _other_cv_floor}, id="config"),
        pytest.param(
            "code_sha256",
            lambda body: body["generator"].update(code_sha256="0" * 64),
            PRESETS,
            id="code",
        ),
        pytest.param("environment", _set_stored_environment, PRESETS, id="environment"),
    ],
)
def test_resuming_with_another_generator_is_refused_before_any_write(
    live_golden, tmp_path, field, mutate, presets
):
    store = _golden_copy(live_golden, tmp_path)
    if mutate is not None:
        support.rewrite_json(store.root / pub.run_manifest_path(live_golden.request.run_id), mutate)
    before = _tree(store.root)
    with pytest.raises(RunConflictError) as raised:
        _resume(store, live_golden, presets=presets)
    assert [name for name in sorted(CONFLICT_FIELDS) if name in str(raised.value)] == [field]
    assert _tree(store.root) == before


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda path: path.write_bytes(path.read_bytes()[:200]), id="partial"),
        pytest.param(
            lambda path: support.rewrite_json(
                path, lambda body: body.update(contract_version="2.0")
            ),
            id="unsupported-version",
        ),
        pytest.param(
            lambda path: support.rewrite_json(
                path, lambda body: body["generator"]["config"].update(n_cells=3)
            ),
            id="config-digest-mismatch",
        ),
    ],
)
def test_an_unusable_run_manifest_is_refused_before_any_write(live_golden, tmp_path, corrupt):
    store = _golden_copy(live_golden, tmp_path)
    corrupt(store.root / pub.run_manifest_path(live_golden.request.run_id))
    before = _tree(store.root)
    with pytest.raises(RunConflictError, match="unusable run.json"):
        _resume(store, live_golden)
    assert _tree(store.root) == before


def test_a_refused_resume_is_recorded_as_a_failed_attempt_without_raising(live_golden, tmp_path):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    support.rewrite_json(store.root / pub.run_manifest_path(run_id), _set_stored_environment)
    before = _tree(store.root)
    result = run_generation_or_refuse(
        store,
        live_golden.request,
        worker=live_golden.worker,
        clock=support.clock_after(live_golden.end_time),
    )
    assert result == {
        "status": "refused",
        "run_id": run_id,
        "attempt_id": result["attempt_id"],
        "reason": result["reason"],
    }
    assert "environment" in result["reason"]
    added = _new_files(before, _tree(store.root))
    assert added == {pub.attempt_record_path(run_id, result["attempt_id"])}
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert attempt.state == "failed"
    assert attempt.error == pub.ErrorInfo("RunConflictError", result["reason"])


def test_a_request_for_another_run_than_announced_is_refused(live_golden, tmp_path, no_generation):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    announced = "golden-000000000000"
    before = _tree(store.root)
    result = run_generation_or_refuse(
        store,
        live_golden.request,
        worker=live_golden.worker,
        expected_run_id=announced,
        clock=support.clock_after(live_golden.end_time),
    )
    assert (result["status"], result["run_id"]) == ("refused", run_id)
    assert announced in result["reason"] and run_id in result["reason"]
    added = _new_files(before, _tree(store.root))
    assert added == {pub.attempt_record_path(run_id, result["attempt_id"])}
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert (attempt.state, attempt.error.type) == ("failed", "RunConflictError")

    # Without a manifest of the request's run the refusal cannot be recorded anywhere.
    empty = tmp_path / "empty"
    refused = run_generation_or_refuse(
        DirectoryStore(empty),
        live_golden.request,
        worker=live_golden.worker,
        expected_run_id=announced,
        clock=support.FakeClock(),
    )
    assert refused == {**result, "attempt_id": None}
    assert not empty.exists()


@pytest.mark.parametrize(
    ("request_args", "kwargs"),
    [
        pytest.param(("big", "small", 402, 2, 1), {}, id="too-many-worlds"),
        pytest.param(("big", "small", 22, 22, 1), {}, id="too-many-worlds-per-task"),
        pytest.param(("odd", "unknown", 2, 2, 1), {}, id="unknown-preset"),
        pytest.param(("odd", "small", 2, 2, 1), {"attempt_id": "attempt-1"}, id="bad-attempt-id"),
    ],
)
def test_invalid_requests_are_rejected_before_any_write(
    tmp_path, no_generation, request_args, kwargs
):
    with pytest.raises(ValueError):
        run_generation(
            DirectoryStore(tmp_path / "store"),
            pub.RunRequest(*request_args),
            worker=support.WORKER,
            clock=support.FakeClock(),
            **kwargs,
        )
    assert not (tmp_path / "store").exists()


# -- deadline --------------------------------------------------------------------------


def test_an_attempt_stops_before_a_task_that_could_outlast_the_deadline(tmp_path, monkeypatch):
    clock = support.FakeClock()
    generate = DataGenerator.generate

    def long_task(self, *args, **kwargs):
        clock.advance(100)  # the task takes about 100 s of the attempt's time
        return generate(self, *args, **kwargs)

    monkeypatch.setattr(DataGenerator, "generate", long_task)
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("deadline", "small", 6, 2, 1)
    deadline = support.EPOCH + timedelta(seconds=250)
    result = run_generation(store, request, worker=support.WORKER, clock=clock, deadline=deadline)
    # About 145 s are left before task 1: more than the 60 s floor, less than twice 100 s.
    assert (result["status"], result["published"]) == ("stopped", ["task-00000"])
    attempt = _attempt_record(
        store, support.load_manifest(store, request.run_id), result["attempt_id"]
    )
    assert (attempt.state, attempt.deadline_at, attempt.error) == ("finished", deadline, None)
    assert store.list_dirs(pub.tasks_dir(request.run_id)) == ["task-00000"]


def test_an_attempt_with_less_than_a_minute_left_generates_nothing(
    live_golden, tmp_path, no_generation
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.task_record_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    clock = support.clock_after(live_golden.end_time)
    result = run_generation(
        store,
        live_golden.request,
        worker=live_golden.worker,
        clock=clock,
        deadline=clock.now + timedelta(seconds=59),
    )
    assert (result["status"], result["published"], result["reused"]) == (
        "stopped",
        [],
        ["task-00000"],
    )
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert attempt.state == "finished"


# -- generator identity and presets ----------------------------------------------------


def test_missing_packages_are_recorded_as_unknown_versions(monkeypatch):
    cfg = PRESETS["small"](2)

    def version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    info = generator_info("small", cfg)
    assert info.environment == {
        "python": platform.python_version(),
        "machine": platform.machine(),
        **dict.fromkeys(("numpy", "scipy", "pytensor", "pymc", "pymc-marketing"), "unknown"),
    }


def test_the_deployment_revision_takes_precedence_over_git(tmp_path, monkeypatch):
    cfg = PRESETS["small"](2)
    monkeypatch.setenv("LIVE_DASHBOARD_CODE_REVISION", "  abc123  ")
    assert generator_info("small", cfg).code_revision == "abc123"
    # A blank value falls back to git, which is unavailable here.
    monkeypatch.setenv("LIVE_DASHBOARD_CODE_REVISION", "   ")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert generator_info("small", cfg).code_revision is None


def test_the_code_digest_covers_generator_sources_wherever_they_live(tmp_path, monkeypatch):
    cfg = PRESETS["small"](2)
    original = generator_info("small", cfg).code_sha256
    copy = tmp_path / "elsewhere" / "pymc_generator"
    shutil.copytree(
        Path(pymc_generator.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "stale.py").write_text("x = 1\n")
    (copy / "notes.txt").write_text("not source")
    monkeypatch.setattr(pymc_generator, "__file__", str(copy / "__init__.py"))
    assert generator_info("small", cfg).code_sha256 == original
    (copy / "slots.py").write_bytes((copy / "slots.py").read_bytes() + b"\n")
    assert generator_info("small", cfg).code_sha256 != original


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_rebuilding_a_preset_yields_the_same_recorded_generator(preset):
    # A resume rebuilds the configuration; any drift would refuse every resume.
    first = generator_info(preset, PRESETS[preset](4))
    request = pub.RunRequest("preset", preset, 4, 4, 29)
    manifest = pub.RunManifest(
        run_id=request.run_id, created_at=support.EPOCH, request=request, generator=first
    )
    decoded = pub.decode_run_manifest(manifest.encode(), run_id=request.run_id)
    again = generator_info(preset, PRESETS[preset](4))
    assert decoded.generator.config_sha256 == again.config_sha256
    assert decoded.generator.config["n_cells"] == 4


def test_the_showcase_preset_keeps_the_notebook_design_per_task():
    cfg = PRESETS["showcase-variety"](4)
    assert (cfg.n_cells, cfg.draws_per_cell, cfg.active_count_allocation) == (4, 1, "independent")
    assert cfg.record_walk_parameters and cfg.outcome_std_mode == "relative"
    # The notebook's fixed design values: zeros come only from flighting.
    for role in ("treatment", "covariate"):
        for gate in ("onset", "offset"):
            assert getattr(cfg, f"{role}_{gate}_inclusion_prob") == 0.0
    assert {name for name, p in cfg.saturation_family_probs.items() if p} == {"michaelis_menten"}
    assert {name for name, p in cfg.carryover_family_probs.items() if p} == {"geometric"}
    assert cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] == (0.05, 20.0)
    assert {edge: n for edge, n in cfg.edge_budget.items() if n != (0, 0)} == {
        "cy": (10, 10),
        "zy": (10, 10),
    }
    # Building the preset must not leak its saturation range into the library defaults.
    assert SCMPrior().saturation_prior_ranges["michaelis_menten"]["kappa_mult"] != (0.05, 20.0)


@pytest.mark.slow
def test_a_showcase_task_is_published_and_extracted(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    request = pub.RunRequest("showcase", "showcase-variety", 2, 2, 29)
    result = run_generation(store, request, worker=support.WORKER, clock=support.FakeClock())
    assert (result["status"], result["published"]) == ("finished", ["task-00000"])
    record = _task_record(
        store, support.load_manifest(store, request.run_id), "task-00000", result["attempt_id"]
    )
    metrics = extract_payload(store.read_bytes(record.payload.path))
    assert metrics.n_worlds == record.accepted_worlds == 2
