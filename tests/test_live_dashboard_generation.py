"""Generation writer: layout, accounting, resume, refusals and deadlines.

The writer runs the real generator (``small`` preset): byte-identical regeneration,
exact accounting and zero-world exhaustion only exist in real output. Tasks are published
as ``pymc_generator.publication`` completions. Later writer calls start from a copy of
the session's golden run (``live_golden``), never from the golden tree itself, with a
clock that starts after the golden attempt.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pymc_generator
from live_dashboard import publication as pub
from live_dashboard.collection import Collector
from live_dashboard.deployment import DEMO_MAX_WORLDS, DEMO_MAX_WORLDS_PER_TASK
from live_dashboard.extract import extract_payload
from live_dashboard.generation import (
    MIXES,
    PRESETS,
    RequestError,
    RunConflictError,
    generator_info,
    preset_cycle,
    run_generation,
    run_generation_or_refuse,
)
from live_dashboard.storage import DirectoryStore
from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus
from pymc_generator.publication import PublicationError
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
    data = store.read_bytes(pub.completion_path(manifest.run_id, task_id, attempt_id))
    return pub.task_record_from_completion(manifest, json.loads(data))


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
        expected.add(pub.completion_path(run_id, task_id, attempt_id))
        expected.add(pub.completion_payload_path(run_id, task_id, attempt_id))
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
        # Membership is deferred: every world is unassigned (is_val=2) until a snapshot.
        assert DataGenerator.validate_corpus(corpus, allow_deferred_split=True) == []
        assert (corpus["is_val"] == 2).all()
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
            surplus_discarded=evaluated - rejected - record.accepted_worlds,
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
        n_tasks=2, seed=live_golden.request.seed + 2, defer_cell_split=True
    )
    save_corpus(batch, tmp_path / "batch.npz")
    assert (tmp_path / "batch.npz").read_bytes() == store.read_bytes(record.payload.path)


def test_a_reader_follows_the_run_task_by_task(live_golden):
    pending, published = "pending", "published"
    assert live_golden.reader_views == (
        ("running", (pending, pending, pending), 0),
        ("running", (published, pending, pending), 1),
        ("running", (published, published, pending), 2),
        ("complete", (published, published, published), None),
    )


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


def test_a_crash_before_the_completion_is_resumed_with_identical_content(live_golden, tmp_path):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    manifest = support.load_manifest(store, run_id)
    lost = _task_record(store, manifest, "task-00001", live_golden.attempt_id)
    (store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    orphan = store.read_bytes(lost.payload.path)

    result = _resume(store, live_golden)
    assert (result["published"], result["reused"]) == (["task-00001"], ["task-00000", "task-00002"])
    record = _task_record(store, manifest, "task-00001", result["attempt_id"])
    assert record.payload.path == pub.completion_payload_path(
        run_id, "task-00001", result["attempt_id"]
    )
    # Same digest as the orphan: readers see one scientific content, never a conflict.
    assert record.payload.sha256 == pub.sha256_hex(orphan)
    assert (record.content_key, record.world_ids) == (lost.content_key, lost.world_ids)
    assert store.read_bytes(lost.payload.path) == orphan


def test_a_partial_completion_stops_the_resume_until_reconciled(
    live_golden, tmp_path, no_generation
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    path = store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)
    path.write_bytes(path.read_bytes()[:100])  # a crash while the completion was written
    before = _tree(store.root)

    # An unresolved completion is never overwritten or bypassed by a new attempt.
    with pytest.raises(PublicationError, match="task-00001 has a completion that does not"):
        _resume(store, live_golden)
    (added,) = _new_files(before, _tree(store.root))
    attempt = _attempt_record(store, support.load_manifest(store, run_id), Path(added).stem)
    assert (attempt.state, attempt.reused_task_ids) == ("failed", ("task-00000",))


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
def test_a_task_whose_payload_no_longer_verifies_stops_the_resume_until_reconciled(
    live_golden, tmp_path, damage, no_generation
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    manifest = support.load_manifest(store, run_id)
    damaged = _task_record(store, manifest, "task-00000", live_golden.attempt_id)
    damage(store.root / damaged.payload.path)
    before = _tree(store.root)

    # pymc_generator.publication never publishes past an unusable earlier attempt: the
    # resume stops before generating, and records why.
    with pytest.raises(PublicationError, match="task-00000 has a completion that does not"):
        _resume(store, live_golden)
    (added,) = _new_files(before, _tree(store.root))
    attempt = _attempt_record(store, manifest, Path(added).stem)
    assert (attempt.state, attempt.error.type) == ("failed", "PublicationError")
    assert support.completion_files(store.root, run_id, "task-00000") == [
        store.root / pub.completion_path(run_id, "task-00000", live_golden.attempt_id)
    ]

    collector = Collector(store, clock=support.clock_after(live_golden.end_time, hours=2))
    collector.refresh()
    run = collector.snapshot.run_index[run_id]
    # Every task has a final record, so the run is complete; task 0's payload is shown
    # unavailable rather than replaced.
    assert (run.state, run.tasks[0].status, run.published_worlds) == ("complete", "unavailable", 4)


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
    failure = accounting.generation_failure
    assert failure.type == "RuntimeError"
    # The counts are the tallies the library reports for the cell that could not fill.
    tally = re.search(
        r"tasks accepted after \d+ rounds .*rejected (\d+)/(\d+) evaluated candidates, "
        r"and (\d+)/\d+ rounds",
        failure.message,
    )
    assert tally is not None
    rejected, evaluated, failures = map(int, tally.groups())
    assert evaluated > 0
    assert (
        accounting.candidates_evaluated,
        accounting.rejected_candidates,
        accounting.draw_failures,
        accounting.surplus_discarded,
    ) == (evaluated, rejected, failures, evaluated - rejected)
    assert record.generation_seconds > 0
    completion_dir = tmp_path / pub.completion_dir(
        request.run_id, "task-00000", result["attempt_id"]
    )
    assert sorted(path.name for path in completion_dir.iterdir()) == [pub.COMPLETION_NAME]

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
        KeyboardInterrupt("cancelled"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_other_generation_failures_fail_the_attempt_without_a_record(
    live_golden, tmp_path, monkeypatch, error
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
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


def test_refusing_never_swallows_a_failure_a_retry_could_fix(live_golden, tmp_path, monkeypatch):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    before = _tree(store.root)

    def generate(self, *args, **kwargs):
        raise TypeError("unexpected keyword")

    monkeypatch.setattr(DataGenerator, "generate", generate)
    with pytest.raises(TypeError):
        run_generation_or_refuse(
            store,
            live_golden.request,
            worker=live_golden.worker,
            expected_run_id=run_id,
            clock=support.clock_after(live_golden.end_time),
        )
    (added,) = _new_files(before, _tree(store.root))
    attempt = _attempt_record(store, support.load_manifest(store, run_id), Path(added).stem)
    assert (attempt.state, attempt.error.type) == ("failed", "TypeError")


def test_a_completion_readers_would_reject_is_never_written(live_golden, tmp_path, monkeypatch):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    generate = DataGenerator.generate

    def miscounting(self, *args, **kwargs):
        corpus = generate(self, *args, **kwargs)
        corpus["diagnostics"]["n_draws_evaluated"] = 0  # fewer candidates than accepted worlds
        return corpus

    monkeypatch.setattr(DataGenerator, "generate", miscounting)
    with pytest.raises(PublicationError, match="Invalid count"):
        _resume(store, live_golden)
    assert support.completion_files(store.root, run_id, "task-00001") == []
    manifest = support.load_manifest(store, run_id)
    (attempt_name,) = set(store.list_files(pub.attempts_dir(run_id))) - {
        f"{live_golden.attempt_id}.json"
    }
    attempt = _attempt_record(store, manifest, attempt_name.removesuffix(".json"))
    assert (attempt.state, attempt.error.type) == ("failed", "PublicationError")


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
        pytest.param(
            # Still valid JSON, but larger than any reader reads.
            lambda path: path.write_bytes(path.read_bytes() + b" " * pub.MAX_RECORD_BYTES),
            id="oversized",
        ),
    ],
)
def test_an_unusable_run_manifest_is_refused_before_any_write(live_golden, tmp_path, corrupt):
    store = _golden_copy(live_golden, tmp_path)
    corrupt(store.root / pub.run_manifest_path(live_golden.request.run_id))
    before = _tree(store.root)
    with pytest.raises(RunConflictError, match="unusable run.json"):
        _resume(store, live_golden)
    # Without a usable manifest the refusal cannot be recorded under the run.
    refused = run_generation_or_refuse(
        store,
        live_golden.request,
        worker=live_golden.worker,
        clock=support.clock_after(live_golden.end_time),
    )
    assert (refused["status"], refused["attempt_id"]) == ("refused", None)
    assert "unusable run.json" in refused["reason"]
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


def test_a_malformed_attempt_id_is_refused_under_a_fresh_attempt(
    live_golden, tmp_path, no_generation
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    before = _tree(store.root)
    result = run_generation_or_refuse(
        store,
        live_golden.request,
        worker=live_golden.worker,
        attempt_id="attempt-1",
        clock=support.clock_after(live_golden.end_time),
    )
    assert (result["status"], result["run_id"]) == ("refused", run_id)
    assert pub.is_attempt_id(result["attempt_id"])
    assert _new_files(before, _tree(store.root)) == {
        pub.attempt_record_path(run_id, result["attempt_id"])
    }
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert (attempt.state, attempt.error.type) == ("failed", "RequestError")


@pytest.mark.parametrize(
    ("request_args", "kwargs"),
    [
        pytest.param(
            (
                "big",
                "small",
                (DEMO_MAX_WORLDS // DEMO_MAX_WORLDS_PER_TASK + 1) * DEMO_MAX_WORLDS_PER_TASK,
                DEMO_MAX_WORLDS_PER_TASK,
                1,
            ),
            {},
            id="too-many-worlds",
        ),
        pytest.param(
            ("big", "small", 2 * (DEMO_MAX_WORLDS_PER_TASK + 1), DEMO_MAX_WORLDS_PER_TASK + 1, 1),
            {},
            id="too-many-worlds-per-task",
        ),
        pytest.param(("odd", "unknown", 2, 2, 1), {}, id="unknown-preset"),
        pytest.param(("odd", "small", 2, 2, 1), {"attempt_id": "attempt-1"}, id="bad-attempt-id"),
    ],
)
def test_invalid_requests_are_refused_before_any_write(
    tmp_path, no_generation, request_args, kwargs
):
    request = pub.RunRequest(*request_args)
    store = DirectoryStore(tmp_path / "store")
    with pytest.raises(RequestError):
        run_generation(store, request, worker=support.WORKER, clock=support.FakeClock(), **kwargs)
    refused = run_generation_or_refuse(
        store, request, worker=support.WORKER, clock=support.FakeClock(), **kwargs
    )
    assert (refused["status"], refused["run_id"], refused["attempt_id"]) == (
        "refused",
        request.run_id,
        None,
    )
    assert not (tmp_path / "store").exists()


def test_the_largest_demo_request_is_accepted(tmp_path, no_generation):
    store = DirectoryStore(tmp_path)
    largest = DEMO_MAX_WORLDS - DEMO_MAX_WORLDS % DEMO_MAX_WORLDS_PER_TASK
    request = pub.RunRequest("largest", "small", largest, DEMO_MAX_WORLDS_PER_TASK, 1)
    clock = support.FakeClock(step=0.0)
    # A deadline that leaves no time for a task: the run opens without generating.
    result = run_generation(
        store,
        request,
        worker=support.WORKER,
        clock=clock,
        deadline=clock.now + timedelta(seconds=1),
    )
    assert (result["status"], result["published"]) == ("stopped", [])
    manifest = support.load_manifest(store, request.run_id)
    assert (manifest.n_tasks, manifest.worlds_per_task) == (
        largest // DEMO_MAX_WORLDS_PER_TASK,
        DEMO_MAX_WORLDS_PER_TASK,
    )


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
    # About 145 s are left before task 1: more than the 90 s floor, less than twice 100 s.
    assert (result["status"], result["published"]) == ("stopped", ["task-00000"])
    attempt = _attempt_record(
        store, support.load_manifest(store, request.run_id), result["attempt_id"]
    )
    assert (attempt.state, attempt.deadline_at, attempt.error) == ("finished", deadline, None)
    assert store.list_dirs(pub.tasks_dir(request.run_id)) == ["task-00000"]


@pytest.mark.parametrize(
    ("seconds_left", "expected"),
    [
        pytest.param(89, ("stopped", [], ["task-00000"]), id="89s"),
        pytest.param(90, ("finished", ["task-00001"], ["task-00000", "task-00002"]), id="90s"),
    ],
)
def test_a_task_starts_only_with_at_least_90_seconds_left(
    live_golden, tmp_path, seconds_left, expected
):
    store = _golden_copy(live_golden, tmp_path)
    run_id = live_golden.request.run_id
    (store.root / pub.completion_path(run_id, "task-00001", live_golden.attempt_id)).unlink()
    # A stopped clock: exactly ``seconds_left`` remain when task 1 is due.
    clock = support.FakeClock(start=live_golden.end_time + timedelta(hours=1), step=0.0)
    deadline = clock.now + timedelta(seconds=seconds_left)
    result = run_generation(
        store, live_golden.request, worker=live_golden.worker, clock=clock, deadline=deadline
    )
    assert (result["status"], result["published"], result["reused"]) == expected
    attempt = _attempt_record(store, support.load_manifest(store, run_id), result["attempt_id"])
    assert (attempt.state, attempt.deadline_at) == ("finished", deadline)


# -- generator identity and presets ----------------------------------------------------


def _single(preset: str, cfg: SCMPrior) -> pub.GeneratorInfo:
    """The recorded generator of a run of one preset (a one-preset cycle)."""
    return generator_info(preset, (preset,), {preset: cfg})


def test_missing_packages_are_recorded_as_unknown_versions(monkeypatch):
    cfg = PRESETS["small"](2)

    def version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    info = _single("small", cfg)
    assert info.environment == {
        "python": platform.python_version(),
        "machine": platform.machine(),
        **dict.fromkeys(("numpy", "scipy", "pytensor", "pymc", "pymc-marketing"), "unknown"),
    }


def test_the_deployment_revision_takes_precedence_over_git(tmp_path, monkeypatch):
    cfg = PRESETS["small"](2)
    monkeypatch.setenv("LIVE_DASHBOARD_CODE_REVISION", "  abc123  ")
    assert _single("small", cfg).code_revision == "abc123"
    # A blank value falls back to git, which is unavailable here.
    monkeypatch.setenv("LIVE_DASHBOARD_CODE_REVISION", "   ")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _single("small", cfg).code_revision is None


def test_the_code_digest_covers_generator_sources_wherever_they_live(tmp_path, monkeypatch):
    cfg = PRESETS["small"](2)
    original = _single("small", cfg).code_sha256
    copy = tmp_path / "elsewhere" / "pymc_generator"
    shutil.copytree(
        Path(pymc_generator.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "stale.py").write_text("x = 1\n")
    (copy / "notes.txt").write_text("not source")
    monkeypatch.setattr(pymc_generator, "__file__", str(copy / "__init__.py"))
    assert _single("small", cfg).code_sha256 == original
    (copy / "slots.py").write_bytes((copy / "slots.py").read_bytes() + b"\n")
    assert _single("small", cfg).code_sha256 != original


@pytest.mark.parametrize("preset", sorted({*PRESETS, *MIXES}))
def test_rebuilding_a_preset_yields_the_same_recorded_generator(preset):
    # A resume rebuilds the configuration; any drift would refuse every resume.
    cycle = preset_cycle(preset)

    def recorded() -> pub.GeneratorInfo:
        return generator_info(preset, cycle, {name: PRESETS[name](4) for name in cycle})

    request = pub.RunRequest("preset", preset, 4, 4, 29)
    manifest = pub.RunManifest(
        run_id=request.run_id, created_at=support.EPOCH, request=request, generator=recorded()
    )
    decoded = pub.decode_run_manifest(manifest.encode(), run_id=request.run_id)
    assert decoded.generator.config_sha256 == recorded().config_sha256
    config = decoded.generator.config
    members = config["presets"].values() if len(cycle) > 1 else [config]
    assert {member["n_cells"] for member in members} == {4}


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


@pytest.mark.slow
@pytest.mark.parametrize(
    ("preset", "saturation"),
    [("geometric-mm", "michaelis_menten"), ("geometric-logistic", "logistic")],
)
def test_a_variety_task_has_only_direct_positive_effects_of_its_style(tmp_path, preset, saturation):
    store = DirectoryStore(tmp_path / "store")
    request = pub.RunRequest("variety", preset, 2, 2, 1000)
    result = run_generation(store, request, worker=support.WORKER, clock=support.FakeClock())
    assert (result["status"], result["published"]) == ("finished", ["task-00000"])
    record = _task_record(
        store, support.load_manifest(store, request.run_id), "task-00000", result["attempt_id"]
    )
    data = store.read_bytes(record.payload.path)
    corpus = _corpus(data, tmp_path)
    graph = PRESETS[preset](2).layout.unpack(corpus["g"])
    channels = corpus["treatment_active_mask"] == 1
    controls = corpus["covariate_active_mask"] == 1
    # Every active channel and control points to the target, and nothing else has an arrow.
    assert (graph["cy"] == channels).all() and (graph["zy"] == controls).all()
    assert not any(graph[edge].any() for edge in ("cc", "zc", "zz", "dc", "dz", "dy"))
    assert not corpus["indirect_effects"].any()
    assert not corpus["latent_unobserved_contribution"].any()
    assert (corpus["treatment_contribution_raw"] >= 0).all()
    assert (corpus["covariate_contribution"] >= 0).all()
    collector = Collector(store)
    collector.refresh()
    worlds = collector.snapshot.run_index[request.run_id].worlds
    assert [(world.adstock_forms, world.saturation_forms) for world in worlds] == [
        (("geometric",), (saturation,))
    ] * 2


# -- preset mixes ----------------------------------------------------------------------


def _small_with_carryover(family: str) -> Callable[[int], SCMPrior]:
    """The small preset with every channel on one carryover family (visible in payloads)."""

    def build(worlds_per_task: int) -> SCMPrior:
        cfg = PRESETS["small"](worlds_per_task)
        probs = {name: float(name == family) for name in cfg.carryover_family_probs}
        return dataclasses.replace(cfg, carryover_family_probs=probs)

    return build


MIX_PRESETS = {"geo": _small_with_carryover("geometric"), "wei": _small_with_carryover("weibull")}
MIX = {"geo-wei": ("geo", "wei")}


@pytest.mark.slow  # two compiled configurations: about 3 s locally
def test_a_mix_alternates_its_presets_by_task_and_records_each_one(tmp_path):
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("mix", "geo-wei", 6, 2, 7)
    result = run_generation(
        store,
        request,
        worker=support.WORKER,
        clock=support.FakeClock(),
        presets=MIX_PRESETS,
        mixes=MIX,
    )
    assert (result["status"], result["published"]) == ("finished", TASK_IDS)
    collector = Collector(store)
    collector.refresh()
    run = collector.snapshot.run_index[request.run_id]
    assert {world.task_id: world.adstock_forms for world in run.worlds} == {
        "task-00000": ("geometric",),
        "task-00001": ("weibull",),
        "task-00002": ("geometric",),
    }
    config = support.load_manifest(store, request.run_id).generator.config
    assert config["cycle"] == ["geo", "wei"]
    assert {
        name: member["carryover_family_probs"]["weibull"]
        for name, member in config["presets"].items()
    } == {
        "geo": 0.0,
        "wei": 1.0,
    }


def test_resuming_a_mix_whose_member_changed_is_refused_before_any_write(tmp_path):
    store = DirectoryStore(tmp_path)
    request = pub.RunRequest("mix", "geo-wei", 4, 2, 7)
    clock = support.FakeClock(step=0.0)
    # A deadline that leaves no time for a task: the run opens without generating.
    opened = run_generation(
        store,
        request,
        worker=support.WORKER,
        clock=clock,
        deadline=clock.now + timedelta(seconds=1),
        presets=MIX_PRESETS,
        mixes=MIX,
    )
    assert (opened["status"], opened["published"]) == ("stopped", [])
    before = _tree(store.root)
    changed = {**MIX_PRESETS, "wei": _small_with_carryover("none")}
    with pytest.raises(RunConflictError, match="config_sha256"):
        run_generation(
            store, request, worker=support.WORKER, clock=clock, presets=changed, mixes=MIX
        )
    assert _tree(store.root) == before


def test_a_resumed_mix_keeps_every_task_on_its_member_and_seed(live_golden, tmp_path, monkeypatch):
    golden = _corpus(
        DirectoryStore(live_golden.root).read_bytes(
            pub.completion_payload_path(
                live_golden.request.run_id, "task-00000", live_golden.attempt_id
            )
        ),
        tmp_path,
    )
    # A saved corpus drops the in-memory timing a fresh generation carries.
    golden["diagnostics"] = {
        **golden["diagnostics"],
        "timing": {"elapsed_s": 100.0, "tasks_per_sec": 0.02},
    }
    clock = support.FakeClock()
    calls: list[tuple[str, int]] = []

    def recorded_generate(self, *, n_tasks, seed, defer_cell_split):
        assert defer_cell_split  # membership is assigned later, by a training snapshot
        weibull = self.config.carryover_family_probs["weibull"] == 1.0
        calls.append(("wei" if weibull else "geo", seed))
        clock.advance(100)  # each task takes 100 s of the attempt's time
        return golden  # a valid two-world corpus: no generation needed to check the mapping

    monkeypatch.setattr(DataGenerator, "generate", recorded_generate)
    store = DirectoryStore(tmp_path / "store")
    request = pub.RunRequest("mix", "geo-wei", 6, 2, 7)
    options = {"worker": support.WORKER, "clock": clock, "presets": MIX_PRESETS, "mixes": MIX}
    # 150 s are left after task 0: less than twice its 100 s, so the attempt stops there.
    first = run_generation(
        store, request, deadline=support.EPOCH + timedelta(seconds=250), **options
    )
    assert (first["status"], first["published"]) == ("stopped", ["task-00000"])
    second = run_generation(store, request, **options)
    assert (second["reused"], second["published"]) == (["task-00000"], TASK_IDS[1:])
    # Task i uses member i % 2 and seed 7 + i, whichever attempt generates it.
    assert calls == [("geo", 7), ("wei", 8), ("geo", 9)]


@pytest.mark.parametrize(
    ("mixes", "message"),
    [
        ({"broken": ("geo", "gone")}, r"preset mix 'broken' names unknown presets: \['gone'\]"),
        (
            {"broken": ("geo", "geo-wei"), "geo-wei": ("geo", "wei")},
            r"unknown presets: \['geo-wei'\]",
        ),
        ({"broken": ()}, r"preset mix 'broken' must name one or more presets"),
        ({"broken": "geo"}, r"preset mix 'broken' must name one or more presets"),
        ({"geo": ("wei",), "broken": ("geo",)}, r"'geo' names both a preset and a preset mix"),
    ],
    ids=["unknown-member", "mix-of-mixes", "empty", "bare-string", "shadows-a-preset"],
)
def test_an_unusable_mix_is_refused_before_any_write(tmp_path, mixes, message):
    store = DirectoryStore(tmp_path / "store")
    preset = "geo" if "geo" in mixes else "broken"
    request = pub.RunRequest("mix", preset, 4, 2, 7)
    with pytest.raises(RequestError, match=message):
        run_generation(store, request, worker=support.WORKER, presets=MIX_PRESETS, mixes=mixes)
    assert not (tmp_path / "store").exists()
