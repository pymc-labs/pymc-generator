"""Publication contract v1.1: run identities, record round trips and rejection classes.

Readers rely on these properties to discover, deduplicate and reject records without
guessing; a change to the identity digest would orphan every published run.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime, timedelta

import pytest

from live_dashboard import publication as pub

NOW = datetime(2026, 1, 1, 12, 0, 0, 123000, tzinfo=UTC)
WORKER = pub.WorkerInfo(platform="linux", cpu_count=2, gpu_requested="A100", gpu_name="A100-SXM4")
ATTEMPT = pub.new_attempt_id(NOW)


def _manifest(request: pub.RunRequest | None = None) -> pub.RunManifest:
    request = request or pub.RunRequest("demo", "small", 6, 2, 1)
    generator = pub.GeneratorInfo(
        preset=request.preset,
        config={"n_cells": 2, "by_index": {2: "a", 10: "b"}},
        pymc_generator_version="0.0.2",
        corpus_schema_version=5,
        code_sha256="ab" * 32,
        code_revision="deadbeef",
        environment={"numpy": "2.4.6", "python": "3.13.9"},
    )
    return pub.RunManifest(
        run_id=request.run_id, created_at=NOW, request=request, generator=generator
    )


def _task(manifest: pub.RunManifest, index: int = 1, accepted: int = 2) -> pub.TaskRecord:
    task_id = pub.task_id_for(index)
    payload = None
    if accepted:
        payload = pub.PayloadRef(
            path=pub.payload_path(manifest.run_id, task_id, ATTEMPT),
            sha256="0f" * 32,
            size=1234,
            corpus_schema_version=5,
        )
    return pub.TaskRecord(
        run_id=manifest.run_id,
        task_id=task_id,
        task_index=index,
        seed=manifest.request.task_seed(index),
        target_worlds=manifest.worlds_per_task,
        accepted_worlds=accepted,
        world_ids=tuple(pub.world_id_for(manifest.run_id, task_id, row) for row in range(accepted)),
        payload=payload,
        accounting=(
            pub.Accounting(5, 3, 0, None, None)
            if accepted
            else pub.Accounting(None, None, None, None, pub.ErrorInfo("RuntimeError", "cell 0"))
        ),
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
        generation_seconds=1.75,
        attempt_id=ATTEMPT,
        worker=WORKER,
    )


def _decode_task(data: bytes, manifest: pub.RunManifest, index: int = 1) -> pub.TaskRecord:
    return pub.decode_task_record(
        data, manifest=manifest, task_id=pub.task_id_for(index), attempt_id=ATTEMPT
    )


def _edit(data: bytes, **changes: object) -> bytes:
    body = json.loads(data)
    body.update(changes)
    return json.dumps(body).encode()


def test_run_id_is_pinned_to_the_canonical_request_digest():
    assert pub.RunRequest("demo", "small", 6, 2, 1).run_id == "demo-3bf7c7e5736c"


@pytest.mark.parametrize(
    "change",
    [{"label": "other"}, {"preset": "showcase-variety"}, {"worlds": 8}, {"worlds_per_task": 3}],
)
def test_every_request_field_names_a_different_run(change):
    base = pub.RunRequest("demo", "small", 6, 2, 1)
    changed = dataclasses.replace(base, **change)
    assert changed.run_id != base.run_id
    assert dataclasses.replace(base, seed=2).run_id != base.run_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "Demo"},
        {"label": "demo\n"},
        {"label": "-demo"},
        {"label": "a" * 41},
        {"preset": ""},
        {"worlds_per_task": 1, "worlds": 4},
        {"worlds": 5},
        {"worlds": 0},
        {"worlds": True},
        {"seed": -1},
        {"seed": 1.0},
        {"seed": pub.MAX_SEED + 1},
        {"worlds": 2 * (pub.MAX_TASKS + 1)},
        {"worlds_per_task": pub.MAX_WORLDS_PER_TASK + 2, "worlds": pub.MAX_WORLDS_PER_TASK + 2},
    ],
)
def test_invalid_requests_are_rejected(kwargs):
    arguments = {"label": "demo", "preset": "small", "worlds": 6, "worlds_per_task": 2, "seed": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        pub.RunRequest(**arguments)


def test_the_largest_seed_keeps_every_task_seed_json_safe():
    request = pub.RunRequest("demo", "small", 2 * pub.MAX_TASKS, 2, pub.MAX_SEED)
    manifest = _manifest(request)
    assert request.task_seed(request.n_tasks - 1) <= pub.MAX_JSON_INT
    assert pub.decode_run_manifest(manifest.encode(), run_id=manifest.run_id) == manifest


def test_records_round_trip_exactly():
    manifest = _manifest()
    decoded = pub.decode_run_manifest(manifest.encode(), run_id=manifest.run_id)
    assert decoded == manifest
    # The digest written now is the digest a reader recomputes, even for integer keys.
    assert decoded.generator.config_sha256 == manifest.generator.config_sha256
    for accepted in (2, 0):
        record = _task(manifest, accepted=accepted)
        assert _decode_task(record.encode(), manifest) == record
    attempt = pub.AttemptRecord(
        run_id=manifest.run_id,
        attempt_id=ATTEMPT,
        state="running",
        started_at=NOW,
        updated_at=NOW,
        deadline_at=NOW + timedelta(minutes=20),
        current_task_index=2,
        published_task_ids=("task-00000",),
        reused_task_ids=(),
        error=None,
        worker=dataclasses.replace(WORKER, usd_per_hour=0.126),
    )
    assert pub.decode_attempt_record(attempt.encode(), manifest=manifest, attempt_id=ATTEMPT) == (
        attempt
    )


def test_zero_world_records_carry_the_whole_shortfall():
    manifest = _manifest()
    record = _decode_task(_task(manifest, accepted=0).encode(), manifest)
    assert (record.status, record.shortfall_worlds, record.payload) == ("zero", 2, None)
    assert record.content_key == (0, None)


def test_additive_minor_versions_are_accepted():
    manifest = _manifest()
    data = _edit(_task(manifest).encode(), contract_version="1.9", future_field={"x": 1})
    assert _decode_task(data, manifest).accepted_worlds == 2


def _attempt_record(manifest: pub.RunManifest, worker: pub.WorkerInfo) -> bytes:
    return pub.AttemptRecord(
        run_id=manifest.run_id,
        attempt_id=ATTEMPT,
        state="finished",
        started_at=NOW,
        updated_at=NOW + timedelta(minutes=25),
        deadline_at=None,
        current_task_index=None,
        published_task_ids=(),
        reused_task_ids=(),
        error=None,
        worker=worker,
    ).encode()


def test_records_from_before_the_worker_price_decode_as_unpriced():
    manifest = _manifest()
    body = json.loads(_attempt_record(manifest, WORKER))
    del body["worker"]["usd_per_hour"]
    body["contract_version"] = "1.0"
    decoded = pub.decode_attempt_record(
        json.dumps(body).encode(), manifest=manifest, attempt_id=ATTEMPT
    )
    assert decoded.worker == WORKER and decoded.worker.usd_per_hour is None


def _decoded_price(price: object) -> float | None:
    manifest = _manifest()
    body = json.loads(_attempt_record(manifest, WORKER))
    body["worker"]["usd_per_hour"] = price
    data = json.dumps(body).encode()
    return pub.decode_attempt_record(
        data, manifest=manifest, attempt_id=ATTEMPT
    ).worker.usd_per_hour


@pytest.mark.parametrize("price", [-0.01, "0.126", True, [0.126], pub.MAX_USD_PER_HOUR * 1.5])
def test_invalid_worker_prices_are_rejected(price):
    with pytest.raises(pub.RecordError, match="usd_per_hour"):
        _decoded_price(price)


def test_worker_prices_at_the_bounds_decode_as_non_negative_numbers():
    assert _decoded_price(pub.MAX_USD_PER_HOUR) == pub.MAX_USD_PER_HOUR
    zero = _decoded_price(-0.0)
    assert zero == 0.0 and math.copysign(1.0, zero) == 1.0  # never shown as "$-0.000/h"


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda data: data[: len(data) // 2], pub.UnreadableRecordError),
        (lambda data: b"\xff" + data, pub.UnreadableRecordError),
        (lambda data: _edit(data, contract_version="2.0"), pub.UnsupportedVersionError),
        (lambda data: _edit(data, contract_version="v1.0"), pub.RecordError),
        (lambda data: _edit(data, contract_version="1" * 5000 + ".0"), pub.RecordError),
        (lambda data: _edit(data, contract="something-else"), pub.RecordError),
        (lambda data: _edit(data, kind="attempt"), pub.RecordError),
        (lambda data: _edit(data, seed=3), pub.RecordError),
        (lambda data: _edit(data, seed=2.0), pub.RecordError),
        (lambda data: _edit(data, task_index=True), pub.RecordError),
        (lambda data: _edit(data, accounting={"draw_failures": 2**53}), pub.RecordError),
        (lambda data: _edit(data, accepted_worlds=1), pub.RecordError),
        (lambda data: _edit(data, status="complete", accepted_worlds=1), pub.RecordError),
        (lambda data: _edit(data, world_ids=["x", "y"]), pub.RecordError),
        (lambda data: _edit(data, run_id="other-0123456789ab"), pub.RecordError),
        (lambda data: b"[1, 2]", pub.RecordError),
        (lambda data: data.replace(b"1.75", b"NaN"), pub.RecordError),
        (lambda data: data.replace(b"1.75", b"1" + b"0" * 400), pub.RecordError),
    ],
)
def test_unusable_task_records_raise_record_errors(mutate, error):
    manifest = _manifest()
    with pytest.raises(error):
        _decode_task(mutate(_task(manifest).encode()), manifest)


def test_payload_paths_cannot_point_outside_the_publishing_attempt():
    manifest = _manifest()
    body = json.loads(_task(manifest).encode())
    for path in (
        pub.payload_path(manifest.run_id, "task-00000", ATTEMPT),
        f"runs/{manifest.run_id}/payloads/task-00001/../task-00000/{ATTEMPT}.npz",
    ):
        body["payload"]["path"] = path
        with pytest.raises(pub.RecordError):
            _decode_task(json.dumps(body).encode(), manifest)


def test_generation_failures_and_accounting_must_be_consistent():
    manifest = _manifest()
    body = json.loads(_task(manifest).encode())
    body["accounting"]["generation_failure"] = {"type": "RuntimeError", "message": "x"}
    with pytest.raises(pub.RecordError):
        _decode_task(json.dumps(body).encode(), manifest)
    body = json.loads(_task(manifest).encode())
    body["accounting"]["rejected_candidates"] = 99
    with pytest.raises(pub.RecordError):
        _decode_task(json.dumps(body).encode(), manifest)


def test_run_manifests_must_match_their_path_and_request():
    manifest = _manifest()
    data = manifest.encode()
    with pytest.raises(pub.RecordError):
        pub.decode_run_manifest(data, run_id="demo-000000000000")
    with pytest.raises(pub.RecordError):
        pub.decode_run_manifest(_edit(data, n_tasks=3.0), run_id=manifest.run_id)
    body = json.loads(data)
    body["generator"]["config"]["n_cells"] = 3
    with pytest.raises(pub.RecordError):
        pub.decode_run_manifest(json.dumps(body).encode(), run_id=manifest.run_id)
    body["generator"]["config"]["n_cells"] = 1e400
    with pytest.raises(pub.RecordError):
        pub.decode_run_manifest(json.dumps(body).encode(), run_id=manifest.run_id)


def test_messages_are_bounded_and_valid_utf8():
    message = pub.clean_message("bad \ud800 " + "x" * 5000)
    assert len(message) == pub.MAX_MESSAGE_CHARS
    message.encode("utf-8")
    error = pub.ErrorInfo.from_exception(RuntimeError("\udcff"))
    assert error.describe().encode("utf-8")


def test_timestamps_round_trip_at_millisecond_precision():
    text = pub.format_timestamp(NOW)
    assert text == "2026-01-01T12:00:00.123Z"
    assert pub.parse_timestamp(text) == NOW
    with pytest.raises(ValueError):
        pub.format_timestamp(datetime(2026, 1, 1))
    for bad in ("2026-01-01T12:00:00.123+02:00", "2026-01-01T12:00:00", 5):
        with pytest.raises(pub.RecordError):
            pub.parse_timestamp(bad)
