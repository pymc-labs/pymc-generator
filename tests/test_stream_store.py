from __future__ import annotations

import json

import pytest

from live_dashboard import publication as pub
from live_dashboard.stream_store import StreamPublicationStore
from pymc_generator.publication import PublicationError, PublicationUnconfirmedError


def metadata(index=0, attempt=None):
    req = pub.RunRequest("stream-store", "stream-composable-v1", 20, 10, 1)
    now = pub.utc_now()
    m = pub.RunManifest(
        req.run_id, now, req, pub.GeneratorInfo(req.preset, {}, "test", 5, "0" * 64, None, {})
    )
    return pub.completion_metadata(
        m,
        task_index=index,
        member="hill-geometric",
        member_config={},
        attempt_id=attempt or pub.new_attempt_id(),
        worker=pub.WorkerInfo("test", 1, None, None),
        candidates=0,
        rejected=0,
        published=0,
        started_at=now,
        finished_at=now,
        generation_seconds=0,
        draw_failures=0,
        generation_failure=None,
    )


def test_zero_world_completion_and_scoped_discovery(tmp_path):
    store = StreamPublicationStore(tmp_path)
    first = metadata()
    ref = store.publish(first, None)
    record, data = store.read_output(ref)
    assert data is None and record["counts"]["shortfall"] == 10
    # A different task can publish despite an unrelated malformed record. Full
    # run validation (used before resume) still reports the corruption.
    (tmp_path / ref).write_text("{}")
    second = metadata(1)
    store.publish(second, None)
    with pytest.raises(PublicationError):
        store.validated_outputs(first["run_id"])
    with pytest.raises(PublicationError):
        store.publish(metadata(0), None)


def test_task_prefixed_identity_required(tmp_path):
    store = StreamPublicationStore(tmp_path)
    meta = metadata()
    meta["output_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        store.publish(meta, None)
    meta = metadata()
    meta["world_ids"] = ["foreign-world"]
    with pytest.raises(ValueError, match="task-prefixed"):
        store.publish(meta, None)


def test_failed_commit_is_not_success_and_keeps_record_for_reconciliation(tmp_path):
    class Volume:
        def reload(self):
            pass

        def commit(self):
            raise OSError("commit acknowledgement lost")

    store = StreamPublicationStore(tmp_path, Volume())
    meta = metadata()
    with pytest.raises(PublicationUnconfirmedError):
        store.publish(meta, None)
    records = list(tmp_path.rglob("completion.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["task_id"] == meta["task_id"]
    assert store._writing_task is None
