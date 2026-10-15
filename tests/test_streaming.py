"""Streaming orchestration contracts; no remote jobs are started by these tests."""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest

from live_dashboard import publication as pub
from live_dashboard.save_queue import stream_to_writer
from live_dashboard.streaming import StreamPlan, run_partition


def test_queue_backpressure_and_overlap():
    release = threading.Event()
    started = threading.Event()
    produced = []
    saved = []
    errors = []

    def produce():
        for n in range(20):
            produced.append(n)
            yield n

    def save(n):
        if n == 0:
            started.set()
            assert release.wait(3)
        saved.append(n)

    def run():
        try:
            stream_to_writer(produce(), save, capacity=2)
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    assert started.wait(3)
    # Writer owns one, queue owns two, producer can own at most one more.
    assert len(produced) <= 4
    release.set()
    t.join(3)
    assert not t.is_alive() and not errors
    assert saved == list(range(20))


def test_writer_failure_does_not_deadlock_full_queue():
    original = OSError("storage unavailable")

    def write(item):
        raise original

    with pytest.raises(RuntimeError, match="writer failed") as caught:
        stream_to_writer(range(100), write, capacity=1)
    assert caught.value.__cause__ is original


def test_producer_failure_drains_accepted_items():
    saved = []

    def produce():
        yield 1
        yield 2
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        stream_to_writer(produce(), saved.append, capacity=1)
    assert saved == [1, 2]


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_queue_capacity(capacity):
    with pytest.raises(ValueError):
        stream_to_writer([], lambda _: None, capacity=capacity)


def manifest(plan):
    return pub.RunManifest(
        plan.request.run_id,
        pub.utc_now(),
        plan.request,
        pub.GeneratorInfo(
            plan.request.preset, {"configuration": {}}, "test", 5, "0" * 64, None, {}
        ),
    )


class FakeStore:
    def __init__(self):
        self.records = {}
        self.thread_ids = set()

    def validated_outputs(self, run_id):
        return [{"record": r} for r in self.records.values()]

    def publish(self, metadata, corpus):
        self.thread_ids.add(threading.get_ident())
        key = metadata["task_id"]
        assert key not in self.records
        self.records[key] = {**metadata, "payload": None}
        return key

    def read_output(self, ref):
        return self.records[ref], None


class FakeGenerator:
    indices = []

    def __init__(self, plan, partition):
        self.plan = plan
        self.adapter = SimpleNamespace(compile_records=[{}])

    def generate(self, index):
        self.indices.append(index)
        return None, list(range(self.plan.batch_size)), self.plan.batch_size, 0


def test_partition_resume_skips_published_and_owns_disjoint_tasks():
    plan = StreamPlan("test-stream", worlds_per_split=30)
    store = FakeStore()
    m = manifest(plan)
    FakeGenerator.indices = []
    first = run_partition(
        plan, 0, m, pub.new_attempt_id(), store, factory=FakeGenerator, queue_size=1
    )
    assert first["batches_written"] == 3
    assert FakeGenerator.indices == [0, 2, 4]
    again = run_partition(plan, 0, m, pub.new_attempt_id(), store, factory=FakeGenerator)
    assert again["compilations"] == 0 and again["reused_batches"] == 3
    run_partition(plan, 1, m, pub.new_attempt_id(), store, factory=FakeGenerator)
    assert sorted(FakeGenerator.indices) == list(range(6))
    assert threading.get_ident() not in store.thread_ids


def test_stop_drains_queue_then_resume_completes():
    plan = StreamPlan("stop-stream", worlds_per_split=30)
    store = FakeStore()
    FakeGenerator.indices = []
    result = run_partition(
        plan,
        0,
        manifest(plan),
        pub.new_attempt_id(),
        store,
        factory=FakeGenerator,
        stop=lambda: len(FakeGenerator.indices) >= 1,
    )
    assert result["stopped"] and result["batches_written"] == 1
    result = run_partition(
        plan, 0, manifest(plan), pub.new_attempt_id(), store, factory=FakeGenerator
    )
    assert result["reused_batches"] == 1 and result["batches_written"] == 2


def test_manifest_mismatch_rejected():
    p = StreamPlan("mismatch")
    with pytest.raises(ValueError, match="manifest"):
        run_partition(p, 0, manifest(dataclasses.replace(p, seed=9)), "first", FakeStore())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worlds_per_split": 11},
        {"batch_size": 1},
        {"families": ()},
        {"families": (("bad", "geometric"),)},
    ],
)
def test_bad_plan(kwargs):
    with pytest.raises(ValueError):
        StreamPlan("invalid", **kwargs)


def test_200k_plan_is_lazy_and_balanced():
    p = StreamPlan("large", worlds_per_split=100_000)
    assert p.request.worlds == 200_000
    assert isinstance(p.task_indices(0), range)
    assert len(p.task_indices(0)) == len(p.task_indices(1)) == 10_000


@pytest.mark.slow
def test_real_compilation_reuse_replay_publication_and_dashboard(tmp_path):
    import numpy as np

    from live_dashboard.collection import Collector
    from live_dashboard.storage import DirectoryStore
    from live_dashboard.stream_store import StreamPublicationStore
    from live_dashboard.streaming import FamilyGenerator, prepare_run

    plan = StreamPlan("real-stream", worlds_per_split=20)
    directory = DirectoryStore(tmp_path)
    m = prepare_run(directory, plan)
    with pytest.raises(ValueError, match="confirmation"):
        prepare_run(directory, plan, resume=True)
    assert prepare_run(directory, plan, resume=True, previous_stopped=True) == m
    with pytest.raises(ValueError, match="mismatch"):
        prepare_run(
            directory,
            dataclasses.replace(plan, families=tuple(reversed(plan.families))),
            resume=True,
            previous_stopped=True,
        )
    captured = []

    def factory(p, i):
        engine = FamilyGenerator(p, i)
        captured.append(engine)
        return engine

    store = StreamPublicationStore(tmp_path)
    for partition in range(2):
        result = run_partition(
            plan, partition, m, pub.new_attempt_id(), store, factory=factory, queue_size=1
        )
        assert result["compilations"] == 1 and result["published_worlds"] == 20
    reader = Collector(directory)
    reader.refresh()
    assert reader.snapshot.last_error is None
    run = reader.snapshot.runs[0]
    assert len(run.worlds) == 40 and not run.issues
    assert all(t.status == "published" for t in run.tasks)
    # Replay an earlier logical batch after intervening chunks on the SAME compiled model.
    outputs = store.validated_outputs(m.run_id)
    for partition, engine in enumerate(captured):
        previous = next(o for o in outputs if o["record"]["task_id"] == pub.task_id_for(partition))
        _, original = store.read_output(previous["reference"])
        replay, _, _, _ = engine.generate(partition)
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == replay[key].dtype
                assert value.tobytes() == replay[key].tobytes(), key
        assert len(engine.adapter.compile_records) == 1
        assert len(engine.adapter.all_primary_hashes) <= plan.batch_size
        assert len(engine.adapter.executions) <= plan.batch_size
    for partition in range(2):
        assert run_partition(plan, partition, m, pub.new_attempt_id(), store)["compilations"] == 0
