"""Streaming orchestration contracts; no remote jobs are started by these tests."""

from __future__ import annotations

import dataclasses
import json
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from live_dashboard import publication as pub
from live_dashboard import structural
from live_dashboard.save_queue import stream_to_writer
from live_dashboard.storage import DirectoryStore
from live_dashboard.streaming import (
    Batch,
    StreamPlan,
    drop_negative_controls,
    prepare_run,
    run_partition,
)

DESIGN = "structural-splits"
DESIGN_FAMILIES = tuple(
    (split.saturation, split.adstock) for split in structural.DESIGNS[DESIGN].splits
)


def design_plan(label="design-stream", **kwargs):
    """The 1,000-world structural-splits run: 10 splits x 5 regimes, 20 worlds per member."""
    return StreamPlan.for_design(
        label,
        DESIGN,
        **{"worlds_per_split": 100, "batch_size": 10, "seed": 20261015120000, **kwargs},
    )


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

    def validated_task_outputs(self, run_id, task_ids):
        return [{"record": self.records[task]} for task in task_ids if task in self.records]

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
        return Batch(None, list(range(self.plan.batch_size)), self.plan.batch_size, 0)


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
    ("kwargs", "match"),
    [
        ({"worlds_per_split": 11}, "multiple of batch size"),
        ({"batch_size": 1}, "batch size"),
        ({"families": ()}, "nonempty"),
        ({"families": (("bad", "geometric"),)}, "unknown saturation"),
        ({"families": DESIGN_FAMILIES, "design": "no-such-design"}, "unknown design"),
        # A design's families are its splits, in design order.
        ({"design": DESIGN}, "its splits"),
        ({"families": DESIGN_FAMILIES[::-1], "design": DESIGN}, "its splits"),
        # Three tasks per split cannot cover its five regimes evenly.
        ({"families": DESIGN_FAMILIES, "design": DESIGN, "worlds_per_split": 30}, "regimes"),
        ({"compile_mode": "numba"}, "compile mode"),
    ],
)
def test_bad_plan(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StreamPlan("invalid", **kwargs)


def test_200k_plan_is_lazy_and_balanced():
    p = StreamPlan("large", worlds_per_split=100_000)
    assert p.request.worlds == 200_000
    assert isinstance(p.task_indices(0), range)
    assert len(p.task_indices(0)) == len(p.task_indices(1)) == 10_000


def test_design_plan_gives_every_task_its_legacy_member_and_split_seed():
    plan = design_plan()
    design = structural.DESIGNS[DESIGN]
    assert plan.request.worlds == 1000 and plan.request.n_tasks == 100
    # The reader must not apply the legacy design's seed and redraw contract.
    assert plan.request.design is None
    tasks = Counter()
    for partition, member in enumerate(plan.partitions):
        split = design.split_of(partition)
        assert plan.family(partition) == (split.saturation, split.adstock)
        assert plan.root_seed(partition) == split.seed(plan.seed)
        for index in plan.task_indices(partition):
            assert design.member_of(index) == member
            tasks[index] += 1
    assert sorted(tasks) == list(range(100)) and set(tasks.values()) == {1}
    assert Counter(plan.partitions[index % 50] for index in tasks) == dict.fromkeys(
        design.members, 2
    )


def test_design_manifest_is_readable_and_records_member_configs_and_split_seeds(tmp_path):
    from live_dashboard.generation import DESIGN_PRESETS

    plan = design_plan()
    store = DirectoryStore(tmp_path)
    written = prepare_run(store, plan)
    manifest = pub.decode_run_manifest(
        store.read_bytes(pub.run_manifest_path(written.run_id)), run_id=written.run_id
    )
    design = structural.DESIGNS[DESIGN]
    config = manifest.generator.config
    assert manifest.generator.preset == plan.request.preset == "stream-structural-splits"
    assert config["structural_splits"] == [split.to_dict(plan.seed) for split in design.splits]
    assert config["cycle"] == list(design.cycle)
    for partition, member in enumerate(plan.partitions):
        legacy = dataclasses.asdict(DESIGN_PRESETS[member](plan.batch_size))
        legacy.update(seed=plan.root_seed(partition), record_walk_parameters=False)
        assert config["presets"][member] == json.loads(pub.canonical_json(legacy)), member


class FakeDesignGenerator(FakeGenerator):
    def generate(self, index):
        n = self.plan.batch_size
        # Two candidates failed the generator's filters, one had a negative control.
        return Batch(None, list(range(n)), n + 3, 2, negative_controls=1, draw_failures=1)


def test_design_partition_records_member_split_regime_and_negative_controls(tmp_path):
    plan = design_plan()
    m = prepare_run(DirectoryStore(tmp_path), plan)
    design = structural.DESIGNS[DESIGN]
    store = FakeStore()
    # michaelis-menten-geometric-hectic-baseline: a partition past the ten families, and
    # the longest member name.
    partition = 47
    result = run_partition(
        plan, partition, m, pub.new_attempt_id(), store, factory=FakeDesignGenerator
    )
    assert result["batches_written"] == 2
    for index in plan.task_indices(partition):
        record = store.records[pub.task_id_for(index)]
        effective = record["provenance"]["effective_configuration"]
        assert effective["preset"] == design.member_of(index) == plan.partitions[partition]
        assert effective["configuration"] == m.generator.config["presets"][effective["preset"]]
        assert effective["family"] == ["michaelis_menten", "geometric"]
        context = record["provenance"]["seed_context"]
        assert {"split": context["split"], "regime": context["regime"]} == design.seed_context(
            plan.seed, index
        )
        assert context["family_root_seed"] == design.split_of(index).seed(plan.seed)
        assert record["counts"] == {
            "target": 10,
            "candidates": 13,
            "rejected": 3,
            "surplus": 0,
            "published": 10,
            "shortfall": 0,
        }
        assert record["rejection_reasons"] == {"generator_filters": 2, "negative_controls": 1}
        assert record["draw_failures"] == 1


def test_worlds_with_a_negative_control_are_dropped_with_their_candidate_ids():
    import numpy as np

    rows, weeks = 4, 3
    corpus = {
        "covariate_active_mask": np.array([[1, 0]] * rows, dtype=np.uint8),
        "covariates": np.ones((rows, weeks, 2), dtype=np.float32),
        "covariate_contribution": np.zeros((rows, weeks, 2), dtype=np.float32),
        "cell_id": np.arange(rows),
        "diagnostics": {"schema_version": 5},
    }
    corpus["covariates"][1, 2, 0] = -0.5  # a control below zero
    corpus["covariates"][2, 0, 1] = -9.0  # an inactive slot is no control
    corpus["covariate_contribution"][3, 1, 0] = np.nan  # NaN counts as negative
    kept, ids, dropped = drop_negative_controls(corpus, [10, 11, 12, 13])
    assert dropped == 2 and ids == [10, 12]
    assert kept["cell_id"].tolist() == [0, 2] and kept["covariates"].shape == (2, weeks, 2)
    assert kept["diagnostics"] == {"schema_version": 5}
    only_negative = {k: v[1:2] if isinstance(v, np.ndarray) else v for k, v in corpus.items()}
    assert drop_negative_controls(only_negative, [11]) == (None, [], 1)
    with pytest.raises(ValueError, match="rows"):
        drop_negative_controls(corpus, [10, 11])


def test_design_task_rejects_negative_control_worlds_before_taking_its_target(monkeypatch):
    import numpy as np

    from live_dashboard._specialized import packing
    from live_dashboard.streaming import FamilyGenerator
    from pymc_generator import data_generator, sampler

    negative_rows = {0: [1, 4], 1: [0]}  # per allocation: rows with a control below zero
    calls = []

    def generate_fixed(cfg, adapter):
        allocation = len(calls)
        calls.append(allocation)
        # The generator's filters reject cell 7 of the first ten candidates.
        cells = [cell for cell in range(cfg.n_cells) if (allocation, cell) != (0, 7)]
        adapter.evaluated, adapter.rejected = cfg.n_cells, cfg.n_cells - len(cells)
        # One candidate of the first ten failed to draw: the task still fills.
        adapter.draw_failures = 1 - allocation
        adapter.last_draw_error = "FloatingPointError('overflow')"
        adapter.accepted_ids = [(cell, 0) for cell in cells]
        covariates = np.ones((len(cells), 3, 2), dtype=np.float32)
        covariates[negative_rows[allocation], 0, 0] = -1.0
        return {
            "covariate_active_mask": np.ones((len(cells), 2), dtype=np.uint8),
            "covariates": covariates,
            "covariate_contribution": np.zeros((len(cells), 3, 2), dtype=np.float32),
            "cell_id": np.arange(len(cells)),
            "diagnostics": {},
        }

    monkeypatch.setattr(packing, "generate_fixed", generate_fixed)
    monkeypatch.setattr(sampler, "_finalize_corpus", lambda corpus, cfg: corpus)
    monkeypatch.setattr(
        data_generator.DataGenerator, "validate_corpus", staticmethod(lambda corpus, **kw: [])
    )
    engine = FamilyGenerator(design_plan(), 0)
    engine.adapter.compile_records.append({})  # its one compiled model
    batch = engine.generate(3)
    first, second = 3 * 20, 3 * 20 + 10
    # Seven worlds survive the first ten candidates; the next ten fill the remaining three
    # after dropping their negative first world.
    assert batch.candidate_ids == [first + c for c in (0, 2, 3, 5, 6, 8, 9)] + [
        second + c for c in (1, 2, 3)
    ]
    assert (batch.evaluated, batch.rejected, batch.negative_controls) == (20, 1, 3)
    assert batch.draw_failures == 1 and batch.generation_failure is None
    assert batch.corpus["cell_id"].tolist() == list(range(10))
    assert (batch.corpus["covariates"] >= 0).all()


def test_a_task_without_worlds_reports_why_drawing_failed(monkeypatch):
    from live_dashboard._specialized import packing
    from live_dashboard.streaming import FamilyGenerator

    def generate_fixed(cfg, adapter):
        # Four candidates of every ten fail to draw; the generator rejects the others.
        adapter.evaluated = adapter.rejected = cfg.n_cells - 4
        adapter.draw_failures, adapter.last_draw_error = 4, "FloatingPointError('overflow')"
        adapter.accepted_ids = []
        return None

    monkeypatch.setattr(packing, "generate_fixed", generate_fixed)
    engine = FamilyGenerator(design_plan(), 0)
    engine.adapter.compile_records.append({})
    batch = engine.generate(0)
    assert batch.corpus is None and batch.candidate_ids == []
    assert (batch.evaluated, batch.rejected, batch.draw_failures) == (12, 12, 8)
    assert batch.generation_failure.type == "DrawFailure"
    assert "overflow" in batch.generation_failure.message


def _fail_inside_a_node():
    """Raise a numeric error as PyTensor reports a failing node (the library retries it)."""
    scope = {"__name__": "pytensor.link.utils"}
    exec("def raise_with_op():\n    raise FloatingPointError('overflow in a node')", scope)
    scope["raise_with_op"]()


def test_draw_failures_skip_candidates_but_a_task_that_never_draws_stops_the_worker():
    from live_dashboard.streaming import FamilyGenerator

    engine = FamilyGenerator(design_plan(), 0)
    engine.adapter.build = lambda *args, **kwargs: (None, (), ())
    engine.adapter.draw = lambda *args, **kwargs: _fail_inside_a_node()
    with pytest.raises(RuntimeError, match="all 20 candidates of task 0 .*overflow in a node"):
        engine.generate(0)

    def bug(*args, **kwargs):
        raise FloatingPointError("outside any node")

    # Errors outside the model's nodes are bugs, never skipped.
    engine.adapter.draw = bug
    with pytest.raises(FloatingPointError, match="outside any node"):
        engine.generate(0)


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
        replay = engine.generate(partition).corpus
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == replay[key].dtype
                assert value.tobytes() == replay[key].tobytes(), key
        assert len(engine.adapter.compile_records) == 1
        assert len(engine.adapter.all_primary_hashes) <= plan.batch_size
        assert len(engine.adapter.executions) <= plan.batch_size
    for partition in range(2):
        assert run_partition(plan, partition, m, pub.new_attempt_id(), store)["compilations"] == 0


def _ordinary_parity(engine, cell):
    """The ordinary model, given the template's primitive draws for ``cell``, reproduces
    the template's outputs and reports (tests/test_world_model_template.py's check)."""
    import numpy as np
    from pytensor import function
    from pytensor.graph.replace import clone_replace

    from pymc_generator.sampler import _slice_g_active
    from pymc_generator.world_model import build_world_model

    adapter = engine.adapter
    model, outputs, reports = adapter.models[0]
    adapter.current_cell = cell
    flags = set(adapter.flag_names)
    primitives = tuple(rv.name for rv in model.free_RVs if rv.name not in flags)
    drawn = adapter.draw(tuple(dict.fromkeys((*outputs, *reports, *primitives))), 1)
    g = _slice_g_active(adapter.padded_g(cell), *adapter.counts(cell))
    ordinary, names, params = build_world_model(
        g, engine.cfg, adapter.cells[cell].structure, engine.cfg.n_time_steps
    )
    assert set(names) <= set(drawn)
    compared = (*names, *(name for name in params if name in drawn))
    given = {rv: rv.type(name=f"given_{rv.name}") for rv in ordinary.free_RVs}
    evaluate = function(
        list(given.values()),
        clone_replace([ordinary[name] for name in compared], replace=given, rebuild_strict=False),
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    values = []
    for rv in given:
        shape = tuple(int(n) for n in rv.shape.eval())
        value = np.asarray(drawn[rv.name][0], dtype=rv.dtype)
        values.append(value[tuple(slice(0, n) for n in shape)] if shape else value)
    for name, expected in zip(compared, evaluate(*values), strict=True):
        expected = np.asarray(expected)
        actual = np.asarray(drawn[name][0])[tuple(slice(0, n) for n in expected.shape)]
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11, err_msg=name)


@pytest.mark.slow
@pytest.mark.parametrize("compile_mode", ["cvm", "python"])
def test_real_design_members_keep_split_families_static_flags_and_positive_controls(
    tmp_path, compile_mode
):
    import numpy as np

    from live_dashboard.collection import Collector
    from live_dashboard.stream_store import StreamPublicationStore
    from live_dashboard.streaming import COMPILE_MODES, FamilyGenerator
    from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
    from pymc_generator.slots import TRAJECTORY_COMPONENTS

    # One two-world task per member. Hill x geometric with texture off (calm) and with
    # jitter and pulses on every input (hectic): probabilities 0 and 1 build no flag input.
    plan = design_plan(
        "real-design-stream", worlds_per_split=10, batch_size=2, compile_mode=compile_mode
    )
    m = prepare_run(DirectoryStore(tmp_path), plan)
    store = StreamPublicationStore(tmp_path)
    engines = {}

    def factory(p, i):
        engines[i] = FamilyGenerator(p, i)
        return engines[i]

    texture = [TRAJECTORY_COMPONENTS.index(c) for c in ("hf", "pulse")]
    for member, textured in (("hill-geometric-calm", False), ("hill-geometric-hectic", True)):
        partition = plan.partitions.index(member)
        result = run_partition(
            plan, partition, m, pub.new_attempt_id(), store, factory=factory, queue_size=1
        )
        assert result["compilations"] == 1 and result["published_worlds"] == 2
        (record,) = engines[partition].adapter.compile_records
        assert record["mode"] == COMPILE_MODES[compile_mode]
        assert record["c_thunks"] == (compile_mode == "cvm")
        (index,) = plan.task_indices(partition)
        (output,) = store.validated_task_outputs(m.run_id, [pub.task_id_for(index)])
        _, corpus = store.read_output(output["reference"])
        channels = corpus["treatment_active_mask"] == 1
        controls = corpus["covariate_active_mask"] == 1
        assert (corpus["sat_family"][channels] == SATURATION_FAMILY_KEYS.index("hill")).all()
        assert (
            corpus["carryover_family"][channels] == CARRYOVER_FAMILY_KEYS.index("geometric")
        ).all()
        assert (np.where(controls[:, None, :], corpus["covariates"], 0) >= 0).all()
        assert (corpus["covariate_contribution"] >= 0).all()
        for role, active in (("treatment", channels), ("covariate", controls)):
            assert (corpus[f"{role}_components"][..., texture][active] == textured).all(), role
        engine = engines[partition]
        replay = engine.generate(index).corpus
        for key, value in corpus.items():
            if isinstance(value, np.ndarray):
                assert value.tobytes() == replay[key].tobytes(), key
        _ordinary_parity(engine, engine.adapter.accepted_ids[0][0])
    reader = Collector(DirectoryStore(tmp_path))
    reader.refresh()
    assert reader.snapshot.last_error is None
    (run,) = reader.snapshot.runs
    assert len(run.worlds) == 4 and not run.issues
