"""Local publication contract; no Modal credentials or paid execution."""

import copy
import json

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator.publication import FilePublicationStore, PublicationError


@pytest.fixture(scope="module")
def corpus():
    config = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        seed=7,
    )
    return pg.DataGenerator(config).generate()


def metadata(task="task1", attempt="attempt1", n=2):
    return {
        "run_id": "run1",
        "instruction_id": "instruction1",
        "task_id": task,
        "attempt_id": attempt,
        "output_id": f"output_{task}",
        "world_ids": [f"{task}_candidate{i}" for i in range(n)],
        "counts": {
            "target": 2,
            "candidates": 2,
            "rejected": 2 - n,
            "surplus": 0,
            "published": n,
            "shortfall": 2 - n,
        },
        "rejection_reasons": {"gate": 2 - n},
        "provenance": {
            "run_spec_sha256": "a" * 64,
            "generation_mode": "ordinary",
            "generation_version": "test",
            "environment": {"lock": "test"},
            "seed_context": {"seed": 7},
            "effective_configuration": {"fixture": True},
        },
    }


def test_generation_persistence_discovery_and_snapshot(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(), corpus)
    candidates, problems = store.discover("run1")
    assert not problems
    assert [item["reference"] for item in candidates] == [reference]
    record, loaded = store.read_output(reference)
    assert record["counts"]["published"] == 2
    for key, value in corpus.items():
        if isinstance(value, np.ndarray):
            assert loaded[key].dtype == value.dtype
            assert loaded[key].shape == value.shape
            assert loaded[key].tobytes() == value.tobytes()
    snapshot = store.freeze_snapshot("run1", "training1")
    store.publish(metadata("task2"), corpus)
    assert len(store.read_snapshot(snapshot)) == 1
    assert len(store.validated_outputs("run1")) == 2
    with pytest.raises(FileExistsError):
        store.freeze_snapshot("run1", "training1")


def test_zero_world_completion(tmp_path):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(n=0), None)
    record, loaded = store.read_output(reference)
    assert loaded is None
    assert record["counts"]["shortfall"] == 2
    assert not list(tmp_path.rglob("*.npz"))
    assert store.read_snapshot(store.freeze_snapshot("run1", "zero"))[0][1] is None


def test_incomplete_payload_not_discovered_and_retry_is_deduplicated(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    abandoned = tmp_path / "runs/run1/tasks/task1/attempts/crashed"
    abandoned.mkdir(parents=True)
    (abandoned / "corpus.npz").write_bytes(b"interrupted")
    assert store.discover("run1") == ([], {})
    first = store.publish(metadata(), corpus)
    assert store.publish(metadata(attempt="retry"), corpus) == first
    assert len(store.validated_outputs("run1")) == 1


def test_partial_completion_is_reported_not_silently_skipped(tmp_path):
    store = FilePublicationStore(tmp_path)
    path = tmp_path / "runs/run1/tasks/task1/attempts/crashed/completion.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"version":')
    candidates, problems = store.discover("run1")
    assert not candidates and problems
    with pytest.raises(PublicationError, match="Unresolved"):
        store.freeze_snapshot("run1", "bad")


def test_out_of_order_arrivals(tmp_path):
    store = FilePublicationStore(tmp_path)
    store.publish(metadata("task9", n=0), None)
    store.publish(metadata("task1", n=0), None)
    assert {item["record"]["task_id"] for item in store.validated_outputs("run1")} == {
        "task1",
        "task9",
    }


def test_snapshot_missing_payload_fails(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(), corpus)
    snapshot = store.freeze_snapshot("run1", "training")
    (tmp_path / reference).with_name("corpus.npz").unlink()
    with pytest.raises(FileNotFoundError):
        store.read_snapshot(snapshot)


def test_corrupt_payload_fails(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(), corpus)
    (tmp_path / reference).with_name("corpus.npz").write_bytes(b"corrupt")
    with pytest.raises(PublicationError, match="integrity"):
        store.read_output(reference)


def test_conflicting_retry_is_not_published(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    store.publish(metadata(), corpus)
    different = metadata(attempt="retry")
    different["provenance"]["seed_context"] = {"seed": 8}
    with pytest.raises(PublicationError, match="Conflicting retry"):
        store.publish(different, corpus)
    assert len(list(tmp_path.rglob("completion.json"))) == 1


def test_duplicate_completed_retry_and_conflict(tmp_path):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(n=0), None)
    record = json.loads((tmp_path / reference).read_text())
    record["attempt_id"] = "retry"
    path = (tmp_path / reference).parent.parent / "retry/completion.json"
    path.parent.mkdir()
    path.write_text(json.dumps(record))
    assert len(store.validated_outputs("run1")) == 1
    record["provenance"]["seed_context"] = {"seed": 8}
    path.write_text(json.dumps(record))
    with pytest.raises(PublicationError, match="Conflicting results"):
        store.validated_outputs("run1")
    with pytest.raises(PublicationError, match="Conflicting retry"):
        store.publish(metadata(attempt="retry2", n=0), None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(task_id="../escape"),
        lambda m: m["counts"].update(published=1),
        lambda m: m["counts"].update(rejected=True),
        lambda m: m["provenance"].pop("environment"),
    ],
)
def test_invalid_metadata_rejected(tmp_path, mutate):
    store = FilePublicationStore(tmp_path)
    value = copy.deepcopy(metadata(n=0))
    mutate(value)
    with pytest.raises((PublicationError, KeyError)):
        store.publish(value, None)
    assert not list(tmp_path.rglob("completion.json"))


def test_version_and_additive_fields(tmp_path):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(n=0), None)
    path = tmp_path / reference
    record = json.loads(path.read_text())
    record["optional_future_field"] = "ignored"
    path.write_text(json.dumps(record))
    store.read_output(reference)
    record["version"] = 2
    path.write_text(json.dumps(record))
    assert store.discover("run1")[1]
    with pytest.raises(PublicationError, match="Unsupported"):
        store.read_output(reference)


def test_changed_scientific_arrays_are_a_conflicting_retry(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    store.publish(metadata(), corpus)
    config = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        seed=8,
    )
    different = pg.DataGenerator(config).generate()
    with pytest.raises(PublicationError, match="Conflicting retry"):
        store.publish(metadata(attempt="retry"), different)
    assert len(list(tmp_path.rglob("completion.json"))) == 1


def test_reference_cannot_escape_store(tmp_path):
    with pytest.raises(PublicationError, match="escapes"):
        FilePublicationStore(tmp_path).read_output("../outside.json")


def test_unrelated_missing_payload_does_not_block_publication(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(), corpus)
    (tmp_path / reference).with_name("corpus.npz").unlink()
    other = store.publish(metadata("task2", n=0), None)
    assert store.read_output(other)[1] is None
    with pytest.raises(FileNotFoundError):
        store.publish(metadata(attempt="retry"), corpus)


def test_snapshot_rejects_changed_record_and_duplicate_members(tmp_path):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(n=0), None)
    snapshot = store.freeze_snapshot("run1", "training")
    path = tmp_path / snapshot
    value = json.loads(path.read_text())
    value["members"].append(value["members"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(PublicationError, match="Duplicate"):
        store.read_snapshot(snapshot)
    value["members"].pop()
    path.write_text(json.dumps(value))
    completion = tmp_path / reference
    record = json.loads(completion.read_text())
    record["provenance"]["generation_version"] = "changed"
    completion.write_text(json.dumps(record))
    with pytest.raises(PublicationError, match="changed"):
        store.read_snapshot(snapshot)


def test_wrong_payload_world_count_rejected(tmp_path, corpus):
    store = FilePublicationStore(tmp_path)
    value = metadata(n=1)
    with pytest.raises(PublicationError, match="world count"):
        store.publish(value, corpus)
    assert not list(tmp_path.rglob("completion.json"))


def test_missing_completion_record_does_not_shrink_snapshot(tmp_path):
    store = FilePublicationStore(tmp_path)
    reference = store.publish(metadata(n=0), None)
    snapshot = store.freeze_snapshot("run1", "training")
    (tmp_path / reference).unlink()
    with pytest.raises(FileNotFoundError):
        store.read_snapshot(snapshot)
