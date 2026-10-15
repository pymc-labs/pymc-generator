"""Deferred membership keeps immutable worlds independent of training partitions."""

import json
from dataclasses import replace

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator.publication import FilePublicationStore, PublicationError
from pymc_generator.splitting import assign_cell_split


@pytest.fixture(scope="module")
def prior():
    return pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=2,
        seed=7,
    )


def assert_arrays_equal(left, right):
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == right[key].dtype
            assert value.shape == right[key].shape
            assert value.tobytes() == right[key].tobytes(), key


def test_default_matches_explicit_legacy_and_rejects_one(prior):
    default = pg.DataGenerator(prior).generate()
    explicit = pg.DataGenerator(prior).generate(defer_cell_split=False)
    assert_arrays_equal(default, explicit)
    assert "cell_split" not in default["diagnostics"]
    with pytest.raises(ValueError, match="n=1"):
        pg.DataGenerator(prior).generate(n_tasks=1)


def test_single_world_deferred_is_explicit_and_valid(prior):
    corpus = pg.DataGenerator(prior).generate(n_tasks=1, defer_cell_split=True)
    assert corpus["is_val"].tolist() == [2]
    assert corpus["diagnostics"]["cell_split"] == {"mode": "deferred", "version": 1}
    assert not pg.DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
    assert any("deferred" in error for error in pg.DataGenerator.validate_corpus(corpus))
    corpus["is_val"][:] = 0
    assert any(
        "unassigned" in error
        for error in pg.DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
    )


def test_deferred_arrays_masks_and_normalization_ignore_validation_fraction(prior):
    prior = replace(prior, n_cells=4, p_long_horizon=0.0)
    left = pg.DataGenerator(prior).generate(defer_cell_split=True)
    right = pg.DataGenerator(replace(prior, val_cell_frac=0.8)).generate(defer_cell_split=True)
    assert_arrays_equal(left, right)
    assert np.all(left["is_val"] == 2)
    assert np.all(left["is_future"] == 0)  # No validation-only horizon repair.


def test_grouped_assignment_is_order_independent_and_leakage_safe():
    cells = {
        "a": ("grid1", 0),
        "b": ("grid1", 0),
        "c": ("grid1", 1),
        "d": ("grid2", 0),
        "e": ("grid2", 1),
    }
    result = assign_cell_split(cells, validation_fraction=0.5, seed=10)
    assert result == assign_cell_split(
        dict(reversed(list(cells.items()))), validation_fraction=0.5, seed=10
    )
    assert result["a"] == result["b"]
    assert set(result.values()) == {"train", "validation"}
    assert len({result[key] for key in ("a", "c", "d", "e")}) == 2


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), True])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        assign_cell_split({"a": ("grid", 0), "b": ("grid", 1)}, validation_fraction=fraction)


def test_deferred_save_and_one_world_batches(tmp_path, prior):
    generator = pg.DataGenerator(prior)
    path = tmp_path / "one.npz"
    generated = generator.generate_and_save(path, n_tasks=1, defer_cell_split=True)
    from pymc_generator.data_generator import load_corpus

    loaded = load_corpus(path)
    assert_arrays_equal(generated, loaded)
    assert loaded["diagnostics"]["cell_split"]["mode"] == "deferred"
    batches = list(generator.iter_batches(2, batch_size=1, defer_cell_split=True))
    assert [batch["is_val"].tolist() for batch in batches] == [[2], [2]]


def test_one_cell_cannot_be_split_without_leakage():
    with pytest.raises(ValueError, match="two distinct cells"):
        assign_cell_split({"a": ("grid", 0), "b": ("grid", 0)})


@pytest.mark.parametrize("second_namespace", ["grid2", "grid1"])
def test_publish_single_world_then_split_snapshot(tmp_path, prior, second_namespace):
    store = FilePublicationStore(tmp_path)
    references = []
    for task, namespace in [("task1", "grid1"), ("task2", second_namespace)]:
        grid_prior = replace(prior, seed=8) if namespace == "grid2" else prior
        corpus = pg.DataGenerator(grid_prior).generate(n_tasks=1, defer_cell_split=True)
        metadata = {
            "run_id": "run1",
            "instruction_id": "instruction1",
            "task_id": task,
            "attempt_id": "first",
            "output_id": f"output_{task}",
            "world_ids": [task],
            "counts": {
                "target": 2,
                "candidates": 1,
                "rejected": 0,
                "surplus": 0,
                "published": 1,
                "shortfall": 1,
            },
            "rejection_reasons": {},
            "provenance": {
                "run_spec_sha256": "a" * 64,
                "generation_mode": "ordinary-deferred",
                "generation_version": "test",
                "environment": {"lock": "test"},
                "seed_context": {"seed": grid_prior.seed},
                "effective_configuration": {"test": True},
                "cell_namespace": namespace,
            },
        }
        references.append(store.publish(metadata, corpus))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if second_namespace == "grid1":
        with pytest.raises(ValueError, match="two distinct cells"):
            store.freeze_snapshot("run1", "training", validation_fraction=0.5)
        assert not (tmp_path / "snapshots/training.json").exists()
        return
    snapshot = store.freeze_snapshot("run1", "training", validation_fraction=0.5, split_seed=3)
    assert len(store.read_snapshot(snapshot)) == 2
    split = json.loads((tmp_path / snapshot).read_text())["cell_split"]
    assert set(split["membership"].values()) == {"train", "validation"}
    assert store.read_training_split(snapshot) == split["membership"]
    for path, contents in before.items():
        assert path.read_bytes() == contents
    for reference in references:
        assert store.read_output(reference)[1]["is_val"].tolist() == [2]
    value = json.loads((tmp_path / snapshot).read_text())
    value["cell_split"]["membership"]["task1"] = "incorrect"
    (tmp_path / snapshot).write_text(json.dumps(value))
    with pytest.raises(PublicationError, match="Frozen cell split"):
        store.read_snapshot(snapshot)
