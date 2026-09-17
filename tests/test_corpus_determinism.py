"""Reproducibility of the persisted artifact, plus the guards that protect it.

A corpus is supposed to be a pure function of ``(config, seed)``. That claim is
only worth anything if it survives serialization: a training pipeline that
re-generates a shard must get the same *bytes*, so a content hash can stand in
for the shard. This module pins that end to end, together with the three guards
that keep the invariant honest — the schema-version stamp, the separation of
draw failures from realism rejections, and the float32 storage bound.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import pymc_generator as pg
import pymc_generator.world_model as world_model
from pymc_generator import DataGenerator
from pymc_generator.sampler import (
    CARRYOVER_FAMILY_KEYS,
    CORPUS_STORAGE_MAX,
    MAX_TOPUPS_PER_CELL,
    SCMPrior,
)
from pymc_generator.slots import LEGACY_CORPUS_KEYS_V1


def _tiny_config(**overrides):
    """The smallest config that still exercises the full corpus pipeline."""
    kwargs = {
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "n_time_steps": 16,
        "n_cells": 2,
        "draws_per_cell": 1,
        "seed": 20260903,
    }
    kwargs.update(overrides)
    return pg.make_scm_prior(**kwargs)


@pytest.fixture(scope="module")
def tiny_corpus():
    return pg.sample_prior_predictive(_tiny_config())


# ---------------------------------------------------------------------------
# Byte determinism (timing out of the persisted artifact)
# ---------------------------------------------------------------------------


def test_same_seed_generations_save_byte_identical_shards(tmp_path):
    """Independent same-seed generations persist identically despite different timings."""
    cfg = _tiny_config()
    first = pg.sample_prior_predictive(cfg)
    second = pg.sample_prior_predictive(cfg)

    paths = []
    for name, corpus in (("first.npz", first), ("second.npz", second)):
        elapsed = float(len(paths) + 1)
        corpus["diagnostics"]["timing"] = {
            "elapsed_s": elapsed,
            "tasks_per_sec": len(corpus["outcome_raw"]) / elapsed,
        }
        path = tmp_path / name
        pg.save_corpus(corpus, path)
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_timing_is_in_memory_only_and_never_persisted(tmp_path):
    cfg = _tiny_config()
    corpus = pg.sample_prior_predictive(cfg)

    timing = corpus["diagnostics"]["timing"]
    assert set(timing) == {"elapsed_s", "tasks_per_sec"}
    assert timing["elapsed_s"] > 0.0 and timing["tasks_per_sec"] > 0.0
    # The nondeterministic values must not also survive at the top level.
    assert "elapsed_s" not in corpus["diagnostics"]
    assert "tasks_per_sec" not in corpus["diagnostics"]

    before = copy.deepcopy(corpus["diagnostics"])
    path = tmp_path / "corpus.npz"
    pg.save_corpus(corpus, path)
    # save_corpus works on a copy: the caller's diagnostics survive intact.
    assert corpus["diagnostics"]["timing"] is timing
    assert corpus["diagnostics"] == before

    loaded = pg.load_corpus(path)
    assert "timing" not in loaded["diagnostics"]
    assert DataGenerator.validate_corpus(corpus) == []
    assert DataGenerator.validate_corpus(loaded) == []


def test_validate_corpus_rejects_a_malformed_timing_block(tiny_corpus):
    for timing, expected in (
        ([1.0, 2.0], "diagnostics timing must be a mapping"),
        ({"elapsed_s": "slow", "tasks_per_sec": 1.0}, "diagnostics timing elapsed_s"),
        ({"elapsed_s": 1.0}, "diagnostics timing tasks_per_sec"),
    ):
        broken = dict(tiny_corpus)
        diagnostics = dict(tiny_corpus["diagnostics"])
        diagnostics["timing"] = timing
        broken["diagnostics"] = diagnostics
        assert any(expected in error for error in DataGenerator.validate_corpus(broken)), timing


# ---------------------------------------------------------------------------
# M7 — the schema-version stamp is enforced, not merely written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        (None, "missing schema_version"),
        (1, "schema_version 1 is not supported"),
        (99, "schema_version 99 is not supported"),
        ("2", "schema_version must be a non-bool integer"),
        (True, "schema_version must be a non-bool integer"),
    ),
    ids=("missing", "v1", "future", "string", "bool"),
)
def test_unsupported_schema_versions_are_refused_everywhere(
    tmp_path, tiny_corpus, version, expected
):
    """validate reports it, save refuses to write it, load refuses to read it."""
    probe = dict(tiny_corpus)
    diagnostics = dict(tiny_corpus["diagnostics"])
    if version is None:
        diagnostics.pop("schema_version")
    else:
        diagnostics["schema_version"] = version
    probe["diagnostics"] = diagnostics

    assert any(expected in error for error in DataGenerator.validate_corpus(probe))

    path = tmp_path / "bad-version.npz"
    with pytest.raises(ValueError, match=expected):
        pg.save_corpus(probe, path)
    assert not path.exists()

    # A shard already on disk with that version must not be readable either.
    good = tmp_path / "stamped.npz"
    pg.save_corpus(tiny_corpus, good)
    payload = dict(np.load(good, allow_pickle=False))
    on_disk = {
        key: value for key, value in diagnostics.items() if key not in {"timing", "schema_version"}
    }
    if version is not None:
        on_disk["schema_version"] = version
    payload["diagnostics"] = np.array(json.dumps(on_disk))
    np.savez_compressed(good, **payload)
    with pytest.raises(ValueError, match=expected):
        pg.load_corpus(good)


def test_persistence_requires_diagnostics(tmp_path):
    path = tmp_path / "unstamped.npz"
    payload = {"outcome_raw": np.zeros((1, 4))}
    with pytest.raises(ValueError, match="diagnostics"):
        pg.save_corpus(payload, path)
    assert not path.exists()
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="diagnostics"):
        pg.load_corpus(path)


@pytest.mark.parametrize("version", [99, True, "2"])
def test_legacy_keys_cannot_override_an_explicit_version(tmp_path, tiny_corpus, version):
    path = tmp_path / "explicit-version.npz"
    payload = {old: tiny_corpus[new] for old, new in LEGACY_CORPUS_KEYS_V1.items()}
    payload["diagnostics"] = np.array(json.dumps({"schema_version": version}))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="stamped corpora"):
        pg.load_corpus(path)


def test_partial_legacy_vocabulary_is_not_a_migration(tmp_path):
    path = tmp_path / "partial-legacy.npz"
    np.savez(path, K_active=np.array([1]), diagnostics=np.array("{}"))
    with pytest.raises(ValueError, match="incomplete legacy"):
        pg.load_corpus(path)


# ---------------------------------------------------------------------------
# M5 — the burn-in query-overlap boundary follows the ADMITTED response reach
# ---------------------------------------------------------------------------


def test_identity_only_response_accepts_a_short_horizon_with_burn_in():
    """A linear (identity-carryover) preset convolves nothing, so no week is warmup."""
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=8,
        l_max=8,
        nonlinearity="linear",
        n_cells=2,
        draws_per_cell=1,
        seed=2,
    )
    assert cfg.carryover_burn_in == cfg.l_max
    assert cfg.carryover_family_probs[CARRYOVER_FAMILY_KEYS[0]] == 1.0

    corpus = pg.sample_prior_predictive(cfg)
    assert corpus["diagnostics"]["signal"]["response_warmup_weeks"] == 0
    assert DataGenerator.validate_corpus(corpus) == []


def test_zero_decay_geometric_only_response_accepts_a_short_horizon():
    """A geometric family pinned at alpha == 0 is an exact identity too."""
    cfg = _tiny_config(
        n_time_steps=8,
        l_max=8,
        carryover_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        carryover_alpha_range=(0.0, 0.0),
    )
    assert cfg.carryover_burn_in == cfg.l_max
    corpus = pg.sample_prior_predictive(cfg)
    assert corpus["diagnostics"]["signal"]["response_warmup_weeks"] == 0
    assert DataGenerator.validate_corpus(corpus) == []


@pytest.mark.parametrize(
    "carryover_family_probs",
    (
        {"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        {"none": 0.0, "geometric": 0.0, "weibull": 1.0},
        {"none": 0.5, "geometric": 0.0, "weibull": 0.5},
    ),
    ids=("geometric", "weibull", "mixed"),
)
def test_carryover_admitting_families_still_reject_a_short_horizon(carryover_family_probs):
    """Any family that can reach back keeps the boundary at the full l_max - 1."""
    with pytest.raises(ValueError, match="query overlap") as error:
        SCMPrior(
            n_time_steps=8,
            l_max=8,
            carryover_burn_in=8,
            carryover_family_probs=carryover_family_probs,
        ).validate()
    assert "admitted_response_support_weeks = 7" in str(error.value)


# ---------------------------------------------------------------------------
# M3 — draw failures are their own failure mode, not filter rejections
# ---------------------------------------------------------------------------


def _pytensor_evaluation_value_error():
    """Raise the exact shape of a numeric failure inside a PyTensor node.

    Not a hand-rolled ``ValueError``: the retry predicate deliberately requires
    the exception to have escaped a node PyTensor was evaluating, so the fake
    has to travel that path too.
    """
    import pytensor
    import pytensor.tensor as pt

    class _BoomOp(pt.Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            raise ValueError("scale < 0")

    x = pt.dvector("x")
    pytensor.function([x], _BoomOp()(x), mode="FAST_COMPILE")(np.ones(2))


def test_a_programming_error_in_the_draw_propagates_unchanged(monkeypatch):
    """A TypeError from the draw is a bug; it must not be retried or reinterpreted."""

    def boom(*args, **kwargs):
        raise TypeError("simulated programming error")

    monkeypatch.setattr(world_model, "draw_worlds", boom)
    with pytest.raises(TypeError, match="simulated programming error"):
        pg.sample_prior_predictive(_tiny_config())


def test_a_numeric_node_failure_is_retried_and_counted_separately(monkeypatch):
    """The retried batch produced no candidates, so it stays out of the filter stats."""
    real_draw_worlds = world_model.draw_worlds
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            _pytensor_evaluation_value_error()
        return real_draw_worlds(*args, **kwargs)

    monkeypatch.setattr(world_model, "draw_worlds", flaky)
    corpus = pg.sample_prior_predictive(_tiny_config())

    diagnostics = corpus["diagnostics"]
    clean = pg.sample_prior_predictive(_tiny_config())["diagnostics"]
    assert diagnostics["n_draw_failures"] == 1
    assert clean["n_draw_failures"] == 0
    # The failed batch contributed nothing to the realism-filter accounting.
    assert diagnostics["n_draws_evaluated"] == clean["n_draws_evaluated"]
    assert diagnostics["rejection_rate"] == clean["rejection_rate"]


def test_exhaustion_error_separates_filter_rejections_from_draw_failures(monkeypatch):
    """A draw that always fails must not be reported as a strict realism filter."""

    def always_numeric_failure(*args, **kwargs):
        _pytensor_evaluation_value_error()

    monkeypatch.setattr(world_model, "draw_worlds", always_numeric_failure)
    with pytest.raises(RuntimeError) as error:
        pg.sample_prior_predictive(_tiny_config())

    message = str(error.value)
    assert "rejected 0/0 evaluated candidates" in message
    assert f"{MAX_TOPUPS_PER_CELL}/{MAX_TOPUPS_PER_CELL} rounds produced no candidates" in message
    assert "scale < 0" in message


# ---------------------------------------------------------------------------
# L6 — prior ranges the float32 storage cannot hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    (
        "rw_positive_mean_range",
        "beta_additive_range",
        "rw_treatment_std_range",
        "rw_baseline_mean_range",
        "weibull_lam_range",
        "treatment_pulse_amp_range",
    ),
)
def test_validate_rejects_ranges_the_float32_storage_cannot_hold(field):
    """Every persisted array is float32, so an endpoint past its max is unusable."""
    with pytest.raises(ValueError, match=f"{field} bound .* exceeds the float32"):
        SCMPrior(**{field: (1e39, 1e39)}).validate()


def test_validate_rejects_a_shock_level_range_beyond_float32():
    with pytest.raises(
        ValueError, match="treatment_shock_level_range bound .* exceeds the float32"
    ):
        SCMPrior(n_treatment_shocks=1, treatment_shock_level_range=(0.0, 1e39)).validate()


def test_the_largest_representable_bound_is_still_accepted():
    """The rule is representability, not a safety margin: float32 max is legal."""
    SCMPrior(rw_positive_mean_range=(1.0, CORPUS_STORAGE_MAX)).validate()
    with pytest.raises(ValueError, match="exceeds the float32"):
        SCMPrior(rw_positive_mean_range=(1.0, np.nextafter(CORPUS_STORAGE_MAX, np.inf))).validate()
