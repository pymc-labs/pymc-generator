"""Decomposition invariants — the correctness contract of the generator.

These assert behaviour at the public level (``sample_prior_predictive`` /
``sample_scm``), so they hold regardless of how the SCM graph is built or
drawn internally. They are the regression net for any engine refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.scenarios import SCENARIOS


@pytest.fixture(scope="module")
def corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=48,
        n_cells=2,
        draws_per_cell=4,
        seed=11,
        edge_budget={"cy": (3, 4), "cc": (1, 2), "zc": (1, 2), "dc": (1, 2)},
    )
    return pg.sample_prior_predictive(cfg)


def test_additive_identity(corpus):
    """sales_raw == baseline_raw + Σ_k contributions_raw + indirect_effects."""
    err = np.abs(
        corpus["baseline_raw"].astype(np.float64)
        + corpus["contributions_raw"].astype(np.float64).sum(-1)
        + corpus["indirect_effects"].astype(np.float64)
        - corpus["sales_raw"].astype(np.float64)
    ).max()
    # float32 storage rounding on O(sales) magnitudes
    assert err < 1e-3 * float(np.abs(corpus["sales_raw"]).max())


def test_telescoping_split_sums_to_indirect(corpus):
    ibs = corpus["indirect_effects_by_source"].astype(np.float64)
    assert ibs.shape[-1] == 3  # locked order (cc, zc, dc)
    err = np.abs(ibs.sum(-1) - corpus["indirect_effects"].astype(np.float64)).max()
    assert err < 1e-4 * max(float(np.abs(corpus["indirect_effects"]).max()), 1.0)


def test_full_per_node_additivity(corpus):
    """baseline_intrinsic + Σ confounder + Σ control + Σ direct + Σ by_source == sales."""
    f = lambda k: corpus[k].astype(np.float64)  # noqa: E731
    recon = (
        f("baseline_intrinsic")
        + f("confounder_contribution").sum(-1)
        + f("control_contribution").sum(-1)
        + f("contributions_raw").sum(-1)
        + f("indirect_effects_by_source").sum(-1)
    )
    err = np.abs(recon - f("sales_raw")).max()
    assert err < 1e-3 * float(np.abs(corpus["sales_raw"]).max())


def test_diagnostics_report_persisted_identity_error(corpus):
    """Diagnostics quantify reconstruction error from the stored float32 arrays."""
    expected_decomposition = np.abs(
        corpus["baseline_raw"]
        + corpus["contributions_raw"].sum(axis=-1)
        + corpus["indirect_effects"]
        - corpus["sales_raw"]
    ).max()
    expected_telescoping = np.abs(
        corpus["indirect_effects_by_source"].sum(axis=-1) - corpus["indirect_effects"]
    ).max()
    expected_full = np.abs(
        corpus["baseline_intrinsic"]
        + corpus["confounder_contribution"].sum(axis=-1)
        + corpus["control_contribution"].sum(axis=-1)
        + corpus["contributions_raw"].sum(axis=-1)
        + corpus["indirect_effects_by_source"].sum(axis=-1)
        - corpus["sales_raw"]
    ).max()
    expected_baseline = np.abs(
        corpus["baseline_intrinsic"]
        + corpus["confounder_contribution"].sum(axis=-1)
        + corpus["control_contribution"].sum(axis=-1)
        - corpus["baseline_raw"]
    ).max()

    diagnostics = corpus["diagnostics"]
    assert diagnostics["decomposition_max_abs_error"] == float(expected_decomposition)
    assert diagnostics["telescoping_split_max_abs_error"] == float(expected_telescoping)
    assert diagnostics["full_decomposition_max_abs_error"] == float(expected_full)
    assert diagnostics["baseline_decomposition_max_abs_error"] == float(expected_baseline)

    # Equality locks the persisted-float32 measurement contract; the ceiling
    # independently bounds the reconstruction residual itself.
    for name in (
        "decomposition_max_abs_error",
        "telescoping_split_max_abs_error",
        "full_decomposition_max_abs_error",
        "baseline_decomposition_max_abs_error",
    ):
        assert diagnostics[name] < 1e-5


def test_channels_positive_and_finite(corpus):
    assert np.isfinite(corpus["spend_raw"]).all()
    assert (corpus["spend_raw"] >= 0).all()
    assert np.isfinite(corpus["sales_raw"]).all()
    assert (corpus["sales_raw"] >= 0).all()


def test_inactive_channels_zero_padded():
    """Channels beyond the active count contribute nothing and carry no spend."""
    cfg = pg.make_scm_prior(
        n_treatments=6,
        n_covariates=3,
        n_latent=2,
        n_treatments_active_range=(3, 3),
        n_covariates_active_range=(2, 2),
        n_latent_active_range=(1, 1),
        n_time_steps=40,
        n_cells=2,
        draws_per_cell=2,
        seed=5,
    )
    corpus = pg.sample_prior_predictive(cfg)
    acm = corpus["treatment_active_mask"]
    # padded (inactive) channel slots are exactly zero in spend and contribution
    pad = acm == 0
    assert (corpus["spend_raw"][pad[:, None, :].repeat(corpus["spend_raw"].shape[1], 1)] == 0).all()
    contrib = corpus["contributions_raw"]
    assert (contrib[pad[:, None, :].repeat(contrib.shape[1], 1)] == 0).all()


def test_determinism_same_seed():
    kw = {
        "n_treatments": 4,
        "n_covariates": 2,
        "n_latent": 1,
        "n_time_steps": 40,
        "n_cells": 2,
        "draws_per_cell": 3,
        "seed": 99,
    }
    a = pg.sample_prior_predictive(pg.make_scm_prior(**kw))
    b = pg.sample_prior_predictive(pg.make_scm_prior(**kw))
    for key in ("spend_raw", "sales_raw", "g", "contributions_raw", "indirect_effects"):
        assert np.array_equal(a[key], b[key]), f"{key} not reproducible"


def test_different_seed_differs():
    a = pg.sample_prior_predictive(
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=40, seed=1)
    )
    b = pg.sample_prior_predictive(
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=40, seed=2)
    )
    assert not np.array_equal(a["sales_raw"], b["sales_raw"])


def test_direct_only_scenario_has_negligible_indirect():
    """direct_only isolates C->Y with no channel-input interactions -> indirect ~ 0."""
    sc = SCENARIOS[0]
    assert sc.name == "direct_only"
    world = pg.sample_scm(sc.prior(n_time_steps=52, seed=0), seed=0, connect_all=True, name=sc.name)
    assert np.abs(world.data["indirect_effects"]).max() < 1e-6


def test_diverse_texture_targets_not_flat():
    """The supported (diverse) texture must give contribution targets real variation."""
    world = pg.sample_scm(
        pg.make_scm_prior(
            n_treatments=4,
            n_covariates=2,
            n_latent=1,
            n_time_steps=104,
            seed=3,
            edge_budget={"cy": (4, 4)},
        ),
        seed=3,
        connect_all=True,
    )
    contrib = world.data["contributions"]  # (n_time_steps, n_treatments)
    # at least one direct channel varies week to week (std / |mean| not ~0)
    cv = contrib.std(0) / (np.abs(contrib.mean(0)) + 1e-9)
    assert cv.max() > 0.05
