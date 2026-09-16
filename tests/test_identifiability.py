"""Signal behavior and reproducibility under observationally inert options."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _slice_g_active,
    sample_g_additive,
)
from pymc_generator.signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    check_signal_gate,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from pymc_generator.world_model import build_world_model, draw_worlds, sample_structure


def _corpus_arrays(**overrides) -> dict[str, np.ndarray]:
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        n_cells=2,
        draws_per_cell=1,
        seed=17,
        edge_budget={
            "cy": (2, 2),
            "dc": (2, 2),
            "dz": (2, 2),
            "dy": (1, 1),
            "zy": (2, 2),
            "zc": (4, 4),
            "cc": (1, 1),
            "zz": (1, 1),
        },
        **overrides,
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert pg.DataGenerator.validate_corpus(corpus) == []
    arrays = {key: value for key, value in corpus.items() if isinstance(value, np.ndarray)}
    arrays.update(
        {f"identifiability/{key}": value for key, value in corpus["identifiability"].items()}
    )
    return arrays


@pytest.fixture(scope="module")
def default_arrays():
    return _corpus_arrays()


def test_disabled_covariate_pulse_amplitude_is_rng_inert():
    disabled = {
        "covariate_hf_sigma_range": (0.0, 0.0),
        "covariate_pulse_prob_range": (0.0, 0.0),
    }
    expected = _corpus_arrays(**disabled, covariate_pulse_amp_range=(0.0, 0.0))
    actual = _corpus_arrays(**disabled, covariate_pulse_amp_range=(0.5, 3.0))
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("scope", ("intercept", "non_treatment"))
def test_nonbinding_floor_preserves_generated_arrays(scope, default_arrays):
    """Changing a nonbinding clip must not change draws or persisted labels."""
    floored = _corpus_arrays(baseline_floor=-1e6, baseline_floor_scope=scope)
    assert floored.keys() == default_arrays.keys()
    for key in floored:
        np.testing.assert_array_equal(floored[key], default_arrays[key], err_msg=key)


def test_identifiability_labels_are_optional_metadata_not_features():
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        seed=47,
        confounding_strength_range=(0.6, 0.6),
        n_treatment_shocks=1,
        treatment_shock_length_range=(2, 2),
        treatment_shock_level_range=(0.0, 0.5),
    )
    labelled = pg.sample_prior_predictive(cfg)
    feature_only = pg.sample_prior_predictive(replace(cfg, include_identifiability_labels=False))
    assert set(labelled["identifiability"]) == {"signal_metrics", "signal_metric_valid"}
    assert "identifiability" not in feature_only
    for key, value in feature_only.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, labelled[key]), key
    assert feature_only["diagnostics"]["signal"] == labelled["diagnostics"]["signal"]
    assert pg.DataGenerator.validate_corpus(feature_only) == []


def test_metadata_first_draw_preserves_legacy_rng_order():
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=2, n_latent=1, n_time_steps=24)
    rng = np.random.default_rng(23)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=2, n_latent_active=1
    )
    g_active = _slice_g_active(g, 2, 2, 1)
    model, _out_names, _param_names = build_world_model(
        g_active, cfg, sample_structure(g_active, cfg, rng), cfg.n_time_steps
    )

    legacy = draw_worlds(model, _ADDITIVE_OUT_NAMES, seed=29, draws=3)
    combined = draw_worlds(
        model,
        _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES,
        seed=29,
        draws=3,
    )

    for name in _ADDITIVE_OUT_NAMES:
        assert np.array_equal(combined[name], legacy[name]), name


def test_baseline_walk_sigma_default_tracks_shared_sigma_after_replace():
    """The default baseline scale remains dynamic rather than being copied at construction."""
    cfg = pg.SCMPrior(rw_std_sigma=0.8)
    assert cfg.rw_baseline_std_sigma is None
    assert cfg.rw_baseline_std_sigma_effective == 0.8
    assert replace(cfg, rw_std_sigma=0.35).rw_baseline_std_sigma_effective == 0.35
    assert replace(cfg, rw_baseline_std_sigma=0.6).rw_baseline_std_sigma_effective == 0.6


@pytest.mark.parametrize("sigma", (0.0, -0.1, np.nan, np.inf, -np.inf))
def test_baseline_walk_sigma_override_must_be_finite_and_positive(sigma):
    with pytest.raises(ValueError, match="rw_baseline_std_sigma must be finite and > 0"):
        pg.SCMPrior(rw_baseline_std_sigma=sigma).validate()


def test_confounding_strength_monotonically_increases_dense_r2_without_flattening():
    """The calibrated rho fixture moves label confounding without weakening targets."""
    model_seed = 20260725
    draw_seed = 8675309
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    fixture = {
        "n_time_steps": 72,
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "nonlinearity": "linear",
        "carryover_family_probs": {"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        "saturation_family_probs": {
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        "beta_additive_range": (1.0, 1.0),
        "rw_treatment_std_range": (0.25, 0.25),
        "rw_baseline_std_sigma": 0.75,
        "rw_outcome_std_sigma": 1e-4,
        "outcome_std_mode": "absolute",
        "treatment_hf_sigma_range": (0.08, 0.08),
        "treatment_pulse_prob_range": (0.0, 0.0),
        "treatment_pulse_amp_range": (0.0, 0.0),
        # This fixture isolates rho, so every other texture axis is pinned —
        # the covariates here are edge-free anyway, and leaving their texture on
        # would only move the calibrated constants below.
        "covariate_hf_sigma_range": (0.0, 0.0),
        "covariate_pulse_prob_range": (0.0, 0.0),
        "covariate_pulse_amp_range": (0.0, 0.0),
        "rw_covariate_mean_range": (3.0, 3.0),
        "rw_positive_mean_range": (3.0, 3.0),
        "rw_baseline_mean_range": (3.0, 3.0),
        "carryover_burn_in": 0,
    }
    medians = []
    target_cv_medians = []
    target_std_medians = []
    free_rv_orders = []
    r2_index = SIGNAL_METRIC_LAYOUT.index("contrib_r2_explained_by_rest")
    cv_index = SIGNAL_METRIC_LAYOUT.index("contrib_cv")

    # rho enters the mixture as sqrt(1 - rho**2) * eps_c + rho * eps_b, so R2's
    # response to rho is convex: a 0.45 midpoint still leaves 89% of the treatment
    # innovation independent and lands within noise of rho = 0 (measured first
    # step 0.07-0.14 across draw seeds). 0.6 separates the three levels well
    # clear of the 0.10 margin asserted below (measured 0.15-0.21).
    for rho in (0.0, 0.6, 0.9):
        cfg = pg.make_scm_prior(
            **fixture,
            confounding_strength_range=(rho, rho),
        )
        structural = sample_structure(g, cfg, np.random.default_rng(model_seed))
        structural["smoothness_c"][:] = 0.0
        structural["smoothness_b"][:] = 0.0
        model, out_names, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
        drawn = draw_worlds(model, out_names, seed=draw_seed, draws=48)
        metrics, valid = dense_signal_metrics(
            drawn["treatments"],
            drawn["contributions"],
            drawn["outcome"],
            drawn["baseline"],
            np.ones((48, 2), dtype=bool),
            outcome_scale=np.ones(48),
            carryover_family=np.zeros((48, 2), dtype=np.uint8),
            carryover_alpha=np.zeros((48, 2)),
            weibull_lam=np.zeros((48, 2)),
            weibull_k=np.zeros((48, 2)),
            l_max=cfg.l_max,
            carryover_burn_in=cfg.carryover_burn_in,
        )
        r2 = metrics[..., r2_index][valid[..., r2_index].astype(bool)]
        target_cv = metrics[..., cv_index][valid[..., cv_index].astype(bool)]
        medians.append(float(np.median(r2)))
        target_cv_medians.append(float(np.median(target_cv)))
        target_std = drawn["contributions"].std(axis=1).ravel()
        target_std_medians.append(float(np.median(target_std)))
        free_rv_orders.append(tuple(rv.name for rv in model.free_RVs))

    # The realized walk amplitude is now random (the walk is normalized by a
    # constant, not by its own realized sd). At these pinned seeds, the
    # per-draw minima are 0.106 (CV) and 0.107 (std), but those are tail
    # statistics. The median-ratio assertions below carry the "targets do not
    # flatten" property.
    assert np.diff(medians).min() > 0.10
    assert medians[-1] - medians[0] > 0.40
    assert target_cv_medians[-1] >= 0.90 * target_cv_medians[0]
    assert target_std_medians[-1] >= 0.90 * target_std_medians[0]
    assert free_rv_orders[0] == free_rv_orders[1] == free_rv_orders[2]
    assert all("confounding_strength" not in order for order in free_rv_orders)


@pytest.mark.slow
def test_supported_corpus_signal_gate():
    """The supported prior clears the calibrated identifiability signal gate."""
    cfg = pg.make_scm_prior(
        n_treatments=6,
        n_covariates=4,
        n_latent=2,
        n_time_steps=104,
        n_cells=4,
        draws_per_cell=6,
        seed=314159,
    )
    seed_pool = (314159, 1, 2, 3, 4, 5, 6, 7)
    corpora = [pg.sample_prior_predictive(replace(cfg, seed=seed)) for seed in seed_pool]
    signal = summarize_signal_metrics(
        np.concatenate([corpus["identifiability"]["signal_metrics"] for corpus in corpora]),
        np.concatenate([corpus["identifiability"]["signal_metric_valid"] for corpus in corpora]),
        np.concatenate([corpus["outcome_raw"] for corpus in corpora]),
        np.concatenate(
            [
                (corpus["g"][:, cfg.layout.slices["cy"]] == 1)
                & (corpus["treatment_active_mask"] == 1)
                for corpus in corpora
            ]
        ),
        l_max=cfg.l_max,
        carryover_burn_in=cfg.carryover_burn_in,
        outcome_scale=np.concatenate([corpus["outcome_scale"] for corpus in corpora]),
        carryover_family=np.concatenate([corpus["carryover_family"] for corpus in corpora]),
        carryover_alpha=np.concatenate([corpus["carryover_alpha"] for corpus in corpora]),
        weibull_lam=np.concatenate([corpus["weibull_lam"] for corpus in corpora]),
        weibull_k=np.concatenate([corpus["weibull_k"] for corpus in corpora]),
    )
    passed, _ = check_signal_gate(signal)

    # Gated values are Bernoulli fractions over direct treatments. At the old
    # 100-treatment size, their population means sat about one sigma below the
    # thresholds, so one draw failed about a quarter of the time (6/8 passed).
    # Pooling eight independent corpora yields roughly 800 direct treatments and
    # measures the prior's gate compliance rather than one noisy realization.
    assert passed
    assert signal["frac_spearman_lt_03"] <= 0.15
    assert signal["frac_warmup_gt_3"] <= 0.10
