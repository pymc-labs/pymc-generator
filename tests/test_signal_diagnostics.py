"""Deterministic contracts for persisted signal diagnostics."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from prior_generator import DataGenerator, load_corpus, make_scm_prior, mechanisms, save_corpus
from prior_generator.sampler import OUTCOME_NOISE_SEMANTICS, OUTCOME_NOISE_VERSION
from prior_generator.signal_diagnostics import (
    DEFAULT_GATE,
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    _adstock_numpy,
    check_signal_gate,
    contemporaneous_weight,
    dense_signal_metrics,
    per_channel_signal,
    signal_summary,
    summarize_signal_metrics,
)


def _dense(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)[None, :, None]
    y = np.asarray(y, dtype=np.float32)[None, :, None]
    sales = y[..., 0] + 10
    if baseline is None:
        baseline = np.zeros(x.shape[1], dtype=np.float32)
    if "sales_scale" not in kwargs:
        kwargs["sales_scale"] = np.ones(1, dtype=np.float32)
    return dense_signal_metrics(
        x, y, sales, np.asarray(baseline, dtype=np.float32)[None], np.ones((1, 1)), **kwargs
    )


def _metric(name: str) -> int:
    return SIGNAL_METRIC_LAYOUT.index(name)


def _direct_only_edge_budget() -> dict[str, int | tuple[int, int]]:
    """Require one direct channel and exclude upstream channel parents."""
    return {
        "cy": (1, 1),
        "dc": 0,
        "db": 0,
        "zb": 0,
        "dz": 0,
        "zc": 0,
        "cc": 0,
        "zz": 0,
    }


def test_numpy_adstock_matches_pytensor_mechanisms():
    x = np.array([0.2, 1.0, 0.5, 2.0, 0.7], dtype=np.float64)
    x_symbolic = pt.as_tensor_variable(x[:, None])
    geo = pytensor.function([], mechanisms.apply_geometric_adstock(x_symbolic, 0.6, 4))()[:, 0]
    weibull = pytensor.function(
        [], mechanisms.apply_weibull_pdf_adstock(x_symbolic, 3.0, 2.0, 4)
    )()[:, 0]
    assert np.allclose(_adstock_numpy(x, 1, 0.6, 1.0, 1.0, 4), geo)
    assert np.allclose(_adstock_numpy(x, 2, 0.0, 3.0, 2.0, 4), weibull)


def test_single_lag_weibull_is_identity_in_numpy_and_symbolic_paths():
    x = np.array([0.2, 1.0, 0.5], dtype=np.float64)
    with np.errstate(all="raise"):
        numpy_result = _adstock_numpy(x, 2, 0.0, 3.0, 2.0, 1)
        symbolic_result = pytensor.function(
            [],
            mechanisms.apply_weibull_pdf_adstock(pt.as_tensor_variable(x[:, None]), 3.0, 2.0, 1),
        )()[:, 0]
    assert np.array_equal(numpy_result, x)
    assert np.array_equal(symbolic_result, x)


def test_degenerate_weibull_kernel_is_finite_and_consistent():
    x = np.array([0.2, 1.0, 0.5], dtype=np.float64)
    lam = 2.0804050381276453
    shape = 2.0
    with np.errstate(all="raise"):
        numpy_result = _adstock_numpy(x, 2, 0.0, lam, shape, 2)
        symbolic_result = pytensor.function(
            [],
            mechanisms.apply_weibull_pdf_adstock(pt.as_tensor_variable(x[:, None]), lam, shape, 2),
        )()[:, 0]
    assert np.array_equal(numpy_result, np.zeros_like(x))
    assert np.array_equal(symbolic_result, numpy_result)


def test_weibull_guard_matches_numpy_across_compile_modes():
    """Weibull degeneracy decisions must be identical in both symbolic backends."""
    x = np.array([0.2, 1.0, 0.5, 1.5, 0.7, 0.3, 1.2, 0.8], dtype=np.float64)
    compiled_adstock = {}
    for l_max in (4, 8):
        lam_t = pt.dscalar("lam")
        k_t = pt.dscalar("k")
        adstock = mechanisms.apply_weibull_pdf_adstock(
            pt.as_tensor_variable(x[:, None]), lam_t, k_t, l_max
        )
        compiled_adstock[l_max] = (
            pytensor.function([lam_t, k_t], adstock, mode="FAST_COMPILE"),
            pytensor.function([lam_t, k_t], adstock, mode="FAST_RUN"),
        )

    for l_max, lam, k in (
        (4, 1e-6, 60.0),  # Overflowed span is -inf.
        (4, 1.0, 1000.0),  # Near-delta density exercises annihilated lags.
        (4, 1e200, 1e-200),  # Underflowed product creates a degenerate kernel.
        (4, 1e250, 1e-200),  # A second underflow must remain finite.
        (4, 1e6, 1e-6),  # Near-degenerate but valid kernel must not be zeroed.
        (4, 4.0, 2.0),  # Ordinary in-prior Weibull kernel must not be zeroed.
        # Representatives of the 24 historical library-output mode flips on
        # lam=logspace(0, 6, 61), k=logspace(0, 3, 61), l_max=8.
        (8, 10.0**1.3, 10.0**2.9),
        (8, 10.0**2.5, 10.0**2.3),
        (8, 10.0**3.7, 10.0**2.05),
        (8, 1e4, 100.0),
    ):
        fast_compile, fast_run = compiled_adstock[l_max]
        with np.errstate(all="ignore"):
            numpy_result = _adstock_numpy(x, 2, 0.0, lam, k, l_max)
            fast_compile_result = fast_compile(lam, k)[:, 0]
            fast_run_result = fast_run(lam, k)[:, 0]

        # Removing the analytic floor and relying on a library-output isfinite
        # guard makes the l_max=8 representatives disagree between modes.
        np.testing.assert_allclose(fast_compile_result, fast_run_result, rtol=1e-12, atol=1e-12)
        assert np.isfinite(fast_compile_result).all()
        assert np.isfinite(fast_run_result).all()
        numpy_is_zero = np.array_equal(numpy_result, np.zeros_like(numpy_result))
        for symbolic_result in (fast_compile_result, fast_run_result):
            assert np.array_equal(symbolic_result, np.zeros_like(symbolic_result)) == numpy_is_zero
            np.testing.assert_allclose(symbolic_result, numpy_result, rtol=1e-12, atol=1e-12)


def test_adstock_is_a_plain_normalized_causal_convolution():
    x = np.arange(1, 9, dtype=np.float64)
    result = _adstock_numpy(x, 1, 0.5, 1.0, 1.0, 3)
    weights = np.array([1.0, 0.5, 0.25])
    weights /= weights.sum()
    expected = np.convolve(x, weights, mode="full")[: x.size]
    assert np.allclose(result, expected)


def test_spearman_uses_once_adstocked_observed_spend_and_lmax_suffix():
    x = np.array([0.3, 2.0, 0.1, 1.3, 0.8, 3.0, 0.4, 2.4], dtype=np.float32)
    identity, identity_valid = _dense(x, x, l_max=3)
    assert identity_valid[0, 0, _metric("spearman")] == 1
    assert identity[0, 0, _metric("spearman")] == 1.0

    y = _adstock_numpy(x, 1, 0.55, 1.0, 1.0, 3).astype(np.float32)
    metrics, valid = _dense(
        x, y, l_max=3, adstock_family=np.array([[1]]), adstock_alpha=np.array([[0.55]])
    )
    assert valid[0, 0, _metric("spearman")] == 1
    assert metrics[0, 0, _metric("spearman")] == 1.0


def test_flattened_signal_api_remains_compatible():
    x = np.arange(6, dtype=np.float32)[None, :, None]
    flattened = per_channel_signal(x, x, x[..., 0] + 1, np.ones((1, 1)), l_max=3)
    assert flattened["task_idx"].shape == (1,)
    assert flattened["spearman_valid"].shape == (1,)
    assert flattened["contrib_r2_explained_by_rest_valid"][0] == 0
    assert flattened["contrib_corr_baseline_valid"][0] == 0
    with_baseline = per_channel_signal(
        x,
        x,
        x[..., 0] + 1,
        np.ones((1, 1)),
        l_max=3,
        baseline=x[..., 0],
    )
    assert with_baseline["contrib_r2_explained_by_rest_valid"][0] == 1
    assert with_baseline["contrib_corr_baseline_valid"][0] == 1
    summary = signal_summary(x, x, x[..., 0] + 1, np.ones((1, 1)), l_max=3)
    assert summary["n_direct_channels"] == 1


def test_r2_and_signed_baseline_correlation_contracts():
    x = np.arange(6, dtype=np.float32)
    constant, _ = _dense(x, np.full(6, 7.0), baseline=np.arange(6))
    assert constant[0, 0, _metric("contrib_r2_explained_by_rest")] == 1.0

    baseline = np.arange(6, dtype=np.float32)
    single, _ = _dense(x, 0.5 + 2 * baseline, baseline=baseline)
    assert np.isclose(single[0, 0, _metric("contrib_r2_explained_by_rest")], 1.0)
    assert np.isclose(single[0, 0, _metric("contrib_corr_baseline")], 1.0)

    spend = np.ones((1, 6, 2), dtype=np.float32)
    contributions = np.zeros_like(spend)
    contributions[0, :, 1] = np.arange(6)
    contributions[0, :, 0] = 1.0 + 3.0 * baseline + contributions[0, :, 1]
    metrics, _ = dense_signal_metrics(
        spend,
        contributions,
        np.ones((1, 6), dtype=np.float32),
        baseline[None],
        np.ones((1, 2)),
        sales_scale=np.ones(1),
    )
    assert np.isclose(metrics[0, 0, _metric("contrib_r2_explained_by_rest")], 1.0)

    negative, _ = _dense(x, -baseline, baseline=baseline)
    assert np.isclose(negative[0, 0, _metric("contrib_corr_baseline")], -1.0)


def test_r2_is_invariant_to_large_target_offsets():
    x = np.arange(8, dtype=np.float32)
    y = np.tile(np.array([-1.0, 1.0], dtype=np.float32), 4)
    base, _ = _dense(x, y)
    shifted, _ = _dense(x, y + np.float32(1_000_000.0))
    index = _metric("contrib_r2_explained_by_rest")
    assert base[0, 0, index] == 0.0
    assert shifted[0, 0, index] == base[0, 0, index]


def test_r2_is_scale_invariant_for_representably_nonconstant_targets():
    x = np.arange(100, dtype=np.float32)
    index = _metric("contrib_r2_explained_by_rest")
    for amplitude in (1.0, 1e-4, 1e-7):
        y = (amplitude * np.sin(np.arange(100))).astype(np.float32)
        metrics, valid = _dense(x, y)
        assert valid[0, 0, index] == 1
        assert np.isclose(metrics[0, 0, index], 0.0, atol=1e-6)


def test_r2_requires_residual_degrees_of_freedom_not_raw_column_count():
    index = _metric("contrib_r2_explained_by_rest")
    _, saturated_valid = _dense(np.arange(2), np.arange(2), baseline=np.arange(2))
    assert saturated_valid[0, 0, index] == 0

    constant, constant_valid = _dense(np.arange(2), np.ones(2), baseline=np.arange(2))
    assert constant_valid[0, 0, index] == 1
    assert constant[0, 0, index] == 1.0

    spend = np.ones((1, 3, 2), dtype=np.float32)
    baseline = np.arange(3, dtype=np.float32)
    contributions = np.stack([baseline, 2 * baseline], axis=1)[None]
    metrics, valid = dense_signal_metrics(
        spend,
        contributions,
        np.ones((1, 3), dtype=np.float32),
        baseline[None],
        np.ones((1, 2)),
        sales_scale=np.ones(1),
    )
    assert valid[0, 0, index] == 1
    assert metrics[0, 0, index] == 1.0


def test_short_constant_and_warmup_validity_contracts():
    short, short_valid = _dense(np.arange(2), np.arange(2), l_max=3)
    assert short_valid[0, 0, _metric("spend_hf")] == 1
    assert short_valid[0, 0, _metric("spearman")] == 0
    assert short_valid[0, 0, _metric("warmup_ratio")] == 0

    constant, constant_valid = _dense(np.ones(6), np.ones(6), l_max=3)
    assert constant_valid[0, 0, _metric("spearman")] == 1
    assert constant[0, 0, _metric("spearman")] == 0.0

    x = np.arange(8)
    y = np.array([0, 3, 6, 1, 2, 1, 2, 1])
    with_warmup, warmup_valid = _dense(x, y, l_max=3)
    assert warmup_valid[0, 0, _metric("warmup_ratio")] == 1
    assert with_warmup[0, 0, _metric("warmup_ratio")] > 3.0

    flat_suffix, flat_suffix_valid = _dense(x, np.array([0, 3, 6, 1, 1, 1, 1, 1]), l_max=3)
    assert flat_suffix[0, 0, _metric("warmup_ratio")] == 0.0
    assert flat_suffix_valid[0, 0, _metric("warmup_ratio")] == 0

    _, burned_valid = _dense(x, y, l_max=3, adstock_burn_in=3)
    assert burned_valid[0, 0, _metric("warmup_ratio")] == 0


def test_summary_uses_valid_denominators_and_gate_contracts():
    metrics = np.zeros((1, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    cv = _metric("contrib_cv")
    metrics[0, :, cv] = [0.2, 0.0]
    valid[0, 0, cv] = 1
    summary = summarize_signal_metrics(
        metrics,
        valid,
        np.ones((1, 3)),
        np.ones((1, 2)),
        sales_scale=np.ones(1),
        l_max=3,
        adstock_burn_in=3,
    )
    assert summary["frac_contrib_cv_lt_010"] == 0.0
    assert summary["frac_spearman_lt_03"] is None
    assert summary["frac_warmup_gt_3"] == 0.0
    ok, lines = check_signal_gate(summary)
    assert not ok and any("frac_spearman_lt_03 missing" in line for line in lines)


def test_empty_signal_summary_has_none_quantiles():
    metrics = np.zeros((0, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    summary = summarize_signal_metrics(
        metrics,
        valid,
        np.zeros((0, 4)),
        np.zeros((0, 2), dtype=bool),
    )
    assert summary["n_direct_channels"] == 0
    assert all(value is None for value in summary["sales_level_ratio_quantiles"].values())


def test_final_float32_metrics_recompute_after_save_load(tmp_path):
    x = np.array([0.1, 0.7, 2.1, 0.4, 1.3, 0.8], dtype=np.float64)[None, :, None].astype(np.float32)
    y = (1.7 * x).astype(np.float32)
    sales = (y[..., 0] + 10).astype(np.float32)
    baseline = np.zeros_like(sales)
    metrics, valid = dense_signal_metrics(x, y, sales, baseline, np.ones((1, 1)))
    path = tmp_path / "signal.npz"
    save_corpus(
        {"spend_raw": x, "contributions_raw": y, "sales_raw": sales, "baseline_raw": baseline}, path
    )
    loaded = load_corpus(path)
    recomputed, recomputed_valid = dense_signal_metrics(
        loaded["spend_raw"],
        loaded["contributions_raw"],
        loaded["sales_raw"],
        loaded["baseline_raw"],
        np.ones((1, 1)),
    )
    assert np.array_equal(metrics, recomputed)
    assert np.array_equal(valid, recomputed_valid)


def test_generated_shard_labels_match_loaded_array_recomputation(tmp_path):
    from prior_generator import make_scm_prior, sample_prior_predictive
    from prior_generator.slots import EDGE_TYPES_EXTENDED, SlotLayout

    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=16,
        l_max=3,
        adstock_burn_in=3,
        n_cells=2,
        draws_per_cell=1,
        seed=37,
    )
    corpus = sample_prior_predictive(cfg)
    path = tmp_path / "generated.npz"
    save_corpus(corpus, path)
    loaded = load_corpus(path)
    layout = SlotLayout(n_treatments=2, n_covariates=2, n_latent=1, edge_types=EDGE_TYPES_EXTENDED)
    direct = (loaded["g"][:, layout.slices["cy"]] == 1) & (loaded["treatment_active_mask"] == 1)
    signal_config = loaded["diagnostics"]["signal"]
    assert signal_config["adstock_kernel_semantics"] == "normalized-causal-minmax-weibull-density"
    assert signal_config["outcome_noise_semantics"] == OUTCOME_NOISE_SEMANTICS
    assert signal_config["outcome_noise_version"] == OUTCOME_NOISE_VERSION
    assert signal_config["outcome_std_mode"] == "relative"
    metrics, valid = dense_signal_metrics(
        loaded["spend_raw"],
        loaded["contributions_raw"],
        loaded["sales_raw"],
        loaded["baseline_raw"],
        direct,
        sales_scale=loaded["sales_scale"],
        adstock_family=loaded["adstock_family"],
        adstock_alpha=loaded["adstock_alpha"],
        weibull_lam=loaded["weibull_lam"],
        weibull_k=loaded["weibull_k"],
        l_max=signal_config["l_max"],
        adstock_burn_in=signal_config["adstock_burn_in"],
    )
    assert np.array_equal(loaded["identifiability"]["signal_metrics"], metrics)
    assert np.array_equal(loaded["identifiability"]["signal_metric_valid"], valid)


def test_validator_checks_signal_layout_dtype_and_eligibility():
    corpus = {
        "spend_raw": np.ones((1, 4, 1), dtype=np.float32),
        "spend_norm": np.ones((1, 4, 1), dtype=np.float32),
        "spend_share": np.ones((1, 4, 1), dtype=np.float32),
        "controls": np.ones((1, 4, 1), dtype=np.float32),
        "sales_raw": np.ones((1, 4), dtype=np.float32),
        "sales_norm": np.ones((1, 4), dtype=np.float32),
        "support_mask": np.array([[1, 1, 1, 0]], dtype=np.uint8),
        "is_future": np.zeros(1, dtype=np.uint8),
        "g": np.zeros((1, 6), dtype=np.uint8),
        "contributions_raw": np.zeros((1, 4, 1), dtype=np.float32),
        "baseline_raw": np.ones((1, 4), dtype=np.float32),
        "demand": np.ones((1, 4, 1), dtype=np.float32),
        "spend_means": np.ones((1, 1), dtype=np.float32),
        "sales_scale": np.ones(1, dtype=np.float32),
        "is_val": np.zeros(1, dtype=np.uint8),
        "cell_id": np.zeros(1, dtype=np.int32),
        "treatment_active_mask": np.ones((1, 1), dtype=np.uint8),
        "covariate_active_mask": np.ones((1, 1), dtype=np.uint8),
        "latent_active_mask": np.ones((1, 1), dtype=np.uint8),
        "n_treatments_active": np.ones(1, dtype=np.int32),
        "n_covariates_active": np.ones(1, dtype=np.int32),
        "n_latent_active": np.ones(1, dtype=np.int32),
        "confounding_strength": np.zeros(1, dtype=np.float32),
        "indirect_effects": np.zeros((1, 4), dtype=np.float32),
        "channel_active": np.zeros((1, 1), dtype=np.uint8),
        "control_contribution": np.zeros((1, 4, 1), dtype=np.float32),
        "confounder_contribution": np.zeros((1, 4, 1), dtype=np.float32),
        "baseline_intrinsic": np.ones((1, 4), dtype=np.float32),
        "indirect_effects_by_source": np.zeros((1, 4, 3), dtype=np.float32),
        "channel_shock_mask": np.zeros((1, 4, 1), dtype=np.uint8),
        "channel_shock_channel": np.empty((1, 0), dtype=np.int32),
        "channel_shock_start": np.empty((1, 0), dtype=np.int32),
        "channel_shock_length": np.empty((1, 0), dtype=np.int32),
        "channel_shock_level_multiplier": np.empty((1, 0), dtype=np.float32),
        "channel_shock_level": np.empty((1, 0), dtype=np.float32),
        "channel_level": np.ones((1, 1), dtype=np.float32),
        "saturation_scale": np.ones((1, 1), dtype=np.float32),
        "adstock_family": np.zeros((1, 1), dtype=np.uint8),
        "adstock_alpha": np.zeros((1, 1), dtype=np.float32),
        "weibull_lam": np.zeros((1, 1), dtype=np.float32),
        "weibull_k": np.zeros((1, 1), dtype=np.float32),
        "identifiability": {
            "signal_metrics": np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32),
            "signal_metric_valid": np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.uint8),
        },
        "diagnostics": {
            "n_tasks": 1,
            "n_cells": 1,
            "short_horizon_n_query": 1,
            "signal": {
                "metric_version": SIGNAL_METRIC_VERSION,
                "metric_layout": list(SIGNAL_METRIC_LAYOUT),
                "l_max": 8,
                "adstock_burn_in": 0,
                "adstock_kernel_semantics": "normalized-causal-minmax-weibull-density",
                "adstock_kernel_version": 3,
                "outcome_noise_semantics": OUTCOME_NOISE_SEMANTICS,
                "outcome_noise_version": OUTCOME_NOISE_VERSION,
                "outcome_std_mode": "relative",
            },
        },
    }
    corpus["identifiability"]["signal_metrics"][0, 0, 0] = 1
    errors = DataGenerator.validate_corpus(corpus)
    assert any("ineligible" in error for error in errors)
    assert SIGNAL_METRIC_VERSION == 3


def test_scale_free_metrics_match_their_unscaled_definitions():
    tiny_cv = 1e-13 * np.array([1, 2, 1, 2], dtype=np.float32)
    tiny_cv_metrics, tiny_cv_valid = dense_signal_metrics(
        tiny_cv[None, :, None],
        tiny_cv[None, :, None],
        np.ones((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        sales_scale=np.ones(1),
        l_max=1,
    )
    for key in ("spend_cv", "contrib_cv"):
        index = _metric(key)
        assert tiny_cv_valid[0, 0, index] == 1
        assert np.isclose(tiny_cv_metrics[0, 0, index], 1 / 3)

    zero_mean = np.array([-1, 1, -1, 1], dtype=np.float32)
    zero_mean_metrics, zero_mean_valid = dense_signal_metrics(
        zero_mean[None, :, None],
        zero_mean[None, :, None],
        np.ones((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        sales_scale=np.ones(1),
        l_max=1,
    )
    for key in ("spend_cv", "contrib_cv"):
        index = _metric(key)
        assert zero_mean_metrics[0, 0, index] == 0.0
        assert zero_mean_valid[0, 0, index] == 0

    tiny_corr = 1e-7 * np.array([1, 3, 1, 3, 1, 3], dtype=np.float32)
    corr_metrics, corr_valid = dense_signal_metrics(
        tiny_corr[None, :, None],
        tiny_corr[None, :, None],
        np.ones((1, 6), dtype=np.float32),
        tiny_corr[None],
        np.ones((1, 1), dtype=bool),
        sales_scale=np.ones(1),
        l_max=1,
    )
    corr_index = _metric("contrib_corr_baseline")
    assert corr_valid[0, 0, corr_index] == 1
    assert np.isclose(corr_metrics[0, 0, corr_index], 1.0)

    warmup = 1e-13 * np.array([0, 3, 6, 1, 2, 1, 2, 1], dtype=np.float32)
    warmup_metrics, warmup_valid = dense_signal_metrics(
        np.arange(8, dtype=np.float32)[None, :, None],
        warmup[None, :, None],
        np.ones((1, 8), dtype=np.float32),
        np.zeros((1, 8), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        sales_scale=np.ones(1),
        l_max=3,
    )
    warmup_index = _metric("warmup_ratio")
    assert warmup_valid[0, 0, warmup_index] == 1
    expected_warmup_ratio = (warmup[:3].max() - warmup[:3].min()) / warmup[3:].std()
    assert np.isclose(warmup_metrics[0, 0, warmup_index], expected_warmup_ratio)
    assert warmup_metrics[0, 0, warmup_index] > 3.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"adstock_family": np.array([[1.9]])}, "adstock_family"),
        ({"adstock_family": np.array([[3]])}, "adstock_family"),
        (
            {
                "adstock_family": np.array([[1]]),
                "adstock_alpha": np.array([[np.nan]]),
            },
            "adstock_alpha",
        ),
        (
            {
                "adstock_family": np.array([[1]]),
                "adstock_alpha": np.array([[-0.1]]),
            },
            "adstock_alpha",
        ),
        (
            {
                "adstock_family": np.array([[1]]),
                "adstock_alpha": np.array([[1.1]]),
            },
            "adstock_alpha",
        ),
        (
            {
                "adstock_family": np.array([[2]]),
                "weibull_lam": np.array([[np.nan]]),
            },
            "weibull_lam",
        ),
        (
            {
                "adstock_family": np.array([[2]]),
                "weibull_lam": np.array([[0.0]]),
            },
            "weibull_lam",
        ),
        (
            {
                "adstock_family": np.array([[2]]),
                "weibull_k": np.array([[np.nan]]),
            },
            "weibull_k",
        ),
        (
            {
                "adstock_family": np.array([[2]]),
                "weibull_k": np.array([[0.0]]),
            },
            "weibull_k",
        ),
        ({"l_max": 0}, "l_max"),
        ({"l_max": 1.5}, "l_max"),
        ({"adstock_burn_in": -1}, "adstock_burn_in"),
        ({"adstock_burn_in": 0.5}, "adstock_burn_in"),
        ({"sales_scale": np.array([np.nan])}, "sales_scale"),
        ({"sales_scale": np.array([0.0])}, "sales_scale"),
        ({"sales_scale": np.array([-1.0])}, "sales_scale"),
    ],
)
def test_dense_signal_metrics_rejects_invalid_diagnostic_inputs(kwargs, match):
    base_kwargs = {"sales_scale": np.ones(1)}
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        dense_signal_metrics(
            np.ones((1, 8, 1)),
            np.ones((1, 8, 1)),
            np.arange(8, dtype=float)[None],
            np.zeros((1, 8)),
            np.ones((1, 1), dtype=bool),
            **base_kwargs,
        )


def test_summary_validates_sales_and_sales_scale():
    metrics = np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    mask = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="sales"):
        summarize_signal_metrics(
            metrics, valid, np.array([[1.0, np.nan]]), mask, sales_scale=np.ones(1)
        )
    with pytest.raises(ValueError, match="sales_scale"):
        summarize_signal_metrics(metrics, valid, np.ones((1, 2)), mask, sales_scale=np.zeros(1))


def test_response_warmup_and_contemporaneous_diagnostics():
    metrics = np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    sales = np.arange(8, dtype=float)[None]
    mask = np.ones((1, 1), dtype=bool)
    no_burn_in = summarize_signal_metrics(
        metrics,
        valid,
        sales,
        mask,
        sales_scale=np.ones(1),
        l_max=8,
        adstock_burn_in=0,
    )
    assert no_burn_in["response_warmup_weeks"] == 0
    assert no_burn_in["frac_zero_contemporaneous_weight"] is None

    fallback = summarize_signal_metrics(
        metrics,
        valid,
        sales,
        mask,
        sales_scale=np.ones(1),
        l_max=8,
        adstock_burn_in=8,
    )
    assert fallback["response_warmup_weeks"] > 0

    eligible_identity = summarize_signal_metrics(
        np.zeros((1, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32),
        np.zeros((1, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.uint8),
        sales,
        np.array([[True, False]]),
        sales_scale=np.ones(1),
        l_max=8,
        adstock_burn_in=8,
        adstock_family=np.array([[0, 2]], dtype=np.uint8),
    )
    assert eligible_identity["response_warmup_weeks"] == 0

    with_burn_in = summarize_signal_metrics(
        metrics,
        valid,
        sales,
        mask,
        sales_scale=np.ones(1),
        l_max=8,
        adstock_burn_in=8,
        adstock_family=np.array([[2]], dtype=np.uint8),
        adstock_alpha=np.array([[0.0]], dtype=np.float32),
        weibull_lam=np.array([[8.0]], dtype=np.float32),
        weibull_k=np.array([[4.0]], dtype=np.float32),
    )
    assert with_burn_in["response_warmup_weeks"] > 0
    assert with_burn_in["frac_zero_contemporaneous_weight"] == 1.0

    assert contemporaneous_weight(2, 0.0, 8.0, 4.0, 8) == 0.0
    assert contemporaneous_weight(1, 0.5, 1.0, 1.0, 8) > 0.0


def test_response_warmup_matches_persisted_response_boundary():
    """Persisted spend reproduces geometric response only after the reported boundary."""
    cfg = make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_cells=2,
        draws_per_cell=1,
        n_time_steps=8,
        l_max=4,
        adstock_burn_in=4,
        nonlinearity="linear",
        adstock_family_probs={
            "none": 0.0,
            "geometric": 1.0,
            "weibull": 0.0,
        },
        adstock_alpha_range=(0.8, 0.8),
        edge_budget=_direct_only_edge_budget(),
        seed=73,
    )
    corpus = DataGenerator(cfg).generate(validate=True)
    warmup = int(corpus["diagnostics"]["signal"]["response_warmup_weeks"])
    assert 0 < warmup < cfg.n_time_steps
    assert np.all(corpus["adstock_family"][:, 0] == 1)

    for task in range(corpus["spend_raw"].shape[0]):
        spend = corpus["spend_raw"][task, :, 0].astype(np.float64)
        contribution = corpus["contributions_raw"][task, :, 0].astype(np.float64)
        adstock = _adstock_numpy(
            spend,
            int(corpus["adstock_family"][task, 0]),
            float(corpus["adstock_alpha"][task, 0]),
            float(corpus["weibull_lam"][task, 0]),
            float(corpus["weibull_k"][task, 0]),
            cfg.l_max,
        )
        response = adstock / float(corpus["saturation_scale"][task, 0])
        # The corpus does not persist beta. Fit its one linear scale on the
        # reproducible tail, so any early error isolates missing response state.
        tail_response = response[warmup:]
        tail_contribution = contribution[warmup:]
        beta = float(
            np.dot(tail_response, tail_contribution) / np.dot(tail_response, tail_response)
        )
        reconstructed = beta * response
        relative_error = np.abs(reconstructed - contribution) / np.maximum(
            np.abs(contribution), 1e-12
        )
        assert relative_error[:warmup].min() > 0.1
        assert relative_error[warmup:].max() < 1e-4


def test_adstock_corpus_response_warmup_contrasts_and_self_validates():
    """With burn-in, diagnostics distinguish identity from geometric response state."""
    warmups = {}
    for name, family, family_probs in (
        ("identity", 0, {"none": 1.0, "geometric": 0.0, "weibull": 0.0}),
        ("geometric", 1, {"none": 0.0, "geometric": 1.0, "weibull": 0.0}),
    ):
        cfg = make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_cells=2,
            draws_per_cell=1,
            n_time_steps=20,
            l_max=5,
            nonlinearity="linear",
            adstock_family_probs=family_probs,
            edge_budget=_direct_only_edge_budget(),
            seed=74,
        )
        assert cfg.adstock_burn_in == cfg.l_max
        corpus = DataGenerator(cfg).generate(validate=True)

        assert np.all(corpus["adstock_family"] == family)
        warmups[name] = corpus["diagnostics"]["signal"]["response_warmup_weeks"]
        assert DataGenerator.validate_corpus(corpus) == []

    assert warmups == {"identity": 0, "geometric": 4}


def test_default_gate_rejects_low_amplitude_and_collinear_targets():
    metrics = np.zeros((1, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    for key, values in (
        ("contrib_cv", (0.2, 0.2)),
        ("contrib_hf", (0.2, 0.2)),
        ("spearman", (0.5, 0.5)),
        ("contrib_rel_std", (0.005, 0.02)),
        ("contrib_r2_explained_by_rest", (0.96, 0.9)),
    ):
        index = _metric(key)
        metrics[0, :, index] = values
        valid[0, :, index] = 1

    summary = summarize_signal_metrics(
        metrics,
        valid,
        np.arange(3, dtype=float)[None],
        np.ones((1, 2), dtype=bool),
        sales_scale=np.ones(1),
        l_max=3,
        adstock_burn_in=3,
    )
    assert summary["frac_contrib_rel_std_lt_001"] == 0.5
    assert summary["frac_contrib_r2_gt_095"] == 0.5
    assert DEFAULT_GATE["frac_contrib_rel_std_lt_001"] == 0.10
    assert DEFAULT_GATE["frac_contrib_r2_gt_095"] == 0.10
    ok, lines = check_signal_gate(summary)
    assert not ok
    assert any("[FAIL] frac_contrib_rel_std_lt_001" in line for line in lines)
    assert any("[FAIL] frac_contrib_r2_gt_095" in line for line in lines)
