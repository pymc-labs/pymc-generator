"""Deterministic contracts for persisted signal diagnostics."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt

from prior_generator import DataGenerator, load_corpus, mechanisms, save_corpus
from prior_generator.signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    _adstock_numpy,
    _reset_adstock_numpy,
    check_signal_gate,
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
    return dense_signal_metrics(
        x, y, sales, np.asarray(baseline, dtype=np.float32)[None], np.ones((1, 1)), **kwargs
    )


def _metric(name: str) -> int:
    return SIGNAL_METRIC_LAYOUT.index(name)


def test_numpy_adstock_matches_pytensor_mechanisms():
    x = np.array([0.2, 1.0, 0.5, 2.0, 0.7], dtype=np.float64)
    x_symbolic = pt.as_tensor_variable(x[:, None])
    geo = pytensor.function([], mechanisms.apply_geometric_adstock(x_symbolic, 0.6, 4))()[:, 0]
    weibull = pytensor.function(
        [], mechanisms.apply_weibull_pdf_adstock(x_symbolic, 3.0, 2.0, 4)
    )()[:, 0]
    assert np.allclose(_adstock_numpy(x, 1, 0.6, 1.0, 1.0, 4), geo)
    assert np.allclose(_adstock_numpy(x, 2, 0.0, 3.0, 2.0, 4), weibull)


def test_reset_adstock_is_chronological():
    x = np.arange(1, 9, dtype=np.float64)
    result = _reset_adstock_numpy(x, 1, 0.5, 1.0, 1.0, 3, np.array([5, 2]))
    expected = _adstock_numpy(x, 1, 0.5, 1.0, 1.0, 3)
    expected[2:] = _adstock_numpy(x[2:], 1, 0.5, 1.0, 1.0, 3)
    expected[5:] = _adstock_numpy(x[5:], 1, 0.5, 1.0, 1.0, 3)
    assert np.array_equal(result, expected)


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
        spend, contributions, np.ones((1, 6), dtype=np.float32), baseline[None], np.ones((1, 2))
    )
    assert np.isclose(metrics[0, 0, _metric("contrib_r2_explained_by_rest")], 1.0)

    negative, _ = _dense(x, -baseline, baseline=baseline)
    assert np.isclose(negative[0, 0, _metric("contrib_corr_baseline")], -1.0)


def test_short_constant_and_warmup_validity_contracts():
    short, short_valid = _dense(np.arange(2), np.arange(2), l_max=3)
    assert short_valid[0, 0, _metric("spend_hf")] == 1
    assert short_valid[0, 0, _metric("spearman")] == 0
    assert short_valid[0, 0, _metric("warmup_ratio")] == 0

    constant, constant_valid = _dense(np.ones(6), np.ones(6), l_max=3)
    assert constant_valid[0, 0, _metric("spearman")] == 1
    assert constant[0, 0, _metric("spearman")] == 0.0

    x, y = np.arange(8), np.array([0, 3, 6, 1, 1, 1, 1, 1])
    with_warmup, warmup_valid = _dense(x, y, l_max=3)
    assert warmup_valid[0, 0, _metric("warmup_ratio")] == 1
    assert with_warmup[0, 0, _metric("warmup_ratio")] > 3.0
    _, burned_valid = _dense(x, y, l_max=3, adstock_burn_in=3)
    assert burned_valid[0, 0, _metric("warmup_ratio")] == 0


def test_summary_uses_valid_denominators_and_gate_contracts():
    metrics = np.zeros((1, 2, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    cv = _metric("contrib_cv")
    metrics[0, :, cv] = [0.2, 0.0]
    valid[0, 0, cv] = 1
    summary = summarize_signal_metrics(
        metrics, valid, np.ones((1, 3)), np.ones((1, 2)), l_max=3, adstock_burn_in=3
    )
    assert summary["frac_contrib_cv_lt_010"] == 0.0
    assert summary["frac_spearman_lt_03"] is None
    assert summary["frac_warmup_gt_3"] == 0.0
    ok, lines = check_signal_gate(summary)
    assert not ok and any("frac_spearman_lt_03 missing" in line for line in lines)


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
        T=16,
        n_cells=2,
        draws_per_cell=1,
        seed=37,
    )
    corpus = sample_prior_predictive(cfg)
    path = tmp_path / "generated.npz"
    save_corpus(corpus, path)
    loaded = load_corpus(path)
    layout = SlotLayout(K=2, M=2, J=1, edge_types=EDGE_TYPES_EXTENDED)
    direct = (loaded["g"][:, layout.slices["cy"]] == 1) & (loaded["active_c_mask"] == 1)
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
        channel_shock_channel=loaded["channel_shock_channel"],
        channel_shock_start=loaded["channel_shock_start"],
        l_max=cfg.l_max,
        adstock_burn_in=cfg.adstock_burn_in,
    )
    assert np.array_equal(loaded["signal_metrics"], metrics)
    assert np.array_equal(loaded["signal_metric_valid"], valid)


def test_validator_checks_signal_layout_dtype_and_eligibility():
    corpus = {
        "spend_raw": np.ones((1, 4, 1), dtype=np.float32),
        "spend_norm": np.ones((1, 4, 1), dtype=np.float32),
        "spend_share": np.ones((1, 4, 1), dtype=np.float32),
        "controls": np.ones((1, 4, 1), dtype=np.float32),
        "sales_raw": np.ones((1, 4), dtype=np.float32),
        "sales_norm": np.ones((1, 4), dtype=np.float32),
        "support_mask": np.ones((1, 4), dtype=np.uint8),
        "is_future": np.zeros(1, dtype=np.uint8),
        "g": np.zeros((1, 8), dtype=np.uint8),
        "contributions_raw": np.zeros((1, 4, 1), dtype=np.float32),
        "baseline_raw": np.ones((1, 4), dtype=np.float32),
        "demand": np.ones((1, 4, 1), dtype=np.float32),
        "spend_means": np.ones((1, 1), dtype=np.float32),
        "sales_scale": np.ones(1, dtype=np.float32),
        "is_val": np.zeros(1, dtype=np.uint8),
        "cell_id": np.zeros(1, dtype=np.int32),
        "active_c_mask": np.ones((1, 1), dtype=np.uint8),
        "channel_shock_mask": np.zeros((1, 4, 1), dtype=np.uint8),
        "channel_shock_channel": np.empty((1, 0), dtype=np.int32),
        "channel_shock_start": np.empty((1, 0), dtype=np.int32),
        "channel_shock_length": np.empty((1, 0), dtype=np.int32),
        "channel_shock_level_multiplier": np.empty((1, 0), dtype=np.float32),
        "channel_shock_level": np.empty((1, 0), dtype=np.float32),
        "channel_level": np.ones((1, 1), dtype=np.float32),
        "adstock_family": np.zeros((1, 1), dtype=np.uint8),
        "adstock_alpha": np.zeros((1, 1), dtype=np.float32),
        "weibull_lam": np.zeros((1, 1), dtype=np.float32),
        "weibull_k": np.zeros((1, 1), dtype=np.float32),
        "signal_metrics": np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32),
        "signal_metric_valid": np.zeros((1, 1, len(SIGNAL_METRIC_LAYOUT)), dtype=np.uint8),
        "diagnostics": {
            "signal": {
                "metric_version": SIGNAL_METRIC_VERSION,
                "metric_layout": list(SIGNAL_METRIC_LAYOUT),
            }
        },
    }
    corpus["signal_metrics"][0, 0, 0] = 1
    errors = DataGenerator.validate_corpus(corpus)
    assert any("ineligible" in error for error in errors)
    assert SIGNAL_METRIC_VERSION == 1
