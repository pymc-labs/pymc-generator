"""Corpus schema, validation, and .npz persistence (the PFN-consumable format)."""

from __future__ import annotations

import numpy as np
import pytest

import prior_generator as pg
from prior_generator import DataGenerator
from prior_generator.signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from prior_generator.slots import EDGE_TYPES_EXTENDED, SlotLayout


@pytest.fixture(scope="module")
def corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4, n_covariates=2, n_latent=1, T=40, n_cells=2, draws_per_cell=3, seed=7
    )
    return pg.sample_prior_predictive(cfg)


def test_required_keys_and_shapes(corpus):
    N, T, K = corpus["spend_raw"].shape
    M = corpus["controls"].shape[2]
    J = corpus["demand"].shape[2]
    expected = {
        "spend_raw": (N, T, K),
        "spend_norm": (N, T, K),
        "spend_share": (N, T, K),
        "controls": (N, T, M),
        "sales_raw": (N, T),
        "sales_norm": (N, T),
        "support_mask": (N, T),
        "contributions_raw": (N, T, K),
        "baseline_raw": (N, T),
        "indirect_effects": (N, T),
        "indirect_effects_by_source": (N, T, 3),
        "control_contribution": (N, T, M),
        "confounder_contribution": (N, T, J),
        "baseline_intrinsic": (N, T),
        "channel_active": (N, K),
        "active_c_mask": (N, K),
    }
    for key, shape in expected.items():
        assert corpus[key].shape == shape, f"{key}: {corpus[key].shape} != {shape}"


def test_shock_and_mechanism_metadata_schema(corpus):
    N, T, K = corpus["spend_raw"].shape
    expected = {
        "channel_shock_mask": ((N, T, K), np.uint8),
        "channel_shock_channel": ((N, 0), np.int32),
        "channel_shock_start": ((N, 0), np.int32),
        "channel_shock_length": ((N, 0), np.int32),
        "channel_shock_level_multiplier": ((N, 0), np.float32),
        "channel_shock_level": ((N, 0), np.float32),
        "channel_level": ((N, K), np.float32),
        "adstock_family": ((N, K), np.uint8),
        "adstock_alpha": ((N, K), np.float32),
        "weibull_lam": ((N, K), np.float32),
        "weibull_k": ((N, K), np.float32),
    }
    for key, (shape, dtype) in expected.items():
        assert corpus[key].shape == shape
        assert corpus[key].dtype == dtype
    assert not corpus["channel_shock_mask"].any()


def test_g_layout_is_extended_8_block(corpus):
    N, T, K = corpus["spend_raw"].shape
    M, J = corpus["controls"].shape[2], corpus["demand"].shape[2]
    layout = SlotLayout(K=K, M=M, J=J, edge_types=EDGE_TYPES_EXTENDED)
    assert corpus["g"].shape == (N, layout.n_slots)
    assert set(np.unique(corpus["g"])).issubset({0, 1})


def test_validate_corpus_accepts_generated(corpus):
    assert DataGenerator.validate_corpus(corpus) == []


def test_validate_corpus_flags_missing_key(corpus):
    broken = {k: v for k, v in corpus.items() if k != "sales_scale"}
    errors = DataGenerator.validate_corpus(broken)
    assert any("sales_scale" in e for e in errors)


def test_validate_corpus_flags_nan(corpus):
    broken = dict(corpus)
    bad = corpus["spend_raw"].copy()
    bad[0, 0, 0] = np.nan
    broken["spend_raw"] = bad
    errors = DataGenerator.validate_corpus(broken)
    assert any("NaN" in e or "Inf" in e for e in errors)


def test_save_load_roundtrip(tmp_path, corpus):
    path = tmp_path / "corpus.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key, val in corpus.items():
        if key == "diagnostics":
            assert loaded[key]["edge_types"] == list(EDGE_TYPES_EXTENDED)
        else:
            assert np.array_equal(loaded[key], val), f"{key} changed across roundtrip"


def test_datagenerator_generate_n_tasks():
    cfg = pg.make_scm_prior(
        n_treatments=4, n_covariates=2, n_latent=1, T=32, draws_per_cell=5, seed=1
    )
    gen = DataGenerator(cfg)
    corpus = gen.generate(n_tasks=7, seed=1)
    assert corpus["spend_raw"].shape[0] == 7
    assert corpus["is_val"].sum() >= 1
    assert corpus["channel_shock_mask"].shape[0] == 7
    assert corpus["channel_level"].shape[0] == 7


def test_finalization_uses_retained_tasks_for_truncated_public_paths(tmp_path):
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        T=16,
        n_cells=2,
        draws_per_cell=2,
        seed=73,
    )
    n_tasks = 3
    full = pg.sample_prior_predictive(cfg)
    truncated = pg.sample_prior_predictive(cfg, n=n_tasks)
    generated = DataGenerator(cfg).generate(n_tasks=n_tasks, validate=True)

    for corpus in (truncated, generated):
        for value in corpus.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                assert value.shape[0] == n_tasks
        assert not any(key.startswith("_temp_") for key in corpus)
        assert corpus["is_val"].sum() >= 1
        assert corpus["diagnostics"]["n_tasks"] == n_tasks
        assert corpus["diagnostics"]["n_cells"] == len(np.unique(corpus["cell_id"]))

        layout = SlotLayout(K=2, M=2, J=1, edge_types=EDGE_TYPES_EXTENDED)
        direct = (corpus["g"][:, layout.slices["cy"]] == 1) & (corpus["active_c_mask"] == 1)
        expected_metrics, expected_valid = dense_signal_metrics(
            corpus["spend_raw"],
            corpus["contributions_raw"],
            corpus["sales_raw"],
            corpus["baseline_raw"],
            direct,
            sales_scale=corpus["sales_scale"],
            adstock_family=corpus["adstock_family"],
            adstock_alpha=corpus["adstock_alpha"],
            weibull_lam=corpus["weibull_lam"],
            weibull_k=corpus["weibull_k"],
            channel_shock_channel=corpus["channel_shock_channel"],
            channel_shock_start=corpus["channel_shock_start"],
            l_max=cfg.l_max,
            adstock_burn_in=cfg.adstock_burn_in,
        )
        assert np.array_equal(corpus["signal_metrics"], expected_metrics)
        assert np.array_equal(corpus["signal_metric_valid"], expected_valid)
        expected_signal = summarize_signal_metrics(
            expected_metrics,
            expected_valid,
            corpus["sales_raw"],
            direct,
            sales_scale=corpus["sales_scale"],
            l_max=cfg.l_max,
            adstock_burn_in=cfg.adstock_burn_in,
        )
        expected_signal["metric_version"] = SIGNAL_METRIC_VERSION
        expected_signal["metric_layout"] = list(SIGNAL_METRIC_LAYOUT)
        assert corpus["diagnostics"]["signal"] == expected_signal

    for key, value in full.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and key != "is_val":
            assert np.array_equal(truncated[key], value[:n_tasks]), key

    path = tmp_path / "truncated.npz"
    pg.save_corpus(generated, path)
    loaded = pg.load_corpus(path)
    assert DataGenerator.validate_corpus(loaded) == []


def test_validate_corpus_flags_corrupt_shock_metadata(corpus):
    broken = dict(corpus)
    bad = corpus["channel_shock_mask"].copy()
    bad[0, 0, 0] = 2
    broken["channel_shock_mask"] = bad
    errors = DataGenerator.validate_corpus(broken)
    assert any("channel_shock_mask is not binary" in error for error in errors)


def test_shock_metadata_is_reconstructable_and_zero_padded():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        T=24,
        n_cells=2,
        draws_per_cell=1,
        n_treatments_active_range=(2, 3),
        n_channel_shocks=2,
        channel_shock_length_range=(1, 2),
        channel_shock_level_range=(0.0, 1.0),
        seed=14,
    )
    generated = pg.sample_prior_predictive(cfg)
    assert DataGenerator.validate_corpus(generated) == []
    inactive = generated["active_c_mask"] == 0
    for key in ("channel_level", "adstock_family", "adstock_alpha", "weibull_lam", "weibull_k"):
        assert not generated[key][inactive].any(), key

    broken = dict(generated)
    bad_start = generated["channel_shock_start"].copy()
    bad_start[0, 0] = cfg.T
    broken["channel_shock_start"] = bad_start
    errors = DataGenerator.validate_corpus(broken)
    assert any("start/length" in error or "mask does not match" in error for error in errors)

    broken = dict(generated)
    bad_level = generated["channel_shock_level"].copy()
    bad_level[0, 0] += 1.0
    broken["channel_shock_level"] = bad_level
    errors = DataGenerator.validate_corpus(broken)
    assert any("multiplier * channel_level" in error for error in errors)


def test_sales_norm_matches_scale(corpus):
    expected = (
        corpus["sales_raw"].astype(np.float64) / corpus["sales_scale"].astype(np.float64)[:, None]
    )
    assert np.allclose(corpus["sales_norm"].astype(np.float64), expected, rtol=1e-5)
