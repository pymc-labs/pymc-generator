"""Corpus schema, validation, and .npz persistence (the PFN-consumable format)."""

from __future__ import annotations

import os

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


class _PicklePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


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
        "confounding_strength": (N,),
        "saturation_scale": (N, K),
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
        "saturation_scale": ((N, K), np.float32),
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


@pytest.mark.parametrize("key", ("spend_raw", "controls", "demand", "g"))
def test_validate_corpus_returns_errors_for_malformed_core_dimensions(corpus, key):
    broken = dict(corpus)
    broken[key] = corpus[key][0]
    errors = DataGenerator.validate_corpus(broken)
    assert any(key in error and "ndim" in error for error in errors)


def test_validate_corpus_rejects_wrong_graph_slot_width(corpus):
    broken = dict(corpus)
    broken["g"] = np.pad(corpus["g"], ((0, 0), (0, 1)))
    errors = DataGenerator.validate_corpus(broken)
    assert any("Shape mismatch for g" in error for error in errors)


def test_validate_corpus_rejects_non_array_required_fields(corpus):
    broken = dict(corpus)
    broken["sales_raw"] = corpus["sales_raw"].tolist()
    assert DataGenerator.validate_corpus(broken) == ["sales_raw must be an ndarray"]


@pytest.mark.parametrize(
    "value",
    (
        np.array([np.nan] * 6, dtype=np.float32),
        np.array([-0.1] * 6, dtype=np.float32),
        np.array([0.96] * 6, dtype=np.float32),
        np.zeros(6, dtype=np.float64),
    ),
)
def test_validate_corpus_rejects_invalid_confounding_strength(corpus, value):
    broken = dict(corpus)
    broken["confounding_strength"] = value
    errors = DataGenerator.validate_corpus(broken)
    assert any("confounding_strength" in error for error in errors)


def test_validate_corpus_requires_confounding_strength(corpus):
    broken = {key: value for key, value in corpus.items() if key != "confounding_strength"}
    errors = DataGenerator.validate_corpus(broken)
    assert any("confounding_strength" in error for error in errors)


def test_validate_corpus_rejects_nonpositive_active_saturation_scale(corpus):
    broken = dict(corpus)
    bad = corpus["saturation_scale"].copy()
    bad[0, np.flatnonzero(corpus["active_c_mask"][0])[0]] = 0.0
    broken["saturation_scale"] = bad
    errors = DataGenerator.validate_corpus(broken)
    assert any("saturation_scale" in error for error in errors)


def test_single_node_edge_marginals_are_defined_without_empty_mean_warning(recwarn):
    generated = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            T=8,
            n_cells=2,
            draws_per_cell=1,
            seed=91,
        )
    )
    assert generated["diagnostics"]["edge_marginals"]["cc"] == 0.0
    assert generated["diagnostics"]["edge_marginals"]["zz"] == 0.0
    assert not any("Mean of empty slice" in str(item.message) for item in recwarn)


def test_save_load_roundtrip(tmp_path, corpus):
    path = tmp_path / "corpus.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key, val in corpus.items():
        if key == "diagnostics":
            assert loaded[key]["edge_types"] == list(EDGE_TYPES_EXTENDED)
        elif key == "identifiability":
            assert set(loaded[key]) == {"signal_metrics", "signal_metric_valid"}
            for label, value in val.items():
                assert np.array_equal(loaded[key][label], value)
        else:
            assert np.array_equal(loaded[key], val), f"{key} changed across roundtrip"
    assert all(not key.startswith("identifiability__") for key in loaded)
    assert isinstance(loaded["identifiability"], dict)


def test_load_corpus_refuses_pickle_payload_without_execution(tmp_path):
    path = tmp_path / "malicious.npz"
    marker = tmp_path / "executed"
    np.savez(path, payload=np.array([_PicklePayload(marker)], dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        pg.load_corpus(path)
    assert not marker.exists()


def test_numpy_scalar_diagnostics_validate_and_roundtrip(tmp_path, corpus):
    modified = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    signal = dict(diagnostics["signal"])
    signal.update(
        {
            "metric_version": np.int64(SIGNAL_METRIC_VERSION),
            "metric_layout": np.asarray(SIGNAL_METRIC_LAYOUT),
            "l_max": np.int32(signal["l_max"]),
            "adstock_burn_in": np.int64(signal["adstock_burn_in"]),
            "adstock_kernel_version": np.int64(1),
        }
    )
    diagnostics["signal"] = signal
    modified["diagnostics"] = diagnostics
    assert DataGenerator.validate_corpus(modified) == []

    path = tmp_path / "numpy-diagnostics.npz"
    pg.save_corpus(modified, path)
    loaded = pg.load_corpus(path)
    assert DataGenerator.validate_corpus(loaded) == []


def test_validate_corpus_recomputes_labels_and_signal_summary(corpus):
    labels = corpus["identifiability"]
    metric_index = SIGNAL_METRIC_LAYOUT.index("contrib_r2_explained_by_rest")
    eligible = np.argwhere(labels["signal_metric_valid"][..., metric_index] == 1)
    assert eligible.size
    n, k = eligible[0]

    bad_label = dict(corpus)
    bad_label["identifiability"] = dict(labels)
    bad_label["identifiability"]["signal_metrics"] = labels["signal_metrics"].copy()
    old = labels["signal_metrics"][n, k, metric_index]
    bad_label["identifiability"]["signal_metrics"][n, k, metric_index] = 0.0 if old > 0.5 else 1.0
    assert DataGenerator.validate_corpus(bad_label) == [
        "identifiability signal_metrics do not match recomputation"
    ]

    bad_summary = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"]["frac_spearman_lt_03"] = 0.123
    bad_summary["diagnostics"] = diagnostics
    assert DataGenerator.validate_corpus(bad_summary) == [
        "diagnostics signal summary does not match recomputation"
    ]


@pytest.mark.parametrize("diagnostics", ([], {"signal": []}))
def test_validate_corpus_rejects_malformed_diagnostics(corpus, diagnostics):
    broken = dict(corpus)
    broken["diagnostics"] = diagnostics
    errors = DataGenerator.validate_corpus(broken)
    assert errors == [
        "diagnostics must be a mapping"
        if isinstance(diagnostics, list)
        else "diagnostics signal must be a mapping"
    ]


def test_save_corpus_requires_npz_suffix(tmp_path):
    with pytest.raises(ValueError, match="must use the .npz suffix"):
        pg.save_corpus({"x": np.ones(1)}, tmp_path / "corpus.data")


def test_save_corpus_rejects_reserved_prefix_and_empty_identifiability(tmp_path):
    with pytest.raises(ValueError, match="reserved identifiability__ prefix"):
        pg.save_corpus(
            {"identifiability__signal_metrics": np.ones(1)},
            tmp_path / "collision.npz",
        )
    with pytest.raises(ValueError, match="omitted rather than empty"):
        pg.save_corpus({"identifiability": {}}, tmp_path / "empty.npz")


@pytest.mark.parametrize(
    "diagnostics",
    (np.array(["{}"]), np.array("{not json"), np.array("[]")),
)
def test_load_corpus_rejects_malformed_diagnostics(tmp_path, diagnostics):
    path = tmp_path / "malformed.npz"
    np.savez(path, diagnostics=diagnostics)
    with pytest.raises(ValueError, match="diagnostics"):
        pg.load_corpus(path)


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("spearman", -0.1),
        ("contrib_r2_explained_by_rest", 1.1),
        ("contrib_corr_baseline", 1.1),
    ),
)
def test_validate_corpus_rejects_out_of_domain_signal_metrics(corpus, metric, value):
    index = SIGNAL_METRIC_LAYOUT.index(metric)
    labels = corpus["identifiability"]
    eligible = np.argwhere(labels["signal_metric_valid"][..., index] == 1)
    assert eligible.size
    n, k = eligible[0]
    broken = dict(corpus)
    broken["identifiability"] = dict(labels)
    broken["identifiability"]["signal_metrics"] = labels["signal_metrics"].copy()
    broken["identifiability"]["signal_metrics"][n, k, index] = value
    errors = DataGenerator.validate_corpus(broken)
    assert any(metric in error for error in errors)


@pytest.mark.parametrize("key", ("spend_norm", "spend_share", "sales_norm"))
def test_validate_corpus_rejects_nonfinite_normalized_inputs(corpus, key):
    broken = dict(corpus)
    broken[key] = corpus[key].copy()
    broken[key].flat[0] = np.nan
    errors = DataGenerator.validate_corpus(broken)
    assert any(f"{key} contains NaN or Inf" in error for error in errors)


@pytest.mark.parametrize("key", ("spend_norm", "spend_share"))
def test_validate_corpus_rejects_incorrect_normalized_inputs(corpus, key):
    broken = dict(corpus)
    broken[key] = corpus[key].copy()
    broken[key].flat[0] += 0.25
    errors = DataGenerator.validate_corpus(broken)
    assert any(key in error for error in errors)


def test_validate_corpus_rejects_nonpositive_sales_scale(corpus):
    broken = dict(corpus)
    broken["sales_scale"] = corpus["sales_scale"].copy()
    broken["sales_scale"][0] = 0.0
    errors = DataGenerator.validate_corpus(broken)
    assert "sales_scale must be positive" in errors


def test_validate_corpus_rejects_unverifiable_metric_version(corpus):
    broken = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"]["metric_version"] = 1
    broken["diagnostics"] = diagnostics
    errors = DataGenerator.validate_corpus(broken)
    assert "diagnostics signal metric_version is not supported" in errors


def test_validate_corpus_rejects_empty_identifiability_block(corpus):
    broken = dict(corpus)
    broken["identifiability"] = {}
    assert DataGenerator.validate_corpus(broken) == [
        "identifiability signal_metrics and signal_metric_valid must be present together"
    ]


def test_validate_corpus_rejects_corrupt_layout_metrics_and_family(corpus):
    bad_layout = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"]["metric_layout"] = [*SIGNAL_METRIC_LAYOUT, "unknown"]
    bad_layout["diagnostics"] = diagnostics
    assert any("metric_layout" in error for error in DataGenerator.validate_corpus(bad_layout))

    bad_metric = dict(corpus)
    bad_metric["identifiability"] = dict(corpus["identifiability"])
    bad_metric["identifiability"]["signal_metrics"] = corpus["identifiability"][
        "signal_metrics"
    ].copy()
    bad_metric["identifiability"]["signal_metrics"].flat[0] = np.nan
    assert any(
        "signal_metrics contains" in error for error in DataGenerator.validate_corpus(bad_metric)
    )

    bad_family = dict(corpus)
    bad_family["adstock_family"] = corpus["adstock_family"].copy()
    bad_family["adstock_family"][0, 0] = 3
    assert any("adstock_family" in error for error in DataGenerator.validate_corpus(bad_family))


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
        for value in corpus["identifiability"].values():
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
        assert np.array_equal(corpus["identifiability"]["signal_metrics"], expected_metrics)
        assert np.array_equal(corpus["identifiability"]["signal_metric_valid"], expected_valid)
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
        expected_signal["l_max"] = cfg.l_max
        expected_signal["adstock_burn_in"] = cfg.adstock_burn_in
        expected_signal["adstock_kernel_semantics"] = "normalized-causal-reset-aware-weibull-pdf"
        expected_signal["adstock_kernel_version"] = 1
        assert corpus["diagnostics"]["signal"] == expected_signal

    for key, value in full.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and key != "is_val":
            assert np.array_equal(truncated[key], value[:n_tasks]), key
            assert np.array_equal(generated[key], value[:n_tasks]), key
    for label, value in full["identifiability"].items():
        assert np.array_equal(truncated["identifiability"][label], value[:n_tasks])
        assert np.array_equal(generated["identifiability"][label], value[:n_tasks])

    path = tmp_path / "truncated.npz"
    pg.save_corpus(generated, path)
    loaded = pg.load_corpus(path)
    assert DataGenerator.validate_corpus(loaded) == []


def test_datagenerator_generate_and_save_without_identifiability_labels(tmp_path):
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        T=16,
        n_cells=2,
        draws_per_cell=1,
        seed=57,
        include_identifiability_labels=False,
        confounding_strength_range=(0.3, 0.3),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
    )
    path = tmp_path / "feature-only.npz"
    pg.DataGenerator(cfg).generate_and_save(path)
    loaded = pg.load_corpus(path)
    assert "identifiability" not in loaded
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
        channel_shock_length_range=(2, 2),
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
