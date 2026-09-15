"""Corpus schema, validation, and .npz persistence (the PFN-consumable format)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import pymc_generator as pg
import pymc_generator.world_model as world_model
from pymc_generator import DataGenerator
from pymc_generator.sampler import OUTCOME_NOISE_SEMANTICS, OUTCOME_NOISE_VERSION
from pymc_generator.signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from pymc_generator.slots import (
    CORPUS_SCHEMA_VERSION,
    EDGE_TYPES_EXTENDED,
    LEGACY_CORPUS_KEYS_V1,
    SlotLayout,
)


class _PicklePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


@pytest.fixture(scope="module")
def corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=40,
        n_cells=2,
        draws_per_cell=3,
        seed=7,
    )
    return pg.sample_prior_predictive(cfg)


@pytest.fixture(scope="module")
def parentless_baseline_corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=40,
        n_cells=2,
        draws_per_cell=1,
        seed=19,
        edge_budget={"dy": (0, 0), "zy": (0, 0)},
    )
    return pg.sample_prior_predictive(cfg)


@pytest.fixture(scope="module")
def padded_corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=3,
        n_latent=2,
        n_treatments_active_range=(2, 2),
        n_covariates_active_range=(1, 1),
        n_latent_active_range=(1, 1),
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        seed=107,
    )
    return pg.sample_prior_predictive(cfg)


def test_required_keys_and_shapes(corpus):
    n_tasks, n_time_steps, n_treatments = corpus["spend_raw"].shape
    n_covariates = corpus["controls"].shape[2]
    n_latent = corpus["demand"].shape[2]
    expected = {
        "spend_raw": (n_tasks, n_time_steps, n_treatments),
        "spend_norm": (n_tasks, n_time_steps, n_treatments),
        "spend_share": (n_tasks, n_time_steps, n_treatments),
        "controls": (n_tasks, n_time_steps, n_covariates),
        "sales_raw": (n_tasks, n_time_steps),
        "sales_norm": (n_tasks, n_time_steps),
        "support_mask": (n_tasks, n_time_steps),
        "contributions_raw": (n_tasks, n_time_steps, n_treatments),
        "baseline_raw": (n_tasks, n_time_steps),
        "indirect_effects": (n_tasks, n_time_steps),
        "indirect_effects_by_source": (n_tasks, n_time_steps, 3),
        "control_contribution": (n_tasks, n_time_steps, n_covariates),
        "confounder_contribution": (n_tasks, n_time_steps, n_latent),
        "baseline_intrinsic": (n_tasks, n_time_steps),
        "sales_noise": (n_tasks, n_time_steps),
        "channel_active": (n_tasks, n_treatments),
        "treatment_active_mask": (n_tasks, n_treatments),
        "covariate_active_mask": (n_tasks, n_covariates),
        "latent_active_mask": (n_tasks, n_latent),
        "n_treatments_active": (n_tasks,),
        "n_covariates_active": (n_tasks,),
        "n_latent_active": (n_tasks,),
        "confounding_strength": (n_tasks,),
        "saturation_scale": (n_tasks, n_treatments),
    }
    for key, shape in expected.items():
        assert corpus[key].shape == shape, f"{key}: {corpus[key].shape} != {shape}"


def test_shock_and_mechanism_metadata_schema(corpus):
    n_tasks, n_time_steps, n_treatments = corpus["spend_raw"].shape
    expected = {
        "channel_shock_mask": ((n_tasks, n_time_steps, n_treatments), np.uint8),
        "channel_shock_channel": ((n_tasks, 0), np.int32),
        "channel_shock_start": ((n_tasks, 0), np.int32),
        "channel_shock_length": ((n_tasks, 0), np.int32),
        "channel_shock_level_multiplier": ((n_tasks, 0), np.float32),
        "channel_shock_level": ((n_tasks, 0), np.float32),
        "channel_level": ((n_tasks, n_treatments), np.float32),
        "saturation_scale": ((n_tasks, n_treatments), np.float32),
        "adstock_family": ((n_tasks, n_treatments), np.uint8),
        "adstock_alpha": ((n_tasks, n_treatments), np.float32),
        "weibull_lam": ((n_tasks, n_treatments), np.float32),
        "weibull_k": ((n_tasks, n_treatments), np.float32),
    }
    for key, (shape, dtype) in expected.items():
        assert corpus[key].shape == shape
        assert corpus[key].dtype == dtype
    assert not corpus["channel_shock_mask"].any()


def test_g_layout_is_extended_8_block(corpus):
    n_tasks, n_time_steps, n_treatments = corpus["spend_raw"].shape
    n_covariates, n_latent = corpus["controls"].shape[2], corpus["demand"].shape[2]
    layout = SlotLayout(
        n_treatments=n_treatments,
        n_covariates=n_covariates,
        n_latent=n_latent,
        edge_types=EDGE_TYPES_EXTENDED,
    )
    assert corpus["g"].shape == (n_tasks, layout.n_slots)
    assert set(np.unique(corpus["g"])).issubset({0, 1})


def test_validate_corpus_accepts_generated(corpus):
    assert DataGenerator.validate_corpus(corpus) == []


def test_query_windows_do_not_overlap_response_warmup(corpus):
    """Every query suffix begins after the non-reproducible response prefix."""
    n_time_steps = corpus["spend_raw"].shape[1]
    n_query = int(corpus["diagnostics"]["short_horizon_n_query"])
    warmup = int(corpus["diagnostics"]["signal"]["response_warmup_weeks"])
    assert min(n_time_steps - n_query, n_time_steps // 2) >= warmup


def test_persisted_spend_ratios_are_unit_invariant(monkeypatch):
    """Persisted spend features stay ratios when the monetary unit changes."""
    draw_worlds = world_model.draw_worlds
    unit_scale = 1e-16

    def draw_with_rescaled_spend(*args, **kwargs):
        drawn = draw_worlds(*args, **kwargs)
        drawn["channels"] = drawn["channels"] * unit_scale
        return drawn

    monkeypatch.setattr(world_model, "draw_worlds", draw_with_rescaled_spend)
    cfg = pg.make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(2, 2),
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        seed=29,
    )
    generated = pg.sample_prior_predictive(cfg)

    def normalized_spend(spend):
        means = spend.mean(axis=1)
        return np.divide(
            spend, means[:, None, :], out=np.zeros_like(spend), where=means[:, None, :] != 0.0
        )

    def spend_shares(spend):
        active = generated["treatment_active_mask"].astype(np.float64)[:, None, :]
        active_sum = (spend * active).sum(axis=-1, keepdims=True)
        return (
            np.divide(spend, active_sum, out=np.zeros_like(spend), where=active_sum != 0.0) * active
        )

    def spend_cv_quantiles(spend):
        means = spend.mean(axis=1)
        cv = np.divide(
            spend.std(axis=1), means, out=np.zeros_like(means), where=means != 0.0
        ).ravel()
        return np.quantile(cv, (0.1, 0.5, 0.9))

    spend = generated["spend_raw"].astype(np.float64)
    rescaled_spend = spend * 1e8
    np.testing.assert_allclose(
        generated["spend_norm"], normalized_spend(rescaled_spend), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        generated["spend_share"], spend_shares(rescaled_spend), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        spend_cv_quantiles(spend), spend_cv_quantiles(rescaled_spend), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        [
            generated["diagnostics"]["spend_cv_quantiles"][f"q{int(q * 100)}"]
            for q in (0.1, 0.5, 0.9)
        ],
        spend_cv_quantiles(rescaled_spend),
        rtol=1e-5,
        atol=1e-6,
    )
    inactive = generated["treatment_active_mask"] == 0
    for key in ("spend_raw", "spend_norm", "spend_share"):
        assert not generated[key].transpose(0, 2, 1)[inactive].any(), key
    assert DataGenerator.validate_corpus(generated) == []


def test_validate_corpus_flags_missing_key(corpus):
    broken = {k: v for k, v in corpus.items() if k != "sales_scale"}
    errors = DataGenerator.validate_corpus(broken)
    assert any("sales_scale" in e for e in errors)


@pytest.mark.parametrize(
    "key",
    (
        "covariate_active_mask",
        "latent_active_mask",
        "n_treatments_active",
        "n_covariates_active",
        "n_latent_active",
        "indirect_effects",
        "channel_active",
        "control_contribution",
        "confounder_contribution",
        "baseline_intrinsic",
        "sales_noise",
        "indirect_effects_by_source",
    ),
)
def test_validate_corpus_requires_complete_generated_schema(corpus, key):
    broken = {name: value for name, value in corpus.items() if name != key}
    assert any(key in error for error in DataGenerator.validate_corpus(broken))


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


def test_validate_corpus_rejects_non_array_extra_fields(corpus):
    broken = dict(corpus)
    broken["extra"] = "not an array"
    assert DataGenerator.validate_corpus(broken) == ["extra must be an ndarray"]


@pytest.mark.parametrize(
    ("key", "dtype"),
    (("spend_raw", np.float64), ("g", np.int32), ("n_treatments_active", np.int64)),
)
def test_validate_corpus_rejects_wrong_required_dtypes(corpus, key, dtype):
    broken = dict(corpus)
    broken[key] = corpus[key].astype(dtype)
    assert any(f"{key} has dtype" in error for error in DataGenerator.validate_corpus(broken))


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
    bad[0, np.flatnonzero(corpus["treatment_active_mask"][0])[0]] = 0.0
    broken["saturation_scale"] = bad
    errors = DataGenerator.validate_corpus(broken)
    assert any("saturation_scale" in error for error in errors)


def test_single_node_edge_marginals_are_defined_without_empty_mean_warning(recwarn):
    generated = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_time_steps=8,
            adstock_burn_in=0,
            n_cells=2,
            draws_per_cell=1,
            seed=91,
        )
    )
    assert generated["diagnostics"]["edge_marginals"]["cc"] == 0.0
    assert generated["diagnostics"]["edge_marginals"]["zz"] == 0.0
    assert not any("Mean of empty slice" in str(item.message) for item in recwarn)


def test_edge_base_rates_report_legacy_overrides():
    overrides = {"cy": 0.11, "dc": 0.22, "dy": 0.33, "zy": 0.44}
    corpus = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_time_steps=8,
            l_max=1,
            adstock_burn_in=0,
            n_cells=2,
            draws_per_cell=1,
            edge_rate_overrides=overrides,
            seed=419,
        )
    )
    rates = corpus["diagnostics"]["edge_base_rates"]
    assert {edge_type: rates[edge_type] for edge_type in overrides} == overrides


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


def _write_legacy_shard(path, corpus, version, *, diagnostics_overrides=None):
    """Write the historical vocabulary independently of the reader's mapping."""
    import json

    diagnostics = {
        key: value
        for key, value in corpus["diagnostics"].items()
        if key not in {"schema_version", "timing"}
    }
    diagnostics["edge_types"] = ["cy", "dc", "dz", "db", "zb", "zc", "cc", "zz"]
    for field in ("edge_base_rates", "edge_marginals", "edge_budget"):
        values = diagnostics.get(field)
        if values is not None:
            diagnostics[field] = {
                {"dy": "db", "zy": "zb"}.get(key, key): value for key, value in values.items()
            }
    diagnostics["min_dead_channels"] = diagnostics.pop("min_no_direct_effect_channels")
    if version == 2:
        diagnostics["schema_version"] = 2
    diagnostics.update(diagnostics_overrides or {})
    payload = {"diagnostics": np.array(json.dumps(diagnostics))}
    for key, value in corpus.items():
        if key == "diagnostics":
            continue
        if key == "identifiability":
            for label, array in value.items():
                payload[f"identifiability__{label}"] = array
        else:
            payload[key] = value
    if version == 1:
        for old, new in LEGACY_CORPUS_KEYS_V1.items():
            payload[old] = payload.pop(new)
    np.savez_compressed(path, **payload)


@pytest.mark.parametrize("version", (1, 2))
def test_load_corpus_migrates_legacy_metadata_without_changing_arrays(tmp_path, corpus, version):
    path = tmp_path / f"v{version}.npz"
    _write_legacy_shard(path, corpus, version)
    loaded = pg.load_corpus(path)

    assert set(loaded) == set(corpus)
    for key, value in corpus.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[key], value, err_msg=key)
    for key, value in corpus["identifiability"].items():
        np.testing.assert_array_equal(loaded["identifiability"][key], value, err_msg=key)
    assert loaded["diagnostics"] == {
        key: value for key, value in corpus["diagnostics"].items() if key != "timing"
    }
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("edge_types", ["cy", "dc", "dz", "zb", "db", "zc", "cc", "zz"], "edge order"),
        ("edge_types", list(EDGE_TYPES_EXTENDED), "edge order"),
        ("edge_base_rates", {"db": 0.5, "dy": 0.2}, "mixes"),
        ("edge_marginals", {"zb": 0.5, "zy": 0.2}, "mixes"),
        ("edge_budget", {"db": 1, "dy": 1}, "mixes"),
        ("edge_budget", [], "mapping"),
        ("min_no_direct_effect_channels", 7, "conflicting"),
    ),
)
def test_v2_migration_rejects_ambiguous_metadata(tmp_path, corpus, field, value, reason):
    path = tmp_path / "ambiguous-v2.npz"
    _write_legacy_shard(path, corpus, 2, diagnostics_overrides={field: value})
    with pytest.raises(ValueError, match=reason):
        pg.load_corpus(path)


@pytest.mark.parametrize(
    "field", ("edge_types", "edge_base_rates", "edge_marginals", "edge_budget")
)
def test_current_schema_rejects_legacy_edge_names_at_every_boundary(tmp_path, corpus, field):
    import json

    bad = dict(corpus)
    legacy_value = ["db"] if field == "edge_types" else {"db": 1}
    bad["diagnostics"] = {**corpus["diagnostics"], field: legacy_value}
    assert any("pre-v3" in error for error in DataGenerator.validate_corpus(bad))
    path = tmp_path / "mislabeled.npz"
    with pytest.raises(ValueError, match="pre-v3"):
        pg.save_corpus(bad, path)
    assert not path.exists()
    pg.save_corpus(corpus, path)
    with np.load(path, allow_pickle=False) as stored:
        payload = dict(stored)
    payload["diagnostics"] = np.array(json.dumps(bad["diagnostics"]))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="pre-v3"):
        pg.load_corpus(path)


def test_load_corpus_rejects_a_shard_mixing_both_vocabularies(tmp_path, corpus):
    path = tmp_path / "mixed.npz"
    _write_legacy_shard(path, corpus, 1)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["n_treatments_active"] = payload["K_active"]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="mixes v1 and v2 dimension keys"):
        pg.load_corpus(path)


def test_new_corpora_are_stamped_and_free_of_v1_keys(corpus):
    assert corpus["diagnostics"]["schema_version"] == CORPUS_SCHEMA_VERSION
    assert not set(corpus) & set(LEGACY_CORPUS_KEYS_V1)


def test_save_corpus_refuses_v1_dimension_keys(tmp_path, corpus):
    """Invariant: a written shard can only carry canonical names."""
    regressed = dict(corpus)
    regressed["K_active"] = regressed.pop("n_treatments_active")
    path = tmp_path / "regressed.npz"
    with pytest.raises(ValueError, match="symbolic dimension keys"):
        pg.save_corpus(regressed, path)
    assert not path.exists()


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
            "adstock_kernel_version": np.int64(3),
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
    ("payload", "message"),
    (
        ({"x": [1.0]}, "x must be an ndarray"),
        ({"x": np.array([object()], dtype=object)}, "x may not have object dtype"),
        (
            {"identifiability": {"label": [1.0]}},
            "identifiability.label must be an ndarray",
        ),
        (
            {"identifiability": {"label": np.array([object()], dtype=object)}},
            "identifiability.label may not have object dtype",
        ),
    ),
)
def test_save_corpus_rejects_unreadable_payloads(tmp_path, payload, message):
    with pytest.raises((TypeError, ValueError), match=message):
        pg.save_corpus(payload, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"diagnostics": [1, 2]}, "diagnostics must be a mapping"),
        ({"identifiability": [np.ones(1)]}, "identifiability metadata must be a mapping"),
        (
            {
                "diagnostics": {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "bad": np.array([object()], dtype=object),
                }
            },
            "diagnostics arrays may not have object dtype",
        ),
    ),
)
def test_save_corpus_rejects_malformed_nested_payloads(tmp_path, payload, message):
    with pytest.raises((TypeError, ValueError), match=message):
        pg.save_corpus(payload, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_validate_corpus_rejects_explicit_none_identifiability(corpus):
    broken = dict(corpus)
    broken["identifiability"] = None
    assert DataGenerator.validate_corpus(broken) == ["identifiability must be a mapping"]


def test_validate_corpus_returns_error_for_unsupported_extra_array(corpus):
    broken = dict(corpus)
    broken["extra"] = np.array([object()], dtype=object)
    assert any(
        "extra has unsupported dtype" in error for error in DataGenerator.validate_corpus(broken)
    )


def test_validate_and_save_reject_complex_arrays(tmp_path, corpus):
    broken = dict(corpus)
    broken["extra"] = np.array([1 + 2j], dtype=np.complex64)
    assert DataGenerator.validate_corpus(broken) == ["extra has unsupported dtype complex64"]
    with pytest.raises(ValueError, match="extra may not have complex dtype"):
        pg.save_corpus(broken, tmp_path / "complex.npz")


@pytest.mark.parametrize(
    "array",
    (
        np.array(["text"]),
        np.array([b"bytes"]),
        np.zeros(1, dtype=[("field", "f4")]),
    ),
)
def test_save_corpus_rejects_non_numeric_arrays(tmp_path, array):
    with pytest.raises(ValueError, match="x must have a real numeric dtype"):
        pg.save_corpus({"x": array}, tmp_path / "non-numeric.npz")


def test_save_corpus_rejects_non_numeric_identifiability_array(tmp_path):
    with pytest.raises(ValueError, match="identifiability.label must have a real numeric dtype"):
        pg.save_corpus(
            {"identifiability": {"label": np.array(["text"])}},
            tmp_path / "non-numeric-label.npz",
        )


@pytest.mark.parametrize(
    "value",
    ([1.0], np.array([object()]), np.array([1j]), np.array([np.nan])),
)
def test_nested_extensions_follow_numeric_persistence_contract(tmp_path, corpus, value):
    broken = dict(corpus)
    broken["identifiability"] = dict(corpus["identifiability"], extra=value)
    assert any("identifiability.extra" in error for error in DataGenerator.validate_corpus(broken))
    with pytest.raises((TypeError, ValueError), match="identifiability.extra"):
        pg.save_corpus(broken, tmp_path / "invalid-extension.npz")


def test_numeric_nested_extension_roundtrips(tmp_path, corpus):
    extended = dict(corpus)
    values = np.array([0.25, 0.75])
    extended["identifiability"] = dict(corpus["identifiability"], extra=values)
    assert DataGenerator.validate_corpus(extended) == []
    path = tmp_path / "extension.npz"
    pg.save_corpus(extended, path)
    np.testing.assert_array_equal(pg.load_corpus(path)["identifiability"]["extra"], values)


@pytest.mark.parametrize(
    "key",
    (
        "adstock_kernel_version",
        "adstock_kernel_semantics",
        "outcome_noise_version",
        "outcome_noise_semantics",
    ),
)
def test_array_valued_semantics_return_validation_errors(corpus, key):
    broken = dict(corpus)
    signal = dict(corpus["diagnostics"]["signal"])
    signal[key] = np.array([signal[key], signal[key]])
    broken["diagnostics"] = dict(corpus["diagnostics"], signal=signal)
    errors = DataGenerator.validate_corpus(broken)
    assert any("semantics are not supported" in error for error in errors)


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


def test_validate_corpus_checks_temporal_split_and_sales_scale(corpus):
    bad_support = dict(corpus)
    bad_support["support_mask"] = np.ones_like(corpus["support_mask"])
    assert "support_mask does not match the recorded temporal split" in (
        DataGenerator.validate_corpus(bad_support)
    )

    bad_split_type = dict(corpus)
    bad_split_type["is_future"] = corpus["is_future"].copy()
    bad_split_type["is_future"][0] ^= 1
    assert "support_mask does not match the recorded temporal split" in (
        DataGenerator.validate_corpus(bad_split_type)
    )

    bad_scale = dict(corpus)
    bad_scale["sales_scale"] = corpus["sales_scale"].copy()
    bad_scale["sales_scale"][0] += 1.0
    errors = DataGenerator.validate_corpus(bad_scale)
    assert "sales_scale does not match supported sales observations" in errors
    assert "sales_norm != sales_raw / sales_scale" in errors


def test_validate_corpus_rejects_unverifiable_metric_version(corpus):
    broken = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"]["metric_version"] = 2
    broken["diagnostics"] = diagnostics
    errors = DataGenerator.validate_corpus(broken)
    assert "diagnostics signal metric_version is not supported" in errors


def test_validate_corpus_handles_array_metric_version(corpus):
    broken = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"]["metric_version"] = np.array([2, 2])
    broken["diagnostics"] = diagnostics
    assert "diagnostics signal metric_version is not supported" in (
        DataGenerator.validate_corpus(broken)
    )


def test_validate_corpus_requires_current_signal_fields_and_edge_order(corpus):
    missing_warmup = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"].pop("response_warmup_weeks")
    missing_warmup["diagnostics"] = diagnostics
    assert any(
        "response_warmup_weeks" in error for error in DataGenerator.validate_corpus(missing_warmup)
    )

    missing_weight = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"].pop("frac_zero_contemporaneous_weight")
    missing_weight["diagnostics"] = diagnostics
    assert any(
        "summary does not match recomputation" in error
        for error in DataGenerator.validate_corpus(missing_weight)
    )

    reordered_edges = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["edge_types"] = list(reversed(diagnostics["edge_types"]))
    reordered_edges["diagnostics"] = diagnostics
    assert any(
        "edge_types does not match" in error
        for error in DataGenerator.validate_corpus(reordered_edges)
    )


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


@pytest.mark.parametrize("is_val_value", (0, 1))
def test_validate_corpus_requires_nonempty_train_and_validation_sides(corpus, is_val_value):
    broken = dict(corpus)
    broken["is_val"] = np.full_like(corpus["is_val"], is_val_value)

    assert "is_val must contain at least one training and one validation world" in (
        DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("field", "old_value"),
    (
        ("adstock_kernel_semantics", "normalized-causal-weibull-pdf"),
        ("adstock_kernel_version", 2),
    ),
)
def test_validate_corpus_rejects_previous_adstock_kernel_metadata(corpus, field, old_value):
    broken = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"][field] = old_value
    broken["diagnostics"] = diagnostics

    assert "diagnostics signal adstock kernel semantics are not supported" in (
        DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("field", "old_value"),
    (
        ("outcome_noise_semantics", "baseline-walk-random-walk-sales-noise"),
        ("outcome_noise_version", 0),
    ),
)
def test_validate_corpus_rejects_previous_outcome_noise_metadata(corpus, field, old_value):
    broken = dict(corpus)
    diagnostics = dict(corpus["diagnostics"])
    diagnostics["signal"] = dict(diagnostics["signal"])
    diagnostics["signal"][field] = old_value
    broken["diagnostics"] = diagnostics

    assert "diagnostics signal outcome noise semantics are not supported" in (
        DataGenerator.validate_corpus(broken)
    )


def test_datagenerator_generate_n_tasks():
    cfg = pg.make_scm_prior(
        n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=32, draws_per_cell=5, seed=1
    )
    gen = DataGenerator(cfg)
    corpus = gen.generate(n_tasks=7, seed=1)
    assert corpus["spend_raw"].shape[0] == 7
    assert corpus["is_val"].sum() >= 1
    assert corpus["channel_shock_mask"].shape[0] == 7
    assert corpus["channel_level"].shape[0] == 7


@pytest.mark.slow
@pytest.mark.parametrize("seed", (11, 23, 37))
@pytest.mark.parametrize("n_tasks", (2, 3, 6, 12, 20))
def test_datagenerator_small_task_counts_keep_cell_level_split(seed, n_tasks):
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=4,
        l_max=1,
        adstock_burn_in=0,
        draws_per_cell=20,
        nonlinearity="linear",
        spend_cv_floor=0.0,
        seed=seed,
    )
    corpus = DataGenerator(cfg).generate(n_tasks=n_tasks, validate=True)

    assert corpus["spend_raw"].shape[0] == n_tasks
    assert corpus["diagnostics"]["draws_per_cell"] == max(1, n_tasks // 2)
    assert 0 < corpus["is_val"].sum() < n_tasks
    cell_ids = np.unique(corpus["cell_id"])
    assert cell_ids.size >= 2
    for cell_id in cell_ids:
        assert np.unique(corpus["is_val"][corpus["cell_id"] == cell_id]).size == 1


def test_datagenerator_task_count_above_draws_per_cell_keeps_cell_level_split():
    n_tasks = 3
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=4,
        l_max=1,
        adstock_burn_in=0,
        draws_per_cell=2,
        nonlinearity="linear",
        spend_cv_floor=0.0,
        seed=11,
    )
    corpus = DataGenerator(cfg).generate(n_tasks=n_tasks, validate=True)

    assert corpus["spend_raw"].shape[0] == n_tasks
    assert corpus["diagnostics"]["draws_per_cell"] == 2
    assert 0 < corpus["is_val"].sum() < n_tasks
    cell_ids = np.unique(corpus["cell_id"])
    assert cell_ids.size >= 2
    for cell_id in cell_ids:
        assert np.unique(corpus["is_val"][corpus["cell_id"] == cell_id]).size == 1


def test_datagenerator_rejects_one_task_cell_split():
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=8,
        l_max=1,
        adstock_burn_in=0,
        draws_per_cell=2,
    )
    with pytest.raises(ValueError, match="n must be >= 2"):
        DataGenerator(cfg).generate(n_tasks=1, validate=True)
    with pytest.raises(ValueError, match="n must be >= 2"):
        DataGenerator(cfg).iter_batches(n_tasks=1)


@pytest.mark.parametrize(
    ("n_tasks", "batch_size", "expected_sizes"),
    (
        (2, 2, (2,)),
        (3, 2, (3,)),
        (5, 2, (2, 3)),
        (7, 3, (3, 4)),
        (8, 3, (3, 3, 2)),
    ),
)
def test_datagenerator_batches_preserve_tasks_without_one_world_batch(
    n_tasks, batch_size, expected_sizes
):
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=8,
        l_max=1,
        adstock_burn_in=0,
        draws_per_cell=2,
        nonlinearity="linear",
        spend_cv_floor=0.0,
        seed=91,
    )

    batches = DataGenerator(cfg).iter_batches(n_tasks=n_tasks, batch_size=batch_size)
    sizes = tuple(batch["spend_raw"].shape[0] for batch in batches)

    assert sizes == expected_sizes
    assert sum(sizes) == n_tasks
    assert all(size != 1 for size in sizes)


def test_datagenerator_rejects_one_world_batch_size():
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=8,
        l_max=1,
        adstock_burn_in=0,
        draws_per_cell=2,
    )

    with pytest.raises(ValueError, match="batch_size must be at least 2"):
        DataGenerator(cfg).iter_batches(n_tasks=2, batch_size=1)


def test_batch_iterator_releases_yielded_arrays_and_uses_config_seed():
    import weakref

    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=8,
        l_max=1,
        adstock_burn_in=0,
        nonlinearity="linear",
        seed=91,
    )
    generator = DataGenerator(cfg)
    batches = generator.iter_batches(n_tasks=4, batch_size=2)
    first = next(batches)
    expected = generator.generate(n_tasks=2, seed=cfg.seed)
    np.testing.assert_array_equal(first["sales_raw"], expected["sales_raw"])
    array_ref = weakref.ref(first["spend_raw"])
    del first
    assert array_ref() is None
    second = next(batches)
    expected = generator.generate(n_tasks=2, seed=cfg.seed + 1)
    np.testing.assert_array_equal(second["sales_raw"], expected["sales_raw"])
    with pytest.raises(StopIteration):
        next(batches)


def test_finalization_uses_retained_tasks_for_truncated_public_paths(tmp_path):
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=16,
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

        layout = SlotLayout(
            n_treatments=2, n_covariates=2, n_latent=1, edge_types=EDGE_TYPES_EXTENDED
        )
        direct = (corpus["g"][:, layout.slices["cy"]] == 1) & (corpus["treatment_active_mask"] == 1)
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
            adstock_family=corpus["adstock_family"],
            adstock_alpha=corpus["adstock_alpha"],
            weibull_lam=corpus["weibull_lam"],
            weibull_k=corpus["weibull_k"],
        )
        expected_signal["metric_version"] = SIGNAL_METRIC_VERSION
        expected_signal["metric_layout"] = list(SIGNAL_METRIC_LAYOUT)
        expected_signal["l_max"] = cfg.l_max
        expected_signal["adstock_burn_in"] = cfg.adstock_burn_in
        expected_signal["adstock_kernel_semantics"] = "normalized-causal-minmax-weibull-density"
        expected_signal["adstock_kernel_version"] = 3
        expected_signal["outcome_noise_semantics"] = OUTCOME_NOISE_SEMANTICS
        expected_signal["outcome_noise_version"] = OUTCOME_NOISE_VERSION
        expected_signal["outcome_std_mode"] = cfg.outcome_std_mode
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
        n_time_steps=16,
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


@pytest.mark.parametrize(
    ("key", "axis", "expected"),
    (
        ("spend_raw", "channel", "inactive-channel padding"),
        ("spend_norm", "channel", "inactive-channel padding"),
        ("spend_share", "channel", "inactive-channel padding"),
        ("contributions_raw", "channel", "inactive-channel padding"),
        ("channel_shock_mask", "channel", "inactive-channel padding"),
        ("spend_means", "channel", "inactive-channel padding"),
        ("channel_active", "channel", "inactive-channel padding"),
        ("channel_level", "channel", "inactive-channel padding"),
        ("saturation_scale", "channel", "inactive-channel padding"),
        ("adstock_family", "channel", "inactive-channel padding"),
        ("adstock_alpha", "channel", "inactive-channel padding"),
        ("weibull_lam", "channel", "inactive-channel padding"),
        ("weibull_k", "channel", "inactive-channel padding"),
        ("controls", "control", "inactive-control padding"),
        ("control_contribution", "control", "inactive-control padding"),
        ("demand", "demand", "inactive-demand padding"),
        ("confounder_contribution", "demand", "inactive-demand padding"),
    ),
)
def test_validate_corpus_rejects_nonzero_node_padding(padded_corpus, key, axis, expected):
    broken = dict(padded_corpus)
    bad = padded_corpus[key].copy()
    active_key = {
        "channel": "n_treatments_active",
        "control": "n_covariates_active",
        "demand": "n_latent_active",
    }[axis]
    index = int(padded_corpus[active_key][0])
    if bad.ndim == 3:
        bad[0, 0, index] = 1
    else:
        bad[0, index] = 1
    broken[key] = bad
    assert any(expected in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        ("n_treatments_active", "n_treatments_active"),
        ("n_covariates_active", "n_covariates_active"),
        ("n_latent_active", "n_latent_active"),
        ("treatment_active_mask", "n_treatments_active"),
        ("covariate_active_mask", "n_covariates_active"),
        ("latent_active_mask", "n_latent_active"),
    ),
)
def test_validate_corpus_rejects_corrupt_active_counts_and_masks(padded_corpus, key, expected):
    broken = dict(padded_corpus)
    broken[key] = padded_corpus[key].copy()
    if key.endswith("_active"):
        broken[key][0] += 1
    else:
        broken[key][0, int(broken[key][0].sum())] = 1
    assert any(expected in error for error in DataGenerator.validate_corpus(broken))


def test_validate_corpus_rejects_zero_active_count(padded_corpus):
    broken = dict(padded_corpus)
    broken["n_treatments_active"] = padded_corpus["n_treatments_active"].copy()
    broken["n_treatments_active"][0] = 0
    assert (
        "n_treatments_active does not match its active prefix mask"
        in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize("edge_type", EDGE_TYPES_EXTENDED)
def test_validate_corpus_rejects_graph_edges_incident_to_padding(padded_corpus, edge_type):
    broken = dict(padded_corpus)
    layout = SlotLayout(n_treatments=4, n_covariates=3, n_latent=2, edge_types=EDGE_TYPES_EXTENDED)
    # unpack() reshapes the non-square blocks in place, so it hands back VIEWS
    # into the g-vector: writing an edge through them would corrupt this
    # module-scoped fixture for every later test. Copy first.
    parts = layout.unpack(padded_corpus["g"][:1].copy())
    inactive = {"c": 2, "m": 1, "j": 1}
    indices = {
        "cy": (inactive["c"],),
        "dc": (0, inactive["c"]),
        "dz": (0, inactive["m"]),
        "dy": (inactive["j"],),
        "zy": (inactive["m"],),
        "zc": (inactive["m"], 0),
        "cc": (0, inactive["c"]),
        "zz": (0, inactive["m"]),
    }
    parts[edge_type][(0, *indices[edge_type])] = 1
    bad = padded_corpus["g"].copy()
    bad[0] = layout.pack(**{f"g_{name}": value for name, value in parts.items()})[0]
    broken["g"] = bad
    assert any(f"g_{edge_type}" in error for error in DataGenerator.validate_corpus(broken))
    # The fixture itself must survive every parameterized case untouched.
    assert DataGenerator.validate_corpus(padded_corpus) == []


@pytest.mark.parametrize(("edge_type", "index"), (("cc", (1, 0)), ("zz", (1, 0))))
def test_validate_corpus_rejects_non_dag_square_graph_blocks(corpus, edge_type, index):
    broken = dict(corpus)
    layout = SlotLayout(n_treatments=4, n_covariates=2, n_latent=1, edge_types=EDGE_TYPES_EXTENDED)
    parts = layout.unpack(corpus["g"])
    parts[edge_type][(0, *index)] = 1
    broken["g"] = layout.pack(**{f"g_{name}": value for name, value in parts.items()})
    assert f"g_{edge_type} must be strictly upper triangular" in DataGenerator.validate_corpus(
        broken
    )


def test_validate_corpus_rejects_offsetting_impossible_indirect_sources():
    generated = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=3,
            n_covariates=2,
            n_latent=1,
            n_time_steps=10,
            n_cells=2,
            draws_per_cell=1,
            l_max=2,
            adstock_burn_in=2,
            cc_base_rate=0.0,
            zc_base_rate=0.0,
            edge_rate_overrides={"dc": 0.0},
            seed=823,
        )
    )
    broken = dict(generated)
    broken["indirect_effects_by_source"] = generated["indirect_effects_by_source"].copy()
    broken["indirect_effects_by_source"][0, 0, 0] = 1.0
    broken["indirect_effects_by_source"][0, 0, 1] = -1.0
    errors = DataGenerator.validate_corpus(broken)
    assert "indirect_effects_by_source cc column is nonzero without an edge" in errors
    assert "indirect_effects_by_source zc column is nonzero without an edge" in errors


def test_single_treatment_corpus_has_exactly_zero_cc_indirect_source():
    generated = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_time_steps=10,
            n_cells=2,
            draws_per_cell=1,
            l_max=2,
            adstock_burn_in=2,
            seed=7,
        )
    )
    assert np.array_equal(
        generated["indirect_effects_by_source"][..., 0],
        np.zeros((2, 10), dtype=np.float32),
    )
    assert DataGenerator.validate_corpus(generated) == []


def test_validate_corpus_rejects_corrupt_cell_metadata(corpus):
    forged_id = dict(corpus)
    forged_id["cell_id"] = corpus["cell_id"].copy()
    forged_id["cell_id"][1] = 99
    errors = DataGenerator.validate_corpus(forged_id)
    assert "cell_id must contain contiguous nonnegative ids" in errors
    assert "diagnostics n_cells does not match cell_id" in errors

    inconsistent_graph = dict(corpus)
    inconsistent_graph["g"] = corpus["g"].copy()
    inconsistent_graph["g"][1, 0] ^= 1
    assert "g differs within a cell" in DataGenerator.validate_corpus(inconsistent_graph)


def test_validate_corpus_handles_array_valued_signal_diagnostic(corpus):
    broken = dict(corpus)
    broken["diagnostics"] = dict(corpus["diagnostics"])
    broken["diagnostics"]["signal"] = dict(corpus["diagnostics"]["signal"])
    value = broken["diagnostics"]["signal"]["n_direct_channels"]
    broken["diagnostics"]["signal"]["n_direct_channels"] = np.array([value, value])
    assert "diagnostics signal summary does not match recomputation" in (
        DataGenerator.validate_corpus(broken)
    )


def test_validate_corpus_rejects_incorrect_channel_active(corpus):
    broken = dict(corpus)
    broken["channel_active"] = corpus["channel_active"].copy()
    broken["channel_active"][0, 0] ^= 1
    assert any(
        "channel_active does not match" in error for error in DataGenerator.validate_corpus(broken)
    )


def test_validate_corpus_rejects_balanced_null_channel_contribution(corpus):
    layout = SlotLayout(n_treatments=4, n_covariates=2, n_latent=1, edge_types=EDGE_TYPES_EXTENDED)
    direct = corpus["g"][:, layout.slices["cy"]]
    n, k = np.argwhere(direct == 0)[0]
    broken = dict(corpus)
    for key in ("contributions_raw", "indirect_effects", "indirect_effects_by_source"):
        broken[key] = corpus[key].copy()
    broken["contributions_raw"][n, 0, k] += 0.25
    broken["indirect_effects"][n, 0] -= 0.25
    broken["indirect_effects_by_source"][n, 0, 0] -= 0.25
    assert any("channels without C->Y" in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize(
    ("edge_type", "contribution_key", "message"),
    (
        ("zy", "control_contribution", "without a Z->Y edge"),
        ("dy", "confounder_contribution", "without a D->Y edge"),
    ),
)
def test_validate_corpus_rejects_balanced_parentless_baseline_contribution(
    parentless_baseline_corpus, edge_type, contribution_key, message
):
    corpus = parentless_baseline_corpus
    layout = SlotLayout(n_treatments=4, n_covariates=2, n_latent=1, edge_types=EDGE_TYPES_EXTENDED)
    edges = layout.unpack(corpus["g"])[edge_type]
    n, node = np.argwhere(edges == 0)[0]
    broken = dict(corpus)
    broken[contribution_key] = corpus[contribution_key].copy()
    broken["baseline_intrinsic"] = corpus["baseline_intrinsic"].copy()
    broken[contribution_key][n, 0, node] += 0.25
    broken["baseline_intrinsic"][n, 0] -= 0.25
    assert any(message in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize(
    ("key", "index", "expected"),
    (
        ("indirect_effects", (0, 0), "additive decomposition"),
        ("indirect_effects_by_source", (0, 0, 0), "telescoping decomposition"),
        ("baseline_intrinsic", (0, 0), "baseline decomposition"),
        ("control_contribution", (0, 0, 0), "full decomposition"),
    ),
)
def test_validate_corpus_rejects_corrupt_decompositions(corpus, key, index, expected):
    broken = dict(corpus)
    broken[key] = corpus[key].copy()
    broken[key][index] += 1.0
    assert any(expected in error for error in DataGenerator.validate_corpus(broken))


def test_shock_metadata_is_reconstructable_and_zero_padded():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
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
    inactive = generated["treatment_active_mask"] == 0
    for key in ("channel_level", "adstock_family", "adstock_alpha", "weibull_lam", "weibull_k"):
        assert not generated[key][inactive].any(), key

    broken = dict(generated)
    bad_start = generated["channel_shock_start"].copy()
    bad_start[0, 0] = cfg.n_time_steps
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
