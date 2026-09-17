"""Outcome-space distributions: masking, exactness of the share budget, adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator.outcomes import (
    ADDITIVE_QUANTITIES,
    OUTCOME_QUANTITIES,
    outcome_distributions,
)


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


@pytest.fixture(scope="module")
def dist(corpus):
    return outcome_distributions(corpus)


# -- structure ---------------------------------------------------------------


def test_every_quantity_present(dist):
    assert dist.names == OUTCOME_QUANTITIES
    assert len(dist) == len(OUTCOME_QUANTITIES)


def test_scalar_quantity_is_one_unit_per_world(corpus, dist):
    n_tasks = corpus["outcome_raw"].shape[0]
    outcome = dist["outcome"]
    assert outcome.n_units == n_tasks
    assert (outcome.column_index == -1).all()
    assert np.array_equal(outcome.world_index, np.arange(n_tasks))


def test_column_units_count_active_columns_only(corpus, dist):
    """Padded (inactive) treatments/covariates/latents must not become units."""
    for name, mask_key in (
        ("treatment_contribution", "treatment_active_mask"),
        ("treatment", "treatment_active_mask"),
        ("covariate_contribution", "covariate_active_mask"),
        ("latent_unobserved_contribution", "latent_active_mask"),
    ):
        expected = int(corpus[mask_key].sum())
        assert dist[name].n_units == expected, name


def test_series_holds_exactly_the_active_values(corpus, dist):
    """The pooled treatment series is the unpadded corpus contribution block."""
    contrib = corpus["treatment_contribution_raw"].astype(np.float64)
    mask = corpus["treatment_active_mask"].astype(bool)
    expected = contrib.transpose(0, 2, 1)[mask]
    got = dist["treatment_contribution"].series
    assert got is not None
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)
    assert dist["treatment_contribution"].values.size == expected.size


def test_unit_stats_match_direct_numpy(corpus, dist):
    contrib = corpus["treatment_contribution_raw"].astype(np.float64)
    mask = corpus["treatment_active_mask"].astype(bool)
    d = dist["treatment_contribution"]
    for unit in (0, d.n_units // 2, d.n_units - 1):
        w, k = int(d.world_index[unit]), int(d.column_index[unit])
        assert mask[w, k]
        series = contrib[w, :, k]
        assert d.unit_mean[unit] == pytest.approx(series.mean(), rel=1e-6)
        assert d.unit_std[unit] == pytest.approx(series.std(), rel=1e-6)
        assert d.unit_min[unit] == pytest.approx(series.min(), rel=1e-6, abs=1e-9)
        assert d.unit_max[unit] == pytest.approx(series.max(), rel=1e-6, abs=1e-9)


def test_pooled_outcome_distribution_is_the_corpus_outcome(corpus, dist):
    expected = np.sort(corpus["outcome_raw"].astype(np.float64).ravel())
    got = np.sort(dist["outcome"].values.astype(np.float64))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_labels_follow_the_node_vocabulary(dist):
    assert set(dist["treatment_contribution"].labels()) <= {"C1", "C2", "C3", "C4"}
    assert set(dist["covariate_contribution"].labels()) <= {"Z1", "Z2"}
    assert set(dist["latent_unobserved_contribution"].labels()) == {"D1"}
    assert set(dist["indirect_by_source"].labels()) == {"cc", "zc", "dc"}
    assert set(dist["outcome"].labels()) == {""}


# -- share budget ------------------------------------------------------------


def test_additive_shares_sum_to_one_per_world(dist):
    """The generator's decomposition is exact, so the budget closes."""
    total = dist.additive_share_total()
    assert total.shape == (dist.n_worlds,)
    assert np.allclose(total, 1.0, rtol=0, atol=1e-4)


def test_treatment_plus_baseline_share_is_one(dist):
    treatment = np.bincount(
        dist["treatment_total_contribution"].world_index,
        weights=dist["treatment_total_contribution"].unit_share,
        minlength=dist.n_worlds,
    )
    baseline = dist["baseline"].unit_share
    assert np.allclose(treatment + baseline, 1.0, rtol=0, atol=1e-4)


def test_share_equals_total_over_outcome_total(corpus, dist):
    outcome_total = corpus["outcome_raw"].astype(np.float64).sum(axis=1)
    d = dist["treatment_contribution"]
    expected = d.unit_total / outcome_total[d.world_index]
    assert np.allclose(d.unit_share, expected, rtol=1e-6, atol=1e-9)


def test_zero_outcome_world_reports_an_undefined_share_not_zero():
    """A world with no outcome has no budget to split, so every share is NaN.

    Reporting 0.0 would read as a decomposition that lost all of Y, which is
    exactly the alarm this diagnostic exists to raise; the ratio is genuinely
    undefined and NaN is the only honest answer. The config below zeroes every
    driver of Y — no baseline walk, no observation noise, a single treatment
    whose only shock has level 0 — so outcome are identically 0.
    """
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=4,
        l_max=1,
        carryover_burn_in=0,
        nonlinearity="linear",
        edge_budget=dict.fromkeys(("dc", "dz", "zc", "cc", "zz", "dy", "zy"), (0, 0))
        | {"cy": (1, 1)},
        rw_baseline_mean_range=(0.0, 0.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_outcome_std_range=(0.0, 0.0),
        n_treatment_shocks=1,
        treatment_shock_length_range=(4, 4),
        treatment_shock_level_range=(0.0, 0.0),
        seed=13,
    )
    world = pg.sample_scm(cfg, seed=13)
    assert not np.any(world.data["outcome"])

    zero = outcome_distributions([world])
    assert np.isnan(zero["outcome"].unit_share).all()
    assert np.isnan(zero["treatment_contribution"].unit_share).all()
    assert np.isnan(zero.additive_share_total()).all()


def test_exogenous_quantities_have_no_share(dist):
    for name in ("treatment", "covariates", "latent_unobserved"):
        assert not dist[name].on_y_scale
        assert np.isnan(dist[name].unit_share).all(), name


def test_additive_set_is_the_documented_one():
    additive = {name for name in OUTCOME_QUANTITIES if name in ADDITIVE_QUANTITIES}
    assert additive == {
        "baseline_intrinsic",
        "outcome_noise",
        "covariate_contribution",
        "latent_unobserved_contribution",
        "treatment_contribution",
        "indirect_by_source",
    }


def test_incomplete_subset_refuses_to_report_a_budget(corpus):
    partial = outcome_distributions(corpus, quantities=["outcome", "treatment_contribution"])
    with pytest.raises(ValueError, match="incomplete"):
        partial.additive_share_total()


# -- normalization -----------------------------------------------------------


def test_normalize_outcome_scale_divides_y_scale_quantities(corpus, dist):
    scaled = outcome_distributions(corpus, normalize="outcome_scale")
    scale = corpus["outcome_scale"].astype(np.float64)
    assert np.allclose(
        scaled["outcome"].unit_mean,
        dist["outcome"].unit_mean / scale,
        rtol=1e-6,
    )
    # exogenous inputs are not in outcome units and stay untouched
    assert np.allclose(scaled["treatment"].unit_mean, dist["treatment"].unit_mean, rtol=1e-6)


def test_normalization_leaves_shares_invariant(corpus, dist):
    for mode in ("outcome_scale", "outcome_mean"):
        scaled = outcome_distributions(corpus, normalize=mode)
        assert np.allclose(
            scaled["treatment_contribution"].unit_share,
            dist["treatment_contribution"].unit_share,
            rtol=1e-6,
            atol=1e-9,
        ), mode


# -- selection ---------------------------------------------------------------


def test_world_subset_selects_rows_and_records_ids(corpus):
    keep = corpus["cell_id"] == 0
    subset = outcome_distributions(corpus, worlds=keep)
    assert subset.n_worlds == int(keep.sum())
    assert np.array_equal(subset.world_ids, np.flatnonzero(keep))
    expected = np.sort(corpus["outcome_raw"][keep].astype(np.float64).ravel())
    assert np.allclose(np.sort(subset["outcome"].values.astype(np.float64)), expected, rtol=1e-5)
    assert np.allclose(subset.additive_share_total(), 1.0, atol=1e-4)


def test_select_drops_structurally_null_treatments(dist):
    treatment = dist["treatment_contribution"]
    direct = treatment.select(treatment.unit_max > 0.0)
    assert direct.n_units <= treatment.n_units
    assert direct.zero_unit_fraction == 0.0
    assert direct.pooled["max"] == pytest.approx(treatment.pooled["max"], rel=1e-6)
    assert direct.n_units == treatment.n_units - round(
        treatment.zero_unit_fraction * treatment.n_units
    )


def test_select_validates_shape(dist):
    with pytest.raises(ValueError, match="boolean mask"):
        dist["outcome"].select(np.ones(dist["outcome"].n_units + 1, dtype=bool))


def test_uint8_flag_is_not_silently_read_as_world_positions(corpus):
    """The corpus stores its flags as uint8, so a raw flag is ambiguous.

    ``worlds=corpus["is_val"]`` reads as the positions "world 1, world 1,
    world 0, ..." — the right length, made of real rows, and wrong. Nothing
    downstream can catch that, so the caller has to say which reading it meant.
    """
    is_val = corpus["is_val"]
    assert is_val.dtype == np.uint8
    with pytest.raises(ValueError, match="ambiguous world selector") as excinfo:
        outcome_distributions(corpus, worlds=is_val)
    message = str(excinfo.value)
    assert "astype(bool)" in message
    assert "np.flatnonzero" in message

    expected = np.flatnonzero(is_val)
    assert 0 < expected.size < is_val.size
    for selector in (is_val == 1, is_val.astype(bool)):
        subset = outcome_distributions(corpus, worlds=selector)
        assert np.array_equal(subset.world_ids, expected)
    # the rule is narrow: integers that cannot be a mask stay positional
    positions = outcome_distributions(corpus, worlds=np.array([0, 2, 4]))
    assert np.array_equal(positions.world_ids, [0, 2, 4])


def test_uint8_unit_mask_is_not_silently_read_as_unit_positions(dist):
    treatment = dist["treatment_contribution"]
    flag = (treatment.unit_max > 0.0).astype(np.uint8)
    with pytest.raises(ValueError, match="ambiguous unit selector"):
        treatment.select(flag)

    kept = treatment.select(flag.astype(bool))
    assert kept.n_units == int(flag.sum())
    assert kept.n_units < treatment.n_units
    assert np.array_equal(treatment.select(flag == 1).world_index, kept.world_index)
    assert treatment.select(np.array([0, 2])).n_units == 2


@pytest.mark.parametrize(
    "selector", [[1.9], ["2"], [[0, 2]], np.array([2**64 - 1], dtype=np.uint64)]
)
def test_selectors_reject_lossy_or_malformed_positions(corpus, dist, selector):
    with pytest.raises((TypeError, ValueError, IndexError)):
        outcome_distributions(corpus, worlds=selector)
    with pytest.raises((TypeError, ValueError, IndexError)):
        dist["treatment_contribution"].select(selector)


def test_scalar_unit_selection_preserves_unit_axis(dist):
    treatment = dist["treatment_contribution"]
    one = treatment.select(2)
    assert one.series.shape == (1, treatment.series.shape[1])
    np.testing.assert_array_equal(one.series[0], treatment.series[2])
    assert one.world_index.shape == (1,)


def test_select_recomputes_pooled_and_refuses_without_raw_series(corpus, dist):
    """A subset's pooled report describes the subset, or it does not exist.

    Without the raw values there is nothing to re-pool, and handing back the
    full population's statistics under a subset's ``n_units`` is silently
    wrong — so that combination raises instead.
    """
    treatment = dist["treatment_contribution"]
    keep = treatment.unit_max > 0.0
    assert treatment.series is not None
    subset_values = treatment.series[keep].reshape(-1).astype(np.float64)

    subset = treatment.select(keep)
    assert subset.pooled["n"] == subset_values.size
    assert subset.pooled["n"] < treatment.pooled["n"]
    assert subset.pooled["mean"] == pytest.approx(subset_values.mean(), rel=1e-6)
    assert subset.pooled["mean"] != pytest.approx(treatment.pooled["mean"], rel=1e-6)

    lean = outcome_distributions(corpus, keep_series=False)["treatment_contribution"]
    with pytest.raises(ValueError, match="keep_series=True"):
        lean.select(keep)


def test_empty_world_selection_raises(corpus):
    with pytest.raises(ValueError, match="empty"):
        outcome_distributions(corpus, worlds=np.zeros(corpus["outcome_raw"].shape[0], bool))


# -- reports -----------------------------------------------------------------


def test_summary_is_json_serializable(dist):
    payload = json.dumps(dist.summary())
    assert "treatment_contribution" in payload
    summary = dist.summary()
    assert summary["n_worlds"] == dist.n_worlds
    assert summary["quantities"]["outcome"]["share"] is not None
    assert summary["quantities"]["treatment"]["share"] is None


def test_table_lists_quantities_and_share_table_drops_exogenous(dist):
    values = dist.table()
    assert "treatment_contribution" in values
    assert "treatment" in values
    shares = dist.table(of="share")
    assert "treatment_contribution" in shares
    assert "\nspend" not in shares


def test_to_frame_has_one_row_per_unit(dist):
    frame = dist.to_frame()
    assert len(frame) == sum(d.n_units for d in dist)
    treatments = frame[frame["quantity"] == "treatment_contribution"]
    assert set(treatments["label"]) <= {"C1", "C2", "C3", "C4"}
    assert frame["world"].max() < dist.n_worlds


def test_quantiles_of_each_statistic(dist):
    d = dist["outcome"]
    pooled = d.quantiles()
    assert pooled["min"] <= pooled["q50"] <= pooled["max"]
    means = d.quantiles(of="mean")
    assert means["n"] == d.n_units
    with pytest.raises(ValueError, match="unknown stat"):
        d.quantiles(of="nope")


def test_keep_series_false_keeps_stats_but_drops_values(corpus):
    lean = outcome_distributions(corpus, keep_series=False)
    assert lean["outcome"].series is None
    assert lean["outcome"].pooled["q50"] == pytest.approx(
        outcome_distributions(corpus)["outcome"].pooled["q50"], rel=1e-6
    )
    with pytest.raises(ValueError, match="keep_series"):
        _ = lean["outcome"].values
    assert np.allclose(lean.additive_share_total(), 1.0, atol=1e-4)


def test_keep_series_false_still_reports_at_the_stored_levels(corpus, dist):
    """Dropping the raw values must not change the default reports.

    ``table`` substitutes the stored levels and passes them explicitly, so it
    has to recognize them as the levels the pooled report already holds —
    otherwise the default table asks for values that were deliberately freed.
    """
    lean = outcome_distributions(corpus, keep_series=False)
    assert lean.table() == dist.table()
    assert lean.table(levels=lean.quantile_levels) == lean.table()
    assert json.dumps(lean.summary(list(lean.quantile_levels))) == json.dumps(lean.summary())
    # other levels genuinely need the values that were dropped
    with pytest.raises(ValueError, match="keep_series"):
        lean.table(levels=(0.1, 0.9))
    with pytest.raises(ValueError, match="keep_series"):
        lean["outcome"].quantiles(levels=(0.1, 0.9))


# -- validation + SCM adapter ------------------------------------------------


def test_unknown_quantity_and_bad_normalize_raise(corpus):
    with pytest.raises(ValueError, match="unknown quantities"):
        outcome_distributions(corpus, quantities=["nope"])
    with pytest.raises(ValueError, match="normalize"):
        outcome_distributions(corpus, normalize="zscore")
    with pytest.raises(ValueError, match="quantile levels"):
        outcome_distributions(corpus, quantiles=[1.5])


@pytest.mark.parametrize("levels", [[], [np.nan], [np.inf], [[0.5]], 0.5, [-0.1], [0.5, 0.5]])
def test_invalid_quantiles_fail_consistently(corpus, dist, levels):
    with pytest.raises(ValueError):
        outcome_distributions(corpus, quantiles=levels)
    for report in (dist.summary, dist.table, dist["outcome"].summary, dist["outcome"].quantiles):
        with pytest.raises(ValueError):
            report(levels=levels)


def test_numpy_quantiles_can_use_cached_reports_without_series(corpus):
    lean = outcome_distributions(corpus, quantiles=np.array([0.0, 0.5, 1.0]), keep_series=False)
    levels = np.asarray(lean.quantile_levels)
    assert json.dumps(lean.summary(levels)) == json.dumps(lean.summary())
    assert lean.table(levels=levels) == lean.table()


def test_empty_quantity_subset_is_rejected(corpus):
    """An empty selection reports nothing and cannot even build a frame."""
    with pytest.raises(ValueError, match="quantities is empty"):
        outcome_distributions(corpus, quantities=[])


@pytest.mark.parametrize(
    "name, keys",
    [
        ("outcome", ("outcome_raw",)),
        (
            "treatment_total_contribution",
            ("outcome_raw", "treatment_contribution_raw", "indirect_effects"),
        ),
        ("treatment", ("outcome_raw", "treatment_raw", "treatment_active_mask")),
    ],
)
def test_quantity_subset_requires_only_its_dependencies(corpus, name, keys):
    full = outcome_distributions(corpus, normalize="none")
    minimal = {key: corpus[key] for key in keys}
    subset = outcome_distributions(minimal, quantities=(name,), normalize="none")
    np.testing.assert_array_equal(subset[name].values, full[name].values)
    np.testing.assert_array_equal(subset[name].unit_share, full[name].unit_share)


def test_missing_corpus_key_names_it():
    with pytest.raises(KeyError, match="missing keys"):
        outcome_distributions({"outcome_raw": np.zeros((2, 3))})


def test_plot_outcome_distributions_writes_a_figure(dist, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from pymc_generator.viz import plot_outcome_distributions

    for of in ("value", "mean", "share"):
        path = tmp_path / f"outcomes_{of}.png"
        plot_outcome_distributions(dist, str(path), of=of)
        assert path.stat().st_size > 0


def test_scm_sequence_path_matches_its_worlds():
    cfg = pg.make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=32,
        seed=5,
        edge_budget={"cy": (2, 3), "cc": (1, 1), "zc": (1, 1), "dc": (1, 1)},
    )
    worlds = [pg.sample_scm(cfg, seed=s) for s in (1, 2)]
    dist = outcome_distributions(worlds)

    assert dist.n_worlds == 2
    assert dist.n_time_steps == 32
    assert dist["treatment_contribution"].n_units == 2 * 3
    assert np.allclose(dist.additive_share_total(), 1.0, atol=1e-4)
    expected = np.sort(np.concatenate([w.data["outcome"] for w in worlds]))
    assert np.allclose(np.sort(dist["outcome"].values.astype(np.float64)), expected, rtol=1e-5)
    # the exact per-world identity survives the pooling
    for i, world in enumerate(worlds):
        assert dist["outcome"].unit_mean[i] == pytest.approx(world.data["outcome"].mean(), rel=1e-6)


def test_non_sequence_source_rejected():
    with pytest.raises(TypeError, match="corpus mapping or a sequence"):
        outcome_distributions(42)  # type: ignore[arg-type]
