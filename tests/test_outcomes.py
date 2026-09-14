"""Outcome-space distributions: masking, exactness of the share budget, adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.outcomes import (
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
    n_tasks = corpus["sales_raw"].shape[0]
    sales = dist["sales"]
    assert sales.n_units == n_tasks
    assert (sales.column_index == -1).all()
    assert np.array_equal(sales.world_index, np.arange(n_tasks))


def test_column_units_count_active_columns_only(corpus, dist):
    """Padded (inactive) channels/controls/latents must not become units."""
    for name, mask_key in (
        ("channel_contribution", "treatment_active_mask"),
        ("spend", "treatment_active_mask"),
        ("control_contribution", "covariate_active_mask"),
        ("confounder_contribution", "latent_active_mask"),
    ):
        expected = int(corpus[mask_key].sum())
        assert dist[name].n_units == expected, name


def test_series_holds_exactly_the_active_values(corpus, dist):
    """The pooled channel series is the unpadded corpus contribution block."""
    contrib = corpus["contributions_raw"].astype(np.float64)
    mask = corpus["treatment_active_mask"].astype(bool)
    expected = contrib.transpose(0, 2, 1)[mask]
    got = dist["channel_contribution"].series
    assert got is not None
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)
    assert dist["channel_contribution"].values.size == expected.size


def test_unit_stats_match_direct_numpy(corpus, dist):
    contrib = corpus["contributions_raw"].astype(np.float64)
    mask = corpus["treatment_active_mask"].astype(bool)
    d = dist["channel_contribution"]
    for unit in (0, d.n_units // 2, d.n_units - 1):
        w, k = int(d.world_index[unit]), int(d.column_index[unit])
        assert mask[w, k]
        series = contrib[w, :, k]
        assert d.unit_mean[unit] == pytest.approx(series.mean(), rel=1e-6)
        assert d.unit_std[unit] == pytest.approx(series.std(), rel=1e-6)
        assert d.unit_min[unit] == pytest.approx(series.min(), rel=1e-6, abs=1e-9)
        assert d.unit_max[unit] == pytest.approx(series.max(), rel=1e-6, abs=1e-9)


def test_pooled_sales_distribution_is_the_corpus_sales(corpus, dist):
    expected = np.sort(corpus["sales_raw"].astype(np.float64).ravel())
    got = np.sort(dist["sales"].values.astype(np.float64))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_labels_follow_the_node_vocabulary(dist):
    assert set(dist["channel_contribution"].labels()) <= {"C1", "C2", "C3", "C4"}
    assert set(dist["control_contribution"].labels()) <= {"Z1", "Z2"}
    assert set(dist["confounder_contribution"].labels()) == {"D1"}
    assert set(dist["indirect_by_source"].labels()) == {"cc", "zc", "dc"}
    assert set(dist["sales"].labels()) == {""}


# -- share budget ------------------------------------------------------------


def test_additive_shares_sum_to_one_per_world(dist):
    """The generator's decomposition is exact, so the budget closes."""
    total = dist.additive_share_total()
    assert total.shape == (dist.n_worlds,)
    assert np.allclose(total, 1.0, rtol=0, atol=1e-4)


def test_media_plus_baseline_share_is_one(dist):
    media = np.bincount(
        dist["media_contribution"].world_index,
        weights=dist["media_contribution"].unit_share,
        minlength=dist.n_worlds,
    )
    baseline = dist["baseline"].unit_share
    assert np.allclose(media + baseline, 1.0, rtol=0, atol=1e-4)


def test_share_equals_total_over_sales_total(corpus, dist):
    sales_total = corpus["sales_raw"].astype(np.float64).sum(axis=1)
    d = dist["channel_contribution"]
    expected = d.unit_total / sales_total[d.world_index]
    assert np.allclose(d.unit_share, expected, rtol=1e-6, atol=1e-9)


def test_zero_sales_world_reports_an_undefined_share_not_zero():
    """A world with no sales has no budget to split, so every share is NaN.

    Reporting 0.0 would read as a decomposition that lost all of Y, which is
    exactly the alarm this diagnostic exists to raise; the ratio is genuinely
    undefined and NaN is the only honest answer. The config below zeroes every
    driver of Y — no baseline walk, no observation noise, a single channel
    whose only shock has level 0 — so sales are identically 0.
    """
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=4,
        l_max=1,
        adstock_burn_in=0,
        nonlinearity="linear",
        edge_budget=dict.fromkeys(("dc", "dz", "zc", "cc", "zz", "dy", "zy"), (0, 0))
        | {"cy": (1, 1)},
        rw_baseline_mean_range=(0.0, 0.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_sales_std_range=(0.0, 0.0),
        n_channel_shocks=1,
        channel_shock_length_range=(4, 4),
        channel_shock_level_range=(0.0, 0.0),
        seed=13,
    )
    world = pg.sample_scm(cfg, seed=13)
    assert not np.any(world.data["sales"])

    zero = outcome_distributions([world])
    assert np.isnan(zero["sales"].unit_share).all()
    assert np.isnan(zero["channel_contribution"].unit_share).all()
    assert np.isnan(zero.additive_share_total()).all()


def test_exogenous_quantities_have_no_share(dist):
    for name in ("spend", "controls", "demand"):
        assert not dist[name].on_y_scale
        assert np.isnan(dist[name].unit_share).all(), name


def test_additive_set_is_the_documented_one():
    additive = {name for name in OUTCOME_QUANTITIES if name in ADDITIVE_QUANTITIES}
    assert additive == {
        "baseline_intrinsic",
        "sales_noise",
        "control_contribution",
        "confounder_contribution",
        "channel_contribution",
        "indirect_by_source",
    }


def test_incomplete_subset_refuses_to_report_a_budget(corpus):
    partial = outcome_distributions(corpus, quantities=["sales", "channel_contribution"])
    with pytest.raises(ValueError, match="incomplete"):
        partial.additive_share_total()


# -- normalization -----------------------------------------------------------


def test_normalize_sales_scale_divides_y_scale_quantities(corpus, dist):
    scaled = outcome_distributions(corpus, normalize="sales_scale")
    scale = corpus["sales_scale"].astype(np.float64)
    assert np.allclose(
        scaled["sales"].unit_mean,
        dist["sales"].unit_mean / scale,
        rtol=1e-6,
    )
    # exogenous inputs are not in sales units and stay untouched
    assert np.allclose(scaled["spend"].unit_mean, dist["spend"].unit_mean, rtol=1e-6)


def test_normalization_leaves_shares_invariant(corpus, dist):
    for mode in ("sales_scale", "sales_mean"):
        scaled = outcome_distributions(corpus, normalize=mode)
        assert np.allclose(
            scaled["channel_contribution"].unit_share,
            dist["channel_contribution"].unit_share,
            rtol=1e-6,
            atol=1e-9,
        ), mode


# -- selection ---------------------------------------------------------------


def test_world_subset_selects_rows_and_records_ids(corpus):
    keep = corpus["cell_id"] == 0
    subset = outcome_distributions(corpus, worlds=keep)
    assert subset.n_worlds == int(keep.sum())
    assert np.array_equal(subset.world_ids, np.flatnonzero(keep))
    expected = np.sort(corpus["sales_raw"][keep].astype(np.float64).ravel())
    assert np.allclose(np.sort(subset["sales"].values.astype(np.float64)), expected, rtol=1e-5)
    assert np.allclose(subset.additive_share_total(), 1.0, atol=1e-4)


def test_select_drops_structurally_null_channels(dist):
    media = dist["channel_contribution"]
    direct = media.select(media.unit_max > 0.0)
    assert direct.n_units <= media.n_units
    assert direct.zero_unit_fraction == 0.0
    assert direct.pooled["max"] == pytest.approx(media.pooled["max"], rel=1e-6)
    assert direct.n_units == media.n_units - round(media.zero_unit_fraction * media.n_units)


def test_select_validates_shape(dist):
    with pytest.raises(ValueError, match="boolean mask"):
        dist["sales"].select(np.ones(dist["sales"].n_units + 1, dtype=bool))


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
    media = dist["channel_contribution"]
    flag = (media.unit_max > 0.0).astype(np.uint8)
    with pytest.raises(ValueError, match="ambiguous unit selector"):
        media.select(flag)

    kept = media.select(flag.astype(bool))
    assert kept.n_units == int(flag.sum())
    assert kept.n_units < media.n_units
    assert np.array_equal(media.select(flag == 1).world_index, kept.world_index)
    assert media.select(np.array([0, 2])).n_units == 2


@pytest.mark.parametrize("selector", [[1.9], ["2"], [[0, 2]], np.array([2**64 - 1], dtype=np.uint64)])
def test_selectors_reject_lossy_or_malformed_positions(corpus, dist, selector):
    with pytest.raises((TypeError, ValueError, IndexError)):
        outcome_distributions(corpus, worlds=selector)
    with pytest.raises((TypeError, ValueError, IndexError)):
        dist["channel_contribution"].select(selector)


def test_scalar_unit_selection_preserves_unit_axis(dist):
    media = dist["channel_contribution"]
    one = media.select(2)
    assert one.series.shape == (1, media.series.shape[1])
    np.testing.assert_array_equal(one.series[0], media.series[2])
    assert one.world_index.shape == (1,)


def test_select_recomputes_pooled_and_refuses_without_raw_series(corpus, dist):
    """A subset's pooled report describes the subset, or it does not exist.

    Without the raw values there is nothing to re-pool, and handing back the
    full population's statistics under a subset's ``n_units`` is silently
    wrong — so that combination raises instead.
    """
    media = dist["channel_contribution"]
    keep = media.unit_max > 0.0
    assert media.series is not None
    subset_values = media.series[keep].reshape(-1).astype(np.float64)

    subset = media.select(keep)
    assert subset.pooled["n"] == subset_values.size
    assert subset.pooled["n"] < media.pooled["n"]
    assert subset.pooled["mean"] == pytest.approx(subset_values.mean(), rel=1e-6)
    assert subset.pooled["mean"] != pytest.approx(media.pooled["mean"], rel=1e-6)

    lean = outcome_distributions(corpus, keep_series=False)["channel_contribution"]
    with pytest.raises(ValueError, match="keep_series=True"):
        lean.select(keep)


def test_empty_world_selection_raises(corpus):
    with pytest.raises(ValueError, match="empty"):
        outcome_distributions(corpus, worlds=np.zeros(corpus["sales_raw"].shape[0], bool))


# -- reports -----------------------------------------------------------------


def test_summary_is_json_serializable(dist):
    payload = json.dumps(dist.summary())
    assert "channel_contribution" in payload
    summary = dist.summary()
    assert summary["n_worlds"] == dist.n_worlds
    assert summary["quantities"]["sales"]["share"] is not None
    assert summary["quantities"]["spend"]["share"] is None


def test_table_lists_quantities_and_share_table_drops_exogenous(dist):
    values = dist.table()
    assert "channel_contribution" in values
    assert "spend" in values
    shares = dist.table(of="share")
    assert "channel_contribution" in shares
    assert "\nspend" not in shares


def test_to_frame_has_one_row_per_unit(dist):
    frame = dist.to_frame()
    assert len(frame) == sum(d.n_units for d in dist)
    channels = frame[frame["quantity"] == "channel_contribution"]
    assert set(channels["label"]) <= {"C1", "C2", "C3", "C4"}
    assert frame["world"].max() < dist.n_worlds


def test_quantiles_of_each_statistic(dist):
    d = dist["sales"]
    pooled = d.quantiles()
    assert pooled["min"] <= pooled["q50"] <= pooled["max"]
    means = d.quantiles(of="mean")
    assert means["n"] == d.n_units
    with pytest.raises(ValueError, match="unknown stat"):
        d.quantiles(of="nope")


def test_keep_series_false_keeps_stats_but_drops_values(corpus):
    lean = outcome_distributions(corpus, keep_series=False)
    assert lean["sales"].series is None
    assert lean["sales"].pooled["q50"] == pytest.approx(
        outcome_distributions(corpus)["sales"].pooled["q50"], rel=1e-6
    )
    with pytest.raises(ValueError, match="keep_series"):
        _ = lean["sales"].values
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
        lean["sales"].quantiles(levels=(0.1, 0.9))


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
    for report in (dist.summary, dist.table, dist["sales"].summary, dist["sales"].quantiles):
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
        ("sales", ("sales_raw",)),
        ("media_contribution", ("sales_raw", "contributions_raw", "indirect_effects")),
        ("spend", ("sales_raw", "spend_raw", "treatment_active_mask")),
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
        outcome_distributions({"sales_raw": np.zeros((2, 3))})


def test_plot_outcome_distributions_writes_a_figure(dist, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from prior_generator.viz import plot_outcome_distributions

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
    assert dist["channel_contribution"].n_units == 2 * 3
    assert np.allclose(dist.additive_share_total(), 1.0, atol=1e-4)
    expected = np.sort(np.concatenate([w.data["sales"] for w in worlds]))
    assert np.allclose(np.sort(dist["sales"].values.astype(np.float64)), expected, rtol=1e-5)
    # the exact per-world identity survives the pooling
    for i, world in enumerate(worlds):
        assert dist["sales"].unit_mean[i] == pytest.approx(world.data["sales"].mean(), rel=1e-6)


def test_non_sequence_source_rejected():
    with pytest.raises(TypeError, match="corpus mapping or a sequence"):
        outcome_distributions(42)  # type: ignore[arg-type]
