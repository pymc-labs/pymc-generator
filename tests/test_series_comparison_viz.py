"""Numerical coordinates of independently authored reference-comparison figures."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pymc_generator import viz
from pymc_generator.diagnostics import activity_runs, lagged_dependence, temporal_statistics
from pymc_generator.series_comparison import (
    assess_series_similarity,
    compare_series,
    nearest_series,
    normalize_series_quantile,
)


@pytest.fixture(scope="module", autouse=True)
def agg_backend():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def capture(monkeypatch):
    from matplotlib.figure import Figure

    figures = []

    def save(self, *args, **kwargs):
        figures.append(self)

    monkeypatch.setattr(Figure, "savefig", save)
    return figures


def pearson(left, right):
    """A small, ordinary-range oracle independent of diagnostic implementation."""
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return np.corrcoef(left, right)[0, 1]


def overlap_profile(values, lags):
    return np.array(
        [pearson(values[:-lag], values[lag:]) if lag < len(values) else np.nan for lag in lags]
    )


def assert_band(ax, lags, lower, upper):
    vertices = np.concatenate([path.vertices for path in ax.collections[0].get_paths()])
    for lag, low, high in zip(lags, lower, upper, strict=True):
        ordinates = vertices[vertices[:, 0] == lag, 1]
        np.testing.assert_allclose([ordinates.min(), ordinates.max()], [low, high], atol=1e-14)


def test_temporal_bands_and_scalar_ecdfs_use_available_reference_worlds(capture, tmp_path):
    reference_values = np.array(
        [
            [0, 1, 0, 2, 1, 3, 2, 0],
            [1, 0, 2, 1, 0, 3, 1, 2],
            [2, 1, 0, 1, 3, 2, 1, 0],
            [0, 0, 1, 4, 0, 1, 0, 2],
        ],
        dtype=float,
    )
    query_values = np.array([[0, 2, 1, 0, 3, 1, 2, 4]], dtype=float)
    lags = (1, 2, 20)
    reference = temporal_statistics(
        reference_values[..., None],
        lags=lags,
        eligible=[[True], [True], [True], [False]],
        scale_available=[[True], [True], [False], [True]],
    )
    query = temporal_statistics(query_values[..., None], lags=lags)
    viz.plot_temporal_comparison(
        reference, query, str(tmp_path / "temporal.png"), band=(0.25, 0.75)
    )
    axes = capture[-1].axes
    for position, differences in enumerate((False, True)):
        raw = np.diff(reference_values, axis=1) if differences else reference_values
        query_raw = np.diff(query_values, axis=1) if differences else query_values
        profiles = np.array([overlap_profile(row, lags[:2]) for row in raw[:3]])
        expected = np.quantile(profiles, [0.25, 0.5, 0.75], axis=0)
        ax = axes[position]
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), lags)
        np.testing.assert_allclose(ax.lines[0].get_ydata()[:2], expected[1], atol=1e-14)
        assert np.isnan(ax.lines[0].get_ydata()[2])
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), overlap_profile(query_raw[0], lags), atol=1e-14
        )
        assert_band(ax, lags[:2], expected[0], expected[2])
        vertices = np.concatenate([path.vertices for path in ax.collections[0].get_paths()])
        assert not np.any(vertices[:, 0] == lags[2])
    for ax, values in ((axes[2], reference_values), (axes[3], np.diff(reference_values, axis=1))):
        np.testing.assert_allclose(ax.lines[0].get_xdata(), np.sort(np.std(values[:2], axis=1)))
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.5, 1.0])
    expected_time = np.sort([pearson(row, np.arange(row.size)) for row in reference_values[:3]])
    np.testing.assert_allclose(axes[4].lines[0].get_xdata(), expected_time, atol=1e-14)


def timeline_segments(ax):
    from matplotlib.collections import LineCollection

    return np.concatenate(
        [
            collection.get_segments()
            for collection in ax.collections
            if isinstance(collection, LineCollection)
        ]
    )


def test_activity_distributions_and_literal_half_open_query_intervals(capture, tmp_path):
    reference_values = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0, 3, 0], [1, 0, 0, 2, 0, 0, 0, 3]]
    )
    query_values = np.array([[0, 0, 3, 3, 0, -2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    reference = activity_runs(reference_values[..., None])
    query = activity_runs(query_values[..., None])
    viz.plot_activity_runs(
        reference,
        query,
        str(tmp_path / "runs.png"),
        statistics=("complete_inactive_length_median", "active_spell_count"),
    )
    axes = capture[-1].axes
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [1.5, 2.5])
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [0.5, 1])
    np.testing.assert_allclose(axes[0].lines[1].get_xdata(), [1])
    np.testing.assert_allclose(axes[1].lines[0].get_xdata(), [0, 3, 3])
    np.testing.assert_allclose(axes[1].lines[1].get_xdata(), [0, 2])
    np.testing.assert_allclose(
        timeline_segments(axes[2]),
        [
            [[0, 0], [2, 0]],
            [[2, 1], [4, 1]],
            [[4, 0], [5, 0]],
            [[5, 1], [6, 1]],
            [[6, 0], [8, 0]],
        ],
    )
    np.testing.assert_allclose(timeline_segments(axes[3]), [[[0, 0], [8, 0]]])
    np.testing.assert_allclose([line.get_xdata() for line in axes[2].lines], [[0, 0], [8, 8]])
    from matplotlib.collections import PathCollection

    endpoints = np.concatenate(
        [
            collection.get_offsets()
            for collection in axes[3].collections
            if isinstance(collection, PathCollection)
        ]
    )
    np.testing.assert_allclose(endpoints, [[0, 0], [8, 0]])


def pair_medians(values, eligible, pairs, lags, differences):
    block = np.diff(values, axis=1) if differences else values
    result = np.full((values.shape[0], len(lags)), np.nan)
    for world in range(values.shape[0]):
        for position, lag in enumerate(lags):
            correlations = []
            for left_index, right_index in pairs:
                if not (eligible[world, left_index] and eligible[world, right_index]):
                    continue
                left, right = block[world, :, left_index], block[world, :, right_index]
                if lag > 0:
                    left, right = left[:-lag], right[lag:]
                elif lag < 0:
                    left, right = left[-lag:], right[:lag]
                correlation = pearson(left, right)
                if np.isfinite(correlation):
                    correlations.append(correlation)
            if correlations:
                result[world, position] = np.median(correlations)
    return result


def test_dependence_reduces_unequally_available_pairs_before_world_bands(capture, tmp_path):
    base = np.array([0, 1, 0, 2, 1, 3, 2, 0], dtype=float)
    reference_values = np.stack(
        [np.column_stack([base, base, base, base]), np.column_stack([base, -base, -base, -base])]
    )
    query_values = np.column_stack([base, np.roll(base, 1), -base, base + np.arange(base.size)])[
        None
    ]
    eligible = np.array([[True, True, False, False], [True, True, True, True]])
    pairs, lags = [(0, 1), (0, 2), (0, 3)], (-1, 0, 1)
    reference = lagged_dependence(reference_values, pairs=pairs, lags=lags, eligible=eligible)
    query = lagged_dependence(query_values, pairs=pairs, lags=lags)
    viz.plot_dependence_comparison(
        reference, query, str(tmp_path / "dependence.png"), band=(0.25, 0.75)
    )
    for ax, differences in zip(capture[-1].axes, (False, True), strict=True):
        worlds = pair_medians(reference_values, eligible, pairs, lags, differences)
        expected = np.quantile(worlds, [0.25, 0.5, 0.75], axis=0)
        expected_query = pair_medians(
            query_values, np.ones((1, 4), dtype=bool), pairs, lags, differences
        )
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), lags)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected[1], atol=1e-14)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), expected_query[0], atol=1e-14)
        assert_band(ax, lags, expected[0], expected[2])
        np.testing.assert_allclose(ax.lines[0].get_ydata()[1], 0, atol=1e-14)
        np.testing.assert_allclose(expected[[0, 2], 1], [-0.5, 0.5], atol=1e-14)


def waveform_score(query, reference, metric, lags):
    if metric == "mse":
        return np.mean((query - reference) ** 2)
    if metric == "level_r":
        return pearson(query, reference)
    if metric == "diff_r":
        return pearson(np.diff(query), np.diff(reference))
    return np.sqrt(np.mean((overlap_profile(query, lags) - overlap_profile(reference, lags)) ** 2))


@pytest.mark.parametrize("metric", ("mse", "level_r", "diff_r", "overlap_pearson_rmse"))
def test_matched_waveforms_have_the_oracle_selected_rows_and_alignment(metric, capture, tmp_path):
    base = np.array([0, 1, 0, 2, 1, 3, 2, 0], dtype=float)
    query = np.stack([base, np.roll(base, 1)])
    reference = np.stack([-base, base + 0.25, np.roll(base, 1) + 0.5])
    lags = (1, 2)
    comparison = compare_series(query, reference, lags=lags, metrics=(metric,))
    matches = nearest_series(comparison)
    viz.plot_series_matches(
        comparison, matches, str(tmp_path / "matches.png"), metric=metric, query_indices=(1, 0)
    )
    for ax, index in zip(capture[-1].axes, (1, 0), strict=True):
        scores = [waveform_score(query[index], row, metric, lags) for row in reference]
        minimize = metric in ("mse", "overlap_pearson_rmse")
        selected = int(np.argmin(scores) if minimize else np.argmax(scores))
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(base.size))
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), query[index])
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), np.arange(base.size))
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), reference[selected])


def test_waveform_plot_uses_retained_signed_normalized_values(capture, tmp_path):
    query_values = np.array([[-4, -2, -3, -1, -5, -2]], dtype=float)
    reference_values = np.array([[-8, -4, -6, -2, -10, -4], [1, 2, 3, 4, 5, 6]], dtype=float)
    query = normalize_series_quantile(query_values, quantile=0.5, mode="signed")
    reference = normalize_series_quantile(reference_values, quantile=0.5, mode="signed")
    comparison = compare_series(query, reference, lags=(1,), metrics=("mse",))
    viz.plot_series_matches(
        comparison, nearest_series(comparison), str(tmp_path / "scaled.png"), metric="mse"
    )
    ax = capture[-1].axes[0]
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), query_values[0] / np.quantile(query_values[0], 0.5)
    )
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(), reference_values[0] / np.quantile(reference_values[0], 0.5)
    )


@pytest.mark.parametrize("empty_reference", (False, True))
def test_unavailable_match_never_draws_reference_negative_index(empty_reference, capture, tmp_path):
    query = np.ones((1, 6))
    reference = (
        np.empty((0, 6)) if empty_reference else np.array([[0, 1, 0, 2, 1, 3], [9, 9, 9, 9, 9, 9]])
    )
    comparison = compare_series(query, reference, lags=(1,), metrics=("level_r",))
    matches = nearest_series(comparison)
    np.testing.assert_array_equal(matches.indices["level_r"], [-1])
    viz.plot_series_matches(
        comparison, matches, str(tmp_path / "unavailable.png"), metric="level_r"
    )
    ax = capture[-1].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), query[0])
    for line in ax.lines:
        np.testing.assert_array_equal(line.get_xdata(), np.arange(6))
        np.testing.assert_array_equal(line.get_ydata(), query[0])


def test_waveform_plot_rejects_matches_from_an_equal_but_different_comparison(tmp_path):
    values = np.array([[0, 1, 0, 2, 1, 3]], dtype=float)
    first = compare_series(values, values, lags=(1,), metrics=("mse",))
    second = compare_series(values, values, lags=(1,), metrics=("mse",))
    with pytest.raises(ValueError):
        viz.plot_series_matches(
            second, nearest_series(first), str(tmp_path / "wrong.png"), metric="mse"
        )


def score_assessment():
    base = np.array([0, 1, 0, 2, 1, 3], dtype=float)
    bank = np.stack([base, -base])
    calibration = np.stack([base, base + 1, base + 1, base + 2])
    query = np.stack([base + 0.5, base + 1, base + 3])
    assessment = assess_series_similarity(
        query,
        calibration,
        bank,
        lags=(1,),
        metrics=("mse",),
        query_groups=[30, 31, 32],
        calibration_groups=[20, 21, 22, 23],
        bank_groups=[10, 11],
    )
    return assessment, query, calibration, bank


def test_similarity_ecdf_query_midrank_coordinates_and_tie_extents(capture, tmp_path):
    from matplotlib.collections import LineCollection, PathCollection

    assessment, query, calibration, bank = score_assessment()
    order = (2, 0, 1)
    viz.plot_similarity_scores(assessment, str(tmp_path / "scores.png"), query_indices=order)
    ax = capture[-1].axes[0]
    reference_scores = np.min(np.mean((calibration[:, None] - bank[None]) ** 2, axis=2), axis=1)
    query_scores = np.min(np.mean((query[:, None] - bank[None]) ** 2, axis=2), axis=1)[list(order)]
    lower = np.array([np.mean(reference_scores < score) for score in query_scores])
    upper = np.array([np.mean(reference_scores <= score) for score in query_scores])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.sort(reference_scores))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.arange(1, 5) / 4)
    markers = next(
        collection for collection in ax.collections if isinstance(collection, PathCollection)
    )
    np.testing.assert_allclose(
        markers.get_offsets(), np.column_stack([query_scores, (lower + upper) / 2])
    )
    spans = next(
        collection for collection in ax.collections if isinstance(collection, LineCollection)
    )
    expected = [
        [[score, low], [score, high]]
        for score, low, high in zip(query_scores, lower, upper, strict=True)
    ]
    np.testing.assert_allclose(spans.get_segments(), expected)


def test_similarity_plot_omits_unavailable_query_score_instead_of_zero_marker(capture, tmp_path):
    from matplotlib.collections import PathCollection

    base = np.array([0, 1, 0, 2, 1, 3], dtype=float)
    query = normalize_series_quantile(
        np.stack([np.zeros_like(base), base]), quantile=0.5, mode="absolute"
    )
    calibration = normalize_series_quantile(
        np.stack([base, base + 1]), quantile=0.5, mode="absolute"
    )
    bank = normalize_series_quantile(base[None], quantile=0.5, mode="absolute")
    assessment = assess_series_similarity(
        query,
        calibration,
        bank,
        lags=(1,),
        metrics=("mse",),
        query_groups=[30, 31],
        calibration_groups=[20, 21],
        bank_groups=[10],
    )
    viz.plot_similarity_scores(assessment, str(tmp_path / "missing-score.png"))
    ax = capture[-1].axes[0]
    markers = next(
        collection for collection in ax.collections if isinstance(collection, PathCollection)
    )
    np.testing.assert_allclose(markers.get_offsets(), [[0, 0.25]])


@pytest.mark.parametrize("mismatch", ("lags", "time", "series"))
def test_temporal_rejects_incompatible_diagnostic_definitions(mismatch, tmp_path):
    values = np.arange(8, dtype=float)[None, :, None]
    reference = temporal_statistics(values, lags=(1,))
    query_values = values[:, :-1] if mismatch == "time" else values
    if mismatch == "series":
        query_values = np.repeat(query_values, 2, axis=2)
    query = temporal_statistics(query_values, lags=(2,) if mismatch == "lags" else (1,))
    with pytest.raises(ValueError):
        viz.plot_temporal_comparison(reference, query, str(tmp_path / "incompatible.png"))


@pytest.mark.parametrize("mismatch", ("lags", "views", "pairs"))
def test_dependence_rejects_incompatible_diagnostic_definitions(mismatch, tmp_path):
    values = np.stack([np.arange(8), np.arange(8) ** 2], axis=1)[None]
    reference = lagged_dependence(values, pairs=[(0, 1)], lags=(0, 1))
    query = lagged_dependence(
        values,
        pairs=[(1, 0)] if mismatch == "pairs" else [(0, 1)],
        lags=(0, 2) if mismatch == "lags" else (0, 1),
        views=("levels",) if mismatch == "views" else ("levels", "differences"),
    )
    with pytest.raises(ValueError):
        viz.plot_dependence_comparison(reference, query, str(tmp_path / "incompatible.png"))


def test_activity_rejects_incomparable_window_lengths(tmp_path):
    with pytest.raises(ValueError):
        viz.plot_activity_runs(
            activity_runs(np.zeros((1, 8, 1))),
            activity_runs(np.zeros((1, 7, 1))),
            str(tmp_path / "incompatible.png"),
        )


def test_similarity_rejects_a_different_calibration_bank(tmp_path):
    assessment, _, calibration, bank = score_assessment()
    other = nearest_series(compare_series(calibration, bank + 1, lags=(1,), metrics=("mse",)))
    with pytest.raises(ValueError):
        viz.plot_similarity_scores(
            replace(assessment, calibration_matches=other), str(tmp_path / "incompatible.png")
        )


def test_comparison_panel_bounds_require_explicit_selection(tmp_path):
    values = np.repeat(np.arange(8, dtype=float)[None, :, None], 5, axis=2)
    temporal = temporal_statistics(values, lags=(1,))
    runs = activity_runs(values)
    with pytest.raises(ValueError):
        viz.plot_temporal_comparison(temporal, temporal, str(tmp_path / "wide-temporal.png"))
    with pytest.raises(ValueError):
        viz.plot_activity_runs(runs, runs, str(tmp_path / "wide-runs.png"))
    comparison = compare_series(np.zeros((25, 8)), np.zeros((1, 8)), lags=(1,), metrics=("mse",))
    with pytest.raises(ValueError):
        viz.plot_series_matches(
            comparison, nearest_series(comparison), str(tmp_path / "wide-matches.png"), metric="mse"
        )


def test_single_lag_reference_quantiles_keep_their_numeric_interval(capture, tmp_path):
    from matplotlib.collections import LineCollection

    reference_values = np.array(
        [[0, 1, 0, 2, 1, 3], [1, 0, 2, 1, 0, 3], [2, 1, 0, 1, 3, 2]], dtype=float
    )
    reference = temporal_statistics(reference_values[..., None], lags=(1,))
    query = temporal_statistics(reference_values[:1, :, None], lags=(1,))
    viz.plot_temporal_comparison(
        reference,
        query,
        str(tmp_path / "one-lag.png"),
        statistics=("overlap_pearson",),
        band=(0.25, 0.75),
    )
    expected = np.quantile(
        [pearson(row[:-1], row[1:]) for row in reference_values], [0.25, 0.5, 0.75]
    )
    ax = capture[-1].axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xydata(), [[1, expected[1]]])
    interval = next(
        collection for collection in ax.collections if isinstance(collection, LineCollection)
    )
    np.testing.assert_allclose(interval.get_segments(), [[[1, expected[0]], [1, expected[2]]]])
