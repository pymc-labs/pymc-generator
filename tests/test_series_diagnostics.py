"""Independent numerical oracles for focused, world-first series diagnostics."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from pymc_generator.diagnostics import (
    _acf_world,
    _centered_rows,
    activity_runs,
    lagged_dependence,
    temporal_statistics,
)


def _pearson(left, right):
    """Scalar definition, independent of the production row normalizer."""
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right, strict=True)
        if np.isfinite(x) and np.isfinite(y)
    ]
    if len(pairs) < 3:
        return np.nan
    left_mean = math.fsum(x for x, _ in pairs) / len(pairs)
    right_mean = math.fsum(y for _, y in pairs) / len(pairs)
    dx = [x - left_mean for x, _ in pairs]
    dy = [y - right_mean for _, y in pairs]
    xx = math.fsum(x * x for x in dx)
    yy = math.fsum(y * y for y in dy)
    if xx == 0.0 or yy == 0.0:
        return np.nan
    return math.fsum(x * y for x, y in zip(dx, dy, strict=True)) / math.sqrt(xx * yy)


def _population_std(values, *, differences=False):
    """High-precision moments retain extremes without float squared moments."""
    with localcontext() as context:
        context.prec = 1500
        data = [Decimal.from_float(float(value)) for value in values]
        if differences:
            data = [right - left for left, right in zip(data[:-1], data[1:], strict=True)]
        mean = sum(data) / len(data)
        variance = sum((value - mean) ** 2 for value in data) / len(data)
        return float(variance.sqrt())


def _aligned(left, right, lag):
    if abs(lag) >= len(left):
        return left[:0], right[:0]
    if lag >= 0:
        return left[: len(left) - lag], right[lag:]
    return left[-lag:], right[: len(right) + lag]


def test_temporal_moments_and_overlap_correlations_match_scalar_oracles():
    block = np.array(
        [
            [[2, -3], [5, 1], [1, 6], [9, 2], [4, 8], [11, 7], [6, 12]],
            [[8, 4], [3, 9], [10, 2], [1, 6], [7, 5], [2, 11], [13, 3]],
        ],
        dtype=float,
    )
    lags = (2, 1, 6, 9)
    result = temporal_statistics(block, lags=lags)
    assert result.lags == lags
    assert result.n_time_steps == 7
    for world, series in np.ndindex((2, 2)):
        levels = block[world, :, series]
        changes = np.diff(levels)
        assert result.values["level_std"][world, series] == pytest.approx(_population_std(levels))
        assert result.values["difference_std"][world, series] == pytest.approx(
            _population_std(levels, differences=True)
        )
        assert result.values["time_correlation"][world, series] == pytest.approx(
            _pearson(levels, np.arange(len(levels)))
        )
        for name, view in (("overlap_pearson", levels), ("diff_overlap_pearson", changes)):
            for position, lag in enumerate(lags):
                left, right = _aligned(view, view, lag)
                expected = _pearson(left, right)
                index = (world, series, position)
                assert result.counts[name][index] == max(len(view) - lag, 0)
                assert result.valid[name][index] == np.isfinite(expected)
                if np.isfinite(expected):
                    assert result.values[name][index] == pytest.approx(expected)
                else:
                    assert np.isnan(result.values[name][index])
    np.testing.assert_array_equal(result.counts["level_std"], np.full((2, 2), 7))
    np.testing.assert_array_equal(result.counts["difference_std"], np.full((2, 2), 6))


def test_overlap_estimator_is_distinct_from_unchanged_sample_acf():
    levels = np.array([2.0, 7.0, 3.0, 10.0, 5.0, 12.0])
    lag = 2
    centered = levels - math.fsum(levels) / len(levels)
    expected_sample_acf = math.fsum(
        float(x * y) for x, y in zip(centered[:-lag], centered[lag:], strict=True)
    ) / math.fsum(float(x * x) for x in centered)
    expected_overlap = _pearson(levels[:-lag], levels[lag:])
    result = temporal_statistics(levels[None, :, None], lags=(lag,))
    legacy, legacy_valid = _acf_world(levels[None, :], (lag,))
    assert legacy_valid[0, 0]
    assert legacy[0, 0] == pytest.approx(expected_sample_acf)
    assert result.values["overlap_pearson"][0, 0, 0] == pytest.approx(expected_overlap)
    assert not math.isclose(expected_sample_acf, expected_overlap)


def test_source_float32_jitter_is_constant_but_resolved_variation_is_retained():
    eps32 = np.finfo(np.float32).eps
    jitter = np.array([1, 1 + eps32, 1, 1 + 2 * eps32, 1, 1 + eps32], dtype=np.float32)
    trend = np.float32(1.0) + np.arange(6, dtype=np.float32) * np.float32(0.03)
    block = np.stack((jitter, trend), axis=1)[None, :, :]
    result = temporal_statistics(block, lags=(1,))
    assert result.valid["level_std"][0, 0]
    assert result.values["level_std"][0, 0] == 0.0
    assert result.values["difference_std"][0, 0] == 0.0
    assert not result.valid["time_correlation"][0, 0]
    assert not result.valid["overlap_pearson"][0, 0, 0]
    assert result.values["level_std"][0, 1] == pytest.approx(_population_std(trend))
    assert result.values["difference_std"][0, 1] == 0.0
    assert not result.valid["diff_overlap_pearson"][0, 1, 0]
    # Explicit precision describes a source; float64 conversion alone must not
    # reinterpret float32 rounding noise as signal.
    restored = temporal_statistics(block.astype(float), lags=(1,), resolution=eps32)
    np.testing.assert_array_equal(restored.values["level_std"], result.values["level_std"])
    precise = temporal_statistics(block.astype(float), lags=(1,))
    assert precise.values["level_std"][0, 0] > 0.0
    assert precise.values["difference_std"][0, 1] > 0.0


def test_centered_rows_joint_finiteness_translation_and_precision_floor():
    baseline = 1e150
    step = np.spacing(baseline) * 16.0
    translated = baseline + step * np.array([0.0, 1.0, 5.0, 2.0, 8.0])
    rows = np.stack((translated, np.arange(5.0), np.ones(5)))
    rows[1, 1] = np.nan
    finite = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool)
    unit, valid = _centered_rows(rows, np.finfo(float).eps, finite=finite)
    np.testing.assert_array_equal(valid, [True, True, False])
    translated_offsets = (translated - baseline) / step
    expected = translated_offsets - translated_offsets.mean()
    expected /= math.sqrt(math.fsum(float(x * x) for x in expected))
    np.testing.assert_allclose(unit[0], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(unit[1], [-1 / math.sqrt(2), 0, 0, 0, 1 / math.sqrt(2)])
    np.testing.assert_array_equal(unit[2], np.zeros(5))
    short, short_valid = _centered_rows(rows[:1], np.finfo(float).eps, finite=[[1, 1, 0, 0, 0]])
    assert not short_valid[0]
    np.testing.assert_array_equal(short, np.zeros((1, 5)))
    floor_unit, floor_valid = _centered_rows(
        np.arange(5.0)[None, :], np.finfo(float).eps, constant_floor=4.0
    )
    assert not floor_valid[0]
    np.testing.assert_array_equal(floor_unit, np.zeros((1, 5)))


@pytest.mark.parametrize(
    "levels",
    [
        np.array(
            [
                -np.finfo(float).max,
                -np.finfo(float).max,
                np.finfo(float).max,
                np.finfo(float).max,
                np.finfo(float).max,
            ]
        ),
        np.array([1e308, 1e308 + 3e293, 1e308 + 8e293, 1e308 + 2e293, 1e308 + 9e293]),
        np.array([1e-310, 5e-310, 2e-310, 8e-310, 3e-310]),
        1e-200 + np.array([0.0, 2e-213, 9e-213, 3e-213, 7e-213]),
    ],
)
def test_extreme_population_moments_retain_representable_original_units(levels):
    with np.errstate(over="raise", invalid="raise"):
        result = temporal_statistics(levels[None, :, None], lags=(1,))
    for name, differences in (("level_std", False), ("difference_std", True)):
        expected = _population_std(levels, differences=differences)
        assert result.valid[name][0, 0]
        assert expected > 0.0
        np.testing.assert_allclose(result.values[name][0, 0], expected, rtol=2e-13, atol=0.0)


def test_differences_scale_before_overflow_and_restore_only_after_mean():
    maximum = np.finfo(float).max
    levels = np.array([-maximum, maximum, maximum, maximum, maximum, maximum])
    with np.errstate(over="raise", invalid="raise"):
        result = temporal_statistics(levels[None, :, None], lags=(1,))
    expected = _population_std(levels, differences=True)
    assert np.isfinite(expected)
    assert result.valid["difference_std"][0, 0]
    assert result.values["difference_std"][0, 0] == pytest.approx(expected)
    # The first difference is 2 * maxfloat, yet the population standard
    # deviation is 0.8 * maxfloat and is representable.
    assert expected == pytest.approx(0.8 * maximum)
    unrepresentable = temporal_statistics(
        np.array([-maximum, maximum, -maximum, maximum])[None, :, None], lags=()
    )
    assert not unrepresentable.valid["difference_std"][0, 0]
    assert np.isnan(unrepresentable.values["difference_std"][0, 0])


def test_safe_difference_correlations_match_scale_invariant_oracle():
    scaled = np.array([-1.0, 1.0, 1.0, -1.0, 0.5, 0.0, 1.0, -1.0])
    levels = scaled * np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        temporal = temporal_statistics(levels[None, :, None], lags=(1, 2))
        lagged = lagged_dependence(levels[None, :, None], pairs=[(0, 0)], lags=(1, 2))
    changes = np.diff(scaled)
    for position, lag in enumerate((1, 2)):
        expected = _pearson(changes[:-lag], changes[lag:])
        assert temporal.values["diff_overlap_pearson"][0, 0, position] == pytest.approx(expected)
        assert lagged.correlation[0, 1, 0, position] == pytest.approx(expected)


@pytest.mark.parametrize("n_time", [1, 2, 3])
def test_short_temporal_windows_have_honest_validity_and_counts(n_time):
    levels = np.arange(n_time, dtype=float)[None, :, None]
    result = temporal_statistics(levels, lags=(1, 2, 5))
    assert result.valid["level_std"][0, 0]
    assert result.counts["level_std"][0, 0] == n_time
    assert result.valid["difference_std"][0, 0] == (n_time >= 2)
    assert result.counts["difference_std"][0, 0] == n_time - 1
    assert result.valid["time_correlation"][0, 0] == (n_time >= 3)
    assert not result.valid["overlap_pearson"].any()
    np.testing.assert_array_equal(
        result.counts["overlap_pearson"][0, 0], [max(n_time - h, 0) for h in (1, 2, 5)]
    )
    assert not result.valid["diff_overlap_pearson"].any()


def test_eligibility_and_amplitude_availability_are_separate():
    block = np.array([2.0, 6.0, 1.0, 8.0, 3.0, 9.0])[None, :, None]
    block = np.tile(block, (2, 1, 2))
    eligible = np.array([[True, False], [True, True]])
    result = temporal_statistics(
        block, lags=(1,), eligible=eligible, scale_available=[[False], [True]]
    )
    assert not result.valid["level_std"][0].any()
    assert not result.valid["difference_std"][0].any()
    assert result.valid["time_correlation"][0, 0]
    assert result.valid["overlap_pearson"][0, 0, 0]
    for name in result.values:
        assert np.isnan(result.values[name][0, 1]).all()
        assert not result.valid[name][0, 1].any()
        assert not result.counts[name][0, 1].any()
    assert result.counts["level_std"][0, 0] == 6


def test_activity_spell_summaries_and_records_have_independent_literal_oracle():
    sparse = np.array([0, 0, 4, 4, 0, -2, -2, -2, 0, 0, 0, 3, 0, 0])
    result = activity_runs(sparse[None, :, None])
    expected = {
        "leading_inactive_length": 2,
        "trailing_inactive_length": 2,
        "longest_active_length": 3,
        "longest_inactive_length": 3,
        "active_spell_count": 3,
        "inactive_spell_count": 4,
        "complete_active_spell_count": 3,
        "complete_inactive_spell_count": 2,
        "censored_active_spell_count": 0,
        "censored_inactive_spell_count": 2,
        "complete_active_length_median": 2,
        "complete_inactive_length_median": 2,
        "switching_rate": 6 / 13,
        "interior_inactive_fraction": 4 / 14,
    }
    for name, value in expected.items():
        assert result.valid[name][0, 0]
        assert result.values[name][0, 0] == pytest.approx(value)
        count = {
            "complete_active_length_median": 3,
            "complete_inactive_length_median": 2,
            "switching_rate": 13,
        }.get(name, 14)
        assert result.counts[name][0, 0] == count
    expected_runs = [
        (0, 2, False, True, False),
        (2, 4, True, False, False),
        (4, 5, False, False, False),
        (5, 8, True, False, False),
        (8, 11, False, False, False),
        (11, 12, True, False, False),
        (12, 14, False, False, True),
    ]
    assert result.records == tuple(
        {
            "world": 0,
            "series": 0,
            "start": start,
            "stop": stop,
            "active": active,
            "left_censored": left,
            "right_censored": right,
        }
        for start, stop, active, left, right in expected_runs
    )
    frame = result.to_frame()
    assert frame.loc[0, "interior_inactive_fraction"] == pytest.approx(4 / 14)
    assert frame.loc[0, "complete_inactive_length_median_count"] == 2
    assert result.runs_frame().to_dict("records") == list(result.records)


@pytest.mark.parametrize("active", [False, True])
def test_constant_activity_is_one_doubly_censored_spell(active):
    result = activity_runs(np.full((1, 5, 1), 7.0 if active else 0.0))
    assert len(result.records) == 1
    assert result.records[0] == {
        "world": 0,
        "series": 0,
        "start": 0,
        "stop": 5,
        "active": active,
        "left_censored": True,
        "right_censored": True,
    }
    state = "active" if active else "inactive"
    other = "inactive" if active else "active"
    assert result.values[f"{state}_spell_count"][0, 0] == 1
    assert result.values[f"censored_{state}_spell_count"][0, 0] == 1
    assert result.values[f"{other}_spell_count"][0, 0] == 0
    assert result.values["interior_inactive_fraction"][0, 0] == 0.0
    assert result.valid["interior_inactive_fraction"][0, 0]
    assert result.values["switching_rate"][0, 0] == 0.0
    for name in ("complete_active_length_median", "complete_inactive_length_median"):
        assert not result.valid[name][0, 0]
        assert np.isnan(result.values[name][0, 0])
        assert result.counts[name][0, 0] == 0


def test_activity_uses_exact_nonzero_and_respects_eligibility_without_inference():
    tiny = np.nextafter(0.0, 1.0)
    block = np.array([[0.0, 0.0], [tiny, 0.0], [tiny, 0.0], [0.0, 0.0]])[None, :, :]
    result = activity_runs(block)
    assert result.eligible.all()
    assert result.values["longest_active_length"][0, 0] == 2
    assert result.values["complete_active_length_median"][0, 0] == 2
    assert result.values["inactive_spell_count"][0, 1] == 1
    masked = activity_runs(block, eligible=[True, False])
    assert all(record["series"] == 0 for record in masked.records)
    for name in masked.values:
        assert np.isnan(masked.values[name][0, 1])
        assert not masked.valid[name][0, 1]
        assert masked.counts[name][0, 1] == 0
    single = activity_runs(np.zeros((1, 1, 1)))
    assert not single.valid["switching_rate"][0, 0]
    assert single.counts["switching_rate"][0, 0] == 0
    assert single.values["leading_inactive_length"][0, 0] == 1
    assert single.values["trailing_inactive_length"][0, 0] == 1


def test_signed_pair_orientation_and_views_match_world_first_oracles():
    first = np.array(
        [[2, 9, 4], [7, 2, 1], [1, 7, 6], [8, 1, 3], [3, 8, 9], [9, 3, 2], [4, 9, 7], [6, 4, 5]],
        dtype=float,
    )
    second = np.array(
        [[8, -2, 7], [1, 5, 3], [6, 1, 9], [2, 8, 4], [9, 3, 1], [3, 7, 8], [7, 4, 2], [4, 9, 6]],
        dtype=float,
    )
    block = np.stack((first, second))
    pairs = [(0, 1), (1, 0), (0, 2)]
    lags = (1, -2, 0, 2, 20)
    result = lagged_dependence(block, pairs=pairs, lags=lags)
    assert result.correlation.shape == (2, 2, 3, 5)
    for world, view, pair, position in np.ndindex(result.correlation.shape):
        values = block[world] if view == 0 else np.diff(block[world], axis=0)
        left, right = _aligned(values[:, pairs[pair][0]], values[:, pairs[pair][1]], lags[position])
        expected = _pearson(left, right)
        index = (world, view, pair, position)
        assert result.n_observations[index] == len(left)
        assert result.valid[index] == np.isfinite(expected)
        if np.isfinite(expected):
            assert result.correlation[index] == pytest.approx(expected)
        else:
            assert np.isnan(result.correlation[index])
    assert result.correlation[0, 0, 0, 0] != pytest.approx(result.correlation[0, 0, 1, 0])
    frame = result.to_frame()
    row = frame[
        (frame.world == 0) & (frame["view"] == "levels") & (frame.pair == 0) & (frame.lag == 1)
    ].iloc[0]
    assert row["left"] == 0
    assert row["right"] == 1
    assert row["correlation"] == pytest.approx(_pearson(first[:-1, 0], first[1:, 1]))
    temporal = temporal_statistics(block, lags=(1, 2))
    self_pairs = lagged_dependence(block, pairs=[(0, 0), (1, 1), (2, 2)], lags=(1, 2))
    np.testing.assert_allclose(
        self_pairs.correlation[:, 0], temporal.values["overlap_pearson"], equal_nan=True
    )
    np.testing.assert_allclose(
        self_pairs.correlation[:, 1], temporal.values["diff_overlap_pearson"], equal_nan=True
    )


def test_lagged_finite_counts_and_source_precision_are_explicit():
    block = np.array(
        [[[1, 2], [3, 5], [np.nan, 1], [5, 8], [7, np.inf], [4, 3], [8, 9]]], dtype=float
    )
    result = lagged_dependence(block, pairs=[(0, 1)], lags=(0, 1, -1, 6, 7))
    for view in range(2):
        values = block[0] if view == 0 else np.diff(block[0], axis=0)
        for position, lag in enumerate(result.lags):
            left, right = _aligned(values[:, 0], values[:, 1], lag)
            count = sum(np.isfinite(x) and np.isfinite(y) for x, y in zip(left, right, strict=True))
            assert result.n_observations[0, view, 0, position] == count
            expected = _pearson(left, right)
            assert result.valid[0, view, 0, position] == np.isfinite(expected)
            if np.isfinite(expected):
                assert result.correlation[0, view, 0, position] == pytest.approx(expected)
    eps32 = np.finfo(np.float32).eps
    jitter = np.float32(1.0) + np.array([0, 1, 0, 2, 1, 0], dtype=np.float32) * eps32
    other = np.array([2, 5, 1, 7, 3, 8], dtype=np.float32)
    noisy = np.stack((jitter, other), axis=1)[None, :, :]
    unresolved = lagged_dependence(noisy, pairs=[(0, 1)], lags=(0,))
    assert not unresolved.valid.any()
    np.testing.assert_array_equal(unresolved.n_observations[0, :, 0, 0], [6, 5])
    resolved = lagged_dependence(noisy.astype(float), pairs=[(0, 1)], lags=(0,))
    assert resolved.valid.all()
    ineligible = lagged_dependence(noisy, pairs=[(0, 1)], lags=(0,), eligible=[True, False])
    assert not ineligible.eligible.any()
    assert not ineligible.n_observations.any()
    assert np.isnan(ineligible.correlation).all()


def test_lagged_dependence_never_pools_worlds():
    block = np.stack((np.tile([1.0, 2.0], (5, 1)), np.tile([9.0, 18.0], (5, 1))))
    result = lagged_dependence(block, pairs=[(0, 1)], lags=(0, 1))
    assert not result.valid.any()
    assert np.isnan(result.correlation).all()
    np.testing.assert_array_equal(result.n_observations[:, 0, 0], [[5, 4], [5, 4]])


@pytest.mark.parametrize("shape", [(0, 5, 2), (2, 5, 0), (0, 5, 0)])
def test_empty_world_and_series_axes_keep_meaningful_shapes(shape):
    block = np.empty(shape)
    temporal = temporal_statistics(block, lags=(1, 7))
    activity = activity_runs(block)
    lagged = lagged_dependence(block, pairs=[], lags=(-1, 0, 1))
    summary_shape = (shape[0], shape[2])
    assert temporal.eligible.shape == summary_shape
    assert activity.eligible.shape == summary_shape
    assert temporal.values["level_std"].shape == summary_shape
    assert temporal.values["overlap_pearson"].shape == (*summary_shape, 2)
    assert lagged.pairs.shape == (0, 2)
    assert lagged.correlation.shape == (shape[0], 2, 0, 3)
    assert lagged.eligible.shape == (shape[0], 0)
    assert not activity.records
    assert temporal.to_frame().empty
    assert activity.to_frame().empty
    assert activity.runs_frame().empty
    assert lagged.to_frame().empty


def test_empty_lag_axes_and_zero_eligibility_keep_counts_honest():
    block = np.zeros((2, 4, 2))
    temporal = temporal_statistics(block, lags=(), eligible=False)
    assert temporal.values["overlap_pearson"].shape == (2, 2, 0)
    assert all(not count.any() for count in temporal.counts.values())
    activity = activity_runs(block, eligible=False)
    assert not activity.records
    assert all(not count.any() for count in activity.counts.values())
    lagged = lagged_dependence(block, pairs=[(0, 1)], lags=())
    assert lagged.correlation.shape == (2, 2, 1, 0)
    assert lagged.to_frame().empty
    empty, valid = _centered_rows(np.empty((0, 4)), np.finfo(float).eps)
    assert empty.shape == (0, 4)
    assert valid.shape == (0,)


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (temporal_statistics, {"lags": (1,)}),
        (activity_runs, {}),
        (lagged_dependence, {"pairs": [(0, 0)], "lags": (0,)}),
    ],
)
def test_value_entrypoints_reject_masked_arrays_and_empty_time(function, kwargs):
    masked = np.ma.array(np.ones((1, 4, 1)), mask=False)
    with pytest.raises(TypeError, match="masked"):
        function(masked, **kwargs)
    with pytest.raises(ValueError, match="time"):
        function(np.empty((1, 0, 1)), **kwargs)
    with pytest.raises(ValueError, match="shape"):
        function(np.ones((4, 1)), **kwargs)
    with pytest.raises(TypeError, match="numeric"):
        function(np.full((1, 4, 1), "text"), **kwargs)


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "function,kwargs", [(temporal_statistics, {"lags": (1,)}), (activity_runs, {})]
)
def test_temporal_and_activity_require_finite_values_even_when_ineligible(function, kwargs, poison):
    block = np.ones((1, 4, 1))
    block[0, 1, 0] = poison
    with pytest.raises(ValueError, match="finite"):
        function(block, eligible=False, **kwargs)


@pytest.mark.parametrize(
    "function,kwargs,key,value",
    [
        (temporal_statistics, {"lags": (1,)}, "eligible", [[True]]),
        (temporal_statistics, {"lags": (1,)}, "resolution", [[1e-7]]),
        (temporal_statistics, {"lags": (1,)}, "scale_available", [[True]]),
        (temporal_statistics, {}, "lags", [1]),
        (activity_runs, {}, "eligible", [[True]]),
        (lagged_dependence, {"pairs": [(0, 0)], "lags": (0,)}, "eligible", [[True]]),
        (lagged_dependence, {"pairs": [(0, 0)], "lags": (0,)}, "resolution", [[1e-7]]),
        (lagged_dependence, {"lags": (0,)}, "pairs", [[0, 0]]),
        (lagged_dependence, {"pairs": [(0, 0)]}, "lags", [0]),
        (lagged_dependence, {"pairs": [(0, 0)], "lags": (0,)}, "views", ["levels"]),
    ],
)
def test_all_selector_and_precision_entrypoints_reject_masked_arrays(function, kwargs, key, value):
    kwargs = {**kwargs, key: np.ma.array(value, mask=False)}
    with pytest.raises(TypeError, match="masked"):
        function(np.ones((1, 4, 1)), **kwargs)


@pytest.mark.parametrize(
    "key,value",
    [("resolution", [1e-7]), ("constant_floor", [0.0]), ("finite", [[True, True, True]])],
)
def test_centered_rows_reject_masked_metadata(key, value):
    kwargs = {"resolution": np.finfo(float).eps, key: np.ma.array(value, mask=False)}
    with pytest.raises(TypeError, match="masked"):
        _centered_rows(np.arange(3.0)[None, :], **kwargs)
    with pytest.raises(TypeError, match="masked"):
        _centered_rows(np.ma.array(np.arange(3.0)[None, :]), np.finfo(float).eps)


@pytest.mark.parametrize("lags", [(0,), (-1,), (1, 1), (True,), (1.5,), None])
def test_temporal_lags_are_explicit_unique_positive_integers(lags):
    with pytest.raises((TypeError, ValueError)):
        temporal_statistics(np.ones((1, 4, 1)), lags=lags)


@pytest.mark.parametrize(
    "pairs",
    [
        [[0.0, 1.0]],
        [[True, False]],
        [[-1, 0]],
        [[0, 2]],
        [[np.uint64(2**64 - 1), np.uint64(0)]],
        [0, 1],
    ],
)
def test_pair_positions_are_validated_without_lossy_casting(pairs):
    with pytest.raises((TypeError, ValueError)):
        lagged_dependence(np.ones((1, 4, 2)), pairs=pairs, lags=(0,))


@pytest.mark.parametrize("resolution", [0.0, -1e-7, np.nan, np.inf])
def test_resolution_requires_positive_finite_values(resolution):
    with pytest.raises(ValueError, match="resolution"):
        temporal_statistics(np.ones((1, 4, 1)), lags=(), resolution=resolution)


def test_invalid_masks_shapes_views_and_signed_lags_are_rejected():
    block = np.ones((1, 4, 2))
    with pytest.raises(ValueError, match="binary"):
        activity_runs(block, eligible=[1, 2])
    with pytest.raises(ValueError, match="broadcast"):
        temporal_statistics(block, lags=(), eligible=np.ones((3, 3)))
    for views in ((), ("levels", "levels"), ("unknown",)):
        with pytest.raises(ValueError, match="views"):
            lagged_dependence(block, pairs=[(0, 1)], lags=(0,), views=views)
    for lags in ((True,), (0.5,), (0, 0)):
        with pytest.raises((TypeError, ValueError)):
            lagged_dependence(block, pairs=[(0, 1)], lags=lags)


def test_subnormal_resolved_variance_is_not_erased_by_floor_rounding():
    quantum = np.nextafter(0.0, 1.0)
    levels = 1e-308 + np.array([0.0, 2.0, 0.0, 2.0, 0.0]) * quantum
    result = temporal_statistics(levels[None, :, None], lags=(1,))
    for name, differences in (("level_std", False), ("difference_std", True)):
        assert result.valid[name][0, 0]
        assert result.values[name][0, 0] == _population_std(levels, differences=differences)
        assert result.values[name][0, 0] > 0.0
    # A genuinely nonrepresentable positive moment must not become a
    # source-constant zero simply because its restoration underflows.
    underflowed = temporal_statistics(np.array([0.0, quantum, 0.0])[None, :, None], lags=())
    assert not underflowed.valid["level_std"][0, 0]
    assert np.isnan(underflowed.values["level_std"][0, 0])


def test_centered_wide_range_fallback_retains_the_signed_direction():
    scaled = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    levels = scaled * np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        unit, valid = _centered_rows(levels[None, :], np.finfo(float).eps)
    expected = scaled - math.fsum(scaled) / len(scaled)
    expected /= math.sqrt(math.fsum(float(value * value) for value in expected))
    assert valid[0]
    np.testing.assert_allclose(unit[0], expected, rtol=1e-14, atol=1e-14)


def test_temporal_frame_retains_scalar_and_lag_observation_counts():
    levels = np.array([2.0, 6.0, 1.0, 8.0, 3.0, 9.0])
    result = temporal_statistics(levels[None, :, None], lags=(1, 5))
    frame = result.to_frame()
    scalar = frame[frame.statistic == "difference_std"].iloc[0]
    assert scalar["world"] == 0
    assert scalar["series"] == 0
    assert scalar["value"] == pytest.approx(_population_std(levels, differences=True))
    assert scalar["n_observations"] == 5
    assert scalar["valid"]
    lag = frame[(frame.statistic == "overlap_pearson") & (frame.lag == 1)].iloc[0]
    assert lag["value"] == pytest.approx(_pearson(levels[:-1], levels[1:]))
    assert lag["n_observations"] == 5
    short = frame[(frame.statistic == "overlap_pearson") & (frame.lag == 5)].iloc[0]
    assert short["n_observations"] == 1
    assert not short["valid"]
    assert np.isnan(short["value"])


def test_empty_world_pool_can_retain_a_nonempty_pair_axis():
    result = lagged_dependence(np.empty((0, 4, 2)), pairs=[(0, 1)], lags=(-1, 0, 1))
    assert result.correlation.shape == (0, 2, 1, 3)
    assert result.valid.shape == (0, 2, 1, 3)
    assert result.n_observations.shape == (0, 2, 1, 3)
    assert result.eligible.shape == (0, 1)
    assert result.to_frame().empty


def test_single_time_step_pair_differences_have_zero_observations():
    result = lagged_dependence(np.array([[[2.0, 5.0]]]), pairs=[(0, 1)], lags=(0, 1, -1))
    np.testing.assert_array_equal(result.n_observations[0, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(result.n_observations[0, 1, 0], [0, 0, 0])
    assert not result.valid.any()
    assert np.isnan(result.correlation).all()
