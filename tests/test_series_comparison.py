"""Independent numerical oracles for alignment-sensitive waveform comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from pymc_generator.series_comparison import (
    assess_series_similarity,
    compare_series,
    nearest_series,
    normalize_series_quantile,
)


def _overlap(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def test_single_series_extreme_signed_quantile_keeps_zero_divisor_metadata():
    values = np.array([-1e308, 1e308])
    normalized = normalize_series_quantile(values, quantile=0.5, mode="signed")
    assert normalized.divisors == 0
    assert not normalized.scale_available
    np.testing.assert_array_equal(normalized.values, values)


def test_namespaced_unknown_groups_are_not_promoted_to_known_groups():
    values = [[0, 1, 4, 2, 3]]
    with pytest.raises(ValueError, match="unknown group"):
        assess_series_similarity(
            values,
            values,
            values,
            lags=(1,),
            query_groups=[("query", -1)],
            calibration_groups=[("reference", 1)],
            bank_groups=[("reference", 2)],
        )


def test_quantile_scaling_modes_axes_and_negative_divisors():
    values = np.array([[-8.0, -4.0, -2.0, 0.0], [1.0, 2.0, 3.0, 7.0]])
    absolute = normalize_series_quantile(values, quantile=0.75, mode="absolute")
    signed = normalize_series_quantile(values, quantile=0.5, mode="signed")
    np.testing.assert_allclose(absolute.divisors, np.quantile(abs(values), 0.75, axis=1))
    np.testing.assert_allclose(absolute.values, values / absolute.divisors[:, None])
    np.testing.assert_array_equal(signed.divisors, [-3.0, 2.5])
    np.testing.assert_allclose(signed.values, values / np.array([-3.0, 2.5])[:, None])
    transposed = normalize_series_quantile(values.T, quantile=0.75, mode="absolute", axis=0)
    np.testing.assert_array_equal(transposed.values, absolute.values.T)
    np.testing.assert_array_equal(values, [[-8, -4, -2, 0], [1, 2, 3, 7]])
    assert signed.values[0, -1] == 0


def test_zero_quantile_preserves_sparse_values_but_only_disables_amplitude_metric():
    raw = np.array([[0.0, 0.0, -2.0, 0.0, 0.0]])
    result = normalize_series_quantile(raw, quantile=0.5, mode="signed")
    np.testing.assert_array_equal(result.values, raw)
    assert not result.scale_available[0]
    comparison = compare_series(result, result, lags=(1,))
    assert np.isnan(comparison.values["mse"][0, 0])
    assert comparison.values["level_r"][0, 0] == pytest.approx(1)
    assert comparison.values["diff_r"][0, 0] == pytest.approx(1)


def test_extreme_quantile_interpolation_and_unrepresentable_scaling():
    result = normalize_series_quantile(np.array([[-1e308, 1e308]]), quantile=0.5, mode="signed")
    assert result.divisors[0] == 0
    np.testing.assert_array_equal(result.values, [[-1e308, 1e308]])
    # A subnormal minimum must not disappear through whole-row rescaling.
    smallest = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="represented"):
        normalize_series_quantile([[smallest, 1e308]], quantile=0, mode="absolute")
    with pytest.raises(ValueError, match="represented"):
        normalize_series_quantile([[0, 1e-309, 1]], quantile=0.5, mode="absolute")


def test_source_precision_is_retained_by_normalization():
    values = np.array([[1, 1 + np.finfo(np.float32).eps, 1, 1]], dtype=np.float32)
    normalized = normalize_series_quantile(values, quantile=0.5, mode="absolute")
    assert normalized.constant[0]
    result = compare_series(normalized, normalized, lags=(1,))
    assert np.isnan(result.values["level_r"][0, 0])
    assert np.isnan(result.values["diff_r"][0, 0])


def test_all_pair_metrics_match_independent_multirow_oracles():
    query = np.array([[1, 4, 2, 7, 3, 8, 5], [2, 0, 3, 1, 4, 0, 6]], dtype=float)
    reference = np.array(
        [[0, 2, 1, 5, 2, 8, 3], [3, 1, 5, 0, 7, 2, 9], [9, 3, 4, 1, 5, 0, 8]], dtype=float
    )
    lags = (1, 2)
    actual = compare_series(query, reference, lags=lags)
    for i, q in enumerate(query):
        for j, r in enumerate(reference):
            assert actual.values["mse"][i, j] == pytest.approx(np.mean((q - r) ** 2))
            assert actual.values["level_r"][i, j] == pytest.approx(np.corrcoef(q, r)[0, 1])
            assert actual.values["diff_r"][i, j] == pytest.approx(
                np.corrcoef(np.diff(q), np.diff(r))[0, 1]
            )
            expected = np.sqrt(
                np.mean([(_overlap(q, lag) - _overlap(r, lag)) ** 2 for lag in lags])
            )
            assert actual.values["overlap_pearson_rmse"][i, j] == pytest.approx(expected)


def test_signed_best_match_is_not_absolute_correlation_maximum():
    query = np.array([[0, 2, -1, 4, 1, 7, 3]], dtype=float)
    bank = np.vstack((-query[0], query[0] + np.array([0, 0, 0, 0, 0, 0, 1])))
    result = nearest_series(compare_series(query, bank, lags=(1,)))
    assert result.indices["level_r"][0] == 1
    assert result.indices["diff_r"][0] == 1
    assert result.scores["level_r"][0] > 0


def test_mse_preserves_tiny_differences_and_representable_extreme_means():
    query = np.array([[1, 2, 4, 3]], dtype=float)
    bank = np.vstack((query[0] + 1e-7, query[0]))
    comparison = compare_series(query, bank, lags=(), metrics=("mse",))
    assert comparison.values["mse"][0, 0] > 0
    assert nearest_series(comparison).indices["mse"][0] == 1
    extreme = compare_series([[1e154] * 4], [[0.0] * 4], lags=(), metrics=("mse",))
    assert extreme.values["mse"][0, 0] == pytest.approx(1e308)
    underflow = compare_series([[1e-200] * 4], [[0.0] * 4], lags=(), metrics=("mse",))
    assert np.isnan(underflow.values["mse"][0, 0])


def test_extreme_differences_retain_signed_correlations():
    values = np.array([[-1e308, 1e308, -1e308, 1e308, -1e308]])
    comparison = compare_series(values, values, lags=(1,))
    assert comparison.values["level_r"][0, 0] == pytest.approx(1)
    assert comparison.values["diff_r"][0, 0] == pytest.approx(1)


def test_partial_profiles_constants_short_series_and_empty_bank_remain_unavailable():
    values = np.array([[0, 1, 4, 2, 3]], dtype=float)
    partial = compare_series(values, values, lags=(1, 4))
    assert np.isnan(partial.values["overlap_pearson_rmse"][0, 0])
    assert partial.values["level_r"][0, 0] == pytest.approx(1)
    constant = nearest_series(compare_series([[3.0] * 5], [[3.0] * 5], lags=(1,)))
    assert constant.scores["mse"][0] == 0
    assert np.isnan(constant.scores["level_r"][0])
    assert constant.indices["level_r"][0] == -1
    one = compare_series([[2.0]], [[2.0]], lags=(1,))
    assert one.values["mse"][0, 0] == 0
    assert np.isnan(one.values["diff_r"][0, 0])
    empty = nearest_series(compare_series(values, np.empty((0, 5)), lags=(1,)))
    for metric in empty.metrics:
        assert np.isnan(empty.scores[metric][0])
        assert empty.indices[metric][0] == -1
        assert empty.candidate_counts[metric][0] == 0


def test_mixed_normalization_contracts_are_rejected():
    raw = np.array([[1, 2, 4, 3, 5]], dtype=float)
    normalized = normalize_series_quantile(raw, quantile=0.5, mode="absolute")
    with pytest.raises(ValueError, match="same quantile"):
        compare_series(raw, normalized, lags=(1,))
    different = normalize_series_quantile(raw, quantile=0.75, mode="absolute")
    with pytest.raises(ValueError, match="same quantile"):
        compare_series(normalized, different, lags=(1,))


def test_real_bank_ranks_tie_bounds_and_metric_specific_reference_counts():
    query = np.array([[1.0] * 5, [2.0] * 5])
    calibration = np.array([[0.0] * 5, [1.0] * 5, [-1.0] * 5, [3.0] * 5])
    bank = np.zeros((1, 5))
    result = assess_series_similarity(
        query,
        calibration,
        bank,
        lags=(1,),
        query_groups=["q0", "q1"],
        calibration_groups=["c0", "c1", "c1", "c2"],
        bank_groups=["b0"],
    )
    # Actual held-out MSEs are [0,1,1,9]; query MSEs are [1,4].
    np.testing.assert_allclose(result.query_matches.scores["mse"], [1, 4])
    np.testing.assert_allclose(result.calibration_matches.scores["mse"], [0, 1, 1, 9])
    np.testing.assert_allclose(result.percentiles["mse"], [0.5, 0.75])
    np.testing.assert_allclose(result.lower["mse"], [0.25, 0.75])
    np.testing.assert_allclose(result.upper["mse"], [0.75, 0.75])
    np.testing.assert_array_equal(result.tie_counts["mse"], [2, 0])
    assert result.finite_reference_counts["mse"] == 4
    assert result.finite_reference_counts["level_r"] == 0
    assert np.isnan(result.percentiles["level_r"]).all()


@pytest.mark.parametrize("overlap", [("q", "c"), ("q", "b"), ("c", "b")])
def test_all_interpool_group_overlaps_are_rejected(overlap):
    groups = {"q": ["query"], "c": ["calibration"], "b": ["bank"]}
    groups[overlap[1]] = groups[overlap[0]]
    values = [[0, 1, 4, 2, 3]]
    with pytest.raises(ValueError, match="group-disjoint"):
        assess_series_similarity(
            values,
            values,
            values,
            lags=(1,),
            query_groups=groups["q"],
            calibration_groups=groups["c"],
            bank_groups=groups["b"],
        )


def test_unknown_groups_require_explicit_opt_out_and_empty_calibration_has_no_rank():
    values = np.array([[0, 1, 4, 2, 3]], dtype=float)
    with pytest.raises(ValueError, match="unknown group"):
        assess_series_similarity(
            values,
            values,
            values,
            lags=(1,),
            query_groups=[-1],
            calibration_groups=[1],
            bank_groups=[2],
        )
    opted_out = assess_series_similarity(
        np.ones((1, 3)),
        [[0.0, 0, 0], [2.0, 2, 2]],
        np.zeros((1, 3)),
        lags=(),
        metrics=("mse",),
        require_disjoint_groups=False,
    )
    np.testing.assert_array_equal(opted_out.query_matches.scores["mse"], [1.0])
    np.testing.assert_array_equal(opted_out.calibration_matches.scores["mse"], [0.0, 4.0])
    assert opted_out.percentiles["mse"][0] == 0.5
    assert opted_out.finite_reference_counts["mse"] == 2
    assert opted_out.group_disjoint is False
    result = assess_series_similarity(
        values,
        np.empty((0, 5)),
        values,
        lags=(1,),
        query_groups=["q"],
        calibration_groups=[],
        bank_groups=["b"],
    )
    assert result.finite_reference_counts["mse"] == 0
    assert np.isnan(result.percentiles["mse"][0])


@pytest.mark.parametrize("operation", ["normalize", "compare", "groups"])
def test_masked_arrays_never_silently_reinclude_values(operation):
    values = np.ma.array([[0.0, 1, 4, 2, 3]], mask=[[False, True, False, False, False]])
    with pytest.raises(TypeError, match="masked"):
        if operation == "normalize":
            normalize_series_quantile(values, quantile=0.5, mode="absolute")
        elif operation == "compare":
            compare_series(values, np.zeros((1, 5)), lags=(1,))
        else:
            assess_series_similarity(
                [[0, 1, 4, 2, 3]],
                [[1, 2, 3, 4, 5]],
                [[0, 2, 1, 4, 3]],
                lags=(1,),
                query_groups=np.ma.array([1], mask=[True]),
                calibration_groups=[2],
                bank_groups=[3],
            )


def test_normalization_rejects_underflow_that_would_invent_inactive_zeros():
    with pytest.raises(ValueError, match="represented"):
        normalize_series_quantile([[1e-200, 0.0, 1e200, 1e200]], quantile=0.75, mode="absolute")


def test_subnormal_mse_keeps_correct_nearest_candidate():
    from decimal import Decimal, localcontext

    bank = np.array([[2.6e-162, 2.6e-162, 3.4e-162], [2.4e-162] * 3])
    with localcontext() as context:
        context.prec = 1500
        expected = np.array(
            [float(sum(Decimal.from_float(float(x)) ** 2 for x in row) / len(row)) for row in bank]
        )
    comparison = compare_series(np.zeros((1, 3)), bank, lags=(), metrics=("mse",))
    np.testing.assert_array_equal(comparison.values["mse"][0], expected)
    assert nearest_series(comparison).indices["mse"][0] == 1


def test_profile_rmse_remains_available_when_its_square_is_unrepresentable():
    query = np.array([[0.0, 0, 1, 0, -1, 0, 0]])
    bank = query.copy()
    bank[0, 5] = 1e-200
    comparison = compare_series(query, bank, lags=(1,), metrics=("overlap_pearson_rmse",))
    # The one-coordinate correlation difference is delta/2 to float64 precision.
    np.testing.assert_allclose(
        comparison.values["overlap_pearson_rmse"][0, 0], 5e-201, rtol=1e-14, atol=0
    )
    matches = nearest_series(comparison)
    assert matches.indices["overlap_pearson_rmse"][0] == 0
    assert matches.candidate_counts["overlap_pearson_rmse"][0] == 1


@pytest.mark.parametrize(
    ("values", "quantile", "mode"),
    [
        ([-1e308, 1e308], np.nextafter(0.5, 1.0), "signed"),
        ([-1e308, 1e308], np.nextafter(0.5, 0.0), "signed"),
        ([2 * np.nextafter(0.0, 1.0), 3 * np.nextafter(0.0, 1.0)], 0.5, "absolute"),
    ],
)
def test_quantile_interpolation_preserves_cancellation_and_subnormal_midpoints(
    values, quantile, mode
):
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 1500
        fraction = Decimal.from_float(float(quantile))
        expected = float(
            (1 - fraction) * Decimal.from_float(float(values[0]))
            + fraction * Decimal.from_float(float(values[1]))
        )
    result = normalize_series_quantile([values], quantile=quantile, mode=mode)
    np.testing.assert_allclose(result.divisors[0], expected, rtol=1e-15, atol=0)


def test_finite_signed_correlation_ranks_use_only_eligible_reference_scores():
    base = np.array([0.0, 1, 0, 2, 1, 3])
    result = assess_series_similarity(
        base[None],
        np.stack((base, -base, np.ones_like(base))),
        np.stack((base, base)),
        lags=(),
        metrics=("level_r",),
        query_groups=["q"],
        calibration_groups=["c0", "c1", "c2"],
        bank_groups=["b", "b"],
    )
    assert result.percentiles["level_r"][0] == 0.75
    assert result.lower["level_r"][0] == 0.5
    assert result.upper["level_r"][0] == 1
    assert result.tie_counts["level_r"][0] == 1
    assert result.finite_reference_counts["level_r"] == 2
    assert result.query_matches.candidate_counts["level_r"][0] == 2
    assert result.query_matches.indices["level_r"][0] == 0


@pytest.mark.parametrize(
    ("metric", "n_time"),
    [
        ("level_r", 3),
        ("diff_r", 4),
        ("level_r", 10007),
        ("diff_r", 10007),
        ("level_r", 10008),
        ("diff_r", 10008),
    ],
)
def test_correlations_are_invariant_to_query_and_reference_batching(metric, n_time):
    rng = np.random.default_rng(2026)
    query = rng.normal(size=(9, n_time))
    reference = rng.normal(size=(31, n_time))
    full = compare_series(query, reference, lags=(), metrics=(metric,)).values[metric]
    query_rows = np.concatenate(
        [
            compare_series(row[None], reference, lags=(), metrics=(metric,)).values[metric]
            for row in query
        ],
        axis=0,
    )
    reference_columns = np.concatenate(
        [
            compare_series(query, row[None], lags=(), metrics=(metric,)).values[metric]
            for row in reference
        ],
        axis=1,
    )
    np.testing.assert_array_equal(full, query_rows)
    np.testing.assert_array_equal(full, reference_columns)
    single_pairs = np.array(
        [
            [
                compare_series(left[None], right[None], lags=(), metrics=(metric,)).values[metric][
                    0, 0
                ]
                for right in reference
            ]
            for left in query
        ]
    )
    np.testing.assert_array_equal(full, single_pairs)


@pytest.mark.parametrize(
    ("n_time", "metric", "expected"),
    [
        (10007, "level_r", 0.25),
        (10007, "diff_r", 0.75),
        (10008, "level_r", 0.25),
        (10008, "diff_r", 0.25),
    ],
)
def test_long_singleton_assessment_retains_identical_calibration_score_ties(
    n_time, metric, expected
):
    base, bank = np.random.default_rng(744).normal(size=(2, n_time))
    result = assess_series_similarity(
        base[None],
        np.vstack((base, -base, np.ones(n_time))),
        bank[None],
        lags=(),
        metrics=(metric,),
        query_groups=["q"],
        calibration_groups=["c0", "c1", "c2"],
        bank_groups=["b"],
    )
    assert result.query_matches.scores[metric][0] == result.calibration_matches.scores[metric][0]
    assert result.percentiles[metric][0] == expected
    assert result.lower[metric][0] == expected - 0.25
    assert result.upper[metric][0] == expected + 0.25
    assert result.tie_counts[metric][0] == 1
    assert result.finite_reference_counts[metric] == 2


def test_quantile_scaling_cannot_promote_source_constants_into_shape_matches():
    unresolved = 0.75 + np.spacing(0.75) * np.array([0, 6, 1, 2, 0])
    values = np.stack((unresolved, [0.5, 1, 0, 2, 1]))
    normalized = normalize_series_quantile(values, quantile=0.75, mode="absolute")
    np.testing.assert_array_equal(normalized.constant, [True, False])
    shape_metrics = ("level_r", "diff_r", "overlap_pearson_rmse")
    comparison = compare_series(normalized, normalized, lags=(1,), metrics=("mse", *shape_metrics))
    assert comparison.values["mse"][0, 0] == 0
    for metric in shape_metrics:
        np.testing.assert_array_equal(
            np.isfinite(comparison.values[metric]), [[False, False], [False, True]]
        )
    assessment = assess_series_similarity(
        normalized,
        normalized,
        normalized,
        lags=(1,),
        metrics=("mse", *shape_metrics),
        query_groups=["q0", "q1"],
        calibration_groups=["c0", "c1"],
        bank_groups=["b0", "b1"],
    )
    assert assessment.finite_reference_counts["mse"] == 2
    np.testing.assert_array_equal(assessment.percentiles["mse"], [0.5, 0.5])
    for metric in shape_metrics:
        assert assessment.finite_reference_counts[metric] == 1
        assert assessment.query_matches.candidate_counts[metric][0] == 0
        assert np.isnan(assessment.percentiles[metric][0])
        assert assessment.percentiles[metric][1] == 0.5


@pytest.mark.parametrize("normalized", [False, True])
def test_caller_mutation_cannot_change_waveforms_that_produced_retained_scores(normalized):
    query = np.array([[0.0, 1, 4, 2, 3]])
    bank = query.copy()
    expected = query.copy()
    if normalized:
        query = normalize_series_quantile(query, quantile=0.5, mode="absolute")
        bank = normalize_series_quantile(bank, quantile=0.5, mode="absolute")
        expected /= 2
    comparison = compare_series(query, bank, lags=(), metrics=("mse",))
    matches = nearest_series(comparison)
    if normalized:
        query.values[:] = -50
        bank.values[:] = 100
        query.scale_available[:] = False
        bank.scale_available[:] = False
    else:
        query[:] = -50
        bank[:] = 100
    np.testing.assert_array_equal(comparison.query, expected)
    np.testing.assert_array_equal(comparison.reference, expected)
    np.testing.assert_allclose(
        np.mean((comparison.query[0] - comparison.reference[0]) ** 2),
        matches.scores["mse"][0],
    )
    assert comparison.query_scale_available[0] and comparison.reference_scale_available[0]


@pytest.mark.parametrize(
    ("pool", "missing"),
    [("query", ("cell", np.datetime64("NaT"))), ("calibration", np.timedelta64("NaT"))],
)
def test_missing_temporal_group_labels_cannot_establish_disjointness(pool, missing):
    groups = {"query": ["q"], "calibration": ["c"], "bank": ["b"]}
    groups[pool] = [missing]
    values = [[0, 1, 4, 2, 3]]
    with pytest.raises(ValueError, match="unknown group"):
        assess_series_similarity(
            values,
            values,
            values,
            lags=(),
            metrics=("mse",),
            query_groups=groups["query"],
            calibration_groups=groups["calibration"],
            bank_groups=groups["bank"],
        )


@pytest.mark.parametrize("case", ["nullable_series", "namespaced_timestamp"])
def test_nullable_missing_group_metadata_never_establishes_disjointness(case):
    import pandas as pd

    query_groups, bank_groups = ["q"], ["b"]
    if case == "nullable_series":
        query_groups = pd.Series([pd.NA], dtype="Int64")
    else:
        bank_groups = [("cell", pd.NaT)]
    values = [[0, 1, 4, 2, 3]]
    with pytest.raises(ValueError, match="unknown group"):
        assess_series_similarity(
            values,
            values,
            values,
            lags=(),
            metrics=("mse",),
            query_groups=query_groups,
            calibration_groups=["c"],
            bank_groups=bank_groups,
        )


def test_known_negative_duration_groups_are_not_missing_integer_sentinels():
    result = assess_series_similarity(
        [[0, 1, 3]],
        [[1, 0, 3]],
        [[0, 2, 3]],
        lags=(),
        metrics=("mse",),
        query_groups=[("duration", np.timedelta64(-1, "D"))],
        calibration_groups=[("duration", np.timedelta64(0, "D"))],
        bank_groups=[("duration", np.timedelta64(1, "D"))],
    )
    assert result.group_disjoint
    assert result.query_matches.scores["mse"][0] == pytest.approx(1 / 3)
    assert result.percentiles["mse"][0] == 0
