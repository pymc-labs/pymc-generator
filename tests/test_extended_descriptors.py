"""Opt-in descriptor families with independent literal and NumPy oracles."""

from __future__ import annotations

import numpy as np
import pytest

from pymc_generator.descriptors import (
    DEFAULT_STATISTICS,
    DESCRIPTOR_VERSION,
    STATISTICS,
    ObservedWorlds,
    WorldDescriptors,
    world_descriptors,
)
from pymc_generator.diagnostics import ACTIVITY_STATISTICS

_TEMPORAL = (
    "level_std",
    "difference_std",
    "time_correlation",
    "overlap_pearson",
    "diff_overlap_pearson",
)
_LAGGED = ("lagged_pearson", "diff_lagged_pearson")
_REDUCERS = {"min": np.min, "median": np.median, "max": np.max}


def _describe(source, **kwargs) -> WorldDescriptors:
    return world_descriptors(source, source_id="synthetic", **kwargs)


def _value(result, name, row=0):
    assert result.column_status(name)[row] == "valid", name
    return float(result.column(name)[row])


def _pearson(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def _temporal_value(values, statistic, lag):
    if statistic == "level_std":
        return float(np.std(values, ddof=0))
    if statistic == "difference_std":
        return float(np.std(np.diff(values), ddof=0))
    if statistic == "time_correlation":
        return _pearson(values, np.arange(values.size))
    view = np.diff(values) if statistic == "diff_overlap_pearson" else values
    return _pearson(view[:-lag], view[lag:])


def _lagged_value(left, right, lag, *, differences):
    if differences:
        left, right = np.diff(left), np.diff(right)
    if lag > 0:
        left, right = left[:-lag], right[lag:]
    elif lag < 0:
        left, right = left[-lag:], right[:lag]
    return _pearson(left, right)


def _ordinary_world():
    treatments = np.array(
        [
            [0, 3, 1, 7, 2, -1, 4, 8, -2, 5],
            [2, -2, 4, 0, 6, 1, -3, 7, 5, 3],
            [-1, 4, 2, -2, 5, 0, 8, 3, 6, -3],
        ],
        dtype=float,
    ).T
    covariates = np.array(
        [[4, 0, 3, -2, 1, 7, -1, 5, 2, 6], [-2, 5, 0, 4, -1, 2, 7, 1, 6, 3]],
        dtype=float,
    ).T
    outcome = np.array([5, 1, -2, 6, 0, 3, -1, 8, 4, 2], dtype=float)
    return ObservedWorlds(treatments[None], covariates[None], outcome[None])


def test_default_selection_and_legacy_definitions_are_unchanged():
    old_statistics = (
        "level_to_variation",
        "cv",
        "zero_fraction",
        "roughness",
        "spike",
        "acf",
        "diff_acf",
        "constant_fraction",
        "abs_pearson",
        "diff_abs_pearson",
    )
    assert DEFAULT_STATISTICS == old_statistics
    assert DESCRIPTOR_VERSION == 1
    source = _ordinary_world()
    default = _describe(source)
    explicit = _describe(source, statistics=old_statistics)
    assert default.definitions == explicit.definitions
    np.testing.assert_array_equal(default.values, explicit.values)
    np.testing.assert_array_equal(default.status, explicit.status)
    assert len(default.features) == 75
    assert {definition.statistic for definition in default.definitions} == set(old_statistics)
    assert all(definition.version == 1 for definition in default.definitions)
    selected = _describe(source, statistics=STATISTICS, dependence_lags=(-2, 0, 2))
    assert {definition.statistic for definition in selected.definitions} == set(STATISTICS)
    assert set(_TEMPORAL + _LAGGED + ACTIVITY_STATISTICS).isdisjoint(old_statistics)


def test_temporal_values_reducers_and_definitions_match_numpy():
    source = _ordinary_world()
    result = _describe(source, statistics=_TEMPORAL, lags=(3, 1), reducers=("max", "min", "median"))
    roles = {
        "treatment": source.treatment_raw[0].T,
        "covariate": source.covariates[0].T,
        "outcome": source.outcome_raw[0, None],
    }
    for role, series in roles.items():
        for statistic in _TEMPORAL:
            lags = (1, 3) if statistic in ("overlap_pearson", "diff_overlap_pearson") else (None,)
            for lag in lags:
                token = statistic if lag is None else f"{statistic}_lag_{lag}"
                expected = [_temporal_value(values, statistic, lag) for values in series]
                reducers = (None,) if role == "outcome" else tuple(_REDUCERS)
                for reducer in reducers:
                    name = f"{role}_{token}" + ("" if reducer is None else f"_{reducer}")
                    value = expected[0] if reducer is None else _REDUCERS[reducer](expected)
                    assert _value(result, name) == pytest.approx(value, abs=1e-14)
                    definition = result.definitions[result.feature_index(name)]
                    assert definition.statistic == statistic
                    assert definition.view == (
                        "differences"
                        if statistic in ("difference_std", "diff_overlap_pearson")
                        else "levels"
                    )
                    assert definition.lag == lag
                    assert definition.reducer == reducer
                    assert definition.observable and definition.version == 1
                    if statistic not in ("level_std", "difference_std"):
                        assert -1.0 <= _value(result, name) <= 1.0
    assert result.metadata["n_treatments_active"].tolist() == [3]
    assert result.metadata["n_covariates_active"].tolist() == [2]


@pytest.mark.parametrize("factor", [2.0**-700, 2.0**650, -7.0])
def test_amplitude_descriptors_remain_in_original_units(factor):
    values = np.array([0, 2, 8, -1, 5, 3, -3, 4], dtype=float)
    result = _describe(
        ObservedWorlds((factor * values)[None, :, None]),
        statistics=("level_std", "difference_std"),
        reducers=("median",),
    )
    for statistic, view in (("level_std", values), ("difference_std", np.diff(values))):
        expected = abs(factor) * np.std(view, ddof=0)
        actual = _value(result, f"treatment_{statistic}_median")
        assert actual > 0.0 and np.isfinite(actual)
        assert actual == pytest.approx(expected, rel=4e-15, abs=0.0)


def test_representable_extreme_amplitudes_do_not_overflow_intermediate_moments():
    scale = 2.0**1022
    unit_values = np.array([1.0, -1.0, 1.0, -1.0])
    result = _describe(
        ObservedWorlds((scale * unit_values)[None, :, None]),
        statistics=("level_std", "difference_std"),
        reducers=("median",),
    )
    for statistic, unit_view in (
        ("level_std", unit_values),
        ("difference_std", np.diff(unit_values)),
    ):
        actual = _value(result, f"treatment_{statistic}_median")
        assert np.isfinite(actual)
        assert actual == pytest.approx(scale * np.std(unit_view, ddof=0), rel=4e-15)


def test_temporal_precision_floor_uses_each_original_role_dtype():
    one = np.float32(1.0)
    next_value = np.nextafter(one, np.float32(2.0))
    values = np.array([one, next_value, one, next_value, next_value, one, next_value, one])
    source = ObservedWorlds(values[None, :, None], values.astype(float)[None, :, None])
    result = _describe(source, statistics=_TEMPORAL, reducers=("median",))
    for statistic in ("level_std", "difference_std"):
        assert _value(result, f"treatment_{statistic}_median") == 0.0
        assert _value(result, f"covariate_{statistic}_median") > 0.0
    for statistic in ("time_correlation", "overlap_pearson_lag_1", "diff_overlap_pearson_lag_1"):
        assert result.column_status(f"treatment_{statistic}_median")[0] == "undefined"
        assert result.column_status(f"covariate_{statistic}_median")[0] == "valid"


def test_new_node_masks_distinguish_ineligible_undefined_and_valid_reduction():
    treatment = np.zeros((3, 6, 2))
    treatment[1, :, 0] = 4.0
    treatment[2, :, 0] = 4.0
    treatment[2, :, 1] = [0, 3, 1, 5, 2, 4]
    source = ObservedWorlds(treatment, treatment_active_mask=[[0, 0], [1, 0], [1, 1]])
    result = _describe(source, statistics=_TEMPORAL + ACTIVITY_STATISTICS, lags=(1, 99))
    assert result.column_status("treatment_level_std_median").tolist() == [
        "ineligible",
        "valid",
        "valid",
    ]
    assert result.column_status("treatment_time_correlation_median").tolist() == [
        "ineligible",
        "undefined",
        "valid",
    ]
    for definition in result.definitions:
        if definition.role in ("covariate", "outcome"):
            assert result.column_status(definition.name).tolist() == ["ineligible"] * 3
        if definition.lag == 99 and definition.role == "treatment":
            assert result.column_status(definition.name).tolist() == [
                "ineligible",
                "undefined",
                "undefined",
            ]
    variable = treatment[2, :, 1]
    assert _value(result, "treatment_time_correlation_median", 2) == pytest.approx(
        _pearson(variable, np.arange(variable.size))
    )
    assert _value(result, "treatment_level_std_median", 2) == pytest.approx(np.std(variable) / 2)
    assert result.column_status("treatment_complete_inactive_length_median_median").tolist() == [
        "ineligible",
        "undefined",
        "undefined",
    ]


@pytest.mark.parametrize("length", [1, 2])
def test_short_windows_keep_amplitude_and_activity_availability_explicit(length):
    source = ObservedWorlds(np.arange(1.0, length + 1)[None, :, None])
    result = _describe(source, statistics=_TEMPORAL + ACTIVITY_STATISTICS, reducers=("median",))
    assert _value(result, "treatment_level_std_median") == pytest.approx(
        np.std(np.arange(1.0, length + 1), ddof=0)
    )
    assert result.column_status("treatment_difference_std_median")[0] == (
        "undefined" if length == 1 else "valid"
    )
    if length == 2:
        assert _value(result, "treatment_difference_std_median") == 0.0
    for statistic in ("time_correlation", "overlap_pearson_lag_1", "diff_overlap_pearson_lag_1"):
        assert result.column_status(f"treatment_{statistic}_median")[0] == "undefined"
    assert _value(result, "treatment_active_spell_count_median") == 1
    assert _value(result, "treatment_censored_active_spell_count_median") == 1
    assert result.column_status("treatment_complete_active_length_median_median")[0] == "undefined"
    assert result.column_status("treatment_switching_rate_median")[0] == (
        "undefined" if length == 1 else "valid"
    )


def test_activity_summaries_use_exact_spells_and_full_window_denominator():
    first = np.array([0, 0, 2, 3, 0, 4, 4, 0, 0, 0, 5, 0], dtype=float)
    second = np.array([8, 8, 0, 0, 0, 6, 0, 7, 7, 0, 0, 9], dtype=float)
    expected = {
        "leading_inactive_length": [2, 0],
        "trailing_inactive_length": [1, 0],
        "longest_active_length": [2, 2],
        "longest_inactive_length": [3, 3],
        "active_spell_count": [3, 4],
        "inactive_spell_count": [4, 3],
        "complete_active_spell_count": [3, 2],
        "complete_inactive_spell_count": [2, 3],
        "censored_active_spell_count": [0, 2],
        "censored_inactive_spell_count": [2, 0],
        "complete_active_length_median": [2, 1.5],
        "complete_inactive_length_median": [2, 2],
        "switching_rate": [6 / 11, 6 / 11],
        "interior_inactive_fraction": [4 / 12, 6 / 12],
    }
    assert ACTIVITY_STATISTICS == tuple(expected)
    source = ObservedWorlds(np.column_stack((first, second))[None], outcome_raw=first[None])
    result = _describe(source, statistics=ACTIVITY_STATISTICS)
    for statistic, values in expected.items():
        assert _value(result, f"outcome_{statistic}") == pytest.approx(values[0])
        for reducer, reduction in _REDUCERS.items():
            name = f"treatment_{statistic}_{reducer}"
            assert _value(result, name) == pytest.approx(reduction(values))
            definition = result.definitions[result.feature_index(name)]
            assert definition.view == "levels" and definition.lag is None
            assert definition.statistic == statistic and definition.reducer == reducer


def test_all_zero_and_all_active_windows_are_single_censored_spells():
    treatment = np.array([[0.0] * 6, [-(2.0**-700)] * 6])[:, :, None]
    result = _describe(
        ObservedWorlds(treatment), statistics=ACTIVITY_STATISTICS, reducers=("median",)
    )
    expected = {
        "leading_inactive_length": [6, 0],
        "trailing_inactive_length": [6, 0],
        "longest_active_length": [0, 6],
        "longest_inactive_length": [6, 0],
        "active_spell_count": [0, 1],
        "inactive_spell_count": [1, 0],
        "complete_active_spell_count": [0, 0],
        "complete_inactive_spell_count": [0, 0],
        "censored_active_spell_count": [0, 1],
        "censored_inactive_spell_count": [1, 0],
        "switching_rate": [0, 0],
        "interior_inactive_fraction": [0, 0],
    }
    for statistic, values in expected.items():
        np.testing.assert_array_equal(result.column(f"treatment_{statistic}_median"), values)
        assert result.column_status(f"treatment_{statistic}_median").tolist() == ["valid"] * 2
    for statistic in ("complete_active_length_median", "complete_inactive_length_median"):
        assert result.column_status(f"treatment_{statistic}_median").tolist() == ["undefined"] * 2
        assert np.isnan(result.column(f"treatment_{statistic}_median")).all()


def test_tiny_nonzero_activity_survives_legacy_internal_rescaling():
    tiny = np.nextafter(0.0, 1.0)
    huge = 2.0**900
    values = np.array([tiny, 0, huge, -tiny, 0, tiny, 0, huge])
    source = ObservedWorlds(values[None, :, None])
    expected = {
        "leading_inactive_length": 0,
        "trailing_inactive_length": 0,
        "longest_active_length": 2,
        "longest_inactive_length": 1,
        "active_spell_count": 4,
        "inactive_spell_count": 3,
        "complete_active_spell_count": 2,
        "complete_inactive_spell_count": 3,
        "censored_active_spell_count": 2,
        "censored_inactive_spell_count": 0,
        "complete_active_length_median": 1.5,
        "complete_inactive_length_median": 1,
        "switching_rate": 6 / 7,
        "interior_inactive_fraction": 3 / 8,
    }
    for statistics in (ACTIVITY_STATISTICS, ("zero_fraction",) + ACTIVITY_STATISTICS):
        result = _describe(source, statistics=statistics, reducers=("median",))
        for statistic, value in expected.items():
            assert _value(result, f"treatment_{statistic}_median") == pytest.approx(value)


def test_lagged_pair_roles_signed_values_and_reducers_match_numpy():
    source = _ordinary_world()
    result = _describe(source, statistics=_LAGGED, dependence_lags=(2, 0, -2))
    treatment, covariates, outcome = (
        source.treatment_raw[0].T,
        source.covariates[0].T,
        source.outcome_raw[0],
    )
    for statistic in _LAGGED:
        for lag in (-2, 0, 2):
            pairs = {
                "treatment_pair": [
                    (treatment[i], treatment[j])
                    for i in range(3)
                    for j in range(3)
                    if i != j and (lag != 0 or i < j)
                ],
                "covariate_pair": [
                    (covariates[i], covariates[j])
                    for i in range(2)
                    for j in range(2)
                    if i != j and (lag != 0 or i < j)
                ],
                "treatment_covariate": [
                    (left, right) for left in treatment for right in covariates
                ],
                "treatment_outcome": [(left, outcome) for left in treatment],
            }
            for role, role_pairs in pairs.items():
                values = [
                    _lagged_value(left, right, lag, differences=statistic == "diff_lagged_pearson")
                    for left, right in role_pairs
                ]
                for reducer, reduction in _REDUCERS.items():
                    name = f"{role}_{statistic}_lag_{lag}_{reducer}"
                    assert _value(result, name) == pytest.approx(reduction(values), abs=1e-14)
                    assert -1 <= _value(result, name) <= 1
                    definition = result.definitions[result.feature_index(name)]
                    assert definition.statistic == statistic and definition.lag == lag
                    assert definition.reducer == reducer
                    assert definition.view == (
                        "differences" if statistic == "diff_lagged_pearson" else "levels"
                    )
                    assert definition.observable and definition.version == 1


def test_positive_lag_orientation_is_left_then_right_and_correlations_stay_signed():
    left = np.array([0, 2, -1, 5, 3, -2, 4, 1], dtype=float)
    right = np.array([-5, 0, 2, -1, 5, 3, -2, 4], dtype=float)
    source = ObservedWorlds(left[None, :, None], right[None, :, None], -right[None])
    result = _describe(source, statistics=_LAGGED, dependence_lags=(-1, 1), reducers=("median",))
    for statistic in _LAGGED:
        assert _value(result, f"treatment_covariate_{statistic}_lag_1_median") == pytest.approx(1)
        assert _value(result, f"treatment_outcome_{statistic}_lag_1_median") == pytest.approx(-1)
        expected = _lagged_value(left, right, -1, differences=statistic == "diff_lagged_pearson")
        assert _value(result, f"treatment_covariate_{statistic}_lag_-1_median") == pytest.approx(
            expected
        )


def test_nonzero_same_role_lags_are_invariant_to_column_permutations():
    source = _ordinary_world()
    permuted = ObservedWorlds(
        source.treatment_raw[:, :, [2, 0, 1]], source.covariates[:, :, [1, 0]], source.outcome_raw
    )
    options = {
        "statistics": _LAGGED + ("abs_pearson", "diff_abs_pearson"),
        "dependence_lags": (-2, 0, 2),
    }
    first, second = _describe(source, **options), _describe(permuted, **options)
    assert first.definitions == second.definitions
    np.testing.assert_array_equal(first.status, second.status)
    np.testing.assert_allclose(first.values, second.values, atol=1e-14, rtol=1e-14)


def test_lag_zero_signed_pairs_do_not_change_legacy_absolute_pearson():
    values = np.array([1, 4, 0, 3, -2, 5], dtype=float)
    source = ObservedWorlds(np.column_stack((values, -values))[None])
    result = _describe(
        source, statistics=_LAGGED + ("abs_pearson", "diff_abs_pearson"), dependence_lags=(0,)
    )
    for reducer in _REDUCERS:
        assert _value(result, f"treatment_pair_abs_pearson_{reducer}") == pytest.approx(1)
        assert _value(result, f"treatment_pair_diff_abs_pearson_{reducer}") == pytest.approx(1)
        for statistic in _LAGGED:
            assert _value(result, f"treatment_pair_{statistic}_lag_0_{reducer}") == pytest.approx(
                -1
            )


def test_lagged_masks_and_short_overlaps_preserve_status_distinctions():
    treatment = np.zeros((3, 6, 2))
    treatment[0, :, 0] = [1, 4, 0, 2, 5, 3]
    treatment[1, :, 0] = 4
    treatment[1, :, 1] = -2
    treatment[2, :, 0] = [1, 4, 0, 2, 5, 3]
    treatment[2, :, 1] = [2, 0, 5, 1, 3, 4]
    source = ObservedWorlds(treatment, treatment_active_mask=[[1, 0], [1, 1], [1, 1]])
    result = _describe(source, statistics=_LAGGED, dependence_lags=(-99, -3, 0, 3, 99))
    for statistic in _LAGGED:
        for lag in (-99, 99):
            for reducer in _REDUCERS:
                name = f"treatment_pair_{statistic}_lag_{lag}_{reducer}"
                assert result.column_status(name).tolist() == [
                    "ineligible",
                    "undefined",
                    "undefined",
                ]
        assert result.column_status(f"treatment_pair_{statistic}_lag_0_median").tolist() == [
            "ineligible",
            "undefined",
            "valid",
        ]
    for lag in (-3, 3):
        assert result.column_status(f"treatment_pair_lagged_pearson_lag_{lag}_median")[2] == "valid"
        assert (
            result.column_status(f"treatment_pair_diff_lagged_pearson_lag_{lag}_median")[2]
            == "undefined"
        )
    for definition in result.definitions:
        if definition.role != "treatment_pair":
            assert result.column_status(definition.name).tolist() == ["ineligible"] * 3


def test_new_descriptors_do_not_form_pairs_between_worlds():
    values = np.array(
        [[[0, 0], [1, 0], [3, 0], [2, 0], [5, 0]], [[0, 2], [0, 5], [0, 1], [0, 4], [0, 3]]],
        dtype=float,
    )
    source = ObservedWorlds(values, treatment_active_mask=[[1, 0], [0, 1]])
    result = _describe(source, statistics=_LAGGED, dependence_lags=(-1, 0, 1))
    assert (result.status == 1).all()
    assert np.isnan(result.values).all()


def test_empty_roles_keep_all_new_features_ineligible():
    source = ObservedWorlds(np.zeros((1, 5, 0)))
    result = _describe(
        source, statistics=_TEMPORAL + ACTIVITY_STATISTICS + _LAGGED, dependence_lags=(0, 1)
    )
    assert (result.status == 1).all()
    assert np.isnan(result.values).all()


def test_legacy_sample_acf_is_distinct_from_overlap_centered_correlation():
    levels = np.arange(6.0)
    result = _describe(
        ObservedWorlds(levels[None, :, None]),
        statistics=("acf", "diff_acf", "overlap_pearson", "diff_overlap_pearson"),
        reducers=("median",),
    )
    # Full-series centering and denominator give 8.75 / 17.5, whereas the
    # two separately centered, linearly related overlaps have Pearson r = 1.
    assert _value(result, "treatment_acf_lag_1_median") == pytest.approx(0.5)
    assert _value(result, "treatment_overlap_pearson_lag_1_median") == pytest.approx(1)
    for statistic in ("diff_acf", "diff_overlap_pearson"):
        assert result.column_status(f"treatment_{statistic}_lag_1_median")[0] == "undefined"


@pytest.mark.parametrize("dependence_lags", [None, ()])
def test_signed_pair_families_require_explicit_nonempty_lags(dependence_lags):
    with pytest.raises(ValueError, match="dependence_lags"):
        _describe(
            _ordinary_world(), statistics=("lagged_pearson",), dependence_lags=dependence_lags
        )


@pytest.mark.parametrize(
    ("lags", "error"),
    [
        ((1, 1), ValueError),
        ((True,), TypeError),
        ((np.bool_(False),), TypeError),
        ((1.0,), TypeError),
        ("1", TypeError),
        (np.ma.array([1], mask=[False]), TypeError),
    ],
)
def test_signed_pair_lags_reject_ambiguous_selectors(lags, error):
    with pytest.raises(error):
        _describe(_ordinary_world(), statistics=("lagged_pearson",), dependence_lags=lags)


@pytest.mark.parametrize("lags", [(0,), (-1,), (True,), (1.0,)])
def test_signed_pair_selection_never_relaxes_legacy_positive_lag_validation(lags):
    with pytest.raises((TypeError, ValueError)):
        _describe(
            _ordinary_world(), statistics=("lagged_pearson",), lags=lags, dependence_lags=(-1, 0, 1)
        )


def test_new_feature_definitions_and_values_are_canonical_across_chunking_and_selector_order():
    ordinary = _ordinary_world()
    source = ObservedWorlds(
        np.concatenate((ordinary.treatment_raw, ordinary.treatment_raw * 3)),
        np.concatenate((ordinary.covariates, ordinary.covariates * 2)),
        np.concatenate((ordinary.outcome_raw, ordinary.outcome_raw * -2)),
    )
    statistics = _TEMPORAL + ACTIVITY_STATISTICS + _LAGGED
    first = _describe(
        source, statistics=statistics, lags=(2, 1), dependence_lags=(2, 0, -1), chunk_size=1
    )
    second = _describe(
        source,
        statistics=tuple(reversed(statistics)),
        lags=(1, 2),
        dependence_lags=(-1, 0, 2),
        chunk_size=2,
    )
    assert first.definitions == second.definitions
    np.testing.assert_array_equal(first.status, second.status)
    np.testing.assert_allclose(first.values, second.values, atol=1e-14, rtol=1e-14)


def test_finite_original_unit_variability_survives_even_node_median_reduction():
    values = np.array([[-1e308, -1e308], [1e308, 1e308], [-1e308, -1e308], [1e308, 1e308]])[None]
    result = _describe(ObservedWorlds(values), statistics=("level_std",))
    for reducer in ("median", "min", "max"):
        name = f"treatment_level_std_{reducer}"
        np.testing.assert_allclose(result.column(name), [1e308], rtol=1e-15, atol=0)
        assert result.column_status(name)[0] == "valid"
