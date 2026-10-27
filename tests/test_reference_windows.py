"""Existing-source windows reconstruct exactly from their original lineage."""

from __future__ import annotations

import numpy as np
import pytest

from pymc_generator.dataset_typicality import ReferenceWindows, sample_reference_windows
from pymc_generator.descriptors import ObservedWorlds, world_descriptors


def _corpus():
    world = np.arange(6)[:, None, None]
    time = np.arange(11)[None, :, None]
    treatment = (128 * world + 8 * time + np.arange(4)[None, None, :] + 1).astype(np.float32)
    covariates = (-64 * world + 4 * time + np.arange(3)[None, None, :] + 0.5).astype(np.float16)
    outcome = 32 * np.arange(6)[:, None] + np.arange(11)[None, :] ** 2 + 0.25
    treatment_mask = np.array(
        [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]]
    )
    covariate_mask = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]])
    treatment[~np.broadcast_to(treatment_mask[:, None, :], treatment.shape).astype(bool)] = 0
    covariates[~np.broadcast_to(covariate_mask[:, None, :], covariates.shape).astype(bool)] = 0
    return {
        "treatment_raw": treatment,
        "covariates": covariates,
        "outcome_raw": outcome,
        "treatment_active_mask": treatment_mask,
        "covariate_active_mask": covariate_mask,
        "cell_id": np.array([4, 4, -1, 8, 8, -1]),
    }


def _source(corpus, kind):
    if kind == "corpus":
        return corpus
    return ObservedWorlds(
        treatment_raw=corpus["treatment_raw"],
        covariates=corpus["covariates"],
        outcome_raw=corpus["outcome_raw"],
        treatment_active_mask=corpus["treatment_active_mask"],
        covariate_active_mask=corpus["covariate_active_mask"],
    )


def _sample(source, **kwargs):
    options = {"length": 5, "n_windows": 2, "n_treatments": 1, "n_covariates": 1, "seed": 271}
    options.update(kwargs)
    return sample_reference_windows(source, **options)


def _assert_reconstruction(result, original):
    assert isinstance(result, ReferenceWindows)
    observed = result.observed
    n_windows, length, n_treatments = observed.treatment_raw.shape
    n_covariates = observed.covariates.shape[2]
    assert (
        result.world_positions.shape
        == result.world_ids.shape
        == result.group_ids.shape
        == (n_windows,)
    )
    assert result.starts.shape == (n_windows,)
    assert result.treatment_indices.shape == (n_windows, n_treatments)
    assert result.covariate_indices.shape == (n_windows, n_covariates)
    assert np.unique(result.world_positions).size == n_windows
    for row, (world, start) in enumerate(zip(result.world_positions, result.starts)):
        stop = start + length
        assert 0 <= start < stop <= original["outcome_raw"].shape[1]
        for key, columns, mask in (
            ("treatment_raw", result.treatment_indices[row], "treatment_active_mask"),
            ("covariates", result.covariate_indices[row], "covariate_active_mask"),
        ):
            assert np.unique(columns).size == columns.size
            assert original[mask][world, columns].all()
            expected = original[key][world, start:stop][:, columns]
            np.testing.assert_array_equal(getattr(observed, key)[row], expected)
        np.testing.assert_array_equal(
            observed.outcome_raw[row], original["outcome_raw"][world, start:stop]
        )
    assert observed.treatment_active_mask.all()
    assert observed.covariate_active_mask.all()
    expected_resolution = tuple(
        float(np.finfo(original[key].dtype).eps)
        for key in ("treatment_raw", "covariates", "outcome_raw")
    )
    assert observed.resolution == expected_resolution
    frame = result.to_frame()
    np.testing.assert_array_equal(frame["world_position"], result.world_positions)
    np.testing.assert_array_equal(frame["world_id"], result.world_ids)
    np.testing.assert_array_equal(frame["group_id"], result.group_ids)
    np.testing.assert_array_equal(frame["start"], result.starts)
    np.testing.assert_array_equal(frame["stop"], result.starts + length)
    assert frame["treatment_indices"].tolist() == [
        tuple(int(column) for column in row) for row in result.treatment_indices
    ]
    assert frame["covariate_indices"].tolist() == [
        tuple(int(column) for column in row) for row in result.covariate_indices
    ]


@pytest.mark.parametrize("kind", ["corpus", "observed"])
def test_reordered_pool_intersects_both_role_masks_and_preserves_override_ids(kind):
    original = _corpus()
    worlds = np.array([5, 1, 4, 0, 3])
    treatment_eligible = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    )
    covariate_eligible = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1]])
    world_ids = np.array([71, 17, 46, 8, 32])
    group_ids = np.array([-1, 6, 6, 2, 6])
    treatment_intersection = original["treatment_active_mask"][worlds] & treatment_eligible
    covariate_intersection = original["covariate_active_mask"][worlds] & covariate_eligible
    eligible_worlds = [
        int(world)
        for row, world in enumerate(worlds)
        if treatment_intersection[row].sum() >= 2 and covariate_intersection[row].sum() >= 2
    ]
    result = _sample(
        _source(original, kind),
        n_windows=len(eligible_worlds),
        n_treatments=2,
        n_covariates=2,
        worlds=worlds,
        treatment_eligible=treatment_eligible,
        covariate_eligible=covariate_eligible,
        world_ids=world_ids,
        group_ids=group_ids,
    )
    assert result.n_eligible_worlds == len(eligible_worlds) == 3
    assert sorted(result.world_positions.tolist()) == sorted(eligible_worlds)
    for row, world in enumerate(result.world_positions):
        pool_row = int(np.flatnonzero(worlds == world)[0])
        assert result.world_ids[row] == world_ids[pool_row]
        assert result.group_ids[row] == group_ids[pool_row]
        assert treatment_intersection[pool_row, result.treatment_indices[row]].all()
        assert covariate_intersection[pool_row, result.covariate_indices[row]].all()
    assert sorted(result.group_ids.tolist()) == [-1, 6, 6]
    _assert_reconstruction(result, original)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
def test_seed_repeats_all_lineage_and_every_returned_array(kind):
    original = _corpus()
    first = _sample(_source(original, kind), n_treatments=2, n_covariates=2)
    second = _sample(_source(original, kind), n_treatments=2, n_covariates=2)
    assert first.n_eligible_worlds == second.n_eligible_worlds == 5
    for name in (
        "world_positions",
        "world_ids",
        "group_ids",
        "starts",
        "treatment_indices",
        "covariate_indices",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    for key in ("treatment_raw", "covariates", "outcome_raw"):
        np.testing.assert_array_equal(getattr(first.observed, key), getattr(second.observed, key))
    _assert_reconstruction(first, original)
    _assert_reconstruction(second, original)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
@pytest.mark.parametrize("counts", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_full_horizon_and_zero_width_roles_reconstruct_without_special_cases(kind, counts):
    original = _corpus()
    result = _sample(
        _source(original, kind),
        worlds=[3, 5],
        length=11,
        n_treatments=counts[0],
        n_covariates=counts[1],
    )
    assert result.n_eligible_worlds == 2
    np.testing.assert_array_equal(result.starts, np.zeros(2, dtype=np.int64))
    _assert_reconstruction(result, original)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
@pytest.mark.parametrize(
    "selector, expected",
    [
        (slice(1, 6, 2), [1, 3, 5]),
        (np.array([False, True, False, False, True, False]), [1, 4]),
        (3, [3]),
    ],
)
def test_selector_forms_keep_original_positions_not_pool_row_numbers(kind, selector, expected):
    original = _corpus()
    result = _sample(
        _source(original, kind),
        worlds=selector,
        n_windows=len(expected),
        n_treatments=0,
        n_covariates=0,
    )
    assert result.n_eligible_worlds == len(expected)
    assert sorted(result.world_positions.tolist()) == expected
    np.testing.assert_array_equal(result.world_ids, result.world_positions)
    _assert_reconstruction(result, original)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
def test_default_groups_preserve_repeated_cells_and_unknown_independence(kind):
    original = _corpus()
    result = _sample(_source(original, kind), worlds=[5, 0, 3, 1], n_windows=4)
    expected = (
        original["cell_id"][result.world_positions]
        if kind == "corpus"
        else np.full(4, -1, dtype=np.int64)
    )
    np.testing.assert_array_equal(result.group_ids, expected)
    np.testing.assert_array_equal(result.world_ids, result.world_positions)
    _assert_reconstruction(result, original)
    without_groups = dict(original)
    without_groups.pop("cell_id")
    unknown = _sample(without_groups)
    np.testing.assert_array_equal(unknown.group_ids, [-1, -1])
    _assert_reconstruction(unknown, original)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
def test_changing_values_to_all_zero_crops_never_changes_sampling_or_eligibility(kind):
    original = _corpus()
    before = _sample(_source(original, kind), length=4, n_windows=3, n_treatments=2, n_covariates=2)
    cropped = {key: value.copy() for key, value in original.items()}
    for world, start in zip(before.world_positions, before.starts):
        for key in ("treatment_raw", "covariates", "outcome_raw"):
            cropped[key][world, start : start + 4] = 0
    after = _sample(_source(cropped, kind), length=4, n_windows=3, n_treatments=2, n_covariates=2)
    assert after.n_eligible_worlds == before.n_eligible_worlds
    for name in ("world_positions", "starts", "treatment_indices", "covariate_indices"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    for key in ("treatment_raw", "covariates", "outcome_raw"):
        assert not getattr(after.observed, key).any()
    _assert_reconstruction(before, original)
    _assert_reconstruction(after, cropped)


@pytest.mark.parametrize("kind", ["corpus", "observed"])
def test_float32_rounding_floor_survives_extraction_into_downstream_descriptors(kind):
    eps32 = float(np.finfo(np.float32).eps)
    time = np.arange(12)
    nearly_flat = (1.0 + eps32 * (time % 2)).astype(np.float32)
    original = {
        "treatment_raw": np.stack([nearly_flat, nearly_flat[::-1]], axis=1)[None],
        "covariates": (time.astype(np.float64) ** 2)[None, :, None],
        "outcome_raw": (time + 2).astype(np.float16)[None],
        "treatment_active_mask": np.ones((1, 2), dtype=bool),
        "covariate_active_mask": np.ones((1, 1), dtype=bool),
    }
    assert np.unique(nearly_flat).size == 2
    result = _sample(_source(original, kind), length=12, n_windows=1, n_treatments=2)
    _assert_reconstruction(result, original)
    descriptors = world_descriptors(
        result.observed,
        source_id="projection",
        statistics=("constant_fraction", "acf", "abs_pearson"),
        lags=(1,),
        reducers=("max",),
    )
    assert descriptors.column("treatment_constant_fraction")[0] == 1.0
    assert descriptors.column_status("treatment_acf_lag_1_max")[0] == "undefined"
    assert descriptors.column_status("treatment_pair_abs_pearson_max")[0] == "undefined"
    relabeled = ObservedWorlds(
        treatment_raw=result.observed.treatment_raw,
        covariates=result.observed.covariates,
        outcome_raw=result.observed.outcome_raw,
    )
    fine_resolution = world_descriptors(
        relabeled, source_id="fine", statistics=("constant_fraction",)
    )
    assert fine_resolution.column("treatment_constant_fraction")[0] == 0.0


def test_absent_observed_outcome_and_covariates_stay_absent():
    treatment = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    source = ObservedWorlds(treatment_raw=treatment)
    result = _sample(source, length=3, n_covariates=0)
    assert result.observed.outcome_raw is None
    assert result.observed.covariates.shape == (2, 3, 0)
    assert result.observed.resolution == source.resolution
    for row, (world, start) in enumerate(zip(result.world_positions, result.starts)):
        expected = treatment[world, start : start + 3][:, result.treatment_indices[row]]
        np.testing.assert_array_equal(result.observed.treatment_raw[row], expected)


@pytest.mark.parametrize("name", ["length", "n_windows"])
@pytest.mark.parametrize("value", [0, -1])
def test_lengths_and_window_counts_must_be_positive(name, value):
    with pytest.raises(ValueError, match=f"{name}.*positive integer"):
        _sample(_corpus(), **{name: value})


@pytest.mark.parametrize("name", ["length", "n_windows", "n_treatments", "n_covariates"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5])
def test_counts_reject_booleans_and_nonintegers(name, value):
    with pytest.raises(TypeError, match=f"{name}.*integer"):
        _sample(_corpus(), **{name: value})


@pytest.mark.parametrize("name", ["n_treatments", "n_covariates"])
def test_role_counts_reject_negative_values(name):
    with pytest.raises(ValueError, match=f"{name}.*non-negative"):
        _sample(_corpus(), **{name: -1})


def test_numpy_integer_counts_are_supported():
    original = _corpus()
    result = _sample(
        original,
        length=np.int64(11),
        n_windows=np.int32(1),
        n_treatments=np.int64(0),
        n_covariates=np.int32(0),
        seed=np.int64(41),
    )
    _assert_reconstruction(result, original)


@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5])
def test_seed_must_be_an_integer(value):
    with pytest.raises(TypeError, match="seed.*integer"):
        _sample(_corpus(), seed=value)


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="seed.*non-negative"):
        _sample(_corpus(), seed=-1)


def test_short_horizon_empty_pool_and_insufficient_candidates_are_clear_errors():
    original = _corpus()
    with pytest.raises(ValueError, match="length.*exceeds.*horizon"):
        _sample(original, length=12)
    with pytest.raises(ValueError, match="selection is empty"):
        _sample(original, worlds=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="0 eligible source worlds"):
        _sample(original, n_treatments=5)
    with pytest.raises(ValueError, match="1 eligible source worlds"):
        _sample(original, worlds=[2], n_windows=2)
    with pytest.raises(ValueError, match="0 eligible source worlds"):
        _sample(original, treatment_eligible=np.zeros((6, 4), dtype=bool))
    empty_time = dict(original)
    for key in ("treatment_raw", "covariates", "outcome_raw"):
        empty_time[key] = original[key][:, :0]
    with pytest.raises(ValueError, match="no time steps"):
        _sample(empty_time)
    empty_worlds = {key: value[:0] for key, value in original.items()}
    with pytest.raises(ValueError, match="no worlds"):
        _sample(empty_worlds)


@pytest.mark.parametrize("selector", [[1, 1], [-1], [6], np.ma.array([0, 1], mask=False)])
def test_invalid_selectors_keep_existing_strict_selector_semantics(selector):
    with pytest.raises((TypeError, ValueError, IndexError)):
        _sample(_corpus(), worlds=selector)


@pytest.mark.parametrize("name, width", [("treatment_eligible", 4), ("covariate_eligible", 3)])
def test_caller_masks_use_selected_pool_shape_and_require_binary_values(name, width):
    with pytest.raises(ValueError, match="shape"):
        _sample(_corpus(), worlds=[4, 0], **{name: np.ones((6, width))})
    for invalid in (2.0, np.nan):
        with pytest.raises(ValueError, match="binary 0/1"):
            _sample(_corpus(), **{name: np.full((6, width), invalid)})
    with pytest.raises(TypeError, match=f"{name}.*masked"):
        _sample(_corpus(), **{name: np.ma.array(np.ones((6, width)), mask=False)})


@pytest.mark.parametrize(
    "key",
    [
        "treatment_raw",
        "covariates",
        "outcome_raw",
        "treatment_active_mask",
        "covariate_active_mask",
    ],
)
def test_masked_source_values_and_masks_are_rejected_before_conversion(key):
    original = _corpus()
    original[key] = np.ma.array(original[key], mask=False)
    with pytest.raises(TypeError, match=f"{key}.*masked"):
        _sample(original)


@pytest.mark.parametrize("field", ["world_ids", "group_ids", "cell_id"])
@pytest.mark.parametrize("problem", ["masked", "overflow", "floating", "shape", "below_minimum"])
def test_default_and_override_ids_validate_before_int64_cast(field, problem):
    original = _corpus()
    n = 6 if field == "cell_id" else 3
    values = np.arange(n, dtype=np.int64)
    error, match = ValueError, "int64"
    if problem == "masked":
        values = np.ma.array(values, mask=False)
        error, match = TypeError, "masked"
    elif problem == "overflow":
        values = np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64)
    elif problem == "floating":
        values = values.astype(np.float64)
    elif problem == "shape":
        values = values[:-1]
    else:
        values[0] = -1 if field == "world_ids" else -2
        match = ">="
    options = {"worlds": [4, 0, 5], "n_windows": 1}
    if field == "cell_id":
        original[field] = values
    else:
        options[field] = values
    with pytest.raises(error, match=match):
        _sample(original, **options)


@pytest.mark.parametrize("field", ["world_ids", "group_ids", "cell_id"])
def test_largest_int64_ids_are_retained_when_supplied_as_unsigned(field):
    original = _corpus()
    worlds = np.array([4, 0, 5])
    n = 6 if field == "cell_id" else len(worlds)
    values = np.array([np.iinfo(np.int64).max - offset for offset in range(n)], dtype=np.uint64)
    options = {"worlds": worlds, "n_windows": 3}
    if field == "cell_id":
        original[field] = values
    else:
        options[field] = values
    result = _sample(original, **options)
    actual = result.world_ids if field == "world_ids" else result.group_ids
    assert actual.dtype == np.dtype(np.int64)
    for row, world in enumerate(result.world_positions):
        position = world if field == "cell_id" else int(np.flatnonzero(worlds == world)[0])
        assert int(actual[row]) == int(values[position])
    _assert_reconstruction(result, original)


def test_invalid_default_cells_are_not_hidden_by_selection_or_group_overrides():
    original = _corpus()
    original["cell_id"] = np.array([0, 1, 2, 3, 4, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="cell_id.*int64"):
        _sample(original, worlds=[0], n_windows=1, group_ids=[12])


@pytest.mark.parametrize("key", ["treatment_raw", "covariates", "outcome_raw"])
def test_nonfinite_selected_source_values_are_rejected(key):
    original = _corpus()
    original[key][3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _sample(original, worlds=[3], n_windows=1, length=11)


@pytest.mark.parametrize("key, column", [("treatment_raw", 2), ("covariates", 1)])
def test_invalid_padding_is_rejected_even_when_the_role_is_not_requested(key, column):
    original = _corpus()
    original[key][0, 0, column] = 1
    with pytest.raises(ValueError, match="non-zero inactive padding.*world 0"):
        _sample(original, worlds=[0], n_windows=1, n_treatments=0, n_covariates=0)


def test_invalid_source_types_and_missing_observable_keys_are_rejected():
    with pytest.raises(TypeError, match="corpus mapping or ObservedWorlds"):
        _sample([])
    original = _corpus()
    original.pop("outcome_raw")
    with pytest.raises(KeyError, match="outcome_raw"):
        _sample(original)
