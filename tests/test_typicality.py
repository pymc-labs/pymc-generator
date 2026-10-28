"""Dataset typicality: split-conformal p-values and profiles against a reference.

Every expectation is an INDEPENDENT oracle — a hand count, a literal, or the
documented split rule re-derived here — never the helper under test. Tables are
literal (``_table``) or descriptors of synthetic observed worlds
(``_described``); only the ``slow`` end-to-end test generates worlds.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, replace

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator.dataset_typicality import (
    FEATURE_ROLES,
    TYPICALITY_STATUSES,
    Typicality,
    matched_prior,
    typicality,
)
from pymc_generator.descriptors import (
    INELIGIBLE,
    METADATA_COLUMNS,
    TRUTH_FEATURES,
    UNDEFINED,
    FeatureDefinition,
    ObservedWorlds,
    WorldDescriptors,
    world_descriptors,
)

U, INE = "undefined", "ineligible"  # literal cells for the two non-valid states
INF = math.inf
SHAPE = (2, 2, 40)  # (n_treatments_active, n_covariates_active, n_time_steps) of literal rows

#: Every per-row and per-(row, feature) array of a result.
_ARRAYS = tuple(
    f.name
    for f in fields(Typicality)
    if f.name not in ("features", "k", "calibration_fraction", "central", "seed")
)


def _definition(name: str, *, observable: bool = True) -> FeatureDefinition:
    # A statistic of its own per column: literal columns never repeat a reducer.
    return FeatureDefinition(
        name=name,
        role="covariate",
        statistic=name,
        view="levels",
        lag=None,
        reducer="median",
        observable=observable,
        description=f"test column {name}",
    )


def _table(
    columns: dict[str, list],
    *,
    source: str = "ref",
    world_ids=None,
    group_ids=None,
    shapes: list[tuple[int, int, int]] | None = None,
    hidden: tuple[str, ...] = (),
) -> WorldDescriptors:
    """Literal cells (numbers valid, U / INE the other states); one group per row by default."""
    names = list(columns)
    n = len(columns[names[0]])
    values = np.full((n, len(names)), np.nan)
    status = np.zeros((n, len(names)), dtype=np.uint8)
    for j, name in enumerate(names):
        for i, cell in enumerate(columns[name]):
            if cell == U:
                status[i, j] = UNDEFINED
            elif cell == INE:
                status[i, j] = INELIGIBLE
            else:
                values[i, j] = cell
    shape = np.asarray(shapes if shapes is not None else [SHAPE] * n, dtype=np.int64)
    return WorldDescriptors(
        source_ids=np.full(n, source, dtype=object),
        world_ids=np.asarray(world_ids if world_ids is not None else range(n), dtype=np.int64),
        group_ids=np.asarray(group_ids if group_ids is not None else range(n), dtype=np.int64),
        metadata={column: shape[:, c] for c, column in enumerate(METADATA_COLUMNS)},
        definitions=tuple(_definition(name, observable=name not in hidden) for name in names),
        values=values,
        status=status,
    )


def _synthetic(
    n_worlds: int,
    *,
    n_treatments: int = 2,
    n_covariates: int = 2,
    n_time_steps: int = 40,
    seed: int = 0,
    dtype: type = np.float64,
    outcome: bool = True,
) -> ObservedWorlds:
    """Gamma treatments (about 30% with random zero weeks), random-walk covariates, an outcome."""
    rng = np.random.default_rng(seed)
    treatment = rng.gamma(2.0, 1.0, size=(n_worlds, n_time_steps, n_treatments))
    gated = rng.random((n_worlds, 1, n_treatments)) < 0.3
    treatment[gated & (rng.random(treatment.shape) < 0.25)] = 0.0
    covariates = 5.0 + rng.normal(size=(n_worlds, n_time_steps, n_covariates)).cumsum(axis=1)
    y = 10.0 + treatment.sum(axis=2) + covariates.sum(axis=2) + rng.normal(size=treatment.shape[:2])
    return ObservedWorlds(
        treatment_raw=treatment.astype(dtype),
        covariates=covariates.astype(dtype),
        outcome_raw=y.astype(dtype) if outcome else None,
    )


def _described(
    n_worlds: int, *, source: str, draws: int = 1, lags: tuple[int, ...] = (1,), **kwargs
) -> WorldDescriptors:
    """Synthetic worlds described with ``draws`` consecutive worlds per generation group."""
    return world_descriptors(
        _synthetic(n_worlds, **kwargs),
        source_id=source,
        lags=lags,
        group_ids=np.arange(n_worlds) // draws,
    )


def _calibration_groups(seed: int, groups: list[tuple[str, int]], fraction: float = 0.5) -> set:
    """The documented split, re-derived: blake2b keys sorted, the first round(f * n) groups."""
    keyed = sorted(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}|{source}|{group}".encode("utf-8", "surrogatepass"), digest_size=8
                ).digest(),
                "big",
            ),
            source,
            group,
        )
        for source, group in groups
    )
    n = len(keyed)
    n_calibration = min(max(round(fraction * n), 1), n - 1) if n >= 2 else 0
    return {(source, group) for _, source, group in keyed[:n_calibration]}


def _statuses(result: Typicality) -> list[str]:
    return [TYPICALITY_STATUSES[code] for code in result.status]


def _assert_same(expected: Typicality, actual: Typicality, rows=None) -> None:
    """Every array of ``actual`` equals ``expected``'s (rows reindexed), NaN equal to NaN."""
    assert actual.features == expected.features
    for name in _ARRAYS:
        want = getattr(expected, name) if rows is None else getattr(expected, name)[rows]
        np.testing.assert_array_equal(getattr(actual, name), want, err_msg=name)


# ---------------------------------------------------------------------------
# The conformal p-value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_self_query_calibration_p_values_are_exactly_a_rotation(seed):
    # Each calibration row of a self-query faces the same ruler and the other 19
    # calibration rows, so its p-values times 20 are exactly the ranks 1..20.
    rng = np.random.default_rng(100 + seed)
    reference = _table({"x": rng.random(40).tolist(), "y": rng.random(40).tolist()})
    result = typicality(reference, reference, seed=seed)
    assert _statuses(result) == ["ok"] * 40
    calibration = result.n_ruler == result.n_ruler.max()
    n_c = int(calibration.sum())
    assert n_c == 20
    assert (result.n_calibration[calibration] == n_c - 1).all()
    ranks = np.rint(result.p_value[calibration] * n_c).astype(int)
    assert sorted(ranks.tolist()) == list(range(1, n_c + 1))


def test_the_split_follows_the_documented_hash_and_ties_count_against_the_query():
    groups = [("ref", g) for g in range(9)]
    calibration = _calibration_groups(0, groups)
    # Pinned: 0.5 * 9 = 4.5 rounds half to even, to 4 calibration groups (half-up
    # would add group 5, the next key).
    assert calibration == {("ref", 0), ("ref", 1), ("ref", 7), ("ref", 8)}
    ruler = [2, 3, 4, 5, 6]
    x = [0.0] * 9
    # Ruler values are mirror-symmetric with median 0 and IQR 2, so z = x / 2 exactly.
    for g, value in zip(ruler, [-2.0, -1.0, 0.0, 1.0, 2.0]):
        x[g] = value
    # Calibration distances (k=1, z units): 0.125 (the mirror image of the query),
    # 0.0625, 0.5 and 0.03125.
    for g, value in zip([0, 1, 7, 8], [-0.75, 0.125, 3.0, 0.0625]):
        x[g] = value
    reference = _table({"x": x})

    own = typicality(reference, reference, k=1)
    # Leaving a calibration row's own group out keeps the five ruler rows.
    left_out_of_calibration = {("ref", g) for g in range(9) if own.n_calibration[g] == 3}
    assert left_out_of_calibration == calibration
    assert own.n_ruler[[0, 1, 7, 8]].tolist() == [5] * 4
    assert own.n_ruler[ruler].tolist() == [4] * 5

    result = typicality(reference, _table({"x": [0.75]}, source="obs"), k=1)
    assert (result.n_ruler[0], result.n_calibration[0]) == (5, 4)
    assert result.distance[0] == 0.125  # |0.75 / 2 - 1 / 2|
    # At least as atypical: the exact tie (-0.75) and 3.0; not 0.125 or 0.0625.
    assert result.p_value[0] == (1 + 2) / (4 + 1)
    assert result.p_min[0] == 1 / 5


def test_groups_of_several_sources_are_split_by_their_full_identity():
    rng = np.random.default_rng(12)
    reference = WorldDescriptors.concat(
        [
            _table({"x": rng.random(10).tolist(), "y": rng.random(10).tolist()}, source=source)
            for source in ("a", "b")
        ]
    )
    # Twenty groups: group 3 of 'a' is not group 3 of 'b'.
    outside = typicality(reference, _table({"x": [0.5], "y": [0.5]}, source="obs"))
    assert (outside.n_calibration[0], outside.n_ruler[0]) == (10, 10)
    own = typicality(reference, reference)
    left_out = {
        (str(source), int(group))
        for source, group, n in zip(own.source_ids, own.group_ids, own.n_calibration)
        if n == 9
    }
    assert left_out == _calibration_groups(0, [(s, g) for s in ("a", "b") for g in range(10)])


@pytest.mark.parametrize(("fraction", "n_calibration"), [(0.1, 1), (0.95, 4)])
def test_the_split_keeps_at_least_one_group_on_each_side(fraction, n_calibration):
    # Of 5 groups, round(0.5) = 0 is raised to 1 and round(4.75) = 5 lowered to 4.
    reference = _table({"x": np.random.default_rng(13).random(5).tolist()})
    result = typicality(
        reference, _table({"x": [0.5]}, source="obs"), k=1, calibration_fraction=fraction
    )
    assert (result.n_calibration[0], result.n_ruler[0]) == (n_calibration, 5 - n_calibration)


def test_a_calibration_group_contributes_its_lowest_world_id_row():
    calibration = {g for _, g in _calibration_groups(0, [("ref", g) for g in range(10)])}
    x, world_ids = [], []
    for g in range(10):
        # Each group lists its higher world_id first; only the higher row of a
        # calibration group lies far from the ruler.
        x += [1000.0 + g if g in calibration else g + 0.6, g + 0.3]
        world_ids += [2 * g + 1, 2 * g]
    reference = _table({"x": x}, world_ids=world_ids, group_ids=[i // 2 for i in range(20)])
    result = typicality(reference, _table({"x": [500.0]}, source="obs"))
    assert (result.n_calibration[0], result.n_ruler[0]) == (5, 10)
    # Calibrating on the far rows instead would give p = 1.
    assert result.p_value[0] == result.p_min[0] == 1 / 6


def test_a_source_id_with_a_lone_surrogate_is_split_by_its_documented_key():
    source = "ref\udc80"  # e.g. an undecodable file-name byte kept by os.fsdecode
    reference = _table({"x": np.random.default_rng(14).random(10).tolist()}, source=source)
    own = typicality(reference, reference)
    left_out = {(source, int(g)) for g, n in zip(own.group_ids, own.n_calibration) if n == 4}
    assert left_out == _calibration_groups(0, [(source, g) for g in range(10)])


def test_a_query_far_outside_the_reference_gets_the_smallest_p_value():
    rng = np.random.default_rng(7)
    reference = _table({"x": rng.random(30).tolist(), "y": rng.random(30).tolist()})
    result = typicality(reference, _table({"x": [5.0], "y": [5.0]}, source="obs"))
    assert _statuses(result) == ["ok"]
    assert result.p_value[0] * (result.n_calibration[0] + 1) == 1
    assert result.p_value[0] == result.p_min[0]


def test_a_query_tied_with_every_world_on_the_scored_feature_gets_p_one():
    # Equal on the selected feature, distinct on another shared one: not a duplicate.
    reference = _table({"x": [1.0] * 12, "y": np.arange(12.0).tolist()})
    result = typicality(reference, _table({"x": [1.0], "y": [4.5]}, source="obs"), features=["x"])
    assert _statuses(result) == ["ok"]
    assert result.p_value[0] == 1.0
    assert result.distance[0] == 0.0  # x is a constant: no distance feature


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "r", "duplicate"),
    [
        (1.0, 1.0 + 0.5e-4, True),
        (1.0, 1.0 + 2e-4, False),
        # The floor max(|x|, |r|, 1): near zero the tolerance is absolute 1e-4.
        (0.0, 0.5e-4, True),
        (0.0, 2e-4, False),
        # Above 1 the tolerance is relative: 0.5 against 1.00005, 2 against 1.0002.
        (1e4, 1e4 * (1 + 0.5e-4), True),
        (1e4, 1e4 * (1 + 2e-4), False),
        # Infinities match only themselves, never a finite value.
        (INF, INF, True),
        (1e300, INF, False),
    ],
)
def test_duplicates_are_judged_on_every_shared_feature_within_the_tolerance(x, r, duplicate):
    # Row 0 sits at r on every feature; the other rows keep the non-constant guard.
    reference = _table({"a": [r, *range(5, 16)], "b": [r, *range(20, 31)]})
    result = typicality(reference, _table({"a": [x], "b": [x]}, source="obs"), features=["a"])
    assert _statuses(result) == ["duplicate" if duplicate else "ok"]


def test_identical_worlds_cannot_flag_a_duplicate_and_tie_the_query_instead():
    # No shared feature takes two values: equal worlds are not copies, and an
    # identical query is scored (every score ties) rather than called a duplicate.
    reference = _table({"a": [3.0] * 12, "b": [7.0] * 12})
    result = typicality(reference, _table({"a": [3.0], "b": [7.0]}, source="obs"))
    assert _statuses(result) == ["ok"]
    assert result.p_value[0] == 1.0


def test_a_duplicate_must_also_match_on_shared_features_that_are_not_scored():
    reference = _table({"a": [1.0, *range(5, 16)], "b": [1.0, *range(20, 31)]})
    query = _table({"a": [1.0, 1.0], "b": [1.0, 3.0]}, source="obs")
    assert _statuses(typicality(reference, query, features=["a"])) == ["duplicate", "ok"]


def test_reference_copies_are_judged_on_every_observable_feature():
    # Two groups equal on the scored x but not on y hold two worlds, not one twice.
    x = [1.0, 1.0, *np.arange(2.0, 12.0).tolist()]
    reference = _table({"x": x, "y": np.arange(12.0).tolist()})
    result = typicality(reference, _table({"x": [1.5], "y": [3.5]}, source="obs"), features=["x"])
    assert _statuses(result) == ["ok"]


def test_a_duplicate_is_reported_before_a_too_small_ruler():
    # Three groups leave a one-row ruler, too few for any k.
    reference = _table({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    query = _table({"x": [2.0, 2.5], "y": [5.0, 5.5]}, source="obs")
    assert _statuses(typicality(reference, query)) == ["duplicate", "too_few"]


def test_count_statistics_are_no_evidence_of_one_world():
    # Zero and constant fractions take a handful of levels that distinct worlds
    # share: they make neither reference copies nor duplicate queries.
    counts = ("zero_fraction", "constant_fraction")

    def flat_second_treatment(observed: ObservedWorlds, worlds: slice) -> ObservedWorlds:
        treatment = observed.treatment_raw.copy()
        treatment[worlds, :, 1] = 3.0
        return ObservedWorlds(treatment, observed.covariates, observed.outcome_raw)

    reference = world_descriptors(
        flat_second_treatment(_synthetic(20), slice(None, None, 2)),
        source_id="ref",
        statistics=counts,
        group_ids=np.arange(20),
    )
    query = world_descriptors(
        flat_second_treatment(_synthetic(20, seed=1), slice(0, 5)),
        source_id="obs",
        statistics=counts,
    )
    # Every other reference world has one constant treatment of two, so the
    # constant fraction takes two levels too.
    assert set(reference.column("treatment_constant_fraction")) == {0.0, 0.5}
    assert _statuses(typicality(reference, query)) == ["ok"] * 20


def test_one_step_worlds_are_not_copies_of_each_other():
    # At T = 1 every series is constant: cv is 0 and level-to-variation +inf (or
    # undefined at zero) in every world, so no feature tells worlds apart.
    reference = _described(8, source="ref", n_time_steps=1)
    query = _described(1, source="obs", seed=5, n_time_steps=1)
    assert _statuses(typicality(reference, query, k=2)) == ["ok"]


def test_a_dataset_read_back_from_csv_is_a_duplicate_of_its_float32_original(tmp_path):
    observed = _synthetic(12, seed=3, dtype=np.float32)
    reference = world_descriptors(observed, source_id="ref", group_ids=np.arange(12))
    path = tmp_path / "dataset.csv"
    observed.to_frame(4).to_csv(path, index=False)
    query = world_descriptors(ObservedWorlds.from_csv(path), source_id="csv")
    # The printed float32 decimals read back as other float64 values ...
    assert not np.array_equal(query.values[0], reference.values[4], equal_nan=True)
    # ... that still describe the same world within the duplicate tolerance.
    assert _statuses(typicality(reference, query)) == ["duplicate"]


# ---------------------------------------------------------------------------
# Feature roles and violations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_treatments", "n_covariates"), [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (1, 3), (4, 1)]
)
def test_reducers_over_a_single_node_or_pair_are_structural_duplicates(n_treatments, n_covariates):
    shape = {"n_treatments": n_treatments, "n_covariates": n_covariates}
    reference = _described(16, source="ref", lags=(1, 2), **shape)
    result = typicality(reference, _described(1, source="obs", seed=5, lags=(1, 2), **shape))
    k, m = n_treatments, n_covariates
    set_size = {
        "treatment": k,
        "covariate": m,
        "treatment_pair": k * (k - 1) // 2,
        "covariate_pair": m * (m - 1) // 2,
        "treatment_covariate": k * m,
        "treatment_outcome": k,
    }
    first: dict[tuple, str] = {}
    duplicates = []
    for j, name in enumerate(result.features):
        d = reference.definition(name)
        single = d.reducer is not None and set_size.get(d.role) == 1
        key = (d.role, d.statistic, d.view, d.lag)
        expected = single and key in first
        if single:
            first.setdefault(key, name)
        assert (FEATURE_ROLES[result.role[0, j]] == "duplicate") is expected, name
        if expected:
            duplicates.append(j)
            # One way only: a reducer over one element repeats the first one.
            valid = reference.column_status(name) == "valid"
            np.testing.assert_array_equal(
                reference.column(name)[valid], reference.column(first[key])[valid]
            )
    defined = ~np.isnan(result.percentile[0])
    assert result.n_profiled[0] == defined.sum() - defined[duplicates].sum()
    assert not result.violation[0, duplicates].any()


def test_structural_duplicates_never_enter_the_scores_or_the_counts():
    reference = _described(16, source="ref", n_treatments=1)
    observed = _synthetic(2, n_treatments=1, seed=13)
    treatment = observed.treatment_raw.copy()
    treatment[1] = 3.0  # world 1 has one flat treatment
    query = world_descriptors(
        ObservedWorlds(treatment, observed.covariates, observed.outcome_raw), source_id="obs"
    )
    full = typicality(reference, query)
    assert _statuses(full) == ["ok", "ok"]
    redundant = full.role[0] == FEATURE_ROLES.index("duplicate")
    assert redundant.any() and (full.role[1] == full.role[0]).all()
    kept = [name for name, flag in zip(full.features, redundant) if not flag]
    narrow = typicality(reference, query, features=kept)
    for name in (
        "status",
        "p_value",
        "p_min",
        "violations",
        "distance",
        "n_distance_features",
        "n_constant_features",
        "n_profiled",
        "n_inside",
        "expected_outside",
    ):
        np.testing.assert_array_equal(getattr(narrow, name), getattr(full, name), err_msg=name)
    # The flat treatment breaks the first reducer; its copies over one treatment
    # (min and max) are never flagged.
    ltv = [
        full.features.index(f"treatment_level_to_variation_{r}") for r in ("median", "min", "max")
    ]
    assert full.violation[1, ltv].tolist() == [True, False, False]
    assert full.violations[1] > 0 and not full.violation[:, redundant].any()


def test_one_covariate_with_zero_weeks_breaks_the_zero_fraction_max_constant():
    reference = _described(16, source="ref", n_covariates=3)
    observed = _synthetic(1, n_covariates=3, seed=11)
    covariates = observed.covariates.copy()
    covariates[0, ::5, 1] = 0.0  # zero weeks in the second covariate only
    query = world_descriptors(
        ObservedWorlds(observed.treatment_raw, covariates, observed.outcome_raw), source_id="obs"
    )
    result = typicality(reference, query)
    assert _statuses(result) == ["ok"]
    j = result.features.index("covariate_zero_fraction_max")
    assert FEATURE_ROLES[result.role[0, j]] == "constant"
    assert result.violation[0, j]
    # The median and min of the three zero fractions are still the constant 0.
    for name in ("covariate_zero_fraction_median", "covariate_zero_fraction_min"):
        assert not result.violation[0, result.features.index(name)]


def _roles_reference(*, deviating_calibration_row: bool) -> WorldDescriptors:
    """12 single-row groups: x a distance feature, constants c1 = c2 = 1, s +inf on one ruler row."""
    calibration = sorted(g for _, g in _calibration_groups(0, [("ref", g) for g in range(12)]))
    ruler = sorted(set(range(12)) - set(calibration))
    x, s, c1 = [0.0] * 12, [0.0] * 12, [1.0] * 12
    # Ruler x has median 0 and IQR 2, so z = x / 2 exactly.
    ruler_cells = [(-3.0, 0.5), (-1.0, INF), (-1.0, 0.25), (1.0, 0.75), (1.0, 1.0), (3.0, 0.125)]
    for g, (xv, sv) in zip(ruler, ruler_cells):
        x[g], s[g] = xv, sv
    # Calibration distances in z units: 0, 0.375, 0.25, 0.25, 3 and 0.5.
    calibration_cells = [(1.0, 0.1), (0.25, 0.2), (2.5, 0.3), (3.5, 0.4), (-9.0, 0.6), (0.0, 0.7)]
    for g, (xv, sv) in zip(calibration, calibration_cells):
        x[g], s[g] = xv, sv
    if deviating_calibration_row:
        c1[calibration[0]] = 2.0  # one violation at distance 0
    return _table({"x": x, "c1": c1, "c2": [1.0] * 12, "s": s})


_ROLE_QUERY = {
    # 0 seen +inf, 1 unseen -inf, 2-3 finite s values, 4 c1 = +inf,
    # 5 x undefined, 6 c2 off its constant, 7 far on x.
    "x": [1.5, 1.5, 1.5, 1.5, 1.5, U, 1.5, 100.0],
    "c1": [1.0, 1.0, 1.0, 1.0, INF, 1.0, 1.0, 1.0],
    "c2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0],
    "s": [INF, -INF, 0.3, 1000.0, 0.3, 0.3, 0.3, 0.3],
}


def test_roles_violations_and_distance_follow_the_ruler_rows():
    reference = _roles_reference(deviating_calibration_row=False)
    result = typicality(reference, _table(_ROLE_QUERY, source="obs"), k=1)
    assert _statuses(result) == ["ok"] * 8
    roles = [[FEATURE_ROLES[code] for code in row] for row in result.role]
    assert roles == [["distance", "constant", "constant", "state"]] * 8
    assert result.n_distance_features.tolist() == [1] * 8
    assert result.n_constant_features.tolist() == [2] * 8
    # A seen state (+inf) costs nothing, an unseen one (-inf) one violation; the
    # finite value of a state feature never enters the distance; a constant at
    # +inf counts once; an undefined distance feature is a violation at inf.
    assert result.violations.tolist() == [0, 1, 0, 0, 1, 1, 1, 0]
    # k=1 over x alone: |1.5 / 2 - 1 / 2| = 0.25 and |100 / 2 - 3 / 2| = 48.5.
    assert result.distance.tolist() == [0.25, 0.25, 0.25, 0.25, 0.25, INF, 0.25, 48.5]
    expected = np.zeros((8, 4), dtype=bool)
    expected[1, 3] = expected[4, 1] = expected[5, 0] = expected[6, 2] = True
    np.testing.assert_array_equal(result.violation, expected)
    assert "obs:1: s" in result.table().splitlines()
    # No calibration row deviates: a constant deviation is more atypical than all.
    assert result.p_value[6] == result.p_min[6] == 1 / 7


def test_violations_outrank_distance_both_ways():
    reference = _roles_reference(deviating_calibration_row=True)
    result = typicality(reference, _table(_ROLE_QUERY, source="obs"), k=1)
    # Far query, no violation: the deviating calibration row at distance 0 still
    # counts against it, and only that one.
    assert result.p_value[7] == 2 / 7
    # One violation at distance 0.25: calibration rows farther away (0.375, 0.5,
    # even 3) without a violation do not count against it.
    assert result.p_value[6] == 1 / 7


def test_the_distance_averages_the_k_nearest_ruler_worlds():
    reference = _roles_reference(deviating_calibration_row=False)
    result = typicality(reference, _table(_ROLE_QUERY, source="obs"), k=2)
    # Ruler z = x / 2. Nearest to 0.75: 0.5 twice. Nearest to 50: 1.5 at 48.5,
    # then 0.5 at 49.5.
    assert result.distance.tolist() == [0.25] * 5 + [INF, 0.25, (48.5 + 49.5) / 2]


def test_undefined_and_ineligible_are_different_states():
    calibration = {g for _, g in _calibration_groups(0, [("ref", g) for g in range(12)])}
    s = np.arange(12.0).tolist()
    s[sorted(set(range(12)) - calibration)[0]] = U  # the ruler shows s undefined, not ineligible
    reference = _table({"x": np.random.default_rng(15).random(12).tolist(), "s": s})
    result = typicality(reference, _table({"x": [0.5, 0.5], "s": [U, INE]}, source="obs"))
    assert _statuses(result) == ["ok", "ok"]
    assert result.violation[:, 1].tolist() == [False, True]


def test_scores_depend_on_the_ruler_worlds_only():
    reference = _described(20, source="ref")
    query = _described(1, source="obs", seed=8)
    base = typicality(reference, query)
    calibration = {g for _, g in _calibration_groups(0, [("ref", g) for g in range(20)])}
    rows = np.isin(reference.group_ids, list(calibration))  # single-row groups
    values = reference.values.copy()
    values[rows] *= 100.0
    scaled = typicality(replace(reference, values=values), query)
    for name in ("distance", "violations", "n_distance_features", "n_constant_features", "role"):
        np.testing.assert_array_equal(getattr(scaled, name), getattr(base, name), err_msg=name)
    # Only the comparison moves: every calibration world, now far off the ruler, counts.
    assert base.p_value[0] < 1.0 and scaled.p_value[0] == 1.0


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def test_profile_order_statistics_percentiles_and_inside_flags():
    reference = _table({"x": np.arange(20.0).tolist()}, group_ids=[i // 2 for i in range(20)])
    query = _table({"x": [0.5, -1.0, 18.5, 19.5, U]}, source="obs")
    result = typicality(reference, query)
    # inverted_cdf at n=20, central 0.9: positions ceil(1) = 1, 10 and 19.
    assert result.reference_low[:, 0].tolist() == [0.0] * 5
    assert result.reference_median[:, 0].tolist() == [9.0] * 5
    assert result.reference_high[:, 0].tolist() == [18.0] * 5
    np.testing.assert_array_equal(result.percentile[:, 0], [0.05, 0.0, 0.95, 1.0, np.nan])
    assert result.inside[:, 0].tolist() == [True, False, False, False, False]
    assert result.n_profiled.tolist() == [1, 1, 1, 1, 0]
    assert result.n_inside.tolist() == [1, 0, 0, 0, 0]
    frame = result.profile(2)
    assert frame.loc[0, "percentile"] == 0.95 and not frame.loc[0, "inside"]

    # Both ends of the interval are inside: a value equal to low or high counts.
    ends = typicality(reference, _table({"x": [0.0, 18.0]}, source="obs"))
    assert ends.inside[:, 0].tolist() == [True, True]

    # 20 * 0.15 is exactly 3: the low end is the third value, not the fourth.
    narrow = typicality(reference, query, central=0.7)
    assert (narrow.reference_low[0, 0], narrow.reference_high[0, 0]) == (2.0, 16.0)

    # A reference row leaves its own group of two out: 18 values remain.
    own = typicality(reference, reference.take([10]))
    assert own.n_reference[0] == 18
    assert own.percentile[0, 0] == 10 / 18


def test_profile_intervals_and_percentiles_include_infinite_reference_values():
    reference = _table(
        {"x": [*np.arange(18.0).tolist(), INF, INF]}, group_ids=[i // 2 for i in range(20)]
    )
    result = typicality(reference, _table({"x": [100.0, INF]}, source="obs"))
    assert result.reference_high[:, 0].tolist() == [INF, INF]
    assert result.inside[:, 0].tolist() == [True, True]
    np.testing.assert_array_equal(result.percentile[:, 0], [18 / 20, 19 / 20])


@pytest.mark.parametrize("n", [5, 8, 21])
@pytest.mark.parametrize("central", [0.5, 0.7, 0.9])
def test_expected_outside_counts_the_self_query_rows_outside_their_interval(n, central):
    # Over a self-query of n distinct values, the row of overall rank r lies
    # outside its interval for exactly the ranks the expectation counts.
    reference = _table({"x": np.arange(float(n)).tolist()})
    result = typicality(reference, reference, central=central)
    outside = n - int(result.inside[:, 0].sum())
    assert (result.expected_outside == outside / n).all()


# ---------------------------------------------------------------------------
# Panels, strata and order
# ---------------------------------------------------------------------------


def test_a_shuffled_panel_scores_every_row_as_its_own_single_row_call():
    main = _described(24, source="ref", draws=2)
    other = world_descriptors(
        _synthetic(12, n_treatments=3, seed=4),
        source_id="ref",
        world_ids=np.arange(100, 112),
        group_ids=np.arange(100, 112) // 2,
    )
    reference = WorldDescriptors.concat([main, other])
    calibration = sorted(g for _, g in _calibration_groups(0, [("ref", g) for g in range(12)]))
    ruler = sorted(set(range(12)) - set(calibration))
    panel = WorldDescriptors.concat(
        [
            # a ruler-group row and a calibration group's calibration row
            reference.take([2 * ruler[0] + 1, 2 * calibration[0]]),
            _described(2, source="obs", seed=9),
            world_descriptors(_synthetic(1, n_treatments=3, seed=10), source_id="other_shape"),
            world_descriptors(
                _synthetic(1, n_treatments=1, n_covariates=1, seed=12), source_id="unseen_shape"
            ),
        ]
    )
    panel = panel.take(np.random.default_rng(0).permutation(panel.n_worlds))
    result = typicality(reference, panel)
    assert sorted(_statuses(result)) == ["no_reference"] + ["ok"] * 5
    for i in range(panel.n_worlds):
        _assert_same(result, typicality(reference, panel.take([i])), rows=[i])


@pytest.mark.parametrize("column", METADATA_COLUMNS)
def test_each_shape_is_split_and_scored_on_its_own(column):
    position = METADATA_COLUMNS.index(column)
    other = tuple(v + (position == c) for c, v in enumerate(SHAPE))
    unseen = tuple(v + 2 * (position == c) for c, v in enumerate(SHAPE))
    rng = np.random.default_rng(3)
    same_shape = _table({"x": rng.random(12).tolist(), "y": rng.random(12).tolist()})
    other_shape = _table(
        {"x": (5.0 + rng.random(10)).tolist(), "y": rng.random(10).tolist()},
        world_ids=range(100, 110),
        group_ids=range(100, 110),
        shapes=[other] * 10,
    )
    reference = WorldDescriptors.concat([same_shape, other_shape])
    query = _table(
        {"x": [0.5, 5.5, 0.5], "y": [0.5, 0.5, 0.5]},
        source="obs",
        shapes=[SHAPE, other, unseen],
    )
    result = typicality(reference, query)
    assert _statuses(result) == ["ok", "ok", "no_reference"]
    assert result.n_reference.tolist() == [12, 10, 0]
    _assert_same(result, typicality(same_shape, query.take([0])), rows=[0])
    _assert_same(result, typicality(other_shape, query.take([1])), rows=[1])


def test_results_ignore_row_order_and_take_each_group_s_lowest_world_id():
    observed = _synthetic(20, seed=21)
    # Pairs listed highest world_id first: a group's lowest world_id is never its first row.
    world_ids = np.arange(20).reshape(10, 2)[:, ::-1].reshape(-1)
    reference = world_descriptors(
        observed, source_id="ref", world_ids=world_ids, group_ids=np.arange(20) // 2
    )
    query = _described(2, source="obs", seed=22)
    base = typicality(reference, query)
    own = typicality(reference, reference)
    swapped = np.arange(20).reshape(10, 2)[:, ::-1].reshape(-1)
    for order in (swapped, *(np.random.default_rng(s).permutation(20) for s in range(3))):
        shuffled = reference.take(order)
        _assert_same(base, typicality(shuffled, query))
        _assert_same(own, typicality(shuffled, shuffled), rows=order)
    roles = {typicality(reference, query, seed=seed).role.tobytes() for seed in range(5)}
    assert len(roles) > 1


# ---------------------------------------------------------------------------
# Statuses and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_statuses_follow_ruler_and_calibration_sizes(k):
    needed = max(k, 2)

    def two_groups(rows_per_group: int) -> WorldDescriptors:
        # One calibration group (its lowest row) and one ruler group of every row.
        n = 2 * rows_per_group
        return _table(
            {"x": (np.arange(n) / 7.0).tolist(), "c": [1.0] * n},
            group_ids=[0] * rows_per_group + [1] * rows_per_group,
        )

    query = _table({"x": [0.3], "c": [5.0]}, source="obs")
    short = typicality(two_groups(needed - 1), query, k=k)
    assert _statuses(short) == ["too_few"]
    assert (short.n_ruler[0], short.n_calibration[0], short.n_reference[0]) == (
        needed - 1,
        1,
        2 * (needed - 1),
    )
    for name in ("p_value", "p_min", "violations", "distance"):
        assert np.isnan(getattr(short, name)[0]), name
    assert not short.violation.any()
    if needed - 1 == 1:  # a one-row ruler has no constant
        assert FEATURE_ROLES.index("constant") not in short.role

    enough = typicality(two_groups(needed), query, k=k)
    assert _statuses(enough) == ["ok"]
    assert enough.violation[0].tolist() == [False, True]

    one_group = _table({"x": np.arange(6.0).tolist(), "c": [1.0] * 6}, group_ids=[0] * 6)
    alone = typicality(one_group, query, k=k)
    assert _statuses(alone) == ["too_few"]
    assert (alone.n_ruler[0], alone.n_calibration[0]) == (6, 0)


def test_invalid_requests_fail_with_clear_errors():
    reference = _table({"x": np.arange(8.0).tolist(), "y": (np.arange(8.0) ** 2).tolist()})
    query = _table({"x": [0.5], "y": [0.5]}, source="obs")
    with pytest.raises(ValueError, match="group_ids="):
        typicality(_table({"x": [0.0, 1.0]}, group_ids=[0, -1]), query.select(["x"]))
    copies = _table({"x": [1.0, 1.0, 2.0, 3.0], "y": [5.0, 5.0, 6.0, 7.0]})
    with pytest.raises(ValueError, match="bit-identical"):
        typicality(copies, query)
    with pytest.raises(ValueError, match="not one of its rows"):
        typicality(reference, _table({"x": [0.5], "y": [0.5]}, world_ids=[99]))
    changed = _table({"x": [0.0], "y": [0.5]})  # identity ('ref', 0), but y differs
    with pytest.raises(ValueError, match="differs from it on 'y'"):
        typicality(reference, changed)
    with pytest.raises(ValueError, match="share no observable feature"):
        typicality(reference, _table({"z": [0.5]}, source="obs"))
    hidden = _table({"x": [0.0, 1.0], "truth": [0.2, 0.3]}, hidden=("truth",))
    with pytest.raises(ValueError, match="not observable"):
        typicality(
            hidden,
            _table({"x": [0.5], "truth": [0.2]}, source="obs", hidden=("truth",)),
            features=["truth"],
        )
    regrouped = replace(reference.take([3]), group_ids=np.array([7]))
    with pytest.raises(ValueError, match="carries group_id 7"):
        typicality(reference, regrouped)
    # An external dataset described under the reference's source_id collides with it.
    collision = _table({"x": [0.5], "y": [0.5]}, world_ids=[3], group_ids=[-1])
    with pytest.raises(ValueError, match="own source_id"):
        typicality(reference, collision)
    # The reference's own worlds, described again without their generation groups.
    observed = _synthetic(8, seed=2)
    described = world_descriptors(observed, source_id="ref", group_ids=np.arange(8))
    redescribed = world_descriptors(observed, source_id="ref", worlds=[2, 5])
    with pytest.raises(ValueError, match="without its generation group") as raised:
        typicality(described, redescribed)
    assert "own source_id" not in str(raised.value)
    no_treatment = _table({"x": [0.5], "y": [0.5]}, source="obs", shapes=[(0, 2, 40)])
    with pytest.raises(ValueError, match="no active treatment"):
        typicality(reference, no_treatment)
    with pytest.raises(TypeError, match="world_descriptors"):
        typicality(_synthetic(4), query)
    with pytest.raises(TypeError, match="world_descriptors"):
        typicality(reference, _synthetic(1))
    # A shape the reference lacks never reaches a neighbour search: typicality
    # itself must refuse the parameters.
    unseen = _table({"x": [0.5], "y": [0.5]}, source="obs", shapes=[(1, 1, 40)])
    with pytest.raises(ValueError, match="positive"):
        typicality(reference, unseen, k=0)
    with pytest.raises(TypeError, match="positive"):
        typicality(reference, unseen, k=True)
    for fraction in (0.0, 1.0, 1.5, math.nan):
        with pytest.raises(ValueError, match="calibration_fraction"):
            typicality(reference, unseen, calibration_fraction=fraction)
    with pytest.raises(TypeError, match="calibration_fraction"):
        typicality(reference, unseen, calibration_fraction="0.5")
    with pytest.raises(ValueError, match="central"):
        typicality(reference, unseen, central=1.0)
    with pytest.raises(ValueError, match="seed"):
        typicality(reference, unseen, seed=-1)
    with pytest.raises(TypeError, match="seed"):
        typicality(reference, unseen, seed=1.5)


def test_every_dataset_needs_treatments_covariates_and_an_outcome():
    reference = _described(8, source="ref")
    no_outcome = world_descriptors(_synthetic(1, seed=4, outcome=False), source_id="obs")
    with pytest.raises(ValueError, match=r"\('obs', 0\) has no observed outcome"):
        typicality(reference, no_outcome)
    with pytest.raises(ValueError, match="no observed outcome"):
        typicality(reference, no_outcome, features=["treatment_outcome_abs_pearson_max"])
    observed = _synthetic(1, seed=4)
    no_covariates = world_descriptors(
        ObservedWorlds(observed.treatment_raw, None, observed.outcome_raw), source_id="obs"
    )
    with pytest.raises(ValueError, match="no active covariate"):
        typicality(reference, no_covariates)


def _with_truth(table: WorldDescriptors) -> WorldDescriptors:
    """``table`` plus the truth columns ``include_truth=True`` adds (generated worlds only)."""
    truth = tuple(
        FeatureDefinition(
            name=name,
            role="truth",
            statistic="share",
            view="levels",
            lag=None,
            reducer=None,
            observable=False,
            description=f"net share of outcome from contribution row {key!r}",
        )
        for name, key in TRUTH_FEATURES.items()
    )
    n = table.n_worlds
    return replace(
        table,
        definitions=table.definitions + truth,
        values=np.hstack([table.values, np.full((n, len(truth)), 0.2)]),
        status=np.hstack([table.status, np.zeros((n, len(truth)), dtype=np.uint8)]),
    )


def test_default_features_are_the_shared_equal_definitions_in_reference_order():
    reference = _with_truth(
        world_descriptors(_synthetic(12), source_id="ref", lags=(1, 2), group_ids=np.arange(12))
    )
    described = _with_truth(world_descriptors(_synthetic(1, seed=3), source_id="obs"))
    flipped = described.select(described.features[::-1])
    bumped = "covariate_cv_max"
    query = replace(
        flipped,
        definitions=tuple(
            replace(d, version=d.version + 1) if d.name == bumped else d
            for d in flipped.definitions
        ),
    )
    result = typicality(reference, query)
    # Lag-2 columns are missing from the query, truth is not observable, and the
    # bumped version is another definition.
    expected = tuple(
        d.name for d in reference.definitions if d.observable and d.lag != 2 and d.name != bumped
    )
    assert result.features == expected
    with pytest.raises(ValueError, match=bumped):
        typicality(reference, query, features=[bumped])
    with pytest.raises(KeyError, match="treatment_acf_lag_2_median"):
        typicality(reference, query, features=["treatment_acf_lag_2_median"])


def test_summary_is_strict_json_with_an_infinite_distance_and_a_non_ok_row():
    rng = np.random.default_rng(5)
    reference = _table({"x": rng.random(12).tolist(), "y": rng.random(12).tolist()})
    query = _table({"x": [U, 0.5], "y": [0.5, 0.5]}, source="obs", shapes=[SHAPE, (1, 1, 40)])
    result = typicality(reference, query, k=1)
    assert _statuses(result) == ["ok", "no_reference"]
    assert result.distance[0] == INF
    payload = result.summary()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    first, second = payload["rows"]
    assert first["distance"] is None and first["violating_features"] == ["x"]
    assert second["p_value"] is None and second["status"] == "no_reference"
    assert result.to_frame()["status"].tolist() == ["ok", "no_reference"]


# ---------------------------------------------------------------------------
# matched_prior
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prior():
    return pg.make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=40,
        n_treatments_active_range=(1, 3),
        n_covariates_active_range=(1, 2),
        n_cells=10,
        draws_per_cell=4,
        seed=11,
    )


def test_matched_prior_pins_the_dataset_shape_and_keeps_every_other_field(prior):
    observed = _synthetic(1, n_treatments=2, n_covariates=1)
    matched = matched_prior(prior, observed, n_cells=6, seed=99)
    assert matched.active_count_grid == ((2,), (1,))
    assert matched.n_time_steps == 40
    assert matched == replace(
        prior,
        n_treatments_active_range=(2, 2),
        n_covariates_active_range=(1, 1),
        active_count_weights=None,
        n_cells=6,
        draws_per_cell=1,
        seed=99,
    )
    described = world_descriptors(observed, source_id="obs")
    assert matched_prior(prior, described, n_cells=6, seed=99) == matched


def test_matched_prior_refuses_shapes_the_prior_cannot_generate(prior):
    observed = _synthetic(1, n_treatments=2, n_covariates=1)
    with pytest.raises(ValueError, match="n_time_steps=32"):
        matched_prior(
            prior, _synthetic(1, n_treatments=2, n_covariates=1, n_time_steps=32), n_cells=4, seed=1
        )
    with pytest.raises(ValueError, match="active-count grid"):
        matched_prior(prior, _synthetic(1, n_treatments=4, n_covariates=1), n_cells=4, seed=1)
    weights = [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]  # zero at 2 treatments x 1 covariate
    stratified = replace(prior, active_count_allocation="stratified", active_count_weights=weights)
    with pytest.raises(ValueError, match="zero weight"):
        matched_prior(stratified, observed, n_cells=4, seed=1)
    independent = replace(prior, active_count_weights=weights)
    assert matched_prior(independent, observed, n_cells=4, seed=1).active_count_grid == ((2,), (1,))
    mixed = _synthetic(2, n_treatments=2, n_covariates=1)
    treatment = mixed.treatment_raw.copy()
    treatment[1, :, 1] = 0.0
    padded = ObservedWorlds(
        treatment, mixed.covariates, mixed.outcome_raw, treatment_active_mask=[[1, 1], [1, 0]]
    )
    with pytest.raises(ValueError, match="differ"):
        matched_prior(prior, padded, n_cells=4, seed=1)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_periodic_on_off_datasets_sit_at_the_floor_of_a_texture_reference():
    sizes = {"n_treatments": 3, "n_covariates": 2, "n_latent": 1, "n_time_steps": 52}
    datasets = world_descriptors(
        pg.sample_prior_predictive(
            pg.make_scm_prior(
                **sizes, trajectories="periodic_on_off", n_cells=2, draws_per_cell=1, seed=31
            )
        ),
        source_id="datasets",
    )
    reference_prior = matched_prior(pg.make_scm_prior(**sizes), datasets, n_cells=12, seed=32)
    reference = world_descriptors(
        pg.sample_prior_predictive(reference_prior), source_id="reference"
    )
    result = typicality(reference, datasets)
    assert _statuses(result) == ["ok", "ok"]
    assert (result.violations > 0).all()
    np.testing.assert_array_equal(result.p_value, result.p_min)
