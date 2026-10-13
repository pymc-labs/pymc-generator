"""Between-world descriptor analysis: summaries, bins, scales, neighbours, comparisons.

Every expectation is an INDEPENDENT oracle — a hand-computed literal or a
brute-force evaluation of the definition — never the helper under test.
Tables are built directly from literal cells.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pymc_generator import descriptor_analysis as da
from pymc_generator.descriptor_analysis import (
    NEIGHBOR_STATUSES,
    DescriptorScale,
    bin_counts,
    compare_descriptors,
    nearest_worlds,
    summarize_descriptors,
)
from pymc_generator.descriptors import (
    INELIGIBLE,
    METADATA_COLUMNS,
    UNDEFINED,
    FeatureDefinition,
    WorldDescriptors,
)

U, INE = "undefined", "ineligible"  # literal cells for the two non-valid states
INF = math.inf


def _definition(name: str, lag: int | None = None) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        role="covariate",
        statistic="cv",
        view="levels",
        lag=lag,
        reducer="median",
        observable=True,
        description=f"test column {name}",
    )


def _table(
    columns: dict[str, list],
    *,
    source: str = "S",
    sources: list[str] | None = None,
    world_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    n_time_steps: list[int] | None = None,
    definitions: dict[str, FeatureDefinition] | None = None,
) -> WorldDescriptors:
    """A descriptor table from literal cells: numbers are valid, U / INE mark the other states."""
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
    metadata = {column: np.zeros(n, dtype=np.int64) for column in METADATA_COLUMNS}
    if n_time_steps is not None:
        metadata["n_time_steps"] = np.asarray(n_time_steps, dtype=np.int64)
    return WorldDescriptors(
        source_ids=np.asarray(sources if sources is not None else [source] * n, dtype=object),
        world_ids=np.asarray(world_ids if world_ids is not None else range(n), dtype=np.int64),
        group_ids=np.asarray(group_ids if group_ids is not None else [-1] * n, dtype=np.int64),
        metadata=metadata,
        definitions=tuple((definitions or {}).get(name, _definition(name)) for name in names),
        values=values,
        status=status,
    )


def _labels(result: da.NearestWorlds) -> list[str]:
    return [NEIGHBOR_STATUSES[code] for code in result.status]


# ---------------------------------------------------------------------------
# compare_descriptors
# ---------------------------------------------------------------------------


def test_midranks_count_ties_as_half_and_rank_infinity_above_every_finite_value():
    reference = _table({"x": [1.0, 2.0, 2.0, 3.0, INF, -INF, U, INE]}, source="ref")
    query = _table({"x": [2.0, INF, -INF, 0.0, 5.0, U]}, source="qry")
    percentiles = compare_descriptors(reference, query).percentiles[:, 0]
    # Six valid reference values; the infinities are ordered values, NaN cells are not.
    expected = [(2 + 0.5 * 2) / 6, (5 + 0.5) / 6, 0.5 / 6, 1 / 6, 5 / 6, np.nan]
    np.testing.assert_allclose(percentiles, expected)


def test_leave_self_out_drops_exactly_the_row_itself():
    table = _table({"x": [1.0, 2.0, 2.0, 4.0]})
    # Each row against the three OTHER rows: count below plus half the ties.
    np.testing.assert_allclose(
        compare_descriptors(table, table).percentiles[:, 0], [0 / 3, 1.5 / 3, 1.5 / 3, 3 / 3]
    )

    # A shared identity is left out; the same value under another identity is not.
    visitors = _table({"x": [1.0, 1.0]}, sources=["S", "V"], world_ids=[0, 0])
    np.testing.assert_allclose(
        compare_descriptors(table, visitors).percentiles[:, 0], [0 / 3, 0.5 / 4]
    )

    # A shared identity carrying another value is removed with the weight it had:
    # (S, 0) = 1 sat below 3, (S, 3) = 4 sat above 0.
    moved = _table({"x": [3.0, 0.0]}, world_ids=[0, 3])
    np.testing.assert_allclose(compare_descriptors(table, moved).percentiles[:, 0], [2 / 3, 0 / 3])

    # Nothing is left to rank against once the only reference row is the row itself.
    single = _table({"x": [7.0]})
    assert np.isnan(compare_descriptors(single, single).percentiles[0, 0])


def test_comparison_strata_keep_reference_rows_apart():
    reference = _table({"x": [0.0, 2.0, 100.0, 102.0]}, source="ref", n_time_steps=[10, 10, 20, 20])
    query = _table({"x": [2.0, 101.0, 2.0]}, source="qry", n_time_steps=[10, 20, 30])
    result = compare_descriptors(reference, query, by=("n_time_steps",))
    assert result.strata == ((10,), (20,), (30,))
    # (A, 2) ranks among A = {0, 2} only; pooled with B = {100, 102} it would read 1.5 / 4.
    np.testing.assert_allclose(result.percentiles[:, 0], [0.75, 0.5, np.nan])
    query_only = result.stats(2, "x")
    assert query_only["reference"]["n_valid"] == 0
    assert query_only["ks_distance"] is None and query_only["median_shift"] is None


def test_comparison_contrasts_match_hand_computed_values():
    reference = _table({"x": [0.0, 1.0, 2.0, 3.0]}, source="ref")
    query = _table({"x": [-INF, 2.0, 3.0, 4.0, 5.0, U, INE]}, source="qry")
    stats = compare_descriptors(reference, query).stats(0, "x")
    assert stats["median_shift"] == pytest.approx(3.5 - 1.5)  # finite medians only
    assert stats["ks_distance"] == pytest.approx(0.4)  # at t = 3: F_ref 1.0, F_query 0.6
    assert stats["fraction_below_range"] == pytest.approx(1 / 5)  # the -inf
    assert stats["fraction_above_range"] == pytest.approx(2 / 5)  # 4 and 5
    side = stats["query"]
    counts = ("n_valid", "n_finite", "n_neginf", "n_undefined", "n_ineligible")
    assert tuple(side[key] for key in counts) == (5, 4, 1, 1, 1)


# ---------------------------------------------------------------------------
# summarize_descriptors
# ---------------------------------------------------------------------------


def test_summary_keeps_states_apart_and_counts_distinct_known_groups():
    table = _table(
        {"x": [1.0, 2.0, 3.0, INF, -INF, U, U, INE]},
        sources=["A", "A", "B", "B", "A", "A", "B", "B"],
        group_ids=[0, 0, 0, -1, 1, 1, -1, 0],
    )
    stats = summarize_descriptors(table, quantiles=(0.5,)).stats(0, "x")
    assert stats["n_worlds"] == 8
    # (A, 0), (A, 1) and (B, 0): one group id under two sources is two groups.
    assert (stats["n_groups"], stats["n_unknown_group"]) == (3, 2)
    counts = ("n_valid", "n_finite", "n_posinf", "n_neginf", "n_undefined", "n_ineligible")
    assert tuple(stats[key] for key in counts) == (5, 3, 1, 1, 2, 1)
    # Moments over the finite valid values {1, 2, 3} only.
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(math.sqrt(2 / 3))
    assert (stats["min"], stats["q50"], stats["max"]) == (1.0, 2.0, 3.0)


def test_summary_strata_are_sorted_keys_over_their_own_rows():
    table = _table(
        {"x": [1.0, 10.0, 3.0, 30.0, 5.0]},
        sources=["b", "a", "b", "a", "a"],
        n_time_steps=[10, 20, 10, 10, 20],
    )
    summary = summarize_descriptors(table, by=("source_id", "n_time_steps"))
    assert summary.strata == (("a", 10), ("a", 20), ("b", 10))
    assert [summary.stats(s, "x")["mean"] for s in range(3)] == pytest.approx([30.0, 7.5, 2.0])
    with pytest.raises(ValueError, match="feature="):
        summary.table()


# ---------------------------------------------------------------------------
# bin_counts
# ---------------------------------------------------------------------------


def test_bins_are_left_closed_with_a_closed_last_bin_and_count_every_outside_reason():
    table = _table({"x": [0.0, 0.5, 1.0, 2.0, -0.1, 2.1, -INF, INF, U, INE]})
    result = bin_counts(table, {"x": [0.0, 1.0, 2.0]})
    # 0 and 0.5 fall in [0, 1); the interior edge 1 opens [1, 2]; the last edge 2 closes it.
    assert result.counts.tolist() == [[2, 2]]
    outside = {reason: int(n[0]) for reason, n in result.outside["x"].items()}
    assert outside == {"below": 2, "above": 2, "undefined": 1, "ineligible": 1}
    assert (int(result.n_outside[0]), int(result.n_stratum_worlds[0])) == (6, 10)
    assert result.locate(table)[:, 0].tolist() == [0, 0, 1, 1, -1, -1, -1, -1, -1, -1]


def test_two_feature_grid_counts_worlds_and_distinct_known_groups():
    table = _table(
        {
            "x": [0.5, 0.5, 0.5, 0.5, 1.5, 0.5, U, -1.0],
            "y": [5.0, 5.0, 5.0, 5.0, 15.0, 25.0, 5.0, U],
        },
        sources=["A", "A", "B", "A", "A", "A", "A", "B"],
        group_ids=[0, 0, 0, -1, 1, 2, 2, 1],
    )
    result = bin_counts(table, {"x": [0.0, 1.0, 2.0], "y": [0.0, 10.0, 20.0]})
    assert result.counts.tolist() == [[[4, 0], [0, 1]]]
    # Cell (0, 0) holds (A, 0) twice, (B, 0) and one unknown group: two known groups.
    assert result.group_counts.tolist() == [[[2, 0], [0, 1]]]
    # The last row is outside on both features and counted once.
    assert int(result.n_outside[0]) == 3
    x_out = {reason: int(n[0]) for reason, n in result.outside["x"].items()}
    y_out = {reason: int(n[0]) for reason, n in result.outside["y"].items()}
    assert x_out == {"below": 1, "above": 0, "undefined": 1, "ineligible": 0}
    assert y_out == {"below": 0, "above": 1, "undefined": 1, "ineligible": 0}

    other = _table({"x": [1.0, 2.0, 0.0, INF], "y": [10.0, 20.0, 20.0, 5.0]}, source="other")
    assert result.locate(other).tolist() == [[1, 1], [1, 1], [0, 1], [-1, 0]]
    relagged = _table({"x": [0.5], "y": [5.0]}, definitions={"x": _definition("x", lag=2)})
    with pytest.raises(ValueError, match="differ"):
        result.locate(relagged)


def test_bin_fractions_are_per_stratum_and_count_rows_outside_the_grid():
    table = _table({"x": [0.5, 1.5, 3.0, 0.5]}, n_time_steps=[10, 10, 10, 20])
    frame = bin_counts(table, {"x": [0.0, 1.0, 2.0]}, by=("n_time_steps",)).to_frame()
    assert frame["n_time_steps"].tolist() == [10, 10, 20, 20]
    assert frame["x_lower"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert frame["x_upper"].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert frame["n_worlds"].tolist() == [1, 1, 1, 0]
    np.testing.assert_allclose(frame["fraction"], [1 / 3, 1 / 3, 1.0, 0.0])


# ---------------------------------------------------------------------------
# DescriptorScale
# ---------------------------------------------------------------------------


def test_a_constant_reference_feature_cannot_be_standardized():
    # 0.1 is not exactly representable: the population std of identical copies
    # rounds to ~1e-17 rather than 0, so "constant" has to mean min == max. The
    # infinite row is not complete and takes no part in the fit.
    table = _table({"x": [1.0, 2.0, 3.0, 4.0], "c": [0.1, 0.1, 0.1, INF]})
    with pytest.raises(ValueError, match=r"\['c'\].*0\.1"):
        DescriptorScale.fit(table, ["x", "c"])


def test_robust_scale_uses_median_and_iqr_with_a_std_fallback():
    table = _table({"x": [1.0, 2.0, 3.0, 4.0, 100.0], "spike": [0.0, 0.0, 0.0, 0.0, 1.0]})
    robust = DescriptorScale.fit(table, ["x", "spike"])
    # x: median 3, IQR 4 - 2. spike: IQR 0, so the population std sqrt(0.8 / 5).
    np.testing.assert_allclose(robust.center, [3.0, 0.0])
    np.testing.assert_allclose(robust.scale, [2.0, 0.4])
    standard = DescriptorScale.fit(table, ["x", "spike"], method="standard")
    np.testing.assert_allclose(standard.center, [22.0, 0.2])
    np.testing.assert_allclose(standard.scale, [math.sqrt(7610 / 5), 0.4])


# ---------------------------------------------------------------------------
# nearest_worlds
# ---------------------------------------------------------------------------


def test_the_default_scale_is_fitted_on_the_reference_alone():
    reference = _table(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 10.0, 20.0, 30.0, 40.0]}, source="ref"
    )
    near = _table({"x": [2.6], "y": [26.0]}, source="qry")
    wide = _table({"x": [2.6, 1000.0, -1000.0], "y": [26.0, 1e4, -1e4]}, source="qry")
    one = nearest_worlds(reference, near, features=["x", "y"])
    three = nearest_worlds(reference, wide, features=["x", "y"])
    # Reference median (2, 20) and IQR (2, 20): the query sits at z = (0.3, 0.3)
    # and reference world 3 at (0.5, 0.5).
    assert one.neighbor_positions[0, 0] == 3
    assert one.distances[0, 0] == pytest.approx(0.2)
    np.testing.assert_allclose(one.differences[0], [-0.2, -0.2])
    assert three.distances[0, 0] == one.distances[0, 0]


def test_a_world_is_never_its_own_neighbour_and_an_exact_duplicate_is_at_zero():
    table = _table({"x": [0.0, 0.0, 5.0, 9.0]}, world_ids=[10, 11, 12, 13])
    result = nearest_worlds(table, features=["x"])
    assert result.neighbor_positions[:, 0].tolist() == [1, 0, 3, 2]
    # Median 2.5 and IQR 6 - 0 over {0, 0, 5, 9}.
    np.testing.assert_allclose(result.distances[:, 0], [0.0, 0.0, 4 / 6, 4 / 6])
    assert result.distances[0, 0] == 0.0
    sources, worlds, _ = result.neighbor_ids()
    assert worlds[:, 0].tolist() == [11, 10, 13, 12]
    assert sources[:, 0].tolist() == ["S"] * 4


def test_same_group_neighbours_are_excluded_across_sources_by_identity():
    table = _table(
        {"x": [0.0, 0.02, 1.0, 0.05, 3.0]},
        sources=["A", "A", "A", "B", "C"],
        group_ids=[0, 0, 1, 0, -1],
    )
    assert nearest_worlds(table, features=["x"]).neighbor_positions[0, 0] == 1
    grouped = nearest_worlds(table, features=["x"], exclude_same_group=True)
    # (B, 0) shares the group id but not the source, so it is admissible.
    assert grouped.neighbor_positions[0, 0] == 3
    # C's unknown-group row has no same-source candidate, so nothing is ambiguous.
    assert _labels(grouped) == ["ok"] * 5


def test_an_unknown_group_on_either_side_of_a_same_source_pair_is_reported():
    table = _table(
        {"x": [0.0, 0.02, 0.5, 3.0]},
        sources=["A", "A", "A", "B"],
        group_ids=[0, 1, -1, 0],
    )
    grouped = nearest_worlds(table, features=["x"], exclude_same_group=True)
    # Rows 0 and 1 have known groups, but the unknown-group A row might share
    # either; row 2's own group is unknown. Only B has no ambiguous candidate.
    assert _labels(grouped) == ["group_unknown", "group_unknown", "group_unknown", "ok"]
    assert grouped.neighbor_positions[:3].tolist() == [[-1], [-1], [-1]]
    assert np.isnan(grouped.distances[:3, 0]).all()
    # Without the exclusion nothing needs deciding.
    assert _labels(nearest_worlds(table, features=["x"])) == ["ok"] * 4


def test_match_keeps_neighbours_inside_equal_metadata():
    table = _table({"x": [0.0, 5.0, 0.1, 6.0]}, n_time_steps=[10, 10, 20, 20])
    assert nearest_worlds(table, features=["x"]).neighbor_positions[:, 0].tolist() == [2, 3, 0, 1]
    matched = nearest_worlds(table, features=["x"], match=("n_time_steps",))
    assert matched.neighbor_positions[:, 0].tolist() == [1, 0, 3, 2]


def test_too_few_admissible_candidates_and_incomplete_rows_get_their_own_status():
    reference = _table({"x": [0.0, 1.0, 2.0, 5.0], "y": [0.0, 1.0, 4.0, INF]})
    # (S, 0) is reference row 0 itself; the Q rows are missing a finite feature.
    query = _table(
        {"x": [0.5, 0.5, 0.5], "y": [0.5, U, INF]}, sources=["S", "Q", "Q"], world_ids=[0, 1, 2]
    )
    two = nearest_worlds(reference, query, features=["x", "y"], k=2)
    three = nearest_worlds(reference, query, features=["x", "y"], k=3)
    assert two.n_reference_candidates == 3  # the infinite reference row is not a candidate
    assert _labels(two) == ["ok", "missing_feature", "missing_feature"]
    assert two.neighbor_positions[0].tolist() == [1, 2]
    assert _labels(three) == ["no_candidates", "missing_feature", "missing_feature"]
    assert three.neighbor_positions[0].tolist() == [-1, -1, -1]
    assert np.isnan(three.distances[0]).all()


def test_subsamples_are_seeded_drawn_reference_first_and_mark_unsampled_rows():
    x = np.arange(20.0)
    table = _table({"x": x.tolist(), "y": (x**2 % 7).tolist()})

    def run(seed: int, max_query: int | None = 5) -> da.NearestWorlds:
        return nearest_worlds(
            table, features=["x", "y"], max_reference=8, max_query=max_query, seed=seed
        )

    first, again = run(3), run(3)
    np.testing.assert_array_equal(first.status, again.status)
    np.testing.assert_array_equal(first.neighbor_positions, again.neighbor_positions)
    labels = np.asarray(_labels(first))
    assert first.query_sampled
    assert (labels == "not_sampled").sum() == 15 and (labels == "ok").sum() == 5
    searched = first.reference_positions
    assert searched.size == 8 and np.all(np.diff(searched) > 0)
    assert np.isin(first.neighbor_positions[labels == "ok"], searched).all()
    # The scale comes from all 20 reference rows, never from the subsample.
    assert first.scale.center[0] == 9.5
    # The reference sample does not depend on whether the query is sampled too,
    # and sampling the query changes nothing for the rows it keeps.
    unsampled = run(3, max_query=None)
    np.testing.assert_array_equal(unsampled.reference_positions, searched)
    np.testing.assert_array_equal(
        first.neighbor_positions[labels == "ok"], unsampled.neighbor_positions[labels == "ok"]
    )
    assert len({tuple(run(seed).reference_positions) for seed in range(5)}) > 1


@pytest.mark.parametrize("block_elements", [21, 700])
def test_blocked_search_equals_a_brute_force_scan_including_ties(monkeypatch, block_elements):
    # A tiny block budget splits every query row's candidates into many blocks whose
    # running best lists must merge; a larger one packs several query rows into one
    # block. Integer points make exact distance ties common.
    points = np.random.default_rng(0).integers(0, 4, size=(30, 2)).astype(float)
    table = _table({"x": points[:, 0].tolist(), "y": points[:, 1].tolist()})
    unit = DescriptorScale(
        features=("x", "y"),
        center=np.zeros(2),
        scale=np.ones(2),
        definitions=table.definitions,
        method="unit",
    )
    monkeypatch.setattr(da, "_BLOCK_ELEMENTS", block_elements)
    result = nearest_worlds(table, features=["x", "y"], scale=unit, k=4)
    for i in range(len(points)):
        distance = np.sqrt(((points[i] - points) ** 2).mean(axis=1))
        best = sorted((distance[j], j) for j in range(len(points)) if j != i)[:4]
        assert result.neighbor_positions[i].tolist() == [j for _, j in best]
        np.testing.assert_allclose(result.distances[i], [d for d, _ in best])


# ---------------------------------------------------------------------------
# Reports and validation
# ---------------------------------------------------------------------------


def test_every_summary_is_strict_json():
    reference = _table(
        {"x": [1.0, INF, -INF, U, INE, 2.0], "y": [0.0, 1.0, 2.0, 3.0, U, 5.0]},
        group_ids=[0, 0, 1, -1, -1, 2],
        n_time_steps=[10, 10, 10, 20, 20, 20],
    )
    query = _table(
        {"x": [INF, 0.5, U], "y": [1.0, 2.0, INE]}, source="Q", n_time_steps=[10, 20, 30]
    )
    reports = [
        summarize_descriptors(reference, by=("n_time_steps",)),
        bin_counts(reference, {"x": [0.0, 1.0, 2.0], "y": [0.0, 3.0, 6.0]}, by=("source_id",)),
        nearest_worlds(reference, query, features=["y"], k=2),
        compare_descriptors(reference, query, by=("n_time_steps",)),
    ]
    for report in reports:
        payload = report.summary()
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_invalid_requests_fail_with_clear_errors():
    table = _table({"x": [0.0, 1.0, 2.0], "y": [1.0, 0.0, 2.0]})
    with pytest.raises(ValueError, match="empty"):
        summarize_descriptors(table, features=[])
    with pytest.raises(TypeError, match="sequence"):
        summarize_descriptors(table, features="x")
    with pytest.raises(KeyError, match="unknown descriptor feature"):
        summarize_descriptors(table, features=["z"])
    with pytest.raises(ValueError, match="positive"):
        nearest_worlds(table, features=["x"], k=0)
    with pytest.raises(ValueError, match="exceeds"):
        nearest_worlds(table, features=["x"], k=4)
    with pytest.raises(TypeError, match="positive"):
        nearest_worlds(table, features=["x"], k=True)
    with pytest.raises(ValueError, match="max_query"):
        nearest_worlds(table, features=["x"], max_query=0)
    with pytest.raises(TypeError, match="seed"):
        nearest_worlds(table, features=["x"], seed=1.5)
    swapped = DescriptorScale.fit(table, ["y", "x"])
    with pytest.raises(ValueError, match="scale covers"):
        nearest_worlds(table, features=["x", "y"], scale=swapped)
    with pytest.raises(ValueError, match="positive"):
        DescriptorScale(
            features=("x",),
            center=[0.0],
            scale=[0.0],
            definitions=table.definitions[:1],
            method="unit",
        )
    with pytest.raises(ValueError, match="source_id"):
        compare_descriptors(table, table, by=("source_id",))
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_counts(table, {"x": [0.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="one or two"):
        bin_counts(table, {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    relagged = _table({"x": [0.0]}, source="Q", definitions={"x": _definition("x", lag=2)})
    with pytest.raises(ValueError, match="lag"):
        compare_descriptors(table, relagged, features=["x"])
    with pytest.raises(KeyError, match="features="):
        compare_descriptors(table, relagged)  # every reference feature, but 'y' is absent


def test_rows_of_another_collection_with_the_same_world_id_are_neighbours():
    # Identity is (source_id, world_id): observed world 0 is not generated world 0.
    reference = _table({"x": [0.0, 5.0, 9.0, 20.0]}, source="ref")
    query = _table({"x": [0.0]}, source="obs")
    result = nearest_worlds(reference, query, features=["x"])
    assert result.neighbor_positions[0, 0] == 0
    assert result.distances[0, 0] == 0.0


def test_leave_self_out_removes_the_own_value_and_only_a_valid_one():
    table = _table({"x": [1.0, 2.0, 2.0, 4.0]})
    moved = _table({"x": [3.0, 1.5]}, world_ids=[3, 0])
    np.testing.assert_allclose(compare_descriptors(table, moved).percentiles[:, 0], [3 / 3, 0 / 3])
    gap = _table({"x": [1.0, U, 2.0, 4.0]})
    probe = _table({"x": [3.0]}, world_ids=[1])
    np.testing.assert_allclose(compare_descriptors(gap, probe).percentiles[:, 0], [2 / 3])


def test_subsamples_are_drawn_from_complete_rows_only():
    y = [U if i % 4 == 0 else float(i % 7) for i in range(20)]
    table = _table({"x": [float(i) for i in range(20)], "y": y})
    complete = np.array([i % 4 != 0 for i in range(20)])
    for seed in range(5):
        result = nearest_worlds(table, features=["x", "y"], max_reference=6, max_query=4, seed=seed)
        labels = np.asarray(_labels(result))
        assert result.n_reference_candidates == 15
        assert complete[result.reference_positions].all()
        assert (labels[~complete] == "missing_feature").all()
        assert (labels == "ok").sum() == 4 and (labels == "not_sampled").sum() == 11


def test_group_ambiguity_is_judged_only_among_matched_candidates():
    table = _table({"x": [0.0, 1.0, 0.5]}, group_ids=[0, 1, -1], n_time_steps=[10, 10, 20])
    result = nearest_worlds(table, features=["x"], match=("n_time_steps",), exclude_same_group=True)
    assert _labels(result) == ["ok", "ok", "no_candidates"]


def test_a_fitted_scale_uses_only_rows_complete_on_every_feature():
    table = _table({"x": [0.0, 1.0, 2.0, 3.0, 100.0], "y": [0.0, 1.0, 2.0, 3.0, INF]})
    scale = DescriptorScale.fit(table, ["x", "y"])
    np.testing.assert_allclose(scale.center, [1.5, 1.5])
    np.testing.assert_allclose(scale.scale, [1.5, 1.5])


def test_distances_between_huge_standardized_values_stay_finite_and_ordered():
    scale = DescriptorScale(
        features=("x", "y"),
        center=[0.0, 0.0],
        scale=[1.0, 1.0],
        definitions=(_definition("x"), _definition("y")),
        method="unit",
    )
    table = _table({"x": [0.0, 1e200, 3e200], "y": [0.0, 1e200, 3e200]})
    result = nearest_worlds(table, features=["x", "y"], scale=scale)
    # Squared differences overflow float64; the RMS itself must not.
    assert _labels(result) == ["ok", "ok", "ok"]
    assert result.neighbor_positions[:, 0].tolist() == [1, 0, 1]
    np.testing.assert_allclose(result.distances[:, 0], [1e200, 1e200, 2e200])
    # |z| near the float64 limit: z_q - z_c would overflow; the distance must not.
    edge = _table({"x": [1e308, -1e308, -1e308], "y": [0.0, 0.0, 1.0]}, source="ref")
    probe = _table({"x": [1e308], "y": [0.5]}, source="obs")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        far = nearest_worlds(edge, probe, features=["x", "y"], scale=scale, k=2)
    assert _labels(far) == ["ok"]
    assert far.neighbor_positions[0].tolist() == [0, 1]
    np.testing.assert_allclose(far.distances[0], [np.sqrt(0.125), np.sqrt(2.0) * 1e308])


def test_equal_distances_are_exactly_equal_and_go_to_the_lower_position():
    names = ["a", "b", "c", "d"]
    unit = DescriptorScale(
        features=tuple(names),
        center=[0.0] * 4,
        scale=[1.0] * 4,
        definitions=tuple(_definition(n) for n in names),
        method="unit",
    )
    # Both rows have squared norm 102: sqrt(102 / 4) exactly, whatever the offsets.
    columns = ([6.0, 4.0], [-4.0, 7.0], [-5.0, 1.0], [5.0, -6.0])
    reference = _table(dict(zip(names, columns)))
    origin = _table(dict.fromkeys(names, [0.0]), source="Q")
    result = nearest_worlds(reference, origin, features=names, scale=unit, k=2)
    assert result.neighbor_positions[0].tolist() == [0, 1]
    assert result.distances[0, 0] == result.distances[0, 1] == np.sqrt(102.0 / 4.0)


def _unit_scale(*names: str) -> DescriptorScale:
    return DescriptorScale(
        features=names,
        center=[0.0] * len(names),
        scale=[1.0] * len(names),
        definitions=tuple(_definition(n) for n in names),
        method="unit",
    )


def test_tiny_distances_keep_their_order():
    # Squares of 1e-170 underflow to zero; the distances must not.
    table = _table({"x": [0.0, 3e-170, 1e-170]})
    result = nearest_worlds(table, features=["x"], scale=_unit_scale("x"))
    assert result.neighbor_positions[0, 0] == 2
    np.testing.assert_allclose(result.distances[0, 0], 1e-170, rtol=1e-12)


def test_differences_at_the_float64_limit_are_signed_infinities_without_warnings():
    table = _table({"x": [1e308, -1e308]})
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = nearest_worlds(table, features=["x"], scale=_unit_scale("x"))
    assert _labels(result) == ["ok", "ok"]
    assert result.differences[:, 0].tolist() == [np.inf, -np.inf]


def test_transform_blanks_incomplete_rows_and_fit_names_unscalable_features():
    table = _table({"x": [1.0, INF, U, 1e308]})
    z, complete = _unit_scale("x").transform(table)
    assert complete.tolist() == [True, False, False, True]
    assert np.isnan(z[1:3, 0]).all() and z[0, 0] == 1.0
    tiny = DescriptorScale(
        features=("x",), center=[0.0], scale=[1e-10], definitions=(_definition("x"),), method="unit"
    )
    with np.errstate(over="raise"):
        assert tiny.transform(table)[1].tolist() == [True, False, False, False]  # z overflows
    with pytest.raises(ValueError, match=r"\['x'\]"):  # IQR = 3.4e308 overflows
        DescriptorScale.fit(_table({"x": [-1.7e308, -1.7e308, 1.7e308, 1.7e308]}), ["x"])
