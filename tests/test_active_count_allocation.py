"""Stratified coverage of active treatment × covariate counts (issue #26), without corpora.

``active_count_allocation="stratified"`` allocates corpus cells over the grid of
(active treatments, active covariates) combinations by weight. These tests pin:

* the allocation rule — exact targets and the complete set of allowed outcomes
  for hand-worked weight rows, unbiased rounding, and invariants over a sweep
  of small grids, with every composition of the cells enumerated here;
* the seed contract — independent allocation draws exactly the legacy per-cell
  counts, both through the helper the generation loops share and in the whole
  template structure loop; stratified allocation seeds its plan with one draw
  from the main stream, spawns nothing and then draws the documented sequence
  (restated here) from its own stream, so the plan follows the generator's
  state alone, and afterwards only the latent draw touches the main stream;
* config validation of the allocation and its weight matrix on an offset,
  non-square grid, the JSON recipe round trip, the template path's checks, and
  ``validate()`` checking the active ranges before the direct-null rule;
* the coverage summaries on hand-made masks.

Targets, outcome sets and counts are worked out by hand from the rule in
:mod:`pymc_generator.active_counts`. Randomness is checked by support over
fixed seed lists, never by frequency, except the documented mean tolerance of
the unbiasedness test.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
import pytest

from pymc_generator import active_count_coverage, make_scm_prior
from pymc_generator.active_counts import (
    active_count_coverage_errors,
    allocation_targets,
    draw_active_counts,
    is_stratified_allocation,
    plan_active_counts,
    stratified_cell_counts,
    summarize_active_count_coverage,
)
from pymc_generator.sampler import SCMPrior, _slice_g_active, sample_g_additive
from pymc_generator.world_model import sample_structure
from pymc_generator.world_model_template import build_cell_inputs, sample_cell_structures

#: Seeds for support checks. The rarest outcome listed below has probability
#: about 0.06, so 200 seeds miss it with probability about 2e-6.
SEEDS = range(200)

#: An offset, non-square grid: the treatment range (2, 5) clamps to rows {2, 3}
#: and the covariate range (2, 6) to columns {2, 3, 4}.
OFFSET = {
    "n_treatments": 3,
    "n_covariates": 4,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 4,
    "n_treatments_active_range": (2, 5),
    "n_covariates_active_range": (2, 6),
}
OFFSET_WEIGHTS = [[0, 1, 0], [2, 0, 1]]
#: Its 4-cell allocation. The targets 1, 2 and 1 are whole numbers, so every
#: seed gives exactly these (n_treatments_active, n_covariates_active) cells.
OFFSET_CELLS = {(2, 3): 1, (3, 2): 2, (3, 4): 1}


def _offset_cfg(**overrides: Any) -> SCMPrior:
    return make_scm_prior(**{**OFFSET, **overrides})


def _stratified_offset_cfg(**overrides: Any) -> SCMPrior:
    return _offset_cfg(
        active_count_allocation="stratified", active_count_weights=OFFSET_WEIGHTS, **overrides
    )


# -- the allocation rule --------------------------------------------------------


def _outcomes(*matrices: list[list[int]]) -> set[tuple[int, ...]]:
    """Hand-listed count matrices as row-major tuples."""
    return {tuple(itertools.chain.from_iterable(matrix)) for matrix in matrices}


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Every ordered split of ``total`` cells into ``parts`` non-negative counts."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


class Row(NamedTuple):
    """Weights, a cell total, the exact targets and every outcome the rule allows."""

    weights: list[list[float]]
    n_cells: int
    targets: list[list[Fraction | int]]
    outcomes: set[tuple[int, ...]]


ROWS = {
    # Whole-number targets are met exactly.
    "whole_targets": Row([[1, 0], [0, 3]], 8, [[2, 0], [0, 6]], _outcomes([[2, 0], [0, 6]])),
    # Lifting takes two passes: (0, 0) has share 7/63, then (0, 1) has 54/62 of
    # the 6 cells left. A single pass would also allow [[1, 0], [2, 4]].
    "repeated_lifting": Row(
        [[1, 9], [16, 37]],
        7,
        [[1, 1], [Fraction(80, 53), Fraction(185, 53)]],
        _outcomes([[1, 1], [1, 4]], [[1, 1], [2, 3]]),
    ),
    # As many cells as positive weights: one each, although weight 3 is lifted
    # only in the second pass (6/13 of the 2 cells left).
    "one_each_after_two_lifts": Row(
        [[1, 1, 1], [1, 3, 10]], 6, [[1, 1, 1], [1, 1, 1]], _outcomes([[1, 1, 1], [1, 1, 1]])
    ),
    # As many cells as positive weights: one each, however uneven the weights.
    "one_each_despite_a_heavy_weight": Row(
        [[1, 100], [1, 1]], 4, [[1, 1], [1, 1]], _outcomes([[1, 1], [1, 1]])
    ),
    # Fewer cells than combinations: every share is below one, so any two
    # combinations get the cells.
    "fewer_cells_than_combinations": Row(
        [[1, 2], [3, 4]],
        2,
        [[Fraction(1, 5), Fraction(2, 5)], [Fraction(3, 5), Fraction(4, 5)]],
        {counts for counts in itertools.product((0, 1), repeat=4) if sum(counts) == 2},
    ),
    # Exact ties: any two combinations get the extra cell.
    "uniform_ties": Row(
        [[1, 1], [1, 1]],
        6,
        [[Fraction(3, 2), Fraction(3, 2)], [Fraction(3, 2), Fraction(3, 2)]],
        set(itertools.permutations((2, 2, 1, 1))),
    ),
    # Zero weights get no cell; the two 3/2 targets share the odd one.
    "zero_weights": Row(
        [[0, 1], [1, 0]],
        3,
        [[0, Fraction(3, 2)], [Fraction(3, 2), 0]],
        _outcomes([[0, 1], [2, 0]], [[0, 2], [1, 0]]),
    ),
    # A float64 sum of these weights overflows to inf; exact fractions do not.
    "float_overflow": Row(
        [[1e308, 1e308], [5e-324, 0]], 3, [[1, 1], [1, 0]], _outcomes([[1, 1], [1, 0]])
    ),
    # Fewer cells than combinations: (0, 0)'s share 300/103 is capped at one
    # cell and the other two cells are shared out again.
    "capped_share": Row(
        [[100, 1], [1, 1]],
        3,
        [[1, Fraction(2, 3)], [Fraction(2, 3), Fraction(2, 3)]],
        {(1, *rest) for rest in itertools.permutations((1, 1, 0))},
    ),
    # Capping takes two passes: (0, 0)'s share 300/112 is capped, then (0, 1)'s
    # 20/12 of the 2 cells left. A single pass would also allow [[1, 2], [0, 0]].
    "repeated_capping": Row(
        [[100, 10], [1, 1]],
        3,
        [[1, 1], [Fraction(1, 2), Fraction(1, 2)]],
        _outcomes([[1, 1], [1, 0]], [[1, 1], [0, 1]]),
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=list(ROWS))
def test_allocation_targets_are_the_hand_worked_fractions(row):
    targets = allocation_targets(np.asarray(row.weights, dtype=np.float64), row.n_cells)
    # Fraction equality with a float is exact, so 0.2 would not pass for 1/5.
    assert targets.tolist() == row.targets


@pytest.mark.parametrize("row", ROWS.values(), ids=list(ROWS))
def test_stratified_cell_counts_reach_exactly_the_allowed_outcomes(row):
    weights = np.asarray(row.weights, dtype=np.float64)
    reached = set()
    for seed in SEEDS:
        counts = stratified_cell_counts(weights, row.n_cells, np.random.default_rng(seed))
        assert counts.shape == weights.shape
        reached.add(tuple(counts.ravel().tolist()))
    assert reached == row.outcomes


@pytest.mark.parametrize("row", ROWS.values(), ids=list(ROWS))
def test_is_stratified_allocation_accepts_exactly_the_allowed_outcomes(row):
    """Every composition of the cells over the grid is checked, not just samples."""
    weights = np.asarray(row.weights, dtype=np.float64)
    compositions = list(_compositions(row.n_cells, weights.size))
    assert len(compositions) == math.comb(row.n_cells + weights.size - 1, weights.size - 1)
    accepted = {
        counts
        for counts in compositions
        if is_stratified_allocation(np.reshape(counts, weights.shape), weights)
    }
    assert accepted == row.outcomes


def test_stratified_counts_are_unbiased_for_their_targets():
    """Mean counts over 4000 fixed seeds lie within 0.04 (over 5 standard errors) of the targets.

    Deterministic largest-remainder rounding would always give the three 6/7
    targets their cell and never the 3/7 one.
    """
    weights = np.array([[1.0, 2.0], [2.0, 2.0]])
    expected = [[Fraction(3, 7), Fraction(6, 7)], [Fraction(6, 7), Fraction(6, 7)]]
    assert allocation_targets(weights, 3).tolist() == expected
    counts = [stratified_cell_counts(weights, 3, np.random.default_rng(s)) for s in range(4000)]
    np.testing.assert_allclose(
        np.mean(counts, axis=0), np.array(expected, dtype=np.float64), rtol=0, atol=0.04
    )


def test_near_equal_weights_never_starve_a_combination():
    """Two cells over weights 1, 1.01, 1.02: each combination is left out for some seed.

    Deterministic largest-remainder rounding would never give weight 1.0 a cell.
    """
    weights = np.array([[1.0, 1.01, 1.02]])
    reached = {
        tuple(stratified_cell_counts(weights, 2, np.random.default_rng(s)).ravel().tolist())
        for s in SEEDS
    }
    assert reached == {(0, 1, 1), (1, 0, 1), (1, 1, 0)}


@pytest.mark.parametrize("shape", ((1, 1), (1, 2), (1, 3), (2, 2)))
def test_stratified_allocation_invariants_over_small_grids(shape):
    """Every weight matrix over {0, 1, 2, 3, 7} with a positive entry, and 0..3G+2 cells."""
    size = math.prod(shape)
    rng = np.random.default_rng(26)
    for flat in itertools.product((0, 1, 2, 3, 7), repeat=size):
        if not any(flat):
            continue
        weights = np.reshape(np.asarray(flat, dtype=np.float64), shape)
        positive = weights > 0
        for n_cells in range(3 * size + 3):
            targets = allocation_targets(weights, n_cells)
            assert all(isinstance(target, Fraction) for target in targets.ravel())
            assert sum(targets.ravel(), Fraction(0)) == n_cells
            counts = stratified_cell_counts(weights, n_cells, rng)
            context = f"weights={flat}, n_cells={n_cells}, counts={counts.ravel().tolist()}"
            assert counts.shape == shape, context
            assert counts.sum() == n_cells, context
            assert (counts[~positive] == 0).all(), context
            if n_cells >= positive.sum():
                assert (counts[positive] >= 1).all(), context
            else:
                assert (counts <= 1).all(), context
            # Equal weights (uniform ones included) differ by at most one cell.
            for weight in set(flat):
                same = counts[weights == weight]
                assert same.max() - same.min() <= 1, context
            assert is_stratified_allocation(counts, weights), context


# -- the seed contract ----------------------------------------------------------


@pytest.mark.parametrize("weights", (None, OFFSET_WEIGHTS), ids=("no_weights", "inert_weights"))
def test_independent_plan_touches_no_randomness(weights):
    cfg = _offset_cfg(active_count_weights=weights)
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    assert plan_active_counts(cfg, rng) is None
    assert rng.bit_generator.state == state
    assert rng.bit_generator.seed_seq.n_children_spawned == 0


#: The configured active ranges of a (3, 3, 2) layout and the effective ranges
#: they clamp to, written out by hand.
_DRAW_CASES = {
    "spread": (((1, 3), (2, 3), (1, 2)), ((1, 3), (2, 3), (1, 2))),
    "clamped": (((2, 5), (1, 9), (2, 4)), ((2, 3), (1, 3), (2, 2))),
    "degenerate": (((2, 2), (3, 3), (1, 1)), ((2, 2), (3, 3), (1, 1))),
}


@pytest.mark.parametrize("ranges, effective", _DRAW_CASES.values(), ids=list(_DRAW_CASES))
def test_independent_draws_are_the_legacy_per_cell_integers(ranges, effective):
    """Treatment, covariate, latent: one ``integers`` call each per cell, in that order."""
    cfg = make_scm_prior(
        n_treatments=3,
        n_covariates=3,
        n_latent=2,
        n_time_steps=16,
        n_cells=6,
        n_treatments_active_range=ranges[0],
        n_covariates_active_range=ranges[1],
        n_latent_active_range=ranges[2],
    )
    rng, twin = np.random.default_rng(13), np.random.default_rng(13)
    plan = plan_active_counts(cfg, rng)
    assert plan is None
    for cell in range(cfg.n_cells):
        expected = tuple(int(twin.integers(lo, hi + 1)) for lo, hi in effective)
        assert draw_active_counts(cfg, rng, plan, cell) == expected
    assert rng.bit_generator.state == twin.bit_generator.state
    assert rng.bit_generator.seed_seq.n_children_spawned == 0


def test_stratified_plan_costs_one_draw_and_leaves_the_latent_draw_on_the_main_stream():
    """The plan takes one ``integers(2**63)`` draw from ``rng`` and spawns nothing."""
    cfg = _stratified_offset_cfg(n_latent=2, n_latent_active_range=(1, 2))
    rng, twin = np.random.default_rng(17), np.random.default_rng(17)
    plan = plan_active_counts(cfg, rng)
    twin.integers(2**63)
    assert rng.bit_generator.state == twin.bit_generator.state
    assert rng.bit_generator.seed_seq.n_children_spawned == 0
    for cell in range(cfg.n_cells):
        n_treatments_active, n_covariates_active, n_latent_active = draw_active_counts(
            cfg, rng, plan, cell
        )
        assert (n_treatments_active, n_covariates_active) == tuple(plan[cell].tolist())
        assert n_latent_active == int(twin.integers(1, 3))
    assert rng.bit_generator.state == twin.bit_generator.state
    assert rng.bit_generator.seed_seq.n_children_spawned == 0


#: The offset weights' targets for five cells, row-major over the (2..3) × (2..4)
#: grid: weights 1, 2 and 1 share the cells as 5/4, 5/2 and 5/4.
_FIVE_CELL_TARGETS = [0, Fraction(5, 4), 0, Fraction(5, 2), 0, Fraction(5, 4)]


def test_stratified_plan_is_the_documented_draw_sequence():
    """The plan, restated on a twin generator: its numpy consumption is the contract.

    One ``integers(2**63)`` draw from ``rng`` seeds the plan's own stream, which
    draws a permutation of the grid, then one uniform offset, then a permutation
    of the cells. The targets' fractional parts are laid end to end in permuted
    order, and each part covering a point ``offset + k`` gets one more cell.
    Five cells over the offset weights leave targets 5/4, 5/2 and 5/4, so the
    draws decide both the rounding and the cell order.
    """
    cfg = _stratified_offset_cfg(n_cells=5)
    combinations = list(itertools.product((2, 3), (2, 3, 4)))
    roundings, first_cells = set(), set()
    for seed in range(10):
        rng, twin = np.random.default_rng(seed), np.random.default_rng(seed)
        plan = plan_active_counts(cfg, rng)
        stream = np.random.default_rng(int(twin.integers(2**63)))
        assert rng.bit_generator.state == twin.bit_generator.state
        order = stream.permutation(len(combinations))
        offset = Fraction(stream.random())
        floors = [math.floor(target) for target in _FIVE_CELL_TARGETS]
        # One point ``offset + k`` per cell the floors leave over.
        points = [offset + k for k in range(cfg.n_cells - sum(floors))]
        counts = list(floors)
        start = Fraction(0)
        for i in order:
            end = start + _FIVE_CELL_TARGETS[i] - floors[i]
            counts[i] += sum(start <= point < end for point in points)
            start = end
        expected = np.repeat(combinations, counts, axis=0)[stream.permutation(cfg.n_cells)]
        np.testing.assert_array_equal(plan, expected, err_msg=f"seed {seed}")
        roundings.add(tuple(counts))
        first_cells.add(tuple(expected[0].tolist()))
    # Every rounding the targets allow occurs, and any combination can come first.
    assert roundings == _outcomes(
        [[0, 2, 0], [2, 0, 1]], [[0, 1, 0], [3, 0, 1]], [[0, 1, 0], [2, 0, 2]]
    )
    assert first_cells == set(OFFSET_CELLS)


def test_stratified_plan_maps_weights_onto_the_offset_grid():
    """Rows are treatment counts 2..3 (from 2..5), columns covariate counts 2..4 (from 2..6)."""
    cfg = _stratified_offset_cfg()
    assert cfg.active_count_grid == ((2, 3), (2, 3, 4))
    for seed in range(10):
        plan = plan_active_counts(cfg, np.random.default_rng(seed))
        assert plan.shape == (4, 2)
        assert Counter(map(tuple, plan.tolist())) == OFFSET_CELLS


def test_stratified_plan_without_weights_is_uniform():
    """Six cells over the 2x3 grid: every combination exactly once."""
    cfg = _offset_cfg(active_count_allocation="stratified", n_cells=6)
    plan = plan_active_counts(cfg, np.random.default_rng(0))
    assert Counter(map(tuple, plan.tolist())) == dict.fromkeys(
        itertools.product((2, 3), (2, 3, 4)), 1
    )


def test_stratified_plan_is_seeded_and_shuffled_over_the_cells():
    """Equal seeds give equal plans; any combination can come first, so cell order is random."""
    cfg = _stratified_offset_cfg()
    for seed in range(5):
        np.testing.assert_array_equal(
            plan_active_counts(cfg, np.random.default_rng(seed)),
            plan_active_counts(cfg, np.random.default_rng(seed)),
        )
    first = {tuple(plan_active_counts(cfg, np.random.default_rng(s))[0].tolist()) for s in SEEDS}
    assert first == set(OFFSET_CELLS)


def test_stratified_plan_depends_only_on_the_generator_state():
    """Generators in one state plan alike, whatever seed sequence they carry.

    A state copied onto a differently seeded generator keeps that generator's
    seed sequence, and every ``PCG64.jumped()`` copy gets a fresh OS-entropy
    one, so a plan drawn from a spawned stream would differ in both pairs. Nine
    cells over the uniform 2x3 grid have fractional targets, so plans vary.
    """
    cfg = _offset_cfg(active_count_allocation="stratified", n_cells=9)
    for seed in range(5):
        copied = np.random.Generator(np.random.PCG64(seed + 1))
        copied.bit_generator.state = np.random.PCG64(seed).state
        np.testing.assert_array_equal(
            plan_active_counts(cfg, copied),
            plan_active_counts(cfg, np.random.Generator(np.random.PCG64(seed))),
        )
        np.testing.assert_array_equal(
            plan_active_counts(cfg, np.random.Generator(np.random.PCG64(seed).jumped())),
            plan_active_counts(cfg, np.random.Generator(np.random.PCG64(seed).jumped())),
        )


def test_stratified_plan_takes_a_numpy_unsigned_cell_count():
    """``n_cells=np.uint64(5)`` plans exactly as ``n_cells=5``.

    ``Generator.permutation`` of a ``np.uint64`` returns float indices, which
    cannot shuffle the cells. Five cells over the offset weights leave
    fractional targets, so the plans vary with the seed.
    """
    unsigned = dataclasses.replace(_stratified_offset_cfg(), n_cells=np.uint64(5))
    signed = dataclasses.replace(unsigned, n_cells=5)
    for seed in range(10):
        np.testing.assert_array_equal(
            plan_active_counts(unsigned, np.random.default_rng(seed)),
            plan_active_counts(signed, np.random.default_rng(seed)),
        )


# -- config validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "allocation", ("Stratified", "uniform", "", None, 1, np.array("stratified"))
)
def test_unknown_allocation_is_rejected(allocation):
    """Only the two strings pass: a 0-d array equal to "stratified" is not a str."""
    with pytest.raises(ValueError, match="active_count_allocation"):
        _offset_cfg(active_count_allocation=allocation)


@pytest.mark.parametrize(
    "weights",
    (
        [[0, 2], [1, 0], [0, 1]],
        [[0, 1, 0], [2, 0, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 1, 0, 1, 1], [2, 0, 1, 1, 1]],
        [0, 1, 0, 2, 0, 1],
    ),
    ids=("transposed_3x2", "raw_treatment_range_4x3", "raw_covariate_range_2x5", "flat"),
)
def test_weights_must_match_the_effective_grid(weights):
    """The error names the expected 2x3 shape and both axes' effective counts."""
    with pytest.raises(ValueError, match="active_count_weights") as raised:
        _offset_cfg(active_count_allocation="stratified", active_count_weights=weights)
    message = str(raised.value)
    assert "2x3" in message
    assert re.search(r"n_treatments_active\D*2\D+3", message)
    assert re.search(r"n_covariates_active\D*2\D+4", message)


_INVALID_WEIGHTS = {
    "negative": [[0, 1, 0], [2, 0, -1]],
    "nan": [[0, 1, 0], [2, 0, math.nan]],
    "inf": [[0, 1, 0], [2, 0, math.inf]],
    "all_zero": [[0, 0, 0], [0, 0, 0]],
    "bool_entry": [[0, True, 0], [2, 0, 1]],
    "str_entry": [[0, "1", 0], [2, 0, 1]],
    "ragged": [[0, 1, 0], [2, 0]],
    "overflowing_int": [[0, 1, 0], [2, 0, 10**400]],
}


@pytest.mark.parametrize("weights", _INVALID_WEIGHTS.values(), ids=list(_INVALID_WEIGHTS))
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="active_count_weights"):
        _offset_cfg(active_count_allocation="stratified", active_count_weights=weights)


#: Arrays of any dtype or subclass are refused with a hint to pass
#: ``weights.tolist()``: they would break ``SCMPrior ==`` and the JSON recipe,
#: and a masked array hides its negative entry from the value checks.
_ARRAY_WEIGHTS = {
    "int_array": np.array(OFFSET_WEIGHTS),
    "float_array": np.array(OFFSET_WEIGHTS, dtype=np.float64),
    "object_array": np.array(OFFSET_WEIGHTS, dtype=object),
    "bool_array": np.array(OFFSET_WEIGHTS, dtype=bool),
    "str_array": np.array(OFFSET_WEIGHTS).astype(str),
    "masked_array": np.ma.masked_less([[0, 1, 0], [2, 0, -1]], 0),
    # A view, because constructing np.matrix warns of its pending deprecation.
    "matrix": np.array(OFFSET_WEIGHTS).view(np.matrix),
}


@pytest.mark.parametrize("weights", _ARRAY_WEIGHTS.values(), ids=list(_ARRAY_WEIGHTS))
def test_weight_arrays_are_rejected_with_a_tolist_hint(weights):
    with pytest.raises(ValueError, match=r"active_count_weights .*weights\.tolist\(\)"):
        _offset_cfg(active_count_allocation="stratified", active_count_weights=weights)


_VALID_WEIGHTS = {
    "nested_lists": OFFSET_WEIGHTS,
    "tuples": ((0, 1, 0), (2, 0, 1)),
    "numpy_scalars": [[np.float32(0), np.int64(1), 0.0], [np.float64(2), 0, np.uint8(1)]],
}


@pytest.mark.parametrize("weights", _VALID_WEIGHTS.values(), ids=list(_VALID_WEIGHTS))
def test_accepted_weight_forms_give_the_same_matrix(weights):
    cfg = _offset_cfg(active_count_allocation="stratified", active_count_weights=weights)
    matrix = cfg.active_count_weight_matrix()
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix, OFFSET_WEIGHTS)


@pytest.mark.parametrize(
    "weights", ([[0, 2], [1, 0], [0, 1]], [[0, 1, 0], [2, 0, -1]]), ids=("shape", "negative")
)
def test_weights_are_validated_without_stratification_too(weights):
    """Inert weights still have to be valid (their inertness is pinned with the seed contract)."""
    with pytest.raises(ValueError, match="active_count_weights"):
        _offset_cfg(active_count_weights=weights)


def test_stratified_config_round_trips_through_a_json_recipe():
    cfg = _stratified_offset_cfg()
    loaded = SCMPrior(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    loaded.validate()
    assert loaded.active_count_allocation == "stratified"
    assert loaded.active_count_weights == OFFSET_WEIGHTS
    np.testing.assert_array_equal(
        plan_active_counts(loaded, np.random.default_rng(3)),
        plan_active_counts(cfg, np.random.default_rng(3)),
    )


@pytest.mark.parametrize(
    "allocation, change",
    (
        ("stratified", {"active_count_allocation": "Stratified"}),
        ("stratified", {"active_count_allocation": np.array("stratified")}),
        ("stratified", {"active_count_weights": [[0, 2], [1, 0], [0, 1]]}),
        ("stratified", {"active_count_weights": [[0, 1, 0], [2, 0, -1]]}),
        ("independent", {"active_count_weights": [[0, 2], [1, 0], [0, 1]]}),
        ("independent", {"active_count_weights": [[0, 1, 0], [2, 0, -1]]}),
    ),
    ids=(
        "allocation",
        "allocation_array",
        "weights_shape",
        "negative_weight",
        "inert_weights_shape",
        "inert_negative_weight",
    ),
)
def test_template_structures_check_the_allocation_themselves(allocation, change):
    """``sample_cell_structures`` never calls ``validate()``, so the plan checks what it would.

    That covers the unused weights of an independent config, and a 0-d array
    that compares equal to ``"stratified"`` without being a ``str``.
    """
    base = _offset_cfg(active_count_allocation=allocation, active_count_weights=OFFSET_WEIGHTS)
    cfg = dataclasses.replace(base, **change)
    (field,) = change  # the error names the changed field
    with pytest.raises(ValueError, match=field):
        sample_cell_structures(cfg, np.random.default_rng(0))


#: Active ranges ``validate()`` rejects: empty, from zero, and with float bounds.
_INVALID_RANGES = {"empty": (3, 2), "from_zero": (0, 2), "float_bounds": (2.0, 3.0)}


@pytest.mark.parametrize("bounds", _INVALID_RANGES.values(), ids=list(_INVALID_RANGES))
@pytest.mark.parametrize("allocation", ("independent", "stratified"))
@pytest.mark.parametrize("field", ("n_treatments_active_range", "n_covariates_active_range"))
def test_template_structures_name_an_invalid_active_range(field, allocation, bounds):
    """A range ``validate()`` rejects fails as a ``ValueError`` naming it, in both modes.

    ``sample_cell_structures`` skips ``validate()``, so the plan checks the
    ranges as it would. Unchecked, (0, 2) can silently build cells with none of
    those slots active, float bounds fail as a ``TypeError``, and (3, 2) fails
    inside ``Generator.integers`` or indexes an empty grid.
    """
    cfg = dataclasses.replace(_offset_cfg(active_count_allocation=allocation), **{field: bounds})
    with pytest.raises(ValueError, match=field):
        sample_cell_structures(cfg, np.random.default_rng(0))


@pytest.mark.parametrize("bounds", (None, ("a", 3)), ids=("none", "str_bound"))
def test_validate_checks_the_active_ranges_before_the_direct_null_rule(bounds):
    """The range error comes first: the direct-null rule reads ``n_treatments_active_range[0]``.

    Checked only by the weight matrix after that rule, these ranges would fail
    there as a ``TypeError`` (``None`` is not indexable, ``"a"`` is not
    comparable with the floor).
    """
    cfg = SCMPrior(n_treatments_active_range=bounds, min_no_direct_effect_treatments=1)
    with pytest.raises(ValueError, match="^n_treatments_active_range must"):
        cfg.validate()


# -- the template path ----------------------------------------------------------


def _legacy_cell_structures(cfg: SCMPrior, rng: np.random.Generator) -> list[dict]:
    """The pre-#26 template structure loop, restated: its numpy consumption is the contract."""
    cells = []
    for _ in range(cfg.n_cells):
        n_treatments_active, n_covariates_active, n_latent_active = (
            int(rng.integers(lo, hi + 1))
            for lo, hi in (
                cfg.n_treatments_active_range_effective,
                cfg.n_covariates_active_range_effective,
                cfg.n_latent_active_range_effective,
            )
        )
        g = sample_g_additive(
            rng,
            cfg,
            cfg.layout,
            n_treatments_active=n_treatments_active,
            n_covariates_active=n_covariates_active,
            n_latent_active=n_latent_active,
        )
        g_act = _slice_g_active(g, n_treatments_active, n_covariates_active, n_latent_active)
        structural = sample_structure(g_act, cfg, rng)
        active = {key: g[key] for key in ("active_treatment", "active_covariate", "active_latent")}
        cells.append(build_cell_inputs(cfg, g, active, structural))
    return cells


@pytest.mark.parametrize(
    "weights", (None, [[1, 0, 2], [0, 3, 1]]), ids=("no_weights", "inert_weights")
)
def test_independent_template_structures_keep_the_legacy_draws(weights):
    """Payloads, final RNG state and spawn count are the pre-#26 ones, weights or not."""
    cfg = make_scm_prior(
        n_treatments=3,
        n_covariates=3,
        n_latent=2,
        n_time_steps=16,
        n_cells=6,
        n_treatments_active_range=(2, 3),
        n_covariates_active_range=(1, 3),
        n_latent_active_range=(1, 2),
        active_count_weights=weights,
    )
    rng, twin = np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed)
    cells = sample_cell_structures(cfg, rng)
    expected = _legacy_cell_structures(cfg, twin)

    assert rng.bit_generator.state == twin.bit_generator.state
    assert rng.bit_generator.seed_seq.n_children_spawned == 0
    assert twin.bit_generator.seed_seq.n_children_spawned == 0
    assert len(cells) == len(expected) == cfg.n_cells
    for got, want in zip(cells, expected, strict=True):
        assert got.keys() == want.keys()
        for name, value in want.items():
            np.testing.assert_array_equal(got[name], value, err_msg=name)
            assert got[name].dtype == value.dtype, name


def test_stratified_template_structures_follow_the_weights():
    """The payload activity masks realise the offset grid's exact allocation."""
    cfg = _stratified_offset_cfg()
    cells = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))
    realised = Counter(
        (int(cell["active_treatment"].sum()), int(cell["active_covariate"].sum())) for cell in cells
    )
    assert realised == OFFSET_CELLS


# -- coverage summaries ---------------------------------------------------------

#: (cell_id, active treatments, active covariates) per task: cell 0 has two
#: tasks at (2, 1), and (3, 1) has a full cell plus a partial one-task cell.
_TASKS = ((0, 2, 1), (0, 2, 1), (1, 3, 3), (1, 3, 3), (2, 3, 1), (2, 3, 1), (3, 3, 1))
#: Over the grid rows 2..3 and columns 1..3.
_CELLS = [[1, 0, 0], [2, 0, 1]]
_WORLDS = [[2, 0, 0], [3, 0, 2]]
#: Tasks whose active covariate counts start at 2: (3, 2) holds cell 0 (two
#: tasks) and cell 3, while (2, 3) and (3, 3) hold one cell each.
_OFFSET_TASKS = ((0, 3, 2), (0, 3, 2), (1, 2, 3), (2, 3, 3), (3, 3, 2))
#: Tasks on 3 treatment slots but only 2 covariate slots: cell 0 at (2, 2), and
#: cells 1 (two tasks) and 2 at (3, 1).
_NARROW_TASKS = ((0, 2, 2), (1, 3, 1), (1, 3, 1), (2, 3, 1))
_NARROW_WIDTHS = (3, 2)


def _masks(tasks, widths=(3, 3)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefix activity masks of (treatment, covariate) ``widths`` and cell ids, in corpus dtypes."""
    n_treatments, n_covariates = widths
    treatment = np.array([[1] * t + [0] * (n_treatments - t) for _, t, _ in tasks], dtype=np.uint8)
    covariate = np.array([[1] * c + [0] * (n_covariates - c) for _, _, c in tasks], dtype=np.uint8)
    cell_id = np.array([cell for cell, _, _ in tasks], dtype=np.int32)
    return treatment, covariate, cell_id


def _corpus(tasks, widths=(3, 3)) -> dict[str, Any]:
    treatment, covariate, cell_id = _masks(tasks, widths)
    return {
        "treatment_active_mask": treatment,
        "covariate_active_mask": covariate,
        "cell_id": cell_id,
        "diagnostics": {},
    }


def _plain_ints(value) -> bool:
    if isinstance(value, list):
        return all(_plain_ints(item) for item in value)
    return type(value) is int


def _prior(
    treatment_range: tuple[int, int], covariate_range: tuple[int, int], **layout: int
) -> SCMPrior:
    """A prior over 3 treatment and 3 covariate slots, unless ``layout`` resizes them."""
    return make_scm_prior(
        **{"n_treatments": 3, "n_covariates": 3, "n_latent": 1, **layout},
        n_treatments_active_range=treatment_range,
        n_covariates_active_range=covariate_range,
    )


def test_summary_counts_distinct_cells_and_tasks_per_combination():
    """A cell counts once however many tasks it holds; a task outside the grid counts nowhere."""
    treatment, covariate, cell_id = _masks((*_TASKS, (4, 1, 2), (4, 1, 2)))
    summary = summarize_active_count_coverage(treatment, covariate, cell_id, (2, 3), (1, 2, 3))
    assert summary == {"n_cells": _CELLS, "n_worlds": _WORLDS}
    assert _plain_ints(list(summary.values()))


def test_coverage_columns_follow_covariate_labels_that_start_at_two():
    """Column ``j`` counts ``covariate_counts[j]`` active covariates, not ``j + 1``.

    The prior's covariate range (2, 6) clamps to the 3 slots: columns 2..3.
    """
    treatment, covariate, cell_id = _masks(_OFFSET_TASKS)
    counts = {"n_cells": [[0, 1], [2, 1]], "n_worlds": [[0, 1], [3, 1]]}
    assert summarize_active_count_coverage(treatment, covariate, cell_id, (2, 3), (2, 3)) == counts
    assert active_count_coverage(_corpus(_OFFSET_TASKS), prior=_prior((2, 3), (2, 6))) == {
        "n_treatments_active": [2, 3],
        "n_covariates_active": [2, 3],
        **counts,
    }


def test_coverage_over_a_wider_prior_grid_reports_unreached_combinations_as_zeros():
    got = active_count_coverage(_corpus(_TASKS), prior=_prior((1, 3), (1, 3)))
    assert got == {
        "n_treatments_active": [1, 2, 3],
        "n_covariates_active": [1, 2, 3],
        "n_cells": [[0, 0, 0], *_CELLS],
        "n_worlds": [[0, 0, 0], *_WORLDS],
    }
    assert _plain_ints(list(got.values()))


def test_coverage_rejects_tasks_outside_the_prior_grid():
    with pytest.raises(ValueError, match="outside"):
        active_count_coverage(_corpus(_TASKS), prior=_prior((2, 3), (1, 2)))


@pytest.mark.parametrize(
    ("treatment_range", "covariate_range"),
    (((2, 4), (1, 2)), ((2, 3), (1, 3))),
    ids=("4-treatments-on-3-slots", "3-covariates-on-2-slots"),
)
def test_coverage_rejects_a_prior_grid_beyond_the_corpus_slots(treatment_range, covariate_range):
    """Every task lies inside the grid, but the grid lists counts the slots cannot hold."""
    prior = _prior(treatment_range, covariate_range, n_treatments=4)
    with pytest.raises(ValueError, match="does not fit"):
        active_count_coverage(_corpus(_NARROW_TASKS, _NARROW_WIDTHS), prior=prior)


def test_coverage_without_a_prior_uses_the_stored_grid_and_recounts():
    """The stored counts are stale on purpose: the masks are what gets counted."""
    corpus = _corpus(_TASKS)
    corpus["diagnostics"]["active_count_coverage"] = {
        "allocation": "stratified",
        "n_treatments_active": [1, 2, 3],
        "n_covariates_active": [1, 2, 3],
        "weights": np.ones((3, 3)).tolist(),
        "n_cells": np.zeros((3, 3), dtype=int).tolist(),
        "n_worlds": np.zeros((3, 3), dtype=int).tolist(),
    }
    assert active_count_coverage(corpus) == {
        "n_treatments_active": [1, 2, 3],
        "n_covariates_active": [1, 2, 3],
        "n_cells": [[0, 0, 0], *_CELLS],
        "n_worlds": [[0, 0, 0], *_WORLDS],
    }
    # A prior's grid wins over the stored one.
    assert active_count_coverage(corpus, prior=_prior((2, 3), (1, 3))) == {
        "n_treatments_active": [2, 3],
        "n_covariates_active": [1, 2, 3],
        "n_cells": _CELLS,
        "n_worlds": _WORLDS,
    }


def test_coverage_needs_a_prior_or_a_stored_grid():
    with pytest.raises(ValueError, match="prior"):
        active_count_coverage(_corpus(_TASKS))


def _stored_block(**changes: Any) -> dict[str, Any]:
    """A stored coverage block over rows 2..3 and columns 1..3 that counts ``_TASKS``."""
    block = {
        "allocation": "stratified",
        "n_treatments_active": [2, 3],
        "n_covariates_active": [1, 2, 3],
        "weights": np.ones((2, 3)).tolist(),
        "n_cells": _CELLS,
        "n_worlds": _WORLDS,
    }
    return {**block, **changes}


#: Stored blocks whose grid cannot be read: not a mapping, an axis missing, or an
#: axis that is not consecutive integers (``int()`` would truncate the floats and
#: parse the string, and a repeated count would leave a falsely empty row).
_MALFORMED_GRIDS = {
    "null_block": None,
    "axis_missing": {k: v for k, v in _stored_block().items() if k != "n_covariates_active"},
    "float_axis": _stored_block(n_treatments_active=[2.7, 3.9]),
    "str_axis": _stored_block(n_treatments_active=["2"]),
    "repeated_axis": _stored_block(n_treatments_active=[2, 2, 3]),
}


@pytest.mark.parametrize("block", _MALFORMED_GRIDS.values(), ids=list(_MALFORMED_GRIDS))
def test_coverage_rejects_a_malformed_stored_grid(block):
    """The error points to ``validate_corpus``; passing the prior still reports the corpus."""
    corpus = _corpus(_TASKS)
    corpus["diagnostics"]["active_count_coverage"] = block
    with pytest.raises(ValueError, match="validate_corpus"):
        active_count_coverage(corpus)
    assert active_count_coverage(corpus, prior=_prior((2, 3), (1, 3)))["n_cells"] == _CELLS


#: Coverage of ``_NARROW_TASKS`` over rows 2..3 and columns 1..2, weighted by its
#: own cells, and the same coverage widened by an empty third covariate column.
_NARROW_BLOCK = {
    "allocation": "stratified",
    "n_treatments_active": [2, 3],
    "n_covariates_active": [1, 2],
    "weights": [[0.0, 1.0], [2.0, 0.0]],
    "n_cells": [[0, 1], [2, 0]],
    "n_worlds": [[0, 1], [3, 0]],
}
_WIDENED_BLOCK = {
    **_NARROW_BLOCK,
    "n_covariates_active": [1, 2, 3],
    "weights": [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
    "n_cells": [[0, 1, 0], [2, 0, 0]],
    "n_worlds": [[0, 1, 0], [3, 0, 0]],
}


def test_validator_bounds_each_axis_by_its_own_mask_width():
    """3 treatment and 2 covariate slots: a third covariate column is out of bounds.

    Bounding either axis by the other mask's width would accept the widened
    block or reject the narrow one.
    """
    treatment, covariate, cell_id = _masks(_NARROW_TASKS, _NARROW_WIDTHS)
    assert active_count_coverage_errors(_NARROW_BLOCK, treatment, covariate, cell_id) == []
    assert active_count_coverage_errors(_WIDENED_BLOCK, treatment, covariate, cell_id) == [
        "diagnostics active_count_coverage n_covariates_active must list consecutive integer "
        "counts within [1, 2]"
    ]


def test_coverage_bounds_each_axis_by_its_own_mask_width():
    """3 treatment and 2 covariate slots: the narrow grid is read, the widened one refused."""
    corpus = _corpus(_NARROW_TASKS, _NARROW_WIDTHS)
    corpus["diagnostics"]["active_count_coverage"] = _NARROW_BLOCK
    report = {
        "n_treatments_active": [2, 3],
        "n_covariates_active": [1, 2],
        "n_cells": [[0, 1], [2, 0]],
        "n_worlds": [[0, 1], [3, 0]],
    }
    assert active_count_coverage(corpus) == report
    assert active_count_coverage(corpus, prior=_prior((2, 3), (1, 2))) == report
    corpus["diagnostics"]["active_count_coverage"] = _WIDENED_BLOCK
    with pytest.raises(ValueError, match="validate_corpus"):
        active_count_coverage(corpus)
