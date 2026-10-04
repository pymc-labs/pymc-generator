"""Stratified coverage of active treatment × covariate counts in corpora (issue #26).

Every oracle here is independent of the production counters: cells and worlds
per (active treatments, active covariates) combination are recounted in each
test from the stored ``cell_id`` and active-mask sums, and the expected
allocations are listed by hand. The validator rows attach a synthetic block to
an ordinary (independent) corpus, so they pin ``validate_corpus`` independently
of the generation code that emits the block. One test checks the template
path's per-cell active counts against a corpus generated from the same seed,
and one restates the corpus RNG's draws up to cell 0's graph.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import replace

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator import DataGenerator, make_scm_prior
from pymc_generator.sampler import sample_g_additive
from pymc_generator.world_model_template import sample_cell_structures

#: Two draws per cell, cut to 13 tasks: ``n=13`` re-derives seven cells from the
#: configured forty, so the allocation covers exactly the cells kept and the
#: last one keeps a single task. The treatment range (2, 5) clamps to
#: n_treatments=3, giving a non-square 2x3 grid on which a transposed block
#: cannot match. Same-seed independent draws leave two combinations empty, so a
#: bypassed allocator fails the coverage test.
_UNIFORM = {
    "n_treatments": 3,
    "n_covariates": 3,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 40,
    "draws_per_cell": 2,
    "seed": 3,
    "n_treatments_active_range": (2, 5),
    "n_covariates_active_range": (1, 3),
    "active_count_allocation": "stratified",
}
_UNIFORM_N = 13
_UNIFORM_GRID = ((2, 3), (1, 2, 3))

#: Four cells over weights [[1, 0], [1, 2]]: whole-number targets, met exactly.
#: The weights are tuples of ints, which the block must echo as lists of floats.
#: Same-seed independent draws give another allocation here (seed 5's happen
#: to match), so a bypassed allocator fails the weighted test.
_WEIGHTED = {
    "n_treatments": 3,
    "n_covariates": 2,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 4,
    "draws_per_cell": 1,
    "seed": 6,
    "n_treatments_active_range": (2, 3),
    "n_covariates_active_range": (1, 2),
    "active_count_allocation": "stratified",
    "active_count_weights": ((1, 0), (1, 2)),
}

#: The default (independent) allocation: four cells cannot cover six combinations.
_INDEPENDENT = {
    "n_treatments": 3,
    "n_covariates": 3,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 4,
    "draws_per_cell": 2,
    "seed": 11,
    "n_treatments_active_range": (2, 3),
    "n_covariates_active_range": (1, 3),
}
_INDEPENDENT_GRID = ((2, 3), (1, 2, 3))

_BLOCK = "diagnostics active_count_coverage"
MAPPING = f"{_BLOCK} must be a mapping"
KEYS = (
    f"{_BLOCK} must have exactly the keys allocation, n_treatments_active, "
    "n_covariates_active, weights, n_cells, n_worlds"
)
ALLOCATION = f"{_BLOCK} allocation must be 'stratified'"
TREATMENT_AXIS = f"{_BLOCK} n_treatments_active must list consecutive integer counts within [1, 3]"
COVARIATE_AXIS = f"{_BLOCK} n_covariates_active must list consecutive integer counts within [1, 3]"
WEIGHTS = (
    f"{_BLOCK} weights must be a 2x3 matrix of finite non-negative numbers with a positive entry"
)
GRID = f"{_BLOCK} grid does not contain every task's active counts"
STALE = f"{_BLOCK} {{}} does not match recomputation"
NOT_ALLOCATION = f"{_BLOCK} n_cells is not a stratified allocation of its weights"


def _by_combination(combinations, counts):
    return {(int(t), int(c)): int(n) for (t, c), n in zip(combinations, counts, strict=True)}


def _recount(corpus):
    """Cells and worlds per realised (active treatments, active covariates), counted here.

    A cell is a distinct ``cell_id``, so a truncated cell counts once like any other.
    """
    counts = np.stack(
        [
            corpus["treatment_active_mask"].sum(axis=1, dtype=np.int64),
            corpus["covariate_active_mask"].sum(axis=1, dtype=np.int64),
        ],
        axis=1,
    )
    combinations, n_worlds = np.unique(counts, axis=0, return_counts=True)
    cells = np.unique(np.column_stack([corpus["cell_id"], counts]), axis=0)
    cell_combinations, n_cells = np.unique(cells[:, 1:], axis=0, return_counts=True)
    return (
        _by_combination(cell_combinations, n_cells),
        _by_combination(combinations, n_worlds),
    )


def _on_grid(counts, grid):
    """``counts`` indexed ``[t - t_lo][c - c_lo]`` over ``grid``, zero where absent."""
    treatments, covariates = grid
    return [[counts.get((t, c), 0) for c in covariates] for t in treatments]


def _report(corpus, grid):
    """The coverage report expected over ``grid``, from the in-test recount."""
    n_cells, n_worlds = _recount(corpus)
    return {
        "n_treatments_active": list(grid[0]),
        "n_covariates_active": list(grid[1]),
        "n_cells": _on_grid(n_cells, grid),
        "n_worlds": _on_grid(n_worlds, grid),
    }


@pytest.fixture(scope="module")
def uniform():
    cfg = make_scm_prior(**_UNIFORM)
    return cfg, pg.sample_prior_predictive(cfg, n=_UNIFORM_N)


@pytest.fixture(params=("in_memory", "loaded"))
def stored_uniform(request, uniform, tmp_path):
    """The uniform corpus as generated, and after a save/load round trip."""
    _, corpus = uniform
    if request.param == "loaded":
        path = tmp_path / "uniform.npz"
        pg.save_corpus(corpus, path)
        corpus = pg.load_corpus(path)
    return corpus


@pytest.fixture(scope="module")
def weighted():
    cfg = make_scm_prior(**_WEIGHTED)
    return cfg, pg.sample_prior_predictive(cfg)


@pytest.fixture(scope="module")
def independent():
    cfg = make_scm_prior(**_INDEPENDENT)
    return cfg, pg.sample_prior_predictive(cfg)


# -- stratified corpora ----------------------------------------------------------


def test_uniform_allocation_covers_every_combination_within_one_cell(uniform):
    """Seven cells over six combinations: one each, plus one extra somewhere.

    Independent per-cell draws give this multiset only ~5% of the time.
    """
    _, corpus = uniform
    n_cells, _ = _recount(corpus)
    grid = list(itertools.product(*_UNIFORM_GRID))
    assert set(n_cells) <= set(grid)
    assert sorted(n_cells.get(combination, 0) for combination in grid) == [1, 1, 1, 1, 1, 2]


def test_coverage_block_counts_match_the_stored_masks(stored_uniform):
    """Indexed ``[t - 2][c - 1]``; on the 2x3 grid a transposed block cannot match."""
    block = stored_uniform["diagnostics"]["active_count_coverage"]
    n_cells, n_worlds = _recount(stored_uniform)
    assert block["n_cells"] == _on_grid(n_cells, _UNIFORM_GRID)
    assert block["n_worlds"] == _on_grid(n_worlds, _UNIFORM_GRID)
    # Counted after truncation: 13 retained tasks, and the one-task last cell is a cell.
    assert sum(map(sum, block["n_worlds"])) == _UNIFORM_N
    assert sum(map(sum, block["n_cells"])) == 7


def test_coverage_block_echoes_the_grid_and_weights_as_plain_values(stored_uniform):
    block = stored_uniform["diagnostics"]["active_count_coverage"]
    assert block["allocation"] == "stratified"
    assert block["n_treatments_active"] == [2, 3]
    assert block["n_covariates_active"] == [1, 2, 3]
    assert block["weights"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert all(type(w) is float for row in block["weights"] for w in row)
    for key in ("n_treatments_active", "n_covariates_active"):
        assert all(type(count) is int for count in block[key])
    for key in ("n_cells", "n_worlds"):
        assert all(type(count) is int for row in block[key] for count in row)


def test_stratified_corpus_validates(stored_uniform):
    assert DataGenerator.validate_corpus(stored_uniform) == []


def test_template_structures_plan_the_corpus_cell_counts(uniform):
    """For one seed the template path plans the corpus's active counts, cell by cell.

    ``docs/reference/config.md`` promises this. ``n=13`` generates seven cells
    instead of the configured forty, so the template plans seven too. Both the
    cell order and the combination holding the extra cell are random, so a plan
    from any other stream, or over any other cell count, would almost never match.
    The template's config carries a different seed: the plan must follow the
    caller's generator, not ``cfg.seed``.
    """
    cfg, corpus = uniform
    _, first_rows = np.unique(corpus["cell_id"], return_index=True)
    corpus_counts = [
        (
            int(corpus["treatment_active_mask"][row].sum()),
            int(corpus["covariate_active_mask"][row].sum()),
        )
        for row in first_rows
    ]
    template_cfg = replace(cfg, n_cells=7, seed=cfg.seed + 1)
    template_counts = [
        (int(cell["active_treatment"].sum()), int(cell["active_covariate"].sum()))
        for cell in sample_cell_structures(template_cfg, np.random.default_rng(cfg.seed))
    ]
    assert corpus_counts == template_counts


@pytest.mark.parametrize(
    ("fixture", "n_cells"),
    # ``n=13`` generates seven of the uniform prior's forty cells.
    (pytest.param("uniform", 7, id="uniform"), pytest.param("weighted", 4, id="weighted")),
)
def test_stratified_corpus_draws_cell_zero_right_after_the_plan_seed(request, fixture, n_cells):
    """Restated consumption: the plan's seed is the only corpus-RNG draw before cell 0.

    On a twin of the corpus stream, the plan's ``integers(2**63)`` seed draw,
    then cell 0's latent count, then its graph at the stored cell-0 counts give
    the stored cell-0 ``g`` row. Both priors have one latent slot, so the latent
    draw ``integers(1, 2)`` consumes nothing and its place is not pinned here.
    """
    cfg, corpus = request.getfixturevalue(fixture)
    cfg = replace(cfg, n_cells=n_cells)
    twin = np.random.default_rng(cfg.seed)
    twin.integers(2**63)
    latent_lo, latent_hi = cfg.n_latent_active_range_effective
    n_latent_active = int(twin.integers(latent_lo, latent_hi + 1))
    g = sample_g_additive(
        twin,
        cfg,
        cfg.layout,
        n_treatments_active=int(corpus["n_treatments_active"][0]),
        n_covariates_active=int(corpus["n_covariates_active"][0]),
        n_latent_active=n_latent_active,
    )
    assert corpus["cell_id"][0] == 0
    assert corpus["n_latent_active"][0] == n_latent_active
    packed = cfg.layout.pack(**{f"g_{edge}": g[f"g_{edge}"] for edge in cfg.layout.edge_types})
    np.testing.assert_array_equal(corpus["g"][0], packed)


def test_weighted_allocation_honours_the_weights(weighted):
    """Independent per-cell draws give this exact allocation only ~5% of the time."""
    _, corpus = weighted
    n_cells, n_worlds = _recount(corpus)
    assert n_cells == n_worlds == {(2, 1): 1, (3, 1): 1, (3, 2): 2}
    block = corpus["diagnostics"]["active_count_coverage"]
    assert all(type(w) is float for row in block["weights"] for w in row)
    assert block["weights"] == [[1.0, 0.0], [1.0, 2.0]]
    assert block["n_cells"] == block["n_worlds"] == [[1, 0], [1, 2]]
    assert DataGenerator.validate_corpus(corpus) == []
    # 3 treatment and 2 covariate slots: each stored axis fits its own mask width.
    assert pg.active_count_coverage(corpus) == _report(corpus, ((2, 3), (1, 2)))


def test_short_cell_error_names_the_allocated_active_counts():
    """A realism floor no draw can reach fails the first cell, named by its counts.

    A non-negative series of n weeks has a coefficient of variation of at most
    sqrt(n - 1), so the CV floor rejects every candidate. The weights put every
    cell on (2, 1), the bottom of both ranges, so a message naming the layout's
    3 treatments and 2 covariates instead would fail.
    """
    cfg = make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        draws_per_cell=1,
        seed=1,
        n_treatments_active_range=(2, 3),
        n_covariates_active_range=(1, 2),
        active_count_allocation="stratified",
        active_count_weights=[[1, 0], [0, 0]],
        treatment_cv_floor=1e6,
    )
    with pytest.raises(RuntimeError) as failure:
        pg.sample_prior_predictive(cfg)
    assert "n_treatments_active=2" in str(failure.value)
    assert "n_covariates_active=1" in str(failure.value)


# -- the default (independent) allocation ---------------------------------------


def test_default_corpus_carries_no_coverage_block(independent):
    _, corpus = independent
    assert "active_count_coverage" not in corpus["diagnostics"]


def test_default_allocation_still_draws_cell_zero_counts_first(independent):
    """Restated legacy consumption: nothing draws from the corpus stream before cell 0."""
    cfg, corpus = independent
    rng = np.random.default_rng(cfg.seed)
    cell_zero = (int(corpus["n_treatments_active"][0]), int(corpus["n_covariates_active"][0]))
    assert cell_zero == (int(rng.integers(2, 4)), int(rng.integers(1, 4)))


# -- the public coverage helper --------------------------------------------------


def test_coverage_helper_reports_a_default_corpus_over_the_prior_grid(independent):
    """Combinations no task reached are reported as zeros."""
    cfg, corpus = independent
    report = pg.active_count_coverage(corpus, prior=cfg)
    assert report == _report(corpus, _INDEPENDENT_GRID)
    assert 0 in itertools.chain.from_iterable(report["n_cells"]), "the fixture covers the grid"


@pytest.mark.parametrize(
    ("treatment_range", "grid"),
    (
        pytest.param(None, _UNIFORM_GRID, id="stored-grid"),
        pytest.param((1, 3), ((1, 2, 3), (1, 2, 3)), id="wider-prior-grid"),
        # The generating prior: its raw (2, 5) treatment range clamps to rows 2..3.
        pytest.param((2, 5), _UNIFORM_GRID, id="clamped-prior-grid"),
    ),
)
def test_coverage_helper_uses_the_prior_grid_else_the_stored_one(uniform, treatment_range, grid):
    cfg, corpus = uniform
    prior = (
        None if treatment_range is None else replace(cfg, n_treatments_active_range=treatment_range)
    )
    assert pg.active_count_coverage(corpus, prior=prior) == _report(corpus, grid)


# -- validate_corpus rules for the block ------------------------------------------


def _attach_coverage_block(corpus, grid):
    """``corpus`` plus a coverage block recounted here, weighted by its own cell counts.

    With weights equal to the realised cells per combination, every positive
    target equals its count, so the block is the unique stratified allocation of
    its weights.
    """
    report = _report(corpus, grid)
    attached = dict(corpus)
    attached["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    attached["diagnostics"]["active_count_coverage"] = {
        "allocation": "stratified",
        "n_treatments_active": report["n_treatments_active"],
        "n_covariates_active": report["n_covariates_active"],
        "weights": [[float(n) for n in row] for row in report["n_cells"]],
        "n_cells": report["n_cells"],
        "n_worlds": report["n_worlds"],
    }
    return attached


@pytest.fixture(scope="module")
def attached(independent):
    _, corpus = independent
    attached = _attach_coverage_block(corpus, _INDEPENDENT_GRID)
    counts = np.array(attached["diagnostics"]["active_count_coverage"]["n_cells"])
    # The rows below need two populated combinations (one loses its weight), an
    # empty one (weighted heavily) and tasks with three active treatments.
    assert (counts > 0).sum() >= 2 and (counts == 0).any() and counts[-1].any()
    return attached


def _edit_block(corpus, edit):
    """Copy of ``corpus`` whose deep-copied coverage block is replaced by ``edit(block)``."""
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    block = broken["diagnostics"]["active_count_coverage"]
    broken["diagnostics"]["active_count_coverage"] = edit(block)
    return broken


def _set(key, value):
    """Edit setting ``block[key]`` to ``value(block)``."""

    def edit(block):
        block[key] = value(block)
        return block

    return edit


def _set_entry(key, index, value):
    """Edit setting ``block[key][i][j]`` to ``value(old entry)``."""

    def edit(block):
        i, j = index
        block[key][i][j] = value(block[key][i][j])
        return block

    return edit


def _transposed(key):
    return _set(key, lambda block: [list(column) for column in zip(*block[key], strict=True)])


def _uint64_n_cells(first=None):
    """Edit storing ``n_cells`` as a uint64 array, its [0][0] entry set to ``first`` if given."""

    def edit(block):
        n_cells = np.array(block["n_cells"], dtype=np.uint64)
        if first is not None:
            n_cells[0, 0] = first
        block["n_cells"] = n_cells
        return block

    return edit


def _drop_last_treatment_count(block):
    """The grid without its top treatment count, otherwise self-consistent."""
    block["n_treatments_active"] = block["n_treatments_active"][:-1]
    for key in ("weights", "n_cells", "n_worlds"):
        block[key] = block[key][:-1]
    return block


def _prepend_empty_treatment_count(block):
    """The grid widened to one active treatment, a row no task reaches, weighted 0."""
    block["n_treatments_active"] = [1, *block["n_treatments_active"]]
    width = len(block["n_covariates_active"])
    block["weights"] = [[0.0] * width, *block["weights"]]
    for key in ("n_cells", "n_worlds"):
        block[key] = [[0] * width, *block[key]]
    return block


def _zero_a_populated_weight(block):
    """Weight 0 on the first combination that holds cells; the others keep theirs."""
    i, j = np.argwhere(np.array(block["n_cells"]) > 0)[0]
    block["weights"][i][j] = 0.0
    return block


def _skew_to_an_empty_combination(block):
    """Weight 1 on every populated combination, 100 on the first empty one, 0 elsewhere.

    No populated combination has weight 0, yet the heavy one's target is at
    least one cell (with two or more cells in total) and it holds none.
    """
    counts = np.array(block["n_cells"])
    weights = (counts > 0).astype(np.float64)
    weights[tuple(np.argwhere(counts == 0)[0])] = 100.0
    block["weights"] = weights.tolist()
    return block


REJECTED_BLOCKS = (
    pytest.param(lambda block: list(block.values()), MAPPING, id="block-not-a-mapping"),
    pytest.param(
        lambda block: {k: v for k, v in block.items() if k != "weights"}, KEYS, id="key-missing"
    ),
    pytest.param(lambda block: {**block, "n_latent_active": [1]}, KEYS, id="key-extra"),
    pytest.param(
        _set("allocation", lambda _: "independent"), ALLOCATION, id="allocation-independent"
    ),
    pytest.param(
        _set("allocation", lambda _: np.array(["stratified", "stratified"])),
        ALLOCATION,
        id="allocation-array",
    ),
    # Axes: consecutive ascending integer counts within [1, mask width].
    pytest.param(_set("n_treatments_active", lambda _: [1, 3]), TREATMENT_AXIS, id="axis-gap"),
    pytest.param(
        _set("n_treatments_active", lambda _: [3, 2]), TREATMENT_AXIS, id="axis-descending"
    ),
    pytest.param(
        _set("n_covariates_active", lambda _: [0, 1, 2]), COVARIATE_AXIS, id="axis-from-zero"
    ),
    pytest.param(
        _set("n_treatments_active", lambda _: [3, 4]), TREATMENT_AXIS, id="axis-beyond-width"
    ),
    pytest.param(
        _set("n_covariates_active", lambda _: [True, 2, 3]), COVARIATE_AXIS, id="axis-bool"
    ),
    pytest.param(
        _set("n_covariates_active", lambda _: [1.0, 2.0, 3.0]), COVARIATE_AXIS, id="axis-float"
    ),
    pytest.param(_set("n_treatments_active", lambda _: []), TREATMENT_AXIS, id="axis-empty"),
    # Weights: a 2x3 matrix of finite non-negative reals with a positive entry.
    pytest.param(_transposed("weights"), WEIGHTS, id="weights-transposed"),
    pytest.param(
        _set("weights", lambda block: [block["weights"][0], block["weights"][1][:2]]),
        WEIGHTS,
        id="weights-ragged",
    ),
    pytest.param(_set_entry("weights", (0, 1), lambda _: -1.0), WEIGHTS, id="weights-negative"),
    pytest.param(_set_entry("weights", (1, 1), lambda _: float("nan")), WEIGHTS, id="weights-nan"),
    pytest.param(_set_entry("weights", (0, 0), lambda _: 10**400), WEIGHTS, id="weights-overflow"),
    pytest.param(_set("weights", lambda _: [[0.0] * 3, [0.0] * 3]), WEIGHTS, id="weights-all-zero"),
    pytest.param(_set_entry("weights", (0, 0), lambda _: "1.0"), WEIGHTS, id="weights-string"),
    pytest.param(_set_entry("weights", (0, 0), lambda _: True), WEIGHTS, id="weights-bool"),
    # Counts: exactly the integer recount from the masks.
    pytest.param(
        _set_entry("n_cells", (0, 0), lambda old: old + 1),
        STALE.format("n_cells"),
        id="n_cells-off-by-one",
    ),
    pytest.param(
        _set_entry("n_worlds", (1, 2), lambda old: old + 1),
        STALE.format("n_worlds"),
        id="n_worlds-off-by-one",
    ),
    pytest.param(_transposed("n_cells"), STALE.format("n_cells"), id="n_cells-transposed"),
    pytest.param(
        _set("n_cells", lambda block: [[float(n) for n in row] for row in block["n_cells"]]),
        STALE.format("n_cells"),
        id="n_cells-float",
    ),
    pytest.param(
        _set_entry("n_cells", (0, 0), lambda _: 10**30),
        STALE.format("n_cells"),
        id="n_cells-beyond-int64",
    ),
    pytest.param(
        _uint64_n_cells(first=2**63), STALE.format("n_cells"), id="n_cells-uint64-beyond-int64"
    ),
    # The grid must hold every task, and n_cells must round its weights' targets.
    pytest.param(_drop_last_treatment_count, GRID, id="grid-misses-tasks"),
    pytest.param(_zero_a_populated_weight, NOT_ALLOCATION, id="populated-zero-weight"),
    pytest.param(_skew_to_an_empty_combination, NOT_ALLOCATION, id="weights-skewed"),
)


@pytest.mark.parametrize(
    "edit",
    (
        pytest.param(lambda block: block, id="as-attached"),
        pytest.param(_prepend_empty_treatment_count, id="grid-wider-than-realised"),
        pytest.param(_uint64_n_cells(), id="n_cells-uint64-array"),
    ),
)
def test_validate_corpus_accepts_a_consistent_coverage_block(attached, edit):
    assert DataGenerator.validate_corpus(_edit_block(attached, edit)) == []


@pytest.mark.parametrize(("edit", "expected"), REJECTED_BLOCKS)
def test_validate_corpus_rejects_an_inconsistent_coverage_block(attached, edit, expected):
    assert expected in DataGenerator.validate_corpus(_edit_block(attached, edit))


def test_validate_corpus_rejects_swapped_cell_and_world_counts(attached):
    """Two draws per cell make every populated world count twice its cell count."""
    broken = _edit_block(
        attached,
        lambda block: {**block, "n_cells": block["n_worlds"], "n_worlds": block["n_cells"]},
    )
    errors = DataGenerator.validate_corpus(broken)
    assert STALE.format("n_cells") in errors
    assert STALE.format("n_worlds") in errors


def _numpy_valued(block):
    """The block with numpy values where its JSON has lists and scalars: a 0-d
    string array, integer-array axes, and matrices as lists of array rows."""
    numpy_valued = {
        "allocation": np.array(block["allocation"]),
        "n_treatments_active": np.array(block["n_treatments_active"]),
        "n_covariates_active": np.array(block["n_covariates_active"]),
    }
    for key in ("weights", "n_cells", "n_worlds"):
        numpy_valued[key] = [np.array(row) for row in block[key]]
    return numpy_valued


def test_numpy_valued_block_is_judged_as_after_a_save_load_round_trip(uniform, tmp_path):
    """``save_corpus`` writes numpy values as plain JSON; in memory they get the same
    verdict and the same coverage report."""
    _, corpus = uniform
    numpy_valued = _edit_block(corpus, _numpy_valued)
    path = tmp_path / "numpy_valued.npz"
    pg.save_corpus(numpy_valued, path)
    loaded = pg.load_corpus(path)
    assert (
        DataGenerator.validate_corpus(numpy_valued) == DataGenerator.validate_corpus(loaded) == []
    )
    expected = _report(corpus, _UNIFORM_GRID)
    assert pg.active_count_coverage(numpy_valued) == pg.active_count_coverage(loaded) == expected
