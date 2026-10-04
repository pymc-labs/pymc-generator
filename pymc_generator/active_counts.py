"""How corpus cells cover the grid of active treatment × covariate counts.

Every corpus cell fixes how many treatment and covariate slots are active. With
``active_count_allocation="independent"`` (the default) each cell draws both
counts independently and uniformly from the effective active ranges, so a
finite corpus covers the joint grid only in expectation. With ``"stratified"``
the cells are allocated over the grid by weight (uniform by default).

The stratified rule gives every grid combination a target number of cells.
The targets sum to ``n_cells`` and follow the weights, but first cover as many
positive-weight combinations as the cells allow:

* A combination with weight 0 has target 0.
* With fewer cells than positive-weight combinations, no combination can get
  more than one cell: the targets are the weight shares, capped at one cell
  (the capped excess is shared out again until no target exceeds one).
* Otherwise every positive-weight combination gets at least one cell: a share
  below one cell is raised to exactly one (the remaining cells are shared out
  again until no share is below one).

Each count is then its target rounded down or up, with the fractional parts
rounded by systematic sampling in a random order. So the counts sum to
``n_cells``, every count's expectation equals its target, and every combination
whose target is a whole number gets exactly that. Under uniform weights the
counts differ by at most one, and every combination appears once there are at
least as many cells as combinations. Targets use exact rational arithmetic, so
equal weights tie exactly. The allocated combinations are finally shuffled over
the cells, so ``cell_id`` (and the cell-level validation split) carries no
information about the counts.

This module is numpy-only, so ``pymc_generator.active_count_coverage`` loads
without PyMC.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .sampler import SCMPrior

#: Accepted values of ``SCMPrior.active_count_allocation``.
ACTIVE_COUNT_ALLOCATIONS: tuple[str, ...] = ("independent", "stratified")

#: Keys of ``diagnostics["active_count_coverage"]``, in insertion order.
ACTIVE_COUNT_COVERAGE_KEYS: tuple[str, ...] = (
    "allocation",
    "n_treatments_active",
    "n_covariates_active",
    "weights",
    "n_cells",
    "n_worlds",
)

_BLOCK = "diagnostics active_count_coverage"


def _number_matrix(value: Any, *, integer: bool) -> np.ndarray | None:
    """``value`` as a rectangular 2-D matrix of plain numbers, or None.

    Accepts nested lists/tuples of ints and floats (numpy scalars included);
    rejects arrays, bools, strings, ragged rows and, when ``integer``, floats.
    A float conversion that overflows becomes ``inf``.
    """
    if not isinstance(value, (list, tuple)) or not value:
        return None
    kinds: tuple[type, ...] = (
        (int, np.integer) if integer else (int, float, np.integer, np.floating)
    )
    rows: list[list[Any]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != len(value[0]) or not row:
            return None
        if not all(isinstance(v, kinds) and not isinstance(v, (bool, np.bool_)) for v in row):
            return None
        rows.append(list(row))
    if integer:
        flat = [int(v) for row in rows for v in row]
        if any(abs(v) > np.iinfo(np.int64).max for v in flat):
            return None
        return np.array(flat, dtype=np.int64).reshape(len(rows), len(rows[0]))
    numbers = []
    for row in rows:
        for v in row:
            try:
                numbers.append(float(v))
            except OverflowError:
                numbers.append(math.inf if v > 0 else -math.inf)
    return np.array(numbers, dtype=np.float64).reshape(len(rows), len(rows[0]))


def _axis_text(name: str, counts: Sequence[int]) -> str:
    return f"{name} {counts[0]}..{counts[-1]}"


def _plain(value: Any) -> Any:
    """``value`` with numpy arrays and scalars as the lists and Python scalars
    ``save_corpus`` persists, so a block is judged the same in memory and on disk."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def active_count_weight_matrix(
    value: Any, treatment_counts: Sequence[int], covariate_counts: Sequence[int]
) -> np.ndarray:
    """Validated ``active_count_weights`` as a float64 matrix (ones for ``None``).

    Rows follow ``treatment_counts`` and columns ``covariate_counts`` (the
    effective active-count ranges, both non-empty). Raises ``ValueError``
    naming the expected shape unless the value is nested lists or tuples
    forming a finite, non-negative real matrix of exactly that shape with at
    least one positive entry.
    """
    shape = (len(treatment_counts), len(covariate_counts))
    if value is None:
        return np.ones(shape, dtype=np.float64)
    expected = (
        f"a {shape[0]}x{shape[1]} matrix (rows: "
        f"{_axis_text('n_treatments_active', treatment_counts)}; columns: "
        f"{_axis_text('n_covariates_active', covariate_counts)})"
    )
    if isinstance(value, np.ndarray):
        # Arrays would break SCMPrior equality and recipe.json serialization.
        raise ValueError(
            f"active_count_weights must be {expected} given as nested lists or tuples "
            "(use weights.tolist() for an array)"
        )
    matrix = _number_matrix(value, integer=False)
    if matrix is None:
        raise ValueError(f"active_count_weights must be {expected} of real numbers, got {value!r}")
    if matrix.shape != shape:
        raise ValueError(f"active_count_weights must be {expected}, got shape {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or not (matrix > 0).any():
        raise ValueError(
            f"active_count_weights must be {expected} of finite, non-negative numbers with "
            f"at least one positive entry, got {value!r}"
        )
    return matrix


def allocation_targets(weights: np.ndarray, n_cells: int) -> np.ndarray:
    """Exact target cells per grid combination (module docstring), as Fractions.

    ``weights`` must be finite and non-negative with a positive entry. The
    result is an object array of :class:`fractions.Fraction` with the shape of
    ``weights`` that sums to ``n_cells``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    flat = [Fraction(float(w)) for w in weights.ravel()]
    targets = [Fraction(0)] * len(flat)
    free = [i for i, w in enumerate(flat) if w > 0]
    capped = n_cells < len(free)
    seats = n_cells
    # Below one cell per positive combination, ``seats < len(free)`` holds
    # throughout, so the free shares average below one and a pass never caps
    # them all. Otherwise ``seats >= len(free)`` holds and a pass never raises
    # them all. Either way the loop ends with a free combination left.
    while True:
        total = sum((flat[i] for i in free), Fraction(0))
        if capped:
            moved = {i for i in free if seats * flat[i] >= total}
        else:
            moved = {i for i in free if seats * flat[i] < total}
        if not moved:
            break
        for i in moved:
            targets[i] = Fraction(1)
        seats -= len(moved)
        free = [i for i in free if i not in moved]
    for i in free:
        targets[i] = seats * flat[i] / total
    out = np.empty(len(flat), dtype=object)
    out[:] = targets
    return out.reshape(weights.shape)


def stratified_cell_counts(
    weights: np.ndarray, n_cells: int, rng: np.random.Generator
) -> np.ndarray:
    """Cells per grid combination: :func:`allocation_targets` rounded without bias.

    Every count is the floor or the ceiling of its target, the counts sum to
    ``n_cells``, and each count's expectation is its target. ``rng`` is consumed
    by exactly one ``rng.permutation(weights.size)`` and one ``rng.random()``.
    """
    targets = allocation_targets(weights, n_cells).ravel()
    order = rng.permutation(targets.size)
    offset = Fraction(float(rng.random()))
    counts = np.array([math.floor(t) for t in targets], dtype=np.int64)
    # Systematic sampling: lay the fractional parts end to end in a random
    # order and give one more cell to each part that covers a point
    # ``offset + k``. A part shorter than one covers at most one point, with
    # probability equal to its length.
    start = Fraction(0)
    for i in order:
        end = start + targets[i] - counts[i]
        counts[i] += math.ceil(end - offset) - math.ceil(start - offset)
        start = end
    return counts.reshape(np.shape(weights))


def is_stratified_allocation(counts: np.ndarray, weights: np.ndarray) -> bool:
    """Whether ``counts`` rounds the stratified targets of ``counts.sum()`` cells.

    True when every count is the floor or the ceiling of its
    :func:`allocation_targets` entry, which is what
    :func:`stratified_cell_counts` guarantees.
    """
    counts = np.asarray(counts)
    if counts.shape != np.shape(weights) or (counts < 0).any():
        return False
    targets = allocation_targets(weights, int(counts.sum()))
    return all(
        math.floor(t) <= int(c) <= math.ceil(t)
        for c, t in zip(counts.ravel(), targets.ravel(), strict=True)
    )


def plan_active_counts(cfg: SCMPrior, rng: np.random.Generator) -> np.ndarray | None:
    """Per-cell ``(n_treatments_active, n_covariates_active)``, or None if independent.

    Independent allocation touches no randomness here: its counts are drawn
    per cell by :func:`draw_active_counts`, exactly as before this option
    existed. Stratified allocation draws one seed from ``rng`` for its own
    stream, so the plan follows the generator's state, allocates
    ``cfg.n_cells`` cells with :func:`stratified_cell_counts`, and shuffles
    them over the cells. Both modes first check the allocation, the active
    ranges and the weights, because the template path never calls
    ``SCMPrior.validate()``.
    """
    allocation = cfg.active_count_allocation
    if not isinstance(allocation, str) or allocation not in ACTIVE_COUNT_ALLOCATIONS:
        raise ValueError(
            f"active_count_allocation must be 'independent' or 'stratified', got {allocation!r}"
        )
    # Checks only: no randomness, so the independent path stays the legacy draw.
    weights = cfg.active_count_weight_matrix()
    if allocation == "independent":
        return None
    n_cells = int(cfg.n_cells)
    allocation_rng = np.random.default_rng(int(rng.integers(2**63)))
    treatment_counts, covariate_counts = cfg.active_count_grid
    counts = stratified_cell_counts(weights, n_cells, allocation_rng)
    grid_t, grid_c = np.meshgrid(treatment_counts, covariate_counts, indexing="ij")
    combinations = np.stack([grid_t.ravel(), grid_c.ravel()], axis=1).astype(np.int64)
    cells = np.repeat(combinations, counts.ravel(), axis=0)
    return cells[allocation_rng.permutation(n_cells)]


def draw_active_counts(
    cfg: SCMPrior, rng: np.random.Generator, plan: np.ndarray | None, cell: int
) -> tuple[int, int, int]:
    """One cell's active treatment, covariate and latent counts.

    Without a plan, the treatment, covariate and latent counts are drawn in
    that order from ``rng``. With a plan, the treatment and covariate counts
    come from it and only the latent count is drawn.
    """
    if plan is None:
        tr = cfg.n_treatments_active_range_effective
        cv = cfg.n_covariates_active_range_effective
        n_treatments_active = int(rng.integers(tr[0], tr[1] + 1))
        n_covariates_active = int(rng.integers(cv[0], cv[1] + 1))
    else:
        n_treatments_active, n_covariates_active = (int(v) for v in plan[cell])
    lt = cfg.n_latent_active_range_effective
    n_latent_active = int(rng.integers(lt[0], lt[1] + 1))
    return n_treatments_active, n_covariates_active, n_latent_active


def summarize_active_count_coverage(
    treatment_active_mask: np.ndarray,
    covariate_active_mask: np.ndarray,
    cell_id: np.ndarray,
    treatment_counts: Sequence[int],
    covariate_counts: Sequence[int],
) -> dict[str, list[list[int]]]:
    """Realised cells and worlds per grid combination, from the stored masks.

    Row ``i`` / column ``j`` counts tasks whose active-treatment mask sums to
    ``treatment_counts[i]`` and active-covariate mask to ``covariate_counts[j]``.
    ``n_worlds`` counts tasks; ``n_cells`` counts distinct ``cell_id`` values, a
    truncated (partial) cell included. Tasks outside the grid are not counted.
    Values are plain ``int`` so the block round-trips through the corpus JSON.
    """
    n_treatments = np.asarray(treatment_active_mask).astype(np.int64).sum(axis=1)
    n_covariates = np.asarray(covariate_active_mask).astype(np.int64).sum(axis=1)
    cells = np.asarray(cell_id).astype(np.int64)
    row = _positions(n_treatments, treatment_counts)
    col = _positions(n_covariates, covariate_counts)
    inside = (row >= 0) & (col >= 0)
    shape = (len(treatment_counts), len(covariate_counts))
    n_worlds = np.zeros(shape, dtype=np.int64)
    np.add.at(n_worlds, (row[inside], col[inside]), 1)
    n_cells = np.zeros(shape, dtype=np.int64)
    if inside.any():
        distinct = np.unique(np.stack([cells[inside], row[inside], col[inside]], axis=1), axis=0)
        np.add.at(n_cells, (distinct[:, 1], distinct[:, 2]), 1)
    return {"n_cells": n_cells.tolist(), "n_worlds": n_worlds.tolist()}


def _positions(values: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Index of every value in ``labels``, or -1 where it is absent."""
    lookup = {int(label): i for i, label in enumerate(labels)}
    return np.array([lookup.get(int(v), -1) for v in values], dtype=np.int64)


def active_count_coverage(
    corpus: Mapping[str, Any], prior: SCMPrior | None = None
) -> dict[str, list[Any]]:
    """Realised cells and worlds per (active treatments, active covariates) combination.

    Works for any corpus, including default (independent) ones, which carry no
    ``diagnostics["active_count_coverage"]`` block. Combinations of the grid
    that no task reached are reported as zeros, which is how missing
    combinations show up.

    Parameters
    ----------
    corpus : mapping
        A corpus from :func:`pymc_generator.sample_prior_predictive` or
        :func:`pymc_generator.load_corpus`.
    prior : SCMPrior, optional
        Report over this prior's effective active-count grid. Required unless
        the corpus carries ``diagnostics["active_count_coverage"]``, whose grid
        is used otherwise.

    Returns
    -------
    dict
        ``n_treatments_active`` and ``n_covariates_active`` (the grid axes),
        and ``n_cells`` / ``n_worlds`` as nested lists indexed
        ``[treatment row][covariate column]``. ``n_cells`` counts distinct
        ``cell_id`` values (a truncated cell included), ``n_worlds`` tasks.

    Raises
    ------
    ValueError
        If neither ``prior`` nor a valid stored grid is available, the prior's
        grid does not fit the corpus's slots, or a task's active counts fall
        outside the grid.
    """
    diagnostics = corpus.get("diagnostics")
    treatment_mask = np.asarray(corpus["treatment_active_mask"])
    covariate_mask = np.asarray(corpus["covariate_active_mask"])
    if prior is not None:
        treatment_axis, covariate_axis = (list(axis) for axis in prior.active_count_grid)
        if (
            _count_axis(treatment_axis, treatment_mask.shape[1]) is None
            or _count_axis(covariate_axis, covariate_mask.shape[1]) is None
        ):
            raise ValueError(
                f"the prior's grid n_treatments_active={treatment_axis}, "
                f"n_covariates_active={covariate_axis} does not fit this corpus's "
                f"{treatment_mask.shape[1]} treatment and {covariate_mask.shape[1]} covariate slots"
            )
    elif isinstance(diagnostics, Mapping) and "active_count_coverage" in diagnostics:
        stored = _plain(diagnostics["active_count_coverage"])
        stored_treatment = stored_covariate = None
        if isinstance(stored, Mapping):
            stored_treatment = _count_axis(
                stored.get("n_treatments_active"), treatment_mask.shape[1]
            )
            stored_covariate = _count_axis(
                stored.get("n_covariates_active"), covariate_mask.shape[1]
            )
        if stored_treatment is None or stored_covariate is None:
            raise ValueError(
                f"{_BLOCK} does not hold a valid grid; check the corpus with "
                "validate_corpus or pass the prior that generated it as prior="
            )
        treatment_axis, covariate_axis = stored_treatment, stored_covariate
    else:
        raise ValueError(
            "this corpus does not record its active-count grid; pass the prior that "
            "generated it as prior="
        )
    summary = summarize_active_count_coverage(
        treatment_mask, covariate_mask, corpus["cell_id"], treatment_axis, covariate_axis
    )
    outside = treatment_mask.shape[0] - sum(sum(row) for row in summary["n_worlds"])
    if outside:
        raise ValueError(
            f"{outside} tasks have active counts outside the grid "
            f"n_treatments_active={treatment_axis}, n_covariates_active={covariate_axis}"
        )
    return {
        "n_treatments_active": treatment_axis,
        "n_covariates_active": covariate_axis,
        **summary,
    }


def _count_axis(labels: Any, width: int) -> list[int] | None:
    """``labels`` as consecutive ascending integer counts within ``[1, width]``."""
    if not isinstance(labels, (list, tuple)) or not labels:
        return None
    if not all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in labels
    ):
        return None
    values = [int(v) for v in labels]
    if values != list(range(values[0], values[0] + len(values))):
        return None
    return values if values[0] >= 1 and values[-1] <= width else None


def active_count_coverage_errors(
    block: Any,
    treatment_active_mask: np.ndarray,
    covariate_active_mask: np.ndarray,
    cell_id: np.ndarray,
) -> list[str]:
    """Validation errors of ``diagnostics["active_count_coverage"]`` against the arrays.

    The masks must already be binary prefix masks and ``cell_id`` contiguous
    with per-cell constant masks (checked earlier by ``validate_corpus``).
    """
    if not isinstance(block, Mapping):
        return [f"{_BLOCK} must be a mapping"]
    if set(block) != set(ACTIVE_COUNT_COVERAGE_KEYS):
        return [f"{_BLOCK} must have exactly the keys {', '.join(ACTIVE_COUNT_COVERAGE_KEYS)}"]
    block = _plain(block)
    errors: list[str] = []
    if not isinstance(block["allocation"], str) or block["allocation"] != "stratified":
        errors.append(f"{_BLOCK} allocation must be 'stratified'")
    treatment_axis = _count_axis(block["n_treatments_active"], treatment_active_mask.shape[1])
    covariate_axis = _count_axis(block["n_covariates_active"], covariate_active_mask.shape[1])
    for key, axis, width in (
        ("n_treatments_active", treatment_axis, treatment_active_mask.shape[1]),
        ("n_covariates_active", covariate_axis, covariate_active_mask.shape[1]),
    ):
        if axis is None:
            errors.append(
                f"{_BLOCK} {key} must list consecutive integer counts within [1, {width}]"
            )
    if treatment_axis is None or covariate_axis is None:
        return errors
    shape = (len(treatment_axis), len(covariate_axis))
    weights = _number_matrix(block["weights"], integer=False)
    if (
        weights is None
        or weights.shape != shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not (weights > 0).any()
    ):
        errors.append(
            f"{_BLOCK} weights must be a {shape[0]}x{shape[1]} matrix of finite non-negative "
            "numbers with a positive entry"
        )
        weights = None
    recount = summarize_active_count_coverage(
        treatment_active_mask, covariate_active_mask, cell_id, treatment_axis, covariate_axis
    )
    if sum(sum(row) for row in recount["n_worlds"]) != treatment_active_mask.shape[0]:
        errors.append(f"{_BLOCK} grid does not contain every task's active counts")
    for key in ("n_cells", "n_worlds"):
        stored = _number_matrix(block[key], integer=True)
        if stored is None or not np.array_equal(stored, np.asarray(recount[key])):
            errors.append(f"{_BLOCK} {key} does not match recomputation")
    if weights is not None and not is_stratified_allocation(
        np.asarray(recount["n_cells"], dtype=np.int64), weights
    ):
        errors.append(f"{_BLOCK} n_cells is not a stratified allocation of its weights")
    return errors
