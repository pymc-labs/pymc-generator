"""Symbolic random-walk noise for the Phase-4 additive causal graph.

Every non-outcome node in the additive SCM (D, Z, C, B) carries a random-walk
noise term rather than flat Gaussian noise (plan doc 03, D1a). The outcome
node ``Y`` instead carries iid observation noise. A walk is always
autocorrelated — a smoothed Brownian motion — and its ``smoothness`` parameter
controls texture:

* ``smoothness -> 0``: raw Brownian motion — jagged, hectic, but cumulative.
* ``smoothness -> 1``: an absolute-week moving average no wider than
  ``rw_smoothness_max_weeks`` (26 weeks by default), capped at the simulated
  horizon, and producing a soft drift.

The symbolic expression lives inside the causal graph so intervention-based
decomposition evaluates the same graph twice. PyTensor is imported lazily so
importing this module stays light.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["symbolic_random_walk", "symbolic_random_walk_by_width", "walk_width_index"]


def _kernel_width(smoothness: float, n_time_steps: int, rw_smoothness_max_weeks: int) -> int:
    """Return the moving-average width for one simulated horizon.

    The unclamped width is ``max(1, round(smoothness *
    rw_smoothness_max_weeks))`` weeks. ``rw_smoothness_max_weeks`` is an
    absolute timescale, so changing ``n_time_steps`` changes only the short-series
    clamp.

    ``n_time_steps`` must be at least 2 — a walk over a single week has no
    amplitude to calibrate (see :func:`_centred_walk_scale`), so every walk
    entry point rejects that horizon here rather than downstream.
    """
    if not 0.0 <= smoothness <= 1.0:
        raise ValueError(f"smoothness must be in [0, 1], got {smoothness}")
    if (
        isinstance(rw_smoothness_max_weeks, (bool, np.bool_))
        or not isinstance(rw_smoothness_max_weeks, (int, np.integer))
        or rw_smoothness_max_weeks < 1
    ):
        raise ValueError(
            f"rw_smoothness_max_weeks must be an integer >= 1, got {rw_smoothness_max_weeks!r}"
        )
    if (
        isinstance(n_time_steps, (bool, np.bool_))
        or not isinstance(n_time_steps, (int, np.integer))
        or n_time_steps < 2
    ):
        raise ValueError(
            "n_time_steps must be an integer >= 2 (a one-week random walk has no "
            f"calibratable amplitude), got {n_time_steps!r}"
        )
    width = max(1, int(round(smoothness * rw_smoothness_max_weeks)))
    return min(width, int(n_time_steps))


@lru_cache(maxsize=64)
def _centred_walk_scale(n_time_steps: int, width: int) -> float:
    """RMS amplitude of the centred, smoothed Brownian path for unit innovations.

    The walk is ``mean + std * (A @ eps) / c`` where ``A = centre . movavg .
    cumsum`` is a FIXED linear operator and ``c`` is the constant returned
    here, ``sqrt(tr(A A^T) / n_time_steps)``. Dividing by this constant instead
    of by the path's own realized standard deviation is what keeps the map
    injective: normalizing by ``walk.std()`` makes ``eps -> walk`` invariant to
    ``eps -> k * eps``, leaving the likelihood exactly flat along the radial
    direction of an ``n_time_steps``-dimensional latent, which no amount of
    step-size tuning can fix.

    The price is that ``std`` becomes the walk's EXPECTED amplitude
    (``E[var(walk)] == std ** 2``) rather than its exact realized one. The gain
    is that the walk is now an ordinary multivariate normal with covariance
    ``(std / c) ** 2 * A A^T`` -- a distribution with a density, which the
    normalized version did not have.

    That calibration is unsatisfiable at ``n_time_steps == 1``: centring a
    one-week path subtracts its only value, so ``A`` is the zero operator,
    ``c`` is 0, and no finite ``std`` scaling can give the path the population
    variance ``std ** 2`` a single point cannot have. Rescaling by ``c`` would
    divide by zero, and defining the path as its ``mean`` instead would quietly
    return a walk with no amplitude and no injectivity — ``eps -> walk`` would
    map every innovation vector to the same constant, the exact flat-likelihood
    failure this normalization exists to avoid. A single-week walk is therefore
    not defined at all: :func:`_kernel_width` rejects the horizon for every
    entry point in this module (the top-level config separately requires
    ``n_time_steps >= 4``).

    In generation, this function's ``n_time_steps`` argument is the FULL horizon
    ``n_time_steps_full`` — the reported ``n_time_steps`` plus ``adstock_burn_in``. Persisted
    ``param_rw_*_std`` labels declare each random walk's expected standard
    deviation over ``n_time_steps_full``. ``param_rw_y_std`` is different: ``RW_Y`` is iid
    observation noise, so its label is its exact per-week Normal standard
    deviation and carries no smoothness or walk operator. Because each random
    walk path is divided by a fixed constant rather than by its own realized
    standard deviation, random-walk scale is realized only in expectation. It
    is therefore neither the realized standard deviation of an individual path
    nor a standard deviation measured only over the reported window.
    ``smoothness`` maps to an absolute kernel width in weeks, governed by
    ``rw_smoothness_max_weeks`` and capped at ``n_time_steps_full``. For positive-only
    walks, ``std`` is the pre-softplus amplitude, so ``rw_c`` is excluded from
    the signed-walk table below rather than reported with a misleadingly wide
    range.

    Across 32 signed random walks from eight worlds at ``n_time_steps=52`` and
    ``adstock_burn_in=8``, reported-window sd / declared ``std`` was:

    * ``rw_d``: [0.396, 0.927], median 0.690
    * ``rw_z``: [0.252, 1.985], median 0.868
    * ``rw_b``: [0.264, 1.809], median 0.814

    This is roughly an 8x spread. Random-walk ``param_rw_*_std`` is therefore
    a weak label for anything measured on the reported window; consumers should
    not score it as if it were the realized reported-window standard deviation.

    Column ``j`` of ``A`` is the smoothed, centred step function ``1[t >= j]``,
    so the whole operator is built in one ``(n_time_steps, n_time_steps)`` pass.
    """
    steps = np.tril(
        np.ones((n_time_steps, n_time_steps))
    )  # steps[t, j] = 1 if t >= j (the cumsum operator)
    columns = _smooth_columns_numpy(steps, width)
    columns = columns - columns.mean(axis=0, keepdims=True)
    return float(np.sqrt((columns**2).sum() / n_time_steps))


def _smooth_columns_numpy(raw: np.ndarray, width: int) -> np.ndarray:
    """Edge-padded moving average down axis 0 used to build the scale operator."""
    if width <= 1:
        return raw
    n_time_steps = raw.shape[0]
    left, right = width // 2, width - 1 - width // 2
    padded = np.concatenate(
        [np.repeat(raw[:1], left, axis=0), raw, np.repeat(raw[-1:], right, axis=0)]
    )
    cumulative = np.cumsum(padded, axis=0)
    head = np.zeros((1, *raw.shape[1:]), dtype=cumulative.dtype)
    window_sums = cumulative[width - 1 :] - np.concatenate([head, cumulative[: n_time_steps - 1]])
    return np.asarray(window_sums / width)


@lru_cache(maxsize=64)
def _walk_basis(n_time_steps: int, width: int) -> np.ndarray:
    """Return the fixed ``B = A / c`` operator of one signed random walk.

    :func:`symbolic_random_walk` applies cumulative sum, edge-padded moving
    average, column centring, and then the fixed
    :func:`_centred_walk_scale` normalization. Its zero-mean walk is therefore
    exactly ``std * B @ eps`` for the plain float64 matrix returned here.
    """
    steps = np.tril(np.ones((n_time_steps, n_time_steps)))
    columns = _smooth_columns_numpy(steps, width)
    columns = columns - columns.mean(axis=0, keepdims=True)
    return np.asarray(columns / _centred_walk_scale(n_time_steps, width))


@lru_cache(maxsize=8)
def _walk_basis_stack(n_time_steps: int, rw_smoothness_max_weeks: int) -> np.ndarray:
    """Every walk operator ``_kernel_width`` can select, stacked on axis 0.

    ``_kernel_width`` maps the continuous ``smoothness`` in [0, 1] onto at most
    ``min(rw_smoothness_max_weeks, n_time_steps)`` distinct integer widths, so
    the whole family of walk operators is a small finite set. Stacking it lets a
    graph pick its kernel by integer index at run time instead of baking one
    width in at compile time — the enabling trick behind the one-compile
    template path (``(26, 108, 108)`` float64 is 2.4 MB).
    """
    n_widths = _kernel_width(1.0, n_time_steps, rw_smoothness_max_weeks=rw_smoothness_max_weeks)
    return np.stack([_walk_basis(n_time_steps, w) for w in range(1, n_widths + 1)])


def walk_width_index(smoothness, n_time_steps: int, *, rw_smoothness_max_weeks: int) -> np.ndarray:
    """0-based indices into :func:`_walk_basis_stack` for each smoothness value.

    Routes through :func:`_kernel_width` so the smoothness-to-width mapping stays
    single-sourced with :func:`symbolic_random_walk`.
    """
    widths = [
        _kernel_width(float(s), n_time_steps, rw_smoothness_max_weeks=int(rw_smoothness_max_weeks))
        for s in np.asarray(smoothness, dtype="float64").ravel()
    ]
    return np.asarray(widths, dtype="int64") - 1


def symbolic_random_walk(
    n_time_steps: int,
    mean,
    std,
    smoothness: float,
    positive_only: bool,
    *,
    rw_smoothness_max_weeks: int,
    eps=None,
):
    """Create a PyTensor symbolic random walk (plan doc 4.0c).

    The expression is part of the causal graph. Concrete values (``eps``,
    ``mean``, ``std``) are supplied at ``.eval()`` time.

    Parameters
    ----------
    n_time_steps : int
        Number of time steps (static — determines graph shape). Must be at
        least 2; a one-week walk has no calibratable amplitude (see
        :func:`_centred_walk_scale`) and is rejected.
    mean, std : pt.TensorVariable | float
        Walk center and amplitude (symbolic scalars or floats).
    smoothness : float
        Static smoothness in [0, 1]. This must be a Python float (not a
        tensor) because the moving-average kernel width is a structural
        property of the graph.
    positive_only : bool
        If True, apply softplus at the end.
    rw_smoothness_max_weeks : int
        Maximum absolute smoothing width in weeks; required so every caller
        uses the configured timescale rather than a horizon-derived default.
    eps : pt.TensorVariable | None
        White-noise input of shape (n_time_steps,). If None, a fresh ``pt.vector``
        named ``"eps"`` is created.

    Returns
    -------
    pt.TensorVariable of shape (n_time_steps,)
    """
    import pytensor.tensor as pt

    if eps is None:
        eps = pt.vector("eps")
    raw = pt.cumsum(eps)
    w = _kernel_width(smoothness, n_time_steps, rw_smoothness_max_weeks=rw_smoothness_max_weeks)
    if w > 1:
        left, right = w // 2, w - 1 - w // 2
        padded = pt.concatenate([pt.tile(raw[0], left), raw, pt.tile(raw[-1], right)])
        c = pt.cumsum(padded)
        window_sums = c[w - 1 :] - pt.concatenate([pt.zeros(1), c[: n_time_steps - 1]])
        walk = window_sums / w
    else:
        walk = raw
    walk = walk - walk.mean()
    walk = walk * std / _centred_walk_scale(n_time_steps, w)
    walk = walk + mean
    if positive_only:
        walk = pt.softplus(walk)
    return walk


def symbolic_random_walk_by_width(
    n_time_steps: int,
    mean,
    std,
    width_index,
    positive_only: bool,
    *,
    rw_smoothness_max_weeks: int,
    eps,
):
    """A random walk whose smoothing kernel is chosen by a symbolic integer index.

    Mathematically identical to :func:`symbolic_random_walk` — it evaluates
    ``mean + std * B @ eps`` for the same operator ``B`` — but ``width_index``
    may be a tensor, so one compiled graph serves every smoothness value.
    :func:`symbolic_random_walk` instead resolves the width while building the
    graph, which bakes one kernel in and forces a recompile per structure.

    ``width_index`` is 0-based (see :func:`walk_width_index`).
    """
    import pytensor.tensor as pt

    basis = pt.constant(
        _walk_basis_stack(n_time_steps, rw_smoothness_max_weeks),
        name="walk_basis_stack",
    )
    walk = std * pt.dot(basis[width_index], eps) + mean
    if positive_only:
        walk = pt.softplus(walk)
    return walk
