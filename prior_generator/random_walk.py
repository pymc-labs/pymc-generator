"""Symbolic random-walk noise for the additive causal graph.

Every non-outcome node in the additive SCM (D, Z, C, B) carries a random-walk
noise term rather than flat Gaussian noise. The outcome
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


def _centred_walk_operator(n_time_steps: int, width: int) -> np.ndarray:
    steps = np.tril(np.ones((n_time_steps, n_time_steps)))
    columns = _smooth_columns_numpy(steps, width)
    return columns - columns.mean(axis=0, keepdims=True)


def _walk_operator_scale(columns: np.ndarray) -> float:
    return float(np.sqrt((columns**2).sum() / columns.shape[0]))


@lru_cache(maxsize=64)
def _centred_walk_scale(n_time_steps: int, width: int) -> float:
    """RMS amplitude of a centred, smoothed path with unit Gaussian innovations.

    For the fixed operator ``A = centre @ movavg @ cumsum``, return
    ``c = sqrt(tr(A @ A.T) / n_time_steps)``. The signed walk
    ``mean + std * A @ eps / c`` then satisfies ``E[var_pop(walk)] = std**2``.
    This is an expected variance, not the realized variance of each draw.

    Constant calibration removes the radial scale invariance introduced by
    dividing each path by its own standard deviation. It does not make the
    map injective: centring gives rank at most ``n_time_steps - 1``, and
    smoothing can reduce it further. The signed walk is a degenerate Gaussian
    on an affine subspace, without a full-dimensional Lebesgue density.
    Independent observation noise with positive variance makes the sales
    covariance full rank; it does not make the latent walk full rank.

    A one-week path has ``A = 0`` and no calibratable amplitude, so
    :func:`_kernel_width` rejects it. In generation the operator spans the
    reported window plus burn-in. Labels describe this full-path calibration,
    not the realized reported-window standard deviation. For positive walks,
    ``std`` is the amplitude before softplus. Unlike these walks, ``RW_Y`` is
    iid observation noise with an exact per-week Normal standard deviation.
    """
    return _walk_operator_scale(_centred_walk_operator(n_time_steps, width))


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
    columns = _centred_walk_operator(n_time_steps, width)
    return columns / _walk_operator_scale(columns)


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
    """Create a PyTensor symbolic random walk.

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
