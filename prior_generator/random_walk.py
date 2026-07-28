"""Symbolic random-walk noise for the Phase-4 additive causal graph.

Every node in the additive SCM (D, Z, C, B, Y) carries a random-walk noise
term instead of flat Gaussian noise (plan doc 03, D1a).
The walk is always autocorrelated — a smoothed Brownian motion — and the
``smoothness`` parameter controls its texture:

* ``smoothness -> 0``: raw Brownian motion — jagged, hectic, but cumulative.
* ``smoothness -> 1``: wide moving-average of the Brownian path — a soft,
  slow sinusoidal-like drift.

The symbolic expression lives inside the causal graph so intervention-based
decomposition evaluates the same graph twice. PyTensor is imported lazily so
importing this module stays light.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["symbolic_random_walk"]


def _kernel_width(smoothness: float, T: int) -> int:
    """Moving-average kernel width for a given smoothness (static int).

    Low smoothness -> width 1 (no smoothing, raw Brownian). High smoothness
    -> width ~T/4 (very smooth drift). Always at least 1.
    """
    if not 0.0 <= smoothness <= 1.0:
        raise ValueError(f"smoothness must be in [0, 1], got {smoothness}")
    return max(1, int(round(smoothness * T / 4.0)))


@lru_cache(maxsize=64)
def _centred_walk_scale(T: int, width: int) -> float:
    """RMS amplitude of the centred, smoothed Brownian path for unit innovations.

    The walk is ``mean + std * (A @ eps) / c`` where ``A = centre . movavg .
    cumsum`` is a FIXED linear operator and ``c`` is the constant returned
    here, ``sqrt(tr(A A^T) / T)``. Dividing by this constant instead of by the
    path's own realized standard deviation is what keeps the map injective:
    normalizing by ``walk.std()`` makes ``eps -> walk`` invariant to
    ``eps -> k * eps``, leaving the likelihood exactly flat along the radial
    direction of a T-dimensional latent, which no amount of step-size tuning
    can fix.

    The price is that ``std`` becomes the walk's EXPECTED amplitude
    (``E[var(walk)] == std ** 2``) rather than its exact realized one. The gain
    is that the walk is now an ordinary multivariate normal with covariance
    ``(std / c) ** 2 * A A^T`` -- a distribution with a density, which the
    normalized version did not have.

    In generation, ``T`` is the full horizon
    ``T_full = T_reported + adstock_burn_in``. Persisted
    ``param_rw_*_std`` labels declare the walk's expected standard deviation
    over ``T_full``. Because each path is divided by a fixed constant rather
    than by its own realized standard deviation, that scale is realized only in
    expectation. It is therefore neither the realized standard deviation of an
    individual path nor a standard deviation measured only over the reported
    window. ``smoothness`` maps to a kernel width in ``T_full`` weeks. For
    positive-only walks, ``std`` is the pre-softplus amplitude, so ``rw_c`` is
    excluded from the signed-walk table below rather than reported with a
    misleadingly wide range.

    Across 40 signed walks from eight worlds at ``T=52`` and
    ``adstock_burn_in=8``, reported-window sd / declared ``std`` was:

    * ``rw_d``: [0.396, 0.927], median 0.690
    * ``rw_z``: [0.252, 1.985], median 0.868
    * ``rw_b``: [0.264, 1.809], median 0.814
    * ``rw_y``: [0.294, 1.335], median 0.757
    * all signed (n=40): [0.25, 1.99], median 0.80

    This is roughly an 8x spread. ``param_rw_*_std`` is therefore a weak label
    for anything measured on the reported window; consumers should not score it
    as if it were the realized reported-window standard deviation.

    Column ``j`` of ``A`` is the smoothed, centred step function
    ``1[t >= j]``, so the whole operator is built in one ``(T, T)`` pass.
    """
    steps = np.tril(np.ones((T, T)))  # steps[t, j] = 1 if t >= j (the cumsum operator)
    columns = _smooth_columns_numpy(steps, width)
    columns = columns - columns.mean(axis=0, keepdims=True)
    return float(np.sqrt((columns**2).sum() / T))


def _smooth_columns_numpy(raw: np.ndarray, width: int) -> np.ndarray:
    """Edge-padded moving average down axis 0 used to build the scale operator."""
    if width <= 1:
        return raw
    T = raw.shape[0]
    left, right = width // 2, width - 1 - width // 2
    padded = np.concatenate(
        [np.repeat(raw[:1], left, axis=0), raw, np.repeat(raw[-1:], right, axis=0)]
    )
    cumulative = np.cumsum(padded, axis=0)
    head = np.zeros((1, *raw.shape[1:]), dtype=cumulative.dtype)
    window_sums = cumulative[width - 1 :] - np.concatenate([head, cumulative[: T - 1]])
    return np.asarray(window_sums / width)


def symbolic_random_walk(
    T: int,
    mean,
    std,
    smoothness: float,
    positive_only: bool,
    eps=None,
):
    """Create a PyTensor symbolic random walk (plan doc 4.0c).

    The expression is part of the causal graph. Concrete values (``eps``,
    ``mean``, ``std``) are supplied at ``.eval()`` time.

    Parameters
    ----------
    T : int
        Number of time steps (static — determines graph shape).
    mean, std : pt.TensorVariable | float
        Walk center and amplitude (symbolic scalars or floats).
    smoothness : float
        Static smoothness in [0, 1]. This must be a Python float (not a
        tensor) because the moving-average kernel width is a structural
        property of the graph.
    positive_only : bool
        If True, apply softplus at the end.
    eps : pt.TensorVariable | None
        White-noise input of shape (T,). If None, a fresh ``pt.vector``
        named ``"eps"`` is created.

    Returns
    -------
    pt.TensorVariable of shape (T,)
    """
    import pytensor.tensor as pt

    if eps is None:
        eps = pt.vector("eps")
    raw = pt.cumsum(eps)
    w = _kernel_width(smoothness, T)
    if w > 1:
        left, right = w // 2, w - 1 - w // 2
        padded = pt.concatenate([pt.tile(raw[0], left), raw, pt.tile(raw[-1], right)])
        c = pt.cumsum(padded)
        window_sums = c[w - 1 :] - pt.concatenate([pt.zeros(1), c[: T - 1]])
        walk = window_sums / w
    else:
        walk = raw
    walk = walk - walk.mean()
    walk = walk * std / _centred_walk_scale(T, w)
    walk = walk + mean
    if positive_only:
        walk = pt.softplus(walk)
    return walk
