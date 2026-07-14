"""Random-walk noise generator for the Phase-4 additive causal graph.

Every node in the additive SCM (D, Z, C, B, Y) carries an independent
random-walk noise term instead of flat Gaussian noise (plan doc 03, D1a).
The walk is always autocorrelated — a smoothed Brownian motion — and the
``smoothness`` parameter controls its texture:

* ``smoothness -> 0``: raw Brownian motion — jagged, hectic, but cumulative.
* ``smoothness -> 1``: wide moving-average of the Brownian path — a soft,
  slow sinusoidal-like drift.

Three entry points:

* :func:`generate_random_walk` — concrete numpy walk (data generation).
* :func:`sample_rw_params` — draw walk parameters from their priors.
* :func:`symbolic_random_walk` — same math as a PyTensor expression so the
  walk can live inside the symbolic causal graph (intervention-based
  decomposition evaluates the same graph twice).

The numpy and symbolic implementations share the identical smoothing
algorithm (edge-padded moving average via cumulative sums) so that
evaluating the symbolic walk with the same white noise reproduces the
numpy walk bit-for-bit (tested in ``tests/data/test_random_walk.py``).

PyTensor is imported lazily inside :func:`symbolic_random_walk` so the
numpy path stays import-light.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "generate_random_walk",
    "sample_rw_params",
    "softplus",
    "symbolic_random_walk",
]


def softplus(x):
    """Numerically stable softplus — THE positivity transform of this module.

    Positive-only walks are ``softplus(pre-activation)``, so ``softplus(mean)``
    is the canonical "level" anchor for anything that scales relative to a
    positive walk (e.g. the channel-texture factors in ``symbolic_graph``).
    Keep both call sites on this helper so they can never diverge.
    """
    return np.logaddexp(0.0, x)


def _kernel_width(smoothness: float, T: int) -> int:
    """Moving-average kernel width for a given smoothness (static int).

    Low smoothness -> width 1 (no smoothing, raw Brownian). High smoothness
    -> width ~T/4 (very smooth drift). Always at least 1.
    """
    if not 0.0 <= smoothness <= 1.0:
        raise ValueError(f"smoothness must be in [0, 1], got {smoothness}")
    return max(1, int(round(smoothness * T / 4.0)))


def generate_random_walk(
    T: int,
    mean: float,
    std: float,
    smoothness: float,
    positive_only: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random-walk time series.

    Parameters
    ----------
    T : int
        Number of time steps.
    mean : float
        Center of the walk.
    std : float
        Amplitude constraint (target standard deviation of the walk before
        the optional positivity transform).
    smoothness : float
        0.0 = raw Brownian motion (jagged), 1.0 = very smooth drift.
        The walk is always autocorrelated — never white noise.
    positive_only : bool
        If True, constrain output to positive values via softplus (smooth,
        differentiable). ``mean`` should be drawn from a positive range so
        the walk stays clearly positive after softplus.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (T,)
        The random walk time series (float64).
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if std < 0:
        raise ValueError(f"std must be >= 0, got {std}")
    eps = rng.standard_normal(T)
    return _walk_from_eps_numpy(eps, mean, std, smoothness, positive_only)


def _walk_from_eps_numpy(
    eps: np.ndarray,
    mean: float,
    std: float,
    smoothness: float,
    positive_only: bool,
) -> np.ndarray:
    """Deterministic walk construction from white noise (numpy)."""
    T = eps.shape[0]
    raw = np.cumsum(eps)  # Brownian base — always a walk
    w = _kernel_width(smoothness, T)
    if w > 1:
        left, right = w // 2, w - 1 - w // 2
        padded = np.concatenate([np.full(left, raw[0]), raw, np.full(right, raw[-1])])
        c = np.cumsum(padded)
        window_sums = c[w - 1 :] - np.concatenate([[0.0], c[: T - 1]])
        walk = window_sums / w
    else:
        walk = raw
    walk = walk - walk.mean()
    walk = walk * std / (walk.std() + 1e-8)
    walk = walk + mean
    if positive_only:
        walk = softplus(walk)
    return np.asarray(walk)


def sample_rw_params(
    positive_only: bool,
    rng: np.random.Generator,
    mean_range: tuple[float, float] = (-1.0, 1.0),
    positive_mean_range: tuple[float, float] = (0.5, 3.0),
    std_sigma: float = 1.0,
    smoothness_alpha: float = 2.0,
    smoothness_beta: float = 2.0,
    std_range: tuple[float, float] | None = None,
    std_relative: bool = False,
) -> dict[str, Any]:
    """Sample random-walk parameters from their priors (plan doc 4.0b).

    Priors:

    * ``mean ~ Uniform(mean_range)`` for signed nodes,
      ``Uniform(positive_mean_range)`` for positive-only nodes (channels).
    * ``std ~ HalfNormal(std_sigma)``, or ``Uniform(std_range)`` when
      ``std_range`` is given. The HalfNormal piles mass at 0 (near-constant
      walks); a uniform range floors the walk amplitude — used for channels,
      whose contribution targets otherwise degenerate to flat lines.
      With ``std_relative=True`` (positive-only walks only) the drawn factor
      is multiplied by the walk's level ``softplus(mean)``, making the
      amplitude scale-free across small and large nodes; the range then reads
      like a CV range.
    * ``smoothness ~ Beta(smoothness_alpha, smoothness_beta)``.

    Returns
    -------
    dict
        Keys ``mean``, ``std``, ``smoothness``, ``positive_only`` — plugs
        directly into :func:`generate_random_walk` as keyword arguments.
    """
    if std_relative and not positive_only:
        raise ValueError("std_relative=True requires positive_only=True (softplus level anchor)")
    lo, hi = positive_mean_range if positive_only else mean_range
    # Draw order (mean, std, smoothness) is LOCKED: it defines the RNG stream
    # of every corpus generated so far, and std_range=None must reproduce
    # legacy corpora byte-for-byte. std_relative multiplies AFTER the draw,
    # consuming no extra RNG.
    mean = float(rng.uniform(lo, hi))
    if std_range is not None:
        std = float(rng.uniform(std_range[0], std_range[1]))
        if std_relative:
            std *= float(softplus(mean))
    else:
        std = float(abs(rng.normal(0.0, std_sigma)))
    return {
        "mean": mean,
        "std": std,
        "smoothness": float(rng.beta(smoothness_alpha, smoothness_beta)),
        "positive_only": positive_only,
    }


def symbolic_random_walk(
    T: int,
    mean,
    std,
    smoothness: float,
    positive_only: bool,
    eps=None,
):
    """Create a PyTensor symbolic random walk (plan doc 4.0c).

    Same construction as :func:`generate_random_walk`, expressed
    symbolically so the walk can be part of the causal graph. Concrete
    values (``eps``, ``mean``, ``std``) are supplied at ``.eval()`` time.

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
    walk = walk * std / (walk.std() + 1e-8)
    walk = walk + mean
    if positive_only:
        walk = pt.softplus(walk)
    return walk
