"""Exact finite-width prior for an oracle that does not know walk smoothness.

Generation maps Beta smoothness onto rounded, clamped integer widths. Integrating
that discrete choice out is exact: it does not substitute mean smoothness or use
the world's realized width. This module has no generation-side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc, betaincc, betainccinv, betaincinv

from .random_walk import _walk_basis_stack


@dataclass(frozen=True)
class WalkWidthPrior:
    """The pushforward of a Beta prior through the generator's width rule."""

    widths: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    alpha: float
    beta: float

    def sample_smoothness(self, width_index: int, rng: np.random.Generator) -> float:
        """Draw continuous smoothness conditional on an inferred 0-based width.

        Use the survival function in the upper tail to avoid subtracting two
        CDF values rounded to one. Widths remain the identifiable quantity.
        """
        if not 0 <= width_index < len(self.widths):
            raise ValueError("width_index is outside the prior support")
        lo, hi = self.lower[width_index], self.upper[width_index]
        if lo > self.alpha / (self.alpha + self.beta):
            s_lo = betaincc(self.alpha, self.beta, lo)
            s_hi = betaincc(self.alpha, self.beta, hi)
            return float(betainccinv(self.alpha, self.beta, rng.uniform(s_hi, s_lo)))
        c_lo = betainc(self.alpha, self.beta, lo)
        c_hi = betainc(self.alpha, self.beta, hi)
        return float(betaincinv(self.alpha, self.beta, rng.uniform(c_lo, c_hi)))


def walk_width_prior(
    n_time_steps: int, max_weeks: int, alpha: float, beta: float
) -> WalkWidthPrior:
    """Integrate Beta mass over ``min(T, max(1, round(s * max_weeks)))``.

    Half-integer rounding ties have zero mass under the continuous prior. When
    the horizon caps width, its final bin includes the entire upper tail.
    """
    if (
        isinstance(n_time_steps, bool)
        or not isinstance(n_time_steps, (int, np.integer))
        or n_time_steps < 2
        or isinstance(max_weeks, bool)
        or not isinstance(max_weeks, (int, np.integer))
        or max_weeks < 1
    ):
        raise ValueError("n_time_steps must be >= 2 and max_weeks >= 1 integers")
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0:
        raise ValueError("Beta prior parameters must be finite and positive")
    widths = np.arange(1, min(n_time_steps, max_weeks) + 1, dtype="int64")
    lower = (widths.astype("float64") - 0.5) / max_weeks
    upper = (widths.astype("float64") + 0.5) / max_weeks
    lower[0], upper[-1] = 0.0, 1.0
    left = betainc(alpha, beta, upper) - betainc(alpha, beta, lower)
    right = betaincc(alpha, beta, lower) - betaincc(alpha, beta, upper)
    mass = np.where(lower > alpha / (alpha + beta), right, left)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Width prior mass is not representable in float64")
    mass /= mass.sum()
    for array in (widths, lower, upper, mass):
        array.flags.writeable = False
    return WalkWidthPrior(widths, mass, lower, upper, float(alpha), float(beta))


def reported_walk_bases(n_reported: int, burn_in: int, max_weeks: int) -> NDArray[np.float64]:
    """All generator baseline operators, retaining its full-path calibration.

    Missing adstock history does not justify changing the baseline process:
    innovations still span the generating burn-in plus the reported window.
    Returned axes are width, reported time, full-horizon innovation.
    """
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if (
        isinstance(n_reported, bool)
        or not isinstance(n_reported, (int, np.integer))
        or n_reported < 2
    ):
        raise ValueError("n_reported must be an integer >= 2")
    return np.asarray(_walk_basis_stack(n_reported + burn_in, max_weeks)[:, burn_in:, :])
