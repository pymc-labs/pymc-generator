"""Media-response mechanism families (adstock + saturation).

Thin κ-relative wrappers over ``pymc_marketing.mmm.transformers``: the
adstock and saturation math is provided by pymc-marketing (``geometric_adstock``
/ ``weibull_adstock`` and the standard saturation curves), not hand-rolled here.
pymc-marketing 1.0's transformers are xtensor/named-dim based, so each wrapper
bridges a plain ``(n_time_steps,)`` time column through ``as_xtensor(dims=("time",))``,
applies the library function along the time dim, and returns the underlying
tensor via ``.values``. Parameters may be concrete floats or symbolic
(pytensor / PyMC RV) scalars — both compose into the graph.

Each family's half-point or scale is relative to the caller-supplied
``reference_level``: either ``kappa = kappa_mult * reference_level`` or input
rescaling by ``x / reference_level``. This anchor depends on structural
parameters, not a reduction over the realized series. Wrappers have signature
``f(x, reference_level, **shape_params) -> tensor`` and are monotone in ``x``.

Every family is normalized to a **unit asymptote** (or, for the unbounded
``root``, to ``f(reference_level) = 1``), so the channel's single amplitude is the
structural coefficient ``beta`` in :mod:`pymc_generator.symbolic_graph`.
This mirrors pymc-marketing's own convention -- its wrappers add a ``beta``
scale exactly to those families whose transformer is bounded, and omit it for
``michaelis_menten`` / ``tanh`` whose transformer already exposes an asymptote.
Carrying both would make ``beta`` and the family asymptote a pure product: the
contribution identifies only ``beta * asymptote``, leaving each factor free to
slide along a ridge.

Family parameterizations below use ``r = reference_level``:

==================  =========================================================
``hill``            ``hill_function(x, slope, κ)`` with κ = kappa_mult·r.
                    f(κ) = 0.5 exactly for any slope; asymptote 1.
``logistic``        ``logistic_saturation(x / r, lam)``; half-point at
                    x = ln(3)/lam · r; asymptote 1.
``michaelis_menten``  ``michaelis_menten(x, 1, κ)`` with
                    κ = kappa_mult·r. f(κ) = 0.5; asymptote 1.
``tanh``            ``tanh_saturation(x / r, 1, c)`` =
                    tanh(x/(r·c)); asymptote 1; initial slope 1/(c·r).
``root``            ``root_saturation(x / r, alpha)`` = (x/r)^alpha;
                    f(r) = 1; concave for alpha < 1, no asymptote.
==================  =========================================================
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytensor.tensor as pt
from pymc_marketing.mmm import transformers as _pmm
from pytensor.tensor import TensorVariable
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable


def _as_time(x: TensorVariable) -> XTensorVariable:
    """Wrap a ``(n_time_steps,)`` time column as an xtensor with a named ``time`` dim."""
    result: XTensorVariable = as_xtensor(pt.as_tensor_variable(x), dims=("time",))
    return result


# --------------------------------------------------------------------------
# Saturation transformers (pymc-marketing, bridged to time-column tensors)
# --------------------------------------------------------------------------


def hill_function(
    x: TensorVariable, slope: TensorVariable, kappa: TensorVariable
) -> TensorVariable:
    """Hill saturation: 1 - kappa^slope / (kappa^slope + x^slope).

    Property: f(kappa) = 0.5 for any slope; asymptote 1 as x -> inf.
    Delegates to ``pymc_marketing.mmm.transformers.hill_function``.
    """
    return _pmm.hill_function(_as_time(x), slope=slope, kappa=kappa).values


def logistic_saturation(x: TensorVariable, lam: TensorVariable) -> TensorVariable:
    """Logistic saturation (asymptote 1); ``pymc_marketing`` ``logistic_saturation``."""
    return _pmm.logistic_saturation(_as_time(x), lam=lam).values


def michaelis_menten(
    x: TensorVariable, alpha: TensorVariable | float, lam: TensorVariable
) -> TensorVariable:
    """Michaelis–Menten: alpha * x / (lam + x); ``pymc_marketing`` ``michaelis_menten``."""
    return _pmm.michaelis_menten(_as_time(x), alpha=alpha, lam=lam).values


def tanh_saturation(
    x: TensorVariable, b: TensorVariable | float, c: TensorVariable
) -> TensorVariable:
    """Tanh saturation: b * tanh(x / (b * c)); ``pymc_marketing`` ``tanh_saturation``."""
    return _pmm.tanh_saturation(_as_time(x), b=b, c=c).values


def root_saturation(x: TensorVariable, alpha: TensorVariable) -> TensorVariable:
    """Root saturation: x^alpha; ``pymc_marketing`` ``root_saturation``."""
    return _pmm.root_saturation(_as_time(x), alpha=alpha).values


# --------------------------------------------------------------------------
# Saturation families — dimensionless, reference-relative wrappers
# --------------------------------------------------------------------------


#: Sensible Uniform prior ranges per family shape parameter. Chosen so that on
#: an input with mean 1 every family is monotone increasing and O(1)-scaled
#: (max ≲ 2 on x ∈ [0, 2]).
SATURATION_PRIOR_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "hill": {"slope": (1.0, 3.0), "kappa_mult": (0.7, 1.5)},
    "logistic": {"lam": (0.5, 3.0)},
    "michaelis_menten": {"kappa_mult": (0.7, 1.5)},
    "tanh": {"c": (0.3, 1.5)},
    "root": {"alpha": (0.3, 0.9)},
}


def hill_kappa_relative(x, reference_level, *, slope, kappa_mult) -> TensorVariable:
    """Hill curve with κ = kappa_mult · reference_level; unit asymptote."""
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return hill_function(x, slope, kappa_mult * safe_reference_level)


def logistic_kappa_relative(x, reference_level, *, lam) -> TensorVariable:
    """Logistic curve with half-point ln(3)/lam · reference_level."""
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return logistic_saturation(x / safe_reference_level, lam)


def michaelis_menten_kappa_relative(x, reference_level, *, kappa_mult) -> TensorVariable:
    """Michaelis-Menten curve with λ = kappa_mult · reference_level.

    The library asymptote is pinned to 1 rather than exposed as a parameter:
    the structural ``beta`` gate is this channel's only amplitude.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return michaelis_menten(x, 1.0, kappa_mult * safe_reference_level)


def tanh_kappa_relative(x, reference_level, *, c) -> TensorVariable:
    """Unit-asymptote tanh(x / (reference_level · c)).

    The library asymptote ``b`` is pinned to 1 for the same reason as
    ``michaelis_menten``: ``(b, c) -> (λb, c/λ)`` scales the response by ``λ``
    without changing its shape, so a free ``b`` only duplicates ``beta``.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return tanh_saturation(x / safe_reference_level, 1.0, c)


def root_kappa_relative(x, reference_level, *, alpha) -> TensorVariable:
    """Root curve (x / reference_level)^alpha; f(reference_level) = 1."""
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    ratio = pt.maximum(x / safe_reference_level, 0.0)
    return root_saturation(ratio, alpha)


#: Name -> wrapper with signature ``f(x, reference_level, **shape_params)``.
SATURATION_FAMILIES: dict[str, Callable[..., TensorVariable]] = {
    "hill": hill_kappa_relative,
    "logistic": logistic_kappa_relative,
    "michaelis_menten": michaelis_menten_kappa_relative,
    "tanh": tanh_kappa_relative,
    "root": root_kappa_relative,
}


# --------------------------------------------------------------------------
# Adstock — pymc-marketing transformers over a single time column (axis 0)
# --------------------------------------------------------------------------


def apply_geometric_adstock(
    x: TensorVariable, alpha: TensorVariable | float, l_max: int
) -> TensorVariable:
    """Normalized geometric adstock of a ``(n_time_steps, 1)`` column over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.geometric_adstock`` (ConvMode
    ``After``, ``normalize=True``). ``alpha`` may be a float or a symbolic scalar.
    """
    out: XTensorVariable = _pmm.geometric_adstock(
        _as_time(x[:, 0]), alpha=alpha, l_max=int(l_max), dim="time", normalize=True
    )
    values: TensorVariable = out.values[:, None]
    return values


def apply_weibull_pdf_adstock(
    x: TensorVariable, lam: TensorVariable | float, k: TensorVariable | float, l_max: int
) -> TensorVariable:
    """Normalized min-max-rescaled Weibull-density adstock over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.weibull_adstock`` with
    ``type="PDF"``, ``normalize=True``. The library min-max rescales the sampled
    density before sum-normalizing, so this kernel is not a Weibull PDF:
    ``min(w) == 0`` exactly and one or more lags are always annihilated. Under
    the default prior (``lam ~ U(2, 8)``, ``k ~ U(1.5, 4)``, ``l_max = 8``),
    45.0% of channels have zero current-week weight and 38.5% peak at lag >= 5
    (measured over 200k draws). For a zero current-week weight,
    ``contributions[t]`` is independent of ``channels[t]``;
    ``frac_zero_contemporaneous_weight`` reports this diagnostic.

    ``l_max == 1`` deliberately returns ``x`` rather than calling the library:
    the singleton causal kernel is an identity, while the library's min-max
    rescaling has ``min == max`` and returns NaN. ``lam``/``k`` may be floats
    or symbolic.
    At extreme scales where ``lam >> l_max`` and ``k`` is large, the analytic
    density can underflow into denormals. An analytic magnitude floor zeroes
    this library-breakdown regime without inspecting library output; the NumPy
    diagnostic uses the same floor, so the symbolic and numeric paths agree.
    """
    if int(l_max) == 1:
        return x
    lag = pt.arange(int(l_max), dtype=x.dtype) + 1
    lam_t = pt.as_tensor_variable(lam)
    k_t = pt.as_tensor_variable(k)
    raw_weights = (k_t / lam_t) * pt.pow(lag / lam_t, k_t - 1) * pt.exp(-pt.pow(lag / lam_t, k_t))
    raw_weight_max = raw_weights.max()
    weight_min = raw_weights.min()
    weight_span = raw_weight_max - weight_min
    minmax_weights = (raw_weights - weight_min) / weight_span
    weight_total = minmax_weights.sum()
    out: XTensorVariable = _pmm.weibull_adstock(
        _as_time(x[:, 0]),
        lam=lam,
        k=k,
        l_max=int(l_max),
        dim="time",
        type="PDF",
        normalize=True,
    )
    values = out.values[:, None]
    # This guard MUST track signal_diagnostics._adstock_weights. The oracle passes
    # symbolic value variables; a backend-dependent library-output guard would make
    # FAST_COMPILE generation and the FAST_RUN oracle disagree whether a channel responds.
    # 1e-300 is above the float64 denormal cliff (~5e-324), yet below any
    # normal-magnitude density, so the analytic replica detects only underflow.
    kernel_is_valid = pt.and_(
        pt.and_(
            pt.and_(pt.isfinite(weight_span), pt.invert(pt.eq(weight_span, 0))),
            pt.and_(pt.isfinite(weight_total), pt.invert(pt.eq(weight_total, 0))),
        ),
        pt.gt(raw_weight_max, np.float64(1e-300)),
    )
    result: TensorVariable = pt.switch(
        pt.invert(kernel_is_valid),
        pt.zeros_like(values),
        values,
    )
    return result
