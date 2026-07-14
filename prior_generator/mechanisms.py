"""Media-response mechanism families (adstock + saturation).

Thin κ-relative wrappers over ``pymc_marketing.mmm.transformers``: the
adstock and saturation math is provided by pymc-marketing (``geometric_adstock``
/ ``weibull_adstock`` and the standard saturation curves), not hand-rolled here.
pymc-marketing 1.0's transformers are xtensor/named-dim based, so each wrapper
bridges a plain ``(T,)`` time column through ``as_xtensor(dims=("time",))``,
applies the library function along the time dim, and returns the underlying
tensor via ``.values``. Parameters may be concrete floats or symbolic
(pytensor / PyMC RV) scalars — both compose into the graph.

κ-relative parameterization (FINDINGS D7, the ``kappa_adstock_adjusted``
lesson): every family's half-point / scale parameter is expressed *relative to
the mean of the (adstocked) input series* — either an explicit
``kappa = kappa_mult * mean_x`` where the library function takes a
half-saturation argument, or by rescaling the input to ``x / mean_x`` so the
remaining shape parameters are scale-free. Each wrapper has signature
``f(x, mean_x, **shape_params) -> tensor``, is monotone increasing in ``x``,
and stays O(1) when ``x`` is on its own mean scale.

Family parameterizations (prior ranges in `SATURATION_PRIOR_RANGES`):

==================  =========================================================
``hill``            ``hill_function(x, slope, κ)`` with κ = kappa_mult·mean_x.
                    f(κ) = 0.5 exactly for any slope; asymptote 1.
``logistic``        ``logistic_saturation(x / mean_x, lam)``; half-point at
                    x = ln(3)/lam · mean_x; asymptote 1.
``michaelis_menten``  ``michaelis_menten(x, alpha, κ)`` with
                    κ = kappa_mult·mean_x. f(κ) = alpha/2; asymptote alpha.
``tanh``            ``tanh_saturation(x / mean_x, b, c)`` =
                    b·tanh(x/(mean_x·b·c)); asymptote b; initial slope
                    1/(c·mean_x).
``root``            ``root_saturation(x / mean_x, alpha)`` = (x/mean_x)^alpha;
                    f(mean_x) = 1; concave for alpha < 1, no asymptote.
==================  =========================================================
"""

from __future__ import annotations

from collections.abc import Callable

import pytensor.tensor as pt
from pymc_marketing.mmm import transformers as _pmm
from pytensor.tensor import TensorVariable
from pytensor.xtensor import as_xtensor


def _as_time(x: TensorVariable) -> TensorVariable:
    """Wrap a ``(T,)`` time column as an xtensor with a named ``time`` dim."""
    return as_xtensor(pt.as_tensor_variable(x), dims=("time",))


# --------------------------------------------------------------------------
# Saturation transformers (pymc-marketing, bridged to time-column tensors)
# --------------------------------------------------------------------------


def hill_function(x: TensorVariable, slope: TensorVariable, kappa: TensorVariable) -> TensorVariable:
    """Hill saturation: 1 - kappa^slope / (kappa^slope + x^slope).

    Property: f(kappa) = 0.5 for any slope; asymptote 1 as x -> inf.
    Delegates to ``pymc_marketing.mmm.transformers.hill_function``.
    """
    return _pmm.hill_function(_as_time(x), slope=slope, kappa=kappa).values


def logistic_saturation(x: TensorVariable, lam: TensorVariable) -> TensorVariable:
    """Logistic saturation (asymptote 1); ``pymc_marketing`` ``logistic_saturation``."""
    return _pmm.logistic_saturation(_as_time(x), lam=lam).values


def michaelis_menten(
    x: TensorVariable, alpha: TensorVariable, lam: TensorVariable
) -> TensorVariable:
    """Michaelis–Menten: alpha * x / (lam + x); ``pymc_marketing`` ``michaelis_menten``."""
    return _pmm.michaelis_menten(_as_time(x), alpha=alpha, lam=lam).values


def tanh_saturation(x: TensorVariable, b: TensorVariable, c: TensorVariable) -> TensorVariable:
    """Tanh saturation: b * tanh(x / (b * c)); ``pymc_marketing`` ``tanh_saturation``."""
    return _pmm.tanh_saturation(_as_time(x), b=b, c=c).values


def root_saturation(x: TensorVariable, alpha: TensorVariable) -> TensorVariable:
    """Root saturation: x^alpha; ``pymc_marketing`` ``root_saturation``."""
    return _pmm.root_saturation(_as_time(x), alpha=alpha).values


# --------------------------------------------------------------------------
# Saturation families — κ-relative wrappers (FINDINGS D7)
# --------------------------------------------------------------------------

#: Canonical family order — index in this tuple is the family id used per
#: channel (e.g. a per-channel categorical over families).
SATURATION_FAMILY_ORDER: tuple[str, ...] = (
    "hill",
    "logistic",
    "michaelis_menten",
    "tanh",
    "root",
)

#: Sensible Uniform prior ranges per family shape parameter. Chosen so that on
#: an input with mean 1 every family is monotone increasing and O(1)-scaled
#: (max ≲ 2 on x ∈ [0, 2]).
SATURATION_PRIOR_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "hill": {"slope": (1.0, 3.0), "kappa_mult": (0.7, 1.5)},
    "logistic": {"lam": (0.5, 3.0)},
    "michaelis_menten": {"alpha": (1.0, 2.0), "kappa_mult": (0.7, 1.5)},
    "tanh": {"b": (0.6, 1.2), "c": (0.3, 1.5)},
    "root": {"alpha": (0.3, 0.9)},
}


def hill_kappa_relative(x, mean_x, *, slope, kappa_mult) -> TensorVariable:
    """`hill_function` with κ = kappa_mult · mean_x; f(κ) = 0.5, asymptote 1."""
    safe_mean = pt.maximum(mean_x, 1e-8)
    return hill_function(x, slope, kappa_mult * safe_mean)


def logistic_kappa_relative(x, mean_x, *, lam) -> TensorVariable:
    """`logistic_saturation` on x / mean_x; half-point ln(3)/lam · mean_x."""
    safe_mean = pt.maximum(mean_x, 1e-8)
    return logistic_saturation(x / safe_mean, lam)


def michaelis_menten_kappa_relative(x, mean_x, *, alpha, kappa_mult) -> TensorVariable:
    """`michaelis_menten` with λ = kappa_mult · mean_x; f(λ) = alpha / 2."""
    safe_mean = pt.maximum(mean_x, 1e-8)
    return michaelis_menten(x, alpha, kappa_mult * safe_mean)


def tanh_kappa_relative(x, mean_x, *, b, c) -> TensorVariable:
    """`tanh_saturation` on x / mean_x: b·tanh(x / (mean_x·b·c)); asymptote b."""
    safe_mean = pt.maximum(mean_x, 1e-8)
    return tanh_saturation(x / safe_mean, b, c)


def root_kappa_relative(x, mean_x, *, alpha) -> TensorVariable:
    """`root_saturation` on x / mean_x: (x / mean_x)^alpha; f(mean_x) = 1."""
    safe_mean = pt.maximum(mean_x, 1e-8)
    ratio = pt.maximum(x / safe_mean, 0.0)
    return root_saturation(ratio, alpha)


#: name -> wrapper with uniform signature ``f(x, mean_x, **shape_params)``.
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


def apply_geometric_adstock(x, alpha, l_max: int) -> TensorVariable:
    """Normalized geometric adstock of a ``(T, 1)`` column over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.geometric_adstock`` (ConvMode
    ``After``, ``normalize=True``). ``alpha`` may be a float or a symbolic scalar.
    """
    out = _pmm.geometric_adstock(
        _as_time(x[:, 0]), alpha=alpha, l_max=int(l_max), dim="time", normalize=True
    )
    return out.values[:, None]


def apply_weibull_pdf_adstock(x, lam, k, l_max: int) -> TensorVariable:
    """Normalized Weibull-PDF adstock of a ``(T, 1)`` column over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.weibull_adstock`` with
    ``type="PDF"``, ``normalize=True``. ``lam``/``k`` may be floats or symbolic.
    """
    out = _pmm.weibull_adstock(
        _as_time(x[:, 0]),
        lam=lam,
        k=k,
        l_max=int(l_max),
        dim="time",
        type="PDF",
        normalize=True,
    )
    return out.values[:, None]
