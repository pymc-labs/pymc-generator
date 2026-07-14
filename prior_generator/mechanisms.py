"""Media-response mechanism families (adstock + saturation), pure PyTensor.

The adstock and saturation transforms are implemented directly in pytensor
with documented math parity to ``pymc_marketing.mmm.transformers``
(``geometric_adstock``/``weibull_adstock`` with ``ConvMode.After``, and the
standard saturation curves) — keeping them local avoids depending on
pymc-marketing's tensor-API compatibility layer while staying faithful to its
definitions. κ-relative wrappers make every family scale-free.

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
from pytensor.tensor import TensorVariable

# --------------------------------------------------------------------------
# Adstock transformers (direct pytensor — no pymc-marketing dependency)
#
# These are faithful to the math of pymc-marketing's
# geometric_adstock / weibull_adstock with axis=0, normalise=True/False.
# Keeping them local avoids the fragile pymc-marketing XTensor/dim API
# compatibility layer.
# --------------------------------------------------------------------------


def _conv1d_right(x: TensorVariable, w: TensorVariable) -> TensorVariable:
    """Causal 1-D convolution along axis 0 (ConvMode.After semantics).

    Matches pymc-marketing ``batched_convolution(..., mode=ConvMode.After)``:
    ``y[t] = sum_{l=0}^{l_max-1} w[l] * x[t - l]`` with ``x[negative] = 0``.

    Parameters
    ----------
    x : (T, K)
    w : (l_max,) or (K, l_max)
        If 2-D, kernel w[k, :] applies to channel k.

    Returns
    -------
    (T, K) — same shape as x, with trailing time convolved.
    """
    l_max = w.type.shape[-1]
    if l_max is None:
        raise ValueError("w must have a static l_max on its last axis")
    # Left-pad x with (l_max - 1) zeros along the time axis
    pad_shape = list(x.type.shape)
    pad_shape[0] = l_max - 1
    pad = pt.zeros(pad_shape, dtype=x.dtype)
    x_pad = pt.concatenate([pad, x], axis=0)  # (T + l_max - 1, K)

    # Sliding windows with the kernel reversed: y[t] = sum_l w[l] * x[t - l]
    # x_pad[n] = x[n - (l_max - 1)] (padding zeros occupy indices < l_max - 1)
    # Want windows[t, l] such that w[l] multiplies x[t - l]:
    #   windows[t, l] = x_pad[t - l + (l_max - 1)] = x_pad[(l_max - 1 - l) + t]
    # Indexing: idx[t, l] = (l_max - 1 - l) + t
    idx = pt.arange(l_max - 1, -1, -1)[None, :] + pt.arange(x.type.shape[0])[:, None]
    windows = x_pad[idx]  # (T, l_max, K)

    # Apply kernel — w: (l_max,) broadcasts against (T, l_max, K)
    if w.type.ndim == 1:
        return (windows * w[None, :, None]).sum(axis=1)
    # w: (K, l_max) — transpose to (l_max, K) for broadcast
    return (windows * w.T[None, :, :]).sum(axis=1)


def geometric_adstock(
    x: TensorVariable, alpha: TensorVariable, l_max: int, normalize: bool = True
) -> TensorVariable:
    """Geometric adstock with weights ``w_l = alpha^l``, optional normalisation.

    Matches pymc-marketing geometric_adstock (ConvMode.After): adstock peaks
    at the exposure period and decays geometrically over l_max lags.

    Parameters
    ----------
    x : (T, K) — input time series (e.g. spend), time on axis 0.
    alpha : (K,) or scalar — retention rate in [0, 1].
    l_max : int — number of lags.
    normalize : bool — if True, divide weights by their sum so the total
        adstocked mass equals the sum of x (useful for interpreting
        alpha as "how spread out" the effect is, not a gain term).
    """
    alpha = pt.as_tensor_variable(alpha)
    lags = pt.arange(l_max, dtype=x.dtype)  # (l_max,)
    # Broadcast: alpha (K,) ** lags (l_max,) -> (K, l_max) when alpha has trailing dim
    if alpha.type.ndim == 0:
        w = alpha**lags  # (l_max,)
    else:
        w = alpha[:, None] ** lags[None, :]  # (K, l_max)
    if normalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return _conv1d_right(x, w)


def weibull_adstock_pdf(
    x: TensorVariable, lam: TensorVariable, k: TensorVariable, l_max: int, normalize: bool = True
) -> TensorVariable:
    """Weibull-PDF adstock with weights ``w_l = Weibull.pdf(l; k, lam)``.

    Matches pymc-marketing ``weibull_adstock(type="PDF")``: kernel is the PDF
    of a Weibull(k, lam) evaluated at integer lags 1..l_max. pymc-marketing
    unconditionally rescales the raw PDF to [0, 1] via min-max before optional
    normalisation — that is replicated here so outputs match bit-for-bit.

    Parameters
    ----------
    x : (T, K) — input time series, time on axis 0.
    lam : (K,) or scalar — Weibull scale (>0).
    k : (K,) or scalar — Weibull shape (>0).
    l_max : int — number of lags.
    normalize : bool — rescale weights to sum to 1 (after min-max).
    """
    lam = pt.as_tensor_variable(lam)
    k = pt.as_tensor_variable(k)
    lags = pt.arange(1, l_max + 1, dtype=x.dtype)  # (l_max,)
    # Weibull PDF at integer lags: f(l) = (k/lam) * (l/lam)^(k-1) * exp(-(l/lam)^k)
    if lam.type.ndim == 0 and k.type.ndim == 0:
        ratio = lags / lam
        pdf_raw = (k / lam) * ratio ** (k - 1) * pt.exp(-(ratio**k))
    else:
        ratio = lags[None, :] / lam[:, None]  # (K, l_max)
        pdf_raw = (
            (k[:, None] / lam[:, None]) * ratio ** (k[:, None] - 1) * pt.exp(-(ratio ** k[:, None]))
        )
    # pymc-marketing applies min-max rescaling unconditionally for PDF type:
    # w in [0, 1] before normalization (helps with numerical stability)
    pdf_min = pdf_raw.min(axis=-1, keepdims=True)
    pdf_max = pdf_raw.max(axis=-1, keepdims=True)
    pdf = (pdf_raw - pdf_min) / (pdf_max - pdf_min + 1e-12)
    if normalize:
        pdf = pdf / (pdf.sum(axis=-1, keepdims=True) + 1e-12)
    return _conv1d_right(x, pdf)


# --------------------------------------------------------------------------
# Saturation transformers (direct pytensor — pymc-marketing-free)
# --------------------------------------------------------------------------


def hill_function(
    x: TensorVariable, slope: TensorVariable, kappa: TensorVariable
) -> TensorVariable:
    """Hill saturation: 1 - kappa^slope / (kappa^slope + x^slope).

    Property: f(kappa) = 0.5 for any slope; asymptote 1 as x -> inf.
    """
    x = pt.as_tensor_variable(x)
    slope = pt.as_tensor_variable(slope)
    kappa = pt.as_tensor_variable(kappa)
    return 1.0 - kappa**slope / (kappa**slope + x**slope)


def logistic_saturation(x: TensorVariable, lam: TensorVariable) -> TensorVariable:
    """Logistic saturation: (1 - exp(-lam*x)) / (1 + exp(-lam*x)); asymptote 1."""
    x = pt.as_tensor_variable(x)
    lam = pt.as_tensor_variable(lam)
    e = pt.exp(-lam * x)
    return (1.0 - e) / (1.0 + e)


def michaelis_menten(
    x: TensorVariable, alpha: TensorVariable, lam: TensorVariable
) -> TensorVariable:
    """Michaelis–Menten: alpha * x / (lam + x); asymptote alpha."""
    x = pt.as_tensor_variable(x)
    alpha = pt.as_tensor_variable(alpha)
    lam = pt.as_tensor_variable(lam)
    return alpha * x / (lam + x)


def tanh_saturation(x: TensorVariable, b: TensorVariable, c: TensorVariable) -> TensorVariable:
    """Tanh saturation: b * tanh(x / (b * c)); asymptote b, initial slope 1/c."""
    x = pt.as_tensor_variable(x)
    b = pt.as_tensor_variable(b)
    c = pt.as_tensor_variable(c)
    return b * pt.tanh(x / (b * c))


def root_saturation(x: TensorVariable, alpha: TensorVariable) -> TensorVariable:
    """Root saturation: x^alpha; no finite asymptote; monotone concave for alpha<1."""
    x = pt.as_tensor_variable(x)
    alpha = pt.as_tensor_variable(alpha)
    return x**alpha


# --------------------------------------------------------------------------
# Saturation families — κ-relative wrappers (FINDINGS D7)
# --------------------------------------------------------------------------

#: Canonical family order — index in this tuple is the family id used with
#: `select_family` (e.g. a per-channel Categorical over families).
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
# Adstock — convenience wrappers over the pytensor implementations above
# (normalized, time axis 0)
# --------------------------------------------------------------------------


def apply_geometric_adstock(x, alpha, l_max: int) -> TensorVariable:
    """Normalized geometric adstock over the time axis (axis=0)."""
    return geometric_adstock(
        pt.as_tensor_variable(x), pt.as_tensor_variable(alpha), l_max=int(l_max), normalize=True
    )


def apply_weibull_pdf_adstock(x, lam, k, l_max: int) -> TensorVariable:
    """Normalized Weibull-PDF adstock over the time axis (axis=0)."""
    return weibull_adstock_pdf(
        pt.as_tensor_variable(x),
        pt.as_tensor_variable(lam),
        pt.as_tensor_variable(k),
        l_max=int(l_max),
        normalize=True,
    )
