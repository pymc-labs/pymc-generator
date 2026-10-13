"""Treatment-response mechanism families (carryover + saturation).

κ-relative wrappers over ``pymc_marketing.mmm.transformers``. Carryover and the
saturation transformers are bridged from plain ``(n_time_steps,)`` time
columns through ``as_xtensor(dims=("time",))``, then returned through ``.values``.
The default :data:`SATURATION_FAMILIES` evaluate those library graphs exactly:
their rounding and the order in which PyMC's draw walk reaches each curve's
inputs are part of the seeded-corpus contract. :data:`STABLE_SATURATION_FAMILIES`,
selected only by opt-in mechanism priors, evaluates the same unit curves with
scalar arithmetic that keeps finite values where the library's intermediate
products or automatic derivatives lose them.
Parameters may be concrete floats or symbolic (pytensor / PyMC RV) scalars —
both compose into the graph.

Each family's half-point or scale is relative to the caller-supplied
``reference_level``: either ``kappa = kappa_mult * reference_level`` or input
rescaling by ``x / reference_level``. This anchor depends on structural
parameters, not a reduction over the realized series. Wrappers have signature
``f(x, reference_level, **shape_params) -> tensor`` and are monotone in ``x``.

Every family is normalized to a **unit asymptote** (or, for the unbounded
``root``, to ``f(reference_level) = 1``), so the treatment's single amplitude is the
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
from functools import cache
from math import frexp

import numpy as np
import pytensor.scalar as ps
import pytensor.tensor as pt
from pymc_marketing.mmm import transformers as _pmm
from pytensor.compile.builders import OpFromGraph
from pytensor.link.numba.dispatch.basic import numba_njit, register_funcify_and_cache_key
from pytensor.scalar import ScalarOp, upgrade_to_float64
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

_FLOAT64_TINY = np.finfo("float64").tiny
_FLOAT64_MAX = np.finfo("float64").max


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


def _product_quotient_impl(a, b, c):
    # A zero pullback of a saturated curve is zero even if its unbounded
    # internal argument overflowed. Keep the ordinary product's zero sign.
    if (a == 0.0 or b == 0.0) and c != 0.0:
        return np.copysign(0.0, a) * np.copysign(1.0, b) * np.copysign(1.0, c)
    ma, ea = frexp(a)
    mb, eb = frexp(b)
    mc, ec = frexp(c)
    return np.ldexp((ma * mb) / mc, ea + eb - ec)


def _product_quotient_c_code(inputs, outputs):
    a, b, c = inputs
    (z,) = outputs
    return f"""{{
            if (({a} == 0.0 || {b} == 0.0) && {c} != 0.0) {{
                {z} = copysign(0.0, (double){a})
                    * copysign(1.0, (double){b}) * copysign(1.0, (double){c});
            }} else {{
                int ea, eb, ec;
                double ma = frexp((double){a}, &ea);
                double mb = frexp((double){b}, &eb);
                double mc = frexp((double){c}, &ec);
                {z} = ldexp((ma * mb) / mc, ea + eb - ec);
            }}
        }}"""


class _ProductQuotient(ScalarOp):
    """Compute a * b / c with only the final result rounded into the exponent range."""

    nin = 3

    impl = staticmethod(_product_quotient_impl)

    def pullback(self, inputs, outputs, grads):
        a, b, c = inputs
        (z,) = outputs
        (gz,) = grads
        return self(gz, b, c), self(gz, a, c), -ps.as_scalar(self(gz, z, c))

    def c_headers(self, **kwargs):
        return ["math.h"]

    def c_code(self, node, name, inputs, outputs, sub):
        return _product_quotient_c_code(inputs, outputs)

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 1)


@register_funcify_and_cache_key(_ProductQuotient)
def _numba_product_quotient(op, node, **kwargs):
    return (
        numba_njit(_product_quotient_impl, fastmath=False),
        "pymc_generator.product_quotient.v1",
    )


_product_quotient_scalar = _ProductQuotient(upgrade_to_float64, name="product_quotient")


def _logistic_subnormal_impl(x, lam, reference):
    # The cubic correction cannot cross a nonzero binary64-input midpoint gap
    # at this scale, but breaks exact half-ULP ties toward zero. Two uint64 limbs
    # compare the original binary products without rounding the half-argument.
    sign = np.copysign(1.0, x) * np.copysign(1.0, lam)
    if x == 0.0 or lam == 0.0:
        return np.copysign(0.0, sign)
    mx, ex = frexp(abs(x))
    ml, el = frexp(abs(lam))
    mr, er = frexp(reference)
    units = np.ldexp((mx * ml) / mr, ex + el - er + 1073)
    if units < 0.25:
        return np.copysign(0.0, sign)
    a = np.uint64(np.ldexp(mx, 53))
    b = np.uint64(np.ldexp(ml, 53))
    r = np.uint64(np.ldexp(mr, 53))
    mask = np.uint64(0xFFFFFFFF)
    product = (a & mask) * (b & mask)
    carry = (product >> 32) + (a >> 32) * (b & mask) + (a & mask) * (b >> 32)
    left_hi = (a >> 32) * (b >> 32) + (carry >> 32)
    left_lo = (product & mask) | ((carry & mask) << 32)
    shift = ex + el - er + 1021
    if shift > 0:
        left_hi = (left_hi << shift) | (left_lo >> (64 - shift))
        left_lo = left_lo << shift
    # The preliminary estimate has <2 minimum-subnormal units of error.
    rounded = np.uint64(max(0.0, np.floor(units) - 2.0))
    while True:
        midpoint = np.uint64(2) * rounded + np.uint64(1)
        product = (r & mask) * (midpoint & mask)
        carry = (product >> 32) + (r >> 32) * (midpoint & mask) + (r & mask) * (midpoint >> 32)
        right_hi = (r >> 32) * (midpoint >> 32) + (carry >> 32)
        right_lo = (product & mask) | ((carry & mask) << 32)
        if shift < 0:
            distance = -shift
            right_hi = (right_hi << distance) | (right_lo >> (64 - distance))
            right_lo = right_lo << distance
        if left_hi < right_hi or (left_hi == right_hi and left_lo <= right_lo):
            break
        rounded += np.uint64(1)
    return np.copysign(np.ldexp(float(rounded), -1074), sign)


_LOGISTIC_SUBNORMAL_C_SUPPORT = """
#include <float.h>
#include <stdint.h>
static void pymc_generator_logistic_mantissa_product(
    uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo
) {
    const uint64_t mask = UINT64_C(0xffffffff);
    uint64_t product = (a & mask) * (b & mask);
    uint64_t carry = (product >> 32)
        + (a >> 32) * (b & mask) + (a & mask) * (b >> 32);
    *hi = (a >> 32) * (b >> 32) + (carry >> 32);
    *lo = (product & mask) | ((carry & mask) << 32);
}
static double pymc_generator_logistic_subnormal(double x, double lam, double reference) {
    double sign = copysign(1.0, x) * copysign(1.0, lam);
    if (x == 0.0 || lam == 0.0) return copysign(0.0, sign);
    int ex, el, er;
    double mx = frexp(fabs(x), &ex);
    double ml = frexp(fabs(lam), &el);
    double mr = frexp(reference, &er);
    double units = ldexp((mx * ml) / mr, ex + el - er + 1073);
    if (units < 0.25) return copysign(0.0, sign);
    uint64_t a = (uint64_t)ldexp(mx, 53);
    uint64_t b = (uint64_t)ldexp(ml, 53);
    uint64_t r = (uint64_t)ldexp(mr, 53);
    uint64_t left_hi, left_lo;
    pymc_generator_logistic_mantissa_product(a, b, &left_hi, &left_lo);
    int shift = ex + el - er + 1021;
    if (shift > 0) {
        left_hi = (left_hi << shift) | (left_lo >> (64 - shift));
        left_lo <<= shift;
    }
    uint64_t rounded = (uint64_t)fmax(0.0, floor(units) - 2.0);
    for (;;) {
        uint64_t midpoint = UINT64_C(2) * rounded + UINT64_C(1);
        uint64_t right_hi, right_lo;
        pymc_generator_logistic_mantissa_product(r, midpoint, &right_hi, &right_lo);
        if (shift < 0) {
            int distance = -shift;
            right_hi = (right_hi << distance) | (right_lo >> (64 - distance));
            right_lo <<= distance;
        }
        if (left_hi < right_hi || (left_hi == right_hi && left_lo <= right_lo)) break;
        ++rounded;
    }
    return copysign(ldexp((double)rounded, -1074), sign);
}
"""


def _logistic_relative_impl(x, lam, reference):
    argument = _product_quotient_impl(x, lam, reference)
    positive = argument >= 0.0
    magnitude = argument if positive else -argument
    if magnitude <= 2.0 * _FLOAT64_TINY:
        return _logistic_subnormal_impl(x, lam, reference)
    exponential = np.expm1(-magnitude)
    response = -exponential / (2.0 + exponential)
    return response if positive else -response


class _LogisticRelative(ScalarOp):
    """The bounded forward law with complete weighted exponential-tail pullbacks."""

    nin = 3
    impl = staticmethod(_logistic_relative_impl)

    def pullback(self, inputs, outputs, grads):
        x, lam, reference = inputs
        (gz,) = grads
        magnitude = ps.as_scalar(ps.abs(_product_quotient_scalar(x, lam, reference)))
        # f'(z) = 2 exp(-|z|) / (1 + exp(-|z|))². The unit tail may
        # underflow while the incoming weight and relative factors rescue it.
        log_tail = ps.as_scalar(ps.log1p(ps.exp(-magnitude)))
        log_weight = ps.as_scalar(ps.log(ps.abs(gz))) + np.log(2.0) - magnitude - 2.0 * log_tail
        log_x = ps.as_scalar(ps.log(ps.abs(x)))
        log_lam = ps.as_scalar(ps.log(ps.abs(lam)))
        log_reference = ps.as_scalar(ps.log(reference))
        signed_weight = ps.as_scalar(ps.sign(gz))
        signed_x = ps.as_scalar(ps.sign(x))
        signed_lam = ps.as_scalar(ps.sign(lam))
        return (
            signed_weight * signed_lam * ps.exp(log_weight + log_lam - log_reference),
            signed_weight * signed_x * ps.exp(log_weight + log_x - log_reference),
            -signed_weight
            * signed_x
            * signed_lam
            * ps.exp(log_weight + log_x + log_lam - 2.0 * log_reference),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h", "stdint.h"]

    def c_support_code(self, **kwargs):
        return _LOGISTIC_SUBNORMAL_C_SUPPORT

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        argument = f"{name}_argument"
        product_quotient = _product_quotient_c_code(inputs, [argument])
        return f"""{{
            double {argument};
            {product_quotient}
            double magnitude = {argument} >= 0.0 ? {argument} : -{argument};
            if (magnitude <= 2.0 * DBL_MIN) {{
                {z} = pymc_generator_logistic_subnormal({inputs[0]}, {inputs[1]}, {inputs[2]});
            }} else {{
                double exponential = expm1(-magnitude);
                double response = -exponential / (2.0 + exponential);
                {z} = {argument} >= 0.0 ? response : -response;
            }}
        }}"""

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 3)


@register_funcify_and_cache_key(_LogisticRelative)
def _numba_logistic_relative(op, node, **kwargs):
    product_quotient = numba_njit(_product_quotient_impl, fastmath=False)
    subnormal = numba_njit(_logistic_subnormal_impl, fastmath=False)

    def relative(x, lam, reference):
        argument = product_quotient(x, lam, reference)
        positive = argument >= 0.0
        magnitude = argument if positive else -argument
        if magnitude <= 2.0 * _FLOAT64_TINY:
            return subnormal(x, lam, reference)
        exponential = np.expm1(-magnitude)
        response = -exponential / (2.0 + exponential)
        return response if positive else -response

    return numba_njit(relative, fastmath=False), "pymc_generator.logistic_relative.v2"


_logistic_relative = Elemwise(_LogisticRelative(upgrade_to_float64, name="logistic_relative"))


def _quotient_product_impl(x, reference, shape):
    ratio = np.divide(x, reference)
    if x == 0.0 or (_FLOAT64_TINY <= abs(ratio) <= _FLOAT64_MAX):
        return np.divide(ratio, shape)
    mx, ex = frexp(x)
    mr, er = frexp(reference)
    ms, es = frexp(shape)
    return np.ldexp(mx / (mr * ms), ex - er - es)


_QUOTIENT_PRODUCT_C_SUPPORT = """
#include <float.h>
static double pymc_generator_quotient_product(double x, double reference, double shape) {
    double ratio = x / reference;
    if (x == 0.0 || (fabs(ratio) >= DBL_MIN && fabs(ratio) <= DBL_MAX)) {
        return ratio / shape;
    }
    int ex, er, es;
    double mx = frexp(x, &ex);
    double mr = frexp(reference, &er);
    double ms = frexp(shape, &es);
    return ldexp(mx / (mr * ms), ex - er - es);
}
"""


class _QuotientProduct(ScalarOp):
    """Retain x / reference / shape until the complete quotient is representable."""

    nin = 3
    impl = staticmethod(_quotient_product_impl)

    def pullback(self, inputs, outputs, grads):
        x, reference, shape = inputs
        (gz,) = grads
        log_weight = ps.as_scalar(ps.log(ps.abs(gz)))
        log_input_weight = log_weight + ps.as_scalar(ps.log(ps.abs(x)))
        log_reference = ps.as_scalar(ps.log(reference))
        log_shape = ps.as_scalar(ps.log(shape))
        signed_weight = ps.as_scalar(ps.sign(gz))
        signed_input_weight = -signed_weight * ps.as_scalar(ps.sign(x))
        return (
            signed_weight * ps.exp(log_weight - log_reference - log_shape),
            signed_input_weight * ps.exp(log_input_weight - 2.0 * log_reference - log_shape),
            signed_input_weight * ps.exp(log_input_weight - log_reference - 2.0 * log_shape),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h"]

    def c_support_code(self, **kwargs):
        return _QUOTIENT_PRODUCT_C_SUPPORT

    def c_code(self, node, name, inputs, outputs, sub):
        x, reference, shape = inputs
        (z,) = outputs
        return f"{z} = pymc_generator_quotient_product({x}, {reference}, {shape});"

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 1)


@register_funcify_and_cache_key(_QuotientProduct)
def _numba_quotient_product(op, node, **kwargs):
    return (
        numba_njit(_quotient_product_impl, fastmath=False),
        "pymc_generator.quotient_product.v1",
    )


_quotient_product_scalar = _QuotientProduct(upgrade_to_float64, name="quotient_product")


def _michaelis_menten_denominator_parts(x, reference, kappa):
    # Keep the library's rounded lambda when it is representable, including
    # subnormals; only an out-of-range product needs its extended exponent.
    lam = reference * kappa
    if 0.0 < lam <= _FLOAT64_MAX:
        ml, el = frexp(lam)
    else:
        mr, er = frexp(reference)
        mk, ek = frexp(kappa)
        ml, el = mr * mk, er + ek
    mx, ex = frexp(x)
    exponent = max(el, ex) if x != 0.0 else el
    denominator = np.ldexp(ml, el - exponent) + np.ldexp(mx, ex - exponent)
    return denominator, exponent


_MM_DENOMINATOR_C_SUPPORT = """
#include <float.h>
static double pymc_generator_mm_denominator(
    double x, double reference, double kappa, int *exponent
) {
    double lam = reference * kappa;
    double ml;
    int el, ex;
    if (lam > 0.0 && lam <= DBL_MAX) {
        ml = frexp(lam, &el);
    } else {
        int er, ek;
        double mr = frexp(reference, &er);
        double mk = frexp(kappa, &ek);
        ml = mr * mk;
        el = er + ek;
    }
    double mx = frexp(x, &ex);
    *exponent = (x != 0.0 && ex > el) ? ex : el;
    return ldexp(ml, el - *exponent) + ldexp(mx, ex - *exponent);
}
"""


def _michaelis_menten_relative_impl(x, kappa, reference):
    lam = reference * kappa
    denominator = lam + x
    if lam > 0.0 and np.isfinite(denominator):
        return np.divide(x, denominator)
    mantissa, exponent = _michaelis_menten_denominator_parts(x, reference, kappa)
    mx, ex = frexp(x)
    return np.ldexp(np.divide(mx, mantissa), ex - exponent)


def _mm_log_denominator_impl(x, reference, kappa):
    lam = reference * kappa
    denominator = lam + x
    if lam > 0.0 and np.isfinite(denominator):
        return np.log(abs(denominator))
    mantissa, exponent = _michaelis_menten_denominator_parts(x, reference, kappa)
    return np.log(abs(mantissa)) + exponent * np.log(2.0)


class _MMLogDenominator(ScalarOp):
    """Log of the library denominator, including an extended-range lambda + x."""

    nin = 3
    impl = staticmethod(_mm_log_denominator_impl)

    def pullback(self, inputs, outputs, grads):
        x, reference, kappa = inputs
        (log_denominator,) = outputs
        (gz,) = grads
        log_weight = ps.as_scalar(ps.log(ps.abs(gz))) - log_denominator
        sign = ps.as_scalar(ps.sign(gz)) * ps.as_scalar(ps.sign(x + reference * kappa))
        return (
            sign * ps.exp(log_weight),
            sign * ps.exp(log_weight + ps.log(kappa)),
            sign * ps.exp(log_weight + ps.log(reference)),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h"]

    def c_support_code(self, **kwargs):
        return _MM_DENOMINATOR_C_SUPPORT

    def c_code(self, node, name, inputs, outputs, sub):
        x, reference, kappa = inputs
        (z,) = outputs
        return f"""{{
            double lam = (double){reference} * (double){kappa};
            double denominator = lam + (double){x};
            if (lam > 0.0 && isfinite(denominator)) {{
                {z} = log(fabs(denominator));
            }} else {{
                int exponent;
                double mantissa = pymc_generator_mm_denominator(
                    {x}, {reference}, {kappa}, &exponent);
                {z} = log(fabs(mantissa)) + exponent * log(2.0);
            }}
        }}"""

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 1)


@register_funcify_and_cache_key(_MMLogDenominator)
def _numba_mm_log_denominator(op, node, **kwargs):
    denominator_parts = numba_njit(_michaelis_menten_denominator_parts, fastmath=False)

    def log_denominator(x, reference, kappa):
        lam = reference * kappa
        denominator = lam + x
        if lam > 0.0 and np.isfinite(denominator):
            return np.log(abs(denominator))
        mantissa, exponent = denominator_parts(x, reference, kappa)
        return np.log(abs(mantissa)) + exponent * np.log(2.0)

    return numba_njit(log_denominator, fastmath=False), "pymc_generator.mm_log_denominator.v1"


_mm_log_denominator_scalar = _MMLogDenominator(upgrade_to_float64, name="mm_log_denominator")


class _MichaelisMentenRelative(ScalarOp):
    """Unit library curve with complete relative-operand, weighted pullbacks."""

    nin = 3
    impl = staticmethod(_michaelis_menten_relative_impl)

    def pullback(self, inputs, outputs, grads):
        x, kappa, reference = inputs
        (gz,) = grads
        log_reference, log_kappa = ps.as_scalar(ps.log(reference)), ps.as_scalar(ps.log(kappa))
        lam = reference * kappa
        log_lambda = ps.switch(
            ps.as_scalar(ps.gt(lam, 0.0)) & ps.as_scalar(ps.le(lam, np.finfo("float64").max)),
            ps.log(lam),
            log_reference + log_kappa,
        )
        log_weight = ps.as_scalar(ps.log(ps.abs(gz))) - 2.0 * ps.as_scalar(
            _mm_log_denominator_scalar(x, reference, kappa)
        )
        signed_weight = ps.as_scalar(ps.sign(gz))
        signed_input_weight = -signed_weight * ps.as_scalar(ps.sign(x))
        log_input_weight = log_weight + ps.as_scalar(ps.log(ps.abs(x)))
        return (
            signed_weight * ps.exp(log_weight + log_lambda),
            signed_input_weight * ps.exp(log_input_weight + log_reference),
            signed_input_weight * ps.exp(log_input_weight + log_kappa),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h"]

    def c_support_code(self, **kwargs):
        return _MM_DENOMINATOR_C_SUPPORT

    def c_code(self, node, name, inputs, outputs, sub):
        x, kappa, reference = inputs
        (z,) = outputs
        return f"""{{
            double lam = (double){reference} * (double){kappa};
            double denominator = lam + (double){x};
            if (lam > 0.0 && isfinite(denominator)) {{
                {z} = (double){x} / denominator;
            }} else {{
                int exponent, ex;
                double mantissa = pymc_generator_mm_denominator(
                    {x}, {reference}, {kappa}, &exponent);
                double mx = frexp((double){x}, &ex);
                {z} = ldexp(mx / mantissa, ex - exponent);
            }}
        }}"""

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 2)


@register_funcify_and_cache_key(_MichaelisMentenRelative)
def _numba_michaelis_menten_relative(op, node, **kwargs):
    denominator_parts = numba_njit(_michaelis_menten_denominator_parts, fastmath=False)

    def relative(x, kappa, reference):
        lam = reference * kappa
        denominator = lam + x
        if lam > 0.0 and np.isfinite(denominator):
            return np.divide(x, denominator)
        mantissa, exponent = denominator_parts(x, reference, kappa)
        mx, ex = frexp(x)
        return np.ldexp(np.divide(mx, mantissa), ex - exponent)

    return numba_njit(relative, fastmath=False), "pymc_generator.michaelis_menten_relative.v2"


_michaelis_menten_relative = Elemwise(
    _MichaelisMentenRelative(upgrade_to_float64, name="michaelis_menten_relative")
)


def _tanh_relative_impl(x, reference, c):
    return np.tanh(_quotient_product_impl(x, reference, c))


class _TanhRelative(ScalarOp):
    """Preserve the unit tanh curve, evaluating its analytic pullback in log space."""

    nin = 3

    impl = staticmethod(_tanh_relative_impl)

    def pullback(self, inputs, outputs, grads):
        x, reference, c = inputs
        (gz,) = grads
        magnitude = ps.as_scalar(ps.abs(_quotient_product_scalar(x, reference, c)))
        # sech²(t) = 4 exp(-2 |t|) / (1 + exp(-2 |t|))². Combining
        # its logarithm with the scale factors before exponentiation avoids
        # both 0 / c² at saturation and premature underflow in a rescuable tail.
        log_tail = ps.as_scalar(ps.log1p(ps.exp(-2.0 * magnitude)))
        log_sech_squared = np.log(4.0) - 2.0 * magnitude - 2.0 * log_tail
        log_weight = ps.as_scalar(ps.log(ps.abs(gz))) + log_sech_squared
        log_reference, log_c = ps.as_scalar(ps.log(reference)), ps.as_scalar(ps.log(c))
        signed_weight = ps.as_scalar(ps.sign(gz))
        signed_input_weight = -signed_weight * ps.as_scalar(ps.sign(x))
        log_input_weight = log_weight + ps.as_scalar(ps.log(ps.abs(x)))
        return (
            signed_weight * ps.exp(log_weight - log_reference - log_c),
            signed_input_weight * ps.exp(log_input_weight - 2.0 * log_reference - log_c),
            signed_input_weight * ps.exp(log_input_weight - log_reference - 2.0 * log_c),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h"]

    def c_support_code(self, **kwargs):
        return _QUOTIENT_PRODUCT_C_SUPPORT

    def c_code(self, node, name, inputs, outputs, sub):
        x, reference, c = inputs
        (z,) = outputs
        return f"{z} = tanh(pymc_generator_quotient_product({x}, {reference}, {c}));"

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 2)


@register_funcify_and_cache_key(_TanhRelative)
def _numba_tanh_relative(op, node, **kwargs):
    quotient_product = numba_njit(_quotient_product_impl, fastmath=False)

    def relative(x, reference, c):
        return np.tanh(quotient_product(x, reference, c))

    return numba_njit(relative, fastmath=False), "pymc_generator.tanh_relative.v2"


_tanh_relative = Elemwise(_TanhRelative(upgrade_to_float64, name="tanh_relative"))


def _root_relative_impl(x, reference, alpha):
    if x <= 0.0:
        return 0.0
    ratio = np.divide(x, reference)
    if _FLOAT64_TINY <= ratio <= _FLOAT64_MAX:
        return ratio**alpha
    return np.exp(alpha * (np.log(x) - np.log(reference)))


class _RootRelative(ScalarOp):
    """The clipped library power with no early range loss in a positive ratio."""

    nin = 3
    impl = staticmethod(_root_relative_impl)

    def pullback(self, inputs, outputs, grads):
        x, reference, alpha = inputs
        (gz,) = grads
        positive = ps.gt(x, 0.0)
        safe_x = ps.switch(positive, x, 1.0)
        log_x, log_reference = ps.as_scalar(ps.log(safe_x)), ps.as_scalar(ps.log(reference))
        ratio = safe_x / reference
        direct = ps.as_scalar(ps.ge(ratio, np.finfo("float64").tiny)) & ps.as_scalar(
            ps.le(ratio, np.finfo("float64").max)
        )
        log_ratio = ps.switch(direct, ps.log(ps.switch(direct, ratio, 1.0)), log_x - log_reference)
        log_weight = ps.as_scalar(ps.log(ps.abs(gz))) + alpha * log_ratio
        signed_weight = ps.as_scalar(ps.sign(gz))
        log_alpha = ps.as_scalar(ps.log(alpha))
        return (
            ps.switch(positive, signed_weight * ps.exp(log_weight + log_alpha - log_x), 0.0),
            ps.switch(
                positive, -signed_weight * ps.exp(log_weight + log_alpha - log_reference), 0.0
            ),
            ps.switch(
                positive,
                signed_weight * ps.sign(log_ratio) * ps.exp(log_weight + ps.log(ps.abs(log_ratio))),
                0.0,
            ),
        )

    def c_headers(self, **kwargs):
        return ["math.h", "float.h"]

    def c_support_code(self, **kwargs):
        return "#include <float.h>"

    def c_code(self, node, name, inputs, outputs, sub):
        x, reference, alpha = inputs
        (z,) = outputs
        return f"""{{
            if ({x} <= 0.0) {{
                {z} = 0.0;
            }} else {{
                double ratio = (double){x} / (double){reference};
                {z} = (ratio >= DBL_MIN && ratio <= DBL_MAX)
                    ? pow(ratio, (double){alpha})
                    : exp((double){alpha} * (log((double){x}) - log((double){reference})));
            }}
        }}"""

    def c_code_cache_version(self):
        return (*super().c_code_cache_version(), 1)


@register_funcify_and_cache_key(_RootRelative)
def _numba_root_relative(op, node, **kwargs):
    return numba_njit(_root_relative_impl, fastmath=False), "pymc_generator.root_relative.v1"


_root_relative = Elemwise(_RootRelative(upgrade_to_float64, name="root_relative"))


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
    the structural ``beta`` gate is this treatment's only amplitude.
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


def _hill_relative_parts(x, reference, slope, kappa):
    positive = pt.gt(x, 0.0)
    safe_x = pt.switch(positive, x, 1.0)
    ratio = safe_x / (kappa * reference)
    # Preserve the rounded ratio at the knee and the separate-log far tail.
    direct = pt.ge(ratio, _FLOAT64_TINY) & pt.le(ratio, _FLOAT64_MAX)
    log_ratio = pt.switch(
        direct,
        pt.log(pt.switch(direct, ratio, 1.0)),
        pt.log(safe_x) - pt.log(reference) - pt.log(kappa),
    )
    return positive, safe_x, log_ratio, slope * log_ratio


def _hill_relative_pullback(inputs, outputs, grads):
    x, reference, slope, kappa = inputs
    (gz,) = grads
    positive, safe_x, log_ratio, log_odds = _hill_relative_parts(*inputs)
    magnitude = pt.abs(log_odds)
    # sigmoid'(z) = exp(-|z|) / (1 + exp(-|z|))². Do not round that
    # tail before the cotangent and each relative/shape factor can rescue it.
    log_weight = pt.log(pt.abs(gz)) - magnitude - 2.0 * pt.log1p(pt.exp(-magnitude))
    signed_weight = pt.sign(gz)
    log_slope = pt.log(slope)
    derivatives = (
        signed_weight * pt.exp(log_weight + log_slope - pt.log(safe_x)),
        -signed_weight * pt.exp(log_weight + log_slope - pt.log(reference)),
        signed_weight * pt.sign(log_ratio) * pt.exp(log_weight + pt.log(pt.abs(log_ratio))),
        -signed_weight * pt.exp(log_weight + log_slope - pt.log(kappa)),
    )
    result = []
    for value, derivative in zip(inputs, derivatives, strict=True):
        derivative = pt.switch(positive, derivative, 0.0)
        # OpFromGraph does not perform Elemwise's broadcast pullback reduction.
        extra_axes = tuple(range(derivative.ndim - value.ndim))
        if extra_axes:
            derivative = derivative.sum(axis=extra_axes)
        broadcast_axes = tuple(
            axis
            for axis, size in enumerate(value.type.shape)
            if size == 1 and derivative.type.shape[axis] != 1
        )
        if broadcast_axes:
            derivative = derivative.sum(axis=broadcast_axes, keepdims=True)
        result.append(derivative)
    return result


@cache
def _build_hill_relative(input_types):
    x, reference, slope, kappa = (value_type() for value_type in input_types)
    positive, _, _, log_odds = _hill_relative_parts(x, reference, slope, kappa)
    zero = pt.as_tensor_variable(0.0)
    zero_input, odds_input = zero.type(), log_odds.type()
    maximum = pt.maximum(zero_input, odds_input)
    maximum = pt.switch(pt.isinf(maximum), 0.0, maximum)
    # Preserve the library's non-inlined, rounded log-sum arithmetic. Spell
    # out its stabilization so the declared backend floor cannot omit it.
    log_denominator = OpFromGraph(
        [zero_input, odds_input],
        [pt.log(pt.exp(zero_input - maximum) + pt.exp(odds_input - maximum)) + maximum],
        inline=False,
        name="hill_logaddexp",
    )(zero, log_odds)
    response = pt.exp(log_odds - log_denominator)
    return OpFromGraph(
        [x, reference, slope, kappa],
        [pt.switch(positive, response, 0.0)],
        inline=True,
        pullback=_hill_relative_pullback,
        name="hill_relative",
    )


def stable_hill_kappa_relative(x, reference_level, *, slope, kappa_mult) -> TensorVariable:
    """The same log-space Hill curve with exact zero and complete weighted tails.

    The inlined forward graph retains its rounded knee and far-tail arithmetic.
    An analytic pullback combines the incoming cotangent, tail and each relative
    factor before exponentiation, rather than losing a rescuable derivative.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    inputs = tuple(
        pt.as_tensor_variable(value) for value in (x, safe_reference_level, slope, kappa_mult)
    )
    result: TensorVariable = _build_hill_relative(tuple(value.type for value in inputs))(*inputs)
    return result


def stable_logistic_kappa_relative(x, reference_level, *, lam) -> TensorVariable:
    """The same unit-asymptote logistic curve without cancellation or early range loss.

    Mantissa/exponent arithmetic forms lambda * x / reference before rounding:
    neither lambda / 2 underflow nor x / reference overflow can erase a finite
    response. The bounded exponential difference retains the current forward
    rounding; its analytic pullback combines weighted tails before exponentiation.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return pt.as_tensor_variable(_logistic_relative(x, lam, safe_reference_level))


def stable_michaelis_menten_kappa_relative(x, reference_level, *, kappa_mult) -> TensorVariable:
    """The same unit-asymptote Michaelis-Menten curve with extended-range arithmetic.

    An extended-range denominator and complete weighted pullbacks preserve finite
    results when λ + x or an intermediate absolute-λ derivative is out of range.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    # Input order (x, kappa, reference) is part of the opt-in stream layout: PyMC's
    # right-first walk reaches reference and kappa before x through this node,
    # whereas the default library graph reaches x first.
    return pt.as_tensor_variable(_michaelis_menten_relative(x, kappa_mult, safe_reference_level))


def stable_tanh_kappa_relative(x, reference_level, *, c) -> TensorVariable:
    """The same unit-asymptote tanh(x / (reference_level · c)) with a complete quotient.

    A complete quotient retains inputs whose intermediate x / reference ratio
    is out of range before the shape rescales it.
    Its analytic pullback combines the exponential tail and scale factors in
    log space, so saturation gives zero rather than an indeterminate 0 / c².
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return pt.as_tensor_variable(_tanh_relative(x, safe_reference_level, c))


def stable_root_kappa_relative(x, reference_level, *, alpha) -> TensorVariable:
    """The same root curve (x / reference_level)^alpha with log-space extended range.

    Ordinary ratios use a scalar power. Positive out-of-range or subnormal
    ratios stay in log space through the power and its complete weighted
    pullback; clipped nonpositive inputs retain an exact zero response.
    """
    safe_reference_level = pt.maximum(reference_level, 1e-8)
    return pt.as_tensor_variable(_root_relative(x, safe_reference_level, alpha))


#: Default name -> wrapper table, with signature ``f(x, reference_level, **shape_params)``.
#: These are the library graphs whose rounding and random-stream order seeded
#: corpora reproduce.
SATURATION_FAMILIES: dict[str, Callable[..., TensorVariable]] = {
    "hill": hill_kappa_relative,
    "logistic": logistic_kappa_relative,
    "michaelis_menten": michaelis_menten_kappa_relative,
    "tanh": tanh_kappa_relative,
    "root": root_kappa_relative,
}


#: Opt-in replacements, selected by ``params["mechanism_priors_enabled"]`` in
#: :func:`pymc_generator.symbolic_graph._saturate_family`. They compute the same
#: unit curves with complete-range arithmetic and analytic pullbacks, so they have
#: their own graph, rounding and random-variable traversal.
STABLE_SATURATION_FAMILIES: dict[str, Callable[..., TensorVariable]] = {
    "hill": stable_hill_kappa_relative,
    "logistic": stable_logistic_kappa_relative,
    "michaelis_menten": stable_michaelis_menten_kappa_relative,
    "tanh": stable_tanh_kappa_relative,
    "root": stable_root_kappa_relative,
}


# --------------------------------------------------------------------------
# Carryover — pymc-marketing transformers over a single time column (axis 0)
# --------------------------------------------------------------------------


def apply_geometric_carryover(
    x: TensorVariable, alpha: TensorVariable | float, l_max: int
) -> TensorVariable:
    """Normalized geometric carryover of a ``(n_time_steps, 1)`` column over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.geometric_adstock`` (ConvMode
    ``After``, ``normalize=True``). ``alpha`` may be a float or a symbolic scalar.
    Normalization divides the kernel weights by their sum; it does not map
    data to [0, 1]. The result stays in the input's units.
    """
    out: XTensorVariable = _pmm.geometric_adstock(
        _as_time(x[:, 0]), alpha=alpha, l_max=int(l_max), dim="time", normalize=True
    )
    values: TensorVariable = out.values[:, None]
    return values


def apply_weibull_pdf_carryover(
    x: TensorVariable, lam: TensorVariable | float, k: TensorVariable | float, l_max: int
) -> TensorVariable:
    """Normalized min-max-rescaled Weibull-density carryover over the time axis.

    Delegates to ``pymc_marketing.mmm.transformers.weibull_adstock`` with
    ``type="PDF"``, ``normalize=True``. The library min-max rescales the sampled
    density before sum-normalizing, so this kernel is not a Weibull PDF:
    ``min(w) == 0`` exactly and one or more lags are always annihilated. Under
    the default prior (``lam ~ U(2, 8)``, ``k ~ U(1.5, 4)``, ``l_max = 8``),
    45.0% of treatments have zero current-week weight and 38.5% peak at lag >= 5
    (measured over 200k draws). For a zero current-week weight,
    ``contributions[t]`` is independent of ``treatments[t]``;
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
    # This guard MUST track signal_diagnostics._carryover_weights. The oracle passes
    # symbolic value variables; a backend-dependent library-output guard would make
    # FAST_COMPILE generation and the FAST_RUN oracle disagree whether a treatment responds.
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
