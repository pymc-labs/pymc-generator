"""Analytic saturation landmarks and fixed walk-amplitude calibration.

A non-unit reference level catches dropped input scaling even when additive
decomposition identities still hold. Expected values are closed forms, not
snapshots of implementation output.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

from pymc_generator import mechanisms
from pymc_generator.random_walk import _kernel_width, _walk_basis, symbolic_random_walk
from pymc_generator.sampler import SATURATION_FAMILY_KEYS
from pymc_generator.symbolic_graph import _saturate_col

#: A deliberately non-unit anchor. Every landmark below is a multiple of
#: the reference level, so an anchor of 1.0 would hide dropped input scaling.
REFERENCE_LEVEL = 3.0

#: One mid-range shape parameterization per family, from
#: :data:`~pymc_generator.mechanisms.SATURATION_PRIOR_RANGES`.
FAMILY_SHAPES: dict[str, dict[str, float]] = {
    "hill": {"slope": 2.0, "kappa_mult": 1.2},
    "logistic": {"lam": 1.7},
    "michaelis_menten": {"kappa_mult": 0.9},
    "tanh": {"c": 0.8},
    "root": {"alpha": 0.6},
}


def _evaluate(expression) -> np.ndarray:
    """Evaluate a symbolic saturation column as a float64 array."""
    return np.asarray(expression.eval(), dtype=np.float64)


def _family(name: str, x, reference_level=REFERENCE_LEVEL, **shape) -> np.ndarray:
    """Evaluate one κ-relative family through the name dispatch table."""
    return _evaluate(mechanisms.SATURATION_FAMILIES[name](x, reference_level, **shape))


@pytest.mark.parametrize(
    ("name", "shape", "landmark_x", "landmark_y"),
    [
        # hill's κ IS its half point, for ANY slope: f(κ) = 0.5 exactly.
        ("hill", {"slope": 0.5, "kappa_mult": 1.0}, REFERENCE_LEVEL, 0.5),
        ("hill", {"slope": 3.0, "kappa_mult": 1.0}, REFERENCE_LEVEL, 0.5),
        # logistic_saturation(u, lam) = tanh(lam·u/2) on u = x/reference_level;
        # half point sits at u = ln(3)/lam.
        ("logistic", {"lam": 0.5}, float(np.log(3.0) / 0.5) * REFERENCE_LEVEL, 0.5),
        ("logistic", {"lam": 3.0}, float(np.log(3.0) / 3.0) * REFERENCE_LEVEL, 0.5),
        # Michaelis-Menten reaches 0.5 at kappa_mult times the reference level.
        ("michaelis_menten", {"kappa_mult": 1.0}, REFERENCE_LEVEL, 0.5),
        # tanh reaches 0.5 at atanh(0.5) times c times the reference level.
        ("tanh", {"c": 0.3}, float(np.arctanh(0.5) * 0.3) * REFERENCE_LEVEL, 0.5),
        ("tanh", {"c": 1.5}, float(np.arctanh(0.5) * 1.5) * REFERENCE_LEVEL, 0.5),
        # Root is normalized to 1 at the reference level, not
        # to an asymptote, for any exponent.
        ("root", {"alpha": 0.3}, REFERENCE_LEVEL, 1.0),
        ("root", {"alpha": 0.9}, REFERENCE_LEVEL, 1.0),
    ],
    ids=lambda value: repr(value) if isinstance(value, dict) else str(value),
)
def test_family_reproduces_its_analytic_landmark(name, shape, landmark_x, landmark_y):
    """Each family's documented anchor value is exact, not approximate.

    These are the identities that let ``beta`` be read as "the channel's
    contribution at its κ anchor": a family whose half point drifted with its
    shape parameter would make ``beta`` mean something different per draw.
    """
    value = _family(name, np.array([landmark_x]), **shape)
    np.testing.assert_allclose(value, landmark_y, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(FAMILY_SHAPES))
def test_family_is_strictly_monotone_in_x(name):
    """Monotonicity is what makes a contribution attributable to its spend.

    Asserted STRICTLY: a family that plateaued in float (or that read the wrong
    shape parameter and collapsed to a constant) would still integrate into the
    graph and still satisfy every decomposition identity.
    """
    x = np.linspace(0.0, 4.0 * REFERENCE_LEVEL, 97)
    y = _family(name, x, **FAMILY_SHAPES[name])
    assert (np.diff(y) > 0.0).all(), f"{name} is not strictly increasing in x"


@pytest.mark.parametrize("scale", [1e-3, 0.5, 2.0, 1e3])
@pytest.mark.parametrize("name", sorted(FAMILY_SHAPES))
def test_family_is_scale_free_in_its_anchor(name, scale):
    """Scaling ``x`` and ``reference_level`` together leaves the response unchanged.

    Under reference-relative parameterization,
    the shape parameters are dimensionless, so the same prior ranges are
    meaningful for a channel spending 10 and one spending 10 million. A wrapper
    that forgot to rescale its input would fail here while still passing every
    landmark at the fixed anchor above.
    """
    x = np.linspace(0.05, 4.0 * REFERENCE_LEVEL, 41)
    shape = FAMILY_SHAPES[name]
    base = _family(name, x, **shape)
    rescaled = _family(name, x * scale, reference_level=REFERENCE_LEVEL * scale, **shape)
    np.testing.assert_allclose(rescaled, base, rtol=1e-12, atol=0.0)


#: Channel index used for the dispatch check. NOT 0: a wrapper that ignored the
#: per-channel index would still read the right value at slot 0.
DISPATCH_K = 1

#: Every shape parameter ``_saturate_family`` can read, for two channels. Slot 0
#: deliberately holds a different value from slot ``DISPATCH_K``.
DISPATCH_PARAMS: dict[str, np.ndarray] = {
    "hill_slope": np.array([1.1, FAMILY_SHAPES["hill"]["slope"]]),
    "hill_kappa_mult": np.array([0.7, FAMILY_SHAPES["hill"]["kappa_mult"]]),
    "logistic_lam": np.array([0.6, FAMILY_SHAPES["logistic"]["lam"]]),
    "mm_kappa_mult": np.array([1.4, FAMILY_SHAPES["michaelis_menten"]["kappa_mult"]]),
    "tanh_c": np.array([1.3, FAMILY_SHAPES["tanh"]["c"]]),
    "root_alpha": np.array([0.35, FAMILY_SHAPES["root"]["alpha"]]),
}


def _expected_family_column(name: str, x: np.ndarray) -> np.ndarray:
    """The column ``_saturate_col`` must produce for family ``name``.

    ``linear`` is the one family with no κ-relative wrapper — it is the plain
    anchor-relative ratio — so it is spelled out here rather than looked up.
    """
    if name == "linear":
        return x / REFERENCE_LEVEL
    return _family(name, x, **FAMILY_SHAPES[name])


@pytest.mark.parametrize("dynamic_family", [False, True], ids=["concrete", "switch"])
@pytest.mark.parametrize(
    "family_id", range(len(SATURATION_FAMILY_KEYS)), ids=SATURATION_FAMILY_KEYS
)
def test_saturate_col_routes_every_family_id_to_its_own_wrapper(family_id, dynamic_family):
    """The integer family id and the name table agree, on both dispatch paths.

    ``params["sat_family"]`` is a persisted integer, so the id -> family map is
    a data-format contract: an off-by-one would silently re-label every stored
    world's mechanism. The ``dynamic_family`` path is the riskier one — it
    builds ALL families and picks with a chain of ``pt.switch`` comparisons
    against the id, which is exactly where an index shift hides.
    """
    name = SATURATION_FAMILY_KEYS[family_id]
    x = np.linspace(0.1, 3.0 * REFERENCE_LEVEL, 12)
    params = dict(DISPATCH_PARAMS, sat_family=np.full(2, family_id))

    routed = _evaluate(
        _saturate_col(
            pt.as_tensor_variable(x),
            pt.as_tensor_variable(REFERENCE_LEVEL),
            params,
            DISPATCH_K,
            dynamic_family=dynamic_family,
        )
    )
    np.testing.assert_allclose(routed, _expected_family_column(name, x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("n_time_steps", "width"),
    [
        (2, 1),  # shortest admissible horizon, no smoothing
        (2, 2),  # width clamped to the horizon
        (5, 3),
        (26, 1),
        (26, 13),
        (26, 26),
        (52, 7),
        (104, 13),
        (104, 26),
    ],
)
def test_walk_basis_carries_unit_mean_square_per_week(n_time_steps, width):
    """``sum(B**2) / n_time_steps == 1`` for every horizon and kernel width.

    ``_walk_basis`` divides the centred, smoothed cumsum operator ``A`` by
    ``_centred_walk_scale = sqrt(tr(A Aᵀ) / n_time_steps)``, so this identity is
    that definition read back off the returned matrix. It is what makes ``std``
    the walk's expected per-week amplitude (``E[var(walk)] == std**2``) instead
    of a free scale: normalizing by a path's own realized standard deviation
    would leave the likelihood flat along ``eps -> k·eps``. A drifted
    normalization would rescale every random walk in every generated world
    while breaking no identity.
    """
    basis = _walk_basis(n_time_steps, width)
    assert basis.shape == (n_time_steps, n_time_steps)
    np.testing.assert_allclose((basis**2).sum() / n_time_steps, 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("smoothness", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_time_steps", [4, 52, 104])
def test_symbolic_walk_equals_the_basis_applied_to_its_innovations(n_time_steps, smoothness):
    """``symbolic_random_walk`` is exactly ``std * (B @ eps) + mean``.

    ``_walk_basis`` documents this equivalence, and two things depend on it:
    ``symbolic_random_walk_by_width`` implements the walk as a literal
    ``dot(basis[width_index], eps)`` (one compiled graph for every smoothness),
    and the oracle model marginalizes the outcome-side walks analytically using
    the same ``B``. If the streaming form and the matrix form drifted apart, the
    template path and the oracle would silently model a different walk from the
    one the per-world path draws.
    """
    eps = np.random.default_rng(0).normal(size=n_time_steps)
    mean, std = 2.0, 0.7
    width = _kernel_width(smoothness, n_time_steps, rw_smoothness_max_weeks=26)

    walk = np.asarray(
        symbolic_random_walk(
            n_time_steps,
            mean=mean,
            std=std,
            smoothness=smoothness,
            positive_only=False,
            rw_smoothness_max_weeks=26,
            eps=pt.as_tensor_variable(eps),
        ).eval()
    )

    np.testing.assert_allclose(
        walk, std * (_walk_basis(n_time_steps, width) @ eps) + mean, rtol=0.0, atol=1e-12
    )
