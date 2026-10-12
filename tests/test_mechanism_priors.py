"""Configurable mechanism measures and reference contributions, not graph wiring.

Expected responses below are independent NumPy/Decimal closed forms. Coefficients and
responses are materialized separately: simplifying ``q / f * f`` symbolically
would hide an infinite intermediate coefficient or a cancelled tiny response.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal, localcontext

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc_marketing.mmm import transformers as _pmm
from pytensor.graph.replace import clone_replace
from pytensor.xtensor import as_xtensor

import pymc_generator as pg
import pymc_generator.world_model as world_model
from pymc_generator import mechanisms
from pymc_generator.describe import world_to_dot
from pymc_generator.sampler import (
    CARRYOVER_FAMILY_KEYS,
    CORPUS_STORAGE_MAX,
    SATURATION_FAMILY_KEYS,
    SCMPrior,
)
from pymc_generator.slots import PRIOR_COND_LAYOUT
from pymc_generator.symbolic_graph import _saturate_col, build_symbolic_graph
from pymc_generator.world_model import (
    build_oracle_model,
    build_world_model,
    draw_worlds,
    sample_prior_cond,
    sample_structure,
)
from pymc_generator.world_model_template import (
    build_cell_inputs,
    build_world_model_template,
    compile_template_draw_fn,
)
from pymc_generator.worlds import _assemble_params, edges_with_coeffs

_EPS = float(np.finfo(np.float64).eps)

SHAPE_FIELDS = {
    "hill_slope": ("hill", "slope"),
    "hill_kappa_mult": ("hill", "kappa_mult"),
    "logistic_lam": ("logistic", "lam"),
    "mm_kappa_mult": ("michaelis_menten", "kappa_mult"),
    "tanh_c": ("tanh", "c"),
    "root_alpha": ("root", "alpha"),
}


def _ranges(**overrides):
    ranges = deepcopy(SCMPrior().saturation_prior_ranges)
    for name, bounds in overrides.items():
        family, parameter = SHAPE_FIELDS[name]
        ranges[family][parameter] = bounds
    return ranges


def _one_hot(family):
    return {name: float(name == family) for name in SATURATION_FAMILY_KEYS}


def _cfg(**overrides):
    base = {
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "n_time_steps": 16,
        "n_cells": 2,
        "draws_per_cell": 2,
        "seed": 91,
        "l_max": 4,
        "carryover_burn_in": 0,
        "treatment_cv_floor": 0.0,
        "treatment_hf_sigma_range": (0.0, 0.0),
        "treatment_pulse_prob_range": (0.0, 0.0),
        "covariate_hf_sigma_range": (0.0, 0.0),
        "covariate_pulse_prob_range": (0.0, 0.0),
    }
    base.update(overrides)
    return pg.make_scm_prior(**base)


def _corpus_cfg(**overrides):
    """``_cfg`` with a supported (non-flat) treatment texture for corpus generation."""
    return _cfg(**{"treatment_hf_sigma_range": (0.05, 0.1), **overrides})


def _graph(n_treatments, n_covariates):
    return {
        "g_cy": np.ones(n_treatments, dtype=int),
        "g_dc": np.zeros((1, n_treatments), dtype=int),
        "g_dz": np.zeros((1, n_covariates), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.ones(n_covariates, dtype=int),
        "g_zc": np.zeros((n_covariates, n_treatments), dtype=int),
        "g_cc": np.zeros((n_treatments, n_treatments), dtype=int),
        "g_zz": np.zeros((n_covariates, n_covariates), dtype=int),
    }


def _structure(g, cfg, families, *, carryover=None):
    structural = sample_structure(g, cfg, np.random.default_rng(17))
    structural["sat_family"] = np.asarray(families, dtype=int)
    structural["carryover_family"] = (
        np.zeros(len(families), dtype=int)
        if carryover is None
        else np.asarray(carryover, dtype=int)
    )
    return structural


def _draw(model, outputs, reports, *, seed=29, draws=1):
    names = tuple(dict.fromkeys((*outputs, *reports, *(rv.name for rv in model.free_RVs))))
    return draw_worlds(model, names, seed=seed, draws=draws)


def _params(drawn, index, structural, reports, cfg):
    return _assemble_params(drawn, index, structural, reports, cfg)


def _numpy_response(family, ratio, params, k):
    """Six independent mathematical families on the dimensionless input x/r."""
    u = np.asarray(ratio, dtype=float)
    if family == "linear":
        return u
    if family == "hill":
        result = np.zeros_like(u)
        positive = u > 0.0
        log_odds = params["hill_slope"][k] * (
            np.log(u[positive]) - np.log(params["hill_kappa_mult"][k])
        )
        result[positive] = np.exp(-np.logaddexp(0.0, -log_odds))
        return result
    if family == "logistic":
        return np.tanh(params["logistic_lam"][k] * u / 2.0)
    if family == "michaelis_menten":
        return u / (params["mm_kappa_mult"][k] + u)
    if family == "tanh":
        return np.tanh(u / params["tanh_c"][k])
    if family == "root":
        return u ** params["root_alpha"][k]
    raise AssertionError(f"unexpected family {family!r}")


def _numpy_contributions(treatments, scale, params, g_cy):
    """Independent identity/geometric kernels followed by each response curve."""
    result = np.zeros_like(treatments, dtype=float)
    for k, family_id in enumerate(params["sat_family"]):
        x = np.asarray(treatments[:, k], dtype=float)
        carryover = int(params["carryover_family"][k])
        if carryover == 1:
            weights = params["carryover_alpha"][k] ** np.arange(params["l_max"])
            weights /= weights.sum()
            x = np.convolve(x, weights, mode="full")[: len(x)]
        else:
            assert carryover == 0
        f = _numpy_response(SATURATION_FAMILY_KEYS[int(family_id)], x / scale[k], params, k)
        result[:, k] = g_cy[k] * params["beta"][k] * f
    return result


def _softplus(x):
    return np.logaddexp(0.0, x)


def _independent_anchors(params, g, use_pulse):
    """Parameter-only treatment anchors with parent and pulse terms, in NumPy."""
    z_levels = []
    for m in range(len(params["rw_z_mean"])):
        upstream = sum(g["g_zz"][j, m] * params["gamma_zz"][j, m] * z_levels[j] for j in range(m))
        z_levels.append(params["rw_z_mean"][m] + upstream)
    c_levels = []
    for k in range(len(params["rw_c_mean"])):
        own = _softplus(params["rw_c_mean"][k])
        if use_pulse[k]:
            own += params["pulse_amp"][k] * params["pulse_prob"][k]
        own += sum(g["g_zc"][m, k] * params["v_zc"][m, k] * z for m, z in enumerate(z_levels))
        own += sum(g["g_cc"][j, k] * params["alpha_cc"][j, k] * c_levels[j] for j in range(k))
        c_levels.append(_softplus(own))
    return np.maximum(np.array(c_levels), 1e-8)


def _point(model, values):
    point = {}
    for rv in model.free_RVs:
        raw = np.asarray(values[rv.name])
        value_var = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms[rv]
        point[value_var.name] = (
            raw if transform is None else transform.forward(raw, *rv.owner.inputs).eval()
        )
    return point


def _evaluate_given(model, values, names):
    point = _point(model, values)
    expressions = model.replace_rvs_by_values([model[name] for name in names])
    evaluate = pytensor.function(
        model.value_vars, expressions, on_unused_input="ignore", mode="FAST_COMPILE"
    )
    return dict(zip(names, evaluate(*[point[var.name] for var in model.value_vars])))


# -- Reject invalid supports at the configuration boundary ----------------------


@pytest.mark.parametrize(
    ("kind", "field"),
    (
        ("not_mapping", "saturation_prior_ranges"),
        ("missing_family", "saturation_prior_ranges"),
        ("extra_family", "saturation_prior_ranges"),
        ("not_parameter_mapping", "hill"),
        ("missing_parameter", "hill"),
        ("extra_parameter", "hill"),
        ("wrong_parameter", "logistic"),
    ),
)
def test_saturation_support_requires_complete_known_family_and_parameter_keys(kind, field):
    ranges = _ranges()
    if kind == "not_mapping":
        ranges = None
    elif kind == "missing_family":
        del ranges["root"]
    elif kind == "extra_family":
        ranges["linear"] = {}
    elif kind == "not_parameter_mapping":
        ranges["hill"] = [1.0, 3.0]
    elif kind == "missing_parameter":
        del ranges["hill"]["kappa_mult"]
    elif kind == "extra_parameter":
        ranges["hill"]["offset"] = (1.0, 2.0)
    else:
        ranges["logistic"] = {"rate": (1.0, 2.0)}
    with pytest.raises(ValueError, match=field):
        _cfg(saturation_prior_ranges=ranges)


@pytest.mark.parametrize(
    ("parameter", "bounds"),
    (
        ("hill_slope", None),
        ("hill_slope", 2.0),
        ("hill_slope", (1.0, 2.0, 3.0)),
        ("hill_kappa_mult", ("0.7", 1.5)),
        ("hill_kappa_mult", (0.7, np.bool_(True))),
        ("logistic_lam", (0.0, 1.0)),
        ("logistic_lam", (-1.0, 1.0)),
        ("mm_kappa_mult", (np.nan, 1.0)),
        ("mm_kappa_mult", (1.0, np.inf)),
        ("mm_kappa_mult", (2.0, 1.0)),
        # kappa * the minimum anchor 1e-8 underflows, so a zero input is 0/0.
        ("mm_kappa_mult", (5e-324, 1.0)),
        ("tanh_c", (1.0, 10**400)),
        ("tanh_c", (1.0, 2.0 * CORPUS_STORAGE_MAX)),
        ("root_alpha", (0.5, 1.01)),
        ("root_alpha", (0.0, 0.5)),
        ("root_alpha", (True, 1.0)),
    ),
)
def test_saturation_bounds_are_real_ordered_positive_and_storage_compatible(parameter, bounds):
    family, key = SHAPE_FIELDS[parameter]
    field = f"saturation_prior_ranges[{family!r}][{key!r}]"
    with pytest.raises(ValueError, match=re.escape(field)):
        _cfg(saturation_prior_ranges=_ranges(**{parameter: bounds}))


@pytest.mark.parametrize("field", ("treatment_reference_multiplier", "covariate_reference_scale"))
@pytest.mark.parametrize(
    "value", (0.0, -1.0, True, "1", np.nan, np.inf, 2 * CORPUS_STORAGE_MAX, 10**400)
)
def test_reference_input_scalars_require_positive_real_representable_values(field, value):
    with pytest.raises(ValueError, match=field):
        _cfg(**{field: value})


@pytest.mark.parametrize(
    ("field", "bounds"),
    (
        ("treatment_reference_contribution_range", (-0.1, 1.0)),
        ("treatment_reference_contribution_range", (0.0, 0.0)),
        ("treatment_reference_contribution_range", (2.0, 1.0)),
        ("treatment_reference_contribution_range", (False, 1.0)),
        ("treatment_reference_contribution_range", ("0", 1.0)),
        ("treatment_reference_contribution_range", (np.nan, 1.0)),
        ("treatment_reference_contribution_range", 1.0),
        ("treatment_reference_contribution_range", (0.0, 10**400)),
        ("covariate_reference_contribution_range", (1.0, -1.0)),
        ("covariate_reference_contribution_range", (-np.inf, 1.0)),
        ("covariate_reference_contribution_range", (-1.0, "1")),
        ("covariate_reference_contribution_range", (-1.0, True)),
        ("covariate_reference_contribution_range", (-1.0,)),
        ("covariate_reference_contribution_range", (-1.0, 0.0, 1.0)),
        ("covariate_reference_contribution_range", (-2 * CORPUS_STORAGE_MAX, 0.0)),
    ),
)
def test_reference_target_supports_reject_invalid_types_domains_and_bounds(field, bounds):
    with pytest.raises(ValueError, match=field):
        _cfg(**{field: bounds})


@pytest.mark.parametrize("mode", (None, False, "lognormal", "LOG_UNIFORM"))
def test_mm_prior_rejects_unknown_probability_measures(mode):
    with pytest.raises(ValueError, match="mm_scale_prior"):
        _cfg(mm_scale_prior=mode)


def test_distinct_mm_bounds_that_collapse_in_log_space_are_rejected():
    lo = 1e20
    hi = np.nextafter(lo, np.inf)
    with pytest.raises(ValueError, match="michaelis_menten.*log"):
        _cfg(
            mm_scale_prior="log_uniform",
            saturation_prior_ranges=_ranges(mm_kappa_mult=(lo, hi)),
        )


def test_numpy_mm_bounds_use_float64_logarithms_and_draw_inside_the_support():
    lo = np.float32(1e20)
    hi = np.nextafter(lo, np.float32(np.inf))
    cfg = _cfg(
        mm_scale_prior="log_uniform",
        saturation_prior_ranges=_ranges(mm_kappa_mult=(lo, hi)),
    )
    g = _graph(2, 1)
    structural = _structure(g, cfg, [3, 3])
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    values = draw_worlds(model, ("param_mm_kappa_mult",), seed=13, draws=32)["param_mm_kappa_mult"]
    assert np.all((float(lo) <= values) & (values <= float(hi)))
    assert values.min() < values.max()


def test_log_uniform_mm_draws_stay_inside_bounds_whose_logarithms_round_outward():
    # Bounds one logarithmic ulp apart: exp of the rounded logarithms can land
    # an ulp outside [lo, hi] on either side unless the draw is kept on support.
    lo = 1e30
    log_hi = np.nextafter(np.log(lo), np.inf)
    hi = np.exp(log_hi) * (1.0 - 64.0 * _EPS)
    while np.log(hi) != log_hi:
        hi = np.nextafter(hi, np.inf)
    cfg = _cfg(
        mm_scale_prior="log_uniform",
        saturation_prior_ranges=_ranges(mm_kappa_mult=(lo, hi)),
    )
    g = _graph(2, 1)
    model, _, _ = build_world_model(g, cfg, _structure(g, cfg, [3, 3]), cfg.n_time_steps)
    values = draw_worlds(model, ("param_mm_kappa_mult",), seed=13, draws=32)["param_mm_kappa_mult"]
    assert np.all((lo <= values) & (values <= hi))


# -- The probability measure, independently of sampler implementation -----------


@pytest.mark.parametrize("path", ("world", "template"))
@pytest.mark.parametrize("mode", ("uniform", "log_uniform"))
def test_mm_draws_follow_configured_cdf_and_primitive_density(path, mode):
    lo, hi = 1e-4, 1e4
    options = {} if mode == "uniform" else {"mm_scale_prior": mode}
    cfg = _cfg(
        n_treatments=1,
        saturation_prior_ranges=_ranges(mm_kappa_mult=(lo, hi)),
        **options,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [3])
    if path == "world":
        model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
        values = draw_worlds(model, ("param_mm_kappa_mult",), seed=53, draws=4096)
    else:
        active = {
            "active_treatment": np.ones(1),
            "active_covariate": np.ones(1),
            "active_latent": np.ones(1),
        }
        cell = build_cell_inputs(cfg, g, active, structural)
        model, outputs, _ = build_world_model_template(cfg, cell, cfg.n_time_steps)
        draw = compile_template_draw_fn(model, (*outputs, "param_mm_kappa_mult"))
        values = draw(cell, seed=53, draws=4096)
    samples = values["param_mm_kappa_mult"].ravel()
    assert np.all((lo <= samples) & (samples <= hi))
    normalized = (
        (samples - lo) / (hi - lo)
        if mode == "uniform"
        else (np.log(samples) - np.log(lo)) / (np.log(hi) - np.log(lo))
    )
    ordered = np.sort(normalized)
    n = ordered.size
    ks_distance = max(
        np.max(np.arange(1, n + 1) / n - ordered),
        np.max(ordered - np.arange(n) / n),
    )
    # Dvoretzky-Kiefer-Wolfowitz bound with false-rejection probability 1e-7.
    assert ks_distance < np.sqrt(np.log(2 / 1e-7) / (2 * n))

    x = np.array([1e-3, 0.2, 25.0, 1000.0])
    primitive_name = "mm_kappa_mult" if mode == "uniform" else "mm_kappa_mult_log"
    primitive = next(rv for rv in model.free_RVs if rv.name == primitive_name)
    if mode == "uniform":
        density = np.array([pm.logp(primitive, np.array([value])).eval()[0] for value in x])
        expected = np.full_like(x, -np.log(hi - lo))
    else:
        # The free variable is log K; the original-scale law is the CDF check above.
        density = np.array([pm.logp(primitive, np.array([np.log(value)])).eval()[0] for value in x])
        expected = np.full_like(x, -np.log(np.log(hi) - np.log(lo)))
    np.testing.assert_allclose(density, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ("uniform", "log_uniform"))
def test_fixed_mm_scale_preserves_the_exact_original_bound(mode):
    # exp(log(1e20)) is not this exact float, and can lie outside a point prior.
    point = 1e20
    cfg = _cfg(
        mm_scale_prior=mode,
        saturation_prior_ranges=_ranges(mm_kappa_mult=(point, point)),
    )
    g = _graph(2, 1)
    structural = _structure(g, cfg, [3, 3])
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, ("param_mm_kappa_mult",), seed=13, draws=4)
    np.testing.assert_array_equal(drawn["param_mm_kappa_mult"], np.full((4, 2), point))


def test_large_integer_settings_obey_reference_laws_without_integer_tensor_conversion():
    cfg = _cfg(
        n_treatments=1,
        saturation_family_probs=_one_hot("michaelis_menten"),
        saturation_prior_ranges=_ranges(mm_kappa_mult=(10**20, 10**21)),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(1.0, 1.0),
        treatment_reference_multiplier=10**20,
        covariate_reference_contribution_range=(1.0, 1.0),
        covariate_reference_scale=10**20,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [3])
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    names = (
        "param_beta",
        "param_rho_zy",
        "param_mm_kappa_mult",
        "param_treatment_reference_input",
        "param_covariate_reference_input",
        "saturation_scale",
    )
    drawn = draw_worlds(model, names, seed=13, draws=4)
    ratio = drawn["param_treatment_reference_input"] / drawn["saturation_scale"]
    # The big integers are used as their float64 values, not truncated or wrapped.
    np.testing.assert_array_equal(
        drawn["param_treatment_reference_input"], 1e20 * drawn["saturation_scale"]
    )
    np.testing.assert_array_equal(drawn["param_covariate_reference_input"], 1e20)
    assert np.all((1e20 <= drawn["param_mm_kappa_mult"]) & (drawn["param_mm_kappa_mult"] <= 1e21))
    response = ratio / (ratio + drawn["param_mm_kappa_mult"])
    np.testing.assert_allclose(drawn["param_beta"] * response, 1.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        drawn["param_rho_zy"] * drawn["param_covariate_reference_input"],
        1.0,
        rtol=1e-12,
        atol=0.0,
    )


def test_fixed_shape_supports_and_targets_give_exact_finite_reference_coefficients():
    shapes = {
        "hill_slope": 2.0,
        "hill_kappa_mult": 1.2,
        "logistic_lam": 1.3,
        "mm_kappa_mult": 0.8,
        "tanh_c": 0.6,
        "root_alpha": 1.0,
    }
    cfg = _cfg(
        n_treatments=6,
        saturation_prior_ranges=_ranges(**{name: (value, value) for name, value in shapes.items()}),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(1.25, 1.25),
        treatment_reference_multiplier=1.6,
    )
    g = _graph(6, 1)
    structural = _structure(g, cfg, np.arange(6))
    model, _, reports = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, reports, seed=11, draws=3)
    for name, value in shapes.items():
        np.testing.assert_array_equal(drawn[f"param_{name}"], np.full((3, 6), value))
    for b in range(3):
        params = _params(drawn, b, structural, reports, cfg)
        beta = np.asarray(params["beta"])
        reference_response = np.array(
            [
                _numpy_response(family, np.array([1.6]), params, k)[0]
                for k, family in enumerate(SATURATION_FAMILY_KEYS)
            ]
        )
        np.testing.assert_allclose(beta * reference_response, 1.25, rtol=1e-12)
        np.testing.assert_array_equal(params["treatment_reference_contribution"], 1.25)


# -- Independent reference identities, gates, precedence, and relative noise ----


@pytest.fixture(scope="module")
def reference_worlds():
    cfg = _cfg(
        n_treatments=12,
        n_covariates=3,
        saturation_prior_ranges=_ranges(
            hill_slope=(1.4, 2.4),
            hill_kappa_mult=(0.4, 0.9),
            logistic_lam=(0.8, 1.8),
            mm_kappa_mult=(0.2, 5.0),
            tanh_c=(0.4, 0.9),
            root_alpha=(0.45, 0.75),
        ),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(0.0, 1.4),
        treatment_reference_multiplier=0.37,
        covariate_reference_contribution_range=(-2.0, 3.0),
        covariate_reference_scale=2.75,
        beta_additive_range=(80.0, 90.0),
        zy_coeff_range=(60.0, 70.0),
        rw_baseline_std_range=(0.05, 0.1),
        rw_outcome_std_range=(0.02, 0.05),
    )
    g = _graph(12, 3)
    g["g_cy"][6:] = 0
    g["g_zy"][2] = 0
    structural = _structure(g, cfg, np.tile(np.arange(6), 2), carryover=np.tile([0, 1], 6))
    model, outputs, reports = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = _draw(model, outputs, reports, draws=3)
    return cfg, g, structural, model, reports, drawn


def test_every_family_uses_its_own_target_and_shapes_before_gates(reference_worlds):
    cfg, g, structural, model, reports, drawn = reference_worlds
    for name, (family, key) in SHAPE_FIELDS.items():
        lo, hi = cfg.saturation_prior_ranges[family][key]
        values = drawn[f"param_{name}"]
        assert np.all((lo <= values) & (values <= hi)), name
        if name == "mm_kappa_mult":
            primitive_name = "mm_kappa_mult_log"
            primitive_values = np.log(values[0])
            width = np.log(hi) - np.log(lo)
        else:
            primitive_name, primitive_values, width = name, values[0], hi - lo
        primitive = next(rv for rv in model.free_RVs if rv.name == primitive_name)
        np.testing.assert_allclose(pm.logp(primitive, primitive_values).eval(), -np.log(width))
    for b in range(3):
        params = _params(drawn, b, structural, reports, cfg)
        beta = np.asarray(params["beta"])
        targets = np.asarray(params["treatment_reference_contribution"])
        scales = drawn["saturation_scale"][b]
        reference = np.asarray(params["treatment_reference_input"])
        independent_scale = np.logaddexp(0.0, np.logaddexp(0.0, params["rw_c"]["mean"]))
        np.testing.assert_allclose(scales, independent_scale, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            reference,
            cfg.treatment_reference_multiplier * independent_scale,
            rtol=1e-12,
            atol=0.0,
        )
        for k, family_id in enumerate(structural["sat_family"]):
            family = SATURATION_FAMILY_KEYS[int(family_id)]
            response = _numpy_response(family, np.array([reference[k] / scales[k]]), params, k)[0]
            np.testing.assert_allclose(beta[k] * response, targets[k], rtol=2e-12, atol=0.0)
        assert np.all(beta < cfg.beta_additive_range[0]), "raw beta prior must not win precedence"
        expected = _numpy_contributions(drawn["treatments"][b], scales, params, g["g_cy"])
        np.testing.assert_allclose(
            drawn["contributions_observed"][b], expected, rtol=2e-12, atol=1e-12
        )
        np.testing.assert_array_equal(drawn["contributions_observed"][b, :, 6:], 0.0)


def test_signed_controls_and_relative_noise_use_derived_gated_coefficients(reference_worlds):
    cfg, g, structural, _, reports, drawn = reference_worlds
    for b in range(3):
        params = _params(drawn, b, structural, reports, cfg)
        target = params["covariate_reference_contribution"]
        rho = params["rho_zy"]
        np.testing.assert_allclose(rho * cfg.covariate_reference_scale, target, rtol=1e-12)
        np.testing.assert_array_equal(params["covariate_reference_input"], np.full(3, 2.75))
        np.testing.assert_allclose(
            drawn["covariate_contribution"][b],
            drawn["covariates"][b] * (g["g_zy"] * rho),
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.all(rho < cfg.zy_coeff_range[0]), "raw control prior must not win precedence"
        amplitude = np.linalg.norm(g["g_cy"] * params["beta"])
        for group, primitive in (("rw_b", "rw_b_std_rel"), ("rw_y", "rw_y_std_rel")):
            np.testing.assert_allclose(
                params[group]["std"], drawn[primitive][b] * amplitude, rtol=1e-12
            )


@pytest.mark.parametrize("latent", ("marginal", "sampled"))
def test_oracle_measures_and_conditioned_reference_effects_match_independent_laws(
    reference_worlds, latent
):
    cfg, g, structural, generative, reports, drawn = reference_worlds
    values = {name: value[0] for name, value in drawn.items()}
    # Force both signs and different targets per input; the expected values must
    # not become a seed lottery or permit a coefficient broadcast from slot 0.
    values["treatment_reference_contribution"] = np.linspace(0.05, 1.35, 12)
    values["covariate_reference_contribution"] = np.array([-1.25, 0.75, -0.5])
    values["mm_kappa_mult_log"] = np.log(np.geomspace(0.3, 3.0, 12))
    data = {
        name: drawn[name][0] for name in ("treatments", "covariates", "outcome", "saturation_scale")
    }
    oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
    for name, bounds in (
        ("treatment_reference_contribution", (0.0, 1.4)),
        ("covariate_reference_contribution", (-2.0, 3.0)),
        ("mm_kappa_mult_log", (np.log(0.2), np.log(5.0))),
    ):
        expected = -np.log(bounds[1] - bounds[0])
        for model in (generative, oracle):
            primitive = next(rv for rv in model.free_RVs if rv.name == name)
            density = np.asarray(pm.logp(primitive, values[name]).eval())
            # Compare the independent mathematical density at its returned
            # precision: exact scalar bounds can yield float32 constant logp.
            np.testing.assert_allclose(
                density, np.asarray(expected, dtype=density.dtype), rtol=1e-12
            )

    names = ("beta", "rho_zy", "rw_b_std", "rw_y_std")
    generated = _evaluate_given(generative, values, (*names, "contributions_observed"))
    inferred = _evaluate_given(oracle, values, (*names, "contributions"))
    for name in names:
        np.testing.assert_allclose(inferred[name], generated[name], rtol=1e-12, atol=1e-12)
    params = _params(drawn, 0, structural, reports, cfg)
    params["mm_kappa_mult"] = np.exp(values["mm_kappa_mult_log"])
    expected_beta = np.array(
        [
            values["treatment_reference_contribution"][k]
            / _numpy_response(SATURATION_FAMILY_KEYS[int(family)], np.array([0.37]), params, k)[0]
            for k, family in enumerate(structural["sat_family"])
        ]
    )
    np.testing.assert_allclose(inferred["beta"], expected_beta, rtol=1e-12)
    np.testing.assert_allclose(
        inferred["rho_zy"], values["covariate_reference_contribution"] / 2.75
    )
    params["beta"] = expected_beta
    expected = _numpy_contributions(data["treatments"], data["saturation_scale"], params, g["g_cy"])
    np.testing.assert_allclose(inferred["contributions"], expected, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(
        generated["contributions_observed"], expected, rtol=2e-12, atol=1e-12
    )
    amplitude = np.linalg.norm(g["g_cy"] * expected_beta)
    for name, primitive in (("rw_b_std", "rw_b_std_rel"), ("rw_y_std", "rw_y_std_rel")):
        np.testing.assert_allclose(inferred[name], values[primitive] * amplitude, rtol=1e-12)
    point = _point(oracle, values)
    assert np.isfinite(oracle.compile_logp(mode="FAST_COMPILE")(point))
    gradient = oracle.compile_dlogp(mode="FAST_COMPILE")(point)
    assert np.isfinite(gradient).all() and np.any(gradient != 0.0)


# -- A genuinely reused padded template, not a static-graph proxy ----------------


def test_template_reuses_reference_priors_with_changed_families_and_inactive_inputs():
    cfg = _cfg(
        n_treatments=6,
        n_covariates=3,
        n_treatments_active_range=(2, 6),
        n_covariates_active_range=(1, 3),
        saturation_prior_ranges=_ranges(hill_slope=(4.0, 5.0), mm_kappa_mult=(0.1, 10.0)),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(0.7, 1.1),
        treatment_reference_multiplier=1.6,
        covariate_reference_contribution_range=(-0.8, 0.5),
        covariate_reference_scale=1.75,
        beta_additive_range=(40.0, 50.0),
        zy_coeff_range=(20.0, 30.0),
    )
    cells = []
    for families, treatment_mask, covariate_mask, g_cy, g_zy in (
        (np.arange(6), [1] * 6, [1] * 3, [1, 1, 0, 1, 1, 1], [1, 1, 0]),
        (
            np.arange(5, -1, -1),
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0],
        ),
    ):
        g = _graph(6, 3)
        g["g_cy"] = np.asarray(g_cy)
        g["g_zy"] = np.asarray(g_zy)
        active = {
            "active_treatment": np.asarray(treatment_mask),
            "active_covariate": np.asarray(covariate_mask),
            "active_latent": np.ones(1),
        }
        structural = _structure(g, cfg, families, carryover=[0, 1, 0, 1, 0, 1])
        cells.append(build_cell_inputs(cfg, g, active, structural))
    model, outputs, reports = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
    noise_primitives = ("rw_b_std_rel", "rw_y_std_rel")
    names = outputs + reports + noise_primitives
    draw = compile_template_draw_fn(model, names)
    results = []
    for cell in (cells[0], cells[1], cells[0]):
        drawn = draw(cell, seed=41, draws=2)
        results.append(drawn)
        for b in range(2):
            params = {name.removeprefix("param_"): drawn[name][b] for name in reports}
            params.update(
                l_max=cfg.l_max,
                sat_family=cell["sat_family"],
                carryover_family=cell["carryover_family"],
            )
            beta = params["beta"]
            for k, family_id in enumerate(cell["sat_family"]):
                response = _numpy_response(
                    SATURATION_FAMILY_KEYS[int(family_id)], np.array([1.6]), params, k
                )[0]
                np.testing.assert_allclose(
                    beta[k] * response,
                    params["treatment_reference_contribution"][k],
                    rtol=2e-12,
                )
            expected = _numpy_contributions(
                drawn["treatments"][b], drawn["saturation_scale"][b], params, cell["g_cy"]
            )
            np.testing.assert_allclose(
                drawn["contributions_observed"][b], expected, rtol=2e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                params["rho_zy"] * 1.75, params["covariate_reference_contribution"], rtol=1e-12
            )
            np.testing.assert_allclose(
                drawn["covariate_contribution"][b],
                drawn["covariates"][b] * (cell["g_zy"] * params["rho_zy"]),
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_array_equal(
                drawn["treatments"][b, :, cell["active_treatment"] == 0], 0.0
            )
            np.testing.assert_array_equal(
                drawn["covariates"][b, :, cell["active_covariate"] == 0], 0.0
            )
            np.testing.assert_array_equal(
                drawn["contributions_observed"][b, :, cell["g_cy"] == 0], 0.0
            )
            amplitude = np.linalg.norm(cell["g_cy"] * beta)
            for group, primitive in zip(("rw_b_std", "rw_y_std"), noise_primitives):
                np.testing.assert_allclose(
                    params[group], drawn[primitive][b] * amplitude, rtol=1e-12
                )
    for name in names:
        np.testing.assert_array_equal(results[2][name], results[0][name], err_msg=name)


def test_inactive_template_coefficients_stay_inside_admitted_family_support():
    cfg = _cfg(
        n_treatments_active_range=(1, 2),
        saturation_family_probs=_one_hot("root"),
        saturation_prior_ranges=_ranges(root_alpha=(0.001, 0.001)),
        treatment_reference_contribution_range=(1e30, 1e30),
        treatment_reference_multiplier=1e-290,
    )
    g = _graph(2, 1)
    g["g_cy"][1] = 0
    structural = _structure(_graph(1, 1), cfg, [5])
    active = {
        "active_treatment": np.array([1, 0]),
        "active_covariate": np.ones(1),
        "active_latent": np.ones(1),
    }
    cell = build_cell_inputs(cfg, g, active, structural)
    model, _, _ = build_world_model_template(cfg, cell, cfg.n_time_steps)
    names = (
        "param_beta",
        "param_treatment_reference_input",
        "param_rw_b_std",
        "param_rw_y_std",
        "rw_b_std_rel",
        "rw_y_std_rel",
        "saturation_scale",
        "contributions_observed",
        "outcome",
    )
    drawn = compile_template_draw_fn(model, names)(cell, seed=41)
    beta = drawn["param_beta"]
    assert np.isfinite(beta).all() and np.all(beta <= CORPUS_STORAGE_MAX)
    ratio = drawn["param_treatment_reference_input"] / drawn["saturation_scale"]
    np.testing.assert_allclose(beta * ratio**0.001, 1e30, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(drawn["contributions_observed"][:, :, 1], 0.0)
    for group in ("b", "y"):
        np.testing.assert_allclose(
            drawn[f"param_rw_{group}_std"],
            drawn[f"rw_{group}_std_rel"] * beta[:, :1],
            rtol=1e-12,
            atol=0.0,
        )
    assert np.isfinite(drawn["outcome"]).all()


_ANCHOR_REPORTS = (
    "saturation_scale",
    "param_treatment_reference_input",
    "param_treatment_reference_contribution",
    "param_beta",
    "param_mm_kappa_mult",
    "param_rw_z_mean",
    "param_rw_c_mean",
    "param_gamma_zz",
    "param_v_zc",
    "param_alpha_cc",
    "param_pulse_amp",
    "param_pulse_prob",
)


def _assert_reference_anchor_laws(drawn, g, use_pulse, multiplier):
    for b in range(drawn["saturation_scale"].shape[0]):
        params = {name.removeprefix("param_"): drawn[name][b] for name in _ANCHOR_REPORTS}
        np.testing.assert_allclose(
            params["saturation_scale"], _independent_anchors(params, g, use_pulse), rtol=1e-12
        )
        np.testing.assert_array_equal(
            params["treatment_reference_input"], multiplier * params["saturation_scale"]
        )
        response = multiplier / (multiplier + params["mm_kappa_mult"])
        np.testing.assert_allclose(
            params["beta"] * response, params["treatment_reference_contribution"], rtol=1e-12
        )


def test_reference_inputs_follow_parent_and_pulse_anchors_in_world_and_template():
    cfg = _cfg(
        n_treatments=3,
        n_covariates=2,
        saturation_family_probs=_one_hot("michaelis_menten"),
        treatment_pulse_prob_range=(0.2, 0.4),
        treatment_reference_contribution_range=(0.5, 1.5),
        treatment_reference_multiplier=1.7,
    )
    g = _graph(3, 2)
    g["g_zc"][0, 1] = g["g_zc"][1, 2] = g["g_cc"][0, 2] = g["g_cc"][1, 2] = 1
    g["g_zz"][0, 1] = 1
    structural = _structure(g, cfg, [3, 3, 3])
    structural["use_pulse"] = np.ones(3, dtype=bool)
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, _ANCHOR_REPORTS, seed=5, draws=3)
    assert np.all(drawn["param_pulse_prob"] > 0.0)
    _assert_reference_anchor_laws(drawn, g, structural["use_pulse"], 1.7)

    active = {
        "active_treatment": np.ones(3),
        "active_covariate": np.ones(2),
        "active_latent": np.ones(1),
    }
    without_parents = _graph(3, 2)
    cells = [build_cell_inputs(cfg, graph, active, structural) for graph in (g, without_parents)]
    template, _, _ = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
    draw = compile_template_draw_fn(template, _ANCHOR_REPORTS)
    for graph, cell in zip((g, without_parents), cells):
        _assert_reference_anchor_laws(draw(cell, seed=7, draws=2), graph, np.ones(3), 1.7)


# -- Conditioning really narrows the configured support, not the old defaults ---


@pytest.mark.parametrize(
    ("quantity", "options"),
    (
        # Neither float64 endpoint moves by any admitted width.
        ("hill_shape", {"saturation_prior_ranges": _ranges(hill_slope=(1e20, 1e20 + 1e10))}),
        # Only the upper endpoint collapses; float32 storage cannot resolve either.
        ("hill_shape", {"saturation_prior_ranges": _ranges(hill_slope=(1.0, 1e17))}),
        # Distinct in float64, but float32 prior_cond labels shift by more than a width.
        (
            "hill_shape",
            {
                "saturation_prior_ranges": _ranges(hill_slope=(1e8, 2e8)),
                "prior_cond_width_ranges": {"hill_shape": (1.0, 2.0)},
            },
        ),
        # The rule follows the support's magnitude, not its narrow span.
        (
            "carryover_alpha",
            {
                "carryover_alpha_range": (0.9, 0.95),
                "prior_cond_width_ranges": {"carryover_alpha": (1e-7, 2e-7)},
            },
        ),
        # Just under the resolution 8 * eps32 * 3 at the default Hill support (1, 3);
        # with the 5e-6 acceptance below this pins the factor of eight.
        ("hill_shape", {"prior_cond_width_ranges": {"hill_shape": (2.5e-6, 1e-5)}}),
    ),
)
def test_conditioning_rejects_widths_float32_labels_cannot_resolve(quantity, options):
    with pytest.raises(ValueError, match=f"{quantity}.*prior_cond_width_ranges.*float32"):
        _cfg(prior_conditioning=True, **options)


def test_hill_conditioning_records_and_uses_disjoint_configured_support():
    cfg = _corpus_cfg(
        saturation_prior_ranges=_ranges(hill_slope=(4.0, 5.0)),
        saturation_family_probs=_one_hot("hill"),
        prior_conditioning=True,
        prior_cond_width_ranges={"hill_shape": (0.2, 0.4)},
        edge_budget={"cy": 2, "zy": 0, "dc": 0, "dz": 0, "dy": 0, "zc": 0, "cc": 0, "zz": 0},
    )
    interval = sample_prior_cond(cfg, np.random.default_rng(23))
    lo, width = interval["hill_shape"]
    assert 4.0 <= lo < lo + width <= 5.0
    assert 0.2 <= width <= 0.4
    g = _graph(2, 1)
    g["g_zy"][:] = 0
    structural = _structure(g, cfg, [1, 1])
    model, outputs, reports = build_world_model(
        g, cfg, structural, cfg.n_time_steps, prior_cond=interval
    )
    drawn = _draw(model, outputs, reports, seed=19, draws=8)
    assert np.all((lo <= drawn["hill_slope"]) & (drawn["hill_slope"] <= lo + width))
    data = {
        name: drawn[name][0] for name in ("treatments", "covariates", "outcome", "saturation_scale")
    }
    for latent in ("marginal", "sampled"):
        oracle = build_oracle_model(g, cfg, structural, data, prior_cond=interval, latent=latent)
        hill = next(rv for rv in oracle.free_RVs if rv.name == "hill_slope")
        np.testing.assert_allclose(pm.logp(hill, np.full(2, lo + width / 2)).eval(), -np.log(width))
        assert np.isneginf(pm.logp(hill, np.array([3.0, lo + width + 0.01])).eval()).all()
    corpus = pg.sample_prior_predictive(cfg)
    recorded = corpus["prior_cond"]
    low = recorded[:, PRIOR_COND_LAYOUT.index("hill_shape_low")]
    span = recorded[:, PRIOR_COND_LAYOUT.index("hill_shape_width")]
    assert np.all((low >= 4.0 - 1e-6) & (low + span <= 5.0 + 1e-6))
    assert corpus["diagnostics"]["prior_cond"]["supports"]["hill_shape"] == [4.0, 5.0]
    assert pg.DataGenerator.validate_corpus(corpus) == []


def test_conditioned_float32_labels_near_the_resolution_contain_the_draws():
    # Widths of 5e-6 sit just above the resolution 8 * eps32 * 3 at support (1, 3).
    cfg = _cfg(
        saturation_family_probs=_one_hot("hill"),
        prior_conditioning=True,
        prior_cond_width_ranges={"hill_shape": (5e-6, 1e-5)},
    )
    world = pg.sample_scm(cfg, seed=7, max_eps_draws=8)
    low, width = (np.float32(x) for x in world.extras["prior_cond"]["hill_shape"])
    slack = np.spacing(np.float32(3.0))
    slopes = np.asarray(world.params["hill_slope"])
    assert np.all((low - slack <= slopes) & (slopes <= low + width + slack))
    # A tiny carryover support is resolved at its own magnitude, not at 1.
    carry = _corpus_cfg(
        carryover_family_probs={key: float(key == "geometric") for key in CARRYOVER_FAMILY_KEYS},
        carryover_alpha_range=(0.0, 5e-7),
        prior_conditioning=True,
        prior_cond_width_ranges={"carryover_alpha": (1e-7, 4e-7)},
    )
    corpus = pg.sample_prior_predictive(carry)
    assert pg.DataGenerator.validate_corpus(corpus) == []
    low = corpus["prior_cond"][:, PRIOR_COND_LAYOUT.index("carryover_alpha_low")][:, None]
    width = corpus["prior_cond"][:, PRIOR_COND_LAYOUT.index("carryover_alpha_width")][:, None]
    slack = np.spacing(np.float32(5e-7))
    alpha = corpus["carryover_alpha"]
    inside = (low - slack <= alpha) & (alpha <= low + width + slack)
    assert np.all(inside[corpus["treatment_active_mask"].astype(bool)])


@pytest.mark.parametrize(
    ("options", "error"),
    (
        # Unconfigured default Hill widths are not checked against a narrow support.
        (
            {
                "prior_cond_width_ranges": {"carryover_alpha": (0.05, 0.1)},
                "saturation_prior_ranges": _ranges(hill_slope=(2.0, 2.5)),
            },
            None,
        ),
        # No labels are written, so narrow configured widths keep their base behaviour.
        ({"prior_cond_width_ranges": {"carryover_alpha": (1e-7, 1e-6)}}, None),
        # Configured widths must still fit their support (0.2, 0.8).
        ({"prior_cond_width_ranges": {"carryover_alpha": (0.5, 0.9)}}, "support width"),
    ),
)
def test_disabled_conditioning_checks_only_configured_widths_against_supports(options, error):
    if error is None:
        assert not _cfg(**options).prior_conditioning
    else:
        with pytest.raises(ValueError, match=f"carryover_alpha.*{error}"):
            _cfg(**options)


def test_opt_in_descriptions_print_conditioning_intervals_exactly():
    # The validated width can sit far below the support's leading digits.
    cfg = _cfg(
        saturation_prior_ranges=_ranges(hill_slope=(100.0, 101.0)),
        saturation_family_probs=_one_hot("linear"),
        prior_conditioning=True,
        prior_cond_width_ranges={"hill_shape": (0.001, 0.002)},
    )
    world = pg.sample_scm(cfg, seed=7, max_eps_draws=8)
    lo, width = world.extras["prior_cond"]["hill_shape"]
    printed = re.search(
        r"hill_shape: U\(([^,]+), ([^)]+)\)\s+width=(\S+)", pg.describe_scm(world)
    ).groups()
    np.testing.assert_array_equal([float(x) for x in printed], [lo, lo + width, width])


# -- Cancellation, overflow, and impossible coefficients ------------------------


def _oracle_data(cfg, scale):
    return {
        "treatments": np.ones((cfg.n_time_steps, 1)),
        "covariates": np.zeros((cfg.n_time_steps, 1)),
        "outcome": np.zeros(cfg.n_time_steps),
        "saturation_scale": np.array([scale]),
    }


@pytest.mark.parametrize(
    ("family", "options", "scale", "variable", "match"),
    (
        (
            "linear",
            {"treatment_reference_multiplier": 1e20},
            1e300,
            "treatment_reference_input",
            "treatment_reference_multiplier.*treatment_reference_input",
        ),
        (
            "root",
            {
                "treatment_reference_multiplier": 1e-299,
                "saturation_prior_ranges": _ranges(root_alpha=(0.001, 0.001)),
            },
            1e-30,
            "treatment_reference_input",
            "treatment_reference_multiplier.*treatment_reference_input",
        ),
    ),
    ids=("input_overflow", "input_underflow"),
)
def test_oracle_rejects_unrepresentable_supplied_reference_inputs(
    family, options, scale, variable, match
):
    cfg = _cfg(
        n_treatments=1,
        saturation_family_probs=_one_hot(family),
        treatment_reference_contribution_range=(1.0, 1.0),
        **options,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index(family)])
    oracle = build_oracle_model(g, cfg, structural, _oracle_data(cfg, scale))
    with pytest.raises(ValueError, match=match):
        oracle[variable].eval()


def test_oracle_accepts_a_float32_stored_minimum_anchor():
    cfg = _cfg(
        n_treatments=1,
        saturation_family_probs=_one_hot("michaelis_menten"),
        saturation_prior_ranges=_ranges(mm_kappa_mult=(1.0, 1.0)),
        treatment_reference_contribution_range=(1.0, 1.0),
        treatment_reference_multiplier=2.0,
    )
    # Corpora persist anchors as float32; the generator's 1e-8 floor rounds below it.
    scale = float(np.float32(1e-8))
    assert scale < 1e-8
    g = _graph(1, 1)
    oracle = build_oracle_model(g, cfg, _structure(g, cfg, [3]), _oracle_data(cfg, scale))
    reference = oracle["treatment_reference_input"].eval()
    np.testing.assert_array_equal(reference, [2.0 * scale])
    response = reference / (reference + max(scale, 1e-8))
    np.testing.assert_allclose(oracle["beta"].eval() * response, 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("family", "multiplier", "shape"),
    (
        ("hill", 1e-8, {"hill_slope": (3.0, 3.0), "hill_kappa_mult": (1.0, 1.0)}),
        ("hill", 2.0, {"hill_slope": (1e16, 1e16), "hill_kappa_mult": (2.0, 2.0)}),
        ("logistic", 1e-20, {"logistic_lam": (2.0, 2.0)}),
        # x / (kappa * r) underflows: the separate-logarithm far tail.
        ("hill", 1e-299, {"hill_slope": (0.001, 0.001), "hill_kappa_mult": (1e30, 1e30)}),
    ),
)
def test_opt_in_curves_preserve_tiny_responses_and_high_slope_half_points(
    family, multiplier, shape
):
    cfg = _cfg(
        n_treatments=1,
        saturation_prior_ranges=_ranges(**shape),
        saturation_family_probs=_one_hot(family),
        treatment_reference_contribution_range=(1.0, 1.0),
        treatment_reference_multiplier=multiplier,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index(family)])
    model, outputs, reports = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = _draw(model, outputs, reports)
    params = _params(drawn, 0, structural, reports, cfg)
    beta = np.asarray(params["beta"])[0]
    ratios = np.array([0.0, multiplier, 1.0, 2.0])
    expected_response = _numpy_response(family, ratios, params, 0)
    for dynamic in (False, True):
        response = np.asarray(
            _saturate_col(
                pt.as_tensor_variable(3.0 * ratios),
                pt.as_tensor_variable(np.float64(3.0)),
                params,
                0,
                dynamic_family=dynamic,
            ).eval()
        )
        np.testing.assert_allclose(response, expected_response, rtol=2e-12, atol=0.0)
        assert response[0] == 0.0 and response[1] > 0.0
        assert np.isfinite(beta) and beta <= CORPUS_STORAGE_MAX
        np.testing.assert_allclose(beta * response[1], 1.0, rtol=2e-12, atol=0.0)
    expected = _numpy_contributions(
        drawn["treatments"][0], drawn["saturation_scale"][0], params, g["g_cy"]
    )
    np.testing.assert_allclose(drawn["contributions_observed"][0], expected, rtol=2e-12, atol=0.0)


@pytest.mark.parametrize("shape", ("concrete", "tensor_scalar", "tensor_vector"))
def test_stable_logistic_preserves_minimum_positive_lambda_with_runtime_inputs(shape):
    x = pt.dvector("x")
    lam = np.nextafter(0.0, 1.0)
    shape_value = lam
    if shape == "tensor_scalar":
        shape_value = pt.as_tensor_variable(lam)
    elif shape == "tensor_vector":
        shape_value = pt.as_tensor_variable(np.array([lam]))
    response = mechanisms.stable_logistic_kappa_relative(x, 1.0, lam=shape_value)
    evaluate = pytensor.function([x], response, mode="FAST_COMPILE")
    inputs = np.array([-1e300, -1e38, -1e20, -2.0, 0.0, 2.0, 1e20, 1e38, 1e300])
    expected = np.tanh((lam * inputs) / 2.0)
    np.testing.assert_allclose(evaluate(inputs), expected, rtol=8.0 * _EPS, atol=0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_logistic_subnormal_response_and_amplification_match_decimal(mode):
    x, reference, lam = pt.dvectors("x", "reference", "lambda")
    response = mechanisms.stable_logistic_kappa_relative(x, reference, lam=lam)
    evaluate = pytensor.function(
        [x, reference, lam], [response, CORPUS_STORAGE_MAX * response], mode=mode
    )
    minimum, tiny = np.nextafter(0.0, 1.0), np.finfo(np.float64).tiny
    cases = [
        (value, 1.0, minimum)
        for value in (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.nextafter(3.0, np.inf))
    ]
    cases += [
        (np.nextafter(4.5, np.inf), np.nextafter(1.5, np.inf), minimum),
        (1.25e308, 1e308, minimum),
        (np.nextafter(2.0, 0.0), 1.0, tiny),
        (2.0, 1.0, tiny),
    ]
    cases = [(sign * value, scale, shape) for value, scale, shape in cases for sign in (-1, 1)]
    expected = []
    with localcontext() as context:
        # Resolve the cubic correction below exact half-ULP ties, not only
        # the first-order term or the already-rounded lambda*input product.
        context.prec = 1200
        for value, scale, shape in cases:
            argument = (
                Decimal.from_float(value) * Decimal.from_float(shape) / Decimal.from_float(scale)
            )
            tail = (-argument).exp()
            expected.append(float((1 - tail) / (1 + tail)))
    values, scaled = evaluate(*[np.array(column) for column in zip(*cases)])
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(scaled, CORPUS_STORAGE_MAX * np.array(expected))


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_stable_logistic_preserves_normal_curve_and_zero_input_gradient(mode):
    x, r, lam = pt.dvector("x"), pt.dscalar("r"), pt.dscalar("lam")
    response = mechanisms.stable_logistic_kappa_relative(x, r, lam=lam)
    derivative = pytensor.grad(response.sum(), x)
    evaluate = pytensor.function([x, r, lam], [response, derivative], mode=mode)
    inputs = np.array([-1e20, -7.0, -0.5, 0.0, 0.5, 7.0, 1e20])
    scale, shape = 3.0, 1.3
    expected = np.tanh((shape * (inputs / scale)) / 2.0)
    expected_derivative = (shape / (2.0 * scale)) * (1.0 - expected**2)
    values, derivatives = evaluate(inputs, scale, shape)
    np.testing.assert_allclose(values, expected, rtol=8.0 * _EPS, atol=0.0)
    np.testing.assert_allclose(derivatives, expected_derivative, rtol=32.0 * _EPS, atol=0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_logistic_weighted_saturated_tail_preserves_complete_derivatives(mode):
    x, reference, lam = pt.dvector("x"), pt.dscalar("reference"), pt.dscalar("lambda")
    response = mechanisms.stable_logistic_kappa_relative(x, reference, lam=lam)
    inputs = np.array([-1e38, 0.0, 1e38])
    weights = np.array([1e200, 3e200, -2e200])
    gradients = pytensor.grad((response * weights).sum(), [x, reference, lam])
    evaluate = pytensor.function([x, reference, lam], [response, *gradients], mode=mode)
    with localcontext() as context:
        context.prec = 100
        scale, shape = (Decimal.from_float(value) for value in (1.0, 8e-36))
        expected_x, expected_reference, expected_lambda = [], Decimal(0), Decimal(0)
        for value, weight in zip(inputs, weights):
            value, weight = Decimal.from_float(float(value)), Decimal.from_float(float(weight))
            tail = (-(shape * value / scale).copy_abs()).exp()
            common = 2 * weight * tail / (1 + tail) ** 2
            expected_x.append(float(common * shape / scale))
            expected_reference -= common * shape * value / scale**2
            expected_lambda += common * value / scale
    values, grad_x, grad_reference, grad_lambda = evaluate(inputs, 1.0, 8e-36)
    np.testing.assert_array_equal(values, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(grad_x, expected_x, rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(grad_reference, float(expected_reference), rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(grad_lambda, float(expected_lambda), rtol=2e-12, atol=0.0)
    assert np.all(np.asarray(expected_x) != 0.0)
    assert expected_reference > 0 and expected_lambda < 0


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_sampled_logistic_oracle_retains_weighted_tail_posterior_gradient(mode):
    cfg = _cfg(
        n_treatments=1,
        nonlinearity="linear",
        beta_additive_range=(1.0, 1.0),
        saturation_family_probs=_one_hot("logistic"),
        saturation_prior_ranges=_ranges(logistic_lam=(7e-36, 9e-36)),
        baseline_floor=None,
        rw_baseline_mean_range=(0.0, 0.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_outcome_std_range=(1e-100, 1e-100),
    )
    g = _graph(1, 1)
    g["g_zy"][:] = 0
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index("logistic")])
    data = _oracle_data(cfg, 1.0)
    data["treatments"][:] = 1e38
    oracle = build_oracle_model(g, cfg, structural, data, latent="sampled")
    variable = oracle["logistic_lam"]
    point = oracle.initial_point()
    point[oracle.rvs_to_values[variable].name] = np.array([0.0])
    assert np.isfinite(oracle.compile_logp(mode=mode)(point))
    actual = oracle.compile_dlogp(vars=[variable], mode=mode)(point)
    # The midpoint prior/Jacobian score is zero. The finite Normal cotangent
    # rescues the tail before the lambda interval derivative scales it down.
    with localcontext() as context:
        context.prec = 100
        lo, hi = (Decimal.from_float(value) for value in (7e-36, 9e-36))
        shape = float((lo + hi) / 2)
        x, shape, sigma = (Decimal.from_float(value) for value in (1e38, shape, 1e-100))
        tail = (-(x * shape)).exp()
        raw_score = -Decimal(cfg.n_time_steps) * 2 * x * tail / (1 + tail) ** 2 / sigma**2
        expected = float(raw_score * (hi - lo) / 4)
    assert expected < 0.0
    np.testing.assert_allclose(actual, [expected], rtol=2e-12, atol=0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_hill_weighted_saturated_tails_preserve_broadcast_pullbacks(mode):
    x, reference = pt.dmatrix("x"), pt.dscalar("reference")
    slope = pt.tensor("slope", dtype="float64", shape=(1, 2))
    kappa = pt.tensor("kappa", dtype="float64", shape=(2, 1))
    response = mechanisms.stable_hill_kappa_relative(x, reference, slope=slope, kappa_mult=kappa)
    inputs = np.array([[np.e, np.exp(-1.0)], [0.0, -1.0]])
    weights = np.array([[1e200, -3e200], [2e200, -4e200]])
    gradients = pytensor.grad((response * weights).sum(), [x, reference, slope, kappa])
    evaluate = pytensor.function([x, reference, slope, kappa], [response, *gradients], mode=mode)
    with localcontext() as context:
        context.prec = 100
        shape = Decimal(800)
        expected_x = np.zeros_like(inputs)
        expected_reference = Decimal(0)
        expected_slope = [Decimal(0), Decimal(0)]
        expected_kappa = [Decimal(0), Decimal(0)]
        for row, column in np.ndindex(inputs.shape):
            value = Decimal.from_float(float(inputs[row, column]))
            if value <= 0:
                continue
            weight = Decimal.from_float(float(weights[row, column]))
            log_ratio = value.ln()
            tail = (-(shape * log_ratio).copy_abs()).exp()
            common = weight * tail / (1 + tail) ** 2
            expected_x[row, column] = float(common * shape / value)
            expected_reference -= common * shape
            expected_slope[column] += common * log_ratio
            expected_kappa[row] -= common * shape
    values, grad_x, grad_reference, grad_slope, grad_kappa = evaluate(
        inputs, 1.0, np.full((1, 2), 800.0), np.ones((2, 1))
    )
    np.testing.assert_array_equal(values, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(grad_x, expected_x, rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(grad_reference, float(expected_reference), rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(
        grad_slope, [[float(value) for value in expected_slope]], rtol=2e-12, atol=0.0
    )
    np.testing.assert_allclose(
        grad_kappa, [[float(value)] for value in expected_kappa], rtol=2e-12, atol=0.0
    )
    assert np.all(expected_x[0] != 0.0)
    assert expected_reference > 0 and all(value > 0 for value in expected_slope)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_sampled_hill_oracle_retains_weighted_tail_posterior_gradient(mode):
    cfg = _cfg(
        n_treatments=1,
        nonlinearity="linear",
        beta_additive_range=(1.0, 1.0),
        saturation_family_probs=_one_hot("hill"),
        saturation_prior_ranges=_ranges(hill_slope=(700.0, 900.0), hill_kappa_mult=(1.0, 1.0)),
        baseline_floor=None,
        rw_baseline_mean_range=(0.0, 0.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_outcome_std_range=(1e-100, 1e-100),
    )
    g = _graph(1, 1)
    g["g_zy"][:] = 0
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index("hill")])
    data = _oracle_data(cfg, 1.0)
    data["treatments"][:] = np.e
    oracle = build_oracle_model(g, cfg, structural, data, latent="sampled")
    variable = oracle["hill_slope"]
    point = oracle.initial_point()
    point[oracle.rvs_to_values[variable].name] = np.array([0.0])
    assert np.isfinite(oracle.compile_logp(mode=mode)(point))
    actual = oracle.compile_dlogp(vars=[variable], mode=mode)(point)
    # The independently retained exponential tail survives the finite Normal
    # cotangent and the midpoint slope interval derivative (900 - 700) / 4.
    with localcontext() as context:
        context.prec = 100
        x, sigma = (Decimal.from_float(value) for value in (float(np.e), 1e-100))
        log_ratio = x.ln()
        tail = (-Decimal(800) * log_ratio).exp()
        response = 1 / (1 + tail)
        raw_score = -Decimal(cfg.n_time_steps) * response * tail * log_ratio / (1 + tail) ** 2
        expected = float(raw_score / sigma**2 * 50)
    assert expected < 0.0
    np.testing.assert_allclose(actual, [expected], rtol=2e-12, atol=0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_hill_tiny_relative_factors_rescue_unweighted_saturated_derivatives(mode):
    x, reference, slope, kappa = (pt.dscalar(name) for name in ("x", "reference", "slope", "kappa"))
    response = mechanisms.stable_hill_kappa_relative(x, reference, slope=slope, kappa_mult=kappa)
    gradients = pytensor.grad(response, [x, reference, slope, kappa])
    evaluate = pytensor.function([x, reference, slope, kappa], [response, *gradients], mode=mode)
    inputs = (float(np.e * 1e-300), 1.0, 800.0, 1e-300)
    with localcontext() as context:
        context.prec = 100
        value, scale, shape, half = (Decimal.from_float(value) for value in inputs)
        log_ratio = (value / (half * scale)).ln()
        tail = (-(shape * log_ratio)).exp()
        common = tail / (1 + tail) ** 2
        expected = [
            float(common * shape / value),
            float(-common * shape / scale),
            float(common * log_ratio),
            float(-common * shape / half),
        ]
    value, *actual = evaluate(*inputs)
    assert value == 1.0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)
    assert expected[0] > 0.0 and expected[3] < 0.0


def _decimal_unit_response_and_gradients(family, x, reference, shape, weight=1.0):
    """Independent closed forms, retaining weighted products and squares outside float64."""
    with localcontext() as context:
        context.prec = 100
        x, r, s, w = (Decimal.from_float(float(value)) for value in (x, reference, shape, weight))
        if family == "michaelis_menten":
            # A representable lambda is the library's rounded product, even
            # when it is subnormal. An overflowing product needs the full law.
            primitive_lambda = float(reference) * float(shape)
            lam = Decimal.from_float(primitive_lambda) if 0.0 < primitive_lambda < np.inf else r * s
            denominator = x + lam
            return (
                x / denominator,
                w * lam / denominator**2,
                -w * x * s / denominator**2,
                -w * x * r / denominator**2,
            )
        if family == "root":
            if x <= 0:
                return Decimal(0), Decimal(0), Decimal(0), Decimal(0)
            log_ratio = (x / r).ln()
            response = (s * log_ratio).exp()
            return (
                response,
                w * s * response / x,
                -w * s * response / r,
                w * response * log_ratio,
            )
        assert family == "tanh"
        z = x / (r * s)
        if abs(z) > 2000:
            # Even finite float64 weights and the smallest admitted shape
            # cannot rescue this tail into a nonzero float64 derivative.
            return Decimal(1 if z > 0 else -1), Decimal(0), Decimal(0), Decimal(0)
        e = (-2 * z).exp()
        response = (1 - e) / (1 + e)
        sech_squared = 4 * e / (1 + e) ** 2
        return (
            response,
            w * sech_squared / (r * s),
            -w * x * sech_squared / (r**2 * s),
            -w * x * sech_squared / (r * s**2),
        )


@pytest.mark.parametrize("family", ("michaelis_menten", "tanh"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_tiny_shape_responses_and_gradients_match_decimal(family, mode):
    x, r, shape = pt.dvectors("x", "reference", "shape")
    parameter = "kappa_mult" if family == "michaelis_menten" else "c"
    response = mechanisms.STABLE_SATURATION_FAMILIES[family](x, r, **{parameter: shape})
    gradients = pytensor.grad(response.sum(), [x, r, shape])
    evaluate = pytensor.function([x, r, shape], [response, *gradients], mode=mode)

    scale, shape_value = 3.0, 1.5e-180
    inputs = np.array(
        [
            0.0,
            0.25 * scale * shape_value,
            scale * shape_value,
            4 * scale * shape_value,
            400 * scale * shape_value,
            1.0,
            1e308,
        ]
    )
    references = np.full(inputs.size, scale)
    shapes = np.full(inputs.size, shape_value)
    expected = np.array(
        [
            [
                float(value)
                for value in _decimal_unit_response_and_gradients(
                    family, input_value, scale, shape_value
                )
            ]
            for input_value in inputs
        ]
    ).T
    actual = evaluate(inputs, references, shapes)
    for observed, independent in zip(actual, expected):
        np.testing.assert_allclose(observed, independent, rtol=2e-12, atol=0.0)
    assert actual[0][0] == 0.0
    assert actual[2][0] == actual[3][0] == 0.0
    if family == "michaelis_menten":
        # A rounded unit response still has a nonzero shape derivative.
        assert actual[0][-1] == 1.0 and actual[3][-1] < 0.0
    else:
        # At z=400, sech²(z) itself underflows but its scale derivative does not.
        assert actual[0][4] == 1.0 and actual[3][4] < 0.0
        np.testing.assert_array_equal(actual[1][-2:], 0.0)
        np.testing.assert_array_equal(actual[3][-2:], 0.0)


@pytest.mark.parametrize("family", ("michaelis_menten", "tanh", "root"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_relative_curves_preserve_complete_values_and_weighted_gradients(family, mode):
    small = float(np.nextafter(0.0, 1.0))
    cases = {
        "tanh": (
            (0.0, 1e24, small, 1e-30),
            (1e-300, 1e24, small, 1e-30),
            (-1e-300, 1e24, small, 1e-30),
            (small, 1e38, small, 1e-30),
            (1e-300, 1e38, small, 1e-30),
        ),
        "michaelis_menten": (
            (0.0, 1e308, 1.0, 1.0),
            (1e308, 1e308, 1.0, 1.0),
            (1e308, 1e300, 1e8, 1.0),
            (1e308, 1e308, 1e8, 1.0),
            (3 * small, 1e-8, 1.5e-315, 1e-20),
            (1e-300, 1e308, 1e8, 1e308),
        ),
        "root": (
            (-small, 1e38, 0.5, 1.0),
            (0.0, 1e38, 0.5, 1.0),
            (1e-300, 1e24, 0.5, 1.0),
            (small, 1e38, small, 1.0),
            (1e308, 1e-8, 0.5, 1.0),
            (1e-280, 1e38, 0.5, 1.0),
            (small, 1e38, 1.0, 1e38),
        ),
    }[family]
    x, r, shape, weight = pt.dvectors("x", "reference", "shape", "weight")
    parameter = {"michaelis_menten": "kappa_mult", "tanh": "c", "root": "alpha"}[family]
    response = mechanisms.STABLE_SATURATION_FAMILIES[family](x, r, **{parameter: shape})
    gradients = pytensor.grad((weight * response).sum(), [x, r, shape])
    evaluate = pytensor.function([x, r, shape, weight], [response, *gradients], mode=mode)
    expected = np.array(
        [
            [float(value) for value in _decimal_unit_response_and_gradients(family, *case)]
            for case in cases
        ]
    ).T
    actual = evaluate(*np.array(cases).T)
    for observed, independent in zip(actual, expected):
        assert np.isfinite(observed).all()
        np.testing.assert_allclose(observed, independent, rtol=2e-12, atol=2 * small)
        assert np.all(observed[independent != 0.0] != 0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_root_clipping_preserves_unrepresentable_and_weight_rescued_positive_derivatives(mode):
    small = float(np.nextafter(0.0, 1.0))
    x = pt.dvector("x")
    r, alpha, weight = pt.dscalars("reference", "alpha", "weight")
    response = mechanisms.stable_root_kappa_relative(x, r, alpha=alpha)
    unit_derivative = pytensor.grad(response.sum(), x)
    weighted_derivative = pytensor.grad(weight * response.sum(), x)
    evaluate = pytensor.function(
        [x, r, alpha, weight], [response, unit_derivative, weighted_derivative], mode=mode
    )
    expected = _decimal_unit_response_and_gradients("root", small, 1e-8, 0.001)
    weighted = _decimal_unit_response_and_gradients("root", small, 1e-8, 0.001, 1e-20)
    assert expected[1] > Decimal.from_float(np.finfo("float64").max)
    values, unit_gradient, weighted_gradient = evaluate(
        np.array([-small, 0.0, small]), 1e-8, 0.001, 1e-20
    )
    np.testing.assert_allclose(values, [0.0, 0.0, float(expected[0])], rtol=2e-12, atol=0.0)
    np.testing.assert_array_equal(unit_gradient[:2], 0.0)
    assert np.isposinf(unit_gradient[-1])
    assert np.isfinite(weighted_gradient).all()
    np.testing.assert_allclose(
        weighted_gradient, [0.0, 0.0, float(weighted[1])], rtol=2e-12, atol=0.0
    )


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_weighted_mm_relative_pullback_retains_finite_shape_and_reference_derivatives(mode):
    small = float(np.nextafter(0.0, 1.0))
    x, r, kappa = pt.dvector("x"), pt.dscalar("reference"), pt.dscalar("kappa")
    response = mechanisms.stable_michaelis_menten_kappa_relative(x, r, kappa_mult=kappa)
    gradients = pytensor.grad(1e-10 * response.sum(), [r, kappa])
    evaluate = pytensor.function([x, r, kappa], [response, *gradients], mode=mode)
    expected = _decimal_unit_response_and_gradients(
        "michaelis_menten", 3 * small, 1e-8, 1.5e-315, 1e-10
    )
    # d/dx is genuinely unrepresentable at this weight. The complete shape
    # and reference derivatives are finite; no absolute-lambda derivative is.
    actual = evaluate(np.array([3 * small]), 1e-8, 1.5e-315)
    for observed, independent in zip(actual, (expected[0], expected[2], expected[3])):
        assert np.isfinite(observed).all()
        np.testing.assert_allclose(observed, float(independent), rtol=2e-12, atol=0.0)


_LIBRARY_SHAPE = {"michaelis_menten": "kappa_mult", "tanh": "c", "root": "alpha"}


def _library_unit_curve(family, x, reference, shape):
    """pymc-marketing's own transformer on a time column, with main's reference guard."""
    safe_reference = pt.maximum(reference, 1e-8)
    if family == "michaelis_menten":
        curve = _pmm.michaelis_menten(
            as_xtensor(x, dims=("time",)), alpha=1.0, lam=shape * safe_reference
        )
    elif family == "tanh":
        curve = _pmm.tanh_saturation(as_xtensor(x / safe_reference, dims=("time",)), b=1.0, c=shape)
    else:
        ratio = pt.maximum(x / safe_reference, 0.0)
        curve = _pmm.root_saturation(as_xtensor(ratio, dims=("time",)), alpha=shape)
    return curve.values


@pytest.mark.parametrize("family", ("michaelis_menten", "tanh", "root"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_relative_curves_keep_regular_library_values_exact(family, mode):
    # Seeded corpora reproduce the library's rounding, alone and once the
    # generator scales the response (a gate times beta lets the canonicalizer fold
    # the scale into the library quotient). Random inputs separate re-associated
    # quotients and scalar powers that a few round inputs would hide.
    x, r, shape, scale = pt.dvector("x"), *pt.dscalars("reference", "shape", "scale")
    default = mechanisms.SATURATION_FAMILIES[family](x, r, **{_LIBRARY_SHAPE[family]: shape})
    library = _library_unit_curve(family, x, r, shape)
    # Separate functions: one graph could merge the two sides into one node.
    observed = pytensor.function([x, r, shape, scale], [default, scale * default], mode=mode)
    expected = pytensor.function([x, r, shape, scale], [library, scale * library], mode=mode)
    rng = np.random.default_rng(28)
    values = np.concatenate([[0.0], rng.uniform(-1.0, 12.0, 4095)])
    references = np.concatenate([rng.uniform(1e-12, 1e-8, 12), rng.uniform(1e-3, 8.0, 52)])
    for reference, shape_value, scale_value in zip(
        references, rng.uniform(0.2, 2.0, 64), rng.uniform(0.5, 2.0, 64), strict=True
    ):
        got = observed(values, reference, shape_value, scale_value)
        want = expected(values, reference, shape_value, scale_value)
        for context, observed_value, expected_value in zip(("alone", "scaled"), got, want):
            assert observed_value.tobytes() == expected_value.tobytes(), (
                context,
                reference,
                shape_value,
            )


@pytest.mark.parametrize(
    ("family", "field", "input_value", "scale", "shape_value"),
    (
        ("tanh", "tanh_c", 1e-300, 1e24, float(np.nextafter(0.0, 1.0))),
        ("tanh", "tanh_c", float(np.nextafter(0.0, 1.0)), 1e38, float(np.nextafter(0.0, 1.0))),
        ("michaelis_menten", "mm_kappa_mult", 1e308, 1e300, 1e8),
        ("michaelis_menten", "mm_kappa_mult", 1e308, 1e308, 1e8),
        ("root", "root_alpha", 1e-300, 1e24, 0.5),
        ("root", "root_alpha", float(np.nextafter(0.0, 1.0)), 1e38, float(np.nextafter(0.0, 1.0))),
        ("root", "root_alpha", 1e308, 1e-8, 0.5),
    ),
)
@pytest.mark.parametrize("latent", ("marginal", "sampled"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_fixed_input_relative_oracles_preserve_complete_curves(
    family, field, input_value, scale, shape_value, latent, mode
):
    cfg = _cfg(
        n_treatments=1,
        nonlinearity="linear",
        beta_additive_range=(1.0, 1.0),
        saturation_family_probs=_one_hot(family),
        saturation_prior_ranges=_ranges(**{field: (shape_value, 2 * shape_value)}),
        mm_scale_prior="uniform",
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        baseline_floor=None,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index(family)])
    data = _oracle_data(cfg, scale)
    data["treatments"][:] = input_value
    oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
    shape = oracle[field].type(name="runtime_shape")
    contributions = clone_replace(
        oracle["contributions"], replace={oracle[field]: shape}, rebuild_strict=False
    )
    evaluate = pytensor.function([shape], contributions, mode=mode)
    expected = float(
        _decimal_unit_response_and_gradients(family, input_value, scale, shape_value)[0]
    )
    np.testing.assert_allclose(
        evaluate(np.array([shape_value])),
        np.full((cfg.n_time_steps, 1), expected),
        rtol=2e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("latent", ("marginal", "sampled"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_supported_weighted_mm_oracle_shape_pullback_matches_decimal(latent, mode):
    cfg = _cfg(
        n_treatments=1,
        nonlinearity="linear",
        beta_additive_range=(1.0, 1.0),
        saturation_family_probs=_one_hot("michaelis_menten"),
        saturation_prior_ranges=_ranges(mm_kappa_mult=(1e-315, 2e-315)),
        mm_scale_prior="uniform",
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        baseline_floor=None,
    )
    small = float(np.nextafter(0.0, 1.0))
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index("michaelis_menten")])
    data = _oracle_data(cfg, 1e-8)
    data["treatments"][:] = 3 * small
    oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
    kappa = oracle["mm_kappa_mult"].type(name="runtime_kappa")
    contributions = clone_replace(
        oracle["contributions"], replace={oracle["mm_kappa_mult"]: kappa}, rebuild_strict=False
    )
    derivative = pytensor.grad(1e-10 * contributions[0, 0], kappa)
    evaluate = pytensor.function([kappa], [contributions, derivative], mode=mode)
    expected = _decimal_unit_response_and_gradients(
        "michaelis_menten", 3 * small, 1e-8, 1.5e-315, 1e-10
    )
    response, observed = evaluate(np.array([1.5e-315]))
    np.testing.assert_array_equal(response, np.full((cfg.n_time_steps, 1), float(expected[0])))
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, [float(expected[3])], rtol=2e-12, atol=0.0)


@pytest.fixture(scope="module", params=("michaelis_menten", "tanh"))
def tiny_shape_world(request):
    family = request.param
    field = "mm_kappa_mult" if family == "michaelis_menten" else "tanh_c"
    cfg = _cfg(
        n_treatments=1,
        n_time_steps=32,
        nonlinearity="linear",
        saturation_family_probs=_one_hot(family),
        saturation_prior_ranges=_ranges(**{field: (1e-180, 2e-180)}),
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        treatment_onset_inclusion_prob=1.0,
        treatment_onset_frac_range=(0.125, 0.125),
        beta_additive_range=(1.0, 1.0),
        outcome_std_mode="relative",
        rw_baseline_mean_range=(0.0, 0.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_outcome_std_range=(1.0, 1.0),
        baseline_floor=None,
    )
    g = _graph(1, 1)
    g["g_zy"][:] = 0
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index(family)])
    generative, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    names = ("treatments", "covariates", "outcome", "saturation_scale")
    drawn = draw_worlds(generative, names, seed=17)
    data = {name: drawn[name][0] for name in names}
    np.testing.assert_array_equal(data["treatments"][:4, 0], 0.0)
    assert np.all(data["treatments"][4:, 0] > 0.0)
    return family, field, cfg, g, structural, data


@pytest.mark.parametrize("latent", ("marginal", "sampled"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_tiny_shape_oracle_posterior_gradients_include_onset_zeros(tiny_shape_world, latent, mode):
    family, field, cfg, g, structural, data = tiny_shape_world
    oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
    shape = oracle[field]
    point = oracle.initial_point()
    point[oracle.rvs_to_values[shape].name] = np.array([0.0])
    assert np.isfinite(oracle.compile_logp(mode=mode)(point))
    actual = oracle.compile_dlogp(vars=[shape], mode=mode)(point)

    # At this interval midpoint the prior/Jacobian derivative is zero. With
    # fixed beta=1, baseline=0 and outcome sigma=1, the likelihood pullback is
    # sum((y - f) * df/dshape), followed by the interval derivative (hi-lo)/4.
    with localcontext() as context:
        context.prec = 100
        lo, hi = (Decimal.from_float(value) for value in (1e-180, 2e-180))
        shape_value = float((lo + hi) / 2)
        likelihood_gradient = Decimal(0)
        for x, y in zip(data["treatments"][:, 0], data["outcome"]):
            response, _, _, derivative = _decimal_unit_response_and_gradients(
                family, x, data["saturation_scale"][0], shape_value
            )
            likelihood_gradient += (Decimal.from_float(float(y)) - response) * derivative
        expected = float(likelihood_gradient * (hi - lo) / 4)
    # The marginal factorization's negligible covariance guard changes this
    # sigma=1 likelihood by ~1e-12, not the mechanism's analytic derivative.
    np.testing.assert_allclose(actual, [expected], rtol=2e-10, atol=0.0)
    if family == "tanh":
        np.testing.assert_array_equal(actual, 0.0)
    else:
        assert expected != 0.0


def _decimal_logistic_response_and_input_derivative(x, reference, lam):
    with localcontext() as context:
        context.prec = 100
        x, r, lam = (Decimal.from_float(float(value)) for value in (x, reference, lam))
        e = (-(lam * x / r)).exp()
        return float((1 - e) / (1 + e)), float(2 * lam * e / (r * (1 + e) ** 2))


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_stable_logistic_overflowing_input_ratio_matches_decimal(mode):
    x, r, lam = pt.dvector("x"), pt.dscalar("reference"), pt.dscalar("lambda")
    response = mechanisms.stable_logistic_kappa_relative(x, r, lam=lam)
    derivative = pytensor.grad(response.sum(), x)
    evaluate = pytensor.function([x, r, lam], [response, derivative], mode=mode)
    inputs = np.array([-1e308, 0.0, 1e308])
    scale, shape = 1e-8, np.nextafter(0.0, 1.0)
    expected = np.array(
        [_decimal_logistic_response_and_input_derivative(value, scale, shape) for value in inputs]
    ).T
    values, derivatives = evaluate(inputs, scale, shape)
    np.testing.assert_allclose(values, expected[0], rtol=32 * _EPS, atol=0.0)
    np.testing.assert_allclose(derivatives, expected[1], rtol=32 * _EPS, atol=2 * shape)


@pytest.mark.parametrize("latent", ("marginal", "sampled"))
@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_logistic_oracle_retains_finite_response_when_input_ratio_overflows(latent, mode):
    small = np.nextafter(0.0, 1.0)
    cfg = _cfg(
        n_treatments=1,
        nonlinearity="linear",
        beta_additive_range=(1.0, 1.0),
        saturation_family_probs=_one_hot("logistic"),
        saturation_prior_ranges=_ranges(logistic_lam=(small, 2 * small)),
        baseline_floor=None,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index("logistic")])
    data = _oracle_data(cfg, 1e-8)
    data["treatments"][:] = 1e308
    oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
    lam = oracle["logistic_lam"].type(name="lambda_value")
    contributions = clone_replace(
        oracle["contributions"], replace={oracle["logistic_lam"]: lam}, rebuild_strict=False
    )
    evaluate = pytensor.function([lam], contributions, mode=mode)
    expected, _ = _decimal_logistic_response_and_input_derivative(1e308, 1e-8, small)
    np.testing.assert_allclose(
        evaluate(np.array([small])),
        np.full((cfg.n_time_steps, 1), expected),
        rtol=32 * _EPS,
        atol=0.0,
    )


def test_minimum_positive_logistic_lambda_reference_target_survives_all_shared_paths():
    lam, multiplier, target = np.nextafter(0.0, 1.0), 1e38, 1e-290
    cfg = _cfg(
        n_treatments=1,
        saturation_prior_ranges=_ranges(logistic_lam=(lam, lam)),
        saturation_family_probs=_one_hot("logistic"),
        treatment_reference_contribution_range=(target, target),
        treatment_reference_multiplier=multiplier,
    )
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index("logistic")])
    names = (
        "param_beta",
        "param_logistic_lam",
        "param_treatment_reference_contribution",
        "param_treatment_reference_input",
        "param_treatment_reference_response",
        "saturation_scale",
    )
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    generated = draw_worlds(model, names, seed=11, draws=2)
    active = {
        "active_treatment": np.ones(1),
        "active_covariate": np.ones(1),
        "active_latent": np.ones(1),
    }
    cell = build_cell_inputs(cfg, g, active, structural)
    template, _, _ = build_world_model_template(cfg, cell, cfg.n_time_steps)
    templated = compile_template_draw_fn(template, names)(cell, seed=11, draws=2)
    for drawn in (generated, templated):
        np.testing.assert_array_equal(drawn["param_logistic_lam"], lam)
        np.testing.assert_array_equal(drawn["param_treatment_reference_contribution"], target)
        np.testing.assert_array_equal(
            drawn["param_treatment_reference_input"], multiplier * drawn["saturation_scale"]
        )
        ratio = drawn["param_treatment_reference_input"] / drawn["saturation_scale"]
        expected_response = np.tanh((lam * ratio) / 2.0)
        beta = drawn["param_beta"]
        assert np.all(np.isfinite(beta) & (beta > 0.0) & (beta <= CORPUS_STORAGE_MAX))
        np.testing.assert_allclose(
            drawn["param_treatment_reference_response"],
            expected_response,
            rtol=8.0 * _EPS,
            atol=0.0,
        )
        np.testing.assert_allclose(beta, target / expected_response, rtol=8.0 * _EPS, atol=0.0)
        np.testing.assert_allclose(
            beta * drawn["param_treatment_reference_response"], target, rtol=8.0 * _EPS, atol=0.0
        )
    reference = generated["param_treatment_reference_input"][0]
    factors = np.tile([0.0, 0.25, 1.0, 2.0], cfg.n_time_steps // 4)
    data = {
        "treatments": factors[:, None] * reference,
        "covariates": np.zeros((cfg.n_time_steps, 1)),
        "outcome": np.zeros(cfg.n_time_steps),
        "saturation_scale": generated["saturation_scale"][0],
    }
    oracle = build_oracle_model(g, cfg, structural, data)
    with oracle:
        beta, contributions = pm.draw(
            [oracle["beta"], oracle["contributions"]], random_seed=29, mode="FAST_COMPILE"
        )
    np.testing.assert_allclose(beta, generated["param_beta"][0], rtol=8.0 * _EPS, atol=0.0)
    expected = generated["param_beta"][0] * np.tanh(
        (lam * (data["treatments"] / data["saturation_scale"])) / 2.0
    )
    np.testing.assert_allclose(contributions, expected, rtol=16.0 * _EPS, atol=0.0)
    np.testing.assert_allclose(contributions[2::4, 0], target, rtol=8.0 * _EPS, atol=0.0)


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
def test_stable_hill_half_points_tails_and_gradients_survive_graph_rewrites(mode):
    # Symbolic inputs keep PyTensor from constant-folding the response before
    # the rewrites that previously moved steep half-points.
    x, r = pt.dvector("x"), pt.dscalar("r")
    slope, kappa = pt.dvector("slope"), pt.dvector("kappa")
    params = {
        "hill_slope": slope,
        "hill_kappa_mult": kappa,
        "sat_family": np.array([1]),
        "mechanism_priors_enabled": True,
    }
    response = _saturate_col(x, r, params, 0)
    gradients = pytensor.grad(response.sum(), [x, slope, kappa])
    evaluate = pytensor.function([x, r, slope, kappa], [response, *gradients], mode=mode)
    for kappa_mult in (2.0, 0.7):
        for scale in (1.0, 3.0, 7.0, 1e20, 1e-7):
            half, half_dx, *_ = evaluate(
                np.array([kappa_mult * scale]), scale, np.array([1e16]), np.array([kappa_mult])
            )
            assert half[0] == 0.5, (kappa_mult, scale, half)
            # df/dx = s f (1 - f) / x = s / (4 x) at the half-point.
            np.testing.assert_allclose(half_dx, 1e16 / (4.0 * kappa_mult * scale), rtol=1e-12)
    inputs = np.array([0.0, 1e-300, 0.5, 6.0, 1e300])
    value, *derivatives = evaluate(inputs, 3.0, np.array([1.001]), np.array([2.0]))
    expected = _numpy_response(
        "hill", inputs / 3.0, {"hill_slope": [1.001], "hill_kappa_mult": [2.0]}, 0
    )
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0.0)
    assert value[0] == 0.0 and value[1] > 0.0
    _assert_hill_gradients(derivatives, inputs, 3.0, 1.001, 2.0, expected)
    # Ratio above the float64 maximum: the separate-logarithm far tail.
    value, *derivatives = evaluate(np.array([1.0]), 1.0, np.array([0.001]), np.array([1e-310]))
    expected = _numpy_response(
        "hill", np.array([1.0]), {"hill_slope": [0.001], "hill_kappa_mult": [1e-310]}, 0
    )
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0.0)
    _assert_hill_gradients(derivatives, np.array([1.0]), 1.0, 0.001, 1e-310, expected)


def _assert_hill_gradients(derivatives, x, scale, slope, kappa_mult, f):
    """Closed-form d/dx, d/dslope and d/dkappa of sigmoid(slope * log(x / (kappa r)))."""
    positive = x > 0.0
    local = np.where(positive, slope * f * (1.0 - f), 0.0)
    log_ratio = np.log(np.where(positive, x, 1.0)) - np.log(scale) - np.log(kappa_mult)
    expected = (
        np.where(positive, local / np.where(positive, x, 1.0), 0.0),
        [np.sum(np.where(positive, f * (1.0 - f) * log_ratio, 0.0))],
        [np.sum(-local / kappa_mult)],
    )
    for actual, wanted in zip(derivatives, expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=0.0)


def test_shape_only_opt_in_uses_stable_curves_in_world_template_and_oracle():
    # The legacy curves lose ~1e-7 (Hill far below kappa) and ~1e-10 (logistic
    # at tiny lam) to cancellation here; the stable forms keep closed-form accuracy.
    cfg = _cfg(
        saturation_prior_ranges=_ranges(
            hill_slope=(3.0, 3.0), hill_kappa_mult=(1e3, 1e3), logistic_lam=(1e-6, 1e-6)
        )
    )
    assert cfg.treatment_reference_contribution_range is None
    g = _graph(2, 1)
    structural = _structure(g, cfg, [1, 2])
    model, outputs, reports = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = _draw(model, outputs, reports)
    params = _params(drawn, 0, structural, reports, cfg)
    expected = _numpy_contributions(
        drawn["treatments"][0], drawn["saturation_scale"][0], params, g["g_cy"]
    )
    np.testing.assert_allclose(drawn["contributions_observed"][0], expected, rtol=2e-12, atol=0.0)
    values = {name: value[0] for name, value in drawn.items()}
    data = {
        name: drawn[name][0] for name in ("treatments", "covariates", "outcome", "saturation_scale")
    }
    for latent in ("marginal", "sampled"):
        oracle = build_oracle_model(g, cfg, structural, data, latent=latent)
        inferred = _evaluate_given(oracle, values, ("contributions",))["contributions"]
        np.testing.assert_allclose(inferred, expected, rtol=2e-12, atol=0.0)
        point = _point(oracle, values)
        assert np.isfinite(oracle.compile_logp(mode="FAST_COMPILE")(point))
        assert np.isfinite(oracle.compile_dlogp(mode="FAST_COMPILE")(point)).all()
    active = {
        "active_treatment": np.ones(2),
        "active_covariate": np.ones(1),
        "active_latent": np.ones(1),
    }
    cell = build_cell_inputs(cfg, g, active, structural)
    template, t_outputs, t_reports = build_world_model_template(cfg, cell, cfg.n_time_steps)
    t_drawn = compile_template_draw_fn(template, (*t_outputs, *t_reports))(cell, seed=3)
    t_params = {name.removeprefix("param_"): t_drawn[name][0] for name in t_reports}
    t_params.update(
        l_max=cfg.l_max, sat_family=cell["sat_family"], carryover_family=cell["carryover_family"]
    )
    expected = _numpy_contributions(
        t_drawn["treatments"][0], t_drawn["saturation_scale"][0], t_params, cell["g_cy"]
    )
    np.testing.assert_allclose(t_drawn["contributions_observed"][0], expected, rtol=2e-12, atol=0.0)


@pytest.mark.parametrize(
    ("family", "shapes", "inputs", "reference"),
    (
        # Cancellation far below kappa / at tiny lam.
        ("hill", {"hill_slope": 3.0, "hill_kappa_mult": 1e3}, [0.0, 0.5, 1.0, 2.0], 1.0),
        ("logistic", {"logistic_lam": 1e-6}, [0.0, 0.5, 1.0, 2.0], 1.0),
        # lambda + x overflows the library denominator: 1e308 / inf = 0, not 0.5.
        ("michaelis_menten", {"mm_kappa_mult": 1.0}, [0.0, 1.0, 1e308], 1e308),
        # reference * c overflows the library quotient: tanh(x / inf) = 0.
        ("tanh", {"tanh_c": 1e10}, [0.0, 1e300], 1e300),
        # x / reference underflows to 0 before the library power.
        ("root", {"root_alpha": 0.5}, [0.0, 1e-300, 2.0], 1e24),
    ),
    ids=("hill", "logistic", "michaelis_menten", "tanh", "root"),
)
@pytest.mark.parametrize("dynamic_family", (False, True), ids=("concrete", "switch"))
def test_unflagged_saturation_keeps_legacy_curves_bit_for_bit(
    family, shapes, inputs, reference, dynamic_family
):
    # Defaults keep the library arithmetic and opt-in mechanism priors select the
    # stable form, on the ordinary graph's concrete family and on the template's
    # switch over every family. Inputs, shapes and (for the switch) the family id
    # are symbolic, as in the generator, so constant folding cannot choose the
    # rounding; at these inputs the two forms differ, so neither equality holds
    # vacuously.
    x, r = pt.dvector("x"), pt.dscalar("reference")
    shape_inputs = {field: pt.dvector(field) for field in SHAPE_FIELDS}
    shape_values = {field: np.array([shapes.get(field, 1.0)]) for field in SHAPE_FIELDS}
    family_id = SATURATION_FAMILY_KEYS.index(family)
    sat_family = pt.lvector("sat_family") if dynamic_family else np.array([family_id])
    params = {"sat_family": sat_family, **shape_inputs}
    kwargs = {SHAPE_FIELDS[field][1]: shape_inputs[field][0] for field in shapes}
    outputs = (
        _saturate_col(x, r, params, 0, dynamic_family=dynamic_family),
        _saturate_col(
            x, r, {**params, "mechanism_priors_enabled": True}, 0, dynamic_family=dynamic_family
        ),
        mechanisms.SATURATION_FAMILIES[family](x, r, **kwargs),
        mechanisms.STABLE_SATURATION_FAMILIES[family](x, r, **kwargs),
    )
    symbolic = [x, r, *shape_inputs.values()]
    values = [np.asarray(inputs), reference, *shape_values.values()]
    if dynamic_family:
        symbolic.append(sat_family)
        values.append(np.array([family_id]))
    with np.errstate(all="ignore"):
        unflagged, flagged, legacy, stable = (
            pytensor.function(symbolic, output, mode="FAST_COMPILE", on_unused_input="ignore")(
                *values
            )
            for output in outputs
        )
    assert unflagged.tobytes() == legacy.tobytes()
    assert flagged.tobytes() == stable.tobytes()
    assert not np.array_equal(stable, legacy)


@pytest.mark.parametrize(
    ("options", "fields"),
    (
        (
            {
                "treatment_reference_contribution_range": (1.0, 1.0),
                "treatment_reference_multiplier": 1e-20,
            },
            ("treatment_reference_contribution_range", "treatment_reference_multiplier", "hill"),
        ),
        (
            {
                "treatment_reference_contribution_range": (1.0, 1.0),
                "treatment_reference_multiplier": 1e-200,
            },
            ("treatment_reference_contribution_range", "treatment_reference_multiplier", "hill"),
        ),
        (
            {
                "treatment_reference_contribution_range": (1e38, 1e38),
                "treatment_reference_multiplier": 0.1,
            },
            ("treatment_reference_contribution_range", "hill"),
        ),
        (
            {
                "treatment_reference_contribution_range": (
                    0.9 * CORPUS_STORAGE_MAX,
                    0.9 * CORPUS_STORAGE_MAX,
                ),
                "treatment_reference_multiplier": 2.0,
                "saturation_prior_ranges": _ranges(
                    hill_slope=(0.01, 1000.0), hill_kappa_mult=(1.0, 1.0)
                ),
            },
            ("treatment_reference_contribution_range", "hill"),
        ),
        (
            {
                "treatment_reference_contribution_range": (
                    0.9 * CORPUS_STORAGE_MAX,
                    0.9 * CORPUS_STORAGE_MAX,
                ),
                "treatment_reference_multiplier": 0.5,
                "saturation_prior_ranges": _ranges(root_alpha=(0.01, 1.0)),
                "saturation_family_probs": _one_hot("root"),
            },
            ("treatment_reference_contribution_range", "root"),
        ),
        (
            {
                "covariate_reference_contribution_range": (-1.0, 2.0),
                "covariate_reference_scale": 1e-50,
            },
            ("covariate_reference_contribution_range", "covariate_reference_scale"),
        ),
        (
            {
                "covariate_reference_contribution_range": (1e-310, 1e-310),
                "covariate_reference_scale": 1e20,
            },
            ("covariate_reference_contribution_range", "covariate_reference_scale"),
        ),
        (
            {
                "treatment_reference_contribution_range": (1e-320, 1e-320),
                "treatment_reference_multiplier": 1e38,
                "saturation_family_probs": _one_hot("linear"),
            },
            (
                "treatment_reference_contribution_range",
                "treatment_reference_multiplier",
                "saturation_family_probs['linear']",
            ),
        ),
        (
            {
                "treatment_reference_contribution_range": (1e-320, 1e-320),
                "treatment_reference_multiplier": 1e38,
                "saturation_family_probs": _one_hot("root"),
            },
            ("treatment_reference_contribution_range", "treatment_reference_multiplier", "root"),
        ),
        (
            {
                "treatment_reference_contribution_range": (1.0, 1.0),
                "treatment_reference_multiplier": 1e-320,
                "saturation_prior_ranges": _ranges(root_alpha=(0.001, 0.001)),
                "saturation_family_probs": _one_hot("root"),
            },
            ("treatment_reference_multiplier", "treatment_reference_input"),
        ),
        (
            # Zero-inclusive targets: interior draws far below the endpoint underflow.
            {
                "treatment_reference_contribution_range": (0.0, 1e-310),
                "treatment_reference_multiplier": 1e13,
                "saturation_family_probs": _one_hot("linear"),
            },
            ("treatment_reference_contribution_range", "linear", "underflows"),
        ),
        (
            {
                "covariate_reference_contribution_range": (-1e-310, 1e-310),
                "covariate_reference_scale": 1e13,
            },
            ("covariate_reference_contribution_range", "underflows"),
        ),
        (
            # One ulp above the float32 storage maximum, invisible in log space.
            {
                "covariate_reference_contribution_range": (CORPUS_STORAGE_MAX, CORPUS_STORAGE_MAX),
                "covariate_reference_scale": 0.9999999999999999,
            },
            ("covariate_reference_contribution_range", "storage maximum"),
        ),
        (
            {
                "treatment_reference_contribution_range": (CORPUS_STORAGE_MAX, CORPUS_STORAGE_MAX),
                "treatment_reference_multiplier": 0.9999999999999999,
                "saturation_family_probs": _one_hot("linear"),
            },
            (
                "treatment_reference_contribution_range",
                "saturation_family_probs",
                "storage maximum",
            ),
        ),
        # Each family's bound must use the end of its shape range that minimizes
        # the reference response; the other end would admit these targets.
        (
            {
                "treatment_reference_contribution_range": (0.3 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(mm_kappa_mult=(1.0, 3.0)),
                "saturation_family_probs": _one_hot("michaelis_menten"),
            },
            ("treatment_reference_contribution_range", "michaelis_menten", "storage maximum"),
        ),
        (
            {
                "treatment_reference_contribution_range": (0.6 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(tanh_c=(0.5, 2.0)),
                "saturation_family_probs": _one_hot("tanh"),
            },
            ("treatment_reference_contribution_range", "tanh", "storage maximum"),
        ),
        (
            {
                "treatment_reference_contribution_range": (0.5 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(logistic_lam=(0.5, 3.0)),
                "saturation_family_probs": _one_hot("logistic"),
            },
            ("treatment_reference_contribution_range", "logistic", "storage maximum"),
        ),
        (
            {
                "treatment_reference_contribution_range": (0.3 * CORPUS_STORAGE_MAX,) * 2,
                "treatment_reference_multiplier": 2.0,
                "saturation_prior_ranges": _ranges(
                    hill_slope=(2.0, 2.0), hill_kappa_mult=(1.0, 4.0)
                ),
            },
            ("treatment_reference_contribution_range", "hill", "storage maximum"),
        ),
        (
            # For multipliers >= 1 the largest root response uses the largest alpha.
            {
                "treatment_reference_contribution_range": (1e-320, 1e-320),
                "treatment_reference_multiplier": 1e10,
                "saturation_prior_ranges": _ranges(root_alpha=(0.3, 0.9)),
                "saturation_family_probs": _one_hot("root"),
            },
            ("treatment_reference_contribution_range", "root", "underflows"),
        ),
        (
            # A slope of 1e15 turns a few ulps of runtime ratio rounding into ~20%
            # of response; the Hill margin scales with the slope.
            {
                "treatment_reference_contribution_range": (
                    CORPUS_STORAGE_MAX
                    * np.exp(-np.logaddexp(0.0, -1e15 * np.log(np.exp(-2e-13))))
                    / (1.0 + 8.0 * _EPS)
                    * (1.0 - 1e-6),
                )
                * 2,
                "treatment_reference_multiplier": 0.7 * np.exp(-2e-13),
                "saturation_prior_ranges": _ranges(
                    hill_slope=(1e15, 1e15), hill_kappa_mult=(0.7, 0.7)
                ),
            },
            ("treatment_reference_contribution_range", "hill", "storage maximum"),
        ),
        (
            # Subnormal reference inputs lose relative precision.
            {
                "treatment_reference_contribution_range": (1.0, 1.0),
                "treatment_reference_multiplier": 1e-305,
                "saturation_family_probs": _one_hot("linear"),
            },
            ("treatment_reference_multiplier", "treatment_reference_input", "normal"),
        ),
        (
            # Generation multiplies by fl(1 / scale), which rounds this above the maximum.
            {
                "covariate_reference_contribution_range": (
                    np.nextafter(np.nextafter(CORPUS_STORAGE_MAX, 0.0), 0.0),
                    np.nextafter(CORPUS_STORAGE_MAX, 0.0),
                ),
                "covariate_reference_scale": 0.9999999999999999,
            },
            ("covariate_reference_contribution_range", "storage maximum"),
        ),
        (
            # A subnormal scale has an infinite reciprocal.
            {
                "covariate_reference_contribution_range": (1e-320, 2e-320),
                "covariate_reference_scale": 1e-310,
            },
            ("covariate_reference_contribution_range", "storage maximum"),
        ),
        (
            # A subnormal lower response leaves too few bits for any rounding margin.
            {
                "treatment_reference_contribution_range": (3e-285, 3e-285),
                "treatment_reference_multiplier": 1.5,
                "saturation_prior_ranges": _ranges(logistic_lam=(2 * 5e-324, 2 * 5e-324)),
                "saturation_family_probs": _one_hot("logistic"),
            },
            ("treatment_reference_contribution_range", "logistic", "unrepresentable"),
        ),
    ),
)
@pytest.mark.filterwarnings("error")
def test_impossible_reference_coefficients_fail_instead_of_clamping_or_resampling(options, fields):
    with pytest.raises(ValueError) as error:
        _cfg(**{"saturation_family_probs": _one_hot("hill"), **options})
    for field in fields:
        assert field in str(error.value)


@pytest.mark.parametrize(
    ("family", "options"),
    (
        (
            "michaelis_menten",
            {
                "treatment_reference_contribution_range": (0.24 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(mm_kappa_mult=(1.0, 3.0)),
            },
        ),
        (
            "tanh",
            {
                "treatment_reference_contribution_range": (0.45 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(tanh_c=(0.5, 2.0)),
            },
        ),
        (
            "logistic",
            {
                "treatment_reference_contribution_range": (0.24 * CORPUS_STORAGE_MAX,) * 2,
                "saturation_prior_ranges": _ranges(logistic_lam=(0.5, 3.0)),
            },
        ),
        (
            "hill",
            {
                "treatment_reference_contribution_range": (0.19 * CORPUS_STORAGE_MAX,) * 2,
                "treatment_reference_multiplier": 2.0,
                "saturation_prior_ranges": _ranges(
                    hill_slope=(2.0, 2.0), hill_kappa_mult=(1.0, 4.0)
                ),
            },
        ),
        (
            "root",
            {
                "treatment_reference_contribution_range": (1e-310, 1e-310),
                "treatment_reference_multiplier": 1e10,
                "saturation_prior_ranges": _ranges(root_alpha=(0.3, 0.9)),
            },
        ),
        (
            # Draws are -1e-300 or at least 2**-53; none cancels to a tinier value.
            "linear",
            {
                "covariate_reference_contribution_range": (-1e-300, 1.0),
                "covariate_reference_scale": 1e10,
            },
        ),
    ),
)
def test_reference_supports_just_inside_numeric_limits_validate_and_draw(family, options):
    cfg = _cfg(n_treatments=1, saturation_family_probs=_one_hot(family), **options)
    g = _graph(1, 1)
    structural = _structure(g, cfg, [SATURATION_FAMILY_KEYS.index(family)])
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, ("param_beta", "param_rho_zy"), seed=11, draws=64)
    for name in ("param_beta", "param_rho_zy"):
        values = drawn[name]
        assert np.isfinite(values).all() and np.all(np.abs(values) <= CORPUS_STORAGE_MAX)
    assert np.all(drawn["param_beta"] != 0.0) and np.all(drawn["param_rho_zy"] != 0.0)


def test_runtime_check_rejects_nonzero_targets_that_derive_zero_loadings():
    cfg = _cfg(covariate_reference_contribution_range=(1.0, 1.0))
    # Bypass configuration validation to reach the runtime backstop directly.
    cfg.covariate_reference_contribution_range = (1e-310, 2e-310)
    cfg.covariate_reference_scale = 1e20
    g = _graph(2, 1)
    model, _, _ = build_world_model(g, cfg, _structure(g, cfg, [0, 0]), cfg.n_time_steps)
    with pytest.raises(ValueError, match="preserve nonzero targets"):
        draw_worlds(model, ("param_rho_zy",), seed=13, draws=4)


def test_runtime_reference_failures_propagate_from_corpora_instead_of_resampling(monkeypatch):
    # Validation leaves only rounding-edge draws for the runtime check; inject one
    # deterministically by tightening the runtime storage bound.
    monkeypatch.setattr(world_model, "CORPUS_STORAGE_MAX", 1e-30)
    cfg = _corpus_cfg(treatment_reference_contribution_range=(0.7, 1.1))
    match = "treatment_reference_contribution_range requires finite float64 coefficients"
    with pytest.raises(ValueError, match=match):
        pg.sample_prior_predictive(cfg)
    with pytest.raises(ValueError, match=match):
        pg.sample_scm(cfg, seed=0, max_eps_draws=8)


@pytest.mark.parametrize("target", (0.0, -CORPUS_STORAGE_MAX))
def test_signed_control_zero_and_storage_boundary_are_exact_not_clamped(target):
    cfg = _cfg(
        covariate_reference_contribution_range=(target, target), covariate_reference_scale=1.0
    )
    g = _graph(2, 1)
    structural = _structure(g, cfg, [0, 0])
    model, _, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, ("param_rho_zy",), seed=13)
    np.testing.assert_array_equal(drawn["param_rho_zy"], [[target]])
    np.testing.assert_array_equal(drawn["param_rho_zy"].astype(np.float32), [[np.float32(target)]])


def test_control_reference_is_nominal_before_absorbing_floor_and_edge_gate():
    cfg = _cfg(
        n_treatments=1,
        n_covariates=2,
        baseline_floor=0.0,
        baseline_floor_scope="non_treatment",
        rw_baseline_mean_range=(1.0, 1.0),
        rw_baseline_std_range=(0.0, 0.0),
        rw_covariate_mean_range=(2.0, 2.0),
        treatment_reference_contribution_range=(1.5, 1.5),
        treatment_reference_multiplier=0.4,
        covariate_reference_contribution_range=(-3.0, -3.0),
        covariate_reference_scale=2.0,
    )
    g = _graph(1, 2)
    g["g_zy"][1] = 0
    structural = _structure(g, cfg, [0])
    model, _, reports = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(model, reports, seed=13)
    params = _params(drawn, 0, structural, reports, cfg)
    params["rw_z"]["std"] = np.zeros(2)
    eps = {
        "eps_d": np.zeros((16, 1)),
        "eps_z": np.zeros((16, 2)),
        "eps_c": np.zeros((16, 1)),
        "eps_c_hf": np.zeros((16, 1)),
        "eps_c_pulse": np.zeros((16, 1), dtype="int8"),
        "eps_z_hf": np.zeros((16, 2)),
        "eps_z_pulse": np.zeros((16, 2), dtype="int8"),
        "eps_b": np.zeros(16),
        "eps_y": np.zeros(16),
    }
    graph = build_symbolic_graph(g, params, 16, 1, 2, 1, eps=eps)["outputs"]
    values = {name: value.eval() for name, value in graph.items()}
    nominal = params["rho_zy"] * cfg.covariate_reference_scale
    np.testing.assert_array_equal(nominal, [-3.0, -3.0])
    np.testing.assert_array_equal(values["covariates"], np.full((16, 2), 2.0))
    np.testing.assert_array_equal(values["covariate_contribution"][:, 0], -1.0)
    np.testing.assert_array_equal(values["covariate_contribution"][:, 1], 0.0)
    np.testing.assert_array_equal(values["baseline_intrinsic"], 1.0)
    np.testing.assert_array_equal(values["baseline"], 0.0)
    np.testing.assert_allclose(values["outcome"], values["contributions"].sum(axis=1))


# -- Persisted opt-in provenance and a sampled world's usable reference audit ----


@pytest.mark.parametrize("scheduled", (False, True), ids=("natural", "trajectory"))
def test_opt_in_corpus_metadata_and_additive_truth_survive_save_load(tmp_path, scheduled):
    options = (
        {
            "treatment_trend_inclusion_prob": 1.0,
            "treatment_trend_log_change_range": (0.3, 0.3),
        }
        if scheduled
        else {}
    )
    cfg = _corpus_cfg(
        saturation_prior_ranges=_ranges(mm_kappa_mult=(0.2, 5.0)),
        saturation_family_probs=_one_hot("michaelis_menten"),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(0.7, 1.1),
        treatment_reference_multiplier=1.6,
        covariate_reference_contribution_range=(-0.4, 0.6),
        covariate_reference_scale=1.75,
        rw_baseline_mean_range=(10.0, 12.0),
        edge_budget={"cy": 2, "zy": 1, "dc": 0, "dz": 0, "dy": 0, "zc": 0, "cc": 0, "zz": 0},
        **options,
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert pg.DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "mechanism-priors.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert loaded["diagnostics"]["mechanism_priors"] == corpus["diagnostics"]["mechanism_priors"]
    assert pg.DataGenerator.validate_corpus(loaded) == []
    for name, value in corpus.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[name], value)
    reconstructed = (
        loaded["baseline_intrinsic"]
        + loaded["outcome_noise"]
        + loaded["covariate_contribution"].sum(axis=2)
        + loaded["latent_unobserved_contribution"].sum(axis=2)
        + loaded["treatment_contribution_raw"].sum(axis=2)
        + loaded["indirect_effects_by_source"].sum(axis=2)
    )
    np.testing.assert_allclose(reconstructed, loaded["outcome_raw"], rtol=2e-6, atol=2e-6)
    if scheduled:
        assert np.ptp(loaded["treatment_log_level_shift"], axis=1).min() > 0.0


def test_unused_reference_scalars_keep_legacy_corpora_and_shape_only_settings_record_provenance():
    legacy = pg.sample_prior_predictive(_corpus_cfg())
    assert "mechanism_priors" not in legacy["diagnostics"]
    unused = pg.sample_prior_predictive(
        _corpus_cfg(treatment_reference_multiplier=2.5, covariate_reference_scale=4.0)
    )
    assert "mechanism_priors" not in unused["diagnostics"]
    for name, value in legacy.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(unused[name], value, err_msg=name)
    for options, slope, law in (
        ({"saturation_prior_ranges": _ranges(hill_slope=(1.0, 3.5))}, [1.0, 3.5], "uniform"),
        ({"mm_scale_prior": "log_uniform"}, [1.0, 3.0], "log_uniform"),
    ):
        corpus = pg.sample_prior_predictive(_corpus_cfg(**options))
        metadata = corpus["diagnostics"]["mechanism_priors"]
        assert metadata["saturation_prior_ranges"]["hill"]["slope"] == slope
        assert metadata["mm_scale_prior"] == law
        assert metadata["treatment_reference_contribution_range"] is None
        assert metadata["covariate_reference_contribution_range"] is None


def test_sampled_reference_audit_description_bundle_and_replay_are_consistent(
    tmp_path, monkeypatch
):
    cfg = _cfg(
        saturation_prior_ranges=_ranges(
            hill_slope=np.array([2.345, 2.345], dtype=np.float32),
            hill_kappa_mult=(2.59e-4, 2.59e-4),
            mm_kappa_mult=(0.2, 5.0),
        ),
        saturation_family_probs=_one_hot("hill"),
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(1e-4, 5e-4),
        treatment_reference_multiplier=1.6,
        covariate_reference_contribution_range=(1e-4, 2e-4),
        covariate_reference_scale=3.0,
        edge_budget={
            "cy": (2, 2),
            "zy": (1, 1),
            **dict.fromkeys(("dc", "dz", "dy", "zc", "cc", "zz"), 0),
        },
    )
    world = pg.sample_scm(cfg, seed=61, max_eps_draws=8)
    audit = world.equation_parameters
    for k in range(2):
        reference = audit[f"C{k + 1}"]["response"]["reference"]
        response = _numpy_response(
            "hill",
            np.array([reference["input"] / world.data["saturation_scale"][k]]),
            world.params,
            k,
        )[0]
        np.testing.assert_allclose(
            world.params["beta"][k] * response, reference["contribution"], rtol=1e-12
        )
        assert reference["stage"] == "post_carryover_pre_gate"
        assert reference["multiplier"] == 1.6
        assert reference["input"] == 1.6 * world.data["saturation_scale"][k]
    # Opt-in descriptions keep the drawn shapes and coefficients auditable.
    printed = re.findall(
        r"saturation=hill\(slope=([^,]+),kappa_mult=([^)]+)\)", pg.describe_scm(world)
    )
    np.testing.assert_allclose(
        np.array(printed, dtype=float),
        np.column_stack([world.params["hill_slope"], world.params["hill_kappa_mult"]]),
        rtol=5e-4,
    )
    printed_beta = re.findall(r"beta=([^\s]+)", pg.describe_scm(world))
    np.testing.assert_allclose(np.array(printed_beta, dtype=float), world.params["beta"], rtol=5e-4)
    # The prior block states the effective configured settings.
    description = pg.describe_scm(world)
    ranges = re.search(r"saturation prior ranges: (\{.*\})", description).group(1)
    assert json.loads(ranges) == {
        family: {name: [float(x) for x in bounds] for name, bounds in shape.items()}
        for family, shape in cfg.saturation_prior_ranges.items()
    }
    for line in (
        "mm_scale_prior=log_uniform",
        "treatment_reference_contribution_range=(0.0001, 0.0005)",
        "treatment_reference_multiplier=1.6",
        "covariate_reference_contribution_range=(0.0001, 0.0002)",
        "covariate_reference_scale=3.0",
    ):
        assert line in description
    # The treatment figure titles show the same small derived beta.
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from pymc_generator import viz

    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    viz.plot_treatments(world, str(tmp_path / "treatments.png"))
    titles = [ax.get_title() for ax in figures[0].axes if "β=" in ax.get_title()]
    np.testing.assert_allclose(
        [float(re.search(r"β=(\S+)", title).group(1)) for title in titles],
        world.params["beta"],
        rtol=5e-4,
    )
    # Edge lists and DOT labels show the small derived beta / rho_zy as well.
    edges = {(src, dst): coef for _, src, dst, coef in edges_with_coeffs(world.g, world.params)}
    assert {("C1", "Y"), ("C2", "Y"), ("Z1", "Y")} <= set(edges)
    for pattern, text in (
        (r"\] (\S+) -> (\S+)\s+coef=(\S+)", pg.describe_scm(world)),
        (r'(\S+) -> (\S+) \[label="([^"]+)"', world_to_dot(world)),
    ):
        printed_edges = {(src, dst): float(coef) for src, dst, coef in re.findall(pattern, text)}
        assert set(printed_edges) == set(edges)
        for key, coef in edges.items():
            np.testing.assert_allclose(printed_edges[key], coef, rtol=5e-4, err_msg=str(key))
    control = audit["Z1"]["outcome_reference"]
    np.testing.assert_allclose(control["rho_zy"] * control["input"], control["contribution"])
    assert control["stage"] == "pre_gate_pre_floor" and control["g_zy"] == 1
    description = pg.describe_scm(world)
    references = re.findall(r"reference\(([^)]+)\)", description)
    reported = []
    for record in references:
        fields = dict(part.split("=", 1) for part in record.split(",") if "=" in part)
        reported.append((float(fields["contribution"]), float(fields["input"])))
    expected = [
        (
            float(world.params["treatment_reference_contribution"][k]),
            float(world.params["treatment_reference_input"][k]),
        )
        for k in range(2)
    ] + [(control["contribution"], control["input"])]
    np.testing.assert_allclose(sorted(reported), sorted(expected), rtol=0.0, atol=0.0)
    bundle = pg.write_scm_bundle(world, tmp_path / "world", plots=False)
    persisted = re.findall(r"reference\(([^)]+)\)", (bundle / "description.txt").read_text())
    persisted_values = []
    for record in persisted:
        fields = dict(part.split("=", 1) for part in record.split(",") if "=" in part)
        persisted_values.append((float(fields["contribution"]), float(fields["input"])))
    np.testing.assert_allclose(sorted(persisted_values), sorted(expected), rtol=0.0, atol=0.0)
    replay = build_symbolic_graph(
        world.g,
        world.params,
        world.n_time_steps,
        world.n_treatments,
        world.n_covariates,
        world.n_latent,
        burn_in=cfg.carryover_burn_in,
        eps=world.exogenous,
    )["outputs"]
    for name, value in replay.items():
        np.testing.assert_allclose(value.eval(), world.data[name], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        world.reconstruction(), world.data["outcome"], rtol=1e-12, atol=1e-12
    )
