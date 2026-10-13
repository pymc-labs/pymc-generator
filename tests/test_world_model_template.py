"""The one-compile-per-shard template path.

The template trades a denser graph for compiling once per shard instead of once
per cell, by making the DAG, the mechanism families, and the random-walk kernel
widths run-time inputs. These tests pin the three equivalences that trade rests
on — the walk-by-width identity, the dynamic-vs-static graph, and per-cell
structure actually taking effect — plus the guards that caught real bugs while it
was being built.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.replace import clone_replace
from pytensor.scalar import Add
from pytensor.tensor.elemwise import Elemwise

import pymc_generator.world_model as world_model
from pymc_generator import make_scm_prior
from pymc_generator.random_walk import (
    _walk_basis_stack,
    symbolic_random_walk,
    symbolic_random_walk_by_width,
    walk_width_index,
)
from pymc_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    CARRYOVER_FAMILY_KEYS,
    SATURATION_FAMILY_KEYS,
    _slice_g_active,
    sample_g_additive,
)
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from pymc_generator.symbolic_graph import build_symbolic_graph
from pymc_generator.trajectories import SCHEDULE_COMPONENTS, structural_key
from pymc_generator.world_model import build_world_model, draw_worlds, sample_structure
from pymc_generator.world_model_template import (
    build_cell_inputs,
    build_world_model_template,
    check_template_supported,
    compile_template_draw_fn,
    sample_cell_structures,
)

CORPUS_NAMES = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES


def _cfg(**overrides: Any):
    base: dict[str, Any] = {
        "n_treatments": 3,
        "n_covariates": 2,
        "n_latent": 2,
        "n_time_steps": 32,
        "n_cells": 3,
        "draws_per_cell": 2,
        "seed": 4242,
        "n_treatments_active_range": (2, 3),
        "n_covariates_active_range": (1, 2),
        "n_latent_active_range": (1, 2),
    }
    base.update(overrides)
    return make_scm_prior(**base)


@pytest.fixture(scope="module")
def template():
    """One compiled template plus the cells it will be driven with.

    Four stratified cells cover the 2x2 active treatment × covariate grid once
    each, so the tests below run every combination through the one compiled
    function. Latent counts are still drawn per cell: seed 4242 gives one cell a
    single active latent, which the zero-slot test needs.
    """
    cfg = _cfg(n_cells=4, active_count_allocation="stratified")
    cells = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))
    model, _, _ = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
    draw = compile_template_draw_fn(model, CORPUS_NAMES)
    return cfg, cells, draw


# -- the walk-by-width identity -------------------------------------------------


@pytest.mark.parametrize("n_time_steps", (10, 40))
@pytest.mark.parametrize("positive_only", (False, True))
def test_walk_by_width_matches_symbolic_random_walk_at_every_width(positive_only, n_time_steps):
    """Selecting a kernel by index equals baking that kernel into the graph.

    This is the identity the template's single compile rests on: if it failed for
    any reachable width, template worlds would differ from per-world worlds in a
    way no shape or dtype check would reveal.
    """
    rw_max = 26
    rng = np.random.default_rng(11)
    eps = rng.normal(size=n_time_steps)
    mean, std = 0.3, 1.7

    n_widths = min(rw_max, n_time_steps)
    for width in range(1, n_widths + 1):
        smoothness = 0.0 if width == 1 else 1.0 if width == n_widths else width / rw_max
        index = int(
            walk_width_index(np.array([smoothness]), n_time_steps, rw_smoothness_max_weeks=rw_max)[
                0
            ]
        )

        baked = symbolic_random_walk(
            n_time_steps,
            mean=mean,
            std=std,
            smoothness=smoothness,
            positive_only=positive_only,
            rw_smoothness_max_weeks=rw_max,
            eps=pt.as_tensor_variable(eps),
        ).eval()
        by_width = symbolic_random_walk_by_width(
            n_time_steps,
            mean=mean,
            std=std,
            width_index=index,
            positive_only=positive_only,
            rw_smoothness_max_weeks=rw_max,
            eps=pt.as_tensor_variable(eps),
        ).eval()
        np.testing.assert_allclose(by_width, baked, rtol=0, atol=1e-12)


# -- the dynamic graph computes the same function as the static one --------------


def _concrete_scm_inputs(n_treatments, n_covariates, n_latent, n_time_steps_full, seed=7):
    """Concrete params/eps/g for one world, usable by both graph modes."""
    rng = np.random.default_rng(seed)
    g = {
        "g_cy": np.ones(n_treatments),
        "g_dc": rng.integers(0, 2, (n_latent, n_treatments)).astype("float64"),
        "g_dz": rng.integers(0, 2, (n_latent, n_covariates)).astype("float64"),
        "g_dy": np.ones(n_latent),
        "g_zy": np.ones(n_covariates),
        "g_zc": rng.integers(0, 2, (n_covariates, n_treatments)).astype("float64"),
        "g_cc": np.triu(np.ones((n_treatments, n_treatments)), 1),
        "g_zz": np.triu(np.ones((n_covariates, n_covariates)), 1),
    }
    carryover_family = np.array([2, 1, 0][:n_treatments], dtype="int64")
    sat_family = np.array([1, 3, 0][:n_treatments], dtype="int64")

    def walk(n, positive):
        return {
            "mean": rng.uniform(0.2, 0.8, n),
            "std": rng.uniform(0.3, 0.7, n),
            "positive_only": positive,
            "smoothness": rng.uniform(0.1, 0.9, n),
            "rw_smoothness_max_weeks": 26,
        }

    params: dict[str, Any] = {
        "l_max": 8,
        "w_dc": rng.uniform(0.1, 0.5, (n_latent, n_treatments)),
        "u_dz": rng.uniform(0.1, 0.5, (n_latent, n_covariates)),
        "v_zc": rng.uniform(0.1, 0.5, (n_covariates, n_treatments)),
        "alpha_cc": np.triu(rng.uniform(0.1, 0.3, (n_treatments, n_treatments)), 1),
        "gamma_zz": np.triu(rng.uniform(0.1, 0.3, (n_covariates, n_covariates)), 1),
        "delta_dy": rng.uniform(0.1, 0.5, n_latent),
        "rho_zy": rng.uniform(0.1, 0.5, n_covariates),
        "beta": rng.uniform(0.5, 1.5, n_treatments),
        "carryover_family": carryover_family,
        "sat_family": sat_family,
        "carryover_alpha": rng.uniform(0.2, 0.7, n_treatments),
        "weibull_lam": rng.uniform(1.0, 3.0, n_treatments),
        "weibull_k": rng.uniform(1.0, 3.0, n_treatments),
        "hill_slope": rng.uniform(1.0, 2.0, n_treatments),
        "hill_kappa_mult": rng.uniform(0.8, 1.5, n_treatments),
        "logistic_lam": rng.uniform(0.5, 1.5, n_treatments),
        "mm_kappa_mult": rng.uniform(0.8, 1.5, n_treatments),
        "tanh_c": rng.uniform(0.8, 1.5, n_treatments),
        "root_alpha": rng.uniform(0.3, 0.8, n_treatments),
        "hf_sigma": rng.uniform(0.05, 0.2, n_treatments),
        "pulse_amp": rng.uniform(0.1, 0.4, n_treatments),
        "pulse_prob": rng.uniform(0.1, 0.3, n_treatments),
        "use_hf": np.ones(n_treatments, dtype=bool),
        "use_pulse": np.ones(n_treatments, dtype=bool),
        "covariate_hf_sigma": rng.uniform(0.05, 0.2, n_covariates),
        "covariate_pulse_amp": rng.uniform(0.1, 0.4, n_covariates),
        "covariate_pulse_prob": rng.uniform(0.1, 0.3, n_covariates),
        "use_covariate_hf": np.ones(n_covariates, dtype=bool),
        "use_covariate_pulse": np.ones(n_covariates, dtype=bool),
        "rw_d": walk(n_latent, False),
        "rw_z": walk(n_covariates, False),
        "rw_c": walk(n_treatments, True),
        "rw_b": walk(1, False),
        "rw_y": {"mean": np.zeros(1), "std": rng.uniform(0.2, 0.4, 1), "positive_only": False},
    }
    eps = {
        "eps_d": rng.normal(size=(n_time_steps_full, n_latent)),
        "eps_z": rng.normal(size=(n_time_steps_full, n_covariates)),
        "eps_c": rng.normal(size=(n_time_steps_full, n_treatments)),
        "eps_b": rng.normal(size=n_time_steps_full),
        "eps_y": rng.normal(size=n_time_steps_full),
        "eps_c_hf": rng.normal(size=(n_time_steps_full, n_treatments)),
        "eps_c_pulse": rng.integers(0, 2, (n_time_steps_full, n_treatments)).astype("float64"),
        "eps_z_hf": rng.normal(size=(n_time_steps_full, n_covariates)),
        "eps_z_pulse": rng.integers(0, 2, (n_time_steps_full, n_covariates)).astype("float64"),
    }
    return g, params, eps


def test_dynamic_graph_matches_static_graph_on_identical_inputs():
    """dynamic_g must be a different graph for the SAME function.

    The dynamic form wires every candidate edge, switches over every mechanism
    family, and selects walk kernels by index. Feeding both forms identical
    concrete values is the only direct check that all three rewrites are
    faithful rather than merely plausible.
    """
    n_treatments, n_covariates, n_latent = 3, 2, 2
    n_time_steps, burn_in = 24, 8
    n_time_steps_full = n_time_steps + burn_in
    g, params, eps = _concrete_scm_inputs(n_treatments, n_covariates, n_latent, n_time_steps_full)

    static = build_symbolic_graph(
        g,
        params,
        n_time_steps,
        n_treatments,
        n_covariates,
        n_latent,
        burn_in=burn_in,
        eps=eps,
    )["outputs"]

    # The dynamic form takes structure as tensors and kernel widths as indices.
    dyn_params = dict(params)
    for group, key in (("rw_d", "smoothness"), ("rw_z", "smoothness"), ("rw_c", "smoothness")):
        dyn_params[group] = dict(params[group])
        dyn_params[group]["width_index"] = pt.as_tensor_variable(
            walk_width_index(params[group][key], n_time_steps_full, rw_smoothness_max_weeks=26)
        )
    dyn_params["rw_b"] = dict(params["rw_b"])
    dyn_params["rw_b"]["width_index"] = pt.as_tensor_variable(
        walk_width_index(
            params["rw_b"]["smoothness"], n_time_steps_full, rw_smoothness_max_weeks=26
        )
    )
    dyn_params["carryover_family"] = pt.as_tensor_variable(params["carryover_family"])
    dyn_params["sat_family"] = pt.as_tensor_variable(params["sat_family"])

    dynamic = build_symbolic_graph(
        {k: pt.as_tensor_variable(v) for k, v in g.items()},
        dyn_params,
        n_time_steps,
        n_treatments,
        n_covariates,
        n_latent,
        burn_in=burn_in,
        eps=eps,
        active={
            "active_treatment": pt.as_tensor_variable(np.ones(n_treatments)),
            "active_covariate": pt.as_tensor_variable(np.ones(n_covariates)),
            "active_latent": pt.as_tensor_variable(np.ones(n_latent)),
        },
        dynamic_g=True,
    )["outputs"]

    assert set(static) == set(dynamic)
    for name in static:
        np.testing.assert_allclose(
            np.asarray(dynamic[name].eval(), dtype="float64"),
            np.asarray(static[name].eval(), dtype="float64"),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"dynamic_g diverged from the static graph for {name!r}",
        )


def test_disabled_covariate_texture_needs_no_covariate_noise_and_enabled_texture_latent_unobserved_it():
    """Direct concrete callers may omit the covariate-texture innovations.

    A config with the texture off builds the pre-texture covariate equation, so
    omitting ``eps_z_hf`` / ``eps_z_pulse`` must be identical to passing them
    with the flags off — while an ENABLED term with no innovation is a caller
    bug and must fail loudly instead of silently defaulting to zero.
    """
    n_treatments, n_covariates, n_latent = 2, 2, 1
    n_time_steps, burn_in = 16, 8
    g, params, eps = _concrete_scm_inputs(
        n_treatments, n_covariates, n_latent, n_time_steps + burn_in
    )
    off = dict(params)
    off["use_covariate_hf"] = np.zeros(n_covariates, dtype=bool)
    off["use_covariate_pulse"] = np.zeros(n_covariates, dtype=bool)
    without_noise = {k: v for k, v in eps.items() if k not in ("eps_z_hf", "eps_z_pulse")}

    def covariates(params_in, eps_in):
        graph = build_symbolic_graph(
            g,
            params_in,
            n_time_steps,
            n_treatments,
            n_covariates,
            n_latent,
            burn_in=burn_in,
            eps=eps_in,
        )
        return np.asarray(graph["outputs"]["covariates"].eval(), dtype="float64")

    np.testing.assert_array_equal(covariates(off, without_noise), covariates(off, eps))
    # ... and the enabled graph really consumes them, so it cannot be equal.
    assert not np.allclose(covariates(params, eps), covariates(off, eps), rtol=0.0, atol=1e-12)

    for flag, missing in (("use_covariate_hf", "eps_z_hf"), ("use_covariate_pulse", "eps_z_pulse")):
        enabled_one = dict(off)
        enabled_one[flag] = np.ones(n_covariates, dtype=bool)
        with pytest.raises(ValueError, match=missing):
            covariates(enabled_one, {k: v for k, v in eps.items() if k != missing})


def test_absent_covariate_flags_are_derived_from_the_concrete_magnitudes():
    """A caller may omit the flags entirely; magnitudes then decide.

    ``build_world_model`` always passes ``use_covariate_*``, but a direct concrete
    caller may not, so the graph derives them: hf from a nonzero sigma, and the
    pulse from a nonzero amplitude AND a positive fire probability (an amplitude
    with probability zero can never fire).
    """
    n_treatments, n_covariates, n_latent = 2, 2, 1
    n_time_steps, burn_in = 16, 8
    g, params, eps = _concrete_scm_inputs(
        n_treatments, n_covariates, n_latent, n_time_steps + burn_in
    )

    def covariates(params_in, eps_in):
        graph = build_symbolic_graph(
            g,
            params_in,
            n_time_steps,
            n_treatments,
            n_covariates,
            n_latent,
            burn_in=burn_in,
            eps=eps_in,
        )
        return np.asarray(graph["outputs"]["covariates"].eval(), dtype="float64")

    derived = {
        k: v for k, v in params.items() if k not in ("use_covariate_hf", "use_covariate_pulse")
    }
    np.testing.assert_array_equal(covariates(derived, eps), covariates(params, eps))

    # A live amplitude with a zero fire probability stays OFF, so its innovation
    # is not even required.
    dead_pulse = dict(derived)
    dead_pulse["covariate_pulse_prob"] = np.zeros(n_covariates)
    dead_pulse["covariate_hf_sigma"] = np.zeros(n_covariates)
    off = dict(params)
    off["use_covariate_hf"] = np.zeros(n_covariates, dtype=bool)
    off["use_covariate_pulse"] = np.zeros(n_covariates, dtype=bool)
    np.testing.assert_array_equal(
        covariates(dead_pulse, {k: v for k, v in eps.items() if k != "eps_z_pulse"}),
        covariates(off, eps),
    )

    # A derived-on term with no innovation is still a loud error.
    with pytest.raises(ValueError, match="eps_z_hf"):
        covariates(derived, {k: v for k, v in eps.items() if k != "eps_z_hf"})


def test_unused_response_priors_do_not_change_concrete_worlds():
    """An unused geometric prior cannot change a Weibull world's random draws."""
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        carryover_family_probs={"none": 0.0, "geometric": 0.0, "weibull": 1.0},
        saturation_family_probs={
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
    )
    rng = np.random.default_rng(3)
    g = sample_g_additive(rng, cfg, cfg.layout)
    g_act = _slice_g_active(g, 2, 2, 1)
    structural = sample_structure(g_act, cfg, rng)
    results = []
    for alpha_range in ((0.2, 0.2), (0.2, 0.8)):
        cfg.carryover_alpha_range = alpha_range
        model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.n_time_steps)
        results.append(draw_worlds(model, out_names, seed=37, draws=2))
    for name in out_names:
        np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)


# -- the template as a whole ----------------------------------------------------


def test_template_fixture_drives_every_active_count_combination_once(template):
    _, cells, _ = template
    combinations = sorted(
        (int(cell["active_treatment"].sum()), int(cell["active_covariate"].sum())) for cell in cells
    )
    assert combinations == [(2, 1), (2, 2), (3, 1), (3, 2)]


def test_template_satisfies_the_additive_identity_on_every_world(template):
    """The exact decomposition must survive the denser dynamic graph."""
    cfg, cells, draw = template
    for i, cell in enumerate(cells):
        drawn = draw(cell, seed=500 + i, draws=cfg.draws_per_cell)
        for d in range(cfg.draws_per_cell):
            w = {k: np.asarray(v[d], dtype="float64") for k, v in drawn.items()}
            residual = (
                w["baseline_intrinsic"]
                + w["outcome_noise"]
                + w["latent_unobserved_contribution"].sum(-1)
                + w["covariate_contribution"].sum(-1)
                + w["contributions"].sum(-1)
                + w["indirect_effects_by_source"].sum(-1)
                - w["outcome"]
            )
            assert np.abs(residual).max() < 1e-9


def test_template_zeroes_inactive_node_slots(template):
    """Padded slots must be exactly zero, not merely small.

    The template runs at max size, so a cell using fewer nodes carries padded
    columns. They are switched off inside the graph rather than trimmed
    afterwards, and consumers read them as real zeros. Every node family is
    tracked separately: covariate texture adds a term inside the covariate mask, so
    a guard satisfied by an inactive treatment alone would prove nothing about it.
    """
    cfg, cells, draw = template
    keys = (
        ("treatments", "active_treatment"),
        ("covariates", "active_covariate"),
        ("covariate_contribution", "active_covariate"),
        ("latent_unobserved", "active_latent"),
    )
    seen_inactive = dict.fromkeys(keys, False)
    for i, cell in enumerate(cells):
        drawn = draw(cell, seed=700 + i, draws=1)
        for key, flags in keys:
            arr = np.asarray(drawn[key][0])
            for node, is_active in enumerate(cell[flags]):
                if is_active:
                    continue
                seen_inactive[(key, flags)] = True
                assert np.abs(arr[:, node]).max() == 0.0, f"{key}[:, {node}] not zeroed"
    unproven = [key for key, seen in seen_inactive.items() if not seen]
    assert not unproven, f"fixture never produced an inactive node for {unproven}"


def test_template_smoothness_changes_the_drawn_world(template):
    """Changing only the kernel widths must change the draw.

    Holds everything else fixed, so a graph that ignored the width input would
    return identical worlds and fail here.
    """
    cfg, cells, draw = template
    base = dict(cells[0])
    shifted = dict(base)
    other = np.asarray(base["rw_width_c"]).copy()
    other[:] = (other + 5) % _walk_basis_stack(
        cfg.n_time_steps + cfg.carryover_burn_in, cfg.rw_smoothness_max_weeks
    ).shape[0]
    shifted["rw_width_c"] = other

    a = draw(base, seed=99, draws=1)["treatments"][0]
    b = draw(shifted, seed=99, draws=1)["treatments"][0]
    assert not np.allclose(a, b), "kernel-width input had no effect on the draw"


def test_template_direct_edge_swap_changes_only_its_direct_contribution(template):
    _, cells, draw = template
    present = dict(cells[0])
    absent = dict(cells[0])
    present["g_cy"] = cells[0]["g_cy"].copy()
    absent["g_cy"] = cells[0]["g_cy"].copy()
    present["g_cy"][0] = 1.0
    absent["g_cy"][0] = 0.0
    with_edge = draw(present, seed=31, draws=1)["contributions"]
    without_edge = draw(absent, seed=31, draws=1)["contributions"]
    assert np.any(with_edge[..., 0] > 0.0)
    assert np.all(without_edge[..., 0] == 0.0)
    np.testing.assert_array_equal(with_edge[..., 1:], without_edge[..., 1:])


def test_template_draws_are_seed_deterministic(template):
    """Same seed and same structure must reproduce the world exactly."""
    cfg, cells, draw = template
    a = draw(cells[0], seed=17, draws=2)
    b = draw(cells[0], seed=17, draws=2)
    for name in CORPUS_NAMES:
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]))


def test_template_treatments_are_non_negative(template):
    """The softplus positivity guard must survive the dynamic path."""
    cfg, cells, draw = template
    for i, cell in enumerate(cells):
        drawn = draw(cell, seed=800 + i)
        assert np.asarray(drawn["treatments"]).min() >= 0.0


@pytest.mark.parametrize(
    "overrides, match",
    (
        ({"n_treatment_shocks": 1}, "treatment shocks"),
        ({"prior_conditioning": True}, "prior conditioning"),
        ({"confounding_strength_range": (0.1, 0.6)}, "fixed confounding_strength"),
    ),
)
def test_template_rejects_unsupported_configs(overrides, match):
    """Unsupported features must fail loudly, not generate a wrong corpus."""
    with pytest.raises(ValueError, match=match):
        check_template_supported(_cfg(**overrides))


def test_template_accepts_a_fixed_confounding_strength():
    """A degenerate confounding range is representable without an input slot."""
    check_template_supported(_cfg(confounding_strength_range=(0.4, 0.4)))


# -- the compiled-draw-function cache ------------------------------------------


def test_compile_cache_is_transparent_to_draws():
    """Caching a compiled function must not change what it produces."""
    cfg = _cfg(n_cells=2)
    cells = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))

    results = {}
    for enabled in (True, False):
        world_model.reset_world_model_caches()
        world_model.set_compile_cache_enabled(enabled)
        try:
            model, _out, _param = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
            draw = compile_template_draw_fn(model, ("outcome", "treatments"))
            results[enabled] = draw(cells[0], seed=5, draws=2)
        finally:
            world_model.set_compile_cache_enabled(True)
    for name in ("outcome", "treatments"):
        np.testing.assert_array_equal(results[True][name], results[False][name])


def test_template_cache_respects_different_outcome_priors():
    """A later model must use its own coefficients, not a prior cached graph."""
    cfg = _cfg(n_cells=2, beta_additive_range=(1.0, 1.0))
    cells = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))
    results = []
    for beta in (1.0, 2.0):
        cfg = _cfg(n_cells=2, beta_additive_range=(beta, beta))
        model, _, _ = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
        draw = compile_template_draw_fn(model, CORPUS_NAMES)
        results.append(draw(cells[0], seed=5, draws=1)["contributions"])
    assert np.any(results[0] > 0.0)
    np.testing.assert_allclose(results[1], 2.0 * results[0], rtol=1e-12, atol=0.0)


def test_build_world_model_is_not_cached_across_differing_configs():
    """Two configs differing only in a prior range must get different models.

    A structure-keyed model cache collided here: the second config silently
    reused the first one's model and reported the first one's parameters.
    """
    models = []
    for hi in (0.0, 0.5):
        cfg = make_scm_prior(
            n_treatments=2,
            n_covariates=2,
            n_latent=1,
            n_time_steps=20,
            seed=71,
            nonlinearity="linear",
            edge_budget={"cy": (2, 2)},
            n_treatment_shocks=1,
            treatment_shock_length_range=(3, 3),
            treatment_shock_level_range=(hi, hi),
        )
        rng = np.random.default_rng(cfg.seed)
        g = sample_g_additive(rng, cfg, cfg.layout)
        g_act = _slice_g_active(g, 2, 2, 1)
        structural = sample_structure(g_act, cfg, rng)
        model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.n_time_steps)
        drawn = draw_worlds(model, ("treatment_shock_level",), seed=5, draws=1)
        models.append(np.asarray(drawn["treatment_shock_level"]))

    assert np.all(models[0] == 0.0), "a zero level multiplier must produce zero levels"
    assert np.all(models[1] > 0.0), "a nonzero level multiplier must produce nonzero levels"


@pytest.mark.parametrize("mode", ("FAST_COMPILE", "FAST_RUN"))
@pytest.mark.parametrize(
    ("reference_target", "relative_std"),
    (
        pytest.param(1e-180, 0.1, id="tiny-reference"),
        pytest.param(np.finfo(np.float64).tiny / 2, 0.1, id="subnormal-reference"),
        pytest.param(np.finfo(np.float64).tiny, 1e-16, id="minimum-final-sigma"),
        pytest.param(np.nextafter(0.0, 1.0), 1e38, id="amplified-minimum-beta"),
        pytest.param(1e38, np.nextafter(0.0, 1.0), id="amplified-minimum-std"),
    ),
)
def test_reused_template_preserves_tiny_relative_outcome_scales(
    reference_target, relative_std, mode
):
    """Runtime edge changes retain tiny scales and restore exact zero-edge behavior."""
    cfg = _cfg(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        n_cells=2,
        n_treatments_active_range=(1, 2),
        n_covariates_active_range=(1, 1),
        n_latent_active_range=(1, 1),
        edge_budget={"cy": (0, 2), "cc": 0, "zc": 0, "dc": 0},
        nonlinearity="linear",
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        carryover_burn_in=0,
        treatment_reference_contribution_range=(reference_target, reference_target),
        treatment_reference_multiplier=1.0,
        rw_baseline_std_range=(relative_std, relative_std),
        rw_outcome_std_range=(relative_std, relative_std),
    )
    rng = np.random.default_rng(13)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=1, n_latent_active=1
    )
    g["g_cy"][:] = 1
    structural = sample_structure(_slice_g_active(g, 2, 1, 1), cfg, rng)
    active = {
        "active_treatment": np.ones(2),
        "active_covariate": np.ones(1),
        "active_latent": np.ones(1),
    }
    full_cell = build_cell_inputs(cfg, g, active, structural)
    inactive_cell = {name: value.copy() for name, value in full_cell.items()}
    inactive_cell["active_treatment"][1] = inactive_cell["g_cy"][1] = 0.0
    zero_cell = {name: value.copy() for name, value in inactive_cell.items()}
    zero_cell["g_cy"][:] = 0.0
    model, _, _ = build_world_model_template(cfg, full_cell, cfg.n_time_steps)
    names = ("param_beta", "param_rw_b_std", "param_rw_y_std")
    draw = compile_template_draw_fn(model, names, mode=mode)
    results = []
    for cell in (full_cell, inactive_cell, zero_cell, full_cell):
        result = draw(cell, seed=14)
        results.append(result)
        with localcontext() as context:
            context.prec = 100
            squared = sum(
                Decimal.from_float(float(v)) ** 2 for v in cell["g_cy"] * result["param_beta"][0]
            )
            expected = float(Decimal.from_float(relative_std) * squared.sqrt())
        for name in ("param_rw_b_std", "param_rw_y_std"):
            np.testing.assert_allclose(
                result[name][0], expected, rtol=32 * np.finfo(np.float64).eps, atol=0.0
            )
    for name in names:
        np.testing.assert_array_equal(results[-1][name], results[0][name])


# -- rich trajectories through the actual ordinary and template builders --------


def _evaluate_joint_draw(model, names, drawn):
    """Bind the same primitive draws, not a second seed or concrete folded graph."""
    replacements = {rv: rv.type(name=f"given_{rv.name}") for rv in model.free_RVs}
    expressions = clone_replace(
        [model[name] for name in names], replace=replacements, rebuild_strict=False
    )
    evaluate = pytensor.function(
        list(replacements.values()), expressions, on_unused_input="ignore", mode="FAST_COMPILE"
    )
    values = []
    for rv in replacements:
        shape = tuple(int(n) for n in rv.shape.eval())
        raw = np.asarray(drawn[rv.name][0], dtype=rv.dtype)
        values.append(raw[tuple(slice(0, n) for n in shape)] if shape else raw)
    return dict(zip(names, evaluate(*values)))


def _assert_ordinary_template_parity(cfg, g, structural, drawn, output_names):
    ordinary, names, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    expected = _evaluate_joint_draw(ordinary, names, drawn)
    assert set(names) == set(output_names)
    n_t, n_c, n_l = len(g["g_cy"]), len(g["g_zy"]), len(g["g_dy"])
    widths = {
        "treatments": n_t,
        "treatments_base": n_t,
        "treatments_natural": n_t,
        "contributions": n_t,
        "contributions_observed": n_t,
        "saturation_scale": n_t,
        "treatment_activity": n_t,
        "treatment_log_level_shift": n_t,
        "treatment_shock_mask": n_t,
        "treatment_shock_mask_full": n_t,
        "covariates": n_c,
        "covariate_contribution": n_c,
        "covariate_activity": n_c,
        "covariate_level_shift": n_c,
        "latent_unobserved": n_l,
        "latent_unobserved_contribution": n_l,
    }
    for name, value in expected.items():
        actual = np.asarray(drawn[name][0])
        if name in widths:
            actual = actual[..., : widths[name]]
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(actual, value, err_msg=name)
        else:
            np.testing.assert_allclose(actual, value, rtol=2e-11, atol=2e-11, err_msg=name)


@pytest.fixture(scope="module")
def rich_template():
    """One compile reused A -> B -> A, including all families and both input roles."""
    cfg = _cfg(
        n_treatments=6,
        n_covariates=3,
        n_treatments_active_range=(2, 6),
        n_covariates_active_range=(1, 3),
        n_time_steps=32,
        l_max=4,
        carryover_burn_in=4,
        confounding_strength_range=(0.3, 0.3),
        treatment_hf_sigma_range=(0.05, 0.15),
        treatment_pulse_prob_range=(0.1, 0.3),
        covariate_hf_sigma_range=(0.05, 0.15),
        covariate_pulse_prob_range=(0.1, 0.3),
        treatment_level_jump_count=3,
        covariate_level_jump_count=2,
        treatment_level_jump_factor_range=(2.0, 2.0),
        treatment_trend_log_change_range=(-0.3, 0.4),
        treatment_seasonal_amplitude_range=(0.1, 0.25),
        saturation_prior_ranges={
            "hill": {"slope": (5.0, 7.0), "kappa_mult": (0.7, 1.5)},
            "logistic": {"lam": (0.5, 2.0)},
            "michaelis_menten": {"kappa_mult": (0.2, 1.2)},
            "tanh": {"c": (0.3, 1.5)},
            "root": {"alpha": (0.3, 0.9)},
        },
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(0.5, 1.1),
        treatment_reference_multiplier=1.6,
        covariate_reference_contribution_range=(-0.2, 0.5),
        covariate_reference_scale=1.25,
        **{
            f"{role}_{component}_inclusion_prob": 0.5
            for role in TRAJECTORY_INPUTS
            for component in TRAJECTORY_COMPONENTS
        },
        **{f"{role}_onset_frac_range": (0.1, 0.15) for role in TRAJECTORY_INPUTS},
        **{f"{role}_flighting_period_weeks_range": (2, 4) for role in TRAJECTORY_INPUTS},
        **{f"{role}_flighting_duty_range": (0.5, 0.8) for role in TRAJECTORY_INPUTS},
    )
    g = {
        "g_cy": np.array([1, 0, 1, 1, 1, 1]),
        "g_dc": np.array([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0]]),
        "g_dz": np.array([[1, 0, 1], [0, 1, 0]]),
        "g_dy": np.ones(2),
        "g_zy": np.ones(3),
        "g_zc": np.zeros((3, 6)),
        "g_cc": np.zeros((6, 6)),
        "g_zz": np.zeros((3, 3)),
    }
    g["g_zc"][0, 1] = g["g_zc"][1, 4] = 1
    g["g_cc"][0, 1] = g["g_cc"][1, 2] = g["g_cc"][2, 5] = 1
    g["g_zz"][0, 1] = g["g_zz"][1, 2] = 1
    cases = []
    for all_off, counts in ((False, (6, 3, 2)), (True, (3, 1, 1))):
        n_t, n_c, n_l = counts
        active = {
            "active_treatment": (np.arange(6) < n_t).astype(float),
            "active_covariate": (np.arange(3) < n_c).astype(float),
            "active_latent": (np.arange(2) < n_l).astype(float),
        }
        padded_g = {key: value.copy() for key, value in g.items()}
        padded_g["g_cy"][n_t:] = padded_g["g_zy"][n_c:] = padded_g["g_dy"][n_l:] = 0
        for key, rows, cols in (
            ("g_dc", n_l, n_t),
            ("g_dz", n_l, n_c),
            ("g_zc", n_c, n_t),
            ("g_cc", n_t, n_t),
            ("g_zz", n_c, n_c),
        ):
            padded_g[key][rows:, :] = 0
            padded_g[key][:, cols:] = 0
        active_g = _slice_g_active(padded_g, n_t, n_c, n_l)
        structural = sample_structure(active_g, cfg, np.random.default_rng(17))
        structural["carryover_family"] = (
            np.array([2, 0, 1]) if all_off else np.arange(6) % len(CARRYOVER_FAMILY_KEYS)
        )
        structural["sat_family"] = (
            np.array([5, 3, 1]) if all_off else np.arange(len(SATURATION_FAMILY_KEYS))
        )
        for key, width in (
            ("smoothness_c", n_t),
            ("smoothness_z", n_c),
            ("smoothness_d", n_l),
            ("smoothness_b", 1),
        ):
            structural[key] = (
                np.linspace(0.85, 0.2, width) if all_off else np.linspace(0.1, 0.7, width)
            )
        for role, width in (("treatment", n_t), ("covariate", n_c)):
            for j, component in enumerate(TRAJECTORY_COMPONENTS):
                flags = np.zeros(width, dtype=bool) if all_off else (np.arange(width) + j) % 3 != 0
                if not all_off:
                    flags[0] = component == "level_jump" if role == "treatment" else True
                    flags[1] = True
                structural[structural_key(role, component)] = flags
        cell = build_cell_inputs(cfg, padded_g, active, structural)
        cases.append((active_g, structural, cell))
    model, outputs, reports = build_world_model_template(cfg, cases[0][2], cfg.n_time_steps)
    names = tuple(dict.fromkeys((*outputs, *reports, *(rv.name for rv in model.free_RVs))))
    draw = compile_template_draw_fn(model, names)
    results = [draw(case[2], seed=61) for case in (cases[0], cases[1], cases[0])]
    return cfg, cases, outputs, reports, draw, results


def test_actual_rich_builders_match_every_forward_output_and_restore_a_cell(rich_template):
    cfg, cases, outputs, _, _, results = rich_template
    for (g, structural, _), result in zip(cases, results):
        _assert_ordinary_template_parity(cfg, g, structural, result, outputs)
    for name in results[0]:
        np.testing.assert_array_equal(results[2][name], results[0][name], err_msg=name)


def _numpy_schedule(cfg, role, cell, result):
    """Independent gates/envelopes, quantized from the same joint primitive draws."""
    n_time_steps = cfg.n_time_steps
    weeks = np.arange(n_time_steps)
    active = cell[f"active_{role}"].astype(bool)
    activity = np.zeros((n_time_steps, len(active)), dtype=np.int8)
    shift = np.zeros_like(activity, dtype=float)
    multiplier = np.ones_like(shift)
    for i in np.flatnonzero(active):
        gate = np.ones(n_time_steps, dtype=bool)
        smooth = np.zeros(n_time_steps)
        carried = {
            component: bool(cell[structural_key(role, component)][i])
            for component in TRAJECTORY_COMPONENTS
        }
        if carried["onset"]:
            gate &= weeks >= np.floor(result[f"{role}_onset_frac"][0, i] * n_time_steps)
        if carried["offset"]:
            gate &= weeks < np.floor(result[f"{role}_offset_frac"][0, i] * n_time_steps)
        if carried["flighting"]:
            period = result[f"{role}_flighting_period"][0, i]
            duty = result[f"{role}_flighting_duty"][0, i]
            on_weeks = np.clip(np.floor(duty * period + 0.5), 1, period - 1)
            phase = min(np.floor(result[f"{role}_flighting_phase_u"][0, i] * period), period - 1)
            gate &= (weeks + phase) % period < on_weeks
        if carried["seasonal"]:
            period = getattr(cfg, f"{role}_seasonal_period_weeks_range")[0]
            smooth += result[f"{role}_seasonal_amplitude"][0, i] * np.sin(
                2 * np.pi * weeks / period + result[f"{role}_seasonal_phase"][0, i]
            )
        if carried["trend"]:
            smooth += result[f"{role}_trend_change"][0, i] * weeks / (n_time_steps - 1)
        activity[:, i] = gate.astype(np.int8)
        shift[:, i] = smooth
        multiplier[:, i] = np.exp(smooth)
        if carried["level_jump"]:
            count = getattr(cfg, f"{role}_level_jump_count")
            edges = 1 + np.arange(count + 1) * (n_time_steps - 1) // count
            jump_weeks = np.minimum(
                edges[:-1] + np.floor(result[f"{role}_level_jump_u"][0, :, i] * np.diff(edges)),
                edges[1:] - 1,
            )
            steps = weeks[:, None] >= jump_weeks[None, :]
            if role == "treatment":
                factors = np.full(count, cfg.treatment_level_jump_factor_range[0])
                shift[:, i] += (steps * np.log(factors)).sum(axis=1)
                multiplier[:, i] *= np.where(steps, factors[None, :], 1.0).prod(axis=1)
            else:
                shift[:, i] += (steps * result["covariate_level_jump_size"][0, :, i]).sum(axis=1)
    return activity, shift, multiplier


def test_reused_template_schedules_follow_numpy_gates_and_envelopes(rich_template):
    cfg, cases, _, _, draw, results = rich_template
    for (_, _, cell), result in zip(cases, results):
        without_schedules = {name: value.copy() for name, value in cell.items()}
        for role in TRAJECTORY_INPUTS:
            for component in SCHEDULE_COMPONENTS:
                without_schedules[structural_key(role, component)][:] = 0
        unscheduled = draw(without_schedules, seed=61)
        for role, shift_key in (
            ("treatment", "treatment_log_level_shift"),
            ("covariate", "covariate_level_shift"),
        ):
            activity, shift, multiplier = _numpy_schedule(cfg, role, cell, result)
            np.testing.assert_array_equal(result[f"{role}_activity"][0], activity)
            np.testing.assert_allclose(result[shift_key][0], shift, rtol=2e-12, atol=1e-12)
            np.testing.assert_array_equal(result[shift_key][0][shift == 0], 0.0)
            observed = result["treatments" if role == "treatment" else "covariates"][0]
            np.testing.assert_array_equal(observed[activity == 0], 0.0)
            if role == "treatment":
                expected = unscheduled["treatments_base"][0] * multiplier * activity
                np.testing.assert_allclose(
                    result["treatments_base"][0], expected, rtol=2e-12, atol=1e-12
                )
            else:
                expected = (unscheduled["covariates"][0, :, 0] + shift[:, 0]) * activity[:, 0]
                np.testing.assert_allclose(observed[:, 0], expected, rtol=2e-12, atol=1e-12)
        # Node 0 carries only held x2 jumps: multiplication must be exact, not exp(sum(log 2)).
        weeks = result["param_trajectory_treatment_level_jump_week"][0, :, 0]
        steps = (np.arange(cfg.n_time_steps)[:, None] >= weeks).sum(axis=1)
        multiplier = 2.0**steps if cell["use_level_jump"][0] else np.ones(cfg.n_time_steps)
        np.testing.assert_array_equal(
            result["treatments_base"][0, :, 0],
            unscheduled["treatments_base"][0, :, 0] * multiplier,
        )


def test_reused_template_zeroes_padded_schedule_outputs_and_reports(rich_template):
    _, cases, _, reports, _, results = rich_template
    cell, result = cases[1][2], results[1]
    for role, shift in (
        ("treatment", "treatment_log_level_shift"),
        ("covariate", "covariate_level_shift"),
    ):
        inactive = cell[f"active_{role}"] == 0
        np.testing.assert_array_equal(result[f"{role}_activity"][0, :, inactive], 0)
        np.testing.assert_array_equal(result[shift], 0.0)
        active = ~inactive
        np.testing.assert_array_equal(result[f"{role}_activity"][0, :, active], 1)
        for name in reports:
            if name.startswith(f"param_trajectory_{role}_"):
                np.testing.assert_array_equal(result[name][0, ..., inactive], 0)
    np.testing.assert_array_equal(result["treatments"], result["treatments_natural"])
    np.testing.assert_array_equal(result["outcome"], result["outcome_natural"])


@pytest.mark.parametrize("component", TRAJECTORY_COMPONENTS)
@pytest.mark.parametrize("role", TRAJECTORY_INPUTS)
def test_runtime_component_flags_are_data_and_move_no_draw(rich_template, role, component):
    """Inside one compiled template, a per-cell flag selects only its component's effect.

    Turning one component on for one input reuses the compiled function, so every
    random variable keeps its stream: the draws of every other component and of
    the legacy parameters and innovations are byte-identical. Admitting the
    component in the config is what reseeds a template, not its per-cell flags.
    """
    _, cases, outputs, reports, draw, _ = rich_template
    cell = cases[0][2]
    on, off = ({name: value.copy() for name, value in cell.items()} for _ in range(2))
    for flags in (on, off):
        for other in TRAJECTORY_COMPONENTS:
            flags[structural_key(role, other)][0] = 0
    on[structural_key(role, component)][0] = 1
    enabled, disabled = draw(on, seed=61), draw(off, seed=61)
    primitives = [name for name in enabled if name not in outputs and name not in reports]
    assert primitives
    for name in primitives:
        assert enabled[name].tobytes() == disabled[name].tobytes(), name
    series = "treatments" if role == "treatment" else "covariates"
    assert not np.array_equal(enabled[series][0, :, 0], disabled[series][0, :, 0])


def test_actual_builders_disable_texture_even_when_its_prior_ranges_are_live():
    cfg = _cfg(
        treatment_hf_sigma_range=(0.05, 0.15),
        treatment_pulse_prob_range=(0.1, 0.3),
        covariate_hf_sigma_range=(0.05, 0.15),
        covariate_pulse_prob_range=(0.1, 0.3),
        **{
            f"{role}_{component}_inclusion_prob": 0.0
            for role in TRAJECTORY_INPUTS
            for component in ("hf", "pulse")
        },
    )
    g, _, _ = _concrete_scm_inputs(3, 2, 2, cfg.n_time_steps + cfg.carryover_burn_in)
    structural = sample_structure(g, cfg, np.random.default_rng(11))
    active = {
        "active_treatment": np.ones(3),
        "active_covariate": np.ones(2),
        "active_latent": np.ones(2),
    }
    cell = build_cell_inputs(cfg, g, active, structural)
    model, outputs, reports = build_world_model_template(cfg, cell, cfg.n_time_steps)
    names = tuple(dict.fromkeys((*outputs, *reports, *(rv.name for rv in model.free_RVs))))
    result = compile_template_draw_fn(model, names)(cell, seed=73)
    _assert_ordinary_template_parity(cfg, g, structural, result, outputs)


def test_template_handles_more_than_32_candidate_parents_with_sparse_edges_and_padding():
    cfg = _cfg(
        n_treatments=1,
        n_covariates=1,
        n_latent=40,
        n_treatments_active_range=(1, 1),
        n_covariates_active_range=(1, 1),
        n_latent_active_range=(33, 40),
        treatment_trend_inclusion_prob=0.5,
        covariate_trend_inclusion_prob=0.5,
    )
    g, _, _ = _concrete_scm_inputs(1, 1, 40, cfg.n_time_steps + cfg.carryover_burn_in)
    g["g_dc"][:] = 0
    g["g_dc"][[0, 17, 32], 0] = 1
    g["g_dy"][:] = 0
    g["g_dy"][[1, 19, 32]] = 1
    for key in ("g_dz", "g_zc", "g_zy"):
        g[key][:] = 0
    active = {
        "active_treatment": np.ones(1),
        "active_covariate": np.ones(1),
        "active_latent": (np.arange(40) < 33).astype(float),
    }
    active_g = _slice_g_active(g, 1, 1, 33)
    structural = sample_structure(active_g, cfg, np.random.default_rng(11))
    structural["use_trend"][:] = False
    cell = build_cell_inputs(cfg, g, active, structural)
    model, outputs, reports = build_world_model_template(cfg, cell, cfg.n_time_steps)
    names = tuple(dict.fromkeys((*outputs, *reports, *(rv.name for rv in model.free_RVs))))
    result = compile_template_draw_fn(model, names, mode="FAST_COMPILE")(cell, seed=79)
    _assert_ordinary_template_parity(cfg, active_g, structural, result, outputs)
    latent = result["latent_unobserved"][0]
    np.testing.assert_array_equal(latent[:, 33:], 0.0)
    own = np.log(np.expm1(result["treatments_base"][0, :, 0]))
    expected = np.logaddexp(0.0, own + latent @ (g["g_dc"][:, 0] * result["param_w_dc"][0, :, 0]))
    np.testing.assert_allclose(result["treatments"][0, :, 0], expected, rtol=2e-12, atol=1e-12)
    expected_baseline = (
        result["baseline_intrinsic"][0]
        + result["outcome_noise"][0]
        + latent @ (g["g_dy"] * result["param_delta_dy"][0])
    )
    np.testing.assert_allclose(result["baseline"][0], expected_baseline, rtol=2e-12, atol=1e-12)


def _add_widths(fn) -> list[int]:
    return [
        len(node.inputs)
        for node in fn.maker.fgraph.toposort()
        if isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, Add)
    ]


_SPLIT_WIDTHS = {
    "31": [31],
    "32": [31, 2],
    "33": [31, 3],
    "61": [31, 31],
    "62": [31, 31, 2],
    "95": [31, 31, 31, 5],
    "bool-first-chunk": [31, 6],
    "int8-first-chunk": [31, 6],
}


@pytest.mark.parametrize("case", sorted(_SPLIT_WIDTHS))
def test_wide_python_addition_is_split_into_bounded_additions(case):
    """The Python backend runs additions past its 32-operand limit, and only those change.

    ``case`` counts every addend. One of them broadcasts from length 1, and the
    narrow-dtype cases put 31 bool or int8 addends first, so a chunk that kept
    its inputs' dtype would add as a logical OR or overflow.
    """
    widths = _SPLIT_WIDTHS[case]
    rng = np.random.default_rng(32)
    if case.endswith("first-chunk"):
        dtype = case.split("-")[0]
        inputs = [pt.vector(f"n{i}", dtype=dtype) for i in range(31)]
        inputs += [pt.dvector(f"x{i}") for i in range(5)]
        values = [np.full(8, 1 if dtype == "bool" else 100, dtype=dtype) for _ in range(31)]
        values += [np.full(8, 0.5) for _ in range(5)]
    else:
        n = int(case)
        inputs = [pt.dvector(f"x{i}") for i in range(n - 1)]
        inputs.append(pt.tensor("b", shape=(1,), dtype="float64"))
        values = [rng.normal(size=8) for _ in range(n - 1)] + [rng.normal(size=1)]
    total = pt.add(*inputs)
    # A predicate that rebuilt an identical node would loop; make that an error.
    with pytensor.config.change_flags(on_opt_error="raise"):
        split = pytensor.function(inputs, total, mode="FAST_COMPILE")
        unsplit = pytensor.function(
            inputs, total, mode=get_mode("FAST_COMPILE").excluding("pymc_generator_split_wide_add")
        )
    assert _add_widths(split) == widths
    assert split.maker.fgraph.outputs[0].type == total.type
    result = split(*values)
    if case.endswith("first-chunk"):
        assert result.tobytes() == np.full(8, 33.5 if case.startswith("bool") else 3102.5).tobytes()
    else:
        columns = np.broadcast_arrays(*values)
        exact = np.array([math.fsum(column[i] for column in columns) for i in range(8)])
        bound = 2 * len(values) * np.finfo(np.float64).eps * np.sum(np.abs(columns), axis=0)
        assert np.all(np.abs(result - exact) <= bound)
    if widths == [31]:
        assert unsplit(*values).tobytes() == result.tobytes()
    else:
        with pytest.raises(NotImplementedError, match="more than 32 operands"):
            unsplit(*values)


@pytest.mark.parametrize("linker", ("cvm", "numba"))
def test_wide_addition_split_never_reaches_c_or_numba_compiles(linker):
    """C and Numba run wide additions natively, so even their FAST_COMPILE optimizer keeps them.

    A split there would re-round results of graphs that already ran.
    """
    mode = Mode(linker=linker, optimizer="fast_compile")
    if "py_only" in mode.linker.required_rewrites:
        pytest.skip(f"{linker} runs Python thunks here (no C compiler), which need the split")
    xs = [pt.dvector(f"x{i}") for i in range(40)]
    values = [np.random.default_rng(i).normal(size=64) for i in range(40)]
    compiled = pytensor.function(xs, pt.add(*xs), mode=mode)
    unsplit = pytensor.function(
        xs, pt.add(*xs), mode=mode.excluding("pymc_generator_split_wide_add")
    )
    assert _add_widths(compiled) == [40]
    assert compiled(*values).tobytes() == unsplit(*values).tobytes()


@pytest.mark.slow
def test_template_runs_layouts_whose_flattened_parent_sums_exceed_the_python_limit():
    """Every parent group is narrow, but the flattened treatment equation is not.

    Canonicalization flattens own + latent + covariate (+ treatment) terms into one
    addition of more than 31 operands, so a per-call width threshold would still
    fail under FAST_COMPILE here.
    """
    cfg = _cfg(
        n_treatments=2,
        n_covariates=15,
        n_latent=16,
        n_treatments_active_range=(2, 2),
        n_covariates_active_range=(15, 15),
        n_latent_active_range=(16, 16),
    )
    g, _, _ = _concrete_scm_inputs(2, 15, 16, cfg.n_time_steps + cfg.carryover_burn_in)
    for key in ("g_dc", "g_dz", "g_zc"):
        g[key][:] = 1
    active = {
        "active_treatment": np.ones(2),
        "active_covariate": np.ones(15),
        "active_latent": np.ones(16),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(11))
    cell = build_cell_inputs(cfg, g, active, structural)
    model, outputs, reports = build_world_model_template(cfg, cell, cfg.n_time_steps)
    names = tuple(dict.fromkeys((*outputs, *reports, *(rv.name for rv in model.free_RVs))))
    result = compile_template_draw_fn(model, names, mode="FAST_COMPILE")(cell, seed=83)
    for name in outputs:
        assert np.isfinite(result[name]).all(), name
    _assert_ordinary_template_parity(cfg, g, structural, result, outputs)
    np.testing.assert_allclose(
        result["outcome"][0],
        result["baseline"][0]
        + result["contributions"][0].sum(axis=-1)
        + result["indirect_effects"][0],
        rtol=0.0,
        atol=1e-12,
    )
