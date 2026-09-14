"""The one-compile-per-shard template path.

The template trades a denser graph for compiling once per shard instead of once
per cell, by making the DAG, the mechanism families, and the random-walk kernel
widths run-time inputs. These tests pin the three equivalences that trade rests
on — the walk-by-width identity, the dynamic-vs-static graph, and per-cell
structure actually taking effect — plus the guards that caught real bugs while it
was being built.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytest

import prior_generator.world_model as world_model
from prior_generator import make_scm_prior
from prior_generator.random_walk import (
    _walk_basis_stack,
    symbolic_random_walk,
    symbolic_random_walk_by_width,
    walk_width_index,
)
from prior_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _slice_g_active,
    sample_g_additive,
)
from prior_generator.symbolic_graph import build_symbolic_graph
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure
from prior_generator.world_model_template import (
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
    """One compiled template plus the cells it will be driven with."""
    cfg = _cfg()
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
    adstock_family = np.array([2, 1, 0][:n_treatments], dtype="int64")
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
        "adstock_family": adstock_family,
        "sat_family": sat_family,
        "adstock_alpha": rng.uniform(0.2, 0.7, n_treatments),
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
        "control_hf_sigma": rng.uniform(0.05, 0.2, n_covariates),
        "control_pulse_amp": rng.uniform(0.1, 0.4, n_covariates),
        "control_pulse_prob": rng.uniform(0.1, 0.3, n_covariates),
        "use_control_hf": np.ones(n_covariates, dtype=bool),
        "use_control_pulse": np.ones(n_covariates, dtype=bool),
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
    dyn_params["adstock_family"] = pt.as_tensor_variable(params["adstock_family"])
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


def test_disabled_control_texture_needs_no_control_noise_and_enabled_texture_demands_it():
    """Direct concrete callers may omit the control-texture innovations.

    A config with the texture off builds the pre-texture control equation, so
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
    off["use_control_hf"] = np.zeros(n_covariates, dtype=bool)
    off["use_control_pulse"] = np.zeros(n_covariates, dtype=bool)
    without_noise = {k: v for k, v in eps.items() if k not in ("eps_z_hf", "eps_z_pulse")}

    def controls(params_in, eps_in):
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
        return np.asarray(graph["outputs"]["controls"].eval(), dtype="float64")

    np.testing.assert_array_equal(controls(off, without_noise), controls(off, eps))
    # ... and the enabled graph really consumes them, so it cannot be equal.
    assert not np.allclose(controls(params, eps), controls(off, eps), rtol=0.0, atol=1e-12)

    for flag, missing in (("use_control_hf", "eps_z_hf"), ("use_control_pulse", "eps_z_pulse")):
        enabled_one = dict(off)
        enabled_one[flag] = np.ones(n_covariates, dtype=bool)
        with pytest.raises(ValueError, match=missing):
            controls(enabled_one, {k: v for k, v in eps.items() if k != missing})


def test_absent_control_flags_are_derived_from_the_concrete_magnitudes():
    """A caller may omit the flags entirely; magnitudes then decide.

    ``build_world_model`` always passes ``use_control_*``, but a direct concrete
    caller may not, so the graph derives them: hf from a nonzero sigma, and the
    pulse from a nonzero amplitude AND a positive fire probability (an amplitude
    with probability zero can never fire).
    """
    n_treatments, n_covariates, n_latent = 2, 2, 1
    n_time_steps, burn_in = 16, 8
    g, params, eps = _concrete_scm_inputs(
        n_treatments, n_covariates, n_latent, n_time_steps + burn_in
    )

    def controls(params_in, eps_in):
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
        return np.asarray(graph["outputs"]["controls"].eval(), dtype="float64")

    derived = {k: v for k, v in params.items() if k not in ("use_control_hf", "use_control_pulse")}
    np.testing.assert_array_equal(controls(derived, eps), controls(params, eps))

    # A live amplitude with a zero fire probability stays OFF, so its innovation
    # is not even required.
    dead_pulse = dict(derived)
    dead_pulse["control_pulse_prob"] = np.zeros(n_covariates)
    dead_pulse["control_hf_sigma"] = np.zeros(n_covariates)
    off = dict(params)
    off["use_control_hf"] = np.zeros(n_covariates, dtype=bool)
    off["use_control_pulse"] = np.zeros(n_covariates, dtype=bool)
    np.testing.assert_array_equal(
        controls(dead_pulse, {k: v for k, v in eps.items() if k != "eps_z_pulse"}),
        controls(off, eps),
    )

    # A derived-on term with no innovation is still a loud error.
    with pytest.raises(ValueError, match="eps_z_hf"):
        controls(derived, {k: v for k, v in eps.items() if k != "eps_z_hf"})


def test_unused_response_priors_do_not_change_concrete_worlds():
    """An unused geometric prior cannot change a Weibull world's random draws."""
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        adstock_family_probs={"none": 0.0, "geometric": 0.0, "weibull": 1.0},
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
        cfg.adstock_alpha_range = alpha_range
        model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.n_time_steps)
        results.append(draw_worlds(model, out_names, seed=37, draws=2))
    for name in out_names:
        np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)


# -- the template as a whole ----------------------------------------------------


def test_template_satisfies_the_additive_identity_on_every_world(template):
    """The exact decomposition must survive the denser dynamic graph."""
    cfg, cells, draw = template
    for i, cell in enumerate(cells):
        drawn = draw(cell, seed=500 + i, draws=cfg.draws_per_cell)
        for d in range(cfg.draws_per_cell):
            w = {k: np.asarray(v[d], dtype="float64") for k, v in drawn.items()}
            residual = (
                w["baseline_intrinsic"]
                + w["sales_noise"]
                + w["confounder_contribution"].sum(-1)
                + w["control_contribution"].sum(-1)
                + w["contributions"].sum(-1)
                + w["indirect_effects_by_source"].sum(-1)
                - w["sales"]
            )
            assert np.abs(residual).max() < 1e-9


def test_template_zeroes_inactive_node_slots(template):
    """Padded slots must be exactly zero, not merely small.

    The template runs at max size, so a cell using fewer nodes carries padded
    columns. They are switched off inside the graph rather than trimmed
    afterwards, and consumers read them as real zeros. Every node family is
    tracked separately: control texture adds a term inside the control mask, so
    a guard satisfied by an inactive channel alone would prove nothing about it.
    """
    cfg, cells, draw = template
    keys = (
        ("channels", "active_treatment"),
        ("controls", "active_covariate"),
        ("control_contribution", "active_covariate"),
        ("demand", "active_latent"),
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
        cfg.n_time_steps + cfg.adstock_burn_in, cfg.rw_smoothness_max_weeks
    ).shape[0]
    shifted["rw_width_c"] = other

    a = draw(base, seed=99, draws=1)["channels"][0]
    b = draw(shifted, seed=99, draws=1)["channels"][0]
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


def test_template_channels_are_non_negative(template):
    """The softplus positivity guard must survive the dynamic path."""
    cfg, cells, draw = template
    for i, cell in enumerate(cells):
        drawn = draw(cell, seed=800 + i)
        assert np.asarray(drawn["channels"]).min() >= 0.0


@pytest.mark.parametrize(
    "overrides, match",
    (
        ({"n_channel_shocks": 1}, "channel shocks"),
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
            draw = compile_template_draw_fn(model, ("sales", "channels"))
            results[enabled] = draw(cells[0], seed=5, draws=2)
        finally:
            world_model.set_compile_cache_enabled(True)
    for name in ("sales", "channels"):
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
            n_channel_shocks=1,
            channel_shock_length_range=(3, 3),
            channel_shock_level_range=(hi, hi),
        )
        rng = np.random.default_rng(cfg.seed)
        g = sample_g_additive(rng, cfg, cfg.layout)
        g_act = _slice_g_active(g, 2, 2, 1)
        structural = sample_structure(g_act, cfg, rng)
        model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.n_time_steps)
        drawn = draw_worlds(model, ("channel_shock_level",), seed=5, draws=1)
        models.append(np.asarray(drawn["channel_shock_level"]))

    assert np.all(models[0] == 0.0), "a zero level multiplier must produce zero levels"
    assert np.all(models[1] > 0.0), "a nonzero level multiplier must produce nonzero levels"
