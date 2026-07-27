"""Focused contracts for the auditable single-world SCM API."""

from __future__ import annotations

import numpy as np
import pytest
from pytensor.graph.traversal import ancestors

import prior_generator.symbolic_graph as symbolic_graph
from prior_generator import make_scm_prior, sample_scm
from prior_generator.describe import describe_scm
from prior_generator.symbolic_graph import build_symbolic_graph
from prior_generator.world_model import (
    _MECHANISM_PARAM_NAMES,
    build_world_model,
    draw_worlds,
    sample_structure,
)
from prior_generator.worlds import (
    _LEGACY_WORLD_PARAM_NAMES,
    SCM,
    _assemble_channel_shock_schedule,
    _assemble_params,
)

_RAW_EPS_NAMES = (
    "eps_d",
    "eps_z",
    "eps_c",
    "eps_b",
    "eps_y",
    "eps_c_hf",
    "eps_c_pulse",
)
_CORE_OUTPUTS = (
    "demand",
    "controls",
    "channels",
    "channels_base",
    "baseline",
    "baseline_intrinsic",
    "contributions",
    "contributions_observed",
    "indirect_effects",
    "indirect_effects_by_source",
    "sales",
)


def _config(**overrides):
    kwargs = {
        "n_treatments": 2,
        "n_covariates": 2,
        "n_latent": 1,
        "T": 16,
        "adstock_burn_in": 4,
        "l_max": 4,
        "edge_budget": {
            "cy": (2, 2),
            "dc": (0, 0),
            "dz": (0, 0),
            "db": (0, 0),
            "zb": (0, 0),
            "zc": (0, 0),
            "cc": (0, 0),
            "zz": (0, 0),
        },
    }
    kwargs.update(overrides)
    return make_scm_prior(**kwargs)


def _fixed_world(g: dict, cfg, *, seed: int = 23) -> SCM:
    rng = np.random.default_rng(seed)
    structural = sample_structure(g, cfg, rng)
    model, out_names, param_names = build_world_model(g, cfg, structural, cfg.T)
    drawn = draw_worlds(model, out_names + param_names + _RAW_EPS_NAMES, seed=seed + 1)
    return SCM(
        data={name: drawn[name][0] for name in out_names},
        g=g,
        params=_assemble_params(drawn, 0, structural, param_names, cfg),
        cfg=cfg,
        extras={"structural": structural},
        _exogenous={name: np.array(drawn[name][0], copy=True) for name in _RAW_EPS_NAMES},
    )


def _replay(
    world: SCM,
    eps_c: np.ndarray,
    **exogenous_overrides: np.ndarray,
) -> dict[str, np.ndarray]:
    eps = world.exogenous
    eps["eps_c"] = eps_c
    eps.update(exogenous_overrides)
    graph = build_symbolic_graph(
        world.g,
        world.params,
        world.T,
        world.K,
        world.M,
        world.J,
        burn_in=world.cfg.adstock_burn_in,
        eps=eps,
    )
    return {name: value.eval() for name, value in graph["outputs"].items()}


def test_sampled_data_owns_accepted_arrays():
    """Accepted outputs must not retain the full candidate-draw batch."""
    world = sample_scm(_config(), seed=17, max_eps_draws=40)

    assert all(values.flags.owndata and values.base is None for values in world.data.values())


def test_channel_shock_schedule_rejects_overlapping_windows():
    """Concrete replay scheduling must retain symbolic sum semantics."""
    cfg = _config(n_channel_shocks=2)
    drawn = {
        "channel_shock_channel": np.array([[0, 0]], dtype="int64"),
        "channel_shock_start": np.array([[0, 1]], dtype="int64"),
        "channel_shock_length": np.array([[2, 2]], dtype="int64"),
        "channel_shock_level": np.array([[1.0, 1.0]]),
        "channel_shock_mask_full": np.zeros((1, cfg.T + cfg.adstock_burn_in, 2), dtype="int8"),
    }

    with pytest.raises(AssertionError, match="must not overlap"):
        _assemble_channel_shock_schedule(drawn, 0, cfg)


def test_expanded_audit_preserves_seeded_single_world_outputs():
    cfg = _config(confounding_strength_range=(0.4, 0.4), spend_cv_floor=0.0)
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 2), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(2, dtype=int),
        "g_zc": np.zeros((2, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((2, 2), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(730))
    model, out_names, param_names = build_world_model(g, cfg, structural, cfg.T)
    legacy_names = out_names + _LEGACY_WORLD_PARAM_NAMES
    expanded_names = out_names + param_names + _RAW_EPS_NAMES

    legacy = draw_worlds(model, legacy_names, seed=731, draws=2)
    expanded = draw_worlds(
        model,
        expanded_names,
        seed=731,
        draws=2,
        rng_reference_names=legacy_names,
    )

    for name in legacy_names:
        np.testing.assert_array_equal(expanded[name], legacy[name])


def test_report_specs_cover_every_continuous_parameter_with_expected_shapes():
    cfg = _config()
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 2), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(2, dtype=int),
        "g_zc": np.zeros((2, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((2, 2), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(5))
    model, _out_names, param_names = build_world_model(g, cfg, structural, cfg.T)
    expected = {
        "beta",
        "w_dc",
        "u_dz",
        "v_zc",
        "alpha_cc",
        "gamma_zz",
        "delta_db",
        "rho_zb",
        *_MECHANISM_PARAM_NAMES,
        "hf_sigma",
        "pulse_amp",
        "pulse_prob",
        "channel_level",
        "confounding_strength",
        *(f"rw_{group}_{stat}" for group in ("d", "z", "c", "b", "y") for stat in ("mean", "std")),
    }
    assert set(param_names) == {f"param_{name}" for name in expected}
    drawn = draw_worlds(model, param_names, seed=7)
    assert drawn["param_w_dc"].shape == (1, 1, 2)
    assert drawn["param_u_dz"].shape == (1, 1, 2)
    # The latent factor is pinned to mean 0 / scale 1 -- still reported, never drawn.
    assert drawn["param_rw_d_mean"] == pytest.approx(0.0)
    assert drawn["param_rw_d_std"] == pytest.approx(1.0)
    assert drawn["param_rw_z_std"].shape == (1, 2)
    assert drawn["param_rw_c_mean"].shape == (1, 2)
    assert drawn["param_rw_b_std"].shape == (1, 1)
    assert drawn["param_rw_y_std"].shape == (1, 1)


def test_combined_confounding_and_shock_world_replays_from_raw_innovations():
    cfg = _config(
        confounding_strength_range=(0.6, 0.6),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.5, 0.5),
    )
    world = sample_scm(cfg, seed=43, max_eps_draws=40)
    T_full = cfg.T + cfg.adstock_burn_in
    exogenous = world.exogenous
    assert {name: value.shape for name, value in exogenous.items()} == {
        "eps_d": (T_full, world.J),
        "eps_z": (T_full, world.M),
        "eps_c": (T_full, world.K),
        "eps_b": (T_full,),
        "eps_y": (T_full,),
        "eps_c_hf": (T_full, world.K),
        "eps_c_pulse": (T_full, world.K),
    }
    schedule = world.params["channel_shock"]
    assert schedule["mask_full"].shape == (T_full, world.K)
    assert np.array_equal(
        schedule["start_full"], world.data["channel_shock_start"] + cfg.adstock_burn_in
    )

    rho = float(world.params["confounding_strength"])
    eps_c_eff = np.sqrt(1.0 - rho**2) * exogenous["eps_c"] + rho * exogenous["eps_b"][:, None]
    replay = _replay(world, eps_c_eff)
    for name in _CORE_OUTPUTS:
        np.testing.assert_allclose(replay[name], world.data[name], rtol=0.0, atol=1e-12)

    raw_replay = _replay(world, exogenous["eps_c"])
    assert not np.allclose(raw_replay["channels"], world.data["channels"], rtol=0.0, atol=1e-12)
    assert not np.allclose(
        world.data["channels"], world.data["channels_unshocked"], rtol=0.0, atol=1e-12
    )
    assert not np.allclose(world.data["sales"], world.data["sales_unshocked"], rtol=0.0, atol=1e-12)
    flipped_pulses = 1 - exogenous["eps_c_pulse"]
    pulse_replay = _replay(world, eps_c_eff, eps_c_pulse=flipped_pulses)
    assert not np.allclose(pulse_replay["channels"], world.data["channels"], rtol=0.0, atol=1e-12)


def test_audit_accessors_are_non_aliasing_and_preserve_replay_and_signal():
    cfg = _config(
        confounding_strength_range=(0.5, 0.5),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.5, 0.5),
    )
    world = sample_scm(cfg, seed=47, max_eps_draws=40)
    rho = float(world.params["confounding_strength"])
    before_eps = world.exogenous
    eps_c_eff = np.sqrt(1.0 - rho**2) * before_eps["eps_c"] + rho * before_eps["eps_b"][:, None]
    replay_before = _replay(world, eps_c_eff)
    signal_before = world.signal()

    exposed_eps = world.exogenous
    exposed_eps["eps_c"][:] = 0.0
    exposed_params = world.equation_parameters
    exposed_params["C1"]["texture"]["hf_sigma"] = 1e9
    exposed_params["channel_shocks"]["mask_full"][:] = 0
    exposed_params["channel_shocks"]["level_full"][:] = 0.0
    exposed_params["C1"]["random_walk"]["std"] = 1e9
    exposed_params["C1"]["response"]["saturation"]["scale"] = 1e9

    assert np.array_equal(world.exogenous["eps_c"], before_eps["eps_c"])
    assert world.equation_parameters["C1"]["texture"]["hf_sigma"] != 1e9
    assert world.equation_parameters["C1"]["random_walk"]["std"] != 1e9
    assert world.equation_parameters["C1"]["response"]["saturation"]["scale"] != 1e9
    assert world.equation_parameters["channel_shocks"]["mask_full"].any()
    replay_after = _replay(world, eps_c_eff)
    for name in _CORE_OUTPUTS:
        np.testing.assert_array_equal(replay_after[name], replay_before[name])
    for name, values in signal_before.items():
        if isinstance(values, np.ndarray):
            np.testing.assert_array_equal(world.signal()[name], values)
        else:
            assert world.signal()[name] == values


def test_equations_use_only_active_parents_and_keep_walk_only_nodes():
    cfg = _config()
    g = {
        "g_cy": np.array([1, 0]),
        "g_dc": np.array([[1, 0]]),
        "g_dz": np.array([[1, 0]]),
        "g_db": np.array([0]),
        "g_zb": np.array([0, 0]),
        "g_zc": np.array([[1, 0], [0, 0]]),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.array([[0, 1], [0, 0]]),
    }
    world = _fixed_world(g, cfg)
    equations = world.equations
    parameters = world.equation_parameters
    assert "D1_full" in equations["Z1"]
    assert "D1_full" not in equations["Z2"]
    assert "Z1_full" in equations["Z2"]
    assert "D1_full" in equations["C1"]
    assert "Z1_full" in equations["C1"]
    assert "D1_full" not in equations["C2"]
    assert "Z1_full" not in equations["C2"]
    assert "parents" not in parameters["C2"]
    assert "parents" not in parameters["B"]
    assert "C2" in equations and "f2" in equations
    assert parameters["C2"]["response"]["gate"]["g_cy"] == 0


def test_response_audit_contains_only_family_specific_shape_parameters():
    cfg = _config()
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 2), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(2, dtype=int),
        "g_zc": np.zeros((2, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((2, 2), dtype=int),
    }
    world = _fixed_world(g, cfg, seed=61)
    cases = (
        (0, 0, {"family", "l_max"}, {"family", "scale"}),
        (1, 1, {"family", "l_max", "alpha"}, {"family", "scale", "slope", "kappa_mult"}),
        (2, 2, {"family", "l_max", "lam", "k"}, {"family", "scale", "lam"}),
        (0, 3, {"family", "l_max"}, {"family", "scale", "kappa_mult"}),
        (1, 4, {"family", "l_max", "alpha"}, {"family", "scale", "c"}),
        (2, 5, {"family", "l_max", "lam", "k"}, {"family", "scale", "alpha"}),
    )
    adstock_sources = {
        0: {},
        1: {"alpha": "adstock_alpha"},
        2: {"lam": "weibull_lam", "k": "weibull_k"},
    }
    saturation_sources = {
        0: {},
        1: {"slope": "hill_slope", "kappa_mult": "hill_kappa_mult"},
        2: {"lam": "logistic_lam"},
        3: {"kappa_mult": "mm_kappa_mult"},
        4: {"c": "tanh_c"},
        5: {"alpha": "root_alpha"},
    }
    for adstock_id, saturation_id, adstock_keys, saturation_keys in cases:
        world.params["adstock_family"][0] = adstock_id
        world.params["sat_family"][0] = saturation_id
        response = world.equation_parameters["C1"]["response"]
        assert set(response["adstock"]) == adstock_keys
        assert set(response["saturation"]) == saturation_keys
        assert response["adstock"]["l_max"] == int(world.params["l_max"])
        assert response["saturation"]["scale"] == float(
            np.asarray(world.data["saturation_scale"])[0]
        )
        for key, source in adstock_sources[adstock_id].items():
            assert response["adstock"][key] == float(np.asarray(world.params[source])[0])
        for key, source in saturation_sources[saturation_id].items():
            assert response["saturation"][key] == float(np.asarray(world.params[source])[0])


def test_description_surfaces_equations_and_audit_locations():
    world = sample_scm(_config(), seed=71, max_eps_draws=40)
    description = describe_scm(world)
    assert "Structural equations (vector-valued; active parents only):" in description
    assert "Exact replay audit:" in description
    assert "world.equation_parameters" in description
    assert "world.exogenous" in description


def test_random_walk_equation_uses_fixed_scale_divisor():
    """The audit equation must match the injective fixed-scale walk implementation."""
    equation = sample_scm(_config(), seed=73, max_eps_draws=40).equations["RW"]

    assert "std(q)" not in equation
    assert "1e-8" not in equation
    assert "centred_walk_scale(T_full, width)" in equation
    assert "sqrt(tr(A A^T) / T_full)" in equation
    assert "world constants:" in equation


def test_description_marks_unestimable_signal_metrics_not_applicable():
    cfg = _config(T=4, l_max=8, adstock_burn_in=0)
    world = _fixed_world(_edgeless_graph(), cfg)

    assert not world.signal()["spearman_valid"].any()
    assert "spearman=n/a" in describe_scm(world)


def _edgeless_graph(n_treatments: int = 2, n_covariates: int = 2) -> dict:
    return {
        "g_cy": np.ones(n_treatments, dtype=int),
        "g_dc": np.zeros((1, n_treatments), dtype=int),
        "g_dz": np.zeros((1, n_covariates), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(n_covariates, dtype=int),
        "g_zc": np.zeros((n_covariates, n_treatments), dtype=int),
        "g_cc": np.zeros((n_treatments, n_treatments), dtype=int),
        "g_zz": np.zeros((n_covariates, n_covariates), dtype=int),
    }


def test_saturation_anchor_has_no_noise_ancestors(monkeypatch):
    """The response anchor must be a function of PARAMETERS, never of the draw.

    Deriving it from the realized series (its window mean) would make the
    "prior" a function of the noise it generates, and -- because the mean spans
    the whole window -- would let spend at a late week move the response at an
    early one. Record every saturation call, then inspect its anchor through
    every graph output, including the audit-only unshocked paths.
    """
    cfg = _config(
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.5, 0.5),
    )
    g = _edgeless_graph()
    structural = sample_structure(g, cfg, np.random.default_rng(5))
    saturation_anchors = []
    original_saturate_col = symbolic_graph._saturate_col

    def record_saturation_anchor(ad_col, mean_ad, params, k):
        saturation_anchors.append(mean_ad)
        return original_saturate_col(ad_col, mean_ad, params, k)

    monkeypatch.setattr(symbolic_graph, "_saturate_col", record_saturation_anchor)
    model, out_names, _param_names = build_world_model(g, cfg, structural, cfg.T)

    assert {"channels_unshocked", "sales_unshocked"} <= set(out_names)
    assert saturation_anchors
    for output_name in out_names:
        output_ancestors = set(ancestors([model[output_name]])) | {model[output_name]}
        output_anchors = [anchor for anchor in saturation_anchors if anchor in output_ancestors]
        if output_name == "sales_unshocked":
            assert output_anchors
        for anchor in output_anchors:
            anchor_ancestors = set(ancestors([anchor])) | {anchor}
            for noise in _RAW_EPS_NAMES:
                assert model[noise] not in anchor_ancestors, f"{output_name}: {noise}"

    # ... while the contributions themselves obviously still depend on the draw.
    assert model["eps_c"] in set(ancestors([model["contributions"]]))


def test_saturation_anchor_equals_the_closed_form_expected_level():
    """Texture-free, upstream-free: the anchor is softplus(softplus(rw_c_mean)).

    The channel walk is already softplus-transformed and the channel equation
    applies a second softplus, so the expected level nests both.
    """
    cfg = _config(channel_hf_sigma_range=(0.0, 0.0), channel_pulse_prob_range=(0.0, 0.0))
    world = sample_scm(cfg, seed=11)
    walk_mean = np.asarray(world.params["rw_c"]["mean"], dtype=float)
    expected = np.logaddexp(0.0, np.logaddexp(0.0, walk_mean))
    np.testing.assert_allclose(
        np.asarray(world.data["saturation_scale"], dtype=float), expected, rtol=1e-12
    )


def test_latent_factor_is_pinned_to_zero_mean_unit_scale():
    """D carries no scale of its own; the loadings do.

    The walk is centred exactly, so the mean is 0 to machine precision. Its
    amplitude is normalized by a CONSTANT rather than by its own realized
    standard deviation, so the realized sd scatters around 1 instead of
    equalling it -- that scatter is the price of keeping the innovations-to-path
    map injective. What is pinned is the expectation.
    """
    cfg = _config(adstock_burn_in=0)
    realized = []
    # One seed from each three-seed block retains a representative moment
    # estimate without retaining 24 costly world draws.
    for seed in (0, 4, 6, 9, 14, 17, 19, 22):
        demand = np.asarray(sample_scm(cfg, seed=seed).data["demand"], dtype=float)
        # burn_in=0 means the reported window IS the simulated horizon.
        np.testing.assert_allclose(demand.mean(axis=0), 0.0, atol=1e-12)
        realized.append(float(demand.std()))
    assert 0.85 <= float(np.mean(np.square(realized))) <= 1.20  # E[var] == 1 by construction
    assert np.ptp(realized) > 0.1  # ... and it is genuinely random, not pinned
