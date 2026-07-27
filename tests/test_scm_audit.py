"""Focused contracts for the auditable single-world SCM API."""

from __future__ import annotations

import numpy as np

from prior_generator import make_scm_prior, sample_scm
from prior_generator.describe import describe_scm
from prior_generator.symbolic_graph import build_symbolic_graph
from prior_generator.world_model import (
    _MECHANISM_PARAM_NAMES,
    build_world_model,
    draw_worlds,
    sample_structure,
)
from prior_generator.worlds import _LEGACY_WORLD_PARAM_NAMES, SCM, _assemble_params

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
    assert drawn["param_rw_d_mean"].shape == (1, 1)
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
