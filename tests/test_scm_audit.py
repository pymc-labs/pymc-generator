"""Focused contracts for the auditable single-world SCM API."""

from __future__ import annotations

import numpy as np
import pytest
from pytensor.graph.traversal import ancestors

import pymc_generator.symbolic_graph as symbolic_graph
from pymc_generator import make_scm_prior, sample_scm
from pymc_generator.describe import describe_scm
from pymc_generator.random_walk import _kernel_width
from pymc_generator.sampler import _additive_task_ok
from pymc_generator.symbolic_graph import build_symbolic_graph
from pymc_generator.world_model import (
    _MECHANISM_PARAM_NAMES,
    _walk_basis,
    build_world_model,
    draw_worlds,
    sample_structure,
)
from pymc_generator.worlds import (
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
    "eps_z_hf",
    "eps_z_pulse",
)
_CORE_OUTPUTS = (
    "demand",
    "controls",
    "channels",
    "channels_base",
    "baseline",
    "baseline_intrinsic",
    "sales_noise",
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
        "n_time_steps": 16,
        "adstock_burn_in": 4,
        "l_max": 4,
        "edge_budget": {
            "cy": (2, 2),
            "dc": (0, 0),
            "dz": (0, 0),
            "dy": (0, 0),
            "zy": (0, 0),
            "zc": (0, 0),
            "cc": (0, 0),
            "zz": (0, 0),
        },
    }
    kwargs.update(overrides)
    return make_scm_prior(**kwargs)


def test_sampled_world_owns_its_configuration():
    cfg = _config(baseline_floor=0.0)
    world = sample_scm(cfg, seed=7)
    before = world.equations
    cfg.baseline_floor = 123.0
    cfg.edge_budget["cy"] = (0, 0)
    assert world.equations == before
    assert world.equation_parameters["B"]["floor"] == 0.0
    assert world.cfg.edge_budget["cy"] == (2, 2)


@pytest.mark.parametrize("scope", ["intercept", "non_media"])
def test_equation_audit_identifies_floor_operation(scope):
    world = sample_scm(_config(baseline_floor=0.0, baseline_floor_scope=scope), seed=7)
    audit = world.equation_parameters["Y"]["non_media"]
    assert audit["floor_scope"] == scope
    assert audit["accumulation_order"] == ["B"]
    assert audit["attribution"] == (
        "sequential_clipped_differences" if scope == "non_media" else "additive"
    )


def _fixed_world(g: dict, cfg, *, seed: int = 23) -> SCM:
    rng = np.random.default_rng(seed)
    structural = sample_structure(g, cfg, rng)
    model, out_names, param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
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
        world.n_time_steps,
        world.n_treatments,
        world.n_covariates,
        world.n_latent,
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
        "channel_shock_mask_full": np.zeros(
            (1, cfg.n_time_steps + cfg.adstock_burn_in, 2), dtype="int8"
        ),
    }

    with pytest.raises(AssertionError, match="must not overlap"):
        _assemble_channel_shock_schedule(drawn, 0, cfg)


def test_expanded_audit_preserves_seeded_single_world_outputs():
    cfg = _config(confounding_strength_range=(0.4, 0.4), spend_cv_floor=0.0)
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 2), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(2, dtype=int),
        "g_zc": np.zeros((2, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((2, 2), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(730))
    model, out_names, param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
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
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(2, dtype=int),
        "g_zc": np.zeros((2, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((2, 2), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(5))
    model, _out_names, param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
    expected = {
        "beta",
        "w_dc",
        "u_dz",
        "v_zc",
        "alpha_cc",
        "gamma_zz",
        "delta_dy",
        "rho_zy",
        *_MECHANISM_PARAM_NAMES,
        "hf_sigma",
        "pulse_amp",
        "pulse_prob",
        "control_hf_sigma",
        "control_pulse_amp",
        "control_pulse_prob",
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
    assert drawn["param_control_hf_sigma"].shape == (1, 2)
    assert drawn["param_control_pulse_amp"].shape == (1, 2)
    assert drawn["param_control_pulse_prob"].shape == (1, 2)


def test_combined_confounding_and_shock_world_replays_from_raw_innovations():
    cfg = _config(
        confounding_strength_range=(0.2, 0.4),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 3),
        channel_shock_level_range=(0.5, 1.0),
    )
    world = sample_scm(cfg, seed=43, max_eps_draws=40)
    n_time_steps_full = cfg.n_time_steps + cfg.adstock_burn_in
    exogenous = world.exogenous
    assert {name: value.shape for name, value in exogenous.items()} == {
        "eps_d": (n_time_steps_full, world.n_latent),
        "eps_z": (n_time_steps_full, world.n_covariates),
        "eps_c": (n_time_steps_full, world.n_treatments),
        "eps_b": (n_time_steps_full,),
        "eps_y": (n_time_steps_full,),
        "eps_c_hf": (n_time_steps_full, world.n_treatments),
        "eps_c_pulse": (n_time_steps_full, world.n_treatments),
        "eps_z_hf": (n_time_steps_full, world.n_covariates),
        "eps_z_pulse": (n_time_steps_full, world.n_covariates),
    }
    schedule = world.params["channel_shock"]
    assert schedule["mask_full"].shape == (n_time_steps_full, world.n_treatments)
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
    exposed_eps["eps_z_hf"][:] = 0.0
    exposed_eps["eps_z_pulse"][:] = 1.0
    exposed_params = world.equation_parameters
    exposed_params["C1"]["texture"]["hf_sigma"] = 1e9
    exposed_params["channel_shocks"]["mask_full"][:] = 0
    exposed_params["channel_shocks"]["level_full"][:] = 0.0
    exposed_params["C1"]["random_walk"]["std"] = 1e9
    exposed_params["C1"]["response"]["saturation"]["scale"] = 1e9
    exposed_params["Z1"]["texture"]["control_hf_sigma"] = 1e9

    assert np.array_equal(world.exogenous["eps_c"], before_eps["eps_c"])
    assert np.array_equal(world.exogenous["eps_z_hf"], before_eps["eps_z_hf"])
    assert np.array_equal(world.exogenous["eps_z_pulse"], before_eps["eps_z_pulse"])
    assert world.equation_parameters["Z1"]["texture"]["control_hf_sigma"] != 1e9
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
        "g_dy": np.array([0]),
        "g_zy": np.array([0, 0]),
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
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(2, dtype=int),
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
    assert "centred_walk_scale(n_time_steps_full, width)" in equation
    assert "sqrt(tr(A A^T) / n_time_steps_full)" in equation
    assert "world constants:" in equation


def test_description_marks_unestimable_signal_metrics_not_applicable():
    cfg = _config(n_time_steps=4, l_max=8, adstock_burn_in=0)
    world = _fixed_world(_edgeless_graph(), cfg)

    assert not world.signal()["spearman_valid"].any()
    assert "spearman=n/a" in describe_scm(world)


def _edgeless_graph(n_treatments: int = 2, n_covariates: int = 2) -> dict:
    return {
        "g_cy": np.ones(n_treatments, dtype=int),
        "g_dc": np.zeros((1, n_treatments), dtype=int),
        "g_dz": np.zeros((1, n_covariates), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(n_covariates, dtype=int),
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

    def record_saturation_anchor(ad_col, saturation_scale, params, k):
        saturation_anchors.append(saturation_scale)
        return original_saturate_col(ad_col, saturation_scale, params, k)

    monkeypatch.setattr(symbolic_graph, "_saturate_col", record_saturation_anchor)
    model, out_names, _param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)

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


def test_relative_outcome_scales_have_no_noise_ancestors():
    """The absolute outcome scales must depend on parameters, never innovations."""
    cfg = _config(
        outcome_std_mode="relative",
        rw_baseline_std_range=(0.04, 0.08),
        rw_sales_std_range=(0.01, 0.03),
    )
    g = _edgeless_graph()
    structural = sample_structure(g, cfg, np.random.default_rng(6))
    model, _out_names, _param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)

    for scale_name in ("rw_b_std", "rw_y_std"):
        scale_ancestors = set(ancestors([model[scale_name]])) | {model[scale_name]}
        for noise in _RAW_EPS_NAMES:
            assert model[noise] not in scale_ancestors, f"{scale_name}: {noise}"


def test_rw_y_is_iid_and_cannot_share_a_walk_operator_with_rw_b():
    """Y has no smoothness metadata and responds pointwise to ``eps_y``."""
    cfg = _config()
    world = _fixed_world(_edgeless_graph(), cfg, seed=29)

    assert "smoothness_y" not in world.extras["structural"]
    assert "smoothness" not in world.params["rw_y"]
    assert "rw_smoothness_max_weeks" not in world.params["rw_y"]
    assert "smoothness" not in world.equation_parameters["Y"]["iid_noise"]
    assert "RW_full(eps_y" not in world.equations["Y"]

    replacement_eps_y = np.linspace(-1.0, 1.0, world.n_time_steps + cfg.adstock_burn_in)
    replay = _replay(
        world,
        world.exogenous["eps_c"],
        eps_y=replacement_eps_y,
    )
    std = float(np.asarray(world.params["rw_y"]["std"])[0])
    expected_change = std * (
        replacement_eps_y[cfg.adstock_burn_in :] - world.exogenous["eps_y"][cfg.adstock_burn_in :]
    )
    # The noise is its own column now, so it moves ``sales_noise`` and
    # ``baseline`` pointwise and leaves the intercept target alone.
    np.testing.assert_allclose(
        replay["sales_noise"] - world.data["sales_noise"],
        expected_change,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        replay["baseline"] - world.data["baseline"], expected_change, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        replay["baseline_intrinsic"], world.data["baseline_intrinsic"], rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("outcome_std_mode", ("relative", "absolute"))
def test_outcome_noise_modes_preserve_the_scm_identity(outcome_std_mode):
    world = sample_scm(
        _config(outcome_std_mode=outcome_std_mode),
        seed=37,
        max_eps_draws=40,
    )
    assert world.identity_error() < 1e-9


def test_saturation_anchor_equals_the_closed_form_reference_level():
    """Texture-free, upstream-free: the anchor is softplus(softplus(rw_c_mean)).

    That closed form is the PARAMETER-ONLY reference level, not ``E[C]``: the
    channel walk is already softplus-transformed and the channel equation
    applies a second softplus, so the anchor nests ``softplus`` around a MEAN
    where the realized channel takes the MEAN of a ``softplus``. Softplus is
    strictly convex, so ``E[C] > anchor`` strictly; this test pins the anchor's
    closed form, not any moment of the drawn series.
    """
    cfg = _config(channel_hf_sigma_range=(0.0, 0.0), channel_pulse_prob_range=(0.0, 0.0))
    world = sample_scm(cfg, seed=11)
    walk_mean = np.asarray(world.params["rw_c"]["mean"], dtype=float)
    expected = np.logaddexp(0.0, np.logaddexp(0.0, walk_mean))
    np.testing.assert_allclose(
        np.asarray(world.data["saturation_scale"], dtype=float), expected, rtol=1e-12
    )


def test_control_texture_is_relative_to_the_control_walk_std():
    """Both magnitudes are drawn as a factor of the control's OWN walk std.

    A signed control has no positive level to anchor on (``rw_z_mean`` straddles
    zero), so the scale-free anchor is its walk amplitude. Recovering the drawn
    factor from the reported magnitude proves the scaling is applied once, with
    the right denominator.
    """
    cfg = _config(
        control_hf_sigma_range=(0.2, 0.6),
        control_pulse_prob_range=(0.05, 0.25),
        control_pulse_amp_range=(0.5, 2.0),
    )
    world = sample_scm(cfg, seed=13)
    walk_std = np.asarray(world.params["rw_z"]["std"], dtype=float)
    assert (walk_std > 0.0).all()

    hf_factor = np.asarray(world.params["control_hf_sigma"], dtype=float) / walk_std
    amp_factor = np.asarray(world.params["control_pulse_amp"], dtype=float) / walk_std
    prob = np.asarray(world.params["control_pulse_prob"], dtype=float)
    for factor, (lo, hi) in (
        (hf_factor, cfg.control_hf_sigma_range),
        (amp_factor, cfg.control_pulse_amp_range),
        (prob, cfg.control_pulse_prob_range),
    ):
        assert ((factor >= lo) & (factor <= hi)).all(), (factor, lo, hi)
    assert world.params["use_control_hf"].all()
    assert world.params["use_control_pulse"].all()

    # Containment alone would survive a per-node mixup, so pin the identity on a
    # degenerate-range world: the magnitude is EXACTLY factor * that control's
    # own walk std, and the fire probability is never scaled.
    pinned = sample_scm(
        _config(
            control_hf_sigma_range=(0.4, 0.4),
            control_pulse_prob_range=(0.2, 0.2),
            control_pulse_amp_range=(1.5, 1.5),
        ),
        seed=13,
    )
    pinned_std = np.asarray(pinned.params["rw_z"]["std"], dtype=float)
    assert np.ptp(pinned_std) > 1e-6  # the controls differ, so alignment is testable
    np.testing.assert_allclose(
        np.asarray(pinned.params["control_hf_sigma"], dtype=float),
        0.4 * pinned_std,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pinned.params["control_pulse_amp"], dtype=float),
        1.5 * pinned_std,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pinned.params["control_pulse_prob"], dtype=float), 0.2, rtol=1e-12
    )


def test_control_pulse_is_centred_on_its_own_fire_probability():
    """The exact per-week own-drive delta is ``amp * (fire - prob)``.

    Centring is what keeps a control's expected level at ``rw_z_mean``, so the
    parameter-only saturation anchors stay valid. An uncentred pulse (the
    channel form) would shift every control by ``+amp * prob``.
    """
    cfg = _config(
        control_hf_sigma_range=(0.3, 0.3),
        control_pulse_prob_range=(0.2, 0.2),
        control_pulse_amp_range=(1.5, 1.5),
    )
    world = _fixed_world(_edgeless_graph(), cfg, seed=31)
    exogenous = world.exogenous
    amp = np.asarray(world.params["control_pulse_amp"], dtype=float)
    sigma = np.asarray(world.params["control_hf_sigma"], dtype=float)
    prob = np.asarray(world.params["control_pulse_prob"], dtype=float)
    zeros = np.zeros_like(exogenous["eps_z_pulse"])

    # The controls are edgeless here, so each column IS its own drive. The
    # reference is the SAME world with both texture flags off, so the deltas
    # below are the texture's exact contribution, not a restatement of it.
    off_params = dict(world.params)
    off_params["use_control_hf"] = np.zeros(world.n_covariates, dtype=bool)
    off_params["use_control_pulse"] = np.zeros(world.n_covariates, dtype=bool)
    walk_only = np.asarray(
        build_symbolic_graph(
            world.g,
            off_params,
            world.n_time_steps,
            world.n_treatments,
            world.n_covariates,
            world.n_latent,
            burn_in=cfg.adstock_burn_in,
            eps=exogenous,
        )["outputs"]["controls"].eval(),
        dtype=float,
    )
    never = _replay(world, exogenous["eps_c"], eps_z_hf=zeros, eps_z_pulse=zeros)["controls"]
    always = _replay(world, exogenous["eps_c"], eps_z_hf=zeros, eps_z_pulse=np.ones_like(zeros))[
        "controls"
    ]

    weeks = never.shape[0]
    np.testing.assert_allclose(
        never - walk_only, np.broadcast_to(-amp * prob, (weeks, amp.size)), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        always - walk_only,
        np.broadcast_to(amp * (1.0 - prob), (weeks, amp.size)),
        rtol=0.0,
        atol=1e-12,
    )
    # Probability-weighted, the pulse term contributes exactly nothing.
    np.testing.assert_allclose(
        prob * (always - walk_only) + (1.0 - prob) * (never - walk_only),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    # One spiked COLUMN: the control must read its own innovation column, at its
    # own magnitude, in that week only.
    spike = np.zeros_like(exogenous["eps_z_hf"])
    week = cfg.adstock_burn_in + 3
    spike[week, 0] = 1.0
    spiked = _replay(world, exogenous["eps_c"], eps_z_hf=spike, eps_z_pulse=zeros)["controls"]
    delta = spiked - never
    expected_week = np.zeros(world.n_covariates)
    expected_week[0] = sigma[0]
    np.testing.assert_allclose(
        delta[week - cfg.adstock_burn_in], expected_week, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.delete(delta, week - cfg.adstock_burn_in, axis=0), 0.0, rtol=0.0, atol=1e-12
    )

    # Same for the pulse: one firing COLUMN moves only that control, by amp.
    one_fire = np.zeros_like(exogenous["eps_z_pulse"])
    one_fire[:, 0] = 1.0
    fired = _replay(world, exogenous["eps_c"], eps_z_hf=zeros, eps_z_pulse=one_fire)["controls"]
    expected_fire = np.zeros(world.n_covariates)
    expected_fire[0] = amp[0]
    np.testing.assert_allclose(
        fired - never,
        np.broadcast_to(expected_fire, (weeks, world.n_covariates)),
        rtol=0.0,
        atol=1e-12,
    )


def test_control_texture_leaves_the_parameter_only_saturation_anchor_exact():
    """Enabled control texture must not be ADDED to the κ anchor.

    The anchor sums PARAMETER-only reference levels, and a control's level claim
    is EXACT — a control applies no activation, and both texture terms are
    mean-zero, so ``E[Z_m]`` is unchanged. The live check here is the "do not
    mirror the uncentred channel pulse" one: a
    ``+ amp * prob`` correction on the control levels would move the anchor by a
    measurable amount, asserted below. The graph-side centring that justifies
    the omission is pinned by
    ``test_control_pulse_is_centred_on_its_own_fire_probability``.
    """
    cfg = _config(
        channel_hf_sigma_range=(0.0, 0.0),
        channel_pulse_prob_range=(0.0, 0.0),
        control_hf_sigma_range=(0.4, 0.8),
        control_pulse_prob_range=(0.2, 0.25),
        control_pulse_amp_range=(2.0, 3.0),
        edge_budget={
            "cy": (2, 2),
            "dc": (0, 0),
            "dz": (0, 0),
            "dy": (0, 0),
            "zy": (2, 2),
            "zc": (2, 2),
            "cc": (0, 0),
            "zz": (0, 0),
        },
    )
    world = sample_scm(cfg, seed=17)
    params, g = world.params, world.g
    z_levels = np.asarray(params["rw_z"]["mean"], dtype=float)  # no Z->Z here
    own = np.logaddexp(0.0, np.asarray(params["rw_c"]["mean"], dtype=float))
    upstream = (np.asarray(g["g_zc"], dtype=float) * np.asarray(params["v_zc"], dtype=float)).T @ (
        z_levels
    )
    expected = np.logaddexp(0.0, own + upstream)

    assert np.asarray(g["g_zc"]).sum() == 2  # the Z->C terms are live, not vacuous
    assert world.params["use_control_pulse"].all()
    np.testing.assert_allclose(
        np.asarray(world.data["saturation_scale"], dtype=float), expected, rtol=1e-12
    )

    # The uncentred-channel-style alternative is materially different, so the
    # assertion above is not satisfied by a negligible term.
    pulse_mean = np.asarray(params["control_pulse_amp"], dtype=float) * np.asarray(
        params["control_pulse_prob"], dtype=float
    )
    mirrored = np.logaddexp(
        0.0,
        own
        + (np.asarray(g["g_zc"], dtype=float) * np.asarray(params["v_zc"], dtype=float)).T
        @ (z_levels + pulse_mean),
    )
    assert np.abs(mirrored - expected).max() > 1e-3


def test_intercept_is_censored_at_the_floor_and_the_parents_stay_exact():
    """A floored intercept hits zero and never goes below it.

    The floor clips ONE additive term, so every other decomposition column is
    untouched: the identity still closes and the per-node baseline terms stay
    exactly ``coefficient x node``. Flooring a sum that contained D and Z could
    not offer either guarantee.
    """
    # Low intercept level + a wide absolute-mode walk, so the floor really binds.
    # Live zy/dy edges too, so the "parents stay exact" check is not vacuous.
    stress = {
        "outcome_std_mode": "absolute",
        "rw_baseline_mean_range": (0.5, 1.5),
        "rw_baseline_std_sigma": 1.5,
        "rw_sales_std_sigma": 0.05,
        "n_time_steps": 52,
        "adstock_burn_in": 4,
        "l_max": 4,
        "edge_budget": {
            "cy": (2, 2),
            "dc": (0, 0),
            "dz": (0, 0),
            "dy": (1, 1),
            "zy": (2, 2),
            "zc": (0, 0),
            "cc": (0, 0),
            "zz": (0, 0),
        },
    }
    signed_cfg = _config(**stress)
    floored_cfg = _config(baseline_floor=0.0, **stress)
    for seed in range(40, 60):
        signed = sample_scm(signed_cfg, seed=seed, max_eps_draws=40)
        signed_intercept = np.asarray(signed.data["baseline_intrinsic"], dtype=float)
        if (signed_intercept < 0.0).any():
            break
    else:  # pragma: no cover - the stress fixture is calibrated to dip
        pytest.fail("no seed produced a negative intercept; the floor would prove nothing")
    floored = sample_scm(floored_cfg, seed=seed, max_eps_draws=40)
    floored_intercept = np.asarray(floored.data["baseline_intrinsic"], dtype=float)
    assert (floored_intercept >= 0.0).all()
    assert (floored_intercept == 0.0).any(), "a censored walk must be able to sit AT the floor"

    # Exactly a clip of the same path: unclipped weeks are bit-identical.
    unclipped = signed_intercept > 0.0
    np.testing.assert_allclose(
        floored_intercept[unclipped], signed_intercept[unclipped], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        floored_intercept, np.maximum(signed_intercept, 0.0), rtol=0.0, atol=1e-12
    )

    # The floor does not disturb any other column.
    assert floored.identity_error() < 1e-9
    params, g = floored.params, floored.g
    for m in range(floored.n_covariates):
        expected = (
            float(np.asarray(g["g_zy"])[m])
            * float(np.asarray(params["rho_zy"])[m])
            * np.asarray(floored.data["controls"], dtype=float)[:, m]
        )
        np.testing.assert_allclose(
            np.asarray(floored.data["control_contribution"], dtype=float)[:, m],
            expected,
            rtol=0.0,
            atol=1e-12,
        )


def test_sales_is_never_censored_and_the_filter_carries_non_negativity():
    """Y stays uncensored: a clamp there would censor the OBSERVATION.

    ``sales == baseline + Σ observed contributions`` exactly, with no clip, so
    every world stays inside the additive function class an MMM likelihood can
    represent. Non-negative sales is the acceptance filter's job.
    """
    world = sample_scm(_config(baseline_floor=0.0), seed=51, max_eps_draws=40)
    d = world.data
    np.testing.assert_allclose(
        np.asarray(d["sales"], dtype=float),
        np.asarray(d["baseline"], dtype=float)
        + np.asarray(d["contributions_observed"], dtype=float).sum(1),
        rtol=0.0,
        atol=1e-12,
    )
    assert (np.asarray(d["sales"], dtype=float) >= 0.0).all()
    # The filter is what rejects a negative-sales draw, so it must still bite.
    assert not _additive_task_ok(
        spend=np.asarray(d["channels"], dtype=float),
        sales=np.asarray(d["sales"], dtype=float) - float(d["sales"].max()) - 1.0,
        arrays={"sales": np.asarray(d["sales"], dtype=float)},
        g_cy_active=world.g["g_cy"],
        cv_floor=0.0,
    )


def _absorbing_stress(**overrides):
    """Live zy/dy edges and a low, wide intercept: the floor really binds."""
    stress = {
        "outcome_std_mode": "absolute",
        "rw_baseline_mean_range": (0.5, 1.5),
        "rw_baseline_std_sigma": 1.5,
        "rw_sales_std_sigma": 0.05,
        "rw_std_sigma": 2.0,
        "n_time_steps": 52,
        "n_covariates": 3,
        "n_latent": 2,
        "edge_budget": {
            "cy": (2, 2),
            "dc": (0, 0),
            "dz": (0, 0),
            "dy": (2, 2),
            "zy": (3, 3),
            "zc": (0, 0),
            "cc": (0, 0),
            "zz": (0, 0),
        },
    }
    stress.update(overrides)
    return _config(**stress)


def _non_media(world) -> np.ndarray:
    d = world.data
    return (
        np.asarray(d["baseline_intrinsic"], dtype=float)
        + np.asarray(d["control_contribution"], dtype=float).sum(1)
        + np.asarray(d["confounder_contribution"], dtype=float).sum(1)
    )


def test_absorbing_floor_makes_the_whole_non_media_total_non_negative():
    """A negative control effect is credited only down to the floor.

    Flooring the intercept alone cannot stop a large negative ``rho_zy * Z``
    from dragging the non-media total under; ``scope="non_media"`` clips the
    running total instead, so the excess is absorbed. The per-node columns
    become the telescoping difference each node caused, so they still sum
    EXACTLY to the total.
    """
    intercept_only = _absorbing_stress(baseline_floor=0.0)
    absorbing = _absorbing_stress(baseline_floor=0.0, baseline_floor_scope="non_media")

    for seed in range(60, 80):
        signed = sample_scm(intercept_only, seed=seed, connect_all=False, max_eps_draws=40)
        if (_non_media(signed) < -1e-12).any():
            break
    else:  # pragma: no cover - the stress fixture is calibrated to dip
        pytest.fail("intercept-only scope never dipped; the comparison would prove nothing")

    floored = sample_scm(absorbing, seed=seed, connect_all=False, max_eps_draws=40)
    total = _non_media(floored)
    assert (total >= -1e-12).all(), "the absorbing scope must hold the whole total at the floor"
    assert np.isclose(total, 0.0, atol=1e-12).any(), "the total must be able to sit AT the floor"

    # The columns are a split of THAT total, not of the unclipped one.
    np.testing.assert_allclose(
        total,
        np.asarray(floored.data["baseline"], dtype=float)
        - np.asarray(floored.data["sales_noise"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    assert floored.identity_error() < 1e-9
    # Clipping can only raise the total, never lower it.
    assert (total >= _non_media(signed) - 1e-12).all()


@pytest.mark.parametrize("scope", ("intercept", "non_media"))
def test_a_floor_that_never_binds_leaves_every_column_alone(scope):
    """Both scopes are clips, so a non-binding floor changes nothing material."""
    plain = sample_scm(_config(), seed=9, max_eps_draws=40)
    assert (np.asarray(plain.data["baseline_intrinsic"], dtype=float) > 0.0).all()
    assert (_non_media(plain) > 0.0).all(), "fixture already dips; pick a calmer one"

    floored = sample_scm(
        _config(baseline_floor=0.0, baseline_floor_scope=scope), seed=9, max_eps_draws=40
    )
    for key in (
        "sales",
        "baseline",
        "baseline_intrinsic",
        "sales_noise",
        "control_contribution",
        "confounder_contribution",
    ):
        np.testing.assert_allclose(
            np.asarray(floored.data[key], dtype=float),
            np.asarray(plain.data[key], dtype=float),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"a non-binding floor moved {key!r}",
        )


def test_intercept_and_sales_noise_are_separate_decomposition_columns():
    """``baseline_intrinsic`` is the intercept ALONE; the noise is its own column."""
    world = sample_scm(_config(), seed=57, max_eps_draws=40)
    d, params = world.data, world.params
    burn_in = world.cfg.adstock_burn_in
    expected_noise = (
        float(np.asarray(params["rw_y"]["std"])[0])
        * np.asarray(world.exogenous["eps_y"], dtype=float)[burn_in:]
    )
    np.testing.assert_allclose(
        np.asarray(d["sales_noise"], dtype=float), expected_noise, rtol=0.0, atol=1e-12
    )
    # intrinsic carries no observation noise any more, and no parent terms.
    assert not np.allclose(
        np.asarray(d["baseline_intrinsic"], dtype=float),
        np.asarray(d["baseline_intrinsic"], dtype=float) + expected_noise,
    )
    np.testing.assert_allclose(
        np.asarray(d["baseline"], dtype=float),
        np.asarray(d["baseline_intrinsic"], dtype=float)
        + np.asarray(d["sales_noise"], dtype=float)
        + np.asarray(d["control_contribution"], dtype=float).sum(1)
        + np.asarray(d["confounder_contribution"], dtype=float).sum(1),
        rtol=0.0,
        atol=1e-12,
    )


def test_disabled_control_texture_renders_the_pre_texture_control_equation():
    """The rendered audit must show the exact executed own drive, or none."""
    enabled = sample_scm(
        _config(
            control_hf_sigma_range=(0.3, 0.3),
            control_pulse_prob_range=(0.2, 0.2),
            control_pulse_amp_range=(1.0, 1.0),
        ),
        seed=5,
    )
    for m in (0, 1):
        assert f"control_hf_sigma[{m}] * eps_z_hf[:, {m}]" in enabled.equations[f"Z{m + 1}"]
        assert (
            f"control_pulse_amp[{m}] * (eps_z_pulse[:, {m}] - control_pulse_prob[{m}])"
            in enabled.equations[f"Z{m + 1}"]
        )

    disabled = sample_scm(
        _config(
            control_hf_sigma_range=(0.0, 0.0),
            control_pulse_prob_range=(0.0, 0.0),
            control_pulse_amp_range=(0.0, 0.0),
        ),
        seed=5,
    )
    for m in (0, 1):
        assert "eps_z_hf" not in disabled.equations[f"Z{m + 1}"]
        assert "eps_z_pulse" not in disabled.equations[f"Z{m + 1}"]
        assert f"use_control_hf[{m}]=False" in disabled.equations[f"Z{m + 1}"]
        assert f"use_control_pulse[{m}]=False" in disabled.equations[f"Z{m + 1}"]


def test_latent_factor_is_pinned_to_zero_mean_unit_scale():
    """D carries no scale of its own; the loadings do.

    The walk is centred exactly, so the mean is 0 to machine precision. Its
    amplitude is normalized by a CONSTANT rather than by its own realized
    standard deviation, so the realized sd scatters around 1 instead of
    equalling it -- that scatter is the price of keeping the innovations-to-path
    map injective. What is pinned is the expectation.
    """
    cfg = _config(adstock_burn_in=0)
    world = sample_scm(cfg, seed=0)
    demand = np.asarray(world.data["demand"], dtype=float)
    np.testing.assert_allclose(demand.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(world.params["rw_d"]["mean"], 0.0, atol=0.0)
    np.testing.assert_allclose(world.params["rw_d"]["std"], 1.0, atol=0.0)

    n_time_steps_full = cfg.n_time_steps + cfg.adstock_burn_in
    width = _kernel_width(
        float(world.params["rw_d"]["smoothness"][0]),
        n_time_steps_full,
        rw_smoothness_max_weeks=cfg.rw_smoothness_max_weeks,
    )
    basis = _walk_basis(n_time_steps_full, width)
    np.testing.assert_allclose(np.sum(basis**2) / n_time_steps_full, 1.0, rtol=0.0, atol=1e-14)
    paths = np.random.default_rng(91).normal(size=(128, n_time_steps_full)) @ basis.T
    np.testing.assert_allclose(paths.mean(axis=1), 0.0, atol=1e-12)
    assert np.ptp(paths.std(axis=1)) > 0.1  # Paths scatter; their scale is not pinned.
