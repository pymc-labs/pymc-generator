"""The PyMC-model draw path: priors are distributions, worlds are pm.draws.

Locks the correctness of the RV engine at the low level (build_world_model +
draw_worlds) — the exact interventional decomposition, positivity, seed
determinism, and batched draws — independently of the high-level wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import prior_generator.world_model as world_model
from prior_generator import make_scm_prior
from prior_generator.random_walk import _kernel_width, symbolic_random_walk
from prior_generator.sampler import _slice_g_active, sample_g_additive
from prior_generator.world_model import (
    _rw_prior_group,
    build_world_model,
    draw_worlds,
    sample_structure,
)


@pytest.fixture(scope="module")
def built():
    cfg = make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=48,
        edge_budget={"cy": (4, 4), "cc": (1, 2), "zc": (1, 2), "dc": (1, 2)},
    )
    rng = np.random.default_rng(0)
    g = sample_g_additive(rng, cfg, cfg.layout)
    g_act = _slice_g_active(g, 4, 2, 1)
    structural = sample_structure(g_act, cfg, rng)
    model, out_names, _param_names = build_world_model(g_act, cfg, structural, cfg.n_time_steps)
    return model, out_names


def test_model_is_pm_model_with_priors_and_outputs(built):
    model, out_names = built
    # continuous priors + noise are real RVs; every graph output is registered
    assert len(model.free_RVs) > 10
    assert out_names[:16] == (
        "demand",
        "controls",
        "channels",
        "channels_base",
        "saturation_scale",
        "baseline",
        "baseline_intrinsic",
        "control_contribution",
        "confounder_contribution",
        "contributions",
        "contributions_observed",
        "indirect_effects",
        "indirect_effects_by_source",
        "sales",
        "sales_noise",
        "confounding_strength",
    )
    assert "sales" in out_names and "indirect_effects_by_source" in out_names


def test_pm_draw_preserves_additive_identity(built):
    model, out_names = built
    d = {k: v[0] for k, v in draw_worlds(model, out_names, seed=123, draws=1).items()}
    s = d["sales"]
    identity = np.abs(s - (d["baseline"] + d["contributions"].sum(1) + d["indirect_effects"])).max()
    telescoping = np.abs(d["indirect_effects_by_source"].sum(1) - d["indirect_effects"]).max()
    full = np.abs(
        d["baseline_intrinsic"]
        + d["sales_noise"]
        + d["confounder_contribution"].sum(1)
        + d["control_contribution"].sum(1)
        + d["contributions"].sum(1)
        + d["indirect_effects_by_source"].sum(1)
        - s
    ).max()
    assert identity < 1e-9
    assert telescoping < 1e-9
    assert full < 1e-9


def test_channels_positive(built):
    model, out_names = built
    d = draw_worlds(model, out_names, seed=1, draws=1)
    assert (d["channels"] >= 0).all()


def test_seed_determinism(built):
    model, out_names = built
    a = draw_worlds(model, out_names, seed=42, draws=1)
    b = draw_worlds(model, out_names, seed=42, draws=1)
    c = draw_worlds(model, out_names, seed=43, draws=1)
    assert np.array_equal(a["sales"], b["sales"])
    assert not np.array_equal(a["sales"], c["sales"])


def test_batched_draws_have_leading_axis(built):
    model, out_names = built
    d = draw_worlds(model, out_names, seed=7, draws=5)
    assert d["sales"].shape == (5, 48)
    assert d["contributions"].shape == (5, 48, 4)


#: Innovation streams the channel texture reads, requested alongside the graph
#: outputs so the paired arms below can be checked for identical draws.
TEXTURE_EPS_NAMES = ("eps_c", "eps_c_hf", "eps_c_pulse")


@pytest.fixture(scope="module")
def texture_arms():
    """One world, three texture wirings, byte-identical draws.

    ``sample_structure`` reads the per-channel texture ENABLE flags off the
    config's ranges, but the MAGNITUDES (``hf_sigma``, ``pulse_amp``) and the
    Bernoulli fires are drawn either way. Flipping only the flags therefore
    leaves the RV set — and, at a fixed seed, every drawn value — identical
    while rewiring the channel equation, which makes these three arms a
    controlled experiment: any difference between them IS the texture.

    ``cc``/``zc``/``dc`` are budgeted out so each channel column is its own
    exogenous drive through the softplus, leaving nothing else that a change
    could be attributed to.
    """
    cfg = make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=104,
        edge_budget={"cy": (3, 3), "cc": 0, "zc": 0, "dc": 0},
        channel_hf_sigma_range=(0.4, 0.4),
        channel_pulse_prob_range=(0.15, 0.15),
        channel_pulse_amp_range=(1.5, 1.5),
    )
    rng = np.random.default_rng(0)
    g = sample_g_additive(rng, cfg, cfg.layout)
    g_act = _slice_g_active(g, 3, 2, 1)
    structural = sample_structure(g_act, cfg, rng)

    def arm(*, use_hf: bool, use_pulse: bool) -> dict[str, np.ndarray]:
        wiring = dict(structural)
        wiring["use_hf"] = np.full(3, use_hf)
        wiring["use_pulse"] = np.full(3, use_pulse)
        model, out_names, param_names = build_world_model(g_act, cfg, wiring, cfg.n_time_steps)
        drawn = draw_worlds(model, out_names + param_names + TEXTURE_EPS_NAMES, seed=5, draws=1)
        return {name: values[0] for name, values in drawn.items()}

    return cfg, {
        "smooth": arm(use_hf=False, use_pulse=False),
        "jittery": arm(use_hf=True, use_pulse=False),
        "pulsed": arm(use_hf=False, use_pulse=True),
    }


def test_texture_arms_differ_only_in_their_wiring(texture_arms):
    """The three arms are the same drawn world, so their deltas are the texture.

    Load-bearing for the two tests below: ``reseed_rngs`` hands out streams by
    graph-traversal order, so a future change that made a disabled texture term
    reach an output (or that drew its magnitude conditionally) would shift every
    stream and turn those paired deltas into a comparison of two DIFFERENT
    worlds — which would still look plausible.
    """
    _cfg, arms = texture_arms
    reference = arms["smooth"]
    shared = (
        "param_beta",
        "param_rw_c_mean",
        "param_rw_c_std",
        "param_channel_level",
        "param_hf_sigma",
        "param_pulse_amp",
        "param_pulse_prob",
        *TEXTURE_EPS_NAMES,
    )
    for arm in ("jittery", "pulsed"):
        for name in shared:
            assert np.array_equal(reference[name], arms[arm][name]), (
                f"{name} differs between the 'smooth' and {arm!r} arms"
            )


def test_high_frequency_texture_moves_every_week_in_the_sign_of_its_innovation(texture_arms):
    """``use_hf`` adds ``hf_sigma * eps_c_hf`` inside the channel softplus.

    The signature is mechanism-specific and exact: the term is iid WEEKLY, so
    every single week moves, and softplus is strictly increasing, so each week
    moves in the sign of its own innovation. A texture that had been wired to
    the wrong innovation, scaled to zero, or smoothed would break this while
    still producing a perfectly plausible-looking jagged channel — which is
    what the previous ``cv.max() > 0.05`` check could not tell apart (it passes
    at 0.71 on the 'smooth' arm below, with the texture OFF).
    """
    cfg, arms = texture_arms
    window = slice(cfg.adstock_burn_in, None)
    delta = arms["jittery"]["channels"] - arms["smooth"]["channels"]
    jitter = arms["smooth"]["eps_c_hf"][window]

    assert (delta != 0.0).all(), "an iid weekly term must move every week"
    assert (np.sign(delta) == np.sign(jitter)).all()


def test_pulse_texture_moves_only_the_weeks_its_bernoulli_fires(texture_arms):
    """``use_pulse`` adds ``pulse_amp * eps_c_pulse`` with a 0/1 fire indicator.

    The complement of the high-frequency signature: a pulse is SPARSE, so the
    channel must be untouched — bit for bit — on every non-fire week, and
    strictly raised on every fire week (``pulse_amp > 0`` and softplus is
    strictly increasing). A pulse implemented as a threshold on continuous
    noise, or centred like the CONTROL pulse, would fail here.
    """
    cfg, arms = texture_arms
    window = slice(cfg.adstock_burn_in, None)
    delta = arms["pulsed"]["channels"] - arms["smooth"]["channels"]
    fired = arms["smooth"]["eps_c_pulse"][window] > 0

    # both classes must be represented, or one of the two claims below is vacuous
    assert fired.any(axis=0).all() and (~fired).any(axis=0).all(), (
        f"every channel needs both fire and quiet weeks, got "
        f"{fired.sum(axis=0)} of {fired.shape[0]}"
    )
    assert (delta[fired] > 0.0).all(), "every fire week must raise the channel"
    assert (delta[~fired] == 0.0).all(), "a non-fire week must be untouched"


def test_single_draw_has_leading_axis(built):
    # regression: draw_worlds always keeps a leading draws axis, so
    # sample_scm(max_eps_draws=1) can index candidate 0 without hitting time.
    model, out_names = built
    d = draw_worlds(model, out_names, seed=9, draws=1)
    assert d["sales"].shape == (1, 48)
    assert d["contributions"].shape == (1, 48, 4)


def test_kernel_width_is_horizon_invariant_except_for_the_short_series_clamp():
    max_weeks = 26
    widths = [
        _kernel_width(0.5, n_time_steps, rw_smoothness_max_weeks=max_weeks)
        for n_time_steps in (52, 104, 156)
    ]

    assert widths == [13, 13, 13]
    assert _kernel_width(0.75, 104, rw_smoothness_max_weeks=max_weeks) > widths[0]
    assert _kernel_width(0.5, 104, rw_smoothness_max_weeks=10) < widths[0]
    assert _kernel_width(1.0, 12, rw_smoothness_max_weeks=max_weeks) == 12


@pytest.mark.slow
def test_absolute_width_reduces_horizon_dependence_of_walk_texture():
    """The absolute-week kernel makes a walk's texture markedly less horizon-dependent.

    Asserted as a PAIRED comparison against the former horizon-proportional rule
    rather than as an absolute threshold. Centring subtracts an ``n_time_steps``-dependent
    mean and ``_centred_walk_scale`` divides by an ``n_time_steps``-dependent constant,
    and a
    centred Brownian path's lag-1 autocorrelation rises with window length
    regardless of the kernel, so perfect invariance is not achievable and an
    absolute bound is a seed lottery: over 25 seeds the single-seed spread under
    the fixed rule ranges 0.0008-0.0179, so 9 of them breach a 0.005 bound.
    Averaged over seeds the improvement is stable — measured mean spread over 30
    seeds, fixed vs proportional: 0.0154 vs 0.0333 (smoothness 0.25), 0.0070 vs
    0.0176 (0.50), 0.0047 vs 0.0102 (0.75), i.e. 2.2-2.5x every time.

    Passing ``rw_smoothness_max_weeks=round(n_time_steps / 4)`` per horizon reproduces
    the old ``round(smoothness * n_time_steps / 4)`` width exactly, which is what makes
    this a
    like-for-like paired contrast on identical innovations.
    """
    horizons = (52, 104, 156)

    def mean_spread(cap_for_horizon):
        spreads = []
        for seed in range(1000, 1012):
            innovations = np.random.default_rng(seed).normal(size=max(horizons))
            autocorrelations = []
            for n_time_steps in horizons:
                walk = symbolic_random_walk(
                    n_time_steps,
                    mean=0.0,
                    std=1.0,
                    smoothness=0.5,
                    positive_only=False,
                    rw_smoothness_max_weeks=cap_for_horizon(n_time_steps),
                    eps=pt.as_tensor_variable(innovations[:n_time_steps]),
                ).eval()
                autocorrelations.append(float(np.corrcoef(walk[:-1], walk[1:])[0, 1]))
            spreads.append(max(autocorrelations) - min(autocorrelations))
        return float(np.mean(spreads))

    absolute_weeks = mean_spread(lambda n_time_steps: 26)
    horizon_proportional = mean_spread(lambda n_time_steps: max(1, round(n_time_steps / 4)))

    assert absolute_weeks < horizon_proportional
    assert horizon_proportional / absolute_weeks > 1.7


def test_walk_scale_rejects_ambiguous_range_and_sigma():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _rw_prior_group(
            "test",
            1,
            False,
            (0.0, 0.0),
            0.5,
            rw_smoothness_max_weeks=26,
            std_sigma=1.0,
            std_range=(1.0, 1.0),
        )


def test_relative_outcome_scales_follow_the_media_amplitude():
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        edge_budget={"cy": (2, 2)},
        outcome_std_mode="relative",
        rw_baseline_std_range=(0.04, 0.08),
        rw_sales_std_range=(0.01, 0.03),
    )
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(13))
    model, _out_names, _param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = draw_worlds(
        model,
        ("beta", "rw_b_std_rel", "rw_y_std_rel", "rw_b_std", "rw_y_std"),
        seed=14,
    )
    media_amplitude = np.sqrt(np.sum(drawn["beta"][0] ** 2))

    np.testing.assert_allclose(
        drawn["rw_b_std"][0],
        drawn["rw_b_std_rel"][0] * media_amplitude,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        drawn["rw_y_std"][0],
        drawn["rw_y_std_rel"][0] * media_amplitude,
        rtol=0.0,
        atol=1e-14,
    )


def test_absolute_outcome_scales_keep_halfnormal_semantics():
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=16,
        edge_budget={"cy": (2, 2)},
        outcome_std_mode="absolute",
        rw_baseline_std_sigma=0.35,
        rw_sales_std_sigma=0.12,
    )
    rng = np.random.default_rng(15)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=1, n_latent_active=1
    )
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)
    model, _out_names, _param_names = build_world_model(g_act, cfg, structural, cfg.n_time_steps)

    assert "rw_b_std_rel" not in model.named_vars
    assert "rw_y_std_rel" not in model.named_vars
    value = np.array([0.2])
    np.testing.assert_allclose(
        pm.logp(model["rw_b_std"], value).eval(),
        pm.logp(pm.HalfNormal.dist(sigma=0.35), value).eval(),
    )
    np.testing.assert_allclose(
        pm.logp(model["rw_y_std"], value).eval(),
        pm.logp(pm.HalfNormal.dist(sigma=0.12), value).eval(),
    )


def test_output_registration_rejects_nonidentity_name_collision(monkeypatch):
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=12,
        adstock_burn_in=0,
        edge_budget={"cy": (2, 2)},
    )
    rng = np.random.default_rng(31)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=1, n_latent_active=1
    )
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)

    def graph_with_colliding_beta(*_args, **_kwargs):
        return {"outputs": {"beta": pt.as_tensor_variable(0.0)}}

    monkeypatch.setattr(
        "prior_generator.world_model.build_symbolic_graph",
        graph_with_colliding_beta,
    )
    with pytest.raises(ValueError, match="collides with a different model variable"):
        build_world_model(g_act, cfg, structural, cfg.n_time_steps)


def test_output_registration_allows_identity_collisions_for_free_rvs(monkeypatch):
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=12,
        adstock_burn_in=0,
        edge_budget={"cy": (2, 2)},
        confounding_strength_range=(0.2, 0.4),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 3),
        channel_shock_level_range=(0.5, 1.0),
    )
    rng = np.random.default_rng(31)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=1, n_latent_active=1
    )
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)

    captured_graph: dict[str, Any] = {}
    original_build_symbolic_graph = world_model.build_symbolic_graph

    def capture_graph(*args, **kwargs):
        graph = original_build_symbolic_graph(*args, **kwargs)
        captured_graph["graph"] = graph
        return graph

    monkeypatch.setattr(world_model, "build_symbolic_graph", capture_graph)
    model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.n_time_steps)

    names = (
        "confounding_strength",
        "channel_shock_length",
        "channel_shock_level_multiplier",
    )
    assert set(names) <= {rv.name for rv in model.free_RVs}
    assert set(names) <= set(out_names)
    graph_outputs = captured_graph["graph"]["outputs"]
    for name in names:
        assert model[name] is graph_outputs[name]
