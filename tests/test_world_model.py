"""The PyMC-model draw path: priors are distributions, worlds are pm.draws.

Locks the correctness of the RV engine at the low level (build_world_model +
draw_worlds) — the exact interventional decomposition, positivity, seed
determinism, and batched draws — independently of the high-level wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
        T=48,
        edge_budget={"cy": (4, 4), "cc": (1, 2), "zc": (1, 2), "dc": (1, 2)},
    )
    rng = np.random.default_rng(0)
    g = sample_g_additive(rng, cfg, cfg.layout)
    g_act = _slice_g_active(g, 4, 2, 1)
    structural = sample_structure(g_act, cfg, rng)
    model, out_names, _param_names = build_world_model(g_act, cfg, structural, cfg.T)
    return model, out_names


def test_model_is_pm_model_with_priors_and_outputs(built):
    model, out_names = built
    # continuous priors + noise are real RVs; every graph output is registered
    assert len(model.free_RVs) > 10
    assert out_names[:15] == (
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


def test_diverse_texture_gives_nonflat_targets(built):
    model, out_names = built
    contrib = draw_worlds(model, out_names, seed=3, draws=1)["contributions"][0]  # (T, K)
    cv = contrib.std(0) / (np.abs(contrib.mean(0)) + 1e-9)
    assert cv.max() > 0.05


def test_single_draw_has_leading_axis(built):
    # regression: draw_worlds always keeps a leading draws axis, so
    # sample_scm(max_eps_draws=1) can index candidate 0 without hitting time.
    model, out_names = built
    d = draw_worlds(model, out_names, seed=9, draws=1)
    assert d["sales"].shape == (1, 48)
    assert d["contributions"].shape == (1, 48, 4)


def test_kernel_width_is_horizon_invariant_except_for_the_short_series_clamp():
    max_weeks = 26
    widths = [_kernel_width(0.5, T, rw_smoothness_max_weeks=max_weeks) for T in (52, 104, 156)]

    assert widths == [13, 13, 13]
    assert _kernel_width(0.75, 104, rw_smoothness_max_weeks=max_weeks) > widths[0]
    assert _kernel_width(0.5, 104, rw_smoothness_max_weeks=10) < widths[0]
    assert _kernel_width(1.0, 12, rw_smoothness_max_weeks=max_weeks) == 12


@pytest.mark.slow
def test_absolute_width_reduces_horizon_dependence_of_walk_texture():
    """The absolute-week kernel makes a walk's texture markedly less horizon-dependent.

    Asserted as a PAIRED comparison against the former horizon-proportional rule
    rather than as an absolute threshold. Centring subtracts a ``T``-dependent
    mean and ``_centred_walk_scale`` divides by a ``T``-dependent constant, and a
    centred Brownian path's lag-1 autocorrelation rises with window length
    regardless of the kernel, so perfect invariance is not achievable and an
    absolute bound is a seed lottery: over 25 seeds the single-seed spread under
    the fixed rule ranges 0.0008-0.0179, so 9 of them breach a 0.005 bound.
    Averaged over seeds the improvement is stable — measured mean spread over 30
    seeds, fixed vs proportional: 0.0154 vs 0.0333 (smoothness 0.25), 0.0070 vs
    0.0176 (0.50), 0.0047 vs 0.0102 (0.75), i.e. 2.2-2.5x every time.

    Passing ``rw_smoothness_max_weeks=round(T / 4)`` per horizon reproduces the
    old ``round(smoothness * T / 4)`` width exactly, which is what makes this a
    like-for-like paired contrast on identical innovations.
    """
    horizons = (52, 104, 156)

    def mean_spread(cap_for_horizon):
        spreads = []
        for seed in range(1000, 1012):
            innovations = np.random.default_rng(seed).normal(size=max(horizons))
            autocorrelations = []
            for T in horizons:
                walk = symbolic_random_walk(
                    T,
                    mean=0.0,
                    std=1.0,
                    smoothness=0.5,
                    positive_only=False,
                    rw_smoothness_max_weeks=cap_for_horizon(T),
                    eps=pt.as_tensor_variable(innovations[:T]),
                ).eval()
                autocorrelations.append(float(np.corrcoef(walk[:-1], walk[1:])[0, 1]))
            spreads.append(max(autocorrelations) - min(autocorrelations))
        return float(np.mean(spreads))

    absolute_weeks = mean_spread(lambda T: 26)
    horizon_proportional = mean_spread(lambda T: max(1, round(T / 4)))

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


def test_output_registration_rejects_nonidentity_name_collision(monkeypatch):
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        T=12,
        adstock_burn_in=0,
        edge_budget={"cy": (2, 2)},
    )
    rng = np.random.default_rng(31)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)

    def graph_with_colliding_beta(*_args, **_kwargs):
        return {"outputs": {"beta": pt.as_tensor_variable(0.0)}}

    monkeypatch.setattr(
        "prior_generator.world_model.build_symbolic_graph",
        graph_with_colliding_beta,
    )
    with pytest.raises(ValueError, match="collides with a different model variable"):
        build_world_model(g_act, cfg, structural, cfg.T)


def test_output_registration_allows_identity_collisions_for_free_rvs(monkeypatch):
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        T=12,
        adstock_burn_in=0,
        edge_budget={"cy": (2, 2)},
        confounding_strength_range=(0.2, 0.4),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 3),
        channel_shock_level_range=(0.5, 1.0),
    )
    rng = np.random.default_rng(31)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)

    captured_graph: dict[str, Any] = {}
    original_build_symbolic_graph = world_model.build_symbolic_graph

    def capture_graph(*args, **kwargs):
        graph = original_build_symbolic_graph(*args, **kwargs)
        captured_graph["graph"] = graph
        return graph

    monkeypatch.setattr(world_model, "build_symbolic_graph", capture_graph)
    model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.T)

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
