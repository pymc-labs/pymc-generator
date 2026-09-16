"""Optional shared baseline/treatment innovation tests."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

import pymc_generator as pg
from pymc_generator.random_walk import symbolic_random_walk
from pymc_generator.sampler import _slice_g_active, sample_g_additive
from pymc_generator.world_model import build_world_model, draw_worlds, sample_structure


@pytest.mark.parametrize(
    "strength_range",
    [(-0.01, 0.1), (0.1, 0.96), (0.5, 0.4), (np.nan, 0.1), (0.1, np.inf)],
)
def test_confounding_strength_range_is_bounded_and_finite(strength_range):
    with pytest.raises(ValueError, match="confounding_strength_range"):
        pg.SCMPrior(confounding_strength_range=strength_range).validate()


def _built(strength_range: tuple[float, float] | None):
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        edge_budget={"cy": (2, 2), "dc": (2, 2), "dy": (1, 1), "zc": (2, 2)},
        confounding_strength_range=strength_range,
    )
    rng = np.random.default_rng(42)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=2, n_latent_active=1
    )
    g_active = _slice_g_active(g, 2, 2, 1)
    return build_world_model(g_active, cfg, sample_structure(g_active, cfg, rng), cfg.n_time_steps)


def test_degenerate_confounding_strength_is_constant_without_free_rv():
    model, out_names, param_names = _built((0.35, 0.35))
    assert "confounding_strength" not in {rv.name for rv in model.free_RVs}
    assert "confounding_strength" in out_names
    assert "param_confounding_strength" in param_names
    drawn = draw_worlds(model, out_names + param_names, seed=8, draws=3)
    assert np.array_equal(drawn["confounding_strength"], np.full(3, 0.35))
    assert np.array_equal(drawn["param_confounding_strength"], np.full(3, 0.35))


def test_zero_confounding_range_preserves_disabled_draw_stream():
    disabled_model, disabled_names, _ = _built(None)
    zero_model, zero_names, _ = _built((0.0, 0.0))

    disabled = draw_worlds(disabled_model, disabled_names, seed=23, draws=1)
    zero = draw_worlds(zero_model, zero_names, seed=23, draws=1)
    assert np.array_equal(zero["confounding_strength"], np.zeros(1))

    for name in ("outcome", "treatments", "contributions"):
        assert disabled[name].tobytes() == zero[name].tobytes()


def test_nondegenerate_confounding_strength_varies_per_batched_world():
    model, out_names, _ = _built((0.1, 0.8))
    assert "confounding_strength" in {rv.name for rv in model.free_RVs}
    rho = draw_worlds(model, out_names, seed=9, draws=8)["confounding_strength"]
    assert np.all((0.1 <= rho) & (rho <= 0.8))
    assert np.unique(rho).size > 1


def test_enabled_confounding_preserves_decomposition_and_nonflat_treatments():
    model, out_names, _ = _built((0.7, 0.7))
    d = {name: value[0] for name, value in draw_worlds(model, out_names, seed=10).items()}
    assert (
        np.abs(
            d["outcome"] - d["baseline"] - d["contributions"].sum(1) - d["indirect_effects"]
        ).max()
        < 1e-9
    )
    assert d["treatments"].std(axis=0).max() > 0.0


def test_shared_innovation_formula_drives_treatment_walk_exactly():
    rho = 0.6
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=24,
        rw_smoothness_max_weeks=1,
        confounding_strength_range=(rho, rho),
        treatment_hf_sigma_range=(0.0, 0.0),
        treatment_pulse_prob_range=(0.0, 0.0),
        treatment_pulse_amp_range=(0.0, 0.0),
    )
    g = {
        "g_cy": np.ones(1, dtype=int),
        "g_dc": np.zeros((1, 1), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zy": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, 1), dtype=int),
        "g_cc": np.zeros((1, 1), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(18))
    model, out_names, param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
    drawn = {
        name: values[0]
        for name, values in draw_worlds(
            model, out_names + param_names + ("eps_c", "eps_b"), seed=19
        ).items()
    }
    effective_eps = np.sqrt(1.0 - rho**2) * drawn["eps_c"][:, 0] + rho * drawn["eps_b"]
    n_time_steps_full = cfg.n_time_steps + cfg.carryover_burn_in
    walk = symbolic_random_walk(
        n_time_steps_full,
        mean=drawn["param_rw_c_mean"][0],
        std=drawn["param_rw_c_std"][0],
        smoothness=float(structural["smoothness_c"][0]),
        rw_smoothness_max_weeks=cfg.rw_smoothness_max_weeks,
        positive_only=True,
        eps=pt.as_tensor_variable(effective_eps),
    ).eval()
    expected_treatment = np.logaddexp(0.0, walk)[cfg.carryover_burn_in :]
    np.testing.assert_allclose(drawn["treatments"][:, 0], expected_treatment, rtol=0.0, atol=1e-12)


def test_confounding_strength_is_persisted_in_corpus():
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        n_cells=2,
        draws_per_cell=1,
        seed=3,
        confounding_strength_range=(0.25, 0.25),
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert corpus["confounding_strength"].shape == (2,)
    assert corpus["confounding_strength"].dtype == np.float32
    assert np.array_equal(corpus["confounding_strength"], np.full(2, 0.25, dtype=np.float32))


def test_confounding_strength_is_exposed_by_single_world_data_and_params():
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        seed=4,
        confounding_strength_range=(0.25, 0.25),
    )
    world = pg.sample_scm(cfg, seed=4, max_eps_draws=4)
    assert float(world.data["confounding_strength"]) == 0.25
    assert float(world.params["confounding_strength"]) == 0.25
