"""The observed-data (NUTS oracle) variant of the world model (to-do 02).

Locks the invariants: (1) the oracle's shared priors are the SAME measure as
the generative model's — per-variable logp equality at the drawn world's
values, and a finite total logp at the truth; (2) generation is untouched by
building an oracle; (3) a short NUTS fit on a small world recovers the drawn
parameters within wide posterior bounds (slow-gated).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

import prior_generator as pg
from prior_generator.sampler import _slice_g_active, sample_g_additive
from prior_generator.world_model import build_oracle_model, build_world_model, sample_structure

#: Free RVs the two models must define identically (same name, same prior).
SHARED_RV_NAMES = (
    "beta",
    "delta_db",
    "rho_zb",
    "adstock_alpha",
    "weibull_lam",
    "weibull_k",
    "hill_slope",
    "hill_kappa_mult",
    "logistic_lam",
    "mm_alpha",
    "mm_kappa_mult",
    "tanh_b",
    "tanh_c",
    "root_alpha",
    "rw_d_mean",
    "rw_d_std",
    "rw_b_mean",
    "rw_b_std",
    "rw_y_std",
    "eps_d",
    "eps_b",
)


def _small_cfg(**overrides):
    return pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, T=28, seed=5, **overrides)


@pytest.fixture(scope="module")
def world_and_oracle():
    cfg = _small_cfg()
    rng = np.random.default_rng(2)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)
    gen_model, out_names, _ = build_world_model(g_act, cfg, structural, cfg.T)
    drawn = pg.draw_worlds(gen_model, out_names + SHARED_RV_NAMES, seed=17, draws=1)
    world = {name: drawn[name][0] for name in drawn}
    data = {
        "channels": world["channels"],
        "controls": world["controls"],
        "sales": world["sales"],
    }
    oracle = build_oracle_model(g_act, cfg, structural, data)
    return gen_model, oracle, world


def test_shared_priors_same_measure(world_and_oracle):
    """Per-variable logp equality at the drawn values: same priors, same measure."""
    gen_model, oracle, world = world_and_oracle
    for name in SHARED_RV_NAMES:
        val = world[name]
        lp_gen = pm.logp(gen_model[name], val).eval()
        lp_oracle = pm.logp(oracle[name], val).eval()
        assert np.allclose(lp_gen, lp_oracle), f"prior for {name!r} drifted between models"


def test_oracle_logp_finite_at_truth(world_and_oracle):
    """Total oracle logp (priors + likelihood) is finite at the drawn world."""
    _gen_model, oracle, world = world_and_oracle
    point = {}
    for rv in oracle.free_RVs:
        val = np.asarray(world[rv.name])
        value_var = oracle.rvs_to_values[rv]
        transform = oracle.rvs_to_transforms[rv]
        if transform is None:
            point[value_var.name] = val
        else:
            point[value_var.name] = transform.forward(val, *rv.owner.inputs).eval()
    logp = oracle.compile_logp()(point)
    assert np.isfinite(logp)


def test_oracle_deterministics_shapes(world_and_oracle):
    _gen_model, oracle, world = world_and_oracle
    T, K = world["channels"].shape
    J = world["demand"].shape[1]
    names = {d.name for d in oracle.deterministics}
    assert {"contributions", "baseline", "sales_mu", "demand"} <= names
    assert tuple(oracle["contributions"].shape.eval()) == (T, K)
    assert tuple(oracle["demand"].shape.eval()) == (T, J)


def test_oracle_rejects_bad_shapes(world_and_oracle):
    gen_model, _oracle, world = world_and_oracle
    cfg = _small_cfg()
    rng = np.random.default_rng(2)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)
    bad = {
        "channels": world["channels"][:, :1],  # wrong K
        "controls": world["controls"],
        "sales": world["sales"],
    }
    with pytest.raises(ValueError, match="data shapes"):
        build_oracle_model(g_act, cfg, structural, bad)


def test_generation_untouched_by_oracle():
    """Building an oracle consumes no RNG and leaves corpora byte-identical."""
    cfg = pg.make_scm_prior(
        n_treatments=4, n_covariates=2, n_latent=1, T=40, n_cells=2, draws_per_cell=3, seed=7
    )
    before = pg.sample_prior_predictive(cfg)
    world = pg.sample_scm(cfg, seed=11)
    world.oracle_model()  # build (and discard) an oracle in between
    after = pg.sample_prior_predictive(cfg)
    for key, val in before.items():
        if isinstance(val, np.ndarray):
            assert np.array_equal(after[key], val), key


def test_scm_oracle_model_roundtrip():
    """SCM.oracle_model() rebuilds the oracle from the recorded extras."""
    cfg = _small_cfg(prior_conditioning=True)
    world = pg.sample_scm(cfg, seed=3)
    oracle = world.oracle_model()
    # the conditioned prior narrows adstock_alpha to the recorded interval
    lo, width = world.extras["prior_cond"]["adstock_alpha"]
    outside = np.full(world.K, lo - 0.05)
    inside = np.full(world.K, lo + width / 2)
    assert not np.isfinite(pm.logp(oracle["adstock_alpha"], outside).eval()).all()
    assert np.isfinite(pm.logp(oracle["adstock_alpha"], inside).eval()).all()


@pytest.mark.slow
def test_nuts_smoke_recovers_params():
    """Short NUTS fit on a small world: truth within wide posterior bounds."""
    cfg = _small_cfg()
    world = pg.sample_scm(cfg, seed=8)
    with world.oracle_model():
        idata = pm.sample(
            draws=200,
            tune=200,
            chains=2,
            cores=1,
            random_seed=42,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    true_beta = np.asarray(world.params["beta"])
    direct = np.asarray(world.g["g_cy"]) == 1
    mean = post["beta"].mean(("chain", "draw")).values
    std = post["beta"].std(("chain", "draw")).values
    # wide bounds: the truth within +/- 4 posterior sd of the posterior mean
    assert (np.abs(mean - true_beta)[direct] <= 4.0 * std[direct] + 0.25).all()
    # the fitted sales mean tracks the observed sales
    mu = post["sales_mu"].mean(("chain", "draw")).values
    r = np.corrcoef(mu, world.data["sales"])[0, 1]
    assert r > 0.8
