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
from prior_generator.signal_diagnostics import _adstock_numpy
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
    "mm_kappa_mult",
    "tanh_c",
    "root_alpha",
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


def test_default_unshocked_oracle_keeps_legacy_free_rvs(world_and_oracle):
    """Disabled schedules leave the existing oracle prior/RV surface untouched."""
    _gen_model, oracle, _world = world_and_oracle
    free_names = {rv.name for rv in oracle.free_RVs}
    assert set(SHARED_RV_NAMES) <= free_names
    assert not any(name.startswith("channel_shock") for name in free_names)


def test_baseline_walk_override_is_shared_by_generation_and_oracle():
    """An explicit baseline scale reaches both models without running NUTS."""
    cfg = _small_cfg(rw_std_sigma=1.2, rw_baseline_std_sigma=0.35)
    rng = np.random.default_rng(2)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)
    gen_model, _out_names, _param_names = build_world_model(g_act, cfg, structural, cfg.T)
    oracle = build_oracle_model(
        g_act,
        cfg,
        structural,
        {
            "channels": np.zeros((cfg.T, 2)),
            "controls": np.zeros((cfg.T, 1)),
            "sales": np.zeros(cfg.T),
        },
    )
    value = np.array([0.4])
    expected = pm.logp(pm.HalfNormal.dist(sigma=0.35), value).eval()

    assert np.allclose(pm.logp(gen_model["rw_b_std"], value).eval(), expected)
    assert np.allclose(pm.logp(oracle["rw_b_std"], value).eval(), expected)


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

    bad_sales = {
        "channels": world["channels"],
        "controls": world["controls"],
        "sales": world["sales"][:, None],
    }
    with pytest.raises(ValueError, match="sales must have shape"):
        build_oracle_model(g_act, cfg, structural, bad_sales)

    empty = {
        "channels": np.empty((0, 2)),
        "controls": np.empty((0, 1)),
        "sales": np.empty(0),
    }
    with pytest.raises(ValueError, match="at least one observation"):
        build_oracle_model(g_act, cfg, structural, empty)

    for key, fill in (("channels", np.nan), ("controls", np.inf), ("sales", np.nan)):
        nonfinite = {
            "channels": world["channels"].copy(),
            "controls": world["controls"].copy(),
            "sales": world["sales"].copy(),
        }
        nonfinite[key].flat[0] = fill
        with pytest.raises(ValueError, match="finite"):
            build_oracle_model(g_act, cfg, structural, nonfinite)


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
    assert world.data["saturation_scale"].shape == (world.K,)
    assert (world.data["saturation_scale"] > 0.0).all()
    oracle = world.oracle_model()
    # the conditioned prior narrows adstock_alpha to the recorded interval
    lo, width = world.extras["prior_cond"]["adstock_alpha"]
    outside = np.full(world.K, lo - 0.05)
    inside = np.full(world.K, lo + width / 2)
    assert not np.isfinite(pm.logp(oracle["adstock_alpha"], outside).eval()).all()
    assert np.isfinite(pm.logp(oracle["adstock_alpha"], inside).eval()).all()


def _shocked_oracle_world(*, n_shocks=1, level=(0.0, 0.0)):
    """A deterministic small direct-only shocked world and its oracle inputs."""
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        T=12,
        adstock_burn_in=0,
        n_channel_shocks=n_shocks,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=level,
        edge_budget={"cy": (1, 1)},
    )
    g = {
        "g_cy": np.array([1]),
        "g_dc": np.zeros((1, 1), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, 1), dtype=int),
        "g_cc": np.zeros((1, 1), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(4))
    structural["adstock_family"][:] = 1
    structural["sat_family"][:] = 0
    model, names, param_names = build_world_model(g, cfg, structural, cfg.T)
    drawn = {
        name: value[0]
        for name, value in pg.draw_worlds(model, names + param_names, seed=13).items()
    }
    data = {
        key: drawn[key]
        for key in (
            "channels",
            "controls",
            "sales",
            "saturation_scale",
            "channel_shock_channel",
            "channel_shock_start",
            "channel_shock_length",
            "channel_shock_level_multiplier",
            "channel_shock_level",
        )
    }
    data["channel_level"] = drawn["param_channel_level"]
    return cfg, g, structural, drawn, data


def test_shocked_oracle_uses_known_schedule_without_schedule_rvs():
    cfg, g, structural, drawn, data = _shocked_oracle_world()
    oracle = build_oracle_model(g, cfg, structural, data)

    assert not any(rv.name.startswith("channel_shock") for rv in oracle.free_RVs)
    # A held window clamps spend and nothing else, so pre-window carryover
    # decays into the intervention window instead of being discarded.
    mask = drawn["channel_shock_mask"][:, 0].astype(bool)
    start = int(np.flatnonzero(mask)[0])
    contribution = pm.draw(oracle["contributions"], draws=1, random_seed=17)
    assert (contribution[mask, 0] >= 0.0).all()
    if start > 0 and data["channels"][start - 1, 0] > 0.0:
        assert contribution[start, 0] > 0.0


def test_shocked_oracle_response_matches_plain_adstock_of_the_clamped_series():
    cfg, g, structural, _drawn, data = _shocked_oracle_world(n_shocks=2, level=(0.5, 0.5))
    # Make the response independent of a random schedule realization.
    data.update(
        {
            "channels": np.array(
                [[4.0], [3.0], [0.5], [0.5], [7.0], [8.0], [0.5], [0.5], [1.0], [1.0], [1.0], [1.0]]
            ),
            "channel_shock_channel": np.array([0, 0], dtype="int64"),
            "channel_shock_start": np.array([2, 6], dtype="int64"),
            "channel_shock_length": np.array([2, 2], dtype="int64"),
            "channel_shock_level_multiplier": np.array([0.5, 0.5]),
            "channel_shock_level": np.array([0.5, 0.5]),
            "channel_level": np.array([1.0]),
            "saturation_scale": np.array([3.0]),
        }
    )
    oracle = build_oracle_model(g, cfg, structural, data)
    with oracle:
        got, alpha, beta = pm.draw(
            [oracle["contributions"][:, 0], oracle["adstock_alpha"], oracle["beta"]],
            draws=1,
            random_seed=29,
        )
    adstock = _adstock_numpy(
        data["channels"][:, 0],
        family=1,
        alpha=alpha[0],
        lam=1.0,
        shape=1.0,
        l_max=cfg.l_max,
    )
    expected = beta[0] * adstock / data["saturation_scale"][0]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)


def test_shocked_oracle_rejects_missing_or_malformed_schedule_metadata():
    cfg, g, structural, _drawn, data = _shocked_oracle_world(level=(0.0, 1.0))
    missing = dict(data)
    del missing["channel_shock_start"]
    with pytest.raises(ValueError, match="channel_shock_start"):
        build_oracle_model(g, cfg, structural, missing)
    malformed = dict(data)
    malformed["channel_shock_length"] = np.array([1, 2], dtype="int64")
    with pytest.raises(ValueError, match="channel_shock_length"):
        build_oracle_model(g, cfg, structural, malformed)
    malformed_scale = dict(data)
    malformed_scale["saturation_scale"] = np.array([np.nan])
    with pytest.raises(ValueError, match="saturation_scale"):
        build_oracle_model(g, cfg, structural, malformed_scale)
    contradictory = dict(data)
    contradictory["channels"] = contradictory["channels"].copy()
    start = int(contradictory["channel_shock_start"][0])
    contradictory["channels"][start : start + 2, 0] = 1.0
    with pytest.raises(ValueError, match="held level does not match observed spend"):
        build_oracle_model(g, cfg, structural, contradictory)
    contradictory_multiplier = dict(data)
    original_multiplier = float(data["channel_shock_level_multiplier"][0])
    contradictory_multiplier["channel_shock_level_multiplier"] = np.array(
        [0.0 if original_multiplier > 0.5 else 1.0]
    )
    with pytest.raises(ValueError, match=r"multiplier \* channel_level"):
        build_oracle_model(g, cfg, structural, contradictory_multiplier)


def test_scm_oracle_model_roundtrip_with_shocks():
    cfg = _small_cfg(
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.0, 0.0),
    )
    world = pg.sample_scm(cfg, seed=15, max_eps_draws=4)
    oracle = world.oracle_model()
    assert {"baseline", "contributions", "sales_mu", "demand"} <= {
        deterministic.name for deterministic in oracle.deterministics
    }
    assert not any(rv.name.startswith("channel_shock") for rv in oracle.free_RVs)


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
