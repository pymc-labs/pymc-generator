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
import pytensor
import pytest
from scipy.stats import Covariance, multivariate_normal

import prior_generator as pg
from prior_generator.random_walk import (
    _centred_walk_scale,
    _kernel_width,
    _smooth_columns_numpy,
    symbolic_random_walk,
)
from prior_generator.sampler import SCMPrior, _slice_g_active, sample_g_additive
from prior_generator.signal_diagnostics import _adstock_numpy
from prior_generator.world_model import (
    _MECHANISM_PARAM_NAMES,
    ADSTOCK_FAMILY_PARAM_NAMES,
    SATURATION_FAMILY_PARAM_NAMES,
    _live_mechanism_param_names,
    _walk_basis,
    build_oracle_model,
    build_world_model,
    sample_structure,
)

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
    "rw_b_std_rel",
    "rw_y_std_rel",
    "eps_d",
    "eps_b",
)


#: Priors shared with generation for every oracle mode before live mechanisms.
_SHARED_ORACLE_PRIOR_NAMES = (
    "beta",
    "delta_db",
    "rho_zb",
    "rw_b_mean",
)


def _shared_oracle_rv_names(oracle: pm.Model, *, latent: str) -> tuple[str, ...]:
    """Return the generated RVs whose shared priors must remain identical."""
    free_names = {rv.name for rv in oracle.free_RVs}
    names = [
        *_SHARED_ORACLE_PRIOR_NAMES,
        *(
            name
            for name in ("rw_b_std_rel", "rw_y_std_rel", "rw_b_std", "rw_y_std")
            if name in free_names
        ),
        *(name for name in _MECHANISM_PARAM_NAMES if name in free_names),
    ]
    if latent == "sampled":
        names.extend(("eps_d", "eps_b"))
    return tuple(names)


def _small_cfg(**overrides):
    base = {"n_treatments": 2, "n_covariates": 1, "n_latent": 1, "T": 28, "seed": 5}
    return pg.make_scm_prior(**{**base, **overrides})


def _direct_only_graph(n_treatments: int = 2) -> dict[str, np.ndarray]:
    """Return a one-control, one-latent graph with direct-only media effects."""
    return {
        "g_cy": np.ones(n_treatments, dtype=int),
        "g_dc": np.zeros((1, n_treatments), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, n_treatments), dtype=int),
        "g_cc": np.zeros((n_treatments, n_treatments), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }


def _oracle_truth_point(oracle: pm.Model, values: dict[str, object]) -> dict[str, np.ndarray]:
    """Encode a world draw as the oracle's transformed value-variable point."""
    point = {}
    for rv in oracle.free_RVs:
        value = np.asarray(values[rv.name])
        value_var = oracle.rvs_to_values[rv]
        transform = oracle.rvs_to_transforms[rv]
        if transform is None:
            point[value_var.name] = value
        else:
            point[value_var.name] = transform.forward(value, *rv.owner.inputs).eval()
    return point


def _oracle_deterministic_at_truth(
    oracle: pm.Model, values: dict[str, object], name: str
) -> np.ndarray:
    """Evaluate one oracle deterministic at a generator draw's true parameters."""
    point = _oracle_truth_point(oracle, values)
    deterministic = oracle.replace_rvs_by_values([oracle[name]])[0]
    evaluate = pytensor.function(oracle.value_vars, deterministic, on_unused_input="ignore")
    return np.asarray(evaluate(*[point[value_var.name] for value_var in oracle.value_vars]))


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
        "saturation_scale": world["saturation_scale"],
    }
    oracle_marginal = build_oracle_model(g_act, cfg, structural, data)
    oracle_sampled = build_oracle_model(g_act, cfg, structural, data, latent="sampled")
    return gen_model, oracle_marginal, oracle_sampled, world


def test_shared_priors_same_measure(world_and_oracle):
    """Every mode shares generation's priors, restricted to live mechanisms."""
    gen_model, oracle_marginal, oracle_sampled, world = world_and_oracle
    for latent, oracle in (("marginal", oracle_marginal), ("sampled", oracle_sampled)):
        for name in _shared_oracle_rv_names(oracle, latent=latent):
            val = world[name]
            lp_gen = pm.logp(gen_model[name], val).eval()
            lp_oracle = pm.logp(oracle[name], val).eval()
            assert np.allclose(lp_gen, lp_oracle), f"prior for {name!r} drifted in {latent} mode"


def test_default_unshocked_oracle_uses_marginal_free_rvs(world_and_oracle):
    """The default drops only outcome-walk innovations, never schedule state."""
    _gen_model, oracle, _sampled_oracle, _world = world_and_oracle
    free_names = {rv.name for rv in oracle.free_RVs}
    assert set(_SHARED_ORACLE_PRIOR_NAMES) <= free_names
    assert {"rw_b_std_rel", "rw_y_std_rel"} <= free_names
    assert not {"rw_b_std", "rw_y_std"} & free_names
    assert not {"eps_d", "eps_b"} & free_names
    assert not any(name.startswith("channel_shock") for name in free_names)


def test_baseline_walk_override_is_shared_by_generation_and_oracle():
    """An explicit baseline scale reaches both models without running NUTS."""
    cfg = _small_cfg(
        outcome_std_mode="absolute",
        rw_std_sigma=1.2,
        rw_baseline_std_sigma=0.35,
    )
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
            "saturation_scale": np.ones(2),
        },
    )
    value = np.array([0.4])
    expected = pm.logp(pm.HalfNormal.dist(sigma=0.35), value).eval()

    assert np.allclose(pm.logp(gen_model["rw_b_std"], value).eval(), expected)
    assert np.allclose(pm.logp(oracle["rw_b_std"], value).eval(), expected)


def test_relative_outcome_scale_is_shared_by_generation_and_oracle():
    """Fixed relative parameters produce one absolute scale in both builders."""
    cfg = _small_cfg(
        beta_additive_range=(1.2, 1.2),
        rw_baseline_std_range=(0.07, 0.07),
        rw_sales_std_range=(0.02, 0.02),
    )
    g = _direct_only_graph()
    structural = sample_structure(g, cfg, np.random.default_rng(16))
    generative, _out_names, _param_names = build_world_model(g, cfg, structural, cfg.T)
    oracle = build_oracle_model(
        g,
        cfg,
        structural,
        {
            "channels": np.zeros((cfg.T, 2)),
            "controls": np.zeros((cfg.T, 1)),
            "sales": np.zeros(cfg.T),
            "saturation_scale": np.ones(2),
        },
    )
    media_amplitude = np.sqrt(2.0 * 1.2**2)

    for group, relative_std in (("rw_b", 0.07), ("rw_y", 0.02)):
        generated = np.asarray(pm.draw(generative[f"{group}_std"], draws=1, random_seed=17))
        inferred = np.asarray(pm.draw(oracle[f"{group}_std"], draws=1, random_seed=18))
        np.testing.assert_allclose(generated, relative_std * media_amplitude, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(inferred, generated, rtol=0.0, atol=1e-14)


def test_oracle_logp_finite_at_truth(world_and_oracle):
    """Total oracle logp (priors + likelihood) is finite at the drawn world."""
    _gen_model, oracle, _sampled_oracle, world = world_and_oracle
    point = _oracle_truth_point(oracle, world)
    logp = oracle.compile_logp()(point)
    assert np.isfinite(logp)


def test_oracle_deterministics_shapes(world_and_oracle):
    _gen_model, marginal, sampled, world = world_and_oracle
    T, K = world["channels"].shape
    J = world["demand"].shape[1]

    marginal_names = {d.name for d in marginal.deterministics}
    assert {"contributions", "sales_mu"} <= marginal_names
    assert not {"baseline", "demand"} & marginal_names
    assert tuple(marginal["contributions"].shape.eval()) == (T, K)
    assert tuple(marginal["sales_mu"].shape.eval()) == (T,)

    sampled_names = {d.name for d in sampled.deterministics}
    assert {"contributions", "baseline", "sales_mu", "demand"} <= sampled_names
    assert {"eps_d", "eps_b"} <= {rv.name for rv in sampled.free_RVs}
    assert tuple(sampled["contributions"].shape.eval()) == (T, K)
    assert tuple(sampled["demand"].shape.eval()) == (T, J)


@pytest.mark.parametrize(
    ("smoothness", "expected_width"),
    ((0.0, 1), (0.5, 13), (1.0, 26)),
    ids=("width-1", "width-13", "width-26"),
)
def test_walk_basis_reproduces_symbolic_random_walk(smoothness: float, expected_width: int):
    """The NumPy operator is exact for every representative smoothing width."""
    T = 40
    std = 0.73
    eps = np.random.default_rng(11).normal(size=T)
    width = _kernel_width(smoothness, T, rw_smoothness_max_weeks=26)
    assert width == expected_width

    expected = np.asarray(
        symbolic_random_walk(
            T,
            mean=0.0,
            std=std,
            smoothness=smoothness,
            positive_only=False,
            rw_smoothness_max_weeks=26,
            eps=eps,
        ).eval()
    )
    actual = std * _walk_basis(T, width) @ eps
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_marginal_sales_logp_matches_numpy_mvn():
    """The marginal sales factor is the independently assembled exact MvNormal."""
    cfg = _small_cfg(
        n_treatments=1,
        T=12,
        l_max=2,
        adstock_burn_in=2,
        nonlinearity="linear",
        outcome_std_mode="absolute",
    )
    g = _direct_only_graph(1)
    g["g_db"][:] = 1
    g["g_zb"][:] = 1
    structural = sample_structure(g, cfg, np.random.default_rng(31))
    structural["adstock_family"][:] = 0
    structural["sat_family"][:] = 0
    structural["smoothness_d"][:] = 0.0
    structural["smoothness_b"][:] = 0.5

    T_full = cfg.T + cfg.adstock_burn_in
    rows = np.arange(cfg.adstock_burn_in, T_full)
    values = {
        "beta": np.array([1.2]),
        "delta_db": np.array([0.3]),
        "rho_zb": np.array([0.2]),
        "rw_b_mean": np.array([4.0]),
        "rw_b_std": np.array([0.4]),
        "rw_y_std": np.array([0.1]),
    }
    channels = np.linspace(0.5, 3.0, cfg.T)[:, None]
    controls = np.linspace(-0.4, 0.6, cfg.T)[:, None]
    mean_for_sales = values["rw_b_mean"][0] + controls[:, 0] * values["rho_zb"][0]
    mean_for_sales = mean_for_sales + values["beta"][0] * channels[:, 0] / 2.0
    residual = 0.15 * np.sin(np.arange(cfg.T))
    sales = mean_for_sales + residual - residual.mean()
    oracle = build_oracle_model(
        g,
        cfg,
        structural,
        {
            "channels": channels,
            "controls": controls,
            "sales": sales,
            "saturation_scale": np.array([2.0]),
        },
    )
    likelihood = oracle.compile_logp(vars=[oracle["sales"]])(_oracle_truth_point(oracle, values))

    def gram(smoothness: float) -> np.ndarray:
        width = _kernel_width(
            smoothness,
            T_full,
            rw_smoothness_max_weeks=cfg.rw_smoothness_max_weeks,
        )
        steps = np.tril(np.ones((T_full, T_full)))
        columns = _smooth_columns_numpy(steps, width)
        columns = columns - columns.mean(axis=0, keepdims=True)
        basis = columns / _centred_walk_scale(T_full, width)
        restricted = basis[rows]
        return restricted @ restricted.T

    mu = values["rw_b_mean"][0] + controls[:, 0] * values["rho_zb"][0]
    mu = mu + values["beta"][0] * channels[:, 0] / 2.0
    covariance = (values["rw_b_std"][0] ** 2) * gram(float(structural["smoothness_b"][0]))
    covariance = covariance + (values["rw_y_std"][0] ** 2) * np.eye(rows.size)
    covariance = covariance + (values["delta_db"][0] ** 2) * gram(
        float(structural["smoothness_d"][0])
    )
    covariance = covariance + 1e-12 * np.eye(rows.size)
    expected = multivariate_normal.logpdf(
        sales,
        mean=mu,
        cov=Covariance.from_cholesky(np.linalg.cholesky(covariance)),
    )

    np.testing.assert_allclose(likelihood, expected, rtol=1e-8, atol=1e-8)


def test_oracle_registers_only_live_mechanism_shape_params():
    """Mechanism priors follow the concrete family union, not all families."""
    cfg = _small_cfg(n_treatments=2, T=12, adstock_burn_in=0)
    g = _direct_only_graph(2)
    data = {
        "channels": np.zeros((cfg.T, 2)),
        "controls": np.zeros((cfg.T, 1)),
        "sales": np.zeros(cfg.T),
        "saturation_scale": np.ones(2),
    }

    def oracle_with_families(adstock_family: int, sat_family: int):
        structural = sample_structure(g, cfg, np.random.default_rng(32))
        structural["adstock_family"][:] = adstock_family
        structural["sat_family"][:] = sat_family
        return structural, build_oracle_model(g, cfg, structural, data)

    assert ADSTOCK_FAMILY_PARAM_NAMES["geometric"] == ("adstock_alpha",)
    assert SATURATION_FAMILY_PARAM_NAMES["michaelis_menten"] == ("mm_kappa_mult",)

    geometric_mm_structure, geometric_mm = oracle_with_families(1, 3)
    geometric_mm_names = {rv.name for rv in geometric_mm.free_RVs}
    assert _live_mechanism_param_names(geometric_mm_structure) == (
        "adstock_alpha",
        "mm_kappa_mult",
    )
    assert {"adstock_alpha", "mm_kappa_mult"} <= geometric_mm_names
    assert (
        not {
            "weibull_lam",
            "weibull_k",
            "hill_slope",
            "hill_kappa_mult",
            "logistic_lam",
            "tanh_c",
            "root_alpha",
        }
        & geometric_mm_names
    )

    weibull_hill_structure, weibull_hill = oracle_with_families(2, 1)
    weibull_hill_names = {rv.name for rv in weibull_hill.free_RVs}
    assert _live_mechanism_param_names(weibull_hill_structure) == (
        "weibull_lam",
        "weibull_k",
        "hill_slope",
        "hill_kappa_mult",
    )
    assert {"weibull_lam", "weibull_k", "hill_slope", "hill_kappa_mult"} <= weibull_hill_names
    assert not {"adstock_alpha", "mm_kappa_mult"} & weibull_hill_names


def test_oracle_likelihood_starts_at_first_reproducible_response_week():
    """The likelihood begins after, but not before, unpersisted adstock history."""
    cfg = _small_cfg(
        T=8,
        l_max=4,
        adstock_burn_in=4,
        nonlinearity="linear",
        adstock_family_probs={
            "none": 0.0,
            "geometric": 1.0,
            "weibull": 0.0,
        },
        adstock_alpha_range=(0.79, 0.81),
    )
    g = _direct_only_graph()
    structural = sample_structure(g, cfg, np.random.default_rng(20))
    assert np.all(structural["adstock_family"] == 1)
    generative, output_names, _ = build_world_model(g, cfg, structural, cfg.T)
    drawn = {
        name: value[0]
        for name, value in pg.draw_worlds(
            generative, output_names + SHARED_RV_NAMES, seed=21, draws=1
        ).items()
    }
    oracle = build_oracle_model(
        g,
        cfg,
        structural,
        {key: drawn[key] for key in ("channels", "controls", "sales", "saturation_scale")},
        latent="sampled",
    )

    warmup = cfg.l_max - 1
    oracle_contributions = _oracle_deterministic_at_truth(oracle, drawn, "contributions")
    truth = drawn["contributions_observed"]
    assert tuple(oracle["sales"].shape.eval()) == (cfg.T - warmup,)
    assert tuple(oracle["contributions"].shape.eval()) == (cfg.T, 2)
    assert tuple(oracle["baseline"].shape.eval()) == (cfg.T,)
    assert tuple(oracle["sales_mu"].shape.eval()) == (cfg.T,)
    np.testing.assert_allclose(oracle_contributions[warmup], truth[warmup], rtol=0.0, atol=1e-15)
    before_warmup_error = np.abs(oracle_contributions[warmup - 1] - truth[warmup - 1]).max()
    assert before_warmup_error > 1e-3


def test_oracle_rejects_likelihood_without_reproducible_weeks():
    # L1 rejects this horizon at SCMPrior.validate(). Construct it directly to
    # keep the direct build_oracle_model defensive guard covered.
    cfg = SCMPrior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        T=4,
        l_max=5,
        adstock_burn_in=5,
        adstock_family_probs={
            "none": 0.0,
            "geometric": 1.0,
            "weibull": 0.0,
        },
    )
    g = _direct_only_graph()
    structural = sample_structure(g, cfg, np.random.default_rng(21))
    assert np.all(structural["adstock_family"] == 1)

    with pytest.raises(ValueError, match="no reproducible observations"):
        build_oracle_model(
            g,
            cfg,
            structural,
            {
                "channels": np.zeros((cfg.T, 2)),
                "controls": np.zeros((cfg.T, 1)),
                "sales": np.zeros(cfg.T),
                "saturation_scale": np.ones(2),
            },
        )


def test_oracle_warmup_exempts_identity_adstock():
    """With burn-in, only non-identity direct adstock loses response history."""
    shapes = {}
    for name, family, family_probs in (
        ("identity", 0, {"none": 1.0, "geometric": 0.0, "weibull": 0.0}),
        ("geometric", 1, {"none": 0.0, "geometric": 1.0, "weibull": 0.0}),
    ):
        cfg = pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            T=20,
            l_max=5,
            nonlinearity="linear",
            adstock_family_probs=family_probs,
            edge_budget={
                "cy": (1, 1),
                "dc": 0,
                "db": 0,
                "zb": 0,
                "dz": 0,
                "zc": 0,
                "cc": 0,
                "zz": 0,
            },
        )
        assert cfg.adstock_burn_in == cfg.l_max
        world = pg.sample_scm(cfg, seed=23, max_eps_draws=4)
        assert np.array_equal(world.params["adstock_family"], np.array([family]))

        oracle = world.oracle_model()
        shapes[name] = tuple(oracle["sales"].shape.eval())
        if family == 0:
            truth_values = {**world.params, **world.exogenous}
            media_amplitude = np.sqrt(np.sum((world.g["g_cy"] * world.params["beta"]) ** 2))
            for group in ("rw_b", "rw_y"):
                truth_values[f"{group}_mean"] = world.params[group]["mean"]
                truth_values[f"{group}_std"] = world.params[group]["std"]
                truth_values[f"{group}_std_rel"] = world.params[group]["std"] / media_amplitude
            oracle_contributions = _oracle_deterministic_at_truth(
                oracle, truth_values, "contributions"
            )
            np.testing.assert_allclose(
                oracle_contributions,
                world.data["contributions_observed"],
                rtol=0.0,
                atol=1e-15,
            )

    assert shapes == {"identity": (20,), "geometric": (16,)}


@pytest.mark.parametrize(
    ("adstock_family", "has_gradient"),
    [(0, True), (1, True), (2, False)],
    ids=("identity", "geometric", "weibull"),
)
def test_oracle_logp_and_gradient_capability_by_adstock_family(
    adstock_family: int, has_gradient: bool
):
    """Only pymc-marketing's Weibull normalization prevents an oracle gradient."""
    cfg = _small_cfg(n_treatments=1, T=12, adstock_burn_in=0)
    g = _direct_only_graph(1)
    structural = sample_structure(g, cfg, np.random.default_rng(12))
    structural["adstock_family"][:] = adstock_family
    structural["sat_family"][:] = 0
    model, output_names, parameter_names = build_world_model(g, cfg, structural, cfg.T)
    drawn = pg.draw_worlds(model, output_names + parameter_names, seed=19, draws=1)
    data = {name: drawn[name][0] for name in ("channels", "controls", "sales", "saturation_scale")}
    oracle = build_oracle_model(g, cfg, structural, data)

    assert np.isfinite(oracle.compile_logp()(oracle.initial_point()))
    if has_gradient:
        gradient = oracle.compile_dlogp()(oracle.initial_point())
        assert np.isfinite(gradient).all()
        assert np.max(np.abs(gradient)) > 0.0
    else:
        with pytest.raises(NotImplementedError, match=r"Min\{axis=0\}"):
            oracle.compile_dlogp()


def test_oracle_rejects_bad_shapes(world_and_oracle):
    _gen_model, _oracle, _sampled_oracle, world = world_and_oracle
    cfg = _small_cfg()
    rng = np.random.default_rng(2)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=1, J_active=1)
    g_act = _slice_g_active(g, 2, 1, 1)
    structural = sample_structure(g_act, cfg, rng)
    missing_scale = {
        "channels": world["channels"],
        "controls": world["controls"],
        "sales": world["sales"],
    }
    with pytest.raises(ValueError, match="saturation_scale"):
        build_oracle_model(g_act, cfg, structural, missing_scale)

    bad = {
        "channels": world["channels"][:, :1],  # wrong K
        "controls": world["controls"],
        "sales": world["sales"],
        "saturation_scale": world["saturation_scale"],
    }
    with pytest.raises(ValueError, match="data shapes"):
        build_oracle_model(g_act, cfg, structural, bad)

    bad_sales = {
        "channels": world["channels"],
        "controls": world["controls"],
        "sales": world["sales"][:, None],
        "saturation_scale": world["saturation_scale"],
    }
    with pytest.raises(ValueError, match="sales must have shape"):
        build_oracle_model(g_act, cfg, structural, bad_sales)

    empty = {
        "channels": np.empty((0, 2)),
        "controls": np.empty((0, 1)),
        "sales": np.empty(0),
        "saturation_scale": np.ones(2),
    }
    with pytest.raises(ValueError, match="at least one observation"):
        build_oracle_model(g_act, cfg, structural, empty)

    for key, fill in (("channels", np.nan), ("controls", np.inf), ("sales", np.nan)):
        nonfinite = {
            "channels": world["channels"].copy(),
            "controls": world["controls"].copy(),
            "sales": world["sales"].copy(),
            "saturation_scale": world["saturation_scale"].copy(),
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
    cfg = _small_cfg(
        prior_conditioning=True,
        adstock_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
    )
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
    marginal = world.oracle_model()
    marginal_names = {deterministic.name for deterministic in marginal.deterministics}
    assert {"contributions", "sales_mu"} <= marginal_names
    assert not {"baseline", "demand"} & marginal_names

    sampled = world.oracle_model(latent="sampled")
    assert {"baseline", "contributions", "sales_mu", "demand"} <= {
        deterministic.name for deterministic in sampled.deterministics
    }
    for oracle in (marginal, sampled):
        assert not any(rv.name.startswith("channel_shock") for rv in oracle.free_RVs)


@pytest.mark.slow
def test_nuts_smoke_recovers_params():
    """Short NUTS fit on a small world: truth within wide posterior bounds."""
    cfg = _small_cfg(
        outcome_std_mode="absolute",
        rw_baseline_std_sigma=0.05,
        rw_sales_std_sigma=0.02,
    )
    world = pg.sample_scm(cfg, seed=8)
    with world.oracle_model(latent="sampled"):
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
    # The fitted sales mean tracks the observed likelihood window.
    warmup = (
        cfg.l_max - 1
        if cfg.adstock_burn_in > 0
        and np.any(direct & (np.asarray(world.params["adstock_family"]) != 0))
        else 0
    )
    mu = post["sales_mu"].mean(("chain", "draw")).values
    r = np.corrcoef(mu[warmup:], world.data["sales"][warmup:])[0, 1]
    assert r > 0.8
