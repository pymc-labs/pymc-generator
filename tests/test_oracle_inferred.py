"""Inferred-width likelihood parity, isolation and joint reconstruction."""

from itertools import product

import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution

from pymc_generator.oracle_inferred import sample_inferred_walk_predictions
from pymc_generator.oracle_sampling import OracleSamplingConfig
from pymc_generator.oracle_smoothness import walk_width_prior
from pymc_generator.random_walk import _kernel_width
from pymc_generator.sampler import SCMPrior
from pymc_generator.world_model import build_oracle_model


def case(n_latent=1, floor=0.0, scope="intercept"):
    cfg = SCMPrior(
        n_time_steps=8,
        n_treatments=1,
        n_covariates=1,
        n_latent=n_latent,
        l_max=3,
        carryover_burn_in=2,
        rw_smoothness_max_weeks=3,
        rw_smoothness_alpha=2.0,
        rw_smoothness_beta=3.0,
        baseline_floor=floor,
        baseline_floor_scope=scope,
        outcome_std_mode="absolute",
    )
    graph = {"g_cy": np.ones(1), "g_zy": np.ones(1), "g_dy": np.ones(n_latent)}
    structure = {"carryover_family": np.array([0]), "sat_family": np.array([0])}
    data = {
        "treatments": np.linspace(0.1, 1.0, 8)[:, None],
        "covariates": np.linspace(-2.0, 2.0, 8)[:, None],
        "outcome": np.linspace(2.0, 4.0, 8),
        "saturation_scale": np.ones(1),
    }
    return cfg, graph, structure, data


def build(args, **kwargs):
    cfg, graph, structure, data = args
    return build_oracle_model(
        graph,
        cfg,
        structure,
        data,
        latent="sampled",
        walk_smoothness="inferred",
        observed_indices=[0, 2, 4],
        **kwargs,
    )


def likelihood_at(model, point):
    return model.compile_logp(vars=model.potentials or model.observed_RVs)(point)


@pytest.mark.parametrize("n_latent", [0, 1])
@pytest.mark.parametrize(
    "floor,scope", [(None, "intercept"), (0.0, "intercept"), (0.0, "non_treatment")]
)
def test_mixture_equals_enumerated_supplied_width_likelihood(n_latent, floor, scope):
    args = case(n_latent, floor, scope)
    cfg, graph, structure, data = args
    # Force real floor crossings rather than testing a clip that never activates.
    cfg.rw_baseline_mean_range = (0.0, 0.0)
    cfg.zy_coeff_range = (1.0, 1.0)
    cfg.dy_coeff_range = (1.0, 1.0)
    model = build(args)
    point = model.initial_point()
    # Nonzero innovations distinguish widths; negative excursions exercise floors.
    point["eps_b"] = np.linspace(-2.0, 2.0, 10)
    if n_latent:
        point["eps_d"] = np.linspace(1.0, -2.0, 10)[:, None]
    # Known-width no-carryover models retain the same observation indices.
    prior = walk_width_prior(10, 3, 2.0, 3.0)
    terms = []
    for indices in product(range(3), repeat=n_latent + 1):
        smooth = (prior.lower[list(indices)] + prior.upper[list(indices)]) / 2
        known_structure = dict(structure, smoothness_b=smooth[:1], smoothness_d=smooth[1:])
        # Legacy sampled builder cannot stack zero latent columns; the independent
        # zero-latent case below instead uses a zero-gated latent of known width.
        known_graph = graph
        known_point = dict(point)
        if not n_latent:
            known_graph = dict(graph, g_dy=np.zeros(1))
            known_structure["smoothness_d"] = np.array([0.5])
            known_point["eps_d"] = np.zeros((10, 1))
            known_point["delta_dy_interval__"] = np.zeros(1)
        known = build_oracle_model(
            known_graph, cfg, known_structure, data, latent="sampled", observed_indices=[0, 2, 4]
        )
        # The compiled likelihood consumes only its graph inputs.
        expression = known.logp(vars=known.observed_RVs)
        fn = pytensor.function(known.value_vars, expression, on_unused_input="ignore")
        values = known.initial_point()
        values.update(known_point)
        terms.append(
            fn(*[values[v.name] for v in known.value_vars])
            + np.log(prior.probabilities[list(indices)]).sum()
        )
    expression = model.logp(vars=model.potentials)
    fn = pytensor.function(model.value_vars, expression, on_unused_input="ignore")
    actual = fn(*[point[v.name] for v in model.value_vars])
    np.testing.assert_allclose(actual, logsumexp(terms), rtol=1e-11)
    probabilities = model.replace_rvs_by_values([model["walk_width_probabilities"]])[0]
    fn = pytensor.function(model.value_vars, probabilities, on_unused_input="ignore")
    np.testing.assert_allclose(
        fn(*[point[v.name] for v in model.value_vars]), np.exp(terms - logsumexp(terms))
    )
    baseline = model.replace_rvs_by_values([model._inferred_walk_components["outputs"][1]])[0]
    paths = pytensor.function(model.value_vars, baseline, on_unused_input="ignore")(
        *[point[v.name] for v in model.value_vars]
    )
    if floor is None:
        assert paths.min() < 0
    elif scope == "non_treatment":
        assert paths.min() >= 0


def test_truth_and_query_values_are_not_inputs_and_context_changes_likelihood():
    args = case()
    model = build(args)
    point = model.initial_point()
    original = model.compile_logp()(point)
    cfg, graph, structure, data = args
    data = dict(data, outcome=data["outcome"].copy())
    data["outcome"][[1, 3, 5, 6, 7]] = np.nan
    structure = dict(structure, smoothness_b=object(), smoothness_d=object())
    other = build((cfg, graph, structure, data))
    assert not any(
        name.startswith("walk_width_") and name.endswith("_data") for name in other.named_vars
    )
    np.testing.assert_equal(other.compile_logp()(point), original)
    data["outcome"][0] += 10
    changed = build((cfg, graph, structure, data))
    assert changed.compile_logp()(point) != original
    assert np.isfinite(model.compile_dlogp()(point)).all()


def test_width_mass_and_smoothness_reconstruction():
    prior = walk_width_prior(5, 26, 1.0, 20.0)
    np.testing.assert_allclose(
        prior.probabilities,
        beta_distribution.cdf(prior.upper, 1.0, 20.0)
        - beta_distribution.cdf(prior.lower, 1.0, 20.0),
        atol=1e-15,
    )
    rng = np.random.default_rng(71)
    for i, width in enumerate(prior.widths):
        for _ in range(10):
            smooth = prior.sample_smoothness(i, rng)
            assert _kernel_width(smooth, 5, rw_smoothness_max_weeks=26) == width


def test_joint_predictions_preserve_draws_and_width_choice():
    model = build(case())
    # A small prior draw is sufficient to test reconstruction, not convergence.
    values = pm.draw(model.free_RVs, draws=2, random_seed=32)
    posterior = xr.Dataset(
        {
            rv.name: (
                ("chain", "draw", *[f"{rv.name}_{i}" for i in range(value.ndim - 1)]),
                value[None],
            )
            for rv, value in zip(model.free_RVs, values, strict=True)
        },
        coords={"chain": [3], "draw": [7, 11]},
    )
    result = sample_inferred_walk_predictions(model, posterior, random_seed=15)
    xr.testing.assert_equal(
        result, sample_inferred_walk_predictions(model, posterior, random_seed=15)
    )
    np.testing.assert_array_equal(result.draw, [7, 11])
    assert result.outcome.shape == (1, 2, 8)
    np.testing.assert_allclose(
        result.outcome_mu, result.baseline + result.contributions.sum("channel")
    )
    for width, smooth in zip(
        result.walk_width.values.ravel(), result.walk_smoothness.values.ravel(), strict=True
    ):
        assert _kernel_width(smooth, 10, rw_smoothness_max_weeks=3) == width


def test_mode_is_explicit_and_discrete_rvs_are_not_sampled():
    with pytest.raises(ValueError, match="requires"):
        OracleSamplingConfig(walk_smoothness="inferred")
    config = OracleSamplingConfig(latent="sampled", walk_smoothness="inferred")
    assert config.to_dict()["walk_smoothness"] == "inferred"
    model = build(case())
    assert all(rv.dtype.startswith("float") for rv in model.free_RVs)


@pytest.mark.slow
def test_inferred_one_shot_nutpie_execution():
    model = build(case())
    trace = pm.sample(
        model=model,
        nuts_sampler="nutpie",
        draws=8,
        tune=12,
        chains=1,
        cores=1,
        random_seed=53,
        progressbar=False,
        compute_convergence_checks=False,
    )
    paired = sample_inferred_walk_predictions(model, trace["posterior"], random_seed=54)
    assert paired.outcome.shape == (1, 8, 8)
    assert trace["posterior"]["walk_width_probabilities"].shape == (1, 8, 9)
    np.testing.assert_allclose(
        trace["posterior"]["walk_width_probabilities"].sum("walk_combination"), 1.0
    )
    print("one-shot smoke divergences:", int(trace["sample_stats"]["diverging"].sum()))
