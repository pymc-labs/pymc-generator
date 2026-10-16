"""Scientific response parity, active shapes, and paired prediction retention."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_generator.oracle_family import FamilyOracle, OracleInputs, family_response
from pymc_generator.oracle_sampling import OracleSamplingConfig
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior
from pymc_generator.signal_diagnostics import _carryover_weights
from pymc_generator.world_model import build_oracle_model


@pytest.mark.parametrize("saturation", SATURATION_FAMILY_KEYS)
@pytest.mark.parametrize("carryover", CARRYOVER_FAMILY_KEYS)
def test_vectorized_response_matches_numpy(saturation, carryover):
    parameters = {
        "carryover_alpha": np.array([0.3, 0.7]),
        "weibull_lam": np.array([2.1, 3.0]),
        "weibull_k": np.array([1.5, 2.5]),
        "hill_slope": np.array([1.1, 2.1]),
        "hill_kappa_mult": np.array([0.8, 1.2]),
        "logistic_lam": np.array([0.7, 1.5]),
        "mm_kappa_mult": np.array([0.8, 1.2]),
        "tanh_c": np.array([0.7, 1.3]),
        "root_alpha": np.array([0.4, 0.7]),
    }
    x = np.linspace(0, 4, 24).reshape(12, 2)
    anchors = np.array([1.3, 2.0])
    symbolic_x = pt.dmatrix("x")
    response = family_response(
        symbolic_x, anchors, parameters, saturation=saturation, carryover=carryover, l_max=4
    )
    evaluate = pytensor.function([symbolic_x], response, mode="FAST_COMPILE")
    actual = evaluate(x)
    expected = []
    for k in range(2):
        weights = _carryover_weights(
            CARRYOVER_FAMILY_KEYS.index(carryover),
            parameters["carryover_alpha"][k],
            parameters["weibull_lam"][k],
            parameters["weibull_k"][k],
            4,
        )
        ad = x[:, k] if weights is None else np.convolve(x[:, k], weights)[: len(x)]
        ratio = ad / anchors[k]
        if saturation == "linear":
            value = ratio
        elif saturation == "hill":
            a = ratio ** parameters["hill_slope"][k]
            value = a / (a + parameters["hill_kappa_mult"][k] ** parameters["hill_slope"][k])
        elif saturation == "logistic":
            value = np.tanh(parameters["logistic_lam"][k] * ratio / 2)
        elif saturation == "michaelis_menten":
            value = ratio / (ratio + parameters["mm_kappa_mult"][k])
        elif saturation == "tanh":
            value = np.tanh(ratio / parameters["tanh_c"][k])
        else:
            value = ratio ** parameters["root_alpha"][k]
        expected.append(value)
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), rtol=1e-10, atol=1e-12)


def inputs(p=2, c=2, context=6):
    return OracleInputs(
        channels=np.linspace(0.1, 2.0, 12 * p).reshape(12, p),
        controls=np.linspace(-1.0, 1.0, 12 * c).reshape(12, c),
        context_indices=np.arange(context),
        context_outcome=np.linspace(2.0, 4.0, context),
        saturation_scale=np.ones(p),
        channel_ids=tuple(range(5, 5 + p)),
        control_ids=tuple(range(10, 10 + c)),
        channel_gates=np.ones(p),
        control_gates=np.ones(c),
        latent_gates=np.ones(1),
    )


@pytest.fixture(scope="module")
def worker():
    return FamilyOracle(
        SCMPrior(
            n_time_steps=12,
            n_latent=1,
            l_max=3,
            rw_smoothness_max_weeks=3,
            outcome_std_mode="absolute",
            # Exact scalar avoids a legacy constant-folded float32 log(sigma)
            # differing from the runtime float64 prior normalization.
            rw_outcome_std_sigma=1.0,
            baseline_floor=0.0,
            baseline_floor_scope="non_treatment",
        ),
        saturation="linear",
        carryover="none",
    )


def test_family_matches_one_shot_model(worker):
    data = inputs()
    bound = worker.bind(data)
    cfg = worker.cfg
    observed = np.zeros(12)
    observed[data.context_indices] = data.context_outcome
    model = build_oracle_model(
        {"g_cy": data.channel_gates, "g_zy": data.control_gates, "g_dy": data.latent_gates},
        cfg,
        {"carryover_family": np.zeros(2, dtype=int), "sat_family": np.zeros(2, dtype=int)},
        {
            "treatments": data.channels,
            "covariates": data.controls,
            "outcome": observed,
            "saturation_scale": data.saturation_scale,
        },
        latent="sampled",
        walk_smoothness="inferred",
        observed_indices=data.context_indices,
    )
    point = {name: np.full_like(value, 0.01) for name, value in model.initial_point().items()}
    np.testing.assert_allclose(
        bound.evaluate(bound.initial_point)[0], model.compile_logp()(point), rtol=1e-11
    )
    # Check real derivatives, not just finite values.
    q = bound.initial_point.copy()
    lp, gradient, _ = bound.evaluate(q)
    for i in range(len(q)):
        step = np.zeros_like(q)
        step[i] = 1e-5
        numeric = (bound.evaluate(q + step)[0] - bound.evaluate(q - step)[0]) / 2e-5
        np.testing.assert_allclose(gradient[i], numeric, rtol=3e-5, atol=1e-6)


def test_counts_context_gates_and_regime_priors_rebind_without_compilation(worker):
    import copy

    first = worker.bind(inputs())
    original = first.evaluate(first.initial_point)
    signatures = worker.kernel.signatures
    cfg = copy.deepcopy(worker.cfg)
    cfg.rw_baseline_mean_range = (5.0, 11.0)
    cfg.rw_baseline_std_sigma = 2.5
    cfg.rw_outcome_std_sigma = 0.8
    cfg.rw_smoothness_alpha, cfg.rw_smoothness_beta = 1.0, 20.0
    # Input-only generator settings cannot prevent reuse of an outcome model.
    cfg.n_treatments, cfg.n_covariates = 10, 10
    changed = inputs(p=3, c=1, context=9)
    changed.channel_gates[1] = 0
    second = worker.bind(changed, cfg=cfg)
    third = worker.bind(inputs(p=1, c=0, context=4), cfg=cfg)
    assert worker.kernel.signatures == signatures
    assert second.output_shapes["beta"] == (3,)
    assert third.output_shapes["rho_zy"] == (0,)
    # Linear/no-carryover has exactly one coefficient per active input.
    assert third.layout.size == first.layout.size - 3
    assert np.isfinite(second.evaluate(second.initial_point)[0])
    np.testing.assert_equal(first.evaluate(first.initial_point)[0], original[0])
    same_inputs_new_priors = worker.bind(inputs(), cfg=cfg)
    assert same_inputs_new_priors.evaluate(same_inputs_new_priors.initial_point)[0] != original[0]
    predicted = worker.predict(second, second.initial_point, np.random.default_rng(8))
    np.testing.assert_allclose(
        predicted["outcome_mu"], predicted["baseline"] + predicted["contributions"].sum(axis=1)
    )
    cfg.l_max += 1
    with pytest.raises(ValueError, match="incompatible"):
        worker.bind(inputs(), cfg=cfg)


@pytest.mark.slow
def test_real_nutpie_fits_two_dynamic_bindings(worker):
    traces = []
    worker.bind(inputs())  # First execution materializes the one lazy Numba signature.
    signatures = worker.kernel.signatures
    for data in (inputs(), inputs(p=3, c=1, context=9)):
        bound = worker.bind(data)
        config = OracleSamplingConfig(
            latent="sampled",
            walk_smoothness="inferred",
            draws=8,
            tune=12,
            chains=1,
            cores=1,
            random_seed=47,
        )
        trace = worker.sample(bound, config)
        assert trace["posterior"].sizes["draw"] == 8
        assert trace["sample_stats"].sizes["draw"] == 8
        assert trace["posterior"]["beta"].shape[-1] == data.channels.shape[1]
        np.testing.assert_array_equal(trace["posterior"].channel, data.channel_ids)
        state = trace["posterior"]["_state"].values[0, 0]
        prediction = worker.predict(bound, state, np.random.default_rng(8))
        assert prediction["outcome"].shape == (12,)
        paired = worker.predict_posterior(bound, trace["posterior"], random_seed=8)
        assert paired.outcome.shape == (1, 8, 12)
        np.testing.assert_array_equal(paired.channel, data.channel_ids)
        np.testing.assert_array_equal(paired.draw, trace["posterior"].draw)
        np.testing.assert_allclose(
            paired.outcome_mu, paired.baseline + paired.contributions.sum("channel")
        )
        np.testing.assert_equal(paired.outcome.values[0, 0], prediction["outcome"])
        traces.append(trace)
    assert worker.kernel.signatures == signatures
    with pytest.raises(ValueError, match="immutable binding"):
        worker.predict_posterior(worker.bind(inputs()), traces[-1]["posterior"], random_seed=8)
    # Tiny fits validate execution, not convergence; retain actual diagnostics.
    print("smoke divergences:", [int(t["sample_stats"]["diverging"].sum()) for t in traces])


@pytest.mark.slow
@pytest.mark.parametrize("saturation", SATURATION_FAMILY_KEYS)
@pytest.mark.parametrize("carryover", CARRYOVER_FAMILY_KEYS)
def test_family_gradient_and_reuse_all_response_pairs(saturation, carryover):
    instance = FamilyOracle(
        SCMPrior(
            n_time_steps=12,
            n_latent=1,
            l_max=3,
            rw_smoothness_max_weeks=2,
            outcome_std_mode="absolute",
            rw_outcome_std_sigma=1.0,
        ),
        saturation=saturation,
        carryover=carryover,
    )
    data = inputs()
    data.channels[:3] = 0.0  # Recorded inputs include zero-spend intervals.
    bound = instance.bind(data)
    q = bound.initial_point.copy()
    gradient = bound.evaluate(q)[1]
    for i in range(len(q)):
        step = np.zeros_like(q)
        step[i] = 1e-5
        numeric = (bound.evaluate(q + step)[0] - bound.evaluate(q - step)[0]) / 2e-5
        np.testing.assert_allclose(gradient[i], numeric, rtol=3e-5, atol=2e-6)
    signatures = instance.kernel.signatures
    instance.bind(inputs(p=3, c=1, context=9))
    assert instance.kernel.signatures == signatures


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_reference_priors_and_no_latent_path(mode):
    from dataclasses import replace

    instance = FamilyOracle(
        SCMPrior(
            n_time_steps=12,
            n_latent=0,
            l_max=3,
            rw_smoothness_max_weeks=2,
            outcome_std_mode=mode,
            treatment_reference_contribution_range=(0.1, 3.9),
            covariate_reference_contribution_range=(0.1, 3.9),
            mm_scale_prior="log_uniform",
            baseline_floor=0.0,
            rw_baseline_mean_range=(4.0, 4.0),
        ),
        saturation="michaelis_menten",
        carryover="geometric",
    )
    bound = instance.bind(replace(inputs(), latent_gates=np.zeros(0)))
    assert "eps_d" not in bound.layout.names
    assert np.isfinite(bound.evaluate(bound.initial_point)[1]).all()
    signatures = instance.kernel.signatures
    instance.bind(replace(inputs(p=3, c=1, context=9), latent_gates=np.zeros(0)))
    assert instance.kernel.signatures == signatures
