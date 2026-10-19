"""Native family masks, joint likelihood, and compile-once sampling checks."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytest

from pymc_generator.oracle_family import OracleInputs
from pymc_generator.oracle_native import NativeFamilyTemplate
from pymc_generator.oracle_sampling import CompiledOracle, OracleSamplingConfig
from pymc_generator.sampler import SCMPrior


def config():
    return SCMPrior(
        n_time_steps=8,
        n_latent=0,
        l_max=2,
        rw_smoothness_max_weeks=2,
        outcome_std_mode="absolute",
        rw_outcome_std_sigma=1,
    )


def inputs(p=1, c=0, context=4):
    return OracleInputs(
        channels=np.linspace(0.2, 2, 8 * p).reshape(8, p),
        controls=np.linspace(-1, 1, 8 * c).reshape(8, c),
        context_indices=np.arange(context),
        context_outcome=np.linspace(2, 3, context),
        saturation_scale=np.ones(p),
        channel_ids=tuple(range(9, 9 + p)),
        control_ids=tuple(range(20, 20 + c)),
        channel_gates=np.ones(p),
        control_gates=np.ones(c),
    )


def test_masks_remove_inactive_parameters_from_likelihood_not_priors():
    template = NativeFamilyTemplate.build(
        config(), saturation="linear", carryover="none", max_channels=2, max_controls=1
    )
    assert not template.model.potentials
    assert [rv.name for rv in template.model.observed_RVs] == ["outcome"]
    payload, _ = template.payload(inputs())
    with template.model:
        pm.set_data(payload)
    likelihood = template.model.compile_logp(vars=template.model.observed_RVs)
    prior = template.model.compile_logp(vars=template.model.free_RVs)
    point = template.model.initial_point()
    changed = {k: np.array(v, copy=True) for k, v in point.items()}
    changed["beta_interval__"][1] += 2
    changed["rho_zy_interval__"] += 2
    original_logp = likelihood(point)
    assert original_logp == pytest.approx(likelihood(changed), abs=1e-12)
    assert prior(point) != pytest.approx(prior(changed))
    # Masked weeks and columns cannot affect the context likelihood.
    other = {k: np.array(v, copy=True) for k, v in payload.items()}
    other["context_outcome"][4:] = 100
    other["channels"][:, 1] = 100
    other["controls"][:] = 100
    with template.model:
        pm.set_data(other)
    assert original_logp == pytest.approx(likelihood(point), abs=1e-12)
    assert original_logp == pytest.approx(likelihood(changed), abs=1e-12)
    # The active coefficient must still affect the likelihood.
    changed["beta_interval__"][0] += 1
    assert likelihood(point) != pytest.approx(likelihood(changed))


def test_capacity_and_configuration_checks():
    template = NativeFamilyTemplate.build(
        config(), saturation="linear", carryover="none", max_channels=2, max_controls=1
    )
    with pytest.raises(ValueError, match="capacity"):
        template.payload(inputs(p=3))
    with pytest.raises(ValueError, match="incompatible"):
        cfg = config()
        cfg.l_max = 3
        template.payload(inputs(), cfg)
    first, identity = template.payload(inputs())
    changed = inputs()
    changed.channel_gates[0] = 0
    second, new_identity = template.payload(changed)
    assert identity != new_identity
    assert first["channel_gates"][0] == 1
    assert second["channel_gates"][0] == 0


@pytest.mark.slow
@pytest.mark.parametrize("n_latent,max_controls", [(0, 1), (1, 0)])
def test_native_sampling_reuses_compilation_and_preserves_prediction(
    monkeypatch, n_latent, max_controls
):
    import nutpie
    import nutpie.compiled_pyfunc

    def no_python_callbacks(*args, **kwargs):
        raise AssertionError("native sampling must not use from_pyfunc")

    monkeypatch.setattr(nutpie.compiled_pyfunc, "from_pyfunc", no_python_callbacks)
    original = nutpie.compile_pymc_model
    calls = []

    def compile_once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(nutpie, "compile_pymc_model", compile_once)
    cfg = config()
    cfg.n_latent = n_latent
    compiled = CompiledOracle.for_family(
        cfg, saturation="linear", carryover="none", max_channels=2, max_controls=max_controls
    )

    def observed(p=1, context=4):
        return replace(
            inputs(p=p, c=max_controls if p == 2 else 0, context=context),
            latent_gates=np.ones(n_latent),
        )

    settings = OracleSamplingConfig(
        latent="sampled",
        walk_smoothness="inferred",
        chains=2,
        cores=2,
        tune=12,
        draws=8,
        adaptation="draw_diag",
        random_seed=42,
    )
    traces = []
    for data in (observed(), observed(p=2, context=7)):
        result = compiled.fit(data, settings)
        traces.append(result.idata)
        assert result.idata["posterior"].sizes["channel"] == len(data.channel_ids)
        assert result.idata["posterior"].sizes["control"] == len(data.control_ids)
        assert "_state" not in result.idata["posterior"]
        assert result.idata["oracle_state"].sizes["draw"] == 8
        assert result.receipt.to_dict()["oracle"]["walk_smoothness"] == "inferred"
        predicted = compiled.predict_posterior(data, result.idata, random_seed=17)
        assert predicted.outcome.shape == (2, 8, 8)
        np.testing.assert_allclose(
            predicted.outcome_mu, predicted.baseline + predicted.contributions.sum("channel")
        )
        assert calls == [1]
    # A later fit cannot change an earlier prediction binding.
    again = compiled.predict_posterior(observed(), traces[0], random_seed=17)
    assert again.outcome.shape == (2, 8, 8)
    with pytest.raises(ValueError, match="immutable binding"):
        compiled.predict_posterior(observed(), traces[1], random_seed=17)
    with pytest.raises(ValueError, match="inferred"):
        compiled.fit(observed(), replace(settings, walk_smoothness="known"))
    print("smoke divergences:", [int(t["sample_stats"].diverging.sum()) for t in traces])


@pytest.mark.parametrize("saturation", ["linear", "hill", "michaelis_menten"])
def test_masked_joint_likelihood_matches_active_only_family(saturation):
    from pymc_generator.oracle_family import FamilyOracle

    cfg = config()
    cfg.n_latent = 1
    cfg.outcome_std_mode = "relative"
    cfg.baseline_floor = 0.0
    cfg.baseline_floor_scope = "non_treatment"
    data = replace(inputs(p=1, c=1), latent_gates=np.ones(1))
    native = NativeFamilyTemplate.build(
        cfg, saturation=saturation, carryover="geometric", max_channels=2, max_controls=2
    )
    payload, _ = native.payload(data)
    active = FamilyOracle(cfg, saturation=saturation, carryover="geometric")
    active_payload = {k: v.copy() for k, v in payload.items() if k != "context_mask"}
    active_payload.update(
        channels=data.channels,
        controls=data.controls,
        anchor=data.saturation_scale,
        channel_gates=data.channel_gates,
        control_gates=data.control_gates,
        context_indices=data.context_indices,
        context_outcome=data.context_outcome,
    )
    with active.model:
        pm.set_data(active_payload)
    with native.model:
        pm.set_data(payload)
    a_point, n_point = active.model.initial_point(), native.model.initial_point()
    # Use nonzero innovations so smoothness changes the likelihood.
    for name in a_point:
        a_point[name] = np.full_like(a_point[name], 0.17)
        n_point[name] = np.full_like(n_point[name], 0.17)
    a_logp = active.model.compile_logp(vars=active.model.potentials)
    n_logp = native.model.compile_logp(vars=native.model.observed_RVs)
    np.testing.assert_allclose(a_logp(a_point), n_logp(n_point), rtol=1e-11)
    # The public smoothness prior is inferred, not replaced with generating truth.
    changed_cfg = config()
    changed_cfg.n_latent = 1
    changed_cfg.outcome_std_mode = "relative"
    changed_cfg.baseline_floor = 0.0
    changed_cfg.baseline_floor_scope = "non_treatment"
    changed_cfg.rw_smoothness_alpha, changed_cfg.rw_smoothness_beta = 10.0, 1.0
    other, _ = native.payload(data, changed_cfg)
    assert not np.array_equal(other["width_probs"], payload["width_probs"])


def test_gradient_matches_finite_differences():
    native = NativeFamilyTemplate.build(
        config(), saturation="linear", carryover="none", max_channels=2, max_controls=1
    )
    payload, _ = native.payload(inputs())
    with native.model:
        pm.set_data(payload)
    point = native.model.initial_point()
    logp, gradient = native.model.compile_logp(), native.model.compile_dlogp()
    analytical = gradient(point)
    numerical = []
    for name, values in point.items():
        for i in range(values.size):
            lo = {k: np.array(v, copy=True) for k, v in point.items()}
            hi = {k: np.array(v, copy=True) for k, v in point.items()}
            lo[name].flat[i] -= 1e-5
            hi[name].flat[i] += 1e-5
            numerical.append((logp(hi) - logp(lo)) / 2e-5)
    np.testing.assert_allclose(analytical, numerical, rtol=3e-5, atol=1e-5)
