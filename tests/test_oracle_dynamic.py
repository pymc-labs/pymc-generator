"""Active-only shape rebinding and unknown smoothness contracts."""

import numpy as np
import pymc as pm
import pytest
from scipy.special import logsumexp

from pymc_generator.oracle_dynamic import DynamicOracleKernel, ValueLayout
from pymc_generator.oracle_smoothness import reported_walk_bases, walk_width_prior
from pymc_generator.random_walk import _kernel_width, _walk_basis


def test_width_prior_uniform_and_clamped():
    np.testing.assert_allclose(
        walk_width_prior(104, 4, 1, 1).probabilities, [0.375, 0.25, 0.25, 0.125]
    )
    np.testing.assert_allclose(walk_width_prior(2, 4, 1, 1).probabilities, [0.375, 0.625])
    np.testing.assert_array_equal(walk_width_prior(104, 1, 0.6, 0.6).probabilities, [1])


def test_width_prior_matches_generator_pushforward():
    rng = np.random.default_rng(563)
    prior = walk_width_prior(8, 26, 0.6, 2.3)
    samples = rng.beta(0.6, 2.3, 200_000)
    observed = np.bincount([_kernel_width(s, 8, 26) - 1 for s in samples], minlength=8) / len(
        samples
    )
    np.testing.assert_allclose(prior.probabilities, observed, atol=0.003, rtol=0)
    for i in range(8):
        for _ in range(10):
            assert _kernel_width(prior.sample_smoothness(i, rng), 8, 26) == i + 1


@pytest.mark.parametrize("args", [(1, 26, 1, 1), (10, 0, 1, 1), (10, 3, 0, 1), (10, 3, 1, np.nan)])
def test_invalid_width_prior(args):
    with pytest.raises(ValueError):
        walk_width_prior(*args)


def test_walk_bases_keep_burn_in_calibration_without_dropping_reported_rows():
    bases = reported_walk_bases(10, 3, 4)
    assert bases.shape == (4, 10, 13)
    for i in range(4):
        np.testing.assert_array_equal(bases[i], _walk_basis(13, i + 1)[3:])


def test_layout_includes_scalars_and_no_padded_dimensions():
    layout = ValueLayout(("b", "noise"), ((3,), ()))
    assert layout.size == 4
    assert [v.shape for v in layout.unpack(np.arange(4.0))] == [(3,), ()]
    with pytest.raises(ValueError, match="matching"):
        layout.unpack(np.zeros(5))


@pytest.fixture(scope="module")
def dynamic_kernel():
    with pm.Model() as model:
        x = pm.Data("x", np.ones((6, 2)))
        y = pm.Data("y", np.zeros(6))
        beta = pm.Normal("beta", shape=x.shape[1])
        pm.Normal("observed", x @ beta, 1, observed=y)
    return DynamicOracleKernel(model, output_names=["beta"])


def bind(kernel, p, t, *, offset=0.0):
    x = np.arange(t * p, dtype="float64").reshape(t, p) / (t * p) + offset
    y = np.linspace(-0.1, 0.2, t)
    result = kernel.bind(
        value_shapes={"beta": (p,)},
        data={"x": x, "y": y},
        output_shapes={"beta": (p,)},
        coords={"channel": list(range(10, 10 + p))},
        dims={"beta": ("channel",)},
        initial_point=np.linspace(0.01, 0.02, p),
    )
    return result, x, y


def test_dynamic_active_dimensions_density_gradient_and_stale_binding(dynamic_kernel):
    a, x, y = bind(dynamic_kernel, 3, 12)
    q = np.linspace(-0.2, 0.2, 3)
    lp, grad, outputs = a.evaluate(q)
    residual = y - x @ q
    expected = -0.5 * (q @ q + residual @ residual + (len(y) + len(q)) * np.log(2 * np.pi))
    np.testing.assert_allclose(lp, expected, rtol=1e-12)
    np.testing.assert_allclose(grad, x.T @ residual - q, rtol=1e-12)
    np.testing.assert_array_equal(outputs["beta"], q)
    signatures = dynamic_kernel.signatures
    b, _, _ = bind(dynamic_kernel, 7, 8, offset=2)
    assert b.layout.size == 7
    b.evaluate(np.ones(7))
    assert dynamic_kernel.signatures == signatures
    np.testing.assert_array_equal(a.evaluate(q)[1], grad)
    assert a.evaluate(q)[0] == lp
    # Source arrays are not aliases of bound data.
    x[:] = 100
    assert a.evaluate(q)[0] == lp


def test_rebinding_rejects_rank_and_coordinate_mismatch(dynamic_kernel):
    with pytest.raises(ValueError, match="rank"):
        dynamic_kernel.bind(
            value_shapes={"beta": (2,)},
            data={"x": np.ones(6), "y": np.zeros(6)},
            output_shapes={"beta": (2,)},
            coords={"channel": [0, 1]},
            dims={"beta": ("channel",)},
            initial_point=np.ones(2),
        )
    with pytest.raises(ValueError, match="coordinate"):
        dynamic_kernel.bind(
            value_shapes={"beta": (2,)},
            data={"x": np.ones((6, 2)), "y": np.zeros(6)},
            output_shapes={"beta": (2,)},
            coords={"channel": [0]},
            dims={"beta": ("channel",)},
            initial_point=np.ones(2),
        )


def test_nutpie_rebinds_without_compilation(dynamic_kernel):
    import nutpie

    a, _, _ = bind(dynamic_kernel, 2, 10)
    signatures = dynamic_kernel.signatures
    b, _, _ = bind(dynamic_kernel, 4, 7)
    for world, size in ((a, 2), (b, 4), (a, 2)):
        trace = nutpie.sample(
            world.sampler_model(), chains=1, cores=1, tune=10, draws=5, seed=82, progress_bar=False
        )
        assert trace["posterior"]["beta"].shape == (1, 5, size)
        np.testing.assert_array_equal(trace["posterior"].coords["channel"], range(10, 10 + size))
        assert dynamic_kernel.signatures == signatures


def test_width_marginalization_is_a_global_choice_not_per_observation():
    # A baseline width is chosen once per world, not once per week.
    probabilities = walk_width_prior(5, 2, 1, 1).probabilities
    pointwise = np.array([[-1.0, -5.0], [-4.0, -2.0]])
    global_logp = logsumexp(np.log(probabilities) + pointwise.sum(axis=1))
    assert global_logp == pytest.approx(-6)
    assert global_logp != pytest.approx(
        logsumexp(np.log(probabilities)[:, None] + pointwise, axis=0).sum()
    )
