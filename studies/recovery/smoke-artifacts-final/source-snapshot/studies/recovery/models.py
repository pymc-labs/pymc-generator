"""Generator-support oracle likelihoods extracted read-only from ols_reporting.py."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.rewriting.utils import rewrite_graph
from scipy.special import betainc

from pymc_generator import mechanisms
from pymc_generator.random_walk import _walk_basis

from .config import ContractError


@lru_cache(maxsize=8)
def baseline_geometry(
    horizon: int, burn: int, start: int, max_width: int, a: float = 2, b: float = 2
):
    """Exactly integrate the generator's centered/smoothed 112-week walk and width."""
    full = horizon + burn
    widths = np.arange(1, min(full, max_width) + 1)
    edges = np.r_[0, (widths[:-1] + 0.5) / max_width, 1]
    prior = np.diff(betainc(a, b, edges))
    if np.any(prior <= 0) or not np.isclose(prior.sum(), 1):
        raise ContractError("invalid baseline width probabilities")
    bases = np.stack([_walk_basis(full, int(w)) for w in widths])
    fitted = bases[:, burn + start :, :]
    cov = fitted @ fitted.transpose(0, 2, 1)
    eigen, vectors = np.linalg.eigh(cov)
    if eigen.min() < -1e-10:
        raise ContractError("baseline covariance is not positive semidefinite")
    eigen = np.maximum(eigen, 0)
    return widths, prior, eigen, vectors, vectors.transpose(0, 2, 1).reshape(-1, horizon - start)


def build_model(data: dict, cfg, *, start: int, varying: bool):
    """No realized true coefficient, baseline, shape, or noise parameter enters initialization."""
    n, c, horizon = cfg.n_treatments, cfg.n_covariates, cfg.n_time_steps
    if cfg.mm_scale_prior != "log_uniform" or cfg.treatment_reference_multiplier != 1.0:
        raise ContractError("unsupported prior support")
    coords = {
        "week": np.arange(horizon),
        "fit_week": np.arange(start, horizon),
        "media": np.arange(n),
        "control": np.arange(c),
    }
    geometry = None
    if varying:
        geometry = baseline_geometry(
            horizon,
            cfg.carryover_burn_in,
            start,
            cfg.rw_smoothness_max_weeks,
            cfg.rw_smoothness_alpha,
            cfg.rw_smoothness_beta,
        )
        coords["baseline_width_option"] = geometry[0]
    with pm.Model(coords=coords) as model:
        x = pm.Data("media_data", data["media_data"], dims=("week", "media"))
        z = pm.Data("control_data", data["control_data"], dims=("week", "control"))
        media_scale = pm.Data("media_scale", data["media_scale"], dims="media")
        control_scale = pm.Data("control_scale", data["control_scale"], dims="control")
        media_mask = pm.Data("media_mask", data["media_mask"], dims="media")
        control_mask = pm.Data("control_mask", data["control_mask"], dims="control")
        media_effect_mask = pm.Data("media_effect_mask", data["media_effect_mask"], dims="media")
        control_effect_mask = pm.Data(
            "control_effect_mask", data["control_effect_mask"], dims="control"
        )
        anchor = pm.Data("saturation_anchor", data["saturation_anchor"], dims="media")
        y_scale = pm.Data("outcome_scale", data["outcome_scale"])
        y = pm.Data("outcome_data", data["outcome_data"], dims="fit_week")
        intercept = pm.Uniform("intercept", *cfg.rw_baseline_mean_range)
        alpha = pm.Uniform("alpha", *cfg.carryover_alpha_range, dims="media")
        k_lo, k_hi = cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]
        log_kappa = pm.Uniform("log_kappa", np.log(k_lo), np.log(k_hi), dims="media")
        kappa = pm.Deterministic(
            "mm_kappa_mult", pt.clip(pt.exp(log_kappa), k_lo, k_hi), dims="media"
        )
        half = pm.Deterministic("half_saturation", kappa * anchor, dims="media")
        half_ratio = pm.Deterministic("half_ratio", half / media_scale, dims="media")
        amplitude = pm.Uniform(
            "amplitude", *cfg.treatment_reference_contribution_range, dims="media"
        )
        beta = pm.Deterministic("beta", amplitude * (1 + kappa) * media_effect_mask, dims="media")
        control_reference = pm.Uniform(
            "control_reference_contribution",
            *cfg.covariate_reference_contribution_range,
            dims="control",
        )
        rho = pm.Deterministic(
            "rho",
            control_reference / cfg.covariate_reference_scale * control_effect_mask,
            dims="control",
        )
        control_amplitude = pm.Deterministic(
            "control_amplitude", rho * control_scale, dims="control"
        )
        responses = pt.stack(
            [
                mechanisms.stable_michaelis_menten_kappa_relative(
                    mechanisms.apply_geometric_carryover(x[:, i : i + 1], alpha[i], cfg.l_max)[
                        :, 0
                    ],
                    1.0,
                    kappa_mult=half_ratio[i],
                )
                for i in range(n)
            ],
            axis=1,
        )[start:]
        if varying:
            responses = rewrite_graph(responses, include=("lower_xtensor",), clone=False)
        media_effect = responses * beta * media_mask
        control_effect = z[start:] * control_amplitude * control_mask
        pm.Deterministic("media_path", media_effect, dims=("fit_week", "media"))
        pm.Deterministic("control_path", control_effect, dims=("fit_week", "control"))
        pm.Deterministic("mean_media_contribution", media_effect.mean(axis=0), dims="media")
        pm.Deterministic("mean_control_contribution", control_effect.mean(axis=0), dims="control")
        mu = pm.Deterministic(
            "mu", intercept + media_effect.sum(axis=1) + control_effect.sum(axis=1), dims="fit_week"
        )
        sigma_relative = pm.Uniform("sigma_relative", *cfg.rw_outcome_std_range)
        sigma = pm.Deterministic("sigma", sigma_relative * pt.sqrt(pt.sum(beta**2)))
        if varying:
            widths, prior, eigen, _, projection = geometry
            baseline_relative = pm.Uniform("baseline_sd_relative", *cfg.rw_baseline_std_range)
            tau = pm.Deterministic("baseline_sd", baseline_relative * pt.sqrt(pt.sum(beta**2)))
            residual = y * y_scale - mu
            projected = pt.dot(pt.as_tensor_variable(projection), residual).reshape(
                (len(widths), horizon - start)
            )
            variance = tau**2 * pt.as_tensor_variable(eigen) + sigma**2
            width_logp = -0.5 * pt.sum(
                projected**2 / variance + pt.log(variance) + np.log(2 * np.pi), axis=1
            )
            logits = pt.log(pt.as_tensor_variable(prior)) + width_logp
            marginal_logp = pt.logsumexp(logits)
            pm.Deterministic(
                "baseline_width_probabilities",
                pt.exp(logits - marginal_logp),
                dims="baseline_width_option",
            )
            pm.Potential("baseline_marginal_likelihood", marginal_logp)
        else:
            pm.Normal(
                "outcome", mu=mu / y_scale, sigma=sigma / y_scale, observed=y, dims="fit_week"
            )
    return model, geometry


def reconstruct(idata, data: dict, geometry, seed: int):
    """Conditional Gaussian baseline paths, NOT independent prior-walk overlays."""
    widths, _, eigen, vectors, _ = geometry
    post = idata["posterior"]
    shape = post["intercept"].shape
    sigma = post["sigma"].values.reshape(-1)
    tau = post["baseline_sd"].values.reshape(-1)
    probs = (
        post["baseline_width_probabilities"]
        .transpose("chain", "draw", "baseline_width_option")
        .values.reshape(-1, len(widths))
    )
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-10)
    rng = np.random.default_rng(seed)
    indices = np.minimum(
        (rng.random(len(sigma))[:, None] > probs.cumsum(axis=1)).sum(axis=1), len(widths) - 1
    )
    static_mu = post["mu"].values.reshape(-1, len(data["outcome_data"]))
    residual = data["outcome_data"] * data["outcome_scale"] - static_mu
    deviation = np.empty_like(residual)
    for width in np.unique(indices):
        ix = np.flatnonzero(indices == width)
        u, spectrum = vectors[width], eigen[width]
        latent_variance = tau[ix, None] ** 2 * spectrum
        variance = latent_variance + sigma[ix, None] ** 2
        gain = latent_variance / variance
        conditional_mean = (residual[ix] @ u) * gain
        conditional_sd = np.sqrt(latent_variance * sigma[ix, None] ** 2 / variance)
        deviation[ix] = (
            conditional_mean + conditional_sd * rng.normal(size=conditional_mean.shape)
        ) @ u.T
    import xarray as xr

    baseline = post["intercept"].values.reshape(-1, 1) + deviation
    post["baseline_path"] = xr.DataArray(
        baseline.reshape(*shape, -1), dims=("chain", "draw", "fit_week")
    )
    post["baseline_expected_width"] = xr.DataArray(
        (probs @ widths).reshape(shape), dims=("chain", "draw")
    )
    np.testing.assert_allclose(
        baseline
        + post["media_path"].values.reshape(*baseline.shape, -1).sum(axis=-1)
        + post["control_path"].values.reshape(*baseline.shape, -1).sum(axis=-1),
        static_mu + deviation,
        rtol=1e-10,
        atol=1e-10,
    )
    return indices.reshape(shape)
