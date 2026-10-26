"""Outcome-side reference with unknown walk widths, not evaluation orchestration.

Widths are shared across all times of a world, not independent per observation.
The likelihood exactly sums their Beta-induced prior mass over continuous walk
innovations. It conditions on observed input paths, not their generative density.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from .oracle_smoothness import reported_walk_bases, walk_width_prior
from .sampler import SCMPrior


def _joint_width_logp(value, means, noise, log_prior, mask):
    pointwise = -0.5 * ((value - means) / noise) ** 2 - pt.log(noise) - 0.5 * np.log(2 * np.pi)
    return pt.logsumexp(log_prior + (pointwise * mask).sum(axis=-1))


def add_inferred_walk_likelihood(
    *,
    model: pm.Model,
    cfg: SCMPrior,
    rw: dict,
    n_time_steps: int,
    n_latent: int,
    n_covariates: int,
    contributions: Any,
    controls: Any,
    control_loadings: Any,
    latent_loadings: Any,
    outcome: Any,
    observed_indices: Any,
    width_probabilities: Any = None,
    observed_mask: Any = None,
) -> None:
    """Attach a world-level finite mixture, preserving generator floor order.

    Enumeration is bounded at 4096 width combinations to fail explicitly rather
    than accidentally build an exponential-size graph. The current one-latent,
    26-width population needs 676 components. No width is a sampled discrete RV.
    """
    prior = walk_width_prior(
        n_time_steps + cfg.carryover_burn_in,
        cfg.rw_smoothness_max_weeks,
        cfg.rw_smoothness_alpha,
        cfg.rw_smoothness_beta,
    )
    count = len(prior.widths) ** (n_latent + 1)
    if count > 4096:
        raise ValueError(f"inferred walk widths require {count} combinations; limit is 4096")
    combinations = np.array(list(product(range(len(prior.widths)), repeat=n_latent + 1)))
    bases = pt.as_tensor_variable(
        reported_walk_bases(
            n_time_steps,
            cfg.carryover_burn_in,
            cfg.rw_smoothness_max_weeks,
        )
    )
    full_time = n_time_steps + cfg.carryover_burn_in
    eps_b = pm.Normal("eps_b", shape=(full_time,))
    walk_b = rw["rw_b"]["mean"][0] + rw["rw_b"]["std"][0] * pt.tensordot(
        bases, eps_b, axes=[[2], [0]]
    )
    latent_paths = []
    if n_latent:
        eps_d = pm.Normal("eps_d", shape=(full_time, n_latent))
        for j in range(n_latent):
            latent_paths.append(pt.tensordot(bases, eps_d[:, j], axes=[[2], [0]]))

    def clip(value):
        return value if cfg.baseline_floor is None else pt.maximum(value, cfg.baseline_floor)

    baseline = clip(walk_b[combinations[:, 0]])
    selected_latents = []
    absorbing = cfg.baseline_floor is not None and cfg.baseline_floor_scope == "non_treatment"
    for j, path in enumerate(latent_paths):
        selected = path[combinations[:, j + 1]]
        selected_latents.append(selected)
        baseline = baseline + latent_loadings[j] * selected
        if absorbing:
            baseline = clip(baseline)
    if absorbing:
        # Sequential reflection at the floor, expressed through prefix minima.
        # Unlike an unrolled loop this handles a runtime active-control count,
        # including zero columns, without changing the graph.
        increments = controls * control_loadings
        prefixes = pt.concatenate(
            [pt.zeros((n_time_steps, 1)), pt.cumsum(increments, axis=1)], axis=1
        )
        total = prefixes[:, -1]
        baseline = pt.maximum(
            baseline + total, cfg.baseline_floor + total - pt.min(prefixes, axis=1)
        )
    else:
        baseline = baseline + pt.dot(controls, control_loadings)
    means = baseline + contributions.sum(axis=1)
    noise = rw["rw_y"]["std"][0]
    residual = (outcome[observed_indices] - means[:, observed_indices]) / noise
    pointwise = -0.5 * residual**2 - pt.log(noise) - 0.5 * np.log(2 * np.pi)
    if observed_mask is not None:
        pointwise = pointwise * observed_mask
    log_likelihood = pointwise.sum(axis=1)
    probabilities_data = prior.probabilities if width_probabilities is None else width_probabilities
    log_prior = pt.log(pt.as_tensor_variable(probabilities_data)[combinations]).sum(axis=1)
    log_weights = log_prior + log_likelihood
    if observed_mask is None:
        pm.Potential("outcome_width_likelihood", pt.logsumexp(log_weights))
    else:
        # One joint observation: the width is shared across the context.
        # This symbolic logp is compiled by nutpie, not called from Python while sampling.
        pm.CustomDist(
            "outcome",
            means[:, observed_indices],
            noise,
            log_prior,
            observed_mask,
            logp=_joint_width_logp,
            signature="(k,t),(),(k),(t)->(t)",
            observed=outcome[observed_indices],
        )
    model.add_coord(
        "walk_combination",
        tuple(",".join(str(int(width)) for width in prior.widths[row]) for row in combinations),
    )
    probabilities = pm.Deterministic(
        "walk_width_probabilities", pt.special.softmax(log_weights), dims="walk_combination"
    )
    # Avoid storing every width-specific path per draw. These means are clearly
    # distinguished from the jointly reconstructed trajectories below.
    pm.Deterministic("baseline", (probabilities[:, None] * baseline).sum(axis=0))
    pm.Deterministic("outcome_mu", (probabilities[:, None] * means).sum(axis=0))
    latent = pt.stack(selected_latents, axis=2) if n_latent else pt.zeros((count, n_time_steps, 0))
    pm.Deterministic("latent_unobserved", (probabilities[:, None, None] * latent).sum(axis=0))
    # Private expressions for bounded-memory prediction. They do not become
    # posterior deterministics of size combinations × horizon × draws.
    model._inferred_walk_components = {
        "outputs": [
            probabilities,
            baseline,
            means,
            latent,
            noise,
            contributions,
            controls * control_loadings,
        ],
        "combinations": combinations,
        "prior": prior,
    }


def sample_inferred_walk_predictions(
    model: pm.Model,
    posterior: xr.Dataset | xr.DataTree,
    *,
    random_seed: int,
) -> xr.Dataset:
    """Reconstruct paired full-window paths and noisy outcomes for every draw.

    Pass the posterior group of a fit to an inferred-width model. One joint
    width combination is drawn conditional on the whole context for each chain
    and draw, then reused for baseline, latent paths and the full outcome path.
    Smoothness is drawn from its Beta prior truncated to that selected bin.
    No outcomes beyond the model's context or scientific truth are consumed.
    This does not thin, score, qualify, select worlds or launch a fit.
    """
    components = getattr(model, "_inferred_walk_components", None)
    if components is None:
        raise ValueError("model must use walk_smoothness='inferred'")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()
    if not {"chain", "draw"}.issubset(posterior.dims):
        raise ValueError("posterior must have chain and draw dimensions")
    names = [rv.name for rv in model.free_RVs]
    missing = set(names) - set(posterior.data_vars)
    if missing:
        raise ValueError(f"posterior is missing model parameters: {sorted(missing)}")
    evaluate = pytensor.function(model.free_RVs, components["outputs"], on_unused_input="ignore")
    rng = np.random.default_rng(random_seed)
    combinations, prior = components["combinations"], components["prior"]
    rows: dict[str, list] = {
        name: []
        for name in (
            "baseline",
            "outcome_mu",
            "outcome",
            "latent_unobserved",
            "contributions",
            "control_contributions",
            "walk_width",
            "walk_smoothness",
        )
    }
    for chain in range(posterior.sizes["chain"]):
        for draw in range(posterior.sizes["draw"]):
            point = posterior.isel(chain=chain, draw=draw)
            probs, baseline, means, latent, noise, channels, controls = evaluate(
                *[point[name].values for name in names]
            )
            index = rng.choice(len(probs), p=probs / probs.sum())
            widths = combinations[index]
            values = {
                "baseline": baseline[index],
                "outcome_mu": means[index],
                "outcome": rng.normal(means[index], noise),
                "latent_unobserved": latent[index],
                "contributions": channels,
                "control_contributions": controls,
                "walk_width": prior.widths[widths],
                "walk_smoothness": [prior.sample_smoothness(int(i), rng) for i in widths],
            }
            for name, value in values.items():
                rows[name].append(value)
    dims = {
        "baseline": ("time",),
        "outcome_mu": ("time",),
        "outcome": ("time",),
        "latent_unobserved": ("time", "latent"),
        "contributions": ("time", "channel"),
        "control_contributions": ("time", "control"),
        "walk_width": ("walk",),
        "walk_smoothness": ("walk",),
    }
    sizes = (posterior.sizes["chain"], posterior.sizes["draw"])
    result = xr.Dataset(
        {
            name: (
                ("chain", "draw", *dims[name]),
                np.asarray(value).reshape((*sizes, *np.asarray(value).shape[1:])),
            )
            for name, value in rows.items()
        },
        coords={
            "chain": posterior.chain,
            "draw": posterior.draw,
            "walk": ["baseline", *[f"latent_{j}" for j in range(combinations.shape[1] - 1)]],
        },
    )
    result.attrs.update(
        walk_smoothness="inferred",
        random_seed=random_seed,
        boundary="zero-initialized adstock; all selected context rows retained",
        baseline="intrinsic baseline plus latent and control contributions; excludes noise",
    )
    return result
