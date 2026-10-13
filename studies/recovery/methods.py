"""Method-specific estimators; no inference at import time."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.optimize import least_squares, lsq_linear

from pymc_generator import mechanisms

from .config import ContractError
from .models import build_model, reconstruct


@lru_cache(maxsize=8)
def media_transform(lag: int):
    x, a, k = pt.dvector("spend"), pt.dscalar("alpha"), pt.dscalar("half")
    carry = mechanisms.apply_geometric_carryover(x[:, None], a, lag)[:, 0]
    value = mechanisms.stable_michaelis_menten_kappa_relative(carry, 1.0, kappa_mult=k)
    return pytensor.function([x, a, k], value, mode="FAST_RUN")


def bounded_ols(data, cfg, *, start: int, starts: int, max_nfev: int, seed: int):
    """Estimated-transform bounded nonlinear least squares, not unconstrained OLS."""
    if cfg.rw_baseline_std_range[1] != 0:
        raise ContractError("OLS requires a constant baseline")
    began = perf_counter()
    media_mask = (data["media_mask"] * data["media_effect_mask"]).astype(bool)
    control_mask = (data["control_mask"] * data["control_effect_mask"]).astype(bool)
    media = data["media_data"][:, media_mask] * data["media_scale"][media_mask]
    controls = data["control_data"][:, control_mask] * data["control_scale"][control_mask]
    anchors = data["saturation_anchor"][media_mask]
    target = data["outcome_data"] * data["outcome_scale"]
    m, c = media.shape[1], controls.shape[1]
    assert m >= 1
    alpha_range = cfg.carryover_alpha_range
    k_range = cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]
    amplitude_range = cfg.treatment_reference_contribution_range
    reference_range = cfg.covariate_reference_contribution_range
    lower = np.r_[
        cfg.rw_baseline_mean_range[0],
        np.full(m, amplitude_range[0]),
        np.full(c, reference_range[0]),
    ]
    upper = np.r_[
        cfg.rw_baseline_mean_range[1],
        np.full(m, amplitude_range[1]),
        np.full(c, reference_range[1]),
    ]
    lo = np.r_[lower, np.full(m, alpha_range[0]), np.full(m, np.log(k_range[0]))]
    hi = np.r_[upper, np.full(m, alpha_range[1]), np.full(m, np.log(k_range[1]))]

    def design_at(alpha, kappa):
        response = np.column_stack(
            [
                media_transform(cfg.l_max)(media[:, i], alpha[i], kappa[i] * anchors[i])
                for i in range(m)
            ]
        )[start:]
        return np.column_stack(
            [
                np.ones(len(target)),
                response * (1 + kappa),
                controls[start:] / cfg.covariate_reference_scale,
            ]
        )

    def solve_box(design):
        scale = np.sqrt(np.mean(design**2, axis=0))
        scale = np.where(scale > 0, scale, 1.0)
        result = lsq_linear(
            design / scale,
            target,
            bounds=(lower * scale, upper * scale),
            method="bvls",
            tol=1e-10,
            max_iter=1000,
        )
        return result.x / scale

    rng = np.random.default_rng(seed)
    target_scale = max(float(target.std()), 1e-8)
    attempts = []
    candidates = []
    calls = 0
    wcount = len(lower)

    def residual(theta):
        nonlocal calls
        calls += 1
        weight, alpha, kappa = (
            theta[:wcount],
            theta[wcount : wcount + m],
            np.exp(theta[wcount + m :]),
        )
        return (design_at(alpha, kappa) @ weight - target) / target_scale

    for index in range(starts):
        alpha = (
            np.full(m, np.mean(alpha_range)) if index == 0 else rng.uniform(*alpha_range, size=m)
        )
        lk = (
            np.full(m, np.mean(np.log(k_range)))
            if index == 0
            else rng.uniform(*np.log(k_range), size=m)
        )
        weight = solve_box(design_at(alpha, np.exp(lk)))
        initial = np.clip(np.r_[weight, alpha, lk], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        candidate = least_squares(
            residual,
            initial,
            bounds=(lo, hi),
            x_scale=hi - lo,
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        candidates.append(candidate)
        attempts.append(
            {
                "start": index + 1,
                "success": bool(candidate.success),
                "nfev": int(candidate.nfev),
                "cost": float(candidate.cost),
            }
        )
    best = min(candidates, key=lambda item: item.cost)
    weights, alpha, kappa = (
        best.x[:wcount],
        best.x[wcount : wcount + m],
        np.exp(best.x[wcount + m :]),
    )
    design = design_at(alpha, kappa)
    components = np.zeros((len(target), 1 + cfg.n_treatments + cfg.n_covariates))
    components[:, 0] = weights[0]
    # Preserve padded, oracle-known null columns as exact zeros.
    components[:, 1 + np.flatnonzero(media_mask)] = (design * weights)[:, 1 : 1 + m]
    components[:, 1 + cfg.n_treatments + np.flatnonzero(control_mask)] = (design * weights)[
        :, 1 + m :
    ]
    prediction = components.sum(axis=1)
    if not np.allclose(prediction, design @ weights, atol=1e-10):
        raise ContractError("bounded fit reconstruction differs from fitted design")
    return {
        "components": components,
        "prediction": prediction,
        "alpha_active": alpha,
        "kappa_active": kappa,
        "weights": weights,
    }, {
        "method": "generator-bounded estimated-transform nonlinear least squares",
        "success": bool(best.success),
        "optimizer_status": str(best.message),
        "attempts": attempts,
        "nfev": sum(item["nfev"] for item in attempts),
        "residual_calls": calls,
        "optimality": float(best.optimality),
        "boundary_count": int(np.count_nonzero(best.active_mask)),
        "observed_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "seconds": perf_counter() - began,
    }


def bayes_fit(data, cfg, *, start: int, varying: bool, method: str, budget: dict, seed: int):
    import pandas as pd
    import pymc as pm
    from threadpoolctl import threadpool_limits

    began = perf_counter()
    model, geometry = build_model(data, cfg, start=start, varying=varying)
    point = model.initial_point()
    logp = model.compile_logp()(point)
    grad = model.compile_dlogp()(point)
    if not np.isfinite(logp) or not np.isfinite(grad).all():
        raise ContractError("nonfinite initial log density/gradient")
    compile_seconds = perf_counter() - began
    if method == "advi":
        with model, threadpool_limits(limits=1):
            inference = pm.FullRankADVI(random_seed=seed)
            fit = inference.fit(n=budget["steps"], progressbar=False)
            idata = fit.sample(draws=budget["draws"], random_seed=seed)
        history = np.asarray(fit.hist, dtype=float)
        report = {
            "method": "fullrank_advi",
            "seed": seed,
            "steps": budget["steps"],
            "draws": budget["draws"],
            "loss_finite": bool(np.isfinite(history).all()),
            "loss_last": float(history[-1]),
            "loss_first": float(history[0]),
            "optimization_warning": "short budget; optimization not certified",
            "diagnostics": "Rhat/ESS do not apply to variational draws",
        }
    elif method == "mcmc":
        import nutpie

        with pd.option_context("future.infer_string", False), threadpool_limits(limits=1):
            compiled = nutpie.compile_pymc_model(
                model, backend="numba", gradient_backend="pytensor"
            )
            idata = nutpie.sample(
                compiled,
                chains=budget["chains"],
                cores=budget["chains"],
                tune=budget["warmup"],
                draws=budget["draws"],
                seed=seed,
                adaptation="draw_diag",
                target_accept=budget["target_accept"],
                maxdepth=budget["maxdepth"],
                progress_bar=False,
            )
        stats = idata["sample_stats"]
        report = {
            "method": "nutpie",
            "adaptation": "draw_diag",
            "seed": seed,
            "chains": budget["chains"],
            "warmup": budget["warmup"],
            "draws": budget["draws"],
            "target_accept": budget["target_accept"],
            "divergences": int(stats["diverging"].sum()) if "diverging" in stats else None,
            "maxdepth_hits": int(stats["maxdepth_reached"].sum())
            if "maxdepth_reached" in stats
            else None,
        }
    else:
        raise ContractError("unsupported method")
    if varying:
        widths = reconstruct(idata, data, geometry, seed=seed + 10000)
        report["reconstructed_width_count"] = int(len(widths.reshape(-1)))
    report.update(
        {
            "compile_seconds": compile_seconds,
            "seconds": perf_counter() - began,
            "posterior_finite": bool(
                all(np.isfinite(v.values).all() for v in idata["posterior"].data_vars.values())
            ),
        }
    )
    return idata, report
