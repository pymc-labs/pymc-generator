"""Strict configuration contract for the supported direct-effect geometric/MM study."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import pymc_generator as pg


class ContractError(ValueError):
    """Unsupported or internally inconsistent study request."""


def _table(value, allowed: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(f"{label}: unknown fields {sorted(unknown)}")
    return value


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label: str, minimum: float = 0, maximum: float | None = None) -> float:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ContractError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _range(value, label: str, minimum: int = 0) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{label} must be a two-element inclusive range")
    low, high = (_integer(x, label, minimum) for x in value)
    if low > high:
        raise ContractError(f"{label} has reversed bounds")
    return low, high


@dataclass(frozen=True)
class StudyConfig:
    """Complete resolved prior and methods; no external config read after loading."""

    resolved: dict

    @classmethod
    def load(cls, path: Path) -> StudyConfig:
        path = path.resolve()
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        generator = _table(raw.get("generator", {}), {"prior_file"}, "generator")
        name = generator.get("prior_file")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ContractError("generator.prior_file must be a nonempty config-relative path")
        source = (path.parent / name).resolve()
        try:
            prior = json.loads(source.read_text())
        except (OSError, ValueError) as exc:
            raise ContractError(f"invalid full generator prior: {source}") from exc
        return cls.parse({**raw, "generator": {"prior": prior}})

    @classmethod
    def parse(cls, raw: dict) -> StudyConfig:
        _table(raw, {"study", "generator", "methods", "score", "diagnostics"}, "root")
        generator = _table(raw.get("generator", {}), {"prior"}, "generator")
        prior = generator.get("prior")
        required = {field.name for field in fields(pg.SCMPrior)}
        if not isinstance(prior, dict) or set(prior) != required:
            raise ContractError(
                f"generator.prior requires the full SCMPrior field inventory (missing {sorted(required - set(prior or {})) if isinstance(prior, dict) else 'all'})"
            )
        # Reject extra or incompatible parameters BEFORE creating stage directories. A full
        # explicit prior prevents unrecorded library-default changes or silent overrides.
        fixed = {
            "n_latent": 1,
            "n_latent_active_range": [1, 1],
            "n_cells": 2,
            "draws_per_cell": 1,
            "seed": 0,
            "confounding_strength_range": None,
            "baseline_floor": None,
            "baseline_floor_scope": "intercept",
            "n_treatment_shocks": 0,
            "prior_conditioning": False,
            "prior_cond_width_ranges": None,
            "edge_rate_overrides": None,
            "outcome_std_mode": "relative",
            "mm_scale_prior": "log_uniform",
            "treatment_reference_multiplier": 1.0,
            "carryover_family_probs": {"none": 0.0, "geometric": 1.0, "weibull": 0.0},
            "saturation_family_probs": {
                "linear": 0.0,
                "hill": 0.0,
                "logistic": 0.0,
                "michaelis_menten": 1.0,
                "tanh": 0.0,
                "root": 0.0,
            },
        }
        for key, value in fixed.items():
            if prior[key] != value:
                raise ContractError(
                    f"unsupported generator prior {key}: direct-effect geometric/MM adapter requires {value!r}"
                )
        horizon = _integer(prior["n_time_steps"], "n_time_steps", 16)
        lag = _integer(prior["l_max"], "l_max", 2)
        if lag != 8:
            raise ContractError("l_max must be 8 for this geometric adapter")
        media = _integer(prior["n_treatments"], "n_treatments", 1)
        controls = _integer(prior["n_covariates"], "n_covariates", 1)
        media_range = _range(prior["n_treatments_active_range"], "n_treatments_active_range", 1)
        control_range = _range(prior["n_covariates_active_range"], "n_covariates_active_range", 1)
        if media_range[1] > media or control_range[1] > controls:
            raise ContractError("active counts exceed padded treatment/covariate slots")
        edges = _table(
            prior["edge_budget"], {"cy", "zy", "dc", "dz", "dy", "zc", "cc", "zz"}, "edge_budget"
        )
        if set(edges) != {"cy", "zy", "dc", "dz", "dy", "zc", "cc", "zz"}:
            raise ContractError("all direct and absent edge budgets must be explicit")
        edge_ranges = {key: _range(value, f"edge_budget.{key}") for key, value in edges.items()}
        if (
            edge_ranges["cy"][0] < 1
            or edge_ranges["cy"][0] > media_range[0]
            or edge_ranges["cy"][1] > media
            or edge_ranges["zy"][0] > control_range[0]
            or edge_ranges["zy"][1] > controls
            or any(edge_ranges[k] != (0, 0) for k in edges if k not in ("cy", "zy"))
        ):
            raise ContractError("only feasible direct cy/zy outcome edges are supported")
        if (
            prior["treatment_reference_contribution_range"] is None
            or prior["covariate_reference_contribution_range"] is None
            or prior["covariate_reference_scale"] <= 0
            or set(prior["saturation_prior_ranges"]["michaelis_menten"]) != {"kappa_mult"}
        ):
            raise ContractError(
                "known parameter/reference anchors and MM kappa bounds are required"
            )
        study = _table(
            raw.get("study", {}),
            {"worlds", "seed", "baseline", "reported_weeks", "score_weeks", "burn_in"},
            "study",
        )
        worlds = _integer(study.get("worlds", 2), "worlds", 2)
        if worlds % 2:
            raise ContractError("two distinct cells per batch require an even world count")
        seed = _integer(study.get("seed", 20271), "study seed")
        baseline = study.get("baseline", "constant")
        if baseline not in ("constant", "varying"):
            raise ContractError("baseline must be constant or varying")
        baseline_range = prior["rw_baseline_std_range"]
        if (
            not isinstance(baseline_range, list)
            or len(baseline_range) != 2
            or (baseline == "constant" and baseline_range != [0, 0])
            or (
                baseline == "varying"
                and not (0 <= baseline_range[0] <= baseline_range[1] and baseline_range[1] > 0)
            )
        ):
            raise ContractError("baseline mode disagrees with explicit baseline SD support")
        reported = list(_range(study.get("reported_weeks", [0, horizon - 1]), "reported_weeks"))
        scored = list(_range(study.get("score_weeks", [lag - 1, horizon - 1]), "score_weeks"))
        burn_in = _integer(study.get("burn_in", 8), "burn_in", lag)
        if (
            reported != [0, horizon - 1]
            or scored != [lag - 1, horizon - 1]
            or prior["carryover_burn_in"] != burn_in
        ):
            raise ContractError(
                "report full horizon, score l_max-1 through end and match prior burn-in"
            )
        try:
            pg.SCMPrior(**prior).validate()
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid full generator prior: {exc}") from exc
        methods = _table(raw.get("methods", {}), {"enabled", "ols", "advi", "mcmc"}, "methods")
        enabled = methods.get(
            "enabled", ["ols", "advi", "mcmc"] if baseline == "constant" else ["advi", "mcmc"]
        )
        if (
            not isinstance(enabled, list)
            or not enabled
            or any(type(x) is not str for x in enabled)
            or len(set(enabled)) != len(enabled)
            or set(enabled) - {"ols", "advi", "mcmc"}
            or (baseline == "varying" and "ols" in enabled)
        ):
            raise ContractError("enabled methods must be unique ols/advi/mcmc (OLS constant only)")
        ols = _table(methods.get("ols", {}), {"starts", "max_nfev", "seed"}, "methods.ols")
        advi = _table(methods.get("advi", {}), {"steps", "draws", "seeds"}, "methods.advi")
        mcmc = _table(
            methods.get("mcmc", {}),
            {
                "chains",
                "warmup",
                "draws",
                "target_accept",
                "maxdepth",
                "seed",
                "sampler",
                "adaptation",
            },
            "methods.mcmc",
        )
        seeds = advi.get("seeds", [93001, 93002])
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ContractError("ADVI seeds must be distinct nonnegative integers")
        if (
            mcmc.get("sampler", "nutpie") != "nutpie"
            or mcmc.get("adaptation", "draw_diag") != "draw_diag"
        ):
            raise ContractError("only Nutpie draw_diag sampling is supported")
        budgets = {
            "enabled": enabled,
            "ols": {
                "starts": _integer(ols.get("starts", 2), "starts", 1),
                "max_nfev": _integer(ols.get("max_nfev", 120), "max_nfev", 1),
                "seed": _integer(ols.get("seed", 2026), "OLS seed"),
            },
            "advi": {
                "steps": _integer(advi.get("steps", 20000), "steps", 1),
                "draws": _integer(advi.get("draws", 4000), "draws", 1),
                "seeds": seeds,
            },
            "mcmc": {
                "chains": _integer(mcmc.get("chains", 4), "chains", 2),
                "warmup": _integer(mcmc.get("warmup", 1000), "warmup", 1),
                "draws": _integer(mcmc.get("draws", 1000), "draws", 1),
                "seed": _integer(mcmc.get("seed", 92001), "MCMC seed"),
                "maxdepth": _integer(mcmc.get("maxdepth", 10), "maxdepth", 1),
                "target_accept": _number(
                    mcmc.get("target_accept", 0.95), "target_accept", maximum=1
                ),
                "sampler": "nutpie",
                "adaptation": "draw_diag",
            },
        }
        if budgets["mcmc"]["target_accept"] >= 1:
            raise ContractError("MCMC target_accept must be <1")
        score = _table(raw.get("score", {}), {"relative", "absolute"}, "score")
        diagnostics = _table(
            raw.get("diagnostics", {}),
            {"rhat", "ess", "bfmi", "divergences", "maxdepth_hits"},
            "diagnostics",
        )
        resolved = {
            "schema": 2,
            "study": {
                "worlds": worlds,
                "seed": seed,
                "baseline": baseline,
                "reported_weeks": reported,
                "score_weeks": scored,
                "burn_in": burn_in,
            },
            "generator": prior,
            "methods": budgets,
            "score": {
                "relative": _number(score.get("relative", 0.1), "relative"),
                "absolute": _number(score.get("absolute", 0.01), "absolute"),
            },
            "diagnostics": {
                "rhat": _number(diagnostics.get("rhat", 1.01), "rhat", minimum=1),
                "ess": _integer(diagnostics.get("ess", 400), "ess", 1),
                "bfmi": _number(diagnostics.get("bfmi", 0.3), "bfmi"),
                "divergences": _integer(diagnostics.get("divergences", 0), "divergences"),
                "maxdepth_hits": _integer(diagnostics.get("maxdepth_hits", 0), "maxdepth_hits"),
            },
        }
        return cls(json.loads(json.dumps(resolved, sort_keys=True, allow_nan=False)))
