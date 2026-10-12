"""Strict, side-effect-free configuration preflight for the recovery study."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An unsupported, ambiguous, or incompatible study request."""


def _fields(value: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(f"{name}: unknown fields {sorted(unknown)}")
    return value


def _int(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _real(value: Any, name: str, *, minimum: float = 0, maximum: float | None = None) -> float:
    if type(value) not in (float, int):
        raise ContractError(f"{name} must be finite and >= {minimum}")
    from math import isfinite

    if not isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _range(value: Any, name: str, *, minimum: int = 0) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{name} must be a two-element inclusive range")
    lo, hi = (_int(v, name, minimum) for v in value)
    if lo > hi:
        raise ContractError(f"{name} has reversed endpoints")
    return lo, hi


@dataclass(frozen=True)
class StudyConfig:
    """Resolved config. The generator override is separate from the inference settings."""

    resolved: dict

    @classmethod
    def load(cls, path: Path) -> StudyConfig:
        with path.open("rb") as stream:
            return cls.parse(tomllib.load(stream))

    @classmethod
    def parse(cls, raw: dict) -> StudyConfig:
        _fields(raw, {"study", "generator", "methods", "score", "diagnostics", "pairing"}, "root")
        study = _fields(
            raw.get("study", {}),
            {"worlds", "seed", "baseline", "reported_weeks", "score_weeks", "burn_in"},
            "study",
        )
        worlds = _int(study.get("worlds", 50), "worlds")
        seed = _int(study.get("seed", 10029), "seed", 0)
        baseline = study.get("baseline", "constant")
        if baseline not in ("constant", "varying"):
            raise ContractError("baseline must be constant or varying")
        reported = _range(study.get("reported_weeks", [0, 103]), "reported_weeks")
        scored = _range(study.get("score_weeks", [7, 103]), "score_weeks")
        burn_in = _int(study.get("burn_in", 8), "burn_in", 0)
        if not (reported[0] <= scored[0] <= scored[1] <= reported[1]):
            raise ContractError("score_weeks must be inside reported_weeks")
        generator = _fields(
            raw.get("generator", {}),
            {
                "n_time_steps",
                "n_treatments",
                "n_covariates",
                "l_max",
                "n_treatments_active_range",
                "n_covariates_active_range",
                "carryover_family",
                "saturation_family",
                "edge_budget",
                "rw_baseline_std_range",
                "media_smoothness_beta",
                "rw_smoothness_beta",
            },
            "generator",
        )
        horizon = _int(generator.get("n_time_steps", 104), "n_time_steps")
        media_slots = _int(generator.get("n_treatments", 15), "n_treatments")
        control_slots = _int(generator.get("n_covariates", 10), "n_covariates")
        lag = _int(generator.get("l_max", 8), "l_max")
        if lag != 8:
            raise ContractError("l_max must be 8 for the currently supported geometric adapter")
        if reported != (0, horizon - 1):
            raise ContractError("reported_weeks must span the configured n_time_steps from week 0")
        if burn_in < lag:
            raise ContractError("burn_in must be >= l_max (geometric carryover lag)")
        media_range = _range(
            generator.get("n_treatments_active_range", [min(5, media_slots), media_slots]),
            "n_treatments_active_range",
        )
        controls_range = _range(
            generator.get("n_covariates_active_range", [min(3, control_slots), control_slots]),
            "n_covariates_active_range",
            minimum=1,
        )
        if media_range[0] < 1 or media_range[1] > media_slots:
            raise ContractError(
                "media active range requires at least one media channel and cannot exceed n_treatments"
            )
        if controls_range[1] > control_slots:
            raise ContractError("control active range cannot exceed n_covariates")
        if (
            generator.get("carryover_family", "geometric") != "geometric"
            or generator.get("saturation_family", "michaelis_menten") != "michaelis_menten"
        ):
            raise ContractError(
                "only geometric carryover and Michaelis–Menten saturation are supported"
            )
        edge_keys = {"cy", "zy", "dc", "dz", "dy", "zc", "cc", "zz"}
        supplied = _fields(generator.get("edge_budget", {}), edge_keys, "edge_budget")
        edges = {
            key: list(_range(supplied.get(key, default), f"edge_budget.{key}"))
            for key, default in {
                "cy": [1, media_range[1]],
                "zy": [0, controls_range[1]],
                **{k: [0, 0] for k in edge_keys - {"cy", "zy"}},
            }.items()
        }
        if edges["cy"][0] < 1 or edges["cy"][0] > media_range[0] or edges["cy"][1] > media_slots:
            raise ContractError("media effect budget must be feasible for every media active count")
        if edges["zy"][0] > controls_range[0] or edges["zy"][1] > control_slots:
            raise ContractError(
                "control effect budget must be feasible for every control active count"
            )
        if any(edges[key] != [0, 0] for key in edge_keys - {"cy", "zy"}):
            raise ContractError("only direct media/control outcome effects are supported")
        pair = generator.get(
            "rw_baseline_std_range", [0, 0] if baseline == "constant" else [0, 0.093]
        )
        if not isinstance(pair, list) or len(pair) != 2:
            raise ContractError("rw_baseline_std_range must have two endpoints")
        lo, hi = (_real(v, "baseline standard deviation") for v in pair)
        if lo > hi or (baseline == "constant" and hi != 0) or (baseline == "varying" and hi == 0):
            raise ContractError("baseline standard deviation range conflicts with baseline mode")
        generator = {
            **generator,
            "n_time_steps": horizon,
            "n_treatments": media_slots,
            "n_covariates": control_slots,
            "l_max": lag,
            "n_treatments_active_range": list(media_range),
            "n_covariates_active_range": list(controls_range),
            "carryover_family": "geometric",
            "saturation_family": "michaelis_menten",
            "edge_budget": edges,
            "rw_baseline_std_range": [lo, hi],
        }
        if "rw_smoothness_beta" in generator:
            generator["rw_smoothness_beta"] = _real(
                generator["rw_smoothness_beta"], "rw_smoothness_beta", minimum=1e-12
            )
        if "media_smoothness_beta" in generator:
            generator["media_smoothness_beta"] = _real(
                generator["media_smoothness_beta"], "media_smoothness_beta", minimum=1e-12
            )
        if "rw_smoothness_beta" in generator and "media_smoothness_beta" in generator:
            raise ContractError("shared rw_smoothness_beta cannot represent a media-only override")
        methods = _fields(raw.get("methods", {}), {"enabled", "ols", "advi", "mcmc"}, "methods")
        enabled = methods.get(
            "enabled", ["ols", "advi", "mcmc"] if baseline == "constant" else ["advi", "mcmc"]
        )
        if (
            not isinstance(enabled, list)
            or not enabled
            or any(type(item) is not str for item in enabled)
            or len(set(enabled)) != len(enabled)
            or set(enabled) - {"ols", "advi", "mcmc"}
        ):
            raise ContractError("enabled methods must be a nonempty unique list of ols/advi/mcmc")
        if baseline == "varying" and "ols" in enabled:
            raise ContractError("OLS is not supported for a varying baseline")
        ols = _fields(methods.get("ols", {}), {"starts", "max_nfev", "seed"}, "methods.ols")
        advi = _fields(methods.get("advi", {}), {"steps", "draws", "seeds"}, "methods.advi")
        mcmc = _fields(
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
        budgets = {
            "ols": {
                "starts": _int(ols.get("starts", 2), "starts"),
                "max_nfev": _int(ols.get("max_nfev", 120), "max_nfev"),
                "seed": _int(ols.get("seed", 2026), "OLS seed", 0),
            },
            "advi": {
                "steps": _int(advi.get("steps", 20000), "steps"),
                "draws": _int(advi.get("draws", 4000), "draws"),
            },
            "mcmc": {
                "chains": _int(mcmc.get("chains", 4), "chains"),
                "warmup": _int(mcmc.get("warmup", 1000), "warmup"),
                "draws": _int(mcmc.get("draws", 1000), "draws"),
                "seed": _int(mcmc.get("seed", 92001), "mcmc seed", 0),
                "maxdepth": _int(mcmc.get("maxdepth", 10), "maxdepth"),
                "target_accept": _real(mcmc.get("target_accept", 0.95), "target_accept", maximum=1),
                "sampler": "nutpie",
                "adaptation": "draw_diag",
            },
        }
        if (
            budgets["mcmc"]["target_accept"] >= 1
            or mcmc.get("sampler", "nutpie") != "nutpie"
            or mcmc.get("adaptation", "draw_diag") != "draw_diag"
        ):
            raise ContractError("MCMC requires Nutpie draw_diag with target_accept < 1")
        seeds = advi.get("seeds", [93001, 93002])
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ContractError("ADVI requires distinct explicit seeds")
        budgets["advi"]["seeds"] = [_int(s, "ADVI seed", 0) for s in seeds]
        score = _fields(raw.get("score", {}), {"relative", "absolute"}, "score")
        diagnostics = _fields(
            raw.get("diagnostics", {}),
            {"rhat", "ess", "bfmi", "divergences", "maxdepth_hits"},
            "diagnostics",
        )
        gate = {
            "relative": _real(score.get("relative", 0.1), "relative"),
            "absolute": _real(score.get("absolute", 0.01), "absolute"),
        }
        diagnostic_gate = {
            "rhat": _real(diagnostics.get("rhat", 1.01), "rhat", minimum=1),
            "ess": _int(diagnostics.get("ess", 400), "ess"),
            "bfmi": _real(diagnostics.get("bfmi", 0.3), "bfmi"),
            "divergences": _int(diagnostics.get("divergences", 0), "divergences", 0),
            "maxdepth_hits": _int(diagnostics.get("maxdepth_hits", 0), "maxdepth_hits", 0),
        }
        pairing = _fields(
            raw.get("pairing", {}),
            {"mode", "media_smoothness_beta", "coupling_manifest"},
            "pairing",
        )
        mode = pairing.get("mode", "independent")
        if type(mode) is not str or mode not in ("independent", "matched"):
            raise ContractError("pairing mode must be independent or matched")
        if mode == "matched":
            if (
                type(pairing.get("coupling_manifest")) is not str
                or not pairing["coupling_manifest"]
                or "media_smoothness_beta" not in pairing
            ):
                raise ContractError(
                    "matched media override requires an explicit coupling manifest and media beta"
                )
            if "rw_smoothness_beta" in generator:
                raise ContractError(
                    "shared smoothness changes controls/baseline; matched media override requires isolated draws"
                )
        for value in (generator.get("media_smoothness_beta"), pairing.get("media_smoothness_beta")):
            if value is not None:
                _real(value, "media_smoothness_beta", minimum=1e-12)
        return cls(
            json.loads(
                json.dumps(
                    {
                        "schema": 1,
                        "study": {
                            "worlds": worlds,
                            "seed": seed,
                            "baseline": baseline,
                            "reported_weeks": reported,
                            "score_weeks": scored,
                            "burn_in": burn_in,
                        },
                        "generator": generator,
                        "methods": {"enabled": enabled, **budgets},
                        "score": gate,
                        "diagnostics": diagnostic_gate,
                        "pairing": {
                            "mode": mode,
                            **{k: v for k, v in pairing.items() if k != "mode"},
                        },
                    },
                    sort_keys=True,
                )
            )
        )
