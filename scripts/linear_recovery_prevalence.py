"""Configuration-driven C1 calibration for linear-recovery reports.

This is validation-only code.  It deliberately separates generator settings from
study settings, records the resolved configuration before sampling, and treats a
cell (not a sibling world) as the independent unit for binary summaries.

The runner is intentionally a calibration scaffold: it measures generation and
post-hoc diagnostics separately and does not write prevalence CSVs or claim a
pilot estimate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pymc_generator import __version__, data_diagnostics, make_scm_prior, sample_prior_predictive
from pymc_generator.active_counts import summarize_active_count_coverage
from pymc_generator.sampler import SCMPrior

SCHEMA_VERSION = "linear-recovery-c1/v1"
CI_METHOD = "wilson"
DEFAULT_GENERATOR: dict[str, Any] = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,
    "n_treatments_active_range": [1, 10],
    "n_covariates_active_range": [1, 10],
    "n_latent_active_range": [1, 1],
    "n_time_steps": 104,
    "trajectories": "composable",
    "nonlinearity": "diverse",
}
DEFAULT_STUDY: dict[str, Any] = {
    "seed": 20261005,
    "n_cells": 4,
    "draws_per_cell": 2,
    "rank_tolerance": 1e-10,
    "ci_method": CI_METHOD,
    "confidence_level": 0.95,
    "cell_binary_estimand": "first_world",
    "diagnostic_views": ["levels", "differences"],
    "thresholds": {
        "absolute_correlation": [0.90, 0.95],
        "vif": [5.0, 10.0],
        "condition": [30.0, 100.0],
    },
    "treatment_count_bands": [[1, 2], [3, 5], [6, 8], [9, 10]],
    "control_count_bands": [[1, 2], [3, 5], [6, 8], [9, 10]],
}
STUDY_KEYS = frozenset(DEFAULT_STUDY)
_FACTORY_KEYS = frozenset(("edge_budget", "nonlinearity", "trajectories"))
_STUDY_ONLY_GENERATOR_KEYS = frozenset(("seed", "n_cells", "draws_per_cell"))


def _plain(value: Any) -> Any:
    """Convert numpy/dataclass values into deterministic JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    """Return the canonical serialization used for configuration digests."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(value: Any) -> str:
    """Hash a resolved configuration, independent of dictionary insertion order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _as_range(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in value)
        or int(value[0]) > int(value[1])
    ):
        raise ValueError(f"{name} must be an ascending two-integer range")
    return [int(value[0]), int(value[1])]


def validate_generator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generator section and return a normalized copy.

    Unknown fields are rejected before calling the factory.  In particular,
    sampling seed and cell/world budget belong to the separate study section.
    """
    if not isinstance(config, Mapping):
        raise TypeError("generator configuration must be a mapping")
    fields = {field.name for field in dataclasses.fields(SCMPrior)}
    allowed = fields | _FACTORY_KEYS
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unsupported generator configuration keys: {unknown}")
    forbidden = sorted(set(config) & _STUDY_ONLY_GENERATOR_KEYS)
    if forbidden:
        raise ValueError(f"study-only keys must be supplied separately: {forbidden}")
    required = ("n_treatments", "n_covariates", "n_latent")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"generator configuration is missing required keys: {missing}")
    normalized = _plain(dict(config))
    for name in (
        "n_treatments_active_range",
        "n_covariates_active_range",
        "n_latent_active_range",
    ):
        if name in normalized:
            normalized[name] = _as_range(normalized[name], name)
    for name in required:
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Factory validation catches all cross-field and preset constraints.
    make_scm_prior(**normalized, n_cells=2, draws_per_cell=1, seed=0)
    return normalized


def _validate_bands(value: Any, name: str) -> list[list[int]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(band, list)
            or len(band) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in band)
            or band[0] > band[1]
            for band in value
        )
    ):
        raise ValueError(f"{name} must contain four ascending integer bands")
    return [[int(band[0]), int(band[1])] for band in value]


def validate_study_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize the independent study protocol settings."""
    supplied = {} if settings is None else dict(settings)
    unknown = sorted(set(supplied) - STUDY_KEYS)
    if unknown:
        raise ValueError(f"unsupported study settings keys: {unknown}")
    resolved = _plain({**DEFAULT_STUDY, **supplied})
    if isinstance(resolved["seed"], bool) or not isinstance(resolved["seed"], int):
        raise ValueError("study seed must be an integer")
    for name in ("n_cells", "draws_per_cell"):
        if (
            isinstance(resolved[name], bool)
            or not isinstance(resolved[name], int)
            or resolved[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if resolved["n_cells"] < 2:
        raise ValueError("n_cells must be at least 2 for cell-level accounting")
    if resolved["ci_method"] != CI_METHOD:
        raise ValueError(f"only {CI_METHOD!r} confidence intervals are supported in C1")
    confidence = float(resolved["confidence_level"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if resolved["cell_binary_estimand"] not in {"first_world", "any_sibling"}:
        raise ValueError("cell_binary_estimand must be 'first_world' or 'any_sibling'")
    views = resolved["diagnostic_views"]
    if (
        not isinstance(views, list)
        or not views
        or not all(v in {"levels", "differences"} for v in views)
    ):
        raise ValueError("diagnostic_views must contain levels and/or differences")
    thresholds = resolved["thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "absolute_correlation",
        "vif",
        "condition",
    }:
        raise ValueError("thresholds must declare absolute_correlation, vif, and condition")
    for name, values in thresholds.items():
        if (
            not isinstance(values, list)
            or len(values) != 2
            or not 0 < float(values[0]) <= float(values[1])
        ):
            raise ValueError(f"thresholds[{name!r}] must be two positive ascending values")
    resolved["treatment_count_bands"] = _validate_bands(
        resolved["treatment_count_bands"], "treatment_count_bands"
    )
    resolved["control_count_bands"] = _validate_bands(
        resolved["control_count_bands"], "control_count_bands"
    )
    return resolved


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def split_config(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept the documented envelope, while rejecting ambiguous top-level keys."""
    if "generator" in raw:
        allowed = {"schema_version", "generator", "study"}
        unknown = sorted(set(raw) - allowed)
        if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config schema {raw.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
            )
        if unknown:
            raise ValueError(f"unsupported config envelope keys: {unknown}")
        generator = raw["generator"]
        embedded_study = raw.get("study", {})
    else:
        generator = raw
        embedded_study = {}
    return validate_generator_config(generator), validate_study_settings(embedded_study)


def resolve_config(
    config: Mapping[str, Any], study_settings: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve generator + separate study settings into a hashable report config."""
    generator, embedded_study = split_config(config)
    if study_settings is not None and embedded_study != validate_study_settings(study_settings):
        # A supplied sidecar is authoritative; embedded defaults are allowed, but
        # conflicting values are an error rather than a silent protocol change.
        merged = {**embedded_study, **dict(study_settings)}
        study = validate_study_settings(merged)
    else:
        study = embedded_study
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "generator_factory": generator,
        "generator_resolved": None,
        "study": study,
        "provenance": {
            "package": "pymc-generator",
            "package_version": __version__,
            "git_revision": _git_revision(),
        },
    }
    prior = make_prior(generator, study)
    resolved["generator_resolved"] = _plain(dataclasses.asdict(prior))
    # Hash excludes the hash field itself and includes package/git provenance.
    resolved["config_hash"] = config_hash(resolved)
    return resolved


def make_prior(generator: Mapping[str, Any], study: Mapping[str, Any]) -> SCMPrior:
    """Build a validated prior with the budget supplied by study settings."""
    generator = validate_generator_config(generator)
    study = validate_study_settings(study)
    return make_scm_prior(
        **generator,
        n_cells=study["n_cells"],
        draws_per_cell=study["draws_per_cell"],
        seed=study["seed"],
    )


def wilson_interval(
    successes: int, trials: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Wilson interval for independent cell indicators (not sibling worlds)."""
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes must be in [0, trials] and trials must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(centre - half), float(centre + half)


def cell_binary_summary(
    cell_ids: Any,
    successes: Any,
    *,
    estimand: str = "first_world",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize one binary outcome per cell without treating siblings as iid."""
    cells = np.asarray(cell_ids)
    values = np.asarray(successes, dtype=bool)
    if cells.ndim != 1 or values.shape != cells.shape:
        raise ValueError("cell_ids and successes must be one-dimensional arrays of equal length")
    if estimand not in {"first_world", "any_sibling"}:
        raise ValueError("estimand must be 'first_world' or 'any_sibling'")
    unique = np.unique(cells)
    indicators = []
    for cell in unique:
        members = values[cells == cell]
        indicators.append(members[0] if estimand == "first_world" else bool(np.any(members)))
    count = int(np.count_nonzero(indicators))
    interval = wilson_interval(count, len(indicators), confidence_level)
    return {
        "estimand": estimand,
        "unit": "cell",
        "n_cells": len(indicators),
        "n_successes": count,
        "rate": count / len(indicators),
        "interval": {
            "method": CI_METHOD,
            "confidence": confidence_level,
            "lower": interval[0],
            "upper": interval[1],
        },
    }


def design_rank_condition(
    design: Any,
    *,
    active_mask: Any | None = None,
    rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Calculate rank while excluding only padded columns, never constant active inputs."""
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("design must be a two-dimensional matrix with columns")
    if active_mask is None:
        active = np.ones(matrix.shape[1], dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.ndim != 1 or active.size != matrix.shape[1]:
            raise ValueError("active_mask must have one entry per design column")
    selected = matrix[:, active]
    if selected.shape[1] == 0:
        raise ValueError("design has no active columns")
    constant = np.ptp(selected, axis=0) <= rank_tolerance
    rank = int(np.linalg.matrix_rank(selected, tol=rank_tolerance))
    condition = float(np.linalg.cond(selected))
    return classify_design(
        rank,
        selected.shape[1],
        condition,
        rank_tolerance=rank_tolerance,
        constant_active_columns=int(np.count_nonzero(constant)),
    )


def classify_graph_paths(graph: Mapping[str, Any]) -> dict[str, bool]:
    """Separate direct outcome reachability from a graph-confounding indicator."""

    def any_edge(name: str) -> bool:
        return bool(np.asarray(graph.get(name, []), dtype=bool).any())

    latent_to_treatment = any_edge("g_dc")
    latent_to_outcome = any_edge("g_dy")
    return {
        "direct_treatment_outcome_reachability": any_edge("g_cy"),
        "latent_to_treatment": latent_to_treatment,
        "latent_to_outcome": latent_to_outcome,
        "shared_latent_confounding": latent_to_treatment and latent_to_outcome,
        "mediated_or_observed_path": any_edge("g_zc") and any_edge("g_zy"),
    }


def classify_design(
    rank: int,
    n_columns: int,
    condition: float,
    *,
    rank_tolerance: float,
    constant_active_columns: int = 0,
    causal_path: bool = False,
) -> dict[str, Any]:
    """Classify rank/conditioning while keeping causal reachability separate."""
    if rank < 0 or n_columns < 1 or rank > n_columns:
        raise ValueError("rank must lie in [0, n_columns]")
    if rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be positive")
    exact = rank < n_columns
    return {
        "exact_rank": int(rank),
        "n_columns": int(n_columns),
        "rank_deficient": exact,
        "rank_tolerance": float(rank_tolerance),
        "practical_conditioning": (
            "not_applicable"
            if not np.isfinite(condition)
            else ("high" if condition >= 100 else "moderate" if condition >= 30 else "low")
        ),
        "condition": float(condition) if np.isfinite(condition) else None,
        "constant_active_columns": int(constant_active_columns),
        "constant_active_inputs_retained": True,
        "padded_columns_excluded_by_mask": True,
        "causal_path_present": bool(causal_path),
        "causal_path_interpretation": "graph reachability/confounding flag; not a multicollinearity result",
    }


def analysis_status(analysis: str, status: str, reason: str) -> dict[str, str]:
    """Represent available, unavailable, or ineligible analyses without substitution."""
    if (
        not analysis
        or status not in {"available", "unavailable", "ineligible", "failure"}
        or not reason
    ):
        raise ValueError("analysis, supported status, and reason are required")
    return {"analysis": analysis, "status": status, "reason": reason}


def unsupported_status(analysis: str, reason: str) -> dict[str, str]:
    """Return an explicit status for analyses unavailable from corpus truth."""
    return analysis_status(analysis, "unavailable", reason)


def _band_accounting(
    corpus: Mapping[str, Any], treatment_bands: list[list[int]], control_bands: list[list[int]]
) -> dict[str, Any]:
    treatment_counts = np.asarray(corpus["treatment_active_mask"], dtype=bool).sum(axis=1)
    control_counts = np.asarray(corpus["covariate_active_mask"], dtype=bool).sum(axis=1)
    cells = np.asarray(corpus["cell_id"])
    cell_rows = np.array([np.flatnonzero(cells == cell)[0] for cell in np.unique(cells)])
    cell_treatments = treatment_counts[cell_rows]
    cell_controls = control_counts[cell_rows]
    cell_matrix = np.zeros((len(treatment_bands), len(control_bands)), dtype=int)
    world_matrix = np.zeros_like(cell_matrix)
    for row, column in zip(treatment_counts, control_counts, strict=True):
        for i, (t_low, t_high) in enumerate(treatment_bands):
            for j, (c_low, c_high) in enumerate(control_bands):
                if t_low <= row <= t_high and c_low <= column <= c_high:
                    world_matrix[i, j] += 1
    for row, column in zip(cell_treatments, cell_controls, strict=True):
        for i, (t_low, t_high) in enumerate(treatment_bands):
            for j, (c_low, c_high) in enumerate(control_bands):
                if t_low <= row <= t_high and c_low <= column <= c_high:
                    cell_matrix[i, j] += 1
    return {
        "treatment_bands": treatment_bands,
        "control_bands": control_bands,
        "n_cells": cell_matrix.tolist(),
        "n_worlds": world_matrix.tolist(),
    }


def count_accounting(
    corpus: Mapping[str, Any], prior: SCMPrior, study: Mapping[str, Any]
) -> dict[str, Any]:
    """Return realized cell/world counts, including rejected/failure placeholders."""
    summary = summarize_active_count_coverage(
        corpus["treatment_active_mask"],
        corpus["covariate_active_mask"],
        corpus["cell_id"],
        list(range(prior.n_treatments_active_range[0], prior.n_treatments_active_range[1] + 1)),
        list(range(prior.n_covariates_active_range[0], prior.n_covariates_active_range[1] + 1)),
    )
    n_worlds = int(np.asarray(corpus["cell_id"]).size)
    n_cells = int(np.unique(corpus["cell_id"]).size)
    generation_diagnostics = corpus.get("diagnostics", {})
    evaluated = int(generation_diagnostics.get("n_draws_evaluated", n_worlds))
    failures = int(generation_diagnostics.get("n_draw_failures", 0))
    rejected = max(0, evaluated - n_worlds - failures)
    return {
        "requested_cells": int(prior.n_cells),
        "realized_cells": n_cells,
        "requested_worlds": int(prior.n_cells * prior.draws_per_cell),
        "realized_worlds": n_worlds,
        "accepted_worlds": n_worlds,
        "evaluated_candidates": evaluated,
        "rejected_candidates": rejected,
        "generation_failures": failures,
        "draws_per_cell": int(prior.draws_per_cell),
        "coverage": summary,
        "count_band_accounting": _band_accounting(
            corpus,
            study["treatment_count_bands"],
            study["control_count_bands"],
        ),
        "unit": "cell for binary inference; world for generation accounting",
    }


def _low_dimension_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the bounded low-dimensional calibration variant."""
    low = dict(generator)
    low["n_treatments"] = min(int(low["n_treatments"]), 2)
    low["n_covariates"] = min(int(low["n_covariates"]), 2)
    low["n_latent"] = min(int(low["n_latent"]), 1)
    low["n_treatments_active_range"] = [1, low["n_treatments"]]
    low["n_covariates_active_range"] = [1, low["n_covariates"]]
    low["n_latent_active_range"] = [1, low["n_latent"]]
    low["n_time_steps"] = min(int(low.get("n_time_steps", 104)), 32)
    low["trajectories"] = "texture"
    return low


def _measure_calibration(generator: Mapping[str, Any], study: Mapping[str, Any]) -> dict[str, Any]:
    prior = make_prior(generator, study)
    generation_start = time.perf_counter()
    corpus = sample_prior_predictive(prior)
    generation_seconds = time.perf_counter() - generation_start
    diagnostic_start = time.perf_counter()
    diagnostics = data_diagnostics(
        corpus,
        views=tuple(study["diagnostic_views"]),
        keep_series=False,
    )
    diagnostic_seconds = time.perf_counter() - diagnostic_start
    return {
        "timing": {
            "generation_seconds": generation_seconds,
            "diagnostics_seconds": diagnostic_seconds,
            "timing_scope": "wall-clock calibration only; no acceptance threshold",
        },
        "accounting": count_accounting(corpus, prior, study),
        "diagnostics": _diagnostic_summary(diagnostics),
    }


def _diagnostic_summary(diagnostics: Any) -> dict[str, Any]:
    # Keep the report compact and JSON-safe while retaining explicit raw-input
    # scope. The complete object remains available to callers of data_diagnostics.
    return {
        "source_kind": diagnostics.source_kind,
        "n_worlds": diagnostics.n_worlds,
        "n_time_steps": diagnostics.n_time_steps,
        "views": list(diagnostics.views),
        "scope": "raw observed-node diagnostics; not true mechanism features",
    }


def run_calibration(
    config_path: str | os.PathLike[str] | Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str],
    study_settings_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Generate one bounded calibration report and write it without clobbering runs."""
    raw = load_json(config_path) if not isinstance(config_path, Mapping) else dict(config_path)
    sidecar = load_json(study_settings_path) if study_settings_path is not None else None
    resolved = resolve_config(raw, sidecar)
    generator = resolved["generator_factory"]
    study = resolved["study"]
    low_measurement = _measure_calibration(_low_dimension_generator(generator), study)
    selected_measurement = _measure_calibration(generator, study)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": resolved["config_hash"],
        "resolved_config": resolved,
        "seed_schedule": {"generator_seed": study["seed"]},
        "timing": selected_measurement["timing"],
        "accounting": selected_measurement["accounting"],
        "diagnostics": selected_measurement["diagnostics"],
        "calibration": [
            {"label": "bounded_low", **low_measurement},
            {"label": "selected_configuration", **selected_measurement},
        ],
        "analyses": {
            "raw_input_conditioning": {
                "status": "available",
                "scope": "observed raw levels/differences",
            },
            "true_feature_recovery": unsupported_status(
                "true_feature_recovery",
                "corpus schema does not persist every saturation/coefficient truth required for exact reconstruction",
            ),
            "causal_confounding": {
                "status": "available",
                "scope": "graph/path flags; D->Y reachability is not itself confounding",
            },
        },
        "pilot_claim": "calibration only; no prevalence estimate",
        "proposed_pilot": {
            "unit": "cell",
            "configs": 10,
            "count_band_strata_per_config": 16,
            "cells_per_count_band_stratum": 2,
            "draws_per_cell": 2,
            "cells_total": 320,
            "worlds_total": 640,
            "budget_status": "proposal_only_pending_human_approval",
            "stop_rule": "stop before the sweep if any named stratum exceeds 2x the measured selected-configuration runtime, generation failures exceed 10%, or projected wall time exceeds 30 minutes; seek human review before scaling",
        },
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"linear-recovery-c1-{resolved['config_hash'][:16]}.json"
    payload = canonical_json(report) + "\n"
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite existing report {report_path}")
    else:
        report_path.write_text(payload, encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON generator configuration")
    parser.add_argument("--study-settings", help="JSON sidecar with study settings")
    parser.add_argument(
        "--output-dir", required=True, help="directory for the config-specific report"
    )
    args = parser.parse_args(argv)
    report = run_calibration(
        args.config, output_dir=args.output_dir, study_settings_path=args.study_settings
    )
    print(
        json.dumps(
            {"report": report["report_path"], "config_hash": report["config_hash"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
