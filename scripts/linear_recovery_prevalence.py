"""Configuration-driven C1 calibration and bounded C2 pilot runner.

This is validation-only code. It separates generator, study, and output concerns,
records the resolved configuration before sampling, and treats a cell (not a
sibling world) as the independent unit for binary summaries. C1 calibration and
C2 pilot artifacts are compact JSON; no raw corpus or final prevalence CSV is
written by this runner.

The supported Linux pilot launch sets thread limits before Python starts::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    PYTHONPATH=. uv run --no-sync python scripts/linear_recovery_prevalence.py \\
      --mode pilot --config docs/examples/data/linear-recovery-prevalence-config.json \\
      --output-dir /home/teemu/pymc-labs/prior-generator-artifacts/issue-30-c2 \\
      --wall-time-seconds 5400

These environment variables are not proof of safety. Before every fork the runner
verifies the Linux ``/proc/self/task`` count, including after imports and prior
construction, and fails closed if that count cannot be verified or is not one.
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
PILOT_SCHEMA_VERSION = "linear-recovery-c2/v1"
CI_METHOD = "wilson"
PILOT_WALL_LIMIT_SECONDS = 90 * 60
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
    # C2 is opt-in: C1 keeps its four-cell calibration budget, while pilot
    # settings declare the approved complete 10-config budget separately.
    "pilot_cells_per_config": 32,
    "pilot_siblings_per_cell": 2,
    "pilot_wall_time_seconds": PILOT_WALL_LIMIT_SECONDS,
    "pilot_checkpoint_every_cells": 1,
}
STUDY_KEYS = frozenset(DEFAULT_STUDY)
_FACTORY_KEYS = frozenset(("edge_budget", "nonlinearity", "trajectories"))
_STUDY_ONLY_GENERATOR_KEYS = frozenset(("seed", "n_cells", "draws_per_cell"))
TRAJECTORY_STRESS_CONFIGS = (
    "texture",
    "always_on_spikes",
    "periodic_on_off",
    "delayed_start",
    "ramp_up",
    "decay_to_zero",
    "level_doubling",
    "seasonal",
    "trend",
)
CALIBRATION_SEED_OFFSETS = {
    "primary_low_active": 1,
    "primary_high_active": 2,
    **{f"stress_{name}": 10 + index for index, name in enumerate(TRAJECTORY_STRESS_CONFIGS)},
}


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


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _source_provenance() -> dict[str, Any]:
    """Record source identity separately from the content-addressed config."""
    return {
        "package": "pymc-generator",
        "package_version": __version__,
        "source_commit": _git_revision(),
        "source_dirty": _git_dirty(),
    }


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
        or not value
        or any(
            not isinstance(band, list)
            or len(band) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in band)
            or band[0] > band[1]
            for band in value
        )
    ):
        raise ValueError(f"{name} must contain ascending integer bands")
    bands = [[int(band[0]), int(band[1])] for band in value]
    if any(left[1] >= right[0] for left, right in zip(bands, bands[1:])):
        raise ValueError(f"{name} must be sorted and non-overlapping")
    return bands


def _validate_band_coverage(generator: Mapping[str, Any], study: Mapping[str, Any]) -> None:
    """Require count bands to partition each effective active-count range."""
    for dimension, bands in (
        ("treatments", study["treatment_count_bands"]),
        ("covariates", study["control_count_bands"]),
    ):
        field = f"n_{dimension}_active_range"
        n_field = f"n_{dimension}"
        active_range = generator.get(field, [1, generator[n_field]])
        expected = [int(active_range[0]), int(active_range[1])]
        if bands[0][0] != expected[0] or bands[-1][1] != expected[1]:
            raise ValueError(
                f"{dimension} count bands must completely cover effective active range "
                f"{expected}, got {bands}"
            )
        for left, right in zip(bands, bands[1:]):
            if right[0] != left[1] + 1:
                raise ValueError(f"{dimension} count bands contain a gap: {bands}")


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
    for name in (
        "pilot_cells_per_config",
        "pilot_siblings_per_cell",
        "pilot_checkpoint_every_cells",
    ):
        if (
            isinstance(resolved[name], bool)
            or not isinstance(resolved[name], int)
            or resolved[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if resolved["pilot_cells_per_config"] < 2:
        raise ValueError("pilot_cells_per_config must be at least 2 for generation")
    wall = resolved["pilot_wall_time_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not 0 < wall <= PILOT_WALL_LIMIT_SECONDS
    ):
        raise ValueError(f"pilot_wall_time_seconds must be in (0, {PILOT_WALL_LIMIT_SECONDS}]")
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
    }
    prior = make_prior(generator, study)
    resolved["generator_resolved"] = _plain(dataclasses.asdict(prior))
    _validate_band_coverage(
        {
            "n_treatments": prior.n_treatments,
            "n_covariates": prior.n_covariates,
            "n_treatments_active_range": list(prior.n_treatments_active_range),
            "n_covariates_active_range": list(prior.n_covariates_active_range),
        },
        study,
    )
    # The digest is content-addressed configuration only; source identity is separate.
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
        padded_columns_excluded_by_mask=bool(np.any(~active)),
    )


def classify_graph_paths(
    graph: Mapping[str, Any], focal_treatment: int | None = None
) -> dict[str, Any]:
    """Classify backdoor paths per latent and focal treatment.

    A latent is a potential unobserved confounder only when it reaches both the
    focal treatment and the outcome along a path that bypasses that treatment.
    This avoids treating unrelated latent variables, or ``D -> C -> Y`` alone,
    as confounding. Observed backdoor paths are reported separately because
    their causal identification depends on adjustment.
    """

    def array(name: str, ndim: int) -> np.ndarray:
        value = np.asarray(graph.get(name, []), dtype=bool)
        if value.size == 0:
            return np.zeros((0,) * ndim, dtype=bool)
        if value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        return value

    g_dc = array("g_dc", 2)
    g_dz = array("g_dz", 2)
    g_dy = array("g_dy", 1)
    g_zc = array("g_zc", 2)
    g_zy = array("g_zy", 1)
    g_cc = array("g_cc", 2)
    g_zz = array("g_zz", 2)
    g_cy = array("g_cy", 1)
    n_latents = max(g_dc.shape[0], g_dz.shape[0], g_dy.size)
    n_covariates = max(g_dz.shape[1] if g_dz.ndim == 2 else 0, g_zc.shape[0], g_zy.size)
    n_treatments = max(
        g_dc.shape[1] if g_dc.ndim == 2 else 0, g_zc.shape[1], g_cc.shape[0], g_cy.size
    )
    if focal_treatment is not None and not 0 <= focal_treatment < n_treatments:
        raise ValueError("focal_treatment is outside the graph treatment range")

    adjacency: dict[int, set[int]] = {}
    n_nodes = n_latents + n_covariates + n_treatments + 1
    outcome = n_nodes - 1
    latent_offset = 0
    covariate_offset = n_latents
    treatment_offset = n_latents + n_covariates

    def add_edges(source_offset: int, target_offset: int, matrix: np.ndarray) -> None:
        for source, target in zip(*np.nonzero(matrix)):
            adjacency.setdefault(source_offset + int(source), set()).add(
                target_offset + int(target)
            )

    add_edges(latent_offset, treatment_offset, g_dc)
    add_edges(latent_offset, covariate_offset, g_dz)
    add_edges(covariate_offset, treatment_offset, g_zc)
    add_edges(covariate_offset, covariate_offset, g_zz)
    add_edges(treatment_offset, treatment_offset, g_cc)
    for latent in np.flatnonzero(g_dy):
        adjacency.setdefault(latent_offset + int(latent), set()).add(outcome)
    for covariate in np.flatnonzero(g_zy):
        adjacency.setdefault(covariate_offset + int(covariate), set()).add(outcome)
    for treatment in np.flatnonzero(g_cy):
        adjacency.setdefault(treatment_offset + int(treatment), set()).add(outcome)

    def reachable(start: int, target: int, blocked: set[int] | None = None) -> bool:
        blocked = blocked or set()
        if start in blocked or target in blocked:
            return False
        pending = [start]
        seen = {start}
        while pending:
            current = pending.pop()
            if current == target:
                return True
            for child in adjacency.get(current, ()):
                if child not in blocked and child not in seen:
                    seen.add(child)
                    pending.append(child)
        return False

    focal = range(n_treatments) if focal_treatment is None else (focal_treatment,)
    focal_results: dict[str, Any] = {}
    for treatment in focal:
        treatment_node = treatment_offset + treatment
        latent_results = []
        for latent in range(n_latents):
            treatment_path = reachable(latent_offset + latent, treatment_node)
            outcome_path = reachable(latent_offset + latent, outcome, blocked={treatment_node})
            latent_results.append(
                {
                    "latent": latent,
                    "reaches_focal_treatment": treatment_path,
                    "reaches_outcome_bypassing_focal_treatment": outcome_path,
                    "potential_unobserved_confounding": treatment_path and outcome_path,
                }
            )
        observed_results = []
        for covariate in range(n_covariates):
            covariate_node = covariate_offset + covariate
            treatment_path = reachable(covariate_node, treatment_node)
            outcome_path = reachable(covariate_node, outcome, blocked={treatment_node})
            observed_results.append(
                {
                    "covariate": covariate,
                    "reaches_focal_treatment": treatment_path,
                    "reaches_outcome_bypassing_focal_treatment": outcome_path,
                    "adjustment_dependent_causal_identification": treatment_path and outcome_path,
                }
            )
        focal_results[str(treatment)] = {
            "latent_paths": latent_results,
            "observed_backdoor_paths": observed_results,
            "potential_unobserved_confounding": any(
                item["potential_unobserved_confounding"] for item in latent_results
            ),
            "adjustment_dependent_causal_identification": any(
                item["adjustment_dependent_causal_identification"] for item in observed_results
            ),
        }

    def any_reachable(starts: Any, target_nodes: Any) -> bool:
        return any(
            reachable(int(start), int(target)) for start in starts for target in target_nodes
        )

    latent_nodes = range(latent_offset, latent_offset + n_latents)
    covariate_nodes = range(covariate_offset, covariate_offset + n_covariates)
    treatment_nodes = range(treatment_offset, treatment_offset + n_treatments)
    mediated_latent_to_treatment = any(
        reachable(latent, covariate) and reachable(covariate, treatment)
        for latent in latent_nodes
        for covariate in covariate_nodes
        for treatment in treatment_nodes
    )
    mediated_latent_to_outcome = any(
        reachable(latent, covariate) and reachable(covariate, outcome)
        for latent in latent_nodes
        for covariate in covariate_nodes
    )
    direct_treatment_outcome = any_reachable(treatment_nodes, (outcome,))
    latent_to_treatment = any_reachable(latent_nodes, treatment_nodes)
    latent_to_outcome = any_reachable(latent_nodes, (outcome,))

    return {
        "direct_treatment_outcome_reachability": direct_treatment_outcome,
        "raw_reachability": {
            "latent_to_treatment": latent_to_treatment,
            "latent_to_outcome": latent_to_outcome,
            "mediated_latent_to_treatment": mediated_latent_to_treatment,
            "mediated_latent_to_outcome": mediated_latent_to_outcome,
        },
        "focal_treatments": focal_results,
        "potential_unobserved_confounding": any(
            item["potential_unobserved_confounding"]
            for result in focal_results.values()
            for item in result["latent_paths"]
        ),
        "adjustment_dependent_causal_identification": any(
            result["adjustment_dependent_causal_identification"]
            for result in focal_results.values()
        ),
        # Compatibility labels retained with corrected semantics.
        "latent_to_treatment": latent_to_treatment,
        "latent_to_outcome": latent_to_outcome,
        "shared_latent_confounding": any(
            result["potential_unobserved_confounding"] for result in focal_results.values()
        ),
        "mediated_or_observed_path": any(
            result["adjustment_dependent_causal_identification"]
            for result in focal_results.values()
        ),
    }


def classify_design(
    rank: int,
    n_columns: int,
    condition: float,
    *,
    rank_tolerance: float,
    constant_active_columns: int = 0,
    causal_path: bool = False,
    padded_columns_excluded_by_mask: bool = False,
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
        "padded_columns_excluded_by_mask": bool(padded_columns_excluded_by_mask),
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

    def locate(row: int, column: int) -> tuple[int, int] | None:
        treatment = next(
            (i for i, (low, high) in enumerate(treatment_bands) if low <= row <= high), None
        )
        control = next(
            (j for j, (low, high) in enumerate(control_bands) if low <= column <= high), None
        )
        return None if treatment is None or control is None else (treatment, control)

    unmatched_worlds = 0
    for row, column in zip(treatment_counts, control_counts, strict=True):
        location = locate(int(row), int(column))
        if location is None:
            unmatched_worlds += 1
        else:
            world_matrix[location] += 1
    unmatched_cells = 0
    for row, column in zip(cell_treatments, cell_controls, strict=True):
        location = locate(int(row), int(column))
        if location is None:
            unmatched_cells += 1
        else:
            cell_matrix[location] += 1
    if unmatched_worlds or unmatched_cells:
        raise ValueError(
            "count bands do not cover realized active counts: "
            f"{unmatched_cells} cells and {unmatched_worlds} worlds"
        )
    empty_strata = [
        [i, j]
        for i in range(cell_matrix.shape[0])
        for j in range(cell_matrix.shape[1])
        if cell_matrix[i, j] == 0
    ]
    return {
        "treatment_bands": treatment_bands,
        "control_bands": control_bands,
        "n_cells": cell_matrix.tolist(),
        "n_worlds": world_matrix.tolist(),
        "empty_cell_band_strata": empty_strata,
        "n_empty_cell_band_strata": len(empty_strata),
        "unmatched_cells": unmatched_cells,
        "unmatched_worlds": unmatched_worlds,
    }


def count_accounting(
    corpus: Mapping[str, Any], prior: SCMPrior, study: Mapping[str, Any]
) -> dict[str, Any]:
    """Return realized cell/world counts, including rejected/failure placeholders."""
    _validate_band_coverage(
        {
            "n_treatments": prior.n_treatments,
            "n_covariates": prior.n_covariates,
            "n_treatments_active_range": list(prior.n_treatments_active_range),
            "n_covariates_active_range": list(prior.n_covariates_active_range),
        },
        study,
    )
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
    rejected = evaluated - n_worlds
    if rejected < 0:
        raise ValueError("evaluated candidates cannot be fewer than accepted worlds")
    return {
        "requested_cells": int(prior.n_cells),
        "realized_cells": n_cells,
        "requested_worlds": int(prior.n_cells * prior.draws_per_cell),
        "realized_worlds": n_worlds,
        "accepted_worlds": n_worlds,
        "evaluated_candidates": evaluated,
        "rejected_candidates": rejected,
        "generation_failures": failures,
        "evaluated": evaluated,
        "accepted": n_worlds,
        "rejected": rejected,
        "failures": failures,
        "draws_per_cell": int(prior.draws_per_cell),
        "coverage": summary,
        "count_band_accounting": _band_accounting(
            corpus,
            study["treatment_count_bands"],
            study["control_count_bands"],
        ),
        "unit": "cell for binary inference; world for generation accounting",
    }


def _active_dimension_generator(
    generator: Mapping[str, Any], level: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build low/high variants by pinning only resolved active-count ranges."""
    if level not in {"low", "high"}:
        raise ValueError("active-dimension level must be low or high")
    base = dict(generator)
    treatment_range = _as_range(
        base.get("n_treatments_active_range", [1, base["n_treatments"]]),
        "n_treatments_active_range",
    )
    covariate_range = _as_range(
        base.get("n_covariates_active_range", [1, base["n_covariates"]]),
        "n_covariates_active_range",
    )
    latent_range = _as_range(
        base.get("n_latent_active_range", [1, base["n_latent"]]),
        "n_latent_active_range",
    )
    assert treatment_range is not None and covariate_range is not None and latent_range is not None
    active = (
        treatment_range[0] if level == "low" else treatment_range[1],
        covariate_range[0] if level == "low" else covariate_range[1],
        latent_range[0] if level == "low" else latent_range[1],
    )
    variant = {
        **base,
        "n_treatments_active_range": [active[0], active[0]],
        "n_covariates_active_range": [active[1], active[1]],
        "n_latent_active_range": [active[2], active[2]],
    }
    overrides = {key: value for key, value in variant.items() if base.get(key) != value}
    for key, value in base.items():
        if key not in overrides:
            assert variant[key] == value
    assert set(overrides) <= {
        "n_treatments_active_range",
        "n_covariates_active_range",
        "n_latent_active_range",
    }
    return variant, {
        "active_dimensions": {
            "treatments": active[0],
            "covariates": active[1],
            "latent": active[2],
        },
        "overrides": overrides,
        "study_seed_disclosed_separately": True,
        "label": "runtime_stress_variant",
    }


def _low_dimension_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the deterministic low-active variant."""
    return _active_dimension_generator(generator, "low")[0]


def _high_dimension_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    return _active_dimension_generator(generator, "high")[0]


def _calibration_study(study: Mapping[str, Any], seed: int) -> dict[str, Any]:
    resolved = dict(study)
    resolved["seed"] = int(seed)
    return resolved


def _stress_configurations(generator: Mapping[str, Any], base_seed: int) -> list[dict[str, Any]]:
    """Serialize the primary recipe and every named trajectory stress variant."""
    configurations = [
        {
            "name": "primary_composable",
            "label": "selected_primary_recipe",
            "generator": dict(generator),
            "seed": int(base_seed),
        }
    ]
    for name in TRAJECTORY_STRESS_CONFIGS:
        configurations.append(
            {
                "name": name,
                "label": "runtime_stress_variant",
                "base_recipe": "primary_composable",
                "override": {"trajectories": name},
                "generator": {**generator, "trajectories": name},
                "seed": int(base_seed + CALIBRATION_SEED_OFFSETS[f"stress_{name}"]),
            }
        )
    return configurations


def _study_for_generator(generator: Mapping[str, Any], study: Mapping[str, Any]) -> dict[str, Any]:
    """Use complete one-band coverage for fixed active-dimension stress variants."""
    resolved = dict(study)
    for dimension, field in (
        ("treatments", "treatment_count_bands"),
        ("covariates", "control_count_bands"),
    ):
        active_range = generator.get(
            f"n_{dimension}_active_range", [1, generator[f"n_{dimension}"]]
        )
        bands = resolved[field]
        if bands[0][0] != active_range[0] or bands[-1][1] != active_range[1]:
            resolved[field] = [[int(active_range[0]), int(active_range[1])]]
    return resolved


def _measure_calibration(generator: Mapping[str, Any], study: Mapping[str, Any]) -> dict[str, Any]:
    study = _study_for_generator(generator, study)
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


def _artifact_size_estimate(
    fixed_report: Mapping[str, Any], proposed: Mapping[str, Any]
) -> dict[str, Any]:
    """Estimate compact pilot JSON size without counting generated raw arrays."""
    fixed_bytes = len(canonical_json(fixed_report).encode("utf-8"))
    representative_world = {
        "config_index": 0,
        "cell_index": 0,
        "sibling_index": 0,
        "accepted": True,
        "generation_failed": False,
    }
    representative_cell = {
        "config_index": 0,
        "cell_index": 0,
        "success": False,
        "estimand": "first_world",
    }
    world_bytes = len(canonical_json(representative_world).encode("utf-8"))
    cell_bytes = len(canonical_json(representative_cell).encode("utf-8"))
    world_count = int(proposed["worlds_total"])
    cell_count = int(proposed["cells_total"])
    projected_bytes = fixed_bytes + world_count * world_bytes + cell_count * cell_bytes
    return {
        "method": "fixed canonical UTF-8 JSON schema/config/summary overhead plus compact records",
        "assumptions": {
            "serialization": "canonical JSON with sorted keys and no whitespace",
            "persisted_arrays": False,
            "fixed_overhead": "schema, resolved config, provenance, seed schedule, calibration summaries, analyses, and runtime summary",
            "representative_records": "one compact per-world accounting record and one compact per-cell metric record",
            "expected_counts": {
                "configs": int(proposed["configs"]),
                "cells": cell_count,
                "worlds": world_count,
            },
        },
        "fixed_schema_config_summary_bytes": fixed_bytes,
        "representative_world_record": {
            "schema": representative_world,
            "bytes": world_bytes,
        },
        "representative_cell_record": {
            "schema": representative_cell,
            "bytes": cell_bytes,
        },
        "arithmetic": (
            f"{fixed_bytes} + ({world_count} * {world_bytes}) + "
            f"({cell_count} * {cell_bytes}) = {projected_bytes} bytes"
        ),
        "projected_bytes": projected_bytes,
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
    effective_generator = dict(generator)
    for field in (
        "n_treatments_active_range",
        "n_covariates_active_range",
        "n_latent_active_range",
    ):
        effective_generator[field] = resolved["generator_resolved"][field]
    low_generator, low_variant = _active_dimension_generator(effective_generator, "low")
    high_generator, high_variant = _active_dimension_generator(effective_generator, "high")
    low_seed = study["seed"] + CALIBRATION_SEED_OFFSETS["primary_low_active"]
    high_seed = study["seed"] + CALIBRATION_SEED_OFFSETS["primary_high_active"]
    low_measurement = _measure_calibration(low_generator, _calibration_study(study, low_seed))
    high_measurement = _measure_calibration(high_generator, _calibration_study(study, high_seed))
    selected_measurement = _measure_calibration(effective_generator, study)
    stress_configs = _stress_configurations(effective_generator, study["seed"])
    stress_seed_schedule = {
        item["name"]: item["seed"]
        for item in stress_configs
        if item["name"] != "primary_composable"
    }
    proposed = {
        "unit": "cell",
        "configs": len(stress_configs),
        "independent_cells_per_config": 32,
        "siblings_per_cell": 2,
        "cells_total": len(stress_configs) * 32,
        "worlds_total": len(stress_configs) * 32 * 2,
        "allocation": "independent active counts; band occupancy is realized, not fixed",
        "count_band_strata_per_config": len(study["treatment_count_bands"])
        * len(study["control_count_bands"]),
        "balanced_stratification": "not proposed; requires separate approval",
        "uncertainty": "Wilson intervals use 32 independent cell indicators; empty or sparsely realized bands have wide or unavailable intervals",
        "budget_status": "proposal_only_pending_human_approval",
        "stop_rule": "pilot wall cap is 90 minutes; stop before the next config if generation failures exceed 10% of evaluated candidates; active-count bands are not balanced, so stratum overrun is reported rather than silently stopped",
    }
    high_seconds = (
        high_measurement["timing"]["generation_seconds"]
        + high_measurement["timing"]["diagnostics_seconds"]
    )
    high_worlds = max(high_measurement["accounting"]["realized_worlds"], 1)
    projected_seconds = high_seconds / high_worlds * proposed["worlds_total"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": resolved["config_hash"],
        "resolved_config": resolved,
        "source_provenance": _source_provenance(),
        "seed_schedule": {
            "primary_composable": study["seed"],
            "primary_low_active": low_seed,
            "primary_high_active": high_seed,
            "stress_variants": stress_seed_schedule,
        },
        "timing": selected_measurement["timing"],
        "accounting": selected_measurement["accounting"],
        "diagnostics": selected_measurement["diagnostics"],
        "calibration": [
            {
                "label": "low_active_runtime_stress_variant",
                "variant": low_variant,
                "seed": low_seed,
                **low_measurement,
            },
            {
                "label": "high_active_runtime_stress_variant",
                "variant": high_variant,
                "seed": high_seed,
                **high_measurement,
            },
            {
                "label": "selected_primary_recipe",
                "seed": study["seed"],
                **selected_measurement,
            },
        ],
        "stress_configurations": stress_configs,
        "projected_runtime": {
            "method": "scale measured high-active primary wall time per realized world to 10 configurations x 32 cells x 2 siblings; this is a budget estimate, not a promise",
            "seconds": projected_seconds,
            "minutes": projected_seconds / 60.0,
        },
        "analyses": {
            "raw_input_conditioning": {
                "status": "available",
                "scope": "observed raw levels/differences; raw reachability is not causal confounding",
            },
            "true_feature_recovery": unsupported_status(
                "true_feature_recovery",
                "corpus schema does not persist every saturation/coefficient truth required for exact reconstruction",
            ),
            "causal_confounding": {
                "status": "available",
                "scope": "per-latent/per-focal-treatment graph backdoor classification; potential unobserved confounding is separate from adjustment-dependent identification",
            },
        },
        "pilot_claim": "calibration only; no prevalence estimate",
        "proposed_pilot": proposed,
    }
    report["expected_artifact_size"] = _artifact_size_estimate(report, proposed)
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


def _finite_max(values: Any) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


def _max_off_diagonal(values: Any, valid: Any) -> float | None:
    matrix = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if matrix.ndim != 2 or matrix.shape != mask.shape:
        raise ValueError("diagnostic matrices and validity masks must have equal rank-2 shape")
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    selected = np.abs(matrix[mask & off_diagonal])
    return _finite_max(selected)


def _compact_world_metric(
    corpus: Mapping[str, Any],
    prior: SCMPrior,
    diagnostic: Any,
    row: int,
    *,
    config_index: int,
    cell_index: int,
    sibling_index: int,
    rank_tolerance: float,
) -> dict[str, Any]:
    """Create one compact, raw-input-labelled metric row for a corpus world."""
    treatment_mask = np.asarray(corpus["treatment_active_mask"][row], dtype=bool)
    covariate_mask = np.asarray(corpus["covariate_active_mask"][row], dtype=bool)
    treatment = np.asarray(corpus["treatment_raw"][row], dtype=float)
    covariates = np.asarray(corpus["covariates"][row], dtype=float)
    columns = [treatment[:, treatment_mask], covariates[:, covariate_mask]]
    predictors = np.column_stack([item for item in columns if item.shape[1]])
    constants = [
        *(f"C{index + 1}" for index, active in enumerate(treatment_mask) if active),
        *(f"Z{index + 1}" for index, active in enumerate(covariate_mask) if active),
    ]

    views: dict[str, dict[str, Any]] = {}
    for view_name, view in diagnostic.views.items():
        # Dependence diagnostics are computed on the requested view.  The
        # design diagnostics must use that same representation: in particular,
        # differencing happens before rank/condition calculations.
        view_predictors = np.diff(predictors, axis=0) if view_name == "differences" else predictors
        view_design = np.column_stack([view_predictors, np.ones(view_predictors.shape[0])])
        view_rank = design_rank_condition(view_design, rank_tolerance=rank_tolerance)
        view_scales = np.std(view_predictors, axis=0)
        view_constant_inputs = [
            name
            for name, scale in zip(constants, view_scales, strict=True)
            if scale <= rank_tolerance
        ]
        view_standardized = np.zeros_like(view_predictors)
        view_nonconstant = view_scales > rank_tolerance
        if np.any(view_nonconstant):
            view_standardized[:, view_nonconstant] = (
                view_predictors[:, view_nonconstant]
                - view_predictors[:, view_nonconstant].mean(axis=0)
            ) / view_scales[view_nonconstant]
        view_standardized_metric = design_rank_condition(
            np.column_stack([view_standardized, np.ones(view_standardized.shape[0])]),
            rank_tolerance=rank_tolerance,
        )
        dependence = view.dependence
        keys = [descriptor.key for descriptor in dependence.descriptors]
        observed = [i for i, key in enumerate(keys) if key.startswith(("C", "Z"))]
        index = np.ix_(observed, observed)
        pearson = dependence.matrices["pearson"][row][index]
        pearson_valid = dependence.valid["pearson"][row][index]
        vif = view.vif["observed"]
        vif_values = vif.vif[row][vif.valid[row]]
        views[view_name] = {
            "max_abs_pearson_correlation": _max_off_diagonal(pearson, pearson_valid),
            "max_vif": _finite_max(vif_values),
            "vif_infinite": bool(np.isposinf(vif.vif[row]).any()),
            "rank": view_rank,
            "standardized": view_standardized_metric,
            "constant_active_inputs": view_constant_inputs,
        }
    graph = prior.layout.unpack(np.asarray(corpus["g"][row]))
    graph_flags = classify_graph_paths(graph)
    components = {}
    for name in ("treatment_components", "covariate_components"):
        if name in corpus:
            values = np.asarray(corpus[name][row])
            mask = treatment_mask if name.startswith("treatment") else covariate_mask
            components[name] = values[mask].astype(int).tolist()
    if "sat_family" in corpus:
        components["sat_family"] = (
            np.asarray(corpus["sat_family"][row])[treatment_mask].astype(int).tolist()
        )
    return {
        "config_index": config_index,
        "cell_index": cell_index,
        "sibling_index": sibling_index,
        "world_index": int(row),
        "cell_id": int(corpus["cell_id"][row]),
        "active_counts": {
            "treatments": int(treatment_mask.sum()),
            "covariates": int(covariate_mask.sum()),
            "latents": int(np.asarray(corpus["latent_active_mask"][row]).sum()),
        },
        "active_masks": {
            "treatments": treatment_mask.astype(int).tolist(),
            "covariates": covariate_mask.astype(int).tolist(),
            "latents": np.asarray(corpus["latent_active_mask"][row], dtype=bool)
            .astype(int)
            .tolist(),
        },
        "components": components,
        "raw_input_diagnostics": {
            "scope": "observed raw inputs; configured views only; not true mechanism features",
            "views": views,
        },
        "graph_flags": graph_flags,
    }


def _pilot_configurations(generator: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    """Return the preregistered primary plus nine labelled stress recipes."""
    return _stress_configurations(generator, seed)


def pilot_seed_schedule(generator: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    """Expose the deterministic config schedule without drawing any worlds."""
    return [
        {"index": index, "name": item["name"], "seed": item["seed"], "generator": item["generator"]}
        for index, item in enumerate(_pilot_configurations(generator, seed))
    ]


def _threshold_key(value: float) -> str:
    return str(value).replace(".", "p")


def _band_for_count(count: int, bands: list[list[int]]) -> list[int] | None:
    return next((band for band in bands if band[0] <= count <= band[1]), None)


def _world_indicators(
    record: Mapping[str, Any], thresholds: Mapping[str, list[float]]
) -> dict[str, dict[str, bool]]:
    raw = record["raw_input_diagnostics"]
    result: dict[str, dict[str, bool]] = {}
    for view_name, view in raw.get("views", {}).items():
        rank = view.get("rank", {})
        condition = rank.get("condition")
        indicators = {
            "rank_deficient": bool(rank.get("rank_deficient", False)),
        }
        for threshold in thresholds["condition"]:
            indicators[f"condition_ge_{_threshold_key(threshold)}"] = (
                condition is not None and condition >= threshold
            )
        correlation = view.get("max_abs_pearson_correlation")
        for threshold in thresholds["absolute_correlation"]:
            indicators[f"absolute_correlation_ge_{_threshold_key(threshold)}"] = (
                correlation is not None and correlation >= threshold
            )
        vif = view.get("max_vif")
        vif_infinite = bool(view.get("vif_infinite", False))
        for threshold in thresholds["vif"]:
            indicators[f"vif_ge_{_threshold_key(threshold)}"] = vif_infinite or (
                vif is not None and vif >= threshold
            )
        result[view_name] = indicators
    return result


def _pilot_cell_records(
    world_records: list[dict[str, Any]],
    *,
    thresholds: Mapping[str, list[float]] | None = None,
    treatment_bands: list[list[int]] | None = None,
    control_bands: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or DEFAULT_STUDY["thresholds"]
    treatment_bands = treatment_bands or DEFAULT_STUDY["treatment_count_bands"]
    control_bands = control_bands or DEFAULT_STUDY["control_count_bands"]
    by_cell: dict[int, list[dict[str, Any]]] = {}
    for record in world_records:
        by_cell.setdefault(int(record["cell_id"]), []).append(record)
    cells = []
    for cell_id, siblings in sorted(by_cell.items()):
        first = min(siblings, key=lambda item: item["sibling_index"])
        first_indicators = _world_indicators(first, thresholds)
        sibling_indicators = [_world_indicators(item, thresholds) for item in siblings]
        any_indicators = {
            view: {
                metric: any(item[view][metric] for item in sibling_indicators)
                for metric in first_indicators[view]
            }
            for view in first_indicators
        }
        counts = first.get("active_counts", {})
        treatment_band = _band_for_count(int(counts.get("treatments", 0)), treatment_bands)
        control_band = _band_for_count(int(counts.get("covariates", 0)), control_bands)
        cells.append(
            {
                "cell_id": cell_id,
                "n_siblings": len(siblings),
                "active_counts": counts,
                "count_band": {
                    "treatments": treatment_band,
                    "controls": control_band,
                },
                "first_world": {
                    "world_index": first.get("world_index"),
                    "sibling_index": first["sibling_index"],
                    "views": first["raw_input_diagnostics"].get("views", {}),
                },
                "binary": first_indicators,
                "any_sibling": any_indicators,
            }
        )
    return cells


def _summarize_indicator_set(
    cells: list[dict[str, Any]], estimand: str, confidence: float, view: str, metric: str
) -> dict[str, Any]:
    values = []
    for cell in cells:
        binary = cell["binary"] if estimand == "first_world" else cell["any_sibling"]
        if view not in binary or metric not in binary[view]:
            raise ValueError(f"diagnostic view {view!r} is unavailable")
        values.append(bool(binary[view][metric]))
    return cell_binary_summary(
        [cell["cell_id"] for cell in cells],
        values,
        estimand=estimand,
        confidence_level=confidence,
    )


def _pilot_summaries(
    cells: list[dict[str, Any]],
    confidence: float,
    *,
    thresholds: Mapping[str, list[float]] | None = None,
    treatment_bands: list[list[int]] | None = None,
    control_bands: list[list[int]] | None = None,
) -> dict[str, Any]:
    if not cells:
        return {"status": "unavailable", "reason": "no complete cells"}
    thresholds = thresholds or DEFAULT_STUDY["thresholds"]
    binary = cells[0]["binary"]
    views = [view for view in ("levels", "differences") if view in binary]
    metrics = list(binary[views[0]]) if views else []
    summary: dict[str, Any] = {
        "status": "available",
        "unit": "cell",
        "thresholds": _plain(thresholds),
        "estimands": {},
    }
    for estimand in ("first_world", "any_sibling"):
        summary["estimands"][estimand] = {
            view: {
                metric: _summarize_indicator_set(cells, estimand, confidence, view, metric)
                for metric in metrics
            }
            for view in views
        }
    if treatment_bands is not None and control_bands is not None:
        summary["count_band_wilson"] = {}
        for i, treatment_band in enumerate(treatment_bands):
            for j, control_band in enumerate(control_bands):
                selected = [
                    cell
                    for cell in cells
                    if cell["count_band"]["treatments"] == treatment_band
                    and cell["count_band"]["controls"] == control_band
                ]
                key = f"treatments_{treatment_band[0]}_{treatment_band[1]}__controls_{control_band[0]}_{control_band[1]}"
                summary["count_band_wilson"][key] = {
                    "treatment_band": treatment_band,
                    "control_band": control_band,
                    "status": "available" if selected else "unavailable",
                    "n_cells": len(selected),
                    "estimands": {
                        estimand: {
                            view: {
                                metric: (
                                    _summarize_indicator_set(
                                        selected, estimand, confidence, view, metric
                                    )
                                    if selected
                                    else {
                                        "estimand": estimand,
                                        "unit": "cell",
                                        "n_cells": 0,
                                        "status": "unavailable",
                                    }
                                )
                                for metric in metrics
                            }
                            for view in views
                        }
                        for estimand in ("first_world", "any_sibling")
                    },
                }
        summary["empty_band_policy"] = "zero-denominator bands are unavailable/missing, never zero"
    summary["sensitivity_note"] = (
        "any_sibling is a separate sensitivity estimand; siblings are not pooled as independent cells"
    )
    return summary


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _required_source_provenance(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} provenance is missing")
    required = ("package", "package_version", "source_commit", "source_dirty")
    if any(key not in value for key in required):
        raise ValueError(f"{label} provenance is incomplete")
    return {key: value[key] for key in required}


def validate_pilot_report(
    report: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate a report against its immutable run manifest.

    A report may legitimately come from an older source checkout.  Its source
    identity is therefore compared with the archived manifest for this run,
    rather than with the current checkout, while config/schema identity is
    still checked against the requested resolved configuration.
    """
    if report.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ValueError("pilot report schema mismatch")
    if report.get("config_hash") != resolved.get("config_hash"):
        raise ValueError("pilot report config hash mismatch")
    if report.get("resolved_config") != resolved:
        raise ValueError("pilot report resolved configuration mismatch")
    manifest_path = report.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError("pilot report manifest path is missing")
    if manifest is None:
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError("pilot report manifest cannot be loaded") from exc
    else:
        try:
            archived_manifest = load_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError("pilot report manifest cannot be loaded") from exc
        if archived_manifest != manifest:
            raise ValueError("pilot report manifest does not match its archived path")
    if manifest.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ValueError("pilot manifest schema mismatch")
    if manifest.get("config_hash") != resolved.get("config_hash"):
        raise ValueError("pilot manifest config hash mismatch")
    if manifest.get("resolved_config") != resolved:
        raise ValueError("pilot manifest resolved configuration mismatch")
    manifest_provenance = _required_source_provenance(
        manifest.get("source_provenance"), label="manifest"
    )
    report_provenance = _required_source_provenance(report.get("source_provenance"), label="report")
    if report_provenance != manifest_provenance:
        raise ValueError("pilot report provenance does not match its manifest")
    manifest_hash = config_hash(manifest)
    if report.get("manifest_hash") != manifest_hash:
        raise ValueError("pilot report manifest hash mismatch")
    if not isinstance(report.get("schedule"), list) or not report["schedule"]:
        raise ValueError("pilot report seed schedule is missing")
    if report.get("schedule") != manifest.get("schedule"):
        raise ValueError("pilot report schedule does not match its manifest")
    if report.get("status") not in {"complete", "partial"}:
        raise ValueError("pilot report status is invalid")


class _PilotDeadlineExceededError(RuntimeError):
    pass


class _PilotInterruptedError(RuntimeError):
    pass


# These are deliberately finite: a phase that ignores TERM must not hold the
# parent in cleanup indefinitely.  The child is owned by this invocation only.
_PHASE_TERM_GRACE_SECONDS = 0.25
_PHASE_KILL_JOIN_SECONDS = 0.25
_MINIMAL_FINALIZATION_ALLOWANCE_SECONDS = 0.5


def _actual_os_thread_count() -> int:
    """Return the Linux thread count, failing closed when it is unavailable."""
    if os.name != "posix":
        raise RuntimeError("bounded pilot phases require Linux /proc thread accounting")
    task_path = Path("/proc/self/task")
    try:
        count = sum(1 for entry in task_path.iterdir() if entry.is_dir())
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("cannot verify Linux OS thread count before fork") from exc
    if count < 1:
        raise RuntimeError("cannot verify Linux OS thread count before fork")
    return count


def _require_single_os_thread() -> None:
    """Reject native or Python multithreading before an inherited fork."""
    count = _actual_os_thread_count()
    if count != 1:
        raise RuntimeError(
            f"bounded pilot phases require exactly one OS thread before fork; found {count}"
        )


def _run_phase_child(function: Any, result_path: str) -> None:
    """Execute one inherited callable and serialize its result out of band."""
    import pickle
    import signal

    # The parent installs a TERM handler while running the pilot.  Inheriting
    # it across fork would run parent cleanup code in the child and can leave
    # blocking C work alive; a phase child has the normal default semantics.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        result = ("ok", function())
    except BaseException as exc:  # transfer the phase exception exactly once
        result = ("error", type(exc).__name__, str(exc))
    with open(result_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _bounded_phase(function: Any, deadline: float, clock: Any) -> Any:
    """Run blocking work in an owned process with a hard parent-side timeout."""
    if float(clock()) >= deadline:
        raise _PilotDeadlineExceededError("pilot wall deadline reached before phase")
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("bounded pilot phases require fork process isolation on this host")
    # Python's threading.active_count() misses native BLAS/OpenMP workers.  This
    # check is deliberately immediately before every fork.
    _require_single_os_thread()
    import pickle
    import tempfile

    context = multiprocessing.get_context("fork")
    result_fd, result_path = tempfile.mkstemp(prefix="linear-recovery-phase-")
    os.close(result_fd)
    process = context.Process(target=_run_phase_child, args=(function, result_path), daemon=True)
    try:
        process.start()
    except BaseException:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass
        raise

    def stop_owned_child() -> None:
        if not process.is_alive():
            process.join(_PHASE_KILL_JOIN_SECONDS)
            return
        process.terminate()
        process.join(_PHASE_TERM_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(_PHASE_KILL_JOIN_SECONDS)
        if process.is_alive():
            raise RuntimeError("owned pilot phase child did not exit after SIGKILL")

    try:
        while process.is_alive():
            remaining = float(deadline) - float(clock())
            if remaining <= 0:
                stop_owned_child()
                raise _PilotDeadlineExceededError("pilot wall deadline reached during phase")
            process.join(min(remaining, 0.05))
        process.join(_PHASE_KILL_JOIN_SECONDS)
        try:
            with open(result_path, "rb") as handle:
                result = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError) as exc:
            raise RuntimeError("bounded pilot phase exited without a result") from exc
        if result[0] == "ok":
            return result[1]
        name, message = result[1], result[2]
        if name == "ValueError":
            raise ValueError(message)
        raise RuntimeError(f"pilot phase {name}: {message}")
    except BaseException:
        if process.is_alive():
            stop_owned_child()
        raise
    finally:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass
        if not process.is_alive():
            process.close()


def _build_pilot_report(
    checkpoint: Mapping[str, Any],
    configs: list[Mapping[str, Any]],
    study: Mapping[str, Any],
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: list[Mapping[str, Any]],
    manifest_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    partial: bool,
    interrupted: bool,
    stop_reason: str | None,
    elapsed: Any,
    finalization_overdue: Any,
    wall_budget_seconds: float,
) -> dict[str, Any]:
    """Build the report from checkpoint state without hiding partial work."""
    world_records = checkpoint.get("worlds", [])
    cell_records = checkpoint.get("cells", [])
    summaries = []
    for config_index, item in enumerate(configs):
        if finalization_overdue():
            partial = True
            stop_reason = "wall deadline reached during finalization"
            break
        worlds = [row for row in world_records if int(row["config_index"]) == config_index]
        cells = _pilot_cell_records(
            worlds,
            thresholds=study["thresholds"],
            treatment_bands=study["treatment_count_bands"],
            control_bands=study["control_count_bands"],
        )
        complete_cells = [
            cell for cell in cells if cell["n_siblings"] == int(study["pilot_siblings_per_cell"])
        ]
        summaries.append(
            {
                "config_index": config_index,
                "name": item["name"],
                "label": item.get("label", "runtime_stress_variant"),
                "generator": item["generator"],
                "seed": item["seed"],
                "worlds": worlds,
                "cells": cells,
                "summary": _pilot_summaries(
                    complete_cells,
                    float(study["confidence_level"]),
                    thresholds=study["thresholds"],
                    treatment_bands=study["treatment_count_bands"],
                    control_bands=study["control_count_bands"],
                ),
                "accounting": next(
                    (
                        row
                        for row in checkpoint.get("config_accounting", [])
                        if int(row["config_index"]) == config_index
                    ),
                    {"status": "unavailable", "reason": "configuration was not completed"},
                ),
            }
        )
    if finalization_overdue():
        partial = True
        stop_reason = "wall deadline reached during finalization"
    requested_cells = len(configs) * int(study["pilot_cells_per_config"])
    complete = sum(1 for row in cell_records if row.get("status") == "complete") == requested_cells
    status = "complete" if complete and not partial else "partial"
    views = list(study["diagnostic_views"])
    scope = ", ".join(views)
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": status,
        "config_hash": resolved["config_hash"],
        "resolved_config": resolved,
        "source_provenance": manifest["source_provenance"],
        "manifest_hash": config_hash(manifest),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "report_path": str(report_path),
        "schedule": schedule,
        "stop_reason": stop_reason,
        "interrupted": interrupted,
        "configs": summaries,
        "accounting": {
            "evaluated_world_metrics": len(world_records),
            "accepted_world_metrics": len(world_records),
            "evaluated_candidates": sum(
                int(item.get("evaluated_candidates", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "accepted_candidates": sum(
                int(item.get("accepted_worlds", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "realized_cells": len(cell_records),
            "complete_cells": sum(1 for item in cell_records if item.get("status") == "complete"),
            "partial_cells": sum(1 for item in cell_records if item.get("status") != "complete"),
            "requested_cells": requested_cells,
            "requested_worlds": len(configs)
            * int(study["pilot_cells_per_config"])
            * int(study["pilot_siblings_per_cell"]),
            "rejected_candidates": sum(
                int(item.get("rejected_candidates", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "generation_failures": sum(
                int(item.get("generation_failures", 0))
                for item in checkpoint.get("config_accounting", [])
            )
            + len(checkpoint.get("config_failures", [])),
            "failures": sum(
                int(item.get("generation_failures", 0))
                for item in checkpoint.get("config_accounting", [])
            )
            + len(checkpoint.get("config_failures", [])),
            "wall_elapsed_seconds": elapsed(),
            "wall_budget_seconds": wall_budget_seconds,
            "empty_band_policy": "zero-denominator bands are unavailable/missing, never zero",
        },
        "analyses": {
            "raw_input_conditioning": {
                "status": "available",
                "scope": f"{scope} only; not true-mechanism identifiability",
                "views": views,
            },
            "true_feature_recovery": unsupported_status(
                "true_feature_recovery",
                "C2 corpus does not persist every truth field required for exact reconstruction",
            ),
            "causal_confounding": {
                "status": "available",
                "scope": "graph flags are reported separately from raw-input dependence",
            },
        },
        "pilot_claim": "partial pilot; no prevalence estimate until all preregistered cells complete"
        if status == "partial"
        else "bounded pilot summaries by configuration and realized count band; not a training-prior prevalence claim",
        "ci_note": "Wilson intervals use one first-world indicator per complete cell; any_sibling is sensitivity only and siblings are not pooled",
    }
    return report


def _minimal_partial_report(
    checkpoint: Mapping[str, Any],
    configs: list[Mapping[str, Any]],
    study: Mapping[str, Any],
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: list[Mapping[str, Any]],
    manifest_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    interrupted: bool,
    stop_reason: str | None,
    elapsed: Any,
    wall_budget_seconds: float,
) -> dict[str, Any]:
    """Build a deadline-safe report from checkpointed records only.

    This path intentionally does not call cell compaction or summary helpers:
    finalization must remain bounded even when a normal summary is slow.
    """
    worlds = list(checkpoint.get("worlds", []))
    cells = list(checkpoint.get("cells", []))
    summaries = []
    for config_index, item in enumerate(configs):
        config_worlds = [row for row in worlds if int(row.get("config_index", -1)) == config_index]
        config_cells = [row for row in cells if int(row.get("config_index", -1)) == config_index]
        summaries.append(
            {
                "config_index": config_index,
                "name": item["name"],
                "label": item.get("label", "runtime_stress_variant"),
                "generator": item["generator"],
                "seed": item["seed"],
                "world_count": len(config_worlds),
                "cell_count": len(config_cells),
                "complete_cell_count": sum(row.get("status") == "complete" for row in config_cells),
                "world_refs": config_worlds,
                "cell_refs": config_cells,
                "summary": {
                    "status": "unavailable",
                    "reason": "finalization deadline reached before aggregate summaries",
                },
                "accounting": next(
                    (
                        row
                        for row in checkpoint.get("config_accounting", [])
                        if int(row.get("config_index", -1)) == config_index
                    ),
                    {"status": "unavailable", "reason": "configuration was not completed"},
                ),
            }
        )
    requested_cells = len(configs) * int(study["pilot_cells_per_config"])
    failures = sum(
        int(item.get("generation_failures", 0)) for item in checkpoint.get("config_accounting", [])
    ) + len(checkpoint.get("config_failures", []))
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "partial",
        "config_hash": resolved["config_hash"],
        "resolved_config": resolved,
        "source_provenance": manifest["source_provenance"],
        "manifest_hash": config_hash(manifest),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "report_path": str(report_path),
        "schedule": schedule,
        "stop_reason": stop_reason,
        "interrupted": interrupted,
        "configs": summaries,
        "accounting": {
            "evaluated_world_metrics": len(worlds),
            "accepted_world_metrics": len(worlds),
            "realized_cells": len(cells),
            "complete_cells": sum(row.get("status") == "complete" for row in cells),
            "partial_cells": sum(row.get("status") != "complete" for row in cells),
            "requested_cells": requested_cells,
            "requested_worlds": requested_cells * int(study["pilot_siblings_per_cell"]),
            "evaluated_candidates": sum(
                int(item.get("evaluated_candidates", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "accepted_candidates": sum(
                int(item.get("accepted_worlds", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "rejected_candidates": sum(
                int(item.get("rejected_candidates", 0))
                for item in checkpoint.get("config_accounting", [])
            ),
            "generation_failures": failures,
            "failures": failures,
            "wall_elapsed_seconds": elapsed(),
            "wall_budget_seconds": wall_budget_seconds,
            "empty_band_policy": "zero-denominator bands are unavailable/missing, never zero",
        },
        "analyses": {
            "raw_input_conditioning": {
                "status": "available",
                "scope": "checkpointed views only; aggregate summaries unavailable",
                "views": list(study["diagnostic_views"]),
            },
            "true_feature_recovery": unsupported_status(
                "true_feature_recovery",
                "C2 corpus does not persist every truth field required for exact reconstruction",
            ),
            "causal_confounding": {
                "status": "available",
                "scope": "checkpointed graph flags only; aggregate summaries unavailable",
            },
        },
        "pilot_claim": "partial pilot; no prevalence estimate until all preregistered cells complete",
        "ci_note": "Aggregate summaries were not recomputed after the finalization deadline.",
    }


def run_pilot(
    config_path: str | os.PathLike[str] | Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str],
    study_settings_path: str | os.PathLike[str] | None = None,
    wall_time_seconds: float | None = None,
    resume: bool = True,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Run a resumable, atomically checkpointed C2 pilot under one wall budget."""
    raw = load_json(config_path) if not isinstance(config_path, Mapping) else dict(config_path)
    sidecar = load_json(study_settings_path) if study_settings_path is not None else None
    resolved = resolve_config(raw, sidecar)
    study = resolved["study"]
    limit = float(
        study["pilot_wall_time_seconds"] if wall_time_seconds is None else wall_time_seconds
    )
    if not 0 < limit <= PILOT_WALL_LIMIT_SECONDS:
        raise ValueError(f"pilot wall limit must be in (0, {PILOT_WALL_LIMIT_SECONDS}] seconds")
    # Begin the total budget before creating any run evidence. Resumed time is
    # added below, so evidence creation and finalization share this budget.
    started = float(clock())
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prefix = f"linear-recovery-c2-{resolved['config_hash'][:16]}"
    checkpoint_path = target / f"{prefix}.checkpoint.json"
    report_path = target / f"{prefix}.json"
    manifest_path = target / f"{prefix}.manifest.json"
    evidence = (manifest_path, checkpoint_path, report_path)
    if not resume and any(path.exists() for path in evidence):
        raise FileExistsError("--no-resume refuses to overwrite existing pilot run evidence")

    generator = resolved["generator_factory"]
    configs = _pilot_configurations(generator, int(study["seed"]))
    schedule = pilot_seed_schedule(generator, int(study["seed"]))
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "config_hash": resolved["config_hash"],
        "resolved_config": resolved,
        "source_provenance": _source_provenance(),
        "schedule": schedule,
        "budget": {
            "configs": len(configs),
            "cells_per_config": int(study["pilot_cells_per_config"]),
            "siblings_per_cell": int(study["pilot_siblings_per_cell"]),
            "worlds_total": len(configs)
            * int(study["pilot_cells_per_config"])
            * int(study["pilot_siblings_per_cell"]),
            "wall_time_seconds": limit,
            "allocation": "independent active counts; not balanced by count bands",
            "stop_rules": {
                "wall_time": "one total generation, diagnostic, checkpoint, and finalization budget; partial reports are not prevalence claims",
                "generation_failure_rate": "stop before the next configuration when generation_failures / evaluated_candidates exceeds 0.10",
                "count_stratum_overrun": "report realized and empty bands instead of treating them as balanced",
            },
        },
    }
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise FileExistsError(
            "existing pilot manifest differs; refusing a seed/config change on resume"
        )
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    checkpoint: dict[str, Any] = {
        "manifest_hash": config_hash(manifest),
        "completed": [],
        "worlds": [],
        "cells": [],
        "config_accounting": [],
        "config_failures": [],
        "wall_elapsed_seconds": 0.0,
    }
    if resume and checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        if checkpoint.get("manifest_hash") != config_hash(manifest):
            raise FileExistsError(
                "existing pilot checkpoint differs; refusing a seed/config change on resume"
            )
    consumed = float(checkpoint.get("wall_elapsed_seconds", 0.0))
    # Reserve room for bounded TERM cleanup plus one minimal partial report.  A
    # finite allowance is explicit; it cannot make arbitrary filesystem stalls
    # impossible.
    finalization_reserve = min(
        30.0,
        max(
            _PHASE_TERM_GRACE_SECONDS
            + _PHASE_KILL_JOIN_SECONDS
            + _MINIMAL_FINALIZATION_ALLOWANCE_SECONDS,
            limit * 0.10,
        ),
    )
    hard_deadline = started + max(0.0, limit - consumed)
    # Normal aggregation gets a soft deadline, leaving the explicit finite
    # reserve for checkpoint-only fallback finalization.  End work phases one
    # reserve earlier still; filesystem stalls cannot be bounded by arithmetic.
    finalization_deadline = max(started, hard_deadline - finalization_reserve)
    phase_deadline = max(started, finalization_deadline - finalization_reserve)
    phase_budget = max(0.0, phase_deadline - started)
    deadline = phase_deadline
    partial = consumed >= limit or phase_budget <= 0.0
    stop_reason = "resumed wall budget already exhausted" if partial else None
    cells_since_checkpoint = 0

    def elapsed() -> float:
        # Never clamp an overrun: artifacts expose actual elapsed accounting.
        return consumed + float(clock()) - started

    def persist() -> None:
        checkpoint["wall_elapsed_seconds"] = elapsed()
        _write_json(checkpoint_path, checkpoint)

    def finalization_overdue() -> bool:
        return float(clock()) >= finalization_deadline

    def remaining_hard_budget() -> float:
        return hard_deadline - float(clock())

    def upsert_failure(config_index: int, item: Mapping[str, Any], reason: str) -> None:
        failures = [
            row
            for row in checkpoint.get("config_failures", [])
            if int(row.get("config_index", -1)) != config_index
        ]
        failures.append(
            {
                "config_index": config_index,
                "name": item["name"],
                "seed": item["seed"],
                "reason": reason,
            }
        )
        checkpoint["config_failures"] = failures

    def clear_failure(config_index: int) -> None:
        checkpoint["config_failures"] = [
            row
            for row in checkpoint.get("config_failures", [])
            if int(row.get("config_index", -1)) != config_index
        ]

    interrupted = False
    old_term = None

    def restore_term_handler() -> None:
        if old_term is None:
            return
        import signal

        signal.signal(signal.SIGTERM, old_term)

    # Install TERM handling before entering the one protected region.  The
    # outer finally below owns restoration for every exit, including writes in
    # exception and fallback handling.
    try:
        import signal

        old_term = signal.getsignal(signal.SIGTERM)

        def on_term(_signum: int, _frame: Any) -> None:
            raise _PilotInterruptedError("pilot received SIGTERM")

        signal.signal(signal.SIGTERM, on_term)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "pilot SIGTERM handling requires execution in the main thread on this host"
        ) from exc

    try:
        completed = {
            (int(row["config_index"]), int(row["cell_id"]))
            for row in checkpoint.get("cells", [])
            if row.get("status") == "complete"
        }
        for config_index, item in enumerate(configs):
            if partial:
                break
            pending_cells = set(range(int(study["pilot_cells_per_config"]))) - {
                cell for index, cell in completed if index == config_index
            }
            if not pending_cells:
                continue
            if float(clock()) >= deadline:
                partial, stop_reason = True, "wall deadline reached before configuration"
                break
            config_generator = dict(item["generator"])
            config_study = _study_for_generator(
                config_generator,
                {
                    **study,
                    "n_cells": int(study["pilot_cells_per_config"]),
                    "draws_per_cell": int(study["pilot_siblings_per_cell"]),
                    "seed": int(item["seed"]),
                },
            )
            prior = make_prior(config_generator, config_study)
            # Prior construction/imports may initialize native BLAS workers;
            # verify again before entering the forked phase boundary.
            _require_single_os_thread()
            try:
                corpus = _bounded_phase(
                    lambda prior=prior: sample_prior_predictive(prior), deadline, clock
                )
            except (_PilotDeadlineExceededError, _PilotInterruptedError) as exc:
                partial, stop_reason = True, str(exc)
                interrupted = isinstance(exc, _PilotInterruptedError)
                persist()
                break
            except Exception as exc:
                upsert_failure(config_index, item, f"{type(exc).__name__}: {exc}")
                partial, stop_reason = True, f"generation failed for {item['name']}"
                persist()
                break
            clear_failure(config_index)
            accounting = count_accounting(corpus, prior, config_study)
            evaluated = max(int(accounting["evaluated_candidates"]), 1)
            failure_rate = int(accounting["generation_failures"]) / evaluated
            checkpoint["config_accounting"] = [
                row
                for row in checkpoint.get("config_accounting", [])
                if int(row["config_index"]) != config_index
            ] + [{"config_index": config_index, "name": item["name"], **accounting}]
            persist()
            if failure_rate > 0.10:
                partial, stop_reason = (
                    True,
                    f"generation failure rate {failure_rate:.3f} exceeded 0.10 for {item['name']}",
                )
                persist()
                break
            try:
                diagnostic = _bounded_phase(
                    lambda corpus=corpus: data_diagnostics(
                        corpus, views=tuple(study["diagnostic_views"]), keep_series=False
                    ),
                    deadline,
                    clock,
                )
            except (_PilotDeadlineExceededError, _PilotInterruptedError) as exc:
                partial, stop_reason = True, str(exc)
                interrupted = isinstance(exc, _PilotInterruptedError)
                persist()
                break
            cell_ids = np.asarray(corpus["cell_id"], dtype=int)
            for cell_id in sorted(pending_cells):
                if float(clock()) >= deadline:
                    partial, stop_reason = True, "wall deadline reached during metric finalization"
                    break
                rows = np.flatnonzero(cell_ids == cell_id)
                for sibling_index, row in enumerate(rows):
                    key = (config_index, int(cell_id), int(sibling_index))
                    if any(
                        (
                            int(existing.get("config_index", -1)),
                            int(existing.get("cell_id", -1)),
                            int(existing.get("sibling_index", -1)),
                        )
                        == key
                        for existing in checkpoint["worlds"]
                    ):
                        continue
                    checkpoint["worlds"].append(
                        _compact_world_metric(
                            corpus,
                            prior,
                            diagnostic,
                            int(row),
                            config_index=config_index,
                            cell_index=int(cell_id),
                            sibling_index=sibling_index,
                            rank_tolerance=float(study["rank_tolerance"]),
                        )
                    )
                records = [
                    row
                    for row in checkpoint["worlds"]
                    if int(row["config_index"]) == config_index and int(row["cell_id"]) == cell_id
                ]
                checkpoint["cells"] = [
                    row
                    for row in checkpoint["cells"]
                    if not (
                        int(row["config_index"]) == config_index and int(row["cell_id"]) == cell_id
                    )
                ] + [
                    {
                        "config_index": config_index,
                        "cell_id": int(cell_id),
                        "n_worlds": len(records),
                        "status": "complete"
                        if len(records) == int(study["pilot_siblings_per_cell"])
                        else "partial",
                    }
                ]
                if len(records) == int(study["pilot_siblings_per_cell"]):
                    completed.add((config_index, int(cell_id)))
                cells_since_checkpoint += 1
                if cells_since_checkpoint >= int(study["pilot_checkpoint_every_cells"]):
                    persist()
                    cells_since_checkpoint = 0
            persist()
            if partial:
                break

        # Persist pending cell changes before finalization.  A TERM raised by
        # this write is caught by the single outer handler below.
        if cells_since_checkpoint:
            persist()

        # Build the checkpoint-only candidate before normal aggregation.  This
        # keeps fallback construction inside the reserved interval even if a
        # slow summary overruns the soft deadline; it uses refs/counts only.
        fallback_report = _minimal_partial_report(
            checkpoint,
            configs,
            study,
            resolved,
            manifest,
            schedule,
            manifest_path,
            checkpoint_path,
            report_path,
            interrupted,
            stop_reason,
            elapsed,
            limit,
        )
        report = _build_pilot_report(
            checkpoint,
            configs,
            study,
            resolved,
            manifest,
            schedule,
            manifest_path,
            checkpoint_path,
            report_path,
            partial,
            interrupted,
            stop_reason,
            elapsed,
            finalization_overdue,
            limit,
        )
        if finalization_overdue():
            raise _PilotDeadlineExceededError("wall deadline reached during finalization")
        _write_json(report_path, report)
        # Account for persistence work explicitly; elapsed() is never clamped.
        report["accounting"]["wall_elapsed_seconds"] = elapsed()
        checkpoint["status"] = report["status"]
        if finalization_overdue():
            raise _PilotDeadlineExceededError("wall deadline reached during finalization")
        persist()
        report["accounting"]["wall_elapsed_seconds"] = elapsed()
        if finalization_overdue():
            raise _PilotDeadlineExceededError("wall deadline reached during finalization")
        _write_json(report_path, report)
    except (_PilotInterruptedError, _PilotDeadlineExceededError) as exc:
        partial = True
        interrupted = interrupted or isinstance(exc, _PilotInterruptedError)
        stop_reason = str(exc)
        # The fallback candidate was built from checkpointed refs/counts before
        # normal aggregation.  If interruption occurred earlier, build it now.
        try:
            report = fallback_report
        except UnboundLocalError:
            report = _minimal_partial_report(
                checkpoint,
                configs,
                study,
                resolved,
                manifest,
                schedule,
                manifest_path,
                checkpoint_path,
                report_path,
                interrupted,
                stop_reason,
                elapsed,
                limit,
            )
        report["stop_reason"] = stop_reason
        report["interrupted"] = interrupted
        checkpoint["status"] = "partial"
        # The finite reserve cannot make arbitrary filesystem/OS stalls
        # impossible.  Check remaining hard budget before writing; once an
        # atomic write has begun, its filesystem latency is inherent.  The
        # candidate was prepared before this check, so no fallback scan starts
        # after the hard deadline.
        hard_budget_remaining = remaining_hard_budget()
        if hard_budget_remaining <= 0.0:
            report["stop_reason"] = stop_reason
        try:
            _write_json(report_path, report)
        except _PilotInterruptedError:
            interrupted = True
            report["interrupted"] = True
            _write_json(report_path, report)
        try:
            persist()
        except _PilotInterruptedError:
            interrupted = True
            report["interrupted"] = True
            # Keep TERM handling installed; the outer finally restores it.
            persist()
        report["accounting"]["wall_elapsed_seconds"] = elapsed()
        try:
            _write_json(report_path, report)
        except _PilotInterruptedError:
            interrupted = True
            report["interrupted"] = True
            _write_json(report_path, report)
    except BaseException:
        if cells_since_checkpoint:
            persist()
        raise
    finally:
        restore_term_handler()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON generator configuration")
    parser.add_argument("--study-settings", help="JSON sidecar with study settings")
    parser.add_argument(
        "--output-dir", required=True, help="directory for the config-specific report"
    )
    parser.add_argument("--mode", choices=("calibration", "pilot"), default="calibration")
    parser.add_argument("--wall-time-seconds", type=float, help="pilot cap, at most 5400 seconds")
    parser.add_argument(
        "--no-resume", action="store_true", help="do not read an existing pilot checkpoint"
    )
    args = parser.parse_args(argv)
    runner = run_pilot if args.mode == "pilot" else run_calibration
    kwargs = {"output_dir": args.output_dir, "study_settings_path": args.study_settings}
    if args.mode == "pilot":
        kwargs.update(wall_time_seconds=args.wall_time_seconds, resume=not args.no_resume)
    report = runner(args.config, **kwargs)
    print(
        json.dumps(
            {
                "report": report["report_path"],
                "status": report.get("status", "calibration"),
                "config_hash": report["config_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
