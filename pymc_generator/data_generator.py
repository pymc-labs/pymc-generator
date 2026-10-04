"""High-level generation facade + corpus persistence.

Provides a clean, tested interface for generating structural causal model
worlds at scale, plus the ``.npz`` persistence format that stacks many worlds
into one padded tensor archive.

Design Principles:
1. Model-independent - generates data without any learned model
2. Reproducible - same seed + config produces an identical corpus
3. Validated - generated corpora are checked for schema correctness
4. Modular - generate in batches or all at once

Usage:
    from pymc_generator import DataGenerator, make_scm_prior

    cfg = make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_cells=10, draws_per_cell=10)
    generator = DataGenerator(cfg)
    corpus = generator.generate(n_tasks=100, seed=42)

    # Or generate in batches
    for batch in generator.iter_batches(n_tasks=1000, batch_size=100, seed=42):
        process(batch)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np

from .active_counts import active_count_coverage_errors
from .sampler import (
    CORPUS_STORAGE_MAX,
    OUTCOME_NOISE_SEMANTICS,
    OUTCOME_NOISE_VERSION,
    SATURATION_FAMILY_KEYS,
    SCMPrior,
    sample_prior_predictive,
)
from .signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from .slots import (
    CORPUS_ARRAY_FIELDS,
    CORPUS_SCHEMA_VERSION,
    EDGE_TYPES_EXTENDED,
    LEGACY_CORPUS_KEYS_V1,
    LEGACY_EDGE_KEYS_V2,
    MECHANISM_ARRAY_FIELDS,
    MECHANISM_PRIOR_FIELDS,
    MECHANISM_REFERENCE_FIELDS,
    PRIOR_COND_LAYOUT,
    PRIOR_COND_QUANTITIES,
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_INPUTS,
    TRAJECTORY_PARAM_FIELDS,
    SlotLayout,
)
from .trajectories import GATE_COMPONENTS, LEVEL_COMPONENTS, summarize_component_prevalence

# ---------------------------------------------------------------------------
# Schema-version and timing guards (shared by validate / save / load)
# ---------------------------------------------------------------------------


def _schema_version_error(diagnostics: object) -> str | None:
    """Why ``diagnostics`` fails the corpus schema stamp, or ``None`` if it passes.

    The stamp is the whole point of versioning the format: a consumer that
    silently accepts an unknown version reads a *different* schema through the
    current vocabulary, which is exactly the failure mode a version field is
    supposed to prevent. Version-less and pre-v5 archives cannot recover realised
    rich truth, so they are rejected rather than stamped or migrated. ``True``
    is an ``int`` in Python, so a bool is refused explicitly rather than compared
    numerically.

    All three entry points share this so they cannot drift: ``validate_corpus``
    reports the string, ``save_corpus`` and ``load_corpus`` raise it.
    """
    if not isinstance(diagnostics, dict):
        return "diagnostics must be a mapping carrying schema_version"
    if "schema_version" not in diagnostics:
        return (
            f"diagnostics is missing schema_version; this corpus must be stamped "
            f"schema_version={CORPUS_SCHEMA_VERSION}"
        )
    version = diagnostics["schema_version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        return (
            f"diagnostics schema_version must be a non-bool integer, got "
            f"{type(version).__name__} {version!r}"
        )
    if int(version) != CORPUS_SCHEMA_VERSION:
        return (
            f"corpus schema_version {int(version)} is not supported; this build reads "
            f"schema_version={CORPUS_SCHEMA_VERSION}"
        )
    for field in ("edge_types", "edge_base_rates", "edge_marginals", "edge_budget"):
        value = diagnostics.get(field)
        if isinstance(value, (dict, list, tuple)) or (
            isinstance(value, np.ndarray) and value.ndim == 1
        ):
            if any(isinstance(key, str) and key in LEGACY_EDGE_KEYS_V2 for key in value):
                return f"diagnostics {field} uses pre-v3 outcome-edge names"
    if "min_dead_channels" in diagnostics:
        return "diagnostics uses the pre-v3 min_dead_channels name"
    return None


def _timing_error(diagnostics: dict) -> str | None:
    """Why ``diagnostics["timing"]`` is malformed, or ``None`` if absent or well-formed.

    ``timing`` is the wall-clock block a freshly generated corpus carries and
    ``save_corpus`` strips (it is the only nondeterministic diagnostic, so
    persisting it would break byte-identical same-seed shards). Both states are
    therefore legal; only a present-but-wrong block is an error.
    """
    if "timing" not in diagnostics:
        return None
    timing = diagnostics["timing"]
    if not isinstance(timing, dict):
        return "diagnostics timing must be a mapping"
    for key in ("elapsed_s", "tasks_per_sec"):
        value = timing.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return f"diagnostics timing {key} must be a real number"
    return None


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


def _is_finite_real(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, ValueError):
        return False


def _finite_bounds(
    value: object, *, positive: bool = False, maximum: float | None = None
) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo, hi = value
    if not _is_finite_real(lo) or not _is_finite_real(hi):
        return None
    lo, hi = float(lo), float(hi)
    if (
        lo > hi
        or (positive and lo <= 0.0)
        or (maximum is not None and max(abs(lo), abs(hi)) > maximum)
    ):
        return None
    return lo, hi


def _field_inventory_matches(value: object, fields: dict) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(key, str) for key in value)
        and value == list(fields)
    )


def _mechanism_field_specs(
    corpus: dict[str, Any], errors: list[str]
) -> dict[str, tuple[tuple[str, ...], type[np.generic]]]:
    """Resolve the recipe-gated rich mechanism block without inventing fields."""
    diagnostics = corpus.get("diagnostics")
    metadata = diagnostics.get("mechanism_priors") if isinstance(diagnostics, dict) else None
    has_metadata = isinstance(diagnostics, dict) and "mechanism_priors" in diagnostics
    possible_fields = set(MECHANISM_ARRAY_FIELDS) | set(MECHANISM_REFERENCE_FIELDS)
    if not has_metadata and not possible_fields.intersection(corpus):
        return {}
    if not isinstance(metadata, dict):
        errors.append("mechanism arrays and diagnostics mechanism_priors must be present together")
        return {}
    metadata_keys = {
        "parameter_fields",
        "saturation_prior_ranges",
        "mm_scale_prior",
        "treatment_reference_contribution_range",
        "treatment_reference_multiplier",
        "covariate_reference_contribution_range",
        "covariate_reference_scale",
    }
    if set(metadata) != metadata_keys:
        errors.append("diagnostics mechanism_priors does not match the required recipe fields")
    fields = dict(MECHANISM_ARRAY_FIELDS)
    for input_type in TRAJECTORY_INPUTS:
        if metadata.get(f"{input_type}_reference_contribution_range") is not None:
            fields.update(
                {
                    key: spec
                    for key, spec in MECHANISM_REFERENCE_FIELDS.items()
                    if key.startswith(f"{input_type}_")
                }
            )
    missing = [key for key in fields if key not in corpus]
    if missing:
        errors.append(
            "mechanism arrays and diagnostics mechanism_priors must be present together; "
            f"missing {', '.join(missing)}"
        )
    if not _field_inventory_matches(metadata.get("parameter_fields"), fields):
        errors.append("diagnostics mechanism_priors parameter_fields does not match the schema")
    return fields


def _trajectory_truth_errors(corpus: dict[str, Any]) -> list[str]:
    """Check selected primitives and reconstruct schedules from their stored truth."""
    errors: list[str] = []
    n_time_steps = corpus["treatment_raw"].shape[1]
    weeks = np.arange(n_time_steps)
    for input_type in TRAJECTORY_INPUTS:
        active = corpus[f"{input_type}_active_mask"] == 1
        flags = corpus[f"{input_type}_components"]
        prefix = f"trajectory_{input_type}_"

        def carries(
            component: str, active: np.ndarray = active, flags: np.ndarray = flags
        ) -> np.ndarray:
            return np.asarray(active & (flags[..., TRAJECTORY_COMPONENTS.index(component)] == 1))

        def leaf(component: str, field: str, prefix=prefix) -> np.ndarray:
            return np.asarray(corpus[f"{prefix}{component}_{field}"])

        for key, (axes, _) in TRAJECTORY_PARAM_FIELDS.items():
            if not key.startswith(prefix):
                continue
            component_leaf = key.removeprefix(prefix)
            component = next(
                name for name in TRAJECTORY_COMPONENTS if component_leaf.startswith(f"{name}_")
            )
            selected = carries(component)
            if len(axes) == 3:
                selected = np.broadcast_to(selected[:, None, :], corpus[key].shape)
            value = corpus[key]
            if (value[~selected] != 0).any():
                errors.append(f"{key} has nonzero unselected-component or inactive-input padding")
            live = value[selected]
            field = component_leaf.removeprefix(f"{component}_")
            if component in ("onset", "offset") or (component == "level_jump" and field == "week"):
                if ((live < 1) | (live >= n_time_steps)).any():
                    errors.append(f"{key} must lie in [1, n_time_steps) for selected inputs")
            elif component == "flighting" and field == "period":
                if ((live < 2) | (live > n_time_steps)).any():
                    errors.append(f"{key} must lie in [2, n_time_steps] for selected inputs")
            elif (component == "seasonal" and field == "period") or field == "factor":
                minimum = 2.0 if component == "seasonal" else 0.0
                if (live <= minimum).any():
                    errors.append(f"{key} must be > {minimum:g} for selected inputs")
            elif field in ("sigma", "amp", "prob", "amplitude"):
                if (live < 0.0).any() or (field == "prob" and (live > 0.5).any()):
                    errors.append(f"{key} is outside its live nonnegative domain")
            elif component == "seasonal" and field == "phase":
                if ((live < 0.0) | (live >= 2.0 * np.pi)).any():
                    errors.append(f"{key} must lie in [0, 2*pi) for selected inputs")
            unscaled_primitive = (
                (component == "seasonal" and field in ("amplitude", "period"))
                or (component == "trend" and field == "change")
                or (input_type == "covariate" and component == "level_jump" and field == "size")
            )
            if unscaled_primitive and (np.abs(live) > CORPUS_STORAGE_MAX).any():
                errors.append(f"{key} magnitude exceeds CORPUS_STORAGE_MAX for selected inputs")
        selected = carries("flighting")
        period = leaf("flighting", "period")
        on_weeks = leaf("flighting", "on_weeks")
        phase = leaf("flighting", "phase")
        if (selected & ((on_weeks < 1) | (on_weeks >= period))).any():
            errors.append(f"{prefix}flighting_on_weeks must lie in [1, period) for selected inputs")
        if (selected & ((phase < 0) | (phase >= period))).any():
            errors.append(f"{prefix}flighting_phase must lie in [0, period) for selected inputs")
        selected = carries("level_jump")
        jump_week = leaf("level_jump", "week")
        count = jump_week.shape[1]
        if selected.any():
            edges = 1 + np.arange(count + 1) * (n_time_steps - 1) // count
            valid = (jump_week >= edges[:-1][None, :, None]) & (
                jump_week < edges[1:][None, :, None]
            )
            if not valid[np.broadcast_to(selected[:, None, :], valid.shape)].all():
                errors.append(f"{prefix}level_jump_week does not match chronological jump slots")
        if input_type == "treatment" and selected.any():
            mask = np.broadcast_to(selected[:, None, :], jump_week.shape)
            factor = leaf("level_jump", "factor")[mask]
            log_factor = leaf("level_jump", "log_factor")[mask]
            if (factor > 0.0).all() and not np.allclose(
                log_factor,
                np.log(factor),
                rtol=8 * np.finfo(np.float64).eps,
                atol=8 * np.finfo(np.float64).eps,
            ):
                errors.append(f"{prefix}level_jump_factor and log_factor are inconsistent")
        if errors:
            continue

        expected_activity = np.broadcast_to(
            active[:, None, :], corpus[f"{input_type}_activity"].shape
        ).copy()
        expected_activity &= ~carries("onset")[:, None, :] | (
            weeks[None, :, None] >= leaf("onset", "start")[:, None, :]
        )
        expected_activity &= ~carries("offset")[:, None, :] | (
            weeks[None, :, None] < leaf("offset", "stop")[:, None, :]
        )
        cycle = (weeks[None, :, None] + phase[:, None, :]) % np.where(
            carries("flighting"), period, 1
        )[:, None, :]
        expected_activity &= ~carries("flighting")[:, None, :] | (cycle < on_weeks[:, None, :])
        if not np.array_equal(corpus[f"{input_type}_activity"], expected_activity):
            errors.append(f"{input_type}_activity does not match realised trajectory parameters")

        seasonal = leaf("seasonal", "amplitude")[:, None, :] * np.sin(
            2.0
            * np.pi
            * weeks[None, :, None]
            / np.where(carries("seasonal"), leaf("seasonal", "period"), 1)[:, None, :]
            + leaf("seasonal", "phase")[:, None, :]
        )
        trend = leaf("trend", "change")[:, None, :] * (
            weeks[None, :, None] / float(n_time_steps - 1)
        )
        jump_size = leaf("level_jump", "log_factor" if input_type == "treatment" else "size")
        jumps = np.einsum(
            "ntki,nki->nti",
            weeks[None, :, None, None] >= jump_week[:, None, :, :],
            jump_size,
        )
        expected_shift = (seasonal + trend + jumps).astype(np.float32)
        shift_key = (
            "treatment_log_level_shift" if input_type == "treatment" else "covariate_level_shift"
        )
        if not np.allclose(
            corpus[shift_key], expected_shift, rtol=8 * np.finfo(np.float32).eps, atol=2e-7
        ):
            errors.append(f"{shift_key} does not match realised trajectory parameters")
    return errors


def _reference_truth_matches(actual: np.ndarray, expected: np.ndarray, target: np.ndarray) -> bool:
    """Match admissible derived loadings without erasing nonzero targets."""
    with np.errstate(over="ignore", invalid="ignore"):
        ulp = np.abs(expected - np.nextafter(expected, 0.0))
        tolerance = np.maximum(32 * np.finfo(np.float64).eps * np.abs(expected), 2 * ulp)
        live_match = (
            (np.abs(actual) <= CORPUS_STORAGE_MAX)
            & np.isfinite(expected)
            & (actual != 0.0)
            & (np.signbit(actual) == np.signbit(expected))
            & (np.abs(actual - expected) <= tolerance)
        )
    return bool(np.where(target == 0.0, (expected == 0.0) & (actual == 0.0), live_match).all())


def _mechanism_truth_errors(corpus: dict[str, Any], fields: dict) -> list[str]:
    errors: list[str] = []
    metadata = corpus["diagnostics"]["mechanism_priors"]
    active = {
        input_type: corpus[f"{input_type}_active_mask"] == 1 for input_type in TRAJECTORY_INPUTS
    }
    for key, (axes, _) in fields.items():
        mask = active[axes[-1]]
        if (corpus[key][~mask] != 0).any():
            errors.append(f"{key} has nonzero inactive-{axes[-1]} padding")
        if (
            key in MECHANISM_PRIOR_FIELDS
            or key.endswith("_reference_input")
            or key == "treatment_reference_response"
            or key == "mechanism_saturation_scale"
        ) and (corpus[key][mask] <= 0.0).any():
            errors.append(f"{key} must be positive for active {axes[-1]}s")
        if (
            key == "treatment_reference_input"
            and (corpus[key][mask] < np.finfo(np.float64).tiny).any()
        ):
            errors.append(f"{key} must be normal positive float64 for active treatments")
        if (key in MECHANISM_PRIOR_FIELDS or key in ("beta", "rho_zy")) and (
            np.abs(corpus[key][mask]) > CORPUS_STORAGE_MAX
        ).any():
            errors.append(
                f"{key} exceeds the configured primitive maximum ({CORPUS_STORAGE_MAX:.9g})"
            )
    if (corpus["beta"][active["treatment"]] < 0.0).any():
        errors.append("beta must be nonnegative for active treatments")
    if (corpus["root_alpha"][active["treatment"]] > 1.0).any():
        errors.append("root_alpha must be <= 1 for active treatments")
    if (corpus["sat_family"][active["treatment"]] >= len(SATURATION_FAMILY_KEYS)).any():
        errors.append("sat_family contains an unsupported saturation family")
    if not np.array_equal(
        corpus["saturation_scale"], corpus["mechanism_saturation_scale"].astype(np.float32)
    ):
        errors.append("saturation_scale does not match the exact mechanism_saturation_scale")
    mode = metadata.get("mm_scale_prior")
    if not isinstance(mode, str) or mode not in ("uniform", "log_uniform"):
        errors.append("diagnostics mechanism_priors mm_scale_prior is not supported")
    ranges = metadata.get("saturation_prior_ranges")
    expected_ranges: dict[str, set[str]] = {}
    for family, parameter in MECHANISM_PRIOR_FIELDS.values():
        expected_ranges.setdefault(family, set()).add(parameter)
    if (
        not isinstance(ranges, dict)
        or set(ranges) != set(expected_ranges)
        or any(
            not isinstance(ranges[family], dict) or set(ranges[family]) != parameters
            for family, parameters in expected_ranges.items()
        )
    ):
        errors.append(
            "diagnostics mechanism_priors saturation_prior_ranges does not match the schema"
        )
    else:
        for key, (family, parameter) in MECHANISM_PRIOR_FIELDS.items():
            bounds = _finite_bounds(
                ranges[family][parameter],
                positive=True,
                maximum=1.0 if family == "root" else CORPUS_STORAGE_MAX,
            )
            if bounds is None:
                errors.append(
                    f"diagnostics mechanism_priors {family}.{parameter} has invalid bounds"
                )
            else:
                values = corpus[key][active["treatment"]]
                if ((values < bounds[0]) | (values > bounds[1])).any():
                    errors.append(f"{key} is outside its declared prior support")
                if family == "michaelis_menten":
                    if bounds[0] * 1e-8 == 0.0:
                        errors.append(
                            f"diagnostics mechanism_priors {family}.{parameter} lower bound "
                            "times the minimum saturation_scale=1e-8 must be positive in float64"
                        )
                    if (
                        mode == "log_uniform"
                        and bounds[0] < bounds[1]
                        and np.log(bounds[0]) >= np.log(bounds[1])
                    ):
                        errors.append(
                            f"diagnostics mechanism_priors {family}.{parameter} bounds "
                            "collapse in log space under mm_scale_prior='log_uniform'"
                        )
    for input_type, setting in (
        ("treatment", "treatment_reference_multiplier"),
        ("covariate", "covariate_reference_scale"),
    ):
        value = metadata.get(setting)
        if not _is_finite_real(value) or not 0.0 < value <= CORPUS_STORAGE_MAX:
            errors.append(
                f"diagnostics mechanism_priors {setting} must be finite, positive, "
                f"and at most {CORPUS_STORAGE_MAX:.9g}"
            )
        elif (
            input_type == "treatment"
            and metadata.get("treatment_reference_contribution_range") is not None
            and float(value) * 1e-8 < np.finfo(np.float64).tiny
        ):
            errors.append(
                "diagnostics mechanism_priors treatment_reference_multiplier * the minimum "
                "saturation_scale=1e-8 must produce a normal positive "
                "treatment_reference_input in float64"
            )
        recipe = metadata.get(f"{input_type}_reference_contribution_range")
        if recipe is None:
            continue
        bounds = _finite_bounds(recipe, maximum=CORPUS_STORAGE_MAX)
        if bounds is None or (input_type == "treatment" and (bounds[0] < 0.0 or bounds[1] <= 0.0)):
            errors.append(
                f"diagnostics mechanism_priors {input_type} reference range has invalid bounds"
            )
        else:
            target = corpus[f"{input_type}_reference_contribution"][active[input_type]]
            if ((target < bounds[0]) | (target > bounds[1])).any():
                errors.append(
                    f"{input_type}_reference_contribution is outside its declared prior support"
                )
    if errors:
        return errors
    for input_type in TRAJECTORY_INPUTS:
        if f"{input_type}_reference_input" not in fields:
            continue
        mask = active[input_type]
        reference = corpus[f"{input_type}_reference_input"][mask]
        target = corpus[f"{input_type}_reference_contribution"][mask]
        if input_type == "covariate":
            scale = float(metadata["covariate_reference_scale"])
            if not np.array_equal(reference, np.full_like(reference, scale)):
                errors.append("covariate_reference_input does not match its recipe scale")
            with np.errstate(over="ignore", invalid="ignore"):
                expected = target * (1.0 / scale)
            if not _reference_truth_matches(corpus["rho_zy"][mask], expected, target):
                errors.append("rho_zy does not match the covariate reference target")
        else:
            multiplier = float(metadata["treatment_reference_multiplier"])
            if (
                corpus["sat_family"][mask] == SATURATION_FAMILY_KEYS.index("michaelis_menten")
            ).any():
                mm_hi = float(
                    metadata["saturation_prior_ranges"]["michaelis_menten"]["kappa_mult"][1]
                )
                if multiplier / (multiplier + mm_hi) < np.finfo(np.float64).tiny:
                    errors.append(
                        "diagnostics mechanism_priors michaelis_menten.kappa_mult with "
                        "treatment_reference_multiplier requires a normal positive "
                        "minimum reference response for represented MM treatments"
                    )
            scale = corpus["mechanism_saturation_scale"][mask]
            expected = multiplier * scale
            if not np.array_equal(reference, expected):
                errors.append(
                    "treatment_reference_input does not match its recipe multiplier and anchor"
                )
            response = corpus["treatment_reference_response"][mask]
            with np.errstate(over="ignore", invalid="ignore"):
                expected_coefficient = target / response
            if not _reference_truth_matches(corpus["beta"][mask], expected_coefficient, target):
                errors.append("beta does not match the treatment reference target")
    return errors


@dataclass
class DataGenerator:
    """Independent data generator for synthetic MMM data.

    Parameters
    ----------
    config : SCMPrior
        Corpus generation configuration.
    """

    config: SCMPrior

    def generate(
        self,
        n_tasks: int | None = None,
        seed: int | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Generate a corpus of synthetic MMM tasks.

        Parameters
        ----------
        n_tasks : int, optional
            Number of tasks to generate. If None, uses config.n_cells * config.draws_per_cell.
        seed : int, optional
            Random seed. If None, uses config.seed.
        validate : bool
            Whether to validate the generated corpus.

        Returns
        -------
        dict
            Corpus dictionary with all generated data.
        """
        if n_tasks is not None and n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")

        cfg = dataclasses.replace(self.config, seed=self.config.seed if seed is None else seed)

        # sample_prior_predictive truncates before deriving retained-corpus
        # diagnostics and signal labels. Do not independently slice a finalized
        # corpus here, or those task-level summaries would become stale.
        corpus = sample_prior_predictive(cfg, n=n_tasks)

        # Validate if requested
        if validate:
            errors = self.validate_corpus(corpus)
            if errors:
                raise ValueError(
                    "Corpus validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )

        return corpus

    def iter_batches(
        self,
        n_tasks: int,
        batch_size: int = 100,
        seed: int | None = None,
        validate: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield batches of at least two worlds without retaining earlier batches.

        Every corpus carries its own cell-level train/validation split, so a
        batch cannot be a single world.

        Parameters
        ----------
        n_tasks : int
            Total number of tasks to generate.
        batch_size : int
            Target tasks per batch; a final single world joins the previous batch.
        seed : int, optional
            Base seed, defaulting to config.seed. Batch i uses base_seed + i.
        validate : bool
            Whether to validate each batch.

        Returns
        -------
        Iterator of dict
            Lazy corpus iterator. Retain batches explicitly only when needed.
        """
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")
        if n_tasks == 1:
            raise ValueError("n=1 cannot form a cell-level train/validation split; n must be >= 2")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size == 1:
            raise ValueError(
                "batch_size must be at least 2 for a cell-level train/validation split"
            )

        n_batches = (n_tasks + batch_size - 1) // batch_size
        if n_tasks % batch_size == 1 and n_batches > 1:
            n_batches -= 1
        base_seed = self.config.seed if seed is None else seed
        return (
            self.generate(
                n_tasks=batch_size if i < n_batches - 1 else n_tasks - i * batch_size,
                seed=base_seed + i,
                validate=validate,
            )
            for i in range(n_batches)
        )

    def generate_and_save(
        self,
        path: str | Path,
        n_tasks: int | None = None,
        seed: int | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Generate corpus and save to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the corpus (.npz file).
        n_tasks : int, optional
            Number of tasks.
        seed : int, optional
            Random seed.
        validate : bool
            Whether to validate.

        Returns
        -------
        dict
            Generated corpus.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        corpus = self.generate(n_tasks=n_tasks, seed=seed, validate=validate)

        # Save
        save_corpus(corpus, path)

        return corpus

    @staticmethod
    def validate_corpus(corpus: dict[str, Any]) -> list[str]:
        """Validate a generated corpus.

        Parameters
        ----------
        corpus : dict
            Corpus dictionary.

        Returns
        -------
        list of str
            List of validation errors. Empty if valid.
        """
        errors = []

        for key in CORPUS_ARRAY_FIELDS:
            if key not in corpus:
                errors.append(f"Missing required key: {key}")

        # Rich arrays and their recipe metadata are atomic optional blocks.
        diagnostics = corpus.get("diagnostics")
        trajectory_parts = {key: key in corpus for key in TRAJECTORY_ARRAY_FIELDS}
        trajectory_parts["diagnostics trajectory"] = (
            isinstance(diagnostics, dict) and "trajectory" in diagnostics
        )
        has_trajectory = all(trajectory_parts.values())
        if any(trajectory_parts.values()) and not has_trajectory:
            missing = ", ".join(part for part, present in trajectory_parts.items() if not present)
            errors.append(
                "trajectory arrays and diagnostics trajectory must be present together; "
                f"missing {missing}"
            )
        mechanism_fields = _mechanism_field_specs(corpus, errors)
        jump_counts: dict[str, int] = {}
        if has_trajectory and isinstance(diagnostics, dict):
            trajectory_diagnostics = diagnostics["trajectory"]
            if not isinstance(trajectory_diagnostics, dict):
                errors.append("diagnostics trajectory must be a mapping")
            else:
                if not _field_inventory_matches(
                    trajectory_diagnostics.get("parameter_fields"), TRAJECTORY_PARAM_FIELDS
                ):
                    errors.append(
                        "diagnostics trajectory parameter_fields does not match the schema"
                    )
                counts = trajectory_diagnostics.get("jump_counts")
                if not isinstance(counts, dict) or set(counts) != set(TRAJECTORY_INPUTS):
                    errors.append("diagnostics trajectory jump_counts must map both input roles")
                else:
                    for input_type, count in counts.items():
                        if (
                            isinstance(count, (bool, np.bool_))
                            or not isinstance(count, (int, np.integer))
                            or count < 1
                        ):
                            errors.append(
                                f"diagnostics trajectory jump_counts {input_type} must be an integer >= 1"
                            )
                        else:
                            jump_counts[input_type] = int(count)

        signal_label_keys = ("signal_metrics", "signal_metric_valid")
        if any(key in corpus for key in signal_label_keys):
            errors.append("signal labels must live under the identifiability metadata block")
        identifiability: Any = corpus.get("identifiability")
        if "identifiability" in corpus and not isinstance(identifiability, dict):
            errors.append("identifiability must be a mapping")
        has_signal_labels = isinstance(identifiability, dict) and all(
            key in identifiability for key in signal_label_keys
        )
        if isinstance(identifiability, dict) and not has_signal_labels:
            errors.append(
                "identifiability signal_metrics and signal_metric_valid must be present together"
            )

        if errors:
            return errors

        for key, value in corpus.items():
            if key in {"diagnostics", "identifiability"}:
                continue
            if not isinstance(value, np.ndarray):
                errors.append(f"{key} must be an ndarray")
        if isinstance(identifiability, dict):
            for key, value in identifiability.items():
                if not isinstance(value, np.ndarray):
                    errors.append(f"identifiability.{key} must be an ndarray")
        if errors:
            return errors

        for key in (
            "treatment_raw",
            "covariates",
            "latent_unobserved",
            "g",
            "treatment_shock_index",
        ):
            ndim = len(CORPUS_ARRAY_FIELDS[key][0])
            value = corpus[key]
            if value.ndim != ndim:
                actual_ndim = value.ndim
                errors.append(f"{key} must have ndim={ndim}, got {actual_ndim}")
        if errors:
            return errors

        # Get dimensions
        n_tasks = corpus["treatment_raw"].shape[0]
        n_time_steps = corpus["treatment_raw"].shape[1]
        n_treatments = corpus["treatment_raw"].shape[2]
        n_covariates = corpus["covariates"].shape[2]
        n_latent = corpus["latent_unobserved"].shape[2]
        layout = SlotLayout(
            n_treatments=n_treatments,
            n_covariates=n_covariates,
            n_latent=n_latent,
            edge_types=EDGE_TYPES_EXTENDED,
        )
        n_treatment_shocks = corpus["treatment_shock_index"].shape[1]

        dimensions = {
            "task": n_tasks,
            "time": n_time_steps,
            "treatment": n_treatments,
            "covariate": n_covariates,
            "latent": n_latent,
            "edge": layout.n_slots,
            "shock": n_treatment_shocks,
            "indirect_source": 3,
            "component": len(TRAJECTORY_COMPONENTS),
        }
        for input_type, count in jump_counts.items():
            dimensions[f"{input_type}_jump"] = count
        array_fields = dict(CORPUS_ARRAY_FIELDS)
        if has_trajectory:
            array_fields.update(TRAJECTORY_ARRAY_FIELDS)
        array_fields.update(mechanism_fields)
        field_specs = {
            key: (tuple(dimensions[axis] for axis in axes), dtype)
            for key, (axes, dtype) in array_fields.items()
        }
        if "prior_cond" in corpus:
            field_specs["prior_cond"] = ((n_tasks, len(PRIOR_COND_LAYOUT)), np.float32)

        for key, (expected, _) in field_specs.items():
            actual = corpus[key].shape
            if len(actual) != len(expected):
                errors.append(
                    f"Shape mismatch for {key}: expected ndim={len(expected)} {expected}, got ndim={len(actual)} {actual}"
                )
                continue
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    errors.append(f"Shape mismatch for {key}: expected dim {i} = {e}, got {a}")
                    break
        if errors:
            return errors

        if has_signal_labels:
            expected_label_shape = (n_tasks, n_treatments, len(SIGNAL_METRIC_LAYOUT))
            for key in signal_label_keys:
                value = identifiability[key]
                if value.shape != expected_label_shape:
                    errors.append(
                        f"Shape mismatch for identifiability {key}: "
                        f"expected {expected_label_shape}, got {value.shape}"
                    )
            if errors:
                return errors

        for key, (_, dtype) in field_specs.items():
            if corpus[key].dtype != dtype:
                errors.append(f"{key} has dtype {corpus[key].dtype}, expected {np.dtype(dtype)}")
        if has_signal_labels:
            for key, dtype in (
                ("signal_metrics", np.float32),
                ("signal_metric_valid", np.uint8),
            ):
                if identifiability[key].dtype != dtype:
                    errors.append(
                        f"identifiability {key} has dtype {identifiability[key].dtype}, "
                        f"expected {np.dtype(dtype)}"
                    )
        if errors:
            return errors

        # Every serialized numeric array must be finite, including normalized
        # model inputs rather than only their raw source arrays.
        for key, value in corpus.items():
            if not isinstance(value, np.ndarray):
                continue
            if value.dtype.hasobject or not (
                np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.bool_)
            ):
                errors.append(f"{key} has unsupported dtype {value.dtype}")
            elif not np.isfinite(value).all():
                errors.append(f"{key} contains NaN or Inf")
        if isinstance(identifiability, dict):
            for key, value in identifiability.items():
                if value.dtype.hasobject or not (
                    np.issubdtype(value.dtype, np.integer)
                    or np.issubdtype(value.dtype, np.floating)
                    or np.issubdtype(value.dtype, np.bool_)
                ):
                    errors.append(f"identifiability.{key} has unsupported dtype {value.dtype}")
                elif not np.isfinite(value).all():
                    errors.append(f"identifiability.{key} contains NaN or Inf")
        unknown_fields = set(corpus) - set(field_specs) - {"diagnostics", "identifiability"}
        if unknown_fields and not errors:
            errors.append(f"Unrecognized corpus fields: {sorted(unknown_fields)}")

        positive_outcome_scale = (corpus["outcome_scale"] > 0.0).all()
        if not positive_outcome_scale:
            errors.append("outcome_scale must be positive")
        treatment_raw = corpus["treatment_raw"].astype(np.float64)
        # Treatments are nonnegative by construction: softplus levels under
        # nonnegative envelopes, nonnegative held shock levels, zero off-weeks.
        if (treatment_raw < 0.0).any():
            errors.append("treatment_raw must be nonnegative")
        expected_treatment_means = treatment_raw.mean(axis=1)
        if not np.allclose(
            corpus["treatment_means"], expected_treatment_means, rtol=1e-6, atol=1e-7
        ):
            errors.append("treatment_means != mean(treatment_raw, axis=1)")
        treatment_means = corpus["treatment_means"].astype(np.float64)
        expected_treatment_norm = np.divide(
            treatment_raw,
            treatment_means[:, None, :],
            out=np.zeros_like(treatment_raw),
            where=treatment_means[:, None, :] != 0.0,
        )
        if not np.allclose(corpus["treatment_norm"], expected_treatment_norm, rtol=1e-5, atol=1e-7):
            errors.append("treatment_norm does not match treatment_raw / treatment_means")
        active_treatment_sum = (treatment_raw * corpus["treatment_active_mask"][:, None, :]).sum(
            axis=-1, keepdims=True
        )
        expected_treatment_share = (
            np.divide(
                treatment_raw,
                active_treatment_sum,
                out=np.zeros_like(treatment_raw),
                where=active_treatment_sum != 0.0,
            )
            * corpus["treatment_active_mask"][:, None, :]
        )
        if not np.allclose(
            corpus["treatment_share"], expected_treatment_share, rtol=1e-5, atol=1e-7
        ):
            errors.append("treatment_share does not match active-treatment treatment shares")

        outcome = corpus["outcome_raw"].astype(np.float64)
        if positive_outcome_scale:
            expected_norm = outcome / corpus["outcome_scale"].astype(np.float64)[:, None]
            if not np.allclose(corpus["outcome_norm"], expected_norm, rtol=1e-5):
                errors.append("outcome_norm != outcome_raw / outcome_scale")

        # These arrays have already been validated as uint8.
        for key in ("support_mask", "is_future", "g"):
            if (corpus[key] > 1).any():
                errors.append(f"{key} is not binary")

        short_n_query = None
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("signal"), dict):
            candidate = diagnostics.get("short_horizon_n_query")
            if (
                isinstance(candidate, (int, np.integer))
                and not isinstance(candidate, (bool, np.bool_))
                and 0 < candidate < n_time_steps
            ):
                short_n_query = int(candidate)
            else:
                errors.append(
                    "diagnostics short_horizon_n_query must be an integer in (0, n_time_steps)"
                )
        if short_n_query is not None:
            expected_support_count = np.where(
                corpus["is_future"] == 1,
                n_time_steps // 2,
                n_time_steps - short_n_query,
            )
            expected_support = (
                np.arange(n_time_steps)[None, :] < expected_support_count[:, None]
            ).astype(np.uint8)
            if not np.array_equal(corpus["support_mask"], expected_support):
                errors.append("support_mask does not match the recorded temporal split")

            expected_outcome_scale = np.asarray(
                [outcome[i, expected_support[i] == 1].std() for i in range(n_tasks)],
                dtype=np.float64,
            )
            bad_scale = ~(np.isfinite(expected_outcome_scale) & (expected_outcome_scale > 0.0))
            if bad_scale.any():
                full_std = outcome[bad_scale].std(axis=1)
                expected_outcome_scale[bad_scale] = np.where(
                    np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0
                )
            if not np.allclose(
                corpus["outcome_scale"], expected_outcome_scale, rtol=1e-5, atol=1e-7
            ):
                errors.append("outcome_scale does not match supported outcome observations")

        is_val = corpus["is_val"]
        if (is_val > 1).any():
            errors.append("is_val is not binary")
        elif not 0 < is_val.sum() < n_tasks:
            errors.append("is_val must contain at least one training and one validation world")
        # Top-level shock audit metadata is intentionally sufficient to
        # reconstruct every held-treatment intervention without persisting the
        # full burn-in mask or natural (unshocked) paths.
        shock_mask = corpus["treatment_shock_mask"]
        if not np.isin(shock_mask, (0, 1)).all():
            errors.append("treatment_shock_mask is not binary")
        if (corpus["treatment_shock_level_multiplier"] < 0).any():
            errors.append("treatment_shock_level_multiplier contains negative values")
        if (corpus["treatment_shock_level"] < 0).any():
            errors.append("treatment_shock_level contains negative values")
        if not (
            (corpus["confounding_strength"] >= 0.0) & (corpus["confounding_strength"] <= 0.95)
        ).all():
            errors.append("confounding_strength must be in [0, 0.95]")
        if not np.isin(corpus["carryover_family"], (0, 1, 2)).all():
            errors.append("carryover_family contains invalid family ids")
        if has_signal_labels:
            signal_metrics = identifiability["signal_metrics"]
            signal_metric_valid = identifiability["signal_metric_valid"]
            if not np.isin(signal_metric_valid, (0, 1)).all():
                errors.append("signal_metric_valid is not binary")
            if not np.isfinite(signal_metrics).all():
                errors.append("signal_metrics contains NaN or Inf")
            if (signal_metrics[signal_metric_valid == 0] != 0).any():
                errors.append("signal_metrics has nonzero invalid values")
            metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
            for name, low, high in (
                ("spearman", 0.0, 1.0),
                ("contrib_r2_explained_by_rest", 0.0, 1.0),
                ("contrib_corr_baseline", -1.0, 1.0),
            ):
                index = metric_index[name]
                values = signal_metrics[..., index][signal_metric_valid[..., index].astype(bool)]
                if ((values < low - 1e-6) | (values > high + 1e-6)).any():
                    errors.append(f"signal metric {name} is outside [{low}, {high}]")
        if not isinstance(diagnostics, dict):
            errors.append("diagnostics must be a mapping")
            return errors
        signal_diagnostics = diagnostics.get("signal")
        if not isinstance(signal_diagnostics, dict):
            errors.append("diagnostics signal must be a mapping")
            return errors
        edge_types: Any = diagnostics.get("edge_types")
        try:
            normalized_edge_types = list(edge_types)
        except TypeError:
            normalized_edge_types = None
        if normalized_edge_types != list(EDGE_TYPES_EXTENDED):
            errors.append("diagnostics edge_types does not match the canonical layout")
        version_problem = _schema_version_error(diagnostics)
        if version_problem is not None:
            errors.append(version_problem)
        timing_problem = _timing_error(diagnostics)
        if timing_problem is not None:
            errors.append(timing_problem)

        metric_version = signal_diagnostics.get("metric_version")

        def _is_integer(value: object) -> TypeGuard[int | np.integer]:
            return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

        def _diagnostic_equal(actual, expected) -> bool:
            if isinstance(actual, np.ndarray):
                actual = actual.item() if actual.ndim == 0 else actual.tolist()
            elif isinstance(actual, np.generic):
                actual = actual.item()
            if isinstance(expected, np.ndarray):
                expected = expected.item() if expected.ndim == 0 else expected.tolist()
            elif isinstance(expected, np.generic):
                expected = expected.item()
            if isinstance(actual, dict) and isinstance(expected, dict):
                return actual.keys() == expected.keys() and all(
                    _diagnostic_equal(actual[key], expected[key]) for key in actual
                )
            if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
                return len(actual) == len(expected) and all(
                    _diagnostic_equal(left, right) for left, right in zip(actual, expected)
                )
            if isinstance(actual, (dict, list, tuple)) or isinstance(expected, (dict, list, tuple)):
                return False
            try:
                result = actual == expected
            except (TypeError, ValueError):
                return False
            return isinstance(result, (bool, np.bool_)) and bool(result)

        version_supported = (
            _is_integer(metric_version) and int(metric_version) == SIGNAL_METRIC_VERSION
        )
        if not version_supported:
            errors.append("diagnostics signal metric_version is not supported")
        metric_layout: Any = signal_diagnostics.get("metric_layout")
        try:
            normalized_layout = list(metric_layout)
        except TypeError:
            normalized_layout = None
        if normalized_layout != list(SIGNAL_METRIC_LAYOUT):
            errors.append("diagnostics signal metric_layout does not match signal_metrics")
        if version_supported:
            if not _is_integer(signal_diagnostics.get("l_max")) or signal_diagnostics["l_max"] < 1:
                errors.append("diagnostics signal l_max must be a positive integer")
            if (
                not _is_integer(signal_diagnostics.get("carryover_burn_in"))
                or signal_diagnostics["carryover_burn_in"] < 0
            ):
                errors.append("diagnostics signal carryover_burn_in must be a nonnegative integer")
            response_warmup_weeks = signal_diagnostics.get("response_warmup_weeks")
            if (
                not _is_integer(response_warmup_weeks)
                or not 0 <= response_warmup_weeks < n_time_steps
            ):
                errors.append(
                    "diagnostics signal response_warmup_weeks must be a nonnegative "
                    "integer below n_time_steps"
                )
            frac_zero_contemporaneous_weight = signal_diagnostics.get(
                "frac_zero_contemporaneous_weight"
            )
            if frac_zero_contemporaneous_weight is not None and (
                not isinstance(frac_zero_contemporaneous_weight, (float, np.floating))
                or not np.isfinite(frac_zero_contemporaneous_weight)
                or not 0.0 <= frac_zero_contemporaneous_weight <= 1.0
            ):
                errors.append(
                    "diagnostics signal frac_zero_contemporaneous_weight must be None or a float "
                    "in [0, 1]"
                )
            for prefix, label, semantics, version in (
                (
                    "carryover_kernel",
                    "carryover kernel",
                    "normalized-causal-minmax-weibull-density",
                    3,
                ),
                ("outcome_noise", "outcome noise", OUTCOME_NOISE_SEMANTICS, OUTCOME_NOISE_VERSION),
            ):
                actual_semantics = signal_diagnostics.get(f"{prefix}_semantics")
                actual_version = signal_diagnostics.get(f"{prefix}_version")
                if (
                    not isinstance(actual_semantics, str)
                    or actual_semantics != semantics
                    or not _is_integer(actual_version)
                    or actual_version != version
                ):
                    errors.append(f"diagnostics signal {label} semantics are not supported")
            outcome_std_mode = signal_diagnostics.get("outcome_std_mode")
            if not isinstance(outcome_std_mode, str) or outcome_std_mode not in (
                "relative",
                "absolute",
            ):
                errors.append("diagnostics signal outcome_std_mode is not supported")

        cell_ids = np.unique(corpus["cell_id"])
        if not _is_integer(diagnostics.get("n_tasks")) or diagnostics["n_tasks"] != n_tasks:
            errors.append("diagnostics n_tasks does not match the corpus")
        if not _is_integer(diagnostics.get("n_cells")) or diagnostics["n_cells"] != len(cell_ids):
            errors.append("diagnostics n_cells does not match cell_id")
        if (corpus["cell_id"] < 0).any() or not np.array_equal(
            cell_ids, np.arange(len(cell_ids), dtype=cell_ids.dtype)
        ):
            errors.append("cell_id must contain contiguous nonnegative ids")
        cell_level_keys: tuple[str, ...] = (
            "g",
            "treatment_active_mask",
            "covariate_active_mask",
            "latent_active_mask",
            "n_treatments_active",
            "n_covariates_active",
            "n_latent_active",
            "treatment_active",
            "is_val",
        )
        if has_trajectory:
            cell_level_keys += ("treatment_components", "covariate_components")
        if mechanism_fields:
            cell_level_keys += ("sat_family",)
        for cell in cell_ids:
            in_cell = corpus["cell_id"] == cell
            for key in cell_level_keys:
                rows = corpus[key][in_cell]
                if len(rows) > 1 and not np.equal(rows, rows[0]).all():
                    errors.append(f"{key} differs within a cell")

        has_prior_cond = "prior_cond" in corpus
        has_prior_cond_diagnostics = "prior_cond" in diagnostics
        if has_prior_cond != has_prior_cond_diagnostics:
            errors.append("prior_cond and diagnostics prior_cond must be present together")
        elif has_prior_cond:
            prior_diagnostics = diagnostics["prior_cond"]
            if not isinstance(prior_diagnostics, dict):
                errors.append("diagnostics prior_cond must be a mapping")
            else:
                prior_layout: Any = prior_diagnostics.get("layout")
                try:
                    normalized_prior_layout = list(prior_layout)
                except TypeError:
                    normalized_prior_layout = None
                if normalized_prior_layout != list(PRIOR_COND_LAYOUT):
                    errors.append("diagnostics prior_cond layout does not match prior_cond")
                supports = prior_diagnostics.get("supports")
                width_ranges = prior_diagnostics.get("width_ranges")
                if not isinstance(supports, dict) or not isinstance(width_ranges, dict):
                    errors.append(
                        "diagnostics prior_cond supports and width_ranges must be mappings"
                    )
                elif set(supports) != set(PRIOR_COND_QUANTITIES) or set(width_ranges) != set(
                    PRIOR_COND_QUANTITIES
                ):
                    errors.append("diagnostics prior_cond quantities do not match the layout")
                else:
                    prior_cond = corpus["prior_cond"].astype(np.float64)
                    for q_idx, quantity in enumerate(PRIOR_COND_QUANTITIES):
                        try:
                            support = np.asarray(supports[quantity], dtype=np.float64)
                            width_range = np.asarray(width_ranges[quantity], dtype=np.float64)
                        except (TypeError, ValueError):
                            errors.append(f"diagnostics prior_cond {quantity} bounds are invalid")
                            continue
                        if (
                            support.shape != (2,)
                            or width_range.shape != (2,)
                            or not np.isfinite(support).all()
                            or not np.isfinite(width_range).all()
                            or support[0] >= support[1]
                            or support[0] < 0.0
                            or (quantity == "carryover_alpha" and support[1] > 1.0)
                            or (
                                quantity == "hill_shape"
                                and (support[0] == 0.0 or support[1] > CORPUS_STORAGE_MAX)
                            )
                            or not 0.0 < width_range[0] <= width_range[1] <= support[1] - support[0]
                        ):
                            errors.append(f"diagnostics prior_cond {quantity} bounds are invalid")
                            continue
                        low = prior_cond[:, 2 * q_idx]
                        width = prior_cond[:, 2 * q_idx + 1]
                        tolerance = (
                            8
                            * np.finfo(np.float32).eps
                            * max(
                                float(np.abs(support).max()), float(np.abs(width_range).max()), 1.0
                            )
                        )
                        if (
                            (width < width_range[0] - tolerance).any()
                            or (width > width_range[1] + tolerance).any()
                            or (low < support[0] - tolerance).any()
                            or (low + width > support[1] + tolerance).any()
                        ):
                            errors.append(
                                f"prior_cond {quantity} intervals are outside diagnostics bounds"
                            )
                    for cell in np.unique(corpus["cell_id"]):
                        rows = corpus["prior_cond"][corpus["cell_id"] == cell]
                        if len(rows) > 1 and not np.equal(rows, rows[0]).all():
                            errors.append("prior_cond rows differ within a cell")
                            break

        active_treatment = corpus["treatment_active_mask"]
        active_covariate = corpus["covariate_active_mask"]
        active_latent = corpus["latent_active_mask"]
        for key, mask in (
            ("treatment_active_mask", active_treatment),
            ("covariate_active_mask", active_covariate),
            ("latent_active_mask", active_latent),
            ("treatment_active", corpus["treatment_active"]),
        ):
            if (mask > 1).any():
                errors.append(f"{key} is not binary")
        for key, count, width, mask in (
            ("n_treatments_active", corpus["n_treatments_active"], n_treatments, active_treatment),
            ("n_covariates_active", corpus["n_covariates_active"], n_covariates, active_covariate),
            ("n_latent_active", corpus["n_latent_active"], n_latent, active_latent),
        ):
            if ((count < 1) | (count > width)).any() or not np.array_equal(
                mask, (np.arange(width)[None, :] < count[:, None]).astype(np.uint8)
            ):
                errors.append(f"{key} does not match its active prefix mask")
        # The optional stratified-coverage block must recount exactly from the
        # masks just checked, and its cells must round the allocation targets
        # of its own weights.
        if isinstance(diagnostics, dict) and "active_count_coverage" in diagnostics:
            errors.extend(
                active_count_coverage_errors(
                    diagnostics["active_count_coverage"],
                    active_treatment,
                    active_covariate,
                    corpus["cell_id"],
                )
            )

        inactive_c, inactive_m, inactive_j = (
            active_treatment == 0,
            active_covariate == 0,
            active_latent == 0,
        )

        def _padded_nonzero(array: np.ndarray, inactive: np.ndarray) -> bool:
            expanded = np.broadcast_to(inactive[:, None, :], array.shape)
            return bool((array[expanded] != 0).any())

        for key in (
            "treatment_raw",
            "treatment_norm",
            "treatment_share",
            "treatment_contribution_raw",
        ):
            if _padded_nonzero(corpus[key], inactive_c):
                errors.append(f"{key} has nonzero inactive-treatment padding")
        if _padded_nonzero(corpus["treatment_shock_mask"], inactive_c):
            errors.append("treatment_shock_mask has nonzero inactive-treatment padding")
        for key in (
            "treatment_means",
            "treatment_active",
            "treatment_level",
            "saturation_scale",
            "carryover_family",
            "carryover_alpha",
            "weibull_lam",
            "weibull_k",
        ):
            if (corpus[key][inactive_c] != 0).any():
                errors.append(f"{key} has nonzero inactive-treatment padding")
        for key in ("covariates", "covariate_contribution"):
            if _padded_nonzero(corpus[key], inactive_m):
                errors.append(f"{key} has nonzero inactive-covariate padding")
        for key in ("latent_unobserved", "latent_unobserved_contribution"):
            if _padded_nonzero(corpus[key], inactive_j):
                errors.append(f"{key} has nonzero inactive-latent_unobserved padding")
        if (corpus["saturation_scale"][active_treatment == 1] <= 0.0).any():
            errors.append("saturation_scale must be positive for active treatments")

        if has_trajectory:
            # Component flags are per input (constant within a cell, checked with
            # the cell-level keys above). On an active input, no gate form means
            # always on and no level form means no shift; an off-week zeroes the
            # observed series, except where a held treatment shock overrides it.
            # Conversely, config validation guarantees that an included gate form
            # switches its input off at least once yet keeps 2 on-weeks inside the
            # support window, that an onset launches at week >= 1, and that an
            # offset stops by the last week.
            gate_columns = [TRAJECTORY_COMPONENTS.index(name) for name in GATE_COMPONENTS]
            level_columns = [TRAJECTORY_COMPONENTS.index(name) for name in LEVEL_COMPONENTS]
            in_support = corpus["support_mask"][:, :, None] == 1
            flags_binary = True
            for input_type, inactive, activity_key, shift_key in (
                ("treatment", inactive_c, "treatment_activity", "treatment_log_level_shift"),
                ("covariate", inactive_m, "covariate_activity", "covariate_level_shift"),
            ):
                flags_key = f"{input_type}_components"
                flags, activity = corpus[flags_key], corpus[activity_key]
                # Both have already been validated as uint8.
                if (flags > 1).any():
                    flags_binary = False
                    errors.append(f"{flags_key} is not binary")
                if (activity > 1).any():
                    errors.append(f"{activity_key} is not binary")
                if (flags[inactive] != 0).any():
                    errors.append(f"{flags_key} has nonzero inactive-{input_type} padding")
                for key in (activity_key, shift_key):
                    if _padded_nonzero(corpus[key], inactive):
                        errors.append(f"{key} has nonzero inactive-{input_type} padding")
                gated = flags[..., gate_columns].any(axis=-1)
                if _padded_nonzero(activity != 1, ~inactive & ~gated):
                    errors.append(
                        f"{activity_key} must be 1 for active {input_type}s "
                        "without a gate component"
                    )
                gated &= ~inactive
                on_weeks = activity != 0
                if (gated & on_weeks.all(axis=1)).any():
                    errors.append(
                        f"{activity_key} never switches off for an active {input_type} "
                        "with a gate component"
                    )
                if (gated & ((on_weeks & in_support).sum(axis=1) < 2)).any():
                    errors.append(
                        f"{activity_key} keeps fewer than 2 on-weeks inside the support window "
                        f"for an active {input_type} with a gate component"
                    )
                for component, week, label in (
                    ("onset", 0, "week 0"),
                    ("offset", -1, "the last week"),
                ):
                    carried = ~inactive & (flags[..., TRAJECTORY_COMPONENTS.index(component)] != 0)
                    if (carried & on_weeks[:, week, :]).any():
                        errors.append(
                            f"{activity_key} must be 0 at {label} for an active {input_type} "
                            f"with an {component} component"
                        )
                unshifted = ~inactive & ~flags[..., level_columns].any(axis=-1)
                if _padded_nonzero(corpus[shift_key], unshifted):
                    errors.append(
                        f"{shift_key} must be 0 for active {input_type}s without a level component"
                    )
            treatment_off = (corpus["treatment_activity"] == 0) & (shock_mask == 0)
            if (corpus["treatment_raw"][treatment_off] != 0).any():
                errors.append(
                    "treatment_raw must be 0 on treatment_activity off-weeks outside shocks"
                )
            if (corpus["covariates"][corpus["covariate_activity"] == 0] != 0).any():
                errors.append("covariates must be 0 on covariate_activity off-weeks")
            trajectory_diagnostics = diagnostics["trajectory"]
            if not isinstance(trajectory_diagnostics, dict):
                errors.append("diagnostics trajectory must be a mapping")
            else:
                trajectory_layout: Any = trajectory_diagnostics.get("components")
                try:
                    normalized_trajectory_layout = list(trajectory_layout)
                except TypeError:
                    normalized_trajectory_layout = None
                if normalized_trajectory_layout != list(TRAJECTORY_COMPONENTS):
                    errors.append(
                        "diagnostics trajectory components do not match the canonical layout"
                    )
                inclusion_probs: Any = trajectory_diagnostics.get("inclusion_probs")
                inclusion_valid = (
                    isinstance(inclusion_probs, dict)
                    and set(inclusion_probs) == set(TRAJECTORY_INPUTS)
                    and all(
                        isinstance(probs, dict)
                        and set(probs) == set(TRAJECTORY_COMPONENTS)
                        and all(
                            isinstance(prob, (float, np.floating)) and 0.0 <= prob <= 1.0
                            for prob in probs.values()
                        )
                        for probs in inclusion_probs.values()
                    )
                )
                if not inclusion_valid:
                    errors.append(
                        "diagnostics trajectory inclusion_probs must map every input and "
                        "component to a float in [0, 1]"
                    )
                # Recounting non-binary flags would only restate the error above.
                if flags_binary:
                    recounted = summarize_component_prevalence(
                        corpus["treatment_components"],
                        corpus["covariate_components"],
                        active_treatment,
                        active_covariate,
                    )
                    for key in ("prevalence", "n_inputs"):
                        if not _diagnostic_equal(trajectory_diagnostics.get(key), recounted[key]):
                            errors.append(
                                f"diagnostics trajectory {key} does not match recomputation"
                            )
                    # An exact echo pins the flags: generation draws none at
                    # probability 0 and flags every active input at probability 1.
                    if inclusion_valid:
                        for input_type in TRAJECTORY_INPUTS:
                            prevalence = recounted["prevalence"][input_type]
                            counted = recounted["n_inputs"][input_type] > 0
                            for component in TRAJECTORY_COMPONENTS:
                                prob = inclusion_probs[input_type][component]
                                if (prob == 0.0 and prevalence[component] != 0.0) or (
                                    prob == 1.0 and counted and prevalence[component] != 1.0
                                ):
                                    errors.append(
                                        "diagnostics trajectory inclusion_probs contradict the "
                                        f"stored {input_type}_components for {component}"
                                    )

        graph = layout.unpack(corpus["g"])
        treatment_present = ~inactive_c
        covariate_present = ~inactive_m
        latent_present = ~inactive_j
        graph_masks = {
            "cy": treatment_present,
            "dc": latent_present[:, :, None] & treatment_present[:, None, :],
            "dz": latent_present[:, :, None] & covariate_present[:, None, :],
            "dy": latent_present,
            "zy": covariate_present,
            "zc": covariate_present[:, :, None] & treatment_present[:, None, :],
            "cc": treatment_present[:, :, None] & treatment_present[:, None, :],
            "zz": covariate_present[:, :, None] & covariate_present[:, None, :],
        }
        for edge_type, edge_mask in graph_masks.items():
            if (graph[edge_type][~edge_mask] != 0).any():
                errors.append(f"g_{edge_type} has edges incident to inactive nodes")
        for edge_type in ("cc", "zz"):
            if np.tril(graph[edge_type], k=-1).any():
                errors.append(f"g_{edge_type} must be strictly upper triangular")
        direct = graph["cy"] == 1
        expected_treatment_active = (
            (direct | (graph["cc"].sum(axis=2) > 0)) & treatment_present
        ).astype(np.uint8)
        if not np.array_equal(corpus["treatment_active"], expected_treatment_active):
            errors.append("treatment_active does not match graph reachability rule")
        if _padded_nonzero(corpus["treatment_contribution_raw"], ~direct):
            errors.append("treatment_contribution_raw is nonzero for treatments without C->Y edges")
        if _padded_nonzero(corpus["covariate_contribution"], graph["zy"] != 1):
            errors.append("covariate_contribution is nonzero without a Z->Y edge")
        if _padded_nonzero(corpus["latent_unobserved_contribution"], graph["dy"] != 1):
            errors.append("latent_unobserved_contribution is nonzero without a D->Y edge")
        for source_index, edge_type in enumerate(("cc", "zc", "dc")):
            source_present = graph[edge_type].reshape(n_tasks, -1).any(axis=1)
            if (corpus["indirect_effects_by_source"][~source_present, :, source_index] != 0).any():
                errors.append(
                    f"indirect_effects_by_source {edge_type} column is nonzero without an edge"
                )

        if has_signal_labels and (
            (signal_metrics[inactive_c] != 0).any() or signal_metric_valid[inactive_c].any()
        ):
            errors.append("signal metrics have nonzero inactive-treatment padding")
        ineligible = ~(direct & (active_treatment == 1))
        if has_signal_labels and (
            (signal_metrics[ineligible] != 0).any() or signal_metric_valid[ineligible].any()
        ):
            errors.append("signal metrics have nonzero ineligible-treatment values")

        treatments = corpus["treatment_shock_index"]
        starts = corpus["treatment_shock_start"]
        lengths = corpus["treatment_shock_length"]
        levels = corpus["treatment_shock_level"]
        multipliers = corpus["treatment_shock_level_multiplier"]
        treatment_level = corpus["treatment_level"]
        for n in range(n_tasks):
            rebuilt = np.zeros((n_time_steps, n_treatments), dtype=np.uint8)
            occupied = np.zeros(n_time_steps, dtype=bool)
            for s_idx in range(n_treatment_shocks):
                treatment, start, length = (
                    int(treatments[n, s_idx]),
                    int(starts[n, s_idx]),
                    int(lengths[n, s_idx]),
                )
                slot_lo = s_idx * n_time_steps // n_treatment_shocks
                slot_hi = (s_idx + 1) * n_time_steps // n_treatment_shocks
                if not (0 <= treatment < n_treatments and direct[n, treatment]):
                    errors.append("treatment_shock_index is not an active direct treatment")
                    continue
                if not (length > 0 and slot_lo <= start and start + length <= slot_hi):
                    errors.append("treatment shock start/length is outside its schedule slot")
                    continue
                if occupied[start : start + length].any():
                    errors.append("treatment shocks overlap globally")
                occupied[start : start + length] = True
                rebuilt[start : start + length, treatment] = 1
                expected_level = multipliers[n, s_idx] * treatment_level[n, treatment]
                if not np.isclose(levels[n, s_idx], expected_level, rtol=1e-6, atol=1e-7):
                    errors.append(
                        "treatment_shock_level does not match multiplier * treatment_level"
                    )
                if not np.array_equal(
                    corpus["treatment_raw"][n, start : start + length, treatment],
                    np.full(length, levels[n, s_idx], dtype=np.float32),
                ):
                    errors.append("held treatment does not equal treatment_shock_level")
            if not np.array_equal(rebuilt, shock_mask[n]):
                errors.append("treatment_shock_mask does not match the schedule")

        baseline = corpus["baseline_raw"].astype(np.float64)
        contributions = corpus["treatment_contribution_raw"].astype(np.float64)
        indirect = corpus["indirect_effects"].astype(np.float64)
        indirect_by_source = corpus["indirect_effects_by_source"].astype(np.float64)
        intrinsic = corpus["baseline_intrinsic"].astype(np.float64)
        outcome_noise = corpus["outcome_noise"].astype(np.float64)
        covariate_contribution = corpus["covariate_contribution"].astype(np.float64)
        latent_unobserved_contribution = corpus["latent_unobserved_contribution"].astype(np.float64)
        tolerance_factor = 32 * np.finfo(corpus["outcome_raw"].dtype).eps
        decomposition_errors = {
            "additive decomposition": (
                np.abs(baseline + contributions.sum(axis=2) + indirect - outcome),
                outcome,
            ),
            "telescoping decomposition": (
                np.abs(indirect_by_source.sum(axis=2) - indirect),
                indirect,
            ),
            "baseline decomposition": (
                np.abs(
                    intrinsic
                    + outcome_noise
                    + latent_unobserved_contribution.sum(axis=2)
                    + covariate_contribution.sum(axis=2)
                    - baseline
                ),
                baseline,
            ),
            "full decomposition": (
                np.abs(
                    intrinsic
                    + outcome_noise
                    + latent_unobserved_contribution.sum(axis=2)
                    + covariate_contribution.sum(axis=2)
                    + contributions.sum(axis=2)
                    + indirect_by_source.sum(axis=2)
                    - outcome
                ),
                outcome,
            ),
        }
        for name, (residual, reference) in decomposition_errors.items():
            tolerance = tolerance_factor * np.maximum(np.abs(reference), 1.0)
            if (residual > tolerance).any():
                errors.append(f"{name} exceeds float32 storage tolerance")

        if not errors and version_supported:
            eligible = direct & (corpus["treatment_active_mask"] == 1)
            expected_metrics, expected_valid = dense_signal_metrics(
                corpus["treatment_raw"],
                corpus["treatment_contribution_raw"],
                corpus["outcome_raw"],
                corpus["baseline_raw"],
                eligible,
                outcome_scale=corpus["outcome_scale"],
                carryover_family=corpus["carryover_family"],
                carryover_alpha=corpus["carryover_alpha"],
                weibull_lam=corpus["weibull_lam"],
                weibull_k=corpus["weibull_k"],
                l_max=int(signal_diagnostics["l_max"]),
                carryover_burn_in=int(signal_diagnostics["carryover_burn_in"]),
            )
            if has_signal_labels:
                if not np.array_equal(signal_metrics, expected_metrics):
                    errors.append("identifiability signal_metrics do not match recomputation")
                if not np.array_equal(signal_metric_valid, expected_valid):
                    errors.append(
                        "identifiability signal_metric_valid does not match recomputation"
                    )
            expected_signal = summarize_signal_metrics(
                expected_metrics,
                expected_valid,
                corpus["outcome_raw"],
                eligible,
                outcome_scale=corpus["outcome_scale"],
                l_max=int(signal_diagnostics["l_max"]),
                carryover_burn_in=int(signal_diagnostics["carryover_burn_in"]),
                carryover_family=corpus["carryover_family"],
                carryover_alpha=corpus["carryover_alpha"],
                weibull_lam=corpus["weibull_lam"],
                weibull_k=corpus["weibull_k"],
            )
            expected_signal.update(
                {
                    "metric_version": SIGNAL_METRIC_VERSION,
                    "metric_layout": list(SIGNAL_METRIC_LAYOUT),
                    "l_max": int(signal_diagnostics["l_max"]),
                    "carryover_burn_in": int(signal_diagnostics["carryover_burn_in"]),
                    "carryover_kernel_semantics": "normalized-causal-minmax-weibull-density",
                    "carryover_kernel_version": 3,
                    "outcome_noise_semantics": OUTCOME_NOISE_SEMANTICS,
                    "outcome_noise_version": OUTCOME_NOISE_VERSION,
                    "outcome_std_mode": signal_diagnostics["outcome_std_mode"],
                }
            )
            actual_signal = dict(signal_diagnostics)
            actual_signal["metric_layout"] = normalized_layout
            if not _diagnostic_equal(actual_signal, expected_signal):
                errors.append("diagnostics signal summary does not match recomputation")

        # This corpus-level invariant enforces beta_additive_range >= 0 downstream.
        if (corpus["treatment_contribution_raw"] < 0).any():
            errors.append("treatment_contribution_raw contains negative values")
        if not errors and has_trajectory:
            errors.extend(_trajectory_truth_errors(corpus))
        if not errors and mechanism_fields:
            errors.extend(_mechanism_truth_errors(corpus, mechanism_fields))

        return errors


# ---------------------------------------------------------------------------
# Save/Load utilities
# ---------------------------------------------------------------------------


def save_corpus(corpus: dict[str, Any], path: str | Path) -> None:
    """Save corpus to disk.

    Parameters
    ----------
    corpus : dict
        Corpus dictionary.
    path : str or Path
        Path to save the corpus (.npz file).
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"corpus path must use the .npz suffix, got {path}")
    legacy = sorted(set(corpus) & set(LEGACY_CORPUS_KEYS_V1))
    if legacy:
        raise ValueError(
            f"corpus uses pre-v{CORPUS_SCHEMA_VERSION} symbolic dimension keys {legacy}; "
            f"regenerate the corpus under schema_version={CORPUS_SCHEMA_VERSION}"
        )
    if any(key.startswith("identifiability__") for key in corpus):
        raise ValueError("top-level corpus keys may not use the reserved identifiability__ prefix")
    identifiability = corpus.get("identifiability")
    if isinstance(identifiability, dict) and not identifiability:
        raise ValueError("identifiability metadata must be omitted rather than empty")

    def _is_real_numeric(array: np.ndarray) -> bool:
        return bool(
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.bool_)
        )

    # Convert diagnostics dict to JSON string if present
    save_dict: dict[str, Any] = {}
    for k, v in corpus.items():
        if k == "diagnostics":
            if not isinstance(v, dict):
                raise TypeError("diagnostics must be a mapping")
            import json

            def _json_default(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, np.ndarray):
                    if value.dtype.hasobject:
                        raise ValueError("diagnostics arrays may not have object dtype")
                    return value.tolist()
                raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

            # Timing is nondeterministic. JSON encoding is read-only, so only
            # the top-level mapping needs rebuilding to leave the caller intact.
            persisted = {key: value for key, value in v.items() if key != "timing"}
            save_dict[k] = np.array(json.dumps(persisted, default=_json_default))
        elif k == "identifiability":
            if not isinstance(v, dict):
                raise TypeError("identifiability metadata must be a mapping")
            for label, value in v.items():
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"identifiability.{label} must be an ndarray")
                if value.dtype.hasobject:
                    raise ValueError(f"identifiability.{label} may not have object dtype")
                if np.issubdtype(value.dtype, np.complexfloating):
                    raise ValueError(f"identifiability.{label} may not have complex dtype")
                if not _is_real_numeric(value):
                    raise ValueError(f"identifiability.{label} must have a real numeric dtype")
                if not np.isfinite(value).all():
                    raise ValueError(f"identifiability.{label} contains NaN or Inf")
                save_dict[f"identifiability__{label}"] = value
        else:
            if not isinstance(v, np.ndarray):
                raise TypeError(f"{k} must be an ndarray")
            if v.dtype.hasobject:
                raise ValueError(f"{k} may not have object dtype")
            if np.issubdtype(v.dtype, np.complexfloating):
                raise ValueError(f"{k} may not have complex dtype")
            if not _is_real_numeric(v):
                raise ValueError(f"{k} must have a real numeric dtype")
            save_dict[k] = v

    version_problem = _schema_version_error(corpus.get("diagnostics"))
    if version_problem is not None:
        raise ValueError(version_problem)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **save_dict)


def load_corpus(path: str | Path) -> dict[str, Any]:
    """Load corpus from disk.

    Parameters
    ----------
    path : str or Path
        Path to the corpus (.npz file).

    Returns
    -------
    dict
        Corpus dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        corpus: dict[str, Any] = {k: data[k] for k in data.files}

    identifiability = {
        key.removeprefix("identifiability__"): corpus.pop(key)
        for key in tuple(corpus)
        if key.startswith("identifiability__")
    }
    if identifiability:
        corpus["identifiability"] = identifiability

    # Parse diagnostics JSON string if present
    if "diagnostics" in corpus:
        import json

        diag_str = corpus["diagnostics"]
        if not isinstance(diag_str, np.ndarray) or diag_str.ndim != 0:
            raise ValueError(f"corpus {path} has a non-scalar diagnostics field")
        try:
            diagnostics = json.loads(str(diag_str.item()))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"corpus {path} has invalid diagnostics JSON") from exc
        if not isinstance(diagnostics, dict):
            raise ValueError(f"corpus {path} diagnostics JSON must contain an object")
        corpus["diagnostics"] = diagnostics

    version_problem = _schema_version_error(corpus.get("diagnostics"))
    if version_problem is not None:
        raise ValueError(f"corpus {path}: {version_problem}")

    return corpus
