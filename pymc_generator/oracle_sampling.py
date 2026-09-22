"""Sampling and provenance for the structure-known oracle posterior.

The public entry point in this module deliberately keeps posterior values out of
its receipt.  The returned DataTree is the authoritative posterior result;
the receipt is a small, JSON-native account of how it was requested and what
could be diagnosed from it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any, Literal, cast

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pymc as pm
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor.random.variable import RandomGeneratorSharedVariable
from pytensor.tensor.sharedvar import SharedVariable
from xarray import DataTree

from .worlds import SCM

_DEFAULT_NUTS_SAMPLER: Literal["nutpie", "pymc"] = "nutpie"
_ALLOWED_NUTS_SAMPLERS = frozenset({"nutpie", "pymc"})
_RESERVED_SAMPLER_KWARGS = frozenset(
    {
        "draws",
        "tune",
        "chains",
        "cores",
        "target_accept",
        "random_seed",
        # These are direct Nutpie controls.  Compiled ownership and the
        # validated config must remain the sole source of these values.
        "seed",
        "save_warmup",
        "progress_bar",
        "blocking",
        "sampler",
        "adaptation",
        "init_mean",
        "return_raw_trace",
        "progress_callback",
        "progress_template",
        "progress_style",
        "progress_rate",
        "zarr_store",
        "store_unconstrained",
        "progressbar",
        "compute_convergence_checks",
        "discard_tuned_samples",
        "nuts_sampler",
        "model",
        "step",
        "trace",
        "backend",
        "return_inferencedata",
        "var_names",
    }
)
_LIMITATIONS = (
    "health_is_not_a_convergence_guarantee",
    "oracle_is_a_structure_known_plugin_reference",
    "posterior_draws_are_not_included",
    "step_method_and_mass_matrix_are_not_observed",
)


def _immutable(value: Any, *, path: str = "value") -> Any:
    """Validate a JSON-native value and recursively make it immutable."""
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _immutable(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item, path=f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(f"{path} must be JSON-native")


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _validate_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_nonnegative(value: Any, name: str, *, integer: bool = False) -> None:
    if integer and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{name} must be an integer")
    if not integer and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class OracleSamplingConfig:
    """Validated options for one automatic Nutpie sampling call."""

    draws: int = 800
    tune: int = 500
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.9
    random_seed: int | None = None
    latent: Literal["marginal", "sampled"] = "marginal"
    discard_tuned_samples: bool = True
    progressbar: bool = False
    compute_convergence_checks: bool = False
    nuts_sampler: Literal["nutpie", "pymc"] = _DEFAULT_NUTS_SAMPLER
    sampler_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "sampler_kwargs", _immutable(self.sampler_kwargs))

    def validate(self) -> None:
        for name in ("draws", "tune", "chains", "cores"):
            _validate_nonnegative(getattr(self, name), name, integer=True)
        if self.draws < 1 or self.chains < 1 or self.cores < 1:
            raise ValueError("draws, chains, and cores must be at least one")
        if not isinstance(self.target_accept, (int, float)) or isinstance(self.target_accept, bool):
            raise TypeError("target_accept must be a number")
        if not math.isfinite(float(self.target_accept)) or not 0 < self.target_accept < 1:
            raise ValueError("target_accept must be finite and strictly between zero and one")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
                raise TypeError("random_seed must be a nonnegative integer or None")
            if self.random_seed < 0:
                raise ValueError("random_seed must be nonnegative")
        if self.latent not in ("marginal", "sampled"):
            raise ValueError("latent must be 'marginal' or 'sampled'")
        for name in ("discard_tuned_samples", "progressbar", "compute_convergence_checks"):
            _validate_bool(getattr(self, name), name)
        if not isinstance(self.nuts_sampler, str):
            raise TypeError("nuts_sampler must be a string")
        if self.nuts_sampler not in _ALLOWED_NUTS_SAMPLERS:
            raise ValueError("nuts_sampler must be 'nutpie' or 'pymc'")
        if not isinstance(self.sampler_kwargs, Mapping):
            raise TypeError("sampler_kwargs must be a mapping")
        for key in self.sampler_kwargs:
            if not isinstance(key, str):
                raise TypeError("sampler_kwargs keys must be strings")
            if key in _RESERVED_SAMPLER_KWARGS:
                raise ValueError(f"sampler_kwargs contains reserved key: {key}")
            _immutable(self.sampler_kwargs[key], path=f"sampler_kwargs.{key}")

    def to_dict(self) -> dict[str, object]:
        return {
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores,
            "target_accept": self.target_accept,
            "random_seed": self.random_seed,
            "latent": self.latent,
            "discard_tuned_samples": self.discard_tuned_samples,
            "progressbar": self.progressbar,
            "compute_convergence_checks": self.compute_convergence_checks,
            "nuts_sampler": self.nuts_sampler,
            "sampler_kwargs": _plain(self.sampler_kwargs),
        }


@dataclass(frozen=True, slots=True)
class OracleHealthCriteria:
    """Thresholds and availability requirements for oracle diagnostics."""

    max_divergences: int | None = 0
    max_rhat: float | None = 1.01
    min_ess_bulk: float | None = 400
    min_ess_tail: float | None = 400
    max_tree_depth_saturation: int | None = 0
    min_bfmi: float | None = 0.3
    require_divergences: bool = True
    require_rhat: bool = True
    require_ess_bulk: bool = True
    require_ess_tail: bool = True
    require_tree_depth: bool = False
    require_bfmi: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "max_divergences",
            "max_tree_depth_saturation",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_nonnegative(value, name, integer=True)
        for name in ("max_rhat", "min_ess_bulk", "min_ess_tail", "min_bfmi"):
            value = getattr(self, name)
            if value is not None:
                _validate_nonnegative(value, name)
        if self.max_rhat is not None and self.max_rhat < 1:
            raise ValueError("max_rhat must be at least one")
        for name in (
            "require_divergences",
            "require_rhat",
            "require_ess_bulk",
            "require_ess_tail",
            "require_tree_depth",
            "require_bfmi",
        ):
            _validate_bool(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "max_divergences",
                "max_rhat",
                "min_ess_bulk",
                "min_ess_tail",
                "max_tree_depth_saturation",
                "min_bfmi",
                "require_divergences",
                "require_rhat",
                "require_ess_bulk",
                "require_ess_tail",
                "require_tree_depth",
                "require_bfmi",
            )
        }


@dataclass(frozen=True, slots=True)
class OracleSamplingReceipt:
    """Posterior-free, canonical provenance for one oracle sampling result."""

    schema_version: int
    kind: str
    package_version: str
    environment: Mapping[str, str]
    identities: Mapping[str, Mapping[str, object]]
    oracle: Mapping[str, object]
    sampling: Mapping[str, object]
    timing: Mapping[str, object]
    posterior: Mapping[str, object]
    diagnostics: Mapping[str, object]
    health: Mapping[str, object]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be the integer 1")
        for name in ("kind", "package_version"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        mapping_names = (
            "environment",
            "identities",
            "oracle",
            "sampling",
            "timing",
            "posterior",
            "diagnostics",
            "health",
        )
        for name in mapping_names:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            frozen = _immutable(value, path=name)
            if name == "environment" and any(not isinstance(item, str) for item in frozen.values()):
                raise TypeError("environment values must be strings")
            if name == "identities" and any(
                not isinstance(item, Mapping) for item in frozen.values()
            ):
                raise TypeError("identity values must be mappings")
            object.__setattr__(self, name, frozen)
        if not isinstance(self.limitations, tuple):
            raise TypeError("limitations must be a tuple")
        if any(not isinstance(item, str) for item in self.limitations):
            raise TypeError("limitations must contain only strings")

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            _plain(
                {
                    "schema_version": self.schema_version,
                    "kind": self.kind,
                    "package_version": self.package_version,
                    "environment": self.environment,
                    "identities": self.identities,
                    "oracle": self.oracle,
                    "sampling": self.sampling,
                    "timing": self.timing,
                    "posterior": self.posterior,
                    "diagnostics": self.diagnostics,
                    "health": self.health,
                    "limitations": self.limitations,
                }
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class OracleSamplingResult:
    """The unmodified posterior DataTree and its posterior-free receipt."""

    idata: DataTree
    receipt: OracleSamplingReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.idata, DataTree):
            raise TypeError("idata must be an xarray.DataTree")


def _metric(status: str, value: Any = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if status == "available":
        result["value"] = value
    else:
        result["reason"] = extra.pop("reason")
    result.update(extra)
    return result


def _group(idata: Any, name: str) -> Any:
    try:
        node = idata[name]
        # ArviZ operates on the Dataset held by each DataTree group.  The
        # group itself is still obtained exclusively with bracket access.
        return getattr(node, "ds", node)
    except (AttributeError, KeyError, TypeError):
        return None


def _present_groups(idata: DataTree) -> list[str]:
    """Return every actual descendant group, using DataTree bracket access."""
    groups: list[str] = []

    def visit(node: DataTree, prefix: str = "") -> None:
        for name in node.children:
            child = cast(DataTree, node[name])
            path = f"{prefix}/{name}" if prefix else str(name)
            groups.append(path)
            visit(child, path)

    visit(idata)
    return sorted(groups)


def _scalar_values(value: Any) -> np.ndarray:
    if hasattr(value, "data_vars"):
        arrays = [
            np.asarray(item.values, dtype=float).reshape(-1) for item in value.data_vars.values()
        ]
        return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    return np.asarray(getattr(value, "values", value), dtype=float).reshape(-1)


def _arviz_metric(
    az: Any,
    fn: str,
    posterior: Any,
    method: str,
    reduction: str,
    *,
    report_method: bool = True,
) -> dict[str, Any]:
    try:
        values = _scalar_values(getattr(az, fn)(posterior, method=method))
    except Exception:
        return _metric("invalid", reason=f"arviz_{fn}_failed")
    finite = values[np.isfinite(values)]
    if not finite.size:
        return _metric("invalid", reason="no_finite_result")
    metadata = {"reduction": reduction}
    if report_method:
        metadata["method"] = method
    return _metric(
        "available",
        float(np.max(finite) if reduction == "max" else np.min(finite)),
        **metadata,
    )


def _valid_sampling_stat(stat: Any, sizes: Mapping[str, Any]) -> bool:
    dims = tuple(getattr(stat, "dims", ()))
    stat_sizes = getattr(stat, "sizes", {})
    return (
        dims == ("chain", "draw")
        and int(stat_sizes.get("chain", 0)) > 0
        and int(stat_sizes.get("draw", 0)) > 0
        and (sizes.get("chain") is None or int(stat_sizes["chain"]) == int(sizes["chain"]))
        and (sizes.get("draw") is None or int(stat_sizes["draw"]) == int(sizes["draw"]))
    )


def _diagnostics(idata: DataTree, az: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    posterior = _group(idata, "posterior")
    sample_stats = _group(idata, "sample_stats")
    posterior_present = posterior is not None
    stats_present = sample_stats is not None
    sizes = getattr(posterior, "sizes", {}) if posterior_present else {}
    chains = sizes.get("chain")
    draws = sizes.get("draw")
    diagnostics: dict[str, Any] = {
        "posterior_present": posterior_present,
        "sample_stats_present": stats_present,
        "n_chains": int(chains) if chains is not None else None,
        "draws_per_chain": int(draws) if draws is not None else None,
        "total_draws": int(chains * draws) if chains is not None and draws is not None else None,
    }
    if not stats_present:
        for name in ("divergences", "tree_depth_max", "tree_depth_saturation", "bfmi"):
            diagnostics[name] = _metric("unavailable", reason="sample_stats_group_missing")
    else:
        for field, name in (("diverging", "divergences"), ("tree_depth", "tree_depth_max")):
            if field not in sample_stats:
                diagnostics[name] = _metric("unavailable", reason=f"{field}_missing")
                continue
            try:
                stat = sample_stats[field]
                values = np.asarray(stat.values)
                if not _valid_sampling_stat(stat, sizes):
                    raise ValueError
                if (
                    name == "divergences"
                    and not np.issubdtype(values.dtype, np.bool_)
                    and not np.issubdtype(values.dtype, np.number)
                ):
                    raise ValueError
                finite = values[np.isfinite(values)]
                if finite.size != values.size:
                    diagnostics[name] = _metric("unavailable", reason="non_finite_result")
                elif name == "divergences":
                    diagnostics[name] = _metric("available", int(np.sum(values.astype(bool))))
                else:
                    diagnostics[name] = _metric("available", int(np.max(finite)))
            except Exception:
                diagnostics[name] = _metric("invalid", reason=f"{field}_malformed")
        if "reached_max_treedepth" in sample_stats:
            try:
                stat = sample_stats["reached_max_treedepth"]
                values = np.asarray(stat.values)
                if not _valid_sampling_stat(stat, sizes):
                    raise ValueError
                finite = values[np.isfinite(values)]
                if finite.size != values.size:
                    diagnostics["tree_depth_saturation"] = _metric(
                        "unavailable", reason="non_finite_result"
                    )
                else:
                    diagnostics["tree_depth_saturation"] = _metric(
                        "available",
                        int(np.sum(finite.astype(bool))),
                        source="reached_max_treedepth",
                    )
            except Exception:
                diagnostics["tree_depth_saturation"] = _metric(
                    "invalid", reason="reached_max_treedepth_malformed"
                )
        else:
            diagnostics["tree_depth_saturation"] = _metric(
                "unavailable", reason="reached_max_treedepth_missing"
            )
        if "energy" not in sample_stats:
            diagnostics["bfmi"] = _metric("unavailable", reason="energy_missing")
        else:
            try:
                energy = sample_stats["energy"]
                if not _valid_sampling_stat(energy, sizes):
                    raise ValueError
                if not np.all(np.isfinite(np.asarray(energy.values, dtype=float))):
                    diagnostics["bfmi"] = _metric("unavailable", reason="non_finite_energy")
                else:
                    values = _scalar_values(az.bfmi(sample_stats))
                    finite = values[np.isfinite(values)]
                    diagnostics["bfmi"] = (
                        _metric("available", float(np.min(finite)), reduction="min")
                        if finite.size
                        else _metric("invalid", reason="no_finite_result")
                    )
            except Exception:
                diagnostics["bfmi"] = _metric("invalid", reason="arviz_bfmi_failed")
    if posterior is None or not stats_present:
        reason = "posterior_group_missing" if posterior is None else "sample_stats_group_missing"
        diagnostics["rhat"] = _metric("unavailable", reason=reason)
        diagnostics["ess_bulk"] = _metric("unavailable", reason=reason)
        diagnostics["ess_tail"] = _metric("unavailable", reason=reason)
    else:
        diagnostics["rhat"] = (
            _metric("unavailable", reason="fewer_than_two_chains")
            if chains is None or chains < 2
            else _arviz_metric(az, "rhat", posterior, "rank", "max")
        )
        diagnostics["ess_bulk"] = _arviz_metric(
            az, "ess", posterior, "bulk", "min", report_method=False
        )
        diagnostics["ess_tail"] = _arviz_metric(
            az, "ess", posterior, "tail", "min", report_method=False
        )
    return diagnostics, sizes


def _health(diagnostics: Mapping[str, Any], criteria: OracleHealthCriteria) -> dict[str, Any]:
    threshold_map = {
        "divergences": ("max_divergences", "divergences_exceed_max", lambda v, t: v > t),
        "rhat": ("max_rhat", "rhat_exceeds_max", lambda v, t: v > t),
        "ess_bulk": ("min_ess_bulk", "ess_bulk_below_min", lambda v, t: v < t),
        "ess_tail": ("min_ess_tail", "ess_tail_below_min", lambda v, t: v < t),
        "tree_depth_saturation": (
            "max_tree_depth_saturation",
            "tree_depth_saturation_exceeds_max",
            lambda v, t: v > t,
        ),
        "bfmi": ("min_bfmi", "bfmi_below_min", lambda v, t: v < t),
    }
    required = {
        "divergences": criteria.require_divergences,
        "rhat": criteria.require_rhat,
        "ess_bulk": criteria.require_ess_bulk,
        "ess_tail": criteria.require_ess_tail,
        "tree_depth_saturation": criteria.require_tree_depth,
        "bfmi": criteria.require_bfmi,
    }
    failures = []
    missing = []
    for metric, (threshold_name, code, crosses) in threshold_map.items():
        threshold = getattr(criteria, threshold_name)
        record = diagnostics.get(metric, {"status": "unavailable"})
        if record.get("status") != "available":
            missing.append(metric)
            continue
        if threshold is not None:
            observed = record["value"]
            if crosses(observed, threshold):
                failures.append(
                    {
                        "code": code,
                        "metric": metric,
                        "observed": observed,
                        "threshold": threshold,
                    }
                )
    failures.sort(key=lambda item: (item["code"], item["metric"]))
    if diagnostics.get("tree_depth_max", {}).get("status") != "available":
        missing.append("tree_depth_max")
    missing = sorted(set(missing))
    return {
        "status": "unhealthy"
        if failures
        else ("unknown" if any(required[m] and m in missing for m in required) else "healthy"),
        "criteria": criteria.to_dict(),
        "failure_reasons": failures,
        "missing_diagnostics": missing,
    }


def _versions() -> dict[str, str]:
    names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "xarray": "xarray",
        "arviz": "arviz",
        "pymc": "pymc",
        "pytensor": "pytensor",
        "pymc_marketing": "pymc-marketing",
        "nutpie": "nutpie",
    }
    result = {"python": platform.python_version()}
    for key, package in names.items():
        try:
            result[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[key] = "unavailable"
    return result


def sample_oracle(
    world: SCM,
    config: OracleSamplingConfig | None = None,
    *,
    criteria: OracleHealthCriteria | None = None,
    world_identity: Mapping[str, object] | None = None,
    source_identity: Mapping[str, object] | None = None,
    configuration_identity: Mapping[str, object] | None = None,
    observed_indices: np.ndarray | list[int] | list[bool] | None = None,
) -> OracleSamplingResult:
    """Build and sample a world's observed-data oracle exactly once."""
    config = OracleSamplingConfig() if config is None else config
    criteria = OracleHealthCriteria() if criteria is None else criteria
    if not isinstance(config, OracleSamplingConfig):
        raise TypeError("config must be an OracleSamplingConfig")
    if not isinstance(criteria, OracleHealthCriteria):
        raise TypeError("criteria must be an OracleHealthCriteria")
    config.validate()
    criteria.validate()
    if not callable(getattr(world, "oracle_model", None)):
        raise TypeError("world must provide a callable oracle_model")
    identities = {}
    for name, identity in (
        ("world", world_identity),
        ("source", source_identity),
        ("configuration", configuration_identity),
    ):
        if identity is not None and not isinstance(identity, Mapping):
            raise TypeError(f"{name}_identity must be a mapping or None")
        identities[name] = _immutable({} if identity is None else identity, path=f"{name}_identity")

    started = time.monotonic()
    if observed_indices is None:
        model = world.oracle_model(latent=config.latent)
    else:
        model = world.oracle_model(latent=config.latent, observed_indices=observed_indices)
    idata = pm.sample(
        model=model,
        draws=config.draws,
        tune=config.tune,
        chains=config.chains,
        cores=config.cores,
        target_accept=config.target_accept,
        random_seed=config.random_seed,
        discard_tuned_samples=config.discard_tuned_samples,
        progressbar=config.progressbar,
        compute_convergence_checks=config.compute_convergence_checks,
        nuts_sampler=config.nuts_sampler,
        **_mutable(config.sampler_kwargs),
    )
    if not isinstance(idata, DataTree):
        raise TypeError("pm.sample must return an xarray.DataTree")
    diagnostics, sizes = _diagnostics(idata, az)
    health = _health(diagnostics, criteria)
    posterior = _group(idata, "posterior")
    n_chains = sizes.get("chain")
    draws = sizes.get("draw")
    requested = config.to_dict()
    requested.pop("latent")
    sampling = {
        "requested": requested,
        "effective": {
            "groups": _present_groups(idata),
            "n_chains": int(n_chains) if n_chains is not None else None,
            "draws_per_chain": int(draws) if draws is not None else None,
            "sample_stat_names": sorted(
                str(name) for name in getattr(_group(idata, "sample_stats"), "data_vars", {})
            ),
            "inference_library": "pymc",
            "inference_library_version": _versions()["pymc"],
            "step_methods": {"status": "unavailable", "reason": "not_exposed_by_sampling_result"},
            "mass_matrix": {"status": "unavailable", "reason": "not_exposed_by_sampling_result"},
        },
    }
    try:
        package_version = metadata.version("pymc-generator")
    except metadata.PackageNotFoundError:
        from ._version import __version__

        package_version = __version__
    receipt = OracleSamplingReceipt(
        schema_version=1,
        kind="pymc_generator.oracle_sampling_receipt",
        package_version=package_version,
        environment=_immutable(_versions()),
        identities=_immutable(identities),
        oracle=_immutable({"builder": "SCM.oracle_model", "latent": config.latent}),
        sampling=_immutable(sampling),
        timing=_immutable({"elapsed_seconds": float(time.monotonic() - started)}),
        posterior=_immutable({"present": posterior is not None, "included_in_receipt": False}),
        diagnostics=_immutable(diagnostics),
        health=_immutable(health),
        limitations=_LIMITATIONS,
    )
    return OracleSamplingResult(idata=idata, receipt=receipt)


# Reusable compiled-oracle API ---------------------------------------------

# These names intentionally mirror the recovered producer.  The first seven
# are the historical shared inputs; the selector is the maintained extension.
ORACLE_SHARED_DATA_NAMES = (
    "channels_data",
    "controls_data",
    "sales_data",
    "saturation_scale_data",
    "g_cy_data",
    "g_db_data",
    "g_zb_data",
    "observed_indices_data",
)
# Numeric values which are intentionally runtime data.  Their shapes and
# distribution families are part of the graph contract; their values are not.
ORACLE_DYNAMIC_DATA_NAMES = (
    "prior_cond_carryover_alpha_data",
    "prior_cond_hill_shape_data",
    "walk_width_d_data",
    "walk_width_b_data",
)
ORACLE_DATA_NAMES = ORACLE_SHARED_DATA_NAMES + ORACLE_DYNAMIC_DATA_NAMES

# ``_world_signature`` must not accidentally become a hash of every SCMPrior
# field.  The first group is graph topology/rank, while the second group is
# consumed while constructing numerical RV bounds and is therefore fixed for a
# compiled template.  The latter is deliberately not silently recompiled: a
# changed value is an incompatibility unless it has a named pm.Data above.
ORACLE_CONFIG_TOPOLOGY_FIELDS = (
    "n_treatments",
    "n_covariates",
    "n_latent",
    "n_time_steps",
    "l_max",
    "carryover_burn_in",
    "rw_smoothness_max_weeks",
    "outcome_std_mode",
    "baseline_floor",
    "baseline_floor_scope",
)
ORACLE_CONFIG_FIXED_NUMERIC_FIELDS = (
    "carryover_alpha_range",
    "weibull_lam_range",
    "weibull_k_range",
    "rw_covariate_mean_range",
    "rw_std_sigma",
    "rw_positive_mean_range",
    "rw_treatment_std_range",
    "rw_baseline_mean_range",
    "rw_baseline_std_sigma",
    "rw_baseline_std_sigma_effective",
    "rw_baseline_std_range",
    "rw_outcome_std_sigma",
    "rw_outcome_std_range",
    "dc_coeff_range",
    "dz_coeff_range",
    "zc_coeff_range",
    "cc_coeff_range",
    "zz_coeff_range",
    "dy_coeff_range",
    "zy_coeff_range",
    "beta_additive_range",
    # These ranges are read while constructing the shared prior-spec table,
    # even when this world's selected mechanism does not use their RVs.
    "treatment_hf_sigma_range",
    "treatment_pulse_prob_range",
    "treatment_pulse_amp_range",
    "covariate_hf_sigma_range",
    "covariate_pulse_prob_range",
    "covariate_pulse_amp_range",
)
# Shock metadata is validated at the oracle builder seam but does not create an
# Oracle RV.  It is still pinned in the template contract so changing the
# validation domain cannot silently reuse a compiled template.
ORACLE_CONFIG_VALIDATION_FIELDS = (
    "n_treatment_shocks",
    "treatment_shock_length_range",
    "treatment_shock_level_range",
)
ORACLE_CONFIG_SCHEMA = {
    "topology": ORACLE_CONFIG_TOPOLOGY_FIELDS,
    "fixed_numeric": ORACLE_CONFIG_FIXED_NUMERIC_FIELDS,
    "runtime_data": ORACLE_DYNAMIC_DATA_NAMES,
    "validation_only": ORACLE_CONFIG_VALIDATION_FIELDS,
}


def _array_identity(value: Any) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
    }


def _signature_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, Mapping):
        return {
            str(key): _signature_value(item)
            for key, item in sorted(value.items(), key=lambda p: str(p[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_signature_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalise_observed_indices(
    observed_indices: np.ndarray | list[int] | list[bool] | None,
    n_reported_rows: int,
) -> np.ndarray:
    if observed_indices is None:
        result = np.arange(n_reported_rows, dtype="int64")
    else:
        raw = np.asarray(observed_indices)
        if raw.ndim != 1:
            raise ValueError("observed_indices must be one-dimensional")
        if np.issubdtype(raw.dtype, np.bool_):
            if raw.size != n_reported_rows:
                raise ValueError("observed_indices boolean mask has the wrong length")
            result = np.flatnonzero(raw).astype("int64")
        elif np.issubdtype(raw.dtype, np.integer):
            result = raw.astype("int64", copy=True)
        else:
            raise TypeError("observed_indices must contain integers or booleans")
    if result.size == 0:
        raise ValueError("observed_indices must select at least one reported row")
    if np.any(result < 0) or np.any(result >= n_reported_rows):
        raise ValueError("observed_indices contains an out-of-range reported row")
    if np.unique(result).size != result.size:
        raise ValueError("observed_indices must not contain duplicates")
    return result


def _signature_structural(value: Any) -> Any:
    """Return graph choices while leaving compatible numeric values as data."""
    if isinstance(value, Mapping):
        return {
            str(key): _signature_structural(item)
            for key, item in value.items()
            if not str(key).startswith("smoothness_")
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_signature_structural(item) for item in np.asarray(value).tolist()]
    return value


def _oracle_config_contract(cfg: Any) -> dict[str, Any]:
    """Capture config values consumed by the Oracle graph.

    Numeric fields are a compatibility contract, not signature inputs: putting
    them in the signature would make a changed value look like a topology
    change and encourage accidental recompilation.  Runtime-varying values
    (prior intervals and walk widths) are represented by ``pm.Data`` instead.
    """
    if cfg is None:
        return {}
    fields = (
        *ORACLE_CONFIG_TOPOLOGY_FIELDS,
        *ORACLE_CONFIG_FIXED_NUMERIC_FIELDS,
        *ORACLE_CONFIG_VALIDATION_FIELDS,
    )
    return {name: _signature_value(getattr(cfg, name)) for name in fields if hasattr(cfg, name)}


def _validate_oracle_config_contract(world: Any, expected: Mapping[str, Any]) -> None:
    if not expected:
        return
    actual = _oracle_config_contract(getattr(world, "cfg", None))
    for name, value in expected.items():
        if name not in actual:
            raise ValueError(
                f"incompatible Oracle config: field {name!r} is missing from the world"
            )
        if actual[name] != value:
            raise ValueError(
                f"incompatible Oracle config field {name!r}: changing this numerical "
                "graph-construction value is unsupported; use a new Oracle template"
            )


def _signature_warmup(world: Any) -> int:
    """Resolve the response-history rank implied by this world's prior bounds."""
    cfg = getattr(world, "cfg", None)
    if cfg is None or int(getattr(cfg, "carryover_burn_in", 0)) <= 0:
        return 0
    structural = getattr(world, "extras", {}).get("structural", {})
    prior_cond = getattr(world, "extras", {}).get("prior_cond")
    alpha_range = cfg.carryover_alpha_range
    if prior_cond is not None and "carryover_alpha" in prior_cond:
        lo, width = prior_cond["carryover_alpha"]
        alpha_range = (float(lo), float(lo) + float(width))
    from .signal_diagnostics import admitted_response_support_weeks

    families = np.asarray(structural.get("carryover_family", ()))
    g_cy = np.asarray(getattr(world, "g", {}).get("g_cy", ()))
    direct = families[g_cy != 0.0]
    return int(
        admitted_response_support_weeks(
            direct,
            cfg.l_max,
            carryover_alpha_range=alpha_range,
        )
    )


def _world_signature(world: Any, *, latent: str, observed_count: int) -> str:
    """Hash topology/rank choices, excluding compatible data values."""
    cfg = getattr(world, "cfg", None)
    structural = getattr(world, "extras", {}).get("structural", {})
    graph = getattr(world, "g", {})
    data = getattr(world, "data", {})
    # Only topology/rank/shape/family choices belong in this hash.  In
    # particular, prior interval endpoints and smoothness values are runtime
    # data; fixed config numerics are checked separately at fit time.
    cfg_fields = {}
    if cfg is not None:
        for name in ORACLE_CONFIG_TOPOLOGY_FIELDS:
            if not hasattr(cfg, name):
                continue
            value = getattr(cfg, name)
            # The floor's numerical value is an embedded constant and is
            # checked by the config contract; only its presence is topology.
            cfg_fields[name] = (value is not None) if name == "baseline_floor" else value
    prior_cond = getattr(world, "extras", {}).get("prior_cond") or {}
    prior_topology = {}
    for key, value in prior_cond.items():
        # Intervals (including the unconditioned support) are runtime data and
        # therefore intentionally absent from the signature.  Point masses
        # have a different RV topology and are recorded explicitly.
        if (
            isinstance(value, (tuple, list, np.ndarray))
            and len(value) == 2
            and float(value[1]) == 0.0
        ):
            prior_topology[str(key)] = "point_mass"
    payload = {
        "latent": latent,
        "observed_count": int(observed_count),
        "response_warmup": _signature_warmup(world),
        "prior_conditioning_topology": prior_topology,
        "graph_shapes": {name: np.asarray(value).shape for name, value in graph.items()},
        "graph_topology": {
            name: np.asarray(value).astype(float).tolist()
            for name, value in graph.items()
            if name in ("g_cy", "g_dy", "g_zy")
        },
        # Oracle graph construction consumes only mechanism families from the
        # sampled structure. Walk smoothness is runtime data; texture flags and
        # other generation-only metadata must not split reusable templates.
        "structural": {
            name: _signature_structural(structural[name])
            for name in ("carryover_family", "sat_family")
            if name in structural
        },
        "config_topology": cfg_fields,
        "n_time_steps": int(np.asarray(data["outcome"]).shape[0]),
    }
    encoded = json.dumps(_signature_value(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dynamic_payload(world: Any, names: set[str]) -> dict[str, np.ndarray]:
    """Resolve compatible numeric prior/walk values for a model's pm.Data."""
    extras = getattr(world, "extras", {})
    prior = extras.get("prior_cond") or {}
    supported = {"carryover_alpha", "hill_shape"}
    if set(prior) - supported:
        raise ValueError(f"unsupported prior conditioning keys: {sorted(set(prior) - supported)!r}")
    result: dict[str, np.ndarray] = {}
    for quantity, name in (
        ("carryover_alpha", "prior_cond_carryover_alpha_data"),
        ("hill_shape", "prior_cond_hill_shape_data"),
    ):
        if name not in names:
            continue
        if quantity in prior:
            raw = prior[quantity]
            if (
                isinstance(raw, (tuple, list, np.ndarray))
                and len(raw) == 2
                and float(raw[1]) == 0.0
            ):
                raise ValueError(
                    f"incompatible prior conditioning for {quantity!r}: zero-width "
                    "point masses are unsupported by reusable Oracle templates"
                )
        cfg = world.cfg
        if quantity == "carryover_alpha":
            support = cfg.carryover_alpha_range
        else:
            from . import mechanisms

            support = mechanisms.SATURATION_PRIOR_RANGES["hill"]["slope"]
        if quantity in prior:
            raw = prior[quantity]
            if not isinstance(raw, (tuple, list, np.ndarray)) or len(raw) != 2:
                raise ValueError(f"unsupported prior conditioning for {quantity!r}")
            lo, width = float(raw[0]), float(raw[1])
            if (
                not np.isfinite([lo, width]).all()
                or width < 0
                or lo < support[0]
                or lo + width > support[1]
            ):
                raise ValueError(f"incompatible prior conditioning for {quantity!r}")
            result[name] = np.asarray([1.0, lo, width], dtype="float64")
        else:
            result[name] = np.asarray([0.0, support[0], support[1] - support[0]], dtype="float64")
    if "walk_width_d_data" in names or "walk_width_b_data" in names:
        from .random_walk import walk_width_index

        structural = extras.get("structural", {})
        n_full = int(np.asarray(world.data["outcome"]).shape[0]) + int(world.cfg.carryover_burn_in)
        for key, name in (
            ("smoothness_d", "walk_width_d_data"),
            ("smoothness_b", "walk_width_b_data"),
        ):
            if name in names:
                values = np.asarray(structural.get(key, ()), dtype="float64")
                result[name] = walk_width_index(
                    values,
                    n_full,
                    rw_smoothness_max_weeks=int(world.cfg.rw_smoothness_max_weeks),
                )
    return result


def _required_dynamic_names(world: Any) -> set[str]:
    extras = getattr(world, "extras", {})
    prior = extras.get("prior_cond") or {}
    required = {
        {
            "carryover_alpha": "prior_cond_carryover_alpha_data",
            "hill_shape": "prior_cond_hill_shape_data",
        }[key]
        for key in prior
        if key in {"carryover_alpha", "hill_shape"}
    }
    structural = extras.get("structural", {})
    required.update(
        name
        for key, name in (
            ("smoothness_d", "walk_width_d_data"),
            ("smoothness_b", "walk_width_b_data"),
        )
        if key in structural
    )
    return required


def _oracle_payload(
    world: Any, observed_indices: np.ndarray, *, dynamic_names: set[str] | None = None
) -> dict[str, np.ndarray]:
    try:
        data = world.data
        graph = world.g
        payload = {
            # Own each buffer before handing it to Nutpie. A bound compiled
            # view must not alias a mutable SCM payload across sequential fits.
            "channels_data": np.array(data["treatments"], copy=True),
            "controls_data": np.array(data["covariates"], copy=True),
            "sales_data": np.array(data["outcome"], copy=True),
            "saturation_scale_data": np.array(data["saturation_scale"], copy=True),
            "g_cy_data": np.array(graph["g_cy"], copy=True),
            "g_db_data": np.array(graph["g_dy"], copy=True),
            "g_zb_data": np.array(graph["g_zy"], copy=True),
            "observed_indices_data": np.array(observed_indices, dtype="int64", copy=True),
        }
    except (AttributeError, KeyError) as error:
        raise TypeError("world does not expose the required Oracle data") from error
    if dynamic_names:
        payload.update(_dynamic_payload(world, dynamic_names))
    return payload


def _receipt_for_compiled_fit(
    idata: DataTree,
    *,
    config: OracleSamplingConfig,
    criteria: OracleHealthCriteria,
    started: float,
    template_signature: str,
    compile_identity: Mapping[str, object],
    observed_indices: np.ndarray,
    payload: Mapping[str, np.ndarray],
    world_identity: Mapping[str, object] | None,
    source_identity: Mapping[str, object] | None,
    configuration_identity: Mapping[str, object] | None,
) -> OracleSamplingResult:
    diagnostics, sizes = _diagnostics(idata, az)
    health = _health(diagnostics, criteria)
    posterior = _group(idata, "posterior")
    versions = _versions()
    try:
        package_version = metadata.version("pymc-generator")
    except metadata.PackageNotFoundError:
        from ._version import __version__

        package_version = __version__
    n_chains, draws = sizes.get("chain"), sizes.get("draw")
    requested = config.to_dict()
    requested.pop("latent")
    identity_values = {
        "world": {} if world_identity is None else world_identity,
        "source": {} if source_identity is None else source_identity,
        "configuration": {} if configuration_identity is None else configuration_identity,
        "template": {"signature": template_signature},
        "compile": dict(compile_identity),
        "observed_indices": _array_identity(observed_indices),
        "data": {name: _array_identity(value) for name, value in sorted(payload.items())},
    }
    receipt = OracleSamplingReceipt(
        schema_version=1,
        kind="pymc_generator.oracle_compiled_sampling_receipt",
        package_version=package_version,
        environment=_immutable(versions),
        identities=_immutable(identity_values),
        oracle=_immutable(
            {
                "builder": "SCM.oracle_model",
                "latent": config.latent,
                "observed_indices": "reported_rows",
            }
        ),
        sampling=_immutable(
            {
                "requested": requested,
                "effective": {
                    "groups": _present_groups(idata),
                    "n_chains": int(n_chains) if n_chains is not None else None,
                    "draws_per_chain": int(draws) if draws is not None else None,
                    "sample_stat_names": sorted(
                        str(name)
                        for name in getattr(_group(idata, "sample_stats"), "data_vars", {})
                    ),
                    "inference_library": "nutpie",
                    "inference_library_version": versions.get("nutpie", "unavailable"),
                    "step_methods": {"status": "unavailable", "reason": "not_exposed"},
                    "mass_matrix": {"status": "unavailable", "reason": "not_exposed"},
                },
                "compile": dict(compile_identity),
            }
        ),
        timing=_immutable({"elapsed_seconds": float(time.monotonic() - started)}),
        posterior=_immutable({"present": posterior is not None, "included_in_receipt": False}),
        diagnostics=_immutable(diagnostics),
        health=_immutable(health),
        limitations=_LIMITATIONS + ("compiled_model_ownership_is_process_local",),
    )
    return OracleSamplingResult(idata=idata, receipt=receipt)


def _graph_shared_variables(model: pm.Model) -> tuple[SharedVariable, ...]:
    """Return non-RNG shared leaves reachable from the model's PyTensor graph."""
    outputs: list[Any] = []
    for attr in ("basic_RVs", "observed_RVs", "deterministics", "potentials"):
        outputs.extend(getattr(model, attr, ()))
    logp = model.logp()
    outputs.extend(logp if isinstance(logp, list) else (logp,))
    return tuple(
        variable
        for variable in graph_inputs(outputs)
        if isinstance(variable, SharedVariable)
        and not isinstance(variable, RandomGeneratorSharedVariable)
    )


@dataclass(frozen=True, slots=True)
class OracleTemplate:
    """Full-horizon graph and its explicit reusable structural signature."""

    model: pm.Model
    signature: str
    latent: Literal["marginal", "sampled"]
    observed_indices: tuple[int, ...]
    shared_names: tuple[str, ...] = ORACLE_SHARED_DATA_NAMES
    reported_rows: int | None = None
    data_contract: tuple[str, ...] | None = None
    config_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.shared_names != ORACLE_SHARED_DATA_NAMES:
            raise ValueError("OracleTemplate shared variable names are part of its contract")
        if self.data_contract is None:
            raise ValueError("OracleTemplate requires an explicit data_contract")
        contract = tuple(self.data_contract)
        if len(set(contract)) != len(contract) or not set(self.shared_names) <= set(contract):
            raise ValueError("OracleTemplate data contract must include required shared variables")
        if not set(contract) <= set(ORACLE_DATA_NAMES):
            unexpected = sorted(set(contract) - set(ORACLE_DATA_NAMES))
            raise ValueError(f"OracleTemplate data contract has unknown variables: {unexpected}")
        object.__setattr__(self, "data_contract", contract)
        object.__setattr__(
            self,
            "config_contract",
            {} if self.config_contract is None else dict(self.config_contract),
        )
        if self.latent not in ("marginal", "sampled"):
            raise ValueError("OracleTemplate latent must be 'marginal' or 'sampled'")
        if not self.observed_indices:
            raise ValueError("OracleTemplate must select at least one observed row")
        reported_rows = (
            len(self.observed_indices) if self.reported_rows is None else self.reported_rows
        )
        if not isinstance(reported_rows, int) or reported_rows < len(self.observed_indices):
            raise ValueError("reported_rows must be an integer covering observed_indices")
        if any(index < 0 or index >= reported_rows for index in self.observed_indices):
            raise ValueError("observed_indices contains an out-of-range reported row")
        object.__setattr__(self, "reported_rows", reported_rows)
        missing = [
            name
            for name in contract
            if not isinstance(self.model.named_vars.get(name), SharedVariable)
        ]
        extras = [
            name
            for name, variable in self.model.named_vars.items()
            if isinstance(variable, SharedVariable) and name not in contract
        ]
        if missing:
            raise ValueError(f"oracle template is missing required shared variables: {missing}")
        if extras:
            raise ValueError(f"oracle template has data outside its explicit contract: {extras}")

        # ``named_vars`` is only a registry: raw ``pytensor.shared`` leaves
        # connected to a likelihood (and unnamed leaves) need not appear there.
        graph_shared = _graph_shared_variables(self.model)
        graph_by_name: dict[str, SharedVariable] = {}
        duplicate_names: set[str] = set()
        unnamed = 0
        for variable in graph_shared:
            name = variable.name
            if name is None:
                unnamed += 1
                continue
            if name in graph_by_name and graph_by_name[name] is not variable:
                duplicate_names.add(name)
            graph_by_name[name] = variable
        if unnamed:
            raise ValueError(
                "oracle template has unnamed shared variables outside its explicit contract"
            )
        if duplicate_names:
            raise ValueError(
                "oracle template has duplicate shared variable names in its graph: "
                f"{sorted(duplicate_names)}"
            )
        graph_extras = sorted(set(graph_by_name) - set(contract))
        if graph_extras:
            raise ValueError(
                "oracle template has graph-connected data outside its explicit contract: "
                f"{graph_extras}"
            )
        mismatched = sorted(
            name
            for name, variable in graph_by_name.items()
            if self.model.named_vars.get(name) is not variable
        )
        if mismatched:
            raise ValueError(
                "oracle template has shared variables whose graph object does not match "
                f"the declared contract: {mismatched}"
            )
        bad_names = sorted(
            name
            for name, variable in self.model.named_vars.items()
            if isinstance(variable, SharedVariable) and variable.name != name
        )
        if bad_names:
            raise ValueError(
                f"oracle template has unnamed or mismatched shared declarations: {bad_names}"
            )


@dataclass(slots=True)
class CompiledOracle:
    """One process-local Nutpie compilation, safely reused for sequential fits."""

    template: OracleTemplate
    compiled: Any
    sampler: Any | None = None
    backend: str = "numba"
    _owner_pid: int = field(default_factory=os.getpid, init=False, repr=False)
    _instance_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend != "numba":
            raise ValueError("the maintained compiled Oracle backend is numba")
        missing = [
            name
            for name in self.template.data_contract or ()
            if not isinstance(self.template.model.named_vars.get(name), SharedVariable)
        ]
        if missing:
            raise ValueError(f"compiled oracle missing required shared variables: {missing}")

    def fit(
        self,
        world: SCM,
        config: OracleSamplingConfig | None = None,
        *,
        criteria: OracleHealthCriteria | None = None,
        observed_indices: np.ndarray | list[int] | list[bool] | None = None,
        world_identity: Mapping[str, object] | None = None,
        source_identity: Mapping[str, object] | None = None,
        configuration_identity: Mapping[str, object] | None = None,
    ) -> OracleSamplingResult:
        config = OracleSamplingConfig() if config is None else config
        criteria = OracleHealthCriteria() if criteria is None else criteria
        if not isinstance(config, OracleSamplingConfig) or not isinstance(
            criteria, OracleHealthCriteria
        ):
            raise TypeError("config and criteria must use the Oracle sampling types")
        config.validate()
        criteria.validate()
        if config.latent != self.template.latent:
            raise ValueError("config.latent must match the compiled Oracle template latent mode")
        if config.nuts_sampler != "nutpie":
            raise ValueError("CompiledOracle requires nuts_sampler='nutpie'")
        for name, identity in (
            ("world", world_identity),
            ("source", source_identity),
            ("configuration", configuration_identity),
        ):
            if identity is not None and not isinstance(identity, Mapping):
                raise TypeError(f"{name}_identity must be a mapping or None")
            if identity is not None:
                _immutable(identity, path=f"{name}_identity")
        if observed_indices is None:
            indices = np.asarray(self.template.observed_indices, dtype="int64")
        else:
            indices = _normalise_observed_indices(
                observed_indices, self.template.reported_rows or 0
            )
        if len(indices) != len(self.template.observed_indices):
            raise ValueError("a compiled Oracle cannot change the observed-index rank")
        _validate_oracle_config_contract(world, self.template.config_contract or {})
        expected_signature = _world_signature(
            world, latent=self.template.latent, observed_count=len(indices)
        )
        if expected_signature != self.template.signature:
            raise ValueError(
                "world structural signature does not match the compiled Oracle template"
            )
        contract = self.template.data_contract or ORACLE_SHARED_DATA_NAMES
        dynamic_names = set(contract) - set(ORACLE_SHARED_DATA_NAMES)
        missing_dynamic = _required_dynamic_names(world) - dynamic_names
        if missing_dynamic:
            raise ValueError(
                "Oracle template data contract is missing required runtime inputs: "
                + repr(sorted(missing_dynamic))
            )
        payload = _oracle_payload(world, indices, dynamic_names=dynamic_names)
        if set(payload) != set(contract):
            raise ValueError("Oracle data payload does not exactly match the required shared set")
        for name in ORACLE_SHARED_DATA_NAMES + tuple(sorted(dynamic_names)):
            variable = self.template.model.named_vars[name]
            expected = np.asarray(variable.get_value())
            shape = tuple(int(size) for size in expected.shape)
            if payload[name].shape != shape:
                raise ValueError(
                    f"{name} shape {payload[name].shape} does not match template {shape}"
                )
            if payload[name].dtype != expected.dtype:
                if name != "observed_indices_data" or not np.issubdtype(expected.dtype, np.integer):
                    raise ValueError(
                        f"{name} dtype {payload[name].dtype} does not match template {expected.dtype}"
                    )
                limits = np.iinfo(expected.dtype)
                if np.any(payload[name] < limits.min) or np.any(payload[name] > limits.max):
                    raise ValueError(f"{name} values do not fit template dtype {expected.dtype}")
                payload[name] = payload[name].astype(expected.dtype, copy=True)
        if not callable(self.sampler):
            raise TypeError("CompiledOracle requires the injected Nutpie sampling callable")
        if os.getpid() != self._owner_pid:
            raise RuntimeError("CompiledOracle is process-local and cannot be used after fork")
        started = time.monotonic()
        # Nutpie's with_data returns a new bound compiled model; it does not
        # recompile. The lock also prevents accidental sharing of a bound view.
        with self._lock:
            bound = self.compiled.with_data(**payload)
            kwargs = {
                "draws": config.draws,
                "tune": config.tune,
                "chains": config.chains,
                "cores": config.cores,
                "target_accept": config.target_accept,
                "seed": config.random_seed,
                "save_warmup": not config.discard_tuned_samples,
                "progress_bar": config.progressbar,
                "blocking": True,
            }
            kwargs.update(_mutable(config.sampler_kwargs))
            idata = self.sampler(bound, **kwargs)
        if not isinstance(idata, DataTree):
            raise TypeError("Nutpie compiled sampling must return an xarray.DataTree")
        return _receipt_for_compiled_fit(
            idata,
            config=config,
            criteria=criteria,
            started=started,
            template_signature=self.template.signature,
            compile_identity={
                "engine": "nutpie",
                "backend": self.backend,
                "instance_id": self._instance_id,
                "owner_pid": self._owner_pid,
            },
            observed_indices=indices,
            payload=payload,
            world_identity=world_identity,
            source_identity=source_identity,
            configuration_identity=configuration_identity,
        )


def build_oracle_template(
    world: SCM,
    *,
    latent: Literal["marginal", "sampled"] = "marginal",
    observed_indices: np.ndarray | list[int] | list[bool] | None = None,
) -> OracleTemplate:
    """Build one full-horizon Oracle graph without compiling or sampling.

    Observation selectors are expressed in the reported-row coordinate system,
    which excludes the model's carryover warmup.  Build the default graph first
    to obtain that domain from ``observed_indices_data``; this avoids guessing
    warmup from private configuration details and fixes non-zero-warmup worlds.
    """
    prior_cond = getattr(world, "extras", {}).get("prior_cond") or {}
    if any(
        isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2 and float(value[1]) == 0.0
        for value in prior_cond.values()
    ):
        raise ValueError(
            "reusable Oracle templates do not support zero-width prior point masses; "
            "use a nonzero interval or build a one-shot model"
        )
    model = world.oracle_model(latent=latent)
    try:
        default_indices_var = model.named_vars["observed_indices_data"]
        default_indices = np.asarray(default_indices_var.get_value(), dtype="int64")
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("oracle_model must expose observed_indices_data as shared data") from error
    n_reported_rows = int(default_indices.size)
    if n_reported_rows < 1:
        raise ValueError("oracle model must expose at least one reported row")
    if observed_indices is None:
        indices = default_indices
    else:
        indices = _normalise_observed_indices(observed_indices, n_reported_rows)
        if not np.array_equal(indices, default_indices):
            model = world.oracle_model(latent=latent, observed_indices=indices)
    signature = _world_signature(world, latent=latent, observed_count=len(indices))
    contract = tuple(name for name in ORACLE_DATA_NAMES if name in model.named_vars)
    if set(contract) != set(ORACLE_DATA_NAMES):
        missing = sorted(set(ORACLE_DATA_NAMES) - set(contract))
        raise ValueError(f"oracle model data contract is incomplete; missing {missing}")
    return OracleTemplate(
        model=model,
        signature=signature,
        latent=latent,
        observed_indices=tuple(int(index) for index in indices),
        reported_rows=n_reported_rows,
        data_contract=contract,
        config_contract=_oracle_config_contract(getattr(world, "cfg", None)),
    )


def compile_oracle(template: OracleTemplate | SCM, **kwargs: Any) -> CompiledOracle:
    """Compile exactly once with Nutpie/Numba; use :meth:`CompiledOracle.fit` thereafter."""
    if isinstance(template, SCM):
        template = build_oracle_template(template, **kwargs)
    elif kwargs:
        raise TypeError("kwargs are only accepted when compiling directly from an SCM")
    if not isinstance(template, OracleTemplate):
        raise TypeError("template must be an OracleTemplate or SCM")
    try:
        import nutpie  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError("compile_oracle requires the optional nutpie dependency") from error
    compiled = nutpie.compile_pymc_model(template.model, backend="numba")
    return CompiledOracle(template=template, compiled=compiled, sampler=nutpie.sample)


# Explicit aliases make the lifecycle discoverable while retaining a compact name.
OracleCompiledModel = CompiledOracle
OracleModelTemplate = OracleTemplate


__all__ = [
    "ORACLE_SHARED_DATA_NAMES",
    "ORACLE_DYNAMIC_DATA_NAMES",
    "ORACLE_DATA_NAMES",
    "ORACLE_CONFIG_SCHEMA",
    "OracleHealthCriteria",
    "OracleSamplingConfig",
    "OracleSamplingReceipt",
    "OracleSamplingResult",
    "OracleTemplate",
    "OracleCompiledModel",
    "CompiledOracle",
    "OracleModelTemplate",
    "build_oracle_template",
    "compile_oracle",
    "sample_oracle",
]
