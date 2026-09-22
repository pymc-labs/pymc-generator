"""Sampling and provenance for the structure-known oracle posterior.

The public entry point in this module deliberately keeps posterior values out of
its receipt.  The returned DataTree is the authoritative posterior result;
the receipt is a small, JSON-native account of how it was requested and what
could be diagnosed from it.
"""

from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

_NUTS_SAMPLER = "nutpie"
_RESERVED_SAMPLER_KWARGS = frozenset(
    {
        "draws",
        "tune",
        "chains",
        "cores",
        "target_accept",
        "random_seed",
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

    draws: int = 500
    tune: int = 500
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.9
    random_seed: int | None = None
    latent: Literal["marginal", "sampled"] = "marginal"
    discard_tuned_samples: bool = True
    progressbar: bool = False
    compute_convergence_checks: bool = False
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
        for name in (
            "environment",
            "identities",
            "oracle",
            "sampling",
            "timing",
            "posterior",
            "diagnostics",
            "health",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name), path=name))
        limitations = tuple(self.limitations)
        if any(not isinstance(item, str) for item in limitations):
            raise TypeError("limitations must contain only strings")
        object.__setattr__(self, "limitations", limitations)

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

    idata: Any
    receipt: OracleSamplingReceipt


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


def _present_groups(idata: Any) -> list[str]:
    # DataTree group access is intentionally bracket-only.  These are the
    # groups emitted by PyMC's InferenceData/DataTree contract.
    known = ("posterior", "sample_stats", "prior", "prior_predictive", "observed_data", "constant_data")
    return sorted(name for name in known if _group(idata, name) is not None)


def _scalar_values(value: Any) -> np.ndarray:
    if hasattr(value, "data_vars"):
        arrays = [np.asarray(item.values, dtype=float).reshape(-1) for item in value.data_vars.values()]
        return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    return np.asarray(getattr(value, "values", value), dtype=float).reshape(-1)


def _arviz_metric(az: Any, fn: str, posterior: Any, method: str, reduction: str) -> dict[str, Any]:
    try:
        values = _scalar_values(getattr(az, fn)(posterior, method=method))
    except Exception:
        return _metric("invalid", reason=f"arviz_{fn}_failed")
    finite = values[np.isfinite(values)]
    if not finite.size:
        return _metric("invalid", reason="no_finite_result")
    return _metric("available", float(np.max(finite) if reduction == "max" else np.min(finite)), **{"method": method, "reduction": reduction})


def _diagnostics(idata: Any, az: Any) -> tuple[dict[str, Any], dict[str, Any]]:
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
                dims = set(getattr(stat, "dims", ()))
                values = np.asarray(stat.values)
                if values.size == 0 or not {"chain", "draw"}.issubset(dims):
                    raise ValueError
                if (
                    name == "divergences"
                    and not np.issubdtype(values.dtype, np.bool_)
                    and not np.issubdtype(values.dtype, np.number)
                ):
                    raise ValueError
                finite = values[np.isfinite(values)]
                if finite.size != values.size:
                    diagnostics[name] = _metric("invalid", reason="non_finite_result")
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
                if not {"chain", "draw"}.issubset(set(getattr(stat, "dims", ()) )):
                    raise ValueError
                finite = values[np.isfinite(values)]
                if finite.size != values.size:
                    diagnostics["tree_depth_saturation"] = _metric(
                        "invalid", reason="non_finite_result"
                    )
                else:
                    diagnostics["tree_depth_saturation"] = _metric(
                        "available", int(np.sum(finite.astype(bool))), source="reached_max_treedepth"
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
                values = _scalar_values(az.bfmi(sample_stats))
                finite = values[np.isfinite(values)]
                diagnostics["bfmi"] = (
                    _metric("available", float(np.min(finite)), reduction="min")
                    if finite.size
                    else _metric("invalid", reason="no_finite_result")
                )
            except Exception:
                diagnostics["bfmi"] = _metric("invalid", reason="arviz_bfmi_failed")
    if posterior is None:
        diagnostics["rhat"] = _metric("unavailable", reason="posterior_group_missing")
        diagnostics["ess_bulk"] = _metric("unavailable", reason="posterior_group_missing")
        diagnostics["ess_tail"] = _metric("unavailable", reason="posterior_group_missing")
    else:
        diagnostics["rhat"] = (
            _metric("unavailable", reason="fewer_than_two_chains")
            if chains is None or chains < 2
            else _arviz_metric(az, "rhat", posterior, "rank", "max")
        )
        diagnostics["ess_bulk"] = _arviz_metric(az, "ess", posterior, "bulk", "min")
        diagnostics["ess_tail"] = _arviz_metric(az, "ess", posterior, "tail", "min")
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
        if threshold is None:
            continue
        record = diagnostics.get(metric, {"status": "unavailable"})
        if record.get("status") == "available":
            observed = record["value"]
            if crosses(observed, threshold):
                failures.append({"code": code, "metric": metric, "observed": observed, "threshold": threshold})
        elif required[metric]:
            missing.append(metric)
        else:
            missing.append(metric)
    failures.sort(key=lambda item: (item["code"], item["metric"]))
    missing.sort()
    return {
        "status": "unhealthy" if failures else ("unknown" if any(required[m] and m in missing for m in required) else "healthy"),
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
    }
    result = {"python": platform.python_version()}
    for key, package in names.items():
        try:
            result[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[key] = "unavailable"
    return result


def sample_oracle(
    world: Any,
    config: OracleSamplingConfig | None = None,
    *,
    criteria: OracleHealthCriteria | None = None,
    world_identity: Mapping[str, object] | None = None,
    source_identity: Mapping[str, object] | None = None,
    configuration_identity: Mapping[str, object] | None = None,
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
    identities = {}
    for name, identity in (
        ("world", world_identity),
        ("source", source_identity),
        ("configuration", configuration_identity),
    ):
        identities[name] = _immutable({} if identity is None else identity, path=f"{name}_identity")

    import arviz as az  # type: ignore[import-untyped]
    import pymc as pm
    import xarray as xr

    started = time.monotonic()
    model = world.oracle_model(latent=config.latent)
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
        nuts_sampler=_NUTS_SAMPLER,
        **_mutable(config.sampler_kwargs),
    )
    if not isinstance(idata, xr.DataTree):
        raise TypeError("pm.sample must return an xarray.DataTree")
    diagnostics, sizes = _diagnostics(idata, az)
    health = _health(diagnostics, criteria)
    posterior = _group(idata, "posterior")
    n_chains = sizes.get("chain")
    draws = sizes.get("draw")
    requested = config.to_dict()
    requested.pop("latent")
    requested["nuts_sampler"] = _NUTS_SAMPLER
    sampling = {
        "requested": requested,
        "effective": {
            "groups": _present_groups(idata),
            "n_chains": int(n_chains) if n_chains is not None else None,
            "draws_per_chain": int(draws) if draws is not None else None,
            "sample_stat_names": sorted(str(name) for name in getattr(_group(idata, "sample_stats"), "data_vars", {})),
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


__all__ = [
    "OracleHealthCriteria",
    "OracleSamplingConfig",
    "OracleSamplingReceipt",
    "OracleSamplingResult",
    "sample_oracle",
]
