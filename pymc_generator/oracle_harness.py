"""Deterministic, Generator-owned execution and provenance for the frozen Oracle cohort.

This module deliberately knows nothing about PFN worlds or corpus files.  Callers
bind opaque source locators to ``SCM`` instances and may inject every expensive
execution seam in tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .oracle_sampling import (
    CompiledOracle,
    OracleHealthCriteria,
    OracleSamplingConfig,
    OracleSamplingReceipt,
    OracleSamplingResult,
    build_oracle_template,
    compile_oracle,
)

_FROZEN_CASES = frozenset(
    {
        "constant_intercept",
        "smooth_baseline",
        "sparse_inputs",
        "omitted_channel",
        "omitted_control",
        "logistic_saturation",
        "five_controls",
    }
)
_NAMESPACE_MARKER = ".oracle-namespace.json"
_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _json_value(value: Any, path: str = "value") -> Any:
    """Convert only JSON-native values, rejecting ambiguous identity values."""
    if value is None or isinstance(value, (str, bool, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} must contain only JSON-native values")


def _freeze_json(value: Any, path: str = "value") -> Any:
    """Recursively copy JSON values into immutable containers."""
    native = _json_value(value, path)
    if isinstance(native, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item, f"{path}.{key}") for key, item in native.items()}
        )
    if isinstance(native, list):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(native))
    return native


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_indices(indices: tuple[int, ...]) -> str:
    return _sha256({"dtype": "int64", "shape": [len(indices)], "values": list(indices)})


def _group(idata: Any, name: str) -> Any:
    try:
        node = idata[name]
    except (KeyError, TypeError, AttributeError):
        return None
    return getattr(node, "ds", node)


def _canonical_cell(value: Any) -> bytes:
    """Encode the deliberately small, deterministic mixed-array value domain."""
    if value is None:
        return b"n"
    if isinstance(value, (str, np.str_)):
        raw = str(value).encode("utf-8")
        return b"s" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, (bytes, np.bytes_)):
        raw = bytes(value)
        return b"b" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, (bool, np.bool_)):
        return b"t" + (b"1" if bool(value) else b"0")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        raw = str(int(value)).encode("ascii")
        return b"i" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            token = b"nan"
        elif np.isposinf(number):
            token = b"+inf"
        elif np.isneginf(number):
            token = b"-inf"
        else:
            token = number.hex().encode("ascii")
        return b"f" + struct.pack(">Q", len(token)) + token
    raise TypeError(f"unsupported mixed-array value type: {type(value).__name__}")


def _hash_mixed_array(digest: Any, array: np.ndarray) -> None:
    for value in array.ravel(order="C"):
        encoded = _canonical_cell(value)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)


def _numerical_group_hash(idata: Any, group_name: str) -> str:
    """Hash numerical values and structural array metadata, never xarray attrs."""
    digest = hashlib.sha256()
    dataset = _group(idata, group_name)
    digest.update(_canonical_bytes({"group": group_name}))
    if dataset is None or not hasattr(dataset, "data_vars"):
        raise ValueError(f"idata must contain a {group_name} dataset")
    names = sorted(str(item) for item in dataset.data_vars)
    if not names:
        raise ValueError(f"{group_name} dataset must contain numerical variables")
    for name in names:
        variable = dataset[name]
        array = np.asarray(variable.values)
        is_mixed = array.dtype.kind in {"O", "U", "S"}
        if (
            not is_mixed
            and not np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(f"{group_name}/{name} is not numeric")
        contiguous = np.ascontiguousarray(array)
        metadata_value = {
            "name": name,
            "dims": [str(dim) for dim in variable.dims],
            "dtype": str(contiguous.dtype),
            "shape": [int(size) for size in contiguous.shape],
        }
        digest.update(_canonical_bytes(metadata_value))
        if is_mixed:
            _hash_mixed_array(digest, contiguous)
        else:
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def canonical_numerical_hash(idata: Any) -> str:
    """Return a container-independent hash of posterior and sample statistics."""
    digest = hashlib.sha256()
    for group_name in ("posterior", "sample_stats"):
        digest.update(bytes.fromhex(_numerical_group_hash(idata, group_name)))
    return digest.hexdigest()


def _identity_hash(value: Mapping[str, object]) -> str:
    return _sha256(value)


def _validate_token(value: str, name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a safe single-component identifier")


def _identity_section(identities: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = identities.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_identity must be a mapping")
    return value


def _sha256_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


@dataclass(frozen=True, slots=True)
class OracleWorldSpec:
    """One explicitly frozen source world in the seven-world cohort."""

    case_id: str
    world_id: str
    source_locator: str
    world_seed: int
    observed_indices: tuple[int, ...] | Sequence[int]
    source_digest: str | None = None
    source_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not isinstance(self.world_id, str):
            raise TypeError("case_id and world_id must be strings")
        if not isinstance(self.source_locator, str) or not self.source_locator:
            raise ValueError("source_locator must be a non-empty string")
        if not isinstance(self.world_seed, int) or isinstance(self.world_seed, bool):
            raise TypeError("world_seed must be an integer")
        indices = tuple(self.observed_indices)
        if not indices or any(
            not isinstance(item, int) or isinstance(item, bool) for item in indices
        ):
            raise ValueError("observed_indices must contain at least one integer")
        if any(item < 0 for item in indices) or len(set(indices)) != len(indices):
            raise ValueError("observed_indices must be unique nonnegative integers")
        object.__setattr__(self, "observed_indices", indices)
        if not isinstance(self.source_identity, Mapping):
            raise TypeError("source_identity must be a mapping")
        identity = dict(_json_value(self.source_identity, "source_identity"))
        digest = self.source_digest
        if digest is None:
            for key in ("sha256", "source_sha256", "digest"):
                if key in identity:
                    digest = identity[key]
                    break
        if not isinstance(digest, str):
            raise ValueError("source_digest must be a string")
        digest = _sha256_digest(digest, "source_digest")
        identity.setdefault("sha256", digest)
        if identity["sha256"] != digest:
            raise ValueError("source_identity sha256 does not match source_digest")
        object.__setattr__(self, "source_digest", digest)
        object.__setattr__(self, "source_identity", _freeze_json(identity, "source_identity"))

    @property
    def key(self) -> str:
        return f"{self.case_id}/{self.world_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "world_id": self.world_id,
            "key": self.key,
            "source_locator": self.source_locator,
            "source_digest": self.source_digest,
            "source_identity": dict(self.source_identity),
            "world_seed": self.world_seed,
            "observed_indices": list(self.observed_indices),
            "observed_indices_hash": _hash_indices(tuple(self.observed_indices)),
        }


@dataclass(frozen=True, slots=True)
class OracleRunSpec:
    """Immutable contract for one historical or fresh cohort submission."""

    run_id: str
    worlds: tuple[OracleWorldSpec, ...] | Sequence[OracleWorldSpec]
    config: OracleSamplingConfig
    criteria: OracleHealthCriteria = field(default_factory=OracleHealthCriteria)
    namespace: Literal["historical", "fresh"] = "fresh"
    submission_id: str | None = None
    generator_identity: Mapping[str, object] = field(default_factory=dict)
    package_identity: Mapping[str, object] = field(default_factory=dict)
    environment_identity: Mapping[str, object] = field(default_factory=dict)
    process_identity: Mapping[str, object] = field(default_factory=dict)
    nutpie_version: str = "0.16.11"
    backend: str = "numba"

    def __post_init__(self) -> None:
        _validate_token(self.run_id, "run_id")
        if self.submission_id is not None:
            _validate_token(self.submission_id, "submission_id")
        if self.namespace not in ("historical", "fresh"):
            raise ValueError("namespace must be 'historical' or 'fresh'")
        worlds = tuple(self.worlds)
        if any(not isinstance(world, OracleWorldSpec) for world in worlds):
            raise TypeError("worlds must contain OracleWorldSpec values")
        object.__setattr__(self, "worlds", worlds)
        if not isinstance(self.config, OracleSamplingConfig):
            raise TypeError("config must be an OracleSamplingConfig")
        if not isinstance(self.criteria, OracleHealthCriteria):
            raise TypeError("criteria must be an OracleHealthCriteria")
        for name in (
            "generator_identity",
            "package_identity",
            "environment_identity",
            "process_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if not value:
                raise ValueError(f"{name} must be nonempty provenance identity")
            object.__setattr__(self, name, _freeze_json(value, name))
        if self.nutpie_version != "0.16.11":
            raise ValueError("the frozen cohort requires Nutpie 0.16.11")
        if self.backend != "numba":
            raise ValueError("the frozen cohort requires the Numba backend")

    def validate(self) -> None:
        if len(self.worlds) != 7:
            raise ValueError("the frozen Oracle cohort must contain exactly seven worlds")
        keys = [world.key for world in self.worlds]
        if len(set(keys)) != len(keys):
            raise ValueError("frozen Oracle cohort contains duplicate world identities")
        seeds_by_case: dict[str, set[int]] = {}
        seeds: list[int] = []
        for world in self.worlds:
            seed = world.world_seed + 2
            seeds.append(seed)
            case_seeds = seeds_by_case.setdefault(world.case_id, set())
            if seed in case_seeds:
                raise ValueError("frozen Oracle cohort contains colliding effective random seeds")
            case_seeds.add(seed)
        if any(seed < 0 for seed in seeds):
            raise ValueError("world_seed + 2 must be a nonnegative random seed")
        expected = {f"{case}/w0000" for case in _FROZEN_CASES}
        if set(keys) != expected:
            raise ValueError("worlds must be the owner-approved seven case/w0000 identities")
        config = self.config
        historical = {
            "draws": 1000,
            "tune": 900,
            "chains": 5,
            "cores": 5,
            "target_accept": 0.95,
            "latent": "sampled",
            "discard_tuned_samples": True,
            "nuts_sampler": "nutpie",
            "adaptation": "draw_diag",
            "progressbar": False,
            "compute_convergence_checks": False,
            "sampler_kwargs": {},
        }
        for name, expected_value in historical.items():
            if getattr(config, name) != expected_value:
                raise ValueError(f"frozen cohort config field {name!r} must be {expected_value!r}")
        if config.random_seed is not None:
            raise ValueError("random_seed is derived as world_seed + 2")
        config.validate()
        try:
            installed_nutpie = metadata.version("nutpie")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "Nutpie 0.16.11 must be installed before Oracle publication"
            ) from error
        if installed_nutpie != self.nutpie_version:
            raise RuntimeError(
                f"installed Nutpie must be exactly {self.nutpie_version}; found {installed_nutpie}"
            )
        identities = self.identity_dict()
        generator = _identity_section(identities, "generator")
        package = _identity_section(identities, "package")
        environment = _identity_section(identities, "environment")
        process = _identity_section(identities, "process")
        if not isinstance(generator.get("commit"), str) or not generator["commit"]:
            raise ValueError("generator_identity.commit must be nonempty")
        if generator["commit"] != _runtime_generator_commit():
            raise ValueError("generator_identity.commit does not match the runtime source")
        if package.get("name") != "pymc-generator":
            raise ValueError("package_identity.name must be pymc-generator")
        if package.get("version") != _installed_generator_version():
            raise ValueError("package_identity.version does not match the installed Generator")
        if environment.get("python") != platform.python_version():
            raise ValueError("environment_identity.python does not match the runtime")
        if environment.get("platform") != platform.platform():
            raise ValueError("environment_identity.platform does not match the runtime")
        lock = environment.get("lock_sha256")
        if not isinstance(lock, str):
            raise ValueError("environment_identity.lock_sha256 must be provided")
        _sha256_digest(lock, "environment_identity.lock_sha256")
        if process.get("pid") != os.getpid():
            raise ValueError("process_identity.pid does not match the runtime")
        if process.get("executable") != sys.executable:
            raise ValueError("process_identity.executable does not match the runtime")

    def identity_dict(self) -> dict[str, object]:
        return {
            "generator": dict(self.generator_identity),
            "package": dict(self.package_identity),
            "environment": dict(self.environment_identity),
            "process": dict(self.process_identity),
            "nutpie_version": self.nutpie_version,
            "backend": self.backend,
        }


def _installed_generator_version() -> str:
    try:
        return metadata.version("pymc-generator")
    except metadata.PackageNotFoundError:
        from ._version import __version__

        return __version__


def _git_commit(path: Path) -> str | None:
    for parent in (path, *path.parents):
        if not (parent / ".git").exists():
            continue
        try:
            completed = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        commit = completed.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit):
            return commit
        return None
    return None


def _runtime_generator_commit() -> str:
    """Resolve the installed/editable Generator source commit fail-closed."""
    try:
        distribution = metadata.distribution("pymc-generator")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("installed Generator provenance is unavailable") from error
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info")
        commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
        if isinstance(commit_id, str):
            commit = str(commit_id).lower()
            if re.fullmatch(r"[0-9a-f]{40}", commit):
                return commit
        source_url = direct_url.get("url")
        if isinstance(source_url, str) and source_url.startswith("file://"):
            from urllib.parse import unquote, urlparse

            source_commit = _git_commit(Path(unquote(urlparse(source_url).path)))
            if source_commit is not None:
                return source_commit
    checkout_commit = _git_commit(Path(__file__).resolve().parent)
    if checkout_commit is not None:
        return checkout_commit
    raise RuntimeError("installed Generator source commit is unavailable")


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for key, package in (("pymc_generator", "pymc-generator"), ("nutpie", "nutpie")):
        try:
            result[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[key] = "unavailable"
    return result


def _under_root(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("Oracle artifact path escapes output_root") from error
    return candidate


def _world_path(
    root: Path, kind: str, run_id: str, world: OracleWorldSpec, suffix: str = ""
) -> Path:
    return _under_root(root, kind, run_id, world.case_id, world.world_id, f"a001{suffix}.json")


def _canonical_content(payload: Mapping[str, object] | str) -> bytes:
    content = (
        payload
        if isinstance(payload, str)
        else json.dumps(
            _json_value(payload), allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    )
    if not content.endswith("\n"):
        content += "\n"
    return content.encode("utf-8")


def _write_immutable(path: Path, payload: Mapping[str, object] | str) -> str:
    """Atomically install a canonical JSON file, refusing every overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    content = _canonical_content(payload)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"immutable artifact already exists: {path}")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return _sha256_bytes(content)


def _content_hash(payload: Mapping[str, object] | str) -> str:
    return _sha256_bytes(_canonical_content(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _ensure_namespace(root: Path, namespace: str) -> None:
    marker = root / _NAMESPACE_MARKER
    payload = {"schema_version": _SCHEMA_VERSION, "namespace": namespace}
    if marker.exists():
        if _read_json(marker).get("namespace") != namespace:
            raise ValueError("historical and fresh Oracle namespaces cannot share an output root")
        return
    _write_immutable(marker, payload)


def _check_paths_absent(paths: Sequence[Path]) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable Oracle artifact paths already exist: {collisions}")


def _manifest(spec: OracleRunSpec, root: Path) -> dict[str, object]:
    config = spec.config.to_dict()
    config.pop("random_seed")
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "pymc_generator.oracle_frozen_cohort_manifest",
        "namespace": spec.namespace,
        "run_id": spec.run_id,
        "submission_id": spec.submission_id,
        "cohort": "owner-approved case_id/w0000 representative cohort",
        "worlds": [
            {
                **world.to_dict(),
                "effective_random_seed": world.world_seed + 2,
                "config": {**config, "random_seed": world.world_seed + 2},
            }
            for world in sorted(spec.worlds, key=lambda item: item.key)
        ],
        "config": config,
        "criteria": spec.criteria.to_dict(),
        "identities": spec.identity_dict(),
        "identity_hash": _identity_hash(spec.identity_dict()),
        "output_root": str(root.resolve()),
    }


def _effective_config(spec: OracleRunSpec, seed: int) -> OracleSamplingConfig:
    base = spec.config
    return OracleSamplingConfig(
        draws=base.draws,
        tune=base.tune,
        chains=base.chains,
        cores=base.cores,
        target_accept=base.target_accept,
        random_seed=seed,
        latent=base.latent,
        discard_tuned_samples=base.discard_tuned_samples,
        progressbar=base.progressbar,
        compute_convergence_checks=base.compute_convergence_checks,
        nuts_sampler=base.nuts_sampler,
        adaptation=base.adaptation,
        sampler_kwargs=base.sampler_kwargs,
    )


def _result_parts(result: Any) -> tuple[Any, Any]:
    if isinstance(result, OracleSamplingResult):
        return result.idata, result.receipt
    idata = getattr(result, "idata", None)
    receipt = getattr(result, "receipt", None)
    if idata is None or receipt is None:
        raise TypeError("fit_compiled must return an OracleSamplingResult-like object")
    return idata, receipt


def _compile_identity(compiled: Any, spec: OracleRunSpec) -> dict[str, object]:
    backend = getattr(compiled, "backend", None)
    instance_id = getattr(compiled, "_instance_id", None)
    owner_pid = getattr(compiled, "_owner_pid", None)
    if backend != "numba":
        raise ValueError("compiled Oracle backend provenance must be numba")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("compiled Oracle instance identity is unavailable")
    if owner_pid != os.getpid():
        raise ValueError("compiled Oracle owner PID does not match the runtime")
    return {
        "engine": "nutpie",
        "backend": backend,
        "instance_id": instance_id,
        "owner_pid": owner_pid,
    }


def _selector_identity(indices: tuple[int, ...]) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(indices, dtype="int64"))
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "dtype": str(array.dtype),
        "shape": tuple(int(size) for size in array.shape),
    }


def _default_fit(
    compiled: CompiledOracle,
    world: Any,
    config: OracleSamplingConfig,
    criteria: OracleHealthCriteria,
    world_spec: OracleWorldSpec,
) -> OracleSamplingResult:
    return compiled.fit(
        world,
        config,
        criteria=criteria,
        observed_indices=list(world_spec.observed_indices),
        world_identity={
            "case_id": world_spec.case_id,
            "world_id": world_spec.world_id,
            "world_seed": world_spec.world_seed,
        },
        source_identity=dict(world_spec.source_identity),
        configuration_identity={"effective_config": config.to_dict()},
    )


def _validate_receipt(
    receipt: Any,
    *,
    spec: OracleRunSpec,
    world_spec: OracleWorldSpec,
    config: OracleSamplingConfig,
    template_signature: str,
    compile_identity: Mapping[str, object],
) -> OracleSamplingReceipt:
    if not isinstance(receipt, OracleSamplingReceipt):
        raise TypeError("fit_compiled must return an OracleSamplingReceipt")
    if (
        receipt.schema_version != 1
        or receipt.kind != "pymc_generator.oracle_compiled_sampling_receipt"
    ):
        raise ValueError("fit receipt has an unsupported OracleSamplingReceipt schema or kind")
    if receipt.package_version != spec.package_identity["version"]:
        raise ValueError("fit receipt package version does not match the run identity")
    if receipt.environment.get("python") != spec.environment_identity["python"]:
        raise ValueError("fit receipt Python identity does not match the run identity")
    if receipt.environment.get("nutpie") != spec.nutpie_version:
        raise ValueError("fit receipt Nutpie identity does not match the run identity")
    expected_requested = config.to_dict()
    expected_requested.pop("latent")
    if receipt.sampling.get("requested") != expected_requested:
        raise ValueError("fit receipt effective sampling config does not match the invocation")
    if receipt.oracle.get("latent") != config.latent:
        raise ValueError("fit receipt latent mode does not match the invocation")
    identities = receipt.identities
    expected_world = {
        "case_id": world_spec.case_id,
        "world_id": world_spec.world_id,
        "world_seed": world_spec.world_seed,
    }
    if identities.get("world") != expected_world:
        raise ValueError("fit receipt world identity does not match the invocation")
    if identities.get("source") != dict(world_spec.source_identity):
        raise ValueError("fit receipt source identity does not match the invocation")
    if identities.get("configuration") != {"effective_config": config.to_dict()}:
        raise ValueError("fit receipt configuration identity does not match the invocation")
    if identities.get("template") != {"signature": template_signature}:
        raise ValueError("fit receipt template identity does not match the invocation")
    if identities.get("compile") != dict(compile_identity):
        raise ValueError("fit receipt compile identity does not match the invocation")
    if identities.get("observed_indices") != _selector_identity(tuple(world_spec.observed_indices)):
        raise ValueError("fit receipt selector identity does not match the invocation")
    return receipt


def run_frozen_cohort(
    spec: OracleRunSpec,
    load_world: Callable[[str], Any],
    fit_compiled: Callable[
        [Any, Any, OracleSamplingConfig, OracleHealthCriteria, OracleWorldSpec], Any
    ]
    | None,
    output_root: str | Path,
    *,
    build_template: Callable[..., Any] = build_oracle_template,
    compile_template: Callable[..., Any] = compile_oracle,
) -> dict[str, object]:
    """Execute each frozen world exactly once and emit immutable ledgers."""
    spec.validate()
    root = Path(output_root).resolve()
    fit = _default_fit if fit_compiled is None else fit_compiled
    manifest_path = _under_root(root, "manifests", f"{spec.run_id}.json")
    selected_path = _under_root(root, "selected", f"{spec.run_id}.json")
    expected_paths = [manifest_path, selected_path]
    for world in spec.worlds:
        expected_paths.extend(
            [
                _world_path(root, "attempts", spec.run_id, world),
                _world_path(root, "attempts", spec.run_id, world, ".started"),
                _world_path(root, "attempts", spec.run_id, world, ".committed"),
                _world_path(root, "receipts", spec.run_id, world),
            ]
        )
    _check_paths_absent(expected_paths)
    _ensure_namespace(root, spec.namespace)
    _write_immutable(manifest_path, _manifest(spec, root))

    templates: dict[str, Any] = {}
    compiled_by_signature: dict[str, Any] = {}
    rows: list[dict[str, object]] = []
    for world_spec in sorted(spec.worlds, key=lambda item: item.key):
        attempt_path = _world_path(root, "attempts", spec.run_id, world_spec)
        started_path = _world_path(root, "attempts", spec.run_id, world_spec, ".started")
        committed_path = _world_path(root, "attempts", spec.run_id, world_spec, ".committed")
        receipt_path = _world_path(root, "receipts", spec.run_id, world_spec)
        effective_config = _effective_config(spec, world_spec.world_seed + 2)
        config_hash = _sha256(effective_config.to_dict())
        _write_immutable(
            started_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "kind": "pymc_generator.oracle_attempt",
                "namespace": spec.namespace,
                "run_id": spec.run_id,
                "case_id": world_spec.case_id,
                "world_id": world_spec.world_id,
                "attempt": "a001",
                "status": "started",
                "source_identity": dict(world_spec.source_identity),
                "process_identity": dict(spec.process_identity),
                "config_hash": config_hash,
            },
        )
        attempt_written = False
        try:
            world = load_world(world_spec.source_locator)
            template = build_template(
                world, latent="sampled", observed_indices=tuple(world_spec.observed_indices)
            )
            signature = str(template.signature)
            templates.setdefault(signature, template)
            if signature not in compiled_by_signature:
                compiled_by_signature[signature] = compile_template(template)
            compiled = compiled_by_signature[signature]
            compile_identity = _compile_identity(compiled, spec)
            result = fit(
                compiled,
                world,
                effective_config,
                spec.criteria,
                world_spec,
            )
            idata, receipt = _result_parts(result)
            receipt = _validate_receipt(
                receipt,
                spec=spec,
                world_spec=world_spec,
                config=effective_config,
                template_signature=signature,
                compile_identity=compile_identity,
            )
            numerical_hash = canonical_numerical_hash(idata)
            posterior_hash = _numerical_group_hash(idata, "posterior")
            sample_stats_hash = _numerical_group_hash(idata, "sample_stats")
            receipt_json = receipt.to_json()
            parsed_receipt = json.loads(receipt_json)
            if not isinstance(parsed_receipt, dict):
                raise TypeError("OracleSamplingReceipt.to_json() must encode an object")
            receipt_payload = {
                **parsed_receipt,
                "artifacts": {
                    "numerical_sha256": numerical_hash,
                    "posterior_sha256": posterior_hash,
                    "sample_stats_sha256": sample_stats_hash,
                },
            }
            receipt_hash = _content_hash(receipt_payload)
            compact = {
                "key": world_spec.key,
                "source_digest": world_spec.source_digest,
                "source_identity": dict(world_spec.source_identity),
                "world_seed": world_spec.world_seed,
                "effective_random_seed": world_spec.world_seed + 2,
                "observed_indices": list(world_spec.observed_indices),
                "observed_indices_hash": _hash_indices(tuple(world_spec.observed_indices)),
                "config_hash": config_hash,
                "template_signature": signature,
                "compile_identity": compile_identity,
                "process_identity": dict(spec.process_identity),
                "posterior_sha256": posterior_hash,
                "sample_stats_sha256": sample_stats_hash,
                "numerical_sha256": numerical_hash,
            }
            attempt_payload = {
                "schema_version": _SCHEMA_VERSION,
                "kind": "pymc_generator.oracle_attempt",
                "namespace": spec.namespace,
                "run_id": spec.run_id,
                "case_id": world_spec.case_id,
                "world_id": world_spec.world_id,
                "attempt": "a001",
                "status": "completed",
                "config_hash": compact["config_hash"],
                "source_identity": dict(world_spec.source_identity),
                "process_identity": dict(spec.process_identity),
                "template_signature": signature,
                "compile_identity": compile_identity,
                "posterior_sha256": posterior_hash,
                "sample_stats_sha256": sample_stats_hash,
                "numerical_sha256": numerical_hash,
                "receipt_path": str(receipt_path.relative_to(root)),
                "receipt_sha256": receipt_hash,
            }
            _write_immutable(attempt_path, attempt_payload)
            attempt_written = True
            _write_immutable(receipt_path, receipt_payload)
            _write_immutable(
                committed_path,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "pymc_generator.oracle_attempt_commit",
                    "namespace": spec.namespace,
                    "run_id": spec.run_id,
                    "case_id": world_spec.case_id,
                    "world_id": world_spec.world_id,
                    "attempt": "a001",
                    "status": "committed",
                    "attempt_path": str(attempt_path.relative_to(root)),
                    "receipt_path": str(receipt_path.relative_to(root)),
                    "receipt_sha256": receipt_hash,
                },
            )
            rows.append(
                {
                    **compact,
                    "attempt": "a001",
                    "receipt_path": str(receipt_path.relative_to(root)),
                    "commit_path": str(committed_path.relative_to(root)),
                }
            )
        except Exception as error:
            if attempt_written:
                raise
            _write_immutable(
                attempt_path,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "pymc_generator.oracle_attempt",
                    "namespace": spec.namespace,
                    "run_id": spec.run_id,
                    "case_id": world_spec.case_id,
                    "world_id": world_spec.world_id,
                    "attempt": "a001",
                    "status": "error",
                    "source_identity": dict(world_spec.source_identity),
                    "process_identity": dict(spec.process_identity),
                    "config_hash": config_hash,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise

    selected = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "pymc_generator.oracle_selected_ledger",
        "namespace": spec.namespace,
        "run_id": spec.run_id,
        "selection": "mechanical_a001_only",
        "rows": rows,
    }
    _write_immutable(selected_path, selected)
    return selected


def _pair_compact(selected: Mapping[str, object]) -> dict[str, Any]:
    rows = selected.get("rows")
    if not isinstance(rows, list):
        raise ValueError("selected ledger rows are malformed")
    compact = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("selected ledger row is malformed")
        compact.append(
            {
                key: row[key]
                for key in (
                    "key",
                    "source_digest",
                    "source_identity",
                    "world_seed",
                    "effective_random_seed",
                    "observed_indices",
                    "observed_indices_hash",
                    "config_hash",
                    "template_signature",
                    "process_identity",
                    "posterior_sha256",
                    "sample_stats_sha256",
                    "numerical_sha256",
                )
            }
        )
    return {"rows": sorted(compact, key=lambda row: row["key"])}


def run_pair(
    first: OracleRunSpec,
    second: OracleRunSpec,
    load_world: Callable[[str], Any],
    fit_compiled: Callable[
        [Any, Any, OracleSamplingConfig, OracleHealthCriteria, OracleWorldSpec], Any
    ]
    | None,
    output_root: str | Path,
    *,
    build_template: Callable[..., Any] = build_oracle_template,
    compile_template: Callable[..., Any] = compile_oracle,
    pair_id: str | None = None,
) -> dict[str, object]:
    """Execute two fresh submissions and compare their posterior-free hashes."""
    first.validate()
    second.validate()
    if first.namespace != "fresh" or second.namespace != "fresh":
        raise ValueError("paired submissions must use the fresh namespace")
    if first.run_id == second.run_id:
        raise ValueError("paired submissions require distinct run IDs")
    if (
        not first.submission_id
        or not second.submission_id
        or first.submission_id == second.submission_id
    ):
        raise ValueError("paired submissions require two distinct submission IDs")
    if _canonical_bytes(first.identity_dict()) != _canonical_bytes(second.identity_dict()):
        raise ValueError("paired submissions must have byte-identical execution identities")
    if _canonical_bytes(first.config.to_dict()) != _canonical_bytes(second.config.to_dict()):
        raise ValueError("paired submissions must have byte-identical configs")
    first_sources = [world.to_dict() for world in sorted(first.worlds, key=lambda item: item.key)]
    second_sources = [world.to_dict() for world in sorted(second.worlds, key=lambda item: item.key)]
    if _canonical_bytes(first_sources) != _canonical_bytes(second_sources):
        raise ValueError("paired submissions must have byte-identical frozen source identities")
    root = Path(output_root).resolve()
    pair_id = pair_id or f"{first.submission_id}__{second.submission_id}"
    _validate_token(pair_id, "pair_id")
    comparison_path = _under_root(root, "comparisons", f"{pair_id}.json")
    _check_paths_absent([comparison_path])
    selected_first = run_frozen_cohort(
        first,
        load_world,
        fit_compiled,
        root,
        build_template=build_template,
        compile_template=compile_template,
    )
    selected_second = run_frozen_cohort(
        second,
        load_world,
        fit_compiled,
        root,
        build_template=build_template,
        compile_template=compile_template,
    )
    compact_first = _pair_compact(selected_first)
    compact_second = _pair_compact(selected_second)
    if _canonical_bytes(compact_first) != _canonical_bytes(compact_second):
        raise ValueError("paired submissions differ in posterior-free compact fields or hashes")
    records = [
        {
            "key": row["key"],
            "posterior_sha256": row["posterior_sha256"],
            "sample_stats_sha256": row["sample_stats_sha256"],
            "numerical_sha256": row["numerical_sha256"],
        }
        for row in compact_first["rows"]
    ]
    comparison = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "pymc_generator.oracle_pair_comparison",
        "pair_id": pair_id,
        "submission_ids": [first.submission_id, second.submission_id],
        "run_ids": [first.run_id, second.run_id],
        "status": "pass",
        "rows": records,
    }
    _write_immutable(comparison_path, comparison)
    return comparison


__all__ = [
    "OracleRunSpec",
    "OracleWorldSpec",
    "canonical_numerical_hash",
    "run_frozen_cohort",
    "run_pair",
]
