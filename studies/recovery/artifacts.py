"""Versioned stage identity and verifiable lifecycle, without running any stage."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ContractError, StudyConfig


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def environment() -> dict:
    versions = {}
    for name in (
        "pymc-generator",
        "numpy",
        "scipy",
        "pymc",
        "nutpie",
        "arviz",
        "pytensor",
        "xarray",
        "h5netcdf",
        "h5py",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def identity(
    stage: str,
    config: StudyConfig,
    *,
    source_files: dict[str, Path],
    dataset_files: dict[str, Path] | None = None,
    expected_outputs: set[str] | None = None,
) -> dict:
    """Record resolved settings, relevant source hashes and upstream input hashes."""
    relevant = {
        "generate": ("study", "generator"),
        "fit": ("study", "generator", "methods"),
        "analyze": ("study", "generator", "methods", "score", "diagnostics"),
        "report": ("study", "generator", "methods", "score", "diagnostics"),
    }
    if stage not in relevant:
        raise ContractError(f"unknown stage: {stage}")
    if stage != "generate" and not dataset_files:
        raise ContractError(f"{stage} requires explicit upstream artifact hashes")

    def hashes(files: dict[str, Path]) -> dict[str, str]:
        if not files:
            raise ContractError("empty source/input file inventory")
        return {name: digest(path) for name, path in sorted(files.items())}

    if expected_outputs is not None:
        _names(expected_outputs)
    return {
        "schema": 1,
        "stage": stage,
        "expected_outputs": sorted(expected_outputs) if expected_outputs is not None else None,
        "config": {key: config.resolved[key] for key in relevant[stage]},
        "sources": hashes(source_files),
        "inputs": hashes(dataset_files) if dataset_files else {},
        "environment": environment(),
    }


@dataclass(frozen=True)
class Checkpoint:
    state: str  # new, initialized, partial, complete

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    def __bool__(self) -> bool:
        """Prevent accidental use as the old manifest-presence completion boolean."""
        raise TypeError("inspect checkpoint.state or checkpoint.complete explicitly")


def _outputs(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in directory.iterdir()
        if p.name not in ("manifest.json", "completion.json")
        and not p.name.startswith(".completion-")
    }


def _names(names: set[str]) -> None:
    if not names or any(
        not isinstance(name, str)
        or name in ("manifest.json", "completion.json")
        or name.startswith(".")
        or Path(name).name != name
        or name in ("", ".", "..")
        for name in names
    ):
        raise ContractError("expected outputs must be nonempty plain filenames")


def checkpoint(directory: Path, expected: dict) -> Checkpoint:
    """Inspect stage lifecycle. Completed outputs must still match their recorded hashes."""
    manifest = directory / "manifest.json"
    if not directory.exists():
        return Checkpoint("new")
    if not directory.is_dir():
        raise ContractError(f"stage path is not a directory: {directory}")
    if not manifest.exists():
        if any(directory.iterdir()):
            raise ContractError(f"artifacts without matching manifest: {directory}")
        return Checkpoint("new")
    try:
        recorded = json.loads(manifest.read_text())
    except (ValueError, OSError) as exc:
        raise ContractError(f"invalid stage manifest: {manifest}") from exc
    if recorded != expected:
        raise ContractError(f"stale/incompatible artifact identity: {manifest}")
    outputs = _outputs(directory)
    marker = directory / "completion.json"
    if not marker.exists():
        return Checkpoint("partial" if outputs else "initialized")
    try:
        receipt = json.loads(marker.read_text())
    except (ValueError, OSError) as exc:
        raise ContractError(f"invalid completion receipt: {marker}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != 1
        or not isinstance(receipt.get("outputs"), dict)
    ):
        raise ContractError(f"invalid completion receipt: {marker}")
    inventory = expected.get("expected_outputs")
    if (
        not isinstance(inventory, list)
        or not inventory
        or any(type(name) is not str for name in inventory)
        or len(set(inventory)) != len(inventory)
    ):
        raise ContractError("completed stage identity has missing or malformed output inventory")
    _names(set(inventory))
    hashes = receipt["outputs"]
    _names(set(hashes))
    if set(hashes) != set(inventory):
        raise ContractError("completion receipt inventory differs from stage identity inventory")
    if set(outputs) != set(hashes):
        raise ContractError(
            f"completed output inventory mismatch: missing={sorted(set(hashes) - set(outputs))}; unexpected={sorted(set(outputs) - set(hashes))}"
        )
    for name, expected_hash in hashes.items():
        if (
            not isinstance(expected_hash, str)
            or not outputs[name].is_file()
            or digest(outputs[name]) != expected_hash
        ):
            raise ContractError(f"completed output hash mismatch: {name}")
    return Checkpoint("complete")


def write_manifest(directory: Path, expected: dict) -> None:
    """Initialize a new stage, without treating initialization as completion."""
    state = checkpoint(directory, expected)
    if state.state != "new":
        return
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{directory.name}.manifest-{os.getpid()}"
    try:
        with temporary.open("x") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, directory / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def complete_checkpoint(directory: Path, expected: dict, expected_outputs: set[str]) -> None:
    """Seal an inventory only after every expected output exists; atomic receipt publish."""
    _names(expected_outputs)
    if expected.get("expected_outputs") != sorted(expected_outputs):
        raise ContractError("expected output inventory must equal the stage identity inventory")
    state = checkpoint(directory, expected)
    if state.complete:
        if set(_outputs(directory)) != expected_outputs:
            raise ContractError("completed checkpoint has a different expected output inventory")
        return
    if state.state == "new":
        raise ContractError("initialize the stage manifest before completion")
    outputs = _outputs(directory)
    missing = expected_outputs - set(outputs)
    unexpected = set(outputs) - expected_outputs
    if missing or unexpected:
        raise ContractError(
            f"cannot complete: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if any(not path.is_file() for path in outputs.values()):
        raise ContractError("completed outputs must be regular files")
    receipt = {
        "schema": 1,
        "outputs": {name: digest(outputs[name]) for name in sorted(expected_outputs)},
    }
    temporary = directory / f".completion-{os.getpid()}"
    try:
        with temporary.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, directory / "completion.json")
    finally:
        temporary.unlink(missing_ok=True)
