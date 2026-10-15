"""File-publication prototype with mounted-Volume hooks; not a distributed lock.

The caller guarantees one driver per run and no overlapping attempts per task.
Scientific generation, selection, seed allocation and scheduling remain upstream.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus
from pymc_generator.slots import CORPUS_SCHEMA_VERSION
from pymc_generator.splitting import assign_cell_split

VERSION = 1


class PublicationError(ValueError):
    """Invalid, conflicting, unsupported or unavailable publication."""


class PublicationUnconfirmedError(RuntimeError):
    """Persistence acknowledgment failed; reconcile before attempting more work."""


class VolumeHandle(Protocol):
    """Synchronous mounted-Volume operations, supplied by the owning Modal app."""

    def commit(self) -> None: ...

    def reload(self) -> None: ...


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise PublicationError(f"Invalid identifier: {value!r}")
    return value


def _write(path: Path, data: bytes) -> None:
    # Exclusive creation prevents local accidental overwrite, not distributed races.
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PublicationError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict) or type(value.get("version")) is not int:
        raise PublicationError("Missing integer metadata version")
    if value["version"] != VERSION:
        raise PublicationError(f"Unsupported metadata version: {value['version']}")
    _json(value)  # Reject non-finite JSON numbers.
    return value


def _scientific_digest(corpus: dict) -> str:
    """Bind persisted values, not ZIP container bytes or timing diagnostics."""

    def encode(value):
        if isinstance(value, np.ndarray):
            return {
                "dtype": value.dtype.str,
                "shape": value.shape,
                "sha256": _digest(value.tobytes(order="C")),
            }
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    return _digest(_json(encode(corpus)))


def _validate_record(record: dict) -> None:
    if record.get("kind") != "completion" or record.get("version") != VERSION:
        raise PublicationError("Not a supported completion record")
    for key in ("run_id", "instruction_id", "task_id", "output_id", "attempt_id"):
        _identifier(record[key])
    worlds = record["world_ids"]
    if not isinstance(worlds, list) or len(set(worlds)) != len(worlds):
        raise PublicationError("World identities must be an ordered unique list")
    for world in worlds:
        _identifier(world)
    counts = record["counts"]
    for key in ("target", "candidates", "rejected", "surplus", "published", "shortfall"):
        if type(counts[key]) is not int or counts[key] < 0:
            raise PublicationError(f"Invalid count: {key}")
    if (
        counts["published"] != len(worlds)
        or counts["target"] != counts["published"] + counts["shortfall"]
        or counts["candidates"] != counts["rejected"] + counts["surplus"] + len(worlds)
    ):
        raise PublicationError("Inconsistent completion accounting")
    reasons = record["rejection_reasons"]
    if (
        not isinstance(reasons, dict)
        or any(type(v) is not int or v < 0 for v in reasons.values())
        or sum(reasons.values()) != counts["rejected"]
    ):
        raise PublicationError("Rejection reasons must partition rejected candidates")
    provenance = record["provenance"]
    for key in (
        "run_spec_sha256",
        "generation_mode",
        "generation_version",
        "environment",
        "seed_context",
        "effective_configuration",
    ):
        if key not in provenance:
            raise PublicationError(f"Missing provenance: {key}")
    if not isinstance(provenance["run_spec_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance["run_spec_sha256"]
    ):
        raise PublicationError("Invalid run specification SHA-256")
    for key in ("generation_mode", "generation_version"):
        if not isinstance(provenance[key], str) or not provenance[key]:
            raise PublicationError(f"Invalid provenance: {key}")
    for key in ("environment", "seed_context", "effective_configuration"):
        if not isinstance(provenance[key], dict) or not provenance[key]:
            raise PublicationError(f"Invalid provenance: {key}")
    if record["corpus_schema_version"] != CORPUS_SCHEMA_VERSION:
        raise PublicationError("Unsupported corpus schema")
    payload = record["payload"]
    if not worlds:
        if payload is not None or record["scientific_sha256"] is not None:
            raise PublicationError("Zero-world completion must not have a payload")
    else:
        if (
            payload["name"] != "corpus.npz"
            or type(payload["bytes"]) is not int
            or payload["bytes"] <= 0
        ):
            raise PublicationError("Invalid payload reference")
        for digest in (payload["sha256"], record["scientific_sha256"]):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise PublicationError("Invalid SHA-256")


class FilePublicationStore:
    """Opt-in local prototype of the dashboard-facing file contract.

    Discovery is metadata-only and returns candidates plus explicit read problems.
    Only :meth:`read_output` validates a candidate as consumable.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _refresh(self) -> None:
        """Local reads need no remote view refresh."""

    def _persist(self) -> None:
        """Local writes are already closed/fsynced; no remote commit needed."""

    def _confirm_persistence(self, stage: str) -> None:
        try:
            self._persist()
        except Exception as exc:
            raise PublicationUnconfirmedError(
                f"{stage} persistence unconfirmed; stop and reconcile before retry"
            ) from exc

    def _path(self, reference: str) -> Path:
        path = (self.root / reference).resolve()
        if not path.is_relative_to(self.root):
            raise PublicationError("Reference escapes store")
        return path

    def discover(self, run_id: str) -> tuple[list[dict], dict[str, str]]:
        """Rescan all records; no highest-ID cursor that could miss late arrivals."""
        self._refresh()
        run = self.root / "runs" / _identifier(run_id)
        outputs, problems = [], {}
        for path in sorted(run.glob("tasks/*/attempts/*/completion.json")):
            reference = path.relative_to(self.root).as_posix()
            try:
                record = _read_json(path)
                _validate_record(record)
                expected = self._reference(record)
                if reference != expected or record["run_id"] != run_id:
                    raise PublicationError("Record identity does not match location")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems[reference] = str(exc)
                continue
            outputs.append({"reference": reference, "record": record})
        return outputs, problems

    @staticmethod
    def _reference(record: dict) -> str:
        return (
            f"runs/{record['run_id']}/tasks/{record['task_id']}/attempts/"
            f"{record['attempt_id']}/completion.json"
        )

    def read_output(self, reference: str) -> tuple[dict, dict | None]:
        """Validate record, payload bytes, corpus and ordered world count."""
        self._refresh()
        path = self._path(reference)
        record = _read_json(path)
        _validate_record(record)
        if reference != self._reference(record):
            raise PublicationError("Record identity does not match location")
        if record["payload"] is None:
            return record, None
        payload = record["payload"]
        payload_path = self._path(str(path.parent.relative_to(self.root) / payload["name"]))
        data = payload_path.read_bytes()
        if len(data) != payload["bytes"] or _digest(data) != payload["sha256"]:
            raise PublicationError("Payload integrity mismatch")
        corpus = load_corpus(payload_path)
        errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
        if errors:
            raise PublicationError("Invalid corpus: " + "; ".join(errors))
        if corpus["treatment_raw"].shape[0] != len(record["world_ids"]):
            raise PublicationError("Payload world count mismatch")
        if _scientific_digest(corpus) != record["scientific_sha256"]:
            raise PublicationError("Scientific content mismatch")
        return record, corpus

    @staticmethod
    def _contribution(record: dict) -> bytes:
        # Retry locations/container encoding are not scientific identity.
        return _json(
            {key: value for key, value in record.items() if key not in ("attempt_id", "payload")}
        )

    def validated_outputs(self, run_id: str) -> list[dict]:
        """Fail closed on unresolved records/conflicts; deduplicate matching retries."""
        candidates, problems = self.discover(run_id)
        if problems:
            raise PublicationError(f"Unresolved completion records: {problems}")
        tasks: dict[str, dict] = {}
        output_ids: set[str] = set()
        world_ids: set[str] = set()
        for candidate in candidates:
            record, _ = self.read_output(candidate["reference"])
            task = record["task_id"]
            if task in tasks:
                if self._contribution(record) != self._contribution(tasks[task]["record"]):
                    raise PublicationError(f"Conflicting results for task {task}")
                continue
            if record["output_id"] in output_ids or world_ids.intersection(record["world_ids"]):
                raise PublicationError("Output/world identity reused across tasks")
            tasks[task] = candidate
            output_ids.add(record["output_id"])
            world_ids.update(record["world_ids"])
        return list(tasks.values())

    def publish(self, metadata: dict, corpus: dict | None) -> str:
        """Publish already selected worlds; never generates or infers accounting.

        Caller must reconcile earlier attempts and confirm they stopped first.
        Metadata supplies stable identities, counts, rejection reasons and provenance.
        """
        record = json.loads(_json(metadata))
        record.update(
            version=VERSION, kind="completion", corpus_schema_version=CORPUS_SCHEMA_VERSION
        )
        for key in ("run_id", "task_id", "attempt_id"):
            _identifier(record[key])
        existing, problems = self.discover(record["run_id"])
        task_prefix = f"runs/{record['run_id']}/tasks/{record['task_id']}/"
        if any(ref.startswith(task_prefix) for ref in problems):
            raise PublicationError("Reconcile unresolved completion records before retry")
        # Failure of an unrelated task must not stop this task from publishing.
        for candidate in existing:
            if candidate["record"]["task_id"] == record["task_id"]:
                self.read_output(candidate["reference"])
        reference = self._reference(record)
        directory = self._path(reference).parent
        directory.mkdir(parents=True, exist_ok=False)
        if corpus is None:
            record.update(payload=None, scientific_sha256=None)
        else:
            errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
            if errors:
                raise PublicationError("Invalid corpus: " + "; ".join(errors))
            payload_path = directory / "corpus.npz"
            save_corpus(corpus, payload_path)
            with payload_path.open("rb") as stream:
                os.fsync(stream.fileno())
            data = payload_path.read_bytes()
            record["payload"] = {"name": "corpus.npz", "bytes": len(data), "sha256": _digest(data)}
            persisted = load_corpus(payload_path)
            record["scientific_sha256"] = _scientific_digest(persisted)
            if persisted["treatment_raw"].shape[0] != len(record["world_ids"]):
                raise PublicationError("Payload world count mismatch")
        _validate_record(record)
        reusable: str | None = None
        for candidate in existing:
            previous = candidate["record"]
            if previous["task_id"] == record["task_id"]:
                if self._contribution(previous) != self._contribution(record):
                    raise PublicationError(
                        "Conflicting retry; pause instruction and quarantine attempt"
                    )
                reusable = cast(str, candidate["reference"])
            elif previous["output_id"] == record["output_id"] or set(
                previous["world_ids"]
            ).intersection(record["world_ids"]):
                raise PublicationError("Output/world identity reused across tasks")
        if reusable is not None:
            return reusable
        if corpus is not None:
            self._confirm_persistence("Payload")
        _write(directory / "completion.json", _json(record))
        self._confirm_persistence("Completion")
        return reference

    def freeze_snapshot(
        self,
        run_id: str,
        snapshot_id: str,
        *,
        validation_fraction: float | None = None,
        split_seed: int = 0,
    ) -> str:
        """Freeze all currently validated completions, including zero-world accounting."""
        outputs = self.validated_outputs(run_id)
        members = [{"reference": item["reference"], "record": item["record"]} for item in outputs]
        snapshot = {
            "version": VERSION,
            "kind": "snapshot",
            "run_id": run_id,
            "snapshot_id": _identifier(snapshot_id),
            "members": members,
        }
        if validation_fraction is not None:
            results = [self.read_output(member["reference"]) for member in members]
            snapshot["cell_split"] = self._snapshot_split(results, validation_fraction, split_seed)
        reference = f"snapshots/{snapshot_id}.json"
        path = self._path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, _json(snapshot))
        self._confirm_persistence("Snapshot")
        return reference

    def read_snapshot(self, reference: str) -> list[tuple[dict, dict | None]]:
        """Never substitute or omit a missing/changed frozen member."""
        self._refresh()
        snapshot = _read_json(self._path(reference))
        if snapshot.get("kind") != "snapshot":
            raise PublicationError("Not a snapshot")
        _identifier(snapshot["run_id"])
        _identifier(snapshot["snapshot_id"])
        if reference != f"snapshots/{snapshot['snapshot_id']}.json":
            raise PublicationError("Snapshot identity does not match location")
        if not isinstance(snapshot["members"], list):
            raise PublicationError("Snapshot members must be a list")
        results = []
        tasks: set[str] = set()
        outputs: set[str] = set()
        worlds: set[str] = set()
        for member in snapshot["members"]:
            record, corpus = self.read_output(member["reference"])
            if (
                record["run_id"] != snapshot["run_id"]
                or record["task_id"] in tasks
                or record["output_id"] in outputs
                or worlds.intersection(record["world_ids"])
            ):
                raise PublicationError("Duplicate or foreign snapshot member")
            tasks.add(record["task_id"])
            outputs.add(record["output_id"])
            worlds.update(record["world_ids"])
            if _json(record) != _json(member["record"]):
                raise PublicationError("Frozen completion record changed")
            results.append((record, corpus))
        if "cell_split" in snapshot:
            split = snapshot["cell_split"]
            expected = self._snapshot_split(results, split["validation_fraction"], split["seed"])
            if split != expected:
                raise PublicationError("Frozen cell split does not match snapshot members")
        return results

    def read_training_split(self, reference: str) -> dict[str, str]:
        """Read verified frozen membership without changing published ``is_val``."""
        self.read_snapshot(reference)
        snapshot = _read_json(self._path(reference))
        if "cell_split" not in snapshot:
            raise PublicationError("Snapshot has no training split")
        return cast(dict[str, str], snapshot["cell_split"]["membership"])

    @staticmethod
    def _snapshot_split(results, validation_fraction, seed) -> dict:
        world_cells = {}
        for record, corpus in results:
            if corpus is None:
                continue
            namespace = record["provenance"].get("cell_namespace")
            if not isinstance(namespace, str) or not namespace:
                raise PublicationError(
                    "Split assignment requires explicit provenance.cell_namespace"
                )
            for world, cell in zip(record["world_ids"], corpus["cell_id"], strict=True):
                world_cells[world] = (namespace, int(cell))
        return {
            "version": 1,
            "algorithm": "sha256-cell-rank-v1",
            "seed": seed,
            "validation_fraction": validation_fraction,
            "membership": assign_cell_split(
                world_cells, validation_fraction=validation_fraction, seed=seed
            ),
        }


class MountedVolumePublicationStore(FilePublicationStore):
    """Publication inside an already-mounted Modal Volume, with explicit barriers.

    The owning app supplies a synchronous ``modal.Volume`` handle and mount path.
    This class never creates a Volume, deploys an app, or starts remote work.
    Serialize all access within this container: reload requires no open handles
    anywhere on the mount, including handles owned by other application code.
    Other containers may publish distinct tasks to distinct attempt directories.

    These hooks are locally tested, not a guarantee of cross-container durability.
    Single-driver and non-overlapping-attempt preconditions still apply.
    """

    def __init__(self, root: str | Path, volume: VolumeHandle):
        super().__init__(root)
        self.volume = volume

    def _refresh(self) -> None:
        self.volume.reload()

    def _persist(self) -> None:
        self.volume.commit()
