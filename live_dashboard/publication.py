"""Publication contract ``pymc-generator-live-publication`` v1.2.

Identities, the store layout and the three JSON records a generation run publishes:

- ``runs/<run_id>/run.json``: the immutable run manifest (request, effective generator
  configuration, code and environment identity);
- ``runs/<run_id>/tasks/<task_id>/<attempt_id>.json``: an immutable completion record of
  one logical generation task, written by the attempt that generated it (accepted worlds,
  payload reference and digest, accounting, timing). A task that accepted no world
  publishes a record without payload. Records of different attempts never share a path:
  readers treat records with the same scientific content as duplicates and records with
  different content as a conflict;
- ``runs/<run_id>/attempts/<attempt_id>.json``: provisional progress of one writer
  invocation, rewritten by that writer only and never authoritative.

Every record's ``worker`` (task and attempt records) describes the machine. Since 1.1 it
carries ``usd_per_hour``: the list price of the worker's reservation, or null for a
worker that is not billed (a local run). 1.0 records omit it; readers treat null and
absent alike.

Since 1.2, a run whose preset names a structural-split design (``live_dashboard.structural``)
seeds its tasks per split instead of ``seed + task_index``: the manifest's ``seed_rule``
states the rule and its ``structural_splits`` lists every split's seed. Every completion's
``seed_context`` records the index of the task's last draw (``redraws``: draws the worker
discarded before it) and that draw's seed (``generation_seed``); a design task also
records its split and regime. A 1.1 reader would reject every task of a design run (it
re-derives seeds the old way); the dashboard and the worker deploy together, as one app.

Payloads live at ``runs/<run_id>/payloads/<task_id>/<attempt_id>.npz``. Identities derive
from the run request and task index only, never from scheduling, arrival order or paths.

Compatibility: readers accept contract major version 1 and ignore unknown fields
(additive minor versions); any other major version is rejected, never guessed. Adding a
value to an enumerated field (task status, attempt state) requires a new major version.

Stdlib only: the Modal launcher imports this module with the Modal CLI's interpreter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from live_dashboard import structural

CONTRACT = "pymc-generator-live-publication"
CONTRACT_VERSION = "1.2"
SUPPORTED_MAJOR = 1
PAYLOAD_FORMAT = "pymc-generator-corpus-npz"
MAX_USD_PER_HOUR = 1000.0  # far above any single Modal container's list price
SEED_RULE = "seed + task_index (DataGenerator.iter_batches batches)"
RUNS_DIR = "runs"
MAX_TASKS = 100_000
MAX_WORLDS_PER_TASK = 10_000
MAX_RECORD_BYTES = 1 << 20
MAX_MESSAGE_CHARS = 1000
MAX_TEXT_CHARS = 200
# Every integer a record carries stays exact in any JSON reader (IEEE-754 doubles).
MAX_JSON_INT = 2**53 - 1
MAX_SEED = MAX_JSON_INT - MAX_TASKS

_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}-[0-9a-f]{12}")
_TASK_ID = re.compile(r"task-([0-9]{5})")
_ATTEMPT_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"([0-9]{1,6})\.([0-9]{1,6})")

TaskStatus = Literal["complete", "shortfall", "zero"]
AttemptState = Literal["running", "finished", "failed"]
ATTEMPT_STATES: tuple[AttemptState, ...] = ("running", "finished", "failed")


class RecordError(ValueError):
    """A publication record cannot be used as published."""


class UnreadableRecordError(RecordError):
    """A record file is not (yet) a JSON document, e.g. a partially persisted write."""


class UnsupportedVersionError(RecordError):
    """A record uses a contract major version this reader does not support."""


# -- encoding helpers ---------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Deterministic UTF-8 JSON used for digests (sorted keys, no whitespace)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> datetime:
    """Current UTC time at the millisecond precision records store."""
    moment = datetime.now(UTC)
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def format_timestamp(moment: datetime) -> str:
    """UTC ISO-8601 with milliseconds and a ``Z`` suffix."""
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    text = moment.astimezone(UTC).isoformat(timespec="milliseconds")
    return text.removesuffix("+00:00") + "Z"


def parse_timestamp(text: Any) -> datetime:
    if not isinstance(text, str) or not text.endswith("Z"):
        raise RecordError(f"timestamp must be an ISO-8601 UTC string ending in Z, got {text!r}")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise RecordError(f"invalid timestamp {text!r}") from error
    if moment.utcoffset() != UTC.utcoffset(None):
        raise RecordError(f"timestamp {text!r} is not UTC")
    return moment.astimezone(UTC)


def clean_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """A message safe to store and display: valid UTF-8, at most ``limit`` characters."""
    text = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _encode(kind: str, body: Mapping[str, Any]) -> bytes:
    record = {"contract": CONTRACT, "contract_version": CONTRACT_VERSION, "kind": kind, **body}
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


# -- identities and layout -----------------------------------------------------------


def task_id_for(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_TASKS:
        raise ValueError(f"task index must be an integer in [0, {MAX_TASKS}), got {index!r}")
    return f"task-{index:05d}"


def task_index_of(task_id: Any) -> int:
    match = _TASK_ID.fullmatch(task_id) if isinstance(task_id, str) else None
    if match is None:
        raise RecordError(f"invalid task id {task_id!r}")
    return int(match.group(1))


def is_run_id(text: Any) -> bool:
    return isinstance(text, str) and _RUN_ID.fullmatch(text) is not None


def is_task_id(text: Any) -> bool:
    return isinstance(text, str) and _TASK_ID.fullmatch(text) is not None


def is_attempt_id(text: Any) -> bool:
    return isinstance(text, str) and _ATTEMPT_ID.fullmatch(text) is not None


def new_attempt_id(now: datetime | None = None) -> str:
    moment = (now or utc_now()).astimezone(UTC)
    return f"{moment:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def world_id_for(run_id: str, task_id: str, row: int) -> str:
    return f"{run_id}/{task_id}/w{row}"


def run_manifest_path(run_id: str) -> str:
    return f"{RUNS_DIR}/{run_id}/run.json"


def tasks_dir(run_id: str) -> str:
    return f"{RUNS_DIR}/{run_id}/tasks"


def task_dir(run_id: str, task_id: str) -> str:
    return f"{tasks_dir(run_id)}/{task_id}"


def task_record_path(run_id: str, task_id: str, attempt_id: str) -> str:
    return f"{task_dir(run_id, task_id)}/{attempt_id}.json"


def payload_path(run_id: str, task_id: str, attempt_id: str) -> str:
    return f"{RUNS_DIR}/{run_id}/payloads/{task_id}/{attempt_id}.npz"


def attempts_dir(run_id: str) -> str:
    return f"{RUNS_DIR}/{run_id}/attempts"


def attempt_record_path(run_id: str, attempt_id: str) -> str:
    return f"{attempts_dir(run_id)}/{attempt_id}.json"


@dataclass(frozen=True)
class RunRequest:
    """What a run generates; its digest names the run.

    ``worlds`` accepted worlds are split into ``worlds // worlds_per_task`` logical tasks.
    Task ``i`` generates with seed ``seed + i``: exactly batch ``i`` of
    ``DataGenerator.iter_batches(n_tasks=worlds, batch_size=worlds_per_task, seed=seed)``
    (reproducible, but not the same worlds as one larger generation call). A preset that
    names a structural-split design (``structural.DESIGNS``) seeds per split instead, and
    its tasks must divide evenly over the splits. The code and environment that generate
    a run are recorded in its manifest and checked on resume.
    """

    label: str
    preset: str
    worlds: int
    worlds_per_task: int
    seed: int

    def __post_init__(self) -> None:
        for name in ("label", "preset"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
                raise ValueError(
                    f"{name} must match [a-z0-9][a-z0-9-]{{0,39}} (lowercase slug), got {value!r}"
                )
        for name in ("worlds", "worlds_per_task", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.worlds_per_task < 2:
            raise ValueError("worlds_per_task must be >= 2 (a corpus needs a train/val split)")
        if self.worlds < self.worlds_per_task or self.worlds % self.worlds_per_task:
            raise ValueError(
                f"worlds ({self.worlds}) must be a positive multiple of worlds_per_task "
                f"({self.worlds_per_task})"
            )
        if self.worlds // self.worlds_per_task > MAX_TASKS:
            raise ValueError(f"at most {MAX_TASKS} tasks per run")
        if self.worlds_per_task > MAX_WORLDS_PER_TASK:
            raise ValueError(f"at most {MAX_WORLDS_PER_TASK} worlds per task")
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {self.seed}")
        if self.design is not None:
            self.design.check(self.seed, self.n_tasks)

    @property
    def n_tasks(self) -> int:
        return self.worlds // self.worlds_per_task

    @property
    def design(self) -> structural.SplitDesign | None:
        """The structural-split design the preset names, if any."""
        return structural.DESIGNS.get(self.preset)

    @property
    def seed_rule(self) -> str:
        return SEED_RULE if self.design is None else structural.SEED_RULE

    def task_seed(self, index: int) -> int:
        """The seed of task ``index``'s first draw."""
        if self.design is None:
            return self.seed + index
        return self.design.task_seed(self.seed, index)

    def seed_context(self, index: int) -> dict[str, Any]:
        """How task ``index`` is seeded: rule, base seed, task seed (and split, regime)."""
        context: dict[str, Any] = {
            "rule": self.seed_rule,
            "base_seed": self.seed,
            "task_index": index,
            "task_seed": self.task_seed(index),
        }
        if self.design is not None:
            context.update(self.design.seed_context(self.seed, index))
        return context

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "preset": self.preset,
            "worlds": self.worlds,
            "worlds_per_task": self.worlds_per_task,
            "seed": self.seed,
        }

    @property
    def request_sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    @property
    def run_id(self) -> str:
        return f"{self.label}-{self.request_sha256[:12]}"

    @classmethod
    def from_dict(cls, data: Any) -> RunRequest:
        body = _as_object(data, "request")
        try:
            return cls(
                label=body.get("label"),
                preset=body.get("preset"),
                worlds=body.get("worlds"),
                worlds_per_task=body.get("worlds_per_task"),
                seed=body.get("seed"),
            )
        except ValueError as error:
            raise RecordError(f"invalid run request: {error}") from error


# -- records ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerInfo:
    platform: str
    cpu_count: int | None
    gpu_requested: str | None
    gpu_name: str | None
    usd_per_hour: float | None = None  # list price of the reservation; None if not billed

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "gpu_requested": self.gpu_requested,
            "gpu_name": self.gpu_name,
            "usd_per_hour": self.usd_per_hour,
        }


@dataclass(frozen=True)
class ErrorInfo:
    type: str
    message: str

    @classmethod
    def from_exception(cls, error: BaseException) -> ErrorInfo:
        return cls(type=type(error).__name__, message=clean_message(str(error)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "message": self.message}

    def describe(self) -> str:
        return f"{self.type}: {self.message}" if self.message else self.type


@dataclass(frozen=True)
class GeneratorInfo:
    """The effective generator: configuration, code and numerical environment.

    ``code_sha256`` digests the generating source files and ``environment`` the
    numerical package versions; together with ``config`` they must match when a run is
    resumed. ``code_revision`` (git) is provenance only.
    """

    preset: str
    config: Mapping[str, Any]
    pymc_generator_version: str
    corpus_schema_version: int
    code_sha256: str
    code_revision: str | None
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        # Store the configuration as its JSON round trip, so the digest written now is
        # the digest a reader recomputes (e.g. integer keys become strings).
        object.__setattr__(self, "config", json.loads(canonical_json(dict(self.config))))

    @property
    def config_sha256(self) -> str:
        return sha256_hex(canonical_json(dict(self.config)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "config": dict(self.config),
            "config_sha256": self.config_sha256,
            "pymc_generator_version": self.pymc_generator_version,
            "corpus_schema_version": self.corpus_schema_version,
            "code_sha256": self.code_sha256,
            "code_revision": self.code_revision,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: datetime
    request: RunRequest
    generator: GeneratorInfo
    contract_version: str = CONTRACT_VERSION

    @property
    def n_tasks(self) -> int:
        return self.request.n_tasks

    @property
    def worlds_per_task(self) -> int:
        return self.request.worlds_per_task

    @property
    def target_worlds(self) -> int:
        return self.request.worlds

    def encode(self) -> bytes:
        request = self.request
        body: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": format_timestamp(self.created_at),
            "request": request.to_dict(),
            "request_sha256": request.request_sha256,
            "n_tasks": self.n_tasks,
            "worlds_per_task": self.worlds_per_task,
            "target_worlds": self.target_worlds,
            "seed_rule": request.seed_rule,
        }
        if request.design is not None:
            body["structural_splits"] = [
                split.to_dict(request.seed) for split in request.design.splits
            ]
        body["generator"] = self.generator.to_dict()
        return _encode("run", body)


@dataclass(frozen=True)
class PayloadRef:
    path: str
    sha256: str
    size: int
    corpus_schema_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.size,
            "format": PAYLOAD_FORMAT,
            "corpus_schema_version": self.corpus_schema_version,
        }


@dataclass(frozen=True)
class Accounting:
    """Candidate accounting as reported by the generator.

    ``None`` means "not reported", never zero: the library does not count
    drawn-but-unevaluated surplus candidates, and a task whose generation call failed
    has no diagnostics at all.
    """

    candidates_evaluated: int | None
    rejected_candidates: int | None
    draw_failures: int | None
    surplus_discarded: int | None
    generation_failure: ErrorInfo | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_evaluated": self.candidates_evaluated,
            "rejected_candidates": self.rejected_candidates,
            "draw_failures": self.draw_failures,
            "surplus_discarded": self.surplus_discarded,
            "generation_failure": (
                None if self.generation_failure is None else self.generation_failure.to_dict()
            ),
        }


@dataclass(frozen=True)
class TaskRecord:
    run_id: str
    task_id: str
    task_index: int
    seed: int
    target_worlds: int
    accepted_worlds: int
    world_ids: tuple[str, ...]
    payload: PayloadRef | None
    accounting: Accounting
    started_at: datetime
    finished_at: datetime
    generation_seconds: float | None
    attempt_id: str
    worker: WorkerInfo
    contract_version: str = CONTRACT_VERSION

    @property
    def shortfall_worlds(self) -> int:
        return self.target_worlds - self.accepted_worlds

    @property
    def status(self) -> TaskStatus:
        if self.accepted_worlds == 0:
            return "zero"
        return "complete" if self.accepted_worlds == self.target_worlds else "shortfall"

    @property
    def content_key(self) -> tuple[int, str | None]:
        """Scientific content: equal keys are duplicates, different keys conflict."""
        return (self.accepted_worlds, None if self.payload is None else self.payload.sha256)

    def encode(self) -> bytes:
        return _encode(
            "task",
            {
                "run_id": self.run_id,
                "task_id": self.task_id,
                "task_index": self.task_index,
                "seed": self.seed,
                "target_worlds": self.target_worlds,
                "accepted_worlds": self.accepted_worlds,
                "shortfall_worlds": self.shortfall_worlds,
                "status": self.status,
                "world_ids": list(self.world_ids),
                "payload": None if self.payload is None else self.payload.to_dict(),
                "accounting": self.accounting.to_dict(),
                "timing": {
                    "started_at": format_timestamp(self.started_at),
                    "finished_at": format_timestamp(self.finished_at),
                    "generation_seconds": self.generation_seconds,
                },
                "attempt_id": self.attempt_id,
                "worker": self.worker.to_dict(),
            },
        )


@dataclass(frozen=True)
class AttemptRecord:
    """Provisional progress of one writer invocation.

    ``deadline_at`` is the time after which the attempt cannot still be running (its
    platform timeout), or ``None`` when no such bound exists.
    """

    run_id: str
    attempt_id: str
    state: AttemptState
    started_at: datetime
    updated_at: datetime
    deadline_at: datetime | None
    current_task_index: int | None
    published_task_ids: tuple[str, ...]
    reused_task_ids: tuple[str, ...]
    error: ErrorInfo | None
    worker: WorkerInfo
    contract_version: str = CONTRACT_VERSION

    def encode(self) -> bytes:
        return _encode(
            "attempt",
            {
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "state": self.state,
                "started_at": format_timestamp(self.started_at),
                "updated_at": format_timestamp(self.updated_at),
                "deadline_at": (
                    None if self.deadline_at is None else format_timestamp(self.deadline_at)
                ),
                "current_task_index": self.current_task_index,
                "published_task_ids": list(self.published_task_ids),
                "reused_task_ids": list(self.reused_task_ids),
                "error": None if self.error is None else self.error.to_dict(),
                "worker": self.worker.to_dict(),
            },
        )


# -- decoding --------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise RecordError(f"non-finite JSON number {name}")


def _load(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except RecordError:
        raise
    except (ValueError, RecursionError) as error:
        raise UnreadableRecordError(f"not a UTF-8 JSON document: {error}") from error


def _total[T](decode: Callable[[], T]) -> T:
    """Run a decoder so that unusable input can only surface as ``RecordError``."""
    try:
        return decode()
    except RecordError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise RecordError(f"unusable record: {type(error).__name__}: {error}") from error


def _as_object(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordError(f"{what} must be a JSON object")
    return value


def _envelope(data: bytes, kind: str) -> tuple[dict[str, Any], str]:
    if len(data) > MAX_RECORD_BYTES:
        raise RecordError(f"{kind} record exceeds {MAX_RECORD_BYTES} bytes")
    body = _as_object(_load(data), f"{kind} record")
    if body.get("contract") != CONTRACT:
        raise RecordError(f"contract must be {CONTRACT!r}, got {body.get('contract')!r}")
    version = body.get("contract_version")
    match = _VERSION.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise RecordError(f"invalid contract_version {version!r}")
    if int(match.group(1)) != SUPPORTED_MAJOR:
        raise UnsupportedVersionError(
            f"contract_version {version} is not supported (reader supports {SUPPORTED_MAJOR}.x)"
        )
    if body.get("kind") != kind:
        raise RecordError(f"kind must be {kind!r}, got {body.get('kind')!r}")
    return body, version


def _int(body: Mapping[str, Any], key: str, *, minimum: int = 0, optional: bool = False) -> Any:
    value = body.get(key)
    if value is None and optional:
        return None
    if type(value) is not int or not minimum <= value <= MAX_JSON_INT:
        raise RecordError(f"{key} must be an integer in [{minimum}, {MAX_JSON_INT}], got {value!r}")
    return value


def _exact(body: Mapping[str, Any], key: str, expected: int | str) -> None:
    value = body.get(key)
    if type(value) is not type(expected) or value != expected:
        raise RecordError(f"{key} must be {expected!r}, got {value!r}")


def _str(body: Mapping[str, Any], key: str, *, optional: bool = False) -> Any:
    value = body.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise RecordError(f"{key} must be a non-empty string, got {value!r}")
    return clean_message(value, limit=MAX_TEXT_CHARS)


def _non_negative(body: Mapping[str, Any], key: str) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise RecordError(f"{key} must be a number, got {value!r}")
    number = float(value) + 0.0  # + 0.0 turns -0.0 into 0.0
    if not math.isfinite(number) or number < 0:
        raise RecordError(f"{key} must be a finite number >= 0, got {value!r}")
    return number


def _error(value: Any, what: str) -> ErrorInfo | None:
    if value is None:
        return None
    body = _as_object(value, what)
    message = body.get("message", "")
    if not isinstance(message, str):
        raise RecordError(f"{what}.message must be a string")
    return ErrorInfo(type=_str(body, "type"), message=clean_message(message))


def _worker(value: Any) -> WorkerInfo:
    body = _as_object(value, "worker")
    platform = body.get("platform")
    if not isinstance(platform, str):
        raise RecordError(f"worker.platform must be a string, got {platform!r}")
    usd_per_hour = _non_negative(body, "usd_per_hour")
    if usd_per_hour is not None and usd_per_hour > MAX_USD_PER_HOUR:
        raise RecordError(f"usd_per_hour must be at most {MAX_USD_PER_HOUR}, got {usd_per_hour}")
    return WorkerInfo(
        platform=clean_message(platform, limit=MAX_TEXT_CHARS),
        cpu_count=_int(body, "cpu_count", minimum=1, optional=True),
        gpu_requested=_str(body, "gpu_requested", optional=True),
        gpu_name=_str(body, "gpu_name", optional=True),
        usd_per_hour=usd_per_hour,
    )


def _task_ids(value: Any, what: str, n_tasks: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RecordError(f"{what} must be a list")
    for task_id in value:
        if task_index_of(task_id) >= n_tasks:
            raise RecordError(f"{what} names {task_id}, outside the run's {n_tasks} tasks")
    return tuple(value)


def _environment(value: Any) -> dict[str, str]:
    body = _as_object(value, "generator.environment")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in body.items()):
        raise RecordError("generator.environment must map names to version strings")
    return {key: clean_message(item, limit=MAX_TEXT_CHARS) for key, item in body.items()}


def decode_run_manifest(data: bytes, *, run_id: str) -> RunManifest:
    """Validate ``runs/<run_id>/run.json``; any unusable input raises ``RecordError``."""
    return _total(lambda: _decode_run_manifest(data, run_id=run_id))


def _decode_run_manifest(data: bytes, *, run_id: str) -> RunManifest:
    body, version = _envelope(data, "run")
    _exact(body, "run_id", run_id)
    request = RunRequest.from_dict(body.get("request"))
    if request.run_id != run_id:
        raise RecordError(f"run_id {run_id} does not match its request digest ({request.run_id})")
    _exact(body, "request_sha256", request.request_sha256)
    _exact(body, "n_tasks", request.n_tasks)
    _exact(body, "worlds_per_task", request.worlds_per_task)
    _exact(body, "target_worlds", request.worlds)
    generator_body = _as_object(body.get("generator"), "generator")
    code_sha256 = generator_body.get("code_sha256")
    if not isinstance(code_sha256, str) or _SHA256.fullmatch(code_sha256) is None:
        raise RecordError("generator.code_sha256 must be 64 lowercase hex digits")
    generator = GeneratorInfo(
        preset=_str(generator_body, "preset"),
        config=_as_object(generator_body.get("config"), "generator.config"),
        pymc_generator_version=_str(generator_body, "pymc_generator_version"),
        corpus_schema_version=_int(generator_body, "corpus_schema_version", minimum=1),
        code_sha256=code_sha256,
        code_revision=_str(generator_body, "code_revision", optional=True),
        environment=_environment(generator_body.get("environment")),
    )
    if generator.preset != request.preset:
        raise RecordError("generator.preset does not match the request")
    _exact(generator_body, "config_sha256", generator.config_sha256)
    design = request.design
    if design is not None:
        # The reader re-derives every task's seed and member from the design: a run whose
        # recorded splits, rule or cycle differ was made by another version of it.
        _exact(body, "seed_rule", request.seed_rule)
        if body.get("structural_splits") != [
            split.to_dict(request.seed) for split in design.splits
        ]:
            raise RecordError("structural_splits do not match the design: it changed since the run")
        if generator.config.get("cycle") != list(design.cycle):
            raise RecordError("generator.config.cycle does not match the design's member cycle")
    return RunManifest(
        run_id=run_id,
        created_at=parse_timestamp(body.get("created_at")),
        request=request,
        generator=generator,
        contract_version=version,
    )


def _payload(value: Any, *, run_id: str, task_id: str, attempt_id: str) -> PayloadRef:
    body = _as_object(value, "payload")
    expected_path = payload_path(run_id, task_id, attempt_id)
    _exact(body, "path", expected_path)
    digest = body.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RecordError(f"payload.sha256 must be 64 lowercase hex digits, got {digest!r}")
    _exact(body, "format", PAYLOAD_FORMAT)
    return PayloadRef(
        path=expected_path,
        sha256=digest,
        size=_int(body, "bytes", minimum=1),
        corpus_schema_version=_int(body, "corpus_schema_version", minimum=1),
    )


def decode_task_record(
    data: bytes, *, manifest: RunManifest, task_id: str, attempt_id: str
) -> TaskRecord:
    """Validate ``runs/<run_id>/tasks/<task_id>/<attempt_id>.json`` against its run.

    Raises ``UnreadableRecordError`` for bytes that are not a JSON document,
    ``UnsupportedVersionError`` for another contract major version and ``RecordError``
    for anything else that makes the record unusable.
    """
    return _total(
        lambda: _decode_task_record(data, manifest=manifest, task_id=task_id, attempt_id=attempt_id)
    )


def _decode_task_record(
    data: bytes, *, manifest: RunManifest, task_id: str, attempt_id: str
) -> TaskRecord:
    body, version = _envelope(data, "task")
    run_id = manifest.run_id
    if not is_attempt_id(attempt_id):
        raise RecordError(f"invalid attempt_id {attempt_id!r}")
    _exact(body, "run_id", run_id)
    _exact(body, "task_id", task_id)
    _exact(body, "attempt_id", attempt_id)
    index = task_index_of(task_id)
    if index >= manifest.n_tasks:
        raise RecordError(f"{task_id} is outside the run's {manifest.n_tasks} tasks")
    _exact(body, "task_index", index)
    seed = manifest.request.task_seed(index)
    _exact(body, "seed", seed)
    target = manifest.worlds_per_task
    _exact(body, "target_worlds", target)
    accepted = _int(body, "accepted_worlds")
    if accepted > target:
        raise RecordError(f"accepted_worlds {accepted} exceeds target_worlds {target}")
    _exact(body, "shortfall_worlds", target - accepted)
    world_ids = body.get("world_ids")
    if not isinstance(world_ids, list) or len(world_ids) != accepted:
        raise RecordError("world_ids must list one id per accepted world")
    if world_ids != [world_id_for(run_id, task_id, row) for row in range(accepted)]:
        raise RecordError("world_ids must name rows 0..accepted_worlds-1 of this task")
    payload_value = body.get("payload")
    if (payload_value is None) != (accepted == 0):
        raise RecordError("payload must be null exactly when no world was accepted")
    payload = (
        None
        if payload_value is None
        else _payload(payload_value, run_id=run_id, task_id=task_id, attempt_id=attempt_id)
    )
    accounting_body = _as_object(body.get("accounting"), "accounting")
    accounting = Accounting(
        candidates_evaluated=_int(accounting_body, "candidates_evaluated", optional=True),
        rejected_candidates=_int(accounting_body, "rejected_candidates", optional=True),
        draw_failures=_int(accounting_body, "draw_failures", optional=True),
        surplus_discarded=_int(accounting_body, "surplus_discarded", optional=True),
        generation_failure=_error(
            accounting_body.get("generation_failure"), "accounting.generation_failure"
        ),
    )
    if accounting.generation_failure is not None and accepted:
        raise RecordError("generation_failure is only allowed when no world was accepted")
    evaluated, rejected = accounting.candidates_evaluated, accounting.rejected_candidates
    if evaluated is not None and evaluated < accepted + (rejected or 0):
        raise RecordError("candidates_evaluated is smaller than accepted + rejected candidates")
    timing = _as_object(body.get("timing"), "timing")
    started_at = parse_timestamp(timing.get("started_at"))
    finished_at = parse_timestamp(timing.get("finished_at"))
    if finished_at < started_at:
        raise RecordError("timing.finished_at precedes timing.started_at")
    record = TaskRecord(
        run_id=run_id,
        task_id=task_id,
        task_index=index,
        seed=seed,
        target_worlds=target,
        accepted_worlds=accepted,
        world_ids=tuple(world_ids),
        payload=payload,
        accounting=accounting,
        started_at=started_at,
        finished_at=finished_at,
        generation_seconds=_non_negative(timing, "generation_seconds"),
        attempt_id=attempt_id,
        worker=_worker(body.get("worker")),
        contract_version=version,
    )
    _exact(body, "status", record.status)
    return record


def decode_attempt_record(data: bytes, *, manifest: RunManifest, attempt_id: str) -> AttemptRecord:
    """Validate ``runs/<run_id>/attempts/<attempt_id>.json`` (provisional progress)."""
    return _total(lambda: _decode_attempt_record(data, manifest=manifest, attempt_id=attempt_id))


def _decode_attempt_record(data: bytes, *, manifest: RunManifest, attempt_id: str) -> AttemptRecord:
    body, version = _envelope(data, "attempt")
    if not is_attempt_id(attempt_id):
        raise RecordError(f"invalid attempt_id {attempt_id!r}")
    _exact(body, "run_id", manifest.run_id)
    _exact(body, "attempt_id", attempt_id)
    state = body.get("state")
    if state not in ATTEMPT_STATES:
        raise RecordError(f"state must be one of {ATTEMPT_STATES}, got {state!r}")
    current = _int(body, "current_task_index", optional=True)
    if current is not None and current >= manifest.n_tasks:
        raise RecordError(f"current_task_index {current} is outside the run")
    deadline = body.get("deadline_at")
    return AttemptRecord(
        run_id=manifest.run_id,
        attempt_id=attempt_id,
        state=state,
        started_at=parse_timestamp(body.get("started_at")),
        updated_at=parse_timestamp(body.get("updated_at")),
        deadline_at=None if deadline is None else parse_timestamp(deadline),
        current_task_index=current,
        published_task_ids=_task_ids(
            body.get("published_task_ids"), "published_task_ids", manifest.n_tasks
        ),
        reused_task_ids=_task_ids(body.get("reused_task_ids"), "reused_task_ids", manifest.n_tasks),
        error=_error(body.get("error"), "error"),
        worker=_worker(body.get("worker")),
        contract_version=version,
    )


# -- task outputs as pymc_generator.publication completions ----------------------------
#
# Runs written by this worker publish each task through ``pymc_generator.publication``
# (version-1 completion records): ``runs/<run_id>/tasks/<task_id>/attempts/<attempt_id>/``
# holds ``corpus.npz`` and ``completion.json``. The run manifest and the attempt records
# above stay the driver's records. Task records of the layout above (``<attempt_id>.json``
# next to ``payloads/``) remain readable for runs published before.

COMPLETION_NAME = "completion.json"
COMPLETION_PAYLOAD_NAME = "corpus.npz"
GENERATION_MODE = "DataGenerator.generate(defer_cell_split=True)"


def completion_dir(run_id: str, task_id: str, attempt_id: str) -> str:
    return f"{task_dir(run_id, task_id)}/attempts/{attempt_id}"


def completion_path(run_id: str, task_id: str, attempt_id: str) -> str:
    return f"{completion_dir(run_id, task_id, attempt_id)}/{COMPLETION_NAME}"


def completion_payload_path(run_id: str, task_id: str, attempt_id: str) -> str:
    return f"{completion_dir(run_id, task_id, attempt_id)}/{COMPLETION_PAYLOAD_NAME}"


def completion_world_id(task_id: str, row: int) -> str:
    """A world's id in a completion: unique within its run, and a publication identifier."""
    return f"{task_id}_w{row}"


def run_spec_sha256(manifest: RunManifest) -> str:
    """Digest of what fixes a run's outputs: its request and its recorded generator."""
    generator = manifest.generator
    spec = {
        "request_sha256": manifest.request.request_sha256,
        "config_sha256": generator.config_sha256,
        "code_sha256": generator.code_sha256,
        "corpus_schema_version": generator.corpus_schema_version,
        "environment": dict(generator.environment),
    }
    return sha256_hex(canonical_json(spec))


def completion_metadata(
    manifest: RunManifest,
    *,
    task_index: int,
    member: str,
    member_config: Mapping[str, Any],
    attempt_id: str,
    worker: WorkerInfo,
    candidates: int,
    rejected: int,
    published: int,
    started_at: datetime,
    finished_at: datetime,
    generation_seconds: float | None,
    draw_failures: int | None,
    generation_failure: ErrorInfo | None,
    redraws: int = 0,
    negative_controls: int = 0,
) -> dict[str, Any]:
    """What ``pymc_generator.publication`` publishes for one task (version 1).

    ``member`` is the preset that generated the task (a mix's member) and
    ``member_config`` its configuration as the manifest records it. Counts follow the
    publication's accounting (candidates = rejected + surplus + published, target =
    published + shortfall). ``rejected`` counts the candidates the library's filters
    rejected (the library reports no finer reasons: ``generator_filters``) and
    ``negative_controls`` the worlds the library accepted but the worker discarded
    because one of their controls went below zero (rejected too). ``redraws`` is the
    index of the task's last draw (0: its first), whose seed ``seed_context`` records as
    ``generation_seed``. ``timing``, ``draw_failures`` and ``generation_failure`` are
    additive fields the dashboard reads.
    """
    request = manifest.request
    run_id = manifest.run_id
    task_id = task_id_for(task_index)
    target = request.worlds_per_task
    generator = manifest.generator
    reasons = {"generator_filters": rejected, "negative_controls": negative_controls}
    rejected += negative_controls
    return {
        "run_id": run_id,
        "instruction_id": task_id,
        "task_id": task_id,
        "output_id": task_id,
        "attempt_id": attempt_id,
        "world_ids": [completion_world_id(task_id, row) for row in range(published)],
        "counts": {
            "target": target,
            "candidates": candidates,
            "rejected": rejected,
            "surplus": candidates - rejected - published,
            "published": published,
            "shortfall": target - published,
        },
        "rejection_reasons": {reason: count for reason, count in reasons.items() if count},
        "provenance": {
            "run_spec_sha256": run_spec_sha256(manifest),
            "generation_mode": GENERATION_MODE,
            "generation_version": generator.code_revision or generator.code_sha256,
            "environment": {
                **generator.environment,
                "pymc_generator": generator.pymc_generator_version,
                "code_sha256": generator.code_sha256,
                "worker": worker.to_dict(),
            },
            "seed_context": {
                **request.seed_context(task_index),
                "generation_seed": structural.redraw_seed(request.task_seed(task_index), redraws),
                "redraws": redraws,
            },
            "effective_configuration": {
                "run_preset": request.preset,
                "preset": member,
                "configuration": dict(member_config),
            },
            "cell_namespace": f"{run_id}_{task_id}",
        },
        "timing": {
            "started_at": format_timestamp(started_at),
            "finished_at": format_timestamp(finished_at),
            "generation_seconds": generation_seconds,
        },
        "draw_failures": draw_failures,
        "generation_failure": None if generation_failure is None else generation_failure.to_dict(),
    }


def task_record_from_completion(manifest: RunManifest, completion: Mapping[str, Any]) -> TaskRecord:
    """The task record the dashboard reads for one completion of ``manifest``'s run.

    ``completion`` has passed ``pymc_generator.publication`` discovery (schema, accounting
    and location); this checks that it is a task of this run, with its seed and target,
    and reads the fields this worker adds. Raises ``RecordError`` otherwise.
    """
    return _total(lambda: _task_record_from_completion(manifest, completion))


def _task_record_from_completion(
    manifest: RunManifest, completion: Mapping[str, Any]
) -> TaskRecord:
    request = manifest.request
    task_id = completion["task_id"]
    index = task_index_of(task_id)
    if index >= manifest.n_tasks:
        raise RecordError(f"{task_id} is not a task of this run")
    if completion["run_id"] != manifest.run_id:
        raise RecordError(f"run_id must be {manifest.run_id!r}, got {completion['run_id']!r}")
    counts = completion["counts"]
    _exact(counts, "target", manifest.worlds_per_task)
    provenance = completion["provenance"]
    context = provenance["seed_context"]
    task_seed = request.task_seed(index)
    _exact(context, "task_seed", task_seed)
    design = request.design
    if design is not None or "redraws" in context:  # 1.2 records; earlier ones omit them
        redraws = _int(context, "redraws")
        if redraws > (0 if design is None else structural.MAX_REDRAWS):
            raise RecordError(f"redraws {redraws} exceeds what this run allows")
        _exact(context, "generation_seed", structural.redraw_seed(task_seed, redraws))
    if design is not None:
        expected = design.seed_context(request.seed, index)
        if context.get("split") != expected["split"] or context.get("regime") != expected["regime"]:
            raise RecordError(f"{task_id}'s split or regime is not the design's")
    timing = _as_object(completion.get("timing"), "timing")
    started_at = parse_timestamp(timing.get("started_at"))
    finished_at = parse_timestamp(timing.get("finished_at"))
    if finished_at < started_at:
        raise RecordError("timing.finished_at precedes timing.started_at")
    attempt_id = completion["attempt_id"]
    payload = None
    if completion["payload"] is not None:
        payload = PayloadRef(
            path=completion_payload_path(manifest.run_id, task_id, attempt_id),
            sha256=completion["payload"]["sha256"],
            size=completion["payload"]["bytes"],
            corpus_schema_version=completion["corpus_schema_version"],
        )
    return TaskRecord(
        run_id=manifest.run_id,
        task_id=task_id,
        task_index=index,
        seed=request.task_seed(index),
        target_worlds=counts["target"],
        accepted_worlds=counts["published"],
        world_ids=tuple(completion["world_ids"]),
        payload=payload,
        accounting=Accounting(
            candidates_evaluated=counts["candidates"],
            rejected_candidates=counts["rejected"],
            draw_failures=_int(completion, "draw_failures", optional=True),
            surplus_discarded=counts["surplus"],
            generation_failure=_error(completion.get("generation_failure"), "generation_failure"),
        ),
        started_at=started_at,
        finished_at=finished_at,
        generation_seconds=_non_negative(timing, "generation_seconds"),
        attempt_id=attempt_id,
        worker=_worker(provenance["environment"]["worker"]),
    )
