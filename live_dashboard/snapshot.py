"""Immutable view of the published collection, built by the collector, read by the UI.

Only ``published`` tasks contribute worlds and impact units; every other task status is
reported separately so progress, shortfalls and failures never mix with the
distributions of the consumable dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

TaskStatus = Literal[
    "published",  # authoritative record, payload verified and extracted: worlds consumable
    "loading",  # authoritative record whose payload this collector has not read yet
    "zero",  # authoritative record that accepted no world (completed, full shortfall)
    "pending",  # no record yet
    "unreadable",  # record file(s) present but not decodable as JSON (e.g. still partial)
    "invalid",  # record or payload fails validation/extraction
    "unsupported",  # record uses an unsupported contract major version
    "unavailable",  # authoritative record whose payload is missing or fails its digest
    "conflict",  # records with different scientific content for the same task
]
AttemptStatus = Literal["running", "finished", "failed", "lost", "superseded"]
RunState = Literal["waiting", "running", "complete", "stopped", "failed", "lost"]


@dataclass(frozen=True)
class TaskState:
    """One logical task; timing and accounting come from its authoritative record."""

    task_id: str
    task_index: int
    status: TaskStatus
    target_worlds: int
    accepted_worlds: int = 0  # authoritative record's count (0 without one)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    generation_seconds: float | None = None
    candidates_evaluated: int | None = None
    rejected_candidates: int | None = None
    draw_failures: int | None = None
    generation_failure: str | None = None  # "Type: message" of a zero-world outcome
    attempt_id: str | None = None  # attempt that wrote the authoritative record
    duplicate_records: int = 0  # extra records with identical content (deduplicated)
    detail: str | None = None  # why a task is unreadable/invalid/unsupported/...

    @property
    def consumable_worlds(self) -> int:
        return self.accepted_worlds if self.status == "published" else 0

    @property
    def has_timing(self) -> bool:
        return self.started_at is not None and self.finished_at is not None


@dataclass(frozen=True)
class WorldMetrics:
    """Composition of one consumable world: signed shares of its total outcome.

    ``adstock_forms`` and ``saturation_forms`` are the families its active channels use,
    in the library's order, and ``structural_pairs`` their distinct (saturation, adstock)
    pairs, sorted; ``saturation_forms`` and ``structural_pairs`` are ``None`` when its
    payload does not record saturation families. ``active_channels`` of ``channel_slots``
    channels and ``active_controls`` of ``control_slots`` controls are active (its
    columns); ``effect_channels`` and ``effect_controls`` of them have an effect (a direct
    edge to the target). ``third_weeks`` are the weeks of the first, middle and last third
    of its series, and ``channel_zero_weeks`` / ``control_zero_weeks`` hold, per active
    channel / control in slot order, its weeks at exactly 0 in each third.
    """

    world_id: str
    task_id: str
    channel_share: float  # direct + indirect treatment contributions
    control_share: float  # covariate contributions (net, can be negative)
    baseline_share: float  # intrinsic baseline + latent contributions + noise
    adstock_forms: tuple[str, ...]
    saturation_forms: tuple[str, ...] | None
    structural_pairs: tuple[tuple[str, str], ...] | None
    active_channels: int
    active_controls: int
    effect_channels: int
    effect_controls: int
    channel_slots: int
    control_slots: int
    third_weeks: tuple[int, int, int] = (0, 0, 0)
    channel_zero_weeks: tuple[tuple[int, int, int], ...] = ()
    control_zero_weeks: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True)
class ImpactUnit:
    """One channel with an effect of one consumable world: its direct contribution share.

    Channels with an effect are the active channels with a direct edge to the target.
    """

    world_id: str
    task_id: str
    channel: str  # slot label "C1".."CK"
    share: float


@dataclass(frozen=True)
class ControlUnit:
    """One control with an effect of one consumable world: its direct contribution share.

    Controls with an effect are the active controls with a direct edge to the target.
    """

    world_id: str
    task_id: str
    control: str  # slot label "Z1".."ZM"
    share: float


@dataclass(frozen=True)
class AttemptView:
    attempt_id: str
    status: AttemptStatus
    started_at: datetime
    updated_at: datetime
    deadline_at: datetime | None
    current_task_index: int | None
    error: str | None
    gpu_requested: str | None
    gpu_name: str | None
    usd_per_hour: float | None = None  # the worker's recorded list price, if billed


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    label: str
    preset: str
    created_at: datetime
    state: RunState
    n_tasks: int
    worlds_per_task: int
    target_worlds: int
    seed: int
    code_revision: str | None
    tasks: tuple[TaskState, ...]  # one per task index, pending placeholders included
    worlds: tuple[WorldMetrics, ...]  # consumable worlds, ordered by (task_index, row)
    impacts: tuple[ImpactUnit, ...]  # consumable worlds' channels with an effect, same order
    attempts: tuple[AttemptView, ...]  # newest first
    issues: tuple[str, ...] = ()  # e.g. undecodable attempt records
    control_impacts: tuple[ControlUnit, ...] = ()  # consumable worlds' controls with an effect

    @property
    def published_worlds(self) -> int:
        return sum(task.consumable_worlds for task in self.tasks)

    @property
    def recorded_tasks(self) -> int:
        """Tasks with any record (every status except pending)."""
        return sum(task.status != "pending" for task in self.tasks)

    @property
    def loading_tasks(self) -> int:
        """Published tasks whose payloads the collector has not read yet."""
        return sum(task.status == "loading" for task in self.tasks)


@dataclass(frozen=True)
class CollectionSnapshot:
    """Everything the UI shows; replaced as a whole after each refresh."""

    sequence: int  # increments on every completed refresh attempt (cache key, not shown)
    refreshed_at: datetime | None  # last successful refresh
    last_error: str | None  # store-level failure of the latest refresh, if any
    runs: tuple[RunSnapshot, ...]  # newest first
    issues: tuple[str, ...] = ()  # run manifests that could not be used
    run_index: Mapping[str, RunSnapshot] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.run_index:
            object.__setattr__(self, "run_index", {run.run_id: run for run in self.runs})

    @property
    def generating(self) -> bool:
        """Whether any run currently has a live writer."""
        return any(run.state == "running" for run in self.runs)

    @property
    def loading(self) -> bool:
        """Whether published payloads still wait to be read (a bounded refresh)."""
        return any(run.loading_tasks for run in self.runs)
