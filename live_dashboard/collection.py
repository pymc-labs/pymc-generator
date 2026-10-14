"""Follow the published collection: each ``Collector.refresh()`` rebuilds the snapshot.

The collector re-derives everything from the store on every refresh, so arrival order,
restarts and several readers cannot change what it shows: two collectors refreshing the
same store at the same time produce equal runs. Only expensive, content-addressed work
is cached, per collector: payload digest verification (keyed by path and digest, trusted
while the file keeps its verified size) and extraction (keyed by digest and
``EXTRACTION_VERSION``). Nothing that failed for a possibly transient reason is cached.

Trade-off: an in-place, same-size change of an already verified payload is not noticed
until the process restarts. The protocol never rewrites payloads (they are create-only
and named by attempt), so this only matters for out-of-protocol edits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime

from live_dashboard import publication as pub
from live_dashboard.extract import (
    EXTRACTION_VERSION,
    ExtractionError,
    PayloadMetrics,
    extract_payload,
)
from live_dashboard.snapshot import (
    AttemptView,
    CollectionSnapshot,
    ImpactUnit,
    RunSnapshot,
    RunState,
    TaskState,
    TaskStatus,
    WorldMetrics,
)
from live_dashboard.storage import DirectoryStore

# Record problems, strongest first: the status of a task without a valid record. A newer
# contract version needs a reader upgrade, invalid content a writer fix, and a file that
# is not JSON yet may still be completing, so it ranks last.
_PROBLEMS: tuple[TaskStatus, ...] = ("unsupported", "invalid", "unreadable")
_RECORD_SUFFIX = ".json"
_RUN_STATE_OF_NEWEST_ATTEMPT: dict[str, RunState] = {
    "running": "running",
    "lost": "lost",
    "failed": "failed",
    "finished": "stopped",
    "superseded": "stopped",
}

type _TaskResult = tuple[TaskState, tuple[WorldMetrics, ...], tuple[ImpactUnit, ...]]


def _describe(error: BaseException) -> str:
    return pub.ErrorInfo.from_exception(error).describe()


class Collector:
    """Builds ``CollectionSnapshot``s from a publication store.

    ``refresh()`` never raises: a store-level failure keeps the previous runs and sets
    ``last_error``; problems of one run, task or record are reported in the snapshot
    without affecting the others. Call ``refresh()`` from one thread only (it owns the
    caches). Other threads read ``snapshot``, an immutable object replaced by a single
    attribute assignment, so they always see one complete refresh.
    """

    def __init__(
        self,
        store: DirectoryStore,
        *,
        clock: Callable[[], datetime] = pub.utc_now,
        extractor: Callable[[bytes], PayloadMetrics] = extract_payload,
    ) -> None:
        self._store = store
        self._clock = clock
        self._extractor = extractor
        # (payload path, sha256) -> size of the bytes whose digest matched.
        self._verified: dict[tuple[str, str], int] = {}
        # (sha256, EXTRACTION_VERSION) -> metrics, or the deterministic content error.
        self._extracted: dict[tuple[str, int], PayloadMetrics | ExtractionError] = {}
        self.snapshot = CollectionSnapshot(sequence=0, refreshed_at=None, last_error=None, runs=())

    def refresh(self) -> None:
        """Re-read the store and replace ``snapshot``."""
        previous = self.snapshot
        try:
            self._store.refresh()
            now = self._clock()
            runs, issues = self._collect(now)
        except Exception as error:
            self.snapshot = CollectionSnapshot(
                sequence=previous.sequence + 1,
                refreshed_at=previous.refreshed_at,
                last_error=_describe(error),
                runs=previous.runs,
                issues=previous.issues,
            )
            return
        self.snapshot = CollectionSnapshot(
            sequence=previous.sequence + 1,
            refreshed_at=now,
            last_error=None,
            runs=runs,
            issues=issues,
        )

    # -- runs ----------------------------------------------------------------------

    def _collect(self, now: datetime) -> tuple[tuple[RunSnapshot, ...], tuple[str, ...]]:
        runs: list[RunSnapshot] = []
        issues: list[str] = []
        for name in self._store.list_dirs(pub.RUNS_DIR):
            if not pub.is_run_id(name):
                issues.append(pub.clean_message(f"{pub.RUNS_DIR}/{name}: not a run id (ignored)"))
                continue
            try:
                run = self._collect_run(name, now)
            except pub.RecordError as error:
                issues.append(pub.clean_message(f"{pub.RUNS_DIR}/{name}/run.json: {error}"))
                continue
            except Exception as error:
                issues.append(pub.clean_message(f"{pub.RUNS_DIR}/{name}: {_describe(error)}"))
                continue
            if run is not None:
                runs.append(run)
        runs.sort(key=lambda run: (run.created_at, run.run_id), reverse=True)
        return tuple(runs), tuple(issues)

    def _read_record(self, path: str) -> bytes | None:
        """A record's bytes, ``None`` when absent; oversized files are refused unread."""
        size = self._store.size(path)
        if size is None:
            return None
        if size > pub.MAX_RECORD_BYTES:
            raise pub.RecordError(f"record exceeds {pub.MAX_RECORD_BYTES} bytes ({size} bytes)")
        try:
            return self._store.read_bytes(path)
        except FileNotFoundError:
            return None

    def _collect_run(self, name: str, now: datetime) -> RunSnapshot | None:
        data = self._read_record(pub.run_manifest_path(name))
        if data is None:
            return None  # not published yet: the writer commits run.json first
        manifest = pub.decode_run_manifest(data, run_id=name)
        issues: list[str] = []
        task_ids = [pub.task_id_for(index) for index in range(manifest.n_tasks)]
        present = set(self._store.list_dirs(pub.tasks_dir(name)))
        for stray in sorted(present.difference(task_ids)):
            issues.append(pub.clean_message(f"tasks/{stray}: not a task of this run (ignored)"))
        tasks: list[TaskState] = []
        worlds: list[WorldMetrics] = []
        impacts: list[ImpactUnit] = []
        for index, task_id in enumerate(task_ids):
            if task_id not in present:
                tasks.append(TaskState(task_id, index, "pending", manifest.worlds_per_task))
                continue
            try:
                state, task_worlds, task_impacts = self._collect_task(manifest, task_id)
            except Exception as error:
                state = TaskState(
                    task_id,
                    index,
                    "invalid",
                    manifest.worlds_per_task,
                    detail=pub.clean_message(f"unexpected error: {_describe(error)}"),
                )
                task_worlds, task_impacts = (), ()
            tasks.append(state)
            worlds.extend(task_worlds)
            impacts.extend(task_impacts)
        attempts = self._attempts(manifest, now, issues)
        return RunSnapshot(
            run_id=manifest.run_id,
            label=manifest.request.label,
            preset=manifest.request.preset,
            created_at=manifest.created_at,
            state=_run_state(tasks, attempts),
            n_tasks=manifest.n_tasks,
            worlds_per_task=manifest.worlds_per_task,
            target_worlds=manifest.target_worlds,
            seed=manifest.request.seed,
            code_revision=manifest.generator.code_revision,
            tasks=tuple(tasks),
            worlds=tuple(worlds),
            impacts=tuple(impacts),
            attempts=attempts,
            issues=tuple(issues),
        )

    def _attempts(
        self, manifest: pub.RunManifest, now: datetime, issues: list[str]
    ) -> tuple[AttemptView, ...]:
        directory = pub.attempts_dir(manifest.run_id)
        records: list[pub.AttemptRecord] = []
        for name in self._store.list_files(directory):
            attempt_id = name.removesuffix(_RECORD_SUFFIX)
            try:
                if attempt_id == name:
                    raise pub.RecordError("not an <attempt_id>.json record")
                data = self._read_record(f"{directory}/{name}")
                if data is not None:
                    records.append(
                        pub.decode_attempt_record(data, manifest=manifest, attempt_id=attempt_id)
                    )
            except pub.RecordError as error:
                issues.append(pub.clean_message(f"attempts/{name}: {error}"))
            except Exception as error:
                issues.append(pub.clean_message(f"attempts/{name}: {_describe(error)}"))
        records.sort(key=lambda record: (record.started_at, record.attempt_id), reverse=True)
        views = []
        for position, record in enumerate(records):
            status = record.state
            if status == "running":
                if record.deadline_at is not None and now > record.deadline_at:
                    status = "lost"  # its platform timeout has passed: it cannot still run
                elif position > 0:
                    status = "superseded"  # a newer attempt took over without closing it
            views.append(
                AttemptView(
                    attempt_id=record.attempt_id,
                    status=status,
                    started_at=record.started_at,
                    updated_at=record.updated_at,
                    deadline_at=record.deadline_at,
                    current_task_index=record.current_task_index,
                    error=None if record.error is None else record.error.describe(),
                    gpu_requested=record.worker.gpu_requested,
                    gpu_name=record.worker.gpu_name,
                )
            )
        return tuple(views)

    # -- tasks ---------------------------------------------------------------------

    def _task_record(
        self, manifest: pub.RunManifest, task_id: str, name: str
    ) -> pub.TaskRecord | tuple[TaskStatus, str] | None:
        """Decode one file of a task directory: a record, a (problem, message) or None."""
        attempt_id = name.removesuffix(_RECORD_SUFFIX)
        if attempt_id == name:
            return "invalid", "not an <attempt_id>.json record"
        try:
            data = self._read_record(f"{pub.task_dir(manifest.run_id, task_id)}/{name}")
            if data is None:
                return None  # removed since the listing
            return pub.decode_task_record(
                data, manifest=manifest, task_id=task_id, attempt_id=attempt_id
            )
        except pub.UnreadableRecordError as error:
            return "unreadable", str(error)
        except pub.UnsupportedVersionError as error:
            return "unsupported", str(error)
        except pub.RecordError as error:
            return "invalid", str(error)

    def _collect_task(self, manifest: pub.RunManifest, task_id: str) -> _TaskResult:
        index = pub.task_index_of(task_id)
        target = manifest.worlds_per_task
        records: list[pub.TaskRecord] = []
        problems: list[tuple[TaskStatus, str, str]] = []
        for name in self._store.list_files(pub.task_dir(manifest.run_id, task_id)):
            outcome = self._task_record(manifest, task_id, name)
            if isinstance(outcome, pub.TaskRecord):
                records.append(outcome)
            elif outcome is not None:
                problems.append((outcome[0], name, outcome[1]))
        counts = Counter(kind for kind, _, _ in problems)
        if not records:
            if not problems:
                return TaskState(task_id, index, "pending", target), (), ()
            kind = next(kind for kind in _PROBLEMS if counts[kind])
            name, message = next((name, message) for k, name, message in problems if k == kind)
            detail = f"{name}: {message}"
            if len(problems) > 1:
                detail += f"; problem records: {_count_text(counts)}"
            return TaskState(task_id, index, kind, target, detail=pub.clean_message(detail)), (), ()
        notes = [f"ignored problem records: {_count_text(counts)}"] if problems else []
        groups: dict[tuple[int, str | None], list[pub.TaskRecord]] = {}
        for record in sorted(records, key=lambda record: record.attempt_id):
            groups.setdefault(record.content_key, []).append(record)
        if len(groups) > 1:
            versions = " vs ".join(
                f"{', '.join(record.attempt_id for record in group)} ({_content_text(content)})"
                for content, group in groups.items()
            )
            detail = "; ".join([f"records disagree: {versions}", *notes])
            conflict = TaskState(
                task_id, index, "conflict", target, detail=pub.clean_message(detail)
            )
            return conflict, (), ()
        (group,) = groups.values()
        authoritative, duplicates = group[0], len(group) - 1

        def state(status: TaskStatus, *extra: str) -> TaskState:
            detail = "; ".join([*extra, *notes]) or None
            return _record_state(authoritative, status, duplicates, detail)

        if authoritative.payload is None:
            return state("zero"), (), ()
        key = (authoritative.payload.sha256, EXTRACTION_VERSION)
        found = self._verified_payload(group, need_bytes=key not in self._extracted)
        if found is None:
            return state("unavailable", "no payload matches its recorded size and sha256"), (), ()
        record, data = found
        # Equal bytes under another path or task reuse the extraction; verifying this
        # path may still have read them.
        result = self._extracted.get(key)
        if result is None:
            try:
                result = self._extract(key, data)
            except Exception as error:
                # Not a content verdict: reported now and retried on the next refresh.
                return state("invalid", f"payload extraction failed: {_describe(error)}"), (), ()
        if isinstance(result, ExtractionError):
            return state("invalid", f"payload cannot be used: {result}"), (), ()
        if result.n_worlds != record.accepted_worlds:
            mismatch = (
                f"payload holds {result.n_worlds} worlds, the record accepts "
                f"{record.accepted_worlds}"
            )
            return state("invalid", mismatch), (), ()
        return state("published"), *_world_metrics(record, result)

    def _verified_payload(
        self, group: Sequence[pub.TaskRecord], *, need_bytes: bool
    ) -> tuple[pub.TaskRecord, bytes | None] | None:
        """The first record (attempt order) whose payload matches its size and digest.

        Every record of ``group`` has a payload: they share a content key with worlds.
        Without ``need_bytes`` a cached verification is trusted while the file keeps the
        verified size, and no bytes are returned. Otherwise the payload is read (after
        a size check) and its digest recomputed. Failures are never cached: a payload
        that is missing or damaged now may be repaired by a later publication.
        """
        for record in group:
            reference = record.payload
            key = (reference.path, reference.sha256)
            size = self._store.size(reference.path)
            verified = self._verified.get(key)
            if verified is not None and verified != size:
                del self._verified[key]
                verified = None
            if verified is not None and not need_bytes:
                return record, None
            if size != reference.size:
                continue
            try:
                data = self._store.read_bytes(reference.path)
            except FileNotFoundError:
                continue
            if len(data) != reference.size or pub.sha256_hex(data) != reference.sha256:
                self._verified.pop(key, None)
                continue
            self._verified[key] = len(data)
            return record, data
        return None

    def _extract(self, key: tuple[str, int], data: bytes) -> PayloadMetrics | ExtractionError:
        """Extract verified bytes once; only deterministic outcomes are cached."""
        try:
            result: PayloadMetrics | ExtractionError = self._extractor(data)
        except ExtractionError as error:
            result = error
        self._extracted[key] = result
        return result


def _count_text(counts: Counter[TaskStatus]) -> str:
    return ", ".join(f"{counts[kind]} {kind}" for kind in _PROBLEMS if counts[kind])


def _content_text(content: tuple[int, str | None]) -> str:
    accepted, digest = content
    return f"{accepted} worlds, " + ("no payload" if digest is None else f"sha256 {digest}")


def _record_state(
    record: pub.TaskRecord, status: TaskStatus, duplicates: int, detail: str | None
) -> TaskState:
    """A task's state carrying its authoritative record's timing and accounting."""
    accounting = record.accounting
    failure = accounting.generation_failure
    return TaskState(
        task_id=record.task_id,
        task_index=record.task_index,
        status=status,
        target_worlds=record.target_worlds,
        accepted_worlds=record.accepted_worlds,
        started_at=record.started_at,
        finished_at=record.finished_at,
        generation_seconds=record.generation_seconds,
        candidates_evaluated=accounting.candidates_evaluated,
        rejected_candidates=accounting.rejected_candidates,
        draw_failures=accounting.draw_failures,
        generation_failure=None if failure is None else failure.describe(),
        attempt_id=record.attempt_id,
        duplicate_records=duplicates,
        detail=None if detail is None else pub.clean_message(detail),
    )


def _world_metrics(
    record: pub.TaskRecord, metrics: PayloadMetrics
) -> tuple[tuple[WorldMetrics, ...], tuple[ImpactUnit, ...]]:
    """Attach this task's world identities to identity-free payload metrics."""
    worlds = tuple(
        WorldMetrics(
            world_id=record.world_ids[row],
            task_id=record.task_id,
            channel_share=metrics.channel[row],
            control_share=metrics.control[row],
            baseline_share=metrics.baseline[row],
            adstock_forms=metrics.adstock_forms[row],
            saturation_forms=(
                None if metrics.saturation_forms is None else metrics.saturation_forms[row]
            ),
        )
        for row in range(metrics.n_worlds)
    )
    impacts = tuple(
        ImpactUnit(
            world_id=record.world_ids[row], task_id=record.task_id, channel=label, share=share
        )
        for row, label, share in zip(
            metrics.impact_rows, metrics.impact_labels, metrics.impact_shares, strict=True
        )
    )
    return worlds, impacts


def _run_state(tasks: Sequence[TaskState], attempts: Sequence[AttemptView]) -> RunState:
    if not any(task.status in ("pending", "unreadable") for task in tasks):
        return "complete"
    if not attempts:
        return "waiting"
    return _RUN_STATE_OF_NEWEST_ATTEMPT[attempts[0].status]
