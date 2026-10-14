"""Generation worker: publish the logical tasks of one run into a store, resumably.

A run is a fixed list of logical tasks (``RunRequest``); task ``i`` generates
``worlds_per_task`` worlds with seed ``seed + i``. For each task the writer persists and
commits the payload first and the immutable completion record second, so a reader never
sees a record whose payload is not yet published. Every record is decoded with the
contract decoder before it is written: the writer never publishes what readers would
reject. Rerunning the same request reuses every task that already has a usable record,
so a crashed or timed-out attempt is resumed by running it again. The run manifest pins
the generator configuration, source digest and numerical environment; resuming with a
different generator is refused because one run must never mix two generators' outputs.

``pymc_generator`` is imported inside functions only, so importing this module (e.g. by
the CLI of the CPU-only dashboard image) does not load PyMC or PyTensor.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata
import os
import platform
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from live_dashboard.deployment import REVISION_ENV, check_demo_bounds
from live_dashboard.publication import (
    MAX_RECORD_BYTES,
    Accounting,
    AttemptRecord,
    AttemptState,
    ErrorInfo,
    GeneratorInfo,
    PayloadRef,
    RecordError,
    RunManifest,
    RunRequest,
    TaskRecord,
    WorkerInfo,
    attempt_record_path,
    clean_message,
    decode_attempt_record,
    decode_run_manifest,
    decode_task_record,
    is_attempt_id,
    new_attempt_id,
    payload_path,
    run_manifest_path,
    sha256_hex,
    task_dir,
    task_id_for,
    task_record_path,
    utc_now,
    world_id_for,
)
from live_dashboard.revision import code_revision
from live_dashboard.storage import DirectoryStore

if TYPE_CHECKING:
    from pymc_generator.data_generator import DataGenerator
    from pymc_generator.sampler import SCMPrior

# The library's cell-fill failure: a cell could not accept its worlds within the top-up
# rounds. Only this outcome is a scientific result (a task with zero accepted worlds);
# any other exception is an infrastructure or code failure and fails the attempt.
_EXHAUSTED = re.compile(r"tasks accepted after \d+ rounds")
# Stop before a task when less than this much time, or twice the longest task so far,
# is left before the deadline (the time the platform kills the attempt), so the attempt
# ends cleanly instead of being killed in the middle of a task.
_MIN_DEADLINE_MARGIN_S = 90.0
_ENVIRONMENT_PACKAGES = ("numpy", "scipy", "pytensor", "pymc", "pymc-marketing")
_CONFLICT_FIELDS = ("config_sha256", "code_sha256", "environment")


class RequestError(ValueError):
    """The request cannot be run as asked (demo bounds, unknown preset, bad attempt id).

    Raised before anything is written; retrying the same request cannot succeed.
    """


class RunConflictError(Exception):
    """The stored run cannot be resumed by this generator (or this request).

    Raised before anything is written: resuming would mix outputs of two generators.
    """


# -- presets ---------------------------------------------------------------------------

# Verbatim from the trajectory-showcase notebook (docs/examples/dev/trajectory-showcase.ipynb,
# cell 7), so a live run shows the same variety the notebook reports on.
_SHOWCASE_COUNT_RANGES = {
    "n_treatments_active_range": (1, 10),
    "n_covariates_active_range": (1, 10),
}
_SHOWCASE_PER_INPUT = {  # applied to channels and controls alike
    # Zeros come only from flighting: on for W of every P weeks, at a random phase.
    "flighting_inclusion_prob": 10 / 11,
    "flighting_period_weeks_range": (14, 26),  # P: integer uniform
    "flighting_duty_range": (0.03, 0.99),  # W = clip(floor(duty * P + 0.5), 1, P - 1)
    "onset_inclusion_prob": 0.0,  # fixed by the design (the notebook report asserts it)
    "offset_inclusion_prob": 0.0,  # fixed by the design (the notebook report asserts it)
    "trend_inclusion_prob": 1.0,
    "seasonal_inclusion_prob": 0.5,
    "seasonal_period_weeks_range": (52.0, 52.0),
    "level_jump_inclusion_prob": 0.5,
    "level_jump_count": 1,  # held steps per input that carries a jump
    "pulse_inclusion_prob": 0.5,
    "hf_inclusion_prob": 1.0,
}
_SHOWCASE_KNOBS = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,
    "n_time_steps": 104,
    **_SHOWCASE_COUNT_RANGES,
    "draws_per_cell": 1,  # fixed by the design (the notebook report asserts it)
    "outcome_std_mode": "relative",  # fixed by the design (the notebook report asserts it)
    # Effect sizes: each input's contribution at its reference level, and the baseline level.
    "treatment_reference_contribution_range": (0.0, 3.0),
    "covariate_reference_contribution_range": (0.0, 3.0),
    "covariate_reference_scale": 1.0,
    "rw_baseline_mean_range": (0.05, 8.0),
    "baseline_floor": 0.0,
    # Input levels, volatility and smoothness.
    "rw_positive_mean_range": (0.1, 8.0),  # channel walk mean, before softplus
    "rw_treatment_std_range": (0.1, 1.2),  # channel walk std, relative to its level
    "rw_covariate_mean_range": (0.1, 2.0),
    "rw_std_sigma": 0.5,  # control walk std ~ HalfNormal(rw_std_sigma)
    "rw_smoothness_alpha": 1.0,  # walk smoothness ~ Beta(alpha, beta)
    "rw_smoothness_beta": 1.0,
    "rw_smoothness_max_weeks": 26,  # smoothness 1 = a 26-week moving average
    # Baseline walk and outcome noise std, relative to the channels' joint effect size.
    "rw_baseline_std_range": (0.0, 0.093),
    "rw_outcome_std_range": (0.01, 0.028),
    # Texture (the make_scm_prior defaults, echoed so every range is visible here).
    "treatment_hf_sigma_range": (0.08, 0.6),
    "treatment_pulse_prob_range": (0.0, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "covariate_hf_sigma_range": (0.1, 0.8),
    "covariate_pulse_prob_range": (0.0, 0.25),
    "covariate_pulse_amp_range": (0.5, 3.0),
    # Level forms.
    "treatment_trend_log_change_range": (-1.0, 1.0),
    "covariate_trend_change_range": (-1.0, 1.0),
    "treatment_seasonal_amplitude_range": (0.1, 0.5),
    "covariate_seasonal_amplitude_range": (0.2, 1.0),
    "treatment_level_jump_factor_range": (0.5, 2.0),  # log-uniform
    "covariate_level_jump_size_range": (-1.0, 1.0),
    # Response mechanisms: geometric carryover and Michaelis–Menten saturation.
    "carryover_alpha_range": (0.05, 0.95),
    **{
        f"{role}_{key}": value
        for role in ("treatment", "covariate")
        for key, value in _SHOWCASE_PER_INPUT.items()
    },
}
_SHOWCASE_MM_KAPPA_MULT_RANGE = (0.05, 20.0)  # log-uniform


def _showcase_variety(worlds_per_task: int) -> SCMPrior:
    """The notebook's variety design, one world per cell, ``worlds_per_task`` cells.

    Documented deviation: the notebook allocates active counts with a stratified design
    over its whole cohort, which a stream of small independent tasks cannot carry, so
    counts are drawn independently per cell (``active_count_allocation="independent"``).
    The notebook's config seed is not set: every task passes its own seed.
    """
    from pymc_generator import make_scm_prior
    from pymc_generator.sampler import SATURATION_FAMILY_KEYS, SCMPrior

    shapes = SCMPrior().saturation_prior_ranges
    shapes["michaelis_menten"]["kappa_mult"] = _SHOWCASE_MM_KAPPA_MULT_RANGE
    edges = dict.fromkeys(("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"), (0, 0))
    edges.update(
        cy=(_SHOWCASE_KNOBS["n_treatments"],) * 2, zy=(_SHOWCASE_KNOBS["n_covariates"],) * 2
    )
    return make_scm_prior(
        **_SHOWCASE_KNOBS,
        n_cells=worlds_per_task,
        edge_budget=edges,
        trajectories="composable",
        active_count_allocation="independent",
        carryover_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        saturation_family_probs={
            name: float(name == "michaelis_menten") for name in SATURATION_FAMILY_KEYS
        },
        saturation_prior_ranges=shapes,
        mm_scale_prior="log_uniform",
        record_walk_parameters=True,
    )


def _small(worlds_per_task: int) -> SCMPrior:
    """A small layout (about a second per two-world task) for tests and smoke runs."""
    from pymc_generator import make_scm_prior

    return make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        n_cells=worlds_per_task,
        draws_per_cell=1,
        n_treatments_active_range=(1, 3),
        n_covariates_active_range=(1, 2),
        treatment_cv_floor=0.5,
    )


# Preset name -> configuration for a task of ``worlds_per_task`` worlds (one per cell, so
# a successful task accepts exactly its target). Read-only: runs are named by preset.
PRESETS: Mapping[str, Callable[[int], SCMPrior]] = MappingProxyType(
    {"showcase-variety": _showcase_variety, "small": _small}
)


# -- generator identity ----------------------------------------------------------------


def _code_sha256() -> str:
    """Digest of the source that determines generated bytes and published records.

    Covers every ``*.py`` of the imported ``pymc_generator`` package plus this module and
    the contract, by relative POSIX path, so the digest is the same on a laptop and in a
    container that copied the same sources. Each file is framed by its path and length,
    which keeps the concatenation unambiguous.
    """
    import pymc_generator

    package = Path(pymc_generator.__file__).resolve().parent
    files = {
        path.relative_to(package.parent).as_posix(): path
        for path in package.rglob("*.py")
        if "__pycache__" not in path.relative_to(package).parts
    }
    here = Path(__file__).resolve().parent
    for name in ("generation.py", "publication.py"):
        files[f"{here.name}/{name}"] = here / name
    digest = hashlib.sha256()
    for name in sorted(files):
        data = files[name].read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    """Interpreter, machine and numerical package versions (never the project itself)."""
    environment = {"python": platform.python_version(), "machine": platform.machine()}
    for name in _ENVIRONMENT_PACKAGES:
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = "unknown"
    return environment


def _code_revision() -> str | None:
    """Git provenance: the deployment's recorded revision, else the local checkout's."""
    configured = os.environ.get(REVISION_ENV, "").strip()
    if configured:
        return configured
    return code_revision(Path(__file__).resolve().parent.parent)


def generator_info(preset: str, cfg: SCMPrior) -> GeneratorInfo:
    """The effective generator of a run: what must match for a resume to be allowed."""
    import pymc_generator
    from pymc_generator.slots import CORPUS_SCHEMA_VERSION

    return GeneratorInfo(
        preset=preset,
        config=dataclasses.asdict(cfg),
        pymc_generator_version=pymc_generator.__version__,
        corpus_schema_version=CORPUS_SCHEMA_VERSION,
        code_sha256=_code_sha256(),
        code_revision=_code_revision(),
        environment=_environment(),
    )


# -- run manifest and attempt record ---------------------------------------------------


def _stored_manifest(store: DirectoryStore, run_id: str) -> RunManifest | None:
    """The committed manifest of ``run_id``; ``None`` if absent; ``RecordError`` if unusable."""
    path = run_manifest_path(run_id)
    size = store.size(path)
    if size is None:
        return None
    if size > MAX_RECORD_BYTES:
        raise RecordError(f"run.json exceeds {MAX_RECORD_BYTES} bytes")
    return decode_run_manifest(store.read_bytes(path), run_id=run_id)


def _open_run(
    store: DirectoryStore, request: RunRequest, generator: GeneratorInfo, created_at: datetime
) -> RunManifest:
    """Create the run manifest, or check that the stored one names this generator."""
    run_id = request.run_id
    data = RunManifest(
        run_id=run_id, created_at=created_at, request=request, generator=generator
    ).encode()
    # Decoding also normalises the current generator exactly like the stored one, so the
    # comparison below cannot report a difference that only the encoding introduced.
    current = decode_run_manifest(data, run_id=run_id)
    try:
        stored = _stored_manifest(store, run_id)
    except RecordError as error:
        raise RunConflictError(f"run {run_id} has an unusable run.json: {error}") from error
    if stored is None:
        store.write_new(run_manifest_path(run_id), data)
        store.commit()
        return current
    differing = [
        name
        for name in _CONFLICT_FIELDS
        if getattr(stored.generator, name) != getattr(current.generator, name)
    ]
    if differing:
        raise RunConflictError(
            f"run {run_id} was created by a different generator ({', '.join(differing)} "
            "differ); resuming would mix the outputs of two generators"
        )
    return stored


@dataclass
class _Attempt:
    """The provisional attempt record this writer owns and rewrites as it progresses."""

    store: DirectoryStore
    manifest: RunManifest
    attempt_id: str
    worker: WorkerInfo
    started_at: datetime
    deadline_at: datetime | None
    published: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    current_task_index: int | None = None

    def __post_init__(self) -> None:
        # Records reject empty names: a worker that found no GPU name reports none.
        self.worker = dataclasses.replace(self.worker, gpu_name=self.worker.gpu_name or None)

    def save(
        self, state: AttemptState, updated_at: datetime, error: ErrorInfo | None = None
    ) -> None:
        data = AttemptRecord(
            run_id=self.manifest.run_id,
            attempt_id=self.attempt_id,
            state=state,
            started_at=self.started_at,
            updated_at=updated_at,
            deadline_at=self.deadline_at,
            current_task_index=self.current_task_index,
            published_task_ids=tuple(self.published),
            reused_task_ids=tuple(self.reused),
            error=error,
            worker=self.worker,
        ).encode()
        decode_attempt_record(data, manifest=self.manifest, attempt_id=self.attempt_id)
        self.store.write_replace(attempt_record_path(self.manifest.run_id, self.attempt_id), data)
        self.store.commit()


# -- tasks -----------------------------------------------------------------------------


def _payload_verifies(store: DirectoryStore, payload: PayloadRef) -> bool:
    return store.size(payload.path) == payload.size and (
        sha256_hex(store.read_bytes(payload.path)) == payload.sha256
    )


def _already_published(store: DirectoryStore, manifest: RunManifest, task_id: str) -> bool:
    """Whether a valid record of ``task_id`` exists that a reader can count as published.

    A record whose payload is missing or damaged does not count: the task is generated
    again, and the reader deduplicates the new record (same digest) or reports the
    conflict (different digest).
    """
    for name in store.list_files(task_dir(manifest.run_id, task_id)):
        attempt_id = name.removesuffix(".json")
        if attempt_id == name or not is_attempt_id(attempt_id):
            continue
        path = task_record_path(manifest.run_id, task_id, attempt_id)
        size = store.size(path)
        if size is None or size > MAX_RECORD_BYTES:
            continue
        try:
            record = decode_task_record(
                store.read_bytes(path), manifest=manifest, task_id=task_id, attempt_id=attempt_id
            )
        except RecordError:
            continue
        if record.payload is None or _payload_verifies(store, record.payload):
            return True
    return False


def _payload_bytes(corpus: dict[str, Any], accepted: int) -> bytes:
    """The saved ``.npz`` of ``corpus``, checked by reading it back like a reader will."""
    from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus

    with tempfile.TemporaryDirectory(prefix="live-dashboard-payload-") as directory:
        path = Path(directory) / "payload.npz"
        save_corpus(corpus, path)
        loaded = load_corpus(path)
        errors = DataGenerator.validate_corpus(loaded)
        if errors:
            raise ValueError("saved payload fails validation: " + "; ".join(errors))
        rows = len(loaded["cell_id"])
        if rows != accepted:
            raise ValueError(f"saved payload has {rows} worlds, the generator accepted {accepted}")
        return path.read_bytes()


def _generate_task(
    store: DirectoryStore,
    manifest: RunManifest,
    generator: DataGenerator,
    index: int,
    attempt: _Attempt,
    clock: Callable[[], datetime],
) -> TaskRecord:
    """Generate task ``index``, publish its payload (if any), then its record."""
    request = manifest.request
    run_id = manifest.run_id
    task_id = task_id_for(index)
    started_at = clock()
    t0 = time.perf_counter()
    try:
        corpus = generator.generate(n_tasks=request.worlds_per_task, seed=request.task_seed(index))
    except RuntimeError as error:
        if type(error) is not RuntimeError or _EXHAUSTED.search(str(error)) is None:
            raise
        accepted = 0
        payload = None
        accounting = Accounting(None, None, None, None, ErrorInfo.from_exception(error))
        generation_seconds = time.perf_counter() - t0
    else:
        diagnostics = corpus["diagnostics"]
        accepted = int(diagnostics["n_tasks"])
        data = _payload_bytes(corpus, accepted)
        path = payload_path(run_id, task_id, attempt.attempt_id)
        store.write_new(path, data)
        store.commit()
        payload = PayloadRef(
            path=path,
            sha256=sha256_hex(data),
            size=len(data),
            corpus_schema_version=manifest.generator.corpus_schema_version,
        )
        evaluated = int(diagnostics["n_draws_evaluated"])
        accounting = Accounting(
            candidates_evaluated=evaluated,
            # The library reports a rate; the count is exact for any realistic total.
            rejected_candidates=round(float(diagnostics["rejection_rate"]) * evaluated),
            draw_failures=int(diagnostics["n_draw_failures"]),
            surplus_discarded=None,  # drawn-but-unevaluated candidates are not counted
            generation_failure=None,
        )
        generation_seconds = float(diagnostics["timing"]["elapsed_s"])
    record = TaskRecord(
        run_id=run_id,
        task_id=task_id,
        task_index=index,
        seed=request.task_seed(index),
        target_worlds=request.worlds_per_task,
        accepted_worlds=accepted,
        world_ids=tuple(world_id_for(run_id, task_id, row) for row in range(accepted)),
        payload=payload,
        accounting=accounting,
        started_at=started_at,
        finished_at=max(clock(), started_at),
        generation_seconds=generation_seconds,
        attempt_id=attempt.attempt_id,
        worker=attempt.worker,
    )
    data = record.encode()
    decode_task_record(data, manifest=manifest, task_id=task_id, attempt_id=attempt.attempt_id)
    store.write_new(task_record_path(run_id, task_id, attempt.attempt_id), data)
    store.commit()
    return record


def _publish_tasks(
    store: DirectoryStore,
    manifest: RunManifest,
    cfg: SCMPrior,
    attempt: _Attempt,
    clock: Callable[[], datetime],
    deadline: datetime | None,
) -> str:
    """Reuse or generate every task in order; ``"stopped"`` when the deadline is near."""
    from pymc_generator.data_generator import DataGenerator

    generator = DataGenerator(cfg)
    longest = 0.0
    for index in range(manifest.n_tasks):
        task_id = task_id_for(index)
        if _already_published(store, manifest, task_id):
            attempt.reused.append(task_id)
            continue
        if deadline is not None:
            remaining = (deadline - clock()).total_seconds()
            if remaining < max(_MIN_DEADLINE_MARGIN_S, 2 * longest):
                return "stopped"
        attempt.current_task_index = index
        attempt.save("running", clock())
        record = _generate_task(store, manifest, generator, index, attempt, clock)
        attempt.published.append(task_id)
        # Measured on the deadline's clock, so both sides of the stop rule agree.
        longest = max(longest, (record.finished_at - record.started_at).total_seconds())
    return "finished"


# -- entry points ----------------------------------------------------------------------


def _check_request(
    request: RunRequest, attempt_id: str | None, presets: Mapping[str, Callable[[int], SCMPrior]]
) -> None:
    """Validate what the request alone decides, before the store is touched."""
    try:
        check_demo_bounds(request)
    except ValueError as error:
        raise RequestError(str(error)) from error
    if attempt_id is not None and not is_attempt_id(attempt_id):
        raise RequestError(
            f"attempt_id must look like 20260101T000000Z-0a1b2c3d, got {attempt_id!r}"
        )
    if request.preset not in presets:
        raise RequestError(
            f"unknown preset {request.preset!r}; known: {', '.join(sorted(presets))}"
        )


def run_generation(
    store: DirectoryStore,
    request: RunRequest,
    *,
    worker: WorkerInfo,
    clock: Callable[[], datetime] = utc_now,
    deadline: datetime | None = None,
    attempt_id: str | None = None,
    presets: Mapping[str, Callable[[int], SCMPrior]] = PRESETS,
) -> dict[str, Any]:
    """Publish every task of ``request`` that is not yet published, as one attempt.

    An invalid request (demo bounds, unknown preset, malformed ``attempt_id``) raises
    ``RequestError`` and a stored run of another generator raises ``RunConflictError``,
    both before anything is written. Any other failure marks the attempt ``failed``
    (best effort) and propagates. ``deadline`` is the time after which the attempt
    cannot still be running (the platform's kill time), recorded as the attempt's
    ``deadline_at``: the attempt stops cleanly, instead of starting a task, once less
    than 90 s or twice its longest task so far is left. Returns ``{"status": "finished"
    | "stopped", "run_id", "attempt_id", "published": [task ids], "reused": [task ids]}``.
    """
    _check_request(request, attempt_id, presets)
    store.refresh()
    cfg = presets[request.preset](request.worlds_per_task)
    started_at = clock()
    manifest = _open_run(store, request, generator_info(request.preset, cfg), started_at)
    attempt = _Attempt(
        store=store,
        manifest=manifest,
        attempt_id=attempt_id or new_attempt_id(started_at),
        worker=worker,
        started_at=started_at,
        deadline_at=deadline,
    )
    try:
        attempt.save("running", started_at)
        status = _publish_tasks(store, manifest, cfg, attempt, clock, deadline)
        attempt.current_task_index = None
        attempt.save("finished", clock())
    except BaseException as error:
        # Best effort: the original failure matters more than a failure to record it.
        with contextlib.suppress(Exception):
            attempt.save("failed", clock(), ErrorInfo.from_exception(error))
        raise
    return {
        "status": status,
        "run_id": manifest.run_id,
        "attempt_id": attempt.attempt_id,
        "published": list(attempt.published),
        "reused": list(attempt.reused),
    }


def run_generation_or_refuse(
    store: DirectoryStore,
    request: RunRequest,
    *,
    worker: WorkerInfo,
    expected_run_id: str | None = None,
    clock: Callable[[], datetime] = utc_now,
    deadline: datetime | None = None,
    attempt_id: str | None = None,
    presets: Mapping[str, Callable[[int], SCMPrior]] = PRESETS,
) -> dict[str, Any]:
    """``run_generation``, but a request that must not run is refused, not raised.

    A platform retry cannot fix an invalid request (``RequestError``) or a run that must
    not be resumed (``RunConflictError``), so both are returned as ``{"status":
    "refused", "run_id", "attempt_id", "reason"}``. The refusal is recorded as a
    ``failed`` attempt of the run when the run has a usable manifest (under a fresh
    attempt id if the injected one is malformed; ``attempt_id`` is ``None`` without a
    manifest); the manifest itself is never written. ``expected_run_id`` is the run the
    launcher announced: a request naming another run is refused. Every other failure
    propagates, so the platform can retry it.
    """
    try:
        if expected_run_id is not None and expected_run_id != request.run_id:
            raise RunConflictError(
                f"the launcher expected run {expected_run_id}, the request names run "
                f"{request.run_id}"
            )
        return run_generation(
            store,
            request,
            worker=worker,
            clock=clock,
            deadline=deadline,
            attempt_id=attempt_id,
            presets=presets,
        )
    except (RequestError, RunConflictError) as error:
        return _refuse(store, request, error, worker, clock, deadline, attempt_id)


def _refuse(
    store: DirectoryStore,
    request: RunRequest,
    error: RequestError | RunConflictError,
    worker: WorkerInfo,
    clock: Callable[[], datetime],
    deadline: datetime | None,
    attempt_id: str | None,
) -> dict[str, Any]:
    """Record a refusal as a failed attempt of the request's run, when it has a manifest."""
    store.refresh()
    try:
        manifest = _stored_manifest(store, request.run_id)
    except RecordError:
        manifest = None
    recorded_id = None
    if manifest is not None:
        now = clock()
        attempt = _Attempt(
            store=store,
            manifest=manifest,
            attempt_id=attempt_id if is_attempt_id(attempt_id) else new_attempt_id(now),
            worker=worker,
            started_at=now,
            deadline_at=deadline,
        )
        attempt.save("failed", now, ErrorInfo.from_exception(error))
        recorded_id = attempt.attempt_id
    return {
        "status": "refused",
        "run_id": request.run_id,
        "attempt_id": recorded_id,
        "reason": clean_message(str(error)),
    }
