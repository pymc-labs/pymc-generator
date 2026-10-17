"""Opt-in reusable-family streaming, independent of the legacy demo generator.

One orchestrator per run; disjoint partition workers, one compiled model each; one
storage-owning writer per worker. Do not overlap attempts. Published batches are
immutable and only fully validated batches are skipped on resume. Raw data memory is
batch-bounded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import publication as pub
from . import structural
from .save_queue import stream_to_writer
from .storage import DirectoryStore

PRESET = "stream-composable-v1"
MODE = "specialized-internal-lazy-stream-v1"
_SATURATIONS = ("linear", "hill", "logistic", "michaelis_menten", "tanh", "root")
_ADSTOCKS = ("none", "geometric", "weibull")
# How a partition's model executes (``StudyAdapter`` modes), measured on Modal: C thunks
# compile for minutes per cold container, then draw a world in about 10 ms; Python thunks
# in the same C loop compile in tens of seconds and draw a world in about 0.4 s, the faster
# choice for a few dozen worlds per model.
COMPILE_MODES = {"cvm": "cvm_compile", "python": "python_compile"}


@dataclass(frozen=True)
class StreamPlan:
    """A run of ``worlds_per_split`` worlds for each (saturation, adstock) family.

    Without a ``design`` every family is a partition: one worker and one compiled model of
    the composable prior. With a structural-split ``design`` (``structural.DESIGNS``),
    ``families`` are the design's splits in order and every member (split x regime) is a
    partition, with its ``generation.DESIGN_PRESETS`` configuration and its split's seed:
    task ``i`` belongs to member ``design.member_of(i)``, as in the legacy design runs.
    ``compile_mode`` (``COMPILE_MODES``) changes execution speed, not the prior.
    ``baseline_scale`` multiplies a design's baseline (``scale_baseline``).
    """

    label: str
    worlds_per_split: int = 100
    batch_size: int = 10
    seed: int = 631227
    families: tuple[tuple[str, str], ...] = (("hill", "geometric"), ("logistic", "geometric"))
    design: str | None = None
    compile_mode: str = "cvm"
    baseline_scale: float = 1.0

    def __post_init__(self):
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and distinct")
        for saturation, adstock in self.families:
            if saturation not in _SATURATIONS or adstock not in _ADSTOCKS:
                raise ValueError("unknown saturation/adstock family")
        if self.batch_size < 2 or self.batch_size > 100:
            raise ValueError("publication batch size must be 2..100")
        if self.worlds_per_split < self.batch_size or self.worlds_per_split % self.batch_size:
            raise ValueError("worlds per split must be a positive multiple of batch size")
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile mode must be one of {sorted(COMPILE_MODES)}")
        if not (math.isfinite(self.baseline_scale) and self.baseline_scale > 0):
            raise ValueError("the baseline scale must be finite and positive")
        if self.baseline_scale != 1.0 and self.design is None:
            raise ValueError("a baseline scale applies to a design's members")
        if self.design is not None:
            design = structural.DESIGNS.get(self.design)
            if design is None:
                raise ValueError(f"unknown design {self.design!r}")
            if self.families != tuple((s.saturation, s.adstock) for s in design.splits):
                raise ValueError("a design's families are its splits, in design order")
            if (self.worlds_per_split // self.batch_size) % len(design.regimes):
                raise ValueError("a split's tasks must divide evenly over the design's regimes")
        _ = self.request  # validate request limits too

    @classmethod
    def for_design(cls, label: str, design: str, **kwargs) -> StreamPlan:
        """A plan of ``design``: its families are the design's splits."""
        if design not in structural.DESIGNS:
            raise ValueError(f"unknown design {design!r}")
        splits = structural.DESIGNS[design].splits
        families = tuple((split.saturation, split.adstock) for split in splits)
        return cls(label, families=families, design=design, **kwargs)

    @property
    def split_design(self) -> structural.SplitDesign | None:
        return None if self.design is None else structural.DESIGNS[self.design]

    @property
    def preset(self) -> str:
        """The run's preset. A design run is a plain run to the reader: its seeds and
        redraws do not follow the legacy design contract."""
        return PRESET if self.design is None else f"stream-{self.design}"

    @property
    def partitions(self) -> tuple[str, ...]:
        """The member name of every partition."""
        design = self.split_design
        if design is not None:
            return design.cycle
        return tuple(
            f"{saturation}-{adstock}".replace("_", "-") for saturation, adstock in self.families
        )

    def family(self, partition: int) -> tuple[str, str]:
        """The (saturation, adstock) pair of ``partition``'s model."""
        return self.families[partition % len(self.families)]

    def root_seed(self, partition: int) -> int:
        """The seed of ``partition``'s candidate streams: a design shares its split's seed
        over the split's regimes (candidate IDs differ)."""
        design = self.split_design
        if design is None:
            return self.seed + partition
        return design.splits[partition % len(design.splits)].seed(self.seed)

    @property
    def request(self) -> pub.RunRequest:
        return pub.RunRequest(
            self.label,
            self.preset,
            self.worlds_per_split * len(self.families),
            self.batch_size,
            self.seed,
        )

    def task_indices(self, partition: int):
        n_partitions = len(self.partitions)
        if not 0 <= partition < n_partitions:
            raise ValueError("invalid partition")
        return range(partition, self.request.n_tasks, n_partitions)


def scale_baseline(cfg, scale: float):
    """``cfg`` with its baseline multiplied by ``scale`` in the prior.

    The baseline is ``max(0, level + spread * walk)``, level ~ U(``rw_baseline_mean_range``)
    and spread ~ HalfNormal(``rw_baseline_std_sigma``) on the absolute outcome scale, so
    scaling both scales it exactly and keeps its shape; channels, controls and outcome
    noise are drawn as before. Published worlds also pass the realism filter, which reads
    the outcome (never below 0, no extreme spike), so a lower baseline rejects slightly
    more candidates.
    """
    floored = cfg.baseline_floor is not None
    if (
        cfg.outcome_std_mode != "absolute"
        or (floored and cfg.baseline_floor != 0.0)
        or (floored and cfg.baseline_floor_scope != "intercept")
    ):
        raise ValueError(
            "an exact baseline scale needs absolute walks and at most a floor of 0 on the intercept"
        )
    low, high = cfg.rw_baseline_mean_range
    return dataclasses.replace(
        cfg,
        rw_baseline_mean_range=(low * scale, high * scale),
        rw_baseline_std_sigma=cfg.rw_baseline_std_sigma_effective * scale,
    )


def member_config(plan: StreamPlan, partition: int):
    """The configuration ``partition``'s worker compiles; the manifest records the same."""
    if plan.design is None:
        from ._specialized.inputs import config

        return config(plan.batch_size, plan.root_seed(partition))
    from .generation import DESIGN_PRESETS

    cfg = DESIGN_PRESETS[plan.partitions[partition]](plan.batch_size)
    if cfg.active_count_allocation != "independent":
        raise ValueError("streaming draws every world's active counts independently")
    # Walk truth is a report this backend cannot record; the priors are unchanged.
    cfg = dataclasses.replace(cfg, seed=plan.root_seed(partition), record_walk_parameters=False)
    if plan.baseline_scale != 1.0:
        cfg = scale_baseline(cfg, plan.baseline_scale)
    cfg.validate()
    return cfg


def generator_info(plan: StreamPlan) -> pub.GeneratorInfo:
    import importlib.metadata

    import pymc_generator

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "pymc_generator").glob("*.py")) + sorted(
        (root / "live_dashboard").rglob("*.py")
    )
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    effective = {
        "mode": MODE,
        "plan": dataclasses.asdict(plan),
        "candidate_limit_per_batch": 2 * plan.batch_size,
        "source_hashes": hashes,
    }
    design = plan.split_design
    if design is None:
        effective.update(
            configuration=dataclasses.asdict(member_config(plan, 0)),
            seed_contract="family root = base seed + partition; candidate id = task index * 2 * batch size + offset",
        )
    else:
        effective.update(
            cycle=list(design.cycle),
            presets={
                member: dataclasses.asdict(member_config(plan, partition))
                for partition, member in enumerate(plan.partitions)
            },
            structural_splits=[split.to_dict(plan.seed) for split in design.splits],
            positive_controls="a world whose control value or control contribution goes below zero is rejected",
            seed_contract=(
                "family root = the split's seed: the first 31 bits of "
                "sha256('<base seed>:<saturation>:<adstock>'), shared by the split's regimes; "
                "candidate id = task index * 2 * batch size + offset"
            ),
        )
    return pub.GeneratorInfo(
        plan.preset,
        effective,
        pymc_generator.__version__,
        5,
        hashlib.sha256(pub.canonical_json(hashes)).hexdigest(),
        None,
        {
            **{k: importlib.metadata.version(k) for k in ("numpy", "pymc", "pytensor")},
            "python": platform.python_version(),
        },
    )


def prepare_run(
    store: DirectoryStore, plan: StreamPlan, *, resume: bool = False, previous_stopped: bool = False
) -> pub.RunManifest:
    """Record an immutable plan; resume never silently changes code, priors or IDs."""
    store.refresh()
    path = pub.run_manifest_path(plan.request.run_id)
    info = generator_info(plan)
    if store.exists(path):
        if not resume or not previous_stopped:
            raise ValueError(
                "existing run requires resume and confirmation all previous workers stopped"
            )
        manifest = pub.decode_run_manifest(store.read_bytes(path), run_id=plan.request.run_id)
        if manifest.generator.to_dict() != info.to_dict():
            raise ValueError(
                "resume configuration/source/environment mismatch; use a new run label"
            )
        return manifest
    if resume:
        raise ValueError("cannot resume a missing run")
    manifest = pub.RunManifest(plan.request.run_id, pub.utc_now(), plan.request, info)
    data = manifest.encode()
    # Refuse a manifest the reader would refuse before the run directory exists.
    pub.decode_run_manifest(data, run_id=manifest.run_id)
    store.write_new(path, data)
    store.commit()
    return manifest


@dataclass
class Batch:
    """One task's worlds and its candidate accounting."""

    corpus: dict | None
    candidate_ids: list[int]  # logical candidate ID of every published world, in row order
    evaluated: int
    rejected: int  # by the generator's filters
    negative_controls: int = 0  # accepted by the generator, then dropped (designs only)
    draw_failures: int = 0
    generation_failure: pub.ErrorInfo | None = None  # why a task has no world, if drawing failed


def drop_negative_controls(corpus: dict, ids: list[int]) -> tuple[dict | None, list[int], int]:
    """``corpus`` without its worlds whose control values or contributions go below zero
    (``generation.negative_rows``), the remaining candidate IDs, and how many were dropped.

    ``ids`` are the corpus rows' candidate IDs in row order. ``None`` if no world remains.
    Only per-world arrays are filtered: nested blocks (``diagnostics``, identifiability)
    keep the input's rows until ``_finalize_corpus`` rebuilds them, as ``generate`` does.
    """
    import numpy as np

    from .generation import negative_rows

    negative = negative_rows(corpus)
    if len(negative) != len(ids):
        raise ValueError("candidate IDs do not match the corpus rows")
    dropped = int(negative.sum())
    if not dropped:
        return corpus, list(ids), 0
    keep = ~negative
    kept_ids = [ident for ident, kept in zip(ids, keep, strict=True) if kept]
    if not kept_ids:
        return None, [], dropped
    kept = {
        key: value[keep] if isinstance(value, np.ndarray) else value
        for key, value in corpus.items()
    }
    return kept, kept_ids, dropped


class FamilyGenerator:
    """One live compiled model of a partition. Audit caches are discarded between chunks."""

    def __init__(self, plan: StreamPlan, partition: int):
        from ._specialized.template import StudyAdapter

        self.plan, self.partition = plan, partition
        self.cfg = member_config(plan, partition)
        self.adapter = StudyAdapter(
            "specialized_internal_lazy",
            self.cfg,
            lambda *a, **k: None,
            mode_name=COMPILE_MODES[plan.compile_mode],
            seed_contract="logical-id",
        )

    def generate(self, index: int) -> Batch:
        import numpy as np

        from pymc_generator.data_generator import DataGenerator
        from pymc_generator.sampler import (
            CARRYOVER_FAMILY_KEYS,
            SATURATION_FAMILY_KEYS,
            _finalize_corpus,
        )

        from ._specialized.inputs import make_cell
        from ._specialized.packing import generate_fixed

        n = self.plan.batch_size
        s, a = self.plan.family(self.partition)
        s, a = SATURATION_FAMILY_KEYS.index(s), CARRYOVER_FAMILY_KEYS.index(a)
        chunks, selected = [], []
        evaluated = rejected = negative = draw_failures = 0
        for allocation in range(2):
            if len(selected) == n:
                break
            ids = list(range(index * 2 * n + allocation * n, index * 2 * n + (allocation + 1) * n))
            cells = [make_cell(self.cfg, ident, True) for ident in ids]
            for cell in cells:
                for key, value in (("sat_family", s), ("carryover_family", a)):
                    cell.structure[key][:] = value
                    cell.template_inputs[key][:] = value
            # Research diagnostics must not grow with the total number of worlds.
            self.adapter.all_primary_hashes.clear()
            self.adapter.realised_flags.clear()
            self.adapter.executions.clear()
            self.adapter.set_chunk(cells, ids)
            batch: dict[str, Any] | None = generate_fixed(
                dataclasses.replace(self.cfg, n_cells=n), self.adapter
            )
            evaluated += self.adapter.evaluated
            rejected += self.adapter.rejected
            draw_failures += self.adapter.draw_failures
            accepted = [ids[cell] for cell, _ in self.adapter.accepted_ids]
            if batch is not None and self.plan.design is not None:
                # Design worlds keep every control non-negative, like the legacy design runs.
                batch, accepted, dropped = drop_negative_controls(batch, accepted)
                negative += dropped
            if batch is not None:
                take = min(n - len(selected), len(accepted))
                # Copy truncated views: queued batches must not pin surplus backing arrays.
                for key, value in list(batch.items()):
                    if isinstance(value, np.ndarray):
                        batch[key] = value[:take].copy()
                chunks.append(batch)
                selected.extend(accepted[:take])
        corpus = None
        if chunks:
            corpus = {
                k: np.concatenate([c[k] for c in chunks]) if isinstance(v, np.ndarray) else v
                for k, v in chunks[0].items()
            }
            corpus["cell_id"][:] = np.arange(len(selected))
            corpus["diagnostics"]["n_draws_evaluated"] = evaluated
            corpus["diagnostics"]["rejection_rate"] = rejected / max(1, evaluated)
            corpus["diagnostics"]["n_draw_failures"] = draw_failures
            corpus = _finalize_corpus(corpus, self.cfg)
            errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
            if errors:
                raise ValueError(errors)
        failure = None
        if draw_failures:
            last = self.adapter.last_draw_error
            if not evaluated:
                # Not one candidate drew: a repeatable failure, which must not publish empty
                # tasks (the library also raises once its resamples are spent).
                raise RuntimeError(
                    f"all {draw_failures} candidates of task {index} failed to draw; last: {last}"
                )
            if not selected:
                # The dashboard shows a generation failure for tasks without worlds only.
                failure = pub.ErrorInfo(
                    "DrawFailure",
                    f"no world: {evaluated} candidates evaluated ({rejected} rejected by the "
                    f"generator, {negative} with a negative control), {draw_failures} failed "
                    f"to draw; last: {last}",
                )
        if len(self.adapter.compile_records) != 1:
            raise RuntimeError("expected one compilation per partition worker")
        return Batch(corpus, selected, evaluated, rejected, negative, draw_failures, failure)


def run_partition(
    plan: StreamPlan,
    partition: int,
    manifest: pub.RunManifest,
    attempt_id: str,
    publication,
    *,
    queue_size: int = 2,
    stop: Callable[[], bool] = lambda: False,
    factory=FamilyGenerator,
    worker: pub.WorkerInfo | None = None,
) -> dict:
    """Validate resume state before starting threads; writer owns all subsequent IO.

    stop is checked between batches (e.g. deadline margin or signal). Graceful stop
    drains handed-off batches; a hard container timeout can lose only unpublished work.
    """
    if manifest.request != plan.request:
        raise ValueError("manifest does not match plan")
    indices = plan.task_indices(partition)
    member = plan.partitions[partition]
    recorded = manifest.generator.config
    member_cfg = (
        recorded["presets"][member]
        if plan.design is not None
        else {**recorded["configuration"], "seed": plan.root_seed(partition)}
    )
    # Records are small; read_output validates one payload at a time, never all raw worlds.
    # Only this partition's tasks: identities are task-prefixed and owners disjoint, and the
    # coordinator validates the whole run when it reports.
    completed = {
        item["record"]["task_id"]
        for item in publication.validated_task_outputs(
            manifest.run_id, [pub.task_id_for(index) for index in indices]
        )
    }
    worker = worker or pub.WorkerInfo(platform.platform(), 1, None, None)
    engine = None
    reused = published = shortfall = 0
    generation_seconds = publication_seconds = 0.0
    stopped = False
    started = time.monotonic()

    def batches() -> Iterator[tuple[dict, dict | None]]:
        nonlocal engine, reused, stopped, generation_seconds
        for index in indices:
            task_id = pub.task_id_for(index)
            if task_id in completed:
                reused += 1
                continue
            if stop():
                stopped = True
                break
            if engine is None:
                engine = factory(plan, partition)
            begin = pub.utc_now()
            t0 = time.monotonic()
            batch = engine.generate(index)
            generation_seconds += time.monotonic() - t0
            meta = pub.completion_metadata(
                manifest,
                task_index=index,
                member=member,
                member_config=member_cfg,
                attempt_id=attempt_id,
                worker=worker,
                candidates=batch.evaluated,
                rejected=batch.rejected,
                published=len(batch.candidate_ids),
                started_at=begin,
                finished_at=pub.utc_now(),
                generation_seconds=time.monotonic() - t0,
                draw_failures=batch.draw_failures,
                generation_failure=batch.generation_failure,
                negative_controls=batch.negative_controls,
            )
            meta["provenance"]["generation_mode"] = MODE
            # Keep the dashboard's logical task-seed identity, but do not claim
            # the legacy generator's redraw/generation_seed algorithm was used.
            context = meta["provenance"]["seed_context"]
            context.pop("redraws", None)
            context.pop("generation_seed", None)
            context.update(
                rule="family-root and logical-candidate SeedSequence streams (100/200/300/400/4401)",
                family_root_seed=plan.root_seed(partition),
                logical_candidate_ids=batch.candidate_ids,
                internal_contract=recorded.get("seed_contract"),
            )
            design = plan.split_design
            if design is not None:
                # The split (with its seed) and the regime, as legacy design tasks record them.
                context.update(design.seed_context(plan.seed, index))
            meta["provenance"]["effective_configuration"]["family"] = list(plan.family(partition))
            corpus = batch.corpus
            del batch
            yield meta, corpus
            # Do not retain a previously yielded corpus in a suspended generator frame.
            del meta, corpus

    def write(item):
        nonlocal published, shortfall, publication_seconds
        t0 = time.monotonic()
        metadata, corpus = item
        reference = publication.publish(metadata, corpus)
        record, loaded = publication.read_output(reference)
        pub.task_record_from_completion(manifest, record)
        published += record["counts"]["published"]
        shortfall += record["counts"]["shortfall"]
        publication_seconds += time.monotonic() - t0
        del loaded

    batches_written = stream_to_writer(batches(), write, capacity=queue_size)
    return {
        "partition": partition,
        "batches_written": batches_written,
        "reused_batches": reused,
        "published_worlds": published,
        "shortfall": shortfall,
        "stopped": stopped,
        "compilations": 0 if engine is None else len(engine.adapter.compile_records),
        "wall_seconds": time.monotonic() - started,
        "generation_seconds": generation_seconds,
        "publication_seconds": publication_seconds,
        "stage_times": {} if engine is None else dict(getattr(engine.adapter, "times", {})),
    }
