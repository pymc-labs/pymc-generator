"""Opt-in reusable-family streaming, independent of the legacy demo generator.

One orchestrator per run; disjoint family workers; one storage-owning writer per
worker. Do not overlap attempts. Published batches are immutable and only fully
validated batches are skipped on resume. Raw data memory is batch-bounded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import publication as pub
from .save_queue import stream_to_writer
from .storage import DirectoryStore

PRESET = "stream-composable-v1"
MODE = "specialized-internal-lazy-stream-v1"


@dataclass(frozen=True)
class StreamPlan:
    label: str
    worlds_per_split: int = 100
    batch_size: int = 10
    seed: int = 631227
    families: tuple[tuple[str, str], ...] = (("hill", "geometric"), ("logistic", "geometric"))

    def __post_init__(self):
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and distinct")
        for saturation, adstock in self.families:
            if saturation not in (
                "linear",
                "hill",
                "logistic",
                "michaelis_menten",
                "tanh",
                "root",
            ) or adstock not in ("none", "geometric", "weibull"):
                raise ValueError("unknown saturation/adstock family")
        if self.batch_size < 2 or self.batch_size > 100:
            raise ValueError("publication batch size must be 2..100")
        if self.worlds_per_split < self.batch_size or self.worlds_per_split % self.batch_size:
            raise ValueError("worlds per split must be a positive multiple of batch size")
        _ = self.request  # validate request limits too

    @property
    def request(self) -> pub.RunRequest:
        return pub.RunRequest(
            self.label,
            PRESET,
            self.worlds_per_split * len(self.families),
            self.batch_size,
            self.seed,
        )

    def task_indices(self, partition: int):
        if not 0 <= partition < len(self.families):
            raise ValueError("invalid family partition")
        return range(partition, self.request.n_tasks, len(self.families))


def generator_info(plan: StreamPlan) -> pub.GeneratorInfo:
    import importlib.metadata

    import pymc_generator

    from ._specialized.inputs import config

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "pymc_generator").glob("*.py")) + sorted(
        (root / "live_dashboard").rglob("*.py")
    )
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    effective = {
        "mode": MODE,
        "plan": dataclasses.asdict(plan),
        "configuration": dataclasses.asdict(config(plan.batch_size, plan.seed)),
        "candidate_limit_per_batch": 2 * plan.batch_size,
        "source_hashes": hashes,
        "seed_contract": "family root = base seed + partition; candidate id = task index * 2 * batch size + offset",
    }
    return pub.GeneratorInfo(
        PRESET,
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
    store.write_new(path, manifest.encode())
    store.commit()
    return manifest


class FamilyGenerator:
    """One live compiled family model. Audit caches are discarded between chunks."""

    def __init__(self, plan: StreamPlan, partition: int):
        from ._specialized.inputs import config
        from ._specialized.template import StudyAdapter

        self.plan, self.partition = plan, partition
        self.cfg = config(plan.batch_size, plan.seed + partition)
        self.adapter = StudyAdapter(
            "specialized_internal_lazy",
            self.cfg,
            lambda *a, **k: None,
            mode_name="cvm_compile",
            seed_contract="logical-id",
        )

    def generate(self, index: int):
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
        s, a = self.plan.families[self.partition]
        s, a = SATURATION_FAMILY_KEYS.index(s), CARRYOVER_FAMILY_KEYS.index(a)
        chunks, selected = [], []
        evaluated = rejected = 0
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
            batch = generate_fixed(dataclasses.replace(self.cfg, n_cells=n), self.adapter)
            evaluated += self.adapter.evaluated
            rejected += self.adapter.rejected
            accepted = [ids[cell] for cell, _ in self.adapter.accepted_ids]
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
            corpus = _finalize_corpus(corpus, self.cfg)
            errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
            if errors:
                raise ValueError(errors)
        if len(self.adapter.compile_records) != 1:
            raise RuntimeError("expected one compilation per family worker")
        return corpus, selected, evaluated, rejected


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
    # Records are small; read_output validates one payload at a time, never all raw worlds.
    completed = {
        item["record"]["task_id"] for item in publication.validated_outputs(manifest.run_id)
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
            corpus, ids, evaluated, rejected = engine.generate(index)
            generation_seconds += time.monotonic() - t0
            meta = pub.completion_metadata(
                manifest,
                task_index=index,
                member="-".join(plan.families[partition]).replace("_", "-"),
                member_config={
                    **manifest.generator.config["configuration"],
                    "seed": plan.seed + partition,
                },
                attempt_id=attempt_id,
                worker=worker,
                candidates=evaluated,
                rejected=rejected,
                published=len(ids),
                started_at=begin,
                finished_at=pub.utc_now(),
                generation_seconds=time.monotonic() - t0,
                draw_failures=0,
                generation_failure=None,
            )
            meta["provenance"]["generation_mode"] = MODE
            # Keep the dashboard's logical task-seed identity, but do not claim
            # the legacy generator's redraw/generation_seed algorithm was used.
            context = meta["provenance"]["seed_context"]
            context.pop("redraws", None)
            context.pop("generation_seed", None)
            context.update(
                rule="family-root and logical-candidate SeedSequence streams (100/200/300/400/4401)",
                family_root_seed=plan.seed + partition,
                logical_candidate_ids=ids,
                internal_contract=manifest.generator.config.get("seed_contract"),
            )
            meta["provenance"]["effective_configuration"]["family"] = list(plan.families[partition])
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
