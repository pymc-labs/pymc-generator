"""Bounded integration run against a pinned research worker implementation.

Not a production scheduler. May use an already-mounted Modal Volume; never provisions it.
Run with a single-thread environment and an external runtime/memory envelope.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

RESEARCH_REF = "d61f98f596c5de885947eb5144897ed0909e7c5d"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds-per-pair", type=int, default=10)
    parser.add_argument("--seed", type=int, default=631227)
    parser.add_argument("--partition-index", type=int, choices=range(18))
    parser.add_argument("--research-manifest", type=Path)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--volume-name")
    parser.add_argument("--run-id", default="structural_smoke")
    args = parser.parse_args()
    target = args.worlds_per_pair
    if target not in (10, 100):
        parser.error("Only the approved 10/100-world smoke envelopes are supported")
    if target == 100 and (args.partition_index is None or not args.volume_name):
        parser.error("100-world partitions require the bounded Modal fan-out entry point")
    research = args.research_dir.resolve()
    if args.research_manifest:
        manifest = json.loads(args.research_manifest.read_text())
        if manifest["ref"] != RESEARCH_REF:
            raise RuntimeError("Research manifest revision mismatch")
        actual_files = {path.name for path in (research / "benchmarks").glob("*.py")}
        if actual_files != set(manifest["files"]):
            raise RuntimeError("Research manifest file inventory mismatch")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((research / "benchmarks" / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Research source digest mismatch: {name}")
    else:
        actual = subprocess.check_output(
            ["git", "-C", str(research), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != RESEARCH_REF:
            raise RuntimeError(f"Research source must be pinned to {RESEARCH_REF}")
        subprocess.run(
            ["git", "-C", str(research), "diff", "--exit-code", RESEARCH_REF, "--", "benchmarks"],
            check=True,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["COMPARISON_OUTPUT"] = str(output)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(research / "benchmarks"))

    import numpy as np
    from comparison_preflight import plain
    from study_200k_methods import StudyAdapter
    from study_200k_run import config, make_cell

    from pymc_generator.data_generator import DataGenerator
    from pymc_generator.publication import FilePublicationStore
    from pymc_generator.sampler import (
        CARRYOVER_FAMILY_KEYS,
        SATURATION_FAMILY_KEYS,
        _finalize_corpus,
    )

    started = time.monotonic()

    def save(name, value):
        (output / name).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")

    def event(kind, **values):
        row = plain({"event": kind, "elapsed_s": time.monotonic() - started, **values})
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    # Keep the research branch immutable. Save the exact small adaptation as a
    # run artifact: remove legacy membership/temporal-mask repair BEFORE scales
    # are computed, and mark deferred membership before final diagnostics.
    packing = (research / "benchmarks/study_200k_packing.py").read_text()
    start = packing.index("    # -- cell-level validation split")
    end = packing.index("    # Derived from the FLOAT32", start)
    packing = packing[:start] + '    corpus["is_val"][:] = 2\n\n' + packing[end:]
    anchor = '    corpus["diagnostics"] = diagnostics\n'
    assert packing.count(anchor) == 1
    packing = packing.replace(
        anchor, '    diagnostics["cell_split"] = {"mode": "deferred", "version": 1}\n' + anchor
    )
    packing_path = output / "deferred_research_packing.py"
    packing_path.write_text(packing)
    module = types.ModuleType("deferred_research_packing")
    exec(compile(packing, str(packing_path), "exec"), module.__dict__)

    cfg = config(target, args.seed)
    spec = {
        "kind": "balanced-family-integration-smoke",
        "seed": args.seed,
        "target_per_pair": target,
        "initial_candidates_per_pair": target,
        "one_topup_candidates_per_pair": target,
        "max_total_candidates": 36 * target,
        "max_runtime_seconds": 1800,
        "approach": "specialized_internal_lazy",
        "compiler": "cvm_compile",
        "seed_contract": "logical-id-research",
        "research_ref": RESEARCH_REF,
        "configuration": plain(dataclasses.asdict(cfg)),
        "deferred_cell_split": True,
    }
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    sources = {}
    for path in sorted((ROOT / "pymc_generator").glob("*.py")):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((research / "benchmarks").glob("*.py")):
        sources["research/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources["driver"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources["adapted_packing"] = hashlib.sha256(packing.encode()).hexdigest()
    environment = {
        "python": sys.version,
        "source_hashes": sources,
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pymc", "pytensor")
        },
    }
    save("run-spec.json", spec)
    save("environment.json", environment)
    if args.volume_name:
        import modal

        from pymc_generator.publication import MountedVolumePublicationStore

        if args.store_root is None:
            parser.error("Mounted storage requires --store-root")
        volume = modal.Volume.from_name(args.volume_name)
        store = MountedVolumePublicationStore(args.store_root, volume)
    else:
        store = FilePublicationStore(args.store_root or output / "store")
    summaries = []
    try:
        for s, saturation in enumerate(SATURATION_FAMILY_KEYS):
            for a, carryover in enumerate(CARRYOVER_FAMILY_KEYS):
                partition = s * len(CARRYOVER_FAMILY_KEYS) + a
                if args.partition_index is not None and args.partition_index != partition:
                    continue
                task = f"{saturation}_{carryover}"
                adapter = StudyAdapter(
                    "specialized_internal_lazy",
                    cfg,
                    event,
                    mode_name="cvm_compile",
                    seed_contract="logical-id",
                )
                chunks, selected_ids = [], []
                rejected = evaluated = 0
                reasons = {}
                event("partition_start", task=task)
                for allocation in range(2):
                    if len(selected_ids) >= target:
                        break
                    ids = list(
                        range(
                            partition * 2 * target + allocation * target,
                            partition * 2 * target + allocation * target + target,
                        )
                    )
                    cells = [make_cell(cfg, ident, True) for ident in ids]
                    for cell in cells:
                        for key, family in (("sat_family", s), ("carryover_family", a)):
                            cell.structure[key][:] = family
                            cell.template_inputs[key][:] = family
                    adapter.set_chunk(cells, ids)
                    batch = module.generate_fixed(
                        dataclasses.replace(cfg, n_cells=len(ids)), adapter
                    )
                    evaluated += adapter.evaluated
                    rejected += adapter.rejected
                    for reason, count in adapter.rejection_reasons.items():
                        reasons[reason] = reasons.get(reason, 0) + count
                    accepted = [ids[cell] for cell, _ in adapter.accepted_ids]
                    event(
                        "allocation",
                        task=task,
                        allocation=allocation,
                        candidates=ids,
                        accepted=accepted,
                        rejections=dict(adapter.rejection_reasons),
                    )
                    if batch is None:
                        continue
                    take = min(target - len(selected_ids), len(accepted))
                    for key, value in list(batch.items()):
                        if isinstance(value, np.ndarray):
                            batch[key] = value[:take]
                    batch["cell_id"][:] = accepted[:take]
                    chunks.append(batch)
                    selected_ids.extend(accepted[:take])
                corpus = None
                if chunks:
                    corpus = {
                        key: np.concatenate([chunk[key] for chunk in chunks])
                        if isinstance(value, np.ndarray)
                        else value
                        for key, value in chunks[0].items()
                    }
                    # Corpus cell IDs are contiguous local indices. Global
                    # candidate identity lives in world_ids; each partition has
                    # its own namespace so unrelated local cell zeroes cannot leak.
                    corpus["cell_id"][:] = np.arange(len(selected_ids))
                    corpus["diagnostics"]["n_draws_evaluated"] = evaluated
                    corpus["diagnostics"]["rejection_rate"] = rejected / max(evaluated, 1)
                    corpus = _finalize_corpus(corpus, cfg)
                    errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
                    if errors:
                        raise ValueError(errors)
                    # Saturation IDs are only a corpus field when optional
                    # mechanism reporting is enabled; the specialized compile key
                    # and input structures carry them in this training preset.
                    assert all(np.all(cell.structure["sat_family"] == s) for cell in cells)
                    assert np.all(
                        corpus["carryover_family"][corpus["treatment_active_mask"] == 1] == a
                    )
                count = len(selected_ids)
                metadata = {
                    "run_id": args.run_id,
                    "instruction_id": task,
                    "task_id": task,
                    "attempt_id": "first",
                    "output_id": task,
                    "world_ids": [f"candidate_{ident}" for ident in selected_ids],
                    "counts": {
                        "target": target,
                        "candidates": evaluated,
                        "rejected": rejected,
                        "surplus": evaluated - rejected - count,
                        "published": count,
                        "shortfall": target - count,
                    },
                    "rejection_reasons": reasons,
                    "provenance": {
                        "run_spec_sha256": spec_hash,
                        "generation_mode": "specialized_internal_lazy_deferred",
                        "generation_version": RESEARCH_REF,
                        "environment": environment,
                        "seed_context": {
                            "root": args.seed,
                            "logical_ids": selected_ids,
                            "contract": "logical-id-research",
                        },
                        "effective_configuration": {
                            **spec,
                            "saturation": saturation,
                            "carryover": carryover,
                        },
                        "cell_namespace": f"{args.run_id}_{task}",
                    },
                }
                reference = store.publish(metadata, corpus)
                record, loaded = store.read_output(reference)
                assert record["counts"]["published"] == count
                if loaded is not None:
                    for key, value in corpus.items():
                        if isinstance(value, np.ndarray):
                            assert (
                                value.dtype == loaded[key].dtype
                                and value.tobytes() == loaded[key].tobytes()
                            ), key
                assert len(adapter.compile_records) == 1
                summary = {
                    "task": task,
                    "counts": metadata["counts"],
                    "reference": reference,
                    "builds": len(adapter.build_records),
                    "compilations": len(adapter.compile_records),
                    "stage_times": dict(adapter.times),
                }
                summaries.append(summary)
                save("progress.json", summaries)
                event("partition_complete", **summary)
                del adapter, corpus, loaded, chunks, batch, cells
                gc.collect()
        snapshot = None
        membership = {}
        if args.partition_index is None:
            snapshot = store.freeze_snapshot(
                args.run_id, "training", validation_fraction=0.2, split_seed=17
            )
            store.read_snapshot(snapshot)
            membership = store.read_training_split(snapshot)
        report = {
            "status": "pass"
            if all(p["counts"]["shortfall"] == 0 for p in summaries)
            else "shortfall",
            "target": target * len(summaries),
            "published": sum(p["counts"]["published"] for p in summaries),
            "partitions": summaries,
            "snapshot": snapshot,
            "training": sum(v == "train" for v in membership.values()),
            "validation": sum(v == "validation" for v in membership.values()),
            "elapsed_seconds": time.monotonic() - started,
        }
        save("result.json", report)
        if args.volume_name:
            volume.commit()
        event("complete", **{k: v for k, v in report.items() if k != "partitions"})
    except BaseException as exc:
        save("failure.json", {"error": repr(exc), "completed_partitions": len(summaries)})
        raise


if __name__ == "__main__":
    main()
