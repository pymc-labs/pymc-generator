"""Run one workload/seed-set invocation, with both draw groupings and all methods."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from comparison_methods import Adapter, equivalence_gate
from comparison_preflight import plain
from fixed_allocation import generate_fixed
from template_vs_per_world import CellWorkload, dea_prior

from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus

OUT = Path(os.environ["COMPARISON_OUTPUT"])
START = time.perf_counter()


def event(kind, **kwargs):
    with (OUT / "events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                plain({"event": kind, "elapsed_s": time.perf_counter() - START, **kwargs}),
                allow_nan=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def save(name, value):
    with (OUT / name).open("x") as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)
        f.write("\n")


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"dtype", "shape", "values"}:
            return np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def arrays(value, prefix=""):
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from arrays(v, prefix + "/" + k)


def memory():
    group = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[-1].lstrip("/")
    peak = Path("/sys/fs/cgroup") / group / "memory.peak"
    return {
        "self_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope_peak_bytes": int(peak.read_text()) if peak.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("commit prototype before measurements: " + dirty)
    save(
        "environment.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "workload": args.workload,
            "repeat": args.repeat,
            "cache": "empty per invocation, shared across methods within invocation",
            "compiler_mode": "FAST_COMPILE",
            "method_order": ["ordinary", "dynamic", "specialized"],
            "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "thread_environment": {
                k: os.environ.get(k)
                for k in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "PYTENSOR_FLAGS",
                )
            },
        },
    )
    t = time.perf_counter()
    manifest = json.loads(
        (Path(__file__).parent / "comparison-preflight-evidence/workloads.json").read_text()
    )
    entry = next(
        x for x in manifest if x["workload"] == args.workload and x["repeat"] == args.repeat
    )
    source = entry["config"]
    cfg = dataclasses.replace(
        dea_prior(n_cells=5, draws_per_cell=1, seed=source["seed"]),
        n_treatments=15,
        n_covariates=10,
        n_treatments_active_range=tuple(source["n_treatments_active_range"]),
        n_covariates_active_range=tuple(source["n_covariates_active_range"]),
        saturation_family_probs=source["saturation_family_probs"],
        carryover_family_probs=source["carryover_family_probs"],
    )
    cfg.validate()
    assert plain(dataclasses.asdict(cfg)) == source, "configuration drift from immutable manifest"
    cells = [CellWorkload(**decode(x)) for x in entry["cells"]]
    prepare_s = time.perf_counter() - t
    event(
        "inputs_loaded", seconds=prepare_s, original_structure_preparation_s=entry["preparation_s"]
    )
    adapters = {}
    for method in ("ordinary", "dynamic", "specialized"):
        adapter = Adapter(method, cfg, cells, event)
        adapters[method] = adapter
        for draws in (1, 10):
            adapter.accepted_ids = []
            before = dict(adapter.times)
            t = time.perf_counter()
            corpus = generate_fixed(dataclasses.replace(cfg, draws_per_cell=draws), adapter)
            generation_s = time.perf_counter() - t
            summary = {
                "method": method,
                "workload": args.workload,
                "repeat": args.repeat,
                "draws_per_structure": draws,
                "candidate_allocation": 5 * draws,
                "evaluated": adapter.evaluated,
                "rejected": adapter.rejected,
                "accepted": len(adapter.accepted_ids),
                "logical_candidate_ids": adapter.accepted_ids,
                "physical_to_logical_cells": adapter.retained_cell_ids,
                "generation_including_build_compile_acceptance_packing_s": generation_s,
                "input_loading_s": prepare_s,
                "original_structure_preparation_s": entry["preparation_s"],
                "stages": {k: v - before.get(k, 0) for k, v in adapter.times.items()},
                "compiler_cache": "shared within invocation; see events for compile positions",
                "model_state": "fresh models"
                if draws == 1
                else "reused models after one-draw grouping and replay",
                "rejection_reasons": "aggregate shared realism/support-scale gate; no retries or replenishment",
            }
            if corpus is not None:
                t = time.perf_counter()
                errors = DataGenerator.validate_corpus(corpus)
                summary["validation_s"] = time.perf_counter() - t
                event("corpus_validation", method=method, draws=draws, errors=errors)
                if errors:
                    raise AssertionError(errors)
                path = OUT / f"{method}-{draws}.npz"
                t = time.perf_counter()
                save_corpus(corpus, path)
                summary["serialization_s"] = time.perf_counter() - t
                t = time.perf_counter()
                back = load_corpus(path)
                left = dict(arrays(corpus))
                right = dict(arrays(back))
                assert left.keys() == right.keys()
                for name, a in left.items():
                    b = right[name]
                    assert (
                        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                    ), name
                assert not DataGenerator.validate_corpus(back)
                summary["readback_and_revalidation_s"] = time.perf_counter() - t
                summary["persisted_bytes"] = path.stat().st_size
                summary["pipeline_s"] = generation_s + sum(
                    summary[k]
                    for k in ("validation_s", "serialization_s", "readback_and_revalidation_s")
                )
                summary["accepted_locally_persisted_per_s"] = (
                    summary["accepted"] / summary["pipeline_s"]
                )
            else:
                summary["pipeline_s"] = generation_s
                summary["accepted_locally_persisted_per_s"] = 0
                summary["no_corpus_reason"] = (
                    "zero accepted candidates; no replenishment authorized"
                )
            adapter.replay(draws)
            summary["replay_pass"] = True
            summary["memory_invocation_cumulative"] = memory()
            save(f"{method}-{draws}.json", summary)
            event(
                "group_complete",
                method=method,
                draws=draws,
                accepted=summary["accepted"],
                pipeline_s=summary["pipeline_s"],
            )
    # Gate one concrete structure per workload and seed-set. This is a small
    # controlled-input screen, not a proof of distributional equivalence.
    equivalence_gate(adapters, event)
    save(
        "result.json",
        {
            "status": "pass",
            "memory": memory(),
            "elapsed_s": time.perf_counter() - START,
            "executions": {k: v.executions for k, v in adapters.items()},
            "gate_scope": "one controlled-input structure; all primary tasks replayed, both groupings; not statistical equivalence",
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event(
            "failure",
            exception=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
            memory=memory(),
        )
        raise
