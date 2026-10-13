"""Protocol workload manifest and existing-template replay gate; not throughput evidence.

Run only inside the resource envelope in comparison_preflight.sh. This first
invocation intentionally does not implement the specialized candidate or corpus
packing. A failure blocks proceeding to speed comparisons; success is only a
preflight, not completion of the comparison protocol.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np

from template_vs_per_world import CORPUS_NAMES, dea_prior, make_workload
from pymc_generator.sampler import SATURATION_FAMILY_KEYS
from pymc_generator.world_model_template import (
    build_world_model_template, check_template_supported, compile_template_draw_fn,
)

OUT = Path(os.environ["COMPARISON_OUTPUT"])
START = time.monotonic()


def plain(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(name, data):
    with (OUT / name).open("x") as f:
        json.dump(plain(data), f, indent=2, allow_nan=False)
        f.write("\n")


def event(kind, **data):
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(plain({"event": kind, "elapsed_s": time.monotonic()-START, **data}), allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    OUT.mkdir(exist_ok=True)
    save("environment.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline": "b941d3a222fb91475cc2b59db14281e2b6498c49",
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": Path("/proc/cpuinfo").read_text(),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "environment": {k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTENSOR_FLAGS")},
        "scope": "local preflight only; no Modal, cost, accepted-output throughput or specialization evidence",
    })
    cases = [(k, m, "michaelis_menten", "geometric") for k,m in [(5,3),(5,10),(15,3),(15,10),(10,6)]]
    cases += [(10,6,s,"geometric") for s in SATURATION_FAMILY_KEYS if s != "michaelis_menten"]
    cases += [(10,6,"michaelis_menten",c) for c in ["weibull", "mixed"]]
    assert len(cases) == 12
    manifest = []
    first = None
    # Seed sets are fixed before any model/draw results are observed. Both draw
    # groupings use the same five structures; grouping remains part of identity.
    for case_index, (k,m,s,c) in enumerate(cases):
        for repeat in range(3):
            seed = 999000 + 1000*case_index + repeat
            cfg = dataclasses.replace(dea_prior(n_cells=5, draws_per_cell=1, seed=seed),
                n_treatments=15, n_covariates=10,
                n_treatments_active_range=(k,k), n_covariates_active_range=(m,m),
                saturation_family_probs={key: float(key == s) for key in SATURATION_FAMILY_KEYS},
                carryover_family_probs={"none":0.0,"geometric":0.5 if c=="mixed" else float(c=="geometric"),"weibull":0.5 if c=="mixed" else float(c=="weibull")})
            cfg.validate()
            check_template_supported(cfg)
            t = time.monotonic()
            cells = make_workload(cfg)
            prep_s = time.monotonic()-t
            entry = {"workload":case_index,"repeat":repeat,"active_counts":[k,m],"saturation":s,"carryover":c,
                     "draw_groupings":[1,10], "config":dataclasses.asdict(cfg),"preparation_s":prep_s,
                     "cells":[dataclasses.asdict(cell) for cell in cells]}
            manifest.append(entry)
            if first is None:
                first = cfg, cells
    save("workloads.json", manifest)
    event("manifest_complete", configurations=12, seed_sets=3, structures=180,
          planned_primary_candidates=5940, primary_candidates_executed=0)
    cfg, cells = first
    event("model_build_start", method="dynamic",workload=0,repeat=0)
    t=time.monotonic()
    model, _, _ = build_world_model_template(cfg,cells[0].template_inputs,cfg.n_time_steps)
    event("model_build_complete", seconds=time.monotonic()-t)
    event("compile_start", cache="empty isolated directory", mode="FAST_COMPILE")
    t=time.monotonic()
    draw = compile_template_draw_fn(model,CORPUS_NAMES)
    event("compile_complete",seconds=time.monotonic()-t)
    outputs=[]
    for index in [0,1,0]:
        cell=cells[index]
        event("diagnostic_draw_start",cell=index,draws=1,seed=cell.seed)
        t=time.monotonic()
        result=draw(cell.template_inputs,seed=cell.seed,draws=1)
        event("diagnostic_draw_complete",cell=index,seconds=time.monotonic()-t,draws=1)
        outputs.append(result)
    for name in CORPUS_NAMES:
        a,b=outputs[0][name],outputs[2][name]
        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(), name
        assert np.isfinite(a).all(),name
    event("same_mode_intervening_task_replay_pass", fields=len(CORPUS_NAMES))
    result=outputs[0]
    archive=OUT/"dynamic-replay.npz"
    np.savez_compressed(archive,**result)
    with np.load(archive,allow_pickle=False) as loaded:
        for name,value in result.items():
            back=loaded[name]
            assert value.dtype==back.dtype and value.shape==back.shape and value.tobytes()==back.tobytes(),name
    event("raw_array_readback_pass", sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    save("status.json",{"status":"preflight_pass_only", "diagnostic_draws":3,"primary_candidates":0,
                        "remaining":"specialized prototype, controlled-input equivalence, all-method replay, invariants, identical acceptance, corpus normalization/validation, full bounded comparison"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event("failure",exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
