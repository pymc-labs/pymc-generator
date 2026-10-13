"""Replay only cells containing recorded rejects and identify their exact gates."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from comparison_methods import Adapter
from comparison_run import decode, event, memory, save
from template_vs_per_world import CORPUS_NAMES, CellWorkload, dea_prior

from pymc_generator.data_generator import load_corpus
from pymc_generator.sampler import _additive_task_ok, _make_support_mask


def main():
    source = Path(os.environ["AUDIT_SOURCE"])
    summary = json.loads((source / "specialized-10.json").read_text())
    corpus = load_corpus(source / "specialized-10.npz")
    manifest = json.loads(
        (Path(__file__).parent / "comparison-preflight-evidence/workloads.json").read_text()
    )
    entry = next(
        x
        for x in manifest
        if (x["workload"], x["repeat"]) == (summary["workload"], summary["repeat"])
    )
    config = entry["config"]
    cfg = dataclasses.replace(
        dea_prior(n_cells=5, draws_per_cell=10, seed=config["seed"]),
        n_treatments=15,
        n_covariates=10,
        n_treatments_active_range=tuple(config["n_treatments_active_range"]),
        n_covariates_active_range=tuple(config["n_covariates_active_range"]),
        saturation_family_probs=config["saturation_family_probs"],
        carryover_family_probs=config["carryover_family_probs"],
    )
    cfg.validate()
    cells = [CellWorkload(**decode(x)) for x in entry["cells"]]
    accepted = {tuple(x): i for i, x in enumerate(summary["logical_candidate_ids"])}
    rejected = [(c, d) for c in range(5) for d in range(10) if (c, d) not in accepted]

    def audit_event(kind, **data):
        if kind == "draw":
            data["diagnostic"] = True
        event(kind, **data)

    adapter = Adapter("specialized", cfg, cells, audit_event)
    reasons = []
    for cell in sorted({c for c, d in rejected}):
        adapter.current_cell = cell
        adapter.build(cells[cell].g, cfg, cells[cell].structure, cfg.n_time_steps)
        result = adapter.draw(CORPUS_NAMES, 10)
        # Adapter's default classification is overridden here: every execution
        # in this audit is diagnostic, never another primary contribution.
        for d in range(10):
            value = {k: v[d] for k, v in result.items()}
            treatment, outcome = value["treatments"], value["outcome"]
            support, _ = _make_support_mask(
                np.random.default_rng(np.random.SeedSequence([cfg.seed, cell, d, 4401])),
                cfg.n_time_steps,
                cfg.n_query,
                cfg.p_long_horizon,
            )
            scale = float(np.std(outcome[support == 1]))
            good = (
                _additive_task_ok(
                    treatment, outcome, value, cells[cell].g["g_cy"], cfg.treatment_cv_floor
                )
                and np.isfinite(scale)
                and scale > 0
            )
            assert bool(good) == ((cell, d) in accepted), "acceptance replay disagrees"
            if good:
                np.testing.assert_array_equal(
                    outcome.astype("float32"), corpus["outcome_raw"][accepted[cell, d]]
                )
                continue
            failures = []
            if not all(np.isfinite(v).all() for v in value.values()):
                failures.append("nonfinite")
            if (outcome < 0).any():
                failures.append("negative_outcome")
            direct = cells[cell].g["g_cy"] == 1
            cv = treatment.std(axis=0) / (treatment.mean(axis=0) + 1e-12)
            if direct.any() and cv[direct].min() < cfg.treatment_cv_floor:
                failures.append("direct_treatment_cv_below_floor")
            med = np.median(outcome)
            ratio = float(outcome.max() / (med if med > 0 else 1.0))
            if ratio >= 8.0:
                failures.append("outcome_spike_ratio")
            tm = np.median(treatment, axis=0)
            tr = float((treatment.max(axis=0) / np.where(tm > 0, tm, 1.0)).max())
            if tr >= 50.0:
                failures.append("treatment_spike_ratio")
            if not np.isfinite(scale) or scale <= 0:
                failures.append("invalid_support_outcome_scale")
            assert failures
            record = {
                "cell": cell,
                "draw": d,
                "failures": failures,
                "minimum_direct_cv": float(cv[direct].min()),
                "cv_floor": cfg.treatment_cv_floor,
                "outcome_spike_ratio": ratio,
                "treatment_spike_ratio": tr,
                "support_scale": scale,
            }
            reasons.append(record)
            np.savez_compressed(
                Path(os.environ["COMPARISON_OUTPUT"]) / f"rejected-{cell}-{d}.npz", **value
            )
    save(
        "rejections.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "workload": summary["workload"],
            "repeat": summary["repeat"],
            "status": "pass",
            "diagnostic_draws": 10 * len({c for c, d in rejected}),
            "primary_draws": 0,
            "rejections": reasons,
            "memory": memory(),
            "scope": "acceptance replay and accepted outcome float32 equality",
        },
    )


if __name__ == "__main__":
    main()
