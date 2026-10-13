"""Recompute descriptive summaries and SHA-256 inventory from saved evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "comparison-evidence"
METHODS = ("ordinary", "dynamic", "specialized")


def main():
    rows = []
    peaks = []
    raw_executions = []
    gate_count = 0
    for w in range(12):
        for rep in range(3):
            path = EVIDENCE / f"w{w}-r{rep}"
            result = json.loads((path / "result.json").read_text())
            assert result["status"] == "pass"
            peaks.append(result["memory"]["scope_peak_bytes"])
            for runs in result["executions"].values():
                raw_executions.extend(runs)
            events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
            assert sum(e["event"] == "controlled_equivalence_pass" for e in events) == 1
            gate_count += 1
            for method in METHODS:
                for grouping in (1, 10):
                    value = json.loads((path / f"{method}-{grouping}.json").read_text())
                    assert value["replay_pass"]
                    assert value["evaluated"] == value["candidate_allocation"] == 5 * grouping
                    assert value["accepted"] + value["rejected"] == value["evaluated"]
                    assert len(value["logical_candidate_ids"]) == value["accepted"]
                    assert (
                        len({tuple(x) for x in value["logical_candidate_ids"]}) == value["accepted"]
                    )
                    rows.append(value)
    fields = [
        "workload",
        "repeat",
        "method",
        "draws_per_structure",
        "evaluated",
        "accepted",
        "rejected",
        "pipeline_s",
        "accepted_locally_persisted_per_s",
        "validation_s",
        "serialization_s",
        "readback_and_revalidation_s",
    ]
    with (ROOT / "comparison-summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ledger = json.loads((EVIDENCE / "budget-final.json").read_text())
    summary = {
        "invocations_passed": 36,
        "controlled_input_gates_passed": gate_count,
        "primary_draws": sum(x["draws"] for x in raw_executions if not x["diagnostic"]),
        "replay_draws": sum(x["draws"] for x in raw_executions if x["diagnostic"]),
        "controlled_joint_primitive_draws": 36,
        "rejection_audit_draws": 20,
        "preflight_diagnostic_draws": 3,
        "charged_seconds_including_preflight_reservation": ledger["charged_s"],
        "remaining_seconds": 7200 - ledger["charged_s"],
        "max_invocation_memory_peak_bytes": max(peaks),
        "methods": {
            m: {
                k: sum(r[k] for r in rows if r["method"] == m)
                for k in ("evaluated", "accepted", "rejected")
            }
            for m in METHODS
        },
        "statistics": [],
    }
    table = [
        "# Descriptive results (three seed sets)",
        "",
        "Warm rate = accepted, validated, NPZ-persisted and read-back worlds/s, including normalization. Models reused from the one-draw grouping. Values are median [min, max], not confidence intervals.",
        "",
        "| Workload | Ordinary warm rate | Dynamic warm rate | Specialized warm rate |",
        "|---|---:|---:|---:|",
    ]
    manifest = json.loads((ROOT / "comparison-preflight-evidence/workloads.json").read_text())
    for w in range(12):
        entry = next(x for x in manifest if x["workload"] == w)
        label = f"{tuple(entry['active_counts'])} {entry['saturation']} / {entry['carryover']}"
        cells = []
        for m in METHODS:
            all_rows = [r for r in rows if r["workload"] == w and r["method"] == m]
            rates = [
                r["accepted_locally_persisted_per_s"]
                for r in all_rows
                if r["draws_per_structure"] == 10
            ]
            first = [r["pipeline_s"] for r in all_rows if r["draws_per_structure"] == 1]
            item = {
                "workload": w,
                "label": label,
                "method": m,
                "warm_rate_median": statistics.median(rates),
                "warm_rate_min": min(rates),
                "warm_rate_max": max(rates),
                "first_use_pipeline_median_s": statistics.median(first),
                "first_use_pipeline_min_s": min(first),
                "first_use_pipeline_max_s": max(first),
            }
            summary["statistics"].append(item)
            cells.append(f"{item['warm_rate_median']:.2f} [{min(rates):.2f}, {max(rates):.2f}]")
        table.append("| " + label + " | " + " | ".join(cells) + " |")
    table += [
        "",
        "First-use pipeline = five candidates (one per structure), including build/compile and persistence. This is NOT an independent empty-cache result per method: methods execute ordinary → dynamic → specialized in each fresh process and share that invocation's compiler cache.",
        "",
        "| Workload | Ordinary first use (s) | Dynamic first use (s) | Specialized first use (s) |",
        "|---|---:|---:|---:|",
    ]
    for w in range(12):
        items = [x for x in summary["statistics"] if x["workload"] == w]
        table.append(
            "| "
            + items[0]["label"]
            + " | "
            + " | ".join(
                f"{x['first_use_pipeline_median_s']:.2f} [{x['first_use_pipeline_min_s']:.2f}, {x['first_use_pipeline_max_s']:.2f}]"
                for x in items
            )
            + " |"
        )
    for path, value in ((ROOT / "comparison-summary.json", summary),):
        path.write_text(json.dumps(value, indent=2) + "\n")
    (ROOT / "comparison-tables.md").write_text("\n".join(table) + "\n")
    inventory = []
    for path in sorted(EVIDENCE.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (ROOT / "comparison-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "statistics"}, indent=2))


if __name__ == "__main__":
    main()
