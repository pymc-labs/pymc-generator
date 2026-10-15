"""18-way bounded Modal structural smoke, followed by an independent reader.

Run explicitly with `modal run --env main scripts/modal_structural_smoke.py`.
This is an integration harness, not a deployed service or production scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_REF = "d61f98f596c5de885947eb5144897ed0909e7c5d"
VOLUME_NAME = "pymc-generator-smoke-20261015"
ENVIRONMENT = "main"
MOUNT = "/vol"
WORKER_TIMEOUT = 900
CPU_RATE = 0.0000131
GIB_RATE = 0.00000222
MAX_COST = 5.0
# Price full resource/time allocations, including startup and scaledown reserve.
ESTIMATE = (
    18 * (WORKER_TIMEOUT + 600 + 2) * (CPU_RATE + 2 * GIB_RATE)
    + 600 * (CPU_RATE + 2 * GIB_RATE)
    + 1.0
)

app = modal.App("pymc-generator-structural-smoke")
volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential", "libgomp1")
    .uv_sync(str(ROOT), frozen=True)
    .env(
        {
            "PYTHONPATH": "/workspace",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
)
if modal.is_local():
    research = Path(os.environ.get("PYMC_SMOKE_RESEARCH_DIR", "/tmp/pymc-generator-200k-study"))
    actual = subprocess.check_output(
        ["git", "-C", str(research), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != RESEARCH_REF:
        raise RuntimeError("Incorrect pinned research revision")
    subprocess.run(
        ["git", "-C", str(research), "diff", "--exit-code", RESEARCH_REF, "--", "benchmarks"],
        check=True,
    )
    manifest = {
        "ref": RESEARCH_REF,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((research / "benchmarks").glob("*.py"))
        },
    }
    manifest_path = ROOT / "local-runs/modal-research-manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    image = (
        image.add_local_dir(
            ROOT / "pymc_generator", "/workspace/pymc_generator", ignore=["**/__pycache__/**"]
        )
        .add_local_file(
            ROOT / "scripts/local_structural_smoke.py",
            "/workspace/scripts/local_structural_smoke.py",
        )
        .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
        .add_local_file(manifest_path, "/research/manifest.json")
    )
    for path in sorted((research / "benchmarks").glob("*.py")):
        image = image.add_local_file(path, f"/research/benchmarks/{path.name}")


@app.function(
    image=image,
    volumes={MOUNT: volume},
    cpu=(1.0, 1.0),
    memory=(2048, 2048),
    timeout=WORKER_TIMEOUT,
    startup_timeout=600,
    retries=0,
    max_containers=18,
    scaledown_window=2,
    single_use_containers=True,
)
def generate_partition(run_id: str, partition: int) -> dict:
    """Exactly one family partition per container; no shared compiled RNG state."""
    started = time.time()
    output = Path(MOUNT) / run_id / "evidence" / f"partition-{partition:02d}"
    command = [
        sys.executable,
        "/workspace/scripts/local_structural_smoke.py",
        "--research-dir",
        "/research",
        "--research-manifest",
        "/research/manifest.json",
        "--output",
        str(output),
        "--store-root",
        f"{MOUNT}/{run_id}/store",
        "--volume-name",
        VOLUME_NAME,
        "--run-id",
        run_id,
        "--worlds-per-pair",
        "100",
        "--partition-index",
        str(partition),
    ]
    # Stdout stays in Modal logs; no open Volume log handle can block reload.
    result = subprocess.run(command, timeout=WORKER_TIMEOUT - 30, check=False)
    if result.returncode:
        volume.commit()  # Preserve diagnostics, not a claim of completed output.
        raise RuntimeError(f"Partition {partition} failed with exit {result.returncode}")
    volume.reload()
    report = json.loads((output / "result.json").read_text())
    report.update(
        partition=partition,
        started_at=started,
        finished_at=time.time(),
        container_id=os.environ.get("MODAL_TASK_ID"),
    )
    (output / "worker-result.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.function(
    image=image,
    volumes={MOUNT: volume},
    cpu=(1.0, 1.0),
    memory=(2048, 2048),
    timeout=600,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def verify_store(run_id: str, workers: list[dict]) -> dict:
    """A fresh container reloads and validates every persisted world and snapshot."""
    from pymc_generator.publication import MountedVolumePublicationStore

    started = time.time()
    store = MountedVolumePublicationStore(f"{MOUNT}/{run_id}/store", volume)
    outputs = store.validated_outputs(run_id)
    if len(outputs) != 18:
        raise RuntimeError(f"Expected 18 independently published partitions, found {len(outputs)}")
    counts = {
        key: sum(item["record"]["counts"][key] for item in outputs)
        for key in ("target", "candidates", "rejected", "surplus", "published", "shortfall")
    }
    if counts["published"] != 1800 or counts["shortfall"]:
        raise RuntimeError(f"Incomplete target: {counts}")
    snapshot = store.freeze_snapshot(run_id, "training", validation_fraction=0.2, split_seed=17)
    store.read_snapshot(snapshot)
    membership = store.read_training_split(snapshot)
    events = sorted(
        [(w["started_at"], 1) for w in workers] + [(w["finished_at"], -1) for w in workers]
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    worker_seconds = sum(w["finished_at"] - w["started_at"] for w in workers)
    report = {
        "status": "pass",
        "run_id": run_id,
        "volume": VOLUME_NAME,
        "environment": ENVIRONMENT,
        "partitions": len(outputs),
        "counts": counts,
        "peak_overlapping_worker_calls": peak,
        "worker_container_ids": [w["container_id"] for w in workers],
        "worker_seconds": worker_seconds,
        "reader_seconds": time.time() - started,
        "snapshot": snapshot,
        "training": sum(v == "train" for v in membership.values()),
        "validation": sum(v == "validation" for v in membership.values()),
        "bytes": sum(p.stat().st_size for p in (Path(MOUNT) / run_id / "store").rglob("*.npz")),
        "spend": {
            "approved_threshold_usd": MAX_COST,
            "preflight_reserved_estimate_usd": ESTIMATE,
            "requested_resource_compute_estimate_usd": (worker_seconds + time.time() - started)
            * (CPU_RATE + 2 * GIB_RATE),
            "note": "Estimate, not invoice or hard cap; startup/build/storage/billing lag separate",
            "pricing_source": "https://modal.com/pricing",
            "pricing_checked": "2026-10-15",
        },
    }
    report_path = Path(MOUNT) / run_id / "modal-result.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(run_id: str = "modal_smoke_20261015", max_cost: float = 5.0):
    if max_cost > MAX_COST or max_cost < ESTIMATE:
        raise ValueError(f"Approved threshold <= ${MAX_COST}; reserved estimate is ${ESTIMATE:.2f}")
    if not run_id.replace("_", "").isalnum():
        raise ValueError("run_id must be alphanumeric with underscores")
    local = ROOT / "local-runs" / run_id
    local.mkdir(exist_ok=False)
    (local / "budget.json").write_text(
        json.dumps(
            {
                "threshold": max_cost,
                "reserved_estimate": ESTIMATE,
                "workers": 18,
                "worker_timeout": WORKER_TIMEOUT,
            },
            indent=2,
        )
    )
    # Reserve the whole bounded wave before dispatch. No retries, replenished
    # worker wave or additional calls after failures; no sequential .remote loop.
    results = []
    for result in generate_partition.map(
        [run_id] * 18, range(18), order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, BaseException):
            results.append({"error": repr(result)})
        else:
            results.append(result)
        (local / "workers.json").write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in result for result in results):
        raise RuntimeError(
            "Worker failures; preserved results, no automatic retry or reader launch"
        )
    report = verify_store.remote(run_id, results)
    (local / "modal-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
