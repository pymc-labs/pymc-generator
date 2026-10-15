"""Explicit opt-in streaming launch. Never deploys/replaces the dashboard service.

modal run --env pfn --detach scripts/modal_streaming.py --label stream-test
See live_dashboard/STREAMING.md before changing quotas or resuming.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "pfn"
MOUNT = "/vol"
WORKERS = int(os.environ.get("STREAM_WORKERS", "2"))
if not 1 <= WORKERS <= 16:
    raise ValueError("STREAM_WORKERS must be 1..16")
WORKER_TIMEOUT = 1800
RATE = (0.0000131 + 2 * 0.00000222) * 3600
app = modal.App("pymc-generator-streaming")
volume = modal.Volume.from_name("pymc-generator-live-dashboard", environment_name=ENVIRONMENT)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential", "libgomp1")
    .uv_sync(str(ROOT), frozen=True)
    .env(
        {
            "PYTHONPATH": "/workspace",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)
if modal.is_local():
    image = image.add_local_dir(
        ROOT / "pymc_generator", "/workspace/pymc_generator", ignore=["**/__pycache__/**"]
    ).add_local_dir(
        ROOT / "live_dashboard", "/workspace/live_dashboard", ignore=["**/__pycache__/**"]
    )


def parse_plan(data):
    from live_dashboard.streaming import StreamPlan

    return StreamPlan(**{**data, "families": tuple(tuple(p) for p in data["families"])})


@app.function(
    image=image,
    volumes={MOUNT: volume},
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=600,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def coordinate(
    data: dict,
    resume: bool = False,
    previous_stopped: bool = False,
    attempt_id: str = "",
    results: list | None = None,
):
    from live_dashboard import publication as pub
    from live_dashboard.storage import DirectoryStore
    from live_dashboard.streaming import prepare_run
    from pymc_generator.publication import MountedVolumePublicationStore

    plan = parse_plan(data)
    store = DirectoryStore(MOUNT, on_commit=volume.commit, on_refresh=volume.reload)
    if not attempt_id:
        manifest = prepare_run(store, plan, resume=resume, previous_stopped=previous_stopped)
        now = pub.utc_now()
        attempt_id = pub.new_attempt_id(now)
        # Local coordinator is not a continuously billed generation container.
        record = pub.AttemptRecord(
            manifest.run_id,
            attempt_id,
            "running",
            now,
            now,
            now + timedelta(seconds=len(plan.families) * (WORKER_TIMEOUT + 600) + 1200),
            None,
            (),
            (),
            None,
            pub.WorkerInfo("stream coordinator", None, None, None),
        )
        store.write_new(pub.attempt_record_path(manifest.run_id, attempt_id), record.encode())
        store.commit()
        return {"run_id": manifest.run_id, "attempt_id": attempt_id}
    store.refresh()
    manifest = pub.decode_run_manifest(
        store.read_bytes(pub.run_manifest_path(plan.request.run_id)), run_id=plan.request.run_id
    )
    record = pub.decode_attempt_record(
        store.read_bytes(pub.attempt_record_path(manifest.run_id, attempt_id)),
        manifest=manifest,
        attempt_id=attempt_id,
    )
    outputs = MountedVolumePublicationStore(MOUNT, volume).validated_outputs(manifest.run_id)
    count = sum(item["record"]["counts"]["published"] for item in outputs)
    failed = any("error" in result for result in results)
    state = "failed" if failed else "finished" if count == plan.request.worlds else "stopped"
    record = dataclasses.replace(
        record,
        state=state,
        updated_at=pub.utc_now(),
        published_task_ids=tuple(item["record"]["task_id"] for item in outputs),
        error=pub.ErrorInfo("WorkerFailure", str(results)) if failed else None,
    )
    store.write_replace(pub.attempt_record_path(manifest.run_id, attempt_id), record.encode())
    store.write_new(
        f"runs/{manifest.run_id}/stream-{attempt_id}.json", json.dumps(results, indent=2).encode()
    )
    store.commit()
    return {
        "run_id": manifest.run_id,
        "published_worlds": count,
        "target": plan.request.worlds,
        "state": state,
        "workers": results,
    }


@app.function(
    image=image,
    volumes={MOUNT: volume},
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=WORKER_TIMEOUT,
    startup_timeout=600,
    retries=0,
    max_containers=WORKERS,
    scaledown_window=2,
    single_use_containers=True,
)
def generate(data: dict, partition: int, attempt_id: str, queue_size: int):
    import platform
    import resource

    from live_dashboard import publication as pub
    from live_dashboard.storage import DirectoryStore
    from live_dashboard.stream_store import StreamPublicationStore
    from live_dashboard.streaming import run_partition

    start = time.monotonic()
    plan = parse_plan(data)
    store = DirectoryStore(MOUNT, on_refresh=volume.reload)
    store.refresh()
    manifest = pub.decode_run_manifest(
        store.read_bytes(pub.run_manifest_path(plan.request.run_id)), run_id=plan.request.run_id
    )
    # Main thread touches storage only before/after run_partition's writer lifetime.
    result = run_partition(
        plan,
        partition,
        manifest,
        attempt_id,
        StreamPublicationStore(MOUNT, volume),
        queue_size=queue_size,
        stop=lambda: time.monotonic() - start > WORKER_TIMEOUT - 180,
        worker=pub.WorkerInfo(platform.platform(), 1, None, None, RATE),
    )
    result.update(
        container_id=os.environ.get("MODAL_TASK_ID"),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        estimated_compute_usd=(time.monotonic() - start) * RATE / 3600,
    )
    return result


@app.local_entrypoint()
def main(
    label: str,
    worlds_per_split: int = 100,
    batch_size: int = 10,
    seed: int = 631227,
    families: str = "hill:geometric,logistic:geometric",
    queue_size: int = 2,
    max_cost: float = 1.0,
    resume: bool = False,
    previous_stopped: bool = False,
):
    from live_dashboard.streaming import StreamPlan

    plan = StreamPlan(
        label,
        worlds_per_split,
        batch_size,
        seed,
        tuple(tuple(pair.split(":")) for pair in families.split(",")),
    )
    if not 1 <= queue_size <= 8:
        raise ValueError("queue_size must be 1..8")
    # Reserve a complete bounded wave, not an optimistic throughput extrapolation.
    estimate = (
        len(plan.families) * (WORKER_TIMEOUT + 600 + 2) + 2 * (600 + 600 + 2)
    ) * RATE / 3600 + 0.5
    if not math.isfinite(max_cost) or max_cost <= 0 or estimate > max_cost:
        raise ValueError(f"reserved estimate ${estimate:.3f} exceeds approved ${max_cost:.3f}")
    output = ROOT / "local-runs" / plan.request.run_id / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    data = dataclasses.asdict(plan)
    (output / "budget.json").write_text(
        json.dumps({"approved": max_cost, "reserved_estimate": estimate, "plan": data}, indent=2)
    )
    info = coordinate.remote(data, resume, previous_stopped)
    (output / "launch.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    results = []
    for result in generate.starmap(
        [(data, i, info["attempt_id"], queue_size) for i in range(len(plan.families))],
        order_outputs=False,
        return_exceptions=True,
    ):
        results.append({"error": repr(result)} if isinstance(result, BaseException) else result)
        (output / "workers.json").write_text(json.dumps(results, indent=2))
    report = coordinate.remote(data, attempt_id=info["attempt_id"], results=results)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["state"] == "failed":
        raise RuntimeError("worker failure; no automatic retry")
