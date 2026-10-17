"""Modal deployment: a CPU generation worker and a CPU dashboard, in environment pfn.

Deploy (the image is built from the repository's pyproject.toml and uv.lock):

    modal deploy -e pfn live_dashboard/modal_app.py

``generate_run`` generates on a CPU container (one container at a time, tasks in order)
and publishes into the Volume; ``dashboard`` is a separate CPU container serving the
authenticated web app at a stable URL. Start runs with ``live_dashboard/launch.py``.

No GPU: PyTensor generation has no CUDA backend, and CPU-only workers measured as fast
as A100 workers for about 1/20 of the cost (the decision is in live_dashboard/README.md).

Module scope imports only the standard library, ``modal`` and the stdlib-only
``live_dashboard.deployment``/``live_dashboard.revision``: the Modal CLI imports this
file with its own interpreter, and containers re-import it without git or the repository.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import modal

from live_dashboard.deployment import (
    APP_NAME,
    DASHBOARD_LABEL,
    ENVIRONMENT,
    GENERATION_CPU,
    GENERATION_MEMORY_MIB,
    GENERATION_TIMEOUT_S,
    GENERATION_USD_PER_HOUR,
    REVISION_ENV,
    SECRET_NAME,
    VOLUME_NAME,
)
from live_dashboard.revision import code_revision

MOUNT = "/publications"
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO = _PACKAGE_DIR.parent
_IGNORE = ["**/__pycache__", "**/__pycache__/**", "**/*.pyc"]
# A cold dashboard container rereads the whole store: about 30 ms per file on the Volume,
# so minutes for a 200,000-world run. DASHBOARD_KEEP_WARM=1 at deploy time keeps one
# container running (and its caches) while a large run is supervised; any deploy without
# it scales to zero again (and restarts the container cold).
_KEEP_WARM = os.environ.get("DASHBOARD_KEEP_WARM", "0") == "1"
# Seconds of payload reads and extractions per refresh (listings and records not
# counted), going on until one unread payload's outcome is cached; the rest show as
# loading and are read by the next refreshes, newest run first (live_dashboard.collection).
# 10 s read 100-world payloads (about 1 s each) only as fast as five workers publish them.
DASHBOARD_EXTRACTION_BUDGET_S = 30.0
_base_image = modal.Image.debian_slim(python_version="3.13")
if modal.is_local():
    if modal.config.config.get("environment") != ENVIRONMENT:
        raise RuntimeError(
            f"the live dashboard lives in the Modal environment {ENVIRONMENT!r}: "
            f"run `modal deploy -e {ENVIRONMENT} live_dashboard/modal_app.py`"
        )
    image = (
        _base_image.uv_sync(str(_REPO), extras=["dashboard"], frozen=True)
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                # Match the single-threaded numerics the generator was benchmarked with.
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                REVISION_ENV: code_revision(_REPO) or "",
            }
        )
        .add_local_dir(_REPO / "pymc_generator", remote_path="/root/pymc_generator", ignore=_IGNORE)
        .add_local_dir(_PACKAGE_DIR, remote_path="/root/live_dashboard", ignore=_IGNORE)
    )
else:
    # Inside a container the image is already built; local paths and git do not exist.
    image = _base_image

# Volume v1 (the durable, generally available version); this app is its only writer.
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT, create_if_missing=True, version=1
)
# The dashboard only reads generation-owned data; reload() works on the read-only mount.
dashboard_volume = volume.with_mount_options(read_only=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=GENERATION_CPU,
    memory=GENERATION_MEMORY_MIB,
    volumes={MOUNT: volume},
    timeout=GENERATION_TIMEOUT_S,
    retries=modal.Retries(max_retries=1, initial_delay=5.0),
    max_containers=1,
    scaledown_window=10,
)
def generate_run(
    label: str,
    preset: str,
    worlds: int,
    worlds_per_task: int,
    seed: int,
    expected_run_id: str | None = None,
) -> dict:
    """Generate and publish one run (resuming its unpublished tasks) into the Volume."""
    from datetime import timedelta

    from live_dashboard.generation import run_generation_or_refuse
    from live_dashboard.publication import RunRequest, WorkerInfo, utc_now
    from live_dashboard.storage import DirectoryStore

    # Modal kills this attempt at the timeout: record that bound; the writer stops starting
    # tasks early enough to end before it.
    deadline = utc_now() + timedelta(seconds=GENERATION_TIMEOUT_S)
    try:
        request = RunRequest(
            label=label, preset=preset, worlds=worlds, worlds_per_task=worlds_per_task, seed=seed
        )
    except ValueError as error:
        return {"status": "refused", "reason": str(error)}
    worker = WorkerInfo(
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        gpu_requested=None,
        gpu_name=None,
        usd_per_hour=GENERATION_USD_PER_HOUR,
    )
    store = DirectoryStore(MOUNT, on_commit=volume.commit, on_refresh=volume.reload)
    try:
        return run_generation_or_refuse(
            store, request, worker=worker, deadline=deadline, expected_run_id=expected_run_id
        )
    except TimeoutError as error:
        # A caller's FunctionCall.get raises the builtin TimeoutError while the call is still
        # running; a worker timeout (e.g. ETIMEDOUT on the Volume) must not look like that.
        raise RuntimeError(f"{type(error).__name__}: {error}") from error


@app.function(
    cpu=1.0,
    memory=4096,  # a 200,000-world run's snapshot, extraction cache and figures
    volumes={MOUNT: dashboard_volume},
    secrets=[
        modal.Secret.from_name(
            SECRET_NAME,
            environment_name=ENVIRONMENT,
            required_keys=["DASHBOARD_USERNAME", "DASHBOARD_PASSWORD_HASH"],
        )
    ],
    min_containers=1 if _KEEP_WARM else 0,
    max_containers=1,
    scaledown_window=600,
    timeout=300,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label=DASHBOARD_LABEL)
def dashboard():
    """The authenticated dashboard; one container, one poller over the read-only Volume."""
    from live_dashboard.collection import Collector
    from live_dashboard.storage import DirectoryStore
    from live_dashboard.web import create_app, settings_from_env

    settings = settings_from_env()
    store = DirectoryStore(MOUNT, on_refresh=dashboard_volume.reload)
    return create_app(
        Collector(store, extraction_budget_s=DASHBOARD_EXTRACTION_BUDGET_S),
        credentials=settings.credentials,
        poll_seconds=settings.poll_seconds,
        active_poll_seconds=settings.active_poll_seconds,
    )
