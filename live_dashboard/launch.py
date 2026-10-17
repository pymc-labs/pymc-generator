"""Start a generation run on the deployed live dashboard app (Modal environment pfn).

From the repository root, after ``modal deploy -e pfn live_dashboard/modal_app.py``:

    modal run -e pfn live_dashboard/launch.py --label demo --worlds 20 --worlds-per-task 2

The run id derives from the request (label, preset, worlds, worlds per task, seed), so
it is known before the worker starts. The seed defaults to 29; a structural-split design
(``live_dashboard.structural``) defaults to the launch time as ``YYYYMMDDHHMMSS`` (UTC),
from which every split's seed derives. The launcher prints the seed: resuming needs the
same ``--seed``. A run that already exists is refused unless ``--resume`` is given
(resuming only generates its unpublished tasks, and the worker refuses if the code,
configuration or numerical environment changed). For a new run the launcher waits until
the worker has published the run's manifest (or reports why it refused or failed first);
``--wait`` blocks until the worker returns its summary.

This launcher is a function-less Modal app: it only looks up the deployed functions.
"""

from __future__ import annotations

import re
import time
from pathlib import PurePosixPath

import modal

from live_dashboard import structural
from live_dashboard.deployment import (
    APP_NAME,
    DASHBOARD_FUNCTION,
    ENVIRONMENT,
    GENERATE_FUNCTION,
    LAUNCHER_APP_NAME,
    VOLUME_NAME,
    check_demo_bounds,
)
from live_dashboard.generation import RequestError, preset_cycle
from live_dashboard.publication import RUNS_DIR, RunRequest, utc_now

if modal.is_local() and modal.config.config.get("environment") != ENVIRONMENT:
    raise RuntimeError(
        f"the live dashboard lives in the Modal environment {ENVIRONMENT!r}: "
        f"run `modal run -e {ENVIRONMENT} live_dashboard/launch.py ...`"
    )

app = modal.App(LAUNCHER_APP_NAME)
# How long a new run may take to appear (worker cold start, image pull, imports).
START_WAIT_S = 600
_POLL_S = 5
DEFAULT_SEED = 29


def default_seed(preset: str) -> int:
    """The seed of a launch without ``--seed``: the launch time for a design, else 29."""
    return structural.launch_seed(utc_now()) if preset in structural.DESIGNS else DEFAULT_SEED


def _published_runs(volume: modal.Volume) -> set[str]:
    """Run ids present on the Volume (an absent ``runs`` directory means none)."""
    top = {PurePosixPath(entry.path).name for entry in volume.listdir("/")}
    if RUNS_DIR not in top:
        return set()
    return {PurePosixPath(entry.path).name for entry in volume.listdir(RUNS_DIR)}


def spawn_arguments(request: RunRequest) -> tuple[str, str, int, int, int, str]:
    """Positional arguments of the deployed ``generate_run`` for ``request``."""
    return (
        request.label,
        request.preset,
        request.worlds,
        request.worlds_per_task,
        request.seed,
        request.run_id,
    )


@app.local_entrypoint()
def main(
    label: str,
    worlds: int = 20,
    worlds_per_task: int = 2,
    seed: int = -1,
    preset: str = "showcase-variety",
    resume: bool = False,
    wait: bool = False,
) -> None:
    explicit_seed = seed != -1
    if not explicit_seed:
        seed = default_seed(preset)
    try:
        request = RunRequest(
            label=label, preset=preset, worlds=worlds, worlds_per_task=worlds_per_task, seed=seed
        )
        check_demo_bounds(request)
    except ValueError as error:
        raise SystemExit(f"invalid request: {error}") from error
    try:
        preset_cycle(request.preset)
    except RequestError as error:
        raise SystemExit(str(error)) from error
    volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
    runs = _published_runs(volume)
    exists = request.run_id in runs
    labelled = sorted(
        run for run in runs if re.fullmatch(rf"{re.escape(label)}-[0-9a-f]{{12}}", run)
    )
    if request.design is not None and not explicit_seed and labelled:
        # Without --seed every launch of a design is a new sample (a new run id).
        raise SystemExit(
            f"runs labelled {label!r} exist ({', '.join(labelled)}); pass --seed <its base seed> "
            "--resume to continue one, or choose another --label for a new sample"
        )
    if exists and not resume:
        raise SystemExit(
            f"run {request.run_id} already exists; pass --resume to continue it, or choose "
            "another --label for a fresh run"
        )
    if resume and not exists:
        raise SystemExit(f"--resume: run {request.run_id} does not exist (same --seed?)")
    dashboard = modal.Function.from_name(APP_NAME, DASHBOARD_FUNCTION, environment_name=ENVIRONMENT)
    generate = modal.Function.from_name(APP_NAME, GENERATE_FUNCTION, environment_name=ENVIRONMENT)
    call = generate.spawn(*spawn_arguments(request))
    print(f"run {request.run_id}: {request.n_tasks} tasks x {request.worlds_per_task} worlds")
    print(f"seed {request.seed} (resume with --seed {request.seed} --resume)")
    print(f"dashboard: {dashboard.get_web_url()}/?run={request.run_id}")
    print(f"call: {call.object_id}")
    if wait:
        _report(call.get())
    elif not exists:
        _await_start(call, volume, request.run_id)


def _report(result: dict) -> None:
    """Print the worker's summary; a refusal ends the launcher with a non-zero status."""
    if result.get("status") == "refused":
        raise SystemExit(f"worker refused the run: {result.get('reason')}")
    print(result)


def _manifest_published(volume: modal.Volume, run_id: str) -> bool:
    if run_id not in _published_runs(volume):
        return False
    return any(
        PurePosixPath(entry.path).name == "run.json"
        for entry in volume.listdir(f"{RUNS_DIR}/{run_id}")
    )


def _await_start(call: modal.FunctionCall, volume: modal.Volume, run_id: str) -> None:
    """Return once the run's manifest is published; exit if the worker refused the run.

    ``call.get`` raises the builtin ``TimeoutError`` while the call is running (the worker
    re-raises its own timeouts as other errors, so this cannot be mistaken for its result);
    every poll still lasts at least ``_POLL_S`` so nothing unexpected can spin the loop.
    """
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        polled = time.monotonic()
        try:
            result = call.get(timeout=_POLL_S)
        except TimeoutError:
            if _manifest_published(volume, run_id):
                print("worker started the run; follow it on the dashboard")
                return
            time.sleep(max(0.0, polled + _POLL_S - time.monotonic()))
            continue
        _report(result)
        return
    print(
        f"call {call.object_id} has not started run {run_id} yet (it may be queued behind "
        "another run or waiting for a container); it starts on its own: follow the dashboard link"
    )
