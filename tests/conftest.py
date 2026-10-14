"""Shared pytest configuration.

Keeps the --runslow gate and the autouse seeding fixture. There are no torch
seeding or model fixtures: this package has no torch dependency. ``live_golden`` is
the live dashboard's session-wide published run (see its docstring).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from live_dashboard.publication import RunRequest, WorkerInfo


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def seeded():
    """Deterministic global seeds for every test.

    Tests that care about specific streams should still construct their own
    ``np.random.default_rng(seed)`` — this fixture only pins the legacy
    global state so incidental randomness is reproducible.
    """
    random.seed(42)
    np.random.seed(42)
    yield


def _load(name: str) -> ModuleType:
    """Load the plain helper module ``tests/<name>.py`` once (``tests`` is not a package).

    The module is registered before it executes, so every test module that loads it
    shares one module object (one ``FakeClock`` class, one ``WORKER``).
    """
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class GoldenRun:
    """The session's published ``small`` run (3 tasks of 2 worlds), committed and read-only.

    Tests copy ``root`` with ``copy_store`` and start later writer attempts at
    ``clock_after(end_time)``, so those attempts sort after the golden one.
    """

    root: Path  # the committed store tree
    request: RunRequest
    worker: WorkerInfo
    attempt_id: str
    end_time: datetime  # the run clock's time after the run
    diagnostics: dict[int, dict[str, Any]]  # generator diagnostics (incl. timing) by seed


@pytest.fixture(scope="session")
def live_golden(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """Publish one real run through a Volume-like store, once per session.

    Generation dominates the cost of writer and reader scenarios; sharing one immutable
    run keeps each test fast while still exercising real payloads. The imports stay
    inside so collecting unrelated tests never loads the dashboard or the generator.
    """
    from live_dashboard.deployment import REVISION_ENV
    from live_dashboard.generation import run_generation
    from live_dashboard.publication import RunRequest
    from pymc_generator.data_generator import DataGenerator

    support = _load("_live_dashboard_support")
    store = support.VolumeLikeStore(tmp_path_factory.mktemp("live-golden"))
    request = RunRequest(label="golden", preset="small", worlds=6, worlds_per_task=2, seed=1)
    clock = support.FakeClock()
    diagnostics: dict[int, dict[str, Any]] = {}
    generate = DataGenerator.generate

    def spy(self, n_tasks=None, seed=None, validate=True):
        corpus = generate(self, n_tasks=n_tasks, seed=seed, validate=validate)
        diagnostics[seed] = corpus["diagnostics"]
        return corpus

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(REVISION_ENV, raising=False)
        patch.setattr(DataGenerator, "generate", spy)
        run_generation(
            store, request, worker=support.WORKER, clock=clock, attempt_id=support.GOLDEN_ATTEMPT
        )
    support.make_read_only(store.committed)
    return GoldenRun(
        root=store.committed,
        request=request,
        worker=support.WORKER,
        attempt_id=support.GOLDEN_ATTEMPT,
        end_time=clock.now,
        diagnostics=diagnostics,
    )
