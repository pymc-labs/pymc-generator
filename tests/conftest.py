"""Shared pytest configuration.

Keeps the --runslow gate and the autouse seeding fixture. There are no torch
seeding or model fixtures: this package has no torch dependency. ``live_golden`` is
the live dashboard's session-wide published run (see its docstring).
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType, ModuleType
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


def _read_only(value: Any) -> Any:
    """A deep read-only copy of JSON-like data, so no test can alter what others compare."""
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value


def _tree_digest(root: Path) -> str:
    """Digest of every name and file content under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        if path.is_file() and not path.is_symlink():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


# What a reader saw: (run state, task statuses, newest attempt's current_task_index).
ReaderView = tuple[str, tuple[str, ...], int | None]


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
    diagnostics: Mapping[int, Mapping[str, Any]]  # generator diagnostics (incl. timing) by seed
    # A reader's view before each generation call (task order), then after the run.
    reader_views: tuple[ReaderView, ...]


@pytest.fixture(scope="session")
def live_golden(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GoldenRun]:
    """Publish one real run through a Volume-like store, once per session.

    Generation dominates the cost of writer and reader scenarios; sharing one immutable
    run keeps each test fast while still exercising real payloads. A collector follows
    the run through a separate reader view, as the dashboard follows a live worker. The
    imports stay inside so collecting unrelated tests never loads the dashboard or the
    generator. Teardown fails if any test changed the shared tree (``make_read_only``
    cannot stop a test running as root).
    """
    from live_dashboard.collection import Collector
    from live_dashboard.deployment import REVISION_ENV
    from live_dashboard.generation import run_generation
    from live_dashboard.publication import RunRequest
    from pymc_generator.data_generator import DataGenerator

    support = _load("_live_dashboard_support")
    store = support.VolumeLikeStore(tmp_path_factory.mktemp("live-golden"))
    request = RunRequest(label="golden", preset="small", worlds=6, worlds_per_task=2, seed=1)
    clock = support.FakeClock()
    # The reader has its own frozen clock: its views never depend on the current date, and
    # reading does not advance the writer's clock (golden timings stay fixed).
    collector = Collector(
        store.reader(tmp_path_factory.mktemp("live-golden-reader")),
        clock=support.FakeClock(step=0.0),
    )
    reader_views: list[ReaderView] = []
    diagnostics: dict[int, dict[str, Any]] = {}
    generate = DataGenerator.generate

    def observe() -> None:
        collector.refresh()
        snapshot = collector.snapshot
        assert snapshot.last_error is None, snapshot.last_error
        run = snapshot.run_index[request.run_id]
        statuses = tuple(task.status for task in run.tasks)
        reader_views.append((run.state, statuses, run.attempts[0].current_task_index))

    def spy(self, n_tasks=None, seed=None, validate=True, *, defer_cell_split=False):
        observe()
        corpus = generate(
            self, n_tasks=n_tasks, seed=seed, validate=validate, defer_cell_split=defer_cell_split
        )
        diagnostics[seed] = corpus["diagnostics"]
        return corpus

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(REVISION_ENV, raising=False)
        patch.setattr(DataGenerator, "generate", spy)
        run_generation(
            store, request, worker=support.WORKER, clock=clock, attempt_id=support.GOLDEN_ATTEMPT
        )
    observe()
    support.make_read_only(store.committed)
    digest = _tree_digest(store.committed)
    yield GoldenRun(
        root=store.committed,
        request=request,
        worker=support.WORKER,
        attempt_id=support.GOLDEN_ATTEMPT,
        end_time=clock.now,
        diagnostics=_read_only(diagnostics),
        reader_views=tuple(reader_views),
    )
    assert _tree_digest(store.committed) == digest, "a test modified the shared golden run"
