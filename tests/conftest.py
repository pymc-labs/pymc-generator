"""Shared pytest configuration.

Adapted from structural-pfn's conftest: keeps the --runslow gate and the
autouse seeding fixture, drops the torch seeding and model fixtures (this
package has no torch dependency).
"""

from __future__ import annotations

import random

import numpy as np
import pytest


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
