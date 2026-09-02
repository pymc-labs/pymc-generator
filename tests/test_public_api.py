"""Public API surface: every advertised symbol resolves, and import stays light."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import prior_generator as pg


def test_all_public_symbols_resolve():
    # guards the lazy __getattr__ map — a stale entry would only fail on access.
    for name in pg.__all__:
        assert getattr(pg, name) is not None, f"{name!r} in __all__ did not resolve"


def test_version_matches_the_installed_distribution():
    # __version__ is hand-maintained in prior_generator/__init__.py while the
    # packaged version lives in pyproject [project].version, and release.yml
    # gates a tag against pyproject alone — so a bumped pyproject with a stale
    # __init__ would ship a wheel that misreports itself.
    assert pg.__version__ == importlib.metadata.version("prior-generator")


def test_import_is_light():
    # importing the package must not pull the heavy modeling / plotting stack.
    code = (
        "import sys, prior_generator; "
        "heavy = [m for m in ('pytensor', 'pymc', 'pymc_marketing', 'pandas', 'matplotlib') "
        "if m in sys.modules]; "
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code])
    assert r.returncode == 0, "import prior_generator pulled a heavy dependency at import time"
