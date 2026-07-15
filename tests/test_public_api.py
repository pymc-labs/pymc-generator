"""Public API surface: every advertised symbol resolves, and import stays light."""

from __future__ import annotations

import subprocess
import sys

import prior_generator as pg


def test_all_public_symbols_resolve():
    # guards the lazy __getattr__ map — a stale entry would only fail on access.
    for name in pg.__all__:
        assert getattr(pg, name) is not None, f"{name!r} in __all__ did not resolve"


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
