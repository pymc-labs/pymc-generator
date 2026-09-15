"""Public API surface: every advertised symbol resolves, and import stays light."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import pytest

import pymc_generator as pg

# `pymc_generator.diagnostics` is a submodule AND `data_diagnostics` is a lazy
# public callable: importing the submodule (explicitly, or as the side effect of
# the lazy __getattr__) binds it as a package attribute. A submodule named after
# a public callable would shadow it, so the callable must stay a function under
# both import orders and across repeated attribute access.
_DIAGNOSTICS_EXPORTS_HOLD = (
    "import inspect; "
    "f = pg.data_diagnostics; "
    "assert inspect.isfunction(f), f'data_diagnostics resolved to {f!r}'; "
    "assert pg.data_diagnostics is f, 'a second access changed data_diagnostics'; "
    "assert f is pg.diagnostics.data_diagnostics, 'export is not the module function'; "
    "assert inspect.isclass(pg.DataDiagnostics), f'DataDiagnostics is {pg.DataDiagnostics!r}'"
)


def test_all_public_symbols_resolve():
    # guards the lazy __getattr__ map — a stale entry would only fail on access.
    for name in pg.__all__:
        assert getattr(pg, name) is not None, f"{name!r} in __all__ did not resolve"


def test_version_matches_the_installed_distribution():
    assert pg.__version__ == importlib.metadata.version("pymc-generator")


@pytest.mark.parametrize(
    "prologue",
    (
        "import pymc_generator as pg",
        "import pymc_generator.diagnostics; import pymc_generator as pg",
    ),
    ids=("package-first", "submodule-first"),
)
def test_diagnostics_exports_hold_under_both_import_orders(prologue):
    # one fresh interpreter per order: binding a submodule as a package
    # attribute is a one-shot import-time effect an imported package would hide.
    r = subprocess.run(
        [sys.executable, "-c", f"{prologue}; {_DIAGNOSTICS_EXPORTS_HOLD}"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_all_advertises_the_diagnostics_entry_points():
    # test_all_public_symbols_resolve only walks __all__, so a lazy map entry
    # that was never advertised there would go unchecked.
    assert {"data_diagnostics", "DataDiagnostics"} <= set(pg.__all__)


def test_import_is_light():
    # importing the package must not pull the heavy modeling / plotting stack.
    code = (
        "import sys, pymc_generator; "
        "heavy = [m for m in ('pytensor', 'pymc', 'pymc_marketing', 'scipy', 'pandas', 'matplotlib') "
        "if m in sys.modules]; "
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code])
    assert r.returncode == 0, "import pymc_generator pulled a heavy dependency at import time"


def test_importing_the_diagnostics_module_stays_light():
    # diagnostics is pure numpy over retained arrays: pandas is TYPE_CHECKING-only
    # there and the frame helpers import it on demand.
    code = (
        "import sys, pymc_generator.diagnostics; "
        "heavy = [m for m in ('pytensor', 'pymc', 'pymc_marketing', 'scipy', 'pandas', 'matplotlib') "
        "if m in sys.modules]; "
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code])
    assert r.returncode == 0, "pymc_generator.diagnostics pulled a heavy dependency at import time"
