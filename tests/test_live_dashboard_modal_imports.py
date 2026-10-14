"""The Modal entry points import the way the Modal CLI and containers import them.

The Modal CLI imports ``modal_app.py``/``launch.py`` with its own interpreter (standard
library + modal, no numpy or pymc_generator), loading the file as a top-level module with
the working directory on ``sys.path``; containers re-import ``modal_app`` without git or
the repository checkout. A heavy module-scope import or an unguarded local path breaks
``modal deploy``/``modal run`` only, so these subprocesses run with ``python -I -S`` and
a minimal stand-in ``modal`` package.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

STUB = textwrap.dedent(
    """
    import os

    class _Chain:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class Image:
        @staticmethod
        def debian_slim(**kwargs):
            return _Chain()

    class _Volume(_Chain):
        pass

    class Volume:
        @staticmethod
        def from_name(*args, **kwargs):
            return _Volume()

    class Secret:
        @staticmethod
        def from_name(*args, **kwargs):
            return object()

    class Function:
        @staticmethod
        def from_name(*args, **kwargs):
            return _Chain()

    class Retries:
        def __init__(self, **kwargs):
            pass

    def _decorator(*args, **kwargs):
        return lambda function: function

    class App:
        def __init__(self, *args, **kwargs):
            pass

        function = staticmethod(_decorator)
        local_entrypoint = staticmethod(_decorator)

    concurrent = _decorator
    asgi_app = _decorator

    def is_local():
        return os.environ["STUB_MODAL_LOCAL"] == "1"
    """
)
STUB_CONFIG = textwrap.dedent(
    """
    import os

    class _Config:
        def get(self, key):
            return os.environ.get("STUB_MODAL_ENVIRONMENT") if key == "environment" else None

    config = _Config()
    """
)
PROBE = textwrap.dedent(
    """
    import importlib.util, sys
    sys.path[:0] = {path!r}
    spec = importlib.util.spec_from_file_location({module!r}, {file!r})
    module = importlib.util.module_from_spec(spec)
    sys.modules[{module!r}] = module
    spec.loader.exec_module(module)
    heavy = sorted({{"numpy", "pymc_generator", "fastapi", "plotly"}} & set(sys.modules))
    assert not heavy, heavy
    """
)


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("stub")
    (root / "modal").mkdir()
    (root / "modal" / "__init__.py").write_text(STUB + "\nfrom modal import config\n")
    (root / "modal" / "config.py").write_text(STUB_CONFIG)
    return root


def _probe(stub_dir, module, file, *, local, environment, cwd, path, git=True):
    env = {"STUB_MODAL_LOCAL": "1" if local else "0", "STUB_MODAL_ENVIRONMENT": environment}
    env["PATH"] = os.environ["PATH"] if git else str(cwd)
    code = PROBE.format(path=[str(stub_dir), *map(str, path)], module=module, file=str(file))
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("module", ["modal_app", "launch"])
def test_cli_imports_entry_points_with_stdlib_and_modal_only(stub_dir, module):
    # `modal deploy/run -e pfn live_dashboard/<file>.py` from the repository root.
    result = _probe(
        stub_dir,
        module,
        REPO / "live_dashboard" / f"{module}.py",
        local=True,
        environment="pfn",
        cwd=REPO,
        path=[REPO, REPO / "live_dashboard"],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["modal_app", "launch"])
@pytest.mark.parametrize("environment", ["", "main"])
def test_cli_refuses_environments_other_than_pfn(stub_dir, module, environment):
    result = _probe(
        stub_dir,
        module,
        REPO / "live_dashboard" / f"{module}.py",
        local=True,
        environment=environment,
        cwd=REPO,
        path=[REPO, REPO / "live_dashboard"],
    )
    assert result.returncode != 0
    assert "Modal environment 'pfn'" in result.stderr


def test_container_reimports_the_app_without_git_or_the_checkout(stub_dir, tmp_path):
    # Inside a container: /root holds modal_app.py and the packages; no git, no repository.
    root = tmp_path / "root"
    root.mkdir()
    (root / "live_dashboard").symlink_to(REPO / "live_dashboard", target_is_directory=True)
    (root / "modal_app.py").write_text((REPO / "live_dashboard" / "modal_app.py").read_text())
    result = _probe(
        stub_dir,
        "modal_app",
        root / "modal_app.py",
        local=False,
        environment="",
        cwd=tmp_path,
        path=[root],
        git=False,
    )
    assert result.returncode == 0, result.stderr
