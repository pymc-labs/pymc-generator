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

    LOOKUPS = []  # Modal objects looked up at call time (not at import), in order

    class _Volume(_Chain):
        pass

    class Volume:
        @staticmethod
        def from_name(*args, **kwargs):
            LOOKUPS.append(("Volume", args))
            return _Volume()

    class Secret:
        @staticmethod
        def from_name(*args, **kwargs):
            return object()

    class Function:
        @staticmethod
        def from_name(*args, **kwargs):
            LOOKUPS.append(("Function", args))
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
    root = _container_root(tmp_path)
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


def test_launcher_arguments_bind_to_the_worker_signature_by_name(stub_dir):
    # A reordered generate_run parameter would only fail on Modal (or silently swap values).
    code = PROBE.format(
        path=[str(stub_dir), str(REPO), str(REPO / "live_dashboard")],
        module="modal_app",
        file=str(REPO / "live_dashboard" / "modal_app.py"),
    ) + textwrap.dedent(
        """
        import inspect
        launch_spec = importlib.util.spec_from_file_location("launch", {launch!r})
        launch = importlib.util.module_from_spec(launch_spec)
        launch_spec.loader.exec_module(launch)
        request = launch.RunRequest("demo", "small", 6, 2, 7)
        bound = inspect.signature(module.generate_run).bind(*launch.spawn_arguments(request))
        assert dict(bound.arguments) == {{
            "label": "demo", "preset": "small", "worlds": 6, "worlds_per_task": 2,
            "seed": 7, "expected_run_id": request.run_id,
        }}, bound.arguments
        """
    ).format(launch=str(REPO / "live_dashboard" / "launch.py"))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=REPO,
        env={"STUB_MODAL_LOCAL": "1", "STUB_MODAL_ENVIRONMENT": "pfn", "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def _container_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "live_dashboard").symlink_to(REPO / "live_dashboard", target_is_directory=True)
    (root / "modal_app.py").write_text((REPO / "live_dashboard" / "modal_app.py").read_text())
    return root


def test_worker_entry_records_the_kill_time_and_refuses_bad_requests(stub_dir, tmp_path):
    root = _container_root(tmp_path)
    code = PROBE.format(
        path=[str(stub_dir), str(root)], module="modal_app", file=str(root / "modal_app.py")
    ) + textwrap.dedent(
        """
        from datetime import timedelta
        import live_dashboard.generation as generation
        from live_dashboard.deployment import GENERATION_TIMEOUT_S
        from live_dashboard.publication import RunRequest, utc_now
        calls = []
        def record(store, request, **kwargs):
            calls.append((request, kwargs))
            return {"status": "finished"}
        generation.run_generation_or_refuse = record
        before = utc_now()
        result = module.generate_run("demo", "small", 6, 2, 7, "demo-0123456789ab")
        after = utc_now()
        assert result == {"status": "finished"}, result
        ((request, kwargs),) = calls
        limit = timedelta(seconds=GENERATION_TIMEOUT_S)
        assert before + limit <= kwargs["deadline"] <= after + limit + timedelta(milliseconds=1)
        assert request == RunRequest("demo", "small", 6, 2, 7)
        assert kwargs["expected_run_id"] == "demo-0123456789ab"
        refused = module.generate_run("demo", "small", 5, 2, 7, None)
        assert refused["status"] == "refused" and len(calls) == 1, refused
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=tmp_path,
        env={"STUB_MODAL_LOCAL": "0", "STUB_MODAL_ENVIRONMENT": "", "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_refuses_an_unknown_preset_before_touching_modal(stub_dir):
    code = PROBE.format(
        path=[str(stub_dir), str(REPO), str(REPO / "live_dashboard")],
        module="launch",
        file=str(REPO / "live_dashboard" / "launch.py"),
    ) + textwrap.dedent(
        """
        import modal
        try:
            module.main(label="demo", preset="nope", worlds=6, worlds_per_task=2)
        except SystemExit as refusal:
            assert "unknown preset 'nope'" in str(refusal), refusal
        else:
            raise AssertionError("the launcher accepted an unknown preset")
        assert modal.LOOKUPS == [], modal.LOOKUPS
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=REPO,
        env={"STUB_MODAL_LOCAL": "1", "STUB_MODAL_ENVIRONMENT": "pfn", "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
