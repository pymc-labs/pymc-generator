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

    class App:  # keeps what it deploys, like Modal's registry of an app's functions
        def __init__(self, name=None, **kwargs):
            self.name, self.registered = name, {}

        def function(self, *args, **kwargs):
            def register(function):
                self.registered[function.__name__] = function
                return function
            return register

        local_entrypoint = staticmethod(_decorator)

    concurrent = _decorator

    def asgi_app(*args, **kwargs):
        def web_endpoint(function):
            function.web_endpoint = True
            return function
        return web_endpoint

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
    smi = tmp_path / "nvidia-smi"
    smi.write_text(
        '#!/bin/sh\n[ "$*" = "--query-gpu=name --format=csv,noheader" ] || exit 2\n'
        "echo 'NVIDIA A100-SXM4-40GB'\n"
    )
    smi.chmod(0o755)
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
        assert kwargs["worker"].gpu_name == "NVIDIA A100-SXM4-40GB", kwargs["worker"]
        import os, pathlib
        smi = pathlib.Path(os.environ["PATH"]) / "nvidia-smi"
        # No driver: nvidia-smi explains on stdout and exits 9; still generates, without a name.
        smi.write_text('#!/bin/sh\\necho "NVIDIA-SMI has failed: no driver"\\nexit 9\\n')
        module.generate_run("demo", "small", 6, 2, 7, None)
        assert calls[-1][1]["worker"].gpu_name is None, calls[-1][1]["worker"]
        smi.unlink()  # no nvidia-smi at all
        module.generate_run("demo", "small", 6, 2, 7, None)
        assert calls[-1][1]["worker"].gpu_name is None, calls[-1][1]["worker"]
        del calls[1:]
        refused = module.generate_run("demo", "small", 5, 2, 7, None)
        assert refused["status"] == "refused" and len(calls) == 1, refused
        # A worker timeout must not reach the launcher as the builtin TimeoutError, which
        # FunctionCall.get raises while a call is still running.
        def time_out(store, request, **kwargs):
            import errno
            raise OSError(errno.ETIMEDOUT, "volume commit timed out")  # builds a TimeoutError
        generation.run_generation_or_refuse = time_out
        try:
            module.generate_run("demo", "small", 6, 2, 7, None)
        except TimeoutError:
            raise AssertionError("a worker timeout escaped as TimeoutError")
        except RuntimeError as error:
            assert "timed out" in str(error), error
            assert isinstance(error.__cause__, TimeoutError), error.__cause__
        else:
            raise AssertionError("the worker timeout was swallowed")
        # The worker's store publishes through the Volume (commit) and sees it (reload).
        hooks = []
        module.volume.commit = lambda: hooks.append("commit")
        module.volume.reload = lambda: hooks.append("reload")
        def touches_the_store(store, request, **kwargs):
            store.commit()
            store.refresh()
            return {"status": "finished"}
        generation.run_generation_or_refuse = touches_the_store
        module.generate_run("demo", "small", 6, 2, 7, None)
        assert hooks == ["commit", "reload"], hooks
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


LAUNCH_SCENARIOS = textwrap.dedent(
    """
    from types import SimpleNamespace as Entry
    import modal

    RUN = "demo-0123456789ab"

    class Clock:  # replaces launch's `time`: polls and sleeps advance virtual time only
        now = 0.0
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    module.time = clock

    class Volume:  # stage i is what the Volume shows after i polls
        def __init__(self, stages):
            self.stages, self.polls = stages, 0
        def listdir(self, path):
            files = self.stages[min(self.polls, len(self.stages) - 1)]
            if files is None:
                return []
            if path in ("/", "runs"):
                return [Entry(path="runs" if path == "/" else f"runs/{RUN}")]
            return [Entry(path=f"runs/{RUN}/{name}") for name in files]

    class Call:
        object_id = "fc-test"
        def __init__(self, outcomes, volume=None, waits=True):
            self.outcomes, self.volume, self.waits, self.gets = outcomes, volume, waits, 0
        def get(self, timeout=None):
            self.gets += 1
            assert self.gets <= 1000, "the launcher spins"
            if self.volume is not None:
                self.volume.polls += 1
            if self.waits and timeout:
                clock.now += timeout
            outcome = self.outcomes.pop(0) if self.outcomes else TimeoutError()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    def exits_with(text, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except SystemExit as refusal:
            assert text in str(refusal), refusal
        else:
            raise AssertionError(f"no SystemExit containing {text!r}")

    import contextlib, io

    def output_of(function, *args, **kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            function(*args, **kwargs)
        return buffer.getvalue()

    def deploy(call):
        # What `modal deploy` registers from modal_app.py in the CLI's environment, looked
        # up like Modal's Function.from_name: by app name, function name and environment.
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "modal_app", pathlib.Path(module.__file__).with_name("modal_app.py"))
        deployed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployed)
        environment = modal.config.config.get("environment")
        modal.LOOKUPS.clear()  # the deployment's own lookups, not the launcher's
        spawned = []
        def cannot_spawn(*args):
            raise TypeError("a web endpoint cannot be spawned")
        def from_name(app_name, name, *, environment_name=None):
            function = deployed.app.registered.get(name)
            where = (app_name, environment_name or environment)
            if function is None or where != (deployed.app.name, environment):
                raise LookupError(f"no deployed function {name!r} in {where}")
            if getattr(function, "web_endpoint", False):
                return Entry(get_web_url=lambda: "https://dash", spawn=cannot_spawn)
            return Entry(get_web_url=lambda: "", spawn=lambda *args: spawned.append(args) or call)
        modal.Function.from_name = staticmethod(from_name)
        return spawned

    case = __CASE__
    if case == "manifest-after-staging":
        # Poll 1 shows only a staging file (not a manifest); poll 2 shows run.json.
        volume = Volume([None, [".run.json.ab12.partial"], ["run.json"]])
        call = Call([], volume)
        module._await_start(call, volume, RUN)
        assert volume.polls == 2, volume.polls
    elif case == "refused-new-run":
        exits_with("worker refused the run: r", module._await_start,
                   Call([{"status": "refused", "reason": "r"}]), Volume([None]), RUN)
    elif case in ("gives-up-after-the-window", "gives-up-after-the-window-fast-get"):
        # A get() that returns at once must still be paced: no spin, same window.
        call = Call([], Volume([None]), waits=case == "gives-up-after-the-window")
        out = output_of(module._await_start, call, call.volume, RUN)
        assert call.gets == module.START_WAIT_S // module._POLL_S, call.gets
        assert clock.now == module.START_WAIT_S, clock.now
        assert f"has not started run {RUN} yet" in out, out
    elif case == "worker-error-propagates":
        call = Call([RuntimeError("TimeoutError: [Errno 60] timed out")], Volume([None]))
        try:
            module._await_start(call, call.volume, RUN)
        except RuntimeError as error:
            assert "Errno 60" in str(error), error
        else:
            raise AssertionError("a worker failure was taken for a running call")
        assert call.gets == 1, call.gets
    elif case == "finished-before-manifest":
        call = Call([{"status": "finished", "run_id": RUN}], Volume([None]))
        out = output_of(module._await_start, call, call.volume, RUN)
        assert call.gets == 1 and "'finished'" in out and "not started" not in out, out
    elif case == "existing-run-needs-resume":
        run_id = module.RunRequest("demo", "small", 6, 2, 7).run_id
        module._published_runs = lambda volume: {run_id}
        exits_with("already exists; pass --resume", module.main,
                   label="demo", preset="small", worlds=6, worlds_per_task=2, seed=7)
        assert [kind for kind, _ in modal.LOOKUPS] == ["Volume"], modal.LOOKUPS
    elif case == "resume-of-missing-run":
        module._published_runs = lambda volume: set()
        exits_with("does not exist", module.main, label="demo", preset="small", worlds=6,
                   worlds_per_task=2, seed=7, resume=True)
        assert [kind for kind, _ in modal.LOOKUPS] == ["Volume"], modal.LOOKUPS
    elif case == "resume-does-not-wait":
        run_id = module.RunRequest("demo", "small", 6, 2, 7).run_id
        module._published_runs = lambda volume: {run_id}
        spawned = deploy(Call([]))
        waited = []
        module._await_start = lambda *args: waited.append(args)
        out = output_of(module.main, label="demo", preset="small", worlds=6, worlds_per_task=2,
                        seed=7, resume=True)
        assert spawned == [module.spawn_arguments(module.RunRequest("demo", "small", 6, 2, 7))]
        assert f"dashboard: https://dash/?run={run_id}" in out, out
        assert waited == [], waited
    elif case == "demo-bounds":
        from live_dashboard.deployment import DEMO_MAX_WORLDS
        exits_with(f"invalid request: at most {DEMO_MAX_WORLDS} worlds", module.main,
                   label="demo", worlds=DEMO_MAX_WORLDS + 2, worlds_per_task=2)
        assert modal.LOOKUPS == [], modal.LOOKUPS
    elif case == "invalid-request":
        exits_with("invalid request:", module.main, label="demo", worlds=5, worlds_per_task=2)
        assert modal.LOOKUPS == [], modal.LOOKUPS
    elif case == "new-run-waits-for-start":
        module._published_runs = lambda volume: set()
        spawned = deploy(Call([]))
        waited = []
        module._await_start = lambda call, volume, run_id: waited.append(run_id)
        out = output_of(module.main, label="demo", preset="small", worlds=6, worlds_per_task=2,
                        seed=7)
        request = module.RunRequest("demo", "small", 6, 2, 7)
        assert spawned == [module.spawn_arguments(request)], spawned
        assert f"dashboard: https://dash/?run={request.run_id}" in out, out
        assert waited == [request.run_id], waited
    elif case == "wait-reports-refusal":
        module._published_runs = lambda volume: set()
        refusal = Call([{"status": "refused", "reason": "code_sha256 differs"}])
        deploy(refusal)
        exits_with("worker refused the run: code_sha256 differs", module.main,
                   label="demo", preset="small", worlds=6, worlds_per_task=2, wait=True)
    else:
        raise AssertionError(case)
    """
)


@pytest.mark.parametrize(
    "case",
    [
        "manifest-after-staging",
        "refused-new-run",
        "invalid-request",
        "new-run-waits-for-start",
        "wait-reports-refusal",
        "gives-up-after-the-window",
        "gives-up-after-the-window-fast-get",
        "worker-error-propagates",
        "finished-before-manifest",
        "existing-run-needs-resume",
        "resume-of-missing-run",
        "resume-does-not-wait",
        "demo-bounds",
    ],
)
def test_launcher_waits_for_the_run_or_reports_why_not(stub_dir, case):
    code = PROBE.format(
        path=[str(stub_dir), str(REPO), str(REPO / "live_dashboard")],
        module="launch",
        file=str(REPO / "live_dashboard" / "launch.py"),
    ) + LAUNCH_SCENARIOS.replace("__CASE__", repr(case))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=REPO,
        env={"STUB_MODAL_LOCAL": "1", "STUB_MODAL_ENVIRONMENT": "pfn", "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_entry_reloads_the_volume_and_uses_the_configured_intervals(stub_dir, tmp_path):
    from live_dashboard.auth import hash_password

    root = _container_root(tmp_path)
    code = PROBE.format(
        path=[str(stub_dir), str(root)], module="modal_app", file=str(root / "modal_app.py")
    ) + textwrap.dedent(
        """
        import live_dashboard.web as web
        reloads = []
        module.dashboard_volume.reload = lambda: reloads.append(1)
        module.MOUNT = "publications"  # relative to the test's own working directory
        created = []
        web.create_app = lambda collector, **kwargs: created.append((collector, kwargs))
        module.dashboard()
        ((collector, kwargs),) = created
        assert (kwargs["poll_seconds"], kwargs["active_poll_seconds"]) == (30.0, 3.0), kwargs
        assert kwargs["credentials"].verify("demo", "correct horse battery")
        collector.refresh()
        assert reloads == [1], reloads
        """
    )
    environment = {
        "STUB_MODAL_LOCAL": "0",
        "STUB_MODAL_ENVIRONMENT": "",
        "PATH": str(tmp_path),
        "DASHBOARD_USERNAME": "demo",
        "DASHBOARD_PASSWORD_HASH": hash_password("correct horse battery"),
        "DASHBOARD_POLL_SECONDS": "30",
    }
    # -I without -S: the dashboard imports the web stack from this environment.
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
