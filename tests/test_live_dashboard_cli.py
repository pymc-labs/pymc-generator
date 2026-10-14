"""``python -m live_dashboard``: credentials files and local generation limits.

``hash-password`` writes the file that becomes the Modal Secret: it must never hold the
password, must not be readable by other users, must not replace an existing file, and
must be accepted by the dashboard's own settings parser. ``generate`` must hand the
writer the requested time limit and refuse unusable ones before touching the store.
"""

from __future__ import annotations

import io
import json
import stat
from datetime import timedelta

import pytest

from live_dashboard import generation
from live_dashboard.__main__ import main
from live_dashboard.deployment import DEMO_MAX_WORLDS
from live_dashboard.publication import utc_now
from live_dashboard.web import settings_from_env

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


def _hash(monkeypatch, output, password=PASSWORD, username="demo"):
    monkeypatch.setattr("sys.stdin", io.StringIO(password + "\n"))
    main(["hash-password", "--username", username, "--output", str(output), "--password-stdin"])


def test_credentials_file_is_private_hash_only_and_accepted_by_the_app(monkeypatch, tmp_path):
    output = tmp_path / "new-config-dir" / "auth.json"
    _hash(monkeypatch, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    text = output.read_text()
    assert PASSWORD not in text
    settings = settings_from_env(json.loads(text))
    assert settings.credentials.verify("demo", PASSWORD)
    assert not settings.credentials.verify("demo", PASSWORD + "x")


def test_an_existing_credentials_file_is_never_replaced(monkeypatch, tmp_path):
    output = tmp_path / "auth.json"
    output.write_text("keep")
    with pytest.raises(SystemExit) as exit_info:
        _hash(monkeypatch, output)
    assert exit_info.value.code == 2
    assert output.read_text() == "keep"


@pytest.mark.parametrize(("username", "password"), [("demo", "short"), ("de:mo", PASSWORD)])
def test_credentials_the_dashboard_would_refuse_are_not_written(
    monkeypatch, tmp_path, username, password
):
    output = tmp_path / "auth.json"
    with pytest.raises(SystemExit) as exit_info:
        _hash(monkeypatch, output, password=password, username=username)
    assert exit_info.value.code == 2
    assert not output.exists()


def _generate(tmp_path, *extra):
    main(["generate", "--data-dir", str(tmp_path / "store"), "--label", "cli", *extra])


@pytest.mark.parametrize(
    ("extra", "seconds"),
    [
        ((), 3600),
        (("--max-seconds", "600"), 600),
        (("--max-seconds", "120"), 120),
        (("--max-seconds", "604800"), 604800),
    ],
)
def test_generate_hands_the_writer_its_time_limit(monkeypatch, tmp_path, extra, seconds):
    calls = []
    monkeypatch.setattr(
        generation,
        "run_generation",
        lambda store, request, **kwargs: calls.append(kwargs) or {"status": "finished"},
    )
    before = utc_now()
    _generate(tmp_path, *extra)
    after = utc_now()
    (call,) = calls
    limit = timedelta(seconds=seconds)
    assert before + limit <= call["deadline"] <= after + limit + timedelta(milliseconds=1)


def _writer_must_not_run(*args, **kwargs):
    raise AssertionError("the writer was called")


@pytest.mark.parametrize("value", ["119", "604801", "nan", "inf", "-5", "ten"])
def test_generate_refuses_unusable_time_limits_before_writing(monkeypatch, tmp_path, value):
    monkeypatch.setattr(generation, "run_generation", _writer_must_not_run)
    with pytest.raises(SystemExit) as exit_info:
        _generate(tmp_path, "--max-seconds", value)
    assert exit_info.value.code == 2
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (("--preset", "showcase"), "unknown preset 'showcase'"),
        (("--worlds", "3"), "must be a positive multiple"),
        (("--worlds", str(DEMO_MAX_WORLDS + 2)), f"at most {DEMO_MAX_WORLDS} worlds"),
    ],
)
def test_generate_refuses_bad_requests_with_a_message(monkeypatch, tmp_path, capsys, extra, reason):
    from pymc_generator import DataGenerator

    # The real writer refuses these before generating; never generate if it stops doing so.
    monkeypatch.setattr(DataGenerator, "generate", _writer_must_not_run)
    with pytest.raises(SystemExit) as exit_info:
        _generate(tmp_path, *extra)
    assert exit_info.value.code == 2
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "store").exists()


def test_generate_reports_a_conflicting_run_without_a_traceback(monkeypatch, tmp_path, capsys):
    def conflict(store, request, **kwargs):
        raise generation.RunConflictError("code_sha256 differs")

    monkeypatch.setattr(generation, "run_generation", conflict)
    with pytest.raises(SystemExit) as exit_info:
        _generate(tmp_path)
    assert exit_info.value.code == 2
    assert "error: code_sha256 differs" in capsys.readouterr().err


def test_generation_failures_keep_their_traceback(monkeypatch, tmp_path):
    def library_bug(store, request, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr(generation, "run_generation", library_bug)
    with pytest.raises(ValueError, match="broadcast"):
        _generate(tmp_path)


@pytest.fixture
def served(monkeypatch, tmp_path):
    """``serve`` up to the point it would start the server.

    Returns the store roots it opened, ``create_app``'s keyword arguments and
    ``uvicorn.run``'s keyword arguments.
    """
    import uvicorn

    from live_dashboard import storage, web

    roots, created, started = [], [], []
    monkeypatch.setattr(storage, "DirectoryStore", lambda root, **kwargs: roots.append(root))
    monkeypatch.setattr(web, "create_app", lambda collector, **kwargs: created.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.append(kwargs))
    credentials = tmp_path / "auth.json"
    _hash(monkeypatch, credentials)

    def serve(*extra, credentials_file=credentials):
        main(["serve", "--data-dir", str(tmp_path), "--credentials", str(credentials_file), *extra])
        return roots, created, started

    return serve


def test_serve_follows_the_data_dir_on_loopback_with_the_requested_intervals(served, tmp_path):
    roots, (kwargs,), (server,) = served("--poll-seconds", "30")
    assert roots == [tmp_path]
    assert (kwargs["poll_seconds"], kwargs["active_poll_seconds"]) == (30.0, 3.0)
    assert kwargs["credentials"].verify("demo", PASSWORD)
    # Plain HTTP: only the local machine may reach it unless the operator chooses otherwise.
    assert server == {"host": "127.0.0.1", "port": 8000}


@pytest.mark.parametrize(
    ("extra", "accepted"),
    [
        (("--poll-seconds", "1"), {"poll_seconds": 1.0}),
        (("--poll-seconds", "300"), {"poll_seconds": 300.0}),
        (("--port", "0"), {"port": 0}),
        (("--port", "65535"), {"port": 65535}),
        (("--host", "::1"), {"host": "::1"}),
    ],
)
def test_serve_accepts_the_range_limits(served, extra, accepted):
    _, (kwargs,), (server,) = served(*extra)
    assert accepted.items() <= {**kwargs, **server}.items()


@pytest.mark.parametrize(
    "extra",
    [
        ("--poll-seconds", "0.5"),
        ("--poll-seconds", "301"),
        ("--port", "65536"),
        ("--port", "-1"),
        ("--port", "http"),
    ],
)
def test_serve_refuses_unusable_settings(served, extra):
    with pytest.raises(SystemExit) as exit_info:
        served(*extra)
    assert exit_info.value.code == 2


@pytest.mark.parametrize("content", [None, "[1, 2]", "{not json", '{"DASHBOARD_USERNAME": 5}'])
def test_serve_refuses_unusable_credentials_files(served, tmp_path, capsys, content):
    bad = tmp_path / "bad.json"
    if content is not None:
        bad.write_text(content)
    with pytest.raises(SystemExit) as exit_info:
        served(credentials_file=bad)
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
