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


@pytest.mark.parametrize(("extra", "seconds"), [((), 3600), (("--max-seconds", "600"), 600)])
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


@pytest.mark.parametrize("value", ["119", "604801", "nan", "inf", "-5", "ten"])
def test_generate_refuses_unusable_time_limits_before_writing(tmp_path, value):
    with pytest.raises(SystemExit) as exit_info:
        _generate(tmp_path, "--max-seconds", value)
    assert exit_info.value.code == 2
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "extra", [("--preset", "showcase"), ("--worlds", "3"), ("--worlds", "402")]
)
def test_generate_refuses_bad_requests_with_a_message(tmp_path, capsys, extra):
    with pytest.raises(SystemExit) as exit_info:
        _generate(tmp_path, *extra)
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "store").exists()
