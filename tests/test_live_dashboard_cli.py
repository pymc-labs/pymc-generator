"""``python -m live_dashboard hash-password``: the file that becomes the Modal Secret.

The file must never hold the password, must not be readable by other users, must not
replace an existing credentials file, and must be accepted by the dashboard's own
settings parser (otherwise the deployed app refuses to start).
"""

from __future__ import annotations

import io
import json
import stat

import pytest

from live_dashboard.__main__ import main
from live_dashboard.web import settings_from_env

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ("DASHBOARD_USERNAME", "DASHBOARD_PASSWORD_HASH", "DASHBOARD_POLL_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def _hash(monkeypatch, output, password=PASSWORD, username="demo"):
    monkeypatch.setattr("sys.stdin", io.StringIO(password + "\n"))
    main(["hash-password", "--username", username, "--output", str(output), "--password-stdin"])


def test_credentials_file_is_private_hash_only_and_accepted_by_the_app(monkeypatch, tmp_path):
    output = tmp_path / "auth.json"
    _hash(monkeypatch, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    text = output.read_text()
    assert PASSWORD not in text
    settings = settings_from_env(json.loads(text))
    assert settings.credentials.verify("demo", PASSWORD)
    assert not settings.credentials.verify("demo", PASSWORD + "x")


def test_an_existing_credentials_file_is_never_replaced(monkeypatch, tmp_path):
    output = tmp_path / "auth.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        _hash(monkeypatch, output)
    assert output.read_text() == "keep"


@pytest.mark.parametrize(("username", "password"), [("demo", "short"), ("de:mo", PASSWORD)])
def test_credentials_the_dashboard_would_refuse_are_not_written(
    monkeypatch, tmp_path, username, password
):
    output = tmp_path / "auth.json"
    with pytest.raises(ValueError):
        _hash(monkeypatch, output, password=password, username=username)
    assert not output.exists()
