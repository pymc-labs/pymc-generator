"""The web app: Basic auth on every path, JSON API semantics, page assets, settings, poller."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import importlib.util
import json
import math
import re
import sys
import threading
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import plotly
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from live_dashboard import web
from live_dashboard.auth import Credentials, hash_password
from live_dashboard.snapshot import (
    AttemptView,
    CollectionSnapshot,
    ImpactUnit,
    RunSnapshot,
    TaskState,
    WorldMetrics,
)


def _load(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None and spec.loader is not None
        module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
        spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")

USERNAME = "viewer"
# Its Basic token contains "/" and "=" padding, so lax base64 variants can be tested.
PASSWORD = "correct?horse battery"
RUN_ID = "demo-0123456789ab"
ATTEMPT = "20260101T000000Z-0000abcd"
CHALLENGE = 'Basic realm="PyMC Generator live dashboard", charset="UTF-8"'
ENV_NAMES = (
    "DASHBOARD_USERNAME",
    "DASHBOARD_PASSWORD_HASH",
    "DASHBOARD_POLL_SECONDS",
    "LIVE_DASHBOARD_CODE_REVISION",
)


@pytest.fixture(autouse=True)
def _no_dashboard_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def password_hash() -> str:
    """One real PBKDF2 hash at the minimum cost, shared by the module."""
    return hash_password(PASSWORD)


@pytest.fixture
def credentials(password_hash) -> Credentials:
    return Credentials.from_hash(USERNAME, password_hash)


def basic(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")


TOKEN = basic(USERNAME, PASSWORD).removeprefix("Basic ")
AUTH = {"Authorization": f"Basic {TOKEN}"}


# -- snapshots ---------------------------------------------------------------------------


def make_run(
    published: int,
    *,
    state: str = "running",
    attempt_status: str = "running",
    gpu_name: str | None = "NVIDIA A100-SXM4-40GB",
    detail: str | None = None,
    issues: tuple[str, ...] = (),
    share: float = 0.4,
) -> RunSnapshot:
    """A three-task run whose first ``published`` tasks are published (two worlds each);
    a ``detail`` turns the last task into an invalid one carrying that text."""
    tasks = []
    worlds = []
    impacts = []
    for index in range(3):
        task_id = f"task-{index:05d}"
        start = support.EPOCH + timedelta(minutes=index)
        if index < published:
            tasks.append(
                TaskState(
                    task_id=task_id,
                    task_index=index,
                    status="published",
                    target_worlds=2,
                    accepted_worlds=2,
                    started_at=start,
                    finished_at=start + timedelta(seconds=40),
                    generation_seconds=39.5,
                    candidates_evaluated=3,
                    rejected_candidates=1,
                    draw_failures=0,
                    attempt_id=ATTEMPT,
                )
            )
            for row in range(2):
                world_id = f"{RUN_ID}/{task_id}/w{row}"
                worlds.append(WorldMetrics(world_id, task_id, share, 0.35, 0.25))
                impacts.append(ImpactUnit(world_id, task_id, "C1", share / 2))
        elif index == 2 and detail is not None:
            tasks.append(TaskState(task_id, index, "invalid", 2, detail=detail))
        else:
            tasks.append(TaskState(task_id, index, "pending", 2))
    attempt = AttemptView(
        attempt_id=ATTEMPT,
        status=attempt_status,  # type: ignore[arg-type]
        started_at=support.EPOCH,
        updated_at=support.EPOCH + timedelta(minutes=published),
        deadline_at=support.EPOCH + timedelta(minutes=20),
        current_task_index=published,
        error=None,
        gpu_requested="A100",
        gpu_name=gpu_name,
    )
    return RunSnapshot(
        run_id=RUN_ID,
        label="demo",
        preset="small",
        created_at=support.EPOCH,
        state=state,  # type: ignore[arg-type]
        n_tasks=3,
        worlds_per_task=2,
        target_worlds=6,
        seed=1,
        code_revision="abc123",
        tasks=tuple(tasks),
        worlds=tuple(worlds),
        impacts=tuple(impacts),
        attempts=(attempt,),
        issues=issues,
    )


def make_snapshot(
    sequence: int,
    *runs: RunSnapshot,
    refreshed: bool = True,
    last_error: str | None = None,
    issues: tuple[str, ...] = (),
) -> CollectionSnapshot:
    return CollectionSnapshot(
        sequence=sequence,
        refreshed_at=support.EPOCH + timedelta(hours=1) if refreshed else None,
        last_error=last_error,
        runs=runs,
        issues=issues,
    )


class StubCollector:
    """A collector whose refreshes replay queued outcomes: a snapshot replaces the
    current one, an exception is raised, and an empty queue is an idle refresh (same
    content, next sequence) like a real collector over an unchanged store."""

    def __init__(self, *outcomes: CollectionSnapshot | BaseException) -> None:
        self.snapshot = CollectionSnapshot(sequence=0, refreshed_at=None, last_error=None, runs=())
        self.outcomes = list(outcomes)
        self.calls = 0

    def refresh(self) -> None:
        self.calls += 1
        if not self.outcomes:
            self.snapshot = dataclasses.replace(self.snapshot, sequence=self.snapshot.sequence + 1)
            return
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.snapshot = outcome


@contextlib.contextmanager
def serve(
    collector: StubCollector, credentials: Credentials, **options: Any
) -> Iterator[TestClient]:
    app = web.create_app(collector, credentials=credentials, start_poller=False, **options)
    with TestClient(app) as client:
        yield client


def strict_json(response) -> Any:
    def refuse(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    return json.loads(response.text, parse_constant=refuse)


# -- authentication ----------------------------------------------------------------------

PROTECTED = [
    "/",
    "/static/app.js",
    "/static/style.css",
    web.PLOTLY_PATH,
    "/static/missing.js",
    "/api/runs",
    f"/api/runs/{RUN_ID}",
    "/api/runs/other-000000000000",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/no/such/page",
]


@pytest.mark.parametrize("path", PROTECTED)
def test_every_path_requires_credentials(credentials, path):
    with serve(StubCollector(make_snapshot(1, make_run(1))), credentials) as client:
        for method in ("GET", "POST"):
            response = client.request(method, path)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == CHALLENGE
            assert response.headers["cache-control"] == "no-store"
            assert response.content == b"authentication required\n"


def test_websockets_are_closed_without_credentials(credentials):
    with serve(StubCollector(), credentials) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws"):
                pass
    assert closed.value.code == 1008


@pytest.mark.parametrize(
    ("username", "password"),
    [
        (USERNAME, "correct?horse batter"),
        (USERNAME, PASSWORD + " "),
        (USERNAME, ""),
        ("Viewer", PASSWORD),
        ("someone", PASSWORD),
        ("", ""),
    ],
)
def test_wrong_credentials_are_rejected(credentials, username, password):
    with serve(StubCollector(), credentials) as client:
        response = client.get("/", headers={"Authorization": basic(username, password)})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == CHALLENGE


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"Basic", id="scheme-only"),
        pytest.param(b"Basic ", id="empty-token"),
        pytest.param(TOKEN.encode(), id="no-scheme"),
        pytest.param(b"Bearer " + TOKEN.encode(), id="other-scheme"),
        pytest.param(b"Basic " + TOKEN.rstrip("=").encode(), id="missing-padding"),
        pytest.param(b"Basic " + TOKEN.replace("/", "_").encode(), id="url-safe-alphabet"),
        pytest.param(b"Basic " + TOKEN[:8].encode() + b" " + TOKEN[8:].encode(), id="inner-space"),
        pytest.param(b"Basic " + TOKEN.encode() + b"AAAA", id="data-after-padding"),
        pytest.param(b"Basic " + base64.b64encode(USERNAME.encode()), id="no-colon"),
        pytest.param(
            b"Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}é".encode()), id="non-ascii-pair"
        ),
        pytest.param(b"Basic \xc3\xa9" + TOKEN.encode(), id="non-ascii-header"),
    ],
)
def test_malformed_authorization_headers_are_rejected(credentials, header):
    with serve(StubCollector(), credentials) as client:
        response = client.get("/", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == CHALLENGE


def test_duplicate_authorization_headers_are_rejected(credentials):
    with serve(StubCollector(), credentials) as client:
        response = client.get("/", headers=[("Authorization", AUTH["Authorization"])] * 2)
    assert response.status_code == 401


@pytest.mark.parametrize("scheme", ["basic", "BASIC", "Basic"])
def test_scheme_is_case_insensitive(credentials, scheme):
    with serve(StubCollector(), credentials) as client:
        response = client.get("/", headers={"Authorization": f"{scheme} {TOKEN}"})
    assert response.status_code == 200


def test_wrong_credentials_after_a_success_are_still_rejected(credentials):
    with serve(StubCollector(), credentials) as client:
        assert client.get("/", headers=AUTH).status_code == 200
        wrong = client.get(
            "/", headers={"Authorization": basic(USERNAME, "correct?horse batteryX")}
        )
        assert wrong.status_code == 401
        assert client.get("/", headers=AUTH).status_code == 200


# -- page and assets ---------------------------------------------------------------------


def test_page_references_assets_that_load(credentials):
    with serve(StubCollector(), credentials) as client:
        page = client.get("/", headers=AUTH)
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert page.headers["cache-control"] == "no-cache, private"
        referenced = set(re.findall(r'(?:src|href)="([^"]+)"', page.text))
        plotly_path = f"/static/plotly-{plotly.__version__}.min.js"
        assert referenced == {"/static/style.css", "/static/app.js", plotly_path}
        for path in referenced:
            asset = client.get(path, headers=AUTH)
            assert asset.status_code == 200, path
            assert asset.content, path
        assert client.get("/static/app.js", headers=AUTH).headers["cache-control"] == (
            "no-cache, private"
        )
        assert client.get("/static/style.css", headers=AUTH).headers["cache-control"] == (
            "no-cache, private"
        )
        bundle = client.get(plotly_path, headers=AUTH)
        assert bundle.headers["cache-control"] == "private, max-age=31536000, immutable"
        assert bundle.headers["content-type"].startswith("text/javascript")
        assert "Plotly" in bundle.text[:2000]


def test_unknown_static_path_is_not_found_after_authentication(credentials):
    with serve(StubCollector(make_snapshot(1)), credentials) as client:
        assert client.get("/static/missing.js", headers=AUTH).status_code == 404
        assert client.get("/docs", headers=AUTH).status_code == 404
        assert client.get("/openapi.json", headers=AUTH).status_code == 404


# -- JSON API ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "poll_seconds"), [("running", 3.0), ("complete", 10.0)], ids=["generating", "idle"]
)
def test_run_list_reports_refresh_state_and_poll_interval(credentials, state, poll_seconds):
    snapshot = make_snapshot(1, make_run(1, state=state), issues=("run.json of x is unsupported",))
    with serve(StubCollector(snapshot), credentials) as client:
        response = client.get("/api/runs", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = strict_json(response)
    assert body["refreshed_at"] == "2026-01-01T01:00:00.000Z"
    assert body["last_error"] is None
    assert body["poll_seconds"] == poll_seconds
    assert body["issues"] == ["run.json of x is unsupported"]
    assert [(run["run_id"], run["state"], run["published_worlds"]) for run in body["runs"]] == [
        (RUN_ID, state, 2)
    ]


def test_unknown_run_is_not_found_after_authentication(credentials):
    with serve(StubCollector(make_snapshot(1, make_run(1))), credentials) as client:
        response = client.get("/api/runs/other-000000000000", headers=AUTH)
    assert response.status_code == 404
    assert strict_json(response) == {"detail": "unknown run"}


def test_run_etag_changes_when_a_task_is_published(credentials):
    collector = StubCollector(make_snapshot(1, make_run(1)))
    with serve(collector, credentials) as client:
        first = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-cache, private"
        assert re.fullmatch(r'"[0-9a-f]{24}"', first.headers["etag"])
        assert strict_json(first)["cards"][0]["value"] == 2

        collector.outcomes.append(make_snapshot(2, make_run(2)))
        collector.refresh()
        second = client.get(
            f"/api/runs/{RUN_ID}", headers={**AUTH, "If-None-Match": first.headers["etag"]}
        )
    assert second.status_code == 200
    assert second.headers["etag"] != first.headers["etag"]
    assert strict_json(second)["cards"][0]["value"] == 4


@pytest.mark.parametrize(
    "if_none_match",
    [
        pytest.param(lambda etag: etag, id="strong"),
        pytest.param(lambda etag: f"W/{etag}", id="weak"),
        pytest.param(lambda etag: f'"0123", {etag}', id="list"),
        pytest.param(lambda etag: f'W/"0123",W/{etag}', id="weak-list"),
        pytest.param(lambda etag: "*", id="any"),
    ],
)
def test_idle_refresh_keeps_the_etag_and_revalidation_is_304(credentials, if_none_match):
    collector = StubCollector(make_snapshot(1, make_run(1)))
    with serve(collector, credentials) as client:
        first = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
        etag = first.headers["etag"]
        collector.refresh()  # idle: next sequence, same content
        assert collector.snapshot.sequence == 2
        revalidated = client.get(
            f"/api/runs/{RUN_ID}", headers={**AUTH, "If-None-Match": if_none_match(etag)}
        )
        again = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
    assert revalidated.status_code == 304
    assert revalidated.content == b""
    assert revalidated.headers["etag"] == etag
    assert revalidated.headers["cache-control"] == "no-cache, private"
    assert again.status_code == 200
    assert again.content == first.content
    assert again.headers["etag"] == etag


def test_a_stale_etag_gets_the_full_body(credentials):
    with serve(StubCollector(make_snapshot(1, make_run(1))), credentials) as client:
        response = client.get(
            f"/api/runs/{RUN_ID}", headers={**AUTH, "If-None-Match": '"000000000000000000000000"'}
        )
    assert response.status_code == 200
    assert strict_json(response)["run"]["run_id"] == RUN_ID


def test_run_bodies_are_identical_across_app_instances(credentials):
    bodies = []
    for _ in range(2):
        with serve(StubCollector(make_snapshot(7, make_run(2))), credentials) as client:
            response = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
            bodies.append((response.content, response.headers["etag"]))
    assert bodies[0] == bodies[1]


def test_lost_attempt_changes_the_etag(credentials):
    collector = StubCollector(make_snapshot(1, make_run(1)))
    with serve(collector, credentials) as client:
        running = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
        collector.outcomes.append(
            make_snapshot(2, make_run(1, state="lost", attempt_status="lost"))
        )
        collector.refresh()
        lost = client.get(
            f"/api/runs/{RUN_ID}", headers={**AUTH, "If-None-Match": running.headers["etag"]}
        )
    assert lost.status_code == 200
    assert lost.headers["etag"] != running.headers["etag"]
    body = strict_json(lost)
    assert body["run"]["state"] == "lost"
    assert "latest attempt: lost" in body["status"]["line"]


def test_first_refresh_failure_serves_503_until_a_refresh_succeeds(credentials):
    failed = make_snapshot(1, refreshed=False, last_error="OSError: volume unavailable")
    collector = StubCollector(failed)
    with serve(collector, credentials) as client:
        listing = client.get("/api/runs", headers=AUTH)
        assert listing.status_code == 503
        assert listing.headers["retry-after"] == "2"
        assert listing.headers["cache-control"] == "no-store"
        assert strict_json(listing) == {
            "refreshed_at": None,
            "last_error": "OSError: volume unavailable",
            "poll_seconds": 10.0,
            "issues": [],
            "runs": [],
        }
        detail = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
        assert detail.status_code == 503
        assert detail.headers["retry-after"] == "2"
        assert strict_json(detail)["last_error"] == "OSError: volume unavailable"
        assert client.get("/", headers=AUTH).status_code == 200

        collector.outcomes.append(make_snapshot(2, make_run(1)))
        collector.refresh()
        assert client.get("/api/runs", headers=AUTH).status_code == 200
        assert client.get(f"/api/runs/{RUN_ID}", headers=AUTH).status_code == 200


def test_a_refresh_raising_unexpectedly_still_starts_the_app(credentials):
    collector = StubCollector(RuntimeError("collector bug"))
    with serve(collector, credentials) as client:
        assert collector.calls == 1
        listing = client.get("/api/runs", headers=AUTH)
        assert listing.status_code == 503
        assert strict_json(listing)["last_error"] is None
        collector.outcomes.append(make_snapshot(1, make_run(1)))
        collector.refresh()
        assert client.get("/api/runs", headers=AUTH).status_code == 200


def test_lone_surrogates_and_nan_from_records_are_escaped(credentials):
    lone = "\ud800"
    snapshot = make_snapshot(
        1,
        make_run(
            2, gpu_name=f"gpu {lone}", detail=f"bad {lone}", issues=(f"run {lone}",), share=math.nan
        ),
        last_error=f"store {lone}",
        issues=(f"collection {lone}",),
    )
    with serve(StubCollector(snapshot), credentials) as client:
        listing = client.get("/api/runs", headers=AUTH)
        detail = client.get(f"/api/runs/{RUN_ID}", headers=AUTH)
    assert listing.status_code == 200
    assert detail.status_code == 200
    assert "\\ud800" in listing.text
    assert "\\ud800" in detail.text
    body = strict_json(listing)
    assert body["last_error"] == f"store {lone}"
    assert body["issues"] == [f"collection {lone}"]
    run = strict_json(detail)
    assert f"GPU: A100 requested, gpu {lone} detected" in run["status"]["line"]
    assert f"issues: run {lone}" in run["status"]["line"]
    assert {"task_id": "task-00002", "status": "invalid", "detail": f"bad {lone}"} in run["status"][
        "problems"
    ]


# -- settings ----------------------------------------------------------------------------


def test_settings_from_env_reads_credentials_and_poll_intervals(monkeypatch, password_hash):
    monkeypatch.setenv("DASHBOARD_USERNAME", USERNAME)
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", password_hash)
    defaults = web.settings_from_env()
    assert (defaults.poll_seconds, defaults.active_poll_seconds) == (10.0, 3.0)
    assert defaults.credentials.username == USERNAME
    assert defaults.credentials.verify(USERNAME, PASSWORD)
    monkeypatch.setenv("DASHBOARD_POLL_SECONDS", "30")
    assert (web.settings_from_env().poll_seconds, web.settings_from_env().active_poll_seconds) == (
        30.0,
        3.0,
    )
    monkeypatch.setenv("DASHBOARD_POLL_SECONDS", "1.5")
    assert web.settings_from_env().active_poll_seconds == 1.5


@pytest.mark.parametrize(
    ("changes", "field", "secret"),
    [
        pytest.param({"DASHBOARD_USERNAME": None}, "DASHBOARD_USERNAME", None, id="no-username"),
        pytest.param(
            {"DASHBOARD_PASSWORD_HASH": None}, "DASHBOARD_PASSWORD_HASH", None, id="no-hash"
        ),
        pytest.param({"DASHBOARD_USERNAME": ""}, "DASHBOARD_USERNAME", None, id="empty-username"),
        pytest.param({"DASHBOARD_USERNAME": "ad:min"}, "DASHBOARD_USERNAME", "ad:min", id="colon"),
        pytest.param(
            {"DASHBOARD_USERNAME": "viewer\n"}, "DASHBOARD_USERNAME", "viewer\n", id="newline"
        ),
        pytest.param(
            {"DASHBOARD_USERNAME": "vïewer"}, "DASHBOARD_USERNAME", "vïewer", id="non-ascii"
        ),
        pytest.param(
            {"DASHBOARD_PASSWORD_HASH": "correct?horse battery"},
            "DASHBOARD_PASSWORD_HASH",
            "correct?horse battery",
            id="plain-password",
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": "0.5"}, "DASHBOARD_POLL_SECONDS", None, id="poll-low"
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": "301"}, "DASHBOARD_POLL_SECONDS", "301", id="poll-high"
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": "nan"}, "DASHBOARD_POLL_SECONDS", "nan", id="poll-nan"
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": "inf"}, "DASHBOARD_POLL_SECONDS", "inf", id="poll-inf"
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": "ten"}, "DASHBOARD_POLL_SECONDS", "ten", id="poll-text"
        ),
        pytest.param(
            {"DASHBOARD_POLL_SECONDS": ""}, "DASHBOARD_POLL_SECONDS", None, id="poll-empty"
        ),
    ],
)
def test_settings_from_env_fail_closed(monkeypatch, password_hash, changes, field, secret):
    monkeypatch.setenv("DASHBOARD_USERNAME", USERNAME)
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", password_hash)
    for name, value in changes.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as refused:
        web.settings_from_env()
    message = str(refused.value)
    assert field in message
    assert password_hash not in message
    if secret is not None:
        assert secret not in message
    assert refused.value.__cause__ is None


def _replace_part(text: str, index: int, value: str) -> str:
    parts = text.split(":")
    parts[index] = value
    return ":".join(parts)


_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def _non_canonical(encoded: str, unused_bits: int) -> str:
    """Same decoded bytes, but with a nonzero unused trailing bit."""
    last = _ALPHABET.index(encoded[-1]) | (1 << (unused_bits - 1))
    return encoded[:-1] + _ALPHABET[last]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda h: h.replace("pbkdf2-sha256", "pbkdf2_sha256"), id="scheme"),
        pytest.param(lambda h: h.replace("pbkdf2-sha256", "PBKDF2-SHA256"), id="scheme-case"),
        pytest.param(lambda h: _replace_part(h, 1, "599999"), id="too-few-iterations"),
        pytest.param(lambda h: _replace_part(h, 1, "10000001"), id="too-many-iterations"),
        pytest.param(lambda h: _replace_part(h, 1, "0600000"), id="leading-zero"),
        pytest.param(lambda h: _replace_part(h, 1, "600000.0"), id="float-iterations"),
        pytest.param(lambda h: _replace_part(h, 2, h.split(":")[2][:-1]), id="short-salt"),
        pytest.param(lambda h: _replace_part(h, 2, h.split(":")[2] + "=="), id="padded-salt"),
        pytest.param(lambda h: _replace_part(h, 2, "+" + h.split(":")[2][1:]), id="std-alphabet"),
        pytest.param(
            lambda h: _replace_part(h, 2, _non_canonical(h.split(":")[2], 4)), id="salt-bits"
        ),
        pytest.param(
            lambda h: _replace_part(h, 3, _non_canonical(h.split(":")[3], 2)), id="hash-bits"
        ),
        pytest.param(lambda h: _replace_part(h, 3, h.split(":")[3][:-1]), id="short-hash"),
        pytest.param(lambda h: h + "\n", id="trailing-newline"),
        pytest.param(lambda h: " " + h, id="leading-space"),
        pytest.param(lambda h: h + ":extra", id="extra-field"),
    ],
)
def test_password_hash_format_is_strict(password_hash, mutate):
    malformed = mutate(password_hash)
    assert malformed != password_hash
    with pytest.raises(ValueError) as refused:
        Credentials.from_hash(USERNAME, malformed)
    assert "DASHBOARD_PASSWORD_HASH" in str(refused.value)
    assert malformed.strip() not in str(refused.value)


def test_hash_password_round_trip_and_password_policy(password_hash):
    assert re.fullmatch(r"pbkdf2-sha256:600000:[A-Za-z0-9_-]{22}:[A-Za-z0-9_-]{43}", password_hash)
    credentials = Credentials.from_hash(USERNAME, password_hash)
    assert credentials.verify(USERNAME, PASSWORD)
    assert not credentials.verify(USERNAME, PASSWORD.upper())
    assert not credentials.verify("viewer2", PASSWORD)
    assert PASSWORD not in repr(credentials)
    for weak in ("short pass!", "pässword-long-enough", "tab\there-long-enough"):
        with pytest.raises(ValueError) as refused:
            hash_password(weak)
        assert weak not in str(refused.value)
    with pytest.raises(ValueError):
        hash_password(PASSWORD, iterations=599_999)


# -- poller lifecycle --------------------------------------------------------------------


class SignallingCollector(StubCollector):
    """Signals once it has been refreshed ``signal_after`` times."""

    def __init__(self, *outcomes: CollectionSnapshot | BaseException, signal_after: int) -> None:
        super().__init__(*outcomes)
        self.signal_after = signal_after
        self.reached = threading.Event()

    def refresh(self) -> None:
        try:
            super().refresh()
        finally:
            if self.calls >= self.signal_after:
                self.reached.set()


@pytest.mark.parametrize(
    ("state", "poll_seconds", "active_poll_seconds"),
    [("running", 60.0, 0.01), ("complete", 0.01, 60.0)],
    ids=["generating-uses-active-interval", "idle-uses-poll-interval"],
)
def test_poller_refreshes_survives_errors_and_is_joined_on_shutdown(
    credentials, state, poll_seconds, active_poll_seconds
):
    # Startup refresh, a failing poll, then two idle polls: only reachable within the
    # bounded wait if the poller uses the short interval for this snapshot.
    collector = SignallingCollector(
        make_snapshot(1, make_run(1, state=state)),
        RuntimeError("transient store failure"),
        signal_after=4,
    )
    app = web.create_app(
        collector,
        credentials=credentials,
        poll_seconds=poll_seconds,
        active_poll_seconds=active_poll_seconds,
    )
    assert app.state.collector is collector
    with TestClient(app) as client:
        poller = app.state.poller
        assert poller.name == "live-dashboard-poller"
        assert poller.is_alive()
        assert collector.reached.wait(timeout=5.0)
        assert client.get("/api/runs", headers=AUTH).status_code == 200
    assert not poller.is_alive()
