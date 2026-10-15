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
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import plotly
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from live_dashboard import auth, web
from live_dashboard import publication as pub
from live_dashboard.auth import Credentials, hash_password
from live_dashboard.collection import Collector
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
                worlds.append(
                    WorldMetrics(
                        world_id,
                        task_id,
                        share,
                        0.35,
                        0.25,
                        ("geometric",),
                        ("logistic",),
                        structural_pairs=(("logistic", "geometric"),),
                        active_channels=1,
                        active_controls=2,
                        effect_channels=1,
                        effect_controls=2,
                        channel_slots=10,
                        control_slots=10,
                    )
                )
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


def test_success_cache_is_bounded_and_forgets_the_least_recently_used(credentials, monkeypatch):
    # PBKDF2 is replaced by a counting equality check: this test is about which headers
    # are re-verified, and 600k-iteration hashes for 66 headers would dominate its time.
    verified: list[str] = []

    def counting_verify(self, username: str, password: str) -> bool:
        verified.append(username)
        return (username, password) == (USERNAME, PASSWORD)

    monkeypatch.setattr(Credentials, "verify", counting_verify)
    size = auth.ACCEPTED_CACHE_SIZE
    # Distinct but equally valid headers: any number of spaces may follow the scheme.
    headers = [{"Authorization": "Basic" + " " * n + TOKEN} for n in range(1, size + 2)]
    with serve(StubCollector(), credentials) as client:

        def accepted(header: dict[str, str]) -> bool:
            return client.get("/", headers=header).status_code == 200

        assert all(accepted(header) for header in headers)
        assert len(verified) == size + 1
        # The newest ``size`` headers are remembered (the first one was dropped).
        assert accepted(headers[-1]) and accepted(headers[1])
        assert len(verified) == size + 1
        assert accepted(headers[0])
        assert len(verified) == size + 2
        # Re-admitting it dropped the least recently used header, which was headers[2]
        # (headers[1] had just been used).
        assert accepted(headers[1])
        assert len(verified) == size + 2
        assert accepted(headers[2])
        assert len(verified) == size + 3


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


def test_every_figure_the_api_returns_has_a_place_on_the_page(credentials):
    with serve(StubCollector(make_snapshot(1, make_run(2))), credentials) as client:
        page = client.get("/", headers=AUTH)
        detail = strict_json(client.get(f"/api/runs/{RUN_ID}", headers=AUTH))
    # The page draws a figure only where an anchor with its id exists, in page order.
    anchors = re.findall(r'data-figure="([^"]+)"', page.text)
    assert anchors == [figure["id"] for figure in detail["figures"]]


def test_plotly_bundle_is_gzip_compressed_when_accepted(credentials):
    with serve(StubCollector(), credentials) as client:
        compressed = client.get(web.PLOTLY_PATH, headers={**AUTH, "Accept-Encoding": "gzip"})
        identity = client.get(web.PLOTLY_PATH, headers={**AUTH, "Accept-Encoding": "identity"})
    assert compressed.status_code == identity.status_code == 200
    assert compressed.headers["content-encoding"] == "gzip"
    assert "content-encoding" not in identity.headers
    assert int(compressed.headers["content-length"]) < len(identity.content) // 2
    assert compressed.content == identity.content  # httpx decodes the gzip body


def test_unknown_static_path_is_not_found_after_authentication(credentials):
    with serve(StubCollector(make_snapshot(1)), credentials) as client:
        assert client.get("/static/missing.js", headers=AUTH).status_code == 404
        assert client.get("/docs", headers=AUTH).status_code == 404
        assert client.get("/openapi.json", headers=AUTH).status_code == 404


# -- JSON API ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "last_error", "poll_seconds"),
    [
        ("running", None, 3.0),
        ("complete", None, 10.0),
        ("complete", "OSError: volume unavailable", 3.0),
    ],
    ids=["generating", "idle", "latest-refresh-failed"],
)
def test_run_list_reports_refresh_state_and_poll_interval(
    credentials, state, last_error, poll_seconds
):
    snapshot = make_snapshot(
        1,
        make_run(1, state=state),
        last_error=last_error,
        issues=("run.json of x is unsupported",),
    )
    with serve(StubCollector(snapshot), credentials) as client:
        response = client.get("/api/runs", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = strict_json(response)
    assert body["refreshed_at"] == "2026-01-01T01:00:00.000Z"
    assert body["last_error"] == last_error
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


def test_attempt_passing_its_deadline_changes_the_etag(credentials, live_golden, tmp_path):
    # A real collector over a copy of the golden run whose last task is still being
    # generated by a newer, running attempt; nothing is written after startup.
    store = support.copy_store(live_golden.root, tmp_path / "store")
    run_id = live_golden.request.run_id
    for completion in support.completion_files(store.root, run_id, "task-00002"):
        completion.unlink()
    started = live_golden.end_time + timedelta(hours=1)
    attempt = pub.AttemptRecord(
        run_id=run_id,
        attempt_id=f"{started:%Y%m%dT%H%M%SZ}-0000beef",
        state="running",
        started_at=started,
        updated_at=started,
        deadline_at=started + timedelta(minutes=20),
        current_task_index=2,
        published_task_ids=(),
        reused_task_ids=("task-00000", "task-00001"),
        error=None,
        worker=support.WORKER,
    )
    store.write_replace(pub.attempt_record_path(run_id, attempt.attempt_id), attempt.encode())
    clock = support.FakeClock(start=started + timedelta(minutes=1))
    collector = Collector(store, clock=clock)
    url = f"/api/runs/{run_id}"
    with serve(collector, credentials) as client:
        running = client.get(url, headers=AUTH)
        assert running.status_code == 200
        assert strict_json(running)["run"]["state"] == "running"
        etag = running.headers["etag"]
        collector.refresh()  # same store, a little later, still before the deadline
        idle = client.get(url, headers={**AUTH, "If-None-Match": etag})
        assert (idle.status_code, idle.headers["etag"]) == (304, etag)
        clock.advance(20 * 60)
        collector.refresh()
        lost = client.get(url, headers={**AUTH, "If-None-Match": etag})
    assert lost.status_code == 200
    assert lost.headers["etag"] != etag
    body = strict_json(lost)
    assert body["run"]["state"] == "lost"
    assert body["run"]["published_worlds"] == 4
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
            "poll_seconds": 3.0,
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
        body = strict_json(listing)
        assert body["last_error"] is None
        assert body["poll_seconds"] == 3.0  # never refreshed: the page polls fast
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

FAST = 0.01  # seconds; a poller using it refreshes dozens of times per QUIET window
SLOW = 60.0  # seconds; a poller using it never refreshes during a test
QUIET = 0.3  # seconds a test waits to show that no refresh happens


class SignallingCollector(StubCollector):
    """Sets ``reached`` during its ``signal_after``-th refresh, then blocks for
    ``block_seconds`` (a slow store read in progress) and sets ``finished``."""

    def __init__(
        self,
        *outcomes: CollectionSnapshot | BaseException,
        signal_after: int,
        block_seconds: float = 0.0,
    ) -> None:
        super().__init__(*outcomes)
        self.signal_after = signal_after
        self.block_seconds = block_seconds
        self.reached = threading.Event()
        self.finished = threading.Event()

    def refresh(self) -> None:
        try:
            super().refresh()
        finally:
            if self.calls == self.signal_after:
                self.reached.set()
                time.sleep(self.block_seconds)
                self.finished.set()


class UnreadableSnapshotCollector(SignallingCollector):
    """Its snapshot cannot be read the first ``failures`` times it is asked for."""

    def __init__(self, *outcomes: CollectionSnapshot, failures: int, signal_after: int) -> None:
        self.failures = failures
        super().__init__(*outcomes, signal_after=signal_after)

    @property
    def snapshot(self) -> CollectionSnapshot:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("snapshot unavailable")
        return self._snapshot

    @snapshot.setter
    def snapshot(self, value: CollectionSnapshot) -> None:
        self._snapshot = value


def poller_app(collector: StubCollector, credentials: Credentials, poll: float, active: float):
    return web.create_app(
        collector, credentials=credentials, poll_seconds=poll, active_poll_seconds=active
    )


@pytest.mark.parametrize(
    "startup",
    [
        pytest.param(make_snapshot(1, make_run(1, state="running")), id="generating"),
        pytest.param(RuntimeError("collector bug"), id="first-refresh-raised"),
        pytest.param(
            make_snapshot(1, refreshed=False, last_error="OSError: volume unavailable"),
            id="first-refresh-failed",
        ),
        pytest.param(
            make_snapshot(1, make_run(1, state="complete"), last_error="OSError: volume gone"),
            id="latest-refresh-failed",
        ),
    ],
)
def test_poller_uses_the_active_interval_while_generating_or_failing(credentials, startup):
    collector = SignallingCollector(startup, signal_after=2)
    with TestClient(poller_app(collector, credentials, poll=SLOW, active=FAST)):
        assert collector.reached.wait(timeout=5.0)


def test_poller_uses_the_idle_interval_when_nothing_happens(credentials):
    # Not the shorter of the two intervals: an idle, healthy collection is left alone.
    collector = SignallingCollector(make_snapshot(1, make_run(1, state="complete")), signal_after=2)
    with TestClient(poller_app(collector, credentials, poll=SLOW, active=FAST)):
        assert not collector.reached.wait(timeout=QUIET)
    assert collector.calls == 1  # the startup refresh only


def test_poller_survives_a_raising_refresh(credentials):
    collector = SignallingCollector(
        make_snapshot(1, make_run(1, state="running")),
        RuntimeError("collector bug"),
        signal_after=4,
    )
    app = poller_app(collector, credentials, poll=SLOW, active=FAST)
    with TestClient(app) as client:
        assert collector.reached.wait(timeout=5.0)
        assert app.state.poller.is_alive()
        assert client.get("/api/runs", headers=AUTH).status_code == 200


def test_poller_survives_an_unreadable_snapshot_and_retries_fast(credentials):
    # Healthy and idle, so only the failure fallback (the active interval) brings the
    # second refresh within the wait.
    collector = UnreadableSnapshotCollector(
        make_snapshot(1, make_run(1, state="complete")), failures=1, signal_after=2
    )
    with TestClient(poller_app(collector, credentials, poll=SLOW, active=FAST)):
        assert collector.reached.wait(timeout=5.0)
        # Once a step succeeds again the idle (SLOW) interval applies: no further refresh.
        time.sleep(QUIET)
        assert collector.calls == 2
    assert collector.failures == 0


def test_shutdown_waits_for_a_refresh_in_progress(credentials):
    # The blocked refresh makes the run generating, so the poller's next wait is SLOW:
    # nothing but shutdown can end the thread, however fast or slow this thread runs.
    collector = SignallingCollector(
        make_snapshot(1, make_run(1, state="complete")),
        make_snapshot(2, make_run(1, state="running")),
        signal_after=2,
        block_seconds=QUIET,
    )
    app = poller_app(collector, credentials, poll=FAST, active=SLOW)
    with TestClient(app):
        poller = app.state.poller
        assert collector.reached.wait(timeout=5.0)  # the poller is now inside a refresh
    assert collector.finished.is_set()
    assert not poller.is_alive()
    assert collector.calls == 2


@pytest.mark.parametrize(
    ("poll", "active"),
    [
        (math.inf, 3.0),
        (math.nan, 3.0),
        (0.0, 3.0),
        (301.0, 3.0),
        (1e10, 3.0),
        (10.0, 0.0),
        (10.0, -1.0),
        (10.0, math.nan),
        (10.0, 301.0),
    ],
)
def test_create_app_refuses_intervals_out_of_range(credentials, poll, active):
    with pytest.raises(ValueError, match=r"poll_seconds must be a number of seconds in \(0, 300\]"):
        poller_app(StubCollector(), credentials, poll=poll, active=active)


def test_the_largest_interval_is_accepted_and_reported(credentials):
    collector = StubCollector(make_snapshot(1, make_run(1, state="complete")))
    app = poller_app(collector, credentials, poll=web.MAX_POLL_SECONDS, active=FAST)
    with TestClient(app) as client:
        listing = client.get("/api/runs", headers=AUTH)
    assert strict_json(listing)["poll_seconds"] == web.MAX_POLL_SECONDS
