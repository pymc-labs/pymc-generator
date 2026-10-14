"""The dashboard web app: one page, its assets and a small JSON API, all behind Basic auth.

Request handlers never touch the store. Startup refreshes the collector once (the first
request after a cold start waits for it); then one poller thread refreshes it (faster
while a run is generating) and handlers read its latest immutable ``CollectionSnapshot``,
so a slow or failing store can delay data but never a response.

Every JSON body this module builds is serialised with ``ensure_ascii=True,
allow_nan=False``: lone surrogates from record text are escaped instead of failing the
response, and a NaN that slipped through is a loud bug rather than invalid JSON sent to
the page. A run body is built once per collector refresh and identified by an ETag over
its bytes, so the page's polling costs a 304 until something it shows actually changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import threading
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anyio.to_thread
from fastapi import FastAPI, Request
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response

from live_dashboard import figures
from live_dashboard.auth import BasicAuthMiddleware, Credentials
from live_dashboard.publication import format_timestamp
from live_dashboard.snapshot import CollectionSnapshot, RunSnapshot

logger = logging.getLogger(__name__)

POLL_SECONDS_ENV = "DASHBOARD_POLL_SECONDS"
DEFAULT_POLL_SECONDS = 10.0
DEFAULT_ACTIVE_POLL_SECONDS = 3.0
MIN_POLL_SECONDS = 1.0
MAX_POLL_SECONDS = 300.0
POLLER_THREAD_NAME = "live-dashboard-poller"
RETRY_AFTER_SECONDS = 2
STATIC_DIR = Path(__file__).resolve().parent / "static"
PLOTLY_PATH = f"/static/plotly-{figures.PLOTLY_VERSION}.min.js"
_PLOTLY_PLACEHOLDER = b"__PLOTLY_PATH__"  # in index.html, replaced by PLOTLY_PATH

_NO_STORE = {"Cache-Control": "no-store"}
_REVALIDATE = {"Cache-Control": "no-cache, private"}
_IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}


class SnapshotSource(Protocol):
    """What the app needs from a collector (``live_dashboard.collection.Collector``)."""

    @property
    def snapshot(self) -> CollectionSnapshot: ...

    def refresh(self) -> None: ...


@dataclass(frozen=True)
class Settings:
    """Deployment settings read from the environment (see :func:`settings_from_env`)."""

    credentials: Credentials
    poll_seconds: float
    active_poll_seconds: float


def _poll_seconds(text: str | None) -> float:
    if text is None:
        return DEFAULT_POLL_SECONDS
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not (math.isfinite(seconds) and MIN_POLL_SECONDS <= seconds <= MAX_POLL_SECONDS):
        raise ValueError(
            f"{POLL_SECONDS_ENV} must be a number of seconds in "
            f"[{MIN_POLL_SECONDS:g}, {MAX_POLL_SECONDS:g}]"
        )
    return seconds


def settings_from_env(environ: Mapping[str, str] | None = None) -> Settings:
    """Credentials and polling intervals; any malformed value refuses to start the app.

    The fast interval (see :func:`poll_interval`) is ``DEFAULT_ACTIVE_POLL_SECONDS``, or
    ``DASHBOARD_POLL_SECONDS`` if that is shorter.
    """
    env = os.environ if environ is None else environ
    credentials = Credentials.from_env(env)
    poll_seconds = _poll_seconds(env.get(POLL_SECONDS_ENV))
    return Settings(
        credentials=credentials,
        poll_seconds=poll_seconds,
        active_poll_seconds=min(DEFAULT_ACTIVE_POLL_SECONDS, poll_seconds),
    )


def poll_interval(
    snapshot: CollectionSnapshot, poll_seconds: float, active_poll_seconds: float
) -> float:
    """How long to wait before the next refresh, for the poller and the page alike.

    Fast while a run is generating (new tasks are expected) and while the store has
    never been read or its latest read failed: a transient failure, typically at cold
    start, then recovers within seconds instead of after a full idle interval.
    """
    fast = snapshot.generating or snapshot.refreshed_at is None or snapshot.last_error is not None
    return active_poll_seconds if fast else poll_seconds


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return text.encode("ascii")


def _json_response(value: Any, headers: Mapping[str, str], *, status_code: int = 200) -> Response:
    return Response(
        _json_bytes(value), status_code=status_code, headers=headers, media_type="application/json"
    )


def _etag_matches(header: str | None, etag: str) -> bool:
    """``If-None-Match`` uses weak comparison: ``W/`` prefixes are ignored."""
    if header is None:
        return False
    for candidate in header.split(","):
        tag = candidate.strip()
        if tag == "*" or tag.removeprefix("W/") == etag:
            return True
    return False


class _RunBodies:
    """Encoded run bodies and their ETags for the newest snapshot sequence only.

    A body is a pure function of its run snapshot, so caching per sequence is exact;
    older sequences are dropped as soon as a newer snapshot is served.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = -1
        self._bodies: dict[str, tuple[bytes, str]] = {}

    def get(self, sequence: int, run: RunSnapshot) -> tuple[bytes, str]:
        with self._lock:
            if sequence > self._sequence:
                self._sequence = sequence
                self._bodies = {}
            current = sequence == self._sequence
            cached = self._bodies.get(run.run_id) if current else None
            if cached is not None:
                return cached
            body = _json_bytes(figures.run_body(run))
            entry = (body, f'"{hashlib.sha256(body).hexdigest()[:24]}"')
            if current:
                self._bodies[run.run_id] = entry
            return entry


def _poll(
    collector: SnapshotSource,
    stop: threading.Event,
    poll_seconds: float,
    active_poll_seconds: float,
) -> None:
    """Refresh until ``stop`` is set; nothing else ends the loop.

    Each step (choose the interval, wait, refresh) is guarded as a whole: an exception is
    logged and the next step waits the active interval, as after a refresh that recorded
    a store error. A misbehaving collector can thus neither end the thread nor make it
    spin, and setting ``stop`` still ends any wait at once.
    """
    failed = False
    while True:
        try:
            if failed:
                interval = active_poll_seconds
            else:
                interval = poll_interval(collector.snapshot, poll_seconds, active_poll_seconds)
            failed = False
            if stop.wait(interval):
                return
            collector.refresh()
        except Exception:
            logger.exception("poller step failed; retrying after the active interval")
            failed = True


def _static(name: str) -> bytes:
    return (STATIC_DIR / name).read_bytes()


def create_app(
    collector: SnapshotSource,
    *,
    credentials: Credentials,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    active_poll_seconds: float = DEFAULT_ACTIVE_POLL_SECONDS,
    start_poller: bool = True,
) -> FastAPI:
    """The dashboard app over ``collector`` (anything with ``.snapshot`` and ``.refresh()``).

    Startup refreshes once (a failure leaves the API answering 503 until a refresh
    succeeds) and then, if ``start_poller``, runs the poller thread ``app.state.poller``
    (waiting :func:`poll_interval` between refreshes) until shutdown stops and joins it.
    Both intervals must be finite, positive and at most ``MAX_POLL_SECONDS``: a longer
    wait would leave the page stale (and a huge one overflows the poller's timer). The
    operator range ``[MIN_POLL_SECONDS, MAX_POLL_SECONDS]`` applies to
    :func:`settings_from_env`.
    """
    for name, value in (
        ("poll_seconds", poll_seconds),
        ("active_poll_seconds", active_poll_seconds),
    ):
        if not (math.isfinite(value) and 0 < value <= MAX_POLL_SECONDS):
            raise ValueError(f"{name} must be a number of seconds in (0, {MAX_POLL_SECONDS:g}]")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await anyio.to_thread.run_sync(collector.refresh)
        except Exception:
            logger.exception("first collector refresh raised; serving 503 until one succeeds")
        stop = threading.Event()
        poller = None
        if start_poller:
            poller = threading.Thread(
                target=_poll,
                args=(collector, stop, poll_seconds, active_poll_seconds),
                name=POLLER_THREAD_NAME,
                daemon=True,
            )
            poller.start()
        app.state.poller = poller
        try:
            yield
        finally:
            stop.set()
            if poller is not None:
                await anyio.to_thread.run_sync(poller.join)

    app = FastAPI(
        title="PyMC Generator live dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.collector = collector
    app.state.poller = None
    # Starlette wraps later additions around earlier ones: authentication is outermost,
    # so an unauthenticated request reaches neither compression nor any route.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(BasicAuthMiddleware, credentials=credentials)

    page = _static("index.html").replace(_PLOTLY_PLACEHOLDER, PLOTLY_PATH.encode("ascii"))
    script = _static("app.js")
    stylesheet = _static("style.css")
    plotly_js = figures.plotly_js()
    bodies = _RunBodies()

    def listing(snapshot: CollectionSnapshot) -> dict[str, Any]:
        refreshed_at = snapshot.refreshed_at
        return {
            "refreshed_at": None if refreshed_at is None else format_timestamp(refreshed_at),
            "last_error": snapshot.last_error,
            "poll_seconds": poll_interval(snapshot, poll_seconds, active_poll_seconds),
            "issues": list(snapshot.issues),
            "runs": [figures.run_summary(run) for run in snapshot.runs],
        }

    def not_refreshed(snapshot: CollectionSnapshot) -> Response:
        headers = {**_NO_STORE, "Retry-After": str(RETRY_AFTER_SECONDS)}
        return _json_response(listing(snapshot), headers, status_code=503)

    @app.get("/")
    def index() -> Response:
        return Response(page, headers=_REVALIDATE, media_type="text/html")

    @app.get("/static/app.js")
    def app_script() -> Response:
        return Response(script, headers=_REVALIDATE, media_type="text/javascript")

    @app.get("/static/style.css")
    def app_stylesheet() -> Response:
        return Response(stylesheet, headers=_REVALIDATE, media_type="text/css")

    @app.get(PLOTLY_PATH)
    def plotly_bundle() -> Response:
        return Response(plotly_js, headers=_IMMUTABLE, media_type="text/javascript")

    @app.get("/api/runs")
    def run_list() -> Response:
        snapshot = collector.snapshot
        if snapshot.refreshed_at is None:
            return not_refreshed(snapshot)
        return _json_response(listing(snapshot), _NO_STORE)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, request: Request) -> Response:
        snapshot = collector.snapshot
        if snapshot.refreshed_at is None:
            return not_refreshed(snapshot)
        run = snapshot.run_index.get(run_id)
        if run is None:
            return _json_response({"detail": "unknown run"}, _NO_STORE, status_code=404)
        body, etag = bodies.get(snapshot.sequence, run)
        headers = {**_REVALIDATE, "ETag": etag}
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        return Response(body, headers=headers, media_type="application/json")

    return app
