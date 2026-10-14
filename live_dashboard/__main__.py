"""Local commands of the live dashboard.

  python -m live_dashboard hash-password --username NAME --output auth.json
  python -m live_dashboard serve --data-dir DIR --credentials auth.json
  python -m live_dashboard generate --data-dir DIR --label demo --worlds 20

``hash-password`` writes the username and a PBKDF2 hash (never the password) to a new
0600 JSON file, usable as ``serve --credentials`` and as
``modal secret create -e pfn pymc-generator-dashboard-auth --from-json``. ``serve``
follows the publications under ``--data-dir`` (credentials from ``--credentials`` or
the ``DASHBOARD_USERNAME``/``DASHBOARD_PASSWORD_HASH`` environment variables).
``generate`` publishes a run into ``--data-dir`` task by task.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import platform
import sys
from datetime import timedelta
from pathlib import Path

from live_dashboard.deployment import ENVIRONMENT, SECRET_NAME
from live_dashboard.generation import RequestError, RunConflictError


class _RefusedError(Exception):
    """Input or files the command refuses: reported in one line, not as a traceback."""


def _time_limit(text: str) -> float:
    """A generation time limit: finite seconds the writer can plan tasks within."""
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not (math.isfinite(seconds) and 120.0 <= seconds <= 7 * 24 * 3600.0):
        raise argparse.ArgumentTypeError("must be a number of seconds from 120 to 604800")
    return seconds


def _poll_seconds(text: str) -> float:
    """The dashboard's idle refresh interval, in the range its settings accept."""
    from live_dashboard.web import MAX_POLL_SECONDS, MIN_POLL_SECONDS

    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not (math.isfinite(seconds) and MIN_POLL_SECONDS <= seconds <= MAX_POLL_SECONDS):
        raise argparse.ArgumentTypeError(
            f"must be a number of seconds from {MIN_POLL_SECONDS:g} to {MAX_POLL_SECONDS:g}"
        )
    return seconds


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be a TCP port number from 0 to 65535")
    return port


def _hash_password(args: argparse.Namespace) -> None:
    from live_dashboard.auth import PASSWORD_HASH_ENV, USERNAME_ENV, Credentials, hash_password

    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = getpass.getpass("Dashboard password: ")
        if getpass.getpass("Repeat password: ") != password:
            raise SystemExit("passwords differ")
    try:
        values = {USERNAME_ENV: args.username, PASSWORD_HASH_ENV: hash_password(password)}
        Credentials.from_env(values)  # refuse a username the dashboard would refuse
        args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except (ValueError, OSError) as error:
        raise _RefusedError(str(error)) from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(values, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.output} (username and password hash; mode 0600)")
    print(f"Modal: modal secret create -e {ENVIRONMENT} {SECRET_NAME} --from-json {args.output}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from live_dashboard.collection import Collector
    from live_dashboard.storage import DirectoryStore
    from live_dashboard.web import create_app, settings_from_env

    try:
        environ: dict[str, str] = dict(os.environ)
        if args.credentials is not None:
            loaded = json.loads(Path(args.credentials).read_text())
            if not isinstance(loaded, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
            ):
                raise ValueError(f"{args.credentials} must hold a JSON object of strings")
            environ.update(loaded)
        if args.poll_seconds is not None:
            environ["DASHBOARD_POLL_SECONDS"] = str(args.poll_seconds)
        settings = settings_from_env(environ)
    except (ValueError, OSError) as error:
        raise _RefusedError(str(error)) from error
    collector = Collector(DirectoryStore(args.data_dir))
    app = create_app(
        collector,
        credentials=settings.credentials,
        poll_seconds=settings.poll_seconds,
        active_poll_seconds=settings.active_poll_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def _generate(args: argparse.Namespace) -> None:
    from live_dashboard.generation import run_generation
    from live_dashboard.publication import RunRequest, WorkerInfo, utc_now
    from live_dashboard.storage import DirectoryStore

    try:
        request = RunRequest(
            label=args.label,
            preset=args.preset,
            worlds=args.worlds,
            worlds_per_task=args.worlds_per_task,
            seed=args.seed,
        )
    except ValueError as error:
        raise _RefusedError(str(error)) from error
    worker = WorkerInfo(
        platform=platform.platform(), cpu_count=os.cpu_count(), gpu_requested=None, gpu_name=None
    )
    print(f"run {request.run_id}: {request.n_tasks} tasks x {request.worlds_per_task} worlds")
    try:
        summary = run_generation(
            DirectoryStore(args.data_dir),
            request,
            worker=worker,
            deadline=utc_now() + timedelta(seconds=args.max_seconds),
        )
    except (RequestError, RunConflictError) as error:
        # Refusals happen before any task runs; every other failure keeps its traceback.
        raise _RefusedError(str(error)) from error
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m live_dashboard", description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    commands = parser.add_subparsers(dest="command", required=True)

    hashing = commands.add_parser("hash-password", help="write a credentials JSON file")
    hashing.add_argument("--username", required=True)
    hashing.add_argument("--output", type=Path, required=True, help="new file (never overwritten)")
    hashing.add_argument("--password-stdin", action="store_true", help="read one line from stdin")
    hashing.set_defaults(handler=_hash_password)

    serving = commands.add_parser("serve", help="serve the dashboard for a local store")
    serving.add_argument("--data-dir", type=Path, required=True)
    serving.add_argument("--credentials", type=Path, help="JSON written by hash-password")
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=_port, default=8000)
    serving.add_argument(
        "--poll-seconds", type=_poll_seconds, help="idle refresh interval (1-300 s)"
    )
    serving.set_defaults(handler=_serve)

    generating = commands.add_parser("generate", help="publish a run into a local store")
    generating.add_argument("--data-dir", type=Path, required=True)
    generating.add_argument("--label", required=True)
    generating.add_argument("--preset", default="showcase-variety")
    generating.add_argument("--worlds", type=int, default=20)
    generating.add_argument("--worlds-per-task", type=int, default=2)
    generating.add_argument("--seed", type=int, default=29)
    generating.add_argument(
        "--max-seconds",
        type=_time_limit,
        default=3600.0,
        help=(
            "time limit of this attempt (120 s to 7 days, default 3600); tasks stop starting "
            "early enough to finish before it, and readers report the attempt lost after it"
        ),
    )
    generating.set_defaults(handler=_generate)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except _RefusedError as refusal:
        parser.exit(2, f"{parser.prog} {args.command}: error: {refusal}\n")


if __name__ == "__main__":
    main()
