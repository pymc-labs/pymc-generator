"""HTTP Basic authentication in front of every request the dashboard serves.

The check is a pure ASGI middleware, not a FastAPI dependency: dependencies do not
cover documentation routes, static assets, unknown paths or websockets, and one outermost
check cannot be forgotten on a future route.

The password is stored only as a PBKDF2-SHA256 hash
(``pbkdf2-sha256:<iterations>:<salt>:<hash>``, unpadded URL-safe base64), parsed strictly
so that a mistyped secret fails at startup instead of locking everyone out (or in).
Verification costs the same whatever was wrong: a wrong username still runs PBKDF2 (on a
per-process dummy hash with the configured iterations) and both comparisons are always
made. The key derivation runs in worker threads behind a dedicated limiter of two, so
guessing cannot starve the event loop or the request thread pool, and a header that
already verified is remembered only as an HMAC under a per-process key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import secrets
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field

import anyio
import anyio.to_thread
from starlette.types import ASGIApp, Receive, Scope, Send

USERNAME_ENV = "DASHBOARD_USERNAME"
PASSWORD_HASH_ENV = "DASHBOARD_PASSWORD_HASH"
REALM = "PyMC Generator live dashboard"
MIN_ITERATIONS = 600_000
MAX_ITERATIONS = 10_000_000
MIN_PASSWORD_CHARS = 12
SALT_BYTES = 16
DIGEST_BYTES = 32
VERIFY_THREADS = 2
ACCEPTED_CACHE_SIZE = 64

_HASH = re.compile(r"pbkdf2-sha256:([0-9]{6,8}):([A-Za-z0-9_-]{22}):([A-Za-z0-9_-]{43})")
_HASH_FORM = "pbkdf2-sha256:<iterations>:<salt>:<hash>"
_CHALLENGE = f'Basic realm="{REALM}", charset="UTF-8"'.encode("ascii")
_UNAUTHORIZED_BODY = b"authentication required\n"
# A wrong username is checked against this hash, so it costs exactly one real PBKDF2.
_DUMMY_SALT = secrets.token_bytes(SALT_BYTES)
_DUMMY_DIGEST = secrets.token_bytes(DIGEST_BYTES)


def _printable_ascii(text: str) -> bool:
    return all(" " <= character <= "~" for character in text)


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(text: str, size: int) -> bytes | None:
    """Canonical unpadded URL-safe base64 of exactly ``size`` bytes, else ``None``."""
    data = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    return data if len(data) == size and _encode(data) == text else None


def _derive(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=DIGEST_BYTES
    )


def _check_iterations(iterations: object) -> None:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS
    ):
        raise ValueError(f"iterations must be an integer in [{MIN_ITERATIONS}, {MAX_ITERATIONS}]")


def hash_password(password: str, iterations: int = MIN_ITERATIONS) -> str:
    """The ``DASHBOARD_PASSWORD_HASH`` value for ``password`` (fresh random salt)."""
    if (
        not isinstance(password, str)
        or len(password) < MIN_PASSWORD_CHARS
        or not _printable_ascii(password)
    ):
        raise ValueError(
            f"the password must be at least {MIN_PASSWORD_CHARS} printable ASCII characters"
        )
    _check_iterations(iterations)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, iterations)
    return f"pbkdf2-sha256:{iterations}:{_encode(salt)}:{_encode(digest)}"


@dataclass(frozen=True)
class Credentials:
    """The single dashboard account. Error messages name the setting, never its value."""

    username: str
    iterations: int
    salt: bytes = field(repr=False)
    digest: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.username, str)
            or not self.username
            or ":" in self.username
            or not _printable_ascii(self.username)
        ):
            raise ValueError(f"{USERNAME_ENV} must be non-empty printable ASCII without ':'")
        try:
            _check_iterations(self.iterations)
        except ValueError:
            raise ValueError(
                f"{PASSWORD_HASH_ENV} iterations must be in [{MIN_ITERATIONS}, {MAX_ITERATIONS}]"
            ) from None
        if not isinstance(self.salt, bytes) or len(self.salt) != SALT_BYTES:
            raise ValueError(f"{PASSWORD_HASH_ENV} salt must be {SALT_BYTES} bytes")
        if not isinstance(self.digest, bytes) or len(self.digest) != DIGEST_BYTES:
            raise ValueError(f"{PASSWORD_HASH_ENV} hash must be {DIGEST_BYTES} bytes")

    @classmethod
    def from_hash(cls, username: str, password_hash: str) -> Credentials:
        """Parse a :func:`hash_password` value; anything else is refused."""
        match = _HASH.fullmatch(password_hash) if isinstance(password_hash, str) else None
        if match is None:
            raise ValueError(f"{PASSWORD_HASH_ENV} must have the form {_HASH_FORM}")
        iterations_text, salt_text, digest_text = match.groups()
        iterations = int(iterations_text)
        if str(iterations) != iterations_text:
            raise ValueError(f"{PASSWORD_HASH_ENV} iterations must not have leading zeros")
        salt = _decode(salt_text, SALT_BYTES)
        digest = _decode(digest_text, DIGEST_BYTES)
        if salt is None or digest is None:
            raise ValueError(
                f"{PASSWORD_HASH_ENV} salt and hash must be canonical unpadded URL-safe base64"
            )
        return cls(username=username, iterations=iterations, salt=salt, digest=digest)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Credentials:
        """Read ``DASHBOARD_USERNAME`` and ``DASHBOARD_PASSWORD_HASH``, failing closed."""
        env = os.environ if environ is None else environ
        missing = [name for name in (USERNAME_ENV, PASSWORD_HASH_ENV) if not env.get(name)]
        if missing:
            raise ValueError(f"{' and '.join(missing)} must be set")
        return cls.from_hash(env[USERNAME_ENV], env[PASSWORD_HASH_ENV])

    def verify(self, username: str, password: str) -> bool:
        """Whether the pair matches, with the same work whichever part is wrong."""
        username_ok = hmac.compare_digest(
            hashlib.sha256(username.encode("utf-8")).digest(),
            hashlib.sha256(self.username.encode("ascii")).digest(),
        )
        salt, expected = (self.salt, self.digest) if username_ok else (_DUMMY_SALT, _DUMMY_DIGEST)
        password_ok = hmac.compare_digest(_derive(password, salt, self.iterations), expected)
        return username_ok & password_ok


def _basic_pair(header: bytes) -> tuple[str, str] | None:
    """``(username, password)`` of a well-formed Basic header, else ``None``."""
    try:
        scheme, _, token = header.decode("ascii").partition(" ")
        if scheme.lower() != "basic":
            return None
        decoded = base64.b64decode(token.strip(" "), validate=True).decode("ascii")
    except (UnicodeDecodeError, binascii.Error):
        return None
    username, colon, password = decoded.partition(":")
    return (username, password) if colon else None


class BasicAuthMiddleware:
    """Answer 401 to every HTTP request (and close every websocket) without valid
    credentials; lifespan events pass through untouched."""

    def __init__(self, app: ASGIApp, *, credentials: Credentials) -> None:
        self.app = app
        self._credentials = credentials
        self._limiter = anyio.CapacityLimiter(VERIFY_THREADS)
        self._cache_key = secrets.token_bytes(32)
        self._accepted: OrderedDict[bytes, None] = OrderedDict()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or await self._authorized(scope):
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": ""})
        else:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"www-authenticate", _CHALLENGE),
                        (b"cache-control", b"no-store"),
                        (b"content-type", b"text/plain; charset=utf-8"),
                        (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})

    async def _authorized(self, scope: Scope) -> bool:
        headers = [value for name, value in scope["headers"] if name == b"authorization"]
        if len(headers) != 1:
            return False
        key = hmac.new(self._cache_key, headers[0], hashlib.sha256).digest()
        if key in self._accepted:
            self._accepted.move_to_end(key)
            return True
        pair = _basic_pair(headers[0])
        if pair is None:
            return False
        if not await anyio.to_thread.run_sync(
            self._credentials.verify, *pair, limiter=self._limiter
        ):
            return False
        self._accepted[key] = None
        if len(self._accepted) > ACCEPTED_CACHE_SIZE:
            self._accepted.popitem(last=False)
        return True
