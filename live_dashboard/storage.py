"""The small storage interface behind publication: a directory of immutable files.

The same implementation serves a local folder and a mounted Modal Volume. On Modal both
sides pass ``on_refresh=volume.reload`` (an already running container sees later commits
only after a reload) and the writer also passes ``on_commit=volume.commit`` (its changes
become visible to other containers only after a commit). Locally both hooks are no-ops.

Paths are relative POSIX paths. Names starting with ``.`` are reserved for staging
files: writes never target them and listings never return them, so a partially written
file that a background commit happens to persist is never mistaken for a publication.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path, PurePosixPath


class StoragePathError(ValueError):
    """A store path is absolute, escapes the root, or names a reserved staging file."""


def _segments(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise StoragePathError(f"invalid store path {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise StoragePathError(f"store paths are relative, got {path!r}")
    parts = path.split("/")
    for part in parts:
        if part in ("", ".", "..") or part.startswith("."):
            raise StoragePathError(f"invalid store path {path!r}")
    return tuple(parts)


class DirectoryStore:
    """Publication storage rooted at ``root``.

    ``write_new`` creates immutable files (records, payloads) and refuses to overwrite;
    ``write_replace`` replaces a file its single owner rewrites (a provisional attempt
    record). Both stage the bytes under a hidden name, flush them to disk, then rename
    into place, so readers see either nothing or the complete file. ``write_new``'s
    existence check and rename are not atomic across processes: the protocol relies
    on one writer per run (see the README).
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        on_commit: Callable[[], None] | None = None,
        on_refresh: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self._on_commit = on_commit
        self._on_refresh = on_refresh

    def _resolve(self, path: str) -> Path:
        return self.root.joinpath(*_segments(path))

    def refresh(self) -> None:
        """Make the latest committed state visible to this process (reader side)."""
        if self._on_refresh is not None:
            self._on_refresh()

    def commit(self) -> None:
        """Persist and publish everything written so far (writer side)."""
        if self._on_commit is not None:
            self._on_commit()

    def exists(self, path: str) -> bool:
        return self.size(path) is not None

    def read_bytes(self, path: str) -> bytes:
        """The complete file; raises ``FileNotFoundError`` when absent."""
        return self._resolve(path).read_bytes()

    def size(self, path: str) -> int | None:
        """Size in bytes of the regular file ``path``; ``None`` when absent or not a file.

        Symbolic links are not followed: the store only consists of regular files.
        """
        try:
            status = os.stat(self._resolve(path), follow_symlinks=False)
        except OSError:
            return None
        return status.st_size if stat.S_ISREG(status.st_mode) else None

    def _children(self, prefix: str, *, directories: bool) -> list[str]:
        try:
            with os.scandir(self._resolve(prefix)) as iterator:
                entries = list(iterator)
        except (FileNotFoundError, NotADirectoryError):
            return []
        names = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                wanted = (
                    entry.is_dir(follow_symlinks=False)
                    if directories
                    else entry.is_file(follow_symlinks=False)
                )
            except OSError:
                continue
            if wanted:
                names.append(entry.name)
        return sorted(names)

    def list_files(self, prefix: str) -> list[str]:
        """Names of the regular files directly inside directory ``prefix`` (sorted)."""
        return self._children(prefix, directories=False)

    def list_dirs(self, prefix: str) -> list[str]:
        """Names of the directories directly inside directory ``prefix`` (sorted)."""
        return self._children(prefix, directories=True)

    def _stage(self, target: Path, data: bytes) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{secrets.token_hex(6)}.partial")
        with open(staging, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return staging

    def write_new(self, path: str, data: bytes) -> None:
        """Create ``path`` with ``data``; ``FileExistsError`` if it already exists."""
        target = self._resolve(path)
        if target.exists():
            raise FileExistsError(f"{path} already exists")
        staging = self._stage(target, data)
        try:
            if target.exists():
                raise FileExistsError(f"{path} already exists")
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)

    def write_replace(self, path: str, data: bytes) -> None:
        """Create or replace ``path`` with ``data``."""
        target = self._resolve(path)
        staging = self._stage(target, data)
        try:
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
