"""Git provenance of the source tree (stdlib only; provenance, never identity)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=30
    ).stdout


def code_revision(repo: Path) -> str | None:
    """``HEAD`` of ``repo``, plus ``+dirty.<digest>`` when the tree has local changes.

    The digest covers ``git diff HEAD`` and the names of untracked files, so two
    different dirty trees on the same commit get different revisions. ``None`` when git
    or the repository is unavailable (e.g. inside a Modal container).
    """
    try:
        head = _git(repo, "rev-parse", "HEAD").decode().strip()
        if not _git(repo, "status", "--porcelain", "--untracked-files=all").strip():
            return head
        changes = _git(repo, "diff", "HEAD", "--binary")
        untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    digest = hashlib.sha256(changes + b"\0" + untracked).hexdigest()[:12]
    return f"{head}+dirty.{digest}"
