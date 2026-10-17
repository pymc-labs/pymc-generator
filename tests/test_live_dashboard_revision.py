"""Git provenance of a run: clean commits, dirty trees and missing git.

A run manifest records the revision it was generated from; two different dirty trees on
the same commit must not be recorded as the same revision.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from live_dashboard.revision import code_revision


def _load(name: str) -> ModuleType:
    """Load the plain helper module ``tests/<name>.py`` once (``tests`` is not a package)."""
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)
    # Neither the developer's git configuration nor an enclosing repository may leak in.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Live Dashboard Tests")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "tracked.txt").write_text("one\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _dirty_digest(revision: str | None, head: str) -> str:
    assert revision is not None
    match = re.fullmatch(rf"{head}\+dirty\.([0-9a-f]{{12}})", revision)
    assert match is not None, revision
    return match.group(1)


def test_a_clean_checkout_is_its_head_commit(repo):
    head = _git(repo, "rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert code_revision(repo) == head


def test_different_dirty_trees_get_different_revisions(repo):
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("two\n")
    edited = _dirty_digest(code_revision(repo), head)
    assert _dirty_digest(code_revision(repo), head) == edited  # stable for the same tree

    (repo / "untracked.txt").write_text("new\n")
    with_untracked = _dirty_digest(code_revision(repo), head)
    assert with_untracked != edited

    (repo / "tracked.txt").write_text("three\n")
    assert _dirty_digest(code_revision(repo), head) != with_untracked

    (repo / "tracked.txt").write_text("one\n")
    (repo / "untracked.txt").unlink()
    assert code_revision(repo) == head


def test_an_untracked_file_alone_makes_the_tree_dirty(repo):
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("draft\n")
    _dirty_digest(code_revision(repo), head)


def test_without_git_or_a_repository_there_is_no_revision(repo, tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert code_revision(plain) is None
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert code_revision(repo) is None
