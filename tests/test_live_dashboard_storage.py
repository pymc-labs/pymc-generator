"""DirectoryStore: create-only publication, staging invisibility and path containment."""

from __future__ import annotations

import os

import pytest

from live_dashboard.storage import DirectoryStore, StoragePathError


def test_write_new_never_overwrites_and_leaves_no_staging_files(tmp_path):
    store = DirectoryStore(tmp_path)
    store.write_new("runs/r/tasks/t/a.json", b"first")
    with pytest.raises(FileExistsError):
        store.write_new("runs/r/tasks/t/a.json", b"second")
    assert store.read_bytes("runs/r/tasks/t/a.json") == b"first"
    assert os.listdir(tmp_path / "runs/r/tasks/t") == ["a.json"]


def test_write_replace_replaces_the_whole_file(tmp_path):
    store = DirectoryStore(tmp_path)
    store.write_replace("runs/r/attempts/a.json", b"running")
    store.write_replace("runs/r/attempts/a.json", b"done")
    assert store.read_bytes("runs/r/attempts/a.json") == b"done"
    assert os.listdir(tmp_path / "runs/r/attempts") == ["a.json"]


def test_listings_skip_staging_names_symlinks_and_missing_directories(tmp_path):
    store = DirectoryStore(tmp_path)
    store.write_new("runs/r/tasks/t/a.json", b"{}")
    (tmp_path / "runs/r/tasks/t/.a.json.1234.partial").write_bytes(b"{")
    (tmp_path / "runs/r/tasks/t/link.json").symlink_to(tmp_path / "runs/r/tasks/t/a.json")
    (tmp_path / "runs/linked").symlink_to(tmp_path / "runs/r", target_is_directory=True)
    assert store.list_files("runs/r/tasks/t") == ["a.json"]
    assert store.list_dirs("runs") == ["r"]
    assert store.list_files("runs/missing") == []
    assert store.list_dirs("nothing/here") == []


def test_size_reports_only_regular_files(tmp_path):
    store = DirectoryStore(tmp_path)
    store.write_new("runs/r/run.json", b"12345")
    (tmp_path / "runs/r/alias.json").symlink_to(tmp_path / "runs/r/run.json")
    assert store.size("runs/r/run.json") == 5
    assert store.size("runs/r") is None
    assert store.size("runs/r/alias.json") is None
    assert store.size("runs/r/absent.json") is None
    assert not store.exists("runs/r")


@pytest.mark.parametrize(
    "path", ["", "/etc/passwd", "../x", "runs/../x", "runs//x", "runs/.hidden", "a\\b", "a/./b"]
)
def test_paths_outside_the_store_or_reserved_names_are_rejected(tmp_path, path):
    store = DirectoryStore(tmp_path)
    with pytest.raises(StoragePathError):
        store.write_new(path, b"x")
