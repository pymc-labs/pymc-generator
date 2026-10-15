"""Offline tests of mounted-Volume barriers; not Modal platform evidence."""

import pytest
from test_publication import corpus as generated_corpus
from test_publication import metadata

from pymc_generator.publication import (
    MountedVolumePublicationStore,
    PublicationUnconfirmedError,
)

corpus = generated_corpus


class RecordingVolume:
    def __init__(self, root, fail_commit=None):
        self.root = root
        self.events = []
        self.commits = []
        self.fail_commit = fail_commit
        self.fail_reload = False

    def reload(self):
        self.events.append("reload")
        if self.fail_reload:
            raise RuntimeError("volume busy")

    def commit(self):
        self.events.append("commit")
        self.commits.append(
            {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
        )
        if len(self.commits) == self.fail_commit:
            raise RuntimeError("lost commit acknowledgment")


def test_payload_commit_precedes_completion_and_reads_refresh(tmp_path, corpus):
    volume = RecordingVolume(tmp_path)
    store = MountedVolumePublicationStore(tmp_path, volume)
    reference = store.publish(metadata(), corpus)
    assert volume.events == ["reload", "commit", "commit"]
    assert len(volume.commits) == 2
    assert any(p.endswith("corpus.npz") for p in volume.commits[0])
    assert not any(p.endswith("completion.json") for p in volume.commits[0])
    assert reference in volume.commits[1]
    store.read_output(reference)
    assert volume.events[-1] == "reload"
    snapshot = store.freeze_snapshot("run1", "training")
    assert snapshot in volume.commits[-1]
    before = len(volume.events)
    assert len(store.read_snapshot(snapshot)) == 1
    assert volume.events[before:] == ["reload", "reload"]


def test_payload_commit_failure_cannot_create_completion(tmp_path, corpus):
    volume = RecordingVolume(tmp_path, fail_commit=1)
    store = MountedVolumePublicationStore(tmp_path, volume)
    with pytest.raises(PublicationUnconfirmedError, match="Payload"):
        store.publish(metadata(), corpus)
    assert not list(tmp_path.rglob("completion.json"))
    assert store.discover("run1") == ([], {})


def test_completion_ack_failure_requires_reconciliation(tmp_path, corpus):
    volume = RecordingVolume(tmp_path, fail_commit=2)
    store = MountedVolumePublicationStore(tmp_path, volume)
    with pytest.raises(PublicationUnconfirmedError, match="Completion"):
        store.publish(metadata(), corpus)
    # An acknowledgment failure does not prove absence. This fake models the
    # case where the write is visible; reconcile and reuse rather than duplicate.
    reference = store.validated_outputs("run1")[0]["reference"]
    assert store.publish(metadata(attempt="retry"), corpus) == reference
    assert len(volume.commits) == 2
    assert len(list(tmp_path.rglob("completion.json"))) == 1


def test_zero_world_completion_and_snapshot_commit_failures(tmp_path):
    volume = RecordingVolume(tmp_path)
    store = MountedVolumePublicationStore(tmp_path, volume)
    store.publish(metadata(n=0), None)
    assert len(volume.commits) == 1
    assert not list(tmp_path.rglob("*.npz"))
    volume.fail_commit = 2
    with pytest.raises(PublicationUnconfirmedError, match="Snapshot"):
        store.freeze_snapshot("run1", "training")
    # A possibly persisted snapshot must never be overwritten after failure.
    with pytest.raises(FileExistsError):
        store.freeze_snapshot("run1", "training")


def test_refresh_failure_does_not_return_stale_success_or_write(tmp_path):
    volume = RecordingVolume(tmp_path)
    volume.fail_reload = True
    store = MountedVolumePublicationStore(tmp_path, volume)
    with pytest.raises(RuntimeError, match="volume busy"):
        store.discover("run1")
    with pytest.raises(RuntimeError, match="volume busy"):
        store.publish(metadata(n=0), None)
    assert not list(tmp_path.rglob("completion.json"))
    assert not volume.commits
