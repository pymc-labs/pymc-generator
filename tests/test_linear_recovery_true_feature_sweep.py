import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import scripts.linear_recovery_true_feature_sweep as sweep
from pymc_generator import make_scm_prior, sample_scm
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
from pymc_generator.slots import TRAJECTORY_COMPONENTS
from scripts.linear_recovery_true_feature_sweep import (
    count_stream,
    fit_design,
    frozen_schedule,
    validate_config,
    wilson,
)


def config():
    return json.load(open("docs/examples/data/linear-recovery-true-feature-sweep-config.json"))


def test_config_and_deterministic_natural_count_schedule():
    cfg = validate_config(config())
    first = count_stream(cfg, phase="science")
    assert first == count_stream(cfg, phase="science")
    assert first == sweep.study_schedules(cfg)["science"]
    assert count_stream(cfg, phase="calibration") == sweep.study_schedules(cfg)["calibration"]
    assert all(1 <= x["n_treatments"] <= 10 and 1 <= x["n_covariates"] <= 10 for x in first)
    assert {x["n_treatments"] for x in first} != {10}


def test_config_schema_rejects_missing_unknown_invalid_and_nonfinite():
    cfg = config()
    del cfg["population"]
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config()
    cfg["study"]["surprise"] = 1
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config()
    cfg["study"]["rank_tolerance"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config()
    cfg["study"]["easy_condition_threshold"] = float("nan")
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config()
    cfg["population"]["count_ranges"]["n_latent"] = [1, 2]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_external_archive_noclobber_world_digest_and_catalog(tmp_path):
    cfg = config()
    outside = tmp_path / "external"
    archive, metadata = sweep.create_run_archive(
        outside,
        cfg,
        environment={"python": "test"},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    assert metadata["catalog_schema_sha256"] == sweep.CATALOG_SCHEMA_SHA256
    with pytest.raises(FileExistsError):
        sweep.create_run_archive(
            outside,
            cfg,
            environment={},
            source_hashes=sweep.analysis_source_hashes(),
            seed_manifest=sweep.study_schedules(cfg),
        )
    record = sweep.persist_world(
        archive, {"attempt": 0}, {"status": {"generation": "failed"}, "why": "fixture"}
    )
    saved = json.loads((archive / record["world_locator"]).read_text())
    assert saved["attempt"] == {"attempt": 0}
    assert (
        record["world_sha256"]
        == hashlib.sha256((archive / record["world_locator"]).read_bytes()).hexdigest()
    )
    # Failed records remain valid persistence evidence, never fabricated sampled truth.
    with pytest.raises(ValueError):
        sweep.catalog_entry(archive=archive, world_reference=record)
    with pytest.raises(TypeError):
        sweep.catalog_entry(source_digest="s", config_digest="c", y1=[3.0])
    with pytest.raises(ValueError, match="nonfinite"):
        sweep.persist_world(archive, {"attempt": 1}, {"bad": np.array([np.nan])})


def test_numpy_world_roundtrip_and_logical_attempt_noclobber(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    result = sweep.analyze_world(controlled_world())
    result["empty"] = np.empty((3, 0), dtype=np.float32)
    result["scalar"] = np.int16(7)
    saved = sweep.persist_world(archive, {"attempt": 9}, result)
    restored = sweep.read_world(archive, saved)
    np.testing.assert_array_equal(restored["result"]["empty"], result["empty"])
    assert restored["result"]["empty"].dtype == np.float32
    assert restored["result"]["scalar"] == np.int16(7)
    assert (
        restored["result"]["fits"]["y0"]["coefficients"].dtype
        == result["fits"]["y0"]["coefficients"].dtype
    )
    before = (archive / saved["world_locator"]).read_bytes()
    with pytest.raises(FileExistsError):
        sweep.persist_world(archive, {"attempt": 9}, {"different": True})
    assert (archive / saved["world_locator"]).read_bytes() == before


def test_read_world_rejects_tampering(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    record = sweep.persist_world(archive, {"attempt": 0}, {"a": np.arange(3)})
    with (archive / record["world_locator"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest"):
        sweep.read_world(archive, record)


def test_config_rejects_mutated_grounded_threshold():
    cfg = config()
    cfg["study"]["tolerances"]["identity"] = 1e-5
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_fit_and_rank_accounting():
    x = np.column_stack([np.arange(8.0), np.ones(8)])
    y = 2 * x[:, 0] + 3
    result = fit_design(x, y, 1e-10)
    assert result["rank"] == result["p"] == 2
    np.testing.assert_allclose(result["coefficients"], [2, 3])
    deficient = fit_design(np.ones((8, 2)), y, 1e-10)
    assert deficient["rank"] < deficient["p"]


def test_wilson_denominators_and_budget_manifest():
    assert wilson(0, 0)["denominator"] == 0
    assert wilson(3, 5)["numerator"] == 3
    result = frozen_schedule(config(), 100, 60)
    assert result["frozen_attempt_count"] == len(result["manifest"])
    assert result["formula"] == "min(1000,max(0,floor((5400-E-300)/(2*max(C0,C1,C2)))))"


def _archive_inventory(archive):
    return {
        str(path.relative_to(archive)): (
            ("symlink", str(path.readlink()))
            if path.is_symlink()
            else ("directory", None)
            if path.is_dir()
            else ("file", path.read_bytes())
        )
        for path in archive.rglob("*")
    }


def _assert_recursive_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, np.generic):
        assert type(actual) is type(expected)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_recursive_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for restored, original in zip(actual, expected):
            _assert_recursive_equal(restored, original)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def _reseal_fixture(archive, reference, body):
    """Deliberately trust a malformed fixture to exercise validation beyond its outer hashes."""
    reference = dict(reference)
    payload = (sweep.canonical_json(body) + "\n").encode()
    reference["world_sha256"] = hashlib.sha256(payload).hexdigest()
    reference["world_id"] = hashlib.sha256(
        b"ols-world-persistence/v1/world\0" + payload
    ).hexdigest()
    reference["world_locator"] = f"world-{reference['world_id']}.json"
    (archive / reference["world_locator"]).write_bytes(payload)
    reservation = {key: value for key, value in reference.items() if key != "reservation_sha256"}
    raw = (sweep.canonical_json(reservation) + "\n").encode()
    (archive / reference["reservation_locator"]).write_bytes(raw)
    reference["reservation_sha256"] = hashlib.sha256(raw).hexdigest()
    return reference


class TestWorldPersistence:
    def test_real_analyze_world_roundtrip(self, tmp_path):
        result = sweep.analyze_world(controlled_world())
        assert result["status"]["truth_eligibility"] == "eligible"
        result["extra"] = {
            "empty": np.empty((3, 0), dtype=np.float32),
            "scalar": np.int16(7),
            "float": np.float32(-0.0),
            "zero_dim": np.array(3, dtype=np.int8),
            "noncontiguous": np.arange(20, dtype=">i4").reshape(4, 5)[:, ::2],
        }
        attempt = {"attempt": np.int64(3), "seed": np.uint32(8)}
        reference = sweep.persist_world(tmp_path, attempt, result)
        restored = sweep.read_world(tmp_path, reference)
        _assert_recursive_equal(restored["attempt"], attempt)
        _assert_recursive_equal(restored["result"], result)

    @pytest.mark.parametrize(
        "bad",
        [
            {"bad": float("inf")},
            {"bad": np.array([np.nan])},
            {"bad": object()},
            {1: "non-string key"},
            {"nested": {False: 1}},
            {"bad": {1, 2}},
            {"bad": np.array([object()], dtype=object)},
            {"__ndarray_ref__": {}},
        ],
    )
    def test_prepare_is_side_effect_free(self, tmp_path, bad):
        before = _archive_inventory(tmp_path)
        with pytest.raises(ValueError):
            sweep.persist_world(tmp_path, {"attempt": 0}, {"valid": np.arange(3), **bad})
        assert _archive_inventory(tmp_path) == before
        with pytest.raises(ValueError):
            sweep.persist_world(tmp_path, {"attempt": 0, "bad": bad}, {"valid": np.arange(3)})
        assert _archive_inventory(tmp_path) == before
        sweep._prepare_world(tmp_path, {"attempt": 0}, {"valid": np.arange(3)})
        assert _archive_inventory(tmp_path) == before

    @pytest.mark.parametrize("index", [None, True, np.bool_(False), -1, np.int64(-2), 1.0, "1"])
    def test_invalid_logical_attempt_is_side_effect_free(self, tmp_path, index):
        before = _archive_inventory(tmp_path)
        with pytest.raises(ValueError, match="attempt index"):
            sweep.persist_world(tmp_path, {"attempt": index}, {"array": np.arange(3)})
        assert _archive_inventory(tmp_path) == before

    def test_preflight_existing_blob_and_permissions(self, tmp_path, monkeypatch):
        result = {"array": np.arange(3)}
        _, _, _, _, blobs = sweep._prepare_world(tmp_path, {"attempt": 0}, result)
        (tmp_path / "arrays").mkdir()
        for locator in blobs:
            (tmp_path / locator).write_bytes(b"wrong")
        before = _archive_inventory(tmp_path)
        with pytest.raises(ValueError, match="tampering"):
            sweep.persist_world(tmp_path, {"attempt": 0}, result)
        assert _archive_inventory(tmp_path) == before
        monkeypatch.setattr(sweep.os, "access", lambda *args: False)
        with pytest.raises(PermissionError):
            sweep.persist_world(tmp_path, {"attempt": 1}, {})
        assert _archive_inventory(tmp_path) == before

    def test_reservation_is_first_write(self, tmp_path, monkeypatch):
        writes = []
        original_write = sweep._write_world_file
        original_mkdir = sweep._world_array_directory

        def write(archive, locator, kind, payload):
            writes.append((kind, locator, _archive_inventory(archive)))
            return original_write(archive, locator, kind, payload)

        def mkdir(archive):
            writes.append(("directory", "arrays", _archive_inventory(archive)))
            return original_mkdir(archive)

        monkeypatch.setattr(sweep, "_write_world_file", write)
        monkeypatch.setattr(sweep, "_world_array_directory", mkdir)
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {"a": np.arange(3)})
        assert writes[0] == ("reservation", reference["reservation_locator"], {})
        assert [entry[0] for entry in writes] == ["reservation", "directory", "array", "world"]

    def test_logical_attempt_identity_and_duplicate_preservation(self, tmp_path):
        attempt = {"attempt": np.int64(9), "world_seed": 12, "metadata": "first"}
        reference = sweep.persist_world(tmp_path, attempt, {"array": np.arange(3)})
        expected_id = hashlib.sha256(b"ols-world-persistence/v1/attempt\0" + b"9").hexdigest()
        assert reference["attempt_id"] == expected_id
        before = _archive_inventory(tmp_path)
        for changed in [
            {"attempt": 9, "world_seed": 99, "metadata": "different"},
            {"attempt": np.int32(9), "other": [1, 2]},
            {"attempt": 9},
        ]:
            assert sweep._logical_attempt_id(changed) == expected_id
            with pytest.raises(FileExistsError):
                sweep.persist_world(
                    tmp_path, changed, {"new": np.arange(100), "status": "different"}
                )
            assert _archive_inventory(tmp_path) == before

    def test_exact_canonical_byte_and_domain_digests(self, tmp_path):
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {"v": np.array([1])})
        payload = (tmp_path / reference["world_locator"]).read_bytes()
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
        assert payload == (sweep.canonical_json(json.loads(payload)) + "\n").encode()
        assert reference["world_sha256"] == hashlib.sha256(payload).hexdigest()
        assert (
            reference["world_id"]
            == hashlib.sha256(b"ols-world-persistence/v1/world\0" + payload).hexdigest()
        )
        assert reference["world_locator"] == f"world-{reference['world_id']}.json"
        raw = (tmp_path / reference["reservation_locator"]).read_bytes()
        assert reference["reservation_sha256"] == hashlib.sha256(raw).hexdigest()
        assert json.loads(raw) == {
            key: value for key, value in reference.items() if key != "reservation_sha256"
        }

    def test_reader_requires_trusted_reference(self, tmp_path):
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {"value": 1})
        with pytest.raises(TypeError):
            sweep.read_world(tmp_path)
        with pytest.raises(ValueError, match="trusted"):
            sweep.read_world(tmp_path, reference["world_locator"])
        with pytest.raises(TypeError):
            sweep.read_world(tmp_path, reference["world_locator"], reference["world_sha256"])
        with pytest.raises(TypeError):
            sweep.read_world(tmp_path, reference["world_locator"], expected_digest=None)
        for bad in [
            dict(reference, extra=True),
            {k: v for k, v in reference.items() if k != "reservation_sha256"},
        ]:
            with pytest.raises(ValueError, match="trusted"):
                sweep.read_world(tmp_path, bad)
        body = json.loads((tmp_path / reference["world_locator"]).read_bytes())
        body["result"]["value"] = 2
        raw = (sweep.canonical_json(body) + "\n").encode()
        (tmp_path / reference["world_locator"]).write_bytes(raw)
        with pytest.raises(ValueError, match="digest"):
            sweep.read_world(tmp_path, reference)
        # Even updating ordinary hashes and reservation cannot retain the trusted world ID.
        changed = dict(reference, world_sha256=hashlib.sha256(raw).hexdigest())
        reservation = {key: value for key, value in changed.items() if key != "reservation_sha256"}
        raw = (sweep.canonical_json(reservation) + "\n").encode()
        (tmp_path / reference["reservation_locator"]).write_bytes(raw)
        changed["reservation_sha256"] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError, match="world domain digest"):
            sweep.read_world(tmp_path, changed)

    @pytest.mark.parametrize(
        "case",
        [
            "missing_blob",
            "changed_blob",
            "data_hash",
            "dtype",
            "shape",
            "nbytes",
            "unknown",
            "missing",
            "version",
            "locator",
            "negative_shape",
            "bool_nbytes",
        ],
    )
    def test_array_reference_and_blob_integrity(self, tmp_path, case):
        reference = sweep.persist_world(
            tmp_path,
            {"attempt": 0},
            {
                "array": np.arange(3, dtype=np.int64),
                "empty": np.empty((3, 0), dtype=np.float32),
            },
        )
        restored = sweep.read_world(tmp_path, reference)
        assert restored["result"]["empty"].shape == (3, 0)
        assert restored["result"]["empty"].dtype == np.float32
        body = json.loads((tmp_path / reference["world_locator"]).read_bytes())
        ref = body["result"]["array"]["__ndarray_ref__"]
        assert set(ref) == {"version", "locator", "data_sha256", "dtype", "shape", "nbytes"}
        if case == "missing_blob":
            (tmp_path / ref["locator"]).unlink()
        elif case == "changed_blob":
            (tmp_path / ref["locator"]).write_bytes(b"x" * ref["nbytes"])
        else:
            if case == "missing":
                del ref["dtype"]
            else:
                field, value = {
                    "data_hash": ("data_sha256", "0" * 64),
                    "dtype": ("dtype", "<f4"),
                    "shape": ("shape", [4]),
                    "nbytes": ("nbytes", 25),
                    "unknown": ("extra", 1),
                    "version": ("version", "unsupported"),
                    "locator": ("locator", "../escape.bin"),
                    "negative_shape": ("shape", [-3]),
                    "bool_nbytes": ("nbytes", True),
                }[case]
                ref[field] = value
            reference = _reseal_fixture(tmp_path, reference, body)
        with pytest.raises((ValueError, FileNotFoundError)):
            sweep.read_world(tmp_path, reference)

    @pytest.mark.parametrize(
        "locator",
        [
            "/tmp/world.json",
            "../world.json",
            "./world.json",
            "arrays/../world.json",
            "world-" + "0" * 64 + ".json.bak",
            "nested/world-" + "0" * 64 + ".json",
            "world-" + "A" * 64 + ".json",
        ],
    )
    def test_locator_and_archive_confinement(self, tmp_path, locator):
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {})
        before = _archive_inventory(tmp_path)
        with pytest.raises(ValueError):
            sweep.read_world(tmp_path, dict(reference, world_locator=locator))
        with pytest.raises(ValueError):
            sweep._world_path(tmp_path, locator, "world")
        with pytest.raises(ValueError):
            sweep._world_path(tmp_path, locator, "array")
        with pytest.raises(ValueError, match="outside checkout"):
            sweep.persist_world(Path(sweep.__file__).resolve().parents[1], {"attempt": 0}, {})
        assert _archive_inventory(tmp_path) == before

    @pytest.mark.parametrize(
        "case", ["archive", "ancestor", "world", "reservation", "blob", "arrays"]
    )
    def test_symlink_confinement(self, tmp_path, case):
        archive, outside = tmp_path / "archive", tmp_path / "outside"
        archive.mkdir()
        outside.mkdir()
        (outside / "sentinel").write_bytes(b"unchanged")
        result = {"array": np.arange(3)}
        reference = sweep.persist_world(archive, {"attempt": 0}, result)
        if case in {"archive", "ancestor"}:
            link = tmp_path / "link"
            link.symlink_to(archive if case == "archive" else tmp_path, target_is_directory=True)
            archive = link if case == "archive" else link / "archive"
        elif case == "arrays":
            (archive / "arrays").rename(outside / "blobs")
            (archive / "arrays").symlink_to(outside / "blobs", target_is_directory=True)
        else:
            if case == "blob":
                path = next((archive / "arrays").iterdir())
            else:
                path = archive / reference[f"{case}_locator"]
            dest = outside / path.name
            path.rename(dest)
            path.symlink_to(dest)
        before = _archive_inventory(outside)
        with pytest.raises(ValueError):
            sweep.read_world(archive, reference)
        if case in {"archive", "ancestor", "arrays", "blob"}:
            with pytest.raises(ValueError):
                sweep.persist_world(archive, {"attempt": 1}, result)
        assert _archive_inventory(outside) == before

    @pytest.mark.parametrize(
        "case",
        [
            "missing_world",
            "missing_reservation",
            "world_bytes",
            "reservation_bytes",
            "world_noncanonical",
            "reservation_noncanonical",
            "linkage",
            "digest",
        ],
    )
    def test_missing_and_corrupt_world_or_reservation(self, tmp_path, case):
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {})
        world = tmp_path / reference["world_locator"]
        reservation = tmp_path / reference["reservation_locator"]
        if case.startswith("missing"):
            (world if case == "missing_world" else reservation).unlink()
        elif case == "world_bytes":
            world.write_bytes(b"corrupt")
        elif case == "reservation_bytes":
            reservation.write_bytes(b"corrupt")
        elif case == "world_noncanonical":
            raw = world.read_bytes() + b" "
            reference["world_id"] = hashlib.sha256(
                b"ols-world-persistence/v1/world\0" + raw
            ).hexdigest()
            reference["world_sha256"] = hashlib.sha256(raw).hexdigest()
            reference["world_locator"] = f"world-{reference['world_id']}.json"
            (tmp_path / reference["world_locator"]).write_bytes(raw)
            saved = {key: value for key, value in reference.items() if key != "reservation_sha256"}
            reservation.write_bytes((sweep.canonical_json(saved) + "\n").encode())
            reference["reservation_sha256"] = hashlib.sha256(reservation.read_bytes()).hexdigest()
        elif case in {"reservation_noncanonical", "linkage"}:
            saved = json.loads(reservation.read_bytes())
            if case == "linkage":
                saved["world_id"] = "0" * 64
            reservation.write_bytes(
                (
                    sweep.canonical_json(saved)
                    + (" " if case == "reservation_noncanonical" else "\n")
                ).encode()
            )
            reference["reservation_sha256"] = hashlib.sha256(reservation.read_bytes()).hexdigest()
        else:
            reference["reservation_sha256"] = "0" * 64
        with pytest.raises((ValueError, FileNotFoundError)):
            sweep.read_world(tmp_path, reference)

    def test_failed_record_roundtrip(self, tmp_path):
        result = {"status": {"generation": "failed"}, "reason": "generation_failed: fixture"}
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, result)
        assert sweep.read_world(tmp_path, reference)["result"] == result
        assert not (tmp_path / "arrays").exists()

    def test_unattempted_record_roundtrip(self, tmp_path):
        result = {"status": {"scheduling": "unattempted_budget"}, "reason": "clock exhausted"}
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, result)
        assert sweep.read_world(tmp_path, reference)["result"] == result
        assert not (tmp_path / "arrays").exists()

    def test_writer_self_verifies(self, tmp_path, monkeypatch):
        calls = []
        original = sweep.read_world

        def reader(archive, reference):
            calls.append((archive, dict(reference)))
            return original(archive, reference)

        monkeypatch.setattr(sweep, "read_world", reader)
        reference = sweep.persist_world(tmp_path, {"attempt": 0}, {})
        assert calls == [(tmp_path, reference)]

        def fail(archive, reference):
            raise ValueError("injected readback failure")

        monkeypatch.setattr(sweep, "read_world", fail)
        with pytest.raises(ValueError, match="injected readback failure"):
            sweep.persist_world(tmp_path, {"attempt": 1}, {"a": np.arange(3)})
        before = _archive_inventory(tmp_path)
        assert len(list(tmp_path.glob("attempt-*.json"))) == 2
        assert len(list(tmp_path.glob("world-*.json"))) == 2
        with pytest.raises(FileExistsError):
            sweep.persist_world(tmp_path, {"attempt": 1, "changed": True}, {})
        assert _archive_inventory(tmp_path) == before

    def test_failed_write_preserves_partial_evidence(self, tmp_path, monkeypatch):
        original = sweep._write_world_file

        def fail(archive, locator, kind, payload):
            if kind == "world":
                raise OSError("injected disk failure")
            return original(archive, locator, kind, payload)

        monkeypatch.setattr(sweep, "_write_world_file", fail)
        with pytest.raises(OSError, match="injected disk failure"):
            sweep.persist_world(tmp_path, {"attempt": 0}, {"a": np.arange(3)})
        assert len(list(tmp_path.glob("attempt-*.json"))) == 1
        assert len(list((tmp_path / "arrays").glob("*.bin"))) == 1
        assert not list(tmp_path.glob("world-*.json"))
        before = _archive_inventory(tmp_path)
        with pytest.raises(FileExistsError):
            sweep.persist_world(tmp_path, {"attempt": 0}, {"a": np.arange(8)})
        assert _archive_inventory(tmp_path) == before

    def test_catalog_mutation_cannot_affect_world_persistence(self, tmp_path, monkeypatch):
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        result = {"a": np.arange(3)}
        reference = sweep.persist_world(first, {"attempt": 0}, result)
        with monkeypatch.context() as patch:
            patch.setattr(
                sweep, "CATALOG_SCHEMA", {"version": "mutated", "fields": {"bad": object()}}
            )
            patch.setattr(sweep, "CATALOG_SCHEMA_BYTES", b"changed")
            patch.setattr(sweep, "CATALOG_SCHEMA_SHA256", "changed")
            assert sweep.persist_world(second, {"attempt": 0}, result) == reference
            _assert_recursive_equal(sweep.read_world(second, reference)["result"], result)
        assert _archive_inventory(first) == _archive_inventory(second)


def _config_leaf_paths(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _config_leaf_paths(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _config_leaf_paths(item, (*path, i))
    else:
        yield path


@pytest.mark.parametrize("path", list(_config_leaf_paths(config())))
def test_every_frozen_config_leaf_is_strict(path):
    cfg = config()
    node = cfg
    for part in path[:-1]:
        node = node[part]
    original = node[path[-1]]
    node[path[-1]] = False if type(original) is not bool else 1
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["population"].update(retained_generator_axes=[42]),
        lambda c: c["study"]["categories"].append("easy"),
        lambda c: c["study"].update(easy_condition_threshold=1.0, hard_condition_threshold=2.0),
        lambda c: c["study"]["seed_streams"]["science"].update(world_seed=20261011),
        lambda c: c["study"].update(conservative_cost_rule="mean calibration duration"),
        lambda c: c["study"].update(frozen_count_formula="maximum_schedule"),
        lambda c: c["study"]["fit_rows"].reverse(),
        lambda c: c["study"]["example_selection"].update(rule="best recovered"),
        lambda c: c["study"].update(minimum_subgroup_size=1),
        lambda c: c["population"]["retained_generator_axes"]["resolved_prior"].update(
            rw_outcome_std_range=[0.0, 0.0]
        ),
        lambda c: c["population"]["retained_generator_axes"]["resolved_prior"][
            "carryover_family_probs"
        ].update(none=float("inf")),
        lambda c: c["population"].update(carryover_burn_in=True),
    ],
)
def test_config_rejects_contract_drift(mutation):
    cfg = config()
    mutation(cfg)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_frozen_count_world_schedules_and_prefit_selection():
    cfg = config()
    schedules = sweep.study_schedules(cfg)
    assert schedules == sweep.study_schedules(cfg)
    assert len(schedules["science"]) == 1000 and len(schedules["calibration"]) == 3
    all_seeds = [row["world_seed"] for rows in schedules.values() for row in rows]
    assert len(set(all_seeds)) == 1003
    assert [row["attempt"] for row in schedules["science"]] == list(range(1000))
    assert [row["attempt"] for row in schedules["calibration"]] == list(range(1000, 1003))
    assert len({row["attempt"] for rows in schedules.values() for row in rows}) == 1003
    for phase, rows in schedules.items():
        streams = cfg["study"]["seed_streams"][phase]
        pairs = (
            np.random.Generator(np.random.PCG64(streams["count_seed"])).integers(
                1, 11, size=(len(rows), 2)
            )
            if phase == "science"
            else np.full((3, 2), 10)
        )
        seeds = np.random.Generator(np.random.PCG64(streams["world_seed"])).integers(
            0, 2**32, size=len(rows), dtype=np.uint32
        )
        assert [(r["n_treatments"], r["n_covariates"]) for r in rows] == [
            tuple(pair) for pair in pairs
        ]
        assert [r["world_seed"] for r in rows] == list(seeds)
        assert all(r["n_latent"] == 1 for r in rows)
    selection = sweep.selection_manifest(cfg)
    seen = set()
    for row in selection["rows"]:
        pair = sweep.configured_band_pair(cfg, schedules["science"][row["attempt"]])
        assert row["example_selected"] == (pair not in seen)
        seen.add(pair)
        assert row["fit_rows"] == list(range(104))
        assert "configuration-only" in row["rationale"]
    cfg["study"]["seed_streams"]["calibration"]["count_seed"] = 20261009
    with pytest.raises(ValueError):
        sweep.study_schedules(cfg)


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -1, "2"])
def test_timing_rejects_nonfinite_boolean_negative(bad):
    with pytest.raises(ValueError):
        sweep.frozen_schedule(config(), bad, 2)
    with pytest.raises(ValueError):
        sweep.frozen_schedule(config(), 0, bad)
    with pytest.raises(ValueError):
        sweep.conservative_world_cost(config(), [1, 2, bad])


def test_exact_cost_and_inclusive_formula():
    cfg = config()
    assert sweep.conservative_world_cost(cfg, [1, 4, 3]) == 8
    for durations in ([], [1, 2], [1, 2, 0], [1, 2, 3, 4], [1.0, 2.0, 1e308]):
        with pytest.raises(ValueError):
            sweep.conservative_world_cost(cfg, durations)
    for elapsed, cost, count in [
        (0, 1, 1000),
        (5100, 1, 0),
        (5401, 1, 0),
        (5090, 3, 3),
        (5091, 3, 3),
    ]:
        result = frozen_schedule(cfg, elapsed, cost)
        assert result["frozen_attempt_count"] == count
        assert result["manifest"] == sweep.study_schedules(cfg)["science"][:count]
        assert result["formula"] == cfg["study"]["frozen_count_formula"]
        assert result["inputs"] == {
            "maximum_schedule": 1000,
            "wall_time_seconds": 5400,
            "elapsed_after_calibration": elapsed,
            "reserve": 300,
            "conservative_per_world_cost": cost,
        }


def test_generation_settings_bound_before_sampling(monkeypatch):
    import pymc_generator

    cfg = config()
    attempt = sweep.study_schedules(cfg)["science"][0]
    resolved = sweep.resolved_generation_config(cfg, attempt)
    assert resolved["rw_outcome_std_range"] == [0.01, 0.028]
    assert resolved["n_treatments_active_range"] == [attempt["n_treatments"]] * 2
    original = pymc_generator.make_scm_prior

    def drift(**kwargs):
        prior = original(**kwargs)
        prior.rw_outcome_std_range = (0.0, 0.0)
        return prior

    monkeypatch.setattr(pymc_generator, "make_scm_prior", drift)
    monkeypatch.setattr(
        sweep, "generate_world", lambda *a, **k: pytest.fail("sampling must not start on drift")
    )
    with pytest.raises(ValueError, match="resolved generation config"):
        sweep.generate_configured_world(cfg, attempt)


@pytest.fixture(scope="module")
def catalog_analysis_fixture():
    cfg = config()
    attempt = sweep.study_schedules(cfg)["science"][0]
    # Exactly one fixed controlled contract world, not calibration or a scientific sweep.
    world = sweep.generate_configured_world(cfg, attempt)
    result = sweep.analyze_configured_world(world, cfg, attempt)
    assert result["status"]["truth_eligibility"] == "eligible", result["reasons"]
    return cfg, attempt, world, result


def _catalog_fixture(tmp_path, catalog_analysis_fixture):
    cfg, attempt, _, result = catalog_analysis_fixture
    archive, metadata = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={"kind": "controlled-test-fixture"},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    world_ref = sweep.persist_world(archive, attempt, result)
    entry = sweep.catalog_entry(archive=archive, world_reference=world_ref)
    return archive, metadata, world_ref, entry


def test_real_analyzed_world_external_catalog_roundtrip(tmp_path, catalog_analysis_fixture):
    archive, metadata, world_ref, entry = _catalog_fixture(tmp_path, catalog_analysis_fixture)
    catalog_ref = sweep.persist_catalog_entry(archive, entry)
    assert sweep.read_catalog_entry(archive, catalog_ref) == entry
    assert entry["world_reference"] == world_ref
    assert entry["world_content_digest"] == world_ref["world_sha256"]
    assert entry["source_commit"] == sweep._source_commit()
    assert entry["analysis_file_hashes"] == sweep.analysis_source_hashes()
    assert entry["parent_id"] == hashlib.sha256((archive / "run.json").read_bytes()).hexdigest()
    assert (
        entry["selection_manifest_id"]
        == hashlib.sha256((archive / entry["selection_manifest_locator"]).read_bytes()).hexdigest()
    )
    assert entry["noise_replicate_id"] is None and entry["replicate_kind"] == "fresh_world"
    assert entry["pair_targets"] == ["y0", "y1"]
    assert entry["parameter_draw_id"] != entry["data_draw_id"] != entry["world_id"]
    known = entry["known_sampled_truth"]
    assert known["prior_cond"] is None and known["prior_cond_status"] == "unknown_not_retained"
    assert known["kind"] != entry["inferred_diagnostics"]["kind"]
    raw = json.loads((archive / world_ref["world_locator"]).read_bytes())
    for name in (
        "coefficients",
        "direct_structure",
        "design",
        "contributions_observed",
        "y1",
        "noise",
        "scales",
        "public_history",
        "rows",
        "row_mask",
    ):
        ref = known[name]
        node = raw
        for part in ref["path"]:
            node = node[part]
        assert ref["sha256"] == sweep.digest(node)
        assert ref["world_locator"] == world_ref["world_locator"]
        if ref["array"] is not None:
            assert ref["array"] == node["__ndarray_ref__"]
            assert (
                hashlib.sha256((archive / ref["array"]["locator"]).read_bytes()).hexdigest()
                == ref["array"]["data_sha256"]
            )
    assert metadata["catalog_schema_sha256"] == entry["schema_sha256"]
    before = _archive_inventory(archive)
    with pytest.raises(FileExistsError):
        sweep.persist_catalog_entry(archive, entry)
    assert _archive_inventory(archive) == before


def test_schema_bytes_hash_path_and_mutable_global_independence(tmp_path, monkeypatch):
    original = sweep.CATALOG_SCHEMA_BYTES
    view = sweep.catalog_schema()
    view["properties"].clear()
    assert sweep.catalog_schema()["properties"]
    with pytest.raises(TypeError):
        sweep.CATALOG_SCHEMA["title"] = "changed"
    with pytest.raises(TypeError):
        sweep.CATALOG_SCHEMA["properties"]["schema_version"]["enum"][0] = "changed"
    monkeypatch.setattr(sweep, "CATALOG_SCHEMA", {"bad": "not authoritative"})
    monkeypatch.setattr(sweep, "CATALOG_SCHEMA_SHA256", "stale")
    cfg = config()
    archive, metadata = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    assert (archive / "schemas/ols-world-catalog-v1.json").read_bytes() == original
    assert metadata["catalog_schema_sha256"] == hashlib.sha256(original).hexdigest()
    assert sweep.CATALOG_SCHEMA_BYTES == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e.update(world_id="s"),
        lambda e: e.update(source_commit="0" * 40),
        lambda e: e.update(resolved_config_digest="0" * 64),
        lambda e: e.update(generation_seed=True),
        lambda e: e.update(parameter_draw_id=e["data_draw_id"]),
        lambda e: e.update(data_draw_id=e["world_id"]),
        lambda e: e.update(noise_replicate_id="replicate-1"),
        lambda e: e.update(replicate_kind="outcome_only"),
        lambda e: e.update(pair_targets=["y1", "y1"]),
        lambda e: e["known_sampled_truth"].update(y1=[1.0] * 104),
        lambda e: e["known_sampled_truth"].update(prior_cond={}),
        lambda e: e["known_sampled_truth"]["design"]["array"].update(shape=[104, 100]),
        lambda e: e["known_sampled_truth"]["y1"].update(world_locator="../outside.json"),
        lambda e: e["known_sampled_truth"]["coefficients"].update(
            path=["result", "fits", "y1", "coefficients"]
        ),
        lambda e: e["world_reference"].update(reservation_sha256="0" * 64),
        lambda e: e["selection"].update(example_selected=False),
        lambda e: e["selection"]["fit_rows"].reverse(),
        lambda e: e["inferred_diagnostics"]["labels"].update(conditioning="made_up"),
        lambda e: e.update(unknown="field"),
        lambda e: e.pop("source_id"),
    ],
)
def test_catalog_strict_types_locators_shapes_ids_and_bindings(
    tmp_path, catalog_analysis_fixture, mutation
):
    archive, _, _, entry = _catalog_fixture(tmp_path, catalog_analysis_fixture)
    mutation(entry)
    before = _archive_inventory(archive)
    with pytest.raises(ValueError):
        sweep.persist_catalog_entry(archive, entry)
    assert _archive_inventory(archive) == before


@pytest.mark.parametrize(
    "kind", ["schema", "run", "selection", "catalog", "world", "reservation", "blob"]
)
@pytest.mark.parametrize("change", ["tamper", "missing", "symlink"])
def test_catalog_reopen_rejects_tampered_missing_or_symlinked_evidence(
    tmp_path, catalog_analysis_fixture, kind, change
):
    archive, _, world_ref, entry = _catalog_fixture(tmp_path, catalog_analysis_fixture)
    catalog_ref = sweep.persist_catalog_entry(archive, entry)
    locators = {
        "schema": entry["schema_path"],
        "run": "run.json",
        "selection": entry["selection_manifest_locator"],
        "catalog": catalog_ref["locator"],
        "world": world_ref["world_locator"],
        "reservation": world_ref["reservation_locator"],
        "blob": entry["known_sampled_truth"]["design"]["array"]["locator"],
    }
    path = archive / locators[kind]
    if change == "tamper":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        payload = path.read_bytes()
        path.unlink()
        if change == "symlink":
            outside = tmp_path / "outside"
            outside.write_bytes(payload)
            path.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        sweep.read_catalog_entry(archive, catalog_ref)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(design=r["design"][:, :-1]),
        lambda r: r.update(truth=r["truth"][:-1]),
        lambda r: r["generation"].update(generation_seed=1),
        lambda r: r["source"]["data"].update(contributions_observed=np.zeros((1, 1))),
        lambda r: r["audit"].update(burn_in=8),
        lambda r: r["status"].update(
            conditioning="easy" if r["status"]["conditioning"] != "easy" else "hard"
        ),
        lambda r: r["fits"]["y0"].update(coefficients=r["truth"] + 1),
        lambda r: r.update(identities={}),
    ],
)
def test_catalog_rejects_even_resealed_malformed_analyzed_world(
    tmp_path, catalog_analysis_fixture, mutation
):
    cfg, attempt, _, original = catalog_analysis_fixture
    result = deepcopy(original)
    mutation(result)
    archive, _ = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    reference = sweep.persist_world(archive, attempt, result)
    with pytest.raises(ValueError):
        sweep.catalog_entry(archive=archive, world_reference=reference)


def test_archive_rejects_dummy_sources_and_wrong_schedule_without_writes(tmp_path):
    cfg = config()
    for hashes, seeds in [
        ({"script": "abc"}, sweep.study_schedules(cfg)),
        (sweep.analysis_source_hashes(), []),
    ]:
        before = _archive_inventory(tmp_path)
        with pytest.raises(ValueError):
            sweep.create_run_archive(
                tmp_path, cfg, environment={}, source_hashes=hashes, seed_manifest=seeds
            )
        assert _archive_inventory(tmp_path) == before


def test_calibrated_schedule_cost_binding_and_subgroup_unavailability():
    cfg = config()
    frozen = sweep.freeze_calibrated_schedule(cfg, 5000, [1, 4, 3])
    assert frozen["inputs"]["conservative_per_world_cost"] == 8
    assert frozen["frozen_attempt_count"] == 12
    assert frozen["calibration_durations"] == [1, 4, 3]
    assert frozen["cost_rule"] == cfg["study"]["conservative_cost_rule"]
    with pytest.raises(ValueError):
        sweep.freeze_calibrated_schedule(cfg, 7, [1, 4, 3])
    for count in (0, 1, 9):
        assert sweep.subgroup_available(cfg, count) is False
    for count in (10, 20):
        assert sweep.subgroup_available(cfg, count) is True
    for count in (True, -1, 1.0):
        with pytest.raises(ValueError):
            sweep.subgroup_available(cfg, count)
    assert "not a MAP allocation" in cfg["study"]["maximum_schedule_rationale"]


def test_count_stream_rejects_arbitrary_seed_and_double_authority():
    with pytest.raises(TypeError):
        count_stream(20261008, 20)
    with pytest.raises(ValueError):
        count_stream(20261008)
    with pytest.raises(TypeError):
        count_stream(config(), seed=20261008)
    with pytest.raises(TypeError):
        count_stream(config(), maximum=20)
    with pytest.raises(ValueError):
        count_stream(config(), phase="other")
    cfg = config()
    cfg["study"]["seed"] = 20261008
    with pytest.raises(ValueError, match="wrong frozen keys"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="wrong frozen keys"):
        count_stream(cfg)


def test_schedule_world_seed_collision_rejects_without_redraw(monkeypatch):
    calls = []

    class CollisionGenerator:
        def integers(self, low, high, *, size, dtype=None):
            calls.append((low, high, size))
            return np.ones(size, dtype=int) if dtype is None else np.zeros(size, dtype=dtype)

    monkeypatch.setattr(np.random, "Generator", lambda seed: CollisionGenerator())
    with pytest.raises(ValueError, match="world seed collision: never redraw"):
        sweep.study_schedules(config())
    assert calls == [(0, 2**32, 3), (1, 11, (1000, 2)), (0, 2**32, 1000)]


def test_phase_attempts_share_archive_without_collisions(tmp_path, catalog_analysis_fixture):
    cfg, science, _, science_result = catalog_analysis_fixture
    schedules = sweep.study_schedules(cfg)
    calibration = schedules["calibration"][0]
    assert science["attempt"] == 0
    assert calibration["attempt"] == cfg["study"]["maximum_schedule"]
    for phase, rows in schedules.items():
        for attempt in rows:
            sweep._scheduled_attempt(cfg, attempt, phase)
            with pytest.raises(ValueError):
                sweep._scheduled_attempt(
                    cfg, attempt, "calibration" if phase == "science" else "science"
                )
    for index in (-1, 0, 19, 23, True):
        with pytest.raises(ValueError):
            sweep._scheduled_attempt(cfg, dict(calibration, attempt=index), "calibration")
    # One fixed phase-binding fixture, not a timed calibration or study execution.
    world = sweep.generate_configured_world(cfg, calibration, phase="calibration")
    calibration_result = sweep.analyze_configured_world(
        world, cfg, calibration, phase="calibration"
    )
    archive, _ = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=schedules,
    )
    calibration_ref = sweep.persist_world(archive, calibration, calibration_result)
    science_ref = sweep.persist_world(archive, science, science_result)
    assert calibration_ref["attempt_id"] != science_ref["attempt_id"]
    for attempt, result, reference in (
        (calibration, calibration_result, calibration_ref),
        (science, science_result, science_ref),
    ):
        body = sweep.read_world(archive, reference)
        _assert_recursive_equal(body["attempt"], attempt)
        _assert_recursive_equal(body["result"], result)
        before = _archive_inventory(archive)
        with pytest.raises(FileExistsError):
            sweep.persist_world(archive, attempt, result)
        assert _archive_inventory(archive) == before
    entry = sweep.catalog_entry(archive=archive, world_reference=science_ref)
    assert entry["selection"]["attempt"] == science["attempt"]
    assert sweep.read_catalog_entry(archive, sweep.persist_catalog_entry(archive, entry)) == entry
    # Calibration remains excluded from the science-only MAP catalog.
    before = _archive_inventory(archive)
    with pytest.raises(ValueError):
        sweep.catalog_entry(archive=archive, world_reference=calibration_ref)
    assert _archive_inventory(archive) == before


_CATALOG_NUMERIC_PATHS = [
    *(
        ("fits", target, field)
        for target in ("y0", "y1")
        for field in (
            "rank",
            "singular_values",
            "condition",
            "p",
            "n_fit",
            "coefficients",
            "coefficient_error",
            "max_abs_error",
            "rmse",
            "residual",
        )
    ),
    *(
        (design, field)
        for design in ("raw_design", "standardized_design")
        for field in (
            "rank",
            "singular_values",
            "condition",
            "rank_tolerance",
            "rank_cutoff",
            "shape",
            "conditioning",
        )
    ),
    *(
        (field,)
        for field in (
            "p",
            "n_fit",
            "n_full",
            "rank_tolerance",
            "standardized_X",
            "column_means",
            "column_scales",
            "constant_columns",
        )
    ),
    *(
        ("target_metrics", target, scope, metric)
        for target in (
            "y0",
            "y1",
            "u",
            "outcome_noise",
            "raw_observed_unadjusted",
        )
        for scope in ("all", "fit")
        for metric in ("mean", "std", "max_abs")
    ),
    *(
        ("shapes", field)
        for field in (
            "treatments",
            "treatments_base",
            "covariates",
            "latent_unobserved",
            "contributions_observed",
            "contributions",
            "covariate_contribution",
            "latent_unobserved_contribution",
            "baseline_intrinsic",
            "outcome_noise",
            "outcome",
            "baseline",
            "indirect_effects",
            "indirect_effects_by_source",
            "saturation_scale",
        )
    ),
    *(
        ("targets", target)
        for target in ("y0", "y1", "u", "outcome_noise", "raw_observed_unadjusted")
    ),
]


@pytest.mark.parametrize("path", _CATALOG_NUMERIC_PATHS, ids=lambda p: ".".join(p))
def test_catalog_recomputes_every_retained_numeric_field(tmp_path, catalog_analysis_fixture, path):
    cfg, attempt, _, original = catalog_analysis_fixture
    result = deepcopy(original)
    node = result
    for part in path[:-1]:
        node = node[part]
    field = path[-1]
    value = node[field]
    if field == "rank":
        assert value > 0
        node[field] = 0
    elif field == "constant_columns":
        node[field] = np.array([result["p"] - 1], dtype=value.dtype)
    elif isinstance(value, tuple):
        node[field] = (value[0] + 1, *value[1:])
    elif isinstance(value, str):
        node[field] = "hard" if value != "hard" else "easy"
    else:
        node[field] = value + 1
    archive, _ = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    # Persisting the forged body refreshes every blob/world/reservation digest legitimately.
    reference = sweep.persist_world(archive, attempt, result)
    sweep.read_world(archive, reference)
    before = _archive_inventory(archive)
    with pytest.raises(ValueError):
        sweep.catalog_entry(archive=archive, world_reference=reference)
    assert _archive_inventory(archive) == before


@pytest.mark.parametrize(
    "probe",
    [
        "raw_singular_2x",
        "raw_cutoff_2x",
        "raw_both_2x",
        "standardized_both_2x",
        "coherent_fake_fit",
    ],
)
def test_catalog_rejects_resealed_crossfield_fabrication(tmp_path, catalog_analysis_fixture, probe):
    cfg, attempt, _, original = catalog_analysis_fixture
    result = deepcopy(original)
    if probe == "coherent_fake_fit":
        fitted = result["fits"]["y0"]
        fitted["coefficients"] = result["truth"] + 1
        error = fitted["coefficients"] - result["truth"]
        fitted.update(
            coefficient_error=error,
            max_abs_error=float(np.max(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))),
            residual=result["targets"]["y0"] - result["design"] @ fitted["coefficients"],
        )
    else:
        info = result["standardized_design" if probe == "standardized_both_2x" else "raw_design"]
        if probe != "raw_cutoff_2x":
            info["singular_values"] *= 2
        if probe != "raw_singular_2x":
            info["rank_cutoff"] *= 2
    archive, _ = sweep.create_run_archive(
        tmp_path,
        cfg,
        environment={},
        source_hashes=sweep.analysis_source_hashes(),
        seed_manifest=sweep.study_schedules(cfg),
    )
    reference = sweep.persist_world(archive, attempt, result)
    sweep.read_world(archive, reference)
    before = _archive_inventory(archive)
    with pytest.raises(ValueError):
        sweep.catalog_entry(archive=archive, world_reference=reference)
    assert _archive_inventory(archive) == before


def _completed_calibration(durations=(1, 4, 3)):
    return [
        {"attempt": row, "phase": "calibration", "status": "complete", "duration": duration}
        for row, duration in zip(
            sweep.study_schedules(config())["calibration"], durations, strict=True
        )
    ]


def test_scientific_input_v2_canonical_manifest_and_fixed_calibration():
    cfg = config()
    schedules = sweep.study_schedules(cfg)
    assert cfg["input_contract_version"] == "ols-scientific-input/v3"
    assert sweep.digest(cfg) == "e1f16af94ae3af632875df5ba7233f08aa9f5fa616529d25b6b9ac33d8655f0a"
    assert (
        sweep.digest(schedules)
        == "95dd82f378da6890b72e82e7ff931568219e10ce6f5a213fba334e219e7d3291"
    )
    assert schedules["calibration"] == [
        {"attempt": i, "n_treatments": 10, "n_covariates": 10, "n_latent": 1, "world_seed": seed}
        for i, seed in zip(range(1000, 1003), [2901121193, 1825999679, 2105574551], strict=True)
    ]
    assert cfg["study"]["seed_streams"]["calibration"] == {"world_seed": 20261011}
    assert cfg["study"]["seed_streams"]["science"] == {
        "count_seed": 20261008,
        "world_seed": 20261010,
    }
    assert len({r["attempt"] for phase in schedules.values() for r in phase}) == 1003


@pytest.mark.parametrize("count", [0, 1, 12, 399, 400, 999, 1000])
def test_input_prefix_selection_absent_bands_and_exact_nine_firsts(count):
    cfg = config()
    manifest = sweep.study_schedules(cfg)["science"][:count]
    selected = sweep.prefit_selection(cfg, manifest)
    assert sweep.validate_science_prefix(cfg, manifest) == manifest
    assert len(selected["rows"]) == count
    assert len(selected["band_examples"]) == 9
    expected_firsts = [0, 6, 1, 15, 14, 27, 5, 3, 4]
    for example, first in zip(selected["band_examples"], expected_firsts, strict=True):
        assert example["attempt"] == (first if first < count else None)
        assert example["availability_reason"] == (
            None if first < count else "band_absent_from_manifest"
        )
    chosen = [row["attempt"] for row in selected["rows"] if row["example_selected"]]
    assert len(chosen) == len(set(chosen))
    assert set(chosen) == {i for i in expected_firsts if i < count}
    # Prefix changes may remove later examples, but never reselect an earlier one.
    assert selected == sweep.prefit_selection(cfg, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows, cal: rows.reverse(),
        lambda rows, cal: rows.append(rows[0]),
        lambda rows, cal: rows.__setitem__(0, cal[0]),
        lambda rows, cal: rows[0].update(attempt=True),
        lambda rows, cal: rows[0].update(attempt=-1),
        lambda rows, cal: rows[0].update(world_seed=42),
        lambda rows, cal: rows[0].update(n_treatments=10),
        lambda rows, cal: rows.pop(0),
    ],
)
def test_input_prefix_rejects_phase_ids_order_seeds_and_replacement(mutation):
    cfg = config()
    schedules = sweep.study_schedules(cfg)
    rows = schedules["science"][:3]
    mutation(rows, schedules["calibration"])
    with pytest.raises(ValueError):
        sweep.prefit_selection(cfg, rows)


@pytest.mark.parametrize("phase,number", [("science", 1000), ("calibration", 0), ("bad", 0)])
def test_input_phase_validation(phase, number):
    row = dict(sweep.study_schedules(config())["science"][0], attempt=number)
    with pytest.raises(ValueError):
        sweep._scheduled_attempt(config(), row, phase)


@pytest.mark.parametrize(
    "elapsed,expected",
    [
        (24, 846),
        (2699, 400),
        (2700, 400),
        (2700.01, 399),
        (5093.99, 1),
        (5094, 1),
        (5094.01, 0),
        (5100, 0),
        (5401, 0),
    ],
)
def test_input_allocation_boundaries_and_400_is_only_flag(elapsed, expected):
    # Fake clock begins before setup (24s elapsed already includes setup/calibration).
    ticks = iter([123.0, 123.0 + elapsed])
    start, end = next(ticks), next(ticks)
    result = sweep.allocate_from_calibration(
        config(), end - start, _completed_calibration((1, 2, 3))
    )
    assert result["frozen_attempt_count"] == expected
    assert result["manifest"] == sweep.study_schedules(config())["science"][:expected]
    assert result["inputs"]["conservative_per_world_cost"] == 6
    assert result["inputs"]["elapsed_after_calibration"] == elapsed
    assert result["pooled_precision_adequate"] is (expected >= 400)
    assert result["allocation_reason"] == ("allocated" if expected else "budget_exhausted")
    assert result["allocation_excluded"] == 1000 - expected
    assert result["calibration_invalid_reason"] is None
    assert len(result["selection"]["rows"]) == expected


@pytest.mark.parametrize(
    "bad", [None, True, False, 0, -1, "2", float("nan"), float("inf"), -float("inf"), 1e308]
)
def test_input_allocation_invalid_cost_has_no_fallback(bad):
    records = _completed_calibration()
    records[1]["duration"] = bad
    result = sweep.allocate_from_calibration(config(), 100, records)
    assert result["allocation_reason"] == "calibration_invalid"
    assert result["calibration_invalid_reason"]
    assert result["frozen_attempt_count"] == 0 and result["manifest"] == []
    assert result["inputs"]["conservative_per_world_cost"] is None
    assert result["pooled_precision_adequate"] is False
    assert result["allocation_excluded"] == 1000
    assert all(row["attempt"] is None for row in result["selection"]["band_examples"])
    sweep.canonical_json(result)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0]),
        lambda rows: rows.reverse(),
        lambda rows: rows[0].update(status="failed"),
        lambda rows: rows[0].update(status="started"),
        lambda rows: rows[0].pop("duration"),
        lambda rows: rows[0].update(phase="science"),
        lambda rows: rows[0]["attempt"].update(attempt=0),
        lambda rows: rows[0]["attempt"].update(world_seed=1),
        lambda rows: rows[0].update(extra=True),
    ],
)
def test_input_allocation_requires_exact_complete_calibration(mutation):
    records = _completed_calibration()
    mutation(records)
    result = sweep.allocate_from_calibration(config(), 100, records)
    assert result["allocation_reason"] == "calibration_invalid"
    assert result["frozen_attempt_count"] == 0
    assert result["inputs"]["conservative_per_world_cost"] is None


def test_input_allocation_inclusive_elapsed_and_tiny_finite_cost():
    invalid = sweep.allocate_from_calibration(config(), 7, _completed_calibration())
    assert invalid["allocation_reason"] == "calibration_invalid"
    assert "cannot exclude" in invalid["calibration_invalid_reason"]
    result = sweep.allocate_from_calibration(config(), 10, _completed_calibration((1e-320,) * 3))
    assert result["frozen_attempt_count"] == 1000
    for bad in (None, True, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sweep.allocate_from_calibration(config(), bad, _completed_calibration())


def _projection_fixture(*, eligible=True, rank=2, recovered=True, reasons=None):
    return {
        "status": sweep.orthogonal_status(
            scheduled=True,
            attempted=True,
            generated=True,
            truth_eligible=eligible,
            rank=rank,
            p=2,
            noisefree_recovered=recovered,
            paired_noisy_available=True,
        ),
        "reasons": [] if reasons is None else reasons,
    }


@pytest.mark.parametrize(
    "state,analysis,stage,expected",
    [
        ("eligible_full_rank", _projection_fixture(), None, (True, True, True, True, True, True)),
        (
            "eligible_full_rank",
            _projection_fixture(recovered=False),
            None,
            (True, True, True, True, False, True),
        ),
        (
            "eligible_rank_deficient",
            _projection_fixture(rank=1, recovered=False, reasons=["rank_deficient"]),
            None,
            (True, True, True, False, None, True),
        ),
        (
            "identity_or_truth_ineligible",
            _projection_fixture(
                eligible=False, reasons=["identity_or_truth_ineligible:target_truth"]
            ),
            None,
            (True, True, False, False, None, None),
        ),
        (
            "unsupported_truth",
            _projection_fixture(eligible=False, reasons=["unsupported_truth:missing"]),
            None,
            (True, True, False, False, None, None),
        ),
        # Accepted analyze_world can catch exceptions after it has set truth_eligible.
        (
            "identity_or_truth_ineligible",
            _projection_fixture(reasons=["identity_or_truth_ineligible:SVD failed"]),
            None,
            (True, True, False, False, None, None),
        ),
        ("analysis_failed", None, "analysis_failed", (True, True, False, False, None, None)),
    ],
)
def test_input_denominator_projection_keeps_graph_failures_and_rank_deficiency(
    state, analysis, stage, expected
):
    cfg = config()
    manifest = sweep.study_schedules(cfg)["science"][:1]
    projection = sweep.project_attempt_state(
        cfg,
        manifest,
        manifest[0],
        started=True,
        graph={"g_cy": [1]},
        analysis=analysis,
        failure_stage=stage,
    )
    assert projection["state"] == state
    assert (
        tuple(
            projection[k]
            for k in (
                "A",
                "G",
                "E",
                "F",
                "noisefree_recovered_over_F",
                "paired_noisy_available_over_E",
            )
        )
        == expected
    )
    assert projection["eligible_rank_deficient"] is (state == "eligible_rank_deficient")


def test_input_projection_real_accepted_analysis_and_immutable_failed_example(
    catalog_analysis_fixture,
):
    cfg, attempt, world, analysis = catalog_analysis_fixture
    manifest = sweep.study_schedules(cfg)["science"][:28]
    chosen = sweep.prefit_selection(cfg, manifest)
    before = sweep.canonical_json(chosen)
    projection = sweep.project_attempt_state(
        cfg, manifest, attempt, started=True, graph=world.g, analysis=analysis
    )
    assert projection["E"] and projection["F"]
    # Failure cannot select a later good example, and no outcome enters selection.
    failed = sweep.project_attempt_state(
        cfg,
        manifest,
        manifest[0],
        started=True,
        graph=None,
        analysis=None,
        failure_stage="generation_failed",
    )
    assert failed["A"] and not failed["G"] and failed["state"] == "generation_failed"
    unattempted = sweep.project_attempt_state(
        cfg, manifest, manifest[1], started=False, graph=None, analysis=None
    )
    assert not unattempted["A"] and unattempted["state"] == "unattempted_budget"
    assert before == sweep.canonical_json(sweep.prefit_selection(cfg, manifest))
    assert chosen["band_examples"][0]["attempt"] == failed["attempt"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"started": 1},
        {"started": False},
        {"graph": None},
        {"graph": {}},
        {"analysis": None},
        {"failure_stage": "generation_failed"},
        {"failure_stage": "arbitrary"},
        {"analysis": {}},
        {"analysis": _projection_fixture(rank=None)},
        {"analysis": _projection_fixture(rank=1)},
    ],
)
def test_input_projection_rejects_inconsistent_states(overrides):
    cfg = config()
    manifest = sweep.study_schedules(cfg)["science"][:1]
    args = {"started": True, "graph": {"g_cy": [1]}, "analysis": _projection_fixture()}
    args.update(overrides)
    with pytest.raises(ValueError):
        sweep.project_attempt_state(cfg, manifest, manifest[0], **args)
    with pytest.raises(ValueError):
        sweep.project_attempt_state(cfg, [], manifest[0], started=False, graph=None, analysis=None)


@pytest.mark.parametrize(
    "counts",
    [(0, 0, 0, 0, 0, 0), (399, 350, 300, 200, 100, 100), (1000, 1000, 1000, 1000, 1000, 0)],
)
def test_input_accounting_conservation_and_distinct_budget_exclusions(counts):
    names = ("ncap", "attempted", "generated", "eligible", "full_rank", "rank_deficient")
    result = sweep.validate_accounting(config(), **dict(zip(names, counts, strict=True)))
    n, a, g, e, _, _ = counts
    assert result == {
        "allocation_excluded": 1000 - n,
        "runtime_unattempted": n - a,
        "generation_failed": a - g,
        "evidence_unavailable": 0,
        "generated_ineligible": g - e,
    }
    assert result["allocation_excluded"] + result["runtime_unattempted"] + a == 1000


@pytest.mark.parametrize(
    "field,value",
    [
        ("ncap", 1001),
        ("ncap", 9),
        ("attempted", 11),
        ("generated", 11),
        ("eligible", 11),
        ("full_rank", 11),
        ("rank_deficient", 1),
        ("attempted", True),
        ("eligible", -1),
        ("generated", 10.0),
    ],
)
def test_input_accounting_rejects_broken_conservation(field, value):
    args = {
        "ncap": 10,
        "attempted": 10,
        "generated": 10,
        "eligible": 10,
        "full_rank": 10,
        "rank_deficient": 0,
    }
    args[field] = value
    with pytest.raises(ValueError):
        sweep.validate_accounting(config(), **args)


def test_input_reporting_contract_sparse_frequency_and_separate_precision():
    cfg = config()
    reporting = cfg["study"]["reporting"]
    assert reporting["configured_count_bands"] == [[1, 3], [4, 7], [8, 10]]
    assert reporting["direct_dimension_bands"] == [[0, 0], [1, 3], [4, 7], [8, 10]]
    assert reporting["configured_views"] == [
        "T_marginal",
        "M_marginal",
        "exact_T_M_cells",
        "nine_T_M_band_pairs",
    ]
    assert reporting["frequency_denominators"] == {
        "planned_configured": "Ncap",
        "attempted_configured": "A",
        "direct_dimensions": "G",
        "graph_patterns": "G",
    }
    assert reporting["indicators"]["noisefree_recovered"]["denominator"] == "F"
    assert reporting["indicators"]["paired_noisy_available"]["denominator"] == "E"
    assert "non-simultaneous" in reporting["sparse_frequency_rule"]
    assert "only" in cfg["study"]["subgroup_rule"]
    assert "no marginal or subgroup precision guarantee" in cfg["study"]["pooled_precision_rule"]
    assert wilson(1, 3)["rate"] == 1 / 3  # sparse frequencies remain available
    assert sweep.subgroup_available(cfg, 3) is False  # only conditional outcome rate
    assert all(wilson(0, 0)[key] is None for key in ("rate", "lower", "upper"))


def test_input_freeze_before_calibration_has_no_outcomes_or_external_writes(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("input freeze must not generate, analyze, calibrate or persist")

    for name in ("generate_world", "analyze_world", "create_run_archive", "persist_world"):
        monkeypatch.setattr(sweep, name, forbidden)
    cfg = config()
    environment = sweep.runtime_input_environment()
    sources = sweep.analysis_source_hashes()
    seeds = sweep.study_schedules(cfg)
    result = sweep.freeze_input_contract(
        cfg, environment=environment, source_hashes=sources, seed_manifest=seeds
    )
    contract = result["contract"]
    assert result["digest"] == sweep.digest(contract)
    assert contract["version"] == "ols-scientific-input/v3"
    assert (
        contract["seed_manifest_digest"]
        == "95dd82f378da6890b72e82e7ff931568219e10ce6f5a213fba334e219e7d3291"
    )
    assert (
        contract["catalog_schema_sha256"]
        == "a7625fcd4430af84cd48990bcb83108d88f772957beed0de736dc297c5ad2057"
    )
    assert contract["source_hashes"] == sources and contract["environment"] == environment
    assert list(tmp_path.iterdir()) == []
    environment.clear()
    sources.clear()
    seeds.clear()
    cfg.clear()
    assert result["digest"] == sweep.digest(contract)


@pytest.mark.parametrize("field", ["environment", "source_hashes", "seed_manifest"])
def test_input_freeze_rejects_arbitrary_provenance(field):
    args = {
        "environment": sweep.runtime_input_environment(),
        "source_hashes": sweep.analysis_source_hashes(),
        "seed_manifest": sweep.study_schedules(config()),
    }
    args[field] = {}
    with pytest.raises(ValueError):
        sweep.freeze_input_contract(config(), **args)


def test_frozen_core_and_protected_hashes():
    import re

    blocks = [
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def create_run_archive(",
            "\ndef _exclusive_json",
            "5ca0b699053e0b696f5438c7a50cbd3103c95d74276903a97414524ac5041520",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def _catalog_numeric_check(",
            "\ndef frozen_schedule",
            "b722983b0b565f0b8fd7bc178749f5ad77b8f46e66c83be2089afac1fd8e6ba9",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "CATALOG_SCHEMA_BYTES =",
            "\ndef analysis_source_hashes",
            "fc39c0ed118c4181ab003d6803e9ad041936898441f4785910229573c5f96879",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "WORLD_VERSION =",
            "\ndef _schema_check",
            "e42ef39fe930947218826918ae81faa230355f9a43d63ea2542829fb1f1433d2",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def generate_world(",
            "    return sample_scm",
            "07f6893fc86371e503b54ed514b39d79535c2c23b0808fdbf36be3ac28eb5131",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "SCHEMA_VERSION =",
            "\n\n\ndef canonical_json",
            "0ce026429f55a10f0010c5430b3900548e6b50e21dbf40cc02ed1b0300c3c9c6",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def wilson(",
            "\ndef denominator_summary",
            "6f1f66941b537638cd0a7426d52e68a039e7a4aa41d73c2aee7e4241f97ba9e8",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def denominator_summary(",
            "\ndef exclusive_output_dir",
            "a1918179b605abc9677d83e23e8930714920f70aa9f14a4c396cd318249b39ea",
        ),
        (
            "scripts/linear_recovery_true_feature_sweep.py",
            "def _finite_array(",
            None,
            "96fb448e19d97dc55a3c459fe50634978d4410fe90654a0f188872f47e2eab7d",
        ),
        (
            "tests/test_linear_recovery_true_feature_sweep.py",
            "# All seeds and cases below",
            None,
            "c061e47b5145a02aec2ec9219c3f60534cd332a3b4cf0625a556f93a7774f418",
        ),
    ]
    for filename, start, end, expected in blocks:
        raw = Path(filename).read_text()
        begin = re.search("^" + re.escape(start), raw, flags=re.MULTILINE).start()
        finish = len(raw) if end is None else raw.index(end, begin)
        block = raw[begin:finish]
        if start == "SCHEMA_VERSION =":
            block += "\n"
        assert hashlib.sha256(block.encode()).hexdigest() == expected
    protected = {
        "scripts/linear_recovery_prevalence.py": "737f72565466d68ca8949822b4a1bd85fcb0ba7df3ce6cd989aebcf77e71578c",
        "tests/test_linear_recovery.py": "a31764fc5b9bb72eca1d21be07b75660b31e9867eb3b7fd76375c49a3b37400c",
        "docs/examples/data/linear-recovery-prevalence-config.json": "2c45ea12691245c69882423fba721ab376739cad635b3c8ff883cb5f8289ab5a",
        "docs/examples/data/linear-recovery-prevalence.csv": "f0fb827b740a8b0d1da6c9c7e7d90d588538762185042f854b48609a0e85a47e",
        "docs/examples/linear-recovery.ipynb": "d1b65883287449df19ccc0298a66ee4f1ea02f664d36343f97acf584c127426d",
        "docs/examples/data/linear-recovery-true-feature-sweep-config.json": "bd6eddccb89b921c60b969277f2c6fd3ef23ae6a9b53c11b39a8a5e141e31289",
        "docs/examples/linear-recovery-true-feature-sweep.ipynb": "b84a02ad9cc20d5bf2a73b5e56cd217e31182ba15240d2648a7e3be16a4dfe2e",
    }
    for filename, expected in protected.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected


# Summary fixtures are ENGINEERING evidence only: never calibration or prevalence.
def _summary_setup(tmp_path, *, ncap=12, calibration=None):
    cfg = config()
    environment = sweep.runtime_input_environment()
    sources = sweep.analysis_source_hashes()
    seeds = sweep.study_schedules(cfg)
    inputs = sweep.freeze_input_contract(
        cfg, environment=environment, source_hashes=sources, seed_manifest=seeds
    )
    archive, _ = sweep.create_run_archive(
        tmp_path, cfg, environment=environment, source_hashes=sources, seed_manifest=seeds
    )
    calibration = _completed_calibration((1, 1, 1)) if calibration is None else calibration
    allocation = sweep.allocate_from_calibration(cfg, 5100 - 2 * ncap, calibration)
    return archive, {
        "input_contract": inputs,
        "calibration": calibration,
        "allocation": allocation,
        "ledger": [],
        "timing": {
            "clock": "monotonic_inclusive_before_setup",
            "started_count": 0,
            "elapsed_seconds": 5100 - 2 * ncap,
            "finalization_seconds": 0,
        },
    }


def _summary_start(
    archive, args, *, outcome="unknown", errors=None, world=None, graph=None, catalog=None
):
    i = len(args["ledger"])
    attempt = args["allocation"]["manifest"][i]
    start = {
        "attempt": attempt,
        "phase": "science",
        "input_digest": args["input_contract"]["digest"],
        "allocation_digest": sweep.digest(args["allocation"]),
        "started_elapsed": args["timing"]["elapsed_seconds"],
        "resource": {"host": "ENGINEERING-FIXTURE", "pid": 1},
    }
    sr = sweep.persist_report_evidence(archive, "start", start)
    graph_ref = None
    if graph is not None:
        graph_ref = sweep.persist_report_evidence(
            archive, "graph", {"attempt": attempt, "phase": "science", "graph": graph}
        )
    end = {
        "attempt": attempt,
        "phase": "science",
        "start_sha256": sr["sha256"],
        "finished_elapsed": start["started_elapsed"] + 0.1,
        "status": "recovered_unknown" if outcome == "unknown" else "complete",
        "generation_outcome": outcome,
        "errors": errors or [],
        "world_reference": world,
        "catalog_reference": catalog,
        "graph_reference": graph_ref,
        "partial_references": [],
    }
    er = sweep.persist_report_evidence(archive, "terminal", end)
    args["ledger"].append({"start": sr, "terminal": er})
    args["timing"]["started_count"] += 1
    args["timing"]["elapsed_seconds"] = end["finished_elapsed"]
    return end


def _empty_graph(attempt):
    t, m = attempt["n_treatments"], attempt["n_covariates"]
    return {
        k: np.zeros(shape, dtype=int).tolist()
        for k, shape in {
            "g_cy": (t,),
            "g_zy": (m,),
            "g_dy": (1,),
            "g_dc": (1, t),
            "g_dz": (1, m),
            "g_zc": (m, t),
            "g_cc": (t, t),
            "g_zz": (m, m),
        }.items()
    }


def _stage(stage):
    return {"stage": stage, "type": "EngineeringError", "message": "retained controlled failure"}


def test_summary_correction_preserves_old_confirmed_failure_and_disjoint_unknown():
    cfg = config()
    assert cfg["input_contract_version"] == "ols-scientific-input/v3"
    assert (
        cfg["accounting_correction"]["previous_config_digest"]
        == "e0e13db341c6e3b692e7b57cb07cd0effcfff3655a308bf342ef1a92f2ea05e0"
    )
    prefix = sweep.study_schedules(cfg)["science"][:1]
    old = sweep.project_attempt_state(
        cfg,
        prefix,
        prefix[0],
        started=True,
        graph=None,
        analysis=None,
        failure_stage="generation_failed",
    )
    assert old["state"] == "generation_failed" and old["A"] and not old["G"]
    unknown = sweep.project_attempt_state(
        cfg, prefix, prefix[0], started=True, graph=None, analysis=None
    )
    assert unknown["state"] == "evidence_unavailable" and not unknown["E"]
    counters = sweep.validate_accounting(
        cfg,
        ncap=12,
        attempted=8,
        generated=4,
        eligible=3,
        full_rank=2,
        rank_deficient=1,
        evidence_unavailable=3,
    )
    assert counters["generation_failed"] == 1 and counters["evidence_unavailable"] == 3
    assert 4 + counters["generation_failed"] + counters["evidence_unavailable"] == 8


def test_summary_mixed_durable_missing_evidence_and_partial_graph(tmp_path):
    archive, args = _summary_setup(tmp_path)
    _summary_start(archive, args, outcome="failed", errors=[_stage("generation")])
    _summary_start(archive, args, outcome="completed", errors=[_stage("persistence")])
    _summary_start(archive, args, errors=[_stage("timeout"), _stage("persistence")])
    graph = _empty_graph(args["allocation"]["manifest"][3])
    _summary_start(
        archive,
        args,
        outcome="completed",
        graph=graph,
        errors=[_stage("analysis"), _stage("persistence")],
    )
    result = sweep.build_compact_summary(archive=archive, **args)
    assert result["counts"] == {
        "maximum": 1000,
        "Ncap": 12,
        "A": 4,
        "G": 1,
        "E": 0,
        "F": 0,
        "R": 0,
        "rank_deficient": 0,
        "confirmed_generation_failed": 1,
        "allocation_excluded": 988,
        "runtime_unattempted": 8,
        "evidence_unavailable": 2,
        "generated_ineligible": 1,
    }
    assert result["stage_failure_frequencies"]["persistence"]["numerator"] == 3
    assert result["stage_failure_frequencies"]["timeout"]["numerator"] == 1
    assert result["direct_dimensions"]["g_cy"]["exact"][0]["rate"] == 1
    assert result["graph_patterns"]["frequencies"]["00000000"]["numerator"] == 1
    assert result["graph_patterns"]["frequencies"]["11111111"]["upper"] > 0
    assert result["rates"]["full_rank_over_E"]["rate"] is None
    assert all(x["details"] is None for x in result["examples"])
    assert (
        result["examples"][0]["attempt"]
        == args["allocation"]["selection"]["band_examples"][0]["attempt"]
    )
    assert sweep.validate_compact_summary(result) == result


@pytest.mark.parametrize("ncap", [0, 1, 399, 400])
def test_summary_allocation_flags_sparse_zero_and_one_counts(tmp_path, ncap):
    archive, args = _summary_setup(tmp_path, ncap=ncap)
    result = sweep.build_compact_summary(archive=archive, **args)
    assert result["precision"]["planned_400_adequate"] is (ncap >= 400)
    assert result["precision"]["execution_stop"] is False
    assert result["precision"]["actual_eligible_N"] == 0
    assert result["counts"]["allocation_excluded"] + result["counts"]["runtime_unattempted"] == 1000
    for section, size in (
        ("T_marginal", 10),
        ("M_marginal", 10),
        ("exact_T_M_cells", 100),
        ("nine_T_M_band_pairs", 9),
    ):
        cells = result["planned_configured"][section]
        assert len(cells) == size
        assert sum(c["frequency"]["numerator"] for c in cells) == ncap
        assert all(c["frequency"]["denominator"] == ncap for c in cells)
        assert all(c["frequency"]["rate"] is None for c in result["attempted_configured"][section])
    assert len(result["examples"]) == 9
    assert all(
        x["availability_reason"] in ("unattempted_budget", "band_absent_from_manifest")
        for x in result["examples"]
    )
    if ncap == 1:
        zero = next(
            c["frequency"]
            for c in result["planned_configured"]["exact_T_M_cells"]
            if c["frequency"]["numerator"] == 0
        )
        assert zero["rate"] == 0 and zero["upper"] > 0


def test_summary_invalid_calibration_has_zero_allocation_no_cost_invention(tmp_path):
    archive, args = _summary_setup(
        tmp_path, calibration=[{"status": "failed", "reason": "ENGINEERING calibration failure"}]
    )
    result = sweep.build_compact_summary(archive=archive, **args)
    assert result["counts"]["Ncap"] == result["counts"]["A"] == 0
    assert result["evidence"]["allocation"]["inputs"]["conservative_per_world_cost"] is None
    assert result["evidence"]["allocation"]["calibration_invalid_reason"]
    assert result["evidence"]["calibration"] == args["calibration"]


def test_summary_real_controlled_world_portable_and_optional_raw_verification(
    tmp_path, catalog_analysis_fixture
):
    archive, args = _summary_setup(tmp_path)
    _, attempt, _, result = catalog_analysis_fixture
    world_ref = sweep.persist_world(archive, attempt, result)
    cat = sweep.catalog_entry(archive=archive, world_reference=world_ref)
    cref = sweep.persist_catalog_entry(archive, cat)
    _summary_start(archive, args, outcome="completed", world=world_ref, catalog=cref)
    summary = sweep.build_compact_summary(archive=archive, **args)
    assert summary["counts"]["E"] == summary["counts"]["F"] == 1
    assert summary["rates"]["full_rank_over_E"]["rate"] == 1
    assert summary["descriptive"]["y0"]["residual_rmse"]["availability"] == "one_world"
    selected = next(e for e in summary["examples"] if e["attempt"] == 0)
    assert selected["details"]["n_fit"] == 104 and selected["details"]["warmup"] == 0
    assert selected["details"]["n_full"] == 104
    assert "residual" not in selected["details"]["fits"]["y0"]
    assert "design" not in selected["details"] and "targets" not in selected["details"]
    portable = tmp_path / "portable"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", summary)
    with pytest.raises(FileExistsError):
        sweep.persist_report_evidence(portable, "summary", summary)
    assert sweep.read_compact_summary(portable, ref)["validation"] == "compact_integrity_only"
    assert (
        sweep.read_compact_summary(portable, ref, raw_archive=archive)["validation"]
        == "raw_science_verified"
    )
    for cell in summary["attempted_configured"]["exact_T_M_cells"]:
        assert cell["outcomes"]["full_rank_over_E"]["rate"] is None
    assert sweep.validate_compact_summary(summary) == summary


@pytest.mark.parametrize(
    "mutation",
    [
        "rate",
        "counter",
        "graphref",
        "graph",
        "selection",
        "duplicates",
        "missing",
        "fake_ref",
        "source",
        "calibration",
        "counterfeit_execution",
    ],
)
def test_summary_resealed_tamper_rejected(tmp_path, catalog_analysis_fixture, mutation):
    archive, args = _summary_setup(tmp_path)
    _, attempt, _, result = catalog_analysis_fixture
    world_ref = sweep.persist_world(archive, attempt, result)
    _summary_start(archive, args, outcome="completed", world=world_ref)
    summary = sweep.build_compact_summary(archive=archive, **args)
    altered = deepcopy(summary)
    cap = altered["evidence"]["ledger"][0]
    if mutation == "rate":
        altered["rates"]["full_rank_over_E"]["rate"] = 0.25
    elif mutation == "counter":
        altered["counts"]["A"] = 8
    elif mutation == "graphref":
        cap["catalog"]["known_sampled_truth"]["direct_structure"]["sha256"] = "0" * 64
    elif mutation == "graph":
        cap["graph"]["g_cy"][0] = 1 - cap["graph"]["g_cy"][0]
    elif mutation == "selection":
        altered["examples"][0]["attempt"] = 999
    elif mutation == "duplicates":
        altered["evidence"]["ledger"].append(deepcopy(cap))
    elif mutation == "missing":
        altered["evidence"]["ledger"].clear()
    elif mutation == "fake_ref":
        cap["terminal"]["world_reference"]["world_id"] = "0" * 64
    elif mutation == "source":
        cap["catalog"]["source_commit"] = "0" * 40
    elif mutation == "calibration":
        altered["evidence"]["allocation"]["frozen_attempt_count"] = 999
    else:
        cap["start"]["independently_verified_execution"] = True
    # Even a newly sealed compact file must recompute its accounting and bindings.
    portable = tmp_path / "tampered"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", altered)
    with pytest.raises(ValueError):
        sweep.read_compact_summary(portable, ref)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reorder",
        "seed",
        "phase",
        "resource",
        "in_progress",
        "false_genfail",
        "timing",
        "counterfeit",
        "fake_partial",
    ],
)
def test_summary_durable_ledger_negative_matrix(tmp_path, mutation):
    archive, args = _summary_setup(tmp_path)
    _summary_start(archive, args)
    _summary_start(archive, args)
    if mutation == "missing":
        args["ledger"].pop()
        args["timing"]["started_count"] -= 1
    elif mutation == "duplicate":
        args["ledger"][1] = args["ledger"][0]
    elif mutation == "reorder":
        args["ledger"].reverse()
    else:
        item = args["ledger"][0]
        end = sweep._report_file(archive, item["terminal"])
        if mutation == "seed":
            end["attempt"]["world_seed"] += 1
        elif mutation == "phase":
            end["phase"] = "calibration"
        elif mutation == "resource":
            start = sweep._report_file(archive, item["start"])
            start["resource"]["pid"] = True
            # Preserve original durable file: cannot replace a started identity.
            item["start"] = sweep.persist_report_evidence(archive, "start", start)
            end["start_sha256"] = item["start"]["sha256"]
        elif mutation == "in_progress":
            end["status"] = "in_progress"
        elif mutation == "false_genfail":
            end["generation_outcome"] = "failed"
        elif mutation == "timing":
            end["finished_elapsed"] -= 5
        elif mutation == "counterfeit":
            end["proven_execution"] = True
        else:
            end["partial_references"] = [
                {"locator": "arrays/" + "0" * 64 + ".bin", "sha256": "0" * 64, "kind": "array"}
            ]
        item["terminal"] = sweep.persist_report_evidence(archive, "terminal", end)
    with pytest.raises((ValueError, FileNotFoundError)):
        sweep.build_compact_summary(archive=archive, **args)


def test_summary_reader_trusted_digest_symlink_and_no_absolute_runtime_dependency(tmp_path):
    archive, args = _summary_setup(tmp_path)
    summary = sweep.build_compact_summary(archive=archive, **args)
    portable = tmp_path / "portable"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", summary)
    target = portable / ref["locator"]
    raw = target.read_bytes()
    target.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="digest"):
        sweep.read_compact_summary(portable, ref)
    target.unlink()
    target.symlink_to(archive / "run.json")
    with pytest.raises(OSError):
        sweep.read_compact_summary(portable, ref)


def test_summary_mocked_mixed_world_unit_rates_and_rank_deficient_nonidentification(
    tmp_path, monkeypatch
):
    archive, args = _summary_setup(tmp_path)
    cfg = config()
    manifest = args["allocation"]["manifest"]
    rows = []
    for i, state in enumerate(
        ("full", "rankdef", "ineligible", "genfail", "analysisfail", "persistfail")
    ):
        analysis = _projection_fixture(
            eligible=state in ("full", "rankdef"),
            rank=1 if state == "rankdef" else 2,
            recovered=state == "full",
        )
        analysis["details"] = {
            "coefficient_order": ["beta[0]", "intercept"],
            "fits": {
                target: {
                    "coefficient_error": [float(i), 0.0],
                    "residual_max_abs": 0.2,
                    "residual_rmse": 0.1,
                }
                for target in ("y0", "y1")
            },
            "raw_design": {"condition": 2.0},
            "standardized_design": {"condition": 1.0},
        }
        graph = None if state in ("genfail", "persistfail") else _empty_graph(manifest[i])
        stage = (
            "generation_failed"
            if state == "genfail"
            else "analysis_failed"
            if state == "analysisfail"
            else "evidence_unavailable"
            if state == "persistfail"
            else None
        )
        if graph is None or state == "analysisfail":
            analysis = None
        projection = sweep.project_attempt_state(
            cfg,
            manifest,
            manifest[i],
            started=True,
            graph=graph,
            analysis=analysis,
            failure_stage=stage,
        )
        rows.append(
            {
                **projection,
                "graph": graph,
                "analysis": analysis,
                "catalog": None,
                "stages": ["persistence"] if state == "persistfail" else [],
                "duration": 1.0,
                "evidence_unavailable": state == "persistfail",
                "confirmed_generation_failed": state == "genfail",
            }
        )
    rates = sweep._outcome_rates(rows)
    assert rates["full_rank_over_E"]["numerator"] == 1
    assert rates["full_rank_over_E"]["denominator"] == 2
    assert rates["rank_deficient_over_E"]["rate"] == 0.5
    assert rates["noisefree_recovered_over_F"]["rate"] == 1
    assert rates["paired_noisy_available_over_E"]["denominator"] == 2
    metrics = sweep._descriptive_metrics(rows)
    assert metrics["coefficient_identification"]["rank_deficient_nonidentified_worlds"] == 1
    assert metrics["y0"]["beta_world_max_abs_error"]["n_worlds"] == 1
    assert metrics["y1"]["rho_zy_world_max_abs_error"]["availability"] == "empty"
    assert metrics["y0"]["residual_rmse"]["n_worlds"] == 2
    # Minimum is only conditional outcome rates, never frequency suppression.
    assert sweep._outcome_rates(rows, conditional=True)["full_rank_over_E"]["rate"] is None
    assert sweep._report_rate(0, 2, "engineering frequency")["upper"] > 0
    assert sweep._quantiles([], "empty")["quantiles"] is None


@pytest.mark.parametrize("stage", ["persistence", "analysis", "identity"])
def test_summary_full_world_failure_stages_and_fixed_selected_example(
    tmp_path, catalog_analysis_fixture, stage
):
    archive, args = _summary_setup(tmp_path)
    _, attempt, _, original = catalog_analysis_fixture
    result = deepcopy(original)
    if stage == "identity":
        result["status"]["truth_eligibility"] = "ineligible"
        result["reasons"] = ["identity_or_truth_ineligible:ENGINEERING fixture"]
    world_ref = sweep.persist_world(archive, attempt, result)
    _summary_start(
        archive,
        args,
        outcome="completed",
        world=world_ref,
        errors=[] if stage == "identity" else [_stage(stage)],
    )
    summary = sweep.build_compact_summary(archive=archive, **args)
    assert summary["counts"]["A"] == summary["counts"]["G"] == 1
    assert summary["counts"]["E"] == (1 if stage == "persistence" else 0)
    assert summary["counts"]["evidence_unavailable"] == 0
    selected = next(e for e in summary["examples"] if e["attempt"] == 0)
    assert (
        selected["availability_reason"]
        == {
            "persistence": "available",
            "analysis": "analysis_failed",
            "identity": "identity_or_truth_ineligible",
        }[stage]
    )
    assert selected["references"]["catalog_reference"] is None  # Never fabricate failed writes.
    assert summary["evidence"]["allocation"]["selection"] == args["allocation"]["selection"]


def test_summary_partial_graph_without_world_unknown_substatus_and_overruns(tmp_path):
    archive, args = _summary_setup(tmp_path)
    _summary_start(
        archive,
        args,
        outcome="completed",
        graph=_empty_graph(args["allocation"]["manifest"][0]),
        errors=[_stage("persistence")],
    )
    _summary_start(archive, args, outcome="completed", errors=[_stage("persistence")])
    _summary_start(archive, args, errors=[_stage("timeout")])
    args["timing"]["elapsed_seconds"] = 5501.0
    args["timing"]["finalization_seconds"] = 350.0
    summary = sweep.build_compact_summary(archive=archive, **args)
    assert summary["counts"]["G"] == 1 and summary["counts"]["E"] == 0
    assert summary["attempts"][0]["state"] == "generated_analysis_unavailable"
    assert summary["attempts"][0]["evidence_unavailable"] is False
    assert (
        summary["attempts"][1]["missing_evidence_status"]
        == "known_generation_completed_missing_evidence"
    )
    assert summary["attempts"][2]["missing_evidence_status"] == "generation_outcome_unknown"
    assert summary["runtime"]["deadline_overrun_seconds"] == 101.0
    assert summary["runtime"]["reserve_overrun_seconds"] == 50.0
    assert summary["failure_frequencies"]["evidence_unavailable"]["rate"] == 2 / 3
    assert sweep.validate_compact_summary(summary) == summary


def test_summary_portable_runtime_paths_are_metadata_not_dependencies(tmp_path):
    archive, args = _summary_setup(tmp_path)
    summary = sweep.build_compact_summary(archive=archive, **args)
    evidence = deepcopy(summary["evidence"])
    contract = evidence["input"]["contract"]
    contract["environment"]["python_executable"] = "unavailable-original-host/python"
    contract["environment"]["environment"]["PYTHONPATH"] = "unavailable-original-checkout"
    evidence["input"]["digest"] = sweep.digest(contract)
    portable_summary = sweep._render_compact(evidence)
    portable = tmp_path / "portable"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", portable_summary)
    assert sweep.read_compact_summary(portable, ref)["validation"] == "compact_integrity_only"
    assert all(sweep._report_rate(0, 0, "empty")[key] is None for key in ("rate", "lower", "upper"))
    assert sweep._report_rate(9, 9, "subgroup", conditional=True)["rate"] is None
    assert sweep._report_rate(10, 10, "subgroup", conditional=True)["rate"] == 1.0


@pytest.mark.parametrize("retain_graph", [False, True])
def test_summary_early_analysis_failure_retains_separate_graph_and_reasons(
    tmp_path, catalog_analysis_fixture, retain_graph
):
    archive, args = _summary_setup(tmp_path)
    _, attempt, _, original = catalog_analysis_fixture
    failed = {
        "generation": original["generation"],
        "status": sweep.orthogonal_status(
            scheduled=True, attempted=True, generated=True, truth_eligible=False
        ),
        "reasons": ["unsupported_truth:early ENGINEERING failure before source retention"],
    }
    world_ref = sweep.persist_world(archive, attempt, failed)
    graph = sweep._report_graph(original["source"]["graph"], attempt) if retain_graph else None
    _summary_start(
        archive,
        args,
        outcome="completed",
        world=world_ref,
        graph=graph,
        errors=[_stage("analysis")],
    )
    summary = sweep.build_compact_summary(archive=archive, **args)
    assert summary["counts"]["G"] == int(retain_graph)
    assert summary["counts"]["E"] == 0
    assert summary["counts"]["evidence_unavailable"] == int(not retain_graph)
    assert summary["attempts"][0]["reasons"] == failed["reasons"]


@pytest.fixture(scope="module")
def strict_summary_fixture(tmp_path_factory, catalog_analysis_fixture):
    archive, args = _summary_setup(tmp_path_factory.mktemp("strict-summary"))
    _, attempt, _, result = catalog_analysis_fixture
    world_ref = sweep.persist_world(archive, attempt, result)
    _summary_start(archive, args, outcome="completed", world=world_ref)
    return sweep.build_compact_summary(archive=archive, **args)


def _mapping_paths(value, path=()):
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from _mapping_paths(child, (*path, key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _mapping_paths(child, (*path, i))


def _at_path(value, path):
    for key in path:
        value = value[key]
    return value


def test_summary_old_failed_recursive_claims_fresh_rerender_and_reseal(
    tmp_path, strict_summary_fixture, monkeypatch
):
    """OLD FAILED: every undeclared nested analysis claim survived fresh rendering."""
    analysis = strict_summary_fixture["evidence"]["ledger"][0]["analysis"]
    paths = list(_mapping_paths(analysis))
    assert ("status",) in paths and ("details", "fits", "y1") in paths
    assert ("details", "parameters", "trajectory", "treatment", "use") in paths
    for path in paths:
        evidence = deepcopy(strict_summary_fixture["evidence"])
        _at_path(evidence["ledger"][0]["analysis"], path)["safe_to_run"] = True
        # Construct the old accepted projection, with fresh derived bindings. Only
        # fixture construction bypasses the new schema; the reader is never patched.
        with monkeypatch.context() as patch:
            patch.setattr(sweep, "_validate_compact_analysis", lambda *args: None)
            counterfeit = sweep._render_compact(evidence)
        ref = sweep.persist_report_evidence(tmp_path, "summary", counterfeit)
        with pytest.raises(ValueError):
            sweep.read_compact_summary(tmp_path, ref)


def test_summary_recursive_missing_keys_wrong_scalar_types_and_nonfinite(strict_summary_fixture):
    analysis = strict_summary_fixture["evidence"]["ledger"][0]["analysis"]
    for path in _mapping_paths(analysis):
        for key in _at_path(analysis, path):
            altered = deepcopy(strict_summary_fixture)
            del _at_path(altered["evidence"]["ledger"][0]["analysis"], path)[key]
            with pytest.raises(ValueError):
                sweep.validate_compact_summary(altered)
    for path, value in [
        (("status", "paired_noisy", "available"), 1),
        (("status", "raw_observed_misspecified"), 0),
        (("details", "fits", "y0", "rank"), True),
        (("details", "raw_design", "rank"), True),
        (("details", "n_full"), 104.0),
        (("details", "parameters", "l_max"), 8.0),
        (("details", "parameters", "beta", 0), True),
        (("details", "parameters", "carryover_family", 0), 0.0),
        (("details", "fit_indices", 0), False),
        (("details", "identities", "scalar_baseline", "passed"), 1),
        (("details", "target_metrics", "y0", "all", "mean"), "0"),
        *[
            (("details", "fits", "y0", "residual_rmse"), v)
            for v in (float("nan"), float("inf"), float("-inf"), True)
        ],
    ]:
        altered = deepcopy(strict_summary_fixture)
        _at_path(altered["evidence"]["ledger"][0]["analysis"], path[:-1])[path[-1]] = value
        with pytest.raises(ValueError):
            sweep.validate_compact_summary(altered)


@pytest.mark.parametrize(
    "section",
    [
        (),
        ("evidence",),
        ("bindings",),
        ("runtime",),
        ("counts",),
        ("precision",),
        ("evidence", "input"),
        ("evidence", "input", "contract"),
        ("evidence", "input", "contract", "environment"),
        ("evidence", "input", "contract", "environment", "environment"),
        ("evidence", "input", "contract", "source_hashes"),
        ("evidence", "input", "contract", "config", "study"),
        ("evidence", "calibration", 0),
        ("evidence", "allocation", "inputs"),
        ("evidence", "timing"),
        ("evidence", "ledger", 0),
        ("evidence", "ledger", 0, "start"),
        ("evidence", "ledger", 0, "start", "resource"),
        ("evidence", "ledger", 0, "terminal"),
        ("evidence", "ledger", 0, "start_reference"),
        ("evidence", "ledger", 0, "terminal_reference"),
        ("evidence", "ledger", 0, "terminal", "world_reference"),
        ("evidence", "ledger", 0, "graph_world_node", "g_cy", "__ndarray_ref__"),
        ("evidence", "ledger", 0, "catalog"),
        ("evidence", "ledger", 0, "catalog", "inferred_diagnostics", "labels"),
        ("evidence", "ledger", 0, "catalog", "known_sampled_truth", "direct_structure"),
        ("graph_patterns", "frequencies"),
    ],
)
def test_summary_resealed_unknown_fields_every_retained_boundary(
    tmp_path, strict_summary_fixture, section
):
    altered = deepcopy(strict_summary_fixture)
    _at_path(altered, section)["safe_to_run"] = True
    cap = altered["evidence"]["ledger"][0]
    for kind in ("start", "terminal"):
        if kind == "terminal":
            cap[kind]["start_sha256"] = cap["start_reference"]["sha256"]
        sha = sweep._sha256(sweep._world_bytes(cap[kind]))
        cap[kind + "_reference"].update(sha256=sha, locator=f"{kind}-{sha}.json")
    altered["bindings"]["ledger_digest"] = sweep.digest(altered["evidence"]["ledger"])
    ref = sweep.persist_report_evidence(tmp_path, "summary", altered)
    with pytest.raises(ValueError):
        sweep.read_compact_summary(tmp_path, ref)


@pytest.mark.parametrize(
    "mutation",
    [
        "orphan",
        "duplicate_omitted",
        "duplicate_chosen",
        "missing_terminal",
        "wrong_phase",
        "wrong_attempt",
        "in_progress",
        "wrong_start_digest",
        "terminal_symlink",
        "start_symlink",
        "bad_digest",
        "malformed_name",
        "counterfeit_field",
        "directory_terminal",
        "duplicate_start",
    ],
)
def test_summary_complete_durable_inventory_cannot_be_filtered(tmp_path, mutation):
    """OLD FAILED: omitted/orphan/conflicting durable terminals were not inventoried."""
    archive, args = _summary_setup(tmp_path)
    end = _summary_start(archive, args, outcome="failed", errors=[_stage("generation")])
    item = args["ledger"][0]
    if mutation.startswith("duplicate_") and mutation != "duplicate_start":
        favorable = dict(end, generation_outcome="completed", errors=[])
        other = sweep.persist_report_evidence(archive, "terminal", favorable)
        if mutation == "duplicate_chosen":
            item["terminal"] = other
    elif mutation == "duplicate_start":
        start = sweep._report_file(archive, item["start"])
        start["resource"]["pid"] += 1
        sweep.persist_report_evidence(archive, "start", start)
    elif mutation == "orphan":
        end["attempt"] = args["allocation"]["manifest"][1]
        sweep.persist_report_evidence(archive, "terminal", end)
    elif mutation == "missing_terminal":
        (archive / item["terminal"]["locator"]).unlink()
    elif mutation in ("terminal_symlink", "start_symlink", "directory_terminal", "bad_digest"):
        kind = "start" if mutation == "start_symlink" else "terminal"
        path = archive / item[kind]["locator"]
        raw = path.read_bytes()
        path.unlink()
        if mutation.endswith("symlink"):
            target = tmp_path / "outside-evidence.json"
            target.write_bytes(raw)
            path.symlink_to(target)
        elif mutation == "directory_terminal":
            path.mkdir()
        else:
            path.write_bytes(raw + b" ")
    elif mutation == "malformed_name":
        (archive / "terminal-not-a-digest.json").write_text("{}")
    else:
        (archive / item["terminal"]["locator"]).unlink()
        if mutation == "wrong_phase":
            end["phase"] = "calibration"
        elif mutation == "wrong_attempt":
            end["attempt"] = dict(end["attempt"], world_seed=1)
        elif mutation == "in_progress":
            end["status"] = "in_progress"
        elif mutation == "wrong_start_digest":
            end["start_sha256"] = "0" * 64
        else:
            end["safe_to_run"] = True
        item["terminal"] = sweep.persist_report_evidence(archive, "terminal", end)
    before = sorted((p.name, p.lstat().st_mode, p.lstat().st_size) for p in archive.iterdir())
    with pytest.raises((ValueError, OSError)):
        sweep.build_compact_summary(archive=archive, **args)
    assert before == sorted(
        (p.name, p.lstat().st_mode, p.lstat().st_size) for p in archive.iterdir()
    )


@pytest.mark.parametrize("calibration", [None, [], [{"status": "failed", "reason": "fixture"}]])
@pytest.mark.parametrize(
    "mutation",
    [
        "beyond_elapsed",
        "overlaps_calibration",
        "negative",
        "boolean",
        "nan",
        "inf",
        "elapsed_before_calibration",
    ],
)
def test_summary_empty_ledger_global_timing_rejects(tmp_path, calibration, mutation):
    """OLD FAILED: finalization constraints disappeared when no row loop ran."""
    archive, args = _summary_setup(tmp_path, ncap=0, calibration=calibration)
    timing = args["timing"]
    if mutation == "beyond_elapsed":
        timing["finalization_seconds"] = timing["elapsed_seconds"] + 1
    elif mutation == "overlaps_calibration":
        timing["finalization_seconds"] = 1
    elif mutation == "elapsed_before_calibration":
        timing["elapsed_seconds"] -= 1
    else:
        timing["finalization_seconds"] = {
            "negative": -1,
            "boolean": True,
            "nan": float("nan"),
            "inf": float("inf"),
        }[mutation]
    with pytest.raises(ValueError):
        sweep.build_compact_summary(archive=archive, **args)


def test_summary_old_failed_resealed_empty_finalization(tmp_path):
    archive, args = _summary_setup(tmp_path, ncap=0)
    summary = sweep.build_compact_summary(archive=archive, **args)
    elapsed = summary["runtime"]["elapsed_seconds"]
    summary["evidence"]["timing"]["finalization_seconds"] = elapsed + 1
    summary["runtime"]["finalization_seconds"] = elapsed + 1
    summary["runtime"]["reserve_overrun_seconds"] = elapsed + 1 - 300
    ref = sweep.persist_report_evidence(tmp_path, "summary", summary)
    with pytest.raises(ValueError):
        sweep.read_compact_summary(tmp_path, ref)


@pytest.mark.parametrize("partial", [False, True])
def test_summary_zero_and_partial_inventory_and_timing_positive(tmp_path, partial):
    archive, args = _summary_setup(tmp_path, ncap=12 if partial else 0)
    if partial:
        _summary_start(
            archive,
            args,
            outcome="completed",
            graph=_empty_graph(args["allocation"]["manifest"][0]),
            errors=[_stage("persistence")],
        )
        _summary_start(archive, args, errors=[_stage("timeout")])
        _summary_start(archive, args, outcome="failed", errors=[_stage("generation")])
    args["timing"]["elapsed_seconds"] += 301
    args["timing"]["finalization_seconds"] = 301
    before = _archive_inventory(archive)
    summary = sweep.build_compact_summary(archive=archive, **args)
    assert _archive_inventory(archive) == before
    assert summary["counts"]["A"] == (3 if partial else 0)
    assert summary["counts"]["A"] == sum(
        summary["counts"][key]
        for key in ("G", "confirmed_generation_failed", "evidence_unavailable")
    )
    assert summary["runtime"]["reserve_overrun_seconds"] == 1
    assert summary["runtime"]["deadline_overrun_seconds"] == max(
        0, args["timing"]["elapsed_seconds"] - 5400
    )
    assert sweep.validate_compact_summary(summary) == summary
    if partial:
        assert summary["counts"]["G"] == 1 and summary["counts"]["E"] == 0
        assert summary["stage_failure_frequencies"]["persistence"]["numerator"] == 1
        assert summary["stage_failure_frequencies"]["timeout"]["numerator"] == 1


# Runner probes use fake clocks/children only; no scheduled world is generated.
class _EngineeringClock:
    def __init__(self, value=10000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _engineering_execution(tmp_path, monkeypatch, clock, *, setup=0):
    # Keep real runtime/source/archive checking; fake only elapsed setup cost.
    original = sweep._execution_sources
    calls = []

    def sources():
        calls.append(clock())
        if len(calls) == 1:
            clock.advance(setup)
        return original()

    monkeypatch.setattr(sweep, "_execution_sources", sources)
    return sweep._freeze_execution(config(), tmp_path, clock(), clock=clock)


def _engineering_launch(clock, events, *, science="unknown", calibration_failure=None):
    def launch(archive, request, deadline, **kwargs):
        assert deadline > clock()
        assert (
            sweep._execution_read(archive, "execution.json")["input"]["contract"][
                "seed_manifest_digest"
            ]
            == "95dd82f378da6890b72e82e7ff931568219e10ce6f5a213fba334e219e7d3291"
        )
        if request["mode"] == "finalize":
            events.append("finalize")
            execution = sweep._execution_read(archive, "execution.json")
            sweep._finalize_execution(archive, execution, clock=clock)
            return {
                "returncode": 0,
                "timed_out": False,
                "launched": True,
                "pid": 123,
                "started": clock(),
                "finished": clock(),
            }
        execution = sweep._execution_read(archive, "execution.json")
        start = sweep._load_attempt_start(archive, execution, request)
        attempt, phase, tag = start["attempt"], start["phase"], request["tag"]
        events.append(("launch", phase, attempt["attempt"]))
        if phase == "science":
            assert (
                sweep._execution_read(archive, "allocation.json")["allocation"]["manifest"][
                    attempt["attempt"]
                ]
                == attempt
            )
        generated = phase == "calibration" and calibration_failure != attempt["attempt"]
        generated |= phase == "science" and science in ("graph", "missing")
        if generated:
            sweep._execution_record(
                archive,
                f"generated-{tag}.json",
                {"attempt": attempt, "phase": phase, "start_sha256": request["start"]["sha256"]},
            )
        if phase == "calibration" and generated:
            resolved = sweep.resolved_generation_config(config(), attempt, phase=phase)
            result = {
                "generation": {
                    "phase": phase,
                    "resolved_config": resolved,
                    "resolved_config_digest": sweep.digest(resolved),
                    "config_digest": sweep.digest(config()),
                    "generation_seed": attempt["world_seed"],
                },
                "status": {"truth_eligibility": "unsupported_truth"},
            }
            sweep.persist_world(archive, attempt, result)
            sweep._execution_record(
                archive, f"completed-{tag}.json", {"start_sha256": request["start"]["sha256"]}
            )
        elif phase == "science" and science == "graph":
            sweep._durable_report(
                archive,
                "graph",
                {"attempt": attempt, "phase": phase, "graph": _empty_graph(attempt)},
            )
        elif phase == "science" and science == "generation_failed":
            sweep._execution_record(
                archive,
                f"error-{tag}.json",
                {"start_sha256": request["start"]["sha256"], "error": _stage("generation")},
            )
        clock.advance(1)
        return {
            "returncode": 0 if generated else -9,
            "timed_out": False,
            "launched": True,
            "pid": 123,
            "started": clock() - 1,
            "finished": clock(),
        }

    return launch


@pytest.mark.parametrize("science", ["unknown", "graph", "missing", "generation_failed"])
def test_runner_durable_order_setup_clock_and_failure_partition(tmp_path, monkeypatch, science):
    clock = _EngineeringClock()
    t0 = clock()
    # E=5095+3, C=2 => one science attempt, with all setup included.
    original = sweep._execution_sources
    first = True
    events = []

    def sources():
        nonlocal first
        if first:
            clock.advance(5095)
            first = False
        return original()

    monkeypatch.setattr(sweep, "_execution_sources", sources)
    original_write = sweep._execution_file

    def durable(archive, name, payload=None):
        value = original_write(archive, name, payload)
        if payload is not None:
            events.append(name)
        return value

    monkeypatch.setattr(sweep, "_execution_file", durable)
    monkeypatch.setattr(
        sweep, "generate_configured_world", lambda *a, **k: pytest.fail("no actual worlds")
    )
    receipt = sweep.execute_science(
        config(),
        tmp_path,
        t0=t0,
        clock=clock,
        launch=_engineering_launch(clock, events, science=science),
    )
    archive = Path(receipt["archive"])
    assert receipt["allocation_count"] == 1
    frozen = sweep._execution_read(archive, "allocation.json")
    assert frozen["allocation"]["inputs"]["elapsed_after_calibration"] == 5098
    assert [r["attempt"]["world_seed"] for r in frozen["calibration"]] == [
        2901121193,
        1825999679,
        2105574551,
    ]
    assert [e for e in events if isinstance(e, tuple)] == [
        ("launch", "calibration", 1000),
        ("launch", "calibration", 1001),
        ("launch", "calibration", 1002),
        ("launch", "science", 0),
    ]
    assert (
        events.index("execution.json")
        < events.index("calibration-start-1000.json")
        < events.index(("launch", "calibration", 1000))
    )
    si = next(i for i, e in enumerate(events) if isinstance(e, str) and e.startswith("start-"))
    ti = next(i for i, e in enumerate(events) if isinstance(e, str) and e.startswith("terminal-"))
    assert (
        events.index("allocation.json")
        < si
        < events.index(("launch", "science", 0))
        < ti
        < events.index("finalize")
    )
    summary = sweep.read_compact_summary(
        archive, receipt["finalization"]["summary"], raw_archive=archive
    )["summary"]
    rows = sweep._report_rows(
        config(),
        summary["evidence"]["input"],
        summary["evidence"]["allocation"],
        summary["evidence"]["ledger"],
        summary["evidence"]["timing"],
    )
    assert len(rows) == 1
    assert rows[0]["G"] == (science == "graph")
    assert rows[0]["E"] is False
    assert rows[0]["confirmed_generation_failed"] == (science == "generation_failed")
    if science == "missing":
        assert rows[0]["missing_evidence_status"] == "known_generation_completed_missing_evidence"
    if science == "unknown":
        assert rows[0]["missing_evidence_status"] == "generation_outcome_unknown"
    assert not list(archive.glob("catalog-*.json"))
    assert not any("1000" in c["start"]["attempt"] for c in summary["evidence"]["ledger"])


@pytest.mark.parametrize("bad", [1000, 1001, 1002])
def test_runner_invalid_calibration_retains_all_three_and_zero_cap(tmp_path, monkeypatch, bad):
    clock, events = _EngineeringClock(), []
    monkeypatch.setattr(
        sweep, "generate_configured_world", lambda *a, **k: pytest.fail("no worlds")
    )
    receipt = sweep.execute_science(
        config(),
        tmp_path,
        t0=clock(),
        clock=clock,
        launch=_engineering_launch(clock, events, calibration_failure=bad),
    )
    frozen = sweep._execution_read(receipt["archive"], "allocation.json")
    assert receipt["allocation_count"] == 0
    assert len(frozen["calibration"]) == 3
    assert frozen["allocation"]["allocation_reason"] == "calibration_invalid"
    assert (
        frozen["allocation"]["calibration_invalid_reason"]
        == "calibration completion: frozen value mismatch"
    )
    assert not list(Path(receipt["archive"]).glob("start-*.json"))
    assert len(list(Path(receipt["archive"]).glob("calibration-terminal-*.json"))) == 3


@pytest.mark.parametrize("ncap", [0, 399, 400, 1000])
def test_runner_count_math_and_precision_flag(ncap):
    elapsed = 5100 - 2 * ncap
    frozen = sweep.allocate_from_calibration(config(), elapsed, _completed_calibration((1, 1, 1)))
    assert frozen["frozen_attempt_count"] == ncap
    assert frozen["pooled_precision_adequate"] == (ncap >= 400)
    assert len(frozen["selection"]["band_examples"]) == 9


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_runner_invalid_cost_never_allocates(duration):
    frozen = sweep.allocate_from_calibration(
        config(), 100, _completed_calibration((1, duration, 1))
    )
    assert frozen["frozen_attempt_count"] == 0
    assert frozen["allocation_reason"] == "calibration_invalid"


def test_runner_parent_exception_preserves_cap_and_recovers_one_terminal(tmp_path, monkeypatch):
    clock, events = _EngineeringClock(), []
    normal = _engineering_launch(clock, events)

    def launch(archive, request, deadline, **kwargs):
        if request.get("phase") == "science":
            clock.advance(1)
            raise RuntimeError("engineering parent launch failure")
        return normal(archive, request, deadline, **kwargs)

    result = sweep.execute_science(config(), tmp_path, t0=clock(), clock=clock, launch=launch)
    assert result["allocation_count"] == 1000
    archive = Path(result["archive"])
    assert (
        len(list(archive.glob("start-*.json"))) == len(list(archive.glob("terminal-*.json"))) == 1
    )
    execution = sweep._verify_execution(archive, clock=clock)
    allocation = sweep._execution_read(archive, "allocation.json")["allocation"]
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert len(list(archive.glob("terminal-*.json"))) == 1
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    verified = sweep.verify_execution(archive, clock=clock)
    assert verified["allocation_count"] == 1000 and verified["started_count"] == 1
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


def test_runner_late_5120_no_science_and_final_reserve(tmp_path):
    clock, events = _EngineeringClock(), []
    normal = _engineering_launch(clock, events)

    def launch(archive, request, deadline, **kwargs):
        result = normal(archive, request, deadline, **kwargs)
        if request.get("tag") == "1002":
            clock.advance(5117)
        return result

    result = sweep.execute_science(config(), tmp_path, t0=clock(), clock=clock, launch=launch)
    assert result["elapsed_seconds"] == 5120
    assert result["allocation_count"] == 0
    assert result["finalization"]["final_deadline_elapsed"] == 5400
    assert not list(Path(result["archive"]).glob("start-*.json"))


def test_runner_recovery_original_clock_and_no_source_switch(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    sweep._execution_record(
        archive,
        "allocation.json",
        {"calibration": _completed_calibration((1, 1, 1)), "allocation": allocation},
    )
    clock.advance(4)
    sweep._attempt_start(
        archive, execution, allocation, allocation["manifest"][0], "science", clock=clock
    )
    events = []
    result = sweep.recover_execution(
        archive, clock=clock, launch=_engineering_launch(clock, events)
    )
    assert result["finalization"] is not None
    assert events == ["finalize"]
    assert len(list(archive.glob("terminal-*.json"))) == 1
    before = len(list(archive.glob("terminal-*.json")))
    sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("no rerun"))
    assert len(list(archive.glob("terminal-*.json"))) == before
    monkeypatch.setattr(sweep, "_execution_sources", lambda: {})
    with pytest.raises(ValueError, match="frozen source bytes"):
        sweep.recover_execution(archive, clock=clock)


def test_runner_recovery_after_original_deadline_refuses_new_clock(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    sweep._execution_record(
        archive,
        "allocation.json",
        {"calibration": _completed_calibration((1, 1, 1)), "allocation": allocation},
    )
    clock.value = execution["final_deadline"]
    with pytest.raises(TimeoutError, match="original"):
        sweep.recover_execution(
            archive, clock=clock, launch=lambda *a, **k: pytest.fail("deadline")
        )


@pytest.mark.parametrize("kind", ["overwrite", "symlink", "directory", "traversal"])
def test_runner_no_clobber_no_follow_execution_files(tmp_path, monkeypatch, kind):
    clock = _EngineeringClock()
    archive, _ = _engineering_execution(tmp_path, monkeypatch, clock)
    target = archive / "probe.json"
    if kind == "overwrite":
        target.write_bytes(b"keep")
    elif kind == "symlink":
        target.symlink_to(archive / "execution.json")
    elif kind == "directory":
        target.mkdir()
    with pytest.raises((OSError, ValueError)):
        sweep._execution_record(
            archive, "../probe.json" if kind == "traversal" else "probe.json", {}
        )
    if kind == "overwrite":
        assert target.read_bytes() == b"keep"
    with pytest.raises(FileExistsError):
        sweep._freeze_execution(config(), tmp_path, clock(), clock=clock)


@pytest.mark.parametrize(
    "variable", ["OPENBLAS_NUM_THREADS", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"]
)
def test_runner_bad_environment_no_archive_write(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, "wrong")
    with pytest.raises(ValueError):
        sweep._freeze_execution(config(), tmp_path, 1, clock=lambda: 2)
    assert list(tmp_path.iterdir()) == []


def test_runner_fsync_start_before_launch(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    calls = []
    original = sweep.os.fsync

    def sync(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(sweep.os, "fsync", sync)
    ref, start = sweep._attempt_start(
        archive, execution, allocation, allocation["manifest"][0], "science", clock=clock
    )
    assert len(calls) >= 2
    assert sweep._report_file(archive, ref) == start


@pytest.mark.parametrize("behavior", ["hang", "crash", "signal", "descendant"])
def test_runner_real_engineering_subprocess_kill_reap_logs(tmp_path, monkeypatch, behavior):
    import os
    import signal
    import subprocess
    import time

    archive = tmp_path / "engineering-process"
    archive.mkdir()
    real = subprocess.Popen
    seen = []
    code = {
        "hang": "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('retained', flush=True); time.sleep(30)",
        "crash": "raise RuntimeError('engineering child crash')",
        "signal": "import os,signal; os.kill(os.getpid(),signal.SIGTERM)",
        "descendant": "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)",
    }[behavior]

    def popen(command, **kwargs):
        assert command[0] == sweep._EXECUTION_PYTHON and command[1] == "-B"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["start_new_session"] is True
        process = real([sweep._EXECUTION_PYTHON, "-B", "-c", code], **kwargs)
        seen.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = sweep._timed_execution_child(
        archive, {"mode": "attempt", "tag": "probe"}, time.monotonic() + 0.5
    )
    assert seen[0].poll() is not None
    assert result["returncode"] is not None
    assert (archive / "child-probe.log").exists()
    if behavior in ("hang", "descendant"):
        assert result["timed_out"] is True and result["returncode"] == -signal.SIGKILL
    if behavior == "descendant":
        pid = int((archive / "child-probe.log").read_text().strip())
        for _ in range(100):
            path = Path(f"/proc/{pid}/stat")
            if not path.exists():
                break
            time.sleep(0.01)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("running orphan descendant")


@pytest.mark.parametrize("command", ["--help", "validate", "dry-input"])
def test_runner_cli_read_only_no_generation(tmp_path, command):
    import os
    import subprocess

    args = [sweep._EXECUTION_PYTHON, "-B", "scripts/linear_recovery_true_feature_sweep.py", command]
    if command != "--help":
        args += ["--config", "docs/examples/data/linear-recovery-true-feature-sweep-config.json"]
    result = subprocess.run(args, capture_output=True, text=True, env=os.environ.copy(), timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    if command == "dry-input":
        body = json.loads(result.stdout)
        assert len(body["contract"]["seed_manifest"]["science"]) == 1000


def test_runner_import_no_clock_no_package_no_execution():
    import os
    import subprocess

    code = "import sys; import scripts.linear_recovery_true_feature_sweep as s; assert s._ENTRY_T0 is None; assert 'pymc_generator' not in sys.modules"
    result = subprocess.run(
        [sweep._EXECUTION_PYTHON, "-B", "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_runner_internal_child_cannot_invent_launch(tmp_path, monkeypatch):
    clock = _EngineeringClock(sweep.time.monotonic())
    archive, _ = _engineering_execution(tmp_path, monkeypatch, clock)
    with pytest.raises(FileNotFoundError):
        sweep._execution_child(
            {
                "archive": str(archive),
                "mode": "attempt",
                "tag": "0",
                "phase": "science",
                "start": {},
            }
        )


def _engineering_science_start(tmp_path, monkeypatch):
    clock = _EngineeringClock(sweep.time.monotonic())
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    clock.advance(3)
    calibration = _completed_calibration((1, 1, 1))
    allocation = sweep.allocate_from_calibration(config(), 3, calibration)
    sweep._execution_record(
        archive, "allocation.json", {"calibration": calibration, "allocation": allocation}
    )
    attempt = allocation["manifest"][0]
    ref, start = sweep._attempt_start(
        archive, execution, allocation, attempt, "science", clock=clock
    )
    request = {"mode": "attempt", "tag": "0", "phase": "science", "start": ref}
    return clock, archive, execution, allocation, request, start


@pytest.mark.parametrize("stage", ["generation", "analysis", "persistence"])
def test_runner_authentic_child_stage_failures_without_science(tmp_path, monkeypatch, stage):
    from types import SimpleNamespace

    clock, archive, execution, allocation, request, start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    attempt = start["attempt"]

    def fail(*args, **kwargs):
        raise RuntimeError("ENGINEERING injected stage failure")

    monkeypatch.setattr(
        sweep,
        "generate_configured_world",
        fail if stage == "generation" else lambda *a, **k: SimpleNamespace(g=_empty_graph(attempt)),
    )
    monkeypatch.setattr(
        sweep, "analyze_configured_world", fail if stage == "analysis" else lambda *a, **k: {}
    )
    if stage == "persistence":
        monkeypatch.setattr(sweep, "persist_world", fail)
    assert sweep._attempt_child(archive, execution, request) == 1
    clock.advance(1)
    ref, cost = sweep._finish_attempt(
        archive,
        execution,
        request["start"],
        start,
        {"returncode": 1, "timed_out": False},
        clock=clock,
    )
    terminal = sweep._report_file(archive, ref)
    assert terminal["errors"][0]["stage"] == stage
    assert terminal["generation_outcome"] == ("failed" if stage == "generation" else "completed")
    assert (terminal["graph_reference"] is not None) == (stage != "generation")
    assert terminal["world_reference"] is None and cost["complete"] is False
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert len(list(archive.glob("terminal-*.json"))) == 1


def test_runner_partial_reservation_and_graph_retained_without_full_world(tmp_path, monkeypatch):
    clock, archive, execution, allocation, request, start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    attempt = start["attempt"]
    sweep._durable_report(
        archive, "graph", {"attempt": attempt, "phase": "science", "graph": _empty_graph(attempt)}
    )
    _, reference, reservation, _, _ = sweep._prepare_world(archive, attempt, {"incomplete": True})
    sweep._write_world_file(archive, reference["reservation_locator"], "reservation", reservation)
    clock.advance(1)
    ref, _ = sweep._finish_attempt(
        archive,
        execution,
        request["start"],
        start,
        {"returncode": -9, "timed_out": True},
        clock=clock,
    )
    end = sweep._report_file(archive, ref)
    assert end["generation_outcome"] == "completed" and end["world_reference"] is None
    assert {e["stage"] for e in end["errors"]} == {"persistence", "timeout"}
    assert end["partial_references"] == [
        {
            "kind": "reservation",
            "locator": reference["reservation_locator"],
            "sha256": reference["reservation_sha256"],
        }
    ]
    assert (archive / reference["reservation_locator"]).read_bytes() == reservation
    inventory = sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert len(inventory) == 1


def test_runner_partial_world_blobs_retained_and_verified(tmp_path, monkeypatch):
    clock, archive, execution, _, request, start = _engineering_science_start(tmp_path, monkeypatch)
    attempt = start["attempt"]
    # Deliberately not an analyzed world: raw evidence may survive a failed analysis.
    reference = sweep.persist_world(archive, attempt, {"partial": np.arange(8.0)})
    clock.advance(1)
    ref, _ = sweep._finish_attempt(
        archive,
        execution,
        request["start"],
        start,
        {"returncode": -9, "timed_out": False},
        clock=clock,
    )
    terminal = sweep._report_file(archive, ref)
    assert terminal["world_reference"] is None
    assert {p["kind"] for p in terminal["partial_references"]} == {"reservation", "world", "array"}
    assert terminal["generation_outcome"] == "unknown"
    for partial in terminal["partial_references"]:
        assert (
            sweep._sha256(sweep._read_world_file(archive, partial["locator"], partial["kind"]))
            == partial["sha256"]
        )
    assert sweep.read_world(archive, reference)["result"]["partial"].tolist() == list(range(8))


def test_runner_parent_calibration_exception_reconciles_start(tmp_path):
    clock = _EngineeringClock()

    def launch(*args, **kwargs):
        clock.advance(2)
        raise RuntimeError("engineering calibration launch exception")

    # Finalizer uses the real engineering helper; only the calibration launch fails.
    normal = _engineering_launch(clock, [])
    result = sweep.execute_science(
        config(),
        tmp_path,
        t0=clock(),
        clock=clock,
        launch=lambda archive, request, deadline, **kwargs: (
            normal(archive, request, deadline, **kwargs)
            if request["mode"] == "finalize"
            else launch()
        ),
    )
    archive = Path(result["archive"])
    assert result["allocation_count"] == 0
    assert (
        len(list(archive.glob("calibration-start-*.json")))
        == len(list(archive.glob("calibration-terminal-*.json")))
        == 1
    )
    frozen = sweep._execution_read(archive, "allocation.json")
    assert len(frozen["calibration"]) == 1 and frozen["calibration"][0]["status"] == "failed"
    assert (
        frozen["allocation"]["calibration_invalid_reason"]
        == "all three calibration records required"
    )


def test_runner_science_slow_child_uses_remaining_cutoff_and_does_not_extend(tmp_path):
    clock, events = _EngineeringClock(), []
    normal = _engineering_launch(clock, events)
    cutoffs = []

    def launch(archive, request, deadline, **kwargs):
        if request.get("phase") == "science":
            cutoffs.append(deadline - clock())
            assert deadline - clock() == 5097  # not heuristic C=2
            clock.value = deadline + 0.25
            return {
                "returncode": -9,
                "timed_out": True,
                "launched": True,
                "started": deadline - 5097,
                "finished": clock(),
                "pid": 12,
            }
        return normal(archive, request, deadline, **kwargs)

    result = sweep.execute_science(config(), tmp_path, t0=clock(), clock=clock, launch=launch)
    assert cutoffs == [5097] and result["allocation_count"] == 1000
    archive = Path(result["archive"])
    assert len(list(archive.glob("start-*.json"))) == 1
    cost = sweep._execution_read(archive, "cost-0.json")
    assert cost["science_overrun_seconds"] == 0.25
    assert result["elapsed_seconds"] == 5100.25
    assert result["finalization"] is not None


def test_runner_finalizer_timeout_preserves_inventory_and_actual_cap(tmp_path):
    clock, events = _EngineeringClock(), []
    normal = _engineering_launch(clock, events)

    def launch(archive, request, deadline, **kwargs):
        if request["mode"] == "finalize":
            assert deadline == 15400
            clock.value = deadline + 0.1
            return {
                "returncode": -9,
                "timed_out": True,
                "launched": True,
                "started": deadline - 300,
                "finished": clock(),
                "pid": 99,
            }
        if request.get("phase") == "science":
            clock.value = deadline
            return {
                "returncode": -9,
                "timed_out": True,
                "launched": True,
                "started": deadline - 1,
                "finished": clock(),
                "pid": 98,
            }
        return normal(archive, request, deadline, **kwargs)

    result = sweep.execute_science(config(), tmp_path, t0=clock(), clock=clock, launch=launch)
    assert result["allocation_count"] == 1000
    assert result["finalization"] is None and result["process"]["timed_out"] is True
    assert result["final_overrun_seconds"] > 0
    assert len(list(Path(result["archive"]).glob("terminal-*.json"))) == 1


@pytest.mark.parametrize("target", ["execution.json", "run.json", "schema"])
def test_runner_archive_drift_rejected_before_launch(tmp_path, monkeypatch, target):
    clock = _EngineeringClock()
    archive, _ = _engineering_execution(tmp_path, monkeypatch, clock)
    if target == "schema":
        path = archive / "schemas/ols-world-catalog-v1.json"
        path.write_bytes(b"{}\n")
    else:
        path = archive / target
        body = json.loads(path.read_bytes())
        if target == "execution.json":
            body["sources"] = {}
        else:
            body["source_commit"] = "0" * 40
        path.write_bytes(sweep._world_bytes(body))
    with pytest.raises(ValueError):
        sweep._verify_execution(archive, clock=clock)


def test_runner_duplicate_terminal_recovery_rejects_without_further_writes(tmp_path, monkeypatch):
    clock, archive, execution, allocation, request, start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    clock.advance(1)
    ref, _ = sweep._finish_attempt(
        archive,
        execution,
        request["start"],
        start,
        {"returncode": -9, "timed_out": False},
        clock=clock,
    )
    end = sweep._report_file(archive, ref)
    end["finished_elapsed"] += 0.5
    sweep._durable_report(archive, "terminal", end)
    before = sorted(p.name for p in archive.iterdir())
    with pytest.raises(ValueError, match="duplicate"):
        sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert before == sorted(p.name for p in archive.iterdir())


# Recovery regressions are fake-child engineering only, never scientific runs.
@pytest.mark.parametrize("window", ["before_cost", "before_terminal", "after_terminal"])
def test_runner_recovery_retains_actual_exit_before_terminal(tmp_path, monkeypatch, window):
    import subprocess

    clock, archive, execution, allocation, request, start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    real = subprocess.Popen
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: real(
            [
                sweep._EXECUTION_PYTHON,
                "-B",
                "-c",
                "import os; os.write(1,b'OUT\\x00'); os.write(2,b'ERR\\xff'); raise SystemExit(23)",
            ],
            **kwargs,
        ),
    )
    process = sweep._timed_execution_child(
        archive, request, execution["science_deadline"], clock=clock
    )
    assert process["returncode"] == 23
    assert process["resources"]["user_seconds"] >= 0
    for stream, expected in (("stdout", b"OUT\x00"), ("stderr", b"ERR\xff")):
        ref = process["streams"][stream]
        raw = sweep._execution_file(archive, ref["locator"])
        assert raw == expected and ref["sha256"] == sweep._sha256(expected)
        assert ref["bytes"] == len(expected)
    assert sweep._execution_read(archive, "process-result-0.json") == process
    original = sweep._execution_file

    class ParentCrash(BaseException):
        pass

    def crash(archive, name, payload=None):
        if payload is not None and (
            (window == "before_cost" and name == "cost-0.json")
            or (window == "before_terminal" and name.startswith("terminal-"))
        ):
            raise ParentCrash()
        result = original(archive, name, payload)
        if payload is not None and name.startswith("terminal-"):
            cost = sweep._execution_read(archive, "cost-0.json")
            assert cost["process"] == process
            if window == "after_terminal":
                raise ParentCrash()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sweep, "_execution_file", crash)
        with pytest.raises(ParentCrash):
            sweep._finish_attempt(archive, execution, request["start"], start, process, clock=clock)
    retained = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    clock.advance(17)
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert len(list(archive.glob("terminal-*.json"))) == 1
    assert sweep._execution_read(archive, "cost-0.json")["process"]["returncode"] == 23
    for name, raw in retained.items():
        assert (archive / name).read_bytes() == raw


@pytest.mark.parametrize("window", ["start", "process_result", "cost", "terminal"])
def test_runner_crashed_calibration_no_allocation_recovers_once(tmp_path, monkeypatch, window):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
    ref, start = sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    clock.advance(7)
    process = {"returncode": 19, "timed_out": False, "resources": {"user_seconds": 0.25}}
    if window != "start":
        sweep._retain_process_result(archive, "1000", process)
    if window in ("cost", "terminal"):
        original = sweep._execution_file

        def stop(archive, name, payload=None):
            if (
                window == "cost"
                and payload is not None
                and name == "calibration-terminal-1000.json"
            ):
                raise KeyboardInterrupt("engineering hard crash")
            return original(archive, name, payload)

        with monkeypatch.context() as patch:
            patch.setattr(sweep, "_execution_file", stop)
            if window == "cost":
                with pytest.raises(KeyboardInterrupt):
                    sweep._finish_attempt(archive, execution, ref, start, process, clock=clock)
            else:
                sweep._finish_attempt(archive, execution, ref, start, process, clock=clock)
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    events = []
    final = sweep.recover_execution(archive, clock=clock, launch=_engineering_launch(clock, events))
    assert final["finalization"] and events == ["finalize"]
    frozen = sweep._execution_read(archive, "allocation.json")
    assert frozen["allocation"]["frozen_attempt_count"] == 0
    assert frozen["allocation"]["allocation_reason"] == "calibration_invalid"
    assert frozen["allocation"]["inputs"]["conservative_per_world_cost"] is None
    assert frozen["allocation"]["inputs"]["elapsed_after_calibration"] == 7
    assert len(frozen["calibration"]) == 1 and frozen["calibration"][0]["duration"] == 7
    cost = sweep._execution_read(archive, "cost-1000.json")
    assert cost["process"]["returncode"] == (None if window == "start" else 19)
    if window == "start":
        assert cost["process"]["exit_unknown_reason"]
        assert cost["process"]["resources"] is None
    after = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("no rerun"))
    assert after == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    assert all(after[name] == raw for name, raw in before.items())
    assert not list(archive.glob("start-*.json"))


@pytest.mark.parametrize("window", ["early", "prepared", "summary"])
def test_runner_finalization_real_launch_retry_preserves_failed_attempts(
    tmp_path, monkeypatch, window
):
    import subprocess

    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    frozen = {"calibration": [], "allocation": sweep.allocate_from_calibration(config(), 0, [])}
    sweep._execution_record(archive, "allocation.json", frozen)
    real = subprocess.Popen
    launches = []
    # A real process runs a fake child, reading the genuine parent gate. No worlds.
    code = """import json,sys,os
import scripts.linear_recovery_true_feature_sweep as s
request=json.load(sys.stdin); archive=request["archive"]
e=s._execution_read(archive,"execution.json")
print("engineering stdout",flush=True); print("engineering stderr",file=sys.stderr,flush=True)
WINDOW=__WINDOW__
if WINDOW == "early": raise SystemExit(29)
original=s._execution_file
def write(archive,name,payload=None):
    if payload is not None and ((WINDOW == "prepared" and name.startswith("summary-")) or (WINDOW == "summary" and name == "finalization.json")):
        raise SystemExit(29)
    return original(archive,name,payload)
s._execution_file=write
s._finalize_execution(archive,e,clock=lambda:e["t0"]+10)
"""

    def popen(command, **kwargs):
        if command[-1] != "_child":
            return real(command, **kwargs)
        selected = window if not launches else "success"
        launches.append(command)
        return real(
            [sweep._EXECUTION_PYTHON, "-B", "-c", code.replace("__WINDOW__", repr(selected))],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", popen)
    failed = sweep._bounded_finalize(archive, execution, clock=clock)
    assert failed["process"]["returncode"] == 29 and failed["finalization"] is None
    assert (
        sweep._execution_read(archive, "process-finalize.json")["deadline"]
        == execution["t0"] + 5400
    )
    old = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    clock.advance(20)
    recovered = sweep.recover_execution(archive, clock=clock)
    assert recovered["process"]["returncode"] == 0 and recovered["finalization"]
    assert (
        sweep._execution_read(archive, "process-finalize-1.json")["deadline"]
        == execution["final_deadline"]
    )
    assert len(list(archive.glob("summary-*.json"))) == 1
    assert len(list(archive.glob("finalization.json"))) == 1
    for name, raw in old.items():
        assert (archive / name).read_bytes() == raw
    snapshot = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    sweep.recover_execution(archive, clock=clock)
    sweep.verify_execution(archive, clock=clock)
    assert len(launches) == 2
    assert snapshot == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


def test_runner_missing_allocation_expired_clock_freezes_honest_failure(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
    sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    clock.value = execution["final_deadline"] + 13
    with pytest.raises(TimeoutError, match="original"):
        sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("expired"))
    frozen = sweep._execution_read(archive, "allocation.json")
    assert frozen["allocation"]["frozen_attempt_count"] == 0
    assert frozen["allocation"]["inputs"]["elapsed_after_calibration"] == 5413
    assert sweep._execution_read(archive, "recovery-finished.json")["final_overrun_seconds"] == 13
    assert not (archive / "finalization.json").exists()


def test_runner_remaining_deadline_measured_after_spawn_receipt(tmp_path, monkeypatch):
    import subprocess

    archive = tmp_path / "fake-process"
    archive.mkdir()
    clock = _EngineeringClock()
    observed = []

    class Child:
        pid = 123456789
        stdin = None
        returncode = 0

        def communicate(self, payload, timeout):
            observed.append(timeout)

        def wait(self):
            return 0

    def popen(*args, **kwargs):
        clock.advance(9)
        return Child()

    monkeypatch.setattr(subprocess, "Popen", popen)
    # Nonexistent-PID orchestration fixture only; real ownership is tested below.
    monkeypatch.setattr(sweep, "_kill_group", lambda *a, **k: {"all_disappeared": True})
    sweep._timed_execution_child(
        archive, {"tag": "remaining", "mode": "finalize"}, clock() + 13, clock=clock
    )
    assert observed == [4]


def test_runner_all_calibration_costs_retained_without_reallocation(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    launch = _engineering_launch(clock, [])
    for attempt in execution["input"]["contract"]["seed_manifest"]["calibration"]:
        sweep._run_attempt(
            archive, execution, None, attempt, "calibration", clock=clock, launch=launch
        )
    old = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    sweep.recover_execution(archive, clock=clock, launch=launch)
    receipt = sweep._execution_read(archive, "calibration-recovery.json")
    assert [r["duration"] for r in receipt["records"]] == [1, 1, 1]
    assert all(r["status"] == "complete" for r in receipt["records"])
    assert receipt["allocation"]["frozen_attempt_count"] == 0
    assert receipt["allocation"]["allocation_reason"] == "calibration_invalid"
    assert "original allocation" in receipt["calibration"][0]["reason"]
    assert all((archive / name).read_bytes() == raw for name, raw in old.items())
    assert not list(archive.glob("start-*.json"))


def test_runner_interrupted_calibration_freeze_replayed_without_new_elapsed(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    original = sweep._execution_record

    def crash(archive, name, body):
        if name == "allocation.json":
            raise KeyboardInterrupt("engineering crash after calibration recovery receipt")
        return original(archive, name, body)

    clock.advance(10)
    with monkeypatch.context() as patch:
        patch.setattr(sweep, "_execution_record", crash)
        with pytest.raises(KeyboardInterrupt):
            sweep.recover_execution(archive, clock=clock)
    clock.advance(90)
    sweep.recover_execution(archive, clock=clock, launch=_engineering_launch(clock, []))
    frozen = sweep._execution_read(archive, "allocation.json")
    assert frozen["allocation"]["inputs"]["elapsed_after_calibration"] == 10
    assert frozen["calibration"] == []
    assert frozen["allocation"]["frozen_attempt_count"] == 0


def test_runner_retry_versions_partial_legacy_logs_and_recovery_receipts(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    sweep._execution_record(
        archive,
        "allocation.json",
        {"calibration": [], "allocation": sweep.allocate_from_calibration(config(), 0, [])},
    )
    sweep._execution_file(archive, "child-finalize.log", b"old combined failure")
    old = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    tags = []

    def failed(archive, request, deadline, **kwargs):
        tags.append(request["tag"])
        assert deadline == execution["final_deadline"]
        return {"returncode": 17, "timed_out": False}

    assert sweep.recover_execution(archive, clock=clock, launch=failed)["finalization"] is None
    assert sweep.recover_execution(archive, clock=clock, launch=failed)["finalization"] is None
    assert tags == ["finalize-1", "finalize-2"]
    assert (archive / "recovery-finished.json").exists()
    assert (archive / "recovery-finished-1.json").exists()
    streams = sweep._execution_read(archive, "process-result-finalize.json")["streams"]
    assert streams["stdout"] is streams["stderr"] is None
    assert streams["layout"] == "unknown" and "legacy_combined" not in streams
    assert streams["unclassified_stdout"] == {
        "locator": "child-finalize.log",
        "sha256": sweep._sha256(b"old combined failure"),
        "bytes": len(b"old combined failure"),
    }
    assert sweep._execution_file(archive, "child-finalize.log") == b"old combined failure"
    assert all((archive / name).read_bytes() == raw for name, raw in old.items())


def test_runner_runtime_cache_paths_and_thread_receipt_before_outcomes(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    runtime = execution["runtime"]
    assert runtime["python_executable"] == sweep._EXECUTION_PYTHON
    assert runtime["cwd"] == str(Path.cwd())
    assert runtime["package_origins"]["pymc_generator"] == str(
        Path.cwd() / "pymc_generator/__init__.py"
    )
    assert runtime["cache"]["compiledir"]
    assert "no purge" in runtime["cache"]["scope"]
    assert all(pool["num_threads"] == 1 for pool in runtime["threadpools"])
    assert not list(archive.glob("calibration-start-*.json"))
    assert not list(archive.glob("start-*.json"))


def test_runner_recovery_kills_reaps_recorded_calibration_group(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess
    import time

    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
    ref, _ = sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    code = "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    with (
        (archive / "child-1000.log").open("xb") as stdout,
        (archive / "child-1000-stderr.log").open("xb") as stderr,
    ):
        child = subprocess.Popen(
            [sweep._EXECUTION_PYTHON, "-B", "-c", code],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            sweep._execution_record(
                archive,
                "process-1000.json",
                {
                    "pid": child.pid,
                    "identity": sweep._process_identity(child.pid),
                    "request": {
                        "mode": "attempt",
                        "phase": "calibration",
                        "tag": "1000",
                        "start": ref,
                    },
                    "started": clock(),
                    "deadline": execution["science_deadline"],
                },
            )
            for _ in range(100):
                raw = (archive / "child-1000.log").read_bytes()
                if raw:
                    break
                time.sleep(0.01)
            assert raw
            descendant = int(raw)
            clock.advance(1)
            sweep.recover_execution(archive, clock=clock, launch=_engineering_launch(clock, []))
            result = sweep._execution_read(archive, "process-result-1000.json")
            assert result["returncode"] == -signal.SIGKILL
            assert result["exit_unknown_reason"] is None
            assert not Path(f"/proc/{child.pid}").exists()
            for _ in range(100):
                path = Path(f"/proc/{descendant}/stat")
                if not path.exists():
                    break
                time.sleep(0.01)
            else:
                pytest.fail("recovery left a running descendant")
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


@pytest.mark.parametrize("orphan", [False, True])
def test_runner_calibration_recovery_rejects_noncanonical_inventory(tmp_path, monkeypatch, orphan):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][1]
    if orphan:
        sweep._execution_record(archive, "calibration-terminal-1001.json", {})
    else:
        sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="nonprefix|orphan"):
        sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("invalid"))
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


# Bounded runtime/provenance seam: engineering fixtures only, no generated worlds.
@pytest.mark.parametrize(
    "field",
    [
        ("python_executable",),
        ("python_version",),
        ("cwd",),
        ("sys_path",),
        ("package_versions", "numpy"),
        ("package_versions", "pytensor"),
        ("package_versions", "pymc"),
        ("package_versions", "pymc-marketing"),
        ("package_origins", "numpy"),
        ("package_origins", "pytensor"),
        ("package_origins", "pymc"),
        ("package_origins", "pymc_generator"),
        ("cache", "pytensor_flags"),
        ("cache", "base_compiledir"),
        ("cache", "compiledir"),
        ("cache", "scope"),
        ("thread_scope",),
        ("threadpools", "num_threads"),
        ("threadpools", "version"),
        ("threadpools", "filepath"),
        ("threadpools", "prefix"),
        ("threadpools", "internal_api"),
        ("threadpools", "user_api"),
        ("threadpools", "threading_layer"),
        ("threadpools", "architecture"),
    ],
)
def test_runner_runtime_seam_drift_rejected_before_writes(tmp_path, monkeypatch, field):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    runtime = execution["runtime"]
    if field == ("sys_path",):
        runtime["sys_path"] = list(reversed(runtime["sys_path"])) + ["/different-path"]
    elif field[0] == "threadpools":
        assert runtime["threadpools"] and field[1] in runtime["threadpools"][0]
        runtime["threadpools"][0][field[1]] = 2 if field[1] == "num_threads" else "drift"
    elif len(field) == 2:
        runtime[field[0]][field[1]] = "drift"
    else:
        runtime[field[0]] = "drift"
    (archive / "execution.json").write_bytes(sweep._world_bytes(execution))
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="runtime"):
        sweep.recover_execution(
            archive, clock=clock, launch=lambda *a, **k: pytest.fail("no launch")
        )
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


def test_runner_runtime_seam_thread_constraint_precedes_archive(tmp_path, monkeypatch):
    runtime = sweep._execution_runtime()
    assert runtime["threadpools"]
    runtime["threadpools"][0]["num_threads"] = 2
    monkeypatch.setattr(sweep, "_execution_runtime", lambda: runtime)
    with pytest.raises(ValueError, match="exactly one thread"):
        sweep._freeze_execution(config(), tmp_path, sweep.time.monotonic())
    assert not list(tmp_path.iterdir())


def test_runner_runtime_seam_pool_order_and_new_pid_accept(tmp_path):
    import subprocess

    code = """
import json, os, sys, time
from pathlib import Path
from scripts import linear_recovery_true_feature_sweep as s
if sys.argv[1] == 'freeze':
    cfg = json.loads(Path('docs/examples/data/linear-recovery-true-feature-sweep-config.json').read_bytes())
    archive, execution = s._freeze_execution(cfg, Path(sys.argv[2]), time.monotonic())
else:
    archive = Path(sys.argv[2])
    execution = s._verify_execution(archive)
runtime = execution['runtime']
assert s._stable_execution_runtime(runtime) == s._stable_execution_runtime({**runtime, 'threadpools': list(reversed(runtime['threadpools']))})
print(json.dumps({'archive': str(archive), 'pid': os.getpid(), 'runtime': runtime}))
"""
    first = json.loads(
        subprocess.check_output(
            [sweep._EXECUTION_PYTHON, "-B", "-c", code, "freeze", str(tmp_path)], text=True
        )
    )
    second = json.loads(
        subprocess.check_output(
            [sweep._EXECUTION_PYTHON, "-B", "-c", code, "verify", first["archive"]], text=True
        )
    )
    assert first["pid"] != second["pid"]
    assert first["runtime"] == second["runtime"]


@pytest.mark.parametrize("window", ["before_stdout", "before_stderr"])
def test_runner_stream_seam_receipt_precedes_partial_opens(tmp_path, monkeypatch, window):
    import os
    import subprocess

    clock, archive, execution, _, request, _ = _engineering_science_start(tmp_path, monkeypatch)
    original = os.open
    stop = "child-0.log" if window == "before_stdout" else "child-0-stderr.log"

    def crash(name, flags, *args, **kwargs):
        if name == stop and flags & os.O_CREAT:
            assert sweep._execution_read(archive, "stream-layout-0.json") == {
                "tag": "0",
                "stdout": "child-0.log",
                "stderr": "child-0-stderr.log",
            }
            raise KeyboardInterrupt("stream open crash")
        return original(name, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", crash)
        patch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no launch"))
        with pytest.raises(KeyboardInterrupt, match="stream open crash"):
            sweep._timed_execution_child(
                archive, request, execution["science_deadline"], clock=clock
            )
    streams = sweep._process_streams(archive, "0")
    assert streams["layout"] == "separate" and streams["stderr"] is None
    assert "legacy_combined" not in streams
    assert streams["stdout"] == (
        None
        if window == "before_stdout"
        else {"locator": "child-0.log", "sha256": sweep._sha256(b""), "bytes": 0}
    )
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    process = sweep._recover_process(archive, "0", clock=clock)
    assert process["streams"] == streams and process["returncode"] is None
    assert process["resources"] is None and process["exit_unknown_reason"]
    assert all((archive / name).read_bytes() == raw for name, raw in before.items())


@pytest.mark.parametrize("receipt", ["missing", "counterfeit", "symlink"])
def test_runner_stream_seam_unknown_or_reject_not_legacy(tmp_path, monkeypatch, receipt):
    clock, archive, _, _, _, _ = _engineering_science_start(tmp_path, monkeypatch)
    raw = b"unknown\x00\xff"
    sweep._execution_file(archive, "child-0.log", raw)
    if receipt == "counterfeit":
        sweep._execution_record(
            archive,
            "stream-layout-0.json",
            {"tag": "0", "stdout": "child-1.log", "stderr": "child-0-stderr.log"},
        )
    elif receipt == "symlink":
        (archive / "stream-layout-0.json").symlink_to(archive / "execution.json")
    if receipt != "missing":
        before = sorted(p.name for p in archive.iterdir())
        with pytest.raises((ValueError, OSError)):
            sweep._recover_process(archive, "0", clock=clock)
        assert before == sorted(p.name for p in archive.iterdir())
    else:
        streams = sweep._process_streams(archive, "0")
        assert streams["layout"] == "unknown"
        assert streams["stdout"] is streams["stderr"] is None
        assert "legacy_combined" not in streams
        assert streams["unclassified_stdout"] == {
            "locator": "child-0.log",
            "sha256": sweep._sha256(raw),
            "bytes": len(raw),
        }


@pytest.mark.parametrize("phase", ["science", "calibration"])
@pytest.mark.parametrize("evidence", ["none", "result", "cost", "both"])
@pytest.mark.parametrize("window", ["before", "after"])
def test_runner_provenance_seam_existing_terminal_crash_idempotence(
    tmp_path, monkeypatch, phase, evidence, window
):
    import subprocess

    if phase == "science":
        clock, archive, execution, allocation, request, start = _engineering_science_start(
            tmp_path, monkeypatch
        )
        ref = request["start"]
    else:
        clock = _EngineeringClock()
        archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
        attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
        ref, start = sweep._attempt_start(archive, execution, None, attempt, phase, clock=clock)
        request = {"mode": "attempt", "phase": phase, "tag": str(attempt["attempt"]), "start": ref}
    tag = request["tag"]
    terminal = {
        "attempt": start["attempt"],
        "phase": phase,
        "start_sha256": ref["sha256"],
        "finished_elapsed": clock() - execution["t0"] + 1,
        "status": "recovered_unknown",
        "generation_outcome": "unknown",
        "errors": [],
        "world_reference": None,
        "catalog_reference": None,
        "graph_reference": None,
        "partial_references": [],
    }
    terminal_ref = (
        sweep._durable_report(archive, "terminal", terminal)
        if phase == "science"
        else sweep._execution_record(archive, f"calibration-terminal-{tag}.json", terminal)
    )
    # Reproduce terminal-before-cost history using a real retained wait, or no wait at all.
    if evidence != "none":
        real = subprocess.Popen
        with monkeypatch.context() as patch:
            patch.setattr(
                subprocess,
                "Popen",
                lambda command, **kwargs: real(
                    [
                        sweep._EXECUTION_PYTHON,
                        "-B",
                        "-c",
                        "import os; os.write(1,b'OUT\\x00'); os.write(2,b'ERR\\xff'); raise SystemExit(23)",
                    ],
                    **kwargs,
                ),
            )
            process = sweep._timed_execution_child(
                archive, request, execution["science_deadline"], clock=clock
            )
        assert process["returncode"] == 23 and process["resources"]["user_seconds"] >= 0
        if evidence in ("cost", "both"):
            sweep._execution_record(
                archive,
                f"cost-{tag}.json",
                {
                    "process": process,
                    "terminal": terminal_ref,
                    "terminal_body": terminal,
                    "duration": 1,
                    "complete": False,
                },
            )
        if evidence == "cost":
            (archive / f"process-result-{tag}.json").unlink()  # Fixture: old cost-only parent.
    else:
        sweep._execution_file(archive, f"child-{tag}.log", b"partial unknown\x00")
    old = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    reconcile = (
        (lambda: sweep._recover_terminals(archive, execution, allocation, clock=clock))
        if phase == "science"
        else (lambda: sweep._reconcile_calibration(archive, execution, clock=clock))
    )
    original = sweep._execution_file
    sidecar = f"process-provenance-{tag}.json"

    def crash(archive, name, payload=None):
        if name == sidecar and payload is not None and window == "before":
            raise KeyboardInterrupt("sidecar crash")
        value = original(archive, name, payload)
        if name == sidecar and payload is not None and window == "after":
            raise KeyboardInterrupt("sidecar crash")
        return value

    with monkeypatch.context() as patch:
        patch.setattr(sweep, "_execution_file", crash)
        with pytest.raises(KeyboardInterrupt, match="sidecar crash"):
            reconcile()
    assert (archive / sidecar).exists() == (window == "after")
    reconcile()
    first = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    clock.advance(9)
    reconcile()
    assert first == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    assert all(first[name] == raw for name, raw in old.items())
    assert set(first) - set(old) == {sidecar}
    receipt = sweep._execution_read(archive, sidecar)
    assert receipt["start"] == ref and receipt["terminal"] == terminal_ref
    if evidence == "none":
        assert receipt["process"]["returncode"] is receipt["process"]["resources"] is None
        assert (
            receipt["process"]["exit_unknown_reason"]
            and receipt["process"]["resource_unknown_reason"]
        )
        assert receipt["process"]["streams"]["layout"] == "unknown"
        assert not (archive / f"cost-{tag}.json").exists()
    else:
        assert receipt["process"] == process
        for stream, raw in (("stdout", b"OUT\x00"), ("stderr", b"ERR\xff")):
            stream_ref = receipt["process"]["streams"][stream]
            assert sweep._execution_file(archive, stream_ref["locator"]) == raw
            assert stream_ref["sha256"] == sweep._sha256(raw) and stream_ref["bytes"] == len(raw)
        for field in ("process_result", "cost"):
            source = receipt[field]
            if source is not None:
                assert source["sha256"] == sweep._sha256(old[source["locator"]])


@pytest.mark.parametrize("allocation_present", [False, True])
def test_runner_provenance_seam_calibration_existing_allocation_recovery(
    tmp_path, monkeypatch, allocation_present
):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
    ref, start = sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    terminal = {
        "attempt": attempt,
        "phase": "calibration",
        "start_sha256": ref["sha256"],
        "finished_elapsed": 1,
        "status": "recovered_unknown",
        "generation_outcome": "unknown",
        "errors": [],
        "world_reference": None,
        "catalog_reference": None,
        "graph_reference": None,
        "partial_references": [],
    }
    sweep._execution_record(archive, "calibration-terminal-1000.json", terminal)
    clock.advance(2)
    if allocation_present:
        sweep._execution_record(
            archive,
            "allocation.json",
            {"calibration": [], "allocation": sweep.allocate_from_calibration(config(), 2, [])},
        )
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    events = []
    receipt = sweep.recover_execution(
        archive, clock=clock, launch=_engineering_launch(clock, events)
    )
    assert receipt["finalization"] and events == ["finalize"]
    sidecar = sweep._execution_read(archive, "process-provenance-1000.json")
    assert sidecar["process"]["returncode"] is sidecar["process"]["resources"] is None
    assert (
        sidecar["process"]["exit_unknown_reason"] and sidecar["process"]["resource_unknown_reason"]
    )
    assert not (archive / "cost-1000.json").exists()
    assert all((archive / name).read_bytes() == raw for name, raw in before.items())
    first = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("no rerun"))
    assert first == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "tamper", ["duplicate", "terminal", "exit", "resource", "stream", "source", "cost"]
)
def test_runner_provenance_seam_counterfeit_rejected_no_writes(tmp_path, monkeypatch, tamper):
    clock, archive, execution, allocation, request, start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    sweep._finish_attempt(
        archive,
        execution,
        request["start"],
        start,
        {"returncode": 23, "timed_out": False, "resources": {"user_seconds": 0.25}},
        clock=clock,
    )
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    path = archive / "process-provenance-0.json"
    body = json.loads(path.read_bytes())
    if tamper == "duplicate":
        (archive / "process-provenance-0-duplicate.json").write_bytes(path.read_bytes())
    elif tamper == "source":
        (archive / "process-result-0.json").write_bytes(
            sweep._world_bytes({"returncode": 0, "timed_out": False})
        )
    elif tamper == "cost":
        cost_path = archive / "cost-0.json"
        cost = json.loads(cost_path.read_bytes())
        cost["terminal"]["sha256"] = "0" * 64
        cost_path.write_bytes(sweep._world_bytes(cost))
    else:
        if tamper == "terminal":
            body["terminal"]["sha256"] = "0" * 64
        elif tamper == "exit":
            body["process"]["returncode"] = 0
        elif tamper == "resource":
            body["process"]["resources"]["user_seconds"] = 0
        else:
            body["process"]["streams"] = {
                "stdout": {"locator": "child-1.log", "sha256": "0" * 64, "bytes": 0}
            }
        path.write_bytes(sweep._world_bytes(body))
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


@pytest.mark.parametrize("phase", ["science", "calibration"])
@pytest.mark.parametrize("tamper", ["exit", "duplicate", "alias"])
def test_runner_provenance_seam_late_counterfeit_preflight(tmp_path, monkeypatch, phase, tamper):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    attempts = execution["input"]["contract"]["seed_manifest"][phase][:2]
    records = []
    for attempt in attempts:
        ref, start = sweep._attempt_start(
            archive, execution, allocation, attempt, phase, clock=clock
        )
        clock.advance(1)
        end_ref, cost = sweep._finish_attempt(
            archive,
            execution,
            ref,
            start,
            {"returncode": 23, "timed_out": False, "resources": {"user_seconds": 0.5}},
            clock=clock,
        )
        records.append((ref, start, end_ref, cost["terminal_body"]))
    receipt = sweep._existing_terminal_provenance(archive, execution, *records[1], write=True)
    sidecar = archive / f"process-provenance-{attempts[1]['attempt']}.json"
    if tamper == "exit":
        receipt["process"]["returncode"] = 0
        sidecar.write_bytes(sweep._world_bytes(receipt))
    elif tamper == "duplicate":
        (archive / f"process-provenance-{attempts[1]['attempt']}-extra.json").write_bytes(
            sidecar.read_bytes()
        )
    else:
        receipt["tag"] = str(attempts[1]["attempt"] + 1)
        (archive / f"process-provenance-{receipt['tag']}.json").write_bytes(
            sweep._world_bytes(receipt)
        )
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    assert not (archive / f"process-provenance-{attempts[0]['attempt']}.json").exists()
    with pytest.raises(ValueError):
        if phase == "science":
            sweep._recover_terminals(archive, execution, allocation, clock=clock)
        else:
            sweep._reconcile_calibration(archive, execution, clock=clock)
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


def test_runner_stream_seam_receipt_only_finalization_retry(tmp_path, monkeypatch):
    clock = _EngineeringClock()
    archive, _ = _engineering_execution(tmp_path, monkeypatch, clock)
    raw = sweep._world_bytes(
        {"tag": "finalize", "stdout": "child-finalize.log", "stderr": "child-finalize-stderr.log"}
    )
    sweep._execution_file(archive, "stream-layout-finalize.json", raw)
    assert sweep._finalization_tag(archive, clock=clock) == "finalize-1"
    assert sweep._execution_file(archive, "stream-layout-finalize.json") == raw
    result = sweep._execution_read(archive, "process-result-finalize.json")
    assert result["returncode"] is None and result["exit_unknown_reason"]
    assert result["streams"] == {"layout": "separate", "stdout": None, "stderr": None}


def test_runner_provenance_seam_new_missing_terminal_idempotent(tmp_path, monkeypatch):
    clock, archive, execution, allocation, _, _ = _engineering_science_start(tmp_path, monkeypatch)
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    clock.advance(1)
    sweep._recover_terminals(archive, execution, allocation, clock=clock)
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    assert (archive / "process-provenance-0.json").is_file()


@pytest.mark.parametrize("expired", [False, True])
def test_runner_provenance_seam_finalized_existing_terminal(tmp_path, monkeypatch, expired):
    clock = _EngineeringClock()
    original = sweep._execution_sources
    calls = []

    def sources():
        if not calls:
            clock.advance(5095)
        calls.append(clock())
        return original()

    monkeypatch.setattr(sweep, "_execution_sources", sources)
    result = sweep.execute_science(
        config(), tmp_path, t0=clock(), clock=clock, launch=_engineering_launch(clock, [])
    )
    archive = Path(result["archive"])
    execution = sweep._execution_read(archive, "execution.json")
    sweep._reconcile_calibration(archive, execution, clock=clock)
    assert result["allocation_count"] == 1 and result["finalization"]
    # Fixture: legacy checkpoint finalized without process reconciliation.
    for name in ("process-provenance-0.json", "cost-0.json", "process-result-0.json"):
        (archive / name).unlink()
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    if expired:
        clock.value = execution["final_deadline"] + 10
    sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("no child"))
    after = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    assert set(after) - set(before) == {"process-provenance-0.json"}
    assert all(after[name] == raw for name, raw in before.items())
    receipt = sweep._execution_read(archive, "process-provenance-0.json")
    assert receipt["process"]["returncode"] is receipt["process"]["resources"] is None
    assert (
        receipt["process"]["exit_unknown_reason"] and receipt["process"]["resource_unknown_reason"]
    )
    sweep.recover_execution(archive, clock=clock, launch=lambda *a, **k: pytest.fail("no child"))
    assert after == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    receipt["process"]["returncode"] = 0
    (archive / "process-provenance-0.json").write_bytes(sweep._world_bytes(receipt))
    counterfeit = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        sweep.recover_execution(
            archive, clock=clock, launch=lambda *a, **k: pytest.fail("no child")
        )
    assert counterfeit == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}


@pytest.mark.parametrize("phase", ["science", "calibration"])
@pytest.mark.parametrize("started", [False, True])
def test_runner_provenance_preflight_zero_terminals_unchanged(
    tmp_path, monkeypatch, phase, started
):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    attempt = execution["input"]["contract"]["seed_manifest"][phase][0]
    if started:
        sweep._attempt_start(archive, execution, allocation, attempt, phase, clock=clock)
    # Exact reviewer reproduction: no terminal exists to trigger the old conditional preflight.
    sweep._execution_record(
        archive,
        f"process-provenance-{attempt['attempt']}-extra.json",
        {"tag": str(attempt["attempt"])},
    )
    before = {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError):
        if phase == "science":
            sweep._recover_terminals(archive, execution, allocation, clock=clock)
        else:
            sweep._reconcile_calibration(archive, execution, clock=clock)
    assert before == {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("phase", ["science", "calibration"])
@pytest.mark.parametrize("tamper", ["duplicate", "orphan", "digest", "phase", "cost", "stream"])
def test_runner_provenance_preflight_mixed_terminal_inventory(tmp_path, monkeypatch, phase, tamper):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    allocation = sweep.allocate_from_calibration(config(), 3, _completed_calibration((1, 1, 1)))
    attempts = execution["input"]["contract"]["seed_manifest"][phase][:2]
    # Attempt 0 needs recovery; a later completed attempt carries the bad receipt.
    sweep._attempt_start(archive, execution, allocation, attempts[0], phase, clock=clock)
    ref, start = sweep._attempt_start(
        archive, execution, allocation, attempts[1], phase, clock=clock
    )
    tag = str(attempts[1]["attempt"])
    end_ref, cost = sweep._finish_attempt(
        archive,
        execution,
        ref,
        start,
        {"returncode": 23, "timed_out": False, "resources": {"user_seconds": 0.5}},
        clock=clock,
    )
    receipt = sweep._existing_terminal_provenance(
        archive, execution, ref, start, end_ref, cost["terminal_body"], write=True
    )
    path = archive / f"process-provenance-{tag}.json"
    if tamper == "duplicate":
        (archive / f"process-provenance-{tag}-extra.json").write_bytes(path.read_bytes())
    elif tamper == "orphan":
        (archive / ref["locator"]).unlink()
    elif tamper == "digest":
        receipt["terminal"]["sha256"] = "0" * 64
        path.write_bytes(sweep._world_bytes(receipt))
    elif tamper == "phase":
        # Bind a real, re-sealed wrong-phase source, not merely a broken digest.
        wrong = {**start, "phase": "calibration" if phase == "science" else "science"}
        (archive / ref["locator"]).write_bytes(sweep._world_bytes(wrong))
        receipt["start"]["sha256"] = sweep._sha256(sweep._world_bytes(wrong))
        path.write_bytes(sweep._world_bytes(receipt))
    else:
        source_name = f"cost-{tag}.json" if tamper == "cost" else f"process-result-{tag}.json"
        source = sweep._execution_read(archive, source_name)
        if tamper == "cost":
            source["terminal"]["locator"] = ref["locator"]
        else:
            source["streams"] = {
                "stdout": {"locator": "child-999.log", "sha256": "0" * 64, "bytes": 0}
            }
            receipt["process"] = source
        (archive / source_name).write_bytes(sweep._world_bytes(source))
        receipt["cost" if tamper == "cost" else "process_result"]["sha256"] = sweep._sha256(
            sweep._world_bytes(source)
        )
        path.write_bytes(sweep._world_bytes(receipt))
    before = {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(sweep, "_recover_process", lambda *a, **k: pytest.fail("no kill/reap"))
    monkeypatch.setattr(sweep, "_finish_attempt", lambda *a, **k: pytest.fail("no archive writes"))
    with pytest.raises((ValueError, FileNotFoundError)):
        if phase == "science":
            sweep._recover_terminals(archive, execution, allocation, clock=clock)
        else:
            sweep._reconcile_calibration(archive, execution, clock=clock)
    assert before == {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "branch", ["missing", "retained_calibration", "allocated", "finalized", "expired"]
)
@pytest.mark.parametrize("sidecar_phase", ["science", "calibration"])
def test_runner_provenance_preflight_all_recovery_branches(
    tmp_path, monkeypatch, branch, sidecar_phase
):
    clock = _EngineeringClock()
    archive, execution = _engineering_execution(tmp_path, monkeypatch, clock)
    attempt = execution["input"]["contract"]["seed_manifest"]["calibration"][0]
    sweep._attempt_start(archive, execution, None, attempt, "calibration", clock=clock)
    frozen = {"calibration": [], "allocation": sweep.allocate_from_calibration(config(), 1, [])}
    if branch == "retained_calibration":
        sweep._execution_record(archive, "calibration-recovery.json", {**frozen, "records": []})
    elif branch in ("allocated", "finalized", "expired"):
        sweep._execution_record(archive, "allocation.json", frozen)
    if branch == "expired":
        clock.value = execution["final_deadline"] + 1
    if branch == "finalized":
        sweep._execution_record(archive, "finalization.json", {"engineering_fixture": True})
    tag = execution["input"]["contract"]["seed_manifest"][sidecar_phase][0]["attempt"]
    sweep._execution_record(archive, f"process-provenance-{tag}-extra.json", {"tag": str(tag)})
    before = {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(sweep, "_recover_process", lambda *a, **k: pytest.fail("no kill/reap"))
    monkeypatch.setattr(sweep, "_finish_attempt", lambda *a, **k: pytest.fail("no archive writes"))
    with pytest.raises(ValueError):
        sweep.recover_execution(
            archive, clock=clock, launch=lambda *a, **k: pytest.fail("no launch")
        )
    assert before == {
        str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
    }


def test_runner_provenance_preflight_1000_linear(tmp_path, monkeypatch):
    """Frozen engineering records only; no generation, calibration or scientific fitting."""
    import os
    import time
    from collections import Counter

    clock, archive, execution, allocation, request, first_start = _engineering_science_start(
        tmp_path, monkeypatch
    )
    n = allocation["frozen_attempt_count"]
    assert n == 1000
    # Publish canonical controlled terminals directly, avoiding science and process launches.
    for attempt in allocation["manifest"]:
        tag = str(attempt["attempt"])
        start = {**first_start, "attempt": attempt}
        raw = sweep._world_bytes(start)
        sha = sweep._sha256(raw)
        (archive / f"start-{sha}.json").write_bytes(raw)
        terminal = {
            "attempt": attempt,
            "phase": "science",
            "start_sha256": sha,
            "finished_elapsed": 1,
            "status": "recovered_unknown",
            "generation_outcome": "unknown",
            "errors": [],
            "world_reference": None,
            "catalog_reference": None,
            "graph_reference": None,
            "partial_references": [],
        }
        raw = sweep._world_bytes(terminal)
        (archive / f"terminal-{sweep._sha256(raw)}.json").write_bytes(raw)
        if attempt["attempt"] % 2 == 0:
            (archive / f"process-result-{tag}.json").write_bytes(
                sweep._world_bytes(
                    {"returncode": 23, "timed_out": False, "resources": {"user_seconds": 0.25}}
                )
            )
    sweep._execution_record(archive, "finalization.json", {"engineering_fixture": True})
    monkeypatch.setattr(sweep, "verify_execution", lambda *a, **k: {"engineering_only": True})
    monkeypatch.setattr(
        sweep, "_recover_process", lambda *a, **k: pytest.fail("no process recovery")
    )
    monkeypatch.setattr(sweep, "_finish_attempt", lambda *a, **k: pytest.fail("no science"))
    file_original, preflight_original, listdir_original = (
        sweep._execution_file,
        sweep._existing_terminal_provenance,
        os.listdir,
    )
    counts = Counter()
    archive_stat = archive.stat()

    def measured_file(root, name, payload=None):
        if payload is None:
            counts["reads"] += 1
            if name.startswith("process-provenance-"):
                counts["sidecar_reads"] += 1
        return file_original(root, name, payload)

    def measured_preflight(*args, **kwargs):
        counts["preflights"] += 1
        return preflight_original(*args, **kwargs)

    def measured_listdir(path):
        if isinstance(path, int) and os.fstat(path).st_ino == archive_stat.st_ino:
            counts["archive_scans"] += 1
        return listdir_original(path)

    monkeypatch.setattr(sweep, "_execution_file", measured_file)
    monkeypatch.setattr(sweep, "_existing_terminal_provenance", measured_preflight)
    monkeypatch.setattr(os, "listdir", measured_listdir)
    for repeat in (False, True):
        before = {
            str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
        }
        counts.clear()
        began = time.monotonic()
        assert sweep.recover_execution(
            archive, clock=clock, launch=lambda *a, **k: pytest.fail("no launch")
        ) == {"engineering_only": True}
        elapsed = time.monotonic() - began
        measured = dict(counts)
        # Exactly one provenance inventory, plus calibration/science/report inventories.
        assert measured["preflights"] == 1
        assert measured["archive_scans"] == 4
        assert measured["sidecar_reads"] == n  # write read-back first; retained read second
        # At most three per-record validations, each <=10 reads, plus references/metadata.
        assert measured["reads"] <= 32 * n + 20
        assert elapsed < 300  # This controlled local probe only, not a runtime guarantee.
        print(
            f"provenance 1000 repeat={repeat} counts={measured} elapsed={elapsed:.6f}s reserve=300s"
        )
        after = {
            str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob("*") if p.is_file()
        }
        assert all(after[name] == raw for name, raw in before.items())
        assert set(after) - set(before) == (
            set() if repeat else {f"process-provenance-{i}.json" for i in range(n)}
        )
    authentic = sweep._execution_read(archive, "process-provenance-0.json")["process"]
    unknown = sweep._execution_read(archive, "process-provenance-1.json")["process"]
    assert authentic["returncode"] == 23 and authentic["resources"]["user_seconds"] == 0.25
    assert unknown["returncode"] is unknown["resources"] is None
    assert unknown["exit_unknown_reason"] and unknown["resource_unknown_reason"]


@pytest.mark.parametrize("behavior", ["normal", "term", "kill", "leader_first"])
def test_owned_cleanup_real_proc_disappearance_and_unrelated(tmp_path, monkeypatch, behavior):
    import ctypes
    import signal
    import subprocess
    import time

    state = ctypes.c_int()
    libc = ctypes.CDLL(None)
    assert libc.prctl(37, ctypes.byref(state), 0, 0, 0) == 0
    previous = state.value
    archive = tmp_path / "owned"
    archive.mkdir()
    real = subprocess.Popen
    unrelated = real([sweep._EXECUTION_PYTHON, "-B", "-c", "import time; time.sleep(30)"])
    unrelated_identity = sweep._owned_stat(unrelated.pid)
    code = """
import os, signal, subprocess, sys, time
behavior = sys.argv[1]
if behavior == 'kill': signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
print(child.pid, flush=True)
if behavior == 'normal':
    child.terminate(); child.wait()
if behavior not in ('normal', 'leader_first'): time.sleep(30)
"""

    def popen(command, **kwargs):
        return real([sweep._EXECUTION_PYTHON, "-B", "-c", code, behavior], **kwargs)

    try:
        monkeypatch.setattr(subprocess, "Popen", popen)
        deadline = time.monotonic() + 0.4
        result = sweep._timed_execution_child(archive, {"tag": "owned"}, deadline)
        descendant = int((archive / "child-owned.log").read_text())
        assert (
            result["returncode"]
            == {"normal": 0, "leader_first": 0, "term": -signal.SIGTERM, "kill": -signal.SIGKILL}[
                behavior
            ]
        )
        assert result["cleanup"]["all_disappeared"] is True
        assert not Path(f"/proc/{result['pid']}").exists()
        assert not Path(f"/proc/{descendant}").exists()
        assert result["finished"] <= deadline + 2.2
        assert result["cleanup"]["deadline_overrun_seconds"] >= 0
        assert unrelated.poll() is None
        assert sweep._owned_stat(unrelated.pid) == unrelated_identity
        assert libc.prctl(37, ctypes.byref(state), 0, 0, 0) == 0
        assert state.value == previous
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_owned_cleanup_recovery_known_escaped_child(tmp_path):
    import subprocess
    import time

    archive = tmp_path / "recover"
    archive.mkdir()
    with sweep._owned_subreaper():
        child = subprocess.Popen(
            [
                sweep._EXECUTION_PYTHON,
                "-B",
                "-c",
                "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); time.sleep(30)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        descendant = int(child.stdout.readline())
        sweep._execution_record(
            archive,
            "process-x.json",
            {
                "pid": child.pid,
                "identity": sweep._process_identity(child.pid),
                "started": time.monotonic(),
            },
        )
        result = sweep._recover_process(archive, "x")
        assert result["returncode"] == -15
        assert result["exit_unknown_reason"] is None
        assert not Path(f"/proc/{child.pid}").exists()
        assert not Path(f"/proc/{descendant}").exists()
        child.returncode = result["returncode"]
        child.stdout.close()


def test_owned_cleanup_rejects_reused_identity_without_signaling(tmp_path, monkeypatch):
    import subprocess

    archive = tmp_path / "reuse"
    archive.mkdir()
    child = subprocess.Popen([sweep._EXECUTION_PYTHON, "-B", "-c", "import time; time.sleep(30)"])
    try:
        item = sweep._owned_stat(child.pid)
        sweep._execution_record(
            archive,
            "process-x.json",
            {"pid": child.pid, "identity": "wrong-start-identity", "started": 0},
        )
        with pytest.raises(RuntimeError, match="identity reused"):
            sweep._recover_process(archive, "x")
        assert child.poll() is None
        with pytest.raises(RuntimeError, match="identity changed"):
            sweep._owned_signal({**item, "identity": "wrong"}, 9)
        with pytest.raises(RuntimeError, match="PID/PGID identity reused"):
            sweep._owned_collect({child.pid: item}, {**item, "pgid": item["pgid"] + 1})
        assert child.poll() is None
        assert not (archive / "process-result-x.json").exists()
    finally:
        child.terminate()
        child.wait(timeout=3)


def test_owned_cleanup_legacy_gone_leader_rejects_without_fake_wait(tmp_path):
    import subprocess

    archive = tmp_path / "legacy"
    archive.mkdir()
    # A local subreaper owns this fixture, NOT the hypothetical legacy recoverer.
    with sweep._owned_subreaper():
        child = subprocess.Popen(
            [
                sweep._EXECUTION_PYTHON,
                "-B",
                "-c",
                "import subprocess,sys; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)']); print(p.pid,flush=True)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        identity = sweep._process_identity(child.pid)
        descendant = int(child.stdout.readline())
        child.wait(timeout=3)
        item = sweep._owned_stat(descendant)
        try:
            sweep._execution_record(
                archive, "process-x.json", {"pid": child.pid, "identity": identity, "started": 0}
            )
            with pytest.raises(RuntimeError, match="legacy leader gone"):
                sweep._recover_process(archive, "x")
            assert sweep._owned_stat(descendant) == item
            assert not (archive / "process-result-x.json").exists()
        finally:
            sweep._owned_signal(item, 9)
            sweep.os.waitpid(descendant, 0)
            child.stdout.close()


def test_owned_cleanup_unsupported_rejects_before_spawn(tmp_path, monkeypatch):
    import subprocess
    import sys

    monkeypatch.setattr(sys, "platform", "unsupported")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unsupported launch"))
    with pytest.raises(RuntimeError, match="requires Linux"):
        sweep._timed_execution_child(tmp_path, {"tag": "x"}, sweep.time.monotonic() + 3)
    assert not list(tmp_path.glob("child-*.log"))


def test_owned_cleanup_old_failed_zombie_case_preserved(tmp_path):
    """Measured old implementation: killpg + leader.wait leaves a zombie, not cleanup."""
    import os
    import signal
    import subprocess
    import time

    with sweep._owned_subreaper():
        leader = subprocess.Popen(
            [
                sweep._EXECUTION_PYTHON,
                "-B",
                "-c",
                "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        pid = int(leader.stdout.readline())
        os.killpg(leader.pid, signal.SIGKILL)  # Retained baseline diagnostic, not new cleanup.
        leader.wait(timeout=3)
        try:
            until = time.monotonic() + 2
            while time.monotonic() < until:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "Z":
                    break
                time.sleep(0.01)
            assert state == "Z"
            assert Path(f"/proc/{pid}").exists()  # Exactly the old unacceptable condition.
        finally:
            os.waitpid(pid, 0)
            leader.stdout.close()
        assert not Path(f"/proc/{pid}").exists()


def test_owned_cleanup_grandchild_term_kill_strict(tmp_path, monkeypatch):
    import subprocess
    import time

    archive = tmp_path / "grandchild"
    archive.mkdir()
    real = subprocess.Popen
    leaf = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
    middle = f"import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-B','-c',{leaf!r}],stdout=subprocess.PIPE,text=True); assert p.stdout.readline().strip()=='ready'; print(p.pid,flush=True); time.sleep(30)"
    code = f"import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-B','-c',{middle!r}],stdout=subprocess.PIPE,text=True); print(p.pid,p.stdout.readline().strip(),flush=True); time.sleep(30)"
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, **kw: real([sweep._EXECUTION_PYTHON, "-B", "-c", code], **kw),
    )
    result = sweep._timed_execution_child(archive, {"tag": "tree"}, time.monotonic() + 0.5)
    pids = [result["pid"], *map(int, (archive / "child-tree.log").read_text().split())]
    assert result["returncode"] == -9
    assert len(result["cleanup"]["owned"]) == 3
    assert len(result["cleanup"]["wait_statuses"]) == 3
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids)


def test_owned_cleanup_nonchild_zombie_is_bounded_failure(tmp_path):
    import subprocess
    import time

    # Another live process remains the zombie's parent; subreaper cannot adopt it.
    parent = subprocess.Popen(
        [
            sweep._EXECUTION_PYTHON,
            "-B",
            "-c",
            "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','pass'],start_new_session=True); print(p.pid,flush=True); time.sleep(4); p.wait()",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    pid = int(parent.stdout.readline())
    try:
        item = sweep._owned_stat(pid)
        began = time.monotonic()
        with sweep._owned_subreaper(), pytest.raises(RuntimeError, match="leaves remain"):
            sweep._kill_group(None, leader=item)
        assert 1.9 <= time.monotonic() - began < 3
        assert Path(f"/proc/{pid}").exists()
        parent.wait(timeout=5)
        assert not Path(f"/proc/{pid}").exists()
    finally:
        parent.wait(timeout=5)
        parent.stdout.close()


def test_owned_cleanup_fork_during_term_after_snapshot(tmp_path, monkeypatch):
    """TERM itself creates the leaf; waitid orders exit after the initial snapshot."""
    import os
    import signal
    import subprocess

    code = """
import os, signal, sys
r, w = os.pipe()
def term(sig, frame):
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(w, b'R')
        while True: signal.pause()
    assert os.read(r, 1) == b'R'
    print(pid, flush=True)
    os._exit(0)
signal.signal(signal.SIGTERM, term)
print('ready', flush=True)
while True: signal.pause()
"""
    with sweep._owned_subreaper():
        leader = subprocess.Popen(
            [sweep._EXECUTION_PYTHON, "-B", "-c", code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        leaf = None
        original = sweep._owned_signal
        assert leader.stdout.readline().strip() == "ready"

        def ordered_signal(item, sig):
            nonlocal leaf
            original(item, sig)
            if item["pid"] == leader.pid and sig == signal.SIGTERM:
                leaf = int(leader.stdout.readline())
                os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOWAIT)

        monkeypatch.setattr(sweep, "_owned_signal", ordered_signal)
        try:
            result = sweep._kill_group(leader, adopted=True)
            assert result["all_disappeared"] is True
            assert {item["pid"] for item in result["owned"]} == {leader.pid, leaf}
            assert set(result["wait_statuses"]) == {leader.pid, leaf}
            assert not Path(f"/proc/{leader.pid}").exists()
            assert not Path(f"/proc/{leaf}").exists()
        finally:
            if leader.poll() is None:
                leader.kill()
                leader.wait(timeout=3)
            if leaf is not None and Path(f"/proc/{leaf}").exists():
                original(sweep._owned_stat(leaf), signal.SIGKILL)
                os.waitpid(leaf, 0)
            leader.stdout.close()


def test_owned_cleanup_discovery_parent_proof_and_trusted_reparent(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess

    with sweep._owned_subreaper():
        unrelated = subprocess.Popen(
            [sweep._EXECUTION_PYTHON, "-B", "-c", "import signal; signal.pause()"]
        )
        leader = subprocess.Popen(
            [
                sweep._EXECUTION_PYTHON,
                "-B",
                "-c",
                "import subprocess,sys,signal; p=subprocess.Popen([sys.executable,'-B','-c','import signal; print(\"ready\",flush=True); signal.pause()'],stdout=subprocess.PIPE,text=True); assert p.stdout.readline().strip()=='ready'; print(p.pid,flush=True); signal.pause()",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        leaf = int(leader.stdout.readline())
        leader_item = sweep._owned_stat(leader.pid)
        unrelated_item = sweep._owned_stat(unrelated.pid)
        owned = {leader.pid: leader_item}
        children = sweep._owned_children

        def stale_children(pid):
            # Deterministic reuse interleaving: a child-list PID now denotes an
            # unrelated live process with a new parent/start identity.
            return [unrelated.pid, leaf] if pid == leader.pid else children(pid)

        monkeypatch.setattr(sweep, "_owned_children", stale_children)
        try:
            assert sweep._owned_stat(unrelated.pid, parent=leader.pid) is None
            sweep._owned_collect(owned, leader_item)
            assert set(owned) == {leader.pid, leaf}
            trusted = owned[leaf]
            leader.terminate()
            leader.wait(timeout=3)
            assert sweep._owned_stat(leaf, parent=leader.pid) is None
            assert sweep._owned_stat(leaf, parent=os.getpid()) == trusted
            sweep._owned_collect(owned, leader_item, adopted=True)
            assert owned[leaf] == trusted
            sweep._owned_signal(trusted, signal.SIGTERM)
            waited, status = os.waitpid(leaf, 0)
            assert waited == leaf and os.waitstatus_to_exitcode(status) == -signal.SIGTERM
            assert not Path(f"/proc/{leaf}").exists()
            assert unrelated.poll() is None
            assert sweep._owned_stat(unrelated.pid) == unrelated_item
        finally:
            if leader.poll() is None:
                leader.kill()
                leader.wait(timeout=3)
            if Path(f"/proc/{leaf}").exists():
                sweep._owned_signal(sweep._owned_stat(leaf), signal.SIGKILL)
                os.waitpid(leaf, 0)
            unrelated.terminate()
            unrelated.wait(timeout=3)
            leader.stdout.close()


@pytest.mark.parametrize("fork_leaf", [False, True])
def test_owned_cleanup_fresh_nonchild_recovery_readonly(tmp_path, fork_leaf):
    import os
    import signal
    import subprocess
    import time

    archive = tmp_path / "fresh-recovery"
    archive.mkdir()
    leader_code = """
import os, signal, sys
r, w = os.pipe()
def term(sig, frame):
    pid = 0
    if sys.argv[1] == 'True':
        pid = os.fork()
        if pid == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.write(w, b'R')
            while True: signal.pause()
        assert os.read(r, 1) == b'R'
    print(pid, flush=True)
    os._exit(0)
signal.signal(signal.SIGTERM, term)
print('ready', flush=True)
while True: signal.pause()
"""
    reader_code = """
import json, os, signal, sys
from pathlib import Path
import scripts.linear_recovery_true_feature_sweep as s
signals, waits, adoption = [], [], []
original_signal, original_wait, original_collect = s._owned_signal, os.waitpid, s._owned_collect
def send(item, sig):
    signals.append(item['pid'])
    original_signal(item, sig)
    if sig == signal.SIGTERM: assert os.read(int(sys.argv[2]), 1) == b'R'
def wait(pid, flags):
    waits.append(pid)
    return original_wait(pid, flags)
def collect(owned, leader, *, adopted=False):
    adoption.append(adopted)
    return original_collect(owned, leader, adopted=adopted)
s._owned_signal, os.waitpid, s._owned_collect = send, wait, collect
try:
    result = s._recover_process(Path(sys.argv[1]), 'fresh')
    output = {'result': result}
except RuntimeError as exc:
    output = {'blocked': str(exc)}
print(json.dumps(dict(output, signals=signals, waits=waits, adoption=adoption)), flush=True)
"""
    with sweep._owned_subreaper():
        leader = subprocess.Popen(
            [sweep._EXECUTION_PYTHON, "-B", "-c", leader_code, str(fork_leaf)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert leader.stdout.readline().strip() == "ready"
        sweep._execution_record(
            archive,
            "process-fresh.json",
            {
                "pid": leader.pid,
                "identity": sweep._process_identity(leader.pid),
                "started": time.monotonic(),
            },
        )
        r, w = os.pipe()
        reader = subprocess.Popen(
            [sweep._EXECUTION_PYTHON, "-B", "-c", reader_code, str(archive), str(r)],
            pass_fds=(r,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(r)
        leaf = None
        try:
            leaf = int(leader.stdout.readline())
            leader.wait(timeout=3)
            os.write(w, b"R")  # Reader observes disappearance only after the true parent reaps.
            stdout, stderr = reader.communicate(timeout=10)
            assert reader.returncode == 0, stderr
            evidence = json.loads(stdout)
            assert evidence["signals"] == [leader.pid]
            assert set(evidence["waits"]) <= {leader.pid}
            assert evidence["adoption"] and not any(evidence["adoption"])
            assert not Path(f"/proc/{leader.pid}").exists()
            if fork_leaf:
                assert "unresolved owned cleanup" in evidence["blocked"]
                assert Path(f"/proc/{leaf}").exists()
                assert not (archive / "process-result-fresh.json").exists()
            else:
                assert evidence["result"]["returncode"] is None
                assert evidence["result"]["exit_unknown_reason"]
                assert not any(
                    item and (item["pgid"] == leader.pid or item["sid"] == leader.pid)
                    for entry in Path("/proc").iterdir()
                    if entry.name.isdecimal()
                    for item in [sweep._owned_stat(int(entry.name))]
                )
        finally:
            os.close(w)
            if reader.poll() is None:
                reader.kill()
                reader.wait(timeout=3)
            if leader.poll() is None:
                leader.kill()
                leader.wait(timeout=3)
            if leaf and Path(f"/proc/{leaf}").exists():
                sweep._owned_signal(sweep._owned_stat(leaf), signal.SIGKILL)
                os.waitpid(leaf, 0)
            leader.stdout.close()
        assert not Path(f"/proc/{leader.pid}").exists()
        assert not Path(f"/proc/{reader.pid}").exists()
        assert not leaf or not Path(f"/proc/{leaf}").exists()


# Synthetic ENGINEERING reader fixtures only: no sampling, reconstruction, or fits.
@pytest.fixture
def notebook_no_science(monkeypatch):
    import pymc_generator

    def forbidden(*args, **kwargs):
        pytest.fail("reader/fixture attempted science, recovery, generation or fitting")

    for name in (
        "execute_science",
        "recover_execution",
        "generate_world",
        "generate_configured_world",
        "analyze_world",
        "analyze_configured_world",
        "fit_design",
        "build_compact_summary",
    ):
        monkeypatch.setattr(sweep, name, forbidden)
    monkeypatch.setattr(pymc_generator, "make_scm_prior", forbidden)
    monkeypatch.setattr(pymc_generator, "sample_scm", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)


def _notebook_synthetic_summary(tmp_path, *, case="empty", ncap=12):
    """Manufactured codec evidence; integrity acceptance is NOT science acceptance."""
    archive, args = _summary_setup(tmp_path, ncap=ncap)
    if case == "failure":
        _summary_start(archive, args, outcome="failed", errors=[_stage("generation")])
    elif case == "missing":
        _summary_start(archive, args, outcome="completed", errors=[_stage("persistence")])
        _summary_start(archive, args, errors=[_stage("timeout")])
    elif case in ("eligible", "ineligible", "rank_deficient"):
        attempt = args["allocation"]["manifest"][0]
        graph = _empty_graph(attempt)
        graph["g_cy"][0] = graph["g_zy"][0] = 1
        world_ref = sweep.persist_world(
            archive, attempt, {"kind": "SYNTHETIC ENGINEERING — NOT SCIENCE", "graph": graph}
        )
        _summary_start(archive, args, outcome="completed", world=world_ref)
    evidence = {
        "input": args["input_contract"],
        "calibration": args["calibration"],
        "allocation": args["allocation"],
        "timing": args["timing"],
        "ledger": [],
    }
    for item in args["ledger"]:
        evidence["ledger"].append(
            {
                "start_reference": item["start"],
                "start": sweep._report_file(archive, item["start"]),
                "terminal_reference": item["terminal"],
                "terminal": sweep._report_file(archive, item["terminal"]),
                "graph": None,
                "graph_capsule": None,
                "graph_world_node": None,
                "analysis": None,
                "catalog": None,
            }
        )
    if case in ("eligible", "ineligible", "rank_deficient"):
        capsule = evidence["ledger"][0]
        capsule["graph"] = graph
        capsule["graph_world_node"] = sweep._normalize_world(
            {k: np.array(v, dtype=np.int64) for k, v in graph.items()}, {}
        )
        eligible = case != "ineligible"
        full_rank = case == "eligible"
        status = {
            "scheduling": "attempted",
            "generation": "generated",
            "truth_eligibility": "eligible" if eligible else "ineligible",
            "rank": ("full_rank" if full_rank else "rank_deficient") if eligible else None,
            "conditioning": "easy" if eligible else None,
            "noisefree_recovery": full_rank if eligible else None,
            "paired_noisy": {"available": eligible},
            "raw_observed_misspecified": False if eligible else None,
            "reason": None if eligible else "identity_or_truth_ineligible",
        }
        capsule["analysis"] = {
            "status": status,
            "reasons": [] if eligible else ["identity_or_truth_ineligible:SYNTHETIC ENGINEERING"],
            "details": None,
        }
        if eligible:
            status["paired_noisy"]["max_abs_error"] = 0.0
            t, m = attempt["n_treatments"], attempt["n_covariates"]
            params = {}
            for keys, shape in (
                (
                    (
                        "beta",
                        "carryover_alpha",
                        "weibull_lam",
                        "weibull_k",
                        "hill_slope",
                        "hill_kappa_mult",
                        "logistic_lam",
                        "mm_kappa_mult",
                        "tanh_c",
                        "root_alpha",
                        "hf_sigma",
                        "pulse_amp",
                        "pulse_prob",
                        "treatment_level",
                    ),
                    (t,),
                ),
                (
                    ("rho_zy", "covariate_hf_sigma", "covariate_pulse_amp", "covariate_pulse_prob"),
                    (m,),
                ),
                (("delta_dy",), (1,)),
                (("w_dc",), (1, t)),
                (("u_dz",), (1, m)),
                (("v_zc",), (m, t)),
                (("alpha_cc",), (t, t)),
                (("gamma_zz",), (m, m)),
            ):
                params.update({key: np.zeros(shape).tolist() for key in keys})
            params.update(
                carryover_family=[0] * t,
                sat_family=[0] * t,
                l_max=8,
                baseline_floor=None,
                baseline_floor_scope="intercept",
                confounding_strength=0.0,
            )
            for key, n in (
                ("use_hf", t),
                ("use_pulse", t),
                ("use_covariate_hf", m),
                ("use_covariate_pulse", m),
            ):
                params[key] = [False] * n
            for suffix, n in (("d", 1), ("z", m), ("c", t), ("b", 1), ("y", 1)):
                params["rw_" + suffix] = {
                    "mean": [0.0] * n,
                    "std": [0.0] * n,
                    "positive_only": suffix == "c",
                }
                if suffix != "y":
                    params["rw_" + suffix].update(smoothness=[0.0] * n, rw_smoothness_max_weeks=26)
            params["trajectory"] = {
                role: {
                    "use": {
                        key: [False] * n
                        for key in (
                            "hf",
                            "pulse",
                            "onset",
                            "offset",
                            "flighting",
                            "level_jump",
                            "seasonal",
                            "trend",
                        )
                    }
                }
                for role, n in (("treatment", t), ("covariate", m))
            }
            design = {
                "rank": 3 if full_rank else 2,
                "singular_values": [1.0, 1.0, 1.0 if full_rank else 0.0],
                "condition": 1.0 if full_rank else None,
                "rank_tolerance": 1e-12,
                "rank_cutoff": 1e-12,
                "shape": [104, 3],
                "conditioning": "easy",
            }
            fit = {k: design[k] for k in ("rank", "singular_values", "condition")}
            fit.update(
                coefficients=[0.0] * 3,
                p=3,
                n_fit=104,
                coefficient_error=[0.0] * 3,
                max_abs_error=0.0,
                rmse=0.0,
                residual_max_abs=0.0,
                residual_rmse=0.0,
            )
            metrics = {"mean": 0.0, "std": 0.0, "max_abs": 0.0}
            details = {
                "truth": [0.0] * 3,
                "coefficient_order": ["beta[0]", "rho_zy[0]", "intercept"],
                "parameters": params,
                "n_full": 104,
                "warmup": 0,
                "n_fit": 104,
                "p": 3,
                "fit_indices": list(range(104)),
                "identities": {
                    key: {
                        "all_max_abs": 0.0,
                        "fit_max_abs": 0.0,
                        "tolerance": 2e-12,
                        "passed": True,
                    }
                    for key in (
                        "reconstructed_D",
                        "reconstructed_Z",
                        "reconstructed_C",
                        "reconstructed_C_base",
                        "scalar_baseline",
                        "observed_contribution",
                        "base_contribution",
                        "control_contribution",
                        "direct_latent",
                        "observed_base_indirect",
                        "indirect_total",
                        "baseline_additive",
                        "outcome_additive",
                        "target_truth",
                        "noise_pair",
                        "audited_noise",
                    )
                },
                "raw_design": deepcopy(design),
                "standardized_design": deepcopy(design),
                "target_metrics": {
                    key: {"all": metrics, "fit": metrics}
                    for key in ("y0", "y1", "u", "outcome_noise", "raw_observed_unadjusted")
                },
                "fits": {"y0": deepcopy(fit), "y1": deepcopy(fit)},
                "history": {
                    "n_full": 104,
                    "burn_in": 0,
                    "source": "catalog known_sampled_truth.public_history; full horizon before slicing once",
                },
            }
            capsule["analysis"]["details"] = details
            contract = evidence["input"]["contract"]
            synthetic_hash = hashlib.sha256(
                "SYNTHETIC ENGINEERING — NOT SCIENCE".encode()
            ).hexdigest()
            node_ref = {
                "world_locator": world_ref["world_locator"],
                "path": ["synthetic-engineering"],
                "sha256": synthetic_hash,
                "array": None,
            }
            known = {
                key: deepcopy(node_ref)
                for key in (
                    "coefficients",
                    "direct_structure",
                    "design",
                    "contributions_observed",
                    "y1",
                    "noise",
                    "scales",
                    "public_history",
                    "rows",
                    "row_mask",
                )
            }
            known.update(
                kind="finite_sampled_truth_not_estimates",
                prior_cond=None,
                prior_cond_status="unknown_not_retained",
            )
            known["direct_structure"].update(
                path=["result", "source", "graph"], sha256=sweep.digest(capsule["graph_world_node"])
            )
            inferred = {
                key: deepcopy(node_ref)
                for key in (
                    "y0_coefficients",
                    "y1_coefficients",
                    "status",
                    "raw_design",
                    "standardized_design",
                )
            }
            inferred.update(
                kind="estimates_not_sampled_truth",
                labels={
                    "rank": status["rank"],
                    "conditioning": status["conditioning"],
                    "noisefree_recovery": status["noisefree_recovery"],
                    "paired_noisy_available": True,
                },
            )
            capsule["catalog"] = {
                "schema_version": "ols-world-catalog/v1",
                "schema_path": contract["catalog_schema_path"],
                "schema_sha256": contract["catalog_schema_sha256"],
                "parent_id": synthetic_hash,
                "source_id": sweep.digest(
                    {"commit": contract["source_commit"], "hashes": contract["source_hashes"]}
                ),
                "source_commit": contract["source_commit"],
                "analysis_file_hashes": contract["source_hashes"],
                "config_digest": contract["config_digest"],
                "resolved_config_digest": synthetic_hash,
                "generation_seed": attempt["world_seed"],
                "world_id": world_ref["world_id"],
                "world_content_digest": world_ref["world_sha256"],
                "world_reference": world_ref,
                "selection_manifest_id": synthetic_hash,
                "selection_manifest_locator": f"selection-{synthetic_hash}.json",
                "selection": contract["maximum_selection"]["rows"][0],
                "parameter_draw_id": synthetic_hash,
                "data_draw_id": synthetic_hash,
                "noise_replicate_id": None,
                "replicate_kind": "fresh_world",
                "pair_targets": ["y0", "y1"],
                "coefficient_order": details["coefficient_order"],
                "n_full": 104,
                "warmup": 0,
                "known_sampled_truth": known,
                "inferred_diagnostics": inferred,
            }
    return sweep.validate_compact_summary(sweep._render_compact(evidence))


def _execute_reader_notebook(monkeypatch, notebook, *, directory=None, path=None, sha=None):
    """Execute actual cells sequentially in consuming Python, NOT a Jupyter kernel."""
    import IPython.display

    output = []
    monkeypatch.setattr(IPython.display, "display", lambda value: output.append(value.data))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"notebook/{cell['id']}", "exec"), namespace)
            if cell["id"] == "reader-01":
                namespace.update(
                    NOTEBOOK_DIRECTORY=directory, SUMMARY_PATH=path, EXPECTED_SUMMARY_SHA256=sha
                )
    return namespace, "\n".join(output)


@pytest.fixture
def reader_notebook():
    return json.loads(Path("docs/examples/linear-recovery-true-feature-sweep.ipynb").read_text())


def test_notebook_clean_json_readonly_cells_and_owner_policy(reader_notebook, notebook_no_science):
    import ast

    import nbformat

    nbformat.validate(nbformat.from_dict(reader_notebook))
    markdown = "\n".join(
        "".join(c["source"]) for c in reader_notebook["cells"] if c["cell_type"] == "markdown"
    )
    assert "all 104 rows 0–103" in markdown and "Wdiscard=0" in markdown and "Issue 42" in markdown
    assert "Startup transients are INCLUDED" in markdown
    assert "These are structural recovery diagnostics, not causal identification." in markdown
    assert "adapter preserves exact bytes and allows the frozen reader to run without writing any result into the checkout" in markdown
    assert "y0 = outcome - u - outcome_noise" in markdown and "y1 = outcome - u" in markdown
    assert "not an execution stop" in markdown and "only to conditional outcome rates" in markdown
    for cell in reader_notebook["cells"]:
        assert cell["metadata"] == {}
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        code = "".join(cell["source"])
        assert "/home/" not in code and "raw_archive=" not in code
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
                assert not any(
                    word in name
                    for word in (
                        "execute_science",
                        "recover",
                        "generate",
                        "fit_design",
                        "sample_scm",
                        "make_scm_prior",
                        "subprocess",
                        "system",
                        "get_ipython",
                        "eval",
                    )
                )


def test_notebook_absent_no_result_no_digest_no_files(
    tmp_path, monkeypatch, reader_notebook, notebook_no_science, capsys
):
    monkeypatch.chdir(tmp_path)
    namespace, html = _execute_reader_notebook(monkeypatch, reader_notebook)
    assert namespace["summary"] is None and namespace["validation"] == "not_attempted"
    assert "NO RESULT" in capsys.readouterr().out and html == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "case,ncap",
    [
        ("empty", 0),
        ("empty", 1),
        ("failure", 12),
        ("missing", 12),
        ("ineligible", 12),
        ("eligible", 12),
        ("rank_deficient", 12),
    ],
)
def test_notebook_accepted_codec_views_no_science(
    tmp_path, monkeypatch, reader_notebook, notebook_no_science, case, ncap
):
    summary = _notebook_synthetic_summary(tmp_path / "synthetic-engineering", case=case, ncap=ncap)
    portable = tmp_path / "copy"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", summary)
    raw = (portable / ref["locator"]).read_bytes()
    alias = portable / "linear-recovery-true-feature-sweep-summary.json"
    alias.write_bytes(raw)
    before = {p.name: p.read_bytes() for p in portable.iterdir()}
    namespace, html = _execute_reader_notebook(
        monkeypatch, reader_notebook, path=alias, sha=ref["sha256"]
    )
    assert namespace["summary"] == summary and namespace["validation"] == "compact_integrity_only"
    assert len(namespace["compact_bytes"]) == len(raw)
    assert not Path(namespace["temporary"]).exists()
    assert before == {p.name: p.read_bytes() for p in portable.iterdir()}
    assert "Actual saved configuration" in html and "All nine frozen selections" in html
    assert "full_rank_over_E" in html and "Wilson 95%" in html
    assert "exact_T_M_cells" in html and "00000000" in html and "11111111" in html
    assert "T_marginal" in html and "M_marginal" in html and "nine_T_M_band_pairs" in html
    assert len(summary["examples"]) == 9
    assert len(namespace["patterns"]) == 256
    if case in ("eligible", "rank_deficient"):
        selected = next(x for x in summary["examples"] if x["attempt"] == 0)
        assert selected["availability_reason"] == "available"
        assert selected["details"]["n_fit"] == 104 and selected["details"]["warmup"] == 0
        assert "rho_zy[0]" in html and "fit_indices" in html and "residual_rmse" in html
        assert "below_minimum_10" in html and "one_world" in html
        assert summary["counts"]["F"] == (case == "eligible")
    elif case == "ineligible":
        assert "identity_or_truth_ineligible" in html
    elif case == "missing":
        assert (
            "known_generation_completed_missing_evidence" in html
            and "generation_outcome_unknown" in html
        )
    elif case == "failure":
        assert "generation_failed" in html
    elif ncap == 0:
        assert "zero_denominator" in html and "band_absent_from_manifest" in html
    else:
        assert "unattempted_budget" in html
        assert any(x["frequency"]["rate"] == 0 for x in summary["planned_configured"]["T_marginal"])
    direct, _ = _execute_reader_notebook(
        monkeypatch, reader_notebook, path=portable / ref["locator"], sha=ref["sha256"]
    )
    assert direct["summary"] == namespace["summary"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_digest",
        "wrong_digest",
        "tamper",
        "malformed",
        "resealed_projection",
        "symlink",
        "directory",
        "fifo",
        "parent_symlink",
        "missing_explicit",
    ],
)
def test_notebook_rejects_bad_present_file_loudly(
    tmp_path, monkeypatch, reader_notebook, notebook_no_science, problem
):
    summary = _notebook_synthetic_summary(tmp_path / "synthetic-engineering", ncap=0)
    portable = tmp_path / "copy"
    portable.mkdir()
    ref = sweep.persist_report_evidence(portable, "summary", summary)
    source = portable / ref["locator"]
    alias = portable / "summary.json"
    alias.write_bytes(source.read_bytes())
    sha = ref["sha256"]
    if problem == "missing_digest":
        sha = None
    elif problem == "wrong_digest":
        sha = "0" * 64
    elif problem == "tamper":
        alias.write_bytes(alias.read_bytes() + b" ")
    elif problem == "malformed":
        alias.write_bytes(b"not JSON\n")
        sha = hashlib.sha256(b"not JSON\n").hexdigest()
    elif problem == "resealed_projection":
        summary["counts"]["A"] = 1
        raw = sweep._world_bytes(summary)
        alias.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
    else:
        alias.unlink()
        if problem == "symlink":
            alias.symlink_to(source)
        elif problem == "directory":
            alias.mkdir()
        elif problem == "fifo":
            import os

            os.mkfifo(alias)
        elif problem == "parent_symlink":
            linked = tmp_path / "linked-parent"
            linked.symlink_to(portable, target_is_directory=True)
            alias = linked / ref["locator"]
    with pytest.raises((ValueError, OSError)) as raised:
        _execute_reader_notebook(monkeypatch, reader_notebook, path=alias, sha=sha)
    if problem == "missing_digest":
        assert "EXPECTED_SUMMARY_SHA256 from independent acceptance" in str(raised.value)
    elif problem in ("wrong_digest", "tamper"):
        assert "SHA256 mismatch" in str(raised.value)


@pytest.mark.parametrize("cwd_kind", ["repository", "notebook", "unrelated", "relative"])
def test_notebook_copied_compact_portability(
    tmp_path, monkeypatch, reader_notebook, notebook_no_science, cwd_kind
):
    summary = _notebook_synthetic_summary(tmp_path / "synthetic-engineering", ncap=1)
    # Stored runtime paths must not be used on this reader's machine.
    evidence = deepcopy(summary["evidence"])
    env = evidence["input"]["contract"]["environment"]
    env["python_executable"] = "/unavailable-original-runtime/python"
    env["environment"]["PYTHONPATH"] = "/unavailable-original-checkout"
    evidence["input"]["digest"] = sweep.digest(evidence["input"]["contract"])
    summary = sweep._render_compact(evidence)
    copied_repo = tmp_path / "copied-repo"
    notebook_dir = copied_repo / "docs/examples"
    data_dir = notebook_dir / "data"
    data_dir.mkdir(parents=True)
    notebook_path = notebook_dir / "linear-recovery-true-feature-sweep.ipynb"
    notebook_path.write_text(json.dumps(reader_notebook))
    raw = sweep._world_bytes(summary)
    sha = hashlib.sha256(raw).hexdigest()
    (data_dir / "linear-recovery-true-feature-sweep-summary.json").write_bytes(raw)
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(
        {
            "repository": copied_repo,
            "notebook": notebook_dir,
            "unrelated": other,
            "relative": other,
        }[cwd_kind]
    )
    namespace, html = _execute_reader_notebook(
        monkeypatch,
        json.loads(notebook_path.read_text()),
        directory=notebook_dir if cwd_kind in ("unrelated", "relative") else None,
        path=Path("data/linear-recovery-true-feature-sweep-summary.json")
        if cwd_kind == "relative"
        else None,
        sha=sha,
    )
    assert namespace["summary"] == summary
    assert "unavailable-original-runtime" in html
    assert namespace["summary_path"] == data_dir / "linear-recovery-true-feature-sweep-summary.json"
    assert sorted(p.name for p in data_dir.iterdir()) == [
        "linear-recovery-true-feature-sweep-summary.json"
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "Path('linear-recovery-true-feature-sweep.ipynb').is_file()",
        "recover_execution()",
        "sweep.recover_execution()",
        "execute_science()",
        "sweep.execute_science()",
        "sample_scm()",
        "module.sample_scm()",
        "fit_design()",
        "module.fit_design()",
    ],
)
def test_notebook_call_symbol_guard_probes(reader_notebook, notebook_no_science, expression):
    # Exercise the actual notebook guard, including the former filename false positive.
    notebook = deepcopy(reader_notebook)
    notebook["cells"].append(
        {
            "cell_type": "code",
            "id": "guard-probe",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [expression],
        }
    )
    if expression.startswith("Path("):
        test_notebook_clean_json_readonly_cells_and_owner_policy(notebook, notebook_no_science)
    else:
        with pytest.raises(AssertionError):
            test_notebook_clean_json_readonly_cells_and_owner_policy(notebook, notebook_no_science)


@pytest.mark.parametrize(
    "source",
    [
        "from scripts.linear_recovery_true_feature_sweep import execute_science as alias; alias()",
        "from scripts.linear_recovery_true_feature_sweep import recover_execution as alias; alias()",
        "from scripts.linear_recovery_true_feature_sweep import fit_design as alias; alias()",
        "from pymc_generator import sample_scm as alias; alias()",
        "from pymc_generator import make_scm_prior as alias; alias()",
    ],
)
def test_notebook_runtime_guard_blocks_imported_aliases(notebook_no_science, source):
    with pytest.raises(pytest.fail.Exception, match="attempted science"):
        exec(source, {})



def test_saved_notebook_trusts_only_the_accepted_compact_digest():
    notebook = json.loads(Path("docs/examples/linear-recovery-true-feature-sweep.ipynb").read_text())
    source = "\n".join(notebook["cells"][1]["source"])
    assert 'EXPECTED_SUMMARY_SHA256 = "d8b6ce0d4c91e6f3da864103e634a7c33ecfba89b9837f83707e9ab49da613c8"' in source
    assert 'EXPECTED_SUMMARY_SHA256 = None' not in source
    assert "independently accepted report SHA-256" in "\n".join(notebook["cells"][0]["source"])


@pytest.mark.parametrize("case", ["accepted", "tampered", "wrong_digest", "none_digest", "missing"])
def test_saved_reader_cell_handles_publication_fixtures(case, tmp_path, monkeypatch, capsys):
    import shutil

    notebook = json.loads(Path("docs/examples/linear-recovery-true-feature-sweep.ipynb").read_text())
    (tmp_path / "data").mkdir()
    source = Path("docs/examples/data/linear-recovery-true-feature-sweep-summary.json")
    report = tmp_path / "data/linear-recovery-true-feature-sweep-summary.json"
    if case != "missing":
        shutil.copyfile(source, report)
    if case == "tampered":
        with report.open("ab") as stream:
            stream.write(b" ")
    namespace = {}
    exec("\n".join(notebook["cells"][1]["source"]), namespace)
    if case == "wrong_digest":
        namespace["EXPECTED_SUMMARY_SHA256"] = "0" * 64
    elif case == "none_digest":
        namespace["EXPECTED_SUMMARY_SHA256"] = None
    monkeypatch.chdir(tmp_path)
    code = "\n".join(notebook["cells"][2]["source"])
    if case == "accepted":
        exec(code, namespace)
        assert namespace["validation"] == "compact_integrity_only"
        assert namespace["summary"]["counts"]["E"] == 139
    elif case == "missing":
        exec(code, namespace)
        assert "NO RESULT" in capsys.readouterr().out
        assert namespace["summary"] is None
    else:
        with pytest.raises(ValueError):
            exec(code, namespace)


# All seeds and cases below are fixed controlled fixtures, not a prevalence sweep.


def controlled_world(*, saturation="hill", carryover="geometric", burn=8, **overrides):
    settings = {
        "n_treatments": 3,
        "n_covariates": 2,
        "n_latent": 1,
        "n_time_steps": 104,
        "carryover_burn_in": burn,
        "l_max": 8,
        "trajectories": "composable",
        "outcome_std_mode": "relative",
        "rw_baseline_std_range": (0.0, 0.0),
        "baseline_floor": None,
        "confounding_strength_range": (0.4, 0.4),
        "edge_budget": {
            "cy": (2, 2),
            "dc": (3, 3),
            "dz": (2, 2),
            "dy": (1, 1),
            "zy": (1, 1),
            "zc": (6, 6),
            "cc": (3, 3),
            "zz": (1, 1),
        },
        "saturation_family_probs": {
            name: float(name == saturation) for name in SATURATION_FAMILY_KEYS
        },
        "carryover_family_probs": {
            name: float(name == carryover) for name in CARRYOVER_FAMILY_KEYS
        },
    }
    settings.update(overrides)
    return sample_scm(make_scm_prior(**settings), seed=43)


@pytest.fixture(scope="module")
def generated_world():
    return controlled_world()


def assert_eligible(result):
    assert result["status"]["truth_eligibility"] == "eligible", result["reasons"]
    assert all(metric["passed"] for metric in result["identities"].values())
    assert all(metric["all_max_abs"] <= 2e-12 for metric in result["identities"].values())


def test_generate_world_actual_public_contract_and_determinism():
    attempt = {"n_treatments": 2, "n_covariates": 2, "n_latent": 1, "world_seed": 43}
    first = sweep.generate_world(attempt, carryover_burn_in=8)
    second = sweep.generate_world(attempt, carryover_burn_in=8)
    assert (first.n_treatments, first.n_covariates, first.n_latent) == (2, 2, 1)
    assert first.cfg.outcome_std_mode == "relative"
    assert first.cfg.baseline_floor is None
    assert first.cfg.rw_baseline_std_range == (0.0, 0.0)
    assert first.cfg.trajectory_components_enabled
    assert first.seed == 43
    for name in first.data:
        np.testing.assert_array_equal(first.data[name], second.data[name])
    np.testing.assert_array_equal(
        first.data["baseline_intrinsic"], np.full(104, first.data["baseline_intrinsic"][0])
    )
    assert_eligible(sweep.analyze_world(first))
    with pytest.raises(ValueError, match="dimensions"):
        sweep.generate_world({"n_treatments": 1, "n_latent": 1, "extra": 0})
    with pytest.raises(ValueError, match="one latent"):
        sweep.generate_world(dict(attempt, n_latent=2))


@pytest.mark.parametrize("saturation", SATURATION_FAMILY_KEYS)
@pytest.mark.parametrize("carryover", CARRYOVER_FAMILY_KEYS)
def test_every_generated_mechanism_pair(saturation, carryover):
    world = controlled_world(saturation=saturation, carryover=carryover)
    result = sweep.analyze_world(world)
    assert_eligible(result)
    assert result["n_full"] == 112
    assert result["n_fit"] == 104
    assert result["status"]["rank"] == "full_rank"
    assert result["status"]["noisefree_recovery"] is True
    assert result["status"]["paired_noisy"]["available"] is True
    assert result["status"]["raw_observed_misspecified"] is True
    np.testing.assert_allclose(
        result["fits"]["y0"]["coefficients"], result["truth"], atol=2e-10, rtol=0
    )


def test_original_order_targets_rows_and_standardization(generated_world):
    world = generated_world
    rows = np.arange(3, 100, 2)
    result = sweep.analyze_world(world, fit_indices=rows)
    assert_eligible(result)
    dc, dz = np.flatnonzero(world.g["g_cy"]), np.flatnonzero(world.g["g_zy"])
    assert len(dc) == 2 and len(dz) == 1  # feeder-only C and indirect-only Z remain upstream
    assert result["coefficient_order"] == [
        *(f"beta[{k}]" for k in dc),
        *(f"rho_zy[{m}]" for m in dz),
        "intercept",
    ]
    np.testing.assert_array_equal(
        result["truth"],
        np.r_[
            world.params["beta"][dc],
            world.params["rho_zy"][dz],
            world.data["baseline_intrinsic"][0],
        ],
    )
    np.testing.assert_array_equal(result["fit_indices"], rows)
    np.testing.assert_array_equal(np.flatnonzero(result["fit_row_mask"]), rows)
    assert result["all_row_mask"].all()
    np.testing.assert_allclose(
        result["design"][:, len(dc) : -1], world.data["covariates"][:, dz], atol=2e-12, rtol=0
    )
    target = result["targets"]
    u = world.data["latent_unobserved_contribution"].sum(axis=1)
    np.testing.assert_array_equal(
        target["y0"], world.data["outcome"] - u - world.data["outcome_noise"]
    )
    np.testing.assert_array_equal(target["y1"], world.data["outcome"] - u)
    np.testing.assert_allclose(
        target["y1"] - target["y0"], world.data["outcome_noise"], atol=2e-12, rtol=0
    )
    assert np.max(np.abs(world.data["indirect_effects"])) > 0
    standardized = result["standardized_X"]
    np.testing.assert_array_equal(standardized[:, -1], np.ones(len(rows)))
    np.testing.assert_allclose(standardized[:, :-1].mean(axis=0), 0, atol=2e-12, rtol=0)
    np.testing.assert_allclose(standardized[:, :-1].std(axis=0), 1, atol=2e-12, rtol=0)
    for name in ("y0", "y1"):
        expected = np.linalg.lstsq(result["design"][rows], target[name][rows], rcond=1e-10)[0]
        np.testing.assert_array_equal(result["fits"][name]["coefficients"], expected)
    expected_delta = np.linalg.lstsq(
        result["design"][rows], target["outcome_noise"][rows], rcond=1e-10
    )[0]
    np.testing.assert_allclose(
        result["fits"]["y1"]["coefficients"] - result["fits"]["y0"]["coefficients"],
        expected_delta,
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("burn", [0, 8, 16])
def test_full_history_not_zero_padded_window(burn):
    world = controlled_world(burn=burn, trajectories="texture")
    result = sweep.analyze_world(world)
    assert_eligible(result)
    full = result["audit"]["full"]
    assert full["C"].shape == (104 + burn, 3)
    for k in result["direct_treatments"]:
        response = result["audit"]["equation_parameters"][f"C{k + 1}"]["response"]
        expected = sweep.saturation_path(
            sweep.carryover_path(full["C"][:, k], response["carryover"]), response["saturation"]
        )[burn:]
        np.testing.assert_array_equal(result["audit"]["reported"]["phi"][:, k], expected)
        if burn:
            wrong = sweep.saturation_path(
                sweep.carryover_path(full["C"][burn:, k], response["carryover"]),
                response["saturation"],
            )
            assert np.max(np.abs(wrong - expected)) > 1e-4


def test_all_composable_components_shocks_and_mixed_innovations():
    overrides = {
        f"{kind}_{component}_inclusion_prob": 1.0
        for kind in ("treatment", "covariate")
        for component in TRAJECTORY_COMPONENTS
    }
    world = controlled_world(
        **overrides,
        n_treatment_shocks=2,
        treatment_shock_length_range=(3, 4),
        treatment_shock_level_range=(0.5, 1.0),
    )
    result = sweep.analyze_world(world)
    assert_eligible(result)
    audit = result["audit"]
    for name in ("C1", "C2", "C3", "Z1", "Z2"):
        assert all(audit["equation_parameters"][name]["trajectory"]["use"].values())
    np.testing.assert_array_equal(
        audit["mixed_eps_c"],
        np.sqrt(1 - 0.4**2) * world.exogenous["eps_c"] + 0.4 * world.exogenous["eps_b"][:, None],
    )
    assert not np.array_equal(audit["mixed_eps_c"], world.exogenous["eps_c"])
    shock = audit["equation_parameters"]["treatment_shocks"]
    mask = shock["mask_full"] != 0
    assert mask.any()
    np.testing.assert_array_equal(audit["full"]["C"][mask], shock["level_full"][mask])
    np.testing.assert_array_equal(audit["full"]["C_base"][mask], shock["level_full"][mask])


@pytest.mark.parametrize("rows", [[0], [0, 1]])
def test_rank_deficiency_is_orthogonal_to_identity_and_noisy_availability(generated_world, rows):
    result = sweep.analyze_world(generated_world, fit_indices=rows)
    assert_eligible(result)
    assert result["status"]["rank"] == "rank_deficient"
    assert result["status"]["noisefree_recovery"] is False
    assert result["status"]["paired_noisy"]["available"] is True
    assert result["status"]["raw_observed_misspecified"] is True
    assert "rank_deficient" in result["reasons"]
    assert result["raw_design"]["rank"] < result["p"]
    if len(rows) == 1:
        np.testing.assert_array_equal(result["constant_columns"], np.arange(result["p"] - 1))
        np.testing.assert_array_equal(
            result["standardized_X"][0], np.r_[np.zeros(result["p"] - 1), 1]
        )


@pytest.mark.parametrize(
    "field",
    [
        "treatments",
        "treatments_base",
        "covariates",
        "baseline_intrinsic",
        "contributions_observed",
        "contributions",
        "covariate_contribution",
        "latent_unobserved_contribution",
        "indirect_effects_by_source",
        "outcome",
        "outcome_noise",
        "saturation_scale",
    ],
)
def test_identity_failure_on_excluded_row_blocks_both_fits(generated_world, monkeypatch, field):
    world = deepcopy(generated_world)
    index = int(np.flatnonzero(world.g["g_cy"])[0]) if field == "saturation_scale" else 0
    world.data[field].flat[index] += 0.1

    def forbidden(*args, **kwargs):
        pytest.fail("OLS ran before identity eligibility")

    monkeypatch.setattr(sweep, "fit_design", forbidden)
    result = sweep.analyze_world(world, fit_indices=np.arange(20, 104))
    assert result["status"]["truth_eligibility"] == "ineligible"
    assert result["fits"] == {}
    assert result["reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_history",
        "short_history",
        "nan_history",
        "shape",
        "graph_order",
        "rows",
        "floor",
        "missing_truth",
    ],
)
def test_pre_fit_contract_failures(generated_world, monkeypatch, mutation):
    world = deepcopy(generated_world)
    kwargs = {}
    if mutation == "missing_history":
        world._exogenous.pop("eps_c")
    elif mutation == "short_history":
        world._exogenous["eps_c"] = world._exogenous["eps_c"][8:]
    elif mutation == "nan_history":
        world._exogenous["eps_c"][0, 0] = np.nan
    elif mutation == "shape":
        world.data["outcome_noise"] = world.data["outcome_noise"][:, None]
    elif mutation == "graph_order":
        world.g["g_cc"][2, 0] = 1
    elif mutation == "rows":
        kwargs["fit_indices"] = [2, 1]
    elif mutation == "floor":
        world.cfg.baseline_floor = 0.0
    else:
        world.params.pop("beta")

    def forbidden(*args, **kwargs):
        pytest.fail("OLS ran before audit validation")

    monkeypatch.setattr(sweep, "fit_design", forbidden)
    result = sweep.analyze_world(world, **kwargs)
    assert result["fits"] == {}
    assert result["status"]["truth_eligibility"] == "ineligible"
    assert result["reasons"]


@pytest.mark.parametrize(
    "family,parameters",
    [
        ("hill", {"slope": (100.0, 100.0), "kappa_mult": (1.0, 1.0)}),
        ("logistic", {"lam": (1e-200, 1e-200)}),
        ("michaelis_menten", {"kappa_mult": (1e30, 1e30)}),
        ("tanh", {"c": (1e-200, 1e-200)}),
        ("root", {"alpha": (0.01, 0.01)}),
    ],
)
def test_generated_extreme_saturation_arithmetic(family, parameters):
    from pymc_generator.sampler import SCMPrior

    ranges = deepcopy(SCMPrior().saturation_prior_ranges)
    ranges[family] = parameters
    world = controlled_world(saturation=family, carryover="none", saturation_prior_ranges=ranges)
    result = sweep.analyze_world(world)
    assert_eligible(result)
    assert np.isfinite(result["audit"]["full"]["phi"]).all()


@pytest.mark.parametrize(
    "carryover,overrides",
    [
        ("geometric", {"carryover_alpha_range": (0.0, 0.0)}),
        ("geometric", {"carryover_alpha_range": (1.0, 1.0)}),
        ("weibull", {"l_max": 1}),
        ("weibull", {"weibull_lam_range": (1e30, 1e30), "weibull_k_range": (12.0, 12.0)}),
    ],
)
def test_generated_carryover_extremes(carryover, overrides):
    world = controlled_world(carryover=carryover, **overrides)
    result = sweep.analyze_world(world)
    assert_eligible(result)
    if overrides.get("weibull_lam_range"):
        np.testing.assert_array_equal(result["audit"]["full"]["phi"], 0)
        assert result["status"]["rank"] == "rank_deficient"


def test_reconstruction_does_not_divide_by_beta_or_recompute_scale(generated_world):
    world = deepcopy(generated_world)
    expected = sweep.reconstruct_features(world)
    world.params["beta"][:] = 0.0
    actual = sweep.reconstruct_features(world)
    np.testing.assert_array_equal(actual["full"]["phi"], expected["full"]["phi"])
    for k in range(world.n_treatments):
        assert (
            actual["equation_parameters"][f"C{k + 1}"]["response"]["saturation"]["scale"]
            == world.data["saturation_scale"][k]
        )
    assert any(
        not np.isclose(world.data["treatments"][:, k].mean(), world.data["saturation_scale"][k])
        for k in range(world.n_treatments)
    )


@pytest.mark.parametrize("rows", [[], [0, 0], [104], [-1], [0.0, 1.0], [[0, 1]]])
def test_invalid_reported_row_selector_is_ineligible(generated_world, rows):
    result = sweep.analyze_world(generated_world, fit_indices=rows)
    assert result["fits"] == {}
    assert result["status"]["truth_eligibility"] == "ineligible"
    assert "fit indices" in result["reasons"][0]


@pytest.mark.parametrize("kind", ["mm", "weibull"])
def test_public_config_rejects_out_of_support_extreme_fixtures(kind):
    from pymc_generator.sampler import SCMPrior

    ranges = deepcopy(SCMPrior().saturation_prior_ranges)
    ranges["michaelis_menten"]["kappa_mult"] = (1e200, 1e200)
    overrides = (
        {"saturation_prior_ranges": ranges}
        if kind == "mm"
        else {"weibull_lam_range": (1e100, 1e100)}
    )
    with pytest.raises(ValueError, match="float32 corpus storage maximum"):
        make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1, **overrides)


@pytest.mark.parametrize("counts", [(0, 1), (11, 1), (1, 0), (1, 11), (True, 1), (1.5, 2)])
def test_generate_world_rejects_out_of_population_counts(counts):
    with pytest.raises(ValueError, match="integer in"):
        sweep.generate_world(
            {"n_treatments": counts[0], "n_covariates": counts[1], "n_latent": 1, "world_seed": 43}
        )


def test_raw_and_standardized_conditioning_are_separate(generated_world):
    result = sweep.analyze_world(generated_world)
    assert_eligible(result)
    for key, design in (
        ("raw_design", result["design"]),
        ("standardized_design", result["standardized_X"]),
    ):
        values = np.linalg.svd(design, compute_uv=False)
        np.testing.assert_array_equal(result[key]["singular_values"], values)
        assert result[key]["rank_cutoff"] == 1e-10 * values[0]
        assert result[key]["condition"] == values[0] / values[-1]
        assert result[key]["conditioning"] in {"easy", "moderate", "hard"}
    assert result["raw_design"]["condition"] != result["standardized_design"]["condition"]
    assert (
        result["target_metrics"]["outcome_noise"]["all"]["std"]
        == generated_world.data["outcome_noise"].std()
    )
