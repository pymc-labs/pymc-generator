"""Fast contract tests for the Generator-owned Oracle cohort harness."""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
from dataclasses import replace
from importlib import metadata
from types import SimpleNamespace

import pytest
import xarray as xr

import pymc_generator.oracle_harness as harness
from pymc_generator.oracle_harness import (
    OracleRunSpec,
    OracleWorldSpec,
    canonical_numerical_hash,
    run_frozen_cohort,
    run_pair,
)
from pymc_generator.oracle_sampling import OracleSamplingConfig, OracleSamplingReceipt

_PACKAGE_VERSION = metadata.version("pymc-generator")

_CASES = (
    "constant_intercept",
    "smooth_baseline",
    "sparse_inputs",
    "omitted_channel",
    "omitted_control",
    "logistic_saturation",
    "five_controls",
)


def _config() -> OracleSamplingConfig:
    return OracleSamplingConfig(
        draws=1000,
        tune=900,
        chains=5,
        cores=5,
        target_accept=0.95,
        latent="sampled",
        progressbar=False,
        compute_convergence_checks=False,
        adaptation="draw_diag",
    )


def _spec(run_id: str = "run", submission_id: str | None = None) -> OracleRunSpec:
    worlds = tuple(
        OracleWorldSpec(
            case,
            "w0000",
            f"source:{case}",
            index,
            (0, 2),
            source_digest=f"{index + 1:064x}",
        )
        for index, case in enumerate(_CASES)
    )
    return OracleRunSpec(
        run_id,
        worlds,
        _config(),
        submission_id=submission_id,
        generator_identity={"commit": harness._runtime_generator_commit()},
        package_identity={"name": "pymc-generator", "version": _PACKAGE_VERSION},
        environment_identity={
            "lock_sha256": "a" * 64,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        process_identity={"pid": os.getpid(), "executable": sys.executable},
    )


def _identity_kwargs(spec: OracleRunSpec) -> dict[str, object]:
    return {
        "generator_identity": spec.generator_identity,
        "package_identity": spec.package_identity,
        "environment_identity": spec.environment_identity,
        "process_identity": spec.process_identity,
    }


def _fake_fitter(calls: list[tuple[str, int]], payload: float = 1.0):
    def fit(compiled, world, config, criteria, world_spec):
        calls.append((world_spec.key, config.random_seed))
        coords = {"chain": [0], "draw": [0]}
        dataset = xr.Dataset({"theta": (("chain", "draw"), [[payload]])}, coords=coords)
        tree = xr.DataTree.from_dict({"/posterior": dataset, "/sample_stats": dataset})
        requested = config.to_dict()
        requested.pop("latent")
        compile_identity = {
            "engine": "nutpie",
            "backend": getattr(compiled, "backend", "numba"),
            "instance_id": getattr(compiled, "_instance_id", None),
            "owner_pid": getattr(compiled, "_owner_pid", None),
        }
        receipt = OracleSamplingReceipt(
            schema_version=1,
            kind="pymc_generator.oracle_compiled_sampling_receipt",
            package_version=_PACKAGE_VERSION,
            environment={"python": platform.python_version(), "nutpie": "0.16.11"},
            identities={
                "world": {
                    "case_id": world_spec.case_id,
                    "world_id": world_spec.world_id,
                    "world_seed": world_spec.world_seed,
                },
                "source": dict(world_spec.source_identity),
                "configuration": {"effective_config": config.to_dict()},
                "template": {"signature": getattr(compiled, "template_signature", "s")},
                "compile": compile_identity,
                "observed_indices": harness._selector_identity(tuple(world_spec.observed_indices)),
            },
            oracle={"latent": config.latent},
            sampling={"requested": requested},
            timing={"elapsed_seconds": 0.0},
            posterior={"present": True, "included_in_receipt": False},
            diagnostics={},
            health={},
            limitations=(),
        )
        return SimpleNamespace(idata=tree, receipt=receipt)

    return fit


def _fake_run_kwargs():
    return {
        "build_template": lambda world, **kw: SimpleNamespace(signature="s"),
        "compile_template": lambda template: SimpleNamespace(
            backend="numba", _instance_id="fake", _owner_pid=os.getpid(), template_signature="s"
        ),
    }


def test_immutable_artifact_is_owner_only(tmp_path):
    artifact = tmp_path / "artifact.json"

    harness._write_immutable(artifact, {"status": "complete"})

    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_frozen_run_compiles_once_per_signature_and_derives_seeds(tmp_path):
    fit_calls: list[tuple[str, int]] = []
    compile_calls: list[str] = []

    def build(world, **kwargs):
        return SimpleNamespace(signature="shared")

    def compile(template):
        compile_calls.append(template.signature)
        return SimpleNamespace(
            backend="numba",
            _instance_id="fake",
            _owner_pid=os.getpid(),
            template_signature=template.signature,
        )

    selected = run_frozen_cohort(
        _spec(),
        lambda locator: object(),
        _fake_fitter(fit_calls),
        tmp_path,
        build_template=build,
        compile_template=compile,
    )
    assert len(selected["rows"]) == 7
    assert compile_calls == ["shared"]
    assert sorted(seed for _, seed in fit_calls) == list(range(2, 9))


def test_invalid_cohort_is_rejected_before_loader(tmp_path):
    spec = _spec()
    invalid = OracleRunSpec(
        spec.run_id,
        spec.worlds[:-1],
        spec.config,
        **_identity_kwargs(spec),
    )
    with pytest.raises(ValueError, match="exactly seven"):
        run_frozen_cohort(
            invalid, lambda _: pytest.fail("loader called"), _fake_fitter([]), tmp_path
        )


def test_hash_excludes_container_metadata():
    coords = {"chain": [0], "draw": [0]}
    dataset = xr.Dataset({"theta": (("chain", "draw"), [[1.0]])}, coords=coords)
    first = xr.DataTree.from_dict({"/posterior": dataset, "/sample_stats": dataset})
    second = first.copy(deep=True)
    second["posterior"].ds.attrs["timestamp"] = "different"
    assert canonical_numerical_hash(first) == canonical_numerical_hash(second)


def test_pair_runs_twice_and_rejects_overwrite(tmp_path):
    calls: list[tuple[str, int]] = []
    result = run_pair(
        _spec("first", "submission-a"),
        _spec("second", "submission-b"),
        lambda locator: object(),
        _fake_fitter(calls),
        tmp_path,
        **_fake_run_kwargs(),
    )
    assert result["status"] == "pass"
    assert len(calls) == 14

    with pytest.raises(FileExistsError):
        run_pair(
            _spec("first", "submission-a"),
            _spec("second", "submission-b"),
            lambda locator: object(),
            _fake_fitter([]),
            tmp_path,
            **_fake_run_kwargs(),
        )


def test_source_digest_is_required():
    with pytest.raises(ValueError, match="source_digest"):
        OracleWorldSpec("constant_intercept", "w0000", "source", 1, (0,))


def test_effective_seed_collisions_are_rejected_before_fitting(tmp_path):
    spec = _spec()
    worlds = list(spec.worlds)
    worlds[1] = replace(worlds[1], world_seed=worlds[0].world_seed)
    invalid = OracleRunSpec(
        spec.run_id,
        worlds,
        spec.config,
        **_identity_kwargs(spec),
    )
    with pytest.raises(ValueError, match="colliding effective random seeds"):
        run_frozen_cohort(
            invalid, lambda _: pytest.fail("loader called"), _fake_fitter([]), tmp_path
        )


def test_frozen_config_selector_and_ledger_are_exact_and_canonical(tmp_path):
    calls: list[tuple[OracleSamplingConfig, tuple[int, ...]]] = []

    def fit(compiled, world, config, criteria, world_spec):
        calls.append((config, tuple(world_spec.observed_indices)))
        return _fake_fitter([])(compiled, world, config, criteria, world_spec)

    selected = run_frozen_cohort(
        _spec(),
        lambda locator: object(),
        fit,
        tmp_path,
        **_fake_run_kwargs(),
    )
    assert len(calls) == 7
    assert {(config.draws, config.tune, config.chains, config.cores) for config, _ in calls} == {
        (1000, 900, 5, 5)
    }
    assert {config.target_accept for config, _ in calls} == {0.95}
    assert {config.latent for config, _ in calls} == {"sampled"}
    assert {config.adaptation for config, _ in calls} == {"draw_diag"}
    assert {config.discard_tuned_samples for config, _ in calls} == {True}
    assert {indices for _, indices in calls} == {(0, 2)}

    manifest = (tmp_path / "manifests" / "run.json").read_text()
    assert (
        manifest == json.dumps(json.loads(manifest), sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert len(selected["rows"]) == 7
    for row in selected["rows"]:
        assert row["attempt"] == "a001"
        assert row["source_identity"]["sha256"]
        assert row["posterior_sha256"] and row["sample_stats_sha256"]
        assert (tmp_path / row["receipt_path"]).is_file()
        attempt = tmp_path / "attempts" / "run" / row["key"] / "a001.json"
        started = tmp_path / "attempts" / "run" / row["key"] / "a001.started.json"
        assert json.loads(attempt.read_text())["status"] == "completed"
        assert json.loads(started.read_text())["status"] == "started"


def test_duplicate_non_frozen_and_deviating_config_rejected_before_loader(tmp_path):
    spec = _spec()
    duplicate = OracleRunSpec(
        spec.run_id,
        (*spec.worlds[:-1], spec.worlds[0]),
        spec.config,
        **_identity_kwargs(spec),
    )
    not_frozen = list(spec.worlds)
    not_frozen[0] = replace(not_frozen[0], case_id="other_case")
    invalid_case = OracleRunSpec(
        spec.run_id,
        not_frozen,
        spec.config,
        **_identity_kwargs(spec),
    )
    invalid_config = OracleRunSpec(
        spec.run_id,
        spec.worlds,
        replace(spec.config, draws=999),
        **_identity_kwargs(spec),
    )
    for invalid, message in (
        (duplicate, "duplicate"),
        (invalid_case, "owner-approved"),
        (invalid_config, "draws"),
    ):
        with pytest.raises(ValueError, match=message):
            run_frozen_cohort(
                invalid, lambda _: pytest.fail("loader called"), _fake_fitter([]), tmp_path
            )


def test_failure_writes_started_and_error_without_receipt_or_retry(tmp_path):
    calls = 0

    def failing_fit(*args):
        nonlocal calls
        calls += 1
        raise RuntimeError("fake failure")

    with pytest.raises(RuntimeError, match="fake failure"):
        run_frozen_cohort(
            _spec(), lambda locator: object(), failing_fit, tmp_path, **_fake_run_kwargs()
        )
    assert calls == 1
    attempt_dir = tmp_path / "attempts" / "run" / "constant_intercept" / "w0000"
    assert json.loads((attempt_dir / "a001.started.json").read_text())["status"] == "started"
    assert json.loads((attempt_dir / "a001.json").read_text())["status"] == "error"
    assert not (tmp_path / "receipts").exists()
    assert not (tmp_path / "selected" / "run.json").exists()


def test_missing_dataset_is_failure_without_success_receipt(tmp_path):
    def fit(compiled, world, config, criteria, world_spec):
        valid = _fake_fitter([])(compiled, world, config, criteria, world_spec)
        dataset = xr.Dataset({"theta": (("chain", "draw"), [[1.0]])})
        return SimpleNamespace(
            idata=xr.DataTree.from_dict({"/posterior": dataset}),
            receipt=valid.receipt,
        )

    with pytest.raises(ValueError, match="sample_stats"):
        run_frozen_cohort(_spec(), lambda locator: object(), fit, tmp_path, **_fake_run_kwargs())
    assert not (tmp_path / "receipts").exists()
    assert not (tmp_path / "selected" / "run.json").exists()


def test_identity_mappings_are_recursively_immutable():
    nested = {"sha256": "a" * 64, "metadata": {"source": ["original"]}}
    world = OracleWorldSpec(
        "constant_intercept", "w0000", "source", 1, (0,), source_identity=nested
    )
    nested["metadata"]["source"].append("changed")
    assert tuple(world.source_identity["metadata"]["source"]) == ("original",)
    spec = _spec()
    generator = {"commit": "original"}
    copied = OracleRunSpec(
        spec.run_id,
        spec.worlds,
        spec.config,
        generator_identity=generator,
        package_identity=spec.package_identity,
        environment_identity=spec.environment_identity,
        process_identity=spec.process_identity,
    )
    generator["commit"] = "changed"
    assert copied.generator_identity["commit"] == "original"


def test_path_tokens_and_pair_id_are_confined(tmp_path):
    with pytest.raises(ValueError, match="single-component"):
        _spec("../outside")
    with pytest.raises(ValueError, match="single-component"):
        run_pair(
            _spec("first", "submission-a"),
            _spec("second", "submission-b"),
            lambda locator: object(),
            _fake_fitter([]),
            tmp_path,
            pair_id="../outside",
        )
    assert not (tmp_path.parent / "outside").exists()


def test_pair_rejects_hash_and_compact_mismatches(tmp_path):
    calls = 0

    def changing_fit(compiled, world, config, criteria, world_spec):
        nonlocal calls
        calls += 1
        return _fake_fitter([], payload=1.0 if calls <= 7 else 2.0)(
            compiled, world, config, criteria, world_spec
        )

    with pytest.raises(ValueError, match="hashes"):
        run_pair(
            _spec("first", "submission-a"),
            _spec("second", "submission-b"),
            lambda locator: object(),
            changing_fit,
            tmp_path,
            **_fake_run_kwargs(),
        )
    assert not (tmp_path / "comparisons").exists()

    compact_root = tmp_path / "compact"
    calls = 0

    def changing_template(world, **kw):
        nonlocal calls
        calls += 1
        return SimpleNamespace(signature="first" if calls <= 7 else "second")

    with pytest.raises(ValueError, match="compact fields"):
        run_pair(
            _spec("first", "submission-a"),
            _spec("second", "submission-b"),
            lambda locator: object(),
            _fake_fitter([]),
            compact_root,
            build_template=changing_template,
            compile_template=lambda template: SimpleNamespace(
                backend="numba",
                _instance_id="fake",
                _owner_pid=os.getpid(),
                template_signature=template.signature,
            ),
        )


def test_provenance_is_required_and_nutpie_version_is_verified(tmp_path, monkeypatch):
    spec = _spec()
    with pytest.raises(ValueError, match="package_identity"):
        OracleRunSpec(
            spec.run_id,
            spec.worlds,
            spec.config,
            generator_identity=spec.generator_identity,
            environment_identity=spec.environment_identity,
            process_identity=spec.process_identity,
        )

    bad_package = replace(
        spec,
        package_identity={"name": "pymc-generator", "version": "false"},
    )
    with pytest.raises(ValueError, match="package_identity.version"):
        run_frozen_cohort(
            bad_package,
            lambda _: pytest.fail("loader called"),
            _fake_fitter([]),
            tmp_path,
        )

    bad_commit = replace(
        spec,
        generator_identity={"commit": "0" * 40},
    )
    with pytest.raises(ValueError, match="generator_identity.commit"):
        run_frozen_cohort(
            bad_commit,
            lambda _: pytest.fail("loader called"),
            _fake_fitter([]),
            tmp_path,
        )

    original_commit = harness._runtime_generator_commit
    monkeypatch.setattr(
        harness,
        "_runtime_generator_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable commit")),
    )
    with pytest.raises(RuntimeError, match="unavailable commit"):
        run_frozen_cohort(
            spec,
            lambda _: pytest.fail("loader called"),
            _fake_fitter([]),
            tmp_path,
        )
    monkeypatch.setattr(harness, "_runtime_generator_commit", original_commit)

    original_version = harness.metadata.version
    monkeypatch.setattr(
        harness.metadata,
        "version",
        lambda name: "0.16.10" if name == "nutpie" else original_version(name),
    )
    with pytest.raises(RuntimeError, match="exactly 0.16.11"):
        run_frozen_cohort(
            spec,
            lambda _: pytest.fail("loader called"),
            _fake_fitter([]),
            tmp_path,
        )
    assert not (tmp_path / "manifests").exists()


def test_compile_provenance_requires_actual_numba_identity(tmp_path):
    for index, (compiled, message) in enumerate(
        (
            (SimpleNamespace(_instance_id="fake", _owner_pid=os.getpid()), "backend"),
            (
                SimpleNamespace(backend="pymc", _instance_id="fake", _owner_pid=os.getpid()),
                "backend",
            ),
            (SimpleNamespace(backend="numba", _instance_id="", _owner_pid=os.getpid()), "instance"),
            (
                SimpleNamespace(backend="numba", _instance_id="fake", _owner_pid=os.getpid() + 1),
                "PID",
            ),
        )
    ):
        with pytest.raises(ValueError, match=message):
            run_frozen_cohort(
                _spec(),
                lambda locator: object(),
                _fake_fitter([]),
                tmp_path / f"compile-{index}",
                build_template=lambda world, **kw: SimpleNamespace(signature="s"),
                compile_template=lambda template, compiled=compiled: compiled,
            )


def test_receipt_write_error_cannot_leave_selection(tmp_path, monkeypatch):
    original_write = harness._write_immutable

    def fail_receipt(path, payload):
        if path.parent.name == "w0000" and path.parent.parent.parent.parent.name == "receipts":
            raise OSError("fake receipt write error")
        return original_write(path, payload)

    monkeypatch.setattr(harness, "_write_immutable", fail_receipt)
    with pytest.raises(OSError, match="fake receipt write error"):
        run_frozen_cohort(
            _spec(),
            lambda locator: object(),
            _fake_fitter([]),
            tmp_path,
            **_fake_run_kwargs(),
        )
    assert not (tmp_path / "selected" / "run.json").exists()
    assert not (tmp_path / "receipts").exists()
    attempt_dir = tmp_path / "attempts" / "run" / "constant_intercept" / "w0000"
    assert json.loads((attempt_dir / "a001.json").read_text())["status"] == "completed"
    assert not (attempt_dir / "a001.committed.json").exists()


def test_same_signature_fits_receive_fresh_world_payloads(tmp_path):
    worlds = {f"source:{case}": SimpleNamespace(payload=index) for index, case in enumerate(_CASES)}
    seen: list[int] = []

    def fit(compiled, world, config, criteria, world_spec):
        seen.append(world.payload)
        return _fake_fitter([])(compiled, world, config, criteria, world_spec)

    run_frozen_cohort(
        _spec(),
        worlds.__getitem__,
        fit,
        tmp_path,
        **_fake_run_kwargs(),
    )
    assert sorted(seen) == list(range(7))


def test_non_receipt_and_stale_bound_receipts_are_rejected(tmp_path):
    def non_receipt(compiled, world, config, criteria, world_spec):
        valid = _fake_fitter([])(compiled, world, config, criteria, world_spec)
        return SimpleNamespace(idata=valid.idata, receipt=SimpleNamespace(to_json=lambda: "{}"))

    with pytest.raises(TypeError, match="OracleSamplingReceipt"):
        run_frozen_cohort(
            _spec(),
            lambda locator: object(),
            non_receipt,
            tmp_path / "non-receipt",
            **_fake_run_kwargs(),
        )

    for field, stale_value, message in (
        ("world", {"case_id": "wrong", "world_id": "w0000", "world_seed": 2}, "world"),
        ("source", {"sha256": "f" * 64}, "source"),
        ("configuration", {"effective_config": {"draws": 1}}, "configuration"),
        ("template", {"signature": "stale"}, "template"),
        (
            "compile",
            {"engine": "nutpie", "backend": "numba", "instance_id": "stale", "owner_pid": None},
            "compile",
        ),
    ):

        def stale_fit(
            compiled,
            world,
            config,
            criteria,
            world_spec,
            field=field,
            stale_value=stale_value,
        ):
            valid = _fake_fitter([])(compiled, world, config, criteria, world_spec)
            identities = {name: dict(value) for name, value in valid.receipt.identities.items()}
            identities[field] = stale_value
            receipt = replace(valid.receipt, identities=identities)
            return SimpleNamespace(idata=valid.idata, receipt=receipt)

        with pytest.raises(ValueError, match=message):
            run_frozen_cohort(
                _spec(),
                lambda locator: object(),
                stale_fit,
                tmp_path / f"stale-{field}",
                **_fake_run_kwargs(),
            )
