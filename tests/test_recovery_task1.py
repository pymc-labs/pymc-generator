"""Fast contract and saved-evidence parity tests; no notebook execution or fitting."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from studies.recovery.analysis import (
    saved_bounded_world,
    saved_summary,
    saved_tv_world,
    score_components,
)
from studies.recovery.artifacts import checkpoint, digest, identity, write_manifest
from studies.recovery.config import ContractError, StudyConfig

ROOT = Path(__file__).resolve().parents[1]


def test_config_preflight():
    cfg = StudyConfig.parse({})
    assert cfg.resolved["study"]["score_weeks"] == [7, 103]
    assert cfg.resolved["study"]["burn_in"] == 8
    with pytest.raises(ContractError, match="OLS"):
        StudyConfig.parse({"study": {"baseline": "varying"}, "methods": {"enabled": ["ols"]}})
    for invalid in (
        {"generator": {"carryover_family": "weibull"}},
        {"generator": {"edge_budget": {"dc": [0, 1]}}},
        {"study": {"score_weeks": [7, 104]}},
        {"generator": {"rw_baseline_std_range": [0, 0.093]}},
        {"pairing": {"mode": "matched", "media_smoothness_beta": 4}},
        {
            "generator": {"rw_smoothness_beta": 4},
            "pairing": {
                "mode": "matched",
                "media_smoothness_beta": 4,
                "coupling_manifest": "paired.json",
            },
        },
        {"methods": {"mcmc": {"sampler": "pymc"}}},
        {"unused": 3},
    ):
        with pytest.raises(ContractError):
            StudyConfig.parse(invalid)


def test_identity_and_stale_cache(tmp_path):
    source = tmp_path / "source.py"
    data = tmp_path / "data.npz"
    source.write_text("source")
    data.write_bytes(b"one")
    cfg = StudyConfig.parse({})
    with pytest.raises(ContractError, match="upstream"):
        identity("fit", cfg, source_files={"model": source})
    first = identity("fit", cfg, source_files={"model": source}, dataset_files={"world": data})
    score_changed = StudyConfig.parse({"score": {"relative": 0.05}})
    assert identity("generate", cfg, source_files={"model": source}) == identity(
        "generate", score_changed, source_files={"model": source}
    )
    assert first == identity(
        "fit", score_changed, source_files={"model": source}, dataset_files={"world": data}
    )
    assert identity(
        "analyze", cfg, source_files={"model": source}, dataset_files={"world": data}
    ) != identity(
        "analyze", score_changed, source_files={"model": source}, dataset_files={"world": data}
    )
    stage = tmp_path / "stage"
    assert checkpoint(stage, first).state == "new"
    write_manifest(stage, first)
    assert checkpoint(stage, first).state == "initialized"
    assert not checkpoint(stage, first).complete
    data.write_bytes(b"two")
    changed = identity("fit", cfg, source_files={"model": source}, dataset_files={"world": data})
    with pytest.raises(ContractError, match="stale"):
        checkpoint(stage, changed)
    assert json.loads((stage / "manifest.json").read_text()) == first
    with pytest.raises(ContractError, match="without matching manifest"):
        checkpoint(tmp_path, first)


def test_review_repro_saved_window_format_and_provenance_rejected(tmp_path):
    tv = ROOT / "notebooks/time_varying_baseline/nuts"
    bounded = ROOT / "notebooks/ols_study_50_seed10029_generator_bounds"
    for directory, kind, config, message in (
        (tv, "tv_nuts", {"study": {"baseline": "varying", "score_weeks": [0, 103]}}, "score_weeks"),
        (
            tv,
            "tv_nuts",
            {"study": {"baseline": "varying", "reported_weeks": [1, 103]}},
            "reported_weeks",
        ),
        (tv, "tv_nuts", {"study": {"baseline": "varying", "seed": 42}}, "seed"),
        (
            tv,
            "tv_nuts",
            {
                "study": {"baseline": "varying", "reported_weeks": [0, 104]},
                "generator": {"n_time_steps": 105},
            },
            "n_time_steps|score_weeks",
        ),
        (bounded, "bounded_ols", {"study": {"baseline": "varying"}}, "baseline"),
        (bounded, "bounded_ols", {"study": {"score_weeks": [0, 103]}}, "score_weeks"),
        (
            tv,
            "tv_nuts",
            {"study": {"baseline": "varying"}, "diagnostics": {"rhat": 1.05}},
            "diagnostics",
        ),
        (
            tv,
            "tv_nuts",
            {"study": {"baseline": "varying"}, "methods": {"enabled": ["advi"]}},
            "method",
        ),
        (
            bounded,
            "bounded_ols",
            {"study": {"worlds": 1}, "methods": {"ols": {"starts": 3}}},
            "starts",
        ),
    ):
        with pytest.raises(ContractError, match=message):
            saved_summary(directory, kind=kind, config=StudyConfig.parse(config))
    with pytest.raises(ContractError, match="manifest"):
        saved_summary(
            tmp_path, kind="tv_nuts", config=StudyConfig.parse({"study": {"baseline": "varying"}})
        )


def test_cli_rejects_legacy_window_before_printing(tmp_path, monkeypatch, capsys):
    from studies.recovery.__main__ import main

    config = tmp_path / "wrong.toml"
    config.write_text('[study]\nbaseline = "varying"\nscore_weeks = [0, 103]\n')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recovery",
            "analyze",
            "--config",
            str(config),
            "--saved",
            str(ROOT / "notebooks/time_varying_baseline/nuts"),
            "--format",
            "tv_nuts",
        ],
    )
    with pytest.raises(ContractError, match="score_weeks"):
        main()
    assert capsys.readouterr().out == ""


def test_review_repro_dgp_feasibility_and_nested_types():
    resolved = StudyConfig.parse({}).resolved
    assert resolved["generator"]["edge_budget"]["cy"][0] >= 1
    assert resolved["generator"]["rw_baseline_std_range"] == [0, 0]
    assert resolved["generator"]["l_max"] == 8
    assert (
        StudyConfig.parse({"study": {"baseline": "varying"}}).resolved["generator"][
            "rw_baseline_std_range"
        ][1]
        > 0
    )
    for raw, message in (
        ({"generator": {"n_treatments_active_range": [0, 0]}}, "media"),
        ({"generator": {"edge_budget": {"cy": [0, 0]}}}, "media"),
        ({"generator": {"edge_budget": {"cy": [6, 15]}}}, "media"),
        (
            {"generator": {"n_covariates_active_range": [0, 0], "edge_budget": {"zy": [1, 1]}}},
            "control|covariates",
        ),
        ({"study": {"burn_in": 1}}, "burn_in"),
        ({"methods": {"enabled": [["ols"]]}}, "enabled"),
        ({"methods": {"advi": {"seeds": [[1]]}}}, "seeds"),
    ):
        with pytest.raises(ContractError, match=message):
            StudyConfig.parse(raw)


def test_receipt_reader_rejects_forged_inventory(tmp_path):
    from studies.recovery.artifacts import complete_checkpoint

    source = tmp_path / "source"
    source.write_text("x")
    ident = identity(
        "generate",
        StudyConfig.parse({}),
        source_files={"src": source},
        expected_outputs={"a", "b"},
    )
    stage = tmp_path / "stage"
    write_manifest(stage, ident)
    (stage / "a").write_bytes(b"a")
    (stage / "b").write_bytes(b"b")
    complete_checkpoint(stage, ident, {"a", "b"})
    assert checkpoint(stage, ident).complete
    receipt = stage / "completion.json"
    complete = json.loads(receipt.read_text())
    (stage / "b").unlink()
    for outputs in ({"a": digest(stage / "a")}, {}, None, ["a"]):
        receipt.write_text(json.dumps({"schema": 1, "outputs": outputs}))
        with pytest.raises(ContractError, match="inventory|receipt|outputs"):
            checkpoint(stage, ident)
    receipt.write_text(json.dumps({"schema": 1}))
    with pytest.raises(ContractError, match="receipt"):
        checkpoint(stage, ident)
    # A forged receipt must also fail when all files are present but the receipt
    # has unexpected entries, or the declared inventory is missing/malformed.
    (stage / "b").write_bytes(b"b")
    receipt.write_text(json.dumps({"schema": 1, "outputs": {**complete["outputs"], "c": "0" * 64}}))
    with pytest.raises(ContractError, match="inventory"):
        checkpoint(stage, ident)
    for inventory in (None, [], ["a", "a"], ["a", 1], "missing"):
        altered = {**ident, "expected_outputs": inventory}
        if inventory == "missing":
            del altered["expected_outputs"]
        (stage / "manifest.json").write_text(json.dumps(altered))
        with pytest.raises(ContractError, match="inventory"):
            checkpoint(stage, altered)


def test_legacy_tv_rejects_contradictory_manifest_metadata(tmp_path):
    from studies.recovery.analysis import _legacy_compatible

    original = ROOT / "notebooks"
    base = tmp_path / "ols_study_50_seed10029"
    tv = tmp_path / "time_varying_baseline"
    nuts = tv / "nuts"
    base.mkdir()
    nuts.mkdir(parents=True)
    (base / "manifest.json").write_bytes((original / base.name / "manifest.json").read_bytes())
    for batch in (original / base.name).glob("batch_*.npz"):
        (base / batch.name).symlink_to(batch)
    data_path = tv / "data_manifest.json"
    fit_path = nuts / "manifest.json"
    data = json.loads((original / tv.name / data_path.name).read_text())
    fit = json.loads((original / tv.name / "nuts" / fit_path.name).read_text())
    config = StudyConfig.parse({"study": {"baseline": "varying"}})

    def save(data_manifest, fit_manifest):
        data_path.write_text(json.dumps(data_manifest))
        fit_path.write_text(json.dumps(fit_manifest))

    save(data, fit)
    _legacy_compatible(nuts, "tv_nuts", config)
    for field, wrong in (
        ("smoothness_beta_prior", [4, 4]),
        ("max_smoothing_width", 3),
        ("confounding_strength", 1),
        ("baseline_floor", 2),
        ("seeds", [1] * 50),
    ):
        changed_data = {**data, field: wrong}
        save(changed_data, {**fit, "data_manifest": changed_data})
        with pytest.raises(ContractError, match=field):
            _legacy_compatible(nuts, "tv_nuts", config)
    missing_floor = {key: value for key, value in data.items() if key != "baseline_floor"}
    save(missing_floor, {**fit, "data_manifest": missing_floor})
    with pytest.raises(ContractError, match="baseline_floor"):
        _legacy_compatible(nuts, "tv_nuts", config)
    for wrong in ([92001, 123], [92001], list(range(92001, 92050)) + [92049]):
        save(data, {**fit, "seeds": wrong})
        with pytest.raises(ContractError, match="seed"):
            _legacy_compatible(nuts, "tv_nuts", config)


@pytest.mark.parametrize("kind", ["tv_nuts", "bounded_ols"])
def test_legacy_input_hash_inventory_is_complete(tmp_path, kind):
    from studies.recovery.analysis import _legacy_compatible

    original = ROOT / "notebooks"
    base = tmp_path / "ols_study_50_seed10029"
    base.mkdir()
    base_manifest = json.loads((original / base.name / "manifest.json").read_text())
    (base / "manifest.json").write_text(json.dumps(base_manifest))
    for batch in (original / base.name).glob("batch_*.npz"):
        (base / batch.name).symlink_to(batch)
    if kind == "tv_nuts":
        tv = tmp_path / "time_varying_baseline"
        saved = tv / "nuts"
        saved.mkdir(parents=True)
        original_data = json.loads((original / tv.name / "data_manifest.json").read_text())
        original_fit = json.loads((original / tv.name / "nuts/manifest.json").read_text())
        config = StudyConfig.parse({"study": {"baseline": "varying"}})
        hash_key = "source_sha256"

        def save_map(hashes):
            data = {**original_data, hash_key: hashes}
            (tv / "data_manifest.json").write_text(json.dumps(data))
            (saved / "manifest.json").write_text(
                json.dumps({**original_fit, "data_manifest": data})
            )

        full = original_data[hash_key]
    else:
        saved = tmp_path / "ols_study_50_seed10029_generator_bounds"
        saved.mkdir()
        original_fit = json.loads((original / saved.name / "manifest.json").read_text())
        config = StudyConfig.parse({})
        hash_key = "data_sha256"

        def save_map(hashes):
            (saved / "manifest.json").write_text(json.dumps({**original_fit, hash_key: hashes}))

        full = original_fit[hash_key]
    save_map(full)
    _legacy_compatible(saved, kind, config)  # authentic read-only data accepted
    subset = {"batch_00.npz": full["batch_00.npz"]}
    save_map(subset)
    with pytest.raises(ContractError, match="inventory"):
        _legacy_compatible(saved, kind, config)
    (base / "batch_01.npz").unlink()  # only disposable symlink removed
    (base / "batch_01.npz").write_bytes(b"corrupted")
    with pytest.raises(ContractError, match="inventory"):
        _legacy_compatible(saved, kind, config)  # exact review subset + unlisted corruption
    for hashes in (
        {**full, "batch_25.npz": "0" * 64},
        {**full, "oops": "0" * 64},
        {**full, "batch_01.npz": "not-a-sha256"},
        {},
        None,
    ):
        save_map(hashes)
        with pytest.raises(ContractError, match="inventory"):
            _legacy_compatible(saved, kind, config)
    save_map(full)
    with pytest.raises(ContractError, match="hash"):
        _legacy_compatible(saved, kind, config)  # full inventory detects corrupted input
    (base / "batch_01.npz").unlink()
    (base / "batch_01.npz").symlink_to(original / base.name / "batch_01.npz")
    _legacy_compatible(saved, kind, config)
    for field, bad in (("worlds", 49), ("cells_per_batch", 1), ("draws_per_cell", 2)):
        (base / "manifest.json").write_text(json.dumps({**base_manifest, field: bad}))
        with pytest.raises(ContractError, match=field):
            _legacy_compatible(saved, kind, config)


def test_review_repro_checkpoint_partial_and_tamper(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x")
    ident = identity(
        "generate",
        StudyConfig.parse({"study": {"worlds": 2}}),
        source_files={"source": source},
        expected_outputs={"world_00.npz", "world_01.npz"},
    )
    stage = tmp_path / "stage"
    assert checkpoint(stage, ident).state == "new"
    write_manifest(stage, ident)
    assert checkpoint(stage, ident).state == "initialized"
    (stage / "world_00.npz").write_bytes(b"first")
    from studies.recovery.artifacts import complete_checkpoint

    with pytest.raises(ContractError, match="inventory"):
        complete_checkpoint(stage, ident, {"world_00.npz"})
    with pytest.raises(ContractError, match="missing"):
        complete_checkpoint(stage, ident, {"world_00.npz", "world_01.npz"})
    assert checkpoint(stage, ident).state == "partial"
    (stage / "world_01.npz").write_bytes(b"second")
    complete_checkpoint(stage, ident, {"world_00.npz", "world_01.npz"})
    assert checkpoint(stage, ident).complete
    (stage / "world_00.npz").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="hash"):
        checkpoint(stage, ident)


def test_score_keeps_nulls_separate_from_nonzero_denominators():
    truth = np.array([[2.0, 1.0, 0.0, 0.2], [2.0, 1.0, 0.0, 0.2]])
    predicted = truth.copy()
    predicted[:, 1] = 0
    score = score_components(truth, predicted, ["Baseline", "C1", "C2", "Z1"])
    assert score["nonzero_media"] == ["C1"]
    assert not score["media_pass"] and score["controls_pass"]
    assert score["baseline"]["pass"]
    assert not score["joint_pass"]
    assert not score["total_media"]["pass"]
    with pytest.raises(ContractError):
        score_components(truth, predicted[:, :2], ["Baseline", "C1", "C2", "Z1"])


def test_saved_parity_full_50_read_only():
    tv = ROOT / "notebooks/time_varying_baseline/nuts"
    bounded = ROOT / "notebooks/ols_study_50_seed10029_generator_bounds"
    assert saved_summary(
        tv, kind="tv_nuts", config=StudyConfig.parse({"study": {"baseline": "varying"}})
    ) == {
        "kind": "tv_nuts",
        "provenance": "legacy manifest/input-hash compatibility only; output and source hashes unavailable",
        "worlds": 50,
        "valid": 50,
        "unresolved": 0,
        "media_pass": 11,
        "joint_pass": 9,
        "total_media_pass": 36,
        "media_channel_failures": 144,
        "media_channels_evaluable": 236,
        "control_channel_failures": 19,
        "control_channels_evaluable": 148,
    }
    assert (
        saved_summary(bounded, kind="bounded_ols", config=StudyConfig.parse({}))["media_pass"] == 13
    )
    assert saved_summary(bounded, kind="bounded_ols", config=StudyConfig.parse({}))["valid"] == 48
    for world in range(50):
        report = json.loads((tv / f"world_{world:02d}_nuts_report.json").read_text())
        gate = next(g for g in report["gates"] if g["relative"] == 0.1 and g["absolute"] == 0.01)
        score = saved_tv_world(tv, world)["score"]
        assert score["media_pass"] == gate["media_point_pass"]
        assert score["joint_pass"] == gate["individual_point_pass"]
        assert score["total_media"]["pass"] == gate["total_media_point_pass"]
        bounded_record = json.loads((bounded / f"world_{world:02d}.json").read_text())
        bounded_gate = next(
            g
            for g in bounded_record["fits"]["bounded_estimated"]["gates"]
            if g["relative"] == 0.1 and g["absolute"] == 0.01
        )
        bounded_score = saved_bounded_world(bounded, world)["score"]
        assert bounded_score["joint_pass"] == bounded_gate["error_gates_pass"]
