"""Artifact-free pipeline contracts. Synthetic receipts are NEVER a scientific fit."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from studies.recovery.artifacts import checkpoint, complete_checkpoint, write_manifest
from studies.recovery.config import ContractError, StudyConfig
from studies.recovery.reporting import _channels, _fit_labels, create_report, read_report
from studies.recovery.runner import _fit_identity, _world_identity, generate

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/recovery"


def test_complete_generator_config_and_capability_fail_before_outputs(tmp_path, monkeypatch):
    import copy

    from studies.recovery import runner

    config = StudyConfig.load(STUDY / "example_constant.toml")
    assert config.resolved["generator"]["n_time_steps"] == 32
    assert config.resolved["generator"]["rw_outcome_std_range"] == [0.015, 0.035]
    assert len(config.resolved["generator"]) > 100
    changed = {
        **config.resolved["generator"],
        "n_treatments": 3,
        "n_covariates": 2,
        "n_treatments_active_range": [2, 3],
        "n_covariates_active_range": [1, 2],
        "rw_outcome_std_range": [0.02, 0.04],
        "treatment_hf_sigma_range": [0.1, 0.3],
        "edge_budget": {**config.resolved["generator"]["edge_budget"], "cy": [1, 2], "zy": [0, 1]},
    }
    altered = StudyConfig.parse(
        {"study": config.resolved["study"], "generator": {"prior": changed}}
    )
    assert altered.resolved["generator"]["n_treatments"] == 3
    assert altered.resolved["generator"]["rw_outcome_std_range"] == [0.02, 0.04]
    assert _fit_labels(config, tuple(config.resolved["methods"]["enabled"])) == (
        ("ols", "ols", 0),
        ("advi_1", "advi", 0),
        ("advi_2", "advi", 1),
        ("mcmc", "mcmc", 0),
    )
    forbidden = ("carryover_family_probs", "edge_budget", "prior_conditioning", "baseline_floor")
    replacements = (
        {"geometric": 0.0, "none": 1.0, "weibull": 0.0},
        {**config.resolved["generator"]["edge_budget"], "dy": [1, 1]},
        True,
        0.0,
    )
    for key, replacement in zip(forbidden, replacements):
        raw = {
            "study": copy.deepcopy(config.resolved["study"]),
            "generator": {"prior": copy.deepcopy(config.resolved["generator"])},
        }
        raw["generator"]["prior"][key] = replacement
        with pytest.raises(ContractError):
            StudyConfig.parse(raw)
    raw["generator"]["prior"] = copy.deepcopy(config.resolved["generator"])
    raw["study"]["worlds"] = 4
    four = StudyConfig.parse(raw)
    stale = tmp_path / "generate/world_02"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        runner, "generate_pair", lambda *a, **kw: pytest.fail("generation before preflight")
    )
    with pytest.raises(ContractError, match="stale"):
        generate(four, tmp_path)
    assert not (tmp_path / "generate/world_00").exists()
    assert (
        StudyConfig.load(STUDY / "example_varying.toml").resolved["study"]["baseline"] == "varying"
    )
    raw["study"]["baseline"] = "varying"
    raw["methods"] = {"enabled": ["ols", "mcmc"]}
    raw["generator"]["prior"]["rw_baseline_std_range"] = [0, 0.04]
    with pytest.raises(ContractError, match="OLS"):
        StudyConfig.parse(raw)


def test_fit_preflights_all_worlds_before_any_optimization(tmp_path, monkeypatch):
    from studies.recovery import runner

    config, root = synthetic_ols_root(tmp_path)
    # Force a different selected fit label so world 0 has no output; world 1 is stale.
    broken = root / "fit/world_01/advi_1"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        runner,
        "_dataset",
        lambda path, cfg, world: (path / "generate" / f"world_{world:02d}", None, None),
    )
    monkeypatch.setattr(
        runner, "bayes_fit", lambda *a, **kw: pytest.fail("inference before preflight")
    )
    with pytest.raises(ContractError, match="stale"):
        runner.fit(config, root, "advi")
    assert not (root / "fit/world_00/advi_1").exists()


def test_channel_partition_and_numerical_status():
    rows = [
        {
            "kind": "media",
            "channel": "C1",
            "patterns": ["pulse", "trend", "trend_up"],
            "nonzero_effect": True,
            "computational_status": "valid",
            "point_pass": True,
        },
        {
            "kind": "media",
            "channel": "C2",
            "patterns": ["pulse", "trend", "trend_down"],
            "nonzero_effect": True,
            "computational_status": "valid",
            "point_pass": False,
        },
        {
            "kind": "media",
            "channel": "C3",
            "patterns": ["pulse"],
            "nonzero_effect": True,
            "computational_status": "unresolved",
            "point_pass": None,
        },
        {
            "kind": "control",
            "channel": "Z1",
            "patterns": ["pulse"],
            "nonzero_effect": False,
            "computational_status": "valid",
            "point_pass": None,
        },
    ]
    rates = {(r["kind"], r["pattern"]): r for r in _channels(rows)}
    assert rates["media", "pulse"]["total"] == 3
    assert rates["media", "pulse"]["evaluable"] == 2
    assert rates["media", "pulse"]["failed"] == 1
    assert rates["media", "pulse"]["unresolved"] == 1
    assert rates["media", "trend_up"]["failed"] == 0
    assert rates["media", "trend_down"]["failed"] == 1
    assert rates["control", "pulse"]["known_zero"] == 1
    assert rates["control", "pulse"]["failure_rate"] is None
    assert rates["control", "seasonal"]["status"] == "not represented"


def synthetic_ols_root(tmp_path, *, pattern_arrays=True, config=None, overlap=False):
    """Toy numbers + source-qualified synthetic receipts; does NOT run optimization."""
    config = config or StudyConfig.load(STUDY / "example_constant.toml")
    root = tmp_path / "SYNTHETIC-only-no-inference"
    for world in range(2):
        source = root / "generate" / f"world_{world:02d}"
        ident = _world_identity(config, world)
        write_manifest(source, ident)
        shape = (2, 32, 2)
        media = np.zeros(shape)
        media[:, :, 0] = 1.0
        flags = np.zeros((2, 2, 8), dtype="uint8")
        flags[:, :, 0] = 1
        changes = np.zeros((2, 2))
        if overlap:
            flags[:, :, 1] = 1  # pulse overlaps HF for every active input
            flags[0, :, 7] = 1  # both media trend-bearing in first cell
            flags[1, 0, 7] = 1  # first media trend-bearing in second cell
            changes[0, 0] = 0.25  # upward trend
            changes[0, 1] = 0.15
            changes[1, 0] = -0.2  # downward trend
        patterns = (
            {
                "treatment_components": flags,
                "covariate_components": flags[:, :1],
                "trajectory_treatment_trend_change": changes,
                "trajectory_covariate_trend_change": np.zeros((2, 1)),
            }
            if pattern_arrays
            else {}
        )
        np.savez_compressed(
            source / "dataset.npz",
            n_treatments_active=np.full(2, 2),
            n_covariates_active=np.full(2, 1),
            baseline_intrinsic=np.full((2, 32), 3.0),
            treatment_contribution_raw=media,
            covariate_contribution=np.zeros((2, 32, 1)),
            treatment_raw=media,
            outcome_noise=np.zeros((2, 32)),
            carryover_alpha=np.full((2, 2), 0.5),
            mm_kappa_mult=np.ones((2, 2)),
            beta=np.ones((2, 2)),
            rho_zy=np.zeros((2, 1)),
            **patterns,
        )
        (source / "metadata.json").write_text(
            json.dumps(
                {
                    "world_id": world,
                    "local_world": world,
                    "generation_seed": config.resolved["study"]["seed"],
                }
            )
        )
        complete_checkpoint(source, ident, set(ident["expected_outputs"]))
        fit = root / "fit" / f"world_{world:02d}" / "ols"
        fid = _fit_identity(config, root, world, "ols", 0)
        write_manifest(fit, fid)
        np.savez_compressed(
            fit / "components.npz", components=np.tile([3.0, 1.0, 0.0, 0.0], (25, 1))
        )
        (fit / "report.json").write_text(
            json.dumps(
                {
                    "world_id": world,
                    "replicate": 0,
                    "effective_seed": fid["method_seed"],
                    "success": True,
                    "SYNTHETIC_fixture_not_a_fit": True,
                }
            )
        )
        complete_checkpoint(fit, fid, set(fid["expected_outputs"]))
        assert checkpoint(fit, fid).complete  # existing complete stage reused, never refit
    return config, root


def test_pattern_labels_when_generator_omits_deterministic_metadata(tmp_path):
    config, source = synthetic_ols_root(tmp_path, pattern_arrays=False)
    report = tmp_path / "report"
    create_report(config, source, report, methods=("ols",))
    evidence = read_report(report)
    assert all(
        "hf" in row["patterns"] and "pulse" in row["patterns"] for row in evidence["channels"]
    )
    assert all("trend" not in row["patterns"] for row in evidence["channels"])
    assert evidence["pattern_rates"]["ols"][0]["total"] > 0
    from studies.recovery.analysis import score_components
    from studies.recovery.reporting import _pattern_rows

    fractional = {**config.resolved["generator"], "treatment_hf_inclusion_prob": 0.5}
    changed = StudyConfig.parse(
        {"study": config.resolved["study"], "generator": {"prior": fractional}}
    )
    truth = np.ones((25, 2))
    with pytest.raises(ContractError, match="missing required realized trajectory"):
        _pattern_rows(
            {},
            0,
            ["Baseline", "C1"],
            truth,
            score_components(truth, truth, ["Baseline", "C1"]),
            "valid",
            changed,
        )


def test_recorded_overlapping_pattern_flags_and_direction(tmp_path):
    base = StudyConfig.load(STUDY / "example_constant.toml")
    prior = {**base.resolved["generator"], "treatment_trend_inclusion_prob": 0.5}
    config = StudyConfig.parse(
        {
            "study": base.resolved["study"],
            "generator": {"prior": prior},
            "methods": base.resolved["methods"],
        }
    )
    _, source = synthetic_ols_root(tmp_path, config=config, overlap=True)
    out = tmp_path / "overlap-report"
    create_report(config, source, out, methods=("ols",))
    evidence = read_report(out)
    media = {
        (row["world_id"], row["channel"]): row
        for row in evidence["channels"]
        if row["kind"] == "media"
    }
    assert {"hf", "pulse", "trend", "trend_up"} <= set(media[0, "C1"]["patterns"])
    assert {"hf", "pulse", "trend", "trend_down"} <= set(media[1, "C1"]["patterns"])
    assert "trend" not in media[1, "C2"]["patterns"]
    rates = {
        row["pattern"]: row for row in evidence["pattern_rates"]["ols"] if row["kind"] == "media"
    }
    assert (rates["pulse"]["total"], rates["pulse"]["eligible"], rates["pulse"]["known_zero"]) == (
        4,
        2,
        2,
    )
    assert (
        rates["trend_up"]["total"],
        rates["trend_up"]["eligible"],
        rates["trend_up"]["known_zero"],
    ) == (2, 1, 1)
    assert (rates["trend_down"]["total"], rates["trend_down"]["eligible"]) == (1, 1)


def test_effectively_disabled_hf_and_missing_required_patterns_fail_before_report(tmp_path):
    base = StudyConfig.load(STUDY / "example_constant.toml")
    prior = {**base.resolved["generator"], "treatment_hf_sigma_range": [0.0, 0.0]}
    config = StudyConfig.parse(
        {
            "study": base.resolved["study"],
            "generator": {"prior": prior},
            "methods": base.resolved["methods"],
        }
    )
    _, source = synthetic_ols_root(tmp_path / "disabled", pattern_arrays=False, config=config)
    out = tmp_path / "disabled-report"
    create_report(config, source, out, methods=("ols",))
    evidence = read_report(out)
    for row in evidence["channels"]:
        if row["kind"] == "media":
            assert "hf" not in row["patterns"] and "pulse" in row["patterns"]
    rate = next(
        row
        for row in evidence["pattern_rates"]["ols"]
        if row["kind"] == "media" and row["pattern"] == "hf"
    )
    assert (rate["status"], rate["total"], rate["failure_rate"]) == ("not represented", 0, None)
    for label, change in (
        ("fractional", {"treatment_pulse_inclusion_prob": 0.5}),
        ("schedule", {"treatment_trend_inclusion_prob": 0.5}),
    ):
        modified = {**base.resolved["generator"], **change}
        request = StudyConfig.parse(
            {
                "study": base.resolved["study"],
                "generator": {"prior": modified},
                "methods": base.resolved["methods"],
            }
        )
        _, saved = synthetic_ols_root(tmp_path / label, pattern_arrays=False, config=request)
        refused = tmp_path / f"{label}-report"
        with pytest.raises(ContractError, match="missing required realized trajectory"):
            create_report(request, saved, refused, methods=("ols",))
        assert not refused.exists()


def test_public_native_analyze_report_and_preflight(tmp_path):
    config, root = synthetic_ols_root(tmp_path)
    cfg = STUDY / "example_constant.toml"
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "studies.recovery", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    common = ("--config", str(cfg), "--saved", str(root))
    missing = tmp_path / "missing"
    result = cli("report", *common, "--out", str(missing))
    assert result.returncode != 0 and "advi_1" in result.stderr and not missing.exists()
    result = cli("analyze", *common, "--report-method", "ols")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["world_method_rows"] == 2 and summary["valid_media_pass"] == 2
    report = tmp_path / "report"
    result = cli("report", *common, "--out", str(report), "--report-method", "ols")
    assert result.returncode == 0, result.stderr
    payload = read_report(report)
    assert payload["provenance"]["current_source_status"] == ["current-source qualified"]
    assert payload["report_methods"] == ["ols"] and len(payload["worlds"]) == 2
    for row in payload["worlds"]:
        truth = np.asarray(payload["trajectories"][f"{row['world_id']}:ols"]["truth"])
        assert row["profile"]["baseline_mean"] == pytest.approx(truth[:, 0].mean())
        assert row["profile"]["baseline_temporal_sd"] == pytest.approx(truth[:, 0].std())
    before = hashlib.sha256((root / "generate/world_00/dataset.npz").read_bytes()).hexdigest()
    assert "pending review" in (report / "interpretation.md").read_text()
    assert (
        before == hashlib.sha256((root / "generate/world_00/dataset.npz").read_bytes()).hexdigest()
    )
    assert "ipywidgets" not in (report / "report.ipynb").read_text()
    with pytest.raises(ContractError, match="stale"):
        changed = StudyConfig.parse(
            {
                "study": {**config.resolved["study"], "seed": 234234},
                "generator": {"prior": config.resolved["generator"]},
                "methods": config.resolved["methods"],
            }
        )
        create_report(changed, root, tmp_path / "stale", methods=("ols",))
    assert not (tmp_path / "stale").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "healthy_flags",
        "loss_false",
        "posterior_false",
        "loss_missing",
        "posterior_missing",
        "nonfinite_draw",
    ),
)
def test_advi_qualification_requires_both_finite_receipts_and_paths(tmp_path, fault, monkeypatch):
    """Synthetic NetCDF/receipts, NOT a fit; full native reader and ledger path."""
    import arviz as az

    from studies.recovery.reporting import analyze

    config, source = synthetic_ols_root(tmp_path)
    for world in range(2):
        for replicate in range(2):
            directory = source / "fit" / f"world_{world:02d}" / f"advi_{replicate + 1}"
            ident = _fit_identity(config, source, world, "advi", replicate)
            write_manifest(directory, ident)
            media = np.zeros((1, 4, 25, 2))
            media[..., 0] = 1.0
            affected = world == 0 and replicate == 0
            if affected and fault == "nonfinite_draw":
                media[0, 0, 0, 0] = np.nan
            posterior = az.from_dict(
                {
                    "posterior": {
                        "intercept": np.full((1, 4), 3.0),
                        "media_path": media,
                        "control_path": np.zeros((1, 4, 25, 1)),
                    }
                }
            )
            posterior.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
            posterior.close()
            flags = {"loss_finite": True, "posterior_finite": True}
            if affected and fault in ("loss_false", "posterior_false"):
                flags["loss_finite" if fault == "loss_false" else "posterior_finite"] = False
            if affected and fault in ("loss_missing", "posterior_missing"):
                flags.pop("loss_finite" if fault == "loss_missing" else "posterior_finite")
            (directory / "report.json").write_text(
                json.dumps(
                    {
                        "world_id": world,
                        "replicate": replicate,
                        "effective_seed": ident["method_seed"],
                        "SYNTHETIC_fixture_not_a_fit": True,
                        **flags,
                    }
                )
            )
            complete_checkpoint(directory, ident, set(ident["expected_outputs"]))
    assert analyze(config, source, ("advi",))["world_method_rows"] == 4
    out = tmp_path / "advi-report"
    create_report(config, source, out, methods=("advi",))
    evidence = read_report(out)
    row = next(r for r in evidence["worlds"] if r["world_id"] == 0 and r["method"] == "advi_1")
    assert row["diagnostics"]["finite_loss"] is (
        None if fault == "loss_missing" else fault != "loss_false"
    )
    assert row["diagnostics"]["posterior_finite"] is (
        None if fault == "posterior_missing" else fault != "posterior_false"
    )
    expected = "finite_only_not_converged" if fault == "healthy_flags" else "unresolved"
    assert row["computational_status"] == expected
    assert row["diagnostics"]["selected_draws_finite"] is (fault != "nonfinite_draw")
    assert all(value is None for value in row["recovery"].values())
    other_seed = next(
        r for r in evidence["worlds"] if r["world_id"] == 0 and r["method"] == "advi_2"
    )
    sensitivity = other_seed["diagnostics"]["max_interseed_channel_discrepancy_over_gate"]
    if fault == "healthy_flags":
        assert sensitivity == 0.0
    else:
        assert sensitivity is None
    channel = next(
        c
        for c in evidence["channels"]
        if c["world_id"] == 0 and c["method"] == "advi_1" and c["channel"] == "C1"
    )
    assert channel["computational_status"] == expected and channel["point_pass"] is None
    pattern = next(
        r
        for r in evidence["pattern_rates"]["advi_1"]
        if r["kind"] == "media" and r["pattern"] == "hf"
    )
    assert (
        pattern["total"],
        pattern["eligible"],
        pattern["known_zero"],
        pattern["unresolved"],
        pattern["evaluable"],
        pattern["failed"],
        pattern["failure_rate"],
    ) == (4, 2, 2, 2, 0, 0, None)
    if fault == "nonfinite_draw":
        import nbformat
        from nbclient import NotebookClient

        assert evidence["trajectories"]["0:advi_1"]["fitted"] is None
        assert row["diagnostics"]["selected_draws_finite"] is False
        assert row["uncertainty"]["channel_pointwise_coverage"] is None
        assert (
            next(r for r in evidence["worlds"] if r["world_id"] == 0 and r["method"] == "advi_2")[
                "diagnostics"
            ]["max_interseed_channel_discrepancy_over_gate"]
            is None
        )
        kernel = tmp_path / "kernel/kernels/recovery-pipeline"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Recovery",
                    "language": "python",
                }
            )
        )
        monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "kernel"))
        book = nbformat.read(out / "report.ipynb", as_version=4)
        executed = NotebookClient(
            book,
            timeout=60,
            kernel_name="recovery-pipeline",
            resources={"metadata": {"path": str(out)}},
        ).execute()
        outputs = [
            o
            for cell in executed.cells
            if cell.cell_type == "code"
            for o in cell.get("outputs", [])
        ]
        assert all(o.output_type != "error" for o in outputs)
        assert sum("image/png" in o.get("data", {}) for o in outputs) == 4
        assert "widgets" not in executed.metadata


def test_static_notebook_persists_normal_tables_and_png(tmp_path, monkeypatch):
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    _, source = synthetic_ols_root(tmp_path)
    report = tmp_path / "report"
    create_report(
        StudyConfig.load(STUDY / "example_constant.toml"), source, report, methods=("ols",)
    )
    spec_root = tmp_path / "kernel"
    kernel = spec_root / "kernels" / "recovery-pipeline"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Recovery isolated Python",
                "language": "python",
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(spec_root))
    assert "recovery-pipeline" in KernelSpecManager().find_kernel_specs()
    notebook = nbformat.read(report / "report.ipynb", as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="recovery-pipeline",
        resources={"metadata": {"path": str(report)}},
    ).execute()
    outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("text/html" in output.get("data", {}) for output in outputs) >= 5
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 2
    assert "widgets" not in executed.metadata
    assert (
        read_report(report)["run_id"]
        == json.loads((report / "evidence.json").read_text())["run_id"]
    )
