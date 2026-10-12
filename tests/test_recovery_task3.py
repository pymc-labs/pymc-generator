"""Artifact-only reporting contracts; no generating, sampling, or inference."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from studies.recovery.config import ContractError, StudyConfig
from studies.recovery.reporting import _channels, create_report, read_report

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/recovery"


def test_pattern_counts_belong_to_bearing_channel_not_world():
    def row(kind, name, patterns, nonzero, status, point):
        return {
            "kind": kind,
            "channel": name,
            "patterns": patterns,
            "nonzero_effect": nonzero,
            "computational_status": status,
            "point_pass": point,
        }

    rows = [
        row("media", "C1", ["pulse", "trend", "trend_up"], True, "valid", True),
        row("media", "C2", ["hf", "trend", "trend_down"], True, "valid", False),
        row("media", "C3", ["pulse", "trend", "trend_up"], True, "unresolved", None),
        row("control", "Z1", ["pulse", "trend", "trend_up"], True, "valid", False),
        row("control", "Z2", ["pulse"], False, "valid", None),
    ]
    rates = {(r["kind"], r["pattern"]): r for r in _channels(rows)}
    pulse = rates["media", "pulse"]
    assert (
        pulse["total"],
        pulse["eligible"],
        pulse["evaluable"],
        pulse["failed"],
        pulse["unresolved"],
        pulse["failure_rate"],
    ) == (2, 2, 1, 0, 1, 0)
    assert rates["media", "trend_up"]["failed"] == 0
    assert rates["media", "trend_down"]["failed"] == 1
    assert rates["media", "trend"]["failed"] == 1  # overlap, not disjoint buckets
    assert rates["control", "pulse"]["total"] == 2
    assert rates["control", "pulse"]["known_zero"] == 1
    assert rates["control", "pulse"]["failure_rate"] == 1
    assert rates["control", "seasonal"]["status"] == "not represented"
    assert rates["control", "seasonal"]["failure_rate"] is None
    assert all(row["small_count_warning"] for row in rates.values() if row["evaluable"])
    unresolved = _channels([row("media", "C1", ["pulse"], True, "unresolved", None)])
    assert (
        next(r for r in unresolved if r["kind"] == "media" and r["pattern"] == "pulse")["status"]
        == "unresolved"
    )


def _mock_bundle(monkeypatch, source):
    from studies.recovery import reporting

    def bundle(config, path, kind, *, historical, methods):
        assert (
            path == source and kind == "bounded_ols" and historical is False and methods == ("ols",)
        )
        input_path = source / "saved.bin"
        truth = np.array([[2.0, 1.0], [2.0, 1.0]])
        return (
            [
                {
                    "world_id": 0,
                    "computational_status": "valid",
                    "diagnostics": {"optimizer_converged": True},
                    "recovery": {
                        "media_pass": True,
                        "control_pass": True,
                        "joint_pass": True,
                        "total_media": {"pass": True},
                        "baseline": {"pass": True},
                    },
                    "uncertainty": {"type": "unavailable"},
                    "profile": {
                        "realized_noise_sd": 0.1,
                        "baseline_temporal_sd": 0.0,
                        "baseline_mean": 2.0,
                        "media_true_temporal_sd": [0.0],
                        "control_true_temporal_sd": [],
                        "carryover_alpha": [0.4],
                        "mm_kappa_mult": [0.7],
                        "media_beta": [1.0],
                        "control_coefficient": [],
                    },
                }
            ],
            [
                {
                    "world_id": 0,
                    "kind": "media",
                    "channel": "C1",
                    "patterns": ["hf"],
                    "nonzero_effect": True,
                    "computational_status": "valid",
                    "point_pass": True,
                    "point_error": 0.0,
                    "point_allowance": 0.3,
                }
            ],
            {
                "0": {
                    "terms": ["Baseline", "C1"],
                    "weeks": [7, 8],
                    "truth": truth.tolist(),
                    "fitted": truth.tolist(),
                    "lower95": None,
                    "upper95": None,
                    "raw_spend": [[1.0], [2.0]],
                }
            },
            {"input": input_path},
            "legacy manifest/input-hash compatibility only; output and source hashes unavailable",
        )

    monkeypatch.setattr(reporting, "_bundle", bundle)


def test_report_no_sample_source_write_relative_input_and_run_identity(tmp_path, monkeypatch):
    import pymc_generator as pg
    from studies.recovery import runner

    for module, name in ((runner, "generate"), (runner, "fit"), (pg, "sample_prior_predictive")):
        monkeypatch.setattr(
            module, name, lambda *a, **kw: pytest.fail("report initiated generation or fitting")
        )
    source = tmp_path / "source"
    source.mkdir()
    original = source / "saved.bin"
    original.write_bytes(b"immutable saved input")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    _mock_bundle(monkeypatch, source)
    config = StudyConfig.parse({"study": {"worlds": 1}})
    out = tmp_path / "bundle" / "report"
    result = create_report(config, source, "bounded_ols", out)
    assert read_report(out)["run_id"] == result["run_id"]
    assert not Path(
        json.loads((out / "evidence.json").read_text())["input_paths"]["input"]
    ).is_absolute()
    assert "pending review" in (out / "interpretation.md").read_text()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    with pytest.raises(ContractError, match="new, empty"):
        create_report(config, source, "bounded_ols", out)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    relocated = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "bundle", relocated / "bundle")
    shutil.copytree(source, relocated / "source")
    # Report identity is path-independent; stored inputs are relative to report root.
    copy = relocated / "bundle" / "report"
    # Preserve the source location relative to the moved report bundle.
    payload = json.loads((copy / "evidence.json").read_text())
    actual = (copy / payload["input_paths"]["input"]).resolve()
    actual.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, actual)
    assert read_report(copy)["run_id"] == result["run_id"]
    changed_config = StudyConfig.parse({"study": {"worlds": 1}, "score": {"relative": 0.2}})
    changed = create_report(changed_config, source, "bounded_ols", tmp_path / "changed-config")
    assert changed["run_id"] != result["run_id"]
    assert "pending review" in (tmp_path / "changed-config/interpretation.md").read_text()
    original.write_bytes(b"modified saved input")
    with pytest.raises(ContractError, match="stale"):
        read_report(out)
    assert read_report(copy)["run_id"] == result["run_id"]
    (copy / "interpretation.md").write_text("old conclusion from a different run\n")
    with pytest.raises(ContractError, match="hash"):
        read_report(copy)


def _assert_profiles_match_scored_truth(evidence):
    for row in evidence["worlds"]:
        key = str(row["world_id"])
        if "method" in row:
            key += ":" + row["method"]
        path = evidence["trajectories"][key]
        truth = np.asarray(path["truth"], dtype=float)
        names = path["terms"]
        n_media = sum(name.startswith("C") for name in names)
        profile = row["profile"]
        assert profile["baseline_mean"] == pytest.approx(truth[:, 0].mean(), abs=1e-12)
        assert profile["baseline_temporal_sd"] == pytest.approx(truth[:, 0].std(), abs=1e-12)
        assert profile["media_true_temporal_sd"] == pytest.approx(
            truth[:, 1 : 1 + n_media].std(axis=0), abs=1e-12
        )
        assert profile["control_true_temporal_sd"] == pytest.approx(
            truth[:, 1 + n_media :].std(axis=0), abs=1e-12
        )


@pytest.mark.parametrize("baseline,record_count", [("constant", 8), ("varying", 6)])
def test_historical_smoke_requires_recorded_receipts_and_never_normalizes_stale(
    tmp_path, monkeypatch, baseline, record_count
):
    from studies.recovery import runner

    config = StudyConfig.load(STUDY / f"smoke_{baseline}.toml")
    source = STUDY / "smoke-artifacts-final" / baseline
    marker = source / "fit/world_00/mcmc/completion.json"
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    out = tmp_path / "historical"
    with pytest.raises(ContractError, match="--historical"):
        create_report(config, source, "smoke", out)
    assert not out.exists()
    monkeypatch.setattr(runner, "fit", lambda *a, **kw: pytest.fail("fit called"))
    monkeypatch.setattr(runner, "generate", lambda *a, **kw: pytest.fail("generate called"))
    result = create_report(config, source, "smoke", out, historical=True)
    assert result["world_records"] == record_count
    evidence = read_report(out)
    _assert_profiles_match_scored_truth(evidence)  # all four worlds and selected methods
    for row in evidence["worlds"]:
        world = row["world_id"]
        with np.load(source / f"generate/world_{world:02d}/dataset.npz") as corpus:
            local = world % 2
            start, stop = config.resolved["study"]["score_weeks"]
            media = int(corpus["n_treatments_active"][local])
            assert row["profile"]["realized_noise_sd"] == pytest.approx(
                corpus["outcome_noise"][local, start : stop + 1].std()
            )
            assert row["profile"]["media_beta"] == pytest.approx(corpus["beta"][local, :media])
    assert evidence["provenance"]["current_source_status"] == ["stale for current-source fitting"]
    assert all(
        row["computational_status"] == "unresolved"
        for row in evidence["worlds"]
        if row["method"] == "mcmc"
    )
    assert all(
        row["diagnostics"]["rhat_ess"] == "not applicable"
        for row in evidence["worlds"]
        if row["method"].startswith("advi")
    )
    assert all(
        "max_interseed_channel_discrepancy_over_gate" in row["diagnostics"]
        for row in evidence["worlds"]
        if row["method"].startswith("advi")
    )
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == before


def test_legacy_science_reports_are_full_population_and_provenance_limited(tmp_path, monkeypatch):
    import pymc_generator as pg

    monkeypatch.setattr(
        pg, "sample_prior_predictive", lambda *a, **kw: pytest.fail("legacy report sampled")
    )
    for kind, path, config, expected_valid, expected_media in (
        (
            "tv_nuts",
            ROOT / "notebooks/time_varying_baseline/nuts",
            STUDY / "saved_tv_example.toml",
            50,
            11,
        ),
        (
            "bounded_ols",
            ROOT / "notebooks/ols_study_50_seed10029_generator_bounds",
            STUDY / "saved_bounded_example.toml",
            48,
            13,
        ),
    ):
        out = tmp_path / kind
        create_report(StudyConfig.load(config), path, kind, out)
        evidence = read_report(out)
        assert (
            evidence["provenance"]
            == "legacy manifest/input-hash compatibility only; output and source hashes unavailable"
        )
        assert len(evidence["worlds"]) == 50
        _assert_profiles_match_scored_truth(evidence)  # all 50 TV and constant-baseline worlds
        original = ROOT / "notebooks/ols_study_50_seed10029"
        for world in range(50):
            batch, local = divmod(world, 2)
            corpus = pg.load_corpus(original / f"batch_{batch:02d}.npz")
            row = evidence["worlds"][world]
            profile = row["profile"]
            n_media, n_controls = (len(profile["media_beta"]), len(profile["control_coefficient"]))
            assert profile["realized_noise_sd"] == pytest.approx(
                np.asarray(corpus["outcome_noise"][local, 7:], float).std(), abs=1e-12
            )
            for label, key, count in (
                ("carryover_alpha", "carryover_alpha", n_media),
                ("mm_kappa_mult", "mm_kappa_mult", n_media),
                ("media_beta", "beta", n_media),
                ("control_coefficient", "rho_zy", n_controls),
            ):
                assert profile[label] == pytest.approx(corpus[key][local, :count], abs=1e-12)
            if kind == "tv_nuts":
                with np.load(path / f"world_{world:02d}_nuts_recovery.npz") as saved:
                    residual = saved["observed"] - saved["truth"].sum(axis=1)
                assert profile["realized_noise_sd"] == pytest.approx(residual.std(), abs=1e-6)
                assert profile["baseline_temporal_sd"] > 0
            else:
                assert profile["baseline_temporal_sd"] == pytest.approx(0)
        assert (
            sum(row["computational_status"] == "valid" for row in evidence["worlds"])
            == expected_valid
        )
        assert (
            sum(row["recovery"]["media_pass"] is True for row in evidence["worlds"])
            == expected_media
        )
        assert len({row["world_id"] for row in evidence["channels"]}) == 50
        for rate in evidence["pattern_rates"][kind]:
            bearing = [
                r
                for r in evidence["channels"]
                if r["kind"] == rate["kind"] and rate["pattern"] in r["patterns"]
            ]
            assert rate["failed"] == sum(
                r["point_pass"] is False for r in bearing if r["nonzero_effect"]
            )
            assert rate["evaluable"] + rate["unresolved"] == rate["eligible"]


def test_review_transition_is_explicit_run_bound_and_separate(tmp_path, monkeypatch):
    import subprocess

    from studies.recovery.review_interpretation import attached_review, create_review, read_review

    source = tmp_path / "source"
    source.mkdir()
    (source / "saved.bin").write_bytes(b"SYNTHETIC review test")
    _mock_bundle(monkeypatch, source)
    config = StudyConfig.parse({"study": {"worlds": 1}})
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = create_report(config, source, "bounded_ols", first)
    create_report(
        StudyConfig.parse({"study": {"worlds": 1}, "score": {"relative": 0.2}}),
        source,
        "bounded_ols",
        second,
    )
    assert attached_review(first) is None
    receipt_before = hashlib.sha256((first / "completion.json").read_bytes()).hexdigest()
    text = tmp_path / "test-only-claim.md"
    text.write_text(
        "# Test-only approved interpretation\nThis is a SYNTHETIC fixture, not science.\n"
    )
    with pytest.raises(ContractError, match="run ID"):
        create_review(first, text, "TEST-ONLY-reviewer", "wrong-run")
    assert not (tmp_path / "first-review").exists()
    cli = subprocess.run(
        [
            str(STUDY / ".venv/bin/python"),
            "-m",
            "studies.recovery.review_interpretation",
            "--report",
            str(first),
            "--interpretation",
            str(text),
            "--reviewer",
            "TEST-ONLY-reviewer",
            "--approve-run",
            one["run_id"],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    review = read_review(first, tmp_path / "first-review")
    assert review["decision"] == "approved" and review["report_run_id"] == one["run_id"]
    assert attached_review(first)["reviewer"] == "TEST-ONLY-reviewer"
    import nbformat
    from nbclient import NotebookClient

    monkeypatch.setenv("JUPYTER_PATH", str(STUDY / ".venv/share/jupyter"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    executed = NotebookClient(
        nbformat.read(first / "report.ipynb", as_version=4),
        timeout=90,
        kernel_name="recovery-study",
        resources={"metadata": {"path": str(first)}},
    ).execute()
    review_outputs = str(executed.cells[-1].get("outputs", []))
    assert "TEST-ONLY-reviewer" in review_outputs and one["run_id"] in review_outputs
    assert read_report(first)["run_id"] == one["run_id"]
    assert hashlib.sha256((first / "completion.json").read_bytes()).hexdigest() == receipt_before
    with pytest.raises(ContractError, match="new canonical"):
        create_review(first, text, "TEST-ONLY-reviewer", one["run_id"])
    wrong = tmp_path / "second-review"
    shutil.copytree(tmp_path / "first-review", wrong)
    with pytest.raises(ContractError, match="review identity|canonical"):
        read_review(second, wrong)
    with pytest.raises(ContractError):
        attached_review(second)
    (wrong / "interpretation.md").write_text("edited but not approved\n")
    with pytest.raises(ContractError):
        read_review(second, wrong)
    (tmp_path / "first-review/interpretation.md").write_text("tampered\n")
    with pytest.raises(ContractError, match="review identity"):
        attached_review(first)
    assert read_report(first)["run_id"] == one["run_id"]  # sealed evidence not touched


@pytest.mark.parametrize("world_count", [2, 4])
def test_native_current_identity_method_subset_and_three_advi_seeds(tmp_path, world_count):
    """SYNTHETIC toy bundle only: no real historical fit is re-labelled current."""
    import subprocess

    from studies.recovery.artifacts import complete_checkpoint, write_manifest
    from studies.recovery.reporting import _current_fit_identity, _fit_labels
    from studies.recovery.runner import _world_identity

    config_text = (
        f"[study]\nworlds={world_count}\nseed=12\nbaseline='constant'\n"
        "[generator]\nn_treatments=1\nn_covariates=1\n"
        "n_treatments_active_range=[1,1]\nn_covariates_active_range=[1,1]\n"
        "[generator.edge_budget]\ncy=[1,1]\nzy=[0,0]\n"
        "[methods]\nenabled=['ols','advi','mcmc']\n"
        "[methods.advi]\nseeds=[101,102,103]\n"
    )
    config_path = tmp_path / "synthetic-only.toml"
    config_path.write_text(config_text)
    config = StudyConfig.load(config_path)
    root = tmp_path / "SYNTHETIC-not-scientific"
    for world in range(world_count):
        directory = root / "generate" / f"world_{world:02d}"
        ident = _world_identity(config, world)
        write_manifest(directory, ident)
        shape = (2, 104, 1)
        media = np.ones(shape)
        controls = np.zeros(shape)
        flags = np.zeros((2, 1, 8), dtype="uint8")
        flags[:, :, 0] = 1
        np.savez_compressed(
            directory / "dataset.npz",
            n_treatments_active=np.ones(2, dtype=int),
            n_covariates_active=np.ones(2, dtype=int),
            baseline_intrinsic=np.full((2, 104), 2.0),
            treatment_contribution_raw=media,
            covariate_contribution=controls,
            treatment_raw=media,
            outcome_noise=np.zeros((2, 104)),
            carryover_alpha=np.full((2, 1), 0.5),
            mm_kappa_mult=np.full((2, 1), 0.8),
            beta=np.ones((2, 1)),
            rho_zy=np.zeros((2, 1)),
            treatment_components=flags,
            covariate_components=flags,
            trajectory_treatment_trend_change=np.zeros((2, 1)),
            trajectory_covariate_trend_change=np.zeros((2, 1)),
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                {"world_id": world, "local_world": world % 2, "generation_seed": 12 + world // 2}
            )
        )
        complete_checkpoint(directory, ident, set(ident["expected_outputs"]))
        fit = root / "fit" / f"world_{world:02d}" / "ols"
        ident_fit = _current_fit_identity(config, root, world, "ols", 0)
        write_manifest(fit, ident_fit)
        components = np.tile([2.0, 1.0, 0.0], (97, 1))
        np.savez_compressed(fit / "components.npz", components=components)
        (fit / "report.json").write_text(
            json.dumps(
                {
                    "world_id": world,
                    "replicate": 0,
                    "effective_seed": ident_fit["method_seed"],
                    "success": True,
                    "synthetic_fixture_not_a_fit": True,
                }
            )
        )
        complete_checkpoint(fit, ident_fit, set(ident_fit["expected_outputs"]))

    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def cli(out, *extra):
        return subprocess.run(
            [
                str(STUDY / ".venv/bin/python"),
                "-m",
                "studies.recovery",
                "report",
                "--config",
                str(config_path),
                "--saved",
                str(root),
                "--format",
                "native",
                "--out",
                str(out),
                *extra,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    selected = tmp_path / "selected"
    result = cli(selected, "--report-method", "ols")
    assert result.returncode == 0, result.stderr
    evidence = read_report(selected)
    _assert_profiles_match_scored_truth(evidence)  # current native, 2/4 synthetic worlds
    assert evidence["report_methods"] == ["ols"] and len(evidence["worlds"]) == world_count
    assert evidence["provenance"]["current_source_status"] == ["current-source qualified"]
    assert all(
        r["method"] == "ols" and r["recovery"]["media_pass"] is True for r in evidence["worlds"]
    )
    assert _fit_labels(config, ("advi",)) == (
        ("advi_1", "advi", 0),
        ("advi_2", "advi", 1),
        ("advi_3", "advi", 2),
    )
    incomplete = tmp_path / "incomplete"
    result = cli(incomplete)
    assert result.returncode != 0 and "advi_1" in result.stderr and not incomplete.exists()
    for world in range(world_count):
        for replicate in (0, 1):
            name = f"advi_{replicate + 1}"
            fit = root / "fit" / f"world_{world:02d}" / name
            ident_fit = _current_fit_identity(config, root, world, "advi", replicate)
            write_manifest(fit, ident_fit)
            (fit / "posterior.nc").write_bytes(b"SYNTHETIC placeholder, not inference")
            (fit / "report.json").write_text(
                json.dumps(
                    {
                        "world_id": world,
                        "replicate": replicate,
                        "effective_seed": ident_fit["method_seed"],
                        "loss_finite": True,
                        "posterior_finite": True,
                        "synthetic_fixture_not_a_fit": True,
                    }
                )
            )
            complete_checkpoint(fit, ident_fit, set(ident_fit["expected_outputs"]))
    third = tmp_path / "missing-third"
    result = cli(third, "--report-method", "advi")
    assert result.returncode != 0 and "advi_3" in result.stderr and not third.exists()
    wrong = tmp_path / "historical-not-native"
    result = cli(wrong, "--report-method", "ols", "--historical")
    assert result.returncode != 0 and not wrong.exists()
    modified = tmp_path / "wrong-config"
    changed = config_text.replace("seed=12", "seed=13")
    config_path.write_text(changed)
    result = cli(modified, "--report-method", "ols")
    assert result.returncode != 0 and "stale" in result.stderr and not modified.exists()
    assert read_report(selected)["run_id"] == evidence["run_id"]


def test_report_notebook_run_all_saved_only(tmp_path, monkeypatch):
    import nbformat
    from nbclient import NotebookClient

    from studies.recovery import runner

    source = tmp_path / "source"
    source.mkdir()
    (source / "saved.bin").write_bytes(b"artifact")
    _mock_bundle(monkeypatch, source)
    from studies.recovery import reporting

    original_bundle = reporting._bundle

    def two_views(*args, **kwargs):
        rows, channels, paths, inputs, provenance = original_bundle(*args, **kwargs)
        # SYNTHETIC selector stress: second world has a known-null C1, nonzero C2.
        paths["1"] = {
            "terms": ["Baseline", "C1", "C2"],
            "weeks": [7, 8],
            "truth": [[3.0, 0.0, 2.0], [3.0, 0.0, 2.0]],
            "fitted": [[3.0, 0.0, 2.0], [3.0, 0.0, 2.0]],
            "lower95": None,
            "upper95": None,
            "raw_spend": [[1.0, 5.0], [2.0, 6.0]],
        }
        rows.append({**rows[0], "world_id": 1})
        return rows, channels, paths, inputs, provenance

    monkeypatch.setattr(reporting, "_bundle", two_views)
    out = tmp_path / "report"
    create_report(StudyConfig.parse({"study": {"worlds": 1}}), source, "bounded_ols", out)
    book = nbformat.read(out / "report.ipynb", as_version=4)
    book.cells.append(
        nbformat.v4.new_code_cell(
            "assert selection.value == '0' and channel.value == 'Baseline'\n"
            "assert any('application/vnd.plotly.v1+json' in o.get('data', {}) for o in interactive.outputs)\n"
            "assert not any(o.get('output_type') == 'error' for o in interactive.outputs)\n"
            "assert any(t.name == 'True baseline' for t in _view(selection.value, channel.value).data)\n"
            "selection.value = '1'\n"
            "assert channel.value == 'Baseline' and list(channel.options) == ['Baseline', 'C2']\n"
            "channel.value = 'C2'\n"
            "figure = _view(selection.value, channel.value)\n"
            "assert figure is not None and any(t.name == 'Raw spend C2' and list(t.y) == [5., 6.] for t in figure.data)\n"
            "assert any(t.name == 'True baseline' for t in figure.data)\n"
            "assert 'C1' not in channel.options\n"
            "selection.value = '0'\n"
            "assert 'C1' in channel.options and channel.value == 'Baseline'\n"
            "channel.value = 'C1'\n"
            "assert any(t.name == 'Raw spend C1' for t in _view(selection.value, channel.value).data)"
        )
    )
    code = "\n".join(c.source for c in book.cells if c.cell_type == "code")
    assert "ols_reporting" not in code and "sample_prior_predictive" not in code
    assert "runner" not in code and "bayes_fit" not in code and "bounded_ols(" not in code
    digest_before = hashlib.sha256((source / "saved.bin").read_bytes()).hexdigest()
    kernel_dir = STUDY / ".venv/share/jupyter"
    monkeypatch.setenv("JUPYTER_PATH", str(kernel_dir))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(runner, "fit", lambda *a, **kw: pytest.fail("fit called"))
    executed = NotebookClient(
        book, timeout=90, kernel_name="recovery-study", resources={"metadata": {"path": str(out)}}
    ).execute()
    assert (
        sum(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")
        >= 6
    )
    assert not any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    state = executed.metadata.widgets["application/vnd.jupyter.widget-state+json"]["state"]
    outputs = [
        model["state"].get("outputs", [])
        for model in state.values()
        if model.get("model_name") == "OutputModel"
    ]
    assert outputs and not any(
        entry.get("output_type") == "error" for group in outputs for entry in group
    )
    assert any(
        "application/vnd.plotly.v1+json" in entry.get("data", {})
        for group in outputs
        for entry in group
    )
    dropdowns = [
        model["state"] for model in state.values() if model.get("model_name") == "DropdownModel"
    ]
    # ipywidgets 8 synchronizes Dropdown.index; serialized `value` is None by design.
    assert len(dropdowns) == 2 and all(model.get("index") is not None for model in dropdowns)
    assert any("0" in model.get("_options_labels", []) for model in dropdowns)
    assert any("C1" in model.get("_options_labels", []) for model in dropdowns)
    assert hashlib.sha256((source / "saved.bin").read_bytes()).hexdigest() == digest_before
    assert read_report(out)["run_id"] == json.loads((out / "evidence.json").read_text())["run_id"]
