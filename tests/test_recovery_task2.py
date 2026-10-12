"""Task 2 contracts: only reuse recorded smoke/fixed-RV fixtures; never sample or fit in tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import pymc_generator as pg
from studies.recovery.artifacts import checkpoint
from studies.recovery.config import ContractError, StudyConfig
from studies.recovery.dgp import prior, validate_world, world_data
from studies.recovery.models import baseline_geometry, build_model
from studies.recovery.runner import _world_identity, fit, generate

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/recovery"
SMOKE = STUDY / "smoke-artifacts-final"


def _smoke(label):
    root = SMOKE / label
    if not (root / "generate/world_00/completion.json").exists():
        pytest.skip("four-world smoke not present; no test may create extra worlds")
    return StudyConfig.load(STUDY / f"smoke_{label}.toml"), root


def _historical_dataset(root, config, world):
    """Verify the sealed FROZEN-SOURCE fixture; never claim current-source completion."""
    snapshot = json.loads((SMOKE / "source-snapshot/inventory.json").read_text())
    for name in ("dgp.py", "models.py", "methods.py", "runner.py"):
        relative = f"studies/recovery/{name}"
        assert (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            == snapshot["files"][relative]["sha256"]
        )
    directory = root / "generate" / f"world_{world:02d}"
    recorded = json.loads((directory / "manifest.json").read_text())
    assert recorded["config"] == {
        key: config.resolved[key] for key in ("study", "generator", "pairing")
    }
    assert checkpoint(directory, recorded).complete  # sealed historical output hashes only
    corpus = pg.load_corpus(directory / "dataset.npz")
    cfg = prior(config, seed=config.resolved["study"]["seed"] + world // 2)
    validate_world(corpus, world % 2, cfg)
    return directory, corpus, cfg


def test_config_capabilities_and_generator_support():
    for raw, message in (
        ({"study": {"baseline": "varying"}, "methods": {"enabled": ["ols"]}}, "OLS"),
        ({"generator": {"n_covariates_active_range": [0, 0]}}, "n_covariates_active_range"),
        ({"generator": {"n_covariates": 0}}, "n_covariates"),
        ({"generator": {"edge_budget": {"cc": [1, 1]}}}, "direct"),
    ):
        with pytest.raises(ContractError, match=message):
            StudyConfig.parse(raw)
    for label in ("constant", "varying"):
        config = StudyConfig.load(STUDY / f"smoke_{label}.toml")
        cfg = prior(config, seed=config.resolved["study"]["seed"])
        assert cfg.n_latent == 1 and cfg.mm_scale_prior == "log_uniform"
        assert cfg.outcome_std_mode == "relative" and cfg.treatment_reference_multiplier == 1
        assert cfg.l_max == cfg.carryover_burn_in == 8
        assert cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] == (
            0.05,
            20.0,
        ) or list(cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]) == [0.05, 20.0]
        assert cfg.n_treatments == 3 and cfg.n_covariates == 2


def test_score_window_preflight_is_the_full_supported_suffix():
    for weeks in ([7, 20], [8, 103], [0, 103]):
        with pytest.raises(ContractError, match="score_weeks"):
            StudyConfig.parse({"study": {"score_weeks": weeks}})
    assert StudyConfig.parse({}).resolved["study"]["score_weeks"] == [7, 103]
    shorter = StudyConfig.parse(
        {
            "study": {"reported_weeks": [0, 20], "score_weeks": [7, 20]},
            "generator": {"n_time_steps": 21},
        }
    )
    assert shorter.resolved["study"]["score_weeks"] == [7, 20]
    with pytest.raises(ContractError, match="media_smoothness_beta"):
        StudyConfig.parse({"pairing": {"mode": "independent", "media_smoothness_beta": 4}})
    with pytest.raises(ContractError, match="media_smoothness_beta"):
        StudyConfig.parse({"generator": {"media_smoothness_beta": 4}})
    assert (
        StudyConfig.load(STUDY / "matched_existing_example.toml").resolved["pairing"]["mode"]
        == "matched"
    )


def test_wrong_score_window_never_starts_cli_execution(tmp_path, monkeypatch):
    from studies.recovery.__main__ import main

    invalid = tmp_path / "wrong.toml"
    invalid.write_text("[study]\nworlds = 2\nscore_weeks = [7, 20]\n")
    target = tmp_path / "untouched"
    for stage in ("generate", "fit"):
        monkeypatch.setattr(
            sys, "argv", ["recovery", stage, "--config", str(invalid), "--out", str(target)]
        )
        with pytest.raises(ContractError, match="score_weeks"):
            main()
        assert not target.exists()


def test_matched_fixture_contract_and_source_inventory(tmp_path, monkeypatch):
    import studies.recovery.matched as matched

    source = ROOT / "notebooks/ols_study_50_seed10029"
    paired = ROOT / "notebooks/media_smoothing_comparison"
    good = StudyConfig.load(STUDY / "matched_existing_example.toml")
    raw = {
        "study": {"worlds": 2, "seed": 10029},
        "generator": {"n_treatments": 3},
        "pairing": good.resolved["pairing"],
    }
    with pytest.raises(ContractError, match="n_treatments|generator"):
        matched.replay_world(StudyConfig.parse(raw), source, paired, 0)
    for mutation in (
        {"n_covariates": 2},
        {"n_treatments_active_range": [3, 15]},
        {"rw_baseline_std_range": [0, 0.01]},
    ):
        with pytest.raises(ContractError, match="generator|baseline"):
            matched.replay_world(
                StudyConfig.parse(
                    {
                        "study": {"worlds": 2, "seed": 10029},
                        "generator": mutation,
                        "pairing": good.resolved["pairing"],
                    }
                ),
                source,
                paired,
                0,
            )
    original = matched._replay_identity(good, source, paired, 0)
    for name in (
        "config.py",
        "artifacts.py",
        "runner.py",
        "matched.py",
        "generator/world_model.py",
        "generator/sampler.py",
        "generator/slots.py",
        "generator/mechanisms.py",
        "generator/random_walk.py",
    ):
        assert name in original["sources"]
    replacement = tmp_path / "world_model.py"
    replacement.write_bytes((ROOT / "pymc_generator/world_model.py").read_bytes())
    sources = matched._sources
    monkeypatch.setattr(
        matched,
        "_sources",
        lambda *names: {**sources(*names), "generator/world_model.py": replacement},
    )
    assert matched._replay_identity(good, source, paired, 0)["sources"] == original["sources"]
    replacement.write_text(replacement.read_text() + "\n# deliberate isolated evaluator change\n")
    assert (
        matched._replay_identity(good, source, paired, 0)["sources"]["generator/world_model.py"]
        != original["sources"]["generator/world_model.py"]
    )


def test_oracle_inputs_do_not_include_realized_truth_or_initialize_from_it(monkeypatch):
    cfg, root = _smoke("constant")
    _, corpus, prior_cfg = _historical_dataset(root, cfg, 0)
    data = world_data(corpus, 0, prior_cfg, 7)
    altered = {
        key: val.copy() if isinstance(val, np.ndarray) else val for key, val in corpus.items()
    }
    for key in ("beta", "rho_zy", "carryover_alpha", "mm_kappa_mult"):
        altered[key] = np.zeros_like(corpus[key])
    from studies.recovery import dgp

    # Deliberately corrupted ground truth must not enter oracle input assembly;
    # generator corpus validation is tested on the authentic fixture above.
    monkeypatch.setattr(dgp, "validate_world", lambda *args: None)
    changed = world_data(altered, 0, prior_cfg, 7)
    for key in data:
        np.testing.assert_array_equal(data[key], changed[key])
    assert set(data) == {
        "media_data",
        "control_data",
        "media_scale",
        "control_scale",
        "media_mask",
        "control_mask",
        "media_effect_mask",
        "control_effect_mask",
        "saturation_anchor",
        "outcome_scale",
        "outcome_data",
    }
    model, _ = build_model(data, prior_cfg, start=7, varying=False)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()
    assert "outcome" in model.named_vars


def test_zero_effect_controls_and_scoring_window():
    cfg, root = _smoke("varying")
    _, corpus, prior_cfg = _historical_dataset(root, cfg, 0)
    data = world_data(corpus, 0, prior_cfg, 7)
    assert data["control_effect_mask"].sum() == 0  # actual fixture with active controls, no effect
    assert data["control_mask"].sum() >= 1
    assert len(data["outcome_data"]) == 97
    np.testing.assert_allclose(
        data["outcome_data"] * data["outcome_scale"], corpus["outcome_raw"][0, 7:]
    )
    model, geometry = build_model(data, prior_cfg, start=7, varying=True)
    assert "outcome" not in model.named_vars  # exact path/width marginalized, not pointwise Normal
    assert len(geometry[0]) == 26 and geometry[2].shape == (26, 97)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()
    # Generator's full 112-week walk; first 15 prior weeks excluded only from likelihood.
    widths, probs, eigen, vectors, _ = baseline_geometry(104, 8, 7, 26)
    from pymc_generator.random_walk import _walk_basis

    for i in (0, 12, 25):
        basis = _walk_basis(112, int(widths[i]))[15:]
        np.testing.assert_allclose(
            (vectors[i] * eigen[i]) @ vectors[i].T, basis @ basis.T, rtol=1e-10, atol=1e-10
        )
    assert np.isclose(probs.sum(), 1)


def test_transform_support_matches_saved_generated_truth_without_truth_fit_initialization():
    from pymc_generator.slots import SlotLayout
    from studies.recovery.methods import media_transform

    cfg, root = _smoke("constant")
    _, corpus, prior_cfg = _historical_dataset(root, cfg, 0)
    graph = SlotLayout(prior_cfg.n_treatments, prior_cfg.n_covariates, 1).unpack(corpus["g"][0])
    for channel in range(prior_cfg.n_treatments):
        if not corpus["treatment_active_mask"][0, channel]:
            continue
        alpha = float(corpus["carryover_alpha"][0, channel])
        half = float(
            corpus["mm_kappa_mult"][0, channel] * corpus["mechanism_saturation_scale"][0, channel]
        )
        beta = float(corpus["beta"][0, channel]) * bool(graph["cy"][channel])
        expected = (
            media_transform(prior_cfg.l_max)(
                corpus["treatment_raw"][0, :, channel].astype(float), alpha, half
            )
            * beta
        )
        np.testing.assert_allclose(
            expected[7:], corpus["treatment_contribution_raw"][0, 7:, channel], rtol=3e-5, atol=3e-5
        )


def test_tv_saved_width_probabilities_match_independent_dense_gaussian():
    import xarray as xr
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    from pymc_generator.random_walk import _walk_basis

    cfg, root = _smoke("varying")
    _, corpus, prior_cfg = _historical_dataset(root, cfg, 0)
    data = world_data(corpus, 0, prior_cfg, 7)
    with xr.open_datatree(root / "fit/world_00/mcmc/posterior.nc", engine="h5netcdf") as saved:
        posterior = saved["posterior"]
        static_mu = posterior["mu"].values[0, 0]
        sigma = float(posterior["sigma"].values[0, 0])
        tau = float(posterior["baseline_sd"].values[0, 0])
        fitted = data["outcome_data"] * data["outcome_scale"]
        widths, prior, _, _, _ = baseline_geometry(104, 8, 7, 26)
        scores = []
        for width, mass in zip(widths, prior):
            basis = _walk_basis(112, int(width))[15:]
            covariance = tau**2 * (basis @ basis.T) + sigma**2 * np.eye(97)
            scores.append(
                np.log(mass) + multivariate_normal.logpdf(fitted, mean=static_mu, cov=covariance)
            )
        expected = np.exp(scores - logsumexp(scores))
        np.testing.assert_allclose(
            expected, posterior["baseline_width_probabilities"].values[0, 0], atol=1e-8, rtol=1e-9
        )
        baseline = posterior["baseline_path"].values
        reconstructed = (
            baseline
            + posterior["media_path"].values.sum(axis=-1)
            + posterior["control_path"].values.sum(axis=-1)
        )
        assert reconstructed.shape == (2, 100, 97)
        assert np.isfinite(reconstructed).all()


def test_historical_receipts_remain_valid_but_current_resume_fails_closed(tmp_path, monkeypatch):
    from studies.recovery import runner

    cfg, root = _smoke("constant")
    target = tmp_path / "copy"
    shutil.copytree(root / "generate", target / "generate")
    shutil.copytree(root / "fit", target / "fit")
    for world in range(2):
        directory = target / "generate" / f"world_{world:02d}"
        recorded = json.loads((directory / "manifest.json").read_text())
        assert recorded["config"] == {
            key: cfg.resolved[key] for key in ("study", "generator", "pairing")
        }
        assert checkpoint(directory, recorded).complete
        with pytest.raises(ContractError, match="stale"):
            checkpoint(directory, _world_identity(cfg, world))
    monkeypatch.setattr(runner, "generate_pair", lambda *a, **k: pytest.fail("no new random draws"))
    monkeypatch.setattr(runner, "bounded_ols", lambda *a, **k: pytest.fail("no refit"))
    with pytest.raises(ContractError, match="stale"):
        generate(cfg, target)
    with pytest.raises(ContractError, match="stale"):
        fit(cfg, target, "ols")
    # A partial historical stage remains untrusted by CURRENT source identity.
    partial = target / "generate/world_01"
    old = json.loads((partial / "manifest.json").read_text())
    (partial / "completion.json").unlink()
    assert checkpoint(partial, old).state == "partial"
    with pytest.raises(ContractError, match="stale"):
        generate(cfg, target)
    # Historical receipt/content checks independently reject corrupted outputs.
    intact = target / "generate/world_00"
    recorded = json.loads((intact / "manifest.json").read_text())
    (intact / "dataset.npz").write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="hash"):
        checkpoint(intact, recorded)


def test_public_matched_replay_preflights_all_worlds_before_writing(tmp_path, monkeypatch):
    import studies.recovery.matched as matched

    source = ROOT / "notebooks/ols_study_50_seed10029"
    paired = ROOT / "notebooks/media_smoothing_comparison"
    valid = StudyConfig.load(STUDY / "matched_existing_example.toml")
    bad = StudyConfig.parse(
        {
            "study": {"worlds": 2, "seed": 10029},
            "generator": {"n_treatments": 3},
            "pairing": valid.resolved["pairing"],
        }
    )
    monkeypatch.setattr(
        matched, "_evaluate", lambda *args: pytest.fail("invalid fixture evaluated")
    )
    fresh = tmp_path / "new-output"
    with pytest.raises(ContractError, match="n_treatments"):
        matched.replay(bad, source, paired, fresh, worlds=(0,))
    assert not fresh.exists()  # exact public-path review reproduction
    mixed = tmp_path / "mixed-output"
    with pytest.raises(ContractError, match="world|population"):
        matched.replay(valid, source, paired, mixed, worlds=(0, 2))
    assert not mixed.exists()  # valid first world must NOT leave a manifest
    altered_pair = tmp_path / "paired-fixture-copy"
    altered_pair.mkdir()
    shutil.copy2(paired / "manifest.json", altered_pair / "manifest.json")
    for world in (0, 1):
        for suffix in ("fixed_rvs.npz", "paired.npz"):
            name = f"world_{world:02d}_{suffix}"
            (altered_pair / name).symlink_to(paired / name)
        shutil.copy2(
            paired / f"world_{world:02d}_report.json",
            altered_pair / f"world_{world:02d}_report.json",
        )
    second = altered_pair / "world_01_report.json"
    wrong = json.loads(second.read_text())
    wrong["world_id"] = 99
    second.write_text(json.dumps(wrong))
    altered_config = StudyConfig.parse(
        {
            "study": {"worlds": 2, "seed": 10029},
            "pairing": {
                **valid.resolved["pairing"],
                "coupling_manifest": str(altered_pair / "manifest.json"),
            },
        }
    )
    invalid_second = tmp_path / "invalid-second"
    with pytest.raises(ContractError, match="fixture world"):
        matched.replay(altered_config, source, altered_pair, invalid_second, worlds=(0, 1))
    assert not invalid_second.exists()  # valid first, incompatible second metadata
    existing = tmp_path / "existing-output"
    existing.mkdir()
    marker = existing / "sentinel.txt"
    marker.write_bytes(b"untouched")
    for request, fixture, worlds in (
        (bad, paired, (0,)),
        (valid, paired, (0, 2)),
        (altered_config, altered_pair, (0, 1)),
    ):
        with pytest.raises(ContractError):
            matched.replay(request, source, fixture, existing, worlds=worlds)
        assert sorted(existing.iterdir()) == [marker]
        assert marker.read_bytes() == b"untouched"


def test_matched_replay_uses_fixed_primitive_rvs_and_real_outputs(tmp_path):
    from studies.recovery.matched import replay_world

    config = StudyConfig.load(STUDY / "matched_existing_example.toml")
    source = ROOT / "notebooks/ols_study_50_seed10029"
    paired = ROOT / "notebooks/media_smoothing_comparison"
    for world in (0, 1):
        result = replay_world(config, source, paired, world)
        assert result["fixed_primitive_rvs_checked"] == 56
        assert (
            result["original_exact"] and result["candidate_exact"] and result["paired_target_exact"]
        )
        assert result["media_changed"]
    # Tamper only a disposable copy of one fixed-RV snapshot; real generator replay must reject it.
    copied = tmp_path / "pair"
    copied.mkdir()
    for filename in ("manifest.json", "world_00_report.json", "world_00_paired.npz"):
        shutil.copy2(paired / filename, copied / filename)
    with np.load(paired / "world_00_fixed_rvs.npz", allow_pickle=False) as src:
        fixed = {key: src[key].copy() for key in src.files}
    fixed["eps_y"].flat[15] += 1  # perturb a fitted reported week, not a discarded burn-in element
    np.savez_compressed(copied / "world_00_fixed_rvs.npz", **fixed)
    matched = StudyConfig.parse(
        {
            "study": {"worlds": 2, "seed": 10029},
            "pairing": {
                "mode": "matched",
                "media_smoothness_beta": 4.0,
                "coupling_manifest": str(copied / "manifest.json"),
            },
        }
    )
    with pytest.raises((AssertionError, ContractError)):
        replay_world(matched, source, copied, 0)
