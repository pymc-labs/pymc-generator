"""One-time, read-only export from the approved HISTORICAL evidence, never a fit.

Not needed to view or rerun report.ipynb. The original tree is an explicit argument;
no historical paths or source data are required by the published analysis notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def export(old: Path, target: Path) -> None:
    if any(
        (target / name).exists()
        for name in (
            "snapshot.json",
            "snapshot.npz",
            "historical-config.json",
            "provenance.json",
            "tv50.md",
            "bounded50.md",
        )
    ):
        raise FileExistsError("refusing to overwrite existing compact publication")
    notebooks = old / "notebooks"
    study = old / "studies/recovery"
    base = read(notebooks / "ols_study_50_seed10029/manifest.json")
    bounded = read(notebooks / "ols_study_50_seed10029_generator_bounds/manifest.json")
    tv_data = read(notebooks / "time_varying_baseline/data_manifest.json")
    nuts = read(notebooks / "time_varying_baseline/nuts/manifest.json")
    assert base["worlds"] == 50 and base["first_seed"] == 10029
    assert bounded["data_sha256"] == tv_data["source_sha256"]
    for name, expected in tv_data["source_sha256"].items():
        assert sha(notebooks / "ols_study_50_seed10029" / name) == expected, name
    assert nuts["data_manifest"] == tv_data
    assert bounded["seed"] == 2026 and nuts["seeds"] == list(range(92001, 92051))
    assert tv_data["seeds"] == list(range(91001, 91051))
    assert tv_data["fit_window"] == [15, 112] and bounded["fit_weeks"] == [7, 103]
    config = {
        "original_full_dgp": base["config"],
        "constant_generation": {
            "worlds": 50,
            "cells_per_batch": 2,
            "draws_per_cell": 1,
            "batch_seeds": list(range(10029, 10054)),
            "source_batch_sha256": tv_data["source_sha256"],
        },
        "matched_tv_intervention": {
            k: tv_data[k]
            for k in (
                "baseline_std_range",
                "smoothness_beta_prior",
                "full_horizon",
                "reported_window",
                "fit_window",
                "max_smoothing_width",
                "baseline_floor",
                "confounding_strength",
                "matched",
                "seeds",
            )
        },
        "bounded_fit": {
            k: bounded[k]
            for k in (
                "bounds",
                "source_sha256",
                "starts",
                "max_nfev_per_start",
                "initialization",
                "seed",
                "objective",
                "oracle_information",
                "fit_weeks",
            )
        },
        "tv_nuts_fit": {
            k: nuts[k]
            for k in (
                "sampler",
                "adaptation",
                "chains",
                "tune",
                "draws",
                "target_accept",
                "maxdepth",
                "seeds",
                "baseline_inference",
            )
        },
        "historical_model_text_sha256": hashlib.sha256(nuts["model_source"].encode()).hexdigest(),
        "tv_conditional_baseline_reconstruction_seeds": list(range(93001, 93051)),
        "tv_predictive_noise_seeds": list(range(94001, 94051)),
        "gate": {
            "relative": 0.1,
            "absolute": 0.01,
            "diagnostic_rhat_lt": 1.01,
            "ess_gt_eq": 400,
            "bfmi_gt_eq": 0.3,
            "max_divergences": 0,
            "max_depth_hits": 0,
        },
        "scored_weeks_inclusive": [7, 103],
    }
    results: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    approvals: dict[str, dict] = {}
    for label, origin in (("tv", "tv50"), ("bounded", "bounded50")):
        report = study / "report-artifacts" / f"{origin}-profile-v2"
        review_dir = study / "report-artifacts" / f"{origin}-profile-v2-review"
        interpretation = study / "interpretations" / f"{origin}.md"
        evidence_path = report / "evidence.json"
        evidence = read(evidence_path)
        review = read(review_dir / "review.json")
        receipt = read(review_dir / "review-receipt.json")
        manifest = read(report / "manifest.json")
        assert manifest["stage"] == "report"
        for index in range(25):
            key = f"batch-{index:02d}"
            assert manifest["inputs"][key] == tv_data["source_sha256"][f"batch_{index:02d}.npz"]
        assert evidence["run_id"] == review["report_run_id"]
        assert review["decision"] == "approved" and review["reviewer"] == "Teemu Säilynoja"
        assert sha(evidence_path) == review["evidence_sha256"]
        assert sha(interpretation) == review["interpretation_sha256"]
        assert sha(report / "manifest.json") == review["report_manifest_sha256"]
        assert sha(report / "completion.json") == review["report_receipt_sha256"]
        assert receipt["outputs"] == {
            "interpretation.md": sha(interpretation),
            "review.json": sha(review_dir / "review.json"),
        }
        assert len(evidence["worlds"]) == len(evidence["trajectories"]) == 50
        assert len(evidence["channels"]) == 799
        assert evidence["score_gate"] == {"relative": 0.1, "absolute": 0.01}
        assert evidence["provenance"] == (
            "legacy manifest/input-hash compatibility only; output and source hashes unavailable"
        )
        terms = []
        segments: dict[str, list[np.ndarray]] = {
            key: [] for key in ("truth", "fitted", "raw_spend", "lower95", "upper95")
        }
        starts = [0]
        media_starts = [0]
        observed_means = []
        for world in range(50):
            trajectory = evidence["trajectories"][str(world)]
            names = trajectory["terms"]
            n_media = sum(name.startswith("C") for name in names)
            assert trajectory["weeks"] == list(range(7, 104))
            assert names[0] == ("Baseline" if label == "tv" else "Intercept")
            terms.append(names)
            starts.append(starts[-1] + len(names))
            media_starts.append(media_starts[-1] + n_media)
            for key in ("truth", "fitted", "raw_spend"):
                value = np.asarray(trajectory[key], dtype="float64")
                assert value.shape == (97, n_media if key == "raw_spend" else len(names))
                assert np.isfinite(value).all()
                segments[key].append(value)
            if label == "tv":
                for key in ("lower95", "upper95"):
                    value = np.asarray(trajectory[key], dtype="float64")
                    assert value.shape == (97, len(names)) and np.isfinite(value).all()
                    segments[key].append(value)
                source = (
                    notebooks / f"time_varying_baseline/nuts/world_{world:02d}_nuts_recovery.npz"
                )
                assert sha(source) == manifest["inputs"][f"recovery-{world:02d}"]
                with np.load(source, allow_pickle=False) as saved:
                    observed = saved["observed"]
                assert observed.shape == (97,) and np.isfinite(observed).all()
                observed_means.append(float(np.mean(observed)))
        for key, values in segments.items():
            if values:
                arrays[f"{label}_{key}"] = np.concatenate(values, axis=1)
        results[label] = {
            "source_report_run_id": evidence["run_id"],
            "source_kind": evidence["kind"],
            "source_provenance": evidence["provenance"],
            "source_diagnostic_gate": evidence["diagnostic_gate"],
            "terms": terms,
            "column_offsets": starts,
            "media_column_offsets": media_starts,
            "observed_mean": observed_means if label == "tv" else None,
            "worlds": [
                {
                    k: row[k]
                    for k in (
                        "world_id",
                        "computational_status",
                        "diagnostics",
                        "profile",
                        "uncertainty",
                    )
                }
                for row in evidence["worlds"]
            ],
            "channels": [
                {k: row[k] for k in ("world_id", "kind", "channel", "patterns")}
                for row in evidence["channels"]
            ],
        }
        approvals[label] = {
            "original_source_commit": "4dc69ce",
            "historical_report_run_id": evidence["run_id"],
            "historical_report_path_at_commit": f"studies/recovery/report-artifacts/{origin}-profile-v2/",
            "historical_interpretation_path_at_commit": f"studies/recovery/interpretations/{origin}.md",
            "original_evidence_sha256": sha(evidence_path),
            "original_report_manifest_sha256": sha(report / "manifest.json"),
            "original_report_receipt_sha256": sha(report / "completion.json"),
            "original_review_json_sha256": sha(review_dir / "review.json"),
            "original_review_receipt_sha256": sha(review_dir / "review-receipt.json"),
            "historical_review_id": review["review_id"],
            "reviewer": review["reviewer"],
            "historical_decision": review["decision"],
            "interpretation_sha256": sha(interpretation),
        }
    target.mkdir(parents=True, exist_ok=True)
    save(target / "historical-config.json", config)
    save(
        target / "snapshot.json",
        {
            "schema": 1,
            "scored_weeks_inclusive": [7, 103],
            "score_gate": {"relative": 0.1, "absolute": 0.01},
            "results": results,
        },
    )
    np.savez_compressed(target / "snapshot.npz", **arrays)
    for label, origin in (("tv", "tv50"), ("bounded", "bounded50")):
        (target / f"{origin}.md").write_bytes(
            (study / "interpretations" / f"{origin}.md").read_bytes()
        )
    save(
        target / "provenance.json",
        {
            "scope": "Historical approvals only; compact publication is not a newly approved/sealed scientific run",
            "original_commit": "4dc69ce",
            "review_session": "Plannotator pn-2a768e",
            "legacy_limit": "Original manifests and input hashes establish compatibility; historical per-fit output and original source hashes unavailable; approved report receipts seal later evidence, not old fitting",
            "historical_approvals": approvals,
            "compact_payload_sha256": {
                name: sha(target / name)
                for name in (
                    "historical-config.json",
                    "snapshot.json",
                    "snapshot.npz",
                    "tv50.md",
                    "bounded50.md",
                )
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.historical_root.resolve(), args.out.resolve())
