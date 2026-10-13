"""Artifact-only study analysis. Reads ONLY the included compact files, never fit archives.

All gates and counts derive from 97 saved true/fitted weeks and the historical
computational-status/pattern metadata. The two regimes are NOT an algorithm ranking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PATTERNS = (
    "hf",
    "pulse",
    "onset",
    "offset",
    "flighting",
    "level_jump",
    "seasonal",
    "trend",
    "trend_up",
    "trend_down",
)
LABELS = {
    "tv": "Historical TV-MCMC",
    "bounded": "Historical constant-baseline bounded least squares",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Check copied-byte provenance, shapes, ledger/world identities and finite data."""
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["compact_payload_sha256"].items():
        if _hash(directory / name) != expected:
            raise ValueError(f"compact snapshot/approved text hash mismatch: {name}")
    metadata = json.loads((directory / "snapshot.json").read_text())
    config = json.loads((directory / "historical-config.json").read_text())
    if metadata["schema"] != 1 or metadata["scored_weeks_inclusive"] != [7, 103]:
        raise ValueError("unknown or unscored compact snapshot")
    if metadata["score_gate"] != {"relative": 0.1, "absolute": 0.01}:
        raise ValueError("historical gate changed")
    if config["scored_weeks_inclusive"] != [7, 103]:
        raise ValueError("configuration/score-window mismatch")
    with np.load(directory / "snapshot.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if set(metadata["results"]) != set(LABELS):
        raise ValueError("missing historical regime")
    needed = {f"{label}_{name}" for label in LABELS for name in ("truth", "fitted", "raw_spend")}
    needed |= {"tv_lower95", "tv_upper95"}
    if set(arrays) != needed:
        raise ValueError("array inventory changed")
    for label, part in metadata["results"].items():
        if (
            part["source_report_run_id"]
            != provenance["historical_approvals"][label]["historical_report_run_id"]
        ):
            raise ValueError("historical approval/report identity mismatch")
        if part["source_provenance"] != (
            "legacy manifest/input-hash compatibility only; output and source hashes unavailable"
        ):
            raise ValueError("overstated historical provenance")
        terms = part["terms"]
        offsets = part["column_offsets"]
        media_offsets = part["media_column_offsets"]
        if (len(terms), len(offsets), len(media_offsets), len(part["worlds"])) != (50, 51, 51, 50):
            raise ValueError("expected 50 complete worlds")
        if offsets[0] or media_offsets[0] or len(part["channels"]) != 799:
            raise ValueError("incomplete world/channel ledger")
        for world, names in enumerate(terms):
            if (
                part["worlds"][world]["world_id"] != world
                or offsets[world + 1] - offsets[world] != len(names)
                or media_offsets[world + 1] - media_offsets[world]
                != sum(n.startswith("C") for n in names)
                or names[0] != ("Baseline" if label == "tv" else "Intercept")
            ):
                raise ValueError(f"invalid term/offset inventory: {label} {world}")
        if part["observed_mean"] is not None and len(part["observed_mean"]) != 50:
            raise ValueError("observation-mean inventory mismatch")
        for key in ("truth", "fitted", "lower95", "upper95", "raw_spend"):
            if f"{label}_{key}" not in arrays:
                continue
            value = arrays[f"{label}_{key}"]
            expected = media_offsets[-1] if key == "raw_spend" else offsets[-1]
            if value.shape != (97, expected) or not np.isfinite(value).all():
                raise ValueError(f"invalid or nonfinite stored {label}_{key}")
        expected_channels = [
            (world, "media" if name.startswith("C") else "control", name)
            for world, names in enumerate(terms)
            for name in names[1:]
        ]
        actual_channels = [
            (row["world_id"], row["kind"], row["channel"]) for row in part["channels"]
        ]
        if actual_channels != expected_channels:
            raise ValueError("saved channel/term identity disagrees")
        if any(set(row["patterns"]) - set(PATTERNS) for row in part["channels"]):
            raise ValueError("unrecognized historical pattern")
    if not np.array_equal(arrays["tv_raw_spend"], arrays["bounded_raw_spend"]):
        raise ValueError("shared original input traces changed between regimes")
    return metadata, arrays


def trajectory(meta: dict, arrays: dict[str, np.ndarray], label: str, world: int) -> dict:
    part = meta["results"][label]
    start, stop = part["column_offsets"][world : world + 2]
    m_start, m_stop = part["media_column_offsets"][world : world + 2]
    return {
        "terms": part["terms"][world],
        "weeks": np.arange(7, 104),
        "truth": arrays[f"{label}_truth"][:, start:stop],
        "fitted": arrays[f"{label}_fitted"][:, start:stop],
        "raw_spend": arrays[f"{label}_raw_spend"][:, m_start:m_stop],
        "lower95": arrays["tv_lower95"][:, start:stop] if label == "tv" else None,
        "upper95": arrays["tv_upper95"][:, start:stop] if label == "tv" else None,
    }


def summarize(meta: dict, arrays: dict[str, np.ndarray]) -> dict:
    """Recompute point gates, partitions, pattern counts, descriptor medians from compact data."""
    world_rows, channel_rows, pattern_rows, descriptors, noise_ratios = [], [], [], [], []
    for label, part in meta["results"].items():
        channel_offset = 0
        for world in range(50):
            values = trajectory(meta, arrays, label, world)
            names, truth, fit = values["terms"], values["truth"], values["fitted"]
            status = part["worlds"][world]["computational_status"]
            if status not in ("valid", "unresolved"):
                raise ValueError("unknown computational qualification")
            n_media = sum(name.startswith("C") for name in names)
            noiseless_norm = np.linalg.norm(truth.sum(axis=1))

            def gate(
                actual: np.ndarray, predicted: np.ndarray, *, noiseless_norm: float = noiseless_norm
            ) -> dict:
                allowance = max(0.1 * np.linalg.norm(actual), 0.01 * noiseless_norm)
                error = float(np.linalg.norm(actual - predicted))
                return {
                    "error": error,
                    "allowance": float(allowance),
                    "pass": bool(error <= allowance),
                }

            true_media = truth[:, 1 : 1 + n_media]
            fit_media = fit[:, 1 : 1 + n_media]
            nonzero_media_pass, nonzero_control_pass = [], []
            for index, name in enumerate(names[1:], 1):
                recorded = part["channels"][channel_offset]
                channel_offset += 1
                nonzero = bool(np.linalg.norm(truth[:, index]) > 0)
                score = gate(truth[:, index], fit[:, index])
                passed = score["pass"] if status == "valid" and nonzero else None
                item = {
                    "regime": label,
                    "world_id": world,
                    "kind": recorded["kind"],
                    "channel": name,
                    "patterns": recorded["patterns"],
                    "nonzero_effect": nonzero,
                    "computational_status": status,
                    "point_pass": passed,
                    "point_error": score["error"] if status == "valid" else None,
                    "point_allowance": score["allowance"] if status == "valid" else None,
                }
                channel_rows.append(item)
                if not nonzero:
                    continue
                (
                    nonzero_media_pass if recorded["kind"] == "media" else nonzero_control_pass
                ).append(passed)
                if label == "tv" and recorded["kind"] == "media":
                    media_index = index - 1
                    profile = part["worlds"][world]["profile"]
                    beta = profile["media_beta"][media_index]
                    spend = values["raw_spend"][:, media_index]
                    contribution = truth[:, index]
                    noise_sd = profile["realized_noise_sd"]
                    if beta == 0 or noise_sd <= 0 or spend.mean() == 0 or contribution.mean() == 0:
                        raise ValueError("undefined TV media descriptor")
                    descriptors.append(
                        {
                            "world_id": world,
                            "channel": name,
                            "point_pass": passed,
                            "saturation_mean": float(np.mean(contribution / beta)),
                            "alpha": profile["carryover_alpha"][media_index],
                            "beta": beta,
                            "spend_cv": float(spend.std() / abs(spend.mean())),
                            "contribution_cv": float(contribution.std() / abs(contribution.mean())),
                            "contribution_sd_per_noise_sd": float(contribution.std() / noise_sd),
                        }
                    )
            if label == "tv":
                observed = part["observed_mean"][world]
                noise_ratios.append(
                    part["worlds"][world]["profile"]["realized_noise_sd"] / observed
                )
            row = {
                "regime": label,
                "world_id": world,
                "computational_status": status,
                "diagnostics": part["worlds"][world]["diagnostics"],
                "media_pass": bool(all(nonzero_media_pass)) if status == "valid" else None,
                "joint_pass": bool(all(nonzero_media_pass + nonzero_control_pass))
                if status == "valid"
                else None,
                "total_media_pass": gate(true_media.sum(axis=1), fit_media.sum(axis=1))["pass"]
                if status == "valid"
                else None,
                "baseline_pass": gate(truth[:, 0], fit[:, 0])["pass"]
                if status == "valid"
                else None,
                "eligible_media": len(nonzero_media_pass),
                "eligible_controls": len(nonzero_control_pass),
            }
            world_rows.append(row)
        if channel_offset != len(part["channels"]):
            raise ValueError("leftover channel ledger entries")
        matching = [row for row in channel_rows if row["regime"] == label]
        for kind in ("media", "control"):
            for pattern in PATTERNS:
                rows = [
                    row for row in matching if row["kind"] == kind and pattern in row["patterns"]
                ]
                eligible = [row for row in rows if row["nonzero_effect"]]
                evaluable = [row for row in eligible if row["computational_status"] == "valid"]
                failed = sum(row["point_pass"] is False for row in evaluable)
                unresolved = len(eligible) - len(evaluable)
                pattern_rows.append(
                    {
                        "regime": label,
                        "kind": kind,
                        "pattern": pattern,
                        "total": len(rows),
                        "eligible": len(eligible),
                        "known_zero": len(rows) - len(eligible),
                        "evaluable": len(evaluable),
                        "failed": failed,
                        "unresolved": unresolved,
                        "failure_rate": failed / len(evaluable) if evaluable else None,
                        "status": "not represented"
                        if not rows
                        else "no eligible nonzero channels"
                        if not eligible
                        else "unresolved"
                        if not evaluable
                        else "evaluable",
                        "small_count_warning": bool(evaluable and len(evaluable) < 10),
                    }
                )
    counts = []
    for label in LABELS:
        worlds = [row for row in world_rows if row["regime"] == label]
        channels = [row for row in channel_rows if row["regime"] == label]
        valid = [row for row in worlds if row["computational_status"] == "valid"]
        item = {
            "regime": LABELS[label],
            "worlds": len(worlds),
            "valid": len(valid),
            "unresolved": len(worlds) - len(valid),
            "unresolved_ids": [
                row["world_id"] for row in worlds if row["computational_status"] != "valid"
            ],
        }
        for key in ("media_pass", "joint_pass", "total_media_pass", "baseline_pass"):
            item[key] = sum(row[key] is True for row in valid)
        for kind in ("media", "control"):
            eligible = [row for row in channels if row["kind"] == kind and row["nonzero_effect"]]
            evaluable = [row for row in eligible if row["computational_status"] == "valid"]
            item[f"{kind}_eligible"] = len(eligible)
            item[f"{kind}_evaluable"] = len(evaluable)
            item[f"{kind}_failed"] = sum(row["point_pass"] is False for row in evaluable)
            item[f"{kind}_unresolved"] = len(eligible) - len(evaluable)
            item[f"{kind}_known_zero"] = sum(
                row["kind"] == kind and not row["nonzero_effect"] for row in channels
            )
        counts.append(item)
    comparisons = []
    for passed, label in ((True, "Passing"), (False, "Failing")):
        group = [row for row in descriptors if row["point_pass"] is passed]
        row = {"TV media": label, "count": len(group)}
        for key in (
            "saturation_mean",
            "alpha",
            "beta",
            "spend_cv",
            "contribution_cv",
            "contribution_sd_per_noise_sd",
        ):
            row[key] = float(np.median([item[key] for item in group]))
        comparisons.append(row)
    diag = [row["diagnostics"] for row in world_rows if row["regime"] == "tv"]
    return {
        "counts": counts,
        "world_rows": world_rows,
        "channel_rows": channel_rows,
        "pattern_rows": pattern_rows,
        "tv_descriptors": descriptors,
        "tv_descriptor_medians": comparisons,
        "tv_noise_fraction_quartiles": np.quantile(noise_ratios, [0.25, 0.5, 0.75]).tolist(),
        "tv_diagnostics": {
            "divergences": sum(r["divergences"] for r in diag),
            "maxdepth_hits": sum(r["maxdepth_hits"] for r in diag),
            "max_rhat": max(r["max_rhat"] for r in diag),
            "min_bulk_ess": min(r["min_bulk_ess"] for r in diag),
            "min_tail_ess": min(r["min_tail_ess"] for r in diag),
            "min_bfmi": min(r["min_bfmi"] for r in diag),
            "all_pass": all(r["diagnostics_pass"] is True for r in diag),
        },
    }
