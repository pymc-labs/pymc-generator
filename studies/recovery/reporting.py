"""Current native recovery extraction and sealed artifact-only reporting."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

import pymc_generator as pg
from pymc_generator.trajectories import TRAJECTORY_COMPONENTS

from .analysis import score_components
from .artifacts import checkpoint, complete_checkpoint, identity, write_manifest
from .config import ContractError, StudyConfig
from .runner import _fit_identity, _world_identity

PATTERNS = (*TRAJECTORY_COMPONENTS, "trend_up", "trend_down")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _source_files() -> dict[str, Path]:
    files = {
        name: Path(__file__).with_name(name)
        for name in (
            "reporting.py",
            "report_notebook.py",
            "execute_report.py",
            "analysis.py",
            "artifacts.py",
            "config.py",
            "__main__.py",
            "runner.py",
            "dgp.py",
            "methods.py",
            "models.py",
        )
    }
    package = Path(pg.__file__).parent
    files.update(
        {f"generator/{path.relative_to(package)}": path for path in sorted(package.rglob("*.py"))}
    )
    return files


def _arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys) <= set(archive.files):
            raise ContractError(f"missing report arrays: {path}")
        return {key: archive[key].copy() for key in keys}


def _channels(rows: list[dict]) -> list[dict]:
    """Count the pattern-bearing channel, never another failure in its world."""
    result = []
    for kind in ("media", "control"):
        for pattern in PATTERNS:
            matching = [row for row in rows if row["kind"] == kind and pattern in row["patterns"]]
            eligible = [row for row in matching if row["nonzero_effect"]]
            evaluable = [row for row in eligible if row["computational_status"] == "valid"]
            unresolved = [row for row in eligible if row["computational_status"] != "valid"]
            failed = sum(row["point_pass"] is False for row in evaluable)
            result.append(
                {
                    "kind": kind,
                    "pattern": pattern,
                    "total": len(matching),
                    "eligible": len(eligible),
                    "evaluable": len(evaluable),
                    "failed": failed,
                    "unresolved": len(unresolved),
                    "known_zero": len(matching) - len(eligible),
                    "status": (
                        "not represented"
                        if not matching
                        else "no eligible nonzero channels"
                        if not eligible
                        else "unresolved"
                        if not evaluable
                        else "evaluable"
                    ),
                    "failure_rate": failed / len(evaluable) if evaluable else None,
                    "small_count_warning": bool(evaluable and len(evaluable) < 10),
                }
            )
    return result


def _profiles(
    corpus: dict, local: int, start: int, truth: np.ndarray, media: int, controls: int
) -> dict:
    """True scored-regime paths, with noise and mechanism inputs from the same world."""
    baseline = truth[:, 0]
    return {
        "realized_noise_sd": float(
            np.asarray(corpus["outcome_noise"][local, start : start + len(truth)], float).std()
        ),
        "baseline_temporal_sd": float(baseline.std()),
        "baseline_mean": float(baseline.mean()),
        "media_true_temporal_sd": truth[:, 1 : 1 + media].std(axis=0).tolist(),
        "control_true_temporal_sd": truth[:, 1 + media : 1 + media + controls].std(axis=0).tolist(),
        "carryover_alpha": np.asarray(corpus["carryover_alpha"][local, :media], float).tolist(),
        "mm_kappa_mult": np.asarray(corpus["mm_kappa_mult"][local, :media], float).tolist(),
        "media_beta": np.asarray(corpus["beta"][local, :media], float).tolist(),
        "control_coefficient": np.asarray(corpus["rho_zy"][local, :controls], float).tolist(),
    }


def _pattern_rows(
    corpus: dict,
    local: int,
    names: list[str],
    truth: np.ndarray,
    score: dict,
    status: str,
    config: StudyConfig,
) -> list[dict]:
    # The generator omits trajectory metadata when inclusion is deterministic
    # (0/1) and no schedule exists. Recover only those exact admitted labels
    # from the SEALED full prior, never infer fractional flags from raw paths.
    have_flags = "treatment_components" in corpus and "covariate_components" in corpus
    prior = pg.SCMPrior(**config.resolved["generator"])
    probabilities = prior.trajectory_inclusion_probs()
    if not have_flags and (
        prior.trajectory_metadata_enabled
        or any(p not in (0.0, 1.0) for group in probabilities.values() for p in group.values())
    ):
        raise ContractError("missing required realized trajectory pattern flags")
    rows = []
    for index, name in enumerate(names[1:], 1):
        kind = "media" if name.startswith("C") else "control"
        channel_index = int(name[1:]) - 1
        if have_flags:
            flags = corpus["treatment_components" if kind == "media" else "covariate_components"][
                local, channel_index
            ]
            labels = [value for i, value in enumerate(TRAJECTORY_COMPONENTS) if flags[i]]
            if "trend" in labels:
                change = corpus[
                    "trajectory_treatment_trend_change"
                    if kind == "media"
                    else "trajectory_covariate_trend_change"
                ][local, channel_index]
                labels.append(
                    "trend_up" if change > 0 else "trend_down" if change < 0 else "trend_flat"
                )
        else:
            group = "treatment" if kind == "media" else "covariate"
            labels = [
                value for value in TRAJECTORY_COMPONENTS if probabilities[group][value] == 1.0
            ]
        nonzero = bool(np.linalg.norm(truth[:, index]) > 0)
        rows.append(
            {
                "kind": kind,
                "channel": name,
                "patterns": labels,
                "nonzero_effect": nonzero,
                "computational_status": status,
                "point_pass": score["channels"][name]["pass"]
                if nonzero and status == "valid"
                else None,
                "point_error": score["channels"][name]["error"] if status == "valid" else None,
                "point_allowance": score["channels"][name]["allowance"]
                if status == "valid"
                else None,
            }
        )
    return rows


def _requested_methods(config: StudyConfig, methods: tuple[str, ...] | None) -> tuple[str, ...]:
    enabled = config.resolved["methods"]["enabled"]
    if methods is None:
        return tuple(enabled)
    if (
        not methods
        or any(type(name) is not str for name in methods)
        or len(set(methods)) != len(methods)
        or set(methods) - set(enabled)
    ):
        raise ContractError(
            "method subset must be nonempty, distinct and enabled in the original fit config"
        )
    return methods


def _fit_labels(config: StudyConfig, methods: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (f"advi_{i + 1}", method, i) if method == "advi" else (method, method, 0)
        for method in methods
        for i in range(len(config.resolved["methods"]["advi"]["seeds"]) if method == "advi" else 1)
    )


def _preflight(config: StudyConfig, root: Path, labels: tuple[tuple[str, str, int], ...]):
    """Current source/fit receipts for all worlds BEFORE any report output writes."""
    inputs = {}
    for world in range(config.resolved["study"]["worlds"]):
        source = root / "generate" / f"world_{world:02d}"
        if not checkpoint(source, _world_identity(config, world)).complete:
            raise ContractError(f"world {world} lacks current generation receipt")
        meta = _json(source / "metadata.json")
        if (
            meta.get("world_id") != world
            or meta.get("local_world") != world % 2
            or meta.get("generation_seed") != config.resolved["study"]["seed"] + world // 2
        ):
            raise ContractError(f"generation metadata disagrees with world {world}")
        for name in ("manifest.json", "completion.json", "dataset.npz", "metadata.json"):
            inputs[f"generate-{world:02d}-{name}"] = source / name
        for label, method, replicate in labels:
            fit = root / "fit" / f"world_{world:02d}" / label
            expected = _fit_identity(config, root, world, method, replicate)
            if not checkpoint(fit, expected).complete:
                raise ContractError(f"world {world} {label} lacks current fit receipt")
            record = _json(fit / "report.json")
            if (
                record.get("world_id") != world
                or record.get("replicate") != replicate
                or record.get("effective_seed") != expected["method_seed"]
            ):
                raise ContractError(f"world {world} {label} fit identity mismatch")
            for name in (
                "manifest.json",
                "completion.json",
                "report.json",
                "components.npz" if method == "ols" else "posterior.nc",
            ):
                inputs[f"fit-{world:02d}-{label}-{name}"] = fit / name
    return inputs


def _per_world(config: StudyConfig, root: Path, methods: tuple[str, ...]):
    import arviz as az

    labels = _fit_labels(config, methods)
    inputs = _preflight(config, root, labels)
    all_rows, channels, trajectories = [], [], {}
    for world in range(config.resolved["study"]["worlds"]):
        source = root / "generate" / f"world_{world:02d}"
        with np.load(source / "dataset.npz", allow_pickle=False) as archive:
            corpus = {key: archive[key].copy() for key in archive.files if key != "diagnostics"}
        local = world % 2
        start, stop = config.resolved["study"]["score_weeks"]
        n_media = int(corpus["n_treatments_active"][local])
        n_controls = int(corpus["n_covariates_active"][local])
        names = [
            "Baseline",
            *(f"C{i + 1}" for i in range(n_media)),
            *(f"Z{i + 1}" for i in range(n_controls)),
        ]
        truth = np.column_stack(
            [
                corpus["baseline_intrinsic"][local, start : stop + 1],
                corpus["treatment_contribution_raw"][local, start : stop + 1, :n_media],
                corpus["covariate_contribution"][local, start : stop + 1, :n_controls],
            ]
        ).astype(float)
        profile = _profiles(corpus, local, start, truth, n_media, n_controls)
        for label, method, replicate in labels:
            fit = root / "fit" / f"world_{world:02d}" / label
            report = _json(fit / "report.json")
            if method == "ols":
                all_components = _arrays(fit / "components.npz", ("components",))["components"]
                estimate = all_components[
                    :,
                    [
                        0,
                        *range(1, n_media + 1),
                        *range(
                            1 + config.resolved["generator"]["n_treatments"],
                            1 + config.resolved["generator"]["n_treatments"] + n_controls,
                        ),
                    ],
                ]
                lower = upper = None
                status = "valid" if report.get("success") is True else "unresolved"
                diagnostic = {
                    "algorithm": "bounded estimated-transform nonlinear least squares",
                    "optimizer_converged": report.get("success"),
                }
                uncertainty = {"type": "unavailable (point optimizer only)"}
            else:
                dt = az.from_netcdf(fit / "posterior.nc", engine="h5netcdf")
                try:
                    posterior = dt["posterior"]
                    media = posterior["media_path"].values[:, :, :, :n_media]
                    control = posterior["control_path"].values[:, :, :, :n_controls]
                    baseline = (
                        posterior["baseline_path"].values
                        if "baseline_path" in posterior
                        else np.broadcast_to(
                            posterior["intercept"].values[..., None], media.shape[:3]
                        )
                    )
                    draws = np.concatenate((baseline[..., None], media, control), axis=-1).reshape(
                        -1, len(truth), len(names)
                    )
                    finite = bool(np.isfinite(draws).all())
                    if finite:
                        estimate = draws.mean(axis=0)
                        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
                        uncertainty = {
                            "type": (
                                "variational draw spread, not calibrated posterior"
                                if method == "advi"
                                else "95% pointwise equal-tail posterior intervals; only interpret after diagnostics"
                            ),
                            "channel_pointwise_coverage": float(
                                np.mean(
                                    (truth[:, 1:] >= lower[:, 1:]) & (truth[:, 1:] <= upper[:, 1:])
                                )
                            ),
                        }
                    else:
                        estimate = lower = upper = None
                        uncertainty = {
                            "type": "unavailable: nonfinite selected component draws",
                            "channel_pointwise_coverage": None,
                        }
                    if method == "advi":
                        diagnostic = {
                            "algorithm": "full-rank ADVI",
                            "finite_loss": report.get("loss_finite"),
                            "posterior_finite": report.get("posterior_finite"),
                            "selected_draws_finite": finite,
                            "rhat_ess": "not applicable",
                            "seed_sensitivity": "compare all configured seeds; finite loss is not convergence",
                        }
                        status = (
                            "finite_only_not_converged"
                            if report.get("loss_finite") is True
                            and report.get("posterior_finite") is True
                            and finite
                            else "unresolved"
                        )
                    else:
                        scalar = [
                            name
                            for name in (
                                "intercept",
                                "sigma_relative",
                                "baseline_sd_relative",
                                "alpha",
                                "log_kappa",
                                "amplitude",
                                "control_reference_contribution",
                            )
                            if name in posterior
                        ]

                        def extremum(ds, fn):
                            return float(
                                fn(
                                    np.concatenate(
                                        [
                                            np.asarray(var.values).ravel()
                                            for var in ds.data_vars.values()
                                        ]
                                    )
                                )
                            )

                        rhat = extremum(az.rhat(dt, var_names=scalar), np.max)
                        bulk = extremum(az.ess(dt, var_names=scalar, method="bulk"), np.min)
                        tail = extremum(az.ess(dt, var_names=scalar, method="tail"), np.min)
                        bfmi = float(np.min(az.bfmi(dt)["energy"].values))
                        stats = dt["sample_stats"]
                        divergences = int(stats["diverging"].values.sum())
                        depth = int(stats["maxdepth_reached"].values.sum())
                        gate = config.resolved["diagnostics"]
                        healthy = (
                            divergences <= gate["divergences"]
                            and depth <= gate["maxdepth_hits"]
                            and rhat < gate["rhat"]
                            and bulk >= gate["ess"]
                            and tail >= gate["ess"]
                            and bfmi >= gate["bfmi"]
                        )
                        diagnostic = {
                            "algorithm": "Nutpie draw_diag",
                            "divergences": divergences,
                            "maxdepth_hits": depth,
                            "max_rhat_screen": rhat,
                            "min_bulk_ess_screen": bulk,
                            "min_tail_ess_screen": tail,
                            "min_bfmi": bfmi,
                            "diagnostics_pass": bool(healthy),
                            "screened_variables": scalar,
                        }
                        status = "valid" if healthy and finite else "unresolved"
                finally:
                    dt.close()
            if estimate is not None and estimate.shape != truth.shape:
                raise ContractError("saved fit components disagree with scored truth dimensions")
            score = (
                score_components(truth, estimate, names, **config.resolved["score"])
                if estimate is not None
                else None
            )
            channels.extend(
                {"world_id": world, "method": label, **row}
                for row in _pattern_rows(corpus, local, names, truth, score, status, config)
            )
            trajectories[f"{world}:{label}"] = {
                "terms": names,
                "weeks": list(range(start, stop + 1)),
                "truth": truth.tolist(),
                "fitted": estimate.tolist() if estimate is not None else None,
                "lower95": lower.tolist() if lower is not None else None,
                "upper95": upper.tolist() if upper is not None else None,
                "raw_spend": np.asarray(
                    corpus["treatment_raw"][local, start : stop + 1, :n_media], float
                ).tolist(),
            }
            all_rows.append(
                {
                    "world_id": world,
                    "method": label,
                    "computational_status": status,
                    "diagnostics": diagnostic,
                    "recovery": {
                        "media_pass": score["media_pass"] if status == "valid" else None,
                        "control_pass": score["controls_pass"] if status == "valid" else None,
                        "joint_pass": score["joint_pass"] if status == "valid" else None,
                        "total_media": score["total_media"] if status == "valid" else None,
                        "baseline": score["baseline"] if status == "valid" else None,
                    },
                    "uncertainty": uncertainty,
                    "profile": profile,
                }
            )
        advi = [trajectories[f"{world}:{label}"] for label, method, _ in labels if method == "advi"]
        if len(advi) >= 2:
            # A missing path OR an invalid optimization/posterior receipt makes
            # the complete configured interseed comparison unqualified.
            if any(path["fitted"] is None for path in advi) or any(
                row["world_id"] == world
                and row["method"].startswith("advi_")
                and row["computational_status"] != "finite_only_not_converged"
                for row in all_rows
            ):
                for row in all_rows:
                    if row["world_id"] == world and row["method"].startswith("advi_"):
                        row["diagnostics"]["max_interseed_channel_discrepancy_over_gate"] = None
                continue
            outcome_norm = np.linalg.norm(truth.sum(axis=1))
            ratios = [
                float(
                    np.linalg.norm(
                        np.asarray(first["fitted"])[:, j] - np.asarray(second["fitted"])[:, j]
                    )
                    / max(
                        config.resolved["score"]["relative"] * np.linalg.norm(truth[:, j]),
                        config.resolved["score"]["absolute"] * outcome_norm,
                    )
                )
                for first in advi
                for second in advi
                for j in range(1, truth.shape[1])
                if np.linalg.norm(truth[:, j]) > 0
            ]
            for row in all_rows:
                if row["world_id"] == world and row["method"].startswith("advi_"):
                    row["diagnostics"]["max_interseed_channel_discrepancy_over_gate"] = max(
                        ratios, default=None
                    )
    return (
        all_rows,
        channels,
        trajectories,
        inputs,
        {
            "type": "current-run: generation and selected fit identities/output receipts verified against CURRENT sources",
            "current_source_status": ["current-source qualified"],
            "warning": "Compatibility is NOT sampling convergence or scientific recovery",
        },
    )


def analyze(config: StudyConfig, source: Path, methods: tuple[str, ...] | None = None) -> dict:
    """Public native extraction; strictly read-only and shares report's verified reader."""
    chosen = _requested_methods(config, methods)
    rows, channels, _, _, provenance = _per_world(config, source.resolve(), chosen)
    return {
        "methods": list(chosen),
        "provenance": provenance,
        "worlds": len({r["world_id"] for r in rows}),
        "world_method_rows": len(rows),
        "status_counts": {
            label: sum(r["computational_status"] == label for r in rows)
            for label in sorted({r["computational_status"] for r in rows})
        },
        "valid_media_pass": sum(r["recovery"]["media_pass"] is True for r in rows),
        "valid_joint_pass": sum(r["recovery"]["joint_pass"] is True for r in rows),
        "pattern_rates": {
            method: _channels([r for r in channels if r["method"] == method])
            for method in sorted({r["method"] for r in channels})
        },
    }


def _identity(config: StudyConfig, inputs: dict[str, Path], methods: tuple[str, ...]):
    ident = identity(
        "report",
        config,
        source_files=_source_files(),
        dataset_files=inputs,
        expected_outputs={"evidence.json", "report.ipynb", "interpretation.md"},
    )
    ident["report_methods"] = list(methods)
    ident["environment"]["report_packages"] = {
        name: version(name)
        for name in ("nbclient", "nbformat", "ipykernel", "matplotlib", "pandas")
    }
    return ident


def create_report(
    config: StudyConfig, source: Path, out: Path, *, methods: tuple[str, ...] | None = None
) -> dict:
    """New root only; report reads fully qualified saved stages and performs NO inference."""
    source, out = source.resolve(), out.resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ContractError("report output must be separate from saved fit/source artifacts")
    if out.exists():
        raise ContractError("report output must be a new path, never overwrite an old run")
    selected = _requested_methods(config, methods)
    rows, channels, trajectories, inputs, provenance = _per_world(config, source, selected)
    ident = _identity(config, inputs, selected)
    run_id = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    patterns = {
        method: _channels([row for row in channels if row["method"] == method])
        for method in sorted({row["method"] for row in channels})
    }
    from .report_notebook import notebook

    payload = {
        "schema": 2,
        "run_id": run_id,
        "kind": "native",
        "report_methods": list(selected),
        "provenance": provenance,
        "resolved_config": config.resolved,
        "score_gate": config.resolved["score"],
        "diagnostic_gate": config.resolved["diagnostics"],
        "worlds": rows,
        "channels": channels,
        "pattern_rates": patterns,
        "trajectories": trajectories,
        "input_paths": {key: os.path.relpath(path, out) for key, path in inputs.items()},
        "capabilities": {
            "optional_logistic_screen": "not supported",
            "optional_tsne": "not supported",
            "optional_correlated_pairs": "not supported",
        },
    }
    text = (
        "# Interpretation · pending review\n\n"
        f"Run identity: `{run_id}`\n\n"
        "Descriptive diagnostics and recovery gates do not imply convergence, causal identification, "
        "scientific interpretation or human approval. New inputs require a new report.\n"
    )
    try:
        encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        rendered = json.dumps(notebook(out), indent=1) + "\n"
    except (TypeError, ValueError) as exc:
        raise ContractError("report evidence is not finite/serializable") from exc
    write_manifest(out, ident)
    (out / "evidence.json").write_text(encoded)
    (out / "interpretation.md").write_text(text)
    (out / "report.ipynb").write_text(rendered)
    complete_checkpoint(out, ident, set(ident["expected_outputs"]))
    return {
        "run_id": run_id,
        "world_records": len(rows),
        "pattern_methods": list(patterns),
        "provenance": provenance,
        "report": str(out / "report.ipynb"),
    }


def read_report(out: Path) -> dict:
    """Run-All: rehash original saved input bytes, CURRENT report source and sealed output."""
    out = out.resolve()
    payload = _json(out / "evidence.json")
    recorded = _json(out / "manifest.json")
    config = StudyConfig(recorded["config"])
    if (
        payload.get("schema") != 2
        or not isinstance(payload.get("resolved_config"), dict)
        or payload["resolved_config"].get("schema") != 2
        or {key: payload["resolved_config"].get(key) for key in config.resolved} != config.resolved
    ):
        raise ContractError("report resolved config differs from recorded identity")
    methods = tuple(payload.get("report_methods", ()))
    _requested_methods(config, methods)
    inputs = {key: (out / value).resolve() for key, value in payload["input_paths"].items()}
    expected = _identity(config, inputs, methods)
    if not checkpoint(out, expected).complete:
        raise ContractError("report evidence incomplete")
    run_id = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if payload["run_id"] != run_id or f"`{run_id}`" not in (out / "interpretation.md").read_text():
        raise ContractError("interpretation/run identity mismatch")
    return payload
