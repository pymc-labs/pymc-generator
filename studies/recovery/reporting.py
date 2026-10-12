"""Artifact-only recovery evidence. Never imports fitting methods or samples worlds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import pymc_generator as pg
from pymc_generator.trajectories import TRAJECTORY_COMPONENTS

from .analysis import _legacy_compatible, saved_bounded_world, saved_tv_world, score_components
from .artifacts import checkpoint, complete_checkpoint, digest, identity, write_manifest
from .config import ContractError, StudyConfig

PATTERNS = (*TRAJECTORY_COMPONENTS, "trend_up", "trend_down")
PROVENANCE_LEGACY = (
    "legacy manifest/input-hash compatibility only; output and source hashes unavailable"
)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys) <= set(archive.files):
            raise ContractError(
                f"missing report arrays in {path}: {sorted(set(keys) - set(archive.files))}"
            )
        return {key: archive[key].copy() for key in keys}


def _source_files() -> dict[str, Path]:
    files = {
        name: Path(__file__).with_name(name)
        for name in (
            "reporting.py",
            "report_notebook.py",
            "execute_report.py",
            "review_interpretation.py",
            "analysis.py",
            "artifacts.py",
            "config.py",
            "__main__.py",
        )
    }
    package = Path(pg.__file__).parent
    files.update(
        {f"generator/{path.relative_to(package)}": path for path in sorted(package.rglob("*.py"))}
    )
    return files


def _channels(rows: list[dict]) -> list[dict]:
    """Count the pattern-bearing channel, not another failure in its world."""
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
    """Scored-regime variation from the exact saved truth; noise/parameters from source corpus."""
    noise = np.asarray(corpus["outcome_noise"][local, start : start + len(truth)], float)
    baseline = truth[:, 0]
    return {
        "realized_noise_sd": float(noise.std()),
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
    supplied: list[dict] | None = None,
) -> list[dict]:
    by_name = {(row["kind"], row["channel"]): row for row in supplied or []}
    rows = []
    for index, name in enumerate(names[1:], 1):
        kind = "media" if name.startswith("C") else "control"
        channel_index = int(name[1:]) - 1
        pattern = by_name.get((kind, name))
        if pattern is None:
            flags = corpus["treatment_components" if kind == "media" else "covariate_components"][
                local, channel_index
            ]
            changes = corpus[
                "trajectory_treatment_trend_change"
                if kind == "media"
                else "trajectory_covariate_trend_change"
            ][local, channel_index]
            labels = [value for j, value in enumerate(TRAJECTORY_COMPONENTS) if flags[j]]
            if "trend" in labels:
                labels.append(
                    "trend_up" if changes > 0 else "trend_down" if changes < 0 else "trend_flat"
                )
        else:
            labels = [value for value in TRAJECTORY_COMPONENTS if pattern.get(value) is True]
            if "trend" in labels:
                labels.append("trend_" + pattern.get("trend_direction", "flat"))
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


def _legacy(config: StudyConfig, directory: Path, kind: str):
    _legacy_compatible(directory, kind, config)
    base = (
        directory.parent.parent if kind == "tv_nuts" else directory.parent
    ) / "ols_study_50_seed10029"
    patterns = _json(directory.parent / "patterns.json") if kind == "tv_nuts" else None
    if kind == "bounded_ols":
        # This is the SAME original 50-world base corpus, with the same pattern vocabulary.
        candidate = base.parent / "time_varying_baseline/patterns.json"
        patterns = _json(candidate) if candidate.is_file() else None
    if patterns is None:
        raise ContractError("legacy pattern records unavailable; do not infer pattern absence")
    if not isinstance(patterns, list):
        raise ContractError("invalid legacy pattern inventory")
    all_rows, channels, trajectories, inputs = [], [], {}, {}
    inputs["source-manifest"] = base / "manifest.json"
    inputs["fit-manifest"] = directory / "manifest.json"
    if kind == "tv_nuts":
        inputs["data-manifest"] = directory.parent / "data_manifest.json"
        inputs["patterns"] = directory.parent / "patterns.json"
    else:
        inputs["patterns"] = base.parent / "time_varying_baseline/patterns.json"
    pattern_by_world = {}
    for row in patterns:
        pattern_by_world.setdefault(row["world_id"], []).append(row)
    for world in range(config.resolved["study"]["worlds"]):
        batch, local = divmod(world, 2)
        batch_path = base / f"batch_{batch:02d}.npz"
        inputs[f"batch-{batch:02d}"] = batch_path
        corpus = pg.load_corpus(batch_path)
        report_path = (
            directory / f"world_{world:02d}_nuts_report.json"
            if kind == "tv_nuts"
            else directory / f"world_{world:02d}.json"
        )
        fit_path = (
            directory / f"world_{world:02d}_nuts_recovery.npz"
            if kind == "tv_nuts"
            else directory / f"world_{world:02d}_fits.npz"
        )
        inputs[f"report-{world:02d}"] = report_path
        inputs[f"recovery-{world:02d}"] = fit_path
        record = _json(report_path)
        if kind == "tv_nuts":
            saved = saved_tv_world(
                directory,
                world,
                relative=config.resolved["score"]["relative"],
                absolute=config.resolved["score"]["absolute"],
            )
            arrays = _arrays(fit_path, ("truth", "mean", "lower95", "upper95", "observed"))
            terms, estimate = record["terms"], arrays["mean"]
            lower, upper = arrays["lower95"], arrays["upper95"]
            status = "valid" if saved["diagnostics_pass"] is True else "unresolved"
            diagnostics = {
                "algorithm": "NUTS",
                "diagnostics_pass": saved["diagnostics_pass"],
                **{
                    k: record.get(k)
                    for k in (
                        "divergences",
                        "maxdepth_hits",
                        "max_rhat",
                        "min_bulk_ess",
                        "min_tail_ess",
                        "min_bfmi",
                    )
                },
            }
            uncertainty = {
                "type": "95% pointwise equal-tail posterior intervals",
                "channel_pointwise_coverage": float(
                    np.mean(
                        (arrays["truth"][:, 1:] >= lower[:, 1:])
                        & (arrays["truth"][:, 1:] <= upper[:, 1:])
                    )
                ),
            }
        else:
            saved = saved_bounded_world(
                directory,
                world,
                relative=config.resolved["score"]["relative"],
                absolute=config.resolved["score"]["absolute"],
            )
            arrays = _arrays(fit_path, ("truth", "fit_bounded_estimated_components"))
            terms, estimate = record["terms"], arrays["fit_bounded_estimated_components"]
            lower = upper = None
            status = "valid" if saved["optimizer_converged"] is True else "unresolved"
            diagnostics = {
                "algorithm": "bounded estimated-transform nonlinear least squares",
                "optimizer_converged": saved["optimizer_converged"],
            }
            uncertainty = {"type": "unavailable (point optimizer only)"}
        actual_patterns = pattern_by_world.get(world, ())
        wanted = {("media" if name.startswith("C") else "control", name) for name in terms[1:]}
        if (
            len(actual_patterns) != len(wanted)
            or {(row["kind"], row["channel"]) for row in actual_patterns} != wanted
        ):
            raise ContractError(f"legacy world {world} pattern/channel inventory mismatch")
        for pattern in actual_patterns:
            channel_kind, index = pattern["kind"], int(pattern["channel"][1:]) - 1
            flags = corpus[
                "treatment_components" if channel_kind == "media" else "covariate_components"
            ][local, index]
            if any(
                bool(flags[j]) != (pattern.get(name) is True)
                for j, name in enumerate(TRAJECTORY_COMPONENTS)
            ):
                raise ContractError(
                    f"legacy world {world} pattern flags disagree with source corpus"
                )
            if pattern.get("trend") is True:
                change = corpus[
                    "trajectory_treatment_trend_change"
                    if channel_kind == "media"
                    else "trajectory_covariate_trend_change"
                ][local, index]
                direction = "up" if change > 0 else "down" if change < 0 else "flat"
                if pattern.get("trend_direction") != direction:
                    raise ContractError(
                        f"legacy world {world} trend direction disagrees with source corpus"
                    )
        truth = arrays["truth"]
        score = saved["score"]
        channels.extend(
            {"world_id": world, **row}
            for row in _pattern_rows(
                corpus, local, terms, truth, score, status, pattern_by_world[world]
            )
        )
        start = config.resolved["study"]["score_weeks"][0]
        if truth.shape[0] != config.resolved["study"]["score_weeks"][1] - start + 1:
            raise ContractError("legacy score window mismatch")
        trajectories[str(world)] = {
            "terms": terms,
            "weeks": list(range(start, start + len(truth))),
            "truth": truth.tolist(),
            "fitted": estimate.tolist(),
            "lower95": lower.tolist() if lower is not None else None,
            "upper95": upper.tolist() if upper is not None else None,
            "raw_spend": np.asarray(corpus["treatment_raw"][local, start:, :], float)[
                :, : sum(n.startswith("C") for n in terms)
            ].tolist(),
        }
        all_rows.append(
            {
                "world_id": world,
                "computational_status": status,
                "diagnostics": diagnostics,
                "recovery": {
                    "media_pass": score["media_pass"] if status == "valid" else None,
                    "control_pass": score["controls_pass"] if status == "valid" else None,
                    "joint_pass": score["joint_pass"] if status == "valid" else None,
                    "total_media": score["total_media"] if status == "valid" else None,
                    "baseline": score["baseline"] if status == "valid" else None,
                },
                "uncertainty": uncertainty,
                "profile": _profiles(
                    corpus,
                    local,
                    start,
                    truth,
                    sum(n.startswith("C") for n in terms),
                    sum(n.startswith("Z") for n in terms),
                ),
            }
        )
    return all_rows, channels, trajectories, inputs, PROVENANCE_LEGACY


def _recorded_stage(path: Path, *, stage: str, config: StudyConfig, world: int) -> dict:
    manifest = _json(path / "manifest.json")
    if manifest.get("stage") != stage or manifest.get("world_id") != world:
        raise ContractError("historical stage/world identity mismatch")
    expected = {
        k: config.resolved[k]
        for k in (
            ("study", "generator", "pairing")
            if stage == "generate"
            else ("study", "generator", "pairing", "methods")
        )
    }
    if manifest.get("config") != expected:
        raise ContractError("historical stage config differs from request")
    if stage == "fit":
        source = path.parents[2] / "generate" / f"world_{world:02d}"
        expected_inputs = {
            "dataset.npz": source / "dataset.npz",
            "dataset-manifest.json": source / "manifest.json",
            "dataset-completion.json": source / "completion.json",
            "dataset-metadata.json": source / "metadata.json",
        }
        if manifest.get("inputs") != {name: digest(file) for name, file in expected_inputs.items()}:
            raise ContractError(
                "historical fit's recorded generation inputs disagree with saved bytes"
            )
    if not checkpoint(path, manifest).complete:
        raise ContractError("historical stage incomplete; no relaxed receipt policy")
    return manifest


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
            "report method subset must be nonempty, distinct and enabled in the original fit config"
        )
    return methods


def _fit_labels(config: StudyConfig, methods: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    labels = []
    for method in methods:
        if method == "advi":
            labels.extend(
                (f"advi_{i + 1}", method, i)
                for i in range(len(config.resolved["methods"]["advi"]["seeds"]))
            )
        else:
            labels.append((method, method, 0))
    return tuple(labels)


def _current_fit_identity(
    config: StudyConfig, root: Path, world: int, method: str, replicate: int
) -> dict:
    """Same deterministic fit identity as runner.fit; NEVER writes a stage."""
    from .runner import _sources

    source = root / "generate" / f"world_{world:02d}"
    seed = (
        config.resolved["methods"]["advi"]["seeds"][replicate] + world
        if method == "advi"
        else config.resolved["methods"][method]["seed"] + world
    )
    ident = identity(
        "fit",
        config,
        source_files=_sources(
            "config.py", "dgp.py", "models.py", "methods.py", "runner.py", "artifacts.py"
        ),
        dataset_files={
            "dataset.npz": source / "dataset.npz",
            "dataset-manifest.json": source / "manifest.json",
            "dataset-completion.json": source / "completion.json",
            "dataset-metadata.json": source / "metadata.json",
        },
        expected_outputs={"report.json", "components.npz" if method == "ols" else "posterior.nc"},
    )
    ident.update(world_id=world, method=method, replicate=replicate, method_seed=seed)
    return ident


def _preflight_stages(
    config: StudyConfig, root: Path, labels: tuple[tuple[str, str, int], ...], *, historical: bool
):
    """All selected stage identities/receipts checked before reading outputs or writing report."""
    from .runner import _world_identity

    if config.resolved["study"]["worlds"] % 2:
        raise ContractError(
            "native/historical per-world report requires an even generated world count"
        )
    inputs, status_current = {}, set()
    for world in range(config.resolved["study"]["worlds"]):
        source = root / "generate" / f"world_{world:02d}"
        if historical:
            _recorded_stage(source, stage="generate", config=config, world=world)
            try:
                checkpoint(source, _world_identity(config, world))
            except ContractError as exc:
                if "stale" not in str(exc):
                    raise
                status_current.add("stale for current-source fitting")
            else:
                status_current.add("current-source compatible")
        elif not checkpoint(source, _world_identity(config, world)).complete:
            raise ContractError(
                f"current generation world {world} has no complete qualified receipt"
            )
        meta = _json(source / "metadata.json")
        if (
            meta.get("world_id") != world
            or meta.get("local_world") != world % 2
            or meta.get("generation_seed") != config.resolved["study"]["seed"] + world // 2
        ):
            raise ContractError("generation metadata does not match selected world/seed")
        for name in ("manifest.json", "completion.json", "dataset.npz", "metadata.json"):
            inputs[f"generate-{world:02d}-{name}"] = source / name
        for label, method, replicate in labels:
            fit = root / "fit" / f"world_{world:02d}" / label
            configured_seed = (
                config.resolved["methods"]["advi"]["seeds"][replicate]
                if method == "advi"
                else config.resolved["methods"][method]["seed"] + world
            )
            effective_seed = configured_seed + world if method == "advi" else configured_seed
            if historical:
                manifest = _recorded_stage(fit, stage="fit", config=config, world=world)
                if (
                    manifest.get("method") != method
                    or manifest.get("replicate") != replicate
                    or manifest.get("method_seed") != effective_seed
                ):
                    raise ContractError("historical fit method/replicate/seed identity mismatch")
            elif not checkpoint(
                fit, _current_fit_identity(config, root, world, method, replicate)
            ).complete:
                raise ContractError(
                    f"current fit world {world} {label} has no complete qualified receipt"
                )
            report = _json(fit / "report.json")
            if (
                report.get("world_id") != world
                or report.get("replicate") != replicate
                or report.get("effective_seed") != effective_seed
            ):
                raise ContractError("fit report world/replicate/seed differs from sealed identity")
            for name in (
                "manifest.json",
                "completion.json",
                "report.json",
                "components.npz" if method == "ols" else "posterior.nc",
            ):
                inputs[f"fit-{world:02d}-{label}-{name}"] = fit / name
    return inputs, status_current


def _per_world(config: StudyConfig, root: Path, *, historical: bool, methods: tuple[str, ...]):
    import arviz as az

    labels = _fit_labels(config, methods)
    inputs, status_current = _preflight_stages(config, root, labels, historical=historical)
    all_rows, channels, trajectories = [], [], {}
    for world in range(config.resolved["study"]["worlds"]):
        source = root / "generate" / f"world_{world:02d}"
        with np.load(source / "dataset.npz", allow_pickle=False) as archive:
            corpus = {key: archive[key].copy() for key in archive.files if key != "diagnostics"}
        local = world % 2
        start, stop = config.resolved["study"]["score_weeks"]
        n_media, n_controls = (
            int(corpus["n_treatments_active"][local]),
            int(corpus["n_covariates_active"][local]),
        )
        names = (
            ["Baseline"]
            + [f"C{i + 1}" for i in range(n_media)]
            + [f"Z{i + 1}" for i in range(n_controls)]
        )
        truth = np.column_stack(
            [
                corpus["baseline_intrinsic"][local, start : stop + 1],
                corpus["treatment_contribution_raw"][local, start : stop + 1, :n_media],
                corpus["covariate_contribution"][local, start : stop + 1, :n_controls],
            ]
        ).astype(float)
        for label, method, replicate in labels:
            fit = root / "fit" / f"world_{world:02d}" / label
            report = _json(fit / "report.json")
            if label == "ols":
                estimate = _arrays(fit / "components.npz", ("components",))["components"][
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
                posterior = dt["posterior"]
                media = posterior["media_path"].values[:, :, :, :n_media]
                control = posterior["control_path"].values[:, :, :, :n_controls]
                baseline = (
                    posterior["baseline_path"].values
                    if "baseline_path" in posterior
                    else np.broadcast_to(posterior["intercept"].values[..., None], media.shape[:3])
                )
                draws = np.concatenate((baseline[..., None], media, control), axis=-1).reshape(
                    -1, len(truth), len(names)
                )
                if not np.isfinite(draws).all():
                    status = "unresolved"
                else:
                    status = (
                        "finite_only_not_converged" if label.startswith("advi") else "unresolved"
                    )
                estimate = draws.mean(axis=0)
                lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
                uncertainty = {
                    "type": "variational draw spread, not calibrated posterior"
                    if label.startswith("advi")
                    else "95% pointwise equal-tail posterior intervals; only interpret after diagnostics",
                    "channel_pointwise_coverage": float(
                        np.mean((truth[:, 1:] >= lower[:, 1:]) & (truth[:, 1:] <= upper[:, 1:]))
                    ),
                }
                if label.startswith("advi"):
                    diagnostic = {
                        "algorithm": "full-rank ADVI",
                        "finite_loss": report.get("loss_finite"),
                        "posterior_finite": report.get("posterior_finite"),
                        "rhat_ess": "not applicable",
                        "seed_sensitivity": "compare paired seeds; finite loss is not convergence",
                    }
                else:
                    scalar = [
                        key
                        for key in (
                            "intercept",
                            "sigma_relative",
                            "baseline_sd_relative",
                            "alpha",
                            "log_kappa",
                            "amplitude",
                            "control_reference_contribution",
                        )
                        if key in posterior
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
                    div = int(stats["diverging"].values.sum())
                    depth = int(stats["maxdepth_reached"].values.sum())
                    gate = config.resolved["diagnostics"]
                    healthy = (
                        div <= gate["divergences"]
                        and depth <= gate["maxdepth_hits"]
                        and rhat < gate["rhat"]
                        and bulk >= gate["ess"]
                        and tail >= gate["ess"]
                        and bfmi >= gate["bfmi"]
                    )
                    diagnostic = {
                        "algorithm": "Nutpie draw_diag",
                        "divergences": div,
                        "maxdepth_hits": depth,
                        "max_rhat_screen": rhat,
                        "min_bulk_ess_screen": bulk,
                        "min_tail_ess_screen": tail,
                        "min_bfmi": bfmi,
                        "diagnostics_pass": bool(healthy),
                        "screened_variables": scalar,
                    }
                    status = "valid" if healthy and np.isfinite(draws).all() else "unresolved"
                dt.close()
            if estimate.shape != truth.shape:
                raise ContractError(
                    "historical fit components do not match recorded truth dimensions"
                )
            score = score_components(
                truth,
                estimate,
                names,
                relative=config.resolved["score"]["relative"],
                absolute=config.resolved["score"]["absolute"],
            )
            channel_rows = _pattern_rows(corpus, local, names, truth, score, status)
            channels.extend({"world_id": world, "method": label, **row} for row in channel_rows)
            trajectories[f"{world}:{label}"] = {
                "terms": names,
                "weeks": list(range(start, stop + 1)),
                "truth": truth.tolist(),
                "fitted": estimate.tolist(),
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
                    "profile": _profiles(corpus, local, start, truth, n_media, n_controls),
                }
            )
        advi = [trajectories[f"{world}:{label}"] for label, method, _ in labels if method == "advi"]
        if len(advi) >= 2:
            truth = np.asarray(advi[0]["truth"])
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
        (
            {
                "type": "sealed historical smoke: recorded identity/output receipts verified",
                "current_source_status": sorted(status_current),
                "warning": "Historical fits are NOT current-source-qualified; no inference/resume allowed with stale identity",
            }
            if historical
            else {
                "type": "current-run: generation and selected fit identities/output receipts verified against CURRENT sources",
                "current_source_status": ["current-source qualified"],
                "warning": "Qualification certifies artifact compatibility, NOT sampling convergence or scientific recovery",
            }
        ),
    )


def _bundle(
    config: StudyConfig, source: Path, kind: str, *, historical: bool, methods: tuple[str, ...]
):
    if kind in ("tv_nuts", "bounded_ols"):
        if historical:
            raise ContractError("--historical is only for sealed smoke, not legacy evidence")
        return _legacy(config, source, kind)
    if kind == "smoke":
        if not historical:
            raise ContractError(
                "frozen smoke requires explicit --historical; normal compatibility is strict"
            )
        return _per_world(config, source, historical=True, methods=methods)
    if kind == "native":
        if historical:
            raise ContractError("native/current-run reporting forbids --historical")
        return _per_world(config, source, historical=False, methods=methods)
    raise ContractError(f"unsupported report format {kind}")


def _identity(config: StudyConfig, inputs: dict[str, Path], methods: tuple[str, ...]):
    from importlib.metadata import version

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
        for name in ("nbclient", "nbformat", "ipykernel", "plotly", "ipywidgets", "pandas")
    }
    return ident


def create_report(
    config: StudyConfig,
    source: Path,
    kind: str,
    out: Path,
    *,
    historical: bool = False,
    methods: tuple[str, ...] | None = None,
) -> dict:
    """Create an isolated, verifiable report; no fitting/generation calls."""
    source, out = source.resolve(), out.resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ContractError("report output must not contain or overwrite source artifacts")
    if out.exists():
        raise ContractError(
            "report output must be a new, empty path; never overwrite prior interpretation"
        )
    if kind in ("tv_nuts", "bounded_ols"):
        if methods is not None:
            raise ContractError(
                "legacy reports use their one recorded method; method selection is unsupported"
            )
        selected = ("mcmc",) if kind == "tv_nuts" else ("ols",)
    else:
        selected = _requested_methods(config, methods)
    rows, channels, trajectories, inputs, provenance = _bundle(
        config, source, kind, historical=historical, methods=selected
    )
    ident = _identity(config, inputs, selected)
    run_id = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    pattern_rates = {
        method: _channels([row for row in channels if row.get("method", kind) == method])
        for method in sorted({row.get("method", kind) for row in channels})
    }
    from .report_notebook import notebook

    payload = {
        "schema": 1,
        "run_id": run_id,
        "kind": kind,
        "report_methods": list(selected),
        "provenance": provenance,
        "resolved_config": config.resolved,
        "score_gate": config.resolved["score"],
        "diagnostic_gate": config.resolved["diagnostics"],
        "worlds": rows,
        "channels": channels,
        "pattern_rates": pattern_rates,
        "trajectories": trajectories,
        "input_paths": {
            key: __import__("os").path.relpath(path, out) for key, path in inputs.items()
        },
        "capabilities": {
            "optional_logistic_screen": "deferred/unsupported",
            "optional_tsne": "deferred/unsupported",
            "optional_correlated_pairs": "deferred/unsupported",
        },
    }
    text = (
        "# Interpretation · pending review\n\n"
        f"Run identity: `{run_id}`\n\n"
        "No scientific conclusion is inherited from another dataset or run. Review the diagnostics, "
        "pattern-channel denominators, point gates and uncertainty in this report before writing "
        "a run-specific interpretation. Any changed input/config/source requires a NEW report identity.\n"
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
        "pattern_methods": list(pattern_rates),
        "provenance": provenance,
        "report": str(out / "report.ipynb"),
    }


def read_report(out: Path) -> dict:
    """Run-All guard: rehash original inputs, current report sources AND receipt."""
    out = out.resolve()
    payload = _json(out / "evidence.json")
    recorded = _json(out / "manifest.json")
    config = StudyConfig(recorded["config"])
    resolved = payload.get("resolved_config")
    if (
        not isinstance(resolved, dict)
        or resolved.get("schema") != 1
        or {key: resolved.get(key) for key in config.resolved} != config.resolved
    ):
        raise ContractError("report resolved config differs from recorded identity")
    methods = tuple(payload.get("report_methods", ()))
    if (
        not methods
        or any(type(name) is not str for name in methods)
        or len(set(methods)) != len(methods)
        or set(methods) - set(config.resolved["methods"]["enabled"])
    ):
        raise ContractError("invalid recorded report method selection")
    inputs = {key: (out / value).resolve() for key, value in payload["input_paths"].items()}
    expected = _identity(config, inputs, methods)
    if checkpoint(out, expected).state != "complete":
        raise ContractError("report evidence is incomplete")
    run_id = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if payload["run_id"] != run_id or f"`{run_id}`" not in (out / "interpretation.md").read_text():
        raise ContractError("interpretation/run identity mismatch")
    return payload
