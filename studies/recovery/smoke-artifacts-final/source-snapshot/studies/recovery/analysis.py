"""Read-only, saved-array scoring. No notebook imports or fitting dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import digest
from .config import ContractError, StudyConfig


def _legacy_compatible(directory: Path, kind: str, config: StudyConfig) -> None:
    """Validate claims supported by the existing manifests, not inferred from 97 rows.

    These old manifests do not prove source-file or per-result output hashes; this
    is a compatibility check, not certification as a new-style completed checkpoint.
    """
    study = config.resolved["study"]
    generator = config.resolved["generator"]
    methods = config.resolved["methods"]
    needed_method = "mcmc" if kind == "tv_nuts" else "ols"
    if kind == "tv_nuts" and config.resolved["diagnostics"] != {
        "rhat": 1.01,
        "ess": 400,
        "bfmi": 0.3,
        "divergences": 0,
        "maxdepth_hits": 0,
    }:
        raise ContractError(
            "legacy diagnostics_pass uses fixed MCMC thresholds; custom diagnostics need fresh analysis"
        )
    if kind == "tv_nuts":
        if study["baseline"] != "varying":
            raise ContractError("tv_nuts requires varying baseline")
        base_dir = directory.parent.parent / "ols_study_50_seed10029"
    else:
        if study["baseline"] != "constant":
            raise ContractError("bounded_ols requires constant baseline")
        base_dir = directory.parent / "ols_study_50_seed10029"
    if needed_method not in methods["enabled"]:
        raise ContractError(f"legacy {kind} requires enabled method {needed_method}")

    def manifest(path: Path) -> dict:
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ContractError(f"missing/invalid legacy manifest: {path}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"invalid legacy manifest table: {path}")
        return value

    base = manifest(base_dir / "manifest.json")
    saved = manifest(directory / "manifest.json")
    if kind == "tv_nuts":
        data = saved.get("data_manifest")
        if data != manifest(directory.parent / "data_manifest.json"):
            raise ContractError("TV data manifest differs from saved fit manifest")
        if not isinstance(data, dict):
            raise ContractError("TV data manifest is missing")
    else:
        data = saved

    def equal(field: str, actual: object, wanted: object) -> None:
        if actual != wanted:
            raise ContractError(f"legacy {field} mismatch: saved {actual!r}, requested {wanted!r}")

    equal("seed", base.get("first_seed"), study["seed"])
    for field, supported in (("worlds", 50), ("cells_per_batch", 2), ("draws_per_cell", 1)):
        if type(base.get(field)) is not int or base[field] != supported:
            raise ContractError(f"legacy base {field} differs from supported 50-world recipe")
    if study["worlds"] > base["worlds"]:
        raise ContractError("legacy worlds exceed recorded population")
    base_cfg = base.get("config")
    if not isinstance(base_cfg, dict):
        raise ContractError("legacy generator config missing")
    for field in (
        "n_time_steps",
        "n_treatments",
        "n_covariates",
        "l_max",
        "n_treatments_active_range",
        "n_covariates_active_range",
        "edge_budget",
    ):
        equal(field, base_cfg.get(field), generator[field])
    equal(
        "carryover_family",
        base_cfg.get("carryover_family_probs"),
        {"none": 0.0, "geometric": 1.0, "weibull": 0.0},
    )
    saturation = base_cfg.get("saturation_family_probs")
    if (
        not isinstance(saturation, dict)
        or saturation.get("michaelis_menten") != 1
        or any(v != 0 for k, v in saturation.items() if k != "michaelis_menten")
    ):
        raise ContractError("legacy saturation family is not exclusively Michaelis–Menten")
    if "rw_smoothness_beta" in generator:
        equal(
            "rw_smoothness_beta",
            base_cfg.get("rw_smoothness_beta"),
            generator["rw_smoothness_beta"],
        )
    if "media_smoothness_beta" in generator or config.resolved["pairing"]["mode"] != "independent":
        raise ContractError(
            "legacy manifest cannot establish requested media-only smoothness/pairing provenance"
        )
    equal("reported_weeks", study["reported_weeks"], [0, base_cfg["n_time_steps"] - 1])
    equal("score_weeks", study["score_weeks"], [7, 103])
    equal("burn_in", study["burn_in"], 8)
    if kind == "tv_nuts":
        for field in (
            "rw_smoothness_alpha",
            "rw_smoothness_beta",
            "rw_smoothness_max_weeks",
            "confounding_strength_range",
            "baseline_floor",
        ):
            if field not in base_cfg:
                raise ContractError(f"legacy base {field} metadata missing")
        for field in (
            "smoothness_beta_prior",
            "max_smoothing_width",
            "confounding_strength",
            "baseline_floor",
            "seeds",
        ):
            if field not in data:
                raise ContractError(f"legacy TV {field} metadata missing")
        # This is the recorded legacy recipe, not a general claim that any
        # jointly edited pair of old manifests proves its own provenance.
        equal("rw_smoothness_alpha", base_cfg.get("rw_smoothness_alpha"), 2.0)
        equal("rw_smoothness_beta", base_cfg.get("rw_smoothness_beta"), 2.0)
        equal("rw_smoothness_max_weeks", base_cfg.get("rw_smoothness_max_weeks"), 26)
        equal("confounding_strength_range", base_cfg.get("confounding_strength_range"), None)
        equal("base baseline_floor", base_cfg.get("baseline_floor"), None)
        equal(
            "smoothness_beta_prior",
            data.get("smoothness_beta_prior"),
            [base_cfg["rw_smoothness_alpha"], base_cfg["rw_smoothness_beta"]],
        )
        equal(
            "max_smoothing_width",
            data.get("max_smoothing_width"),
            base_cfg["rw_smoothness_max_weeks"],
        )
        equal("confounding_strength", data.get("confounding_strength"), 0)
        equal("baseline_floor", data.get("baseline_floor"), base_cfg["baseline_floor"])
        equal("TV generation seeds", data.get("seeds"), list(range(91001, 91001 + base["worlds"])))
        equal(
            "rw_baseline_std_range",
            data.get("baseline_std_range"),
            generator["rw_baseline_std_range"],
        )
        equal(
            "reported_window",
            data.get("reported_window"),
            [study["burn_in"], study["burn_in"] + generator["n_time_steps"]],
        )
        equal(
            "fit_window",
            data.get("fit_window"),
            [
                study["burn_in"] + study["score_weeks"][0],
                study["burn_in"] + study["score_weeks"][1] + 1,
            ],
        )
        equal(
            "full_horizon", data.get("full_horizon"), study["burn_in"] + generator["n_time_steps"]
        )
        for key, name in (
            ("chains", "chains"),
            ("tune", "warmup"),
            ("draws", "draws"),
            ("target_accept", "target_accept"),
            ("maxdepth", "maxdepth"),
        ):
            equal(name, saved.get(key), methods["mcmc"][name])
        equal("sampler", saved.get("sampler"), "nutpie")
        equal("adaptation", saved.get("adaptation"), "draw_diag")
        equal(
            "MCMC seeds",
            saved.get("seeds"),
            list(range(methods["mcmc"]["seed"], methods["mcmc"]["seed"] + base["worlds"])),
        )
    else:
        equal(
            "rw_baseline_std_range",
            base_cfg.get("rw_baseline_std_range"),
            generator["rw_baseline_std_range"],
        )
        equal("fit_weeks", saved.get("fit_weeks"), study["score_weeks"])
        for field, key in (
            ("starts", "starts"),
            ("max_nfev", "max_nfev_per_start"),
            ("seed", "seed"),
        ):
            equal(f"OLS {field}", saved.get(key), methods["ols"][field])
    source = data.get("source_sha256" if kind == "tv_nuts" else "data_sha256")
    batch_size = base["cells_per_batch"] * base["draws_per_cell"]
    if base["worlds"] % batch_size:
        raise ContractError("legacy base batch inventory is not divisible by cells and draws")
    expected_batches = {f"batch_{index:02d}.npz" for index in range(base["worlds"] // batch_size)}
    if not isinstance(source, dict) or set(source) != expected_batches:
        raise ContractError(
            "legacy dataset provenance inventory must contain exactly the recorded base batches"
        )
    if any(
        not isinstance(sha, str)
        or len(sha) != 64
        or any(char not in "0123456789abcdef" for char in sha)
        for sha in source.values()
    ):
        raise ContractError("legacy dataset provenance inventory contains malformed SHA256 digests")
    for name in sorted(expected_batches):
        path = base_dir / name
        if not path.is_file() or digest(path) != source[name]:
            raise ContractError(f"legacy dataset provenance hash mismatch: {path}")


def score_components(
    truth: np.ndarray,
    estimate: np.ndarray,
    terms: list[str],
    *,
    relative: float = 0.1,
    absolute: float = 0.01,
) -> dict:
    """Channel and aggregate L2 gates, excluding baseline from joint effect score."""
    if (
        truth.ndim != 2
        or estimate.shape != truth.shape
        or truth.shape[1] != len(terms)
        or not terms
        or terms[0] not in ("Intercept", "Baseline")
    ):
        raise ContractError("component shape/terms mismatch")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ContractError("nonfinite contribution")
    noiseless_norm = np.linalg.norm(truth.sum(axis=1))

    def gate(actual: np.ndarray, predicted: np.ndarray) -> dict:
        allowance = max(relative * np.linalg.norm(actual), absolute * noiseless_norm)
        error = float(np.linalg.norm(actual - predicted))
        return {"error": error, "allowance": float(allowance), "pass": bool(error <= allowance)}

    channels = {name: gate(truth[:, i], estimate[:, i]) for i, name in enumerate(terms) if i > 0}
    media = [name for name in channels if name.startswith("C")]
    controls = [name for name in channels if name.startswith("Z")]
    if len(media) + len(controls) != len(channels):
        raise ContractError("unexpected term vocabulary")
    return {
        "channels": channels,
        "media_pass": all(channels[n]["pass"] for n in media),
        "controls_pass": all(channels[n]["pass"] for n in controls),
        "joint_pass": all(v["pass"] for v in channels.values()),
        "total_media": gate(
            truth[:, [terms.index(n) for n in media]].sum(axis=1),
            estimate[:, [terms.index(n) for n in media]].sum(axis=1),
        ),
        "baseline": gate(truth[:, 0], estimate[:, 0]),
        "media_count": len(media),
        "control_count": len(controls),
        "nonzero_media": [n for n in media if np.linalg.norm(truth[:, terms.index(n)]) > 0],
        "nonzero_controls": [n for n in controls if np.linalg.norm(truth[:, terms.index(n)]) > 0],
    }


def _load_arrays(path: Path, keys: tuple[str, ...]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys) <= set(archive.files):
            raise ContractError(f"missing arrays in {path}: {set(keys) - set(archive.files)}")
        return {key: archive[key].copy() for key in keys}


def saved_tv_world(
    directory: Path,
    world: int,
    *,
    label: str = "nuts",
    relative: float = 0.1,
    absolute: float = 0.01,
) -> dict:
    """Recompute point gates from saved paths; never trust precomputed gate flags."""
    report = json.loads((directory / f"world_{world:02d}_{label}_report.json").read_text())
    if report["world_id"] != world or report["method"] != label:
        raise ContractError("TV report/world identity mismatch")
    arrays = _load_arrays(
        directory / f"world_{world:02d}_{label}_recovery.npz",
        ("truth", "mean", "noiseless", "total_media_truth", "total_media_mean"),
    )
    truth, mean = arrays["truth"], arrays["mean"]
    if truth.shape[0] != 97:
        raise ContractError("TV saved fit window must contain exactly 97 weeks")
    if not np.allclose(truth.sum(axis=1), arrays["noiseless"], rtol=1e-6, atol=1e-5):
        raise ContractError("TV noiseless trajectory disagrees with truth components")
    score = score_components(truth, mean, report["terms"], relative=relative, absolute=absolute)
    media_indices = [i for i, name in enumerate(report["terms"]) if name.startswith("C")]
    if not np.allclose(
        truth[:, media_indices].sum(axis=1), arrays["total_media_truth"], atol=1e-6
    ) or not np.allclose(mean[:, media_indices].sum(axis=1), arrays["total_media_mean"], atol=1e-6):
        raise ContractError("TV total-media trajectory disagrees with components")
    return {"world_id": world, "diagnostics_pass": report["diagnostics_pass"], "score": score}


def saved_bounded_world(
    directory: Path, world: int, *, relative: float = 0.1, absolute: float = 0.01
) -> dict:
    record = json.loads((directory / f"world_{world:02d}.json").read_text())
    if record["world_id"] != world:
        raise ContractError("bounded report/world identity mismatch")
    arrays = _load_arrays(
        directory / f"world_{world:02d}_fits.npz", ("truth", "fit_bounded_estimated_components")
    )
    if arrays["truth"].shape[0] != 97:
        raise ContractError("bounded saved fit window must contain exactly 97 weeks")
    score = score_components(
        arrays["truth"],
        arrays["fit_bounded_estimated_components"],
        record["terms"],
        relative=relative,
        absolute=absolute,
    )
    return {
        "world_id": world,
        "optimizer_converged": record["fits"]["bounded_estimated"]["success"],
        "score": score,
    }


def saved_summary(directory: Path, *, kind: str, config: StudyConfig) -> dict:
    if kind not in ("tv_nuts", "bounded_ols"):
        raise ContractError("unknown saved-result format")
    _legacy_compatible(directory, kind, config)
    worlds = config.resolved["study"]["worlds"]
    relative = config.resolved["score"]["relative"]
    absolute = config.resolved["score"]["absolute"]
    rows = [
        saved_tv_world(directory, w, relative=relative, absolute=absolute)
        if kind == "tv_nuts"
        else saved_bounded_world(directory, w, relative=relative, absolute=absolute)
        for w in range(worlds)
    ]
    valid_key = "diagnostics_pass" if kind == "tv_nuts" else "optimizer_converged"
    return {
        "kind": kind,
        "provenance": "legacy manifest/input-hash compatibility only; output and source hashes unavailable",
        "worlds": worlds,
        "valid": sum(r[valid_key] is True for r in rows),
        "unresolved": sum(r[valid_key] is not True for r in rows),
        "media_pass": sum(r["score"]["media_pass"] and r[valid_key] is True for r in rows),
        "joint_pass": sum(r["score"]["joint_pass"] and r[valid_key] is True for r in rows),
        "total_media_pass": sum(
            r["score"]["total_media"]["pass"] and r[valid_key] is True for r in rows
        ),
        "media_channel_failures": sum(
            sum(not r["score"]["channels"][name]["pass"] for name in r["score"]["nonzero_media"])
            for r in rows
            if r[valid_key] is True
        ),
        "media_channels_evaluable": sum(
            len(r["score"]["nonzero_media"]) for r in rows if r[valid_key] is True
        ),
        "control_channel_failures": sum(
            sum(not r["score"]["channels"][name]["pass"] for name in r["score"]["nonzero_controls"])
            for r in rows
            if r[valid_key] is True
        ),
        "control_channels_evaluable": sum(
            len(r["score"]["nonzero_controls"]) for r in rows if r[valid_key] is True
        ),
    }
