"""Deterministic matched-media fixture replay from supplied primitive RV snapshots.

No sampler, RNG or notebook execution. This capability checks two existing worlds;
new stochastic matched-population generation is deliberately not implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytensor
from scipy.stats import beta as beta_distribution
from threadpoolctl import threadpool_limits

import pymc_generator as pg
from pymc_generator.sampler import _slice_g_active
from pymc_generator.slots import SlotLayout
from pymc_generator.world_model import build_world_model

from .artifacts import checkpoint, complete_checkpoint, digest, identity, write_manifest
from .config import ContractError, StudyConfig

PROTECTED = (
    "covariates",
    "covariate_contribution",
    "baseline_intrinsic",
    "outcome_noise",
    "saturation_scale",
    "latent_unobserved",
    "confounding_strength",
)
OUTPUTS = ("treatments", "contributions", *PROTECTED)


def _structure(value: dict) -> dict:
    return {key: np.asarray(array) for key, array in value.items()}


def _evaluate(graph: dict, cfg: pg.SCMPrior, structure: dict, fixed: dict) -> dict:
    with threadpool_limits(limits=1):
        model, outputs, _ = build_world_model(
            graph, cfg, structure, cfg.n_time_steps, prior_cond=None
        )
        names = tuple(name for name in OUTPUTS if name in outputs)
        rv_names = {rv.name for rv in model.free_RVs}
        if rv_names != set(fixed) or set(names) != set(OUTPUTS):
            raise ContractError(
                "fixed-RV or protected-output inventory differs from generated graph"
            )
        evaluate = pytensor.function(
            model.free_RVs,
            [model[name] for name in names],
            mode="FAST_COMPILE",
            on_unused_input="ignore",
        )
        return dict(zip(names, evaluate(*[fixed[rv.name] for rv in model.free_RVs])))


def replay_world(config: StudyConfig, source: Path, paired: Path, world: int) -> dict:
    """Rebuild BOTH worlds from identical primitive RV arrays, verify observed paired target."""
    pair = config.resolved["pairing"]
    if pair["mode"] != "matched":
        raise ContractError("matched replay requires pairing.mode=matched")
    if Path(pair["coupling_manifest"]).resolve() != (paired / "manifest.json").resolve():
        raise ContractError("coupling_manifest must identify the supplied paired manifest")
    base_manifest = json.loads((source / "manifest.json").read_text())
    pair_manifest = json.loads((paired / "manifest.json").read_text())
    if base_manifest["config"] != pair_manifest["source_config"]:
        raise ContractError("matched source generator configuration differs")
    if (
        base_manifest["first_seed"] != config.resolved["study"]["seed"]
        or world < 0
        or world >= base_manifest["worlds"]
    ):
        raise ContractError("matched world outside the recorded source population/seed")
    if pair_manifest["media_smoothness_new"] != [2, pair["media_smoothness_beta"]] or pair_manifest[
        "media_smoothness_old"
    ] != [2, 2]:
        raise ContractError("matched smoothing quantile coupling differs")
    batch = world // base_manifest["cells_per_batch"]
    local = world % base_manifest["cells_per_batch"]
    batch_path = source / f"batch_{batch:02d}.npz"
    matching_hash = next(
        (
            sha
            for path, sha in pair_manifest["source_hashes"].items()
            if Path(path).name == batch_path.name and Path(path).parent.name == source.name
        ),
        None,
    )
    if matching_hash != digest(batch_path):
        raise ContractError("matched source batch hash differs from recorded input")
    with np.load(paired / f"world_{world:02d}_fixed_rvs.npz", allow_pickle=False) as rv_file:
        fixed = {key: rv_file[key].copy() for key in rv_file.files}
    with np.load(paired / f"world_{world:02d}_paired.npz", allow_pickle=False) as pair_file:
        saved = {key: pair_file[key].copy() for key in pair_file.files}
    report = json.loads((paired / f"world_{world:02d}_report.json").read_text())
    if report["world_id"] != world or report["source_batch"] != batch or not fixed:
        raise ContractError("paired fixture world/primitive-RV inventory mismatch")
    original_structure = _structure(report["structure_original"])
    new_structure = _structure(report["structure_less_smooth"])
    if set(original_structure) != set(new_structure):
        raise ContractError("matched structural inventory differs")
    for name in original_structure:
        if name != "smoothness_c" and not np.array_equal(
            original_structure[name], new_structure[name]
        ):
            raise ContractError(f"matched protected structural parameter changed: {name}")
    proposed = beta_distribution.ppf(
        beta_distribution.cdf(original_structure["smoothness_c"], 2, 2),
        2,
        pair["media_smoothness_beta"],
    )
    np.testing.assert_array_equal(new_structure["smoothness_c"], proposed)
    corpus = pg.load_corpus(batch_path)
    cfg = pg.SCMPrior(**base_manifest["config"])
    if corpus.get("prior_cond") is not None:
        raise ContractError("conditioned prior replay is not supported by this fixed-RV adapter")
    graph_full = {
        f"g_{key}": val
        for key, val in SlotLayout(cfg.n_treatments, cfg.n_covariates, cfg.n_latent)
        .unpack(corpus["g"][local])
        .items()
    }
    graph = _slice_g_active(
        graph_full,
        int(corpus["n_treatments_active"][local]),
        int(corpus["n_covariates_active"][local]),
        int(corpus["n_latent_active"][local]),
    )
    original = _evaluate(graph, cfg, original_structure, fixed)
    candidate = _evaluate(graph, cfg, new_structure, fixed)  # SAME exact fixed-RV map
    for key, saved_key in (
        ("treatments", "media_original"),
        ("contributions", "contribution_original"),
    ):
        np.testing.assert_array_equal(original[key], saved[saved_key])
    np.testing.assert_array_equal(
        original["treatments"].astype(np.float32),
        corpus["treatment_raw"][local, :, : len(graph["g_cy"])],
    )
    for output, stored in (
        ("covariates", "covariates"),
        ("covariate_contribution", "covariate_contribution"),
        ("baseline_intrinsic", "baseline_intrinsic"),
        ("outcome_noise", "outcome_noise"),
    ):
        reference = corpus[stored][local]
        if output in ("covariates", "covariate_contribution"):
            reference = reference[:, : len(graph["g_zy"])]
        np.testing.assert_array_equal(
            original[output].astype(reference.dtype),
            reference,
            err_msg=f"original fixed-RV replay differs from recorded {stored}",
        )
    for key in PROTECTED:
        np.testing.assert_array_equal(
            candidate[key], original[key], err_msg=f"protected matched output {key}"
        )
    for key, saved_key in (
        ("treatments", "media_less_smooth"),
        ("contributions", "contribution_less_smooth"),
    ):
        np.testing.assert_array_equal(
            candidate[key], saved[saved_key], err_msg=f"paired target {key}"
        )
    np.testing.assert_array_equal(saved["controls"], corpus["covariates"][local])
    np.testing.assert_array_equal(saved["noise"], corpus["outcome_noise"][local])
    delta = (candidate["contributions"] - original["contributions"]).sum(axis=1)
    np.testing.assert_allclose(
        saved["observed_less_smooth"], saved["observed_original"] + delta, atol=1e-12, rtol=1e-10
    )
    return {
        "world_id": world,
        "fixed_primitive_rvs_checked": len(fixed),
        "protected_outputs_checked": list(PROTECTED),
        "original_exact": True,
        "candidate_exact": True,
        "paired_target_exact": True,
        "media_changed": bool(np.any(candidate["treatments"] != original["treatments"])),
        "source_hash": digest(batch_path),
        "fixed_rvs_hash": digest(paired / f"world_{world:02d}_fixed_rvs.npz"),
    }


def replay(
    config: StudyConfig, source: Path, paired: Path, out: Path, *, worlds: tuple[int, ...]
) -> list[dict]:
    if not worlds or len(set(worlds)) != len(worlds):
        raise ContractError("matched replay requires distinct explicit world IDs")
    if out.resolve().is_relative_to(source.resolve()) or out.resolve().is_relative_to(
        paired.resolve()
    ):
        raise ContractError("matched validation output cannot overwrite source artifacts")
    records = []
    for world in worlds:
        files = {
            "base-manifest": source / "manifest.json",
            "paired-manifest": paired / "manifest.json",
            "base-batch": source / f"batch_{world // 2:02d}.npz",
            "paired-rvs": paired / f"world_{world:02d}_fixed_rvs.npz",
            "paired-arrays": paired / f"world_{world:02d}_paired.npz",
            "paired-report": paired / f"world_{world:02d}_report.json",
        }
        sources = {
            "matched-adapter": Path(__file__),
            "reference-config": Path(__file__).resolve().parents[2]
            / "notebooks/interactive_batches_legacy/config_29.json",
        }
        ident = identity(
            "analyze",
            config,
            source_files=sources,
            dataset_files=files,
            expected_outputs={"validation.json"},
        )
        ident["world_id"] = world
        directory = out / f"world_{world:02d}"
        if checkpoint(directory, ident).complete:
            records.append(
                {
                    "world_id": world,
                    "state": "reused",
                    **json.loads((directory / "validation.json").read_text()),
                }
            )
            continue
        write_manifest(directory, ident)
        result = replay_world(config, source, paired, world)
        from .runner import _json

        _json(directory, "validation.json", result)
        complete_checkpoint(directory, ident, {"validation.json"})
        records.append({"world_id": world, "state": "replayed", **result})
    return records
