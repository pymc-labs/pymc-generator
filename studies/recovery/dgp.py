"""Study-local adapter of the oracle's direct-effect geometric/MM prior; no notebook import."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

import pymc_generator as pg
from pymc_generator.slots import SlotLayout

from .config import ContractError, StudyConfig

SHOWCASE = (
    Path(__file__).resolve().parents[2] / "notebooks/interactive_batches_legacy/config_29.json"
)


def prior(config: StudyConfig, *, seed: int) -> pg.SCMPrior:
    study, overrides = config.resolved["study"], config.resolved["generator"]
    base = pg.SCMPrior(**json.loads(SHOWCASE.read_text()))
    accepted = {field.name for field in fields(pg.SCMPrior)}
    allowed = {key: value for key, value in overrides.items() if key in accepted}
    if config.resolved["pairing"]["mode"] != "independent" or "media_smoothness_beta" in overrides:
        raise ContractError(
            "matched media smoothing requires fixed-primitive replay, not a changed generator seed"
        )
    weights = base.active_count_weights
    if (
        list(base.n_treatments_active_range) != overrides["n_treatments_active_range"]
        or list(base.n_covariates_active_range) != overrides["n_covariates_active_range"]
    ):
        weights = None  # original matrix is specific to the original active-count grid
    result = replace(
        base,
        **{
            **allowed,
            "edge_budget": {k: tuple(v) for k, v in overrides["edge_budget"].items()},
            "active_count_weights": weights,
            "n_cells": 2,
            "draws_per_cell": 1,
            "seed": seed,
            "carryover_burn_in": study["burn_in"],
        },
    )
    # Never relax the fixed 1 latent slot or change reference/likelihood supports.
    assert result.n_latent == 1 and result.outcome_std_mode == "relative"
    assert result.mm_scale_prior == "log_uniform" and result.treatment_reference_multiplier == 1
    result.validate()
    return result


def validate_world(corpus: dict, world: int, cfg: pg.SCMPrior) -> None:
    if pg.DataGenerator.validate_corpus(corpus):
        raise ContractError("generated corpus violates generator validation")
    n, c = cfg.n_treatments, cfg.n_covariates
    graph = SlotLayout(n, c, cfg.n_latent).unpack(corpus["g"][world])
    if any(np.any(graph[key]) for key in ("dc", "dz", "dy", "zc", "cc", "zz")):
        raise ContractError("non-direct graph in direct-effect study")
    media_mask = corpus["treatment_active_mask"][world].astype(bool)
    control_mask = corpus["covariate_active_mask"][world].astype(bool)
    if not media_mask.any() or not (graph["cy"] & media_mask).any():
        raise ContractError("missing direct media effect")
    if np.any(graph["zy"] & ~control_mask) or np.any(graph["cy"] & ~media_mask):
        raise ContractError("effect outside active input")
    if np.any(corpus["latent_unobserved_contribution"][world]) or np.any(
        corpus["indirect_effects_by_source"][world]
    ):
        raise ContractError("nonzero latent/indirect contribution")
    if not np.all(corpus["carryover_family"][world, media_mask] == 1) or not np.all(
        corpus["sat_family"][world, media_mask] == 3
    ):
        raise ContractError("unsupported transform family")
    intrinsic = corpus["baseline_intrinsic"][world]
    if cfg.rw_baseline_std_range[1] == 0 and np.ptp(intrinsic) != 0:
        raise ContractError("nonconstant baseline")
    noiseless = corpus["outcome_raw"][world] - corpus["outcome_noise"][world]
    terms = (
        intrinsic
        + corpus["treatment_contribution_raw"][world].sum(axis=1)
        + corpus["covariate_contribution"][world].sum(axis=1)
    )
    np.testing.assert_allclose(terms, noiseless, atol=3e-6, rtol=3e-6)


def generate_pair(config: StudyConfig, *, seed: int) -> dict:
    cfg = prior(config, seed=seed)
    corpus = pg.sample_prior_predictive(cfg)
    if len(corpus["cell_id"]) != 2 or len(np.unique(corpus["cell_id"])) != 2:
        raise ContractError("generator did not provide two distinct cells")
    for world in range(2):
        validate_world(corpus, world, cfg)
    return corpus


def world_data(corpus: dict, world: int, cfg: pg.SCMPrior, score_start: int) -> dict:
    validate_world(corpus, world, cfg)
    media = corpus["treatment_raw"][world].astype(np.float64)
    controls = corpus["covariates"][world].astype(np.float64)
    media_scale = np.array([np.mean(x[x > 0]) if np.any(x > 0) else 1.0 for x in media.T])
    control_scale = np.sqrt(np.mean(controls**2, axis=0))
    control_scale = np.where(control_scale > 0, control_scale, 1.0)
    observed = corpus["outcome_raw"][world].astype(np.float64)
    outcome_scale = max(float(observed[score_start:].std()), 1e-8)
    graph = SlotLayout(cfg.n_treatments, cfg.n_covariates, cfg.n_latent).unpack(corpus["g"][world])
    active_media = corpus["treatment_active_mask"][world].astype(bool)
    anchor = np.where(active_media, corpus["mechanism_saturation_scale"][world], 1.0).astype(
        np.float64
    )
    data = {
        "media_data": media / media_scale,
        "control_data": controls / control_scale,
        "media_scale": media_scale,
        "control_scale": control_scale,
        "media_mask": active_media.astype(float),
        "control_mask": corpus["covariate_active_mask"][world].astype(float),
        "media_effect_mask": graph["cy"].astype(float),
        "control_effect_mask": graph["zy"].astype(float),
        "saturation_anchor": anchor,
        "outcome_scale": np.array(outcome_scale),
        "outcome_data": observed[score_start:] / outcome_scale,
    }
    if any(not np.isfinite(v).all() for v in data.values()) or np.any(anchor <= 0):
        raise ContractError("invalid oracle input scales/anchor")
    return data
