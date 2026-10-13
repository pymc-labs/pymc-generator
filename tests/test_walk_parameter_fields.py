"""Opt-in realised walk and outcome-noise truth (``SCMPrior.record_walk_parameters``).

The flag stores each world's walk means, scales, relative draws and smoothness
next to ``diagnostics["walk_parameters"]``. Pinned here: (A) recording changes
no other array, label or diagnostic across every draw-name layout, under forced
realism rejections; (B) every row holds its accepted draw's exact parameters;
(C) the recorded scales are the walks' calibrated scales in law; (D) the
validator rejects each broken invariant while admitting rounding.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import itertools
import json

import numpy as np
import pytest

import pymc_generator as pg
import pymc_generator.sampler as sampler
import pymc_generator.world_model as world_model
from pymc_generator import DataGenerator
from pymc_generator.sampler import SCMPrior
from pymc_generator.slots import (
    CORPUS_ARRAY_FIELDS,
    MECHANISM_ARRAY_FIELDS,
    MECHANISM_REFERENCE_FIELDS,
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    WALK_ARRAY_FIELDS,
    WALK_RELATIVE_MODE_FIELDS,
)

EPS = float(np.finfo(np.float64).eps)
HF = TRAJECTORY_COMPONENTS.index("hf")

# Deliberately independent of slots.py: the consumer-facing manifest, mapping
# each walk array to its slot axis (None: one value per world).
WALK_MANIFEST = {
    "walk_treatment_mean": "treatment",
    "walk_treatment_std": "treatment",
    "walk_treatment_std_relative": "treatment",
    "walk_treatment_smoothness": "treatment",
    "walk_covariate_mean": "covariate",
    "walk_covariate_std": "covariate",
    "walk_covariate_smoothness": "covariate",
    "walk_baseline_mean": None,
    "walk_baseline_std": None,
    "walk_baseline_smoothness": None,
    "outcome_noise_std": None,
    "walk_baseline_std_relative": None,
    "outcome_noise_std_relative": None,
    "walk_latent_smoothness": "latent",
}
RELATIVE_ONLY = ("walk_baseline_std_relative", "outcome_noise_std_relative")
BLOCK_KEYS = {
    "fields",
    "outcome_std_mode",
    "rw_positive_mean_range",
    "rw_treatment_std_range",
    "rw_covariate_mean_range",
    "rw_std_sigma",
    "rw_baseline_mean_range",
    "rw_baseline_std_range",
    "rw_outcome_std_range",
    "rw_baseline_std_sigma",
    "rw_outcome_std_sigma",
    "rw_smoothness_alpha",
    "rw_smoothness_beta",
    "rw_smoothness_max_weeks",
    "baseline_floor",
    "baseline_floor_scope",
}
ACTIVE_COUNTS = {
    "treatment": "n_treatments_active",
    "covariate": "n_covariates_active",
    "latent": "n_latent_active",
}


def _softplus(x):
    """PyTensor's softplus branch for the moderate walk means these priors draw."""
    return np.log1p(np.exp(x))


def _inverse_softplus(y):
    return y + np.log(-np.expm1(-y))


def _expected_fields(cfg):
    return sorted(
        key
        for key in WALK_MANIFEST
        if cfg.outcome_std_mode == "relative" or key not in RELATIVE_ONLY
    )


def _live(corpus, key):
    """A walk array's values on active slots (every world for per-world fields)."""
    role = WALK_MANIFEST[key]
    if role is None:
        return corpus[key]
    return corpus[key][corpus[f"{role}_active_mask"] == 1]


def _canonical(diagnostics):
    """Diagnostics as JSON text, without the walk block and the wall clock."""

    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    kept = {
        key: value for key, value in diagnostics.items() if key not in ("walk_parameters", "timing")
    }
    return json.dumps(kept, sort_keys=True, default=default)


@contextlib.contextmanager
def _generation_spies(override=None):
    """Reject every other realism verdict and record each cell's structure.

    Accepted rows then come from later draws of their batch, so a row written
    from the wrong draw is caught; both runs of an on/off pair see the same
    deterministic verdicts. ``override`` may rewrite a cell's structure in place.
    """
    structures = []
    verdicts = itertools.count(1)
    original_ok = sampler._additive_task_ok
    original_structure = world_model.sample_structure

    def task_ok(*args, **kwargs):
        keep = next(verdicts) % 2 == 0
        return original_ok(*args, **kwargs) and keep

    def structure(g_active, cfg, rng):
        result = original_structure(g_active, cfg, rng)
        if override is not None:
            override(g_active, result)
        structures.append({key: np.array(value, copy=True) for key, value in result.items()})
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sampler, "_additive_task_ok", task_ok)
        patch.setattr(world_model, "sample_structure", structure)
        yield structures


def _generate(cfg, *, record, override=None, n_tasks=None):
    """Generate through ``DataGenerator`` (which validates) under the spies."""
    generator = DataGenerator(dataclasses.replace(cfg, record_walk_parameters=record))
    with _generation_spies(override) as structures:
        corpus = generator.generate(n_tasks=n_tasks)
    return corpus, structures


def _config(**overrides):
    """Three slots per role with one or two active, so padding is always present."""
    kwargs = {
        "n_treatments": 3,
        "n_covariates": 3,
        "n_latent": 2,
        "n_treatments_active_range": (1, 2),
        "n_covariates_active_range": (1, 2),
        "n_latent_active_range": (1, 1),
        "n_cells": 2,
        "draws_per_cell": 2,
    }
    kwargs.update(overrides)
    return pg.make_scm_prior(**kwargs)


def _split_covariate_hf(g_active, structural):
    """The first active covariate carries hf texture and the second does not."""
    structural["use_covariate_hf"] = np.arange(len(g_active["g_zy"])) == 0


#: One config per draw-name layout: (overrides, structure override).
IDENTITY_CASES = {
    # Texture only: no trajectory or mechanism audit, so the walk names are the
    # only outputs ahead of the legacy ones.
    "texture": ({"n_time_steps": 16, "seed": 101}, None),
    "mechanisms_schedules": (
        {
            "n_time_steps": 32,
            "seed": 202,
            "n_covariates_active_range": (2, 2),
            "treatment_reference_contribution_range": (0.2, 0.6),
            "covariate_reference_contribution_range": (-0.2, 0.2),
            "mm_scale_prior": "log_uniform",
            "treatment_trend_inclusion_prob": 0.5,
            "treatment_level_jump_inclusion_prob": 0.5,
            "covariate_seasonal_inclusion_prob": 0.5,
            "covariate_hf_inclusion_prob": 0.5,
            "covariate_hf_sigma_range": (0.5, 0.5),
            "baseline_floor": 0.0,
        },
        _split_covariate_hf,
    ),
    "schedules_shocks": (
        {
            "n_time_steps": 32,
            "seed": 303,
            "trajectories": "periodic_on_off",
            "n_treatment_shocks": 1,
            "treatment_shock_length_range": (2, 3),
            "treatment_shock_level_range": (0.5, 1.5),
            "baseline_floor": 0.0,
            "baseline_floor_scope": "non_treatment",
        },
        None,
    ),
    "shocks_absolute": (
        {"n_time_steps": 16, "seed": 404, "outcome_std_mode": "absolute", "n_treatment_shocks": 1},
        None,
    ),
    "degenerate": (
        {
            "n_time_steps": 24,
            "seed": 505,
            "trajectories": "level_doubling",
            "rw_treatment_std_range": (0.5, 0.5),
            "rw_baseline_std_range": (0.0, 0.0),
        },
        None,
    ),
}


@pytest.fixture(scope="module")
def pairs():
    """Lazily generate each identity case once with the flag off and on."""
    cache = {}

    def get(name):
        if name not in cache:
            overrides, override = IDENTITY_CASES[name]
            cfg = _config(**overrides)
            off, _ = _generate(cfg, record=False, override=override)
            on, structures = _generate(cfg, record=True, override=override)
            cache[name] = (cfg, off, on, structures)
        return cache[name]

    return get


def _clean_config(**overrides):
    """Walks recoverable from the stored series: no burn-in, parents, texture or floor.

    Means are at least 1 and treatment relative scales at most 0.3, so inverting
    the treatment's two softplus layers stays well conditioned. Each role's mean
    and relative-scale ranges are disjoint from the other roles'.
    """
    kwargs = {
        "n_time_steps": 24,
        "seed": 606,
        "carryover_burn_in": 0,
        "treatment_hf_sigma_range": (0.0, 0.0),
        "treatment_pulse_prob_range": (0.0, 0.0),
        "covariate_hf_sigma_range": (0.0, 0.0),
        "covariate_pulse_prob_range": (0.0, 0.0),
        "dz_base_rate": 0.0,
        "zc_base_rate": 0.0,
        "cc_base_rate": 0.0,
        "zz_base_rate": 0.0,
        "edge_rate_overrides": {"dc": 0.0},
        "treatment_cv_floor": 0.0,
        "rw_positive_mean_range": (1.0, 2.0),
        "rw_treatment_std_range": (0.15, 0.3),
        "rw_covariate_mean_range": (2.5, 3.5),
        "rw_baseline_mean_range": (5.0, 8.0),
        "rw_baseline_std_range": (0.0, 0.05),
        "rw_outcome_std_range": (0.06, 0.09),
    }
    kwargs.update(overrides)
    return _config(**kwargs)


@pytest.fixture(scope="module")
def clean_sample():
    cfg = _clean_config()
    with pytest.warns(FutureWarning, match="flat"):
        corpus, structures = _generate(cfg, record=True)
    return cfg, corpus, structures


# ---------------------------------------------------------------------------
# (A) Recording is invisible to every pre-existing output
# ---------------------------------------------------------------------------


def _assert_case_predicates(name, cfg, off):
    diagnostics = off["diagnostics"]
    if name == "texture":
        assert not cfg.mechanism_priors_enabled and not cfg.trajectory_metadata_enabled
        assert cfg.n_treatment_shocks == 0 and cfg.baseline_floor is None
        assert "trajectory" not in diagnostics and "mechanism_priors" not in diagnostics
    elif name == "mechanisms_schedules":
        assert cfg.mechanism_priors_enabled and cfg.trajectory_components_enabled
        assert cfg.n_treatment_shocks == 0 and cfg.baseline_floor == 0.0
        assert "mechanism_priors" in diagnostics and "trajectory" in diagnostics
    elif name == "schedules_shocks":
        assert cfg.trajectory_components_enabled and cfg.n_treatment_shocks == 1
        assert cfg.baseline_floor == 0.0 and cfg.baseline_floor_scope == "non_treatment"
        active = np.broadcast_to(
            off["treatment_active_mask"][:, None, :] == 1, off["treatment_activity"].shape
        )
        assert (off["treatment_activity"][active] == 0).any()
        assert (off["treatment_shock_mask"] == 1).any()
    elif name == "shocks_absolute":
        assert cfg.n_treatment_shocks == 1 and not cfg.trajectory_metadata_enabled
        assert cfg.outcome_std_mode == "absolute" and cfg.baseline_floor is None
        assert (off["treatment_shock_mask"] == 1).any()
    else:
        assert tuple(cfg.rw_treatment_std_range) == (0.5, 0.5)
        assert tuple(cfg.rw_baseline_std_range) == (0.0, 0.0)
        assert tuple(cfg.rw_covariate_mean_range) == (1.0, 1.0)
        assert cfg.trajectory_components_enabled and cfg.baseline_floor is None


@pytest.mark.parametrize("name", IDENTITY_CASES)
def test_recording_changes_no_existing_array_label_or_diagnostic(pairs, name):
    cfg, off, on, _ = pairs(name)
    _assert_case_predicates(name, cfg, off)
    assert cfg.draws_per_cell >= 2
    for corpus in (off, on):
        assert corpus["diagnostics"]["rejection_rate"] > 0.0
        assert DataGenerator.validate_corpus(corpus) == []

    assert set(on) - set(off) == set(_expected_fields(cfg))
    assert set(off) <= set(on)
    for key, value in off.items():
        if isinstance(value, np.ndarray):
            assert on[key].dtype == value.dtype, key
            np.testing.assert_array_equal(on[key], value, err_msg=key)
    off_labels = off.get("identifiability", {})
    assert set(on.get("identifiability", {})) == set(off_labels)
    for key, value in off_labels.items():
        np.testing.assert_array_equal(on["identifiability"][key], value, err_msg=key)
    assert _canonical(on["diagnostics"]) == _canonical(off["diagnostics"])


@pytest.mark.parametrize("name", IDENTITY_CASES)
def test_default_off_corpora_carry_no_walk_keys_or_block(pairs, name):
    cfg, off, _, _ = pairs(name)
    assert cfg.record_walk_parameters is False
    assert not set(off).intersection(WALK_MANIFEST)
    assert "walk_parameters" not in off["diagnostics"]


@pytest.mark.parametrize("name", IDENTITY_CASES)
def test_walk_block_echoes_the_effective_priors(pairs, name):
    cfg, _, on, _ = pairs(name)
    block = on["diagnostics"]["walk_parameters"]
    relative = cfg.outcome_std_mode == "relative"

    assert set(block) == BLOCK_KEYS
    assert block == dataclasses.replace(cfg, record_walk_parameters=True).walk_parameter_metadata()
    assert block["fields"] == _expected_fields(cfg)
    assert block["outcome_std_mode"] == cfg.outcome_std_mode
    for key in (
        "rw_positive_mean_range",
        "rw_treatment_std_range",
        "rw_covariate_mean_range",
        "rw_baseline_mean_range",
    ):
        assert block[key] == [float(bound) for bound in getattr(cfg, key)], key
    for key in ("rw_baseline_std_range", "rw_outcome_std_range"):
        expected = [float(bound) for bound in getattr(cfg, key)] if relative else None
        assert block[key] == expected, key
    if relative:
        assert block["rw_baseline_std_sigma"] is None and block["rw_outcome_std_sigma"] is None
    else:
        assert block["rw_baseline_std_sigma"] == cfg.rw_baseline_std_sigma_effective
        assert block["rw_outcome_std_sigma"] == cfg.rw_outcome_std_sigma
    assert block["rw_std_sigma"] == cfg.rw_std_sigma
    assert block["rw_smoothness_alpha"] == cfg.rw_smoothness_alpha
    assert block["rw_smoothness_beta"] == cfg.rw_smoothness_beta
    assert block["rw_smoothness_max_weeks"] == cfg.rw_smoothness_max_weeks
    assert block["baseline_floor"] == cfg.baseline_floor
    assert block["baseline_floor_scope"] == cfg.baseline_floor_scope
    assert json.loads(json.dumps(block)) == block


@pytest.mark.parametrize("name", ("mechanisms_schedules", "shocks_absolute"))
def test_walk_arrays_and_block_survive_save_and_load(tmp_path, pairs, name):
    _, _, on, _ = pairs(name)
    path = tmp_path / f"{name}.npz"
    pg.save_corpus(on, path)
    loaded = pg.load_corpus(path)
    for key in on["diagnostics"]["walk_parameters"]["fields"]:
        assert loaded[key].dtype == np.float64, key
        assert loaded[key].shape == on[key].shape, key
        np.testing.assert_array_equal(loaded[key], on[key], err_msg=key)
    assert loaded["diagnostics"]["walk_parameters"] == on["diagnostics"]["walk_parameters"]
    assert DataGenerator.validate_corpus(loaded) == []


def test_n_truncation_slices_the_walk_arrays_with_their_rows(pairs):
    cfg, _, full, _ = pairs("texture")
    truncated, _ = _generate(cfg, record=True, n_tasks=3)
    assert truncated["outcome_raw"].shape[0] == 3
    for key in full["diagnostics"]["walk_parameters"]["fields"]:
        np.testing.assert_array_equal(truncated[key], full[key][:3], err_msg=key)
    assert truncated["diagnostics"]["walk_parameters"] == full["diagnostics"]["walk_parameters"]
    assert DataGenerator.validate_corpus(truncated) == []


# ---------------------------------------------------------------------------
# (B) Every row holds its accepted draw's exact parameters
# ---------------------------------------------------------------------------


def _assert_smoothness_follows_structure(corpus, structures):
    for row, cell in enumerate(corpus["cell_id"]):
        structural = structures[cell]
        for key, source, role in (
            ("walk_treatment_smoothness", "smoothness_c", "treatment"),
            ("walk_covariate_smoothness", "smoothness_z", "covariate"),
            ("walk_latent_smoothness", "smoothness_d", "latent"),
        ):
            count = int(corpus[ACTIVE_COUNTS[role]][row])
            expected = np.zeros(corpus[key].shape[1])
            expected[:count] = structural[source]
            np.testing.assert_array_equal(corpus[key][row], expected, err_msg=key)
        assert corpus["walk_baseline_smoothness"][row] == structural["smoothness_b"][0]


@pytest.mark.parametrize("name", IDENTITY_CASES)
def test_smoothness_is_the_cells_structure_draw(pairs, name):
    _, _, on, structures = pairs(name)
    assert len(structures) == on["diagnostics"]["n_cells"]
    _assert_smoothness_follows_structure(on, structures)


@pytest.mark.parametrize("name", IDENTITY_CASES)
def test_treatment_scale_and_level_follow_the_recorded_mean(pairs, name):
    _, _, on, _ = pairs(name)
    active = on["treatment_active_mask"] == 1
    level = _softplus(on["walk_treatment_mean"][active])
    np.testing.assert_allclose(
        on["walk_treatment_std"][active],
        on["walk_treatment_std_relative"][active] * level,
        rtol=8 * EPS,
        atol=0.0,
    )
    np.testing.assert_array_max_ulp(on["treatment_level"][active], level.astype(np.float32), 1)


def test_relative_outcome_scales_times_the_treatment_amplitude_give_the_absolutes(pairs):
    cfg, _, on, _ = pairs("mechanisms_schedules")
    # Mechanism priors scale by the rescaled norm of the direct amplitudes.
    amplitudes = on["g"][:, cfg.layout.slices["cy"]] * on["beta"]
    scale = np.abs(amplitudes).max(axis=1)
    anchor = scale * np.sqrt(((amplitudes / scale[:, None]) ** 2).sum(axis=1))
    for absolute, relative in (
        ("walk_baseline_std", "walk_baseline_std_relative"),
        ("outcome_noise_std", "outcome_noise_std_relative"),
    ):
        np.testing.assert_allclose(on[absolute], on[relative] * anchor, rtol=8 * EPS, atol=0.0)


def test_covariate_texture_scale_is_half_the_recorded_walk_scale(pairs):
    _, _, on, _ = pairs("mechanisms_schedules")
    active = on["covariate_active_mask"] == 1
    carried = active & (on["covariate_components"][..., HF] == 1)
    assert carried.any() and (active & ~carried).any()
    np.testing.assert_allclose(
        on["trajectory_covariate_hf_sigma"][carried],
        0.5 * on["walk_covariate_std"][carried],
        rtol=1e-15,
        atol=0.0,
    )
    assert not on["trajectory_covariate_hf_sigma"][~carried].any()


def test_degenerate_ranges_record_their_exact_constants(pairs):
    _, _, on, _ = pairs("degenerate")
    treatment = on["treatment_active_mask"] == 1
    covariate = on["covariate_active_mask"] == 1
    assert (on["walk_treatment_std_relative"][treatment] == 0.5).all()
    assert (on["walk_covariate_mean"][covariate] == 1.0).all()
    assert (on["walk_baseline_std_relative"] == 0.0).all()
    assert (on["walk_baseline_std"] == 0.0).all()
    np.testing.assert_allclose(
        on["walk_treatment_std"][treatment],
        0.5 * _softplus(on["walk_treatment_mean"][treatment]),
        rtol=8 * EPS,
        atol=0.0,
    )
    # The live noise draw beside them is still a fresh uniform per world.
    lo, hi = on["diagnostics"]["walk_parameters"]["rw_outcome_std_range"]
    noise = on["outcome_noise_std_relative"]
    assert ((noise >= lo) & (noise <= hi)).all()
    assert np.unique(noise).size == noise.size


def test_recorded_means_recover_the_stored_walks(clean_sample):
    _, corpus, structures = clean_sample
    assert corpus["diagnostics"]["rejection_rate"] > 0.0
    assert DataGenerator.validate_corpus(corpus) == []
    _assert_smoothness_follows_structure(corpus, structures)
    for row in range(corpus["outcome_raw"].shape[0]):
        n_treatments = int(corpus["n_treatments_active"][row])
        n_covariates = int(corpus["n_covariates_active"][row])
        # A treatment is softplus(softplus(walk)) without parents or texture.
        treatment = corpus["treatment_raw"][row, :, :n_treatments].astype(np.float64)
        walks = _inverse_softplus(_inverse_softplus(treatment))
        np.testing.assert_allclose(
            walks.mean(axis=0), corpus["walk_treatment_mean"][row, :n_treatments], rtol=1e-6
        )
        covariates = corpus["covariates"][row, :, :n_covariates].astype(np.float64)
        np.testing.assert_allclose(
            covariates.mean(axis=0), corpus["walk_covariate_mean"][row, :n_covariates], rtol=1e-6
        )
        baseline = corpus["baseline_intrinsic"][row].astype(np.float64)
        np.testing.assert_allclose(baseline.mean(), corpus["walk_baseline_mean"][row], rtol=1e-6)


def test_each_role_records_draws_from_its_own_disjoint_range(clean_sample):
    cfg, corpus, _ = clean_sample
    supports = {
        "walk_treatment_mean": cfg.rw_positive_mean_range,
        "walk_covariate_mean": cfg.rw_covariate_mean_range,
        "walk_baseline_mean": cfg.rw_baseline_mean_range,
        "walk_treatment_std_relative": cfg.rw_treatment_std_range,
        "walk_baseline_std_relative": cfg.rw_baseline_std_range,
        "outcome_noise_std_relative": cfg.rw_outcome_std_range,
    }
    for family in (tuple(supports)[:3], tuple(supports)[3:]):
        bounds = sorted(tuple(supports[key]) for key in family)
        assert all(left[1] < right[0] for left, right in itertools.pairwise(bounds)), family
    for key, (lo, hi) in supports.items():
        values = _live(corpus, key)
        assert ((values >= lo) & (values <= hi)).all(), key


# ---------------------------------------------------------------------------
# (C) The recorded scales are the walks' calibrated scales (selection-free)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selection_free():
    """The clean config at law-check size, with nothing the realism filter rejects."""
    cache = {}

    def get(mode):
        if mode not in cache:
            cfg = _clean_config(
                n_time_steps=104,
                n_cells=10,
                draws_per_cell=10,
                seed=707,
                outcome_std_mode=mode,
                rw_baseline_mean_range=(20.0, 30.0),
                rw_baseline_std_sigma=1.0,
                record_walk_parameters=True,
            )
            with pytest.warns(FutureWarning, match="flat"):
                corpus = pg.sample_prior_predictive(cfg)
            assert corpus["diagnostics"]["rejection_rate"] == 0.0
            cache[mode] = (cfg, corpus)
        return cache[mode]

    return get


@pytest.mark.slow
def test_recorded_scales_are_the_walks_calibrated_scales(selection_free):
    """E[var_pop(walk)] equals the recorded std squared, per role."""
    _, corpus = selection_free("absolute")
    ratios: dict[str, list[float]] = {"treatment": [], "covariate": [], "baseline": []}
    for row in range(corpus["outcome_raw"].shape[0]):
        n_treatments = int(corpus["n_treatments_active"][row])
        n_covariates = int(corpus["n_covariates_active"][row])
        treatment = corpus["treatment_raw"][row, :, :n_treatments].astype(np.float64)
        walks = _inverse_softplus(_inverse_softplus(treatment))
        ratios["treatment"].extend(
            walks.var(axis=0) / corpus["walk_treatment_std"][row, :n_treatments] ** 2
        )
        covariates = corpus["covariates"][row, :, :n_covariates].astype(np.float64)
        ratios["covariate"].extend(
            covariates.var(axis=0) / corpus["walk_covariate_std"][row, :n_covariates] ** 2
        )
        baseline = corpus["baseline_intrinsic"][row].astype(np.float64)
        ratios["baseline"].append(baseline.var() / corpus["walk_baseline_std"][row] ** 2)
    for role, values in ratios.items():
        assert len(values) >= 100, role
        assert 0.7 <= np.mean(values) <= 1.4, (role, np.mean(values))


@pytest.mark.slow
@pytest.mark.parametrize("mode", ("absolute", "relative"))
def test_outcome_noise_has_the_recorded_scale(selection_free, mode):
    cfg, corpus = selection_free(mode)
    sigma = corpus["outcome_noise_std"]
    assert (sigma > 0.0).all()
    if mode == "relative":
        # The relative scale multiplies the direct amplitudes: every world has a C->Y edge.
        assert (corpus["g"][:, cfg.layout.slices["cy"]].sum(axis=1) >= 1).all()
    standardized = corpus["outcome_noise"].astype(np.float64) / sigma[:, None]
    assert standardized.size >= 10_000
    assert abs(standardized.std() - 1.0) <= 0.05


# ---------------------------------------------------------------------------
# (D) Configuration, registry and validator boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", (1, 0, "true", None, np.bool_(True)))
def test_record_walk_parameters_must_be_a_bool(value):
    with pytest.raises(ValueError, match="record_walk_parameters must be a bool"):
        SCMPrior(record_walk_parameters=value).validate()


def test_recipe_json_round_trip_keeps_every_field_and_the_echo():
    cfg = _config(
        n_time_steps=16,
        seed=101,
        outcome_std_mode="absolute",
        baseline_floor=0.0,
        record_walk_parameters=True,
    )
    recipe = json.loads(json.dumps(dataclasses.asdict(cfg)))
    rebuilt = SCMPrior(**recipe)
    rebuilt.validate()
    for field in dataclasses.fields(SCMPrior):
        original = json.loads(json.dumps(getattr(cfg, field.name)))
        assert json.loads(json.dumps(getattr(rebuilt, field.name))) == original, field.name
    assert rebuilt.record_walk_parameters is True
    assert rebuilt.walk_parameter_metadata() == cfg.walk_parameter_metadata()


def test_walk_registry_matches_the_consumer_manifest():
    expected_axes = {
        key: ("task",) if role is None else ("task", role) for key, role in WALK_MANIFEST.items()
    }
    assert {key: axes for key, (axes, _) in WALK_ARRAY_FIELDS.items()} == expected_axes
    assert all(dtype is np.float64 for _, dtype in WALK_ARRAY_FIELDS.values())
    assert set(WALK_RELATIVE_MODE_FIELDS) == set(RELATIVE_ONLY)
    for registry in (
        CORPUS_ARRAY_FIELDS,
        TRAJECTORY_ARRAY_FIELDS,
        MECHANISM_ARRAY_FIELDS,
        MECHANISM_REFERENCE_FIELDS,
    ):
        assert not set(WALK_ARRAY_FIELDS).intersection(registry)


def _first(mask):
    return tuple(np.argwhere(mask)[0])


def _advance(value, steps):
    """Move a float scalar ``steps`` ulps (negative: downwards) in its own dtype."""
    target = value.dtype.type(np.inf if steps > 0 else -np.inf)
    for _ in range(abs(steps)):
        value = np.nextafter(value, target)
    return value


def _drop(key):
    def mutate(corpus):
        del corpus[key]

    return mutate


def _set_block(key, value):
    def mutate(corpus):
        corpus["diagnostics"]["walk_parameters"][key] = value

    return mutate


def _set_live(key, value):
    """Write ``value`` into the first active slot of a walk array."""

    def mutate(corpus):
        role = WALK_MANIFEST[key]
        index = (0,) if role is None else _first(corpus[f"{role}_active_mask"] == 1)
        corpus[key][index] = value

    return mutate


def _set_outside(key, range_name):
    def mutate(corpus):
        _, hi = corpus["diagnostics"]["walk_parameters"][range_name]
        _set_live(key, hi + 1.0)(corpus)

    return mutate


def _drop_block(corpus):
    del corpus["diagnostics"]["walk_parameters"]


def _replace_block(corpus):
    corpus["diagnostics"]["walk_parameters"] = []


def _reverse_inventory(corpus):
    corpus["diagnostics"]["walk_parameters"]["fields"].reverse()


def _add_axis(corpus):
    corpus["walk_baseline_mean"] = corpus["walk_baseline_mean"][:, None]


def _narrow_dtype(corpus):
    corpus["walk_covariate_std"] = corpus["walk_covariate_std"].astype(np.float32)


def _set_padding(corpus):
    corpus["walk_treatment_mean"][_first(corpus["treatment_active_mask"] == 0)] = 1.0


def _set_absolute_mode(corpus):
    corpus["diagnostics"]["walk_parameters"].update(
        outcome_std_mode="absolute",
        rw_baseline_std_range=None,
        rw_outcome_std_range=None,
        rw_baseline_std_sigma=1.0,
        rw_outcome_std_sigma=0.25,
    )


def _flip_signal_mode(corpus):
    corpus["diagnostics"]["signal"]["outcome_std_mode"] = "absolute"


def _advance_level(corpus):
    index = _first(corpus["treatment_active_mask"] == 1)
    corpus["treatment_level"][index] = _advance(corpus["treatment_level"][index], 4)


def _scale_treatment_std(corpus):
    index = _first(corpus["treatment_active_mask"] == 1)
    corpus["walk_treatment_std"][index] *= 1.0 + 1e-9


#: Mutations of the relative-mode "texture" corpus: (mutation, expected error).
MUTATIONS = {
    "block_missing": (
        _drop_block,
        "walk arrays and diagnostics walk_parameters must be present together",
    ),
    "array_missing": (
        _drop("walk_latent_smoothness"),
        "missing walk_latent_smoothness",
    ),
    "relative_array_missing": (
        _drop("outcome_noise_std_relative"),
        "missing outcome_noise_std_relative",
    ),
    "block_not_a_mapping": (
        _replace_block,
        "diagnostics walk_parameters must be a mapping",
    ),
    "unsupported_mode": (
        _set_block("outcome_std_mode", "log"),
        "outcome_std_mode must be 'relative' or 'absolute'",
    ),
    "relative_arrays_in_absolute_mode": (
        _set_absolute_mode,
        "must be present iff",
    ),
    "inventory": (
        _reverse_inventory,
        "fields does not list the stored walk arrays",
    ),
    "shape": (
        _add_axis,
        "Shape mismatch for walk_baseline_mean",
    ),
    "dtype": (
        _narrow_dtype,
        "walk_covariate_std has dtype float32",
    ),
    "non_finite": (
        _set_live("walk_baseline_std", np.nan),
        "walk_baseline_std contains NaN or Inf",
    ),
    "padding": (
        _set_padding,
        "walk_treatment_mean has nonzero inactive-treatment padding",
    ),
    "negative_std": (
        _set_live("walk_covariate_std", -0.5),
        "walk_covariate_std must be nonnegative",
    ),
    "mean_outside_range": (
        _set_outside("walk_baseline_mean", "rw_baseline_mean_range"),
        "walk_baseline_mean is outside diagnostics walk_parameters rw_baseline_mean_range",
    ),
    "relative_draw_outside_range": (
        _set_outside("outcome_noise_std_relative", "rw_outcome_std_range"),
        "outcome_noise_std_relative is outside diagnostics walk_parameters rw_outcome_std_range",
    ),
    "smoothness": (
        _set_live("walk_latent_smoothness", 1.5),
        "walk_latent_smoothness must lie in [0, 1]",
    ),
    "treatment_level": (
        _advance_level,
        "treatment_level is not float32(softplus(walk_treatment_mean)) within 1 ulp",
    ),
    "treatment_std_product": (
        _scale_treatment_std,
        "walk_treatment_std does not match",
    ),
    "json_tuple": (
        _set_block("rw_covariate_mean_range", (-1.0, 1.0)),
        "must survive a JSON round trip unchanged",
    ),
    "json_numpy_scalar": (
        _set_block("rw_std_sigma", np.float32(1.0)),
        "must survive a JSON round trip unchanged",
    ),
    "keys": (
        _set_block("rw_extra_range", [0.0, 1.0]),
        "keys do not match the schema",
    ),
    "signal_mode": (
        _flip_signal_mode,
        "outcome_std_mode does not match diagnostics signal",
    ),
    "absolute_sigma_in_relative_mode": (
        _set_block("rw_outcome_std_sigma", 0.25),
        "rw_outcome_std_sigma must be None in relative mode",
    ),
    "range_bounds": (
        _set_block("rw_baseline_mean_range", [8.0, 3.0]),
        "rw_baseline_mean_range has invalid bounds",
    ),
    "std_range_negative_lower_bound": (
        _set_block("rw_baseline_std_range", [-0.1, 0.1]),
        "rw_baseline_std_range has invalid bounds",
    ),
    "treatment_std_range_without_positive_upper_bound": (
        _set_block("rw_treatment_std_range", [0.0, 0.0]),
        "rw_treatment_std_range has invalid bounds",
    ),
    "positive_mean_range_not_positive": (
        _set_block("rw_positive_mean_range", [0.0, 1.0]),
        "rw_positive_mean_range has invalid bounds",
    ),
    "sigma": (
        _set_block("rw_std_sigma", 0.0),
        "rw_std_sigma must be finite and positive",
    ),
    "max_weeks": (
        _set_block("rw_smoothness_max_weeks", 26.0),
        "rw_smoothness_max_weeks must be an int >= 1",
    ),
    "max_weeks_bool": (
        _set_block("rw_smoothness_max_weeks", True),
        "rw_smoothness_max_weeks must be an int >= 1",
    ),
    "floor": (
        _set_block("baseline_floor", "0"),
        "baseline_floor must be None or finite",
    ),
    "scope": (
        _set_block("baseline_floor_scope", "total"),
        "baseline_floor_scope is not supported",
    ),
}


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_validator_rejects_each_broken_walk_invariant(pairs, mutation):
    _, _, on, _ = pairs("texture")
    broken = copy.deepcopy(on)
    mutate, expected = MUTATIONS[mutation]
    mutate(broken)
    errors = DataGenerator.validate_corpus(broken)
    assert any(expected in error for error in errors), errors


def test_validator_rejects_relative_arrays_in_an_absolute_corpus(pairs):
    _, _, on, _ = pairs("shocks_absolute")
    broken = copy.deepcopy(on)
    for key in RELATIVE_ONLY:
        broken[key] = np.zeros(on["outcome_raw"].shape[0])
    errors = DataGenerator.validate_corpus(broken)
    assert any("must be present iff" in error for error in errors), errors


#: Mutations of the absolute-mode "shocks_absolute" corpus: (mutation, expected error).
ABSOLUTE_MUTATIONS = {
    "relative_range_in_absolute_mode": (
        _set_block("rw_baseline_std_range", [0.0, 0.1]),
        "rw_baseline_std_range must be None in absolute mode",
    ),
    "absolute_sigma_not_positive": (
        _set_block("rw_outcome_std_sigma", 0.0),
        "rw_outcome_std_sigma must be finite and positive",
    ),
}


@pytest.mark.parametrize("mutation", ABSOLUTE_MUTATIONS)
def test_validator_rejects_each_broken_absolute_mode_echo(pairs, mutation):
    _, _, on, _ = pairs("shocks_absolute")
    broken = copy.deepcopy(on)
    mutate, expected = ABSOLUTE_MUTATIONS[mutation]
    mutate(broken)
    errors = DataGenerator.validate_corpus(broken)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("steps", (-2, -1, 1, 2))
def test_validator_admits_ulp_rounding_of_the_treatment_scale_product(pairs, steps):
    _, _, on, _ = pairs("texture")
    perturbed = copy.deepcopy(on)
    index = _first(on["treatment_active_mask"] == 1)
    perturbed["walk_treatment_std"][index] = _advance(on["walk_treatment_std"][index], steps)
    assert perturbed["walk_treatment_std"][index] != on["walk_treatment_std"][index]
    assert DataGenerator.validate_corpus(perturbed) == []


@pytest.mark.parametrize("steps", (-128, 128))
def test_validator_rejects_treatment_scale_rounding_beyond_64_ulps(pairs, steps):
    _, _, on, _ = pairs("texture")
    perturbed = copy.deepcopy(on)
    index = _first(on["treatment_active_mask"] == 1)
    perturbed["walk_treatment_std"][index] = _advance(on["walk_treatment_std"][index], steps)
    errors = DataGenerator.validate_corpus(perturbed)
    assert any("walk_treatment_std does not match" in error for error in errors), errors


def test_validator_admits_a_null_baseline_floor(pairs):
    _, _, on, _ = pairs("mechanisms_schedules")
    assert on["diagnostics"]["walk_parameters"]["baseline_floor"] == 0.0
    relaxed = copy.deepcopy(on)
    relaxed["diagnostics"]["walk_parameters"]["baseline_floor"] = None
    assert DataGenerator.validate_corpus(relaxed) == []
