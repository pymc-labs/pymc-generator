"""Schema-v5 truth fidelity, independent schedules, and corruption boundaries."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import pymc_generator as pg
import pymc_generator.world_model as world_model
from pymc_generator import DataGenerator
from pymc_generator.trajectories import structural_key

# Deliberately independent of slots.py: omissions in the writer's registry must
# not silently shrink this consumer-facing field manifest. The last tuple item
# is the jump-axis size (None for per-input leaves).
TRAJECTORY_MANIFEST = {
    "trajectory_treatment_hf_sigma": ("treatment", "hf", np.float64, None),
    "trajectory_treatment_pulse_amp": ("treatment", "pulse", np.float64, None),
    "trajectory_treatment_pulse_prob": ("treatment", "pulse", np.float64, None),
    "trajectory_treatment_onset_start": ("treatment", "onset", np.int64, None),
    "trajectory_treatment_offset_stop": ("treatment", "offset", np.int64, None),
    "trajectory_treatment_flighting_period": ("treatment", "flighting", np.int64, None),
    "trajectory_treatment_flighting_on_weeks": ("treatment", "flighting", np.int64, None),
    "trajectory_treatment_flighting_phase": ("treatment", "flighting", np.int64, None),
    "trajectory_treatment_level_jump_week": ("treatment", "level_jump", np.int64, 2),
    "trajectory_treatment_level_jump_factor": ("treatment", "level_jump", np.float64, 2),
    "trajectory_treatment_level_jump_log_factor": ("treatment", "level_jump", np.float64, 2),
    "trajectory_treatment_seasonal_amplitude": ("treatment", "seasonal", np.float64, None),
    "trajectory_treatment_seasonal_period": ("treatment", "seasonal", np.float64, None),
    "trajectory_treatment_seasonal_phase": ("treatment", "seasonal", np.float64, None),
    "trajectory_treatment_trend_change": ("treatment", "trend", np.float64, None),
    "trajectory_covariate_hf_sigma": ("covariate", "hf", np.float64, None),
    "trajectory_covariate_pulse_amp": ("covariate", "pulse", np.float64, None),
    "trajectory_covariate_pulse_prob": ("covariate", "pulse", np.float64, None),
    "trajectory_covariate_onset_start": ("covariate", "onset", np.int64, None),
    "trajectory_covariate_offset_stop": ("covariate", "offset", np.int64, None),
    "trajectory_covariate_flighting_period": ("covariate", "flighting", np.int64, None),
    "trajectory_covariate_flighting_on_weeks": ("covariate", "flighting", np.int64, None),
    "trajectory_covariate_flighting_phase": ("covariate", "flighting", np.int64, None),
    "trajectory_covariate_level_jump_week": ("covariate", "level_jump", np.int64, 3),
    "trajectory_covariate_level_jump_size": ("covariate", "level_jump", np.float64, 3),
    "trajectory_covariate_seasonal_amplitude": ("covariate", "seasonal", np.float64, None),
    "trajectory_covariate_seasonal_period": ("covariate", "seasonal", np.float64, None),
    "trajectory_covariate_seasonal_phase": ("covariate", "seasonal", np.float64, None),
    "trajectory_covariate_trend_change": ("covariate", "trend", np.float64, None),
}
MECHANISM_MANIFEST = {
    "sat_family": ("treatment", np.uint8),
    "mechanism_saturation_scale": ("treatment", np.float64),
    "beta": ("treatment", np.float64),
    "rho_zy": ("covariate", np.float64),
    "hill_slope": ("treatment", np.float64),
    "hill_kappa_mult": ("treatment", np.float64),
    "logistic_lam": ("treatment", np.float64),
    "mm_kappa_mult": ("treatment", np.float64),
    "tanh_c": ("treatment", np.float64),
    "root_alpha": ("treatment", np.float64),
    "treatment_reference_contribution": ("treatment", np.float64),
    "treatment_reference_input": ("treatment", np.float64),
    "treatment_reference_response": ("treatment", np.float64),
    "covariate_reference_contribution": ("covariate", np.float64),
    "covariate_reference_input": ("covariate", np.float64),
}
COMPONENTS = ("hf", "pulse", "onset", "offset", "flighting", "level_jump", "seasonal", "trend")
SATURATION_SHAPE_PRIORS = {
    "hill_slope": ("hill", "slope"),
    "hill_kappa_mult": ("hill", "kappa_mult"),
    "logistic_lam": ("logistic", "lam"),
    "mm_kappa_mult": ("michaelis_menten", "kappa_mult"),
    "tanh_c": ("tanh", "c"),
    "root_alpha": ("root", "alpha"),
}


def _shape_maximum(key):
    return 1.0 if key == "root_alpha" else float(np.finfo(np.float32).max)


def _capture_generation(cfg, *, controlled_components=False):
    """Observe real joint batches, never substitute parameter draws or outputs."""
    original_build = world_model.build_world_model
    original_draw = world_model.draw_worlds
    original_structure = world_model.sample_structure
    contexts, records = {}, []
    cell = 0

    def structure(g, prior, rng):
        nonlocal cell
        result = original_structure(g, prior, rng)
        if controlled_components:
            for role, n in (("treatment", len(g["g_cy"])), ("covariate", len(g["g_zy"]))):
                for component in COMPONENTS:
                    slot = 1 if component in ("pulse", "offset", "trend") else 0
                    selected = np.arange(n) == slot
                    if cell == 2 and role == "covariate" and component == "seasonal":
                        selected[:] = False
                    result[structural_key(role, component)] = selected
            result["sat_family"] = np.array([2 * cell, 2 * cell + 1], dtype=np.int64)
        cell += 1
        return result

    def build(g, prior, structural, n_time_steps, **kwargs):
        model, output_names, param_names = original_build(
            g, prior, structural, n_time_steps, **kwargs
        )
        contexts[id(model)] = {
            "model": model,
            "cell": len(contexts),
            "structural": {key: np.asarray(value).copy() for key, value in structural.items()},
        }
        return model, output_names, param_names

    def draw(model, names, seed, **kwargs):
        batch = original_draw(model, names, seed, **kwargs)
        records.append({**contexts[id(model)], "batch": batch})
        return batch

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(world_model, "sample_structure", structure)
        patch.setattr(world_model, "build_world_model", build)
        patch.setattr(world_model, "draw_worlds", draw)
        corpus = pg.sample_prior_predictive(cfg)
    return corpus, records


@pytest.fixture(scope="module")
def rich_sample():
    kwargs = {
        f"{role}_{component}_inclusion_prob": 0.5
        for role in ("treatment", "covariate")
        for component in COMPONENTS
    }
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=4,
        n_latent=1,
        n_treatments_active_range=(2, 2),
        n_covariates_active_range=(2, 2),
        n_time_steps=64,
        n_cells=3,
        draws_per_cell=2,
        seed=227,
        treatment_level_jump_count=2,
        covariate_level_jump_count=3,
        treatment_level_jump_factor_range=(0.9, 1.1),
        treatment_seasonal_amplitude_range=(0.01, 0.03),
        treatment_trend_log_change_range=(-0.03, 0.03),
        treatment_onset_frac_range=(0.1, 0.15),
        covariate_onset_frac_range=(0.1, 0.15),
        treatment_offset_frac_range=(0.7, 0.8),
        covariate_offset_frac_range=(0.7, 0.8),
        treatment_flighting_period_weeks_range=(6, 8),
        covariate_flighting_period_weeks_range=(6, 8),
        treatment_flighting_duty_range=(0.6, 0.8),
        covariate_flighting_duty_range=(0.6, 0.8),
        treatment_reference_contribution_range=(0.2, 0.6),
        treatment_reference_multiplier=0.9,
        covariate_reference_contribution_range=(-0.2, 0.2),
        covariate_reference_scale=3.3,
        mm_scale_prior="log_uniform",
        **kwargs,
    )
    corpus, records = _capture_generation(cfg, controlled_components=True)
    return cfg, corpus, records


def _accepted_candidates(corpus, records):
    matches = []
    for row, cell in enumerate(corpus["cell_id"]):
        candidates = []
        for record in records:
            if record["cell"] != cell:
                continue
            for index, outcome in enumerate(record["batch"]["outcome"]):
                if np.array_equal(outcome.astype(np.float32), corpus["outcome_raw"][row]):
                    candidates.append((record, index))
        assert len(candidates) == 1, (row, len(candidates))
        matches.append(candidates[0])
    return matches


def _trajectory_report(key, role):
    leaf = key.removeprefix(f"trajectory_{role}_")
    if leaf in ("hf_sigma", "pulse_amp", "pulse_prob"):
        return f"param_{leaf}" if role == "treatment" else f"param_covariate_{leaf}"
    return f"param_{key}"


def test_realised_manifest_matches_the_accepted_joint_draw_exactly(rich_sample):
    _, corpus, records = rich_sample
    assert {key for key in corpus if key.startswith("trajectory_")} == set(TRAJECTORY_MANIFEST)
    assert corpus["diagnostics"]["trajectory"]["parameter_fields"] == list(TRAJECTORY_MANIFEST)
    assert corpus["diagnostics"]["trajectory"]["jump_counts"] == {"treatment": 2, "covariate": 3}
    assert corpus["diagnostics"]["mechanism_priors"]["parameter_fields"] == list(MECHANISM_MANIFEST)
    candidates = _accepted_candidates(corpus, records)
    assert any(index > 0 for _, index in candidates)
    widths = {"treatment": 4, "covariate": 4}
    for row, (record, _) in enumerate(candidates):
        for role in widths:
            active_count = int(corpus[f"n_{role}s_active"][row])
            expected_flags = np.zeros((widths[role], len(COMPONENTS)), dtype=np.uint8)
            for column, component in enumerate(COMPONENTS):
                expected_flags[:active_count, column] = record["structural"][
                    structural_key(role, component)
                ]
            np.testing.assert_array_equal(
                corpus[f"{role}_components"][row], expected_flags, err_msg=role
            )
    for key, (role, component, dtype, count) in TRAJECTORY_MANIFEST.items():
        shape = (6, widths[role]) if count is None else (6, count, widths[role])
        assert corpus[key].shape == shape, key
        assert corpus[key].dtype == dtype, key
        for row, (record, index) in enumerate(candidates):
            active_count = int(corpus[f"n_{role}s_active"][row])
            flags = np.asarray(record["structural"][structural_key(role, component)], dtype=bool)
            report = _trajectory_report(key, role)
            selected = np.zeros(widths[role], dtype=bool)
            selected[:active_count] = flags
            values = corpus[key][row]
            if flags.any():
                assert report in record["batch"], report
                np.testing.assert_array_equal(
                    values[..., :active_count][..., flags],
                    record["batch"][report][index][..., flags],
                    err_msg=key,
                )
            assert not values[..., ~selected].any(), key
    for key, (role, dtype) in MECHANISM_MANIFEST.items():
        assert corpus[key].shape == (6, widths[role]), key
        assert corpus[key].dtype == dtype, key
        for row, (record, index) in enumerate(candidates):
            active_count = int(corpus[f"n_{role}s_active"][row])
            expected = np.zeros(widths[role], dtype=dtype)
            if key == "sat_family":
                source = record["structural"]["sat_family"]
            else:
                report = (
                    "saturation_scale" if key == "mechanism_saturation_scale" else f"param_{key}"
                )
                source = record["batch"][report][index]
            expected[:active_count] = source
            np.testing.assert_array_equal(corpus[key][row], expected, err_msg=key)
    # A component can be absent from every active input in one cell. Its fields
    # are explicit zero padding, not unrelated random values from another cell.
    last_cell = corpus["cell_id"] == 2
    for field in ("amplitude", "period", "phase"):
        assert not corpus[f"trajectory_covariate_seasonal_{field}"][last_cell].any()
    assert DataGenerator.validate_corpus(corpus) == []


def _independent_schedules(corpus, role):
    n_tasks, n_time, width = corpus[f"{role}_activity"].shape
    activity = np.zeros((n_tasks, n_time, width), dtype=np.uint8)
    shifts = np.zeros((n_tasks, n_time, width), dtype=np.float64)
    weeks = np.arange(n_time)
    for row in range(n_tasks):
        for slot in range(width):
            if not corpus[f"{role}_active_mask"][row, slot]:
                continue
            flags = dict(zip(COMPONENTS, corpus[f"{role}_components"][row, slot], strict=True))

            def value(component, field, row=row, slot=slot):
                data = corpus[f"trajectory_{role}_{component}_{field}"]
                return data[row, slot] if data.ndim == 2 else data[row, :, slot]

            on = np.ones(n_time, dtype=bool)
            if flags["onset"]:
                on &= weeks >= value("onset", "start")
            if flags["offset"]:
                on &= weeks < value("offset", "stop")
            if flags["flighting"]:
                on &= (weeks + value("flighting", "phase")) % value("flighting", "period") < value(
                    "flighting", "on_weeks"
                )
            activity[row, :, slot] = on
            if flags["seasonal"]:
                shifts[row, :, slot] += value("seasonal", "amplitude") * np.sin(
                    2 * np.pi * weeks / value("seasonal", "period") + value("seasonal", "phase")
                )
            if flags["trend"]:
                shifts[row, :, slot] += value("trend", "change") * weeks / (n_time - 1)
            if flags["level_jump"]:
                sizes = value("level_jump", "log_factor" if role == "treatment" else "size")
                for week, size in zip(value("level_jump", "week"), sizes, strict=True):
                    shifts[row, weeks >= week, slot] += size
    return activity, shifts


def test_stored_truth_reconstructs_schedules_independently(rich_sample):
    _, corpus, _ = rich_sample
    for role in ("treatment", "covariate"):
        activity, shift = _independent_schedules(corpus, role)
        np.testing.assert_array_equal(corpus[f"{role}_activity"], activity)
        shift_key = "treatment_log_level_shift" if role == "treatment" else "covariate_level_shift"
        np.testing.assert_allclose(
            corpus[shift_key], shift.astype(np.float32), rtol=1e-6, atol=2e-7
        )
    selected = corpus["trajectory_treatment_level_jump_factor"] > 0
    np.testing.assert_allclose(
        corpus["trajectory_treatment_level_jump_log_factor"][selected],
        np.log(corpus["trajectory_treatment_level_jump_factor"][selected]),
        rtol=2e-15,
        atol=2e-15,
    )


def test_rich_archive_preserves_every_key_dtype_shape_value_and_metadata(tmp_path, rich_sample):
    _, corpus, _ = rich_sample
    path = tmp_path / "rich-v5.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert set(loaded) == set(corpus)
    for key, value in corpus.items():
        if key == "diagnostics":
            assert loaded[key] == {name: item for name, item in value.items() if name != "timing"}
        elif isinstance(value, dict):
            assert set(loaded[key]) == set(value)
            for name, array in value.items():
                assert loaded[key][name].dtype == array.dtype
                assert loaded[key][name].shape == array.shape
                np.testing.assert_array_equal(loaded[key][name], array)
        else:
            assert loaded[key].dtype == value.dtype
            assert loaded[key].shape == value.shape
            np.testing.assert_array_equal(loaded[key], value, err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []
    for role in ("treatment", "covariate"):
        reconstructed, _ = _independent_schedules(loaded, role)
        np.testing.assert_array_equal(loaded[f"{role}_activity"], reconstructed)


@pytest.mark.parametrize("role", ("treatment", "covariate"))
def test_selected_seasonal_period_retains_primitive_ceiling_and_roundtrips(tmp_path, role):
    maximum = float(np.finfo(np.float32).max)
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=1,
        seed=751,
        treatment_seasonal_inclusion_prob=1.0,
        covariate_seasonal_inclusion_prob=1.0,
        treatment_seasonal_period_weeks_range=(maximum, maximum),
        covariate_seasonal_period_weeks_range=(maximum, maximum),
    )
    corpus = pg.sample_prior_predictive(cfg)
    key = f"trajectory_{role}_seasonal_period"
    active = corpus[f"{role}_active_mask"] == 1
    np.testing.assert_array_equal(corpus[key][active], np.full(active.sum(), maximum))
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / f"seasonal-period-{role}.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert loaded[key].dtype == np.float64
    np.testing.assert_array_equal(loaded[key], corpus[key])
    assert DataGenerator.validate_corpus(loaded) == []

    above_maximum = np.nextafter(maximum, np.inf)
    corrupted = copy.deepcopy(corpus)
    corrupted[key][active] = above_maximum
    assert any(key in error for error in DataGenerator.validate_corpus(corrupted))
    setattr(cfg, f"{role}_seasonal_period_weeks_range", (above_maximum, above_maximum))
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    ("component", "field", "prior", "sign"),
    (
        ("seasonal", "amplitude", "covariate_seasonal_amplitude_range", 1.0),
        ("level_jump", "size", "covariate_level_jump_size_range", 1.0),
        ("level_jump", "size", "covariate_level_jump_size_range", -1.0),
        ("trend", "change", "covariate_trend_change_range", 1.0),
        ("trend", "change", "covariate_trend_change_range", -1.0),
    ),
)
def test_selected_unscaled_trajectory_truth_retains_primitive_ceiling_and_roundtrips(
    tmp_path, component, field, prior, sign
):
    maximum = float(np.finfo(np.float32).max)
    endpoint = sign * maximum
    options = {f"covariate_{component}_inclusion_prob": 1.0, prior: (endpoint, endpoint)}
    if component == "level_jump":
        options["covariate_level_jump_count"] = 1
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=1,
        seed=751,
        outcome_std_mode="absolute",
        edge_budget={"cy": 0, "dc": 0, "dz": 0, "dy": 0, "zy": 0, "zc": 0, "cc": 0, "zz": 0},
        **options,
    )
    corpus = pg.sample_prior_predictive(cfg)
    key = f"trajectory_covariate_{component}_{field}"
    active = corpus["covariate_active_mask"] == 1
    if corpus[key].ndim == 3:
        active = np.broadcast_to(active[:, None, :], corpus[key].shape)
    np.testing.assert_array_equal(corpus[key][active], np.full(active.sum(), endpoint))
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / f"primitive-{component}-{sign}.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert loaded[key].dtype == np.float64
    np.testing.assert_array_equal(loaded[key], corpus[key])
    assert DataGenerator.validate_corpus(loaded) == []

    above_maximum = sign * np.nextafter(maximum, np.inf)
    corrupted = copy.deepcopy(corpus)
    corrupted[key][active] = above_maximum
    assert any(key in error for error in DataGenerator.validate_corpus(corrupted))
    setattr(cfg, prior, (above_maximum, above_maximum))
    with pytest.raises(ValueError):
        cfg.validate()


def _tiny_mm_recipe(**overrides):
    ranges = copy.deepcopy(pg.SCMPrior().saturation_prior_ranges)
    ranges["michaelis_menten"]["kappa_mult"] = (1e-290, 1e-280)
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=1,
        seed=751,
        saturation_family_probs={
            "linear": 0.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 1.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        saturation_prior_ranges=ranges,
        mm_scale_prior="log_uniform",
        treatment_reference_contribution_range=(1e-5, 2e-5),
        treatment_reference_multiplier=1e-50,
        covariate_reference_contribution_range=(-1e-291, 1e-291),
        covariate_reference_scale=1e-290,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _shape_only_mm_recipe(**overrides):
    return _tiny_mm_recipe(
        **{
            "l_max": 4,
            "carryover_burn_in": 0,
            "carryover_family_probs": {"none": 1.0, "geometric": 0.0, "weibull": 0.0},
            "treatment_reference_contribution_range": None,
            "covariate_reference_contribution_range": None,
            **overrides,
        }
    )


@pytest.fixture(scope="module")
def mm_shape_sample():
    cfg = _shape_only_mm_recipe()
    return DataGenerator(cfg).generate()


@pytest.fixture(scope="module")
def conditioned_shape_sample():
    cfg = _shape_only_mm_recipe(prior_conditioning=True)
    return DataGenerator(cfg).generate()


@pytest.fixture(scope="module")
def linear_shape_sample():
    cfg = _shape_only_mm_recipe(
        saturation_family_probs={
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        }
    )
    return DataGenerator(cfg).generate()


@pytest.fixture(scope="module")
def mm_reference_sample():
    cfg = _shape_only_mm_recipe(treatment_reference_contribution_range=(1e-5, 2e-5))
    return cfg, DataGenerator(cfg).generate()


@pytest.mark.parametrize("key", SATURATION_SHAPE_PRIORS)
@pytest.mark.parametrize("corruption", ("zero", "negative", "above_maximum"))
def test_paired_shape_truth_and_recipes_cannot_escape_primitive_domains(
    mm_shape_sample, key, corruption
):
    broken = copy.deepcopy(mm_shape_sample)
    value = {
        "zero": 0.0,
        "negative": -1.0,
        "above_maximum": 1.01 if key == "root_alpha" else 1e100,
    }[corruption]
    active = broken["treatment_active_mask"] == 1
    broken[key][active] = value
    family, parameter = SATURATION_SHAPE_PRIORS[key]
    broken["diagnostics"]["mechanism_priors"]["saturation_prior_ranges"][family][parameter] = [
        value,
        value,
    ]

    errors = DataGenerator.validate_corpus(broken)
    assert any(key in error for error in errors)
    assert any(
        "diagnostics mechanism_priors" in error and f"{family}.{parameter}" in error
        for error in errors
    )


@pytest.mark.parametrize("key", SATURATION_SHAPE_PRIORS)
def test_impossible_shape_recipe_endpoints_cannot_hide_behind_admissible_draws(
    mm_shape_sample, key
):
    broken = copy.deepcopy(mm_shape_sample)
    family, parameter = SATURATION_SHAPE_PRIORS[key]
    ranges = broken["diagnostics"]["mechanism_priors"]["saturation_prior_ranges"]
    lo = ranges[family][parameter][0]
    ranges[family][parameter] = [lo, np.nextafter(_shape_maximum(key), np.inf)]
    errors = DataGenerator.validate_corpus(broken)
    assert any(
        "diagnostics mechanism_priors" in error and f"{family}.{parameter}" in error
        for error in errors
    )


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("setting", ("treatment_reference_multiplier", "covariate_reference_scale"))
def test_reference_recipe_scalars_retain_the_public_magnitude_limit(
    rich_sample, mm_shape_sample, setting, enabled
):
    broken = copy.deepcopy(rich_sample[1] if enabled else mm_shape_sample)
    broken["diagnostics"]["mechanism_priors"][setting] = np.nextafter(
        float(np.finfo(np.float32).max), np.inf
    )
    cfg = copy.deepcopy(rich_sample[0]) if enabled else _shape_only_mm_recipe()
    setattr(cfg, setting, broken["diagnostics"]["mechanism_priors"][setting])
    with pytest.raises(ValueError, match=setting):
        cfg.validate()
    assert any(
        f"diagnostics mechanism_priors {setting}" in error
        for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("role", "endpoint"), (("treatment", 1), ("covariate", 0), ("covariate", 1))
)
def test_reference_recipe_supports_retain_signed_public_endpoint_limits(
    rich_sample, role, endpoint
):
    broken = copy.deepcopy(rich_sample[1])
    metadata = broken["diagnostics"]["mechanism_priors"]
    bounds = list(metadata[f"{role}_reference_contribution_range"])
    above_maximum = np.nextafter(float(np.finfo(np.float32).max), np.inf)
    bounds[endpoint] = -above_maximum if endpoint == 0 else above_maximum
    metadata[f"{role}_reference_contribution_range"] = bounds
    cfg = copy.deepcopy(rich_sample[0])
    setattr(cfg, f"{role}_reference_contribution_range", tuple(bounds))
    with pytest.raises(ValueError, match=f"{role}_reference_contribution_range"):
        cfg.validate()
    assert any(
        f"diagnostics mechanism_priors {role} reference range has invalid bounds" in error
        for error in DataGenerator.validate_corpus(broken)
    )


def test_reference_recipe_support_maxima_remain_admissible(rich_sample):
    corpus = copy.deepcopy(rich_sample[1])
    maximum = float(np.finfo(np.float32).max)
    metadata = corpus["diagnostics"]["mechanism_priors"]
    metadata["treatment_reference_contribution_range"] = [0.0, maximum]
    metadata["covariate_reference_contribution_range"] = [-maximum, maximum]
    assert DataGenerator.validate_corpus(corpus) == []


@pytest.mark.parametrize("fixture", ("mm_shape_sample", "linear_shape_sample"))
def test_mm_recipe_rejects_a_minimum_product_that_underflows(request, fixture):
    corpus = request.getfixturevalue(fixture)
    broken = copy.deepcopy(corpus)
    bounds = broken["diagnostics"]["mechanism_priors"]["saturation_prior_ranges"][
        "michaelis_menten"
    ]["kappa_mult"]
    bounds[0] = np.nextafter(0.0, 1.0)
    active = broken["treatment_active_mask"] == 1
    values = broken["mm_kappa_mult"][active]
    assert ((values >= bounds[0]) & (values <= bounds[1])).all()
    cfg = _shape_only_mm_recipe()
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = tuple(bounds)
    with pytest.raises(ValueError, match="michaelis_menten.*kappa_mult.*minimum saturation_scale"):
        cfg.validate()
    assert any(
        "diagnostics mechanism_priors michaelis_menten.kappa_mult lower bound" in error
        and "minimum saturation_scale" in error
        for error in DataGenerator.validate_corpus(broken)
    )


def test_mm_log_uniform_recipe_rejects_coherently_collapsed_bounds(mm_shape_sample):
    broken = copy.deepcopy(mm_shape_sample)
    lo, hi = 1e20, np.nextafter(1e20, np.inf)
    metadata = broken["diagnostics"]["mechanism_priors"]
    metadata["saturation_prior_ranges"]["michaelis_menten"]["kappa_mult"] = [lo, hi]
    metadata["mm_scale_prior"] = "log_uniform"
    active = broken["treatment_active_mask"] == 1
    broken["mm_kappa_mult"][active] = lo
    cfg = _shape_only_mm_recipe()
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = (lo, hi)
    with pytest.raises(ValueError, match="michaelis_menten.*kappa_mult.*collapse in log space"):
        cfg.validate()
    assert any(
        "diagnostics mechanism_priors michaelis_menten.kappa_mult" in error
        and "collapse in log space" in error
        for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("mode", "bounds"),
    (
        ("log_uniform", (1e20, 1e20)),
        ("uniform", (1e20, np.nextafter(1e20, np.inf))),
        ("log_uniform", (1e-310, 1e-310)),
    ),
)
def test_admissible_mm_boundaries_roundtrip_and_keep_exact_draw_support(tmp_path, mode, bounds):
    cfg = _shape_only_mm_recipe(mm_scale_prior=mode)
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = bounds
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    values = corpus["mm_kappa_mult"][active]
    assert ((values >= bounds[0]) & (values <= bounds[1])).all()
    if bounds[0] == bounds[1]:
        np.testing.assert_array_equal(values, np.full(active.sum(), bounds[0]))
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "admissible-mm-boundaries.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert loaded["mm_kappa_mult"].dtype == np.float64
    np.testing.assert_array_equal(loaded["mm_kappa_mult"], corpus["mm_kappa_mult"])
    assert DataGenerator.validate_corpus(loaded) == []

    broken = copy.deepcopy(loaded)
    broken["mm_kappa_mult"][active] = np.nextafter(bounds[1], np.inf)
    assert any(
        "mm_kappa_mult is outside its declared prior support" in error
        for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("key", "setting", "sign"),
    (
        ("beta", "beta_additive_range", 1),
        ("rho_zy", "zy_coeff_range", -1),
        ("rho_zy", "zy_coeff_range", 1),
    ),
)
def test_raw_coefficients_retain_signed_magnitude_limits_without_references(
    mm_shape_sample, key, setting, sign
):
    broken = copy.deepcopy(mm_shape_sample)
    role = "treatment" if key == "beta" else "covariate"
    value = sign * np.nextafter(float(np.finfo(np.float32).max), np.inf)
    broken[key][broken[f"{role}_active_mask"] == 1] = value
    cfg = _shape_only_mm_recipe(**{setting: (value, value)})
    with pytest.raises(ValueError, match=setting):
        cfg.validate()
    assert any(
        f"{key} exceeds the configured primitive maximum" in error
        for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize("sign", (-1, 1))
def test_raw_coefficient_exact_maxima_are_admissible_and_roundtrip(tmp_path, sign):
    maximum = float(np.finfo(np.float32).max)
    cfg = _shape_only_mm_recipe(
        beta_additive_range=(maximum, maximum),
        zy_coeff_range=(sign * maximum, sign * maximum),
        rw_std_sigma=1e-20,
        rw_covariate_mean_range=(1e-20, 1e-20),
        outcome_std_mode="absolute",
        edge_budget={"cy": 0, "dc": 0, "dz": 0, "dy": 0, "zy": 0, "zc": 0, "cc": 0, "zz": 0},
    )
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = (1e20, 1e20)
    corpus = DataGenerator(cfg).generate()
    for key, role, value in (
        ("beta", "treatment", maximum),
        ("rho_zy", "covariate", sign * maximum),
    ):
        active = corpus[f"{role}_active_mask"] == 1
        np.testing.assert_array_equal(corpus[key][active], np.full(active.sum(), value))
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "raw-coefficient-maxima.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in ("beta", "rho_zy"):
        assert loaded[key].dtype == np.float64
        np.testing.assert_array_equal(loaded[key], corpus[key])
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize(
    ("field", "multiplier"),
    (("treatment_reference_input", 1e-310), ("treatment_reference_multiplier", 1e-302)),
)
def test_coherent_treatment_references_retain_public_input_normal_floors(
    mm_reference_sample, field, multiplier
):
    cfg, corpus = mm_reference_sample
    cfg, broken = copy.deepcopy(cfg), copy.deepcopy(corpus)
    target = 1e-24
    metadata = broken["diagnostics"]["mechanism_priors"]
    metadata["treatment_reference_multiplier"] = multiplier
    metadata["treatment_reference_contribution_range"] = [target, target]
    cfg.treatment_reference_multiplier = multiplier
    cfg.treatment_reference_contribution_range = (target, target)
    active = broken["treatment_active_mask"] == 1
    anchor = broken["mechanism_saturation_scale"][active]
    reference = multiplier * anchor
    response = reference / (broken["mm_kappa_mult"][active] * anchor + reference)
    broken["treatment_reference_input"][active] = reference
    broken["treatment_reference_response"][active] = response
    broken["treatment_reference_contribution"][active] = target
    broken["beta"][active] = target / response
    assert np.isfinite(broken["beta"]).all()
    assert (np.abs(broken["beta"]) <= float(np.finfo(np.float32).max)).all()
    if field == "treatment_reference_multiplier":
        assert (reference >= np.finfo(np.float64).tiny).all()
        assert (response >= np.finfo(np.float64).tiny).all()
        expected = "diagnostics mechanism_priors treatment_reference_multiplier"
    else:
        expected = f"{field} must be normal positive float64"
    with pytest.raises(ValueError, match="normal positive treatment_reference_input"):
        cfg.validate()
    assert any(expected in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("corruption", ("recipe_only", "paired_truth"))
def test_represented_mm_reference_recipes_retain_the_public_minimum_response(
    mm_reference_sample, corruption
):
    cfg, corpus = mm_reference_sample
    cfg, broken = copy.deepcopy(cfg), copy.deepcopy(corpus)
    multiplier = 3e-300
    target = 1e-320 if corruption == "paired_truth" else 1e-24
    metadata = broken["diagnostics"]["mechanism_priors"]
    metadata["treatment_reference_multiplier"] = multiplier
    metadata["treatment_reference_contribution_range"] = [target, target]
    cfg.treatment_reference_multiplier = multiplier
    cfg.treatment_reference_contribution_range = (target, target)
    lo = (
        1e20
        if corruption == "paired_truth"
        else cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"][0]
    )
    metadata["saturation_prior_ranges"]["michaelis_menten"]["kappa_mult"] = [lo, 1e20]
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = (lo, 1e20)
    active = broken["treatment_active_mask"] == 1
    if corruption == "paired_truth":
        broken["mm_kappa_mult"][active] = 1e20
    anchor = broken["mechanism_saturation_scale"][active]
    reference = multiplier * anchor
    response = reference / (broken["mm_kappa_mult"][active] * anchor + reference)
    broken["treatment_reference_input"][active] = reference
    broken["treatment_reference_response"][active] = response
    broken["treatment_reference_contribution"][active] = target
    broken["beta"][active] = target / response
    assert (reference >= np.finfo(np.float64).tiny).all()
    assert np.isfinite(broken["beta"]).all()
    assert (np.abs(broken["beta"]) <= float(np.finfo(np.float32).max)).all()
    if corruption == "recipe_only":
        assert (response >= np.finfo(np.float64).tiny).all()
    with pytest.raises(ValueError, match="unrepresentable reference response"):
        cfg.validate()
    assert any(
        "diagnostics mechanism_priors michaelis_menten.kappa_mult" in error
        and "treatment_reference_multiplier" in error
        and "minimum reference response" in error
        for error in DataGenerator.validate_corpus(broken)
    )


def test_unused_mm_reference_response_bounds_do_not_constrain_a_represented_linear_family(
    tmp_path,
):
    cfg = _shape_only_mm_recipe(
        treatment_reference_multiplier=3e-300,
        treatment_reference_contribution_range=(1e-320, 1e-320),
        saturation_family_probs={
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        outcome_std_mode="absolute",
    )
    cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = (1e20, 1e20)
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    np.testing.assert_array_equal(
        corpus["sat_family"][active], np.zeros(active.sum(), dtype=np.uint8)
    )
    np.testing.assert_allclose(
        corpus["treatment_reference_response"][active],
        3e-300,
        rtol=8 * np.finfo(float).eps,
        atol=0.0,
    )
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "unused-mm-response-bound.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in ("mm_kappa_mult", "treatment_reference_response", "beta"):
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize("boundary", ("input_recipe", "response", "maximum_shape_response"))
def test_treatment_normal_reference_boundaries_remain_admissible(tmp_path, boundary):
    minimum = np.nextafter(0.0, 1.0)
    normal = float(np.finfo(np.float64).tiny)
    cfg = _shape_only_mm_recipe(treatment_reference_contribution_range=(minimum, minimum))
    if boundary == "input_recipe":
        cfg.saturation_family_probs = {
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        }
        cfg.treatment_reference_multiplier = normal / 1e-8
        assert cfg.treatment_reference_multiplier * 1e-8 == normal
    else:
        kappa = (
            float(np.finfo(np.float32).max)
            if boundary == "maximum_shape_response"
            else float(2.0**100)
        )
        cfg.saturation_prior_ranges["michaelis_menten"]["kappa_mult"] = (kappa, kappa)
        cfg.treatment_reference_multiplier = normal * kappa
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    assert (corpus["treatment_reference_input"][active] >= normal).all()
    assert (corpus["treatment_reference_response"][active] >= normal).all()
    assert (corpus["beta"][active] > 0.0).all()
    if boundary != "input_recipe":
        np.testing.assert_array_equal(
            corpus["treatment_reference_response"][active], np.full(active.sum(), normal)
        )
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "normal-treatment-reference.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in (
        "treatment_reference_contribution",
        "treatment_reference_input",
        "treatment_reference_response",
        "beta",
    ):
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize(
    ("family", "parameter", "shape", "multiplier"),
    (
        ("logistic", "lam", 1.5e-8, 2.9667651446762687e-300),
        ("tanh", "c", 1e9, 2.2250738585072014e-299),
        ("hill", "slope", 1e12, 0.9999999992916037),
    ),
)
def test_admitted_support_floor_responses_preserve_rounded_truth(
    tmp_path, family, parameter, shape, multiplier
):
    # Public admission bounds the ideal support response; an actual rounded
    # forward response near that boundary can be subnormal.
    minimum = np.nextafter(0.0, 1.0)
    cfg = _shape_only_mm_recipe(
        n_cells=8,
        treatment_reference_multiplier=multiplier,
        treatment_reference_contribution_range=(minimum, minimum),
        saturation_family_probs={
            name: float(name == family) for name in pg.SCMPrior().saturation_family_probs
        },
    )
    cfg.saturation_prior_ranges[family][parameter] = (shape, shape)
    if family == "hill":
        cfg.saturation_prior_ranges[family]["kappa_mult"] = (1.0, 1.0)
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    response = corpus["treatment_reference_response"][active]
    assert ((response > 0.0) & (response < 2.0 * np.finfo(np.float64).tiny)).all()
    np.testing.assert_array_equal(
        corpus["treatment_reference_contribution"][active], np.full(active.sum(), minimum)
    )
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "rounded-treatment-reference.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in ("treatment_reference_response", "treatment_reference_contribution", "beta"):
        assert loaded[key].dtype == np.float64
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize("sign", (-1, 1))
def test_signed_subnormal_covariate_targets_and_inputs_remain_admissible(tmp_path, sign):
    target = sign * np.nextafter(0.0, 1.0)
    scale = 1e-308
    cfg = _shape_only_mm_recipe(
        covariate_reference_contribution_range=(target, target),
        covariate_reference_scale=scale,
    )
    corpus = DataGenerator(cfg).generate()
    active = corpus["covariate_active_mask"] == 1
    np.testing.assert_array_equal(
        corpus["covariate_reference_contribution"][active], np.full(active.sum(), target)
    )
    np.testing.assert_array_equal(
        corpus["covariate_reference_input"][active], np.full(active.sum(), scale)
    )
    np.testing.assert_array_equal(
        corpus["rho_zy"][active], np.full(active.sum(), target * (1.0 / scale))
    )
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "signed-subnormal-covariate.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in ("covariate_reference_contribution", "covariate_reference_input", "rho_zy"):
        assert loaded[key].dtype == np.float64
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize(
    ("quantity", "support"),
    (
        ("carryover_alpha", (-1.0, 1.0)),
        ("carryover_alpha", (0.0, np.nextafter(1.0, np.inf))),
        ("hill_shape", (0.0, 3.0)),
        ("hill_shape", (1.0, np.nextafter(float(np.finfo(np.float32).max), np.inf))),
    ),
)
def test_conditioned_shape_supports_retain_the_configured_endpoint_domains(
    conditioned_shape_sample, quantity, support
):
    broken = copy.deepcopy(conditioned_shape_sample)
    broken["diagnostics"]["prior_cond"]["supports"][quantity] = list(support)
    assert any(
        f"diagnostics prior_cond {quantity} bounds are invalid" in error
        for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize("boundary", ("tiny", "maximum"))
def test_admissible_shape_endpoints_preserve_float64_truth_and_roundtrip(tmp_path, boundary):
    minimum = np.nextafter(0.0, 1.0)
    maximum = float(np.finfo(np.float32).max)
    values = {
        key: _shape_maximum(key) if boundary == "maximum" else minimum
        for key in SATURATION_SHAPE_PRIORS
    }
    if boundary == "tiny":
        values["mm_kappa_mult"] = 1e-310
    carryover_point = maximum if boundary == "maximum" else minimum
    alpha_point = 1.0 if boundary == "maximum" else minimum
    ranges = copy.deepcopy(pg.SCMPrior().saturation_prior_ranges)
    for key, (family, parameter) in SATURATION_SHAPE_PRIORS.items():
        ranges[family][parameter] = (values[key], values[key])
    cfg = _tiny_mm_recipe(
        l_max=4,
        carryover_burn_in=0,
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        carryover_alpha_range=(alpha_point, alpha_point),
        weibull_lam_range=(carryover_point, carryover_point),
        weibull_k_range=(carryover_point, carryover_point),
        saturation_family_probs={
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        saturation_prior_ranges=ranges,
        mm_scale_prior="uniform",
        treatment_reference_contribution_range=None,
        covariate_reference_contribution_range=None,
    )
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    for key, value in values.items():
        np.testing.assert_array_equal(corpus[key][active], np.full(active.sum(), value))
    for key, value in (
        ("carryover_alpha", 1.0 if boundary == "maximum" else 0.0),
        ("weibull_lam", maximum if boundary == "maximum" else 0.0),
        ("weibull_k", maximum if boundary == "maximum" else 0.0),
    ):
        np.testing.assert_array_equal(corpus[key][active], np.full(active.sum(), value))
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / f"shape-{boundary}.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in (*values, "carryover_alpha", "weibull_lam", "weibull_k"):
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


def test_amplified_reference_inputs_above_primitive_maximum_remain_valid(tmp_path):
    maximum = float(np.finfo(np.float32).max)
    cfg = _tiny_mm_recipe(
        l_max=4,
        carryover_burn_in=0,
        carryover_family_probs={"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        saturation_family_probs={
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        treatment_reference_contribution_range=(1.0, 1.0),
        treatment_reference_multiplier=maximum,
        covariate_reference_contribution_range=None,
        rw_positive_mean_range=(10.0, 10.0),
        edge_budget={"cy": (1, 1), "dc": 0, "dz": 0, "dy": 0, "zy": 0, "zc": 0, "cc": 0, "zz": 0},
    )
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    reference = corpus["treatment_reference_input"][active]
    assert (reference > maximum).all()
    np.testing.assert_array_equal(reference, maximum * corpus["mechanism_saturation_scale"][active])
    assert ((corpus["beta"][active] > 0.0) & (corpus["beta"][active] < 1e-38)).all()
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "amplified-reference.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in ("mechanism_saturation_scale", "treatment_reference_input", "beta"):
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


def test_sub_float32_positive_shapes_and_reference_inputs_are_not_rounded(tmp_path):
    corpus, records = _capture_generation(_tiny_mm_recipe())
    candidates = _accepted_candidates(corpus, records)
    for key in ("mm_kappa_mult", "treatment_reference_input", "covariate_reference_input"):
        role = "covariate" if key.startswith("covariate") else "treatment"
        active = corpus[f"{role}_active_mask"] == 1
        values = corpus[key][active]
        assert (values > 0).all()
        assert not values.astype(np.float32).any()
        for row, (record, index) in enumerate(candidates):
            n = int(corpus[f"n_{role}s_active"][row])
            np.testing.assert_array_equal(
                corpus[key][row, :n], record["batch"][f"param_{key}"][index]
            )
    assert DataGenerator.validate_corpus(corpus) == []
    path = tmp_path / "sub-float32.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key in MECHANISM_MANIFEST:
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert DataGenerator.validate_corpus(loaded) == []


def test_subnormal_reference_coefficients_roundtrip_without_erasing_truth(tmp_path):
    ranges = copy.deepcopy(pg.SCMPrior().saturation_prior_ranges)
    ranges["root"]["alpha"] = (0.9, 0.9)
    cfg = pg.make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=1,
        l_max=4,
        carryover_burn_in=4,
        seed=751,
        saturation_family_probs={
            "linear": 0.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 1.0,
        },
        saturation_prior_ranges=ranges,
        treatment_reference_contribution_range=(1e-310, 1e-310),
        treatment_reference_multiplier=1e10,
        covariate_reference_contribution_range=(-1e-310, -1e-310),
        covariate_reference_scale=1.0,
    )
    corpus = DataGenerator(cfg).generate()
    active = corpus["treatment_active_mask"] == 1
    path = tmp_path / "subnormal-coefficients.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    assert DataGenerator.validate_corpus(loaded) == []
    for key in (
        "beta",
        "treatment_reference_contribution",
        "treatment_reference_input",
        "treatment_reference_response",
        "rho_zy",
        "covariate_reference_contribution",
        "covariate_reference_input",
    ):
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    for key, value in (("beta", 0.0), ("treatment_reference_response", 1e-310)):
        broken = copy.deepcopy(loaded)
        broken[key][active] = value
        assert DataGenerator.validate_corpus(broken)
    for role, coefficient in (("treatment", "beta"), ("covariate", "rho_zy")):
        mask = loaded[f"{role}_active_mask"] == 1
        values = loaded[coefficient][mask]
        assert ((np.abs(values) > 0.0) & (np.abs(values) < np.finfo(np.float64).tiny)).all()

        erased = copy.deepcopy(loaded)
        if role == "treatment":
            erased["treatment_reference_response"][mask] = 1e38
        else:
            erased["diagnostics"]["mechanism_priors"]["covariate_reference_scale"] = 1e38
            erased["covariate_reference_input"][mask] = 1e38
        erased[coefficient][mask] = 0.0
        assert any(coefficient in error for error in DataGenerator.validate_corpus(erased))

        oversized = copy.deepcopy(loaded)
        target = 1.0 if role == "treatment" else -1.0
        oversized["diagnostics"]["mechanism_priors"][f"{role}_reference_contribution_range"] = [
            target,
            target,
        ]
        oversized[f"{role}_reference_contribution"][mask] = target
        if role == "treatment":
            oversized["treatment_reference_response"][mask] = 1e-100
        else:
            oversized["diagnostics"]["mechanism_priors"]["covariate_reference_scale"] = 1e-100
            oversized["covariate_reference_input"][mask] = 1e-100
        oversized[coefficient][mask] = target * 1e100
        assert any(coefficient in error for error in DataGenerator.validate_corpus(oversized))


def test_positive_reference_response_cannot_require_an_infinite_coefficient(rich_sample):
    _, corpus, _ = rich_sample
    broken = {
        **corpus,
        "treatment_reference_response": corpus["treatment_reference_response"].copy(),
    }
    active = corpus["treatment_active_mask"] == 1
    broken["treatment_reference_response"][active] = np.nextafter(0.0, 1.0)
    assert DataGenerator.validate_corpus(broken)


@pytest.mark.parametrize("key", (*TRAJECTORY_MANIFEST, *MECHANISM_MANIFEST))
def test_rich_blocks_require_every_declared_field(rich_sample, key):
    _, corpus, _ = rich_sample
    broken = {name: value for name, value in corpus.items() if name != key}
    assert any(key in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("block", ("trajectory", "mechanism_priors"))
@pytest.mark.parametrize("field", ("parameter_fields", "recipe"))
def test_rich_blocks_require_self_describing_metadata(rich_sample, block, field):
    _, corpus, _ = rich_sample
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    key = "jump_counts" if block == "trajectory" else "saturation_prior_ranges"
    broken["diagnostics"][block].pop("parameter_fields" if field == "parameter_fields" else key)
    assert any(block in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("block", ("trajectory", "mechanism_priors"))
def test_rich_arrays_cannot_outlive_their_recipe_metadata(rich_sample, block):
    _, corpus, _ = rich_sample
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    broken["diagnostics"].pop(block)
    assert any("present together" in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("role", ("treatment", "covariate"))
def test_role_specific_jump_axes_cannot_be_swapped(rich_sample, role):
    _, corpus, _ = rich_sample
    key = f"trajectory_{role}_level_jump_week"
    broken = {**corpus, key: corpus[key].transpose(0, 2, 1)}
    assert any(
        f"Shape mismatch for {key}" in error for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize("key", (*TRAJECTORY_MANIFEST, *MECHANISM_MANIFEST))
def test_rich_truth_dtypes_are_not_lossily_coerced(rich_sample, key):
    _, corpus, _ = rich_sample
    wrong = np.float32 if corpus[key].dtype == np.float64 else np.float64
    broken = {**corpus, key: corpus[key].astype(wrong)}
    assert any(f"{key} has dtype" in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("key", (*TRAJECTORY_MANIFEST, *MECHANISM_MANIFEST))
def test_rich_truth_requires_zero_inactive_input_padding(rich_sample, key):
    _, corpus, _ = rich_sample
    broken = {**corpus, key: corpus[key].copy()}
    broken[key][:, ..., -1] = 1
    assert any(
        key in error and "padding" in error for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize("key", TRAJECTORY_MANIFEST)
def test_unselected_component_leaves_cannot_carry_unrelated_draws(rich_sample, key):
    _, corpus, _ = rich_sample
    role, component, _, _ = TRAJECTORY_MANIFEST[key]
    inactive = corpus[f"{role}_components"][..., COMPONENTS.index(component)] == 0
    inactive &= corpus[f"{role}_active_mask"] == 1
    row, slot = np.argwhere(inactive)[0]
    broken = {**corpus, key: corpus[key].copy()}
    index = (row, slot) if corpus[key].ndim == 2 else (row, 0, slot)
    broken[key][index] = 1
    assert any(
        key in error and "padding" in error for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    "key",
    (
        "trajectory_treatment_seasonal_period",
        "trajectory_covariate_seasonal_period",
        "trajectory_treatment_level_jump_factor",
        "mm_kappa_mult",
        "mechanism_saturation_scale",
        "treatment_reference_input",
        "treatment_reference_response",
        "covariate_reference_input",
    ),
)
def test_live_positive_domains_are_enforced(rich_sample, key):
    _, corpus, _ = rich_sample
    broken = {**corpus, key: corpus[key].copy()}
    index = tuple(np.argwhere(corpus[key] > 0)[0])
    broken[key][index] = 0
    assert any(key in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("role", ("treatment", "covariate"))
def test_schedule_equivalent_flighting_phases_must_stay_canonical(rich_sample, role):
    _, corpus, _ = rich_sample
    key = f"trajectory_{role}_flighting_phase"
    selected = (corpus[f"{role}_active_mask"] == 1) & (
        corpus[f"{role}_components"][..., COMPONENTS.index("flighting")] == 1
    )
    row, slot = np.argwhere(selected)[0]
    broken = {**corpus, key: corpus[key].copy()}
    broken[key][row, slot] += corpus[f"trajectory_{role}_flighting_period"][row, slot]

    activity, shift = _independent_schedules(corpus, role)
    mutated_activity, mutated_shift = _independent_schedules(broken, role)
    np.testing.assert_array_equal(activity, corpus[f"{role}_activity"])
    np.testing.assert_array_equal(mutated_activity, activity)
    np.testing.assert_array_equal(mutated_shift, shift)
    shift_key = "treatment_log_level_shift" if role == "treatment" else "covariate_level_shift"
    np.testing.assert_allclose(shift.astype(np.float32), corpus[shift_key], rtol=1e-6, atol=2e-7)

    assert any(key in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("role", ("treatment", "covariate"))
def test_schedule_equivalent_jump_permutations_must_keep_chronological_slots(rich_sample, role):
    _, corpus, _ = rich_sample
    key = f"trajectory_{role}_level_jump_week"
    selected = (corpus[f"{role}_active_mask"] == 1) & (
        corpus[f"{role}_components"][..., COMPONENTS.index("level_jump")] == 1
    )
    row, slot = np.argwhere(selected)[0]
    assert (np.diff(corpus[key][row, :, slot]) > 0).all()
    fields = ("week", "factor", "log_factor") if role == "treatment" else ("week", "size")
    broken = dict(corpus)
    for field in fields:
        field_key = f"trajectory_{role}_level_jump_{field}"
        broken[field_key] = corpus[field_key].copy()
        broken[field_key][row, :, slot] = corpus[field_key][row, ::-1, slot]

    activity, shift = _independent_schedules(corpus, role)
    mutated_activity, mutated_shift = _independent_schedules(broken, role)
    np.testing.assert_array_equal(activity, corpus[f"{role}_activity"])
    np.testing.assert_array_equal(mutated_activity, activity)
    np.testing.assert_allclose(
        mutated_shift,
        shift,
        rtol=8 * np.finfo(np.float64).eps,
        atol=8 * np.finfo(np.float64).eps,
    )
    shift_key = "treatment_log_level_shift" if role == "treatment" else "covariate_level_shift"
    np.testing.assert_allclose(shift.astype(np.float32), corpus[shift_key], rtol=1e-6, atol=2e-7)

    assert any(key in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize(
    "key", ("trajectory_treatment_trend_change", "mm_kappa_mult", "treatment_reference_input")
)
def test_rich_truth_must_be_finite(rich_sample, key):
    _, corpus, _ = rich_sample
    broken = {**corpus, key: corpus[key].copy()}
    broken[key].flat[0] = np.nan
    assert any(
        f"{key} contains NaN or Inf" in error for error in DataGenerator.validate_corpus(broken)
    )


@pytest.mark.parametrize(
    ("key", "change", "expected"),
    (
        ("trajectory_treatment_onset_start", 1, "treatment_activity does not match"),
        ("trajectory_covariate_trend_change", 0.3, "covariate_level_shift does not match"),
        (
            "trajectory_treatment_level_jump_log_factor",
            0.2,
            "factor and log_factor are inconsistent",
        ),
        ("beta", 0.1, "beta does not match"),
        ("rho_zy", 0.1, "rho_zy does not match"),
        ("treatment_reference_input", 0.1, "treatment_reference_input does not match"),
    ),
)
def test_realised_truth_is_consistent_with_schedules_and_reference_targets(
    rich_sample, key, change, expected
):
    _, corpus, _ = rich_sample
    broken = {**corpus, key: corpus[key].copy()}
    if key in TRAJECTORY_MANIFEST:
        role, component, _, _ = TRAJECTORY_MANIFEST[key]
        row, slot = np.argwhere(
            corpus[f"{role}_components"][..., COMPONENTS.index(component)] == 1
        )[0]
        index = (row, slot) if corpus[key].ndim == 2 else (row, 0, slot)
    else:
        index = (0, 0)
    if key == "trajectory_treatment_onset_start":
        row, slot = index
        on_week = int(np.flatnonzero(corpus["treatment_activity"][row, :, slot])[0])
        broken[key][index] = on_week + 1
    else:
        broken[key][index] += change
    assert any(expected in error for error in DataGenerator.validate_corpus(broken))


def test_schema_rejects_undeclared_truth_fields_and_invented_inventory(rich_sample):
    _, corpus, _ = rich_sample
    key = "trajectory_treatment_flighting_duty"
    broken = {**corpus, key: np.zeros((6, 4), dtype=np.float64)}
    assert any(
        "Unrecognized corpus fields" in error for error in DataGenerator.validate_corpus(broken)
    )
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    broken["diagnostics"]["trajectory"]["parameter_fields"].append(key)
    assert any("parameter_fields" in error for error in DataGenerator.validate_corpus(broken))


@pytest.mark.parametrize("value", (True, 0, 1.5, {"treatment": 2}))
def test_jump_count_metadata_cannot_fabricate_event_axes(rich_sample, value):
    _, corpus, _ = rich_sample
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    if isinstance(value, dict):
        broken["diagnostics"]["trajectory"]["jump_counts"] = value
    else:
        broken["diagnostics"]["trajectory"]["jump_counts"]["covariate"] = value
    assert any("jump_counts" in error for error in DataGenerator.validate_corpus(broken))


def test_shape_only_recipes_do_not_invent_reference_targets():
    cfg = _tiny_mm_recipe(
        treatment_reference_contribution_range=None,
        covariate_reference_contribution_range=None,
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert not any("reference" in key for key in corpus)
    assert (
        corpus["diagnostics"]["mechanism_priors"]["treatment_reference_contribution_range"] is None
    )
    assert (
        corpus["diagnostics"]["mechanism_priors"]["covariate_reference_contribution_range"] is None
    )
    assert DataGenerator.validate_corpus(corpus) == []


def test_default_recipes_omit_optional_rich_blocks():
    corpus = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_time_steps=32,
            n_cells=2,
            draws_per_cell=1,
            seed=57,
        )
    )
    assert not set(corpus).intersection(TRAJECTORY_MANIFEST)
    assert not set(corpus).intersection(MECHANISM_MANIFEST)
    assert "trajectory" not in corpus["diagnostics"]
    assert "mechanism_priors" not in corpus["diagnostics"]
    assert DataGenerator.validate_corpus(corpus) == []
