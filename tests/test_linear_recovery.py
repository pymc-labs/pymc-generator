"""Checkpoint A: noiseless OLS recovery in a controlled generated world.

This is one deterministic world, not a prevalence survey.  The graph is
structure-known and deliberately excludes confounding, mediation, floors, and
latent paths so that the linear target is well specified.  Each regressor is
still computed independently from that treatment's generated carryover and
saturation mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_generator import mechanisms
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
from pymc_generator.symbolic_graph import build_symbolic_graph

SEED = 20261005
N_TIME_STEPS = 64
WARMUP = 4
L_MAX = 4
N_TREATMENTS = 3
N_COVARIATES = 2
N_LATENT = 1
NOISE_LEVELS = (1.2, 0.6, 0.3)
N_REPLICATES = 96
REPLICATE_SEED = 20301005


@dataclass(frozen=True)
class ControlledWorld:
    """Generated arrays and the fixed parameters used to generate them."""

    outputs: dict[str, np.ndarray]
    params: dict[str, object]
    graph: dict[str, np.ndarray]


def _rw(mean, std, *, positive_only: bool) -> dict[str, object]:
    mean_array = np.asarray(mean, dtype=float)
    return {
        "mean": mean_array,
        "std": np.asarray(std, dtype=float),
        "smoothness": np.zeros(mean_array.size),
        "positive_only": positive_only,
        "rw_smoothness_max_weeks": 26,
    }


def _controlled_world() -> ControlledWorld:
    """Build one seeded world through the production symbolic generation graph.

    The first ``WARMUP`` generated rows are retained as carryover history and
    excluded from the fit.  Treatments use geometric, Weibull, and geometric
    carryover; their saturations are Hill, Michaelis--Menten, and root.  The
    two observed controls are direct, while the latent is disconnected.
    """
    n_full = N_TIME_STEPS + WARMUP
    rng = np.random.default_rng(SEED)
    eps = {
        "eps_d": np.zeros((n_full, N_LATENT)),
        "eps_z": rng.normal(size=(n_full, N_COVARIATES)),
        "eps_c": rng.normal(size=(n_full, N_TREATMENTS)),
        "eps_b": np.zeros(n_full),
        "eps_y": np.zeros(n_full),
        "eps_c_hf": np.zeros((n_full, N_TREATMENTS)),
        "eps_c_pulse": np.zeros((n_full, N_TREATMENTS)),
    }
    params: dict[str, object] = {
        "l_max": L_MAX,
        "carryover_family": np.array(
            [CARRYOVER_FAMILY_KEYS.index(name) for name in ("geometric", "weibull", "geometric")]
        ),
        "carryover_alpha": np.array([0.42, 0.61, 0.73]),
        "weibull_lam": np.array([2.4, 3.1, 3.7]),
        "weibull_k": np.array([1.9, 2.3, 2.7]),
        "sat_family": np.array(
            [SATURATION_FAMILY_KEYS.index(name) for name in ("hill", "michaelis_menten", "root")]
        ),
        "hill_slope": np.array([1.8, 2.0, 2.2]),
        "hill_kappa_mult": np.array([0.85, 1.0, 1.15]),
        "logistic_lam": np.ones(N_TREATMENTS),
        "mm_kappa_mult": np.array([0.9, 1.25, 1.0]),
        "tanh_c": np.ones(N_TREATMENTS),
        "root_alpha": np.array([0.65, 0.72, 0.83]),
        "rw_d": _rw([0.0], [0.0], positive_only=False),
        "rw_z": _rw([0.25, -0.45], [0.55, 0.45], positive_only=False),
        "rw_c": _rw([0.45, 0.95, 1.35], [0.55, 0.65, 0.5], positive_only=True),
        "rw_b": _rw([5.0], [0.0], positive_only=False),
        "rw_y": _rw([0.0], [0.0], positive_only=False),
        "u_dz": np.zeros((N_LATENT, N_COVARIATES)),
        "gamma_zz": np.zeros((N_COVARIATES, N_COVARIATES)),
        "w_dc": np.zeros((N_LATENT, N_TREATMENTS)),
        "v_zc": np.zeros((N_COVARIATES, N_TREATMENTS)),
        "alpha_cc": np.zeros((N_TREATMENTS, N_TREATMENTS)),
        "delta_dy": np.zeros(N_LATENT),
        "rho_zy": np.array([0.35, -0.28]),
        "beta": np.array([1.15, 0.82, 1.08]),
        "hf_sigma": np.zeros(N_TREATMENTS),
        "pulse_amp": np.zeros(N_TREATMENTS),
        "pulse_prob": np.zeros(N_TREATMENTS),
        "covariate_hf_sigma": np.zeros(N_COVARIATES),
        "covariate_pulse_amp": np.zeros(N_COVARIATES),
        "covariate_pulse_prob": np.zeros(N_COVARIATES),
        "baseline_floor": None,
    }
    graph = {
        "g_cy": np.ones(N_TREATMENTS, dtype=int),
        "g_dc": np.zeros((N_LATENT, N_TREATMENTS), dtype=int),
        "g_dz": np.zeros((N_LATENT, N_COVARIATES), dtype=int),
        "g_dy": np.zeros(N_LATENT, dtype=int),
        "g_zy": np.ones(N_COVARIATES, dtype=int),
        "g_zc": np.zeros((N_COVARIATES, N_TREATMENTS), dtype=int),
        "g_cc": np.zeros((N_TREATMENTS, N_TREATMENTS), dtype=int),
        "g_zz": np.zeros((N_COVARIATES, N_COVARIATES), dtype=int),
    }
    symbolic = build_symbolic_graph(
        graph,
        params,
        n_full,
        N_TREATMENTS,
        N_COVARIATES,
        N_LATENT,
        burn_in=0,
        eps=eps,
    )["outputs"]
    evaluate = pytensor.function([], list(symbolic.values()))
    outputs = {name: np.asarray(value) for name, value in zip(symbolic, evaluate(), strict=True)}
    return ControlledWorld(outputs=outputs, params=params, graph=graph)


def _evaluate(expression: pt.TensorVariable) -> np.ndarray:
    return np.asarray(pytensor.function([], expression)())


def _mechanism_features(
    treatments: np.ndarray, scales: np.ndarray, params: dict[str, object]
) -> np.ndarray:
    """Apply each channel's true carryover, then its true saturation."""
    features = []
    carryover_family = np.asarray(params["carryover_family"])
    saturation_family = np.asarray(params["sat_family"])
    for k in range(N_TREATMENTS):
        column = pt.as_tensor_variable(treatments[:, k, None])
        carry_id = int(carryover_family[k])
        if carry_id == CARRYOVER_FAMILY_KEYS.index("none"):
            carried = column[:, 0]
        elif carry_id == CARRYOVER_FAMILY_KEYS.index("geometric"):
            carried = mechanisms.apply_geometric_carryover(
                column, np.asarray(params["carryover_alpha"])[k], L_MAX
            )[:, 0]
        else:
            carried = mechanisms.apply_weibull_pdf_carryover(
                column,
                np.asarray(params["weibull_lam"])[k],
                np.asarray(params["weibull_k"])[k],
                L_MAX,
            )[:, 0]
        carried_values = _evaluate(carried)
        family = SATURATION_FAMILY_KEYS[int(saturation_family[k])]
        if family == "linear":
            saturated = pt.as_tensor_variable(carried_values / scales[k])
        elif family == "hill":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                slope=np.asarray(params["hill_slope"])[k],
                kappa_mult=np.asarray(params["hill_kappa_mult"])[k],
            )
        elif family == "logistic":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                lam=np.asarray(params["logistic_lam"])[k],
            )
        elif family == "michaelis_menten":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                kappa_mult=np.asarray(params["mm_kappa_mult"])[k],
            )
        elif family == "tanh":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                c=np.asarray(params["tanh_c"])[k],
            )
        else:
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                alpha=np.asarray(params["root_alpha"])[k],
            )
        features.append(_evaluate(saturated))
    return np.column_stack(features)


def _fit(world: ControlledWorld) -> tuple[np.ndarray, np.ndarray, int, float]:
    outputs = world.outputs
    params = world.params
    features_full = _mechanism_features(outputs["treatments"], outputs["saturation_scale"], params)
    window = slice(WARMUP, None)
    design = np.column_stack(
        [features_full[window], outputs["covariates"][window], np.ones(N_TIME_STEPS)]
    )
    coefficient, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, outputs["outcome"][window], rcond=None
    )
    truth = np.concatenate([np.asarray(params["beta"]), np.asarray(params["rho_zy"]), [5.0]])
    return coefficient, truth, int(rank), float(np.linalg.cond(design))


def test_checkpoint_a_noiseless_ols_recovers_the_generated_coefficients(capsys):
    world = _controlled_world()
    coefficient, truth, rank, condition = _fit(world)
    error = coefficient - truth
    print(
        "seed=20261005 warmup=4 l_max=4 rank="
        f"{rank}/6 raw_condition={condition:.12g} max_abs_error={np.max(np.abs(error)):.12g}"
    )
    print("coefficient truth estimate error")
    for target, estimate, delta in zip(truth, coefficient, error, strict=True):
        print(f"{target:.16g} {estimate:.16g} {delta:.3g}")
    captured = capsys.readouterr().out
    assert "rank=6/6" in captured
    assert rank == 6
    assert condition < 100.0
    np.testing.assert_allclose(coefficient, truth, rtol=2e-12, atol=2e-12)


def _noisy_recovery(world: ControlledWorld) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit independent Gaussian-noise replicates on one fixed generated design.

    The design and noiseless mean stay fixed, so this is a conditional-on-X
    OLS sampling experiment.  Each returned level has shape
    ``(N_REPLICATES, n_coefficients)``.
    """
    outputs = world.outputs
    features_full = _mechanism_features(
        outputs["treatments"], outputs["saturation_scale"], world.params
    )
    window = slice(WARMUP, None)
    design = np.column_stack(
        [features_full[window], outputs["covariates"][window], np.ones(N_TIME_STEPS)]
    )
    truth = np.concatenate(
        [np.asarray(world.params["beta"]), np.asarray(world.params["rho_zy"]), [5.0]]
    )
    mean = outputs["outcome"][window]
    standard_error = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
    estimates = []
    for level_index, sigma in enumerate(NOISE_LEVELS):
        level_estimates = []
        for replicate in range(N_REPLICATES):
            seed = REPLICATE_SEED + level_index * N_REPLICATES + replicate
            noise = np.random.default_rng(seed).normal(0.0, sigma, size=N_TIME_STEPS)
            level_estimates.append(np.linalg.lstsq(design, mean + noise, rcond=None)[0])
        estimates.append(np.asarray(level_estimates))
    return np.asarray(estimates), truth, standard_error


def test_checkpoint_b_noisy_recovery_matches_conditional_ols_standard_errors(capsys):
    world = _controlled_world()
    estimates, truth, standard_error = _noisy_recovery(world)
    print(
        f"replicates={N_REPLICATES} seeds={REPLICATE_SEED}-"
        f"{REPLICATE_SEED + len(NOISE_LEVELS) * N_REPLICATES - 1} "
        f"noise_sigmas={','.join(f'{sigma:g}' for sigma in NOISE_LEVELS)}"
    )
    for sigma, level_estimates in zip(NOISE_LEVELS, estimates, strict=True):
        z_scores = (level_estimates - truth) / (sigma * standard_error)
        spread_ratio = np.std(level_estimates, axis=0, ddof=1) / (sigma * standard_error)
        coverage = np.mean(np.abs(z_scores) <= 1.96)
        median_abs_z = float(np.median(np.abs(z_scores)))
        median_spread_ratio = float(np.median(spread_ratio))
        print(
            f"sigma={sigma:g} median_abs_z={median_abs_z:.4f} median_spread_over_se={median_spread_ratio:.4f} coverage95={coverage:.4f}"
        )
        # These are aggregate distributional checks, not per-realization
        # monotonicity requirements.  The nominal interval is conditional on X.
        assert 0.55 <= median_abs_z <= 0.85
        assert 0.75 <= median_spread_ratio <= 1.25
        assert 0.90 <= coverage <= 0.99
    captured = capsys.readouterr().out
    assert "replicates=96" in captured
    assert "noise_sigmas=1.2,0.6,0.3" in captured


def test_checkpoint_a_features_match_generation_and_use_history():
    world = _controlled_world()
    outputs = world.outputs
    features_full = _mechanism_features(
        outputs["treatments"], outputs["saturation_scale"], world.params
    )
    window = slice(WARMUP, None)
    generated_contributions = outputs["contributions_observed"][window]
    np.testing.assert_allclose(
        generated_contributions,
        features_full[window] * np.asarray(world.params["beta"]),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_array_equal(outputs["outcome_noise"], 0.0)
    np.testing.assert_array_equal(outputs["baseline_intrinsic"], 5.0)
    np.testing.assert_allclose(
        outputs["baseline"],
        outputs["baseline_intrinsic"] + outputs["covariate_contribution"].sum(axis=1),
        rtol=0.0,
        atol=2e-15,
    )
    assert np.all(outputs["saturation_scale"] > 0.0)

    # Re-starting the carryover at the fit window is a silent feature error.
    reset_features = _mechanism_features(
        outputs["treatments"][window], outputs["saturation_scale"], world.params
    )
    assert np.max(np.abs(reset_features[0] - features_full[WARMUP])) > 1e-4


def test_checkpoint_a_graph_has_no_confounding_or_floor_paths():
    world = _controlled_world()
    graph = world.graph
    assert not graph["g_dc"].any()
    assert not graph["g_dz"].any()
    assert not graph["g_dy"].any()
    assert not graph["g_zc"].any()
    assert not graph["g_cc"].any()
    assert not graph["g_zz"].any()
    assert graph["g_cy"].all()
    assert graph["g_zy"].all()
    assert world.params["baseline_floor"] is None


def test_c1_config_roundtrip_hash_and_two_valid_configurations():
    from scripts.linear_recovery_prevalence import (
        DEFAULT_STUDY,
        config_hash,
        resolve_config,
    )

    first = {
        "generator": {
            "n_treatments": 2,
            "n_covariates": 2,
            "n_latent": 1,
            "n_treatments_active_range": [1, 2],
            "n_covariates_active_range": [1, 2],
            "n_time_steps": 16,
            "trajectories": "texture",
            "nonlinearity": "diverse",
        },
        "study": {
            **DEFAULT_STUDY,
            "n_cells": 2,
            "draws_per_cell": 1,
            "treatment_count_bands": [[1, 2]],
            "control_count_bands": [[1, 2]],
        },
    }
    second = {
        **first,
        "generator": {**first["generator"], "n_time_steps": 17},
    }
    resolved_first = resolve_config(first)
    resolved_again = resolve_config(first)
    resolved_second = resolve_config(second)
    assert resolved_first["config_hash"] == resolved_again["config_hash"]
    assert resolved_first["config_hash"] == config_hash(
        {key: value for key, value in resolved_first.items() if key != "config_hash"}
    )
    assert resolved_first["config_hash"] != resolved_second["config_hash"]
    assert resolved_first["generator_resolved"]["n_treatments"] == 2


def test_c1_invalid_and_unsupported_configuration_is_explicit():
    import pytest

    from scripts.linear_recovery_prevalence import (
        resolve_config,
        unsupported_status,
    )

    with pytest.raises(ValueError, match="unsupported generator"):
        resolve_config(
            {"generator": {"n_treatments": 2, "n_covariates": 2, "n_latent": 1, "made_up": 4}}
        )
    with pytest.raises(ValueError, match="study-only"):
        resolve_config(
            {"generator": {"n_treatments": 2, "n_covariates": 2, "n_latent": 1, "seed": 9}}
        )
    status = unsupported_status("true_feature_recovery", "truth fields are absent")
    assert status == {
        "analysis": "true_feature_recovery",
        "status": "unavailable",
        "reason": "truth fields are absent",
    }


def test_c1_cell_accounting_wilson_and_explicit_estimand():
    from scripts.linear_recovery_prevalence import cell_binary_summary, wilson_interval

    first = cell_binary_summary([0, 0, 1, 1], [False, True, True, False])
    assert first["unit"] == "cell"
    assert first["estimand"] == "first_world"
    assert first["n_cells"] == 2
    assert first["n_successes"] == 1
    assert first["interval"]["method"] == "wilson"
    any_sibling = cell_binary_summary(
        [0, 0, 1, 1], [False, True, False, False], estimand="any_sibling"
    )
    assert any_sibling["n_successes"] == 1
    assert wilson_interval(0, 4)[0] == 0.0


def test_c1_design_semantics_keep_constant_active_inputs_and_separate_causal_paths():
    from scripts.linear_recovery_prevalence import (
        classify_design,
        classify_graph_paths,
        design_rank_condition,
    )

    measured = design_rank_condition(
        np.column_stack([np.ones(8), np.arange(8), np.ones(8)]),
        active_mask=[True, True, True],
    )
    assert measured["exact_rank"] == 2
    assert measured["constant_active_columns"] == 2
    result = classify_design(
        2,
        3,
        float("inf"),
        rank_tolerance=1e-10,
        constant_active_columns=1,
        causal_path=True,
    )
    assert result["rank_deficient"]
    assert result["constant_active_inputs_retained"]
    assert not result["padded_columns_excluded_by_mask"]
    assert result["causal_path_present"]
    assert "not a multicollinearity" in result["causal_path_interpretation"]
    paths = classify_graph_paths(
        {"g_cy": [1], "g_dc": [[1]], "g_dy": [1], "g_zc": [[1]], "g_zy": [1]}
    )
    assert paths["direct_treatment_outcome_reachability"]
    assert paths["shared_latent_confounding"]
    assert paths["mediated_or_observed_path"]

    disjoint = classify_graph_paths(
        {
            "g_cy": [1],
            "g_dc": [[1], [0]],
            "g_dy": [0, 1],
            "g_dz": [[0], [0]],
            "g_zc": [[1]],
            "g_zy": [1],
        },
        focal_treatment=0,
    )
    assert not disjoint["shared_latent_confounding"]
    mediated = classify_graph_paths(
        {"g_cy": [1], "g_dc": [[0]], "g_dy": [0], "g_dz": [[1]], "g_zc": [[1]], "g_zy": [1]},
        focal_treatment=0,
    )
    assert mediated["potential_unobserved_confounding"]
    treatment_only = classify_graph_paths(
        {"g_cy": [1], "g_dc": [[1]], "g_dy": [0]}, focal_treatment=0
    )
    assert not treatment_only["potential_unobserved_confounding"]
    disconnected_mediated = classify_graph_paths(
        {
            "g_dz": [[1, 0]],  # D0 -> Z0
            "g_zc": [[0], [1]],  # Z1 -> C0; Z0 has no treatment edge
            "g_cy": [1],
            "g_dy": [0],
            "g_zy": [0, 0],
        },
        focal_treatment=0,
    )
    assert not disconnected_mediated["raw_reachability"]["mediated_latent_to_treatment"]


def test_c1_active_variants_only_override_resolved_ranges():
    from scripts.linear_recovery_prevalence import (
        _active_dimension_generator,
        resolve_config,
    )

    resolved = resolve_config(
        {
            "generator": {
                "n_treatments": 4,
                "n_covariates": 5,
                "n_latent": 2,
                "n_treatments_active_range": [1, 4],
                "n_covariates_active_range": [2, 5],
                "n_latent_active_range": [1, 2],
                "n_time_steps": 104,
                "trajectories": "composable",
                "nonlinearity": "diverse",
            },
            "study": {
                "n_cells": 2,
                "draws_per_cell": 1,
                "treatment_count_bands": [[1, 4]],
                "control_count_bands": [[2, 5]],
            },
        }
    )
    base = {
        **resolved["generator_factory"],
        **{
            name: resolved["generator_resolved"][name]
            for name in (
                "n_treatments_active_range",
                "n_covariates_active_range",
                "n_latent_active_range",
            )
        },
    }
    for level, expected in (("low", (1, 2, 1)), ("high", (4, 5, 2))):
        variant, metadata = _active_dimension_generator(base, level)
        assert metadata["active_dimensions"] == {
            "treatments": expected[0],
            "covariates": expected[1],
            "latent": expected[2],
        }
        assert metadata["overrides"] == {
            key: variant[key] for key in variant if base.get(key) != variant[key]
        }
        assert all(variant[key] == base[key] for key in base if key not in metadata["overrides"])
        assert variant["n_time_steps"] == 104
        assert variant["trajectories"] == "composable"
        assert variant["nonlinearity"] == "diverse"


def test_c1_compact_artifact_size_arithmetic_is_explicit():
    from scripts.linear_recovery_prevalence import _artifact_size_estimate, canonical_json

    proposed = {"configs": 10, "cells_total": 320, "worlds_total": 640}
    estimate = _artifact_size_estimate({"schema_version": "test", "summary": {}}, proposed)
    fixed = estimate["fixed_schema_config_summary_bytes"]
    world = estimate["representative_world_record"]["bytes"]
    cell = estimate["representative_cell_record"]["bytes"]
    assert estimate["projected_bytes"] == fixed + 640 * world + 320 * cell
    assert estimate["representative_world_record"]["bytes"] == len(
        canonical_json(estimate["representative_world_record"]["schema"]).encode("utf-8")
    )
    assert estimate["assumptions"]["persisted_arrays"] is False
    assert "640" in estimate["arithmetic"] and "320" in estimate["arithmetic"]


def test_c1_rejections_exclude_failures_from_evaluated_candidates():
    from types import SimpleNamespace

    from scripts.linear_recovery_prevalence import count_accounting

    prior = SimpleNamespace(
        n_cells=3,
        draws_per_cell=1,
        n_treatments=1,
        n_covariates=1,
        n_treatments_active_range=(1, 1),
        n_covariates_active_range=(1, 1),
    )
    corpus = {
        "treatment_active_mask": np.ones((3, 1), dtype=bool),
        "covariate_active_mask": np.ones((3, 1), dtype=bool),
        "cell_id": np.arange(3),
        "diagnostics": {"n_draws_evaluated": 7, "n_draw_failures": 2},
    }
    study = {
        "treatment_count_bands": [[1, 1]],
        "control_count_bands": [[1, 1]],
    }
    accounting = count_accounting(corpus, prior, study)
    assert accounting["rejected_candidates"] == 4
    assert accounting["generation_failures"] == 2
    prior_with_two_counts = SimpleNamespace(
        n_cells=3,
        draws_per_cell=1,
        n_treatments=2,
        n_covariates=2,
        n_treatments_active_range=(1, 2),
        n_covariates_active_range=(1, 2),
    )
    zero_band = count_accounting(
        {
            **corpus,
            "treatment_active_mask": np.ones((3, 2), dtype=bool),
            "covariate_active_mask": np.ones((3, 2), dtype=bool),
        },
        prior_with_two_counts,
        {"treatment_count_bands": [[1, 1], [2, 2]], "control_count_bands": [[1, 1], [2, 2]]},
    )
    assert zero_band["count_band_accounting"]["n_empty_cell_band_strata"] == 3


def test_c2_pilot_schedule_has_primary_and_nine_labelled_stress_configs():
    from scripts.linear_recovery_prevalence import pilot_seed_schedule

    generator = {
        "n_treatments": 2,
        "n_covariates": 2,
        "n_latent": 1,
        "n_treatments_active_range": [1, 2],
        "n_covariates_active_range": [1, 2],
        "n_latent_active_range": [1, 1],
        "n_time_steps": 16,
        "trajectories": "composable",
        "nonlinearity": "diverse",
    }
    first = pilot_seed_schedule(generator, 20261005)
    second = pilot_seed_schedule(generator, 20261005)
    assert first == second
    assert [row["name"] for row in first] == [
        "primary_composable",
        "texture",
        "always_on_spikes",
        "periodic_on_off",
        "delayed_start",
        "ramp_up",
        "decay_to_zero",
        "level_doubling",
        "seasonal",
        "trend",
    ]
    assert len({row["seed"] for row in first}) == 10


def test_c2_any_sibling_summary_is_distinct_from_first_world_estimand():
    from scripts.linear_recovery_prevalence import _pilot_cell_records, _pilot_summaries

    world = {
        "config_index": 0,
        "cell_id": 0,
        "sibling_index": 0,
        "raw_input_diagnostics": {
            "views": {"levels": {"rank": {"rank_deficient": False, "condition": 1.0}}},
        },
    }
    sibling = {
        **world,
        "sibling_index": 1,
        "raw_input_diagnostics": {
            "views": {"levels": {"rank": {"rank_deficient": True, "condition": 1000.0}}}
        },
    }
    summary = _pilot_summaries(_pilot_cell_records([world, sibling]), 0.95)
    assert summary["estimands"]["first_world"]["levels"]["rank_deficient"]["n_successes"] == 0
    assert summary["estimands"]["any_sibling"]["levels"]["rank_deficient"]["n_successes"] == 1
    assert (
        summary["estimands"]["any_sibling"]["levels"]["rank_deficient"]["estimand"] == "any_sibling"
    )
    assert "not pooled" in summary["sensitivity_note"]


def test_c2_pilot_wall_limit_rejects_cap_above_approved_ninety_minutes():
    import pytest

    from scripts.linear_recovery_prevalence import run_pilot

    with pytest.raises(ValueError, match="5400"):
        run_pilot(
            {
                "generator": {
                    "n_treatments": 10,
                    "n_covariates": 10,
                    "n_latent": 1,
                    "n_treatments_active_range": [1, 10],
                    "n_covariates_active_range": [1, 10],
                    "n_latent_active_range": [1, 1],
                }
            },
            output_dir="/tmp/linear-recovery-c2-test",
            wall_time_seconds=5401,
        )


def test_c1_band_validation_rejects_gaps_and_incomplete_ranges():
    import pytest

    from scripts.linear_recovery_prevalence import resolve_config

    config = {
        "generator": {"n_treatments": 3, "n_covariates": 3, "n_latent": 1},
        "study": {
            "treatment_count_bands": [[1, 1], [3, 3]],
            "control_count_bands": [[1, 3]],
        },
    }
    with pytest.raises(ValueError, match="non-overlapping|gap|cover"):
        resolve_config(config)
    overlap = {
        **config,
        "study": {
            "treatment_count_bands": [[1, 2], [2, 3]],
            "control_count_bands": [[1, 3]],
        },
    }
    with pytest.raises(ValueError, match="non-overlapping"):
        resolve_config(overlap)
    eleven = {
        "generator": {
            "n_treatments": 11,
            "n_covariates": 11,
            "n_latent": 1,
            "n_treatments_active_range": [1, 11],
            "n_covariates_active_range": [1, 11],
        },
        "study": {
            "treatment_count_bands": [[1, 10], [11, 11]],
            "control_count_bands": [[1, 10], [11, 11]],
        },
    }
    assert resolve_config(eleven)["generator_factory"]["n_treatments"] == 11


def _c2_mock_config():
    return {
        "generator": {
            "n_treatments": 2,
            "n_covariates": 2,
            "n_latent": 1,
            "n_treatments_active_range": [1, 2],
            "n_covariates_active_range": [1, 2],
            "n_latent_active_range": [1, 1],
            "n_time_steps": 16,
            "trajectories": "composable",
            "nonlinearity": "diverse",
        },
        "study": {
            "n_cells": 2,
            "draws_per_cell": 2,
            "pilot_cells_per_config": 2,
            "pilot_siblings_per_cell": 2,
            "pilot_checkpoint_every_cells": 1,
            "pilot_wall_time_seconds": 60,
            "treatment_count_bands": [[1, 2]],
            "control_count_bands": [[1, 2]],
        },
    }


def _c2_mock_record(config_index, cell_id, sibling_index):
    return {
        "config_index": config_index,
        "cell_id": cell_id,
        "sibling_index": sibling_index,
        "world_index": sibling_index,
        "active_counts": {"treatments": 1, "covariates": 1},
        "raw_input_diagnostics": {
            "views": {
                "levels": {
                    "rank": {"rank_deficient": sibling_index == 1, "condition": 40.0},
                    "max_abs_pearson_correlation": 0.92,
                    "max_vif": 6.0,
                    "vif_infinite": False,
                },
                "differences": {
                    "rank": {"rank_deficient": sibling_index == 1, "condition": 40.0},
                    "max_abs_pearson_correlation": 0.92,
                    "max_vif": 6.0,
                    "vif_infinite": False,
                },
            },
        },
    }


def test_c2_diagnostic_views_keep_view_specific_rank_condition_and_subset():
    from scripts.linear_recovery_prevalence import _world_indicators

    record = {
        "raw_input_diagnostics": {
            "rank": {"rank_deficient": False, "condition": 2.0},
            "views": {
                "levels": {
                    "rank": {"rank_deficient": False, "condition": 2.0},
                    "max_abs_pearson_correlation": None,
                    "max_vif": None,
                    "vif_infinite": False,
                },
                "differences": {
                    "rank": {"rank_deficient": True, "condition": 200.0},
                    "max_abs_pearson_correlation": None,
                    "max_vif": None,
                    "vif_infinite": False,
                },
            },
        }
    }
    thresholds = {
        "absolute_correlation": [0.8, 0.9],
        "vif": [5.0, 10.0],
        "condition": [30.0, 100.0],
    }
    indicators = _world_indicators(record, thresholds)
    assert not indicators["levels"]["rank_deficient"]
    assert not indicators["levels"]["condition_ge_30p0"]
    assert indicators["differences"]["rank_deficient"]
    assert indicators["differences"]["condition_ge_100p0"]

    subset = {
        "raw_input_diagnostics": {
            "views": {"levels": record["raw_input_diagnostics"]["views"]["levels"]}
        }
    }
    assert list(_world_indicators(subset, thresholds)) == ["levels"]


def test_c2_compaction_uses_actual_predictors_and_only_configured_views():
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    levels = np.array([[0.0, 1.0, 3.0, 6.0, 10.0], [0.0, 2.0, 5.0, 9.0, 14.0]])
    descriptors = [SimpleNamespace(key="C1"), SimpleNamespace(key="C2")]

    def diagnostic_view():
        dependence = SimpleNamespace(
            descriptors=descriptors,
            matrices={"pearson": np.zeros((1, 2, 2))},
            valid={"pearson": np.ones((1, 2, 2), dtype=bool)},
        )
        vif = SimpleNamespace(
            vif=np.ones((1, 2)),
            valid=np.ones((1, 2), dtype=bool),
        )
        return SimpleNamespace(dependence=dependence, vif={"observed": vif})

    graph = {
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 0), dtype=int),
        "g_dy": np.zeros(1, dtype=int),
        "g_zc": np.zeros((0, 2), dtype=int),
        "g_zy": np.zeros(0, dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((0, 0), dtype=int),
        "g_cy": np.zeros(2, dtype=int),
    }
    prior = SimpleNamespace(layout=SimpleNamespace(unpack=lambda _value: graph))
    corpus = {
        "treatment_active_mask": np.array([[True, True]]),
        "covariate_active_mask": np.zeros((1, 0), dtype=bool),
        "latent_active_mask": np.array([[True]]),
        "treatment_raw": levels.T[None, :, :],
        "covariates": np.zeros((1, levels.shape[1], 0)),
        "cell_id": np.array([3]),
        "g": np.zeros((1, 1)),
    }
    both = SimpleNamespace(views={"levels": diagnostic_view(), "differences": diagnostic_view()})
    record = runner._compact_world_metric(
        corpus, prior, both, 0, config_index=0, cell_index=3, sibling_index=0, rank_tolerance=1e-10
    )
    views = record["raw_input_diagnostics"]["views"]
    assert set(views) == {"levels", "differences"}
    assert views["levels"]["rank"]["exact_rank"] == 3
    assert views["differences"]["rank"]["exact_rank"] == 2
    assert "rank" not in record["raw_input_diagnostics"]
    assert "standardized" not in record["raw_input_diagnostics"]

    differences_only = SimpleNamespace(views={"differences": diagnostic_view()})
    subset = runner._compact_world_metric(
        corpus,
        prior,
        differences_only,
        0,
        config_index=0,
        cell_index=3,
        sibling_index=0,
        rank_tolerance=1e-10,
    )
    assert set(subset["raw_input_diagnostics"]["views"]) == {"differences"}
    assert "rank" not in subset["raw_input_diagnostics"]
    assert list(runner._world_indicators(subset, runner.DEFAULT_STUDY["thresholds"])) == [
        "differences"
    ]


def test_c2_configured_indicators_and_realized_band_wilson_tables():
    from scripts.linear_recovery_prevalence import _pilot_cell_records, _pilot_summaries

    records = [
        _c2_mock_record(0, 0, 0),
        _c2_mock_record(0, 0, 1),
        _c2_mock_record(0, 1, 0),
        _c2_mock_record(0, 1, 1),
    ]
    thresholds = {
        "absolute_correlation": [0.8, 0.99],
        "vif": [5.0, 10.0],
        "condition": [30.0, 100.0],
    }
    cells = _pilot_cell_records(
        records,
        thresholds=thresholds,
        treatment_bands=[[1, 1], [2, 2]],
        control_bands=[[1, 1], [2, 2]],
    )
    summary = _pilot_summaries(
        cells,
        0.95,
        thresholds=thresholds,
        treatment_bands=[[1, 1], [2, 2]],
        control_bands=[[1, 1], [2, 2]],
    )
    assert summary["thresholds"] == thresholds
    indicator = summary["estimands"]["any_sibling"]["differences"]
    assert indicator["absolute_correlation_ge_0p8"]["n_successes"] == 2
    assert indicator["vif_ge_5p0"]["n_successes"] == 2
    assert summary["count_band_wilson"]["treatments_1_1__controls_1_1"]["status"] == "available"
    assert summary["count_band_wilson"]["treatments_2_2__controls_2_2"]["status"] == "unavailable"


def test_c2_phase_kills_term_ignoring_child_and_reaps_it(tmp_path):
    import os
    import signal
    import time

    from scripts import linear_recovery_prevalence as runner

    child_pid_path = tmp_path / "child.pid"

    def ignores_term():
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child_pid_path.write_text(str(os.getpid()))
        time.sleep(10)

    with pytest.raises(runner._PilotDeadlineExceededError):
        runner._bounded_phase(ignores_term, time.monotonic() + 0.1, time.monotonic)
    child_pid = int(child_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_c2_runner_executes_real_generation_and_diagnostic_phases_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    generation_marker = tmp_path / "generation.calls"
    diagnostic_marker = tmp_path / "diagnostics.calls"

    def failing_generation(_prior):
        generation_marker.write_text(
            generation_marker.read_text() + "x" if generation_marker.exists() else "x"
        )
        raise ValueError("phase value error")

    monkeypatch.setattr(runner, "sample_prior_predictive", failing_generation)
    report = runner.run_pilot(config, output_dir=tmp_path / "generation", wall_time_seconds=60)
    assert generation_marker.read_text() == "x"
    assert report["status"] == "partial"
    assert report["accounting"]["failures"] == 1

    monkeypatch.setattr(
        runner, "sample_prior_predictive", lambda _prior: {"cell_id": np.array([0, 0, 1, 1])}
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _c, _p, _d, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )

    def failing_diagnostics(*args, **kwargs):
        diagnostic_marker.write_text(
            diagnostic_marker.read_text() + "x" if diagnostic_marker.exists() else "x"
        )
        raise ValueError("phase value error")

    monkeypatch.setattr(runner, "data_diagnostics", failing_diagnostics)
    with pytest.raises(ValueError, match="phase value error"):
        runner.run_pilot(config, output_dir=tmp_path / "diagnostics", wall_time_seconds=60)
    assert diagnostic_marker.read_text() == "x"


def test_c2_notebook_renders_runner_complete_partial_and_rejects_wrong_manifest(
    tmp_path, monkeypatch
):
    import copy
    import json
    import os
    from types import SimpleNamespace

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    from scripts import linear_recovery_prevalence as runner

    notebook = nbformat.read("docs/examples/linear-recovery.ipynb", as_version=4)
    render_source = notebook.cells[11].source
    monkeypatch.setattr(
        runner,
        "_pilot_configurations",
        lambda generator, seed: [
            {"name": "primary_composable", "label": "primary", "seed": seed, "generator": generator}
        ],
    )
    monkeypatch.setattr(
        runner,
        "pilot_seed_schedule",
        lambda generator, seed: [
            {"index": 0, "name": "primary_composable", "seed": seed, "generator": generator}
        ],
    )
    n_rows = 64
    monkeypatch.setattr(
        runner,
        "sample_prior_predictive",
        lambda _prior: {"cell_id": np.repeat(np.arange(32), 2)},
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": n_rows,
            "accepted_worlds": n_rows,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _c, _p, _d, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    protocol = runner.load_json("docs/examples/data/linear-recovery-prevalence-config.json")
    complete_dir = tmp_path / "complete"
    complete = runner.run_pilot(protocol, output_dir=complete_dir, wall_time_seconds=60)
    partial = runner.run_pilot(protocol, output_dir=tmp_path / "partial", wall_time_seconds=0.001)
    assert complete["status"] == "complete"
    assert partial["status"] == "partial"

    def execute(report_path):
        os.environ["LINEAR_RECOVERY_C2_REPORT"] = str(report_path)
        source = f"from pathlib import Path\nrepo_root = Path({str(Path.cwd())!r})\n"
        source += render_source
        test_notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(source)], metadata=notebook.metadata
        )
        NotebookClient(test_notebook, timeout=120, kernel_name="python").execute(
            cwd=str(Path.cwd())
        )

    execute(Path(complete["report_path"]))
    execute(Path(partial["report_path"]))
    manifest_path = Path(complete["manifest_path"])
    original = runner.load_json(manifest_path)
    broken = copy.deepcopy(original)
    broken["schedule"] = []
    runner._write_json(manifest_path, broken)
    with pytest.raises(CellExecutionError):
        execute(Path(complete["report_path"]))
    runner._write_json(manifest_path, original)
    assert "_threshold_key" in render_source
    assert json.loads(Path(complete["report_path"]).read_text())["source_provenance"]


def test_c2_runner_success_accounting_resume_collision_and_schema_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    monkeypatch.setattr(
        runner, "sample_prior_predictive", lambda _prior: {"cell_id": np.array([0, 0, 1, 1])}
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _corpus, _prior, _diagnostic, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    report = runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=60, clock=lambda: 0.0)
    assert report["status"] == "complete"
    assert report["accounting"]["complete_cells"] == 20
    assert report["accounting"]["accepted_world_metrics"] == 40
    assert report["configs"][0]["summary"]["count_band_wilson"]
    with pytest.raises(FileExistsError, match="no-resume"):
        runner.run_pilot(
            config, output_dir=tmp_path, wall_time_seconds=60, resume=False, clock=lambda: 0.0
        )
    import copy

    resolved = runner.resolve_config(config)
    broken = {**report, "config_hash": "wrong"}
    with pytest.raises(ValueError, match="config hash"):
        runner.validate_pilot_report(broken, resolved)
    for field, value in (
        ("schema_version", "wrong-schema"),
        ("resolved_config", {**report["resolved_config"], "config_hash": "wrong"}),
        ("manifest_hash", "wrong-manifest"),
        ("schedule", []),
        ("status", "invalid"),
    ):
        mutated = copy.deepcopy(report)
        mutated[field] = value
        with pytest.raises(ValueError):
            runner.validate_pilot_report(mutated, resolved)
    for provenance_field in ("package", "package_version", "source_commit", "source_dirty"):
        mutated = copy.deepcopy(report)
        mutated["source_provenance"][provenance_field] = "mutated"
        with pytest.raises(ValueError):
            runner.validate_pilot_report(mutated, resolved)

    manifest_path = Path(report["manifest_path"])
    original_manifest = runner.load_json(manifest_path)
    for field, value in (
        ("schema_version", "wrong-schema"),
        ("config_hash", "wrong-config"),
        ("resolved_config", {**original_manifest["resolved_config"], "config_hash": "wrong"}),
        ("source_provenance", {**original_manifest["source_provenance"], "source_commit": "wrong"}),
        ("schedule", []),
    ):
        mutated_manifest = copy.deepcopy(original_manifest)
        mutated_manifest[field] = value
        runner._write_json(manifest_path, mutated_manifest)
        with pytest.raises(ValueError):
            runner.validate_pilot_report(report, resolved)
    runner._write_json(manifest_path, original_manifest)


def test_c2_bounded_phase_isolates_sleep_and_transfers_value_error_once(tmp_path):
    import time

    from scripts import linear_recovery_prevalence as runner

    marker = tmp_path / "phase-calls"

    def failing_phase():
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        raise ValueError("phase value error")

    for phase in ("generation", "diagnostics"):
        with pytest.raises(runner._PilotDeadlineExceededError):
            runner._bounded_phase(lambda: time.sleep(2), time.monotonic() + 0.1, time.monotonic)
    with pytest.raises(ValueError, match="phase value error"):
        runner._bounded_phase(failing_phase, time.monotonic() + 2, time.monotonic)
    assert marker.read_text() == "x"


def test_c2_runner_deduplicates_failure_then_success_on_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    failure_marker = tmp_path / "planned-failure.once"

    def sample(_prior):
        try:
            failure_marker.touch(exist_ok=False)
        except FileExistsError:
            return {"cell_id": np.array([0, 0, 1, 1])}
        raise RuntimeError("planned failure")

    monkeypatch.setattr(runner, "sample_prior_predictive", sample)
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _corpus, _prior, _diagnostic, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    first = runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=60, clock=lambda: 0.0)
    assert first["status"] == "partial"
    checkpoint_path = (
        tmp_path
        / f"linear-recovery-c2-{runner.resolve_config(config)['config_hash'][:16]}.checkpoint.json"
    )
    report_path = (
        tmp_path / f"linear-recovery-c2-{runner.resolve_config(config)['config_hash'][:16]}.json"
    )
    checkpoint = runner.load_json(checkpoint_path)
    partial_report = runner.load_json(report_path)
    assert len(checkpoint["config_failures"]) == 1
    assert partial_report["status"] == "partial"
    second = runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=60, clock=lambda: 0.0)
    assert second["status"] == "complete"
    assert second["accounting"]["failures"] == 0


def test_c2_resume_accumulates_budget_and_replaces_stale_failure_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    monkeypatch.setattr(
        runner, "sample_prior_predictive", lambda _prior: {"cell_id": np.array([0, 0, 1, 1])}
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _c, _p, _d, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    report = runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=60, clock=lambda: 0.0)
    checkpoint = runner.load_json(report["checkpoint_path"])
    checkpoint["wall_elapsed_seconds"] = 59.5
    runner._write_json(Path(report["checkpoint_path"]), checkpoint)
    resumed = runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=60, clock=lambda: 0.0)
    assert resumed["status"] == "partial"
    assert resumed["accounting"]["wall_elapsed_seconds"] >= 59.5
    assert resumed["accounting"]["wall_budget_seconds"] == 60.0


def test_c2_runner_handles_term_during_finalization_after_completed_checkpoint(
    tmp_path, monkeypatch
):
    import json
    import os
    import signal
    import time
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    monkeypatch.setattr(
        runner,
        "_pilot_configurations",
        lambda generator, seed: [
            {"name": "primary_composable", "label": "primary", "seed": seed, "generator": generator}
        ],
    )
    monkeypatch.setattr(
        runner,
        "pilot_seed_schedule",
        lambda generator, seed: [
            {"index": 0, "name": "primary_composable", "seed": seed, "generator": generator}
        ],
    )
    monkeypatch.setattr(
        runner, "sample_prior_predictive", lambda _prior: {"cell_id": np.array([0, 0, 1, 1])}
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _c, _p, _d, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    original_cells = runner._pilot_cell_records

    def slow_finalization(*args, **kwargs):
        time.sleep(2)
        return original_cells(*args, **kwargs)

    monkeypatch.setattr(runner, "_pilot_cell_records", slow_finalization)
    resolved = runner.resolve_config(config)
    prefix = f"linear-recovery-c2-{resolved['config_hash'][:16]}"
    output = tmp_path / "finalization"
    pid = os.fork()
    if pid == 0:
        try:
            runner.run_pilot(config, output_dir=output, wall_time_seconds=60)
        finally:
            os._exit(0)
    checkpoint_path = output / f"{prefix}.checkpoint.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
            except json.JSONDecodeError:
                checkpoint = {}
            if any(row.get("status") == "complete" for row in checkpoint.get("cells", [])):
                break
        time.sleep(0.02)
    os.kill(pid, signal.SIGTERM)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    report = runner.load_json(output / f"{prefix}.json")
    assert report["status"] == "partial"
    assert report["interrupted"]


def test_c2_runner_handles_term_mid_phase_with_honest_partial_artifacts(tmp_path, monkeypatch):
    """Exercise TERM in a single-threaded runner process, not a forked pytest parent."""
    import os
    import signal
    import time
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    monkeypatch.setattr(
        runner,
        "sample_prior_predictive",
        lambda _prior: (time.sleep(2), {"cell_id": np.array([0, 0, 1, 1])})[1],
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    resolved = runner.resolve_config(config)
    prefix = f"linear-recovery-c2-{resolved['config_hash'][:16]}"
    pid = os.fork()
    if pid == 0:
        try:
            runner.run_pilot(config, output_dir=tmp_path, wall_time_seconds=10)
        finally:
            os._exit(0)
    time.sleep(0.2)
    os.kill(pid, signal.SIGTERM)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    report = runner.load_json(tmp_path / f"{prefix}.json")
    assert report["status"] == "partial"
    assert report["interrupted"]
    checkpoint = runner.load_json(report["checkpoint_path"])
    persisted = runner.load_json(report["report_path"])
    assert checkpoint["status"] == "partial"
    assert persisted["status"] == "partial"
    assert persisted["accounting"]["wall_elapsed_seconds"] >= 0.0


def test_c2_runner_handles_deadline_during_generation_and_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import linear_recovery_prevalence as runner

    config = _c2_mock_config()
    monkeypatch.setattr(
        runner, "sample_prior_predictive", lambda _prior: {"cell_id": np.array([0, 0, 1, 1])}
    )
    monkeypatch.setattr(runner, "data_diagnostics", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "count_accounting",
        lambda *args, **kwargs: {
            "evaluated_candidates": 4,
            "accepted_worlds": 4,
            "rejected_candidates": 0,
            "generation_failures": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_compact_world_metric",
        lambda _c, _p, _d, row, **kwargs: _c2_mock_record(
            kwargs["config_index"], kwargs["cell_index"], kwargs["sibling_index"]
        ),
    )
    monkeypatch.setattr(
        runner,
        "_bounded_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            runner._PilotDeadlineExceededError("deadline")
        ),
    )
    generation = runner.run_pilot(
        config, output_dir=tmp_path / "generation", wall_time_seconds=60, clock=lambda: 0.0
    )
    assert generation["status"] == "partial" and generation["accounting"]["complete_cells"] == 0

    calls = {"phase": 0}

    def phase(function, deadline, clock):
        calls["phase"] += 1
        if calls["phase"] == 2:
            raise runner._PilotDeadlineExceededError("deadline")
        return function()

    monkeypatch.setattr(runner, "_bounded_phase", phase)
    diagnostic = runner.run_pilot(
        config, output_dir=tmp_path / "diagnostic", wall_time_seconds=60, clock=lambda: 0.0
    )
    assert diagnostic["status"] == "partial"
    assert diagnostic["accounting"]["accepted_candidates"] == 4


__all__ = [
    "N_TIME_STEPS",
    "SEED",
    "WARMUP",
    "_controlled_world",
    "_fit",
    "_mechanism_features",
]
