"""Validation-only true-feature linear recovery tools (no science runs by default)."""

from __future__ import annotations

import time

# The inclusive CLI clock must precede setup/imports (PM-approved E402 exception).
# Importing this module neither starts a clock nor executes science.
_ENTRY_T0 = time.monotonic() if __name__ == "__main__" else None

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402


def generate_world(attempt: dict[str, int], *, n_time_steps: int = 104, carryover_burn_in: int = 0):
    """Generate one deterministic scalar-baseline world using public APIs only."""
    from pymc_generator import make_scm_prior, sample_scm

    if not {"n_treatments", "n_covariates", "n_latent", "world_seed"} <= set(attempt):
        raise ValueError("attempt must include dimensions and world_seed")
    if attempt["n_latent"] != 1:
        raise ValueError("the frozen population requires exactly one latent node")
    for name in ("n_treatments", "n_covariates"):
        if (
            isinstance(attempt[name], bool)
            or not isinstance(attempt[name], (int, np.integer))
            or not 1 <= attempt[name] <= 10
        ):
            raise ValueError(f"{name} must be an integer in [1,10]")
    prior = make_scm_prior(
        n_treatments=attempt["n_treatments"],
        n_covariates=attempt["n_covariates"],
        n_latent=1,
        nonlinearity="diverse",
        trajectories="composable",
        n_time_steps=n_time_steps,
        carryover_burn_in=carryover_burn_in,
        outcome_std_mode="relative",
        rw_baseline_std_range=(0.0, 0.0),
        baseline_floor=None,
    )
    return sample_scm(prior, seed=attempt["world_seed"])


SCHEMA_VERSION = "linear-recovery-true-feature-sweep/v1"
IDENTITY_TOLERANCE = 2e-12
NOISEFREE_COEFFICIENT_TOLERANCE = 2e-10


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


# Exact JSON literals are the immutable authority; each consumer gets a fresh view.
_FROZEN_CONFIG_BYTES = b'{"accounting_correction":{"authority":"2026-10-07 coordinator truthful missing-evidence correction; PM revisions c6bed304","previous_config_digest":"e0e13db341c6e3b692e7b57cb07cd0effcfff3655a308bf342ef1a92f2ea05e0","previous_input_version":"ols-scientific-input/v2","reason":"Missing verified generation evidence is not a confirmed generation failure. Compact integrity is not execution provenance."},"input_contract_version":"ols-scientific-input/v3","population":{"baseline_floor":null,"carryover_burn_in":0,"count_ranges":{"n_covariates":[1,10],"n_latent":[1,1],"n_treatments":[1,10]},"count_rule":"independent inclusive uniform integers; fixed-size public prior; J=1","excluded_claims":["time-varying-baseline worlds","causal identification"],"label":"scalar-baseline restricted population","n_time_steps":104,"outcome_std_mode":"relative","retained_generator_axes":{"factory":{"nonlinearity":"diverse","trajectories":"composable"},"origin":"C2-derived composable/diverse defaults; scalar-baseline restriction; independently drawn fixed dimensions","resolved_prior":{"active_count_allocation":"independent","active_count_weights":null,"baseline_floor":null,"baseline_floor_scope":"intercept","beta_additive_range":[0.5,2.0],"carryover_alpha_range":[0.2,0.8],"carryover_burn_in":0,"carryover_family_probs":{"geometric":0.425,"none":0.15,"weibull":0.425},"cc_base_rate":0.15,"cc_coeff_range":[0.05,0.3],"confounding_strength_range":null,"covariate_flighting_duty_range":[0.3,0.8],"covariate_flighting_inclusion_prob":0.15,"covariate_flighting_period_weeks_range":[4,13],"covariate_hf_inclusion_prob":0.9,"covariate_hf_sigma_range":[0.1,0.8],"covariate_level_jump_count":1,"covariate_level_jump_inclusion_prob":0.3,"covariate_level_jump_size_range":[-1.0,1.0],"covariate_offset_frac_range":[0.6009615384615384,0.9086538461538461],"covariate_offset_inclusion_prob":0.1,"covariate_onset_frac_range":[0.052884615384615384,0.12980769230769232],"covariate_onset_inclusion_prob":0.1,"covariate_pulse_amp_range":[0.5,3.0],"covariate_pulse_inclusion_prob":0.6,"covariate_pulse_prob_range":[0.0,0.25],"covariate_reference_contribution_range":null,"covariate_reference_scale":1.0,"covariate_seasonal_amplitude_range":[0.2,1.0],"covariate_seasonal_inclusion_prob":0.35,"covariate_seasonal_period_weeks_range":[52.0,52.0],"covariate_trend_change_range":[-1.0,1.0],"covariate_trend_inclusion_prob":0.3,"dc_coeff_range":[0.1,0.5],"draws_per_cell":20,"dy_coeff_range":[0.15,0.45],"dz_base_rate":0.3,"dz_coeff_range":[0.1,0.5],"edge_budget":null,"edge_rate_overrides":null,"include_identifiability_labels":true,"l_max":8,"min_no_direct_effect_treatments":0,"mm_scale_prior":"uniform","n_cells":50,"n_time_steps":104,"n_treatment_shocks":0,"outcome_std_mode":"relative","p_long_horizon":0.5,"prior_cond_width_ranges":null,"prior_conditioning":false,"query_frac":0.25,"rw_baseline_mean_range":[3.0,8.0],"rw_baseline_std_range":[0.0,0.0],"rw_baseline_std_sigma":1.0,"rw_covariate_mean_range":[-3.0,3.0],"rw_outcome_std_range":[0.01,0.028],"rw_outcome_std_sigma":0.25,"rw_positive_mean_range":[0.1,8.0],"rw_smoothness_alpha":2.0,"rw_smoothness_beta":2.0,"rw_smoothness_max_weeks":26,"rw_std_sigma":1.5,"rw_treatment_std_range":[0.1,1.2],"saturation_family_probs":{"hill":0.17,"linear":0.15,"logistic":0.17,"michaelis_menten":0.17,"root":0.17,"tanh":0.17},"saturation_prior_ranges":{"hill":{"kappa_mult":[0.7,1.5],"slope":[1.0,3.0]},"logistic":{"lam":[0.5,3.0]},"michaelis_menten":{"kappa_mult":[0.7,1.5]},"root":{"alpha":[0.3,0.9]},"tanh":{"c":[0.3,1.5]}},"seed":0,"treatment_cv_floor":0.08,"treatment_flighting_duty_range":[0.3,0.8],"treatment_flighting_inclusion_prob":0.2,"treatment_flighting_period_weeks_range":[4,13],"treatment_hf_inclusion_prob":0.9,"treatment_hf_sigma_range":[0.08,0.6],"treatment_level_jump_count":1,"treatment_level_jump_factor_range":[0.5,2.0],"treatment_level_jump_inclusion_prob":0.2,"treatment_offset_frac_range":[0.6009615384615384,0.9086538461538461],"treatment_offset_inclusion_prob":0.1,"treatment_onset_frac_range":[0.052884615384615384,0.12980769230769232],"treatment_onset_inclusion_prob":0.15,"treatment_pulse_amp_range":[0.4,2.5],"treatment_pulse_inclusion_prob":0.7,"treatment_pulse_prob_range":[0.0,0.25],"treatment_reference_contribution_range":null,"treatment_reference_multiplier":1.0,"treatment_seasonal_amplitude_range":[0.1,0.5],"treatment_seasonal_inclusion_prob":0.3,"treatment_seasonal_period_weeks_range":[52.0,52.0],"treatment_shock_length_range":[2,2],"treatment_shock_level_range":[0.0,0.0],"treatment_trend_inclusion_prob":0.3,"treatment_trend_log_change_range":[-1.0,1.0],"val_cell_frac":0.2,"weibull_k_range":[1.5,4.0],"weibull_lam_range":[2.0,8.0],"zc_base_rate":0.3,"zc_coeff_range":[0.05,0.3],"zy_coeff_range":[0.1,0.4],"zz_base_rate":0.1,"zz_coeff_range":[-0.2,0.2]}},"rw_baseline_std_range":[0.0,0.0]},"schema_version":"linear-recovery-true-feature-sweep/v1","study":{"calibration_count":3,"calibration_dimensions":{"n_covariates":10,"n_latent":1,"n_treatments":10},"calibration_rule":"three fixed-dimension timing-only worlds, IDs 1000..1002; excluded from every science denominator; all three must complete with finite positive inclusive costs","categories":["generation_failed","unsupported_truth","identity_or_truth_ineligible","rank_deficient","easy","moderate","hard","noisefree_recovered","paired_noisy_available","unattempted_budget"],"clock_rule":"one monotonic inclusive clock starts before setup/calibration; elapsed E includes setup, cold compilation, generation, analysis, persistence and verification; actual deadline enforcement is independent of cost heuristic","conditioning_rule":"raw condition <100 easy; >=10000 hard; otherwise moderate; null unavailable","conservative_cost_rule":"2*max(all three completed calibration inclusive durations: cold compilation+generation+analysis+persistence+verification)","easy_condition_threshold":100.0,"example_selection":{"absent_rule":"null attempt with band_absent_from_manifest reason","rationale":"configuration-only natural count bands; no outcome screening or replacement of failed, ineligible or unattempted selections","rule":"first scheduled attempt in each of nine configured-count band pairs in manifest order","selected_before_outcomes":true},"finalization_reserve_seconds":300,"fit_rows":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103],"frozen_count_formula":"min(1000,max(0,floor((5400-E-300)/(2*max(C0,C1,C2)))))","hard_condition_threshold":10000.0,"maximum_schedule":1000,"maximum_schedule_rationale":"pre-outcome finite maximum; pooled worst-case 95% Wilson half-width about 3.09 percentage points at 1000; natural exact-cell expected occupancy 10; not a MAP allocation","minimum_subgroup_size":10,"noisefree_recovery_rule":"full_rank and max_abs_coefficient_error <= noisefree_coefficient","pooled_precision_adequacy_count":400,"pooled_precision_rule":"Ncap >= 400 is an allocation adequacy flag only, never an execution or release gate; actual eligible denominator governs precision; no marginal or subgroup precision guarantee","rank_rule":"singular_value > rank_tolerance * largest_singular_value","rank_tolerance":1e-10,"reporting":{"allocation_excluded":"1000-Ncap","configured_count_bands":[[1,3],[4,7],[8,10]],"configured_views":["T_marginal","M_marginal","exact_T_M_cells","nine_T_M_band_pairs"],"conservation":["0 <= F <= E <= G <= A <= Ncap <= 1000","E = F + eligible_rank_deficient","A = G + confirmed_generation_failed + evidence_unavailable","G = E + generated_ineligible","1000 = (1000-Ncap) + (Ncap-A) + A"],"denominator_predicates":{"A":"started science attempt within the frozen prefix","E":"G and analysis.status.truth_eligibility == eligible and no analysis exception (explicit failure stage or unsupported_truth:/identity_or_truth_ineligible: reason prefix)","F":"E and analysis.status.rank == full_rank","G":"A and retained verified generated graph/world, including analysis/persistence failures","confirmed_generation_failed":"A without generated graph and with retained generation-stage error evidence","evidence_unavailable":"A without retained verified graph and without a confirmed generation-stage error"},"direct_dimension_bands":[[0,0],[1,3],[4,7],[8,10]],"frequency_denominators":{"attempted_configured":"A","direct_dimensions":"G","graph_patterns":"G","planned_configured":"Ncap"},"indicators":{"evidence_unavailable":{"denominator":"A","numerator":"evidence_unavailable"},"failure_stage":{"denominator":"A","numerator":"one of generation_failed, unsupported_truth, identity_or_truth_ineligible, analysis_failed"},"full_rank":{"denominator":"E","numerator":"F"},"graph_pattern":{"denominator":"G","numerator":"G and graph matches pattern"},"noisefree_recovered":{"denominator":"F","numerator":"F and analysis.status.noisefree_recovery is true"},"paired_noisy_available":{"denominator":"E","numerator":"E and analysis.status.paired_noisy.available is true"},"persistence_failed":{"denominator":"A","numerator":"retained persistence-stage error; orthogonal to generation evidence partition"},"rank_deficient":{"denominator":"E","numerator":"E and analysis.status.rank == rank_deficient"}},"interval_rule":"95% Wilson, one predeclared binary indicator per independently seeded world per view; never rows, coefficients or paired targets; descriptive coefficient errors have no binomial interval","missing_evidence_substatuses":["known_generation_completed_missing_evidence","generation_outcome_unknown"],"precision_rule":"pooled, marginal, exact-cell and band precision are separate; no forced-balanced science","record_states":["unattempted_budget","generation_failed","unsupported_truth","identity_or_truth_ineligible","analysis_failed","eligible_full_rank","eligible_rank_deficient","evidence_unavailable","generated_analysis_unavailable"],"runtime_unattempted":"Ncap-A","sparse_frequency_rule":"retain all cells including zero counts; descriptive one-vs-rest Wilson frequencies are non-simultaneous; minimum 10 does not suppress frequencies","zero_denominator_rule":"unavailable rate and interval, never zero"},"schedule_rule":"PCG64; independent science count pairs uniform inclusive 1..10 and uint32 world seeds; fixed calibration dimensions and separate uint32 world seeds; reject collisions globally without redraw/reorder/replacement","seed_streams":{"calibration":{"world_seed":20261011},"science":{"count_seed":20261008,"world_seed":20261010}},"subgroup_rule":"minimum_subgroup_size=10 applies only to conditional outcome-rate availability; retain all sparse frequency cells and numerator/denominator; zero denominator unavailable, never zero","tolerances":{"identity":2e-12,"noisefree_coefficient":2e-10,"scalar_identity":2e-12},"wall_time_seconds":5400}}'


def _strict_frozen(value, expected, name="config"):
    if type(value) is not type(expected):
        raise ValueError(f"{name}: wrong frozen type")
    if isinstance(expected, dict):
        if value.keys() != expected.keys():
            raise ValueError(f"{name}: wrong frozen keys")
        for key in expected:
            _strict_frozen(value[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{name}: wrong frozen shape")
        for i, (actual, frozen) in enumerate(zip(value, expected, strict=True)):
            _strict_frozen(actual, frozen, f"{name}[{i}]")
    elif value != expected or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"{name}: frozen value mismatch")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """No post-outcome tuning: exact keys, JSON types, shapes, enums and values."""
    _strict_frozen(config, json.loads(_FROZEN_CONFIG_BYTES))
    return json.loads(canonical_json(config))


def study_schedules(config):
    study = validate_config(config)["study"]
    schedules = {}
    for phase, count in (
        ("calibration", study["calibration_count"]),
        ("science", study["maximum_schedule"]),
    ):
        seeds = study["seed_streams"][phase]
        pairs = (
            np.random.Generator(np.random.PCG64(seeds["count_seed"])).integers(
                1, 11, size=(count, 2)
            )
            if phase == "science"
            else np.full((count, 2), 10, dtype=int)
        )
        worlds = np.random.Generator(np.random.PCG64(seeds["world_seed"])).integers(
            0, 2**32, size=count, dtype=np.uint32
        )
        schedules[phase] = [
            {
                "attempt": i + (study["maximum_schedule"] if phase == "calibration" else 0),
                "n_treatments": int(pair[0]),
                "n_covariates": int(pair[1]),
                "n_latent": 1,
                "world_seed": int(seed),
            }
            for i, (pair, seed) in enumerate(zip(pairs, worlds, strict=True))
        ]
    seeds = [row["world_seed"] for rows in schedules.values() for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("world seed collision: never redraw")
    return schedules


def _scheduled_attempt(config, attempt, phase):
    if (
        phase not in ("science", "calibration")
        or not isinstance(attempt, dict)
        or type(attempt.get("attempt")) is not int
    ):
        raise ValueError("invalid scheduled attempt")
    rows = study_schedules(config)[phase]
    offset = config["study"]["maximum_schedule"] if phase == "calibration" else 0
    i = attempt["attempt"] - offset
    if not 0 <= i < len(rows):
        raise ValueError("attempt outside frozen schedule")
    _strict_frozen(attempt, rows[i], "attempt")


def resolved_generation_config(config, attempt, *, phase="science"):
    """Bind every retained generator axis before generation, detecting default drift."""
    from dataclasses import asdict

    from pymc_generator import make_scm_prior

    cfg = validate_config(config)
    _scheduled_attempt(cfg, attempt, phase)
    pop = cfg["population"]
    dimensions = {key: attempt[key] for key in ("n_treatments", "n_covariates", "n_latent")}
    expected = dict(pop["retained_generator_axes"]["resolved_prior"], **dimensions)
    expected.update({key + "_active_range": [value, value] for key, value in dimensions.items()})
    prior = make_scm_prior(
        **dimensions,
        **pop["retained_generator_axes"]["factory"],
        n_time_steps=pop["n_time_steps"],
        carryover_burn_in=pop["carryover_burn_in"],
        outcome_std_mode=pop["outcome_std_mode"],
        rw_baseline_std_range=tuple(pop["rw_baseline_std_range"]),
        baseline_floor=pop["baseline_floor"],
    )
    actual = json.loads(canonical_json(asdict(prior)))
    _strict_frozen(actual, expected, "resolved generation config")
    return expected


def generate_configured_world(config, attempt, *, phase="science"):
    from dataclasses import asdict

    resolved = resolved_generation_config(config, attempt, phase=phase)
    world = generate_world(
        attempt,
        n_time_steps=resolved["n_time_steps"],
        carryover_burn_in=resolved["carryover_burn_in"],
    )
    _strict_frozen(json.loads(canonical_json(asdict(world.cfg))), resolved, "sampled config")
    if world.seed != attempt["world_seed"]:
        raise ValueError("sampled seed mismatch")
    return world


def analyze_configured_world(world, config, attempt, *, phase="science"):
    from dataclasses import asdict

    resolved = resolved_generation_config(config, attempt, phase=phase)
    _strict_frozen(json.loads(canonical_json(asdict(world.cfg))), resolved, "sampled config")
    if world.seed != attempt["world_seed"]:
        raise ValueError("sampled seed mismatch")
    study = config["study"]
    result = analyze_world(
        world, fit_indices=np.array(study["fit_rows"]), rank_tolerance=study["rank_tolerance"]
    )
    result["generation"] = {
        "phase": phase,
        "resolved_config": resolved,
        "resolved_config_digest": digest(resolved),
        "config_digest": digest(config),
        "generation_seed": attempt["world_seed"],
    }
    return result


def conservative_world_cost(config, durations):
    study = validate_config(config)["study"]
    if not isinstance(durations, list) or len(durations) != study["calibration_count"]:
        raise ValueError("every calibration duration is required")
    for duration in durations:
        _positive_time(duration)
    cost = 2 * max(durations)
    _positive_time(cost)
    return cost


def freeze_calibrated_schedule(config, elapsed_after_calibration, durations):
    """Pure budget calculation; never executes calibration or extends a schedule."""
    cost = conservative_world_cost(config, durations)
    _positive_time(elapsed_after_calibration, zero=True)
    if elapsed_after_calibration < sum(durations):
        raise ValueError("inclusive elapsed time cannot exclude calibration durations")
    result = frozen_schedule(config, elapsed_after_calibration, cost)
    result["calibration_durations"] = list(durations)
    result["cost_rule"] = config["study"]["conservative_cost_rule"]
    return result


def allocate_from_calibration(config, elapsed, calibration):
    """Pure allocation from exactly three completed inclusive timing records.

    The caller measures elapsed from before setup; no clock/deadline orchestration
    occurs here. Invalid calibration retains its reason and allocates no attempts.
    """
    cfg = validate_config(config)
    _positive_time(elapsed, zero=True)
    expected = study_schedules(cfg)["calibration"]
    reason = None
    durations = []
    try:
        if type(calibration) is not list or len(calibration) != len(expected):
            raise ValueError("all three calibration records required")
        for record, attempt in zip(calibration, expected, strict=True):
            if type(record) is not dict or set(record) != {
                "attempt",
                "phase",
                "status",
                "duration",
            }:
                raise ValueError("invalid calibration record fields")
            _strict_frozen(record["attempt"], attempt, "calibration attempt")
            _strict_frozen(record["phase"], "calibration", "calibration phase")
            _strict_frozen(record["status"], "complete", "calibration completion")
            _positive_time(record["duration"])
            durations.append(record["duration"])
        result = freeze_calibrated_schedule(cfg, elapsed, durations)
    except ValueError as exc:
        reason = str(exc)
        result = {
            "frozen_attempt_count": 0,
            "manifest": [],
            "formula": cfg["study"]["frozen_count_formula"],
            "inputs": {
                "maximum_schedule": cfg["study"]["maximum_schedule"],
                "wall_time_seconds": cfg["study"]["wall_time_seconds"],
                "elapsed_after_calibration": elapsed,
                "reserve": cfg["study"]["finalization_reserve_seconds"],
                "conservative_per_world_cost": None,
            },
            "calibration_durations": None,
            "cost_rule": cfg["study"]["conservative_cost_rule"],
        }
    count = result["frozen_attempt_count"]
    result.update(
        allocation_reason=(
            "calibration_invalid" if reason else ("budget_exhausted" if count == 0 else "allocated")
        ),
        calibration_invalid_reason=reason,
        pooled_precision_adequate=count >= cfg["study"]["pooled_precision_adequacy_count"],
        allocation_excluded=cfg["study"]["maximum_schedule"] - count,
        selection=prefit_selection(cfg, result["manifest"]),
    )
    return result


def validate_accounting(
    config,
    *,
    ncap,
    attempted,
    generated,
    eligible,
    full_rank,
    rank_deficient,
    evidence_unavailable=0,
):
    """Conservation only, not a report/aggregation implementation."""
    maximum = validate_config(config)["study"]["maximum_schedule"]
    counts = (ncap, attempted, generated, eligible, full_rank, rank_deficient, evidence_unavailable)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("accounting counts must be nonnegative integers")
    if not 0 <= full_rank <= eligible <= generated <= attempted <= ncap <= maximum:
        raise ValueError("accounting subset conservation violated")
    if eligible != full_rank + rank_deficient:
        raise ValueError("eligible rank partition violated")
    if evidence_unavailable > attempted - generated:
        raise ValueError("generation evidence partition violated")
    return {
        "allocation_excluded": maximum - ncap,
        "runtime_unattempted": ncap - attempted,
        "generation_failed": attempted - generated - evidence_unavailable,
        "evidence_unavailable": evidence_unavailable,
        "generated_ineligible": generated - eligible,
    }


def project_attempt_state(
    config, manifest, attempt, *, started, graph, analysis, failure_stage=None
):
    """Project accepted analyze_world fields, retaining generated graphs on failure.

    The wrapper must retain the graph independently: early analysis failures may
    have no ``source`` at all. Caught post-eligibility exceptions may leave the
    truth flag true, so exception reasons must also be checked for E membership.
    """
    prefix = validate_science_prefix(config, manifest)
    _scheduled_attempt(config, attempt, "science")
    if attempt["attempt"] >= len(prefix) or type(started) is not bool:
        raise ValueError("attempt must belong to prefix and started must be boolean")
    if failure_stage not in (None, "generation_failed", "analysis_failed", "evidence_unavailable"):
        raise ValueError("invalid explicit failure stage")
    generated = graph is not None
    if generated and (not isinstance(graph, dict) or not graph):
        raise ValueError("retained graph must be a nonempty mapping")
    if not started and (generated or analysis is not None or failure_stage is not None):
        raise ValueError("unattempted world cannot have generated or analyzed evidence")
    if not generated and analysis is not None:
        raise ValueError("analysis requires a retained generated graph")
    if failure_stage == "generation_failed" and (generated or not started):
        raise ValueError("generation failure contradicts state")
    if failure_stage == "analysis_failed" and not generated:
        raise ValueError("analysis failure requires generated graph")
    eligible = full_rank = rank_deficient = recovered = noisy = False
    stage = failure_stage
    state = "unattempted_budget"
    if started and not generated:
        state = stage = (
            "generation_failed" if stage == "generation_failed" else "evidence_unavailable"
        )
    elif generated:
        if analysis is None:
            if stage not in ("analysis_failed", "evidence_unavailable"):
                raise ValueError("generated world requires analysis or explicit evidence failure")
            state = "generated_analysis_unavailable" if stage == "evidence_unavailable" else stage
        else:
            if not isinstance(analysis, dict):
                raise ValueError("analysis must be a mapping")
            status, reasons = analysis.get("status"), analysis.get("reasons")
            if (
                not isinstance(status, dict)
                or type(reasons) is not list
                or any(type(reason) is not str for reason in reasons)
            ):
                raise ValueError("analysis requires accepted status and reasons")
            if status.get("scheduling") != "attempted" or status.get("generation") != "generated":
                raise ValueError("analysis scheduling/generation mismatch")
            if status.get("truth_eligibility") not in ("eligible", "ineligible"):
                raise ValueError("invalid analysis truth eligibility")
            if stage is None:
                for prefix_reason in ("unsupported_truth", "identity_or_truth_ineligible"):
                    if any(reason.startswith(prefix_reason + ":") for reason in reasons):
                        stage = prefix_reason
                        break
            eligible = status["truth_eligibility"] == "eligible" and stage is None
            if eligible:
                rank = status.get("rank")
                if rank not in ("full_rank", "rank_deficient"):
                    raise ValueError("nonexception eligible world requires classified rank")
                full_rank, rank_deficient = rank == "full_rank", rank == "rank_deficient"
                recovery = status.get("noisefree_recovery")
                paired = status.get("paired_noisy")
                if (
                    type(recovery) is not bool
                    or not isinstance(paired, dict)
                    or type(paired.get("available")) is not bool
                ):
                    raise ValueError("eligible analysis requires recovery/noisy indicators")
                if rank_deficient and recovery:
                    raise ValueError("rank deficient world cannot be recovered")
                recovered, noisy = full_rank and recovery, paired["available"]
                state = "eligible_full_rank" if full_rank else "eligible_rank_deficient"
            else:
                stage = stage or "identity_or_truth_ineligible"
                state = stage
    return {
        "attempt": attempt["attempt"],
        "state": state,
        "failure_stage": stage,
        "A": started,
        "G": generated,
        "E": eligible,
        "F": full_rank,
        "eligible_rank_deficient": rank_deficient,
        "noisefree_recovered_over_F": recovered if full_rank else None,
        "paired_noisy_available_over_E": noisy if eligible else None,
    }


def runtime_input_environment():
    """Snapshot runtime inputs only; no generation, outcomes, clock or archive I/O."""
    import platform
    import sys

    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONPATH",
            )
        },
    }


def freeze_input_contract(config, *, environment, source_hashes, seed_manifest):
    """Freeze before calibration/outcomes; future orchestration must persist this.

    This input-only helper neither starts a run nor creates a world catalog.
    Actual deadline enforcement and proof of ordering belong to orchestration.
    """
    cfg = validate_config(config)
    _strict_frozen(environment, runtime_input_environment(), "runtime inputs")
    _strict_frozen(source_hashes, analysis_source_hashes(), "analysis source hashes")
    _strict_frozen(seed_manifest, study_schedules(cfg), "seed manifest")
    contract = {
        "version": cfg["input_contract_version"],
        "config": cfg,
        "config_digest": digest(cfg),
        "source_commit": _source_commit(),
        "source_hashes": source_hashes,
        "catalog_schema_path": catalog_schema()["$id"],
        "catalog_schema_sha256": CATALOG_SCHEMA_SHA256,
        "environment": environment,
        "seed_manifest": seed_manifest,
        "seed_manifest_digest": digest(seed_manifest),
        "maximum_selection": selection_manifest(cfg),
        "required_order": "freeze source/schema/config/runtime/seeds before calibration or outcomes; then allocate and freeze prefix selection before science",
    }
    # Round-trip severs caller-owned mutable references.
    return json.loads(canonical_json({"contract": contract, "digest": digest(contract)}))


def subgroup_available(config, denominator):
    minimum = validate_config(config)["study"]["minimum_subgroup_size"]
    if type(denominator) is not int or denominator < 0:
        raise ValueError("subgroup denominator must be a nonnegative integer")
    return denominator >= minimum


def _positive_time(value, *, zero=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError("timing must be finite numeric and in range")


def selection_manifest(config):
    """Maximum-schedule pre-fit selection; catalog row schema remains v1."""
    return prefit_selection(config, study_schedules(config)["science"])


def validate_science_prefix(config, manifest):
    """Reject phase mixing, arbitrary IDs/seeds, duplicates and reordered prefixes."""
    schedules = study_schedules(config)
    if type(manifest) is not list or len(manifest) > len(schedules["science"]):
        raise ValueError("invalid science prefix")
    _strict_frozen(manifest, schedules["science"][: len(manifest)], "science prefix")
    return json.loads(canonical_json(manifest))


def configured_band_pair(config, attempt):
    cfg = validate_config(config)
    _scheduled_attempt(cfg, attempt, "science")
    bands = cfg["study"]["reporting"]["configured_count_bands"]
    return tuple(
        next(i for i, (low, high) in enumerate(bands) if low <= attempt[key] <= high)
        for key in ("n_treatments", "n_covariates")
    )


def prefit_selection(config, manifest):
    """Select from a canonical frozen prefix, without seeing any outcome."""
    cfg = validate_config(config)
    manifest = validate_science_prefix(cfg, manifest)
    chosen = {(t, m): None for t in range(3) for m in range(3)}
    rows = []
    bands = cfg["study"]["reporting"]["configured_count_bands"]
    for attempt in manifest:
        pair = tuple(
            next(i for i, (low, high) in enumerate(bands) if low <= attempt[key] <= high)
            for key in ("n_treatments", "n_covariates")
        )
        selected = chosen[pair] is None
        if selected:
            chosen[pair] = attempt["attempt"]
        rows.append(
            {
                "attempt": attempt["attempt"],
                "group": [attempt[key] for key in ("n_treatments", "n_covariates", "n_latent")],
                "example_selected": selected,
                "fit_rows": cfg["study"]["fit_rows"],
                "rationale": cfg["study"]["example_selection"]["rationale"],
            }
        )
    return {
        "version": "ols-prefit-selection/v2",
        "config_digest": digest(cfg),
        "rule": cfg["study"]["example_selection"]["rule"],
        "rows": rows,
        "band_examples": [
            {
                "bands": [bands[t], bands[m]],
                "attempt": attempt,
                "availability_reason": "band_absent_from_manifest" if attempt is None else None,
            }
            for (t, m), attempt in chosen.items()
        ],
    }


def count_stream(config: dict[str, Any], *, phase: str = "science") -> list[dict[str, int]]:
    """Return the canonical phase schedule; no independent seed/count authority."""
    if phase not in ("science", "calibration"):
        raise ValueError("invalid schedule phase")
    return study_schedules(config)[phase]


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> dict[str, float | int | None]:
    if total <= 0 or not 0 <= successes <= total:
        return {
            "numerator": successes,
            "denominator": total,
            "rate": None,
            "lower": None,
            "upper": None,
        }
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return {
        "numerator": successes,
        "denominator": total,
        "rate": p,
        "lower": center - half,
        "upper": center + half,
    }


def fit_design(X: np.ndarray, y: np.ndarray, rank_tolerance: float) -> dict[str, Any]:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (
        X.ndim != 2
        or y.shape != (X.shape[0],)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("design and target must be finite and shape-compatible")
    coef, _, rank, singular = np.linalg.lstsq(X, y, rcond=rank_tolerance)
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else None
    return {
        "coefficients": coef,
        "rank": int(rank),
        "singular_values": singular,
        "condition": condition,
        "p": int(X.shape[1]),
        "n_fit": int(X.shape[0]),
    }


def denominator_summary(
    records: list[dict[str, Any]], indicators: dict[str, str]
) -> dict[str, Any]:
    """Summarize one predeclared binary per world with explicit denominators."""
    result: dict[str, Any] = {}
    for label, field in indicators.items():
        eligible = [r for r in records if isinstance(r.get(field), bool)]
        result[label] = wilson(sum(r[field] for r in eligible), len(eligible))
    return result


def orthogonal_status(
    *,
    scheduled: bool,
    attempted: bool,
    generated: bool,
    truth_eligible: bool,
    rank: int | None = None,
    p: int | None = None,
    condition: float | None = None,
    noisefree_recovered: bool | None = None,
    paired_noisy_available: bool = False,
    raw_observed_misspecified: bool | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return orthogonal state dimensions; none masks another dimension."""
    return {
        "scheduling": "unattempted_budget"
        if scheduled and not attempted
        else ("attempted" if attempted else "not_scheduled"),
        "generation": "generated" if generated else ("failed" if attempted else "not_attempted"),
        "truth_eligibility": "eligible"
        if truth_eligible
        else ("ineligible" if generated else "unavailable"),
        "rank": None if rank is None else ("full_rank" if rank == p else "rank_deficient"),
        "conditioning": None
        if condition is None
        else ("easy" if condition < 100 else ("hard" if condition >= 10000 else "moderate")),
        "noisefree_recovery": noisefree_recovered,
        "paired_noisy": {"available": paired_noisy_available},
        "raw_observed_misspecified": raw_observed_misspecified,
        "reason": reason,
    }


def exclusive_output_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target


# Schema bytes, including their final newline, are the single external authority.
CATALOG_SCHEMA_BYTES = b'{"$id":"schemas/ols-world-catalog-v1.json","$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"properties":{"analysis_file_hashes":{"additionalProperties":false,"properties":{"docs/examples/data/linear-recovery-true-feature-sweep-config.json":{"pattern":"^[0-9a-f]{64}$","type":"string"},"scripts/linear_recovery_true_feature_sweep.py":{"pattern":"^[0-9a-f]{64}$","type":"string"},"tests/test_linear_recovery_true_feature_sweep.py":{"pattern":"^[0-9a-f]{64}$","type":"string"}},"required":["scripts/linear_recovery_true_feature_sweep.py","tests/test_linear_recovery_true_feature_sweep.py","docs/examples/data/linear-recovery-true-feature-sweep-config.json"],"type":"object"},"coefficient_order":{"items":{"minLength":1,"type":"string"},"type":"array"},"config_digest":{"pattern":"^[0-9a-f]{64}$","type":"string"},"data_draw_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"generation_seed":{"maximum":4294967295,"minimum":0,"type":"integer"},"inferred_diagnostics":{"additionalProperties":false,"properties":{"kind":{"enum":["estimates_not_sampled_truth"]},"labels":{"additionalProperties":false,"properties":{"conditioning":{"enum":["easy","moderate","hard",null]},"noisefree_recovery":{"type":"boolean"},"paired_noisy_available":{"enum":[true]},"rank":{"enum":["full_rank","rank_deficient"]}},"required":["rank","conditioning","noisefree_recovery","paired_noisy_available"],"type":"object"},"raw_design":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"standardized_design":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"status":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"y0_coefficients":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"y1_coefficients":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"}},"required":["kind","y0_coefficients","y1_coefficients","status","raw_design","standardized_design","labels"],"type":"object"},"known_sampled_truth":{"additionalProperties":false,"properties":{"coefficients":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"contributions_observed":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"design":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"direct_structure":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"kind":{"enum":["finite_sampled_truth_not_estimates"]},"noise":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"prior_cond":{"type":"null"},"prior_cond_status":{"enum":["unknown_not_retained"]},"public_history":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"row_mask":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"rows":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"scales":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"},"y1":{"additionalProperties":false,"properties":{"array":{"anyOf":[{"type":"null"},{"additionalProperties":false,"properties":{"data_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"dtype":{"minLength":1,"type":"string"},"locator":{"pattern":"^arrays/[0-9a-f]{64}\\\\.bin$","type":"string"},"nbytes":{"minimum":0,"type":"integer"},"shape":{"items":{"minimum":0,"type":"integer"},"type":"array"},"version":{"enum":["ols-world-array/v1"]}},"required":["version","locator","data_sha256","dtype","shape","nbytes"],"type":"object"}]},"path":{"items":{"minLength":1,"type":"string"},"type":"array"},"sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"}},"required":["world_locator","path","sha256","array"],"type":"object"}},"required":["kind","coefficients","direct_structure","design","contributions_observed","y1","noise","scales","public_history","rows","row_mask","prior_cond","prior_cond_status"],"type":"object"},"n_full":{"minimum":0,"type":"integer"},"noise_replicate_id":{"type":"null"},"pair_targets":{"enum":[["y0","y1"]]},"parameter_draw_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"parent_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"replicate_kind":{"enum":["fresh_world"]},"resolved_config_digest":{"pattern":"^[0-9a-f]{64}$","type":"string"},"schema_path":{"enum":["schemas/ols-world-catalog-v1.json"]},"schema_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"schema_version":{"enum":["ols-world-catalog/v1"]},"selection":{"additionalProperties":false,"properties":{"attempt":{"minimum":0,"type":"integer"},"example_selected":{"type":"boolean"},"fit_rows":{"items":{"minimum":0,"type":"integer"},"type":"array"},"group":{"items":{"minimum":0,"type":"integer"},"type":"array"},"rationale":{"minLength":1,"type":"string"}},"required":["attempt","group","example_selected","fit_rows","rationale"],"type":"object"},"selection_manifest_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"selection_manifest_locator":{"pattern":"^selection-[0-9a-f]{64}\\\\.json$","type":"string"},"source_commit":{"pattern":"^[0-9a-f]{40}$","type":"string"},"source_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"warmup":{"minimum":0,"type":"integer"},"world_content_digest":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_reference":{"additionalProperties":false,"properties":{"attempt_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"reservation_locator":{"pattern":"^attempt-[0-9a-f]{64}\\\\.json$","type":"string"},"reservation_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"},"version":{"enum":["ols-world-persistence/v1"]},"world_id":{"pattern":"^[0-9a-f]{64}$","type":"string"},"world_locator":{"pattern":"^world-[0-9a-f]{64}\\\\.json$","type":"string"},"world_sha256":{"pattern":"^[0-9a-f]{64}$","type":"string"}},"required":["version","attempt_id","reservation_locator","world_id","world_locator","world_sha256","reservation_sha256"],"type":"object"}},"required":["schema_version","schema_path","schema_sha256","parent_id","source_id","source_commit","analysis_file_hashes","config_digest","resolved_config_digest","generation_seed","world_id","world_content_digest","world_reference","selection_manifest_id","selection_manifest_locator","selection","parameter_draw_id","data_draw_id","noise_replicate_id","replicate_kind","pair_targets","coefficient_order","n_full","warmup","known_sampled_truth","inferred_diagnostics"],"title":"ols-world-catalog/v1","type":"object"}\n'
CATALOG_SCHEMA_SHA256 = hashlib.sha256(CATALOG_SCHEMA_BYTES).hexdigest()


def catalog_schema():
    """Return a fresh view; changing it cannot change authoritative bytes."""
    return json.loads(CATALOG_SCHEMA_BYTES)


def _readonly(value):
    from types import MappingProxyType

    if isinstance(value, dict):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    return value


CATALOG_SCHEMA = _readonly(catalog_schema())


def analysis_source_hashes():
    checkout = Path(__file__).resolve().parents[1]
    names = catalog_schema()["properties"]["analysis_file_hashes"]["required"]
    return {name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in names}


def _source_commit():
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def _catalog_file(archive, locator, payload=None):
    """Narrow descriptor-anchored, no-follow external catalog I/O."""
    archive = _world_archive(archive)
    if (
        not isinstance(locator, str)
        or re.fullmatch(
            r"(?:run\.json|schemas/ols-world-catalog-v1\.json|(?:catalog|selection)-[0-9a-f]{64}\.json)",
            locator,
        )
        is None
    ):
        raise ValueError("invalid catalog locator")
    path = archive / locator
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        flags = (
            os.O_RDONLY | os.O_NONBLOCK if payload is None else os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        target = os.open(path.name, flags | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(target, "rb" if payload is None else "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("catalog evidence must be a regular file")
            if payload is None:
                return stream.read()
            stream.write(payload)
    finally:
        os.close(fd)


def _json_safe(value, *, blob_dir: Path | None = None):
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("cannot persist nonfinite array")
        payload = np.ascontiguousarray(value).tobytes()
        sha = hashlib.sha256(payload).hexdigest()
        if blob_dir is None:
            return {
                "__ndarray__": True,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "data": base64.b64encode(payload).decode("ascii"),
            }
        blob = blob_dir / "arrays" / f"{sha}.bin"
        blob.parent.mkdir(exist_ok=True)
        try:
            with blob.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if blob.read_bytes() != payload:
                raise ValueError("array blob digest collision or tampering")
        return {
            "__ndarray_ref__": sha,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": len(payload),
        }
    if isinstance(value, np.generic):
        return {
            "__numpy_scalar__": value.dtype.str,
            "value": _json_safe(value.item(), blob_dir=blob_dir),
        }
    if isinstance(value, dict):
        return {str(k): _json_safe(v, blob_dir=blob_dir) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, blob_dir=blob_dir) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cannot persist nonfinite number")
    return value


def create_run_archive(
    root: str | Path,
    config: dict[str, Any],
    *,
    environment: dict[str, Any],
    source_hashes: dict[str, str],
    seed_manifest: Any,
) -> tuple[Path, dict[str, Any]]:
    """Create a config-digest-named external archive exclusively and persist frozen inputs."""
    config = validate_config(config)
    _strict_frozen(source_hashes, analysis_source_hashes(), "analysis source hashes")
    _strict_frozen(seed_manifest, study_schedules(config), "seed manifest")
    # Validate serialization and all provenance before creating any path.
    canonical_json(environment)
    source_commit = _source_commit()
    cfg_digest = digest(config)
    selection = selection_manifest(config)
    selection_bytes = _world_bytes(selection)
    selection_id = _sha256(selection_bytes)
    metadata = {
        "config": config,
        "config_digest": cfg_digest,
        "environment": environment,
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "source_id": digest({"commit": source_commit, "hashes": source_hashes}),
        "seed_manifest": seed_manifest,
        "selection_manifest_id": selection_id,
        "selection_manifest_locator": f"selection-{selection_id}.json",
        "catalog_schema_path": catalog_schema()["$id"],
        "catalog_schema_sha256": _sha256(CATALOG_SCHEMA_BYTES),
    }
    payload = _world_bytes(metadata)
    root = Path(root).absolute()
    checkout = Path(__file__).resolve().parents[1]
    if root == checkout or checkout in root.parents or ".." in root.parts:
        raise ValueError("archive root must be outside checkout")
    for parent in [*reversed(root.parents), root]:
        if parent.is_symlink():
            raise ValueError("archive root must not contain symlinks")
    archive = exclusive_output_dir(root / f"linear-recovery-{cfg_digest}")
    _world_archive(archive)
    (archive / "schemas").mkdir()
    _catalog_file(archive, metadata["catalog_schema_path"], CATALOG_SCHEMA_BYTES)
    _catalog_file(archive, metadata["selection_manifest_locator"], selection_bytes)
    _catalog_file(archive, "run.json", payload)
    return archive, metadata


def _exclusive_json(path: Path, value: Any) -> str:
    payload = (canonical_json(_json_safe(value)) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


WORLD_VERSION = "ols-world-persistence/v1"
_ARRAY_VERSION = "ols-world-array/v1"
_ATTEMPT_DOMAIN = b"ols-world-persistence/v1/attempt\0"
_WORLD_DOMAIN = b"ols-world-persistence/v1/world\0"
_REFERENCE_FIELDS = frozenset(
    {
        "version",
        "attempt_id",
        "reservation_locator",
        "reservation_sha256",
        "world_id",
        "world_locator",
        "world_sha256",
    }
)
_ARRAY_FIELDS = frozenset({"version", "locator", "data_sha256", "dtype", "shape", "nbytes"})


def _world_bytes(value):
    return (canonical_json(value) + "\n").encode()


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _logical_attempt_id(attempt):
    if not isinstance(attempt, dict):
        raise ValueError("attempt must be a dictionary")
    index = attempt.get("attempt")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError("attempt index must be a nonnegative integer excluding bool")
    return _sha256(_ATTEMPT_DOMAIN + str(int(index)).encode("ascii"))


def _world_archive(archive):
    path = Path(archive).absolute()
    if ".." in path.parts:
        raise ValueError("invalid archive path")
    for component in [*reversed(path.parents), path]:
        mode = component.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("archive components must be real directories, not symlinks")
    checkout = Path(__file__).resolve().parents[1]
    if path == checkout or checkout in path.parents:
        raise ValueError("archive must be outside checkout")
    return path


def _world_path(archive, locator, kind):
    patterns = {
        "world": r"world-[0-9a-f]{64}\.json",
        "reservation": r"attempt-[0-9a-f]{64}\.json",
        "array": r"arrays/[0-9a-f]{64}\.bin",
        "directory": r"arrays",
    }
    if not isinstance(locator, str) or re.fullmatch(patterns[kind], locator) is None:
        raise ValueError(f"invalid {kind} locator")
    archive = _world_archive(archive)
    path = archive / locator
    if not path.resolve().is_relative_to(archive):
        raise ValueError("locator escapes archive")
    parts = locator.split("/")
    current = archive
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        directory = index < len(parts) - 1 or kind == "directory"
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
            raise ValueError("locator components must not be symlinks or special files")
    return path


@contextmanager
def _world_parent(archive, locator, kind):
    """Anchor IO to no-follow directory descriptors, including every archive component."""
    path = _world_path(archive, locator, kind)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, path.name
    finally:
        os.close(fd)


def _read_world_file(archive, locator, kind):
    with _world_parent(archive, locator, kind) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("expected a regular evidence file")
            return stream.read()


def _write_world_file(archive, locator, kind, payload):
    with _world_parent(archive, locator, kind) as (parent, name):
        fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)


def _world_array_directory(archive):
    with _world_parent(archive, "arrays", "directory") as (parent, name):
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            _world_path(archive, name, "directory")


def _preflight_world_path(archive, locator, kind):
    path = _world_path(archive, locator, kind)
    parent = path.parent
    if not parent.exists():
        parent = parent.parent
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(f"archive path is not writable: {parent}")
    return path


def _world_dtype(value):
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        raise ValueError("invalid array dtype")
    if not isinstance(value, str) or dtype.str != value or dtype.kind not in "biufc":
        raise ValueError("unsupported or noncanonical array dtype")
    return dtype


def _normalize_world(value, blobs):
    """Pure normalization: immutable blob bytes are only collected in memory."""
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        _world_dtype(array.dtype.str)
        if array.dtype.metadata or not np.isfinite(array).all():
            raise ValueError("cannot persist nonfinite or unsupported array")
        raw = array.tobytes(order="C")
        sha = _sha256(raw)
        locator = f"arrays/{sha}.bin"
        if locator in blobs and blobs[locator] != raw:
            raise ValueError("array blob digest collision")
        blobs[locator] = raw
        ref = {
            "version": _ARRAY_VERSION,
            "locator": locator,
            "data_sha256": sha,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": len(raw),
        }
        return {"__numpy_scalar__" if isinstance(value, np.generic) else "__ndarray_ref__": ref}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("world dictionary keys must be strings")
        if {"__numpy_scalar__", "__ndarray_ref__"} & value.keys():
            raise ValueError("reserved world encoding key")
        return {key: _normalize_world(item, blobs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_world(item, blobs) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot persist nonfinite number")
        return value
    raise ValueError(f"unsupported world value: {type(value).__name__}")


def _prepare_world(archive, attempt, result):
    archive = _world_archive(archive)
    attempt_id = _logical_attempt_id(attempt)
    if not isinstance(result, dict):
        raise ValueError("result must be a dictionary")
    blobs = {}
    body = {
        "version": WORLD_VERSION,
        "attempt": _normalize_world(attempt, blobs),
        "result": _normalize_world(result, blobs),
    }
    payload = _world_bytes(body)
    world_id = _sha256(_WORLD_DOMAIN + payload)
    reference = {
        "version": WORLD_VERSION,
        "attempt_id": attempt_id,
        "reservation_locator": f"attempt-{attempt_id}.json",
        "world_id": world_id,
        "world_locator": f"world-{world_id}.json",
        "world_sha256": _sha256(payload),
    }
    reservation = _world_bytes(reference)
    reference["reservation_sha256"] = _sha256(reservation)
    for kind in ("reservation", "world"):
        path = _preflight_world_path(archive, reference[f"{kind}_locator"], kind)
        if path.exists():
            raise FileExistsError(path)
    for locator, raw in blobs.items():
        path = _preflight_world_path(archive, locator, "array")
        if path.exists() and _read_world_file(archive, locator, "array") != raw:
            raise ValueError("array blob digest collision or tampering")
    return archive, reference, reservation, payload, blobs


def persist_world(
    archive: str | Path, attempt: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Reserve first, preserve partial failures, and return a self-verified trusted reference."""
    archive, reference, reservation, payload, blobs = _prepare_world(archive, attempt, result)
    _write_world_file(archive, reference["reservation_locator"], "reservation", reservation)
    if blobs:
        _world_array_directory(archive)
    for locator, raw in blobs.items():
        try:
            _write_world_file(archive, locator, "array", raw)
        except FileExistsError:
            if _read_world_file(archive, locator, "array") != raw:
                raise ValueError("array blob digest collision or tampering")
    _write_world_file(archive, reference["world_locator"], "world", payload)
    read_world(archive, reference)
    return reference


def _canonical_world_record(payload):
    try:
        record = json.loads(payload)
        if _world_bytes(record) != payload:
            raise ValueError("noncanonical evidence bytes")
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid canonical evidence bytes") from exc
    return record


def _decode_world(value, archive):
    if isinstance(value, list):
        return [_decode_world(item, archive) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite world number")
        return value
    tags = {"__ndarray_ref__", "__numpy_scalar__"} & value.keys()
    if not tags:
        return {key: _decode_world(item, archive) for key, item in value.items()}
    if len(value) != 1:
        raise ValueError("invalid array encoding")
    tag = next(iter(tags))
    ref = value[tag]
    if not isinstance(ref, dict) or ref.keys() != _ARRAY_FIELDS or ref["version"] != _ARRAY_VERSION:
        raise ValueError("invalid array reference fields/version")
    sha = ref["data_sha256"]
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("invalid array data hash")
    if ref["locator"] != f"arrays/{sha}.bin":
        raise ValueError("array locator/hash mismatch")
    dtype = _world_dtype(ref["dtype"])
    shape, nbytes = ref["shape"], ref["nbytes"]
    if (
        not isinstance(shape, list)
        or any(type(n) is not int or n < 0 for n in shape)
        or type(nbytes) is not int
        or nbytes < 0
        or math.prod(shape) * dtype.itemsize != nbytes
        or (tag == "__numpy_scalar__" and shape != [])
    ):
        raise ValueError("array dtype/shape/nbytes mismatch")
    raw = _read_world_file(archive, ref["locator"], "array")
    if len(raw) != nbytes or _sha256(raw) != sha:
        raise ValueError("array blob hash/nbytes mismatch")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError("nonfinite array blob")
    return array[()] if tag == "__numpy_scalar__" else array


def read_world(archive: str | Path, reference: dict[str, Any]) -> dict[str, Any]:
    """Verify evidence against a caller-retained trusted reference before decoding."""
    archive = _world_archive(archive)
    if not isinstance(reference, dict) or reference.keys() != _REFERENCE_FIELDS:
        raise ValueError("a strict trusted world reference is required")
    if reference["version"] != WORLD_VERSION:
        raise ValueError("invalid world reference version")
    for field in ("attempt_id", "world_id", "world_sha256", "reservation_sha256"):
        if (
            not isinstance(reference[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", reference[field]) is None
        ):
            raise ValueError("invalid reference digest")
    if (
        reference["reservation_locator"] != f"attempt-{reference['attempt_id']}.json"
        or reference["world_locator"] != f"world-{reference['world_id']}.json"
    ):
        raise ValueError("reference locator/identity mismatch")
    reservation = _read_world_file(archive, reference["reservation_locator"], "reservation")
    if _sha256(reservation) != reference["reservation_sha256"]:
        raise ValueError("reservation digest mismatch")
    expected = {key: value for key, value in reference.items() if key != "reservation_sha256"}
    if _canonical_world_record(reservation) != expected:
        raise ValueError("reservation linkage mismatch")
    payload = _read_world_file(archive, reference["world_locator"], "world")
    if _sha256(payload) != reference["world_sha256"]:
        raise ValueError("world digest mismatch")
    if _sha256(_WORLD_DOMAIN + payload) != reference["world_id"]:
        raise ValueError("world domain digest mismatch")
    body = _canonical_world_record(payload)
    if (
        not isinstance(body, dict)
        or set(body) != {"version", "attempt", "result"}
        or body["version"] != WORLD_VERSION
    ):
        raise ValueError("invalid world body schema")
    decoded = _decode_world(body, archive)
    if _logical_attempt_id(decoded["attempt"]) != reference["attempt_id"] or not isinstance(
        decoded["result"], dict
    ):
        raise ValueError("world logical attempt/result mismatch")
    return decoded


def _schema_check(value, schema, name="catalog"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _schema_check(value, option, name)
                return
            except ValueError:
                pass
        raise ValueError(f"{name}: no valid nullable type")
    if "enum" in schema:
        if not any(type(value) is type(item) and value == item for item in schema["enum"]):
            raise ValueError(f"{name}: invalid enum")
        return
    kind = schema["type"]
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name}: invalid finite number")
    elif type(value) is not types[kind]:
        raise ValueError(f"{name}: invalid type")
    if kind == "object":
        if set(value) != set(schema["required"]):
            raise ValueError(f"{name}: invalid keys")
        for key, child in schema["properties"].items():
            _schema_check(value[key], child, f"{name}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise ValueError(f"{name}: invalid array length")
        for item in value:
            _schema_check(item, schema["items"], name)
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or (
            "pattern" in schema and re.fullmatch(schema["pattern"], value) is None
        ):
            raise ValueError(f"{name}: invalid string/hash/locator")
    elif kind in ("integer", "number"):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ValueError(f"{name}: invalid integer range")


def _catalog_inputs(archive):
    schema_bytes = _catalog_file(archive, "schemas/ols-world-catalog-v1.json")
    if schema_bytes != CATALOG_SCHEMA_BYTES:
        raise ValueError("external schema bytes mismatch")
    run_bytes = _catalog_file(archive, "run.json")
    run = _canonical_world_record(run_bytes)
    fields = {
        "config",
        "config_digest",
        "environment",
        "source_commit",
        "source_hashes",
        "source_id",
        "seed_manifest",
        "selection_manifest_id",
        "selection_manifest_locator",
        "catalog_schema_path",
        "catalog_schema_sha256",
    }
    if not isinstance(run, dict) or set(run) != fields:
        raise ValueError("invalid run provenance fields")
    cfg = validate_config(run["config"])
    if (
        run["config_digest"] != digest(cfg)
        or run["catalog_schema_sha256"] != _sha256(schema_bytes)
        or run["catalog_schema_path"] != catalog_schema()["$id"]
    ):
        raise ValueError("run config/schema binding mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", run["source_commit"]) is None:
        raise ValueError("invalid source commit")
    _schema_check(run["source_hashes"], catalog_schema()["properties"]["analysis_file_hashes"])
    if run["source_id"] != digest({"commit": run["source_commit"], "hashes": run["source_hashes"]}):
        raise ValueError("source identity mismatch")
    _strict_frozen(run["seed_manifest"], study_schedules(cfg), "seed manifest")
    selection = _world_bytes(selection_manifest(cfg))
    sid = _sha256(selection)
    if (
        run["selection_manifest_id"] != sid
        or run["selection_manifest_locator"] != f"selection-{sid}.json"
        or _catalog_file(archive, run["selection_manifest_locator"]) != selection
    ):
        raise ValueError("prefit selection manifest mismatch")
    return run, _sha256(run_bytes), json.loads(selection)


def _catalog_numeric_check(actual, expected, name):
    """Require exact recomputation equality, including array dtype/shape/content and keys.

    The persisted inputs and accepted numerical helpers are identical, so validation
    adds no tolerance or classification threshold. Normalization is pure/no-write.
    """
    _strict_frozen(_normalize_world(actual, {}), _normalize_world(expected, {}), name)


def _catalog_world_contract(body, cfg):
    """Validate persisted evidence by recomputation; never replace saved fits or labels."""
    attempt, result = body["attempt"], body["result"]
    _scheduled_attempt(cfg, attempt, "science")
    resolved = resolved_generation_config(cfg, attempt)
    _strict_frozen(
        result.get("generation"),
        {
            "phase": "science",
            "resolved_config": resolved,
            "resolved_config_digest": digest(resolved),
            "config_digest": digest(cfg),
            "generation_seed": attempt["world_seed"],
        },
        "generation binding",
    )
    if result.get("status", {}).get("truth_eligibility") != "eligible":
        raise ValueError(
            "catalog requires eligible finite sampled truth; failures stay in world records"
        )
    n, nc, nz = resolved["n_time_steps"], attempt["n_treatments"], attempt["n_covariates"]
    source = result["source"]
    if source["seed"] != attempt["world_seed"]:
        raise ValueError("source seed mismatch")
    graph = source["graph"]
    for name, shape in {
        "g_cy": (nc,),
        "g_zy": (nz,),
        "g_dy": (1,),
        "g_dc": (1, nc),
        "g_dz": (1, nz),
        "g_zc": (nz, nc),
        "g_cc": (nc, nc),
        "g_zz": (nz, nz),
    }.items():
        a = _finite_array(graph[name], shape, name)
        if not np.isin(a, [0, 1]).all():
            raise ValueError("invalid direct structure")
    dc, dz = np.flatnonzero(graph["g_cy"]), np.flatnonzero(graph["g_zy"])
    order = [*(f"beta[{k}]" for k in dc), *(f"rho_zy[{m}]" for m in dz), "intercept"]
    _strict_frozen(result["coefficient_order"], order, "coefficient order")
    p = len(order)
    shapes = {
        "truth": (p,),
        "design": (n, p),
        "fit_indices": (n,),
        "fit_row_mask": (n,),
        "all_row_mask": (n,),
        "direct_treatments": (len(dc),),
        "direct_controls": (len(dz),),
    }
    for key, shape in shapes.items():
        _finite_array(result[key], shape, key)
    if (
        not np.array_equal(result["fit_indices"], cfg["study"]["fit_rows"])
        or result["fit_indices"].dtype.kind not in "iu"
        or result["fit_row_mask"].dtype.kind != "b"
        or not result["fit_row_mask"].all()
        or result["all_row_mask"].dtype.kind != "b"
        or not result["all_row_mask"].all()
        or not np.array_equal(result["direct_treatments"], dc)
        or not np.array_equal(result["direct_controls"], dz)
    ):
        raise ValueError("row/direct selection mismatch")
    for key, value in {
        "n_full": n + resolved["carryover_burn_in"],
        "n_fit": n,
        "p": p,
        "rank_tolerance": cfg["study"]["rank_tolerance"],
    }.items():
        _strict_frozen(result[key], value, key)
    for key in ("y0", "y1", "outcome_noise", "u", "raw_observed_unadjusted"):
        _finite_array(result["targets"][key], (n,), key)
    _finite_array(source["data"]["contributions_observed"], (n, nc), "contributions_observed")
    _finite_array(source["data"]["saturation_scale"], (nc,), "saturation_scale")
    beta = _finite_array(source["params"]["beta"], (nc,), "beta")
    rho = _finite_array(source["params"]["rho_zy"], (nz,), "rho_zy")
    baseline = _finite_array(source["data"]["baseline_intrinsic"], (n,), "baseline")
    if not np.array_equal(result["truth"], np.r_[beta[dc], rho[dz], baseline[0]]):
        raise ValueError("sampled coefficient truth mismatch")
    audit = result["audit"]
    _strict_frozen(audit["burn_in"], resolved["carryover_burn_in"], "warmup")
    _strict_frozen(audit["n_full"], result["n_full"], "full history")
    for name, width in {"D": 1, "Z": nz, "C": nc, "C_base": nc, "phi": nc, "phi_base": nc}.items():
        _finite_array(audit["full"][name], (result["n_full"], width), name)
    for key in ("exogenous", "equation_parameters"):
        if not isinstance(audit[key], dict) or not audit[key]:
            raise ValueError("public reconstruction history missing")
        _finite_tree(audit[key])
    nf = result["n_full"]
    for name, shape in {
        "eps_d": (nf, 1),
        "eps_z": (nf, nz),
        "eps_c": (nf, nc),
        "eps_b": (nf,),
        "eps_y": (nf,),
        "eps_c_hf": (nf, nc),
        "eps_c_pulse": (nf, nc),
        "eps_z_hf": (nf, nz),
        "eps_z_pulse": (nf, nz),
    }.items():
        _finite_array(audit["exogenous"][name], shape, name)
    identities = {
        "reconstructed_D",
        "reconstructed_Z",
        "reconstructed_C",
        "reconstructed_C_base",
        "scalar_baseline",
        "observed_contribution",
        "base_contribution",
        "control_contribution",
        "direct_latent",
        "observed_base_indirect",
        "indirect_total",
        "baseline_additive",
        "outcome_additive",
        "target_truth",
        "noise_pair",
        "audited_noise",
    }
    if set(result["identities"]) != identities:
        raise ValueError("identity evidence fields mismatch")
    for metric in result["identities"].values():
        if (
            metric["tolerance"] != IDENTITY_TOLERANCE
            or metric["passed"] is not True
            or max(metric["all_max_abs"], metric["fit_max_abs"]) > IDENTITY_TOLERANCE
        ):
            raise ValueError("identity gate mismatch")
    if not result["identities"]:
        raise ValueError("identity evidence missing")
    # Recompute from retained inputs, not from other untrusted fitted/diagnostic fields.
    rows = result["fit_indices"]
    selected = result["design"][rows]
    rank_tolerance = cfg["study"]["rank_tolerance"]
    nonintercept = selected[:, :-1]
    means, scales = nonintercept.mean(axis=0), nonintercept.std(axis=0)
    standardized = np.column_stack(
        [(nonintercept - means) / np.where(scales == 0, 1, scales), np.ones(len(rows))]
    )
    for key, expected in {
        "standardized_X": standardized,
        "column_means": means,
        "column_scales": scales,
        "constant_columns": np.flatnonzero(np.ptp(nonintercept, axis=0) == 0),
    }.items():
        _catalog_numeric_check(result[key], expected, key)
    if not np.array_equal(selected[:, -1], np.ones(len(rows))):
        raise ValueError("explicit intercept mismatch")
    for key, design in (("raw_design", selected), ("standardized_design", standardized)):
        expected = _design_diagnostics(design, rank_tolerance)
        expected["conditioning"] = orthogonal_status(
            scheduled=True,
            attempted=True,
            generated=True,
            truth_eligible=True,
            condition=expected["condition"],
        )["conditioning"]
        _catalog_numeric_check(result[key], expected, key)
    expected_fits = {}
    for name in ("y0", "y1"):
        fitted = fit_design(selected, result["targets"][name][rows], rank_tolerance)
        error = fitted["coefficients"] - result["truth"]
        fitted.update(
            coefficient_error=error,
            max_abs_error=float(np.max(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))),
            residual=result["targets"][name][rows] - selected @ fitted["coefficients"],
        )
        expected_fits[name] = fitted
    _catalog_numeric_check(result["fits"], expected_fits, "fits")
    expected_targets = {
        "raw_observed_unadjusted": source["data"]["outcome"],
        "u": source["data"]["latent_unobserved_contribution"].sum(axis=1),
        "outcome_noise": source["data"]["outcome_noise"],
    }
    expected_targets["y1"] = expected_targets["raw_observed_unadjusted"] - expected_targets["u"]
    expected_targets["y0"] = expected_targets["y1"] - expected_targets["outcome_noise"]
    _catalog_numeric_check(result["targets"], expected_targets, "targets")
    metrics = {
        name: {
            scope: {
                "mean": float(values[index].mean()),
                "std": float(values[index].std()),
                "max_abs": float(np.max(np.abs(values[index]))),
            }
            for scope, index in (("all", np.arange(n)), ("fit", rows))
        }
        for name, values in expected_targets.items()
    }
    _catalog_numeric_check(result["target_metrics"], metrics, "target metrics")
    expected_shapes = {
        "treatments": (n, nc),
        "treatments_base": (n, nc),
        "covariates": (n, nz),
        "latent_unobserved": (n, 1),
        "contributions_observed": (n, nc),
        "contributions": (n, nc),
        "covariate_contribution": (n, nz),
        "latent_unobserved_contribution": (n, 1),
        "baseline_intrinsic": (n,),
        "outcome_noise": (n,),
        "outcome": (n,),
        "baseline": (n,),
        "indirect_effects": (n,),
        "indirect_effects_by_source": (n, 3),
        "saturation_scale": (nc,),
    }
    for name, shape in expected_shapes.items():
        _finite_array(source["data"][name], shape, name)
    _catalog_numeric_check(result["shapes"], expected_shapes, "source dimensions")
    raw = result["raw_design"]
    expected_status = orthogonal_status(
        scheduled=True,
        attempted=True,
        generated=True,
        truth_eligible=True,
        rank=raw["rank"],
        p=p,
        condition=raw["condition"],
        noisefree_recovered=bool(
            raw["rank"] == p
            and result["fits"]["y0"]["max_abs_error"] <= NOISEFREE_COEFFICIENT_TOLERANCE
        ),
        paired_noisy_available=True,
        raw_observed_misspecified=bool(np.any(result["targets"]["u"] != 0)),
        reason=";".join(result["reasons"]) or None,
    )
    expected_status["paired_noisy"]["max_abs_error"] = result["fits"]["y1"]["max_abs_error"]
    _strict_frozen(result["status"], expected_status, "frozen status labels")


def catalog_entry(*, archive, world_reference):
    """Build handoff from verified persisted evidence, never inline arrays or caller labels."""
    run, parent_id, selection = _catalog_inputs(archive)
    body = read_world(archive, world_reference)
    try:
        _catalog_world_contract(body, run["config"])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("missing or malformed analyzed-world evidence") from exc
    raw = _canonical_world_record(
        _read_world_file(archive, world_reference["world_locator"], "world")
    )
    result = body["result"]

    def reference(*path):
        node = raw
        for key in path:
            node = node[key]
        return {
            "world_locator": world_reference["world_locator"],
            "path": list(path),
            "sha256": digest(node),
            "array": node.get("__ndarray_ref__") if isinstance(node, dict) else None,
        }

    truth_paths = {
        "coefficients": ("truth",),
        "direct_structure": ("source", "graph"),
        "design": ("design",),
        "contributions_observed": ("source", "data", "contributions_observed"),
        "y1": ("targets", "y1"),
        "noise": ("targets", "outcome_noise"),
        "scales": ("source", "data", "saturation_scale"),
        "public_history": ("audit",),
        "rows": ("fit_indices",),
        "row_mask": ("fit_row_mask",),
    }
    estimate_paths = {
        "y0_coefficients": ("fits", "y0", "coefficients"),
        "y1_coefficients": ("fits", "y1", "coefficients"),
        "status": ("status",),
        "raw_design": ("raw_design",),
        "standardized_design": ("standardized_design",),
    }
    status = result["status"]
    entry = {
        "schema_version": "ols-world-catalog/v1",
        "schema_path": "schemas/ols-world-catalog-v1.json",
        "schema_sha256": _sha256(CATALOG_SCHEMA_BYTES),
        "parent_id": parent_id,
        "source_id": run["source_id"],
        "source_commit": run["source_commit"],
        "analysis_file_hashes": run["source_hashes"],
        "config_digest": run["config_digest"],
        "resolved_config_digest": result["generation"]["resolved_config_digest"],
        "generation_seed": body["attempt"]["world_seed"],
        "world_id": world_reference["world_id"],
        "world_content_digest": world_reference["world_sha256"],
        "world_reference": dict(world_reference),
        "selection_manifest_id": run["selection_manifest_id"],
        "selection_manifest_locator": run["selection_manifest_locator"],
        "selection": selection["rows"][body["attempt"]["attempt"]],
        "parameter_draw_id": digest(
            {
                "domain": "ols-parameter-draw/v1",
                "source_id": run["source_id"],
                "seed": body["attempt"]["world_seed"],
                "resolved_config_digest": result["generation"]["resolved_config_digest"],
                "params": digest(raw["result"]["source"]["params"]),
                "graph": digest(raw["result"]["source"]["graph"]),
            }
        ),
        "data_draw_id": digest(
            {"domain": "ols-data-draw/v1", "world_id": world_reference["world_id"]}
        ),
        "noise_replicate_id": None,
        "replicate_kind": "fresh_world",
        "pair_targets": ["y0", "y1"],
        "coefficient_order": result["coefficient_order"],
        "n_full": result["n_full"],
        "warmup": result["audit"]["burn_in"],
        "known_sampled_truth": {
            "kind": "finite_sampled_truth_not_estimates",
            **{key: reference("result", *path) for key, path in truth_paths.items()},
            "prior_cond": None,
            "prior_cond_status": "unknown_not_retained",
        },
        "inferred_diagnostics": {
            "kind": "estimates_not_sampled_truth",
            **{key: reference("result", *path) for key, path in estimate_paths.items()},
            "labels": {
                "rank": status["rank"],
                "conditioning": status["conditioning"],
                "noisefree_recovery": status["noisefree_recovery"],
                "paired_noisy_available": status["paired_noisy"]["available"],
            },
        },
    }
    _schema_check(entry, catalog_schema())
    return entry


def validate_catalog_entry(archive, entry):
    _schema_check(entry, catalog_schema())
    expected = catalog_entry(archive=archive, world_reference=entry["world_reference"])
    _strict_frozen(entry, expected, "catalog evidence bindings")
    return json.loads(canonical_json(entry))


def persist_catalog_entry(archive, entry):
    entry = validate_catalog_entry(archive, entry)
    payload = _world_bytes(entry)
    sha = _sha256(payload)
    reference = {"version": "ols-world-catalog/v1", "locator": f"catalog-{sha}.json", "sha256": sha}
    _catalog_file(archive, reference["locator"], payload)
    read_catalog_entry(archive, reference)
    return reference


def read_catalog_entry(archive, reference):
    if (
        not isinstance(reference, dict)
        or set(reference) != {"version", "locator", "sha256"}
        or reference["version"] != "ols-world-catalog/v1"
        or not isinstance(reference["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
        or reference["locator"] != f"catalog-{reference['sha256']}.json"
    ):
        raise ValueError("strict trusted catalog reference required")
    payload = _catalog_file(archive, reference["locator"])
    if _sha256(payload) != reference["sha256"]:
        raise ValueError("catalog digest mismatch")
    return validate_catalog_entry(archive, _canonical_world_record(payload))


def frozen_schedule(
    config: dict[str, Any], elapsed_after_calibration: float, conservative_per_world_cost: float
) -> dict[str, Any]:
    """Budget arithmetic only; final allocation requires allocate_from_calibration."""
    study = validate_config(config)["study"]
    _positive_time(conservative_per_world_cost)
    _positive_time(elapsed_after_calibration, zero=True)
    remaining = max(
        0,
        study["wall_time_seconds"]
        - elapsed_after_calibration
        - study["finalization_reserve_seconds"],
    )
    # Clamp before division to avoid overflow for valid tiny positive costs.
    count = (
        study["maximum_schedule"]
        if remaining / study["maximum_schedule"] >= conservative_per_world_cost
        else math.floor(remaining / conservative_per_world_cost)
    )
    return {
        "frozen_attempt_count": count,
        "manifest": study_schedules(config)["science"][:count],
        "formula": study["frozen_count_formula"],
        "inputs": {
            "maximum_schedule": study["maximum_schedule"],
            "wall_time_seconds": study["wall_time_seconds"],
            "elapsed_after_calibration": elapsed_after_calibration,
            "reserve": study["finalization_reserve_seconds"],
            "conservative_per_world_cost": conservative_per_world_cost,
        },
    }


# Compact reporting is a pure projection of retained evidence, not an execution engine.
COMPACT_VERSION = "ols-compact-summary/v1"
_EVIDENCE_VERSION = "ols-report-evidence/v1"
_COMPACT_LIMITATIONS = [
    "Scalar-baseline restricted population; 104 reported rows; warmup applied once before reported rows.",
    "World-unit descriptive, non-simultaneous Wilson 95% frequencies, not causal identification.",
    "Runtime-dependent dropout: outcomes conditional on retained verified generation and eligibility.",
    "Graph patterns/G exclude missing evidence; evidence_unavailable/A is not a latent graph pattern.",
    "400 is planned allocation adequacy only, not an execution stop or subgroup precision guarantee.",
    "Compact validation establishes integrity and accounting, not independent execution provenance or raw science verification.",
]


@contextmanager
def _portable_report_parent(directory):
    """Read-only compact directories may be inside a portable checkout."""
    path = Path(directory).absolute()
    if ".." in path.parts:
        raise ValueError("invalid compact directory")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, None
    finally:
        os.close(fd)


def _report_file(archive, reference, payload=None, *, portable=False):
    """Reuse descriptor-anchored archive checks; never follow or clobber a report file."""
    if (
        type(reference) is not dict
        or set(reference) != {"version", "kind", "locator", "sha256"}
        or reference["version"] != _EVIDENCE_VERSION
        or reference["kind"] not in ("start", "terminal", "graph", "summary")
        or type(reference["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
        or reference["locator"] != f"{reference['kind']}-{reference['sha256']}.json"
    ):
        raise ValueError("strict trusted report reference required")
    # The existing helper anchors all parent components. The leaf is independently
    # constrained above and opened with O_NOFOLLOW (including on reads).
    if portable and payload is not None:
        raise ValueError("portable compact access is read-only")
    parents = (
        _portable_report_parent(archive)
        if portable
        else _world_parent(archive, f"world-{reference['sha256']}.json", "world")
    )
    with parents as (parent, _):
        flags = (
            os.O_RDONLY | os.O_NONBLOCK if payload is None else os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        fd = os.open(reference["locator"], flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "rb" if payload is None else "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("report evidence must be regular")
            if payload is not None:
                stream.write(payload)
                return None
            raw = stream.read()
    if _sha256(raw) != reference["sha256"]:
        raise ValueError("report evidence digest mismatch")
    return _canonical_world_record(raw)


def persist_report_evidence(archive, kind, body):
    """Engineering persistence seam; a seal is integrity, not execution attestation."""
    raw = _world_bytes(body)
    sha = _sha256(raw)
    reference = {
        "version": _EVIDENCE_VERSION,
        "kind": kind,
        "locator": f"{kind}-{sha}.json",
        "sha256": sha,
    }
    _report_file(archive, reference, raw)
    return reference


def _report_capsule(reference, body, kind):
    if type(reference) is not dict or reference != {
        "version": _EVIDENCE_VERSION,
        "kind": kind,
        "locator": f"{kind}-{_sha256(_world_bytes(body))}.json",
        "sha256": _sha256(_world_bytes(body)),
    }:
        raise ValueError("report capsule reference mismatch")


def _validate_report_input(envelope):
    if type(envelope) is not dict or set(envelope) != {"contract", "digest"}:
        raise ValueError("strict input envelope required")
    c = envelope["contract"]
    cfg = validate_config(c["config"])
    if envelope["digest"] != digest(c):
        raise ValueError("input envelope digest mismatch")
    expected = {
        "version",
        "config",
        "config_digest",
        "source_commit",
        "source_hashes",
        "catalog_schema_path",
        "catalog_schema_sha256",
        "environment",
        "seed_manifest",
        "seed_manifest_digest",
        "maximum_selection",
        "required_order",
    }
    if type(c) is not dict or set(c) != expected or c["version"] != cfg["input_contract_version"]:
        raise ValueError("invalid input contract fields/version")
    if (
        c["config_digest"] != digest(cfg)
        or re.fullmatch(r"[0-9a-f]{40}", c["source_commit"]) is None
    ):
        raise ValueError("invalid config/source binding")
    _schema_check(c["source_hashes"], catalog_schema()["properties"]["analysis_file_hashes"])
    _strict_frozen(c["seed_manifest"], study_schedules(cfg), "seed manifest")
    _strict_frozen(c["maximum_selection"], selection_manifest(cfg), "maximum selection")
    if (
        c["seed_manifest_digest"] != digest(c["seed_manifest"])
        or c["catalog_schema_sha256"] != CATALOG_SCHEMA_SHA256
        or c["catalog_schema_path"] != catalog_schema()["$id"]
    ):
        raise ValueError("schema/schedule binding mismatch")
    _strict_frozen(
        c["required_order"],
        "freeze source/schema/config/runtime/seeds before calibration or outcomes; then allocate and freeze prefix selection before science",
        "required input order",
    )
    env = c["environment"]
    if (
        type(env) is not dict
        or set(env)
        != {
            "python_executable",
            "python_version",
            "numpy_version",
            "dont_write_bytecode",
            "environment",
        }
        or env["dont_write_bytecode"] is not True
        or any(
            type(env[k]) is not str or not env[k]
            for k in ("python_executable", "python_version", "numpy_version")
        )
    ):
        raise ValueError("invalid runtime input binding")
    variables = env["environment"]
    if (
        type(variables) is not dict
        or set(variables)
        != {
            "PYTHONDONTWRITEBYTECODE",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYTHONPATH",
        }
        or any(variables[k] != "1" for k in variables if k != "PYTHONPATH")
        or type(variables["PYTHONPATH"]) is not str
    ):
        raise ValueError("invalid prescribed runtime inputs")
    return cfg


def _report_object_schema(properties):
    return {"type": "object", "required": list(properties), "properties": properties}


def _report_array_schema(items, shape):
    for size in reversed(shape):
        items = {"type": "array", "minItems": size, "maxItems": size, "items": items}
    return items


def _report_parameter_schema(attempt, parameters):
    """Closed sampled-parameter schema for the frozen composable/diverse population."""
    t, m = attempt["n_treatments"], attempt["n_covariates"]
    number, boolean = {"type": "number"}, {"type": "boolean"}
    integer = {"type": "integer"}

    def vector(item, n):
        return _report_array_schema(item, (n,))

    props = {}
    for keys, shape in (
        (
            (
                "beta",
                "carryover_alpha",
                "weibull_lam",
                "weibull_k",
                "hill_slope",
                "hill_kappa_mult",
                "logistic_lam",
                "mm_kappa_mult",
                "tanh_c",
                "root_alpha",
                "hf_sigma",
                "pulse_amp",
                "pulse_prob",
                "treatment_level",
            ),
            (t,),
        ),
        (("rho_zy", "covariate_hf_sigma", "covariate_pulse_amp", "covariate_pulse_prob"), (m,)),
        (("delta_dy",), (1,)),
        (("w_dc",), (1, t)),
        (("u_dz",), (1, m)),
        (("v_zc",), (m, t)),
        (("alpha_cc",), (t, t)),
        (("gamma_zz",), (m, m)),
    ):
        props.update({key: _report_array_schema(number, shape) for key in keys})
    props.update(
        {
            "carryover_family": vector({"enum": [0, 1, 2]}, t),
            "sat_family": vector({"enum": list(range(6))}, t),
            "l_max": {"enum": [8]},
            "baseline_floor": {"type": "null"},
            "baseline_floor_scope": {"enum": ["intercept"]},
            "confounding_strength": number,
        }
    )
    for key, n in (
        ("use_hf", t),
        ("use_pulse", t),
        ("use_covariate_hf", m),
        ("use_covariate_pulse", m),
    ):
        props[key] = vector(boolean, n)
    for suffix, n in (("d", 1), ("z", m), ("c", t), ("b", 1), ("y", 1)):
        rw = {
            "mean": vector(number, n),
            "std": vector(number, n),
            "positive_only": {"enum": [suffix == "c"]},
        }
        if suffix != "y":
            rw.update(smoothness=vector(number, n), rw_smoothness_max_weeks={"enum": [26]})
        props["rw_" + suffix] = _report_object_schema(rw)
    roles = {}
    for role, n in (("treatment", t), ("covariate", m)):
        components = (
            "hf",
            "pulse",
            "onset",
            "offset",
            "flighting",
            "level_jump",
            "seasonal",
            "trend",
        )
        use = _report_object_schema({key: vector(boolean, n) for key in components})
        jumps = {"count": {"enum": [1]}, "week": _report_array_schema(integer, (1, n))}
        jumps.update(
            {
                key: _report_array_schema(number, (1, n))
                for key in (("factor", "log_factor") if role == "treatment" else ("size",))
            }
        )
        schedules = {
            "onset": _report_object_schema({"start": vector(integer, n)}),
            "offset": _report_object_schema({"stop": vector(integer, n)}),
            "flighting": _report_object_schema(
                {key: vector(integer, n) for key in ("period", "on_weeks", "phase")}
            ),
            "level_jump": _report_object_schema(jumps),
            "seasonal": _report_object_schema(
                {key: vector(number, n) for key in ("amplitude", "period", "phase")}
            ),
            "trend": _report_object_schema({"change": vector(number, n)}),
        }
        flags = parameters["trajectory"][role]["use"]
        _schema_check(flags, use, "trajectory flags")
        # sample_scm's static model emits schedule leaves iff a concrete flag is set.
        roles[role] = _report_object_schema(
            {"use": use, **{key: schema for key, schema in schedules.items() if any(flags[key])}}
        )
    props["trajectory"] = _report_object_schema(roles)
    return _report_object_schema(props)


def _validate_compact_analysis(analysis, attempt, graph):
    number = {"type": "number"}
    nonnegative = {"type": "number", "minimum": 0}
    boolean = {"type": "boolean"}
    nullable_number = {"anyOf": [nonnegative, {"type": "null"}]}
    paired = {"available": boolean}
    if type(analysis) is not dict or set(analysis) != {"status", "reasons", "details"}:
        raise ValueError("strict compact analysis required")
    status = analysis["status"]
    # The analyzer emits this diagnostic only after a y1 fit exists.
    if (
        type(status) is dict
        and type(status.get("paired_noisy")) is dict
        and "max_abs_error" in status["paired_noisy"]
    ):
        paired["max_abs_error"] = nonnegative
    _schema_check(
        status,
        _report_object_schema(
            {
                "scheduling": {"enum": ["attempted"]},
                "generation": {"enum": ["generated"]},
                "truth_eligibility": {"enum": ["eligible", "ineligible"]},
                "rank": {"enum": [None, "full_rank", "rank_deficient"]},
                "conditioning": {"enum": [None, "easy", "moderate", "hard"]},
                "noisefree_recovery": {"enum": [None, False, True]},
                "paired_noisy": _report_object_schema(paired),
                "raw_observed_misspecified": {"enum": [None, False, True]},
                "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            }
        ),
        "analysis.status",
    )
    _schema_check(
        analysis["reasons"], {"type": "array", "items": {"type": "string"}}, "analysis.reasons"
    )
    d = analysis["details"]
    if d is None:
        return
    if graph is None or status["truth_eligibility"] != "eligible":
        raise ValueError("diagnostic details require eligible generated graph")
    p = sum(graph["g_cy"]) + sum(graph["g_zy"]) + 1
    vector = _report_array_schema(number, (p,))
    rank = {"type": "integer", "minimum": 0, "maximum": p}
    design = _report_object_schema(
        {
            "rank": rank,
            "singular_values": _report_array_schema(nonnegative, (p,)),
            "condition": nullable_number,
            "rank_tolerance": nonnegative,
            "rank_cutoff": nonnegative,
            "shape": _report_array_schema({"type": "integer", "minimum": 1}, (2,)),
            "conditioning": {"enum": ["easy", "moderate", "hard"]},
        }
    )
    fit = _report_object_schema(
        {
            "coefficients": vector,
            "rank": rank,
            "singular_values": _report_array_schema(nonnegative, (p,)),
            "condition": nullable_number,
            "p": {"enum": [p]},
            "n_fit": {"enum": [104]},
            "coefficient_error": vector,
            "max_abs_error": nonnegative,
            "rmse": nonnegative,
            "residual_max_abs": nonnegative,
            "residual_rmse": nonnegative,
        }
    )
    identity = _report_object_schema(
        {
            "all_max_abs": nonnegative,
            "fit_max_abs": nonnegative,
            "tolerance": nonnegative,
            "passed": boolean,
        }
    )
    identities = (
        "reconstructed_D",
        "reconstructed_Z",
        "reconstructed_C",
        "reconstructed_C_base",
        "scalar_baseline",
        "observed_contribution",
        "base_contribution",
        "control_contribution",
        "direct_latent",
        "observed_base_indirect",
        "indirect_total",
        "baseline_additive",
        "outcome_additive",
        "target_truth",
        "noise_pair",
        "audited_noise",
    )
    metrics = _report_object_schema({"mean": number, "std": nonnegative, "max_abs": nonnegative})
    _schema_check(
        d,
        _report_object_schema(
            {
                "truth": vector,
                "coefficient_order": _report_array_schema(
                    {"type": "string", "pattern": r"(?:beta\[[0-9]\]|rho_zy\[[0-9]\]|intercept)"},
                    (p,),
                ),
                "parameters": _report_parameter_schema(attempt, d["parameters"]),
                "n_full": {"enum": [104]},
                "warmup": {"enum": [0]},
                "n_fit": {"enum": [104]},
                "p": {"enum": [p]},
                "fit_indices": _report_array_schema(
                    {"type": "integer", "minimum": 0, "maximum": 103}, (104,)
                ),
                "identities": _report_object_schema(dict.fromkeys(identities, identity)),
                "raw_design": design,
                "standardized_design": design,
                "target_metrics": _report_object_schema(
                    {
                        key: _report_object_schema({"all": metrics, "fit": metrics})
                        for key in ("y0", "y1", "u", "outcome_noise", "raw_observed_unadjusted")
                    }
                ),
                "fits": _report_object_schema({"y0": fit, "y1": fit}),
                "history": _report_object_schema(
                    {
                        "n_full": {"enum": [104]},
                        "burn_in": {"enum": [0]},
                        "source": {
                            "enum": [
                                "catalog known_sampled_truth.public_history; full horizon before slicing once"
                            ]
                        },
                    }
                ),
            }
        ),
        "analysis.details",
    )
    if "max_abs_error" not in paired:
        raise ValueError("paired fit requires retained noisy diagnostic")


def _validate_report_calibration(calibration, cfg):
    """Retain incomplete/failed calibration, but never arbitrary nested claims."""
    if type(calibration) is not list or len(calibration) > cfg["study"]["calibration_count"]:
        raise ValueError("invalid retained calibration")
    for record in calibration:
        if type(record) is dict and set(record) == {"status", "reason"}:
            _schema_check(
                record,
                _report_object_schema(
                    {"status": {"enum": ["failed"]}, "reason": {"type": "string", "minLength": 1}}
                ),
                "calibration failure",
            )
        else:
            _schema_check(
                record,
                _report_object_schema(
                    {
                        "attempt": _report_object_schema(
                            {
                                key: {"type": "integer", "minimum": 0}
                                for key in (
                                    "attempt",
                                    "n_treatments",
                                    "n_covariates",
                                    "n_latent",
                                    "world_seed",
                                )
                            }
                        ),
                        "phase": {"enum": ["calibration"]},
                        "status": {"enum": ["complete", "failed"]},
                        "duration": {"type": "number"},
                    }
                ),
                "calibration record",
            )
            _scheduled_attempt(cfg, record["attempt"], "calibration")


def _report_graph(graph, attempt):
    shapes = {
        "g_cy": (attempt["n_treatments"],),
        "g_zy": (attempt["n_covariates"],),
        "g_dy": (1,),
        "g_dc": (1, attempt["n_treatments"]),
        "g_dz": (1, attempt["n_covariates"]),
        "g_zc": (attempt["n_covariates"], attempt["n_treatments"]),
        "g_cc": (attempt["n_treatments"], attempt["n_treatments"]),
        "g_zz": (attempt["n_covariates"], attempt["n_covariates"]),
    }
    if type(graph) is not dict or set(graph) != set(shapes):
        raise ValueError("complete minimal graph required")
    result = {}
    for key, shape in shapes.items():
        a = _finite_array(graph[key], shape, key)
        if not np.isin(a, [0, 1]).all() or (key in ("g_cc", "g_zz") and np.tril(a).any()):
            raise ValueError("invalid graph mask/order")
        result[key] = a.astype(int).tolist()
    return result


def _compact_analysis(result):
    """Keep world-level diagnostics/parameters, never design/target/history arrays."""

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    compact = {"status": plain(result["status"]), "reasons": result["reasons"], "details": None}
    if result["status"]["truth_eligibility"] != "eligible" or not all(
        k in result for k in ("truth", "fits", "raw_design", "standardized_design")
    ):
        return compact
    fits = {}
    for target in ("y0", "y1"):
        if target not in result["fits"]:
            return compact
        fit = result["fits"][target]
        fits[target] = {k: plain(v) for k, v in fit.items() if k != "residual"}
        residual = fit["residual"]
        fits[target]["residual_max_abs"] = float(np.max(np.abs(residual)))
        fits[target]["residual_rmse"] = float(np.sqrt(np.mean(residual**2)))
    compact["details"] = plain(
        {
            "truth": result["truth"],
            "coefficient_order": result["coefficient_order"],
            "parameters": result["source"]["params"],
            "n_full": result["n_full"],
            "warmup": result["audit"]["burn_in"],
            "n_fit": result["n_fit"],
            "p": result["p"],
            "fit_indices": result["fit_indices"],
            "identities": result["identities"],
            "raw_design": result["raw_design"],
            "standardized_design": result["standardized_design"],
            "target_metrics": result["target_metrics"],
            "fits": fits,
            "history": {
                "n_full": result["n_full"],
                "burn_in": result["audit"]["burn_in"],
                "source": "catalog known_sampled_truth.public_history; full horizon before slicing once",
            },
        }
    )
    return compact


def _world_reference_shape(ref, attempt):
    if type(ref) is not dict or set(ref) != _REFERENCE_FIELDS or ref["version"] != WORLD_VERSION:
        raise ValueError("strict world reference required")
    for key in ("attempt_id", "world_id", "world_sha256", "reservation_sha256"):
        if type(ref[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", ref[key]) is None:
            raise ValueError("invalid world reference digest")
    if (
        ref["attempt_id"] != _logical_attempt_id(attempt)
        or ref["world_locator"] != f"world-{ref['world_id']}.json"
        or ref["reservation_locator"] != f"attempt-{ref['attempt_id']}.json"
        or ref["reservation_sha256"]
        != _sha256(_world_bytes({k: v for k, v in ref.items() if k != "reservation_sha256"}))
    ):
        raise ValueError("world reference binding mismatch")


def _validate_graph_world_node(graph_node, graph):
    if type(graph_node) is not dict or set(graph_node) != set(graph):
        raise ValueError("missing world graph node")
    for name, node in graph_node.items():
        if type(node) is not dict or set(node) != {"__ndarray_ref__"}:
            raise ValueError("graph node requires retained array reference")
        ref = node["__ndarray_ref__"]
        if type(ref) is not dict:
            raise ValueError("invalid graph array reference")
        dtype = _world_dtype(ref["dtype"])
        array = np.asarray(graph[name], dtype=dtype)
        _strict_frozen(node, _normalize_world(array, {}), "compact graph bytes")


def _compact_catalog_binding(cat, capsule, envelope, attempt):
    """Portable consistency checks, distinct from raw-world numeric recomputation."""
    world_ref = capsule["terminal"]["world_reference"]
    contract = envelope["contract"]
    analysis = capsule["analysis"]
    _schema_check(cat, catalog_schema())
    for key, value in {
        "world_reference": world_ref,
        "world_id": world_ref["world_id"],
        "world_content_digest": world_ref["world_sha256"],
        "generation_seed": attempt["world_seed"],
        "config_digest": contract["config_digest"],
        "source_commit": contract["source_commit"],
        "analysis_file_hashes": contract["source_hashes"],
        "source_id": digest(
            {"commit": contract["source_commit"], "hashes": contract["source_hashes"]}
        ),
        "selection": contract["maximum_selection"]["rows"][attempt["attempt"]],
    }.items():
        _strict_frozen(cat[key], value, "compact catalog binding")
    for section in ("known_sampled_truth", "inferred_diagnostics"):
        for ref in cat[section].values():
            if isinstance(ref, dict) and "world_locator" in ref:
                if ref["world_locator"] != world_ref["world_locator"]:
                    raise ValueError("catalog graph/world reference substitution")
    graph_node = capsule["graph_world_node"]
    _validate_graph_world_node(graph_node, capsule["graph"])
    direct = cat["known_sampled_truth"]["direct_structure"]
    if direct["sha256"] != digest(graph_node) or direct["path"] != ["result", "source", "graph"]:
        raise ValueError("catalog graph digest/path mismatch")
    details = analysis["details"]
    status = analysis["status"]
    labels = {
        "rank": status["rank"],
        "conditioning": status["conditioning"],
        "noisefree_recovery": status["noisefree_recovery"],
        "paired_noisy_available": status["paired_noisy"]["available"],
    }
    _strict_frozen(cat["inferred_diagnostics"]["labels"], labels, "catalog labels")
    for key in ("coefficient_order", "n_full", "warmup"):
        _strict_frozen(cat[key], details[key], "compact dimensions")
    p = sum(capsule["graph"]["g_cy"]) + sum(capsule["graph"]["g_zy"]) + 1
    if (
        details["p"] != p
        or len(details["truth"]) != p
        or details["n_fit"] != 104
        or details["fit_indices"] != list(range(104))
        or details["n_full"] != 104 + details["warmup"]
        or details["warmup"] != contract["config"]["population"]["carryover_burn_in"]
    ):
        raise ValueError("compact paired dimension/warmup mismatch")
    truth = _finite_array(details["truth"], (p,), "compact truth")
    for target in ("y0", "y1"):
        fit = details["fits"][target]
        coef = _finite_array(fit["coefficients"], (p,), "compact coefficients")
        error = coef - truth
        _strict_frozen(fit["coefficient_error"], error.tolist(), "compact coefficient errors")
        _strict_frozen(fit["max_abs_error"], float(np.max(np.abs(error))), "compact max error")
        _strict_frozen(fit["rmse"], float(np.sqrt(np.mean(error**2))), "compact rmse")
        if fit["n_fit"] != 104 or fit["p"] != p:
            raise ValueError("paired fit dimensions mismatch")
    full = details["raw_design"]["rank"] == p
    if status["rank"] != ("full_rank" if full else "rank_deficient"):
        raise ValueError("compact rank mismatch")
    recovered = full and details["fits"]["y0"]["max_abs_error"] <= NOISEFREE_COEFFICIENT_TOLERANCE
    if status["noisefree_recovery"] is not recovered:
        raise ValueError("compact recovery mismatch")


def _validate_report_start(start, attempt, envelope, allocation):
    if type(start) is not dict or set(start) != {
        "attempt",
        "phase",
        "input_digest",
        "allocation_digest",
        "started_elapsed",
        "resource",
    }:
        raise ValueError("invalid durable start fields")
    _strict_frozen(start["attempt"], attempt, "started prefix")
    if (
        start["phase"] != "science"
        or start["input_digest"] != envelope["digest"]
        or start["allocation_digest"] != digest(allocation)
    ):
        raise ValueError("started input/phase/allocation mismatch")
    _schema_check(
        start["resource"],
        _report_object_schema(
            {"host": {"type": "string", "minLength": 1}, "pid": {"type": "integer", "minimum": 1}}
        ),
        "start resource",
    )
    _positive_time(start["started_elapsed"], zero=True)


def _validate_report_terminal(end, attempt, start_reference):
    if type(end) is not dict or set(end) != {
        "attempt",
        "phase",
        "start_sha256",
        "finished_elapsed",
        "status",
        "generation_outcome",
        "errors",
        "world_reference",
        "catalog_reference",
        "graph_reference",
        "partial_references",
    }:
        raise ValueError("invalid terminal fields")
    _strict_frozen(end["attempt"], attempt, "terminal attempt")
    if (
        end["phase"] != "science"
        or end["start_sha256"] != start_reference["sha256"]
        or end["status"] not in ("complete", "recovered_unknown")
        or end["generation_outcome"] not in ("completed", "failed", "unknown")
    ):
        raise ValueError("unreconciled or invalid terminal record")
    if end["generation_outcome"] == "unknown" and end["status"] != "recovered_unknown":
        raise ValueError("unknown must be durably recovered")
    _positive_time(end["finished_elapsed"], zero=True)
    _schema_check(
        end["errors"],
        {
            "type": "array",
            "items": _report_object_schema(
                {
                    "stage": {"enum": ["generation", "analysis", "persistence", "timeout"]},
                    "type": {"type": "string", "minLength": 1},
                    "message": {"type": "string", "minLength": 1},
                }
            ),
        },
        "terminal errors",
    )
    if end["world_reference"] is not None:
        _world_reference_shape(end["world_reference"], attempt)
    if end["catalog_reference"] is not None:
        ref = end["catalog_reference"]
        _schema_check(
            ref,
            _report_object_schema(
                {
                    "version": {"enum": ["ols-world-catalog/v1"]},
                    "locator": {"type": "string", "pattern": r"catalog-[0-9a-f]{64}\.json"},
                    "sha256": {"type": "string", "pattern": r"[0-9a-f]{64}"},
                }
            ),
            "catalog reference",
        )
        if ref["locator"] != f"catalog-{ref['sha256']}.json":
            raise ValueError("catalog reference digest/locator mismatch")
    if end["graph_reference"] is not None:
        ref = end["graph_reference"]
        _schema_check(
            ref,
            _report_object_schema(
                {
                    "version": {"enum": [_EVIDENCE_VERSION]},
                    "kind": {"enum": ["graph"]},
                    "locator": {"type": "string", "pattern": r"graph-[0-9a-f]{64}\.json"},
                    "sha256": {"type": "string", "pattern": r"[0-9a-f]{64}"},
                }
            ),
            "graph reference",
        )
        if ref["locator"] != f"graph-{ref['sha256']}.json":
            raise ValueError("graph reference digest/locator mismatch")
    _schema_check(
        end["partial_references"],
        {
            "type": "array",
            "items": _report_object_schema(
                {
                    "kind": {"enum": ["reservation", "world", "array"]},
                    "locator": {"type": "string"},
                    "sha256": {"type": "string", "pattern": r"[0-9a-f]{64}"},
                }
            ),
        },
        "partial references",
    )
    for ref in end["partial_references"]:
        pattern = {
            "reservation": r"attempt-[0-9a-f]{64}\.json",
            "world": r"world-[0-9a-f]{64}\.json",
            "array": r"arrays/[0-9a-f]{64}\.bin",
        }[ref["kind"]]
        if re.fullmatch(pattern, ref["locator"]) is None:
            raise ValueError("invalid partial locator")


def _report_inventory(archive, envelope, allocation):
    """Every durable start and terminal participates; caller references cannot filter it."""
    with _world_parent(archive, "world-" + "0" * 64 + ".json", "world") as (parent, _):
        names = sorted(
            name
            for name in os.listdir(parent)
            if name.startswith(("start-", "terminal-")) and name.endswith(".json")
        )
    starts, terminals = {}, []
    for name in names:
        match = re.fullmatch(r"(start|terminal)-([0-9a-f]{64})\.json", name)
        if match is None:
            raise ValueError("invalid durable inventory locator")
        kind, sha = match.groups()
        ref = {"version": _EVIDENCE_VERSION, "kind": kind, "locator": name, "sha256": sha}
        body = _report_file(archive, ref)
        if type(body) is not dict or type(body.get("attempt")) is not dict:
            raise ValueError("invalid durable inventory body")
        attempt = body["attempt"]
        _scheduled_attempt(envelope["contract"]["config"], attempt, "science")
        i = attempt["attempt"]
        if i >= len(allocation["manifest"]):
            raise ValueError("durable record outside allocated prefix")
        if kind == "start":
            _validate_report_start(body, allocation["manifest"][i], envelope, allocation)
            if i in starts:
                raise ValueError("duplicate durable started identity")
            starts[i] = (ref, body)
        else:
            terminals.append((i, ref, body))
    if sorted(starts) != list(range(len(starts))):
        raise ValueError("durable starts are not canonical prefix")
    ends = {}
    for i, ref, body in terminals:
        if i not in starts:
            raise ValueError("orphan durable terminal")
        _validate_report_terminal(body, allocation["manifest"][i], starts[i][0])
        if i in ends:
            raise ValueError("duplicate/conflicting durable terminals")
        ends[i] = (ref, body)
    if set(ends) != set(starts):
        raise ValueError("missing durable terminal; unreconciled in-progress attempt")
    return [(starts[i], ends[i]) for i in range(len(starts))]


def _report_rows(cfg, envelope, allocation, capsules, timing):
    """Validate the durable started prefix, separately from the planned manifest."""
    manifest = allocation["manifest"]
    if type(capsules) is not list or len(capsules) > len(manifest):
        raise ValueError("invalid started ledger")
    if type(timing) is not dict or set(timing) != {
        "elapsed_seconds",
        "started_count",
        "clock",
        "finalization_seconds",
    }:
        raise ValueError("strict global timing required")
    for key in ("elapsed_seconds", "finalization_seconds"):
        _positive_time(timing[key], zero=True)
    if (
        type(timing["started_count"]) is not int
        or timing["started_count"] != len(capsules)
        or timing["clock"] != "monotonic_inclusive_before_setup"
        or timing["elapsed_seconds"] < allocation["inputs"]["elapsed_after_calibration"]
        or timing["finalization_seconds"] > timing["elapsed_seconds"]
        or timing["elapsed_seconds"] - timing["finalization_seconds"]
        < allocation["inputs"]["elapsed_after_calibration"]
    ):
        raise ValueError("missing started records or invalid clock")
    previous = allocation["inputs"]["elapsed_after_calibration"]
    rows = []
    for i, capsule in enumerate(capsules):
        if type(capsule) is not dict or set(capsule) != {
            "start_reference",
            "start",
            "terminal_reference",
            "terminal",
            "graph",
            "graph_capsule",
            "graph_world_node",
            "analysis",
            "catalog",
        }:
            raise ValueError("invalid ledger capsule fields")
        start, end = capsule["start"], capsule["terminal"]
        _report_capsule(capsule["start_reference"], start, "start")
        _report_capsule(capsule["terminal_reference"], end, "terminal")
        _validate_report_start(start, manifest[i], envelope, allocation)
        _validate_report_terminal(end, manifest[i], capsule["start_reference"])
        for value in (start["started_elapsed"], end["finished_elapsed"]):
            _positive_time(value, zero=True)
        if (
            not previous
            <= start["started_elapsed"]
            <= end["finished_elapsed"]
            <= timing["elapsed_seconds"] - timing["finalization_seconds"]
        ):
            raise ValueError("invalid monotonic inclusive timing")
        if (
            start["started_elapsed"]
            >= cfg["study"]["wall_time_seconds"] - cfg["study"]["finalization_reserve_seconds"]
        ):
            raise ValueError("attempt started beyond deadline reserve")
        previous = end["finished_elapsed"]
        errors = end["errors"]
        if type(errors) is not list:
            raise ValueError("invalid stage errors")
        for error in errors:
            if (
                type(error) is not dict
                or set(error) != {"stage", "type", "message"}
                or error["stage"] not in ("generation", "analysis", "persistence", "timeout")
                or any(type(error[k]) is not str or not error[k] for k in error)
            ):
                raise ValueError("authentic retained stage/error record required")
        stages = {e["stage"] for e in errors}
        graph = capsule["graph"]
        if graph is not None:
            _strict_frozen(graph, _report_graph(graph, manifest[i]), "graph")
        world_ref, graph_ref = end["world_reference"], end["graph_reference"]
        if world_ref is not None:
            _world_reference_shape(world_ref, manifest[i])
        if graph_ref is not None:
            graph_body = capsule["graph_capsule"]
            _report_capsule(graph_ref, graph_body, "graph")
            _strict_frozen(
                graph_body,
                {"attempt": manifest[i], "phase": "science", "graph": graph},
                "graph binding",
            )
        elif capsule["graph_capsule"] is not None:
            raise ValueError("unreferenced graph capsule")
        if graph is not None and world_ref is None and graph_ref is None:
            raise ValueError("fabricated graph without trusted reference")
        if end["generation_outcome"] == "failed" and (
            "generation" not in stages or graph is not None
        ):
            raise ValueError("confirmed generation failure requires stage error, not graph")
        if "generation" in stages and end["generation_outcome"] != "failed":
            raise ValueError("generation error contradicts terminal outcome")
        if graph is not None and end["generation_outcome"] != "completed":
            raise ValueError("graph contradicts generation outcome")
        if end["generation_outcome"] == "unknown" and end["status"] != "recovered_unknown":
            raise ValueError("unknown must be durably recovered")
        analysis = capsule["analysis"]
        if analysis is not None:
            if world_ref is None:
                raise ValueError("analysis requires verified world")
            _validate_compact_analysis(analysis, manifest[i], graph)
        if capsule["graph_world_node"] is not None:
            if world_ref is None or graph is None:
                raise ValueError("world graph node requires retained world/graph")
            _validate_graph_world_node(capsule["graph_world_node"], graph)
        stage = (
            "generation_failed"
            if end["generation_outcome"] == "failed"
            else "analysis_failed"
            if "analysis" in stages and graph is not None
            else "evidence_unavailable"
            if analysis is None
            else None
        )
        projection = project_attempt_state(
            cfg,
            manifest,
            manifest[i],
            started=True,
            graph=graph,
            analysis=analysis if graph is not None else None,
            failure_stage=stage,
        )
        cat = capsule["catalog"]
        if projection["E"]:
            if analysis["details"] is None:
                raise ValueError("eligible analysis requires paired diagnostic details")
            if cat is None:
                # A catalog write may fail after full world verification. The
                # builder still supplies the recomputed catalog, with no fake ref.
                raise ValueError("eligible world requires verified catalog projection")
            _compact_catalog_binding(cat, capsule, envelope, manifest[i])
        elif cat is not None:
            raise ValueError("ineligible world cannot carry eligible catalog")
        cref = end["catalog_reference"]
        if cref is not None:
            sha = _sha256(_world_bytes(cat))
            _strict_frozen(
                cref,
                {
                    "version": "ols-world-catalog/v1",
                    "locator": f"catalog-{sha}.json",
                    "sha256": sha,
                },
                "catalog reference",
            )
        unavailable = graph is None and end["generation_outcome"] != "failed"
        rows.append(
            {
                **projection,
                "evidence_unavailable": unavailable,
                "missing_evidence_status": (
                    "known_generation_completed_missing_evidence"
                    if end["generation_outcome"] == "completed"
                    else "generation_outcome_unknown"
                )
                if unavailable
                else None,
                "confirmed_generation_failed": end["generation_outcome"] == "failed",
                "stages": sorted(stages),
                "duration": end["finished_elapsed"] - start["started_elapsed"],
                "reasons": analysis["reasons"]
                if analysis is not None
                else [error["message"] for error in errors],
                "graph": graph,
                "analysis": analysis,
                "catalog": cat,
            }
        )
    return rows


def _report_rate(numerator, denominator, population, *, conditional=False):
    result = dict(
        wilson(numerator, denominator),
        population=population,
        unit="independently seeded world",
        interval="Wilson 95%; non-simultaneous",
    )
    result["availability"] = (
        "zero_denominator"
        if denominator == 0
        else "below_minimum_10"
        if conditional and denominator < 10
        else "available"
    )
    if result["availability"] != "available":
        result.update(rate=None, lower=None, upper=None)
    return result


def _outcome_rates(rows, *, conditional=False):
    e, f = sum(r["E"] for r in rows), sum(r["F"] for r in rows)
    return {
        "full_rank_over_E": _report_rate(
            f, e, "truth/identity eligible E", conditional=conditional
        ),
        "rank_deficient_over_E": _report_rate(
            sum(r["eligible_rank_deficient"] for r in rows),
            e,
            "truth/identity eligible E",
            conditional=conditional,
        ),
        "noisefree_recovered_over_F": _report_rate(
            sum(r["noisefree_recovered_over_F"] is True for r in rows),
            f,
            "full-rank eligible F",
            conditional=conditional,
        ),
        "paired_noisy_available_over_E": _report_rate(
            sum(r["paired_noisy_available_over_E"] is True for r in rows),
            e,
            "truth/identity eligible E",
            conditional=conditional,
        ),
    }


def _configured_frequencies(manifest, rows, label):
    def cell(name, predicate):
        selected = [i for i, a in enumerate(manifest) if predicate(a)]
        return {
            "group": name,
            "frequency": _report_rate(len(selected), len(manifest), label),
            "outcomes": _outcome_rates(
                [r for r in rows if r["attempt"] in selected], conditional=True
            ),
        }

    return {
        "T_marginal": [cell(t, lambda a, t=t: a["n_treatments"] == t) for t in range(1, 11)],
        "M_marginal": [cell(m, lambda a, m=m: a["n_covariates"] == m) for m in range(1, 11)],
        "exact_T_M_cells": [
            cell([t, m], lambda a, t=t, m=m: (a["n_treatments"], a["n_covariates"]) == (t, m))
            for t in range(1, 11)
            for m in range(1, 11)
        ],
        "nine_T_M_band_pairs": [
            cell(
                [list(t), list(m)],
                lambda a, t=t, m=m: (
                    t[0] <= a["n_treatments"] <= t[1] and m[0] <= a["n_covariates"] <= m[1]
                ),
            )
            for t in ((1, 3), (4, 7), (8, 10))
            for m in ((1, 3), (4, 7), (8, 10))
        ],
    }


def _quantiles(values, population):
    return {
        "n_worlds": len(values),
        "population": population,
        "unit": "one world-level scalar",
        "availability": "empty" if not values else "one_world" if len(values) == 1 else "available",
        "quantiles": None
        if not values
        else dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                strict=True,
            )
        ),
        "binomial_interval": None,
    }


def _descriptive_metrics(rows):
    eligible = [r for r in rows if r["E"]]
    full = [r for r in eligible if r["F"]]
    result = {
        "coefficient_identification": {
            "full_rank_worlds": len(full),
            "rank_deficient_nonidentified_worlds": len(eligible) - len(full),
            "rule": "coefficient errors only over F; no pooled coefficients/rows; no noisy recovery claim",
        }
    }
    for target in ("y0", "y1"):
        metrics = {}
        for family in ("beta", "rho_zy"):
            values = []
            for r in full:
                d = r["analysis"]["details"]
                errors = [
                    abs(e)
                    for name, e in zip(
                        d["coefficient_order"], d["fits"][target]["coefficient_error"], strict=True
                    )
                    if name.startswith(family + "[")
                ]
                if errors:
                    values.append(max(errors))
            metrics[family + "_world_max_abs_error"] = _quantiles(
                values, f"F with at least one direct {family} coefficient"
            )
        for key in ("residual_max_abs", "residual_rmse"):
            metrics[key] = _quantiles(
                [r["analysis"]["details"]["fits"][target][key] for r in eligible],
                "E; rank-deficient prediction residuals retained",
            )
        result[target] = metrics
    for key in ("raw_design", "standardized_design"):
        result[key + "_condition"] = _quantiles(
            [
                r["analysis"]["details"][key]["condition"]
                for r in eligible
                if r["analysis"]["details"][key]["condition"] is not None
            ],
            "E with finite condition; null excluded, not replaced",
        )
    return result


def _render_compact(evidence):
    if type(evidence) is not dict or set(evidence) != {
        "input",
        "calibration",
        "allocation",
        "ledger",
        "timing",
    }:
        raise ValueError("strict compact evidence required")
    cfg = _validate_report_input(evidence["input"])
    allocation = evidence["allocation"]
    _validate_report_calibration(evidence["calibration"], cfg)
    expected = allocate_from_calibration(
        cfg, allocation["inputs"]["elapsed_after_calibration"], evidence["calibration"]
    )
    _strict_frozen(allocation, expected, "calibration allocation")
    rows = _report_rows(cfg, evidence["input"], allocation, evidence["ledger"], evidence["timing"])
    ncap, a = len(allocation["manifest"]), len(rows)
    g, e, f, rd, u = (
        sum(r[k] for r in rows)
        for k in ("G", "E", "F", "eligible_rank_deficient", "evidence_unavailable")
    )
    partition = validate_accounting(
        cfg,
        ncap=ncap,
        attempted=a,
        generated=g,
        eligible=e,
        full_rank=f,
        rank_deficient=rd,
        evidence_unavailable=u,
    )
    if partition["generation_failed"] != sum(r["confirmed_generation_failed"] for r in rows):
        raise ValueError("started partition not conserved")
    generated = [r for r in rows if r["G"]]
    direct = {}
    for key in ("g_cy", "g_zy"):
        counts = [sum(r["graph"][key]) for r in generated]
        direct[key] = {
            "exact": [
                _report_rate(counts.count(k), g, "retained verified generation G")
                for k in range(11)
            ],
            "bins": [
                {
                    "range": [lo, hi],
                    "frequency": _report_rate(
                        sum(lo <= n <= hi for n in counts), g, "retained verified generation G"
                    ),
                }
                for lo, hi in ((0, 0), (1, 3), (4, 7), (8, 10))
            ],
        }
    patterns = {}
    # Predeclared edge-family presence patterns (eight bits), not dimension-specific
    # graph hashes. All 256 cells, including zeros, remain visible.
    keys = ["g_cy", "g_zy", "g_dy", "g_dc", "g_dz", "g_zc", "g_cc", "g_zz"]
    for bits in range(256):
        pattern = format(bits, "08b")
        n = sum(
            "".join(str(int(np.any(r["graph"][k]))) for k in keys) == pattern for r in generated
        )
        patterns[pattern] = _report_rate(
            n, g, "G conditional on retained verified generation; not all started worlds"
        )
    examples = []
    for selection in allocation["selection"]["band_examples"]:
        i = selection["attempt"]
        row = rows[i] if i is not None and i < a else None
        reason = (
            "band_absent_from_manifest"
            if i is None
            else "unattempted_budget"
            if row is None
            else row["state"]
            if not row["E"]
            else "available"
        )
        examples.append(
            {
                **selection,
                "availability_reason": reason,
                "details": row["analysis"]["details"] if row is not None and row["E"] else None,
                "graph": row["graph"] if row else None,
                "references": evidence["ledger"][i]["terminal"] if row else None,
                "catalog": row["catalog"] if row else None,
            }
        )
    elapsed = evidence["timing"]["elapsed_seconds"]
    return {
        "version": COMPACT_VERSION,
        "evidence": evidence,
        "bindings": {
            "config_digest": digest(cfg),
            "input_digest": evidence["input"]["digest"],
            "allocation_digest": digest(allocation),
            "selection_digest": digest(allocation["selection"]),
            "seed_manifest_digest": evidence["input"]["contract"]["seed_manifest_digest"],
            "catalog_schema_sha256": CATALOG_SCHEMA_SHA256,
            "ledger_digest": digest(evidence["ledger"]),
        },
        "counts": {
            "maximum": 1000,
            "Ncap": ncap,
            "A": a,
            "G": g,
            "E": e,
            "F": f,
            "R": sum(r["noisefree_recovered_over_F"] is True for r in rows),
            "rank_deficient": rd,
            "confirmed_generation_failed": partition.pop("generation_failed"),
            **partition,
        },
        "attempts": [
            {k: v for k, v in row.items() if k not in ("graph", "analysis", "catalog")}
            for row in rows
        ],
        "rates": _outcome_rates(rows),
        "failure_frequencies": {
            k: _report_rate(sum(r[k] for r in rows), a, "all started A")
            for k in ("confirmed_generation_failed", "evidence_unavailable")
        },
        "stage_failure_frequencies": {
            k: _report_rate(
                sum(k in r["stages"] for r in rows),
                a,
                "all started A; orthogonal stages may overlap",
            )
            for k in ("generation", "analysis", "persistence", "timeout")
        },
        "truth_failure_frequencies": {
            k: _report_rate(sum(r["state"] == k for r in rows), a, "all started A")
            for k in ("unsupported_truth", "identity_or_truth_ineligible")
        },
        "conditioning_frequencies": {
            k: _report_rate(
                sum(r["E"] and r["analysis"]["status"]["conditioning"] == k for r in rows),
                e,
                "truth/identity eligible E",
            )
            for k in ("easy", "moderate", "hard")
        },
        "conditioning_counts": {
            k: sum(r["E"] and r["analysis"]["status"]["conditioning"] == k for r in rows)
            for k in ("easy", "moderate", "hard")
        },
        "planned_configured": _configured_frequencies(
            allocation["manifest"],
            rows,
            "planned frozen Ncap; known configuration, not latent graph",
        ),
        "attempted_configured": _configured_frequencies(
            allocation["manifest"][:a], rows, "started A; known configuration, not latent graph"
        ),
        "direct_dimensions": direct,
        "graph_patterns": {"bit_order": keys, "frequencies": patterns},
        "descriptive": _descriptive_metrics(rows),
        "examples": examples,
        "precision": {
            "planned_400_adequate": ncap >= 400,
            "actual_eligible_N": e,
            "execution_stop": False,
            "subgroup_precision_certified": False,
        },
        "runtime": {
            **evidence["timing"],
            "deadline_seconds": 5400,
            "reserve_seconds": 300,
            "deadline_overrun_seconds": max(0, elapsed - 5400),
            "reserve_overrun_seconds": max(0, evidence["timing"]["finalization_seconds"] - 300),
            "attempt_durations": [r["duration"] for r in rows],
        },
        "limitations": _COMPACT_LIMITATIONS,
    }


def build_compact_summary(*, archive, input_contract, calibration, allocation, ledger, timing):
    """Read trusted durable refs; never generate worlds or accept inline fitted results."""
    cfg = _validate_report_input(input_contract)
    run, _, _ = _catalog_inputs(archive)
    for key in ("config", "source_commit", "source_hashes", "seed_manifest", "environment"):
        _strict_frozen(run[key], input_contract["contract"][key], "archive input binding")
    capsules = []
    if type(ledger) is not list:
        raise ValueError("ledger must be canonical started prefix")
    inventory = _report_inventory(archive, input_contract, allocation)
    _strict_frozen(
        ledger,
        [{"start": start[0], "terminal": end[0]} for start, end in inventory],
        "canonical durable ledger",
    )
    for item, (started, terminal) in zip(ledger, inventory, strict=True):
        start, end = started[1], terminal[1]
        graph = analysis = cat = graph_body = graph_node = None
        ref = end["world_reference"]
        if ref is not None:
            body = read_world(archive, ref)
            _strict_frozen(body["attempt"], start["attempt"], "world attempt")
            result = body["result"]
            resolved = resolved_generation_config(cfg, start["attempt"])
            _strict_frozen(
                result.get("generation"),
                {
                    "phase": "science",
                    "resolved_config": resolved,
                    "resolved_config_digest": digest(resolved),
                    "config_digest": digest(cfg),
                    "generation_seed": start["attempt"]["world_seed"],
                },
                "summary generation binding",
            )
            if (
                "source" in result
                and result["source"].get("seed") != start["attempt"]["world_seed"]
            ):
                raise ValueError("summary world seed mismatch")
            if "source" in result and "graph" in result["source"]:
                graph = _report_graph(result["source"]["graph"], start["attempt"])
                raw = _canonical_world_record(
                    _read_world_file(archive, ref["world_locator"], "world")
                )
                graph_node = raw["result"]["source"]["graph"]
            if "status" in result and "reasons" in result:
                analysis = _compact_analysis(result)
        if end["graph_reference"] is not None:
            graph_body = _report_file(archive, end["graph_reference"])
            partial_graph = _report_graph(graph_body["graph"], start["attempt"])
            if graph is not None:
                _strict_frozen(partial_graph, graph, "partial/full graph")
            graph = partial_graph
        if analysis is not None and graph is not None:
            projected = project_attempt_state(
                cfg,
                allocation["manifest"],
                start["attempt"],
                started=True,
                graph=graph,
                analysis=analysis,
                failure_stage="analysis_failed"
                if any(e["stage"] == "analysis" for e in end["errors"])
                else None,
            )
            if projected["E"]:
                # Accepted catalog validator recomputes raw science exactly.
                cat = catalog_entry(archive=archive, world_reference=ref)
        if end["catalog_reference"] is not None:
            _strict_frozen(
                read_catalog_entry(archive, end["catalog_reference"]), cat, "catalog binding"
            )
        for partial in end["partial_references"]:
            raw = _read_world_file(archive, partial["locator"], partial["kind"])
            if _sha256(raw) != partial["sha256"]:
                raise ValueError("partial evidence digest mismatch")
        capsules.append(
            {
                "start_reference": item["start"],
                "start": start,
                "terminal_reference": item["terminal"],
                "terminal": end,
                "graph": graph,
                "graph_capsule": graph_body,
                "graph_world_node": graph_node,
                "analysis": analysis,
                "catalog": cat,
            }
        )
    evidence = json.loads(
        canonical_json(
            {
                "input": input_contract,
                "calibration": calibration,
                "allocation": allocation,
                "ledger": capsules,
                "timing": timing,
            }
        )
    )
    return _render_compact(evidence)


def validate_compact_summary(summary):
    """Portable internal integrity/accounting only; raw arrays are not required."""
    try:
        expected = _render_compact(summary["evidence"])
        _strict_frozen(summary, expected, "compact summary")
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("malformed compact summary") from exc
    return json.loads(canonical_json(summary))


def read_compact_summary(archive, reference, *, raw_archive=None):
    """Reopen trusted compact bytes; optional raw verification is explicitly separate.

    ``archive`` is any directory containing the compact file (including a checkout). Stored runtime
    paths are provenance strings, never required to exist on the reader's machine.
    """
    if reference.get("kind") != "summary":
        raise ValueError("trusted summary reference required")
    summary = validate_compact_summary(_report_file(archive, reference, portable=True))
    if raw_archive is not None:
        evidence = summary["evidence"]
        rebuilt = build_compact_summary(
            archive=raw_archive,
            input_contract=evidence["input"],
            calibration=evidence["calibration"],
            allocation=evidence["allocation"],
            ledger=[
                {"start": c["start_reference"], "terminal": c["terminal_reference"]}
                for c in evidence["ledger"]
            ],
            timing=evidence["timing"],
        )
        _strict_frozen(summary, rebuilt, "optional raw science verification")
    return {
        "summary": summary,
        "validation": "raw_science_verified"
        if raw_archive is not None
        else "compact_integrity_only",
    }


# Execution receipts are deliberately separate from the frozen compact v3 contract.
# They record operational facts, not extra scientific classifications.
_EXECUTION_PYTHON = "/home/teemu/pymc-labs/prior-generator/.worktrees/issue-30-identifiable-linear-recovery/.venv/bin/python3"
_EXECUTION_VERSION = "ols-bounded-execution/v1"


def _execution_environment():
    import sys

    checkout = Path(__file__).absolute().parents[1]
    environment = runtime_input_environment()
    if sys.executable != _EXECUTION_PYTHON or not sys.dont_write_bytecode:
        raise ValueError("execution requires the prescribed interpreter with -B")
    required = dict.fromkeys(environment["environment"], "1")
    required["PYTHONPATH"] = str(checkout)
    _strict_frozen(environment["environment"], required, "execution environment")
    with _portable_report_parent(checkout):
        pass
    import subprocess

    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=checkout, text=True
    ).strip()
    if top != str(checkout) or Path.cwd().resolve() != checkout:
        raise ValueError("execution must consume the actual checkout from its root")
    return environment


def _execution_file(archive, name, payload=None):
    """Closed leaf namespace; every write is exclusive and file+directory fsynced."""
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9-]*\.(json|log)", name) is None:
        raise ValueError("invalid execution locator")
    with _world_parent(archive, "world-" + "0" * 64 + ".json", "world") as (parent, _):
        flags = (
            os.O_RDONLY | os.O_NONBLOCK if payload is None else os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        fd = os.open(name, flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "rb" if payload is None else "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("execution evidence must be regular")
            if payload is None:
                return stream.read()
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent)
    if _execution_file(archive, name) != payload:
        raise ValueError("execution write verification failed")
    return {"locator": name, "sha256": _sha256(payload)}


def _execution_record(archive, name, body):
    return _execution_file(archive, name, _world_bytes(body))


def _execution_read(archive, name):
    return _canonical_world_record(_execution_file(archive, name))


def _execution_optional(archive, name):
    try:
        return _execution_read(archive, name)
    except FileNotFoundError:
        return None


def _sync_archive(archive):
    """Flush existing evidence without following links or rewriting any bytes."""
    with _portable_report_parent(_world_archive(archive)) as (root, _):

        def sync(directory):
            for name in os.listdir(directory):
                mode = os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError("nonregular archive evidence")
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if stat.S_ISDIR(mode):
                    flags |= os.O_DIRECTORY
                fd = os.open(name, flags, dir_fd=directory)
                try:
                    if stat.S_ISDIR(mode):
                        sync(fd)
                    os.fsync(fd)
                finally:
                    os.close(fd)

        sync(root)
        os.fsync(root)


def _durable_report(archive, kind, body):
    raw = _world_bytes(body)
    sha = _sha256(raw)
    ref = {
        "version": _EVIDENCE_VERSION,
        "kind": kind,
        "locator": f"{kind}-{sha}.json",
        "sha256": sha,
    }
    _execution_file(archive, ref["locator"], raw)
    _strict_frozen(_report_file(archive, ref), body, "durable report")
    return ref


def _execution_sources():
    checkout = Path(__file__).absolute().parents[1]
    names = set(analysis_source_hashes())
    names.update(str(p.relative_to(checkout)) for p in (checkout / "pymc_generator").rglob("*.py"))
    # The imported generator must be this checkout, not an installed alternative.
    import importlib.util

    origin = importlib.util.find_spec("pymc_generator").origin
    if origin != str(checkout / "pymc_generator/__init__.py"):
        raise ValueError("generator import is not from the actual checkout")
    result = {}
    for name in sorted(names):
        path = checkout / name
        with _portable_report_parent(path.parent) as (parent, _):
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("source must be regular")
                raw = stream.read()
        result[name] = {"sha256": _sha256(raw), "base64": base64.b64encode(raw).decode("ascii")}
    return result


def _clock_identity():
    import socket

    return {
        "host": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def _execution_runtime():
    """Measured runtime/cache metadata, separate from the frozen scientific envelope."""
    import importlib.metadata
    import importlib.util
    import sys

    import pytensor
    from threadpoolctl import threadpool_info

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "sys_path": list(sys.path),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pytensor", "pymc", "pymc-marketing")
        },
        "package_origins": {
            name: importlib.util.find_spec(name).origin
            for name in ("numpy", "pytensor", "pymc", "pymc_generator")
        },
        "cache": {
            "pytensor_flags": os.environ.get("PYTENSOR_FLAGS"),
            "base_compiledir": str(pytensor.config.base_compiledir),
            "compiledir": str(pytensor.config.compiledir),
            "scope": "actual shared cache; no purge or cold-machine guarantee",
        },
        "threadpools": threadpool_info(),
        "thread_scope": "loaded parent pools at freeze; prescribed environment inherited by children",
    }


def _stable_execution_runtime(runtime):
    """Compare every measured field; only loaded-pool enumeration order is incidental."""
    pools = runtime["threadpools"]
    if any(type(pool.get("num_threads")) is not int or pool["num_threads"] != 1 for pool in pools):
        raise ValueError("runtime thread pools must have exactly one thread")
    return {**runtime, "threadpools": sorted(pools, key=canonical_json)}


def _freeze_execution(config, root, t0, *, clock=time.monotonic):
    """No archive writes until runtime, checkout, config and source checks succeed."""
    environment = _execution_environment()
    runtime = _execution_runtime()
    _stable_execution_runtime(runtime)
    cfg = validate_config(config)
    sources = _execution_sources()
    envelope = freeze_input_contract(
        cfg,
        environment=environment,
        source_hashes=analysis_source_hashes(),
        seed_manifest=study_schedules(cfg),
    )
    for name, sha in envelope["contract"]["source_hashes"].items():
        if sources[name]["sha256"] != sha:
            raise ValueError("source changed during freeze")
    _positive_time(t0, zero=True)
    if clock() < t0 or clock() >= t0 + 5100:
        raise ValueError("setup exhausted science deadline")
    execution = {
        "version": _EXECUTION_VERSION,
        "runtime": runtime,
        "input": envelope,
        "sources": sources,
        "t0": t0,
        "science_deadline": t0 + 5100,
        "final_deadline": t0 + 5400,
        "clock_identity": _clock_identity(),
        "checkout": str(Path(__file__).absolute().parents[1]),
    }
    archive, _ = create_run_archive(
        root,
        cfg,
        environment=environment,
        source_hashes=envelope["contract"]["source_hashes"],
        seed_manifest=envelope["contract"]["seed_manifest"],
    )
    _execution_record(archive, "execution.json", execution)
    _sync_archive(archive)
    _verify_execution(archive, clock=clock)
    return archive, execution


def _verify_execution(archive, *, clock=time.monotonic):
    execution = _execution_read(archive, "execution.json")
    if set(execution) != {
        "version",
        "runtime",
        "input",
        "sources",
        "t0",
        "science_deadline",
        "final_deadline",
        "clock_identity",
        "checkout",
    }:
        raise ValueError("invalid execution record")
    if execution["version"] != _EXECUTION_VERSION:
        raise ValueError("unsupported execution version")
    _strict_frozen(execution["clock_identity"], _clock_identity(), "original monotonic clock")
    _strict_frozen(
        execution["checkout"], str(Path(__file__).absolute().parents[1]), "original checkout"
    )
    _strict_frozen(
        execution["input"]["contract"]["environment"], _execution_environment(), "original runtime"
    )
    _strict_frozen(execution["sources"], _execution_sources(), "frozen source bytes")
    _strict_frozen(
        _stable_execution_runtime(execution["runtime"]),
        _stable_execution_runtime(_execution_runtime()),
        "original measured runtime",
    )
    _positive_time(execution["t0"], zero=True)
    if (
        execution["science_deadline"] != execution["t0"] + 5100
        or execution["final_deadline"] != execution["t0"] + 5400
        or clock() < execution["t0"]
    ):
        raise ValueError("original clock/deadline mismatch")
    cfg = _validate_report_input(execution["input"])
    run, _, _ = _catalog_inputs(archive)
    for key in ("config", "source_commit", "source_hashes", "environment", "seed_manifest"):
        _strict_frozen(run[key], execution["input"]["contract"][key], "frozen archive binding")
    if run["source_commit"] != _source_commit():
        raise ValueError("source commit drift")
    for name, sha in run["source_hashes"].items():
        if execution["sources"][name]["sha256"] != sha:
            raise ValueError("source envelope drift")
    validate_config(cfg)
    return execution


@contextmanager
def _execution_lock(archive):
    import fcntl

    with _portable_report_parent(_world_archive(archive)) as (fd, _):
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _process_identity(pid):
    # Field 22 survives PID recycling; split after comm, which can contain spaces.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


@contextmanager
def _owned_subreaper():
    """Linux, process-local only: never adopts an already orphaned non-child."""
    import ctypes
    import sys

    if sys.platform != "linux":
        raise RuntimeError("owned cleanup requires Linux subreaper and pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise RuntimeError("Linux PR_GET_CHILD_SUBREAPER unavailable")
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Linux PR_SET_CHILD_SUBREAPER unavailable")
    try:
        if not hasattr(libc, "pidfd_open") or not hasattr(libc, "pidfd_send_signal"):
            raise RuntimeError("Linux libc pidfd support unavailable")
        fd = libc.pidfd_open(os.getpid(), 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "Linux pidfd_open feature check failed")
        try:
            if libc.pidfd_send_signal(fd, 0, None, 0) != 0:
                raise OSError(ctypes.get_errno(), "Linux pidfd signal feature check failed")
        finally:
            os.close(fd)
        yield
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise RuntimeError("cannot restore process-local subreaper state")


def _owned_stat(pid, *, parent=None):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        # PPID proves a discovery edge, not identity: trusted children can reparent.
        if parent is not None and int(fields[1]) != parent:
            return None
        return {"pid": pid, "identity": fields[19], "pgid": int(fields[2]), "sid": int(fields[3])}
    except FileNotFoundError:
        return None


def _owned_children(pid):
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def _owned_collect(owned, leader, *, adopted=False):
    """Only tree edges, or adopted direct children still in the original session/group."""
    current = _owned_stat(leader["pid"])
    if current is not None and current != leader:
        raise RuntimeError("owned leader PID/PGID identity reused; refusing signals")
    pending = list(owned)
    if adopted:
        for pid in _owned_children(os.getpid()):
            item = _owned_stat(pid, parent=os.getpid())
            if item and item["pgid"] == leader["pgid"] and item["sid"] == leader["sid"]:
                owned.setdefault(pid, item)
                pending.append(pid)
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if _owned_stat(pid) != owned[pid]:
            continue
        for child in _owned_children(pid):
            item = _owned_stat(child, parent=pid)
            if item is not None and _owned_stat(pid) == owned[pid]:
                owned.setdefault(child, item)
                pending.append(child)


def _owned_signal(item, sig):
    import ctypes
    import errno

    current = _owned_stat(item["pid"])
    if current is None:
        return
    if current != item:
        raise RuntimeError("owned PID identity changed; refusing signal")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.pidfd_open(item["pid"], 0)
    if fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return
        raise OSError(error, "owned pidfd_open failed")
    try:
        if _owned_stat(item["pid"]) != item:
            raise RuntimeError("owned PID identity changed; refusing signal")
        if libc.pidfd_send_signal(fd, sig, None, 0) != 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "owned pidfd_send_signal failed")
    finally:
        os.close(fd)


def _kill_group(process, *, leader=None, adopted=False):
    """Bounded TERM/KILL and exact-PID reaping, never waitpid(-1) or killpg.

    Escaped, unobserved arbitrary daemons are not claimed. A non-child zombie
    cannot be reaped here: failure to disappear is an explicit cleanup failure.
    """
    import signal

    leader = leader or (_owned_stat(process.pid) if process is not None else None)
    if leader is None:
        # A caller without a retained identity cannot establish ownership.
        raise RuntimeError("unresolved owned cleanup: leader identity unavailable")
    owned = {leader["pid"]: leader}
    statuses = {}
    began = time.monotonic()
    for sig, until in ((signal.SIGTERM, began + 0.2), (signal.SIGKILL, began + 2.0)):
        sent = set()
        while True:
            _owned_collect(owned, leader, adopted=adopted)
            for pid, item in list(owned.items()):
                current = _owned_stat(pid)
                if current is None:
                    continue
                if current != item:
                    raise RuntimeError("owned PID identity reused; refusing cleanup")
                if pid not in sent:
                    _owned_signal(item, sig)
                    sent.add(pid)
                # Collect descendants before reaping their parent, including zombies.
                try:
                    waited, status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    waited = 0
                if waited:
                    statuses[pid] = os.waitstatus_to_exitcode(status)
                    if process is not None and pid == process.pid:
                        process.returncode = statuses[pid]
            if all(_owned_stat(pid) is None for pid in owned):
                # A dying parent may fork after the earlier snapshot. Only after
                # it disappears is its final adopted child list authoritative.
                _owned_collect(owned, leader, adopted=adopted)
                if all(_owned_stat(pid) is None for pid in owned):
                    return {
                        "owned": list(owned.values()),
                        "wait_statuses": statuses,
                        "all_disappeared": True,
                        "scope": "tracked tree and adopted original group; not arbitrary escaped daemons",
                        "seconds": time.monotonic() - began,
                    }
            if time.monotonic() >= until:
                break
            time.sleep(0.01)
    raise RuntimeError("unresolved owned cleanup: /proc leaves remain after bounded TERM/KILL")


def _process_streams(archive, tag):
    layout = _execution_optional(archive, f"stream-layout-{tag}.json")
    if layout is not None:
        _strict_frozen(
            layout,
            {"tag": tag, "stdout": f"child-{tag}.log", "stderr": f"child-{tag}-stderr.log"},
            "stream layout receipt",
        )
    streams = {"layout": "separate" if layout is not None else "unknown"}
    for stream, name in (("stdout", f"child-{tag}.log"), ("stderr", f"child-{tag}-stderr.log")):
        try:
            raw = _execution_file(archive, name)
        except FileNotFoundError:
            streams[stream] = None
        else:
            streams[stream] = {"locator": name, "sha256": _sha256(raw), "bytes": len(raw)}
    if layout is None:
        # A receiptless log does not establish whether old parents merged streams.
        streams.update(unclassified_stdout=streams["stdout"], unclassified_stderr=streams["stderr"])
        streams["stdout"] = streams["stderr"] = None
    return streams


def _retain_process_result(archive, tag, result):
    name = f"process-result-{tag}.json"
    retained = _execution_optional(archive, name)
    if retained is None:
        _execution_record(archive, name, result)
    else:
        _strict_frozen(retained, result, "retained process result")
    return result


def _recover_process(archive, tag, *, clock=time.monotonic):
    """Reap a surviving recorded child; never guess an unobserved wait status."""
    retained = _execution_optional(archive, f"process-result-{tag}.json")
    process = _execution_optional(archive, f"process-{tag}.json")
    returncode = None
    if process is not None:
        pid = process["pid"]
        identity = _process_identity(pid)
        if identity is not None and identity != process["identity"]:
            raise RuntimeError("owned leader PID identity reused; refusing recovery")
        if identity is not None:
            leader = _owned_stat(pid)
            if (
                leader is None
                or leader["identity"] != process["identity"]
                or leader["pgid"] != pid
                or leader["sid"] != pid
            ):
                raise RuntimeError("owned recovery leader identity/group changed; refusing signals")
            cleanup_deadline = time.monotonic() + 2.0
            with _owned_subreaper():
                cleanup = _kill_group(None, leader=leader)
            # A fresh reader cannot adopt another parent's children. A child
            # forked during TERM may have escaped the tracked snapshot; inspect
            # the original group/session read-only, using the remaining budget.
            while True:
                remaining = []
                for entry in Path("/proc").iterdir():
                    if time.monotonic() >= cleanup_deadline:
                        raise RuntimeError(
                            "unresolved owned cleanup: recovery disappearance unknown"
                        )
                    if entry.name.isdecimal():
                        item = _owned_stat(int(entry.name))
                        if item and (item["pgid"] == pid or item["sid"] == pid):
                            remaining.append(item)
                if not remaining:
                    break
                if time.monotonic() >= cleanup_deadline:
                    raise RuntimeError("unresolved owned cleanup: recovery group/session remains")
                time.sleep(min(0.01, max(0.0, cleanup_deadline - time.monotonic())))
            returncode = cleanup["wait_statuses"].get(pid)
        else:
            # No persisted descendant identities in legacy receipts. Inspect only;
            # never guess that a recycled PGID denotes the original owned group.
            for entry in Path("/proc").iterdir():
                if entry.name.isdecimal():
                    item = _owned_stat(int(entry.name))
                    if item and (item["pgid"] == pid or item["sid"] == pid):
                        raise RuntimeError(
                            "unresolved owned cleanup: legacy leader gone without descendant identities"
                        )
    if retained is not None:
        if "layout" in retained.get("streams", {}):
            _strict_frozen(
                retained["streams"], _process_streams(archive, tag), "retained stream layout"
            )
        for stream, ref in retained.get("streams", {}).items():
            if stream != "layout" and ref is not None:
                raw = _execution_file(archive, ref["locator"])
                if _sha256(raw) != ref["sha256"] or len(raw) != ref["bytes"]:
                    raise ValueError("retained process stream mismatch")
        return retained
    # Backward compatibility: old parents published actual exits in cost records.
    cost = _execution_optional(archive, f"cost-{tag}.json")
    if cost is not None:
        return _retain_process_result(archive, tag, cost["process"])
    result = {
        "returncode": returncode,
        "exit_unknown_reason": "parent died before durable wait status"
        if returncode is None
        else None,
        "timed_out": False,
        "launched": process is not None,
        "started": process["started"] if process else None,
        "finished": clock(),
        "pid": process["pid"] if process else None,
        "resources": None,
        "resource_unknown_reason": "original parent resource measurement unavailable",
        "streams": _process_streams(archive, tag),
    }
    _sync_archive(archive)
    return _retain_process_result(archive, tag, result)


def _timed_execution_child(archive, request, deadline, *, clock=time.monotonic):
    """Gate computation on a durable PID receipt; retain exit and both raw streams."""
    import resource
    import subprocess

    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    tag = request["tag"]
    before = clock()
    if before >= deadline:
        return _retain_process_result(
            archive,
            tag,
            {
                "returncode": None,
                "exit_unknown_reason": "not launched: original deadline exhausted",
                "timed_out": True,
                "launched": False,
                "started": before,
                "finished": clock(),
                "pid": None,
                "resources": None,
                "streams": {"stdout": None, "stderr": None},
            },
        )
    _execution_record(
        archive,
        f"stream-layout-{tag}.json",
        {"tag": tag, "stdout": f"child-{tag}.log", "stderr": f"child-{tag}-stderr.log"},
    )
    process = None
    timed_out = False
    with (
        _owned_subreaper(),
        _world_parent(archive, "world-" + "0" * 64 + ".json", "world") as (parent, _),
    ):
        # Open each exclusively before Popen. A crash here leaves retained, versioned logs.
        with os.fdopen(
            os.open(
                f"child-{tag}.log",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            ),
            "wb",
        ) as stdout:
            with os.fdopen(
                os.open(
                    f"child-{tag}-stderr.log",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                ),
                "wb",
            ) as stderr:
                try:
                    process = subprocess.Popen(
                        [_EXECUTION_PYTHON, "-B", str(Path(__file__).absolute()), "_child"],
                        cwd=Path(__file__).absolute().parents[1],
                        env=os.environ.copy(),
                        stdin=subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    leader = _owned_stat(process.pid)
                    _execution_record(
                        archive,
                        f"process-{tag}.json",
                        {
                            "pid": process.pid,
                            "identity": _process_identity(process.pid),
                            "request": request,
                            "started": before,
                            "deadline": deadline,
                        },
                    )
                    remaining = deadline - clock()
                    if remaining <= 0:
                        timed_out = True
                    else:
                        try:
                            process.communicate(
                                _world_bytes({"archive": str(archive), **request}),
                                timeout=remaining,
                            )
                        except subprocess.TimeoutExpired:
                            timed_out = True
                finally:
                    if process is not None:
                        cleanup = _kill_group(process, leader=leader, adopted=True)
                        if process.stdin is not None:
                            process.stdin.close()
                    for stream in (stdout, stderr):
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.fsync(parent)
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    result = {
                        "returncode": process.returncode if process else None,
                        "exit_unknown_reason": None if process else "Popen did not return a child",
                        "timed_out": timed_out,
                        "launched": process is not None,
                        "started": before,
                        "finished": clock(),
                        "pid": process.pid if process else None,
                        "cleanup": {
                            **cleanup,
                            "deadline_overrun_seconds": max(0.0, clock() - deadline),
                        }
                        if process
                        else None,
                        "resources": {
                            "user_seconds": usage.ru_utime - usage_before.ru_utime,
                            "system_seconds": usage.ru_stime - usage_before.ru_stime,
                            "children_max_rss_kib": usage.ru_maxrss,
                            "rss_scope": "parent cumulative child high-water mark, not per-world attribution",
                        },
                        "streams": _process_streams(archive, tag),
                    }
                    _retain_process_result(archive, tag, result)
    return result


def _execution_error(stage, exc):
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc) or type(exc).__name__}


def _attempt_start(archive, execution, allocation, attempt, phase, *, clock=time.monotonic):
    import socket

    start = {
        "attempt": attempt,
        "phase": phase,
        "input_digest": execution["input"]["digest"],
        "allocation_digest": digest(allocation) if phase == "science" else None,
        "started_elapsed": clock() - execution["t0"],
        "resource": {"host": socket.gethostname(), "pid": os.getpid()},
    }
    if phase == "science":
        _validate_report_start(start, attempt, execution["input"], allocation)
        ref = _durable_report(archive, "start", start)
    else:
        ref = _execution_record(archive, f"calibration-start-{attempt['attempt']}.json", start)
    return ref, start


def _load_attempt_start(archive, execution, request):
    phase = request["phase"]
    ref = request["start"]
    if phase == "science":
        start = _report_file(archive, ref)
        allocation = _execution_read(archive, "allocation.json")["allocation"]
        _validate_report_start(start, start["attempt"], execution["input"], allocation)
        i = start["attempt"]["attempt"]
        _strict_frozen(start["attempt"], allocation["manifest"][i], "allocated child attempt")
    elif phase == "calibration":
        raw = _execution_file(archive, ref["locator"])
        if _sha256(raw) != ref["sha256"]:
            raise ValueError("calibration start digest mismatch")
        start = _canonical_world_record(raw)
        if (
            ref["locator"] != f"calibration-start-{start['attempt']['attempt']}.json"
            or start["allocation_digest"] is not None
        ):
            raise ValueError("invalid calibration start")
    else:
        raise ValueError("invalid child phase")
    _scheduled_attempt(execution["input"]["contract"]["config"], start["attempt"], phase)
    if (
        start["phase"] != phase
        or start["input_digest"] != execution["input"]["digest"]
        or request["tag"] != str(start["attempt"]["attempt"])
    ):
        raise ValueError("child cannot alter frozen inputs")
    return start


def _attempt_child(archive, execution, request):
    """Internal worker: derive every numerical input from immutable recorded inputs."""
    start = _load_attempt_start(archive, execution, request)
    attempt, phase = start["attempt"], start["phase"]
    cfg, tag = execution["input"]["contract"]["config"], request["tag"]
    stage = "generation"
    try:
        if time.monotonic() >= execution["science_deadline"]:
            raise TimeoutError("science cutoff before generation")
        world = generate_configured_world(cfg, attempt, phase=phase)
        stage = "persistence"
        # Completion survives even if minimal graph persistence itself fails.
        _execution_record(
            archive,
            f"generated-{tag}.json",
            {"attempt": attempt, "phase": phase, "start_sha256": request["start"]["sha256"]},
        )
        graph = _report_graph(world.g, attempt)
        if phase == "science":
            _durable_report(archive, "graph", {"attempt": attempt, "phase": phase, "graph": graph})
        else:
            _execution_record(
                archive, f"calibration-graph-{tag}.json", {"attempt": attempt, "graph": graph}
            )
        stage = "analysis"
        result = analyze_configured_world(world, cfg, attempt, phase=phase)
        stage = "persistence"
        reference = persist_world(archive, attempt, result)
        _sync_archive(archive)
        _execution_record(archive, f"world-reference-{tag}.json", reference)
        read_world(archive, reference)
        if phase == "science" and result["status"]["truth_eligibility"] == "eligible":
            cat = catalog_entry(archive=archive, world_reference=reference)
            cref = persist_catalog_entry(archive, cat)
            _sync_archive(archive)
            _execution_record(archive, f"catalog-reference-{tag}.json", cref)
            read_catalog_entry(archive, cref)
        _execution_record(
            archive, f"completed-{tag}.json", {"start_sha256": request["start"]["sha256"]}
        )
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            stage = "timeout"
        _execution_record(
            archive,
            f"error-{tag}.json",
            {"start_sha256": request["start"]["sha256"], "error": _execution_error(stage, exc)},
        )
        return 1
    return 0


def _attempt_evidence(archive, execution, start_reference, start):
    """Recover only observed, readable evidence; never regenerate a missing world."""
    attempt, phase = start["attempt"], start["phase"]
    tag = str(attempt["attempt"])
    errors, partials = [], []
    error = _execution_optional(archive, f"error-{tag}.json")
    if error is not None:
        _strict_frozen(set(error), {"start_sha256", "error"}, "child error fields")
        if error["start_sha256"] != start_reference["sha256"]:
            raise ValueError("child error start mismatch")
        errors.append(error["error"])
    completed = _execution_optional(archive, f"completed-{tag}.json")
    if completed is not None:
        _strict_frozen(completed, {"start_sha256": start_reference["sha256"]}, "child completion")
    generated = _execution_optional(archive, f"generated-{tag}.json")
    if generated is not None:
        _strict_frozen(
            generated,
            {"attempt": attempt, "phase": phase, "start_sha256": start_reference["sha256"]},
            "generation receipt",
        )
    graph_ref = None
    if phase == "science":
        with _portable_report_parent(archive) as (fd, _):
            names = [
                name
                for name in os.listdir(fd)
                if name.startswith("graph-") and name.endswith(".json")
            ]
        for name in names:
            sha = name.removeprefix("graph-").removesuffix(".json")
            ref = {"version": _EVIDENCE_VERSION, "kind": "graph", "locator": name, "sha256": sha}
            body = _report_file(archive, ref)
            if body.get("attempt") == attempt:
                _strict_frozen(
                    body,
                    {
                        "attempt": attempt,
                        "phase": phase,
                        "graph": _report_graph(body["graph"], attempt),
                    },
                    "retained graph",
                )
                if graph_ref is not None:
                    raise ValueError("duplicate graph evidence")
                graph_ref = ref
    world_ref = cat_ref = None
    locator = f"attempt-{_logical_attempt_id(attempt)}.json"
    try:
        raw = _read_world_file(archive, locator, "reservation")
    except FileNotFoundError:
        raw = None
    if raw is not None:
        partials.append({"kind": "reservation", "locator": locator, "sha256": _sha256(raw)})
        try:
            ref = {**_canonical_world_record(raw), "reservation_sha256": _sha256(raw)}
            _world_reference_shape(ref, attempt)
            body = read_world(archive, ref)
            _strict_frozen(body["attempt"], attempt, "recovered attempt")
            result = body["result"]
            resolved = resolved_generation_config(
                execution["input"]["contract"]["config"], attempt, phase=phase
            )
            _strict_frozen(
                result.get("generation"),
                {
                    "phase": phase,
                    "resolved_config": resolved,
                    "resolved_config_digest": digest(resolved),
                    "config_digest": execution["input"]["contract"]["config_digest"],
                    "generation_seed": attempt["world_seed"],
                },
                "recovered generation",
            )
            if phase == "science" and result["status"]["truth_eligibility"] == "eligible":
                catalog_entry(archive=archive, world_reference=ref)
            world_ref = ref
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(_execution_error("persistence", exc))
        # Retain readable partial raw world/blob references even if decoding failed.
        try:
            reserved = _canonical_world_record(raw)
            payload = _read_world_file(archive, reserved["world_locator"], "world")
            partials.append(
                {"kind": "world", "locator": reserved["world_locator"], "sha256": _sha256(payload)}
            )

            def collect(node):
                if isinstance(node, dict):
                    if "__ndarray_ref__" in node or "__numpy_scalar__" in node:
                        aref = next(iter(node.values()))
                        blob = _read_world_file(archive, aref["locator"], "array")
                        item = {
                            "kind": "array",
                            "locator": aref["locator"],
                            "sha256": _sha256(blob),
                        }
                        if item not in partials:
                            partials.append(item)
                    else:
                        for value in node.values():
                            collect(value)
                elif isinstance(node, list):
                    for value in node:
                        collect(value)

            collect(_canonical_world_record(payload))
        except (ValueError, KeyError, OSError, TypeError):
            pass  # Raw files remain in place; do not fabricate references for unreadable bytes.
    candidate = _execution_optional(archive, f"catalog-reference-{tag}.json")
    if candidate is not None and world_ref is not None:
        try:
            read_catalog_entry(archive, candidate)
            cat_ref = candidate
        except (ValueError, OSError) as exc:
            errors.append(_execution_error("persistence", exc))
    outcome = (
        "completed"
        if generated is not None or graph_ref is not None or world_ref is not None
        else "failed"
        if any(e["stage"] == "generation" for e in errors)
        else "unknown"
    )
    if outcome != "failed" and any(e["stage"] == "generation" for e in errors):
        raise ValueError("contradictory generation evidence")
    return {
        "generation_outcome": outcome,
        "errors": errors,
        "world_reference": world_ref,
        "catalog_reference": cat_ref,
        "graph_reference": graph_ref,
        "partial_references": partials,
    }, completed is not None


def _finish_attempt(archive, execution, start_ref, start, process, *, clock=time.monotonic):
    tag = str(start["attempt"]["attempt"])
    process = _retain_process_result(archive, tag, process)
    retained = _execution_optional(archive, f"cost-{tag}.json")
    if retained is not None and "terminal_body" in retained:
        terminal, ref = retained["terminal_body"], retained["terminal"]
        sha = _sha256(_world_bytes(terminal))
        expected_locator = (
            f"terminal-{sha}.json"
            if start["phase"] == "science"
            else f"calibration-terminal-{tag}.json"
        )
        if ref["sha256"] != sha or ref["locator"] != expected_locator:
            raise ValueError("retained terminal reference mismatch")
        _strict_frozen(retained["process"], process, "retained cost process")
        _validate_report_terminal({**terminal, "phase": "science"}, start["attempt"], start_ref)
        try:
            raw = _execution_file(archive, ref["locator"])
        except FileNotFoundError:
            _execution_file(archive, ref["locator"], _world_bytes(terminal))
        else:
            _strict_frozen(_canonical_world_record(raw), terminal, "retained terminal")
        return ref, retained
    evidence, complete = _attempt_evidence(archive, execution, start_ref, start)
    if process["timed_out"]:
        evidence["errors"].append(
            {
                "stage": "timeout",
                "type": "DeadlineExceeded",
                "message": "child killed at actual remaining science cutoff",
            }
        )
    elif process["returncode"] not in (0, None) and not evidence["errors"]:
        evidence["errors"].append(
            {
                "stage": "persistence",
                "type": "ChildExit",
                "message": f"child exit {process['returncode']}; retained evidence reconciled",
            }
        )
    terminal = {
        "attempt": start["attempt"],
        "phase": start["phase"],
        "start_sha256": start_ref["sha256"],
        "finished_elapsed": clock() - execution["t0"],
        "status": "recovered_unknown"
        if evidence["generation_outcome"] == "unknown"
        else "complete",
        **evidence,
    }
    raw = _world_bytes(terminal)
    sha = _sha256(raw)
    if start["phase"] == "science":
        _validate_report_terminal(terminal, start["attempt"], start_ref)
        ref = {
            "version": _EVIDENCE_VERSION,
            "kind": "terminal",
            "locator": f"terminal-{sha}.json",
            "sha256": sha,
        }
    else:
        ref = {"locator": f"calibration-terminal-{tag}.json", "sha256": sha}
    duration = clock() - execution["t0"] - start["started_elapsed"]
    cost = {
        "process": process,
        "duration": duration,
        "complete": complete
        and process["returncode"] == 0
        and not evidence["errors"]
        and evidence["world_reference"] is not None,
        "terminal": ref,
        "terminal_body": terminal,
        "science_overrun_seconds": max(0, clock() - execution["science_deadline"]),
    }
    _execution_record(archive, f"cost-{tag}.json", cost)
    # Cost, real exit/resources, streams and replayable terminal are durable first.
    _execution_file(archive, ref["locator"], raw)
    return ref, cost


def _run_attempt(
    archive,
    execution,
    allocation,
    attempt,
    phase,
    *,
    clock=time.monotonic,
    launch=_timed_execution_child,
):
    start_ref, start = _attempt_start(archive, execution, allocation, attempt, phase, clock=clock)
    request = {
        "mode": "attempt",
        "tag": str(attempt["attempt"]),
        "phase": phase,
        "start": start_ref,
    }
    process = launch(archive, request, execution["science_deadline"], clock=clock)
    return _finish_attempt(archive, execution, start_ref, start, process, clock=clock)


def _existing_terminal_provenance(
    archive, execution, start_ref=None, start=None, terminal_ref=None, terminal=None, *, write=False
):
    """Preflight every sidecar once; return a recovery-local terminal reconciler."""
    unknown = {
        "returncode": None,
        "exit_unknown_reason": "existing terminal has no retained wait status",
        "resources": None,
        "resource_unknown_reason": "existing terminal has no retained resource measurement",
    }
    # Only this closure owns the index: no unchecked caller-supplied mapping or
    # cross-recovery cache. Store bytes so returned records cannot mutate it.
    receipts = {}

    def reconcile(start_ref, start, terminal_ref, terminal, *, write=False):
        tag = str(start["attempt"]["attempt"])
        name = f"process-provenance-{tag}.json"
        refs, bodies = {}, {}
        for field, locator in (
            ("process_result", f"process-result-{tag}.json"),
            ("cost", f"cost-{tag}.json"),
        ):
            body = _execution_optional(archive, locator)
            bodies[field] = body
            refs[field] = (
                None
                if body is None
                else {"locator": locator, "sha256": _sha256(_execution_file(archive, locator))}
            )
        cost, process = bodies["cost"], bodies["process_result"]
        if cost is not None:
            _strict_frozen(cost["terminal"], terminal_ref, "cost terminal binding")
            if "terminal_body" in cost:
                _strict_frozen(cost["terminal_body"], terminal, "cost terminal bytes")
            if process is not None:
                _strict_frozen(cost["process"], process, "cost process binding")
            else:
                process = cost["process"]
        if process is None:
            process = {**unknown, "streams": _process_streams(archive, tag)}
        streams = process.get("streams", {})
        for stream, ref in streams.items():
            if stream == "layout" or ref is None:
                continue
            expected = (
                f"child-{tag}-stderr.log"
                if stream in ("stderr", "unclassified_stderr")
                else f"child-{tag}.log"
            )
            if (
                stream
                not in (
                    "stdout",
                    "stderr",
                    "legacy_combined",
                    "unclassified_stdout",
                    "unclassified_stderr",
                )
                or ref["locator"] != expected
            ):
                raise ValueError("invalid process stream identity")
            raw = _execution_file(archive, expected)
            _strict_frozen(
                ref,
                {"locator": expected, "sha256": _sha256(raw), "bytes": len(raw)},
                "process stream bytes",
            )
        if "layout" in streams:
            _strict_frozen(streams, _process_streams(archive, tag), "process stream layout")
        expected = {
            "tag": tag,
            "input_digest": execution["input"]["digest"],
            "start": start_ref,
            "terminal": terminal_ref,
            **refs,
            "process": process,
        }
        raw = _world_bytes(expected)
        if tag in receipts:
            if receipts[tag] != raw:
                raise ValueError("existing terminal process provenance mismatch")
        elif write:
            _execution_record(archive, name, expected)
            receipts[tag] = raw
        return expected

    with _portable_report_parent(archive) as (fd, _):
        names = sorted(n for n in os.listdir(fd) if n.startswith("process-provenance-"))
    scheduled = {
        str(attempt["attempt"]): (phase, attempt)
        for phase, attempts in execution["input"]["contract"]["seed_manifest"].items()
        for attempt in attempts
        if phase in ("science", "calibration")
    }
    for candidate in names:
        raw = _execution_file(archive, candidate)
        receipt = _canonical_world_record(raw)
        if set(receipt) != {
            "tag",
            "input_digest",
            "start",
            "terminal",
            "process_result",
            "cost",
            "process",
        }:
            raise ValueError("invalid process provenance")
        tag = receipt["tag"]
        if (
            type(tag) is not str
            or candidate != f"process-provenance-{tag}.json"
            or tag in receipts
            or tag not in scheduled
        ):
            raise ValueError("duplicate or invalid process provenance")
        phase, attempt = scheduled[tag]
        records = {}
        for field in ("start", "terminal"):
            ref = receipt[field]
            if type(ref) is not dict or not {"locator", "sha256"} <= set(ref):
                raise ValueError("invalid process provenance source")
            source_raw = _execution_file(archive, ref["locator"])
            sha = _sha256(source_raw)
            expected_ref = (
                {
                    "version": _EVIDENCE_VERSION,
                    "kind": field,
                    "locator": f"{field}-{sha}.json",
                    "sha256": sha,
                }
                if phase == "science"
                else {"locator": f"calibration-{field}-{tag}.json", "sha256": sha}
            )
            _strict_frozen(ref, expected_ref, "process provenance source")
            record = _canonical_world_record(source_raw)
            _strict_frozen(record["attempt"], attempt, "process provenance attempt")
            if record["phase"] != phase:
                raise ValueError("process provenance phase mismatch")
            records[field] = record
        _strict_frozen(
            records["start"]["input_digest"], execution["input"]["digest"], "provenance input"
        )
        _validate_report_terminal(
            {**records["terminal"], "phase": "science"}, attempt, receipt["start"]
        )
        receipts[tag] = raw
        reconcile(receipt["start"], records["start"], receipt["terminal"], records["terminal"])
    if start_ref is not None:
        return reconcile(start_ref, start, terminal_ref, terminal, write=write)
    return reconcile


def _reconcile_calibration(archive, execution, *, clock=time.monotonic, _provenance=None):
    """Retain a started calibration even when launch/parent failed before its return."""
    provenance = _provenance or _existing_terminal_provenance(archive, execution)
    with _portable_report_parent(archive) as (fd, _):
        names = set(os.listdir(fd))
    scheduled = execution["input"]["contract"]["seed_manifest"]["calibration"]
    start_names = {n for n in names if n.startswith("calibration-start-")}
    terminal_names = {n for n in names if n.startswith("calibration-terminal-")}
    expected_starts = {
        f"calibration-start-{a['attempt']}.json" for a in scheduled[: len(start_names)]
    }
    if start_names != expected_starts or terminal_names - {
        n.replace("-start-", "-terminal-") for n in start_names
    }:
        raise ValueError("nonprefix calibration starts or orphan terminal")
    for attempt in scheduled[: len(start_names)]:
        tag = str(attempt["attempt"])
        locator = f"calibration-start-{tag}.json"
        ref = {"locator": locator, "sha256": _sha256(_execution_file(archive, locator))}
        start = _load_attempt_start(
            archive, execution, {"phase": "calibration", "start": ref, "tag": tag}
        )
        terminal = _execution_optional(archive, f"calibration-terminal-{tag}.json")
        if terminal is not None:
            _validate_report_terminal({**terminal, "phase": "science"}, attempt, ref)
            terminal_ref = {
                "locator": f"calibration-terminal-{tag}.json",
                "sha256": _sha256(_world_bytes(terminal)),
            }
            provenance(ref, start, terminal_ref, terminal)
    records = []
    for attempt in execution["input"]["contract"]["seed_manifest"]["calibration"]:
        tag = str(attempt["attempt"])
        start = _execution_optional(archive, f"calibration-start-{tag}.json")
        if start is None:
            break
        raw = _execution_file(archive, f"calibration-start-{tag}.json")
        ref = {"locator": f"calibration-start-{tag}.json", "sha256": _sha256(raw)}
        _load_attempt_start(archive, execution, {"phase": "calibration", "start": ref, "tag": tag})
        terminal = _execution_optional(archive, f"calibration-terminal-{tag}.json")
        if terminal is None:
            _finish_attempt(
                archive,
                execution,
                ref,
                start,
                _recover_process(archive, tag, clock=clock),
                clock=clock,
            )
            terminal = _execution_read(archive, f"calibration-terminal-{tag}.json")
        _validate_report_terminal({**terminal, "phase": "science"}, attempt, ref)
        terminal_ref = {
            "locator": f"calibration-terminal-{tag}.json",
            "sha256": _sha256(_world_bytes(terminal)),
        }
        provenance(ref, start, terminal_ref, terminal, write=True)
        cost = _execution_optional(archive, f"cost-{tag}.json")
        records.append(
            {
                "attempt": attempt,
                "phase": "calibration",
                "status": "complete" if cost is not None and cost["complete"] else "failed",
                "duration": cost["duration"]
                if cost is not None
                else terminal["finished_elapsed"] - start["started_elapsed"],
            }
        )
    return records


def _recover_terminals(archive, execution, allocation, *, clock=time.monotonic, _provenance=None):
    """Inventory first, reject conflicts, then write exactly one missing terminal."""
    provenance = _provenance or _existing_terminal_provenance(archive, execution)
    with _portable_report_parent(archive) as (fd, _):
        names = sorted(
            name
            for name in os.listdir(fd)
            if name.startswith(("start-", "terminal-")) and name.endswith(".json")
        )
    starts, ends = {}, {}
    for name in names:
        match = re.fullmatch(r"(start|terminal)-([0-9a-f]{64})\.json", name)
        if match is None:
            raise ValueError("invalid durable inventory name")
        kind, sha = match.groups()
        ref = {"version": _EVIDENCE_VERSION, "kind": kind, "locator": name, "sha256": sha}
        body = _report_file(archive, ref)
        i = body["attempt"]["attempt"]
        target = starts if kind == "start" else ends
        if i in target:
            raise ValueError("duplicate durable attempt")
        if type(i) is not int or not 0 <= i < len(allocation["manifest"]):
            raise ValueError("attempt outside frozen allocation")
        target[i] = (ref, body)
    if sorted(starts) != list(range(len(starts))) or set(ends) - set(starts):
        raise ValueError("nonprefix starts or orphan terminal")
    for i, (ref, body) in starts.items():
        _validate_report_start(body, allocation["manifest"][i], execution["input"], allocation)
        if i in ends:
            _validate_report_terminal(ends[i][1], allocation["manifest"][i], ref)
            provenance(ref, body, *ends[i])
    for i in sorted(ends):
        provenance(*starts[i], *ends[i], write=True)
    for i in sorted(set(starts) - set(ends)):
        terminal_ref, cost = _finish_attempt(
            archive,
            execution,
            *starts[i],
            _recover_process(archive, str(i), clock=clock),
            clock=clock,
        )
        provenance(*starts[i], terminal_ref, cost["terminal_body"], write=True)
    return _report_inventory(archive, execution["input"], allocation)


def _finalize_execution(archive, execution, *, clock=time.monotonic):
    """Runs in the reserve-bounded child; no generation or allocation is possible here."""
    frozen = _execution_read(archive, "allocation.json")
    allocation, calibration = frozen["allocation"], frozen["calibration"]
    inventory = _report_inventory(archive, execution["input"], allocation)
    prepared = _execution_optional(archive, "finalization-prepared.json")
    if prepared is None:
        began = clock()
        timing = {
            "clock": "monotonic_inclusive_before_setup",
            "started_count": len(inventory),
            "elapsed_seconds": began - execution["t0"],
            "finalization_seconds": 0,
        }
        summary = build_compact_summary(
            archive=archive,
            input_contract=execution["input"],
            calibration=calibration,
            allocation=allocation,
            ledger=[{"start": start[0], "terminal": end[0]} for start, end in inventory],
            timing=timing,
        )
        # Compact timing is explicitly a snapshot; the final receipt includes serialization,
        # read verification and fsync as well (no impossible self-referential end timestamp).
        now = clock()
        summary["evidence"]["timing"].update(
            elapsed_seconds=now - execution["t0"], finalization_seconds=now - began
        )
        summary = _render_compact(summary["evidence"])
        prepared = {"summary": summary, "began": began}
        _execution_record(archive, "finalization-prepared.json", prepared)
    summary, began = prepared["summary"], prepared["began"]
    validate_compact_summary(summary)
    for key, expected in (
        ("input", execution["input"]),
        ("allocation", allocation),
        ("calibration", calibration),
    ):
        _strict_frozen(summary["evidence"][key], expected, "prepared finalization binding")
    if clock() >= execution["final_deadline"]:
        raise TimeoutError("finalization deadline before compact persistence")
    sha = _sha256(_world_bytes(summary))
    ref = {
        "version": _EVIDENCE_VERSION,
        "kind": "summary",
        "locator": f"summary-{sha}.json",
        "sha256": sha,
    }
    try:
        existing = _report_file(archive, ref)
    except FileNotFoundError:
        ref = _durable_report(archive, "summary", summary)
    else:
        _strict_frozen(existing, summary, "prepared summary")
    read_compact_summary(archive, ref)
    finished = clock()
    receipt = {
        "summary": ref,
        "input_digest": execution["input"]["digest"],
        "allocation_digest": digest(allocation),
        "elapsed_seconds": finished - execution["t0"],
        "finalization_seconds": finished - began,
        "science_deadline_elapsed": 5100,
        "final_deadline_elapsed": 5400,
        "final_overrun_seconds": max(0, finished - execution["final_deadline"]),
        "timing_scope": "compact timing is the post-build snapshot; this receipt includes compact persistence and read verification",
    }
    _execution_record(archive, "finalization.json", receipt)
    return receipt


def _execution_child(request):
    archive = _world_archive(request["archive"])
    execution = _verify_execution(archive)
    mode = request["mode"]
    if mode not in ("attempt", "finalize"):
        raise ValueError("invalid internal operation")
    # stdin is the launch gate, but the durable parent receipt is also required.
    process = _execution_read(archive, f"process-{request['tag']}.json")
    expected = {key: value for key, value in request.items() if key != "archive"}
    _strict_frozen(process["request"], expected, "durable child request")
    if process["pid"] != os.getpid() or process["identity"] != _process_identity(os.getpid()):
        raise ValueError("internal child requires its recorded parent launch")
    if mode == "attempt":
        return _attempt_child(archive, execution, request)
    if time.monotonic() >= execution["final_deadline"]:
        raise TimeoutError("original final deadline exhausted")
    _finalize_execution(archive, execution)
    return 0


def _finalization_tag(archive, *, clock=time.monotonic):
    """Keep every failed attempt, including crashes before the PID receipt exists."""
    with _portable_report_parent(archive) as (fd, _):
        names = os.listdir(fd)
    tags = set()
    for name in names:
        match = re.fullmatch(
            r"(?:child|stream-layout|process|process-result|finalization-attempt)-(finalize(?:-[0-9]+)?)(?:-stderr)?\.(?:json|log)",
            name,
        )
        if match:
            tags.add(match[1])
    for tag in sorted(tags):
        _recover_process(archive, tag, clock=clock)
    i = 0
    while (tag := "finalize" if i == 0 else f"finalize-{i}") in tags:
        i += 1
    return tag


def _recovery_receipt(archive, body):
    with _portable_report_parent(archive) as (fd, _):
        names = set(os.listdir(fd))
    i = 0
    while (name := "recovery-finished.json" if i == 0 else f"recovery-finished-{i}.json") in names:
        i += 1
    return _execution_record(archive, name, body)


def _freeze_interrupted_calibration(archive, execution, *, clock=time.monotonic, _provenance=None):
    """Missing original allocation is not authority to resume calibration/science."""
    provenance = _provenance or _existing_terminal_provenance(archive, execution)
    retained = _execution_optional(archive, "calibration-recovery.json")
    if retained is None:
        records = _reconcile_calibration(archive, execution, clock=clock, _provenance=provenance)
        elapsed = clock() - execution["t0"]
        cfg = execution["input"]["contract"]["config"]
        allocation = allocate_from_calibration(cfg, elapsed, records)
        calibration = records
        if allocation["allocation_reason"] != "calibration_invalid":
            # Preserve all three real costs in the operational receipt, but do not
            # invent an original allocation or resume science after a parent crash.
            calibration = [
                {
                    "status": "failed",
                    "reason": "parent interrupted before original allocation was durably frozen",
                }
            ]
            allocation = allocate_from_calibration(cfg, elapsed, calibration)
        retained = {"records": records, "calibration": calibration, "allocation": allocation}
        _execution_record(archive, "calibration-recovery.json", retained)
    frozen = {key: retained[key] for key in ("calibration", "allocation")}
    _execution_record(archive, "allocation.json", frozen)
    return frozen


def _bounded_finalize(archive, execution, *, clock=time.monotonic, launch=_timed_execution_child):
    if clock() >= execution["final_deadline"]:
        result = {
            "returncode": None,
            "timed_out": True,
            "launched": False,
            "started": clock(),
            "finished": clock(),
            "pid": None,
        }
    else:
        tag = _finalization_tag(archive, clock=clock)
        _execution_record(
            archive,
            f"finalization-attempt-{tag}.json",
            {
                "started": clock(),
                "deadline": execution["final_deadline"],
            },
        )
        result = launch(
            archive,
            {"mode": "finalize", "tag": tag},
            execution["final_deadline"],
            clock=clock,
        )
    return {
        "process": result,
        "finalization": _execution_optional(archive, "finalization.json"),
        "elapsed_seconds": clock() - execution["t0"],
        "final_overrun_seconds": max(0, clock() - execution["final_deadline"]),
    }


def execute_science(config, root, *, t0, clock=time.monotonic, launch=_timed_execution_child):
    """Deliberate execution only. One original clock; no replacement or reallocation."""
    archive, execution = _freeze_execution(config, root, t0, clock=clock)
    calibration, allocation, error = [], None, None
    with _execution_lock(archive):
        try:
            for attempt in execution["input"]["contract"]["seed_manifest"]["calibration"]:
                if clock() >= execution["science_deadline"]:
                    break
                _, cost = _run_attempt(
                    archive, execution, None, attempt, "calibration", clock=clock, launch=launch
                )
                calibration.append(
                    {
                        "attempt": attempt,
                        "phase": "calibration",
                        "status": "complete" if cost["complete"] else "failed",
                        "duration": cost["duration"],
                    }
                )
            allocation = allocate_from_calibration(config, clock() - t0, calibration)
            _execution_record(
                archive, "allocation.json", {"calibration": calibration, "allocation": allocation}
            )
            # Read-back, not caller state, is authority before the first science start.
            frozen = _execution_read(archive, "allocation.json")
            _strict_frozen(
                frozen, {"calibration": calibration, "allocation": allocation}, "allocation freeze"
            )
            for attempt in allocation["manifest"]:
                if (
                    clock() + allocation["inputs"]["conservative_per_world_cost"]
                    > execution["science_deadline"]
                    or clock() >= execution["science_deadline"]
                ):
                    break
                _run_attempt(
                    archive, execution, allocation, attempt, "science", clock=clock, launch=launch
                )
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__}
            _execution_record(archive, "parent-error.json", error)
        # Never reset an already frozen cap after an exception.
        provenance = _existing_terminal_provenance(archive, execution)
        frozen = _execution_optional(archive, "allocation.json")
        if frozen is None:
            calibration = _reconcile_calibration(
                archive, execution, clock=clock, _provenance=provenance
            )
            allocation = allocate_from_calibration(config, clock() - t0, calibration)
            _execution_record(
                archive, "allocation.json", {"calibration": calibration, "allocation": allocation}
            )
        else:
            allocation = frozen["allocation"]
        _recover_terminals(archive, execution, allocation, clock=clock, _provenance=provenance)
        final = _bounded_finalize(archive, execution, clock=clock, launch=launch)
        receipt = {
            "archive": str(archive),
            "allocation_count": allocation["frozen_attempt_count"],
            "parent_error": error,
            **final,
        }
        _execution_record(archive, "execution-finished.json", receipt)
        return receipt


def recover_execution(archive, *, clock=time.monotonic, launch=_timed_execution_child):
    """Finalize only a recorded run, on its original boot/clock/source and frozen cap."""
    execution = _verify_execution(archive, clock=clock)
    with _execution_lock(archive):
        provenance = _existing_terminal_provenance(archive, execution)
        frozen = _execution_optional(archive, "allocation.json")
        if frozen is None:
            frozen = _freeze_interrupted_calibration(
                archive, execution, clock=clock, _provenance=provenance
            )
        else:
            _reconcile_calibration(archive, execution, clock=clock, _provenance=provenance)
        expected = allocate_from_calibration(
            execution["input"]["contract"]["config"],
            frozen["allocation"]["inputs"]["elapsed_after_calibration"],
            frozen["calibration"],
        )
        _strict_frozen(frozen["allocation"], expected, "original allocation")
        _recover_terminals(
            archive, execution, frozen["allocation"], clock=clock, _provenance=provenance
        )
        if _execution_optional(archive, "finalization.json") is not None:
            return verify_execution(archive, clock=clock)
        if clock() >= execution["final_deadline"]:
            _finalization_tag(archive, clock=clock)  # reap only; no new child
            _recovery_receipt(
                archive,
                {
                    "finalization": None,
                    "reason": "original 5400-second clock exhausted",
                    "elapsed_seconds": clock() - execution["t0"],
                    "final_overrun_seconds": max(0, clock() - execution["final_deadline"]),
                },
            )
            raise TimeoutError("recovery cannot restart the original 5400-second clock")
        result = _bounded_finalize(archive, execution, clock=clock, launch=launch)
        _recovery_receipt(archive, result)
        return result


def verify_execution(archive, *, clock=time.monotonic):
    """Read-only verification: never repairs, allocates, generates or writes."""
    execution = _verify_execution(archive, clock=clock)
    frozen = _execution_read(archive, "allocation.json")
    inventory = _report_inventory(archive, execution["input"], frozen["allocation"])
    final = _execution_optional(archive, "finalization.json")
    if final is not None:
        read_compact_summary(archive, final["summary"], raw_archive=archive)
    return {
        "archive": str(archive),
        "started_count": len(inventory),
        "allocation_count": frozen["allocation"]["frozen_attempt_count"],
        "finalization": final,
        "read_only": True,
    }


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(
        description="Explicit bounded OLS execution; help/validate/dry-input/verify never generate worlds"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("validate", "dry-input", "science", "recover", "verify", "_child"),
        default="validate",
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive")
    parser.add_argument("--validate-only", action="store_true", help="legacy read-only validation")
    args = parser.parse_args()
    if args.validate_only and args.command != "validate":
        parser.error("--validate-only cannot be combined with execution")
    if args.command == "_child":
        if sys.stdin.isatty():
            parser.error("internal child requires parent pipe")
        raise SystemExit(_execution_child(json.load(sys.stdin)))
    if args.command in ("recover", "verify"):
        if not args.archive or args.output_dir or args.config:
            parser.error("recover/verify require only --archive")
        result = (
            recover_execution(args.archive)
            if args.command == "recover"
            else verify_execution(args.archive)
        )
    else:
        if not args.config or args.archive:
            parser.error("validate/dry-input/science require --config, not --archive")
        cfg = validate_config(json.loads(Path(args.config).read_text()))
        if args.command == "science":
            if not args.output_dir:
                parser.error(
                    "science requires an explicit external --output-dir; no repository default"
                )
            result = execute_science(cfg, args.output_dir, t0=_ENTRY_T0)
        else:
            if args.output_dir:
                parser.error("read-only commands do not accept --output-dir")
            result = {
                "schema_version": SCHEMA_VERSION,
                "config_sha256": digest(cfg),
                "validated": True,
            }
            if args.command == "dry-input":
                result = freeze_input_contract(
                    cfg,
                    environment=_execution_environment(),
                    source_hashes=analysis_source_hashes(),
                    seed_manifest=study_schedules(cfg),
                )
    print(canonical_json(result))


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite shape {shape}, got {array.shape}")
    return array


def _finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, (np.ndarray, int, float, np.number)):
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("nonfinite audit parameter")


def _walk(eps: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Analysis-local centred, edge-smoothed Brownian operator and fixed RMS scale."""
    n = len(eps)
    width = min(n, max(1, round(spec["smoothness"] * spec["rw_smoothness_max_weeks"])))

    def smooth(raw):
        if width == 1:
            return raw
        left, right = width // 2, width - 1 - width // 2
        padded = np.concatenate(
            [np.repeat(raw[:1], left, axis=0), raw, np.repeat(raw[-1:], right, axis=0)]
        )
        cumulative = np.cumsum(padded, axis=0)
        head = np.zeros((1, *raw.shape[1:]))
        return (cumulative[width - 1 :] - np.concatenate([head, cumulative[: n - 1]])) / width

    operator = smooth(np.tril(np.ones((n, n))))
    operator -= operator.mean(axis=0)
    scale = np.sqrt(np.sum(operator**2) / n)
    path = smooth(np.cumsum(eps))
    path = (path - path.mean()) * spec["std"] / scale + spec["mean"]
    return np.logaddexp(0.0, path) if spec["positive_only"] else path


def _schedule(spec: dict[str, Any], weeks: np.ndarray, n: int, treatment: bool):
    gate = np.ones(len(weeks), dtype=bool)
    smooth = np.zeros(len(weeks))
    use = spec.get("use", {})
    if use.get("onset"):
        gate &= weeks >= spec["onset"]["start"]
    if use.get("offset"):
        gate &= weeks < spec["offset"]["stop"]
    if use.get("flighting"):
        f = spec["flighting"]
        gate &= (weeks + f["phase"]) % f["period"] < f["on_weeks"]
    if use.get("seasonal"):
        s = spec["seasonal"]
        smooth += s["amplitude"] * np.sin(2 * np.pi * weeks / s["period"] + s["phase"])
    if use.get("trend"):
        smooth += spec["trend"]["change"] * np.maximum(weeks, 0) / (n - 1)
    level = np.exp(smooth) if treatment else smooth
    if use.get("level_jump"):
        jump = spec["level_jump"]
        after = weeks[:, None] >= np.asarray(jump["week"])[None, :]
        if treatment:
            level *= np.prod(np.where(after, jump["factor"], 1.0), axis=1)
        else:
            level += after @ np.asarray(jump["size"])
    return gate, level


def carryover_path(x: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Actual causal, sum-normalized kernel on the *entire* audited path."""
    family, length = spec["family"], spec["l_max"]
    if family == "none":
        return x.copy()
    if family not in ("geometric", "weibull"):
        raise ValueError(f"unsupported carryover family: {family}")
    if length == 1:
        return x.copy()
    if family == "geometric":
        weights = spec["alpha"] ** np.arange(length)
    else:
        lag = np.arange(1, length + 1, dtype=float)
        lam, k = spec["lam"], spec["k"]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            raw = (k / lam) * (lag / lam) ** (k - 1) * np.exp(-((lag / lam) ** k))
            span = raw.max() - raw.min()
            weights = (raw - raw.min()) / span
        if (
            not np.isfinite(span)
            or span == 0
            or raw.max() <= 1e-300
            or not np.isfinite(weights.sum())
            or weights.sum() == 0
        ):
            return np.zeros_like(x)
    weights = weights / weights.sum()
    return np.convolve(x, weights, mode="full")[: len(x)]


def saturation_path(x: np.ndarray, spec: dict[str, Any], *, stable: bool = False) -> np.ndarray:
    """Local response law; scale is the generated parameter anchor, never a statistic."""
    family, scale = spec["family"], spec["scale"]
    if scale <= 0:
        raise ValueError("saturation scale must be positive")
    if family == "linear":
        return x / scale
    scale = max(scale, 1e-8)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = x / scale
        if family == "hill":
            if not stable:
                power = (spec["kappa_mult"] * scale) ** spec["slope"]
                return 1 - power / (power + x ** spec["slope"])
            positive = x > 0
            safe = np.where(positive, x, 1.0)
            r = safe / (spec["kappa_mult"] * scale)
            log_r = np.where(
                (r >= np.finfo(float).tiny) & np.isfinite(r),
                np.log(r),
                np.log(safe) - np.log(scale) - np.log(spec["kappa_mult"]),
            )
            odds = spec["slope"] * log_r
            return np.where(positive, np.exp(odds - np.logaddexp(0, odds)), 0.0)
        if family == "logistic":
            if not stable:
                exponential = np.exp(-spec["lam"] * ratio)
                return (1 - exponential) / (1 + exponential)
            # Extended intermediates preserve a finite complete product/quotient.
            argument = np.asarray(x.astype(np.longdouble) * spec["lam"] / scale, dtype=float)
            exponential = np.expm1(-np.abs(argument))
            return np.copysign(-exponential / (2 + exponential), argument)
        if family == "michaelis_menten":
            lam = scale * spec["kappa_mult"]
            denominator = lam + x
            extended = x.astype(np.longdouble) / (
                np.longdouble(scale) * spec["kappa_mult"] + x.astype(np.longdouble)
            )
            return np.where(
                (lam > 0) & np.isfinite(denominator),
                x / denominator,
                np.asarray(extended, dtype=float),
            )
        if family == "tanh":
            argument = np.where(
                (np.abs(ratio) >= np.finfo(float).tiny) & np.isfinite(ratio),
                ratio / spec["c"],
                np.asarray(x.astype(np.longdouble) / scale / spec["c"], dtype=float),
            )
            return np.tanh(argument)
        if family == "root":
            result = np.where(
                (ratio >= np.finfo(float).tiny) & np.isfinite(ratio),
                ratio ** spec["alpha"],
                np.exp(spec["alpha"] * (np.log(x) - np.log(scale))),
            )
            return np.where(x > 0, result, 0.0)
    raise ValueError(f"unsupported saturation family: {family}")


def reconstruct_features(world) -> dict[str, Any]:
    """Replay D/Z/C from public audit inputs; no Oracle, replay helper or beta division."""
    n, burn = world.cfg.n_time_steps, world.cfg.carryover_burn_in
    nf, nc, nz, nd = n + burn, world.cfg.n_treatments, world.cfg.n_covariates, world.cfg.n_latent
    audit, eps = world.equation_parameters, world.exogenous
    _finite_tree(audit)
    shapes = {
        "eps_d": (nf, nd),
        "eps_z": (nf, nz),
        "eps_c": (nf, nc),
        "eps_b": (nf,),
        "eps_y": (nf,),
        "eps_c_hf": (nf, nc),
        "eps_c_pulse": (nf, nc),
        "eps_z_hf": (nf, nz),
        "eps_z_pulse": (nf, nz),
    }
    for name, shape in shapes.items():
        eps[name] = _finite_array(eps[name], shape, name)
    rho = audit["innovations"]["rho"]
    if not -1 <= rho <= 1:
        raise ValueError("invalid innovation correlation")
    mixed = np.sqrt(1 - rho**2) * eps["eps_c"] + rho * eps["eps_b"][:, None]
    nodes, bases = {}, []
    weeks = np.arange(-burn, n)
    shock = audit.get("treatment_shocks")
    if shock is not None:
        for key in ("mask_full", "level_full"):
            _finite_array(shock[key], (nf, nc), key)

    def parents(spec, group):
        selected = [
            (name, coefficient)
            for name, coefficient in spec.get("parents", {}).items()
            if name.startswith(group)
        ]
        if not selected:
            return np.zeros(nf)
        return np.column_stack([nodes[name] for name, _ in selected]) @ np.array(
            [coefficient for _, coefficient in selected]
        )

    for group, count, innovation in (
        ("D", nd, eps["eps_d"]),
        ("Z", nz, eps["eps_z"]),
        ("C", nc, mixed),
    ):
        for i in range(count):
            name = f"{group}{i + 1}"
            spec = audit[name]
            own = _walk(innovation[:, i], spec["random_walk"])
            if group == "D":
                nodes[name] = own
                continue
            texture = spec["texture"]
            prefix = "covariate_" if group == "Z" else ""
            eps_prefix = "eps_z" if group == "Z" else "eps_c"
            if texture[f"use_{prefix}hf"]:
                own += texture[f"{prefix}hf_sigma"] * eps[f"{eps_prefix}_hf"][:, i]
            if texture[f"use_{prefix}pulse"]:
                fire = eps[f"{eps_prefix}_pulse"][:, i]
                if group == "Z":
                    fire = fire - texture["covariate_pulse_prob"]
                own += texture[f"{prefix}pulse_amp"] * fire
            gate, level = _schedule(spec.get("trajectory", {}), weeks, n, group == "C")
            if group == "Z":
                nodes[name] = np.where(
                    gate, parents(spec, "D") + parents(spec, "Z") + own + level, 0
                )
            else:
                path = parents(spec, "D") + parents(spec, "Z") + parents(spec, "C") + own
                path = np.where(gate, np.logaddexp(0, path) * level, 0)
                base = np.where(gate, np.logaddexp(0, own) * level, 0)
                if shock is not None:
                    mask, held = shock["mask_full"][:, i] != 0, shock["level_full"][:, i]
                    path, base = np.where(mask, held, path), np.where(mask, held, base)
                nodes[name] = path
                bases.append(base)
    full = {
        group: np.column_stack([nodes[f"{group}{i + 1}"] for i in range(count)])
        if count
        else np.empty((nf, 0))
        for group, count in (("D", nd), ("Z", nz), ("C", nc))
    }
    full["C_base"] = np.column_stack(bases)
    stable = world.params.get("mechanism_priors_enabled", False)
    for source, target in (("C", "phi"), ("C_base", "phi_base")):
        full[target] = np.column_stack(
            [
                saturation_path(
                    carryover_path(full[source][:, k], audit[f"C{k + 1}"]["response"]["carryover"]),
                    audit[f"C{k + 1}"]["response"]["saturation"],
                    stable=stable,
                )
                for k in range(nc)
            ]
        )
    for name, array in full.items():
        _finite_array(array, array.shape, name)
    return {
        "n_full": nf,
        "burn_in": burn,
        "full": full,
        "reported": {name: array[burn:].copy() for name, array in full.items()},
        "exogenous": eps,
        "mixed_eps_c": mixed,
        "equation_parameters": audit,
    }


def _design_diagnostics(X: np.ndarray, rank_tolerance: float) -> dict[str, Any]:
    singular = np.linalg.svd(X, compute_uv=False)
    cutoff = rank_tolerance * singular[0]
    rank = int(np.sum(singular > cutoff))
    condition = (
        float(singular[0] / singular[-1])
        if len(singular) >= X.shape[1] and singular[-1] > 0
        else math.inf
    )
    return {
        "rank": rank,
        "singular_values": singular,
        "condition": condition,
        "rank_tolerance": rank_tolerance,
        "rank_cutoff": float(cutoff),
        "shape": X.shape,
    }


def analyze_world(world, *, fit_indices=None, rank_tolerance: float = 1e-10) -> dict[str, Any]:
    """Identity-gated paired OLS. Arrays and orthogonal failures are retained for persistence.

    ``fit_indices`` must be a predeclared, strictly increasing reported-row selector.
    Identity residuals use the frozen absolute tolerance, on all rows AND selected rows.
    """
    from copy import deepcopy

    result: dict[str, Any] = {
        "fits": {},
        "identities": {},
        "reasons": [],
        "rank_tolerance": rank_tolerance,
    }
    status = {"scheduled": True, "attempted": True, "generated": True, "truth_eligible": False}
    try:
        n, nc, nz, nd = (
            world.cfg.n_time_steps,
            world.cfg.n_treatments,
            world.cfg.n_covariates,
            world.cfg.n_latent,
        )
        if not 0 < rank_tolerance < 1:
            raise ValueError("rank tolerance must be in (0,1)")
        if (
            world.cfg.outcome_std_mode != "relative"
            or world.cfg.baseline_floor is not None
            or tuple(world.cfg.rw_baseline_std_range) != (0.0, 0.0)
            or nd != 1
        ):
            raise ValueError("world is outside the scalar-baseline restricted population")
        rows = np.arange(n) if fit_indices is None else np.asarray(fit_indices)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or not len(rows)
            or rows[0] < 0
            or rows[-1] >= n
            or np.any(np.diff(rows.astype(int)) <= 0)
        ):
            raise ValueError("fit indices must be nonempty, original-order unique reported rows")
        graph_shapes = {
            "g_cy": (nc,),
            "g_zy": (nz,),
            "g_dy": (nd,),
            "g_dc": (nd, nc),
            "g_dz": (nd, nz),
            "g_zc": (nz, nc),
            "g_cc": (nc, nc),
            "g_zz": (nz, nz),
        }
        for name, shape in graph_shapes.items():
            g = _finite_array(world.g[name], shape, name)
            if not np.isin(g, [0, 1]).all() or (name in ("g_cc", "g_zz") and np.tril(g).any()):
                raise ValueError(f"{name}: invalid original topological order or edge mask")
        shapes = {
            "treatments": (n, nc),
            "treatments_base": (n, nc),
            "covariates": (n, nz),
            "latent_unobserved": (n, nd),
            "contributions_observed": (n, nc),
            "contributions": (n, nc),
            "covariate_contribution": (n, nz),
            "latent_unobserved_contribution": (n, nd),
            "baseline_intrinsic": (n,),
            "outcome_noise": (n,),
            "outcome": (n,),
            "baseline": (n,),
            "indirect_effects": (n,),
            "indirect_effects_by_source": (n, 3),
            "saturation_scale": (nc,),
        }
        data = {
            name: _finite_array(world.data[name], shape, name).copy()
            for name, shape in shapes.items()
        }
        _finite_tree(world.params)
        beta = _finite_array(world.params["beta"], (nc,), "beta")
        rho = _finite_array(world.params["rho_zy"], (nz,), "rho_zy")
        delta = _finite_array(world.params["delta_dy"], (nd,), "delta_dy")
        reconstruction = reconstruct_features(world)
        result["audit"] = reconstruction
        result["source"] = {
            "data": data,
            "graph": deepcopy(world.g),
            "params": deepcopy(world.params),
            "seed": world.seed,
        }
        reported = reconstruction["reported"]
        dc, dz = np.flatnonzero(world.g["g_cy"]), np.flatnonzero(world.g["g_zy"])
        X = np.column_stack([reported["phi"][:, dc], reported["Z"][:, dz], np.ones(n)])
        b = data["baseline_intrinsic"][0]
        truth = np.concatenate([beta[dc], rho[dz], [b]])
        u = data["latent_unobserved_contribution"].sum(axis=1)
        noise, outcome = data["outcome_noise"], data["outcome"]
        y0, y1 = outcome - u - noise, outcome - u
        fit_mask = np.zeros(n, dtype=bool)
        fit_mask[rows] = True
        result.update(
            design=X,
            truth=truth,
            targets={
                "y0": y0,
                "y1": y1,
                "u": u,
                "outcome_noise": noise,
                "raw_observed_unadjusted": outcome,
            },
            coefficient_order=[
                *(f"beta[{k}]" for k in dc),
                *(f"rho_zy[{m}]" for m in dz),
                "intercept",
            ],
            direct_treatments=dc,
            direct_controls=dz,
            fit_indices=rows.copy(),
            all_row_mask=np.ones(n, dtype=bool),
            fit_row_mask=fit_mask,
            n_full=reconstruction["n_full"],
            n_fit=len(rows),
            p=X.shape[1],
            shapes={name: array.shape for name, array in data.items()},
        )

        def identity(name, actual, expected):
            actual, expected = np.asarray(actual), np.asarray(expected)
            if (
                actual.shape != expected.shape
                or not np.isfinite(actual).all()
                or not np.isfinite(expected).all()
            ):
                raise ValueError(f"{name}: nonfinite or incompatible identity arrays")
            residual = np.abs(actual - expected)
            metrics = {
                "all_max_abs": float(residual.max(initial=0)),
                "fit_max_abs": float(residual[rows].max(initial=0)),
                "tolerance": IDENTITY_TOLERANCE,
            }
            metrics["passed"] = (
                max(metrics["all_max_abs"], metrics["fit_max_abs"]) <= IDENTITY_TOLERANCE
            )
            result["identities"][name] = metrics
            if not metrics["passed"]:
                result["reasons"].append(f"identity_or_truth_ineligible:{name}")

        for source, target in (
            ("D", "latent_unobserved"),
            ("Z", "covariates"),
            ("C", "treatments"),
            ("C_base", "treatments_base"),
        ):
            identity(f"reconstructed_{source}", reported[source], data[target])
        identity("scalar_baseline", data["baseline_intrinsic"], np.full(n, b))
        identity(
            "observed_contribution",
            reported["phi"] * (beta * world.g["g_cy"]),
            data["contributions_observed"],
        )
        identity(
            "base_contribution",
            reported["phi_base"] * (beta * world.g["g_cy"]),
            data["contributions"],
        )
        identity(
            "control_contribution",
            reported["Z"] * (rho * world.g["g_zy"]),
            data["covariate_contribution"],
        )
        identity(
            "direct_latent",
            reported["D"] * (delta * world.g["g_dy"]),
            data["latent_unobserved_contribution"],
        )
        indirect = (data["contributions_observed"] - data["contributions"]).sum(axis=1)
        identity("observed_base_indirect", indirect, data["indirect_effects_by_source"].sum(axis=1))
        identity("indirect_total", indirect, data["indirect_effects"])
        baseline = (
            data["baseline_intrinsic"] + u + data["covariate_contribution"].sum(axis=1) + noise
        )
        identity("baseline_additive", baseline, data["baseline"])
        identity("outcome_additive", baseline + data["contributions_observed"].sum(axis=1), outcome)
        identity("target_truth", X @ truth, y0)
        identity("noise_pair", y1 - y0, noise)
        identity(
            "audited_noise",
            reconstruction["exogenous"]["eps_y"][world.cfg.carryover_burn_in :]
            * reconstruction["equation_parameters"]["Y"]["iid_noise"]["std"],
            noise,
        )
        status["raw_observed_misspecified"] = bool(np.any(u != 0))
        if result["reasons"]:
            result["status"] = orthogonal_status(**status, reason="identity_or_truth_ineligible")
            return result
        status["truth_eligible"] = True
        result["target_metrics"] = {
            name: {
                scope: {
                    "mean": float(values[index].mean()),
                    "std": float(values[index].std()),
                    "max_abs": float(np.max(np.abs(values[index]))),
                }
                for scope, index in (("all", np.arange(n)), ("fit", rows))
            }
            for name, values in result["targets"].items()
        }
        selected = X[rows]
        nonintercept = selected[:, :-1]
        means, scales = nonintercept.mean(axis=0), nonintercept.std(axis=0)
        constants = np.ptp(nonintercept, axis=0) == 0
        standardized = np.column_stack(
            [(nonintercept - means) / np.where(scales == 0, 1, scales), np.ones(len(rows))]
        )
        raw_info = _design_diagnostics(selected, rank_tolerance)
        std_info = _design_diagnostics(standardized, rank_tolerance)
        for info in (raw_info, std_info):
            info["conditioning"] = orthogonal_status(**status, condition=info["condition"])[
                "conditioning"
            ]
        result.update(
            raw_design=raw_info,
            standardized_design=std_info,
            standardized_X=standardized,
            column_means=means,
            column_scales=scales,
            constant_columns=np.flatnonzero(constants),
        )
        status.update(rank=raw_info["rank"], p=X.shape[1], condition=raw_info["condition"])
        for name in ("y0", "y1"):
            fitted = fit_design(selected, result["targets"][name][rows], rank_tolerance)
            _finite_array(fitted["coefficients"], truth.shape, f"{name} coefficients")
            error = fitted["coefficients"] - truth
            fitted.update(
                coefficient_error=error,
                max_abs_error=float(np.max(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))),
                residual=result["targets"][name][rows] - selected @ fitted["coefficients"],
            )
            result["fits"][name] = fitted
        status["noisefree_recovered"] = bool(
            raw_info["rank"] == X.shape[1]
            and result["fits"]["y0"]["max_abs_error"] <= NOISEFREE_COEFFICIENT_TOLERANCE
        )
        status["paired_noisy_available"] = True
        if raw_info["rank"] < X.shape[1]:
            result["reasons"].append("rank_deficient")
        elif not status["noisefree_recovered"]:
            result["reasons"].append("noisefree_coefficient_error")
    except (KeyError, AttributeError) as exc:
        result["reasons"].append(f"unsupported_truth:{exc}")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["reasons"].append(f"identity_or_truth_ineligible:{exc}")
    result["status"] = orthogonal_status(**status, reason=";".join(result["reasons"]) or None)
    if result["fits"].get("y1") is not None:
        result["status"]["paired_noisy"]["max_abs_error"] = result["fits"]["y1"]["max_abs_error"]
    return result


if __name__ == "__main__":
    main()
