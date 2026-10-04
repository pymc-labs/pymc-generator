"""Composable per-input trajectories (issue #24).

Every treatment and covariate can carry, independently per input, any subset of
``TRAJECTORY_COMPONENTS``. These tests pin the contract end to end:

* the seed contract — defaults and unwired components reproduce the legacy
  RNG traversal, structure draw and corpora bit for bit;
* the concrete graph — a component changes exactly its own input, by exactly
  its equation, in every decomposition variant;
* the component formulas, the archetype presets and the sign guarantees;
* the optional corpus block, the per-component flag streams, config
  validation, the preset horizon sweep and the natural-path realism filter.

Expected schedules are always recomputed here in numpy from the drawn or given
parameters, never through the production helpers. Tolerances: exact for
integers, booleans, exact zeros, constant-factor products (x2, x3) and untouched
columns; otherwise ``rtol=1e-12`` with ``atol=1e-12·max|operand|``.
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytest

import pymc_generator as pg
from pymc_generator import make_scm_prior
from pymc_generator.presets import TRAJECTORY_ARCHETYPES, TRAJECTORY_PRESETS
from pymc_generator.sampler import (
    _CORPUS_TRAJECTORY_NAMES,
    CARRYOVER_FAMILY_KEYS,
    SATURATION_FAMILY_KEYS,
    SCMPrior,
    _additive_task_ok,
    _warn_flat_texture,
    sample_g_additive,
)
from pymc_generator.slots import (
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_INPUTS,
)
from pymc_generator.symbolic_graph import build_symbolic_graph
from pymc_generator.trajectories import (
    GATE_COMPONENTS,
    LEVEL_COMPONENTS,
    SCHEDULE_COMPONENTS,
    activity_column,
    flighting_on_weeks,
    level_shift_column,
    min_on_weeks,
    sample_component_flags,
    structural_key,
    time_index,
    trajectory_params,
    treatment_multiplier_column,
)
from pymc_generator.world_model import build_world_model, draw_worlds, sample_structure

# -- shared helpers -------------------------------------------------------------


def _assert_close(actual, expected, *, scale: float | None = None, err_msg: str = "") -> None:
    """The module tolerance for computed values: rtol 1e-12, atol 1e-12·max|operand|."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if scale is None:
        scale = max(np.abs(actual).max(initial=0.0), np.abs(expected).max(initial=0.0))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * scale, err_msg=err_msg)


def _evaluate(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compile ONE FAST_COMPILE function over every output and run it."""
    names = list(outputs)
    values = pytensor.function([], [outputs[name] for name in names], mode="FAST_COMPILE")()
    return {name: np.asarray(value) for name, value in zip(names, values)}


def _assert_valid(corpus: dict[str, Any]) -> None:
    assert pg.DataGenerator.validate_corpus(corpus) == []


def _edge_free(n_treatments: int, n_covariates: int, n_latent: int) -> dict[str, np.ndarray]:
    """A cell whose inputs have no parents: every series is its own drive."""
    return {
        "g_cy": np.ones(n_treatments, dtype=int),
        "g_dc": np.zeros((n_latent, n_treatments), dtype=int),
        "g_dz": np.zeros((n_latent, n_covariates), dtype=int),
        "g_dy": np.ones(n_latent, dtype=int),
        "g_zy": np.ones(n_covariates, dtype=int),
        "g_zc": np.zeros((n_covariates, n_treatments), dtype=int),
        "g_cc": np.zeros((n_treatments, n_treatments), dtype=int),
        "g_zz": np.zeros((n_covariates, n_covariates), dtype=int),
    }


def _schedule_probs(p: float) -> dict[str, float]:
    """Every gate and level component of both input types at inclusion probability ``p``."""
    return {f"{x}_{c}_inclusion_prob": p for x in TRAJECTORY_INPUTS for c in SCHEDULE_COMPONENTS}


#: Gate priors that keep every gate valid at a 16-week horizon (support prefix 8).
_SHORT_GATES: dict[str, Any] = {
    **{f"{x}_onset_frac_range": (0.1, 0.2) for x in TRAJECTORY_INPUTS},
    **{f"{x}_flighting_period_weeks_range": (2, 3) for x in TRAJECTORY_INPUTS},
    **{f"{x}_flighting_duty_range": (0.5, 0.8) for x in TRAJECTORY_INPUTS},
}

_SCHEDULE_COLUMNS = [TRAJECTORY_COMPONENTS.index(c) for c in SCHEDULE_COMPONENTS]


def _on_run(duty: float, period: int) -> int:
    """Flighting on-run restated: round half up, at least 1, at most ``period - 1``."""
    return min(max(math.floor(duty * period + 0.5), 1), period - 1)


# -- 1. seed contract -----------------------------------------------------------

_PROBE = {"n_treatments": 2, "n_covariates": 1, "n_latent": 1, "n_time_steps": 16}

#: Shared by the corpus twins: rejections at this CV floor exercise the realism path.
_TWIN = {
    "n_treatments": 2,
    "n_covariates": 1,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 2,
    "draws_per_cell": 2,
    "seed": 5,
    "treatment_cv_floor": 0.2,
}


def _probe_cell() -> dict[str, np.ndarray]:
    return {
        "g_cy": np.array([1, 1]),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_dy": np.ones(1, dtype=int),
        "g_zy": np.ones(1, dtype=int),
        "g_zc": np.ones((1, 2), dtype=int),
        "g_cc": np.array([[0, 1], [0, 0]]),
        "g_zz": np.zeros((1, 1), dtype=int),
    }


@pytest.fixture(scope="module")
def default_corpus():
    return pg.sample_prior_predictive(make_scm_prior(**_TWIN))


@pytest.fixture(scope="module")
def shock_twins():
    """Shocks only, and shocks plus every schedule component at probability ~0."""
    base = pg.sample_prior_predictive(make_scm_prior(**_TWIN, n_treatment_shocks=1))
    inert = pg.sample_prior_predictive(
        make_scm_prior(**_TWIN, n_treatment_shocks=1, **_schedule_probs(1e-9), **_SHORT_GATES)
    )
    return base, inert


_LEGACY_STRUCTURE_KEYS = {
    "carryover_family",
    "sat_family",
    "smoothness_d",
    "smoothness_z",
    "smoothness_c",
    "smoothness_b",
    "use_hf",
    "use_pulse",
    "use_covariate_hf",
    "use_covariate_pulse",
}


def _legacy_structure(g: dict, cfg: SCMPrior, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The pre-#24 structure draw, restated: its numpy consumption is the contract."""
    n_t, n_c, n_l = len(g["g_cy"]), len(g["g_zy"]), len(g["g_dy"])
    carryover = rng.choice(
        len(CARRYOVER_FAMILY_KEYS),
        size=n_t,
        p=np.asarray([cfg.carryover_family_probs[k] for k in CARRYOVER_FAMILY_KEYS]),
    )
    saturation = rng.choice(
        len(SATURATION_FAMILY_KEYS),
        size=n_t,
        p=np.asarray([cfg.saturation_family_probs[k] for k in SATURATION_FAMILY_KEYS]),
    )

    def smooth(n: int) -> np.ndarray:
        return rng.beta(cfg.rw_smoothness_alpha, cfg.rw_smoothness_beta, size=n)

    return {
        "carryover_family": carryover.astype(int),
        "sat_family": saturation.astype(int),
        "smoothness_d": smooth(n_l),
        "smoothness_z": smooth(n_c),
        "smoothness_c": smooth(n_t),
        "smoothness_b": smooth(1),
        "use_hf": np.full(n_t, float(cfg.treatment_hf_sigma_range[1]) > 0.0),
        "use_pulse": np.full(n_t, float(cfg.treatment_pulse_prob_range[1]) > 0.0),
        "use_covariate_hf": np.full(n_c, float(cfg.covariate_hf_sigma_range[1]) > 0.0),
        "use_covariate_pulse": np.full(n_c, float(cfg.covariate_pulse_prob_range[1]) > 0.0),
    }


#: The structure keys schedule components add, only when the config enables one.
_SCHEDULE_STRUCTURE_KEYS = {
    "use_onset",
    "use_offset",
    "use_flighting",
    "use_level_jump",
    "use_seasonal",
    "use_trend",
    "use_covariate_onset",
    "use_covariate_offset",
    "use_covariate_flighting",
    "use_covariate_level_jump",
    "use_covariate_seasonal",
    "use_covariate_trend",
}


@pytest.mark.parametrize(
    "overrides, fractional, legacy_texture",
    (
        ({}, False, True),
        (
            {"treatment_hf_sigma_range": (0.0, 0.0), "covariate_pulse_prob_range": (0.0, 0.0)},
            False,
            True,
        ),
        ({"treatment_hf_inclusion_prob": 0.4, "covariate_pulse_inclusion_prob": 0.0}, True, False),
        # An off range makes the effective probability 0, so nothing is drawn.
        ({"treatment_hf_inclusion_prob": 0.4, "treatment_hf_sigma_range": (0.0, 0.0)}, False, True),
        ({**_schedule_probs(0.5), **_SHORT_GATES}, True, True),
        ({**_schedule_probs(1.0), **_SHORT_GATES}, False, True),
    ),
    ids=("default", "ranges_off", "fractional_hf", "dead_fractional_hf", "half", "all_on"),
)
def test_structure_draw_keeps_the_legacy_numpy_consumption(overrides, fractional, legacy_texture):
    """No inclusion probability moves the main stream or the legacy structure.

    Fractional probabilities draw from spawned child streams: the main
    bit-generator state is exactly the legacy one afterwards and only the seed
    sequence's child counter moves, by one.
    """
    cfg = make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=16, **overrides)
    g = _edge_free(3, 2, 1)
    rng, twin = np.random.default_rng(9), np.random.default_rng(9)
    got = sample_structure(g, cfg, rng)
    expected = _legacy_structure(g, cfg, twin)

    assert rng.bit_generator.state == twin.bit_generator.state
    assert twin.bit_generator.seed_seq.n_children_spawned == 0
    assert rng.bit_generator.seed_seq.n_children_spawned == int(fractional)
    schedules = cfg.trajectory_components_enabled
    assert set(got) == _LEGACY_STRUCTURE_KEYS | (_SCHEDULE_STRUCTURE_KEYS if schedules else set())
    texture_flags = {"use_hf", "use_pulse", "use_covariate_hf", "use_covariate_pulse"}
    compared = _LEGACY_STRUCTURE_KEYS if legacy_texture else _LEGACY_STRUCTURE_KEYS - texture_flags
    for key in compared:
        np.testing.assert_array_equal(got[key], expected[key], err_msg=key)
        assert got[key].dtype == expected[key].dtype, key


_PERTURBED_DISABLED_PRIORS: dict[str, Any] = {
    "treatment_onset_frac_range": (0.15, 0.3),
    "covariate_onset_frac_range": (0.2, 0.35),
    "treatment_offset_frac_range": (0.5, 0.95),
    "covariate_offset_frac_range": (0.45, 0.7),
    "treatment_flighting_period_weeks_range": (2, 9),
    "covariate_flighting_period_weeks_range": (3, 5),
    "treatment_flighting_duty_range": (0.2, 0.9),
    "covariate_flighting_duty_range": (0.6, 1.0),
    "treatment_level_jump_count": 3,
    "covariate_level_jump_count": 2,
    # 3·log 4 would break the 3.0 swing bound if jumps were included: a disabled
    # component's priors must not count toward it.
    "treatment_level_jump_factor_range": (0.25, 4.0),
    "covariate_level_jump_size_range": (-2.0, 0.5),
    "covariate_seasonal_amplitude_range": (0.05, 2.0),
    "covariate_seasonal_period_weeks_range": (8.0, 30.0),
    "treatment_trend_log_change_range": (-0.4, 1.7),
    "covariate_trend_change_range": (0.3, 2.5),
}


def test_disabled_component_priors_are_inert_while_another_component_runs():
    """With treatment seasonality running, no other component's priors reach the world."""
    g = _probe_cell()
    running = make_scm_prior(**_PROBE, treatment_seasonal_inclusion_prob=1.0)
    perturbed = make_scm_prior(
        **_PROBE, treatment_seasonal_inclusion_prob=1.0, **_PERTURBED_DISABLED_PRIORS
    )
    worlds = []
    for cfg in (running, perturbed):
        structural = sample_structure(g, cfg, np.random.default_rng(0))
        model, out_names, param_names = build_world_model(g, cfg, structural, cfg.n_time_steps)
        drawn = draw_worlds(model, out_names + param_names, seed=13, draws=2)
        worlds.append(([rv.name for rv in model.free_RVs], drawn))
    (rvs, drawn), (perturbed_rvs, perturbed_drawn) = worlds

    assert perturbed_rvs == rvs
    default = make_scm_prior(**_PROBE)
    default_model, _, _ = build_world_model(
        g, default, sample_structure(g, default, np.random.default_rng(0)), default.n_time_steps
    )
    # RVs exist only for the component the inputs carry (its period is a constant).
    assert set(rvs) - {rv.name for rv in default_model.free_RVs} == {
        "treatment_seasonal_amplitude",
        "treatment_seasonal_phase",
    }
    assert set(perturbed_drawn) == set(drawn)
    for name, value in drawn.items():
        np.testing.assert_array_equal(perturbed_drawn[name], value, err_msg=name)
    assert np.abs(drawn["treatment_log_level_shift"]).max() > 0.0  # seasonality really runs


def test_zero_texture_inclusion_reproduces_the_zero_range_corpus():
    """hf/pulse inclusion 0 and a (0, 0) texture range are the same world prior."""
    base = {**_TWIN, "seed": 8, "treatment_cv_floor": 0.08}
    by_probability = make_scm_prior(
        **base,
        **{f"{x}_{c}_inclusion_prob": 0.0 for x in TRAJECTORY_INPUTS for c in ("hf", "pulse")},
    )
    by_range = make_scm_prior(
        **base,
        treatment_hf_sigma_range=(0.0, 0.0),
        treatment_pulse_prob_range=(0.0, 0.0),
        covariate_hf_sigma_range=(0.0, 0.0),
        covariate_pulse_prob_range=(0.0, 0.0),
    )
    # Both leave every treatment flat, which the corpus entry point flags.
    with pytest.warns(FutureWarning, match="flat"):
        probability_corpus = pg.sample_prior_predictive(by_probability)
    with pytest.warns(FutureWarning, match="flat"):
        range_corpus = pg.sample_prior_predictive(by_range)

    assert set(probability_corpus) - set(range_corpus) == set(TRAJECTORY_ARRAY_FIELDS)
    for key, value in range_corpus.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(probability_corpus[key], value, err_msg=key)
    strip = {"timing", "trajectory"}
    assert {k: v for k, v in probability_corpus["diagnostics"].items() if k not in strip} == {
        k: v for k, v in range_corpus["diagnostics"].items() if k not in strip
    }
    texture = [TRAJECTORY_COMPONENTS.index(c) for c in ("hf", "pulse")]
    assert not probability_corpus["treatment_components"][..., texture].any()
    assert not probability_corpus["covariate_components"][..., texture].any()
    _assert_valid(probability_corpus)
    _assert_valid(range_corpus)


def test_default_corpus_carries_no_trajectory_block(default_corpus, shock_twins):
    for corpus in (default_corpus, shock_twins[0]):
        assert not set(TRAJECTORY_ARRAY_FIELDS) & set(corpus)
        assert "trajectory" not in corpus["diagnostics"]


def test_unwired_schedules_reproduce_the_shocks_only_corpus(shock_twins):
    """Unselected schedules leave seeded series and realism filtering unchanged."""
    base, inert = shock_twins
    assert not inert["treatment_components"][..., _SCHEDULE_COLUMNS].any()
    assert not inert["covariate_components"][..., _SCHEDULE_COLUMNS].any()

    assert set(inert) - set(base) == set(TRAJECTORY_ARRAY_FIELDS)
    for key, value in base.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(inert[key], value, err_msg=key)
    strip = {"timing", "trajectory"}
    assert {k: v for k, v in inert["diagnostics"].items() if k not in strip} == {
        k: v for k, v in base["diagnostics"].items() if k not in strip
    }


# -- 2. concrete graph ----------------------------------------------------------

_N_T, _N_C, _N_L, _T, _BURN = 2, 2, 1, 24, 8


def _numpy_world(
    *, g_overrides: dict | None = None, carryover=(1, 2), saturation=(1, 3), seed: int = 7
) -> tuple[dict, dict, dict]:
    """Concrete params/eps for one edge-free (unless overridden) world."""
    rng = np.random.default_rng(seed)
    n_t, n_c, n_l, n_full = _N_T, _N_C, _N_L, _T + _BURN
    g = {key: value.astype("float64") for key, value in _edge_free(n_t, n_c, n_l).items()}
    g.update(
        {key: np.asarray(value, dtype="float64") for key, value in (g_overrides or {}).items()}
    )

    def walk(n: int, positive: bool) -> dict[str, Any]:
        return {
            "mean": rng.uniform(0.2, 0.8, n),
            "std": rng.uniform(0.3, 0.7, n),
            "positive_only": positive,
            "smoothness": rng.uniform(0.1, 0.9, n),
            "rw_smoothness_max_weeks": 26,
        }

    params: dict[str, Any] = {
        "l_max": 8,
        "w_dc": rng.uniform(0.3, 0.6, (n_l, n_t)),
        "u_dz": rng.uniform(0.1, 0.5, (n_l, n_c)),
        "v_zc": rng.uniform(0.3, 0.6, (n_c, n_t)),
        "alpha_cc": np.triu(rng.uniform(0.3, 0.6, (n_t, n_t)), 1),
        "gamma_zz": np.triu(rng.uniform(0.1, 0.3, (n_c, n_c)), 1),
        "delta_dy": rng.uniform(0.1, 0.5, n_l),
        "rho_zy": rng.uniform(0.2, 0.5, n_c),
        "beta": rng.uniform(0.5, 1.5, n_t),
        "carryover_family": np.asarray(carryover, dtype="int64"),
        "sat_family": np.asarray(saturation, dtype="int64"),
        "carryover_alpha": rng.uniform(0.2, 0.7, n_t),
        "weibull_lam": rng.uniform(1.0, 3.0, n_t),
        "weibull_k": rng.uniform(1.0, 3.0, n_t),
        "hill_slope": rng.uniform(1.0, 2.0, n_t),
        "hill_kappa_mult": rng.uniform(0.8, 1.5, n_t),
        "logistic_lam": rng.uniform(0.5, 1.5, n_t),
        "mm_kappa_mult": rng.uniform(0.8, 1.5, n_t),
        "tanh_c": rng.uniform(0.8, 1.5, n_t),
        "root_alpha": rng.uniform(0.3, 0.8, n_t),
        "hf_sigma": rng.uniform(0.05, 0.2, n_t),
        "pulse_amp": rng.uniform(0.1, 0.4, n_t),
        "pulse_prob": rng.uniform(0.1, 0.3, n_t),
        "use_hf": np.ones(n_t, dtype=bool),
        "use_pulse": np.ones(n_t, dtype=bool),
        "covariate_hf_sigma": rng.uniform(0.05, 0.2, n_c),
        "covariate_pulse_amp": rng.uniform(0.1, 0.4, n_c),
        "covariate_pulse_prob": rng.uniform(0.1, 0.3, n_c),
        "use_covariate_hf": np.ones(n_c, dtype=bool),
        "use_covariate_pulse": np.ones(n_c, dtype=bool),
        "rw_d": walk(n_l, False),
        "rw_z": walk(n_c, False),
        "rw_c": walk(n_t, True),
        "rw_b": walk(1, False),
        "rw_y": {"mean": np.zeros(1), "std": rng.uniform(0.2, 0.4, 1), "positive_only": False},
    }
    eps = {
        "eps_d": rng.normal(size=(n_full, n_l)),
        "eps_z": rng.normal(size=(n_full, n_c)),
        "eps_c": rng.normal(size=(n_full, n_t)),
        "eps_b": rng.normal(size=n_full),
        "eps_y": rng.normal(size=n_full),
        "eps_c_hf": rng.normal(size=(n_full, n_t)),
        "eps_c_pulse": rng.integers(0, 2, (n_full, n_t)).astype("float64"),
        "eps_z_hf": rng.normal(size=(n_full, n_c)),
        "eps_z_pulse": rng.integers(0, 2, (n_full, n_c)).astype("float64"),
    }
    return g, params, eps


def _numpy_spec(input_type: str, on: tuple[str, ...] = (), index: int = 0) -> dict[str, Any]:
    """A concrete trajectory spec for two inputs; ``on`` components wired on ``index``."""
    use = {c: np.zeros(2, dtype=bool) for c in TRAJECTORY_COMPONENTS}
    for component in on:
        use[component][index] = True
    factor = np.array([[2.0, 1.5], [0.5, 3.0]])
    return {
        "use": use,
        "onset": {"start": np.array([5, 7])},
        "offset": {"stop": np.array([17, 19])},
        "flighting": {
            "period": np.array([5, 4]),
            "on_weeks": np.array([3, 1]),
            "phase": np.array([2, 0]),
        },
        "level_jump": {
            "count": 2,
            "week": np.array([[4, 6], [15, 18]]),
            **(
                {"factor": factor, "log_factor": np.log(factor)}
                if input_type == "treatment"
                else {"size": np.array([[1.5, -0.7], [-2.0, 0.4]])}
            ),
        },
        "seasonal": {
            "amplitude": np.array([0.4, 0.3]),
            "period": np.array([10.5, 13.0]),
            "phase": np.array([0.7, 2.1]),
        },
        "trend": {"change": np.array([0.8, -1.2])},
    }


def _expected_schedule(
    spec: dict[str, Any], i: int, weeks: np.ndarray, n_time_steps: int, *, log_level: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activity, level shift and (treatments) multiplier of input ``i``, from the spec."""
    t = np.asarray(weeks)
    use = spec["use"]
    activity = np.ones(t.shape, dtype=bool)
    if use["onset"][i]:
        activity &= t >= spec["onset"]["start"][i]
    if use["offset"][i]:
        activity &= t < spec["offset"]["stop"][i]
    if use["flighting"][i]:
        flighting = spec["flighting"]
        cycle = np.mod(t + flighting["phase"][i], flighting["period"][i])
        activity &= cycle < flighting["on_weeks"][i]
    smooth = np.zeros(t.shape)
    if use["seasonal"][i]:
        seasonal = spec["seasonal"]
        smooth += seasonal["amplitude"][i] * np.sin(
            2.0 * np.pi * t / seasonal["period"][i] + seasonal["phase"][i]
        )
    if use["trend"][i]:
        smooth += spec["trend"]["change"][i] * np.clip(t, 0, None) / (n_time_steps - 1)
    shift, multiplier = smooth.copy(), np.exp(smooth)
    if use["level_jump"][i]:
        jump = spec["level_jump"]
        for e in range(jump["count"]):
            after = t >= jump["week"][e, i]
            if log_level:
                shift += np.log(jump["factor"][e, i]) * after
                multiplier *= np.where(after, jump["factor"][e, i], 1.0)
            else:
                shift += jump["size"][e, i] * after
    return activity, shift, multiplier


def _graph(g: dict, params: dict, eps: dict) -> dict[str, np.ndarray]:
    outputs = build_symbolic_graph(g, params, _T, _N_T, _N_C, _N_L, burn_in=_BURN, eps=eps)
    return _evaluate(outputs["outputs"])


def _unwired() -> dict[str, Any]:
    return {"treatment": _numpy_spec("treatment"), "covariate": _numpy_spec("covariate")}


@pytest.fixture(scope="module")
def toggle_arms():
    g, params, eps = _numpy_world()
    return (
        g,
        params,
        eps,
        _graph(g, params, eps),
        _graph(g, {**params, "trajectory": _unwired()}, eps),
    )


def test_an_unwired_trajectory_spec_builds_the_legacy_graph(toggle_arms):
    *_, nospec, off = toggle_arms
    assert set(off) - set(nospec) == {
        "treatment_activity",
        "covariate_activity",
        "treatment_log_level_shift",
        "covariate_level_shift",
        "treatments_natural",
        "outcome_natural",
    }
    for name, value in nospec.items():
        np.testing.assert_array_equal(off[name], value, err_msg=name)
    for name in ("treatment_activity", "covariate_activity"):
        assert off[name].dtype == np.int8
        assert (off[name] == 1).all()
    assert (off["treatment_log_level_shift"] == 0.0).all()
    assert (off["covariate_level_shift"] == 0.0).all()
    np.testing.assert_array_equal(off["treatments_natural"], off["treatments"])
    np.testing.assert_array_equal(off["outcome_natural"], off["outcome"])


#: Outputs a component toggled on input 0 may change: (per-input columns, whole arrays).
_TOUCHED = {
    "treatment": (
        {
            "treatments",
            "treatments_base",
            "contributions",
            "contributions_observed",
            "treatment_activity",
            "treatment_log_level_shift",
        },
        {"outcome"},
    ),
    "covariate": (
        {"covariates", "covariate_contribution", "covariate_activity", "covariate_level_shift"},
        {"baseline", "outcome", "outcome_natural"},
    ),
}


@pytest.mark.parametrize("input_type", TRAJECTORY_INPUTS)
@pytest.mark.parametrize("component", SCHEDULE_COMPONENTS)
def test_one_component_changes_exactly_its_own_input_by_its_equation(
    toggle_arms, input_type, component
):
    g, params, eps, _, off = toggle_arms
    trajectory = _unwired()
    trajectory[input_type] = _numpy_spec(input_type, on=(component,))
    on = _graph(g, {**params, "trajectory": trajectory}, eps)
    activity, shift, multiplier = _expected_schedule(
        trajectory[input_type], 0, np.arange(_T), _T, log_level=input_type == "treatment"
    )
    assert activity.any()
    if component in GATE_COMPONENTS:
        assert not activity.all()
    else:
        assert np.abs(shift).max() > 0.0

    per_column, whole = _TOUCHED[input_type]
    for name, value in off.items():
        if name in whole:
            continue
        if name in per_column:
            np.testing.assert_array_equal(on[name][:, 1:], value[:, 1:], err_msg=name)
        else:
            np.testing.assert_array_equal(on[name], value, err_msg=name)

    np.testing.assert_array_equal(on[f"{input_type}_activity"][:, 0], activity.astype(np.int8))
    shift_name = (
        "treatment_log_level_shift" if input_type == "treatment" else "covariate_level_shift"
    )
    _assert_close(on[shift_name][:, 0], shift)
    if input_type == "treatment":
        series = on["treatments"][:, 0]
        _assert_close(series, np.where(activity, off["treatments"][:, 0] * multiplier, 0.0))
        _assert_close(
            on["treatments_base"][:, 0],
            np.where(activity, off["treatments_base"][:, 0] * multiplier, 0.0),
        )
    else:
        series = on["covariates"][:, 0]
        _assert_close(series, np.where(activity, off["covariates"][:, 0] + shift, 0.0))
        assert (on["covariate_contribution"][~activity, 0] == 0.0).all()
    assert (series[~activity] == 0.0).all()
    _assert_close(on["outcome"], on["baseline"] + on["contributions_observed"].sum(axis=1))


#: Per-input flag and magnitude of each texture component.
_TEXTURE_FIELDS = {
    ("treatment", "hf"): ("use_hf", "hf_sigma"),
    ("treatment", "pulse"): ("use_pulse", "pulse_amp"),
    ("covariate", "hf"): ("use_covariate_hf", "covariate_hf_sigma"),
    ("covariate", "pulse"): ("use_covariate_pulse", "covariate_pulse_amp"),
}


@pytest.mark.parametrize(("input_type", "component"), tuple(_TEXTURE_FIELDS))
def test_a_texture_component_toggles_per_input_and_adds_exactly_zero_when_off(
    toggle_arms, input_type, component
):
    """Dropping hf / pulse from input 0 alone is its term at exactly zero.

    The world carries every texture term on every input; un-flagging input 0
    equals, bit for bit, zeroing that input's magnitude, and no other input moves.
    """
    g, params, eps, _, every = toggle_arms
    flag, magnitude = _TEXTURE_FIELDS[(input_type, component)]
    base = {**params, "trajectory": _unwired()}
    assert np.asarray(base[flag]).all()
    use = np.array(base[flag], dtype=bool)
    use[0] = False
    dropped = _graph(g, {**base, flag: use}, eps)
    silent = np.array(base[magnitude], dtype=float)
    silent[0] = 0.0
    silenced = _graph(g, {**base, magnitude: silent}, eps)
    assert set(dropped) == set(silenced)
    for name, value in dropped.items():
        np.testing.assert_array_equal(value, silenced[name], err_msg=name)
    series = "treatments" if input_type == "treatment" else "covariates"
    assert not np.array_equal(dropped[series][:, 0], every[series][:, 0])
    np.testing.assert_array_equal(dropped[series][:, 1:], every[series][:, 1:])


_MEDIATED_SCHEDULE = ("onset", "level_jump", "seasonal", "trend")
_SHOCK_WEEKS = (1, 2, 10, 11)  # two off-weeks before the onset at 7, two on-weeks
_SHOCK_LEVEL = 0.75


@pytest.fixture(scope="module")
def mediated_arms():
    """Treatment 1 is the only treatment with incoming C->C, Z->C and D->C edges.

    A linear response without carryover makes every decomposition variant
    proportional to its treatment series, so each must scale by the envelope.
    """
    g, params, eps = _numpy_world(
        g_overrides={"g_cc": [[0, 1], [0, 0]], "g_zc": [[0, 1], [0, 0]], "g_dc": [[0, 1]]},
        carryover=(0, 0),
        saturation=(0, 0),
    )
    schedule = {
        "treatment": _numpy_spec("treatment", on=_MEDIATED_SCHEDULE, index=1),
        "covariate": _numpy_spec("covariate"),
    }
    mask = np.zeros((_T + _BURN, _N_T), dtype=np.int8)
    level = np.zeros((_T + _BURN, _N_T))
    for week in _SHOCK_WEEKS:
        mask[_BURN + week, 1] = 1
        level[_BURN + week, 1] = _SHOCK_LEVEL
    shock = {"mask_full": mask, "level_full": level}
    arms = {
        "nospec": params,
        "off": {**params, "trajectory": _unwired()},
        "on": {**params, "trajectory": schedule},
        "off_shock": {**params, "trajectory": _unwired(), "treatment_shock": shock},
        "on_shock": {**params, "trajectory": schedule, "treatment_shock": shock},
    }
    return schedule, {name: _graph(g, arm, eps) for name, arm in arms.items()}


def test_mediated_decomposition_scales_by_the_treatment_envelope(mediated_arms):
    schedule, arms = mediated_arms
    off, on = arms["off"], arms["on"]
    activity, _, multiplier = _expected_schedule(
        schedule["treatment"], 1, np.arange(_T), _T, log_level=True
    )
    envelope = np.where(activity, multiplier, 0.0)
    assert activity.any() and not activity.all()

    for name in ("treatments", "treatments_base", "contributions", "contributions_observed"):
        _assert_close(on[name][:, 1], off[name][:, 1] * envelope, err_msg=name)
        assert (on[name][~activity, 1] == 0.0).all(), name
        np.testing.assert_array_equal(on[name][:, 0], off[name][:, 0], err_msg=name)
    scale = np.abs(off["contributions_observed"]).max()
    for source, label in enumerate(("cc", "zc", "dc")):
        assert np.abs(off["indirect_effects_by_source"][:, source]).max() > 0.0, label
        _assert_close(
            on["indirect_effects_by_source"][:, source],
            off["indirect_effects_by_source"][:, source] * envelope,
            scale=scale,
            err_msg=label,
        )
        assert (on["indirect_effects_by_source"][~activity, source] == 0.0).all(), label
    _assert_close(
        on["outcome"], on["baseline"] + on["contributions"].sum(1) + on["indirect_effects"]
    )
    _assert_close(on["indirect_effects_by_source"].sum(1), on["indirect_effects"], scale=scale)


def test_shocks_override_the_gate_and_natural_paths_stay_schedule_free(mediated_arms):
    schedule, arms = mediated_arms
    activity, _, _ = _expected_schedule(schedule["treatment"], 1, np.arange(_T), _T, log_level=True)
    shocked = np.isin(np.arange(_T), _SHOCK_WEEKS)
    assert (~activity[shocked]).any() and activity[shocked].any()

    on_shock, on, nospec = arms["on_shock"], arms["on"], arms["nospec"]
    # A held level replaces the scheduled value whether the gate is on or off.
    assert (on_shock["treatments"][shocked, 1] == _SHOCK_LEVEL).all()
    np.testing.assert_array_equal(on_shock["treatments"][~shocked], on["treatments"][~shocked])
    # The unshocked recursion carries neither shocks nor the schedule ...
    np.testing.assert_array_equal(on_shock["treatments_unshocked"], nospec["treatments"])
    np.testing.assert_array_equal(on_shock["outcome_unshocked"], nospec["outcome"])
    # ... and with shocks it IS the natural realism pair, wired or not.
    for arm in ("on_shock", "off_shock"):
        np.testing.assert_array_equal(
            arms[arm]["treatments_natural"], arms[arm]["treatments_unshocked"]
        )
        np.testing.assert_array_equal(arms[arm]["outcome_natural"], arms[arm]["outcome_unshocked"])
    # Without shocks: nothing wired -> the observed pair; wired -> the schedule-free recursion.
    np.testing.assert_array_equal(arms["off"]["treatments_natural"], arms["off"]["treatments"])
    np.testing.assert_array_equal(arms["off"]["outcome_natural"], arms["off"]["outcome"])
    np.testing.assert_array_equal(on["treatments_natural"], nospec["treatments"])
    np.testing.assert_array_equal(on["outcome_natural"], nospec["outcome"])
    assert not np.array_equal(on["treatments_natural"][:, 1], on["treatments"][:, 1])


@pytest.fixture(scope="module")
def gated_covariate_arms():
    """Covariate 1 has D->Z and Z->Z parents and the only Z->C edge, to treatment 1."""
    g, params, eps = _numpy_world(
        g_overrides={"g_dz": [[0, 1]], "g_zz": [[0, 1], [0, 0]], "g_zc": [[0, 0], [0, 1]]}
    )
    orphan = {**g, "g_dz": np.zeros_like(g["g_dz"]), "g_zz": np.zeros_like(g["g_zz"])}
    childless = {**g, "g_zc": np.zeros_like(g["g_zc"])}
    schedule = {
        "treatment": _numpy_spec("treatment"),
        "covariate": _numpy_spec("covariate", on=("onset", "flighting", "seasonal"), index=1),
    }
    unwired = {**params, "trajectory": _unwired()}
    gated = {**params, "trajectory": schedule}
    arms = {
        "off": _graph(g, unwired, eps),
        "orphan": _graph(orphan, unwired, eps),
        "on": _graph(g, gated, eps),
        "childless": _graph(childless, gated, eps),
    }
    return schedule, arms


def test_a_covariate_gate_wraps_its_parents_and_silences_its_children(gated_covariate_arms):
    schedule, arms = gated_covariate_arms
    off, on = arms["off"], arms["on"]
    activity, shift, _ = _expected_schedule(
        schedule["covariate"], 1, np.arange(_T), _T, log_level=False
    )
    assert activity.any() and (~activity).any()
    # The parents really drive covariate 1 on its off-weeks ...
    assert (off["covariates"][~activity, 1] != arms["orphan"]["covariates"][~activity, 1]).all()
    # ... and the gate zeroes the assembled covariate, parents included.
    assert (on["covariates"][~activity, 1] == 0.0).all()
    assert (on["covariate_contribution"][~activity, 1] == 0.0).all()
    _assert_close(on["covariates"][:, 1], np.where(activity, off["covariates"][:, 1] + shift, 0.0))
    np.testing.assert_array_equal(on["covariate_activity"][:, 1], activity.astype(np.int8))
    np.testing.assert_array_equal(on["covariates"][:, 0], off["covariates"][:, 0])
    # The Z->C child receives exactly nothing from covariate 1 on its off-weeks.
    child, childless_child = on["treatments"][:, 1], arms["childless"]["treatments"][:, 1]
    np.testing.assert_array_equal(child[~activity], childless_child[~activity])
    assert (child[activity] != childless_child[activity]).all()
    np.testing.assert_array_equal(on["treatments"][:, 0], off["treatments"][:, 0])


# -- 3. component formulas ------------------------------------------------------


def _formula_spec(n_time_steps: int) -> dict[str, Any]:
    """Six treatments, one per gate combination, with level components spread across them."""
    use = {c: np.zeros(6, dtype=bool) for c in TRAJECTORY_COMPONENTS}
    for i in (0, 1, 4):
        use["onset"][i] = True
    for i in (2, 4):
        use["offset"][i] = True
    for i in (3, 4):
        use["flighting"][i] = True
    for i in (0, 3):
        use["seasonal"][i] = True
    for i in (1, 3, 5):
        use["trend"][i] = True
    for i in (2, 3):
        use["level_jump"][i] = True
    factor = np.full((2, 6), 1.0)
    factor[:, 2] = (2.0, 0.75)
    factor[:, 3] = (3.0, 0.5)
    return {
        "use": use,
        "onset": {"start": np.array([0, 6, 0, 0, 3, 0])},
        "offset": {"stop": np.array([n_time_steps] * 2 + [12, n_time_steps, 15, n_time_steps])},
        "flighting": {
            "period": np.array([2, 2, 2, 5, 4, 2]),
            "on_weeks": np.array([1, 1, 1, 3, 1, 1]),
            "phase": np.array([0, 0, 0, 2, 3, 0]),
        },
        "level_jump": {
            "count": 2,
            "week": np.array([[1, 1, 3, 2, 1, 1], [2, 2, 11, 9, 2, 2]]),
            "factor": factor,
            "log_factor": np.log(factor),
            "size": np.log(factor) * 1.7,
        },
        "seasonal": {
            "amplitude": np.full(6, 0.4),
            "period": np.array([7.5, 7.5, 7.5, 6.25, 7.5, 7.5]),
            "phase": np.array([1.1, 0.0, 0.0, 4.0, 0.0, 0.0]),
        },
        "trend": {"change": np.array([0.0, 0.9, 0.0, -0.6, 0.0, -1.3])},
    }


def test_gate_and_level_helpers_follow_their_equations_through_burn_in():
    T, burn = 20, 9
    weeks = time_index(T, burn)
    assert np.array_equal(weeks, np.arange(-burn, T))
    spec = _formula_spec(T)

    activity = {i: activity_column(spec, i, weeks) for i in range(6)}
    levels = {i: level_shift_column(spec, i, weeks, T, log_level=False) for i in range(6)}
    log_levels = {i: level_shift_column(spec, i, weeks, T, log_level=True) for i in range(6)}
    multipliers = {i: treatment_multiplier_column(spec, i, weeks, T) for i in range(6)}
    assert activity[5] is None  # no gate: always on
    assert levels[4] is None and log_levels[4] is None and multipliers[4] is None
    outputs = {
        **{f"a{i}": v for i, v in activity.items() if v is not None},
        **{f"l{i}": v for i, v in levels.items() if v is not None},
        **{f"g{i}": v for i, v in log_levels.items() if v is not None},
        **{f"m{i}": v for i, v in multipliers.items() if v is not None},
    }
    got = _evaluate(outputs)

    for i in range(5):
        expected, _, _ = _expected_schedule(spec, i, weeks, T, log_level=True)
        np.testing.assert_array_equal(got[f"a{i}"].astype(bool), expected, err_msg=f"input {i}")
    burn_in = weeks < 0
    # A wired onset is off for every earlier week, burn-in included (even a launch at 0).
    assert not got["a0"][burn_in].any() and got["a0"][~burn_in].all()
    # Inputs without an onset are on through burn-in.
    assert got["a2"][burn_in].all()
    # Flighting continues its cycle across week -1 -> 0 with a Python-style modulus:
    # at week -3 the phase-shifted index is -1, i.e. 4 (mod 5) -> off.
    cycle = got["a3"].astype(bool)
    assert np.array_equal(cycle[:-5], cycle[5:])
    assert not cycle[weeks == -3][0] and cycle[weeks == -1][0] and cycle[weeks == 0][0]
    # The combined gate is the product of its three gates.
    onset = weeks >= 3
    offset = weeks < 15
    flighting = np.mod(weeks + 3, 4) < 1
    np.testing.assert_array_equal(got["a4"].astype(bool), onset & offset & flighting)

    for i in (0, 1, 2, 3, 5):
        _, level, _ = _expected_schedule(spec, i, weeks, T, log_level=False)
        _, log_level, multiplier = _expected_schedule(spec, i, weeks, T, log_level=True)
        _assert_close(got[f"l{i}"], level, err_msg=f"level {i}")
        _assert_close(got[f"g{i}"], log_level, err_msg=f"log level {i}")
        _assert_close(got[f"m{i}"], multiplier, err_msg=f"multiplier {i}")
    # Trend: exactly zero through burn-in and at week 0, exactly B at the last week.
    for i, change in ((1, 0.9), (5, -1.3)):
        assert (got[f"l{i}"][weeks <= 0] == 0.0).all()
        assert got[f"l{i}"][-1] == change
    # Jump factors multiply directly: an exact product, no exp/log round trip.
    np.testing.assert_array_equal(
        got["m2"], np.where(weeks >= 3, 2.0, 1.0) * np.where(weeks >= 11, 0.75, 1.0)
    )
    assert (got["l2"][weeks < 3] == 0.0).all()


def _flags(n_treatments: int, n_covariates: int, components: tuple[str, ...]) -> dict:
    return {
        structural_key(x, c): np.full(n, c in components)
        for x, n in (("treatment", n_treatments), ("covariate", n_covariates))
        for c in TRAJECTORY_COMPONENTS
    }


def test_level_jumps_fall_one_per_stratified_slot_and_never_at_week_zero():
    T, K = 16, 3
    cfg = make_scm_prior(
        **_PROBE,
        nonlinearity="linear",
        treatment_level_jump_inclusion_prob=1.0,
        treatment_level_jump_count=K,
        covariate_level_jump_inclusion_prob=1.0,
        covariate_level_jump_count=T - 1,
    )
    with pm.Model():
        spec = trajectory_params(cfg, _flags(2, 1, ("level_jump",)), 2, 1, T)
    weeks, covariate_weeks = pm.draw(
        [spec["treatment"]["level_jump"]["week"], spec["covariate"]["level_jump"]["week"]],
        draws=400,
        random_seed=11,
        mode="FAST_COMPILE",
    )
    assert weeks.shape == (400, K, 2)
    for e in range(K):
        lo, hi = 1 + e * (T - 1) // K, 1 + (e + 1) * (T - 1) // K
        # Inside its own slot, and every week of the slot is reachable.
        assert weeks[:, e].min() == lo and weeks[:, e].max() == hi - 1, e
    # count = T - 1 leaves one-week slots: every week after 0 carries exactly one jump.
    assert (covariate_weeks == np.arange(1, T)[None, :, None]).all()


def _scheduled_world(cfg: SCMPrior, names: tuple[str, ...] = (), *, draws: int = 8, seed: int = 21):
    """Draw an edge-free 2-treatment / 1-covariate cell of ``cfg`` (geometric + Hill)."""
    g = _edge_free(2, 1, 1)
    structural = sample_structure(g, cfg, np.random.default_rng(0))
    structural["carryover_family"][:] = CARRYOVER_FAMILY_KEYS.index("geometric")
    structural["sat_family"][:] = SATURATION_FAMILY_KEYS.index("hill")
    model, out_names, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
    extra = tuple(name for name in names if name in model.named_vars)
    assert extra == tuple(names), f"missing model variables: {set(names) - set(extra)}"
    return out_names, draw_worlds(model, out_names + extra, seed=seed, draws=draws)


def test_a_jump_factor_of_three_triples_the_series_exactly():
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 24},
        treatment_level_jump_inclusion_prob=1.0,
        treatment_level_jump_factor_range=(3.0, 3.0),
    )
    _, d = _scheduled_world(cfg, draws=4)
    t = np.arange(cfg.n_time_steps)[None, :, None]
    shift = d["treatment_log_level_shift"]
    tau = np.argmax(shift != 0.0, axis=1)[:, None, :]
    assert (tau >= 1).all()
    np.testing.assert_array_equal(shift, np.where(t >= tau, np.log(3.0), 0.0))
    after = np.broadcast_to(t >= tau, shift.shape)
    natural, series = d["treatments_natural"], d["treatments"]
    np.testing.assert_array_equal(series[after], 3.0 * natural[after])
    np.testing.assert_array_equal(series[~after], natural[~after])


def test_drawn_jump_factors_scale_the_series_by_their_own_log_factor():
    """Log-uniform factors: the series multiplier and the stored log shift must agree."""
    T, K = 24, 2
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": T},
        treatment_level_jump_inclusion_prob=1.0,
        treatment_level_jump_count=K,
    )
    lo, hi = cfg.treatment_level_jump_factor_range
    assert lo < 1.0 < hi
    _, d = _scheduled_world(cfg, ("treatment_level_jump_log_factor", "treatment_level_jump_u"))
    log_factor, u = d["treatment_level_jump_log_factor"], d["treatment_level_jump_u"]
    assert ((np.log(lo) <= log_factor) & (log_factor <= np.log(hi))).all()
    assert log_factor.min() < 0.0 < log_factor.max()
    weeks = np.arange(T)[None, :, None]
    expected = np.zeros_like(d["treatment_log_level_shift"])
    for e in range(K):
        slot_lo, slot_hi = 1 + e * (T - 1) // K, 1 + (e + 1) * (T - 1) // K
        tau = np.minimum(slot_lo + np.floor(u[:, e] * (slot_hi - slot_lo)), slot_hi - 1)
        expected += np.where(weeks >= tau[:, None, :], log_factor[:, e][:, None, :], 0.0)
    _assert_close(d["treatment_log_level_shift"], expected)
    _assert_scheduled_series(d)


def test_jump_factors_are_log_uniform():
    """Half the factors on (0.5, 2.0) shrink the series; a uniform draw would shrink a third."""
    lo, hi = 0.5, 2.0
    cfg = make_scm_prior(
        **_PROBE,
        treatment_level_jump_inclusion_prob=1.0,
        treatment_level_jump_factor_range=(lo, hi),
    )
    with pm.Model():
        spec = trajectory_params(cfg, _flags(2, 1, ("level_jump",)), 2, 1, cfg.n_time_steps)
    factor = pm.draw(
        spec["treatment"]["level_jump"]["factor"], draws=1000, random_seed=3, mode="FAST_COMPILE"
    )
    assert factor.size == 2000 and ((lo <= factor) & (factor <= hi)).all()
    below = -np.log(lo) / (np.log(hi) - np.log(lo))  # P(f < 1) for a log-uniform factor
    sigma = math.sqrt(below * (1.0 - below) / factor.size)
    assert abs((factor < 1.0).mean() - below) < 5 * sigma


def test_flighting_on_run_is_capped_below_its_period():
    for period in range(2, 30):
        for duty in np.linspace(0.01, 1.0, 100):
            on_weeks = int(flighting_on_weeks(duty, period))
            assert on_weeks == _on_run(float(duty), period), (duty, period)
            assert 1 <= on_weeks <= period - 1
    # Round half up on exact halves (banker's rounding would give 2 for all three).
    halves = {(0.5, 5): 3, (0.25, 10): 3, (0.625, 4): 3}
    assert {key: int(flighting_on_weeks(*key)) for key in halves} == halves

    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 24},
        treatment_flighting_inclusion_prob=1.0,
        treatment_flighting_period_weeks_range=(2, 5),
        treatment_flighting_duty_range=(0.9, 1.0),
    )
    with pm.Model() as model:
        spec = trajectory_params(cfg, _flags(2, 1, ("flighting",)), 2, 1, cfg.n_time_steps)
    flighting = spec["treatment"]["flighting"]
    period, on_weeks, phase, duty = pm.draw(
        [
            flighting["period"],
            flighting["on_weeks"],
            flighting["phase"],
            model["treatment_flighting_duty"],
        ],
        draws=300,
        random_seed=5,
        mode="FAST_COMPILE",
    )
    uncapped = np.floor(duty * period + 0.5)
    assert (uncapped >= period).any()  # the cap binds on some draws
    np.testing.assert_array_equal(on_weeks, np.clip(uncapped, 1, period - 1))
    assert ((1 <= on_weeks) & (on_weeks <= period - 1)).all()
    assert ((0 <= phase) & (phase < period)).all()


# -- 4. archetypes --------------------------------------------------------------

_ARCHETYPE_COMPONENTS = {
    "always_on_spikes": ("hf", "pulse"),
    "periodic_on_off": ("hf", "pulse", "flighting"),
    "delayed_start": ("hf", "pulse", "onset"),
    "ramp_up": ("hf", "pulse", "onset", "trend"),
    "decay_to_zero": ("hf", "pulse", "offset", "trend"),
    "level_doubling": ("hf", "pulse", "level_jump"),
    "seasonal": ("hf", "pulse", "seasonal"),
    "trend": ("hf", "pulse", "trend"),
}


def _archetype_probs(name: str) -> dict[str, dict[str, float]]:
    """The effective inclusion table archetype ``name`` must produce."""
    expected = {
        x: {c: float(c in _ARCHETYPE_COMPONENTS[name]) for c in TRAJECTORY_COMPONENTS}
        for x in TRAJECTORY_INPUTS
    }
    if name == "level_doubling":
        # The covariate step reads as a doubling only on a quiet, texture-free level.
        expected["covariate"].update(hf=0.0, pulse=0.0)
    return expected


@pytest.mark.parametrize("name", TRAJECTORY_ARCHETYPES)
def test_archetype_inclusion_table(name):
    assert set(_ARCHETYPE_COMPONENTS) == set(TRAJECTORY_ARCHETYPES)
    cfg = make_scm_prior(**{**_PROBE, "n_time_steps": 32}, trajectories=name)
    assert cfg.trajectory_inclusion_probs() == _archetype_probs(name)


def _archetype(name: str, *extra: str):
    cfg = make_scm_prior(**{**_PROBE, "n_time_steps": 32}, trajectories=name)
    out_names, drawn = _scheduled_world(cfg, extra, seed=2024)
    return cfg, out_names, drawn


_WEEKS = np.arange(32)[None, :, None]


def _launch(cfg: SCMPrior, d: dict, x: str) -> np.ndarray:
    start = np.floor(d[f"{x}_onset_frac"] * cfg.n_time_steps).astype(int)[:, None, :]
    lo, hi = getattr(cfg, f"{x}_onset_frac_range")
    assert (start >= max(1, math.floor(lo * cfg.n_time_steps))).all()
    assert (start <= math.floor(hi * cfg.n_time_steps)).all()
    return start


def _trend(cfg: SCMPrior, d: dict, x: str, shift_name: str) -> np.ndarray:
    change = d[f"{x}_trend_change"][:, None, :]
    range_name = (
        "treatment_trend_log_change_range" if x == "treatment" else "covariate_trend_change_range"
    )
    lo, hi = getattr(cfg, range_name)
    assert ((lo <= change) & (change <= hi)).all()
    shift = d[shift_name]
    _assert_close(shift, change * _WEEKS / (cfg.n_time_steps - 1), err_msg=shift_name)
    assert (shift[:, 0] == 0.0).all() and (shift[:, -1] == change[:, 0]).all()
    return change


def _assert_scheduled_series(d: dict) -> None:
    """Treatments are the natural series under the drawn envelope, exactly 0 when off."""
    activity = d["treatment_activity"].astype(bool)
    expected = np.where(
        activity, d["treatments_natural"] * np.exp(d["treatment_log_level_shift"]), 0.0
    )
    _assert_close(d["treatments"], expected)
    assert (d["treatments"][~activity] == 0.0).all()


def test_always_on_spikes_never_gate_and_every_input_pulses():
    cfg, out_names, d = _archetype(
        "always_on_spikes", "pulse_prob", "covariate_pulse_prob", "eps_c_pulse", "eps_z_pulse"
    )
    # Nothing can switch an input off: no gate or level output exists at all.
    assert not set(_CORPUS_TRAJECTORY_NAMES) & set(out_names)
    burn = cfg.carryover_burn_in
    for x, prob, fire in (
        ("treatment", "pulse_prob", "eps_c_pulse"),
        ("covariate", "covariate_pulse_prob", "eps_z_pulse"),
    ):
        assert (d[prob] >= getattr(cfg, f"{x}_pulse_prob_range")[0]).all()
        assert getattr(cfg, f"{x}_pulse_prob_range")[0] > 0.0
        assert (d[fire][:, burn:].sum(axis=(0, 1)) > 0).all(), x


def test_periodic_on_off_follows_its_drawn_flighting_cycle():
    names = tuple(
        f"{x}_flighting_{p}" for x in TRAJECTORY_INPUTS for p in ("period", "duty", "phase_u")
    )
    cfg, _, d = _archetype("periodic_on_off", *names)
    for x in TRAJECTORY_INPUTS:
        period = d[f"{x}_flighting_period"]
        p_lo, p_hi = getattr(cfg, f"{x}_flighting_period_weeks_range")
        assert ((p_lo <= period) & (period <= p_hi)).all()
        on_weeks = np.clip(np.floor(d[f"{x}_flighting_duty"] * period + 0.5), 1, period - 1)
        phase = np.minimum(np.floor(d[f"{x}_flighting_phase_u"] * period), period - 1)
        expected = np.mod(_WEEKS + phase[:, None, :], period[:, None, :]) < on_weeks[:, None, :]
        np.testing.assert_array_equal(d[f"{x}_activity"], expected.astype(np.int8), err_msg=x)
        assert expected.any(axis=1).all() and (~expected).any(axis=1).all()
    activity = d["treatment_activity"].astype(bool)
    np.testing.assert_array_equal(d["treatments"], np.where(activity, d["treatments_natural"], 0.0))
    off = ~d["covariate_activity"].astype(bool)
    assert (d["covariates"][off] == 0.0).all() and (d["covariate_contribution"][off] == 0.0).all()
    assert (d["treatment_log_level_shift"] == 0.0).all()
    assert (d["covariate_level_shift"] == 0.0).all()


def test_delayed_start_is_zero_before_launch_including_geometric_carryover():
    cfg, _, d = _archetype("delayed_start", "treatment_onset_frac", "covariate_onset_frac")
    start = _launch(cfg, d, "treatment")
    launched = np.broadcast_to(_WEEKS >= start, d["treatments"].shape)
    np.testing.assert_array_equal(d["treatment_activity"], launched.astype(np.int8))
    for name in ("treatments", "contributions", "contributions_observed"):
        assert (d[name][~launched] == 0.0).all(), name
    first_week = np.take_along_axis(d["contributions_observed"], start, axis=1)
    assert (first_week > 0.0).all()
    np.testing.assert_array_equal(d["treatments"][launched], d["treatments_natural"][launched])

    covariate_start = _launch(cfg, d, "covariate")
    covariate_launched = np.broadcast_to(_WEEKS >= covariate_start, d["covariates"].shape)
    np.testing.assert_array_equal(d["covariate_activity"], covariate_launched.astype(np.int8))
    assert (d["covariates"][~covariate_launched] == 0.0).all()
    assert (d["covariate_contribution"][~covariate_launched] == 0.0).all()
    assert (d["covariate_contribution"][covariate_launched] != 0.0).all()


def test_ramp_up_launches_late_then_climbs():
    names = (
        "treatment_onset_frac",
        "treatment_trend_change",
        "covariate_onset_frac",
        "covariate_trend_change",
    )
    cfg, _, d = _archetype("ramp_up", *names)
    for x, shift_name in (
        ("treatment", "treatment_log_level_shift"),
        ("covariate", "covariate_level_shift"),
    ):
        start = _launch(cfg, d, x)
        np.testing.assert_array_equal(
            d[f"{x}_activity"],
            np.broadcast_to(_WEEKS >= start, d[f"{x}_activity"].shape).astype(np.int8),
        )
        assert (_trend(cfg, d, x, shift_name) > 0.0).all()
        assert (np.diff(d[shift_name], axis=1) > 0.0).all()
    _assert_scheduled_series(d)


def test_decay_to_zero_stops_but_carryover_outlives_the_stop():
    names = (
        "treatment_offset_frac",
        "treatment_trend_change",
        "covariate_offset_frac",
        "covariate_trend_change",
    )
    cfg, _, d = _archetype("decay_to_zero", *names)
    T = cfg.n_time_steps
    for x, series, shift_name in (
        ("treatment", "treatments", "treatment_log_level_shift"),
        ("covariate", "covariates", "covariate_level_shift"),
    ):
        stop = np.floor(d[f"{x}_offset_frac"] * T).astype(int)[:, None, :]
        lo, hi = getattr(cfg, f"{x}_offset_frac_range")
        assert (stop >= math.floor(lo * T)).all() and (stop <= T - 1).all()
        running = np.broadcast_to(_WEEKS < stop, d[series].shape)
        np.testing.assert_array_equal(d[f"{x}_activity"], running.astype(np.int8))
        assert (d[series][~running] == 0.0).all()
        assert (_trend(cfg, d, x, shift_name) < 0.0).all()
        if x == "treatment":
            # Geometric carryover memory outlives the stop.
            assert (np.take_along_axis(d["contributions_observed"], stop, axis=1) > 0.0).all()
    _assert_scheduled_series(d)


def test_level_doubling_doubles_treatments_and_steps_covariates_by_one():
    cfg, _, d = _archetype("level_doubling", "treatment_level_jump_u", "covariate_level_jump_u")
    T = cfg.n_time_steps
    for x, shift_name, size in (
        ("treatment", "treatment_log_level_shift", np.log(2.0)),
        ("covariate", "covariate_level_shift", 1.0),
    ):
        # One jump in the single slot [1, T): tau = 1 + floor(u·(T - 1)).
        u = d[f"{x}_level_jump_u"][:, 0, :]
        tau = np.minimum(1 + np.floor(u * (T - 1)), T - 1).astype(int)[:, None, :]
        np.testing.assert_array_equal(d[shift_name], np.where(_WEEKS >= tau, size, 0.0), err_msg=x)
        assert (d[f"{x}_activity"] == 1).all()
        if x == "treatment":
            after = np.broadcast_to(_WEEKS >= tau, d["treatments"].shape)
            assert after.any() and (~after).any()
            np.testing.assert_array_equal(
                d["treatments"][after], 2.0 * d["treatments_natural"][after]
            )
            np.testing.assert_array_equal(d["treatments"][~after], d["treatments_natural"][~after])
        else:
            # A pinned level of 1 with a quiet walk: the step reads as a doubling.
            after = np.broadcast_to(_WEEKS >= tau, d["covariates"].shape)
            assert (np.abs(d["covariates"][~after] - 1.0) < 0.5).all()
            assert (np.abs(d["covariates"][after] - 2.0) < 0.5).all()


@pytest.fixture(scope="module")
def level_doubling_corpus():
    """Two covariates and two latents per cell, so D->Z and Z->Z edges would be eligible."""
    cfg = make_scm_prior(
        n_treatments=1,
        n_covariates=2,
        n_latent=2,
        n_time_steps=16,
        n_cells=4,
        draws_per_cell=1,
        seed=1,
        trajectories="level_doubling",
    )
    return cfg, pg.sample_prior_predictive(cfg)


def test_level_doubling_corpus_keeps_covariates_parentless_at_a_doubling_level(
    level_doubling_corpus,
):
    cfg, corpus = level_doubling_corpus
    _assert_valid(corpus)
    for edge_type in ("dz", "zz"):
        assert not corpus["g"][:, cfg.layout.slices[edge_type]].any(), edge_type
    # Four cells could miss a small leftover edge rate; 200 structure draws would not.
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = sample_g_additive(
            rng,
            cfg,
            cfg.layout,
            n_treatments_active=cfg.n_treatments,
            n_covariates_active=cfg.n_covariates,
            n_latent_active=cfg.n_latent,
        )
        assert not g["g_dz"].any() and not g["g_zz"].any()
    covariates = corpus["covariates"].astype(np.float64)
    shift = corpus["covariate_level_shift"]
    before, after = shift == 0.0, shift == 1.0
    assert (before | after).all() and before.any() and after.any()
    assert (np.abs(covariates[before] - 1.0) < 0.5).all()
    assert (np.abs(covariates[after] - 2.0) < 0.5).all()


def test_seasonal_archetype_is_a_sinusoid_of_its_drawn_phase():
    names = tuple(f"{x}_seasonal_{p}" for x in TRAJECTORY_INPUTS for p in ("amplitude", "phase"))
    cfg, _, d = _archetype("seasonal", *names)
    for x, shift_name in (
        ("treatment", "treatment_log_level_shift"),
        ("covariate", "covariate_level_shift"),
    ):
        p_lo, p_hi = getattr(cfg, f"{x}_seasonal_period_weeks_range")
        assert p_lo == p_hi <= cfg.n_time_steps / 2  # at least two full cycles per series
        amplitude = d[f"{x}_seasonal_amplitude"][:, None, :]
        phase = d[f"{x}_seasonal_phase"][:, None, :]
        assert ((0.0 <= phase) & (phase < 2.0 * np.pi)).all(), x
        expected = amplitude * np.sin(2.0 * np.pi * _WEEKS / p_lo + phase)
        _assert_close(d[shift_name], expected, err_msg=x)
    # Phases span the whole cycle, not just its first half.
    assert max(d[f"{x}_seasonal_phase"].max() for x in TRAJECTORY_INPUTS) > np.pi
    _assert_scheduled_series(d)


def test_trend_archetype_is_affine_in_the_reported_week():
    cfg, _, d = _archetype("trend", "treatment_trend_change", "covariate_trend_change")
    _trend(cfg, d, "treatment", "treatment_log_level_shift")
    _trend(cfg, d, "covariate", "covariate_level_shift")
    assert (d["treatment_activity"] == 1).all() and (d["covariate_activity"] == 1).all()
    _assert_scheduled_series(d)


@pytest.mark.parametrize("name", TRAJECTORY_ARCHETYPES)
def test_archetype_corpus_stores_exactly_its_components(request, name):
    if name == "level_doubling":
        _, corpus = request.getfixturevalue("level_doubling_corpus")
    else:
        cfg = make_scm_prior(
            n_treatments=1,
            n_covariates=1,
            n_latent=1,
            n_time_steps=32,
            n_cells=2,
            draws_per_cell=1,
            seed=3,
            trajectories=name,
        )
        corpus = pg.sample_prior_predictive(cfg)
    _assert_valid(corpus)
    if name == "always_on_spikes":
        # Exactly the legacy all-or-none texture: no trajectory knob moves, so no block.
        assert not set(TRAJECTORY_ARRAY_FIELDS) & set(corpus)
        assert "trajectory" not in corpus["diagnostics"]
        return
    expected = _archetype_probs(name)
    for x in TRAJECTORY_INPUTS:
        flags = corpus[f"{x}_components"][corpus[f"{x}_active_mask"] == 1]
        row = np.array([expected[x][c] for c in TRAJECTORY_COMPONENTS], dtype=np.uint8)
        np.testing.assert_array_equal(flags, np.broadcast_to(row, flags.shape), err_msg=x)
    # All-or-none inclusion: realised prevalence is the inclusion table itself.
    assert corpus["diagnostics"]["trajectory"]["prevalence"] == expected


# -- 5. signs -------------------------------------------------------------------


def test_extreme_negative_envelopes_keep_treatments_non_negative_and_covariates_signed():
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 32},
        treatment_flighting_inclusion_prob=1.0,
        treatment_flighting_period_weeks_range=(4, 6),
        treatment_trend_inclusion_prob=1.0,
        treatment_trend_log_change_range=(-1.5, -1.5),
        treatment_level_jump_inclusion_prob=1.0,
        treatment_level_jump_factor_range=(0.5, 0.5),
        treatment_seasonal_inclusion_prob=1.0,
        treatment_seasonal_amplitude_range=(0.5, 0.5),
        covariate_level_jump_inclusion_prob=1.0,
        covariate_level_jump_size_range=(-3.0, -3.0),
        rw_covariate_mean_range=(1.0, 1.0),
        covariate_hf_inclusion_prob=0.0,
        covariate_pulse_inclusion_prob=0.0,
        rw_std_sigma=0.05,
        rw_baseline_std_sigma=1.0,
    )
    _, d = _scheduled_world(cfg)
    activity = d["treatment_activity"].astype(bool)
    assert activity.any() and (~activity).any()
    assert d["treatment_log_level_shift"].min() < -2.0  # the envelope really is extreme
    treatments = d["treatments"]
    assert (treatments >= 0.0).all()
    assert (treatments[activity] > 0.0).all() and (treatments[~activity] == 0.0).all()
    _assert_scheduled_series(d)

    shift = d["covariate_level_shift"]
    tau = np.argmax(shift != 0.0, axis=1)[:, None, :]
    after = np.broadcast_to(_WEEKS >= tau, shift.shape)
    np.testing.assert_array_equal(shift, np.where(after, -3.0, 0.0))
    assert (d["covariates"][~after] > 0.0).all() and (d["covariates"][after] < 0.0).all()


# -- 6. corpus block and prevalence ---------------------------------------------

_COMPOSABLE: dict[str, Any] = {
    "n_treatments": 3,
    "n_covariates": 2,
    "n_latent": 1,
    "n_time_steps": 24,
    "n_cells": 3,
    "draws_per_cell": 2,
    "n_treatments_active_range": (1, 3),
    "n_covariates_active_range": (1, 2),
    "seed": 7,
    "trajectories": "composable",
    **{f"{x}_{c}_inclusion_prob": 0.5 for x in TRAJECTORY_INPUTS for c in TRAJECTORY_COMPONENTS},
}
_TRUNCATED_N = 3


@pytest.fixture(scope="module")
def composable_corpus():
    """Every component at a fractional probability, with padded input slots."""
    cfg = make_scm_prior(**_COMPOSABLE)
    return cfg, pg.sample_prior_predictive(cfg)


@pytest.fixture(scope="module")
def truncated_twins():
    """Two independent same-seed generations truncated with ``n=``."""
    cfg = make_scm_prior(**_COMPOSABLE)
    return [pg.sample_prior_predictive(cfg, n=_TRUNCATED_N) for _ in range(2)]


def _prevalence(corpus: dict) -> tuple[dict, dict]:
    """Recompute the block's prevalence and input counts from the stored arrays."""
    prevalence, n_inputs = {}, {}
    for x in TRAJECTORY_INPUTS:
        active = corpus[f"{x}_active_mask"] == 1
        count = int(active.sum())
        flags = corpus[f"{x}_components"][active]
        n_inputs[x] = count
        prevalence[x] = {
            c: int(flags[:, k].sum()) / count for k, c in enumerate(TRAJECTORY_COMPONENTS)
        }
    return prevalence, n_inputs


def test_prevalence_counts_active_input_slots_only(composable_corpus):
    _, corpus = composable_corpus
    for x in TRAJECTORY_INPUTS:
        active = corpus[f"{x}_active_mask"] == 1
        assert (~active).any(), f"the fixture needs padded {x} slots"
        flags = corpus[f"{x}_components"][active]
        assert (flags.min(axis=0) == 0).all() and (flags.max(axis=0) == 1).all()
    prevalence, n_inputs = _prevalence(corpus)
    block = corpus["diagnostics"]["trajectory"]
    assert block["prevalence"] == prevalence
    assert block["n_inputs"] == n_inputs


def _expected_inclusion(cfg: SCMPrior) -> dict[str, dict[str, float]]:
    """Effective inclusion from the raw fields: hf/pulse are 0 while their texture range is off."""
    live = {"hf": "hf_sigma_range", "pulse": "pulse_prob_range"}
    return {
        x: {
            c: (
                0.0
                if c in live and getattr(cfg, f"{x}_{live[c]}")[1] == 0.0
                else float(getattr(cfg, f"{x}_{c}_inclusion_prob"))
            )
            for c in TRAJECTORY_COMPONENTS
        }
        for x in TRAJECTORY_INPUTS
    }


def test_trajectory_diagnostics_echo_effective_probabilities_as_plain_numbers(composable_corpus):
    """Key order is not pinned here; same-seed byte identity covers its determinism."""
    cfg, corpus = composable_corpus
    block = corpus["diagnostics"]["trajectory"]
    # The layout list names the stored component axis.
    assert block["components"] == list(TRAJECTORY_COMPONENTS)
    assert block["inclusion_probs"] == _expected_inclusion(cfg)
    for key in ("inclusion_probs", "prevalence"):
        assert all(type(p) is float for probs in block[key].values() for p in probs.values())
    assert all(type(count) is int for count in block["n_inputs"].values())


def test_raw_prior_echoes_effective_texture_inclusion():
    """A raw SCMPrior keeps its texture ranges off, so hf/pulse inclusion is 0 whatever the knob."""
    cfg = SCMPrior(
        **_PROBE,
        n_cells=2,
        draws_per_cell=1,
        seed=3,
        n_treatments_active_range=(2, 2),
        n_covariates_active_range=(1, 1),
        n_latent_active_range=(1, 1),
        treatment_trend_inclusion_prob=0.5,
    )
    assert cfg.treatment_hf_inclusion_prob == 1.0 and cfg.treatment_hf_sigma_range[1] == 0.0
    corpus = pg.sample_prior_predictive(cfg)
    _assert_valid(corpus)
    block = corpus["diagnostics"]["trajectory"]
    assert block["inclusion_probs"] == _expected_inclusion(cfg)
    assert block["inclusion_probs"]["treatment"]["hf"] == 0.0
    assert block["inclusion_probs"]["treatment"]["trend"] == 0.5
    texture = [TRAJECTORY_COMPONENTS.index(c) for c in ("hf", "pulse")]
    for x in TRAJECTORY_INPUTS:
        assert not corpus[f"{x}_components"][..., texture].any()
    assert block["prevalence"] == _prevalence(corpus)[0]


def test_truncation_recomputes_prevalence_from_the_retained_rows(
    composable_corpus, truncated_twins
):
    _, full = composable_corpus
    truncated = truncated_twins[0]
    assert truncated["treatment_raw"].shape[0] == _TRUNCATED_N
    full_block, block = full["diagnostics"]["trajectory"], truncated["diagnostics"]["trajectory"]
    assert block["prevalence"] != full_block["prevalence"]
    prevalence, n_inputs = _prevalence(truncated)
    assert block["prevalence"] == prevalence and block["n_inputs"] == n_inputs
    for key in TRAJECTORY_ARRAY_FIELDS:
        np.testing.assert_array_equal(truncated[key], full[key][:_TRUNCATED_N], err_msg=key)


def test_trajectory_block_round_trips_through_save_and_load(tmp_path, composable_corpus):
    _, corpus = composable_corpus
    path = tmp_path / "composable.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key, (_, dtype) in TRAJECTORY_ARRAY_FIELDS.items():
        assert loaded[key].dtype == dtype, key
        np.testing.assert_array_equal(loaded[key], corpus[key], err_msg=key)
    assert loaded["diagnostics"]["trajectory"] == corpus["diagnostics"]["trajectory"]
    _assert_valid(loaded)


def test_same_seed_shards_with_the_trajectory_block_are_byte_identical(tmp_path, truncated_twins):
    """Independent same-seed generations persist identically despite different timings."""
    paths = []
    for i, corpus in enumerate(truncated_twins):
        paths.append(tmp_path / f"shard_{i}.npz")
        pg.save_corpus(corpus, paths[-1])
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_every_gated_input_keeps_two_on_weeks_in_its_support_window(composable_corpus):
    _, corpus = composable_corpus
    gates = [TRAJECTORY_COMPONENTS.index(c) for c in GATE_COMPONENTS]
    support = corpus["support_mask"] == 1
    checked = 0
    for x in TRAJECTORY_INPUTS:
        flags, activity = corpus[f"{x}_components"], corpus[f"{x}_activity"]
        active = corpus[f"{x}_active_mask"] == 1
        for row, i in zip(*np.nonzero(active)):
            if flags[row, i, gates].any():
                assert activity[row, support[row], i].sum() >= 2, (x, row, i)
                checked += 1
    assert checked > 0


def test_level_components_leave_a_stored_shift_on_exactly_their_inputs(composable_corpus):
    _, corpus = composable_corpus
    levels = [TRAJECTORY_COMPONENTS.index(c) for c in LEVEL_COMPONENTS]
    for x, shift_key in (
        ("treatment", "treatment_log_level_shift"),
        ("covariate", "covariate_level_shift"),
    ):
        active = corpus[f"{x}_active_mask"] == 1
        levelled = corpus[f"{x}_components"][..., levels].any(axis=-1) & active
        assert levelled.any() and (active & ~levelled).any(), x
        shifted = (corpus[shift_key] != 0.0).any(axis=1)
        np.testing.assert_array_equal(shifted, levelled, err_msg=x)


def test_onset_flag_closes_week_zero_for_inputs_without_flighting(composable_corpus):
    """A carried onset always delays by >= 1 week; offsets stop late, so week 0 reads the onset."""
    _, corpus = composable_corpus
    onset = TRAJECTORY_COMPONENTS.index("onset")
    flighting = TRAJECTORY_COMPONENTS.index("flighting")
    seen = set()
    for x in TRAJECTORY_INPUTS:
        flags, activity = corpus[f"{x}_components"], corpus[f"{x}_activity"]
        active = corpus[f"{x}_active_mask"] == 1
        for row, i in zip(*np.nonzero(active)):
            if flags[row, i, flighting]:
                continue
            carried = bool(flags[row, i, onset])
            assert carried == (activity[row, 0, i] == 0), (x, row, i)
            seen.add(carried)
    assert seen == {True, False}


_METADATA_ONLY: dict[str, Any] = {
    "n_treatments": 3,
    "n_covariates": 2,
    "n_latent": 1,
    "n_time_steps": 16,
    "n_cells": 3,
    "draws_per_cell": 2,
    "n_treatments_active_range": (1, 3),
    "n_covariates_active_range": (1, 2),
    "seed": 2,
    "treatment_hf_inclusion_prob": 0.5,
    "covariate_pulse_inclusion_prob": 0.5,
}


@pytest.fixture(scope="module")
def metadata_corpus():
    """Fractional texture inclusion and no schedule component, with padded input slots."""
    return pg.sample_prior_predictive(make_scm_prior(**_METADATA_ONLY))


def test_metadata_only_block_keeps_active_inputs_on_and_unshifted(metadata_corpus):
    corpus = metadata_corpus
    for x, shift_key, varied in (
        ("treatment", "treatment_log_level_shift", "hf"),
        ("covariate", "covariate_level_shift", "pulse"),
    ):
        active = corpus[f"{x}_active_mask"] == 1
        assert (~active).any(), f"the fixture needs padded {x} slots"
        flags = corpus[f"{x}_components"]
        assert not flags[~active].any() and not flags[..., _SCHEDULE_COLUMNS].any()
        carried = flags[..., TRAJECTORY_COMPONENTS.index(varied)][active]
        assert carried.any() and not carried.all()
        activity = corpus[f"{x}_activity"]
        np.testing.assert_array_equal(
            activity, np.broadcast_to(active[:, None, :], activity.shape).astype(np.uint8)
        )
        assert (corpus[shift_key] == 0.0).all()
    assert corpus["diagnostics"]["trajectory"]["prevalence"] == _prevalence(corpus)[0]


#: Every corpus a module fixture generates, as a list of corpora.
_FIXTURE_CORPORA = {
    "default": lambda request: [request.getfixturevalue("default_corpus")],
    "shock_twins": lambda request: request.getfixturevalue("shock_twins"),
    "composable": lambda request: [request.getfixturevalue("composable_corpus")[1]],
    "truncated": lambda request: request.getfixturevalue("truncated_twins"),
    "metadata_only": lambda request: [request.getfixturevalue("metadata_corpus")],
}


@pytest.mark.parametrize("name", _FIXTURE_CORPORA)
def test_fixture_corpora_pass_validate_corpus(request, name):
    for corpus in _FIXTURE_CORPORA[name](request):
        _assert_valid(corpus)


def test_one_level_jump_in_every_reported_week_generates_a_valid_corpus():
    """K = T - 1 jumps per input next to every other level form and flighting.

    Guards the jump wiring against the FAST_COMPILE backend's 32-operand limit on
    one fused elementwise op.
    """
    T = 104
    components = ("flighting", "level_jump", "seasonal", "trend")
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=T,
        n_cells=2,
        draws_per_cell=1,
        seed=4,
        **{f"{x}_{c}_inclusion_prob": 1.0 for x in TRAJECTORY_INPUTS for c in components},
        treatment_level_jump_count=T - 1,
        covariate_level_jump_count=T - 1,
        # 0.5 + 1.0 + 103·|log 0.99| stays inside the 3.0 treatment swing bound.
        treatment_level_jump_factor_range=(0.99, 1.01),
        covariate_level_jump_size_range=(-0.05, 0.05),
    )
    corpus = pg.sample_prior_predictive(cfg)
    _assert_valid(corpus)
    columns = [TRAJECTORY_COMPONENTS.index(c) for c in components]
    for x in TRAJECTORY_INPUTS:
        flags = corpus[f"{x}_components"][corpus[f"{x}_active_mask"] == 1]
        assert flags[:, columns].all()


# -- 7. flag streams ------------------------------------------------------------

_ALL_KNOBS = tuple(
    f"{x}_{c}_inclusion_prob" for x in TRAJECTORY_INPUTS for c in TRAJECTORY_COMPONENTS
)


def _flag_cfg(**probs: float) -> SCMPrior:
    base = make_scm_prior(
        n_treatments=2, n_covariates=2, n_latent=1, n_time_steps=16, **_SHORT_GATES
    )
    return replace(base, **probs)


def _knob_key(knob: str) -> str:
    x, component = knob.removesuffix("_inclusion_prob").split("_", 1)
    return structural_key(x, component)


def _draw_flags(cfg: SCMPrior, seed: int = 3, n: int = 64) -> dict[str, np.ndarray]:
    return sample_component_flags(cfg, n, n, np.random.default_rng(seed))


def test_each_component_flag_stream_ignores_every_other_probability():
    half = _flag_cfg(**dict.fromkeys(_ALL_KNOBS, 0.5))
    reference = _draw_flags(half)
    for knob in _ALL_KNOBS:
        key = _knob_key(knob)
        assert 0 < reference[key].sum() < reference[key].size, key
        for other in _ALL_KNOBS:
            if other == knob:
                continue
            for value in (0.0, 0.3, 1.0):
                flags = _draw_flags(replace(half, **{other: value}))
                np.testing.assert_array_equal(
                    flags[key], reference[key], err_msg=f"{key} vs {other}"
                )
        # Alone as the only fractional knob, with every other one at 0 or 1.
        for value in (0.0, 1.0):
            solo = _flag_cfg(**{**dict.fromkeys(_ALL_KNOBS, value), knob: 0.5})
            np.testing.assert_array_equal(_draw_flags(solo)[key], reference[key], err_msg=key)


def test_flags_grow_monotonically_with_their_probability():
    low = _draw_flags(_flag_cfg(**dict.fromkeys(_ALL_KNOBS, 0.3)), n=200)
    high = _draw_flags(_flag_cfg(**dict.fromkeys(_ALL_KNOBS, 0.6)), n=200)
    for key in low:
        assert (high[key] | ~low[key]).all(), key  # low ⊆ high
        assert (high[key] & ~low[key]).any(), key


@pytest.mark.parametrize(
    "probs, fractional",
    (
        ({}, False),
        (dict.fromkeys(_ALL_KNOBS, 1.0), False),
        (dict.fromkeys(_ALL_KNOBS, 0.0), False),
        ({"covariate_trend_inclusion_prob": 0.2}, True),
        ({"treatment_hf_inclusion_prob": 0.7}, True),
    ),
    ids=("default", "all_one", "all_zero", "one_schedule", "one_texture"),
)
def test_flag_draws_spawn_one_child_and_leave_the_main_stream(probs, fractional):
    cfg = _flag_cfg(**probs)
    rng = np.random.default_rng(17)
    state = rng.bit_generator.state
    first = sample_component_flags(cfg, 32, 32, rng)
    assert rng.bit_generator.state == state
    assert rng.bit_generator.seed_seq.n_children_spawned == int(fractional)
    second = sample_component_flags(cfg, 32, 32, rng)
    assert rng.bit_generator.seed_seq.n_children_spawned == 2 * int(fractional)
    expected_keys = {
        structural_key(x, c)
        for x in TRAJECTORY_INPUTS
        for c in (TRAJECTORY_COMPONENTS if cfg.trajectory_components_enabled else ("hf", "pulse"))
    }
    assert set(first) == set(second) == expected_keys
    # Consecutive cells draw fresh flags; probabilities of 0 or 1 draw constants.
    mixed = [key for key in first if 0 < first[key].sum() < first[key].size]
    assert bool(mixed) is fractional
    for key in mixed:
        assert not np.array_equal(first[key], second[key]), key


# Large-N but numpy-only: it runs in well under a second.
def test_flag_marginals_and_pairwise_joints_match_independent_bernoullis():
    probs = dict(zip(_ALL_KNOBS, np.linspace(0.15, 0.85, len(_ALL_KNOBS))))
    n = 20_000
    flags = sample_component_flags(_flag_cfg(**probs), n, n, np.random.default_rng(2026))
    keys = [_knob_key(knob) for knob in _ALL_KNOBS]
    p = {key: float(probs[knob]) for key, knob in zip(keys, _ALL_KNOBS)}
    for key in keys:
        sigma = math.sqrt(p[key] * (1 - p[key]) / n)
        assert abs(flags[key].mean() - p[key]) < 5 * sigma, key
    # Treatment i is paired with covariate i, so cross-input independence is tested too.
    for a, key_a in enumerate(keys):
        for key_b in keys[a + 1 :]:
            joint = p[key_a] * p[key_b]
            sigma = math.sqrt(joint * (1 - joint) / n)
            observed = (flags[key_a] & flags[key_b]).mean()
            assert abs(observed - joint) < 5 * sigma, (key_a, key_b)


# -- 8. config validation and presets -------------------------------------------

_RULES = {
    "n_treatments": 2,
    "n_covariates": 1,
    "n_latent": 1,
    "n_time_steps": 32,
    "nonlinearity": "linear",
}
_F32_MAX = float(np.finfo(np.float32).max)
#: A full treatment trend multiplies the level by e^3; one covariate jump adds 0.6 max.
_TREATMENT_REACH = {
    "treatment_trend_inclusion_prob": 1.0,
    "treatment_trend_log_change_range": (3.0, 3.0),
}
_COVARIATE_JUMP = {
    "covariate_level_jump_inclusion_prob": 1.0,
    "covariate_level_jump_count": 1,
    "covariate_level_jump_size_range": (0.6 * _F32_MAX, 0.6 * _F32_MAX),
}


@pytest.mark.parametrize(
    "overrides, match",
    (
        ({"treatment_onset_inclusion_prob": True}, "must be a probability"),
        ({"covariate_hf_inclusion_prob": 1.5}, "must be a probability"),
        ({"treatment_trend_inclusion_prob": float("nan")}, "must be a probability"),
        ({"covariate_seasonal_inclusion_prob": -0.25}, "must be a probability"),
        ({"treatment_pulse_inclusion_prob": "1.0"}, "must be a probability"),
        ({"treatment_onset_frac_range": (0.1, 1.0)}, "a launch must fall inside"),
        ({"covariate_onset_frac_range": (-0.1, 0.2)}, "covariate_onset_frac_range has invalid"),
        ({"treatment_offset_frac_range": (0.0, 0.5)}, "treatment_offset_frac_range has invalid"),
        ({"covariate_offset_frac_range": (0.5, 1.5)}, "covariate_offset_frac_range has invalid"),
        ({"treatment_flighting_period_weeks_range": (1, 4)}, "integer bounds"),
        ({"covariate_flighting_period_weeks_range": (4.0, 6.0)}, "integer bounds"),
        ({"treatment_flighting_period_weeks_range": (True, 4)}, "integer bounds"),
        ({"covariate_flighting_period_weeks_range": (6, 4)}, "integer bounds"),
        ({"treatment_flighting_duty_range": (0.0, 0.5)}, "flighting_duty_range has invalid"),
        ({"covariate_flighting_duty_range": (0.5, 1.25)}, "flighting_duty_range has invalid"),
        ({"treatment_level_jump_count": 0}, "integer >= 1"),
        ({"covariate_level_jump_count": True}, "integer >= 1"),
        ({"treatment_level_jump_count": 2.0}, "integer >= 1"),
        ({"treatment_level_jump_factor_range": (0.0, 2.0)}, "factor_range has invalid"),
        ({"covariate_level_jump_size_range": (1.0, -1.0)}, "size_range has invalid"),
        ({"treatment_seasonal_amplitude_range": (-0.1, 0.5)}, "amplitude_range has invalid"),
        ({"covariate_seasonal_period_weeks_range": (2.0, 10.0)}, "aliases to an alternation"),
        ({"treatment_trend_log_change_range": (0.0, float("inf"))}, "change_range has invalid"),
        ({"covariate_trend_change_range": (0.5, 0.25)}, "change_range has invalid"),
        ({"covariate_level_jump_size_range": (0.0, 1e39)}, "float32 corpus storage maximum"),
    ),
)
def test_static_trajectory_bounds_are_checked_even_when_disabled(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_scm_prior(**_RULES, **overrides)


@pytest.mark.parametrize(
    "priors, include, match",
    (
        (
            {"treatment_onset_frac_range": (float(np.nextafter(1 / 32, 0.0)), 0.2)},
            {"treatment_onset_inclusion_prob": 0.3},
            "launches at week",
        ),
        (
            {"covariate_offset_frac_range": (0.6, 1.0)},
            {"covariate_offset_inclusion_prob": 1.0},
            "stops the input before",
        ),
        (
            {"covariate_level_jump_count": 32},
            {"covariate_level_jump_inclusion_prob": 0.1},
            "must be <= n_time_steps - 1",
        ),
        (
            {"treatment_level_jump_factor_range": (1.0, 1.0)},
            {"treatment_level_jump_inclusion_prob": 0.5},
            "every jump a no-op",
        ),
        (
            {"covariate_level_jump_size_range": (0.0, 0.0)},
            {"covariate_level_jump_inclusion_prob": 0.5},
            "every jump a no-op",
        ),
        (
            {"treatment_seasonal_amplitude_range": (0.0, 0.0)},
            {"treatment_seasonal_inclusion_prob": 1.0},
            "zero amplitude",
        ),
        (
            {"covariate_trend_change_range": (0.0, 0.0)},
            {"covariate_trend_inclusion_prob": 0.7},
            "identically zero",
        ),
        (
            {"treatment_onset_frac_range": (0.1, 15.5 / 32)},
            {"treatment_onset_inclusion_prob": 0.2},
            "only 1 on-week",
        ),
        (
            {"covariate_offset_frac_range": (1.5 / 32, 0.9)},
            {"covariate_offset_inclusion_prob": 0.2},
            "only 1 on-week",
        ),
        (
            {
                "treatment_flighting_period_weeks_range": (13, 13),
                "treatment_flighting_duty_range": (0.1, 0.2),
            },
            {"treatment_flighting_inclusion_prob": 0.9},
            "only 1 on-week",
        ),
        (
            {"treatment_seasonal_amplitude_range": (0.5, 3.5)},
            {"treatment_seasonal_inclusion_prob": 0.1},
            "swing the log-level",
        ),
        (
            {"treatment_trend_log_change_range": (-3.5, 0.0)},
            {"treatment_trend_inclusion_prob": 0.1},
            "swing the log-level",
        ),
        (
            {"treatment_level_jump_count": 5},  # 5·log 2 > 3
            {"treatment_level_jump_inclusion_prob": 1.0},
            "swing the log-level",
        ),
        # Long periods at a high duty keep two on-weeks in the window, yet some
        # draws would never switch off inside the reported weeks.
        (
            {
                "treatment_flighting_period_weeks_range": (33, 40),
                "treatment_flighting_duty_range": (0.9, 1.0),
            },
            {"treatment_flighting_inclusion_prob": 0.5},
            r"treatment_flighting_period_weeks_range upper bound \(40\) must be <= n_time_steps",
        ),
        (
            {
                "covariate_flighting_period_weeks_range": (20, 64),
                "covariate_flighting_duty_range": (0.95, 1.0),
            },
            {"covariate_flighting_inclusion_prob": 1.0},
            r"covariate_flighting_period_weeks_range upper bound \(64\) must be <= n_time_steps",
        ),
        # Every endpoint is storable, but the summed scheduled level is not.
        (
            {
                "covariate_level_jump_count": 2,
                "covariate_level_jump_size_range": (0.6 * _F32_MAX, 0.6 * _F32_MAX),
            },
            {"covariate_level_jump_inclusion_prob": 1.0},
            "beyond the float32 corpus storage limit",
        ),
        (
            {
                "rw_positive_mean_range": (0.05 * _F32_MAX, 0.05 * _F32_MAX),
                "treatment_trend_log_change_range": (3.0, 3.0),
            },
            {"treatment_trend_inclusion_prob": 0.1},
            "beyond the float32 corpus storage limit",
        ),
    ),
)
def test_effectiveness_and_horizon_rules_wait_for_inclusion(priors, include, match):
    make_scm_prior(**_RULES, **priors)  # a disabled component's priors constrain nothing
    with pytest.raises(ValueError, match=match):
        make_scm_prior(**_RULES, **priors, **include)


_WINDOW_OF_4 = {  # a launch at week 12 leaves 4 weeks of the 16-week support prefix
    "treatment_onset_inclusion_prob": 1.0,
    "treatment_onset_frac_range": (12.5 / 32, 12.5 / 32),
    "treatment_flighting_inclusion_prob": 1.0,
    "treatment_flighting_period_weeks_range": (5, 5),
}
_BELOW_HALF = float(np.nextafter(0.5, 0.0))
#: 15 jumps of max|log f| = 0.1 plus a full trend of 1.0; the seasonal amplitude completes 3.0.
_FIFTEEN_SMALL_JUMPS = {
    "treatment_level_jump_inclusion_prob": 1.0,
    "treatment_level_jump_count": 15,
    "treatment_level_jump_factor_range": (float(np.exp(-0.1)), float(np.exp(0.1))),
    "treatment_seasonal_inclusion_prob": 1.0,
    "treatment_trend_inclusion_prob": 1.0,
    "treatment_trend_log_change_range": (-1.0, 0.5),
}


@pytest.mark.parametrize(
    "accepted, rejected, match",
    (
        (
            {"treatment_onset_inclusion_prob": 1.0, "treatment_onset_frac_range": (1 / 32, 0.2)},
            {
                "treatment_onset_inclusion_prob": 1.0,
                "treatment_onset_frac_range": (float(np.nextafter(1 / 32, 0.0)), 0.2),
            },
            "launches at week",
        ),
        (
            {
                "covariate_offset_inclusion_prob": 1.0,
                "covariate_offset_frac_range": (0.6, float(np.nextafter(1.0, 0.0))),
            },
            {"covariate_offset_inclusion_prob": 1.0, "covariate_offset_frac_range": (0.6, 1.0)},
            "stops the input before",
        ),
        (
            {"covariate_level_jump_inclusion_prob": 1.0, "covariate_level_jump_count": 31},
            {"covariate_level_jump_inclusion_prob": 1.0, "covariate_level_jump_count": 32},
            "must be <= n_time_steps - 1",
        ),
        (
            {"covariate_seasonal_period_weeks_range": (float(np.nextafter(2.0, 3.0)), 10.0)},
            {"covariate_seasonal_period_weeks_range": (2.0, 10.0)},
            "aliases to an alternation",
        ),
        (  # the latest launch at S - 2 leaves exactly two support weeks
            {"treatment_onset_inclusion_prob": 1.0, "treatment_onset_frac_range": (0.1, 14.5 / 32)},
            {"treatment_onset_inclusion_prob": 1.0, "treatment_onset_frac_range": (0.1, 15.5 / 32)},
            "only 1 on-week",
        ),
        (  # duty·P = 2.5 rounds half up to W = 3 of 5: two on-weeks in any 4-week window
            {**_WINDOW_OF_4, "treatment_flighting_duty_range": (0.5, 0.8)},
            {**_WINDOW_OF_4, "treatment_flighting_duty_range": (_BELOW_HALF, 0.8)},
            "only 1 on-week",
        ),
        (  # composite swing 1.0 + 2.0 == 3.0 exactly; 1e-9 above it is past the tolerance
            {
                "treatment_seasonal_inclusion_prob": 1.0,
                "treatment_seasonal_amplitude_range": (0.5, 1.0),
                "treatment_trend_inclusion_prob": 1.0,
                "treatment_trend_log_change_range": (-2.0, 1.5),
            },
            {
                "treatment_seasonal_inclusion_prob": 1.0,
                "treatment_seasonal_amplitude_range": (0.5, 1.0),
                "treatment_trend_inclusion_prob": 1.0,
                "treatment_trend_log_change_range": (-(2.0 + 1e-9), 1.5),
            },
            "swing the log-level",
        ),
        (
            {
                "treatment_seasonal_inclusion_prob": 1.0,
                "treatment_seasonal_amplitude_range": (3.0, 3.0),
            },
            {
                "treatment_seasonal_inclusion_prob": 1.0,
                "treatment_seasonal_amplitude_range": (3.0, 3.0 + 1e-9),
            },
            "swing the log-level",
        ),
        (  # 15·0.1 + 0.5 + 1.0 is 3.0 on paper and rounds just above it: still accepted
            {**_FIFTEEN_SMALL_JUMPS, "treatment_seasonal_amplitude_range": (0.1, 0.5)},
            {**_FIFTEEN_SMALL_JUMPS, "treatment_seasonal_amplitude_range": (0.1, 0.5 + 1e-9)},
            "swing the log-level",
        ),
        (  # an asymmetric range's max|log f| comes from its low end: 2·log 4 fits, 3·log 4 not
            {
                "treatment_level_jump_inclusion_prob": 1.0,
                "treatment_level_jump_factor_range": (0.25, 1.5),
                "treatment_level_jump_count": 2,
            },
            {
                "treatment_level_jump_inclusion_prob": 1.0,
                "treatment_level_jump_factor_range": (0.25, 1.5),
                "treatment_level_jump_count": 3,
            },
            "swing the log-level",
        ),
        (  # ... or from its high end
            {
                "treatment_level_jump_inclusion_prob": 1.0,
                "treatment_level_jump_factor_range": (0.9, 4.0),
                "treatment_level_jump_count": 2,
            },
            {
                "treatment_level_jump_inclusion_prob": 1.0,
                "treatment_level_jump_factor_range": (0.9, 4.0),
                "treatment_level_jump_count": 3,
            },
            "swing the log-level",
        ),
        (  # jumps: 4·log 2 fits, 5·log 2 does not
            {"treatment_level_jump_inclusion_prob": 1.0, "treatment_level_jump_count": 4},
            {"treatment_level_jump_inclusion_prob": 1.0, "treatment_level_jump_count": 5},
            "swing the log-level",
        ),
        (  # a flighting period may span the whole window, but not more
            {
                "treatment_flighting_inclusion_prob": 1.0,
                "treatment_flighting_period_weeks_range": (32, 32),
                "treatment_flighting_duty_range": (0.9, 1.0),
            },
            {
                "treatment_flighting_inclusion_prob": 1.0,
                "treatment_flighting_period_weeks_range": (33, 33),
                "treatment_flighting_duty_range": (0.9, 1.0),
            },
            r"flighting_period_weeks_range upper bound \(33\) must be <= n_time_steps",
        ),
        # Every endpoint is storable, but the summed scheduled level must be too. One
        # covariate jump of 0.6 max leaves 0.4 max for the rest of a covariate's level.
        (
            {**_COVARIATE_JUMP, "rw_covariate_mean_range": (-0.3 * _F32_MAX, 1.0)},
            {**_COVARIATE_JUMP, "rw_covariate_mean_range": (-0.5 * _F32_MAX, 1.0)},
            "beyond the float32 corpus storage limit",
        ),
        (
            {
                **_COVARIATE_JUMP,
                "covariate_seasonal_inclusion_prob": 1.0,
                "covariate_seasonal_amplitude_range": (0.0, 0.3 * _F32_MAX),
            },
            {
                **_COVARIATE_JUMP,
                "covariate_seasonal_inclusion_prob": 1.0,
                "covariate_seasonal_amplitude_range": (0.0, 0.5 * _F32_MAX),
            },
            "beyond the float32 corpus storage limit",
        ),
        (
            {
                **_COVARIATE_JUMP,
                "covariate_trend_inclusion_prob": 1.0,
                "covariate_trend_change_range": (-0.3 * _F32_MAX, 0.0),
            },
            {
                **_COVARIATE_JUMP,
                "covariate_trend_inclusion_prob": 1.0,
                "covariate_trend_change_range": (-0.5 * _F32_MAX, 0.0),
            },
            "beyond the float32 corpus storage limit",
        ),
        (  # jumps count by magnitude, once each: 2 · 0.49 max fits, 2 · 0.6 max does not
            {
                "covariate_level_jump_inclusion_prob": 1.0,
                "covariate_level_jump_count": 2,
                "covariate_level_jump_size_range": (-0.49 * _F32_MAX, 0.0),
            },
            {
                "covariate_level_jump_inclusion_prob": 1.0,
                "covariate_level_jump_count": 2,
                "covariate_level_jump_size_range": (-0.6 * _F32_MAX, 0.0),
            },
            "beyond the float32 corpus storage limit",
        ),
        (  # a treatment reaches its walk mean's upper bound times e^3: 0.049 max fits
            {**_TREATMENT_REACH, "rw_positive_mean_range": (1.0, 0.049 * _F32_MAX)},
            {**_TREATMENT_REACH, "rw_positive_mean_range": (1.0, 0.05 * _F32_MAX)},
            "beyond the float32 corpus storage limit",
        ),
    ),
    ids=(
        "onset_first_week",
        "offset_last_week",
        "jump_count",
        "seasonal_period",
        "latest_launch",
        "round_half_up",
        "swing_exactly_3",
        "seasonal_alone",
        "fifteen_small_jumps",
        "asymmetric_low_factor",
        "asymmetric_high_factor",
        "jump_swing",
        "flighting_period_horizon",
        "covariate_mean_reach",
        "covariate_seasonal_reach",
        "covariate_trend_reach",
        "covariate_jump_reach",
        "treatment_mean_reach",
    ),
)
def test_validation_boundaries(accepted, rejected, match):
    make_scm_prior(**_RULES, **accepted)
    with pytest.raises(ValueError, match=match):
        make_scm_prior(**_RULES, **rejected)


def test_gate_windows_are_validated_jointly_per_input_type():
    onset = {"treatment_onset_inclusion_prob": 0.5, "treatment_onset_frac_range": (0.1, 9.5 / 32)}
    offset = {
        "treatment_offset_inclusion_prob": 0.5,
        "treatment_offset_frac_range": (10.5 / 32, 0.9),
    }
    long_flights = {
        "treatment_flighting_period_weeks_range": (13, 13),
        "treatment_flighting_duty_range": (0.1, 0.2),
    }
    make_scm_prior(**_RULES, **onset)
    make_scm_prior(**_RULES, **offset)
    with pytest.raises(ValueError, match="treatment gates can leave only 1 on-week"):
        make_scm_prior(**_RULES, **onset, **offset)
    with pytest.raises(ValueError, match="treatment gates can leave only 0 on-week"):
        make_scm_prior(**_RULES, **onset, **long_flights, treatment_flighting_inclusion_prob=0.5)
    # Windows on different input types never meet on one input.
    covariate_offset = {key.replace("treatment", "covariate"): v for key, v in offset.items()}
    make_scm_prior(**_RULES, **onset, **covariate_offset)
    # Only included gates count: the same priors at inclusion probability 0 constrain nothing.
    make_scm_prior(**_RULES, **onset, treatment_offset_frac_range=(10.5 / 32, 0.9))
    make_scm_prior(**_RULES, **offset, treatment_onset_frac_range=(0.1, 9.5 / 32))
    make_scm_prior(**_RULES, **onset, **long_flights)


def test_min_on_weeks_matches_phase_enumeration():
    max_window = 30
    t = np.arange(max_window)
    worst: dict[tuple[int, int], np.ndarray] = {}
    for period in range(2, 10):
        for on in range(1, period):
            cycle = np.mod(t[None, :] + np.arange(period)[:, None], period) < on
            counts = np.concatenate(
                [np.zeros((period, 1), dtype=int), cycle.cumsum(axis=1)], axis=1
            )
            worst[period, on] = counts.min(axis=0)  # worst phase, for every window length
    for window in range(-2, max_window + 1):
        assert min_on_weeks(window, None, 0.5) == max(window, 0)
        for p_lo in range(2, 10):
            for p_hi in range(p_lo, 10):
                for duty in (0.05, 0.25, 0.3, 0.5, 0.8, 1.0):
                    expected = (
                        min(int(worst[p, _on_run(duty, p)][window]) for p in range(p_lo, p_hi + 1))
                        if window > 0
                        else 0
                    )
                    assert min_on_weeks(window, (p_lo, p_hi), duty) == expected, (
                        window,
                        p_lo,
                        p_hi,
                        duty,
                    )


@pytest.mark.parametrize(
    "overrides, warns",
    (
        ({}, False),
        ({"treatment_hf_inclusion_prob": 0.0, "treatment_pulse_inclusion_prob": 0.0}, True),
        ({"treatment_hf_sigma_range": (0.0, 0.0), "treatment_pulse_inclusion_prob": 0.0}, True),
        ({"treatment_hf_inclusion_prob": 0.2, "treatment_pulse_inclusion_prob": 0.0}, False),
        (
            {
                "treatment_hf_inclusion_prob": 0.0,
                "treatment_pulse_inclusion_prob": 0.0,
                "treatment_trend_inclusion_prob": 0.3,
            },
            False,
        ),
        (  # covariate schedules leave treatments flat
            {
                "treatment_hf_inclusion_prob": 0.0,
                "treatment_pulse_inclusion_prob": 0.0,
                "covariate_trend_inclusion_prob": 1.0,
            },
            True,
        ),
    ),
    ids=(
        "default",
        "probs_zero",
        "range_and_prob_off",
        "fractional_hf",
        "treatment_schedule",
        "covariate_schedule",
    ),
)
def test_flat_texture_warning_follows_effective_treatment_inclusion(overrides, warns):
    cfg = make_scm_prior(**{**_PROBE, "n_time_steps": 32}, **overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_flat_texture(cfg)
    flat = [w for w in caught if issubclass(w.category, FutureWarning) and "flat" in str(w.message)]
    assert bool(flat) is warns


def test_composable_preset_widens_the_input_level_and_variation_priors():
    texture = make_scm_prior(**{**_PROBE, "n_time_steps": 32})
    composable = make_scm_prior(**{**_PROBE, "n_time_steps": 32}, trajectories="composable")
    for name in ("rw_positive_mean_range", "rw_treatment_std_range", "rw_covariate_mean_range"):
        (lo, hi), (texture_lo, texture_hi) = getattr(composable, name), getattr(texture, name)
        assert lo < texture_lo and texture_hi < hi, name
    assert composable.rw_std_sigma > texture.rw_std_sigma

    def baseline_sigma(cfg: SCMPrior) -> float:
        return cfg.rw_std_sigma if cfg.rw_baseline_std_sigma is None else cfg.rw_baseline_std_sigma

    assert baseline_sigma(composable) == baseline_sigma(texture)  # legacy absolute-mode scale


def _worst_on_weeks(cfg: SCMPrior, x: str) -> int:
    """Fewest on-weeks any draw of the config's gate priors leaves in the support prefix.

    Enumerates every reachable integer outcome — launch and stop weeks, flighting
    period, on-run and phase. A gate only switches weeks off, so the worst case
    carries every gate the input can be drawn with.
    """
    T = cfg.n_time_steps
    support = min(T - cfg.n_query, T // 2)
    probs = cfg.trajectory_inclusion_probs()[x]

    def reachable(name: str) -> np.ndarray:
        lo, hi = getattr(cfg, f"{x}_{name}")
        return np.arange(math.floor(lo * T), math.floor(hi * T) + 1)

    starts = reachable("onset_frac_range") if probs["onset"] > 0 else np.array([0])
    stops = reachable("offset_frac_range") if probs["offset"] > 0 else np.array([T])
    weeks = np.arange(support)
    if probs["flighting"] > 0:
        p_lo, p_hi = getattr(cfg, f"{x}_flighting_period_weeks_range")
        d_lo, d_hi = getattr(cfg, f"{x}_flighting_duty_range")
        on = np.array(
            [
                np.mod(weeks + phase, period) < run
                for period in range(p_lo, p_hi + 1)
                for run in range(_on_run(d_lo, period), _on_run(d_hi, period) + 1)
                for phase in range(period)
            ]
        )
    else:
        on = np.ones((1, support), dtype=bool)
    cumulative = np.concatenate([np.zeros((len(on), 1), dtype=int), on.cumsum(axis=1)], axis=1)
    lo = np.minimum(starts, support)
    hi = np.maximum(lo[:, None], np.minimum(stops, support)[None, :])
    return int((cumulative[:, hi] - cumulative[:, lo[:, None]]).min())


#: Fewest support-prefix weeks S = min(T - n_query, T // 2) each gated preset needs so
#: its gates leave two on-weeks:
#:   delayed_start, ramp_up: a launch at week >= 1, then two weeks   -> S >= 3;
#:   periodic_on_off: two 2-week periods at the preset's duty 0.3    -> S >= 4;
#:   composable: a launch at week >= 1, then two 2-week periods      -> S >= 5.
#: Composable's earliest launch grows with T, so its bound is exact up to T = 29;
#: the sweep's longer horizons all leave S >= 24.
_PRESET_MIN_SUPPORT = {"delayed_start": 3, "ramp_up": 3, "periodic_on_off": 4, "composable": 5}
_SWEEP_N_TIME_STEPS = (*range(4, 17), 22, 49, 103, 104)


def _split(n_time_steps: int, query_frac: float) -> tuple[bool, int]:
    """Whether the query split is valid, and the support prefix both split types leave."""
    n_query = round(query_frac * n_time_steps)
    return 0 < n_query <= n_time_steps - 2, min(n_time_steps - n_query, n_time_steps // 2)


def _expected_preset_outcome(name: str, n_time_steps: int, query_frac: float) -> str:
    valid, support = _split(n_time_steps, query_frac)
    if not valid:
        return "invalid query split"
    need = _PRESET_MIN_SUPPORT.get(name, 0)
    if support >= need:
        return "ok"
    next_fit = next(
        t
        for t in itertools.count(n_time_steps + 1)
        if _split(t, query_frac)[0] and _split(t, query_frac)[1] >= need
    )
    return f"next fit {next_fit}"


def _preset_outcome(**kwargs) -> tuple[str, SCMPrior | None]:
    try:
        return "ok", make_scm_prior(**kwargs)
    except ValueError as exc:
        message = str(exc)
        if f"trajectories={kwargs['trajectories']!r}" in message:
            fit = re.search(r"the next longer horizon that fits is n_time_steps=(\d+)$", message)
            return f"next fit {fit.group(1) if fit else '?'}", None
        if "query weeks" in message:
            return "invalid query split", None
        return f"unexpected: {message}", None


@pytest.mark.parametrize("name", TRAJECTORY_PRESETS)
def test_preset_horizon_sweep(name):
    """Every preset is valid exactly where pinned, and valid presets keep two on-weeks."""
    mismatches = []
    for T in _SWEEP_N_TIME_STEPS:
        for query_frac in (0.1, 0.25, 0.5):
            expected = _expected_preset_outcome(name, T, query_frac)
            for l_max in (8, 52):
                outcome, cfg = _preset_outcome(
                    n_treatments=2,
                    n_covariates=1,
                    n_latent=1,
                    n_time_steps=T,
                    query_frac=query_frac,
                    l_max=l_max,
                    nonlinearity="linear",
                    trajectories=name,
                )
                if outcome != expected:
                    mismatches.append((T, query_frac, l_max, outcome, expected))
                elif cfg is not None:
                    for x in TRAJECTORY_INPUTS:
                        worst = _worst_on_weeks(cfg, x)
                        if worst < 2:
                            mismatches.append((T, query_frac, l_max, x, f"{worst} on-weeks"))
    assert not mismatches


def test_delayed_start_fits_a_short_support_prefix():
    """A large query fraction leaves ten support weeks of 104; the launch window must fit them."""
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 104}, query_frac=0.9, trajectories="delayed_start"
    )
    assert _split(104, 0.9) == (True, 10)
    for x in TRAJECTORY_INPUTS:
        lo, hi = getattr(cfg, f"{x}_onset_frac_range")
        assert 1 <= math.floor(lo * 104) <= math.floor(hi * 104) <= 10 - 2
        assert _worst_on_weeks(cfg, x) >= 2


def test_presets_derive_gate_windows_from_the_same_n_query_as_validation():
    """A float32 query fraction rounds differently in float32 than in float64.

    ``SCMPrior`` takes n_query from the raw value, so the preset must too: with a
    float64 copy it built windows for the wrong support prefix — silently dropping
    a preset that cannot fit, or raising for one that can.
    """
    linear = {**_PROBE, "nonlinearity": "linear"}
    fits = make_scm_prior(
        **{**linear, "n_time_steps": 20},
        query_frac=np.float32(0.575),
        trajectories="delayed_start",
    )
    assert fits.trajectory_components_enabled
    for x in TRAJECTORY_INPUTS:
        assert _worst_on_weeks(fits, x) >= 2
    # float32 0.85 * 10 = 8.5 rounds to even, so n_query is 8 and S = 2: too short
    # for flighting. The old float64 copy gave 9, an invalid split, and silently
    # dropped the preset.
    with pytest.raises(ValueError, match="cannot keep two on-weeks"):
        make_scm_prior(
            **{**linear, "n_time_steps": 10},
            query_frac=np.float32(0.85),
            trajectories="periodic_on_off",
        )
    # A numpy-integer horizon promotes the float32 product to float64, which rounds
    # a different way again; the preset must follow SCMPrior there too.
    for horizon, query_frac, extra in (
        (np.int64(10), np.float32(0.05), {"l_max": 1}),  # was silently dropped
        (np.int64(30), np.float32(0.55), {}),  # was rejected for a 1-on-week window
    ):
        cfg = make_scm_prior(
            **{**linear, "n_time_steps": horizon, **extra},
            query_frac=query_frac,
            trajectories="delayed_start",
        )
        assert cfg.trajectory_components_enabled
        for x in TRAJECTORY_INPUTS:
            assert _worst_on_weeks(cfg, x) >= 2


def test_outcome_norm_guard_checks_both_support_splits():
    """The validation-split repair can move a task to the other split after acceptance."""
    from pymc_generator.sampler import _outcome_norm_overflows

    outcome = np.array([1.0, 1.0 + 1e-6, 1.0, 1.0, _F32_MAX / 2.0, 1.0])
    # A tiny std over the first two weeks overflows; a wide prefix does not.
    assert _outcome_norm_overflows(outcome, (2,))
    assert not _outcome_norm_overflows(np.array([1.0, 2.0, 1.0, 2.0]), (2, 4))
    assert _outcome_norm_overflows(outcome, (5, 2))
    # A zero-std prefix takes the full-series fallback and is not judged here.
    assert not _outcome_norm_overflows(np.array([1.0, 1.0, 3.0, 1.0]), (2,))
    # Storage divides the float32 outcome by its own std: 1000.00009 rounds to
    # 1000.000061 in float32, shrinking the std by a third, so a peak that fits
    # against the float64 std overflows against the stored one.
    start = np.tile([1000.0, 1000.00009], 8)
    peak = 0.9 * _F32_MAX * float(np.std(start))
    assert _outcome_norm_overflows(np.concatenate([start, [peak]]), (16,))


def test_a_flipped_split_never_stores_an_inf_outcome_norm(monkeypatch):
    """A task accepted on one split may be repaired onto the other; neither may overflow.

    Seed 22 used to accept a task whose repaired 16-week support overflowed
    outcome_norm. The spy proves the guard rejected a draw the task's own split
    alone would have accepted, so a drifted seed fails loudly instead of passing
    without exercising the repair.
    """
    import pymc_generator.sampler as sampler

    real = sampler._outcome_norm_overflows
    rescued: list[bool] = []

    def spy(outcome, support_ends):
        overflows = real(outcome, support_ends)
        if overflows and not real(outcome, support_ends[:1]):
            rescued.append(True)
        return overflows

    monkeypatch.setattr(sampler, "_outcome_norm_overflows", spy)
    cfg = make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=2,
        seed=22,
        p_long_horizon=0.0,
        edge_rate_overrides={"zy": 1.0},
        zc_base_rate=0.0,
        dz_base_rate=0.0,
        zy_coeff_range=(1.0, 1.0),
        covariate_level_jump_inclusion_prob=1.0,
        covariate_level_jump_size_range=(0.95 * _F32_MAX, 0.95 * _F32_MAX),
        covariate_hf_inclusion_prob=0.0,
        covariate_pulse_inclusion_prob=0.0,
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert rescued, "no draw needed the other split's check; re-pick the seed"
    outcome = corpus["outcome_raw"].astype(np.float64)
    for end in (cfg.n_time_steps - cfg.n_query, cfg.n_time_steps // 2):
        scale = outcome[:, :end].std(axis=1)
        assert (np.abs(outcome).max(axis=1) <= _F32_MAX * scale).all()
    assert np.isfinite(corpus["outcome_norm"]).all()
    _assert_valid(corpus)


def _error_without_preset(**kwargs) -> Exception | None:
    """The error ``make_scm_prior`` raises for these fields without a trajectory preset."""
    try:
        make_scm_prior(**{**kwargs, "trajectories": "texture"})
    except (TypeError, ValueError) as error:
        return error
    return None


@pytest.mark.parametrize(
    "name, nonlinearity, n_time_steps, overrides",
    (
        # Diverse carryover adds a burn-in requirement the gate windows alone miss.
        ("composable", "diverse", 8, {}),
        ("delayed_start", "diverse", 5, {}),
        ("composable", "linear", 8, {}),
        ("periodic_on_off", "linear", 6, {}),
        # Overrides that tighten the complete config push the first fit further out.
        ("composable", "linear", 8, {"covariate_flighting_period_weeks_range": (4, 4)}),
        (
            "periodic_on_off",
            "linear",
            6,
            {"n_treatment_shocks": 2, "treatment_shock_length_range": (6, 6)},
        ),
        # A 52-week carryover burn-in leaves the first fit about a hundred weeks out.
        ("composable", "diverse", 8, {"l_max": 52}),
        # The hint search computes n_query from the float32 value, as SCMPrior does.
        ("composable", "linear", 17, {"query_frac": np.float32(14.5 / 19)}),
        # A numpy-integer horizon rounds n_query differently again (float64 product).
        ("periodic_on_off", "linear", np.int64(44), {"query_frac": np.float32(0.93)}),
    ),
    ids=(
        "composable-diverse",
        "delayed_start-diverse",
        "composable-linear",
        "periodic-linear",
        "composable-tight-flighting",
        "periodic-long-shocks",
        "composable-long-burn-in",
        "composable-float32-query",
        "periodic-int64-horizon",
    ),
)
def test_an_infeasible_preset_names_the_first_horizon_its_complete_config_accepts(
    name, nonlinearity, n_time_steps, overrides
):
    kwargs = {
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "nonlinearity": nonlinearity,
        "trajectories": name,
        **overrides,
    }
    with pytest.raises(ValueError, match="the next longer horizon that fits is") as raised:
        make_scm_prior(**kwargs, n_time_steps=n_time_steps)
    # The cause is the caller's own config at this horizon, exactly as it fails without a preset.
    own = _error_without_preset(**kwargs, n_time_steps=n_time_steps)
    cause = raised.value.__cause__
    if own is None:
        assert cause is None
    else:
        assert type(cause) is type(own) and str(cause) == str(own)
    named = int(re.search(r"n_time_steps=(\d+)$", str(raised.value)).group(1))
    # Candidates keep the caller's horizon type: n_query rounds in that type.
    typed = type(n_time_steps)
    make_scm_prior(**kwargs, n_time_steps=typed(named))
    for horizon in range(int(n_time_steps) + 1, named):
        with pytest.raises(ValueError):
            make_scm_prior(**kwargs, n_time_steps=typed(horizon))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_hint_search_survives_a_query_fraction_that_overflows_its_type():
    """A float16 product with a long candidate horizon overflows; the search just stops.

    No horizon fits before the overflow, so the error carries no hint, and the
    probe's cast warning is silenced rather than leaked to the caller.
    """
    with pytest.raises(ValueError, match="cannot keep two on-weeks") as raised:
        make_scm_prior(
            **{**_PROBE, "n_time_steps": 52},
            query_frac=np.float16(0.9712),
            trajectories="composable",
        )
    assert "next longer horizon" not in str(raised.value)


def test_overrides_cannot_rescue_an_infeasible_preset():
    """Documented precedence: an infeasible preset raises before ``**overrides`` apply.

    Switching flighting off would leave a valid config, as building it without the
    preset shows, yet the preset still raises; the horizon it names accepts the
    complete config, overrides included.
    """
    kwargs = {
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "nonlinearity": "linear",
        "treatment_flighting_inclusion_prob": 0.0,
        "covariate_flighting_inclusion_prob": 0.0,
    }
    make_scm_prior(**kwargs, n_time_steps=6)
    with pytest.raises(ValueError, match="the next longer horizon that fits is") as raised:
        make_scm_prior(**kwargs, n_time_steps=6, trajectories="periodic_on_off")
    # The caller's fields are valid at the requested horizon, so nothing is chained.
    assert raised.value.__cause__ is None
    named = int(re.search(r"n_time_steps=(\d+)$", str(raised.value)).group(1))
    make_scm_prior(**kwargs, n_time_steps=named, trajectories="periodic_on_off")


def test_an_override_invalid_at_every_horizon_drops_the_hint_and_is_chained_as_the_cause(
    monkeypatch,
):
    """No horizon can fit, so the capped hint search gives up and names none.

    It costs at most 1,000 complete-config validations plus one for the cause
    check, and the caller's own invalid field is chained as the cause instead of
    being hidden behind the preset error.
    """
    calls = []
    validate = SCMPrior.validate

    def counting(self):
        calls.append(self.n_time_steps)
        return validate(self)

    monkeypatch.setattr(SCMPrior, "validate", counting)
    with pytest.raises(ValueError, match="cannot keep two on-weeks") as raised:
        make_scm_prior(
            n_treatments=2,
            n_covariates=1,
            n_latent=1,
            n_time_steps=8,
            trajectories="composable",
            rw_positive_mean_range=(0.0, 1.0),
        )
    # Exactly the 1,000 capped hint validations plus the one cause check.
    assert len(calls) == 1_001
    assert "next longer horizon" not in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "rw_positive_mean_range" in str(raised.value.__cause__)


def test_a_misspelled_override_is_the_cause_of_an_infeasible_preset_error():
    with pytest.raises(ValueError, match="cannot keep two on-weeks") as raised:
        make_scm_prior(
            n_treatments=2,
            n_covariates=1,
            n_latent=1,
            n_time_steps=8,
            trajectories="composable",
            treatmnet_trend_inclusion_prob=0.1,
        )
    assert "next longer horizon" not in str(raised.value)  # no horizon can fit a typo
    assert isinstance(raised.value.__cause__, TypeError)
    assert "treatmnet_trend_inclusion_prob" in str(raised.value.__cause__)


def test_the_hint_search_reaches_a_thousand_horizons_past_the_request():
    """query_frac=0.999 leaves a two-week support prefix from 1500 up to 2500 weeks.

    Every one of those thousand horizons is valid but too short for the preset, so
    neither a short search window nor a cap that counted them could reach 2501.
    """
    kwargs = {
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "nonlinearity": "linear",
        "trajectories": "delayed_start",
        "query_frac": 0.999,
    }
    with pytest.raises(
        ValueError, match=r"the next longer horizon that fits is n_time_steps=2501$"
    ):
        make_scm_prior(**kwargs, n_time_steps=1500)
    with pytest.raises(ValueError, match="cannot keep two on-weeks"):
        make_scm_prior(**kwargs, n_time_steps=2500)
    make_scm_prior(**kwargs, n_time_steps=2501)


@pytest.mark.parametrize("trajectories", ("texture", "composable"))
@pytest.mark.parametrize(
    "horizon, match",
    (
        ({"query_frac": float("inf")}, "query_frac must be a positive real scalar"),
        ({"query_frac": float("nan")}, "query_frac must be a positive real scalar"),
        ({"query_frac": None}, "query_frac must be a positive real scalar"),
        ({"n_time_steps": None}, "n_time_steps must be an integer >= 4"),
        ({"n_time_steps": float("inf")}, "n_time_steps must be an integer >= 4"),
    ),
)
def test_presets_leave_invalid_horizons_to_the_config_validator(trajectories, horizon, match):
    """A preset derives nothing from an invalid horizon, so the base config's own error surfaces."""
    with pytest.raises(ValueError, match=match):
        make_scm_prior(
            n_treatments=2, n_covariates=1, n_latent=1, trajectories=trajectories, **horizon
        )


# -- 9. realism -----------------------------------------------------------------


def test_cv_alternative_rescues_only_the_cv_floor():
    flat = np.full((4, 1), 2.0)  # natural path: CV 0
    scheduled = np.array([[0.0], [3.0], [0.0], [3.0]])  # its schedule applied: CV 1
    outcome = np.full(4, 5.0)
    common = {"g_cy_active": np.array([1]), "cv_floor": 0.1, "realism_outcome": outcome}

    def ok(actual, natural, alternative):
        return _additive_task_ok(
            actual,
            outcome,
            {"treatments": actual},
            realism_treatment=natural,
            cv_alternative_treatment=alternative,
            **common,
        )

    # The natural series fails the CV floor; its scheduled alternative clears it.
    assert ok(scheduled, flat, scheduled)
    # Without an alternative the natural series decides alone.
    assert not ok(scheduled, flat, None)
    varied = np.array([[1.0], [2.0], [1.0], [2.0]])
    # Either series may carry the variation: a flat alternative cannot sink a varied natural.
    assert ok(scheduled, varied, flat)
    # A spike on the actual series alone is accepted: spikes are judged on the natural path.
    assert ok(np.array([[1.0], [1.0], [100.0], [1.0]]), varied, varied)
    # A spike on the natural path is rejected even though the alternative clears the CV floor.
    assert not ok(scheduled, np.array([[1.0], [1.0], [60.0], [1.0]]), scheduled)


def test_storage_bound_rejects_draws_float32_cannot_hold():
    series = np.array([[1.0], [2.0], [1.0], [2.0]])
    outcome = np.full(4, 5.0)
    huge = {"treatments": series, "latent_unobserved": np.array([[1.0], [_F32_MAX * 2.0]])}
    common = {"g_cy_active": np.array([1]), "cv_floor": 0.1}
    assert _additive_task_ok(series, outcome, huge, **common)
    assert not _additive_task_ok(series, outcome, huge, storage_max=_F32_MAX, **common)
    # Covariates are signed: a large negative value overflows the cast just the same.
    negative = {"treatments": series, "covariates": np.array([[1.0], [-_F32_MAX * 2.0]])}
    assert not _additive_task_ok(series, outcome, negative, storage_max=_F32_MAX, **common)
    assert _additive_task_ok(
        series, outcome, {"treatments": series}, storage_max=_F32_MAX, **common
    )


def test_extreme_scheduled_levels_never_reach_storage_as_inf():
    """Validated level priors whose texture pushes a draw past float32 are rejected, not stored.

    The static reach rule counts the mean and the schedule but not texture or
    parents; before the draw-time guard this config died in finalization with
    'signal source arrays must be finite'.
    """
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 32},
        n_cells=2,
        draws_per_cell=1,
        seed=1,
        rw_positive_mean_range=(_F32_MAX / 25, _F32_MAX / 25),
        treatment_trend_inclusion_prob=1.0,
        treatment_trend_log_change_range=(3.0, 3.0),
    )
    corpus = pg.sample_prior_predictive(cfg)
    for key, value in corpus.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            assert np.isfinite(value).all(), key
    _assert_valid(corpus)


def test_a_schedule_rescues_a_texture_free_treatment_from_the_cv_floor():
    """The corpus offers each scheduled treatment its schedule as a CV alternative.

    Parentless treatments with no texture and a nearly constant walk always fail
    the CV floor on their natural path; the same prior with on/off flighting must
    still generate because the gate carries the variation.
    """
    quiet = {
        **_PROBE,
        "n_time_steps": 24,
        "n_cells": 2,
        "draws_per_cell": 2,
        "seed": 2,
        "edge_budget": {"cc": (0, 0), "zc": (0, 0), "dc": (0, 0)},
        "treatment_hf_inclusion_prob": 0.0,
        "treatment_pulse_inclusion_prob": 0.0,
        "rw_treatment_std_range": (0.01, 0.02),
    }
    with pytest.warns(FutureWarning, match="flat"):
        with pytest.raises(RuntimeError, match="realism filter rejected"):
            pg.sample_prior_predictive(make_scm_prior(**quiet))
    corpus = pg.sample_prior_predictive(
        make_scm_prior(**quiet, treatment_flighting_inclusion_prob=1.0)
    )
    assert (corpus["treatment_activity"] == 0).any(axis=1).all()
    _assert_valid(corpus)


def test_large_scheduled_envelopes_are_judged_on_the_natural_path():
    """A deliberate x20 seasonal swing with on/off flighting generates; legacy realism would not."""
    cfg = make_scm_prior(
        **{**_PROBE, "n_time_steps": 32},
        n_cells=2,
        draws_per_cell=2,
        seed=2,
        treatment_flighting_inclusion_prob=1.0,
        treatment_flighting_period_weeks_range=(4, 8),
        treatment_seasonal_inclusion_prob=1.0,
        treatment_seasonal_amplitude_range=(3.0, 3.0),
        treatment_seasonal_period_weeks_range=(16.0, 16.0),
    )
    corpus = pg.sample_prior_predictive(cfg)
    assert np.abs(corpus["treatment_log_level_shift"]).max() > 2.9
    active = np.broadcast_to(
        corpus["treatment_active_mask"][:, None, :] == 1, corpus["treatment_activity"].shape
    )
    assert (corpus["treatment_activity"][active] == 0).any()  # flighting switched inputs off
    cy = cfg.layout.slices["cy"]
    legacy = []
    for row in range(corpus["treatment_raw"].shape[0]):
        n_active = int(corpus["n_treatments_active"][row])
        treatment = corpus["treatment_raw"][row, :, :n_active].astype(np.float64)
        outcome = corpus["outcome_raw"][row].astype(np.float64)
        legacy.append(
            _additive_task_ok(
                treatment,
                outcome,
                {"treatments": treatment, "outcome": outcome},
                corpus["g"][row, cy][:n_active],
                cfg.treatment_cv_floor,
            )
        )
    assert not all(legacy)
    _assert_valid(corpus)


def test_outcome_spikes_from_a_scheduled_treatment_are_judged_on_the_natural_outcome():
    """A x20 seasonal treatment swing may spike the outcome; its natural outcome decides.

    One linear treatment with a large coefficient over a low baseline turns the
    treatment envelope straight into outcome peaks far past the 8x spike ratio.
    """
    cfg = make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=32,
        n_cells=2,
        draws_per_cell=3,
        seed=5,
        nonlinearity="linear",
        beta_additive_range=(2.0, 2.0),
        rw_baseline_mean_range=(1.0, 1.0),
        rw_covariate_mean_range=(1.0, 1.0),
        treatment_seasonal_inclusion_prob=1.0,
        treatment_seasonal_amplitude_range=(3.0, 3.0),
        treatment_seasonal_period_weeks_range=(16.0, 16.0),
    )
    corpus = pg.sample_prior_predictive(cfg)
    _assert_valid(corpus)
    outcome = corpus["outcome_raw"].astype(np.float64)
    assert (outcome.max(axis=1) / np.median(outcome, axis=1)).max() >= 8.0


def test_shocks_never_lift_a_scheduled_treatment_over_the_cv_floor():
    """The CV alternative applies a treatment's schedule to its natural path, never its shocks.

    A near-constant treatment carries a negligible trend, so its schedule cannot
    reach the CV floor, while its held shocks would: generation must fail.
    """
    cfg = make_scm_prior(
        n_treatments=1,
        n_covariates=1,
        n_latent=1,
        n_time_steps=24,
        n_cells=2,
        draws_per_cell=2,
        seed=2,
        edge_budget={"cc": (0, 0), "zc": (0, 0), "dc": (0, 0)},
        treatment_hf_inclusion_prob=0.0,
        treatment_pulse_inclusion_prob=0.0,
        rw_treatment_std_range=(0.01, 0.02),
        treatment_trend_inclusion_prob=1.0,
        treatment_trend_log_change_range=(0.001, 0.002),
        n_treatment_shocks=1,
        treatment_shock_level_range=(0.2, 0.3),
    )
    g = _edge_free(1, 1, 1)
    model, _, _ = build_world_model(g, cfg, sample_structure(g, cfg, np.random.default_rng(0)), 24)
    d = draw_worlds(
        model,
        ("treatment_activity", "treatment_log_level_shift", "treatments", "treatments_natural"),
        seed=9,
        draws=8,
    )

    def cv(series: np.ndarray) -> np.ndarray:
        return series.std(axis=1) / series.mean(axis=1)

    alternative = (
        d["treatment_activity"] * d["treatments_natural"] * np.exp(d["treatment_log_level_shift"])
    )
    assert (cv(alternative) < cfg.treatment_cv_floor).all()
    assert (cv(d["treatments"]) > cfg.treatment_cv_floor).all()  # the shocks would clear it
    with pytest.raises(RuntimeError, match="realism filter rejected"):
        pg.sample_prior_predictive(cfg)
