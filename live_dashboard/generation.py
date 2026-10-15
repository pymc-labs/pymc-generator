"""Generation worker: publish the logical tasks of one run into a store, resumably.

A run is a fixed list of logical tasks (``RunRequest``); task ``i`` generates
``worlds_per_task`` worlds with seed ``seed + i``, using the run's preset (a preset mix
cycles through its member presets by task index; a structural-split design spreads its
tasks over saturation x adstock splits and texture regimes, seeds each split on its own
and publishes only worlds whose controls stay non-negative, see
``live_dashboard.structural``). Each task is generated with deferred train/validation
membership and published through ``pymc_generator.publication``: the payload is
persisted and committed first and the immutable completion record second, so a reader
never sees a completion whose payload is not yet published. This worker is the run's
single driver: it writes the run manifest and its attempt record (progress, deadline,
worker) around the publications. Rerunning the same request reuses every task that
already has a verifying completion, so a crashed or timed-out attempt is resumed by
running it again. The run manifest pins the generator configuration, source digest and
numerical environment; resuming with a different generator is refused because one run
must never mix two generators' outputs.

``pymc_generator`` is imported inside functions only, so importing this module (e.g. by
the launcher, which the Modal CLI runs with its own interpreter) does not load PyMC or
PyTensor.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import os
import platform
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from live_dashboard import structural
from live_dashboard.deployment import REVISION_ENV, check_demo_bounds
from live_dashboard.publication import (
    MAX_RECORD_BYTES,
    AttemptRecord,
    AttemptState,
    ErrorInfo,
    GeneratorInfo,
    RecordError,
    RunManifest,
    RunRequest,
    WorkerInfo,
    attempt_record_path,
    clean_message,
    completion_metadata,
    decode_attempt_record,
    decode_run_manifest,
    is_attempt_id,
    new_attempt_id,
    run_manifest_path,
    task_id_for,
    task_record_from_completion,
    utc_now,
)
from live_dashboard.revision import code_revision
from live_dashboard.storage import DirectoryStore

if TYPE_CHECKING:
    from pymc_generator.data_generator import DataGenerator
    from pymc_generator.publication import FilePublicationStore
    from pymc_generator.sampler import SCMPrior

# The library's cell-fill failure: a cell could not accept its worlds within the top-up
# rounds. Only this outcome is a scientific result (a task with zero accepted worlds);
# any other exception is an infrastructure or code failure and fails the attempt. The
# message carries the failing cell's tallies (the library reports no other cell's), which
# the zero-world completion reports.
_EXHAUSTED = re.compile(
    r"tasks accepted after \d+ rounds\b.*?rejected (?P<rejected>\d+)/(?P<evaluated>\d+) "
    r"evaluated candidates, and (?P<failures>\d+)/\d+ rounds"
)
# Stop before a task when less than this much time, or twice the longest task so far,
# is left before the deadline (the time the platform kills the attempt), so the attempt
# ends cleanly instead of being killed in the middle of a task.
_MIN_DEADLINE_MARGIN_S = 90.0
_ENVIRONMENT_PACKAGES = ("numpy", "scipy", "pytensor", "pymc", "pymc-marketing")
_CONFLICT_FIELDS = ("config_sha256", "code_sha256", "environment")


class RequestError(ValueError):
    """The request cannot be run as asked (demo bounds, unknown preset, bad attempt id).

    Raised before anything is written; retrying the same request cannot succeed.
    """


class RunConflictError(Exception):
    """The stored run cannot be resumed by this generator (or this request).

    Raised before anything is written: resuming would mix outputs of two generators.
    """


# -- presets ---------------------------------------------------------------------------

# Verbatim from the trajectory-showcase notebook (docs/examples/dev/trajectory-showcase.ipynb,
# cell 7), so a live run shows the same variety the notebook reports on.
_SHOWCASE_COUNT_RANGES = {
    "n_treatments_active_range": (1, 10),
    "n_covariates_active_range": (1, 10),
}
_SHOWCASE_PER_INPUT = {  # applied to channels and controls alike
    # Zeros come only from flighting: on for W of every P weeks, at a random phase.
    "flighting_inclusion_prob": 10 / 11,
    "flighting_period_weeks_range": (14, 26),  # P: integer uniform
    "flighting_duty_range": (0.03, 0.99),  # W = clip(floor(duty * P + 0.5), 1, P - 1)
    "onset_inclusion_prob": 0.0,  # fixed by the design (the notebook report asserts it)
    "offset_inclusion_prob": 0.0,  # fixed by the design (the notebook report asserts it)
    "trend_inclusion_prob": 1.0,
    "seasonal_inclusion_prob": 0.5,
    "seasonal_period_weeks_range": (52.0, 52.0),
    "level_jump_inclusion_prob": 0.5,
    "level_jump_count": 1,  # held steps per input that carries a jump
    "pulse_inclusion_prob": 0.5,
    "hf_inclusion_prob": 1.0,
}
_SHOWCASE_KNOBS = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,
    "n_time_steps": 104,
    **_SHOWCASE_COUNT_RANGES,
    "draws_per_cell": 1,  # fixed by the design (the notebook report asserts it)
    "outcome_std_mode": "relative",  # fixed by the design (the notebook report asserts it)
    # Effect sizes: each input's contribution at its reference level, and the baseline level.
    "treatment_reference_contribution_range": (0.0, 3.0),
    "covariate_reference_contribution_range": (0.0, 3.0),
    "covariate_reference_scale": 1.0,
    "rw_baseline_mean_range": (0.05, 8.0),
    "baseline_floor": 0.0,
    # Input levels, volatility and smoothness.
    "rw_positive_mean_range": (0.1, 8.0),  # channel walk mean, before softplus
    "rw_treatment_std_range": (0.1, 1.2),  # channel walk std, relative to its level
    "rw_covariate_mean_range": (0.1, 2.0),
    "rw_std_sigma": 0.5,  # control walk std ~ HalfNormal(rw_std_sigma)
    "rw_smoothness_alpha": 1.0,  # walk smoothness ~ Beta(alpha, beta)
    "rw_smoothness_beta": 1.0,
    "rw_smoothness_max_weeks": 26,  # smoothness 1 = a 26-week moving average
    # Baseline walk and outcome noise std, relative to the channels' joint effect size.
    "rw_baseline_std_range": (0.0, 0.093),
    "rw_outcome_std_range": (0.01, 0.028),
    # Texture (the make_scm_prior defaults, echoed so every range is visible here).
    "treatment_hf_sigma_range": (0.08, 0.6),
    "treatment_pulse_prob_range": (0.0, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "covariate_hf_sigma_range": (0.1, 0.8),
    "covariate_pulse_prob_range": (0.0, 0.25),
    "covariate_pulse_amp_range": (0.5, 3.0),
    # Level forms.
    "treatment_trend_log_change_range": (-1.0, 1.0),
    "covariate_trend_change_range": (-1.0, 1.0),
    "treatment_seasonal_amplitude_range": (0.1, 0.5),
    "covariate_seasonal_amplitude_range": (0.2, 1.0),
    "treatment_level_jump_factor_range": (0.5, 2.0),  # log-uniform
    "covariate_level_jump_size_range": (-1.0, 1.0),
    # Response mechanisms: geometric carryover and Michaelis–Menten saturation.
    "carryover_alpha_range": (0.05, 0.95),
    **{
        f"{role}_{key}": value
        for role in ("treatment", "covariate")
        for key, value in _SHOWCASE_PER_INPUT.items()
    },
}
_SHOWCASE_MM_KAPPA_MULT_RANGE = (0.05, 20.0)  # log-uniform


def _showcase_variety(worlds_per_task: int) -> SCMPrior:
    """The notebook's variety design, one world per cell, ``worlds_per_task`` cells.

    Documented deviation: the notebook allocates active counts with a stratified design
    over its whole cohort, which a stream of small independent tasks cannot carry, so
    counts are drawn independently per cell (``active_count_allocation="independent"``).
    The notebook's config seed is not set: every task passes its own seed.
    """
    from pymc_generator import make_scm_prior
    from pymc_generator.sampler import SATURATION_FAMILY_KEYS, SCMPrior

    shapes = SCMPrior().saturation_prior_ranges
    shapes["michaelis_menten"]["kappa_mult"] = _SHOWCASE_MM_KAPPA_MULT_RANGE
    edges = dict.fromkeys(("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"), (0, 0))
    edges.update(
        cy=(_SHOWCASE_KNOBS["n_treatments"],) * 2, zy=(_SHOWCASE_KNOBS["n_covariates"],) * 2
    )
    return make_scm_prior(
        **_SHOWCASE_KNOBS,
        n_cells=worlds_per_task,
        edge_budget=edges,
        trajectories="composable",
        active_count_allocation="independent",
        carryover_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        saturation_family_probs={
            name: float(name == "michaelis_menten") for name in SATURATION_FAMILY_KEYS
        },
        saturation_prior_ranges=shapes,
        mm_scale_prior="log_uniform",
        record_walk_parameters=True,
    )


def _small(worlds_per_task: int) -> SCMPrior:
    """A small layout (about a second per two-world task) for tests and smoke runs."""
    from pymc_generator import make_scm_prior

    return make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
        n_cells=worlds_per_task,
        draws_per_cell=1,
        n_treatments_active_range=(1, 3),
        n_covariates_active_range=(1, 2),
        treatment_cv_floor=0.5,
    )


# Two variety styles that differ only in the saturation curve: geometric carryover with
# Michaelis-Menten (``geometric-mm``) or logistic (``geometric-logistic``) saturation.
# 10 channels and 10 controls (1-10 of each active per world), every active input acting
# only on the target, positive effects, and wide spreads of trends, zeros, smoothness,
# baseline shapes and channel / control / baseline shares (README, "Two-style variety").
_VARIETY_PER_INPUT: dict[str, Any] = {  # applied to channels and controls alike
    # Zeros: on/off blocks (on W of every P weeks), a late start, an early stop.
    "flighting_inclusion_prob": 0.5,
    "flighting_period_weeks_range": (4, 16),
    "flighting_duty_range": (0.05, 0.95),
    "onset_inclusion_prob": 0.15,
    "offset_inclusion_prob": 0.15,
    # Shapes.
    "trend_inclusion_prob": 0.75,
    "seasonal_inclusion_prob": 0.4,
    "seasonal_period_weeks_range": (52.0, 52.0),
    "level_jump_inclusion_prob": 0.3,
    "level_jump_count": 1,
    # Roughness on top of the smoothed walk: weekly jitter and pulses.
    "hf_inclusion_prob": 0.4,
    "pulse_inclusion_prob": 0.3,
}
_VARIETY_KNOBS: dict[str, Any] = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,  # the library needs >= 1; the edge budget gives it no arrows (contributes 0)
    "n_time_steps": 104,
    "n_treatments_active_range": (1, 10),
    "n_covariates_active_range": (1, 10),
    "draws_per_cell": 1,
    # Baseline walk and noise in target units: the relative mode would scale them by the
    # channels' asymptotic coefficients, which near-linear saturation makes very large.
    "outcome_std_mode": "absolute",
    "rw_baseline_std_sigma": 0.8,
    "rw_outcome_std_sigma": 0.4,
    # Effect sizes: each input's contribution at its usual level; the baseline level.
    # Channels and controls share one range with the mean of 1-3 (so neither dominates
    # and the channel/control/baseline split stays even), widened around that mean down
    # to a 0.1 floor: reference effects up to 39x apart within a world.
    "treatment_reference_contribution_range": (0.1, 3.9),
    "covariate_reference_contribution_range": (0.1, 3.9),
    "covariate_reference_scale": 5.5,  # controls sit near 5.5
    "rw_baseline_mean_range": (2.0, 16.0),
    "baseline_floor": 0.0,
    # Channel levels; smoothness ~ Beta(0.6, 0.6): many very smooth and many jagged walks
    # (smoothness 1 is a 26-week moving average).
    "rw_positive_mean_range": (0.3, 6.0),
    "rw_treatment_std_range": (0.1, 1.0),
    "rw_smoothness_alpha": 0.6,
    "rw_smoothness_beta": 0.6,
    "rw_smoothness_max_weeks": 26,
    # Controls are additive and unfloored: a level well above their swings keeps them
    # positive (no negative value in 5,000+ sampled control series).
    "rw_covariate_mean_range": (4.0, 7.0),
    "rw_std_sigma": 0.2,
    # Texture (control jitter and pulses scale with the control's own walk spread).
    "treatment_hf_sigma_range": (0.03, 0.6),
    "treatment_pulse_prob_range": (0.02, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "covariate_hf_sigma_range": (0.2, 1.6),
    "covariate_pulse_prob_range": (0.02, 0.25),
    "covariate_pulse_amp_range": (0.5, 3.0),
    # Shapes.
    "treatment_trend_log_change_range": (-1.5, 1.5),
    "covariate_trend_change_range": (-2.0, 3.0),
    "treatment_seasonal_amplitude_range": (0.1, 0.6),
    "covariate_seasonal_amplitude_range": (0.2, 0.8),
    "treatment_level_jump_factor_range": (0.5, 2.0),
    "covariate_level_jump_size_range": (-0.6, 1.5),
    "carryover_alpha_range": (0.05, 0.9),  # geometric adstock decay
    **{
        f"{role}_{key}": value
        for role in ("treatment", "covariate")
        for key, value in _VARIETY_PER_INPUT.items()
    },
    "covariate_trend_inclusion_prob": 0.9,  # control trends are additive: give more of them
}
_VARIETY_SHAPES: dict[str, dict[str, tuple[float, float]]] = {
    "michaelis_menten": {"kappa_mult": (0.1, 10.0)},  # log-uniform: saturated to ~linear
    "logistic": {"lam": (0.3, 5.0)},  # uniform: the library has no log-uniform option for lam
}


def _variety(saturation: str, worlds_per_task: int) -> SCMPrior:
    """One two-style variety configuration, one world per cell, ``worlds_per_task`` cells."""
    from pymc_generator import make_scm_prior
    from pymc_generator.sampler import SATURATION_FAMILY_KEYS, SCMPrior

    shapes = SCMPrior().saturation_prior_ranges
    shapes[saturation].update(_VARIETY_SHAPES[saturation])
    # Every active channel and control points only to the target: no channel->channel,
    # control->channel, control->control or latent arrows (those budgets are pinned to 0).
    edges: dict[str, int | tuple[int, int]] = dict.fromkeys(
        ("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"), (0, 0)
    )
    edges.update(cy=(10, 10), zy=(10, 10))
    return make_scm_prior(
        **_VARIETY_KNOBS,
        n_cells=worlds_per_task,
        edge_budget=edges,
        trajectories="composable",
        active_count_allocation="independent",
        carryover_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        saturation_family_probs={
            name: float(name == saturation) for name in SATURATION_FAMILY_KEYS
        },
        saturation_prior_ranges=shapes,
        mm_scale_prior="log_uniform",
        record_walk_parameters=True,
    )


# Structural splits (live_dashboard.structural): the two-style variety above (run
# two-styles-64091b9843a5) on 208 weeks, for every saturation x adstock split of a design,
# in five texture regimes, with columns that may carry no effect. Every world has 1-10
# channel columns and 1-10 control columns; a uniform 1..columns of each have an effect
# (an arrow to the target) and the others are data without effect.
def _per_input(**knobs: Any) -> dict[str, Any]:
    """``name=value`` for channels (``treatment_name``) and controls (``covariate_name``)."""
    return {
        f"{role}_{key}": value
        for role in ("treatment", "covariate")
        for key, value in knobs.items()
    }


_SPLIT_KNOBS: dict[str, Any] = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,  # the library needs >= 1; the edge budget gives it no arrows
    "n_time_steps": 208,
    "n_treatments_active_range": (1, 10),
    "n_covariates_active_range": (1, 10),
    "draws_per_cell": 1,
    "outcome_std_mode": "absolute",
    # Effects at the usual level of each input with an effect; baseline level and spread.
    "treatment_reference_contribution_range": (0.1, 3.9),
    "covariate_reference_contribution_range": (0.1, 3.9),
    "covariate_reference_scale": 8.0,  # controls sit near 8
    "rw_baseline_mean_range": (3.0, 9.0),
    "rw_baseline_std_sigma": 1.5,
    "baseline_floor": 0.0,
    # Channel levels and walk spread (relative to the level); small spreads keep many
    # channels flat. No floor on a channel's variation: the library applies it only to
    # channels with an effect, so a flat column would reveal that it has none.
    "rw_positive_mean_range": (0.3, 6.0),
    "rw_treatment_std_range": (0.05, 0.5),
    "treatment_cv_floor": 0.0,
    "rw_smoothness_max_weeks": 26,
    # Controls are additive and signed in the library: a level of 6-10 keeps a control
    # above zero through a 4.5 down-trend, a 0.6 season and a 0.5 drop; the worker
    # discards the rare world whose control still goes below zero.
    "rw_covariate_mean_range": (6.0, 10.0),
    "rw_std_sigma": 0.2,
    # Trajectories: symmetric trends (up, flat or down), seasons and level jumps.
    "treatment_trend_inclusion_prob": 0.45,
    "covariate_trend_inclusion_prob": 1.0,
    "treatment_trend_log_change_range": (-1.2, 1.2),
    "covariate_trend_change_range": (-4.5, 4.5),
    "treatment_seasonal_amplitude_range": (0.1, 0.6),
    "covariate_seasonal_amplitude_range": (0.2, 0.6),
    "treatment_level_jump_factor_range": (0.5, 2.0),
    "covariate_level_jump_size_range": (-0.5, 1.5),
    "carryover_alpha_range": (0.05, 0.9),  # geometric adstock decay
    # Texture magnitudes of the reference (the hectic regime widens them).
    "treatment_hf_sigma_range": (0.03, 0.6),
    "treatment_pulse_prob_range": (0.02, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "covariate_hf_sigma_range": (0.2, 1.6),
    "covariate_pulse_prob_range": (0.02, 0.25),
    "covariate_pulse_amp_range": (0.5, 3.0),
    **_per_input(
        flighting_period_weeks_range=(4, 16),
        flighting_duty_range=(0.05, 0.95),
        seasonal_inclusion_prob=0.4,
        seasonal_period_weeks_range=(52.0, 52.0),
        level_jump_count=1,
    ),
}
# Walk smoothness ~ Beta(alpha, beta) per walk; smoothness 1 is a 26-week moving average,
# 0 a raw random walk. The library draws every walk of a world (each channel, each control
# and the baseline) from one prior, so walk smoothness is set per regime.
_SMOOTH_WALKS = {"rw_smoothness_alpha": 15.0, "rw_smoothness_beta": 1.0}  # 96 % above 0.8
_RAW_WALKS = {"rw_smoothness_alpha": 1.0, "rw_smoothness_beta": 20.0}  # 91 % below 0.115
# Jitter and pulses on every channel and control, with on/off blocks, starts, stops, jumps.
_SPIKY_INPUTS: dict[str, Any] = {
    "treatment_hf_sigma_range": (0.2, 0.8),
    "treatment_pulse_prob_range": (0.05, 0.3),
    "treatment_pulse_amp_range": (0.8, 3.0),
    "covariate_hf_sigma_range": (0.8, 2.0),
    "covariate_pulse_prob_range": (0.05, 0.3),
    "covariate_pulse_amp_range": (1.0, 3.0),
    **_per_input(
        hf_inclusion_prob=1.0,
        pulse_inclusion_prob=1.0,
        flighting_inclusion_prob=0.3,
        onset_inclusion_prob=0.15,
        offset_inclusion_prob=0.15,
        level_jump_inclusion_prob=0.4,
    ),
}
# No jitter, pulses, on/off blocks, late starts, early stops or jumps.
_PLAIN_INPUTS = _per_input(
    hf_inclusion_prob=0.0,
    pulse_inclusion_prob=0.0,
    flighting_inclusion_prob=0.0,
    onset_inclusion_prob=0.0,
    offset_inclusion_prob=0.0,
    level_jump_inclusion_prob=0.0,
)
# A jagged baseline is visible only when its walk is large next to its level: these
# regimes draw its spread from HalfNormal(2.5) around a level of 5-11.
_JAGGED_BASELINE = {"rw_baseline_mean_range": (5.0, 11.0), "rw_baseline_std_sigma": 2.5}
# Texture regime -> knobs (structural.REGIMES).
_REGIMES: Mapping[str, dict[str, Any]] = MappingProxyType(
    {
        # Channels, controls and baseline all smooth at once.
        "calm": {**_SMOOTH_WALKS, **_PLAIN_INPUTS, "rw_outcome_std_sigma": 0.15},
        # All hectic at once: raw walks, spiky channels and controls, loud outcome noise.
        "hectic": {
            **_RAW_WALKS,
            **_SPIKY_INPUTS,
            **_JAGGED_BASELINE,
            "rw_outcome_std_sigma": 0.8,
        },
        # Spiky channels and controls over a smooth baseline.
        "spiky-inputs": {**_SMOOTH_WALKS, **_SPIKY_INPUTS, "rw_outcome_std_sigma": 0.4},
        # A jagged baseline under smooth channels and controls: every walk is raw, so the
        # channels' and controls' walks are made nearly flat (their trends and seasons
        # carry them).
        "hectic-baseline": {
            **_RAW_WALKS,
            **_PLAIN_INPUTS,
            **_JAGGED_BASELINE,
            "rw_outcome_std_sigma": 0.8,
            "rw_treatment_std_range": (0.01, 0.04),
            "rw_std_sigma": 0.02,
            **_per_input(seasonal_inclusion_prob=0.6),
        },
        # Each input on its own (the reference's walks and gates, less jitter): Beta(0.6,
        # 0.6) gives many very smooth and many raw walks in one world.
        "mixed": {
            "rw_smoothness_alpha": 0.6,
            "rw_smoothness_beta": 0.6,
            "rw_outcome_std_sigma": 0.4,
            **_per_input(
                hf_inclusion_prob=0.25,
                pulse_inclusion_prob=0.2,
                flighting_inclusion_prob=0.5,
                onset_inclusion_prob=0.15,
                offset_inclusion_prob=0.15,
                level_jump_inclusion_prob=0.3,
            ),
        },
    }
)
_SPLIT_SHAPES: dict[str, dict[str, tuple[float, float]]] = {
    **_VARIETY_SHAPES,  # hill and tanh keep the library's shape ranges
}


def _structural(saturation: str, adstock: str, regime: str, worlds_per_task: int) -> SCMPrior:
    """One split's configuration in one texture regime, one world per cell."""
    from pymc_generator import make_scm_prior
    from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior

    shapes = SCMPrior().saturation_prior_ranges
    for family, ranges in _SPLIT_SHAPES.items():
        shapes[family].update(ranges)
    # Channels and controls with an effect point only to the target; how many of a world's
    # columns have one is uniform in 1..columns (the library caps the budget at the columns).
    edges: dict[str, int | tuple[int, int]] = dict.fromkeys(
        ("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"), (0, 0)
    )
    edges.update(cy=(1, 10), zy=(1, 10))
    return make_scm_prior(
        **{**_SPLIT_KNOBS, **_REGIMES[regime]},
        n_cells=worlds_per_task,
        edge_budget=edges,
        trajectories="composable",
        active_count_allocation="independent",
        carryover_family_probs={name: float(name == adstock) for name in CARRYOVER_FAMILY_KEYS},
        saturation_family_probs={
            name: float(name == saturation) for name in SATURATION_FAMILY_KEYS
        },
        saturation_prior_ranges=shapes,
        mm_scale_prior="log_uniform",
        record_walk_parameters=True,
    )


# Member presets of every structural-split design, ``<saturation>-<adstock>-<regime>``.
# Only their design runs them (with its seeds and its control filter), never a run alone.
DESIGN_PRESETS: Mapping[str, Callable[[int], SCMPrior]] = MappingProxyType(
    {
        structural.member_name(split, regime): functools.partial(
            _structural, split.saturation, split.adstock, regime
        )
        for design in structural.DESIGNS.values()
        for split in design.splits
        for regime in design.regimes
    }
)


# Preset name -> configuration for a task of ``worlds_per_task`` worlds (one per cell, so
# a successful task accepts exactly its target). Read-only: runs are named by preset.
PRESETS: Mapping[str, Callable[[int], SCMPrior]] = MappingProxyType(
    {
        "showcase-variety": _showcase_variety,
        "small": _small,
        "geometric-mm": functools.partial(_variety, "michaelis_menten"),
        "geometric-logistic": functools.partial(_variety, "logistic"),
    }
)
# Mix name -> the presets its run cycles through: task ``i`` uses member ``i % n``, so the
# members alternate and, with a multiple of ``n`` tasks, finish with equal world counts.
MIXES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"geometric-mm-logistic": ("geometric-mm", "geometric-logistic")}
)


def preset_cycle(
    name: str,
    presets: Mapping[str, Callable[[int], SCMPrior]] = PRESETS,
    mixes: Mapping[str, tuple[str, ...]] = MIXES,
) -> tuple[str, ...]:
    """The presets a run's tasks cycle through: task ``i`` uses ``cycle[i % len(cycle)]``.

    A preset is a one-preset cycle and a structural-split design (``structural.DESIGNS``)
    the cycle of its members (``DESIGN_PRESETS``). ``RequestError`` if ``name`` is none of
    these, is more than one, or is a mix that does not name one or more known presets (a
    mix of mixes included).
    """
    design = structural.DESIGNS.get(name)
    if design is not None:
        if name in presets or name in mixes:
            raise RequestError(f"{name!r} names both a structural-split design and a preset")
        cycle = design.cycle
        unknown = sorted({member for member in cycle if member not in DESIGN_PRESETS})
        if unknown:
            raise RequestError(f"design {name!r} needs unknown presets: {unknown}")
        return cycle
    if name in mixes:
        if name in presets:
            raise RequestError(f"{name!r} names both a preset and a preset mix")
        members = mixes[name]
        if isinstance(members, str) or not members:
            raise RequestError(f"preset mix {name!r} must name one or more presets")
        cycle = tuple(members)
        unknown = sorted({member for member in cycle if member not in presets})
        if unknown:
            raise RequestError(f"preset mix {name!r} names unknown presets: {unknown}")
        return cycle
    if name in presets:
        return (name,)
    known = ", ".join(sorted({*presets, *mixes, *structural.DESIGNS}))
    raise RequestError(f"unknown preset {name!r}; known: {known}")


# -- generator identity ----------------------------------------------------------------


def _code_sha256() -> str:
    """Digest of the source that determines generated bytes and published records.

    Covers every ``*.py`` of the imported ``pymc_generator`` package plus this module, the
    contract and the structural-split designs, by relative POSIX path, so the digest is the
    same on a laptop and in a container that copied the same sources. Each file is framed
    by its path and length, which keeps the concatenation unambiguous.
    """
    import pymc_generator

    package = Path(pymc_generator.__file__).resolve().parent
    files = {
        path.relative_to(package.parent).as_posix(): path
        for path in package.rglob("*.py")
        if "__pycache__" not in path.relative_to(package).parts
    }
    here = Path(__file__).resolve().parent
    for name in ("generation.py", "publication.py", "structural.py"):
        files[f"{here.name}/{name}"] = here / name
    digest = hashlib.sha256()
    for name in sorted(files):
        data = files[name].read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    """Interpreter, machine and numerical package versions (never the project itself)."""
    environment = {"python": platform.python_version(), "machine": platform.machine()}
    for name in _ENVIRONMENT_PACKAGES:
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = "unknown"
    return environment


def _code_revision() -> str | None:
    """Git provenance: the deployment's recorded revision, else the local checkout's."""
    configured = os.environ.get(REVISION_ENV, "").strip()
    if configured:
        return configured
    return code_revision(Path(__file__).resolve().parent.parent)


def generator_info(
    preset: str, cycle: Sequence[str], configs: Mapping[str, SCMPrior]
) -> GeneratorInfo:
    """The effective generator of a run: what must match for a resume to be allowed.

    ``cycle`` is the run's ``preset_cycle`` and ``configs`` holds each member's
    configuration. A single preset records its configuration; a mix records its cycle and
    every member's configuration, so changing any member refuses a resume of the mix.
    """
    import pymc_generator
    from pymc_generator.slots import CORPUS_SCHEMA_VERSION

    if len(cycle) == 1:
        config = dataclasses.asdict(configs[cycle[0]])
    else:
        config = {
            "cycle": list(cycle),
            "presets": {name: dataclasses.asdict(configs[name]) for name in dict.fromkeys(cycle)},
        }
    return GeneratorInfo(
        preset=preset,
        config=config,
        pymc_generator_version=pymc_generator.__version__,
        corpus_schema_version=CORPUS_SCHEMA_VERSION,
        code_sha256=_code_sha256(),
        code_revision=_code_revision(),
        environment=_environment(),
    )


# -- run manifest and attempt record ---------------------------------------------------


def _stored_manifest(store: DirectoryStore, run_id: str) -> RunManifest | None:
    """The committed manifest of ``run_id``; ``None`` if absent; ``RecordError`` if unusable."""
    path = run_manifest_path(run_id)
    size = store.size(path)
    if size is None:
        return None
    if size > MAX_RECORD_BYTES:
        raise RecordError(f"run.json exceeds {MAX_RECORD_BYTES} bytes")
    return decode_run_manifest(store.read_bytes(path), run_id=run_id)


def _open_run(
    store: DirectoryStore, request: RunRequest, generator: GeneratorInfo, created_at: datetime
) -> RunManifest:
    """Create the run manifest, or check that the stored one names this generator."""
    run_id = request.run_id
    data = RunManifest(
        run_id=run_id, created_at=created_at, request=request, generator=generator
    ).encode()
    # Decoding also normalises the current generator exactly like the stored one, so the
    # comparison below cannot report a difference that only the encoding introduced.
    current = decode_run_manifest(data, run_id=run_id)
    try:
        stored = _stored_manifest(store, run_id)
    except RecordError as error:
        raise RunConflictError(f"run {run_id} has an unusable run.json: {error}") from error
    if stored is None:
        store.write_new(run_manifest_path(run_id), data)
        store.commit()
        return current
    differing = [
        name
        for name in _CONFLICT_FIELDS
        if getattr(stored.generator, name) != getattr(current.generator, name)
    ]
    if differing:
        raise RunConflictError(
            f"run {run_id} was created by a different generator ({', '.join(differing)} "
            "differ); resuming would mix the outputs of two generators"
        )
    return stored


@dataclass
class _Attempt:
    """The provisional attempt record this writer owns and rewrites as it progresses."""

    store: DirectoryStore
    manifest: RunManifest
    attempt_id: str
    worker: WorkerInfo
    started_at: datetime
    deadline_at: datetime | None
    published: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    current_task_index: int | None = None

    def __post_init__(self) -> None:
        # Records reject empty names: a worker that found no GPU name reports none.
        self.worker = dataclasses.replace(self.worker, gpu_name=self.worker.gpu_name or None)

    def save(
        self, state: AttemptState, updated_at: datetime, error: ErrorInfo | None = None
    ) -> None:
        data = AttemptRecord(
            run_id=self.manifest.run_id,
            attempt_id=self.attempt_id,
            state=state,
            started_at=self.started_at,
            updated_at=updated_at,
            deadline_at=self.deadline_at,
            current_task_index=self.current_task_index,
            published_task_ids=tuple(self.published),
            reused_task_ids=tuple(self.reused),
            error=error,
            worker=self.worker,
        ).encode()
        decode_attempt_record(data, manifest=self.manifest, attempt_id=self.attempt_id)
        self.store.write_replace(attempt_record_path(self.manifest.run_id, self.attempt_id), data)
        self.store.commit()


# -- tasks -----------------------------------------------------------------------------


class _Barriers:
    """The store's commit and refresh, as the Volume handle ``pymc_generator.publication``
    drives (no-ops for a local directory, ``volume.commit``/``reload`` on Modal)."""

    def __init__(self, store: DirectoryStore) -> None:
        self._store = store

    def commit(self) -> None:
        self._store.commit()

    def reload(self) -> None:
        self._store.refresh()


def _publication(store: DirectoryStore) -> FilePublicationStore:
    """The task-output store over ``store``'s root, with ``store``'s barriers."""
    from pymc_generator.publication import MountedVolumePublicationStore

    return MountedVolumePublicationStore(store.root, _Barriers(store))


def _published_tasks(
    publication: FilePublicationStore, manifest: RunManifest
) -> tuple[set[str], dict[str, str]]:
    """Tasks with a completion whose record, payload and corpus verify, and the reason
    each other task with a completion (or an unresolved one) cannot be reused.

    A completion must also be a task of this run as the dashboard reads it (its seed,
    split, target and timing), so a run is never resumed past a task the reader rejects.
    ``pymc_generator.publication`` refuses to publish a task past an unresolved or
    unusable earlier attempt until it is reconciled, so such a task stops the attempt
    before anything is generated for it.
    """
    candidates, unresolved = publication.discover(manifest.run_id)
    published: set[str] = set()
    unusable = {reference.split("/")[3]: message for reference, message in unresolved.items()}
    for candidate in candidates:
        task_id = candidate["record"]["task_id"]
        if task_id in published:
            continue
        try:
            task_record_from_completion(manifest, candidate["record"])
            publication.read_output(candidate["reference"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            unusable.setdefault(task_id, f"{type(error).__name__}: {error}")
            continue
        published.add(task_id)
    return published, {task: reason for task, reason in unusable.items() if task not in published}


def _negative_worlds(corpus: Mapping[str, Any]) -> int:
    """How many worlds have a control value or a control contribution below zero.

    Checks every column of every world (columns without an effect too: a control's sign
    must not reveal whether it acts) on the arrays as published. NaN counts as negative;
    padded slots are zero.
    """
    import numpy as np

    present = np.asarray(corpus["covariate_active_mask"]) == 1
    values = np.where(present[:, None, :], np.asarray(corpus["covariates"]), 0.0)
    contributions = np.asarray(corpus["covariate_contribution"])
    usable = (values >= 0.0).all(axis=(1, 2)) & (contributions >= 0.0).all(axis=(1, 2))
    return int((~usable).sum())


@dataclass
class _Budget:
    """The deadline stop rule: no draw starts unless 90 s, and twice the longest draw so
    far, are left (both measured on the deadline's clock)."""

    clock: Callable[[], datetime]
    deadline: datetime | None
    longest_draw: float = 0.0

    def allows_a_draw(self) -> bool:
        if self.deadline is None:
            return True
        remaining = (self.deadline - self.clock()).total_seconds()
        return remaining >= max(_MIN_DEADLINE_MARGIN_S, 2 * self.longest_draw)

    def timed(self, draw: Callable[[], Any]) -> Any:
        started = self.clock()
        try:
            return draw()
        finally:
            self.longest_draw = max(self.longest_draw, (self.clock() - started).total_seconds())


def _generate_task(
    publication: FilePublicationStore,
    manifest: RunManifest,
    generator: DataGenerator,
    member: str,
    member_config: Mapping[str, Any],
    index: int,
    attempt: _Attempt,
    budget: _Budget,
) -> bool:
    """Generate task ``index`` and publish it as a completion; ``False`` if it stopped.

    A structural-split design publishes only worlds whose controls stay non-negative and
    aims at every task's target: a draw with a control below zero, or whose cells the
    library could not fill, is discarded and the task drawn again with its next redraw
    seed, at most ``structural.MAX_REDRAWS`` times; a task whose every draw is discarded
    is published with no worlds (other runs publish an unfillable task at once). Counts
    add up over every draw: worlds with a negative control are rejected candidates and
    the other worlds of a discarded draw are surplus. A redraw the deadline does not
    leave time for stops the task unpublished (a resume draws it again from its first
    seed).
    """
    request = manifest.request
    clock = budget.clock
    max_redraws = 0 if request.design is None else structural.MAX_REDRAWS
    started_at = clock()
    t0 = time.perf_counter()
    corpus = None
    candidates = rejected = draw_failures = published = negative_worlds = 0
    generation_seconds = 0.0
    failure: ErrorInfo | None = None
    exhausted = False
    redraw = 0
    for redraw in range(max_redraws + 1):
        if redraw and not budget.allows_a_draw():
            return False
        seed = structural.redraw_seed(request.task_seed(index), redraw)
        try:
            corpus = budget.timed(
                functools.partial(
                    generator.generate,
                    n_tasks=request.worlds_per_task,
                    seed=seed,
                    defer_cell_split=True,
                )
            )
        except RuntimeError as error:
            tally = _EXHAUSTED.search(str(error)) if type(error) is RuntimeError else None
            if tally is None:
                raise
            candidates += int(tally["evaluated"])
            rejected += int(tally["rejected"])
            draw_failures += int(tally["failures"])
            failure = ErrorInfo.from_exception(error)
            exhausted = True
            corpus = None
            continue
        diagnostics = corpus["diagnostics"]
        evaluated = int(diagnostics["n_draws_evaluated"])
        candidates += evaluated
        # The library reports a rate; the count is exact for any realistic total.
        rejected += round(float(diagnostics["rejection_rate"]) * evaluated)
        draw_failures += int(diagnostics["n_draw_failures"])
        generation_seconds += float(diagnostics["timing"]["elapsed_s"])
        negative = _negative_worlds(corpus) if max_redraws else 0
        if not negative:
            published = int(diagnostics["n_tasks"])
            failure = None
            break
        negative_worlds += negative
        failure = ErrorInfo(
            type="NegativeControls",
            message=f"draw {redraw + 1} of {max_redraws + 1} had a control below zero",
        )
        corpus = None
    if exhausted:
        # An unfillable draw reports no time of its own: count the task's wall time.
        generation_seconds = time.perf_counter() - t0
    finished_at = max(clock(), started_at)
    metadata = completion_metadata(
        manifest,
        task_index=index,
        member=member,
        member_config=member_config,
        attempt_id=attempt.attempt_id,
        worker=attempt.worker,
        candidates=candidates,
        rejected=rejected,
        published=published,
        started_at=started_at,
        finished_at=finished_at,
        generation_seconds=generation_seconds,
        draw_failures=draw_failures,
        generation_failure=failure,
        redraws=redraw,
        negative_controls=negative_worlds,
    )
    publication.publish(metadata, corpus)
    return True


def _publish_tasks(
    store: DirectoryStore,
    manifest: RunManifest,
    cycle: Sequence[str],
    configs: Mapping[str, SCMPrior],
    attempt: _Attempt,
    clock: Callable[[], datetime],
    deadline: datetime | None,
) -> str:
    """Reuse or generate every task in order; ``"stopped"`` when the deadline is near.

    Task ``index`` is generated by the preset ``cycle[index % len(cycle)]``.
    """
    from pymc_generator.data_generator import DataGenerator

    publication = _publication(store)
    generators = {name: DataGenerator(cfg) for name, cfg in configs.items()}
    # Each completion records its member's configuration as the manifest stores it.
    recorded = manifest.generator.config
    member_configs = {
        name: recorded if len(cycle) == 1 else recorded["presets"][name] for name in configs
    }
    published, unusable = _published_tasks(publication, manifest)
    budget = _Budget(clock, deadline)
    for index in range(manifest.n_tasks):
        task_id = task_id_for(index)
        if task_id in published:
            attempt.reused.append(task_id)
            continue
        if task_id in unusable:
            from pymc_generator.publication import PublicationError

            raise PublicationError(
                f"{task_id} has a completion that does not verify ({unusable[task_id]}); "
                "reconcile it before resuming"
            )
        if not budget.allows_a_draw():
            return "stopped"
        attempt.current_task_index = index
        attempt.save("running", clock())
        member = cycle[index % len(cycle)]
        if not _generate_task(
            publication,
            manifest,
            generators[member],
            member,
            member_configs[member],
            index,
            attempt,
            budget,
        ):
            return "stopped"
        attempt.published.append(task_id)
    return "finished"


# -- entry points ----------------------------------------------------------------------


def _check_request(
    request: RunRequest,
    attempt_id: str | None,
    presets: Mapping[str, Callable[[int], SCMPrior]],
    mixes: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Validate what the request alone decides, before the store is touched.

    Returns the presets the run's tasks cycle through (``preset_cycle``).
    """
    try:
        check_demo_bounds(request)
    except ValueError as error:
        raise RequestError(str(error)) from error
    if attempt_id is not None and not is_attempt_id(attempt_id):
        raise RequestError(
            f"attempt_id must look like 20260101T000000Z-0a1b2c3d, got {attempt_id!r}"
        )
    return preset_cycle(request.preset, presets, mixes)


def run_generation(
    store: DirectoryStore,
    request: RunRequest,
    *,
    worker: WorkerInfo,
    clock: Callable[[], datetime] = utc_now,
    deadline: datetime | None = None,
    attempt_id: str | None = None,
    presets: Mapping[str, Callable[[int], SCMPrior]] = PRESETS,
    mixes: Mapping[str, tuple[str, ...]] = MIXES,
) -> dict[str, Any]:
    """Publish every task of ``request`` that is not yet published, as one attempt.

    An invalid request (demo bounds, unknown preset or mix, malformed ``attempt_id``) raises
    ``RequestError`` and a stored run of another generator raises ``RunConflictError``,
    both before anything is written. Any other failure marks the attempt ``failed``
    (best effort) and propagates. ``deadline`` is the time after which the attempt
    cannot still be running (the platform's kill time), recorded as the attempt's
    ``deadline_at``: the attempt stops cleanly, instead of starting a task, once less
    than 90 s or twice its longest task so far is left. Returns ``{"status": "finished"
    | "stopped", "run_id", "attempt_id", "published": [task ids], "reused": [task ids]}``.
    The request's preset may name a mix (``MIXES``): task ``i`` then uses member ``i % n``.
    """
    cycle = _check_request(request, attempt_id, presets, mixes)
    store.refresh()
    source = DESIGN_PRESETS if request.design is not None else presets
    configs = {name: source[name](request.worlds_per_task) for name in dict.fromkeys(cycle)}
    started_at = clock()
    manifest = _open_run(store, request, generator_info(request.preset, cycle, configs), started_at)
    attempt = _Attempt(
        store=store,
        manifest=manifest,
        attempt_id=attempt_id or new_attempt_id(started_at),
        worker=worker,
        started_at=started_at,
        deadline_at=deadline,
    )
    try:
        attempt.save("running", started_at)
        status = _publish_tasks(store, manifest, cycle, configs, attempt, clock, deadline)
        attempt.current_task_index = None
        attempt.save("finished", clock())
    except BaseException as error:
        # Best effort: the original failure matters more than a failure to record it.
        with contextlib.suppress(Exception):
            attempt.save("failed", clock(), ErrorInfo.from_exception(error))
        raise
    return {
        "status": status,
        "run_id": manifest.run_id,
        "attempt_id": attempt.attempt_id,
        "published": list(attempt.published),
        "reused": list(attempt.reused),
    }


def run_generation_or_refuse(
    store: DirectoryStore,
    request: RunRequest,
    *,
    worker: WorkerInfo,
    expected_run_id: str | None = None,
    clock: Callable[[], datetime] = utc_now,
    deadline: datetime | None = None,
    attempt_id: str | None = None,
    presets: Mapping[str, Callable[[int], SCMPrior]] = PRESETS,
    mixes: Mapping[str, tuple[str, ...]] = MIXES,
) -> dict[str, Any]:
    """``run_generation``, but a request that must not run is refused, not raised.

    A platform retry cannot fix an invalid request (``RequestError``) or a run that must
    not be resumed (``RunConflictError``), so both are returned as ``{"status":
    "refused", "run_id", "attempt_id", "reason"}``. The refusal is recorded as a
    ``failed`` attempt of the run when the run has a usable manifest (under a fresh
    attempt id if the injected one is malformed; ``attempt_id`` is ``None`` without a
    manifest); the manifest itself is never written. ``expected_run_id`` is the run the
    launcher announced: a request naming another run is refused. Every other failure
    propagates, so the platform can retry it.
    """
    try:
        if expected_run_id is not None and expected_run_id != request.run_id:
            raise RunConflictError(
                f"the launcher expected run {expected_run_id}, the request names run "
                f"{request.run_id}"
            )
        return run_generation(
            store,
            request,
            worker=worker,
            clock=clock,
            deadline=deadline,
            attempt_id=attempt_id,
            presets=presets,
            mixes=mixes,
        )
    except (RequestError, RunConflictError) as error:
        return _refuse(store, request, error, worker, clock, deadline, attempt_id)


def _refuse(
    store: DirectoryStore,
    request: RunRequest,
    error: RequestError | RunConflictError,
    worker: WorkerInfo,
    clock: Callable[[], datetime],
    deadline: datetime | None,
    attempt_id: str | None,
) -> dict[str, Any]:
    """Record a refusal as a failed attempt of the request's run, when it has a manifest."""
    store.refresh()
    try:
        manifest = _stored_manifest(store, request.run_id)
    except RecordError:
        manifest = None
    recorded_id = None
    if manifest is not None:
        now = clock()
        attempt = _Attempt(
            store=store,
            manifest=manifest,
            attempt_id=attempt_id if is_attempt_id(attempt_id) else new_attempt_id(now),
            worker=worker,
            started_at=now,
            deadline_at=deadline,
        )
        attempt.save("failed", now, ErrorInfo.from_exception(error))
        recorded_id = attempt.attempt_id
    return {
        "status": "refused",
        "run_id": request.run_id,
        "attempt_id": recorded_id,
        "reason": clean_message(str(error)),
    }
