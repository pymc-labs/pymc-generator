"""Stratified corpus sampler for the additive causal SCM.

Each world (task) is an additive structural causal model over latent latent_unobserved
factors D, observed covariates Z, treatment treatments C, a baseline B and outcome Y.
The DAG is drawn per cell by :func:`sample_g_additive` (per-edge-type
Bernoulli rates or "pot" budgets over the extended 8-block layout). Each world
is then a PyMC model (:mod:`pymc_generator.world_model`) whose continuous
priors are pm distributions and whose noise is pm RVs; drawing it yields the
series with exact interventional decomposition targets — direct contributions,
per-covariate / per-confounder contributions, ``baseline_intrinsic`` and the
telescoping 3-source indirect split ``(cc, zc, dc)``.

The corpus schema is the dict-of-ndarrays documented in
:func:`sample_prior_predictive` (persist with ``pymc_generator.save_corpus``).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .active_counts import (
    ACTIVE_COUNT_ALLOCATIONS,
    active_count_weight_matrix,
    draw_active_counts,
    plan_active_counts,
    summarize_active_count_coverage,
)
from .signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    admitted_response_support_weeks,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from .slots import (
    CORPUS_ARRAY_FIELDS,
    CORPUS_SCHEMA_VERSION,
    EDGE_BASE_RATES,
    EDGE_TYPES_EXTENDED,
    MECHANISM_ARRAY_FIELDS,
    MECHANISM_REFERENCE_FIELDS,
    N_COVARIATES_DEMO,
    N_LATENT_DEMO,
    N_TIME_STEPS_DEMO,
    N_TREATMENTS_DEMO,
    PRIOR_COND_LAYOUT,
    PRIOR_COND_QUANTITIES,
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_INPUTS,
    TRAJECTORY_MAX_LOG_SHIFT,
    TRAJECTORY_PARAM_FIELDS,
    TRAJECTORY_PARAM_REPORTS,
    SlotLayout,
)
from .trajectories import (
    GATE_COMPONENTS,
    SCHEDULE_COMPONENTS,
    TEXTURE_COMPONENTS,
    min_on_weeks,
    structural_key,
    summarize_component_prevalence,
)

#: Canonical categorical orders for treatment-response mechanism family ids.
CARRYOVER_FAMILY_KEYS: tuple[str, ...] = ("none", "geometric", "weibull")
SATURATION_FAMILY_KEYS: tuple[str, ...] = (
    "linear",
    "hill",
    "logistic",
    "michaelis_menten",
    "tanh",
    "root",
)

#: Persisted outcome-process contract. Version 1 makes iid ``RW_Y`` and its
#: relative/absolute scale mode explicit so pre-change corpora cannot validate
#: as worlds drawn under the new outcome-noise semantics.
OUTCOME_NOISE_SEMANTICS = "baseline-walk-iid-outcome-noise"
OUTCOME_NOISE_VERSION = 1


def _default_carryover_family_probs() -> dict[str, float]:
    """Return a fresh default categorical distribution over carryover families."""
    return dict(zip(CARRYOVER_FAMILY_KEYS, (0.15, 0.425, 0.425), strict=True))


def _default_saturation_family_probs() -> dict[str, float]:
    """Return a fresh default categorical distribution over saturation families."""
    return dict(
        zip(
            SATURATION_FAMILY_KEYS,
            (0.15, 0.17, 0.17, 0.17, 0.17, 0.17),
            strict=True,
        )
    )


def _default_saturation_prior_ranges() -> dict[str, dict[str, tuple[float, float]]]:
    """Legacy shape supports, independently configurable for every prior."""
    return {
        "hill": {"slope": (1.0, 3.0), "kappa_mult": (0.7, 1.5)},
        "logistic": {"lam": (0.5, 3.0)},
        "michaelis_menten": {"kappa_mult": (0.7, 1.5)},
        "tanh": {"c": (0.3, 1.5)},
        "root": {"alpha": (0.3, 0.9)},
    }


#: Maximum draw rounds per cell before giving up (post-filter top-up loop).
MAX_TOPUPS_PER_CELL = 8

#: Default per-quantity width ranges for the prior-conditioning hyperprior
#: (ACE). Each width spans a useful fraction of that quantity's full prior
#: range, narrow enough to inform a cell without collapsing it to a point.
PRIOR_COND_DEFAULT_WIDTH_RANGES: dict[str, tuple[float, float]] = {
    "carryover_alpha": (0.05, 0.45),
    "hill_shape": (0.2, 1.6),
}

#: Bound the diagnostic search so extreme ``query_frac`` values cannot make
#: configuration validation unbounded.
MAX_QUERY_HORIZON_SEARCH_STEPS = 1_000_000

#: Existing series and base features retain float32 storage. New realised
#: trajectory/mechanism truth uses float64; this limit applies only to the
#: historical float32 arrays.
CORPUS_STORAGE_DTYPE = np.float32
CORPUS_STORAGE_MAX = float(np.finfo(CORPUS_STORAGE_DTYPE).max)


def _n_query(n_time_steps: int, query_frac: float) -> int:
    """Return query weeks from the canonical rounded query-fraction rule."""
    return int(round(query_frac * n_time_steps))


def _minimum_valid_query_horizon(
    n_time_steps: int, query_frac: float, response_support: int
) -> int | None:
    """Return the first valid candidate horizon at or above ``n_time_steps``, if bounded.

    ``response_support`` is the largest lag the admitted carryover kernels can
    reach (see :func:`~pymc_generator.signal_diagnostics.
    admitted_response_support_weeks`), i.e. the number of leading reported
    weeks whose treatment response depends on pre-window treatment.
    """
    for candidate in range(n_time_steps, n_time_steps + MAX_QUERY_HORIZON_SEARCH_STEPS + 1):
        n_query = _n_query(candidate, query_frac)
        if (
            0 < n_query < candidate
            and n_query <= candidate - 2
            and min(candidate - n_query, candidate // 2) >= response_support
        ):
            return candidate
    return None


def _reject_unrepresentable_bounds(name: str, value: object, lo: float, hi: float) -> None:
    """Reject a prior range the float32 corpus storage cannot hold.

    A range endpoint above :data:`CORPUS_STORAGE_MAX` is perfectly finite in the
    float64 draw and only overflows to ``inf`` at the storage cast, several
    hundred lines later — at which point generation dies on a finiteness
    assertion that names a downstream array rather than the config field that
    caused it. Rejecting here keeps the diagnosis attached to the knob the
    caller set. The bound is deliberately NOT clamped: silently shrinking a
    prior would change the world distribution behind the caller's back.
    """
    for bound in (lo, hi):
        if abs(bound) > CORPUS_STORAGE_MAX:
            raise ValueError(
                f"{name} bound {bound!r} exceeds the float32 corpus storage maximum "
                f"({CORPUS_STORAGE_MAX:.9g}); corpus arrays are persisted as "
                f"{np.dtype(CORPUS_STORAGE_DTYPE).name}, so this endpoint would "
                f"overflow to inf. Got {name}={value!r}"
            )


#: Relative slack for treatment coefficient bounds: the runtime response takes
#: a few rounding steps (input product, anchor division) this support
#: calculation does not replay. Hill widens it by its condition number.
_ROUNDING_MARGIN = 1.0 + 8.0 * float(np.finfo(np.float64).eps)


def _smallest_nonzero_uniform_magnitude(lo: float, hi: float) -> float | None:
    """Smallest nonzero ``|q|`` a float64 ``U(lo, hi)`` draw can produce.

    A draw is ``lo + (hi - lo) * u`` with ``u`` on the ``2**-53`` grid, so a
    support containing zero yields nonzero values far below either endpoint.
    Near zero the sum is a multiple of the finer quantum of its two terms;
    cancellation against ``lo`` needs a grid step within a factor two of it.
    ``None`` means the support is exactly zero.
    """
    magnitudes = [abs(x) for x in (lo, hi) if x != 0.0]
    if lo < hi and lo <= 0.0 <= hi:
        step = (hi - lo) * 2.0**-53
        magnitudes.append(step)
        if lo != 0.0 and step <= 2.0 * abs(lo):
            magnitudes.append(float(np.spacing(abs(lo))) / 2.0)
    if not magnitudes:
        return None
    return max(min(magnitudes), float(np.nextafter(0.0, 1.0)))


def _prior_cond_storage_resolution(support) -> float:
    """Narrowest conditioning width float32 ``prior_cond`` labels resolve.

    Storing an endpoint moves it by at most half a float32 spacing at the
    support's magnitude; wider intervals keep that under 1/16 of their width.
    """
    return 8 * float(np.finfo(np.float32).eps) * max(abs(float(x)) for x in support)


class _ReferenceRepresentabilityError(ValueError):
    """A drawn reference input or derived coefficient left its validated domain.

    Configuration validation bounds both on parameter support; this runtime
    check is the backstop for residual rounding. It is never retried, because
    resampling would silently truncate the configured target prior.
    """


#: Frame that marks an exception as having escaped a node PyTensor was
#: evaluating. Every linker funnels its ``except Exception`` through
#: ``pytensor.link.utils.raise_with_op``, which re-raises the ORIGINAL
#: exception object (annotating it only under non-default
#: ``config.exception_verbosity``), so the frame — not the type, the message,
#: or an attribute — is the one verbosity-independent marker available.
_PYTENSOR_RAISE_WITH_OP = ("pytensor.link.utils", "raise_with_op")

#: Exception types a node evaluation raises for a numeric/domain failure that a
#: DIFFERENT parameter draw can step around: an out-of-domain distribution
#: parameter produced by an upstream draw (``ValueError``), or an overflow /
#: divide error escaping ``numpy.errstate`` (``ArithmeticError``, which covers
#: ``FloatingPointError`` and ``OverflowError``). A ``TypeError``,
#: ``AttributeError``, ``KeyError`` or the like from the same place is a bug in
#: the graph, not a draw that landed badly, and must never be retried.
_RETRYABLE_DRAW_ERRORS = (ValueError, ArithmeticError)


def _is_retryable_draw_failure(exc: BaseException) -> bool:
    """Is ``exc`` a sporadic per-draw numeric failure worth resampling?

    The batch draw is the one place in generation where retrying is legitimate:
    a hierarchical draw can hand a downstream distribution an out-of-domain
    parameter, and a fresh seed simply lands elsewhere. Retrying anything else
    turns a repeatable programming error into a silent stream of empty batches,
    which is what the caller then misreads as a strict realism filter — so the
    predicate latent_unobserved BOTH that the failure is numeric AND that it escaped a
    node PyTensor was evaluating (rather than our own call frames around it).
    """
    if isinstance(exc, _ReferenceRepresentabilityError):
        return False
    if not isinstance(exc, _RETRYABLE_DRAW_ERRORS):
        return False
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if (frame.f_globals.get("__name__"), frame.f_code.co_name) == _PYTENSOR_RAISE_WITH_OP:
            return True
        tb = tb.tb_next
    return False


@dataclass
class SCMPrior:
    """Corpus generation knobs + prior-range constants for the additive SCM.

    Supports variable-size DAGs via padding to max sizes. Each cell gets
    active counts (n_treatments_active, n_covariates_active,
    n_latent_active) from configured ranges. Inactive nodes are zero-padded and
    masked via treatment_active_mask / covariate_active_mask /
    latent_active_mask.

    Key fields:
        n_treatments / n_covariates / n_latent: padded graph sizes (treatment
            treatments / observed covariates / hidden confounders).
        n_treatments_active_range / ...: per-cell active-count ranges; inactive
            nodes are zero-padded to the sizes above and masked.
        active_count_allocation / active_count_weights: how cells cover the
            grid of active treatment × covariate counts — independent uniform
            draws per cell (default) or a weighted stratified allocation (see
            :mod:`pymc_generator.active_counts`).

    Prefer building configs through
    :func:`pymc_generator.presets.make_scm_prior`, which pins the
    layout and enables the supported "diverse" treatment texture.
    """

    n_time_steps: int = N_TIME_STEPS_DEMO
    n_treatments: int = N_TREATMENTS_DEMO  # treatment treatments (the interventions / treatments)
    n_covariates: int = N_COVARIATES_DEMO  # observed covariates
    n_latent: int = N_LATENT_DEMO  # hidden confounders (latent latent_unobserved factors)
    n_cells: int = 50
    draws_per_cell: int = 20
    val_cell_frac: float = 0.2
    query_frac: float = 0.25
    p_long_horizon: float = 0.5  # fraction of tasks with 50% horizon (vs 25%)
    treatment_cv_floor: float = 0.08
    l_max: int = 8
    seed: int = 0

    # -- variable-size DAG support ----------------------------------------
    # Per-cell active-count ranges (inclusive). Each cell draws a random number
    # of active nodes in these ranges; nodes beyond the active count are
    # zero-padded to the fixed n_treatments/n_covariates/n_latent sizes and
    # masked. Default to the full size (fixed-size graphs) via the factory.
    n_treatments_active_range: tuple[int, int] = (4, 20)
    n_covariates_active_range: tuple[int, int] = (2, 10)
    n_latent_active_range: tuple[int, int] = (1, 5)
    # How cells cover the (n_treatments_active, n_covariates_active) grid of
    # the effective ranges above. "independent" draws both counts per cell;
    # "stratified" allocates the cells over the grid by ``active_count_weights``
    # (rows: treatment counts, columns: covariate counts; None = uniform), see
    # :mod:`pymc_generator.active_counts`. Latent counts are always drawn per
    # cell. Weights are validated in both modes and used only when stratified.
    active_count_allocation: Literal["independent", "stratified"] = "independent"
    active_count_weights: Sequence[Sequence[float]] | None = None

    # -- prior-range constants -------------------------------------------
    # Per-treatment treatment-response mechanism priors (realized as PyMC
    # distributions in pymc_generator.world_model.build_world_model):
    carryover_alpha_range: tuple[float, float] = (0.2, 0.8)
    # Carryover family probabilities by ``CARRYOVER_FAMILY_KEYS``.
    carryover_family_probs: dict[str, float] = field(
        default_factory=_default_carryover_family_probs
    )
    # Saturation family probabilities by ``SATURATION_FAMILY_KEYS``.
    saturation_family_probs: dict[str, float] = field(
        default_factory=_default_saturation_family_probs
    )
    saturation_prior_ranges: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=_default_saturation_prior_ranges
    )
    mm_scale_prior: Literal["uniform", "log_uniform"] = "uniform"
    # Weibull carryover prior ranges
    weibull_lam_range: tuple[float, float] = (2.0, 8.0)
    weibull_k_range: tuple[float, float] = (1.5, 4.0)

    # -- additive causal graph priors ----------------------------------------
    # New edge-type base rates (extended layout: dz, zc, cc, zz)
    dz_base_rate: float = 0.3  # D->Z base rate
    zc_base_rate: float = 0.3  # Z->C base rate
    cc_base_rate: float = 0.15  # C->C base rate (sparse)
    zz_base_rate: float = 0.1  # Z->Z base rate (sparse)
    # Linear coefficient ranges for the additive structural equations
    dc_coeff_range: tuple[float, float] = (0.1, 0.5)  # D->C loadings
    dz_coeff_range: tuple[float, float] = (0.1, 0.5)  # D->Z loadings
    zc_coeff_range: tuple[float, float] = (0.05, 0.3)  # Z->C loadings
    cc_coeff_range: tuple[float, float] = (0.05, 0.3)  # C->C (positive-only)
    zz_coeff_range: tuple[float, float] = (-0.2, 0.2)  # Z->Z (signed)
    dy_coeff_range: tuple[float, float] = (0.15, 0.45)  # D->Y loadings
    zy_coeff_range: tuple[float, float] = (0.1, 0.4)  # Z->Y loadings
    beta_additive_range: tuple[float, float] = (0.5, 2.0)  # treatment effects
    # Nominal input contributions before edge gates and any absorbing floor.
    # Treatment references are post-carryover, relative to the parameter-only
    # saturation anchor; signed covariates use a fixed positive input scale.
    treatment_reference_contribution_range: tuple[float, float] | None = None
    treatment_reference_multiplier: float = 1.0
    covariate_reference_contribution_range: tuple[float, float] | None = None
    covariate_reference_scale: float = 1.0
    # Random-walk and outcome-noise priors.
    rw_covariate_mean_range: tuple[float, float] = (-1.0, 1.0)  # covariate drive (Z)
    rw_positive_mean_range: tuple[float, float] = (0.5, 3.0)  # treatments
    rw_baseline_mean_range: tuple[float, float] = (3.0, 8.0)  # baseline level
    rw_std_sigma: float = 1.0  # HalfNormal prior for Z and absolute-mode RW_B std
    rw_baseline_std_sigma: float | None = None  # absolute-mode RW_B; None follows rw_std_sigma
    rw_outcome_std_sigma: float = 0.25  # absolute-mode iid outcome-noise std
    outcome_std_mode: Literal["relative", "absolute"] = "relative"
    # Relative-mode outcome amplitudes multiply
    # sqrt(sum_k((g_cy[k] * beta[k]) ** 2)), never a realized series.
    rw_baseline_std_range: tuple[float, float] = (0.000, 0.093)
    rw_outcome_std_range: tuple[float, float] = (0.010, 0.028)
    rw_smoothness_alpha: float = 2.0  # Beta prior alpha for smoothness
    rw_smoothness_beta: float = 2.0  # Beta prior beta for smoothness
    # 26 weeks (half a year) reproduces the CURRENT n_time_steps=104 reference
    # exactly at smoothness=1.0 (round(1.0 * 104 / 4) == 26), preserving its
    # drift character while making every other horizon consistent. This is a
    # config knob, not a constant, so drift timescale stays tunable
    # independently of n_time_steps — that flexibility is the point.
    rw_smoothness_max_weeks: int = 26
    # Optional shared innovation between the baseline and every treatment. When
    # enabled, rho is resolved per world and mixes their already-standardized
    # innovations without changing either marginal innovation variance.
    confounding_strength_range: tuple[float, float] | None = None

    # -- Treatment own-drive variation --------------------------------------
    # ``rw_treatment_std_range`` is the treatment random-walk standard-deviation
    # prior, relative to each treatment's own level (softplus of its walk mean).
    # That keeps treatment variation scale-free across small and large treatments.
    # High-frequency exogenous drive on the treatment's own pre-softplus input
    # comes from iid weekly noise (sigma ~ U(range)) and campaign pulses
    # (per-week probability ~ U(prob_range), amplitude ~ U(amp_range)).
    # Defaults retain the deprecated smooth-walk-only texture; make_scm_prior
    # enables the diverse texture, which is the supported prior.
    rw_treatment_std_range: tuple[float, float] = (0.15, 0.8)
    treatment_hf_sigma_range: tuple[float, float] = (0.0, 0.0)
    treatment_pulse_prob_range: tuple[float, float] = (0.0, 0.0)
    treatment_pulse_amp_range: tuple[float, float] = (0.5, 1.5)

    # -- Covariate texture ---------------------------------------------------
    # A covariate's own drive is otherwise a smoothed random walk, i.e. exactly
    # the function class the equally smooth baseline walk (RW_B) spans, so the
    # Z->Y coefficient trades off against baseline drift and is only weakly
    # identified. These knobs add the high-frequency content a smooth walk
    # cannot mimic — iid weekly noise (sigma ~ U(range)) and calendar pulses
    # (per-week probability ~ U(prob_range), amplitude ~ U(amp_range)) — which
    # is also what real covariates (promos, holidays, price steps) look like.
    # Both magnitudes are drawn RELATIVE to the covariate's own walk std, so they
    # are scale-free like the treatment factors; unlike the treatment pulse, the
    # covariate pulse is CENTRED (``amp * (fire - prob)``), because a covariate is
    # signed and its level is identified by ``rw_z_mean`` alone.
    # Defaults are inert: no graph term, magnitude/probability priors are
    # constants and consume no RNG. Same-environment model arrays are stable,
    # not archive bytes across schemas. make_scm_prior enables these controls.
    covariate_hf_sigma_range: tuple[float, float] = (0.0, 0.0)
    covariate_pulse_prob_range: tuple[float, float] = (0.0, 0.0)
    covariate_pulse_amp_range: tuple[float, float] = (0.0, 0.0)

    # -- Intercept floor ---------------------------------------------------
    # The intercept walk RW_B is signed, so the baseline — and, once the signed
    # D->Y / Z->Y terms are added, outcome — can dip below zero. Rare at the
    # default relative-mode scale (mean 3-8, amplitude <= ~0.12 x the treatment
    # amplitude; measured 0.08% of weeks) but routine in absolute mode with a
    # low mean. A floor CENSORS the intercept: ``B = max(RW_B, baseline_floor)``,
    # so B is >= floor by construction and may sit exactly AT it (a censored
    # walk, not a softplus — zeros are a legitimate baseline).
    #
    # This is a PRIOR-level constraint on one additive term, so it stays inside
    # the function class a consuming MMM can represent. Outcome is deliberately
    # NOT clamped: that would censor the OBSERVATION (a likelihood-level
    # change) and make every additive-Gaussian estimator — including this
    # package's own oracle — misspecified. Non-negative outcome remains enforced
    # exactly by the acceptance filter. Note the oracle's analytic
    # ``latent="marginal"`` mode is unavailable with a floor, because a
    # censored walk is not Gaussian.
    # None (default) keeps the signed walk and byte-identical draws.
    baseline_floor: float | None = None
    # WHAT the floor clips, when one is configured.
    #   "intercept" (default): the intercept walk only. Every other term stays
    #       exactly linear in its node, so ``covariate_contribution[:, m]`` remains
    #       ``g_zy[m]·ρ[m]·Z[:, m]`` — but a large negative ρ·Z can still drag
    #       the non-treatment total, and outcome, below zero.
    #   "non_treatment": the running non-treatment TOTAL, clipped as each parent joins in
    #       the locked order intercept -> confounders -> covariates. A negative
    #       covariate effect is then credited only down to the floor and the excess
    #       is absorbed, so the whole mean function of outcome is >= 0 by
    #       construction (treatment contributions are already >= 0). Per-node columns
    #       become the telescoping differences each node caused — exact, and
    #       identical to the linear split wherever the floor does not bind, but
    #       no longer linear in the node where it does.
    baseline_floor_scope: Literal["intercept", "non_treatment"] = "intercept"
    carryover_burn_in: int = 0

    # Symbolic, per-draw held-level treatment shocks. A shock clamps observed
    # treatment for its window; it never touches the treatment's response state, so
    # the clamped path carryovers with the ordinary normalized causal kernel.
    n_treatment_shocks: int = 0
    treatment_shock_length_range: tuple[int, int] = (2, 2)
    treatment_shock_level_range: tuple[float, float] = (0.0, 0.0)

    # Dense truth-derived attribution labels are metadata, never model inputs.
    # Disable them for feature-only shards without changing generated worlds.
    include_identifiability_labels: bool = True

    # -- Prior-conditioning hyperprior ------------------------------------
    # When enabled, each CELL draws a narrowed prior interval per conditioned
    # quantity (stage-1 concrete numpy, like the rest of the structure):
    #   w  ~ U(w_lo, w_hi);  lo ~ U(S_lo, S_hi - w);  I_q = [lo, lo + w]
    # and stage 2 draws that cell's parameter as pm.Uniform(lo, lo + w)
    # instead of the global support. The draws are recorded in the corpus
    # under the ``prior_cond`` key (see PRIOR_COND_LAYOUT) so consumers can
    # expose them as conditioning features. False retains same-environment
    # unconditioned numerical arrays; interval draws consume no RNG when
    # disabled, but archive bytes can change across schemas.
    prior_conditioning: bool = False
    # Per-quantity (w_lo, w_hi) overrides; keys must be in
    # PRIOR_COND_QUANTITIES. None => PRIOR_COND_DEFAULT_WIDTH_RANGES.
    prior_cond_width_ranges: dict[str, tuple[float, float]] | None = None

    # Prior-shift eval support: optional overrides for the LEGACY edge base
    # rates ("cy", "dc", "dy", "zy") which otherwise come from the slots.py
    # module constants. None (default) => byte-identical legacy behaviour.
    # The dz/zc/cc/zz rates have their own config fields above.
    edge_rate_overrides: dict[str, float] | None = None

    # Per-edge-type arrow budget ("pot"). Maps edge type ->
    # "up to n" cap (int) or (lo, hi) inclusive range; the per-task count is
    # drawn uniformly (int n => {0..n}; use (n, n) for exactly n) and capped at
    # the number of eligible pairs. When a type is present, its arrows are
    # scattered uniformly over eligible node pairs instead of drawn per-pair
    # Bernoulli; each type's pot is independent. Types absent from the dict keep
    # their Bernoulli base rate. None (default) => byte-identical legacy path.
    # Example: {"zc": 5} places up to 5 covariate->treatment arrows however they
    # land (one covariate fanning out, or spread across covariates — capped at 5),
    # while "zy" (covariates' effect on the outcome) is untouched.
    # Note: "cy" keeps its >=1 floor from the degenerate C->Y guard, so a cy
    # budget that draws 0 still yields exactly one C->Y edge.
    edge_budget: dict[str, int | tuple[int, int]] | None = None

    # Reserve active treatments with no direct C->Y edge and zero direct
    # contribution. Unlike a fixed cy budget, the cap follows each cell's
    # active count. Such treatments can still affect Y through C->C paths.
    # Zero leaves the graph-sampling RNG schedule unchanged.
    min_no_direct_effect_treatments: int = 0

    # -- Composable per-input trajectories ----------------------------------
    # Every treatment and covariate carries, independently per input, any subset
    # of TRAJECTORY_COMPONENTS on top of its walk (see pymc_generator.trajectories
    # and docs/reference/config.md). Each component has an inclusion probability:
    # per cell, each input carries it with that probability (a concrete structural
    # draw like the mechanism families). hf/pulse default to 1.0 — the legacy
    # all-or-none texture, still switched off by their (0, 0) ranges — and every
    # schedule component to 0.0, so defaults build byte-identical worlds. The
    # component priors below are read only when the component can be included.
    treatment_hf_inclusion_prob: float = 1.0
    treatment_pulse_inclusion_prob: float = 1.0
    treatment_onset_inclusion_prob: float = 0.0
    treatment_offset_inclusion_prob: float = 0.0
    treatment_flighting_inclusion_prob: float = 0.0
    treatment_level_jump_inclusion_prob: float = 0.0
    treatment_seasonal_inclusion_prob: float = 0.0
    treatment_trend_inclusion_prob: float = 0.0
    covariate_hf_inclusion_prob: float = 1.0
    covariate_pulse_inclusion_prob: float = 1.0
    covariate_onset_inclusion_prob: float = 0.0
    covariate_offset_inclusion_prob: float = 0.0
    covariate_flighting_inclusion_prob: float = 0.0
    covariate_level_jump_inclusion_prob: float = 0.0
    covariate_seasonal_inclusion_prob: float = 0.0
    covariate_trend_inclusion_prob: float = 0.0
    # Gate forms. onset: off before week floor(u*T), u ~ U(range), burn-in included.
    # offset: off from week floor(u*T) on. flighting: on for the first
    # W = clip(floor(duty*P + 0.5), 1, P - 1) of every P weeks (P ~ DiscreteUniform,
    # duty ~ U, random phase).
    treatment_onset_frac_range: tuple[float, float] = (0.05, 0.25)
    covariate_onset_frac_range: tuple[float, float] = (0.05, 0.25)
    treatment_offset_frac_range: tuple[float, float] = (0.6, 0.9)
    covariate_offset_frac_range: tuple[float, float] = (0.6, 0.9)
    treatment_flighting_period_weeks_range: tuple[int, int] = (4, 13)
    covariate_flighting_period_weeks_range: tuple[int, int] = (4, 13)
    treatment_flighting_duty_range: tuple[float, float] = (0.3, 0.8)
    covariate_flighting_duty_range: tuple[float, float] = (0.3, 0.8)
    # Level forms. Jumps: `count` held steps per input, one in each of `count`
    # disjoint slots of weeks [1, T). A treatment step multiplies its series by a
    # log-uniform factor; a covariate step adds a signed size (covariate units).
    treatment_level_jump_count: int = 1
    covariate_level_jump_count: int = 1
    treatment_level_jump_factor_range: tuple[float, float] = (0.5, 2.0)
    covariate_level_jump_size_range: tuple[float, float] = (-1.0, 1.0)
    # Seasonal A*sin(2*pi*t/P + phi), phase ~ U(0, 2*pi). Treatment amplitude is
    # in log-level (the series is multiplied by exp(A*sin)); covariate amplitude
    # is in covariate units.
    treatment_seasonal_amplitude_range: tuple[float, float] = (0.1, 0.5)
    covariate_seasonal_amplitude_range: tuple[float, float] = (0.2, 1.0)
    treatment_seasonal_period_weeks_range: tuple[float, float] = (52.0, 52.0)
    covariate_seasonal_period_weeks_range: tuple[float, float] = (52.0, 52.0)
    # Trend B*max(t, 0)/(T-1): zero through burn-in and reported week 0, B at the
    # last week. Treatment B is a log-level change, covariate B a level change.
    treatment_trend_log_change_range: tuple[float, float] = (-1.0, 1.0)
    covariate_trend_change_range: tuple[float, float] = (-1.0, 1.0)

    def trajectory_inclusion_probs(self) -> dict[str, dict[str, float]]:
        """Effective per-input inclusion probability of every trajectory component.

        ``{input: {component: p}}`` in the canonical ``TRAJECTORY_INPUTS`` /
        ``TRAJECTORY_COMPONENTS`` order. ``hf`` and ``pulse`` are 0.0 whenever
        their texture range disables the term (``*_hf_sigma_range[1] == 0`` /
        ``*_pulse_prob_range[1] == 0``), whatever their inclusion knob says.
        """
        ranges = {"hf": "hf_sigma_range", "pulse": "pulse_prob_range"}
        out: dict[str, dict[str, float]] = {}
        for input_type in TRAJECTORY_INPUTS:
            probs: dict[str, float] = {}
            for component in TRAJECTORY_COMPONENTS:
                p = float(getattr(self, f"{input_type}_{component}_inclusion_prob"))
                if component in ranges and not (
                    float(getattr(self, f"{input_type}_{ranges[component]}")[1]) > 0.0
                ):
                    p = 0.0
                probs[component] = p
            out[input_type] = probs
        return out

    @property
    def trajectory_components_enabled(self) -> bool:
        """Whether any schedule component (gate or level form) can be included."""
        return any(
            float(getattr(self, f"{input_type}_{component}_inclusion_prob")) > 0.0
            for input_type in TRAJECTORY_INPUTS
            for component in SCHEDULE_COMPONENTS
        )

    @property
    def trajectory_metadata_enabled(self) -> bool:
        """Whether corpora carry the optional trajectory arrays and diagnostics.

        True when any trajectory inclusion knob departs from its default: a
        schedule component can be included, or an hf/pulse probability is not 1.
        """
        return self.trajectory_components_enabled or any(
            float(getattr(self, f"{input_type}_{component}_inclusion_prob")) != 1.0
            for input_type in TRAJECTORY_INPUTS
            for component in TEXTURE_COMPONENTS
        )

    @property
    def layout(self) -> SlotLayout:
        return SlotLayout(
            n_treatments=self.n_treatments,
            n_covariates=self.n_covariates,
            n_latent=self.n_latent,
            edge_types=EDGE_TYPES_EXTENDED,
        )

    @property
    def n_treatments_active_range_effective(self) -> tuple[int, int]:
        """n_treatments_active_range clamped to n_treatments."""
        lo = min(self.n_treatments_active_range[0], self.n_treatments)
        hi = min(self.n_treatments_active_range[1], self.n_treatments)
        return (lo, hi)

    @property
    def n_covariates_active_range_effective(self) -> tuple[int, int]:
        """n_covariates_active_range clamped to n_covariates."""
        lo = min(self.n_covariates_active_range[0], self.n_covariates)
        hi = min(self.n_covariates_active_range[1], self.n_covariates)
        return (lo, hi)

    @property
    def n_latent_active_range_effective(self) -> tuple[int, int]:
        """n_latent_active_range clamped to n_latent."""
        lo = min(self.n_latent_active_range[0], self.n_latent)
        hi = min(self.n_latent_active_range[1], self.n_latent)
        return (lo, hi)

    @property
    def active_count_grid(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Active treatment and covariate counts spanned by the effective ranges."""
        tr = self.n_treatments_active_range_effective
        cv = self.n_covariates_active_range_effective
        return tuple(range(tr[0], tr[1] + 1)), tuple(range(cv[0], cv[1] + 1))

    def active_count_weight_matrix(self) -> np.ndarray:
        """Validated ``active_count_weights`` over :attr:`active_count_grid` (ones if None).

        The active ranges are checked first, as :meth:`validate` does, so callers
        that skip :meth:`validate` (the template path) get the same errors.
        """
        self._validate_active_ranges()
        return active_count_weight_matrix(self.active_count_weights, *self.active_count_grid)

    def _validate_active_ranges(self) -> None:
        """Validate the variable-size DAG ranges (called by :meth:`validate`).

        The upper bound may exceed the padded size and is intentionally clamped,
        but both declared bounds must still be ordered integers.
        """
        for range_name, size_name in (
            ("n_treatments_active_range", "n_treatments"),
            ("n_covariates_active_range", "n_covariates"),
            ("n_latent_active_range", "n_latent"),
        ):
            value = getattr(self, range_name)
            try:
                lo, hi = value
            except (TypeError, ValueError):
                raise ValueError(f"{range_name} must be an integer (lo, hi) pair, got {value!r}")
            if (
                isinstance(lo, (bool, np.bool_))
                or isinstance(hi, (bool, np.bool_))
                or not isinstance(lo, (int, np.integer))
                or not isinstance(hi, (int, np.integer))
                or not 1 <= lo <= hi
            ):
                raise ValueError(
                    f"{range_name} must have integer bounds satisfying 1 <= lo <= hi, got {value!r}"
                )
            size = getattr(self, size_name)
            if lo > size:
                raise ValueError(f"{size_name} ({size}) must be >= {range_name}[0] ({lo})")

    @property
    def rw_baseline_std_sigma_effective(self) -> float:
        """Baseline walk scale, following the shared walk scale unless overridden."""
        if self.rw_baseline_std_sigma is None:
            return self.rw_std_sigma
        return self.rw_baseline_std_sigma

    @property
    def n_query(self) -> int:
        return _n_query(self.n_time_steps, self.query_frac)

    @property
    def mechanism_priors_enabled(self) -> bool:
        """Whether effective mechanism priors depart from the legacy path."""
        defaults = _default_saturation_prior_ranges()
        changed_shapes = any(
            tuple(bounds) != defaults[family][name]
            for family, parameters in self.saturation_prior_ranges.items()
            for name, bounds in parameters.items()
        )
        return (
            changed_shapes
            or self.mm_scale_prior != "uniform"
            or self.treatment_reference_contribution_range is not None
            or self.covariate_reference_contribution_range is not None
        )

    def prior_cond_spec(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Effective ``{quantity: {"support": (lo, hi), "width_range": (w_lo, w_hi)}}``.

        Quantities iterate in the LOCKED ``PRIOR_COND_QUANTITIES`` order (the
        order both the interval RNG draws and the ``prior_cond`` columns
        follow). Supports come from the same config ranges the unconditioned
        priors use — ``carryover_alpha_range`` for the geometric carryover decay,
        ``saturation_prior_ranges["hill"]["slope"]`` for the Hill shape — so
        the conditioned interval is nested in the exact global prior by
        construction. Width ranges default to
        :data:`PRIOR_COND_DEFAULT_WIDTH_RANGES`, overridable per quantity via
        ``prior_cond_width_ranges``.
        """
        supports: dict[str, tuple[float, float]] = {
            "carryover_alpha": (
                float(self.carryover_alpha_range[0]),
                float(self.carryover_alpha_range[1]),
            ),
            "hill_shape": (
                float(self.saturation_prior_ranges["hill"]["slope"][0]),
                float(self.saturation_prior_ranges["hill"]["slope"][1]),
            ),
        }
        widths = {**PRIOR_COND_DEFAULT_WIDTH_RANGES, **(self.prior_cond_width_ranges or {})}
        return {
            q: {
                "support": supports[q],
                "width_range": (float(widths[q][0]), float(widths[q][1])),
            }
            for q in PRIOR_COND_QUANTITIES
        }

    def _validate_reference_coefficients(self) -> None:
        """Bound derived coefficients using parameter support, not realized inputs.

        Control bounds replay the runtime ``q * fl(1 / scale)`` exactly; treatment
        bounds divide by the smallest admitted reference response with a rounding
        margin. Supports containing zero are checked at their smallest nonzero draw.
        """
        if self.covariate_reference_contribution_range is not None:
            lo, hi = (float(x) for x in self.covariate_reference_contribution_range)
            # Generation multiplies by this reciprocal, exactly as validated here;
            # Python float arithmetic overflows to inf without a NumPy warning.
            inverse_scale = 1.0 / float(self.covariate_reference_scale)
            smallest = _smallest_nonzero_uniform_magnitude(lo, hi)
            largest_coefficient = max(abs(lo), abs(hi)) * inverse_scale
            if not np.isfinite(inverse_scale) or largest_coefficient > CORPUS_STORAGE_MAX:
                raise ValueError(
                    "covariate_reference_contribution_range / covariate_reference_scale "
                    "requires a coefficient exceeding the float32 corpus storage maximum"
                )
            if smallest is not None and smallest * inverse_scale == 0.0:
                raise ValueError(
                    "covariate_reference_contribution_range / covariate_reference_scale "
                    "requires a nonzero coefficient that underflows in float64"
                )
        if self.treatment_reference_contribution_range is None:
            return
        lo, hi = (float(x) for x in self.treatment_reference_contribution_range)
        smallest = _smallest_nonzero_uniform_magnitude(lo, hi)
        multiplier = float(self.treatment_reference_multiplier)
        if multiplier * 1e-8 < np.finfo(np.float64).tiny:
            raise ValueError(
                "treatment_reference_multiplier * the minimum saturation_scale=1e-8 "
                "must produce a normal positive treatment_reference_input in float64"
            )
        ranges = self.saturation_prior_ranges
        hill_kappa_hi = float(ranges["hill"]["kappa_mult"][1])
        hill_slope = float(ranges["hill"]["slope"][1 if hill_kappa_hi >= multiplier else 0])
        hill_ratio = multiplier / hill_kappa_hi
        hill_log_ratio = (
            np.log(hill_ratio)
            if np.finfo(float).tiny <= hill_ratio <= np.finfo(float).max
            else np.log(multiplier) - np.log(hill_kappa_hi)
        )
        # The runtime ratio carries a few ulps of rounding; steep slopes and deep
        # tails amplify it in the response.
        eps = float(np.finfo(np.float64).eps)
        with np.errstate(over="ignore"):
            hill_condition = 4.0 * hill_slope + 2.0 * abs(hill_slope * hill_log_ratio) + 8.0
            margins = {"hill": float(np.exp(hill_condition * eps))}
        root_alpha = float(ranges["root"]["alpha"][1 if multiplier < 1.0 else 0])
        # Saturation can intentionally be almost flat. Underflow means its
        # requested target cannot be represented by a finite supported beta.
        with np.errstate(over="ignore", under="ignore"):
            lower_response = {
                "linear": multiplier,
                "hill": float(np.exp(-np.logaddexp(0.0, -hill_slope * hill_log_ratio))),
                "logistic": float(
                    np.tanh((float(ranges["logistic"]["lam"][0]) * multiplier) / 2.0)
                ),
                "michaelis_menten": multiplier
                / (multiplier + float(ranges["michaelis_menten"]["kappa_mult"][1])),
                "tanh": float(np.tanh(multiplier / float(ranges["tanh"]["c"][1]))),
                "root": multiplier**root_alpha,
            }
        root_alpha_max = float(ranges["root"]["alpha"][1 if multiplier >= 1.0 else 0])
        upper_response = {
            "linear": multiplier,
            "root": multiplier**root_alpha_max,
        }
        for family, response in lower_response.items():
            if self.saturation_family_probs[family] <= 0.0:
                continue
            # linear has no shape parameters, so name its family probability.
            source = (
                "saturation_family_probs['linear']"
                if family == "linear"
                else f"saturation_prior_ranges[{family!r}]"
            )
            margin = margins.get(family, _ROUNDING_MARGIN)
            largest_coefficient = np.inf
            # Subnormal responses carry too few significant bits for any margin.
            if np.isfinite(response) and response >= np.finfo(np.float64).tiny:
                with np.errstate(over="ignore"):
                    largest_coefficient = float(np.float64(hi) / response * margin)
            if largest_coefficient > CORPUS_STORAGE_MAX:
                raise ValueError(
                    "treatment_reference_contribution_range with "
                    f"treatment_reference_multiplier={multiplier!r} and "
                    f"{source} requires an unrepresentable reference response or a "
                    "coefficient exceeding the float32 corpus storage maximum"
                )
            largest_response = upper_response.get(family, 1.0) * margin
            if smallest is not None and np.float64(smallest) / largest_response == 0.0:
                raise ValueError(
                    "treatment_reference_contribution_range with "
                    f"treatment_reference_multiplier={multiplier!r} and "
                    f"{source} requires a nonzero coefficient that underflows in float64"
                )

    def validate(self) -> None:
        def _integer(name: str, value, *, minimum: int) -> None:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")

        def _finite_real(name: str, value, *, positive: bool = False, nonnegative: bool = False):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                numeric = np.nan
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(numeric)
                or (positive and numeric <= 0)
                or (nonnegative and numeric < 0)
            ):
                domain = "positive" if positive else "nonnegative" if nonnegative else "finite"
                raise ValueError(f"{name} must be a {domain} real scalar, got {value!r}")

        for name, minimum in (
            ("n_treatments", 1),
            ("n_covariates", 1),
            ("n_latent", 1),
            ("n_cells", 2),
            ("draws_per_cell", 1),
            ("n_time_steps", 4),
            ("seed", 0),
            ("min_no_direct_effect_treatments", 0),
        ):
            _integer(name, getattr(self, name), minimum=minimum)
        _finite_real("query_frac", self.query_frac, positive=True)
        _finite_real("val_cell_frac", self.val_cell_frac, positive=True)
        if self.val_cell_frac >= 1.0:
            raise ValueError(f"val_cell_frac must be < 1, got {self.val_cell_frac}")
        _finite_real("treatment_cv_floor", self.treatment_cv_floor, nonnegative=True)
        _finite_real("rw_smoothness_alpha", self.rw_smoothness_alpha, positive=True)
        _finite_real("rw_smoothness_beta", self.rw_smoothness_beta, positive=True)
        _integer("rw_smoothness_max_weeks", self.rw_smoothness_max_weeks, minimum=1)
        if not 0 < self.n_query < self.n_time_steps:
            raise ValueError(
                f"query_frac={self.query_frac} gives {self.n_query} query weeks "
                f"for n_time_steps={self.n_time_steps}; need 0 < n_query < n_time_steps"
            )
        if self.n_query > self.n_time_steps - 2:
            raise ValueError(
                f"query_frac={self.query_frac} gives {self.n_query} query weeks "
                f"for n_time_steps={self.n_time_steps}, leaving only "
                f"{self.n_time_steps - self.n_query} support weeks. "
                f"Need at least 2 support weeks for meaningful statistics."
            )
        if isinstance(self.l_max, bool) or not isinstance(self.l_max, (int, np.integer)):
            raise ValueError("l_max must be an int (not bool)")
        if self.l_max < 1:
            raise ValueError(f"l_max must be >= 1, got {self.l_max}")
        _finite_real("p_long_horizon", self.p_long_horizon, nonnegative=True)
        if self.p_long_horizon > 1.0:
            raise ValueError(f"p_long_horizon must be in [0, 1], got {self.p_long_horizon}")

        # Validate mechanism diversity probabilities.
        def _family_probabilities(name: str, probabilities, family_keys: tuple[str, ...]) -> None:
            if not isinstance(probabilities, dict):
                raise ValueError(
                    f"{name} must be a dict with exactly keys {family_keys}, got {probabilities!r}"
                )
            if set(probabilities) != set(family_keys):
                raise ValueError(
                    f"{name} must have exactly keys {family_keys}, got {tuple(probabilities)!r}"
                )
            if any(
                isinstance(probability, (bool, np.bool_))
                or not isinstance(probability, (int, float, np.integer, np.floating))
                or not np.isfinite(probability)
                or not 0.0 <= probability <= 1.0
                for probability in (probabilities[key] for key in family_keys)
            ):
                raise ValueError(f"{name} entries must be finite probabilities")
            total = sum(probabilities[key] for key in family_keys)
            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(f"{name} must sum to 1.0, got {total}")

        _family_probabilities(
            "carryover_family_probs", self.carryover_family_probs, CARRYOVER_FAMILY_KEYS
        )
        _family_probabilities(
            "saturation_family_probs", self.saturation_family_probs, SATURATION_FAMILY_KEYS
        )

        def _finite_range(
            name: str,
            value: Any = ...,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            minimum_exclusive: bool = False,
            reason: str | None = None,
        ) -> tuple[float, float]:
            if value is ...:
                value = getattr(self, name)
            try:
                lo, hi = value
                if isinstance(lo, (bool, np.bool_)) or isinstance(hi, (bool, np.bool_)):
                    raise TypeError
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name} must be a finite (lo, hi) pair, got {value!r}")
            valid_min = minimum is None or (lo > minimum if minimum_exclusive else lo >= minimum)
            if not (
                np.isfinite(lo)
                and np.isfinite(hi)
                and lo <= hi
                and valid_min
                and (maximum is None or hi <= maximum)
            ):
                message = f"{name} has invalid bounds {value!r}"
                if reason is not None:
                    message += f"; {reason}"
                raise ValueError(message)
            _reject_unrepresentable_bounds(name, value, lo, hi)
            return lo, hi

        defaults = _default_saturation_prior_ranges()
        if not isinstance(self.saturation_prior_ranges, dict) or set(
            self.saturation_prior_ranges
        ) != set(defaults):
            raise ValueError(
                f"saturation_prior_ranges must have exactly families {tuple(defaults)}"
            )
        for family, parameters in defaults.items():
            configured = self.saturation_prior_ranges[family]
            if not isinstance(configured, dict) or set(configured) != set(parameters):
                raise ValueError(
                    f"saturation_prior_ranges[{family!r}] must have exactly parameters "
                    f"{tuple(parameters)}"
                )
            for parameter, bounds in configured.items():
                name = f"saturation_prior_ranges[{family!r}][{parameter!r}]"
                try:
                    lo, hi = bounds
                except (TypeError, ValueError):
                    raise ValueError(f"{name} must be a finite (lo, hi) pair, got {bounds!r}")
                _finite_real(name, lo, positive=True)
                _finite_real(name, hi, positive=True)
                _finite_range(
                    name,
                    value=bounds,
                    minimum=0.0,
                    minimum_exclusive=True,
                    maximum=1.0 if family == "root" else None,
                )
        if not isinstance(self.mm_scale_prior, str) or self.mm_scale_prior not in (
            "uniform",
            "log_uniform",
        ):
            raise ValueError("mm_scale_prior must be 'uniform' or 'log_uniform'")
        mm_lo, mm_hi = (
            float(x) for x in self.saturation_prior_ranges["michaelis_menten"]["kappa_mult"]
        )
        if (
            self.mm_scale_prior == "log_uniform"
            and mm_lo < mm_hi
            and (np.log(mm_lo) >= np.log(mm_hi))
        ):
            raise ValueError(
                "saturation_prior_ranges['michaelis_menten']['kappa_mult'] "
                "bounds collapse in log space under mm_scale_prior='log_uniform'"
            )
        if mm_lo * 1e-8 == 0.0:
            raise ValueError(
                "saturation_prior_ranges['michaelis_menten']['kappa_mult'] lower bound "
                "times the minimum saturation_scale=1e-8 must be positive in float64; "
                "otherwise a zero treatment input evaluates 0/0"
            )
        for name in ("treatment_reference_multiplier", "covariate_reference_scale"):
            _finite_real(name, getattr(self, name), positive=True)
            _reject_unrepresentable_bounds(
                name, getattr(self, name), float(getattr(self, name)), float(getattr(self, name))
            )
        for name in (
            "treatment_reference_contribution_range",
            "covariate_reference_contribution_range",
        ):
            bounds = getattr(self, name)
            if bounds is None:
                continue
            try:
                lo, hi = bounds
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be None or a finite (lo, hi) pair, got {bounds!r}")
            _finite_real(name, lo)
            _finite_real(name, hi)
            if any(float(endpoint) == 0.0 and endpoint != 0.0 for endpoint in (lo, hi)):
                raise ValueError(f"{name} nonzero bounds must remain nonzero in float64")
            _finite_range(
                name,
                minimum=0.0 if name == "treatment_reference_contribution_range" else None,
            )
            if name == "treatment_reference_contribution_range" and float(hi) <= 0.0:
                raise ValueError(f"{name} must have an upper bound > 0")
        self._validate_reference_coefficients()

        _finite_range("carryover_alpha_range", minimum=0.0, maximum=1.0)
        _finite_range("weibull_lam_range", minimum=0.0, minimum_exclusive=True)
        _finite_range("weibull_k_range", minimum=0.0, minimum_exclusive=True)
        for name in (
            "dc_coeff_range",
            "dz_coeff_range",
            "zc_coeff_range",
            "zz_coeff_range",
            "dy_coeff_range",
            "zy_coeff_range",
            "rw_covariate_mean_range",
            "rw_baseline_mean_range",
        ):
            _finite_range(name)
        if not isinstance(self.outcome_std_mode, str) or self.outcome_std_mode not in (
            "relative",
            "absolute",
        ):
            raise ValueError(
                f"outcome_std_mode must be 'relative' or 'absolute', got {self.outcome_std_mode!r}"
            )
        for name in ("rw_baseline_std_range", "rw_outcome_std_range"):
            _finite_range(name, minimum=0.0)
        _finite_range(
            "cc_coeff_range",
            minimum=0.0,
            reason="C->C coefficients amplify, not cannibalize",
        )
        _, beta_additive_hi = _finite_range(
            "beta_additive_range",
            minimum=0.0,
            reason="treatment effect amplitudes must be nonnegative",
        )
        if beta_additive_hi <= 0.0:
            raise ValueError(
                "beta_additive_range must have an upper bound > 0 because a zero-only "
                f"amplitude contradicts every drawn C->Y edge, got {self.beta_additive_range!r}"
            )
        _finite_range(
            "rw_positive_mean_range",
            minimum=0.0,
            minimum_exclusive=True,
            reason="treatment walks stay positive after softplus",
        )
        _, treatment_std_hi = _finite_range(
            "rw_treatment_std_range",
            minimum=0.0,
            reason="treatment walk amplitudes must be nonnegative",
        )
        if treatment_std_hi <= 0.0:
            raise ValueError(
                "rw_treatment_std_range must have an upper bound > 0 because a zero-only "
                "treatment-walk amplitude produces flat treatment paths"
            )
        _finite_range("treatment_hf_sigma_range", minimum=0.0)
        _, treatment_pulse_prob_hi = _finite_range(
            "treatment_pulse_prob_range", minimum=0.0, maximum=0.5
        )
        _finite_range("treatment_pulse_amp_range", minimum=0.0)
        _finite_range("covariate_hf_sigma_range", minimum=0.0)
        _finite_range("covariate_pulse_prob_range", minimum=0.0, maximum=0.5)
        _finite_range("covariate_pulse_amp_range", minimum=0.0)
        if self.baseline_floor is not None:
            floor = self.baseline_floor
            if (
                isinstance(floor, (bool, np.bool_))
                or not isinstance(floor, (int, float, np.integer, np.floating))
                or not np.isfinite(floor)
            ):
                raise ValueError(f"baseline_floor must be None or a finite number, got {floor!r}")
        if self.baseline_floor_scope not in ("intercept", "non_treatment"):
            raise ValueError(
                "baseline_floor_scope must be 'intercept' or 'non_treatment', got "
                f"{self.baseline_floor_scope!r}"
            )

        # Prior-conditioning hyperprior (ACE)
        if self.prior_cond_width_ranges is not None:
            unknown = sorted(set(self.prior_cond_width_ranges) - set(PRIOR_COND_QUANTITIES))
            if unknown:
                raise ValueError(
                    f"prior_cond_width_ranges keys must be in the conditioned set "
                    f"{PRIOR_COND_QUANTITIES}, got {unknown}"
                )
        if self.prior_conditioning or self.prior_cond_width_ranges is not None:
            # Disabled conditioning only checks the widths it was given.
            checked = (
                PRIOR_COND_QUANTITIES
                if self.prior_conditioning
                else tuple(self.prior_cond_width_ranges or ())
            )
            for q, cond_spec in self.prior_cond_spec().items():
                if q not in checked:
                    continue
                s_lo, s_hi = cond_spec["support"]
                w_lo, w_hi = cond_spec["width_range"]
                if not 0.0 < w_lo <= w_hi <= s_hi - s_lo:
                    raise ValueError(
                        f"prior conditioning for {q!r} needs "
                        f"0 < w_lo <= w_hi <= support width; got width range "
                        f"({w_lo}, {w_hi}) against support ({s_lo}, {s_hi})"
                    )
                resolution = _prior_cond_storage_resolution((s_lo, s_hi))
                # Only enabled conditioning writes float32 prior_cond labels.
                if self.prior_conditioning and w_lo <= resolution:
                    raise ValueError(
                        f"prior conditioning for {q!r} needs prior_cond_width_ranges with "
                        f"a lower bound above {resolution:.3g}, the narrowest width float32 "
                        f"prior_cond labels resolve for support ({s_lo}, {s_hi})"
                    )
        # Prior-shift eval: legacy edge-rate overrides
        if self.edge_rate_overrides is not None:
            unknown = sorted(set(self.edge_rate_overrides) - {"cy", "dc", "dy", "zy"})
            if unknown:
                raise ValueError(
                    f"edge_rate_overrides only accepts legacy edge types "
                    f"('cy', 'dc', 'dy', 'zy'), got {unknown}. The dz/zc/cc/zz "
                    f"rates have dedicated config fields."
                )
            for et, rate in self.edge_rate_overrides.items():
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"edge_rate_overrides[{et!r}] must be in [0, 1], got {rate}")
        # Per-edge-type arrow budgets (pots)
        if self.edge_budget is not None:
            unknown = sorted(set(self.edge_budget) - set(EDGE_TYPES_EXTENDED))
            if unknown:
                raise ValueError(
                    f"edge_budget keys must be edge types {EDGE_TYPES_EXTENDED}, got {unknown}"
                )
            for et, spec in self.edge_budget.items():
                if isinstance(spec, bool):
                    raise ValueError(f"edge_budget[{et!r}] must be an int or (lo, hi), got bool")
                if isinstance(spec, (tuple, list)):
                    if len(spec) != 2:
                        raise ValueError(
                            f"edge_budget[{et!r}] range must be (lo, hi), got {spec!r}"
                        )
                    lo, hi = spec
                    if (
                        isinstance(lo, bool)
                        or isinstance(hi, bool)
                        or not (
                            isinstance(lo, (int, np.integer))
                            and isinstance(hi, (int, np.integer))
                            and 0 <= lo <= hi
                        )
                    ):
                        raise ValueError(
                            f"edge_budget[{et!r}] range must be ints with 0 <= lo <= hi, "
                            f"got {spec!r}"
                        )
                elif isinstance(spec, (int, np.integer)):
                    if spec < 0:
                        raise ValueError(f"edge_budget[{et!r}] must be >= 0, got {spec}")
                else:
                    raise ValueError(
                        f"edge_budget[{et!r}] must be an int or (lo, hi) tuple, got {type(spec)}"
                    )
        # Phase 4: additive-SCM priors
        for name in ("dz_base_rate", "zc_base_rate", "cc_base_rate", "zz_base_rate"):
            rate = getattr(self, name)
            if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {rate}")
        for name in ("rw_std_sigma", "rw_outcome_std_sigma"):
            sigma = getattr(self, name)
            if (
                isinstance(sigma, (bool, np.bool_))
                or not isinstance(sigma, (int, float, np.integer, np.floating))
                or not np.isfinite(sigma)
                or sigma <= 0
            ):
                raise ValueError(f"{name} must be finite and > 0, got {sigma}")
        if self.rw_baseline_std_sigma is not None:
            sigma = self.rw_baseline_std_sigma
            if (
                isinstance(sigma, (bool, np.bool_))
                or not isinstance(sigma, (int, float, np.integer, np.floating))
                or not np.isfinite(sigma)
                or sigma <= 0
            ):
                raise ValueError(f"rw_baseline_std_sigma must be finite and > 0, got {sigma}")
        if self.confounding_strength_range is not None:
            try:
                strength_lo, strength_hi = self.confounding_strength_range
                strength_lo, strength_hi = float(strength_lo), float(strength_hi)
            except (TypeError, ValueError):
                raise ValueError(
                    "confounding_strength_range must be a (lo, hi) pair or None, got "
                    f"{self.confounding_strength_range!r}"
                )
            if not (
                np.isfinite(strength_lo)
                and np.isfinite(strength_hi)
                and 0.0 <= strength_lo <= strength_hi <= 0.95
            ):
                raise ValueError(
                    "confounding_strength_range must satisfy finite 0 <= lo <= hi <= 0.95, "
                    f"got {self.confounding_strength_range}"
                )
        # Treatment texture
        if isinstance(self.carryover_burn_in, bool) or not isinstance(
            self.carryover_burn_in, (int, np.integer)
        ):
            raise ValueError("carryover_burn_in must be an int (not bool)")
        if self.carryover_burn_in < 0:
            raise ValueError(f"carryover_burn_in must be >= 0, got {self.carryover_burn_in}")
        if 0 < self.carryover_burn_in < self.l_max:
            # A partial burn-in still convolves the first reported weeks into
            # zero padding — the warmup artifact the knob exists to remove.
            # Also catches dataclasses.replace(cfg, l_max=...) desyncing a
            # preset-built config (presets pin burn_in = l_max).
            raise ValueError(
                f"carryover_burn_in={self.carryover_burn_in} must be 0 (off) or >= "
                f"l_max ({self.l_max}) — a partial burn-in leaves carryover warmup "
                f"in the reported window"
            )
        if self.carryover_burn_in > 0:
            # The boundary is the response's REACH, not the kernel length: an
            # identity-only family mix (nonlinearity="linear") convolves nothing,
            # so no reported week depends on pre-window treatment and a short horizon
            # is perfectly scorable even though the presets still pin
            # carryover_burn_in = l_max. Take the upper bound over every family the
            # config admits (probability > 0) — the families are drawn per world,
            # so validation cannot know which one a given world gets.
            admitted_families = [
                index
                for index, key in enumerate(CARRYOVER_FAMILY_KEYS)
                if self.carryover_family_probs[key] > 0.0
            ]
            warmup_boundary = admitted_response_support_weeks(
                admitted_families, self.l_max, carryover_alpha_range=self.carryover_alpha_range
            )
            short_query_start = self.n_time_steps - self.n_query
            long_query_start = self.n_time_steps // 2
            # Check both split types even at degenerate probabilities: validation-split repair can
            # force either type, and every scored target must have persisted response history.
            if warmup_boundary > 0 and min(short_query_start, long_query_start) < warmup_boundary:
                suggested_n_time_steps = _minimum_valid_query_horizon(
                    self.n_time_steps, self.query_frac, warmup_boundary
                )
                horizon_remedy = (
                    f"raise n_time_steps to at least {suggested_n_time_steps}, "
                    if suggested_n_time_steps is not None
                    else "raise n_time_steps, "
                )
                raise ValueError(
                    "carryover burn-in query overlap: "
                    f"n_time_steps={self.n_time_steps}, l_max={self.l_max}, "
                    f"n_query={self.n_query}; "
                    f"short-horizon query start n_time_steps - n_query={short_query_start}, "
                    f"long-horizon query start n_time_steps // 2={long_query_start}. With "
                    f"burn-in, the first admitted_response_support_weeks = "
                    f"{warmup_boundary} reported weeks carry a "
                    "treatment response that depends on unpersisted pre-window treatment. Both the "
                    "short-horizon (n_time_steps - n_query) and long-horizon "
                    "(n_time_steps // 2) query windows must "
                    "start at or after that boundary, otherwise tasks are scored on targets that "
                    "are not a function of the persisted inputs. To reach this world anyway, "
                    "either set carryover_burn_in=0 (the convolution then zero-pads, which is "
                    "reproducible from persisted treatment at every week), restrict "
                    f"carryover_family_probs to the identity family, {horizon_remedy}or lower "
                    "query_frac / l_max."
                )
        if treatment_pulse_prob_hi > 0.0 and float(self.treatment_pulse_amp_range[1]) <= 0.0:
            raise ValueError(
                "treatment_pulse_prob_range enables pulses but treatment_pulse_amp_range "
                "has zero amplitude — disable pulses via the prob range instead"
            )
        covariate_p_hi = float(self.covariate_pulse_prob_range[1])
        if covariate_p_hi > 0.0 and float(self.covariate_pulse_amp_range[1]) <= 0.0:
            raise ValueError(
                "covariate_pulse_prob_range enables pulses but covariate_pulse_amp_range "
                "has zero amplitude — disable pulses via the prob range instead"
            )
        # Symbolic treatment-shock schedule. Keep this validation explicit rather
        # than relying on PyMC's distribution errors, so invalid schedules fail
        # before a model is built.
        if isinstance(self.n_treatment_shocks, bool) or not isinstance(
            self.n_treatment_shocks, (int, np.integer)
        ):
            raise ValueError("n_treatment_shocks must be an int (not bool)")
        if self.n_treatment_shocks < 0:
            raise ValueError(f"n_treatment_shocks must be >= 0, got {self.n_treatment_shocks}")
        if not isinstance(self.include_identifiability_labels, bool):
            raise ValueError("include_identifiability_labels must be a bool")
        try:
            shock_len_lo, shock_len_hi = self.treatment_shock_length_range
        except (TypeError, ValueError):
            raise ValueError("treatment_shock_length_range must be an (lo, hi) integer pair")
        if (
            isinstance(shock_len_lo, bool)
            or isinstance(shock_len_hi, bool)
            or not isinstance(shock_len_lo, (int, np.integer))
            or not isinstance(shock_len_hi, (int, np.integer))
            or not 1 <= shock_len_lo <= shock_len_hi <= self.n_time_steps
        ):
            raise ValueError(
                "treatment_shock_length_range must have integer bounds satisfying "
                f"1 <= lo <= hi <= n_time_steps, got {self.treatment_shock_length_range!r}"
            )
        try:
            shock_level_lo, shock_level_hi = self.treatment_shock_level_range
            shock_level_lo, shock_level_hi = float(shock_level_lo), float(shock_level_hi)
        except (TypeError, ValueError):
            raise ValueError("treatment_shock_level_range must be a finite (lo, hi) pair")
        if not (
            np.isfinite(shock_level_lo)
            and np.isfinite(shock_level_hi)
            and 0.0 <= shock_level_lo <= shock_level_hi
        ):
            raise ValueError(
                "treatment_shock_level_range must satisfy finite 0 <= lo <= hi, "
                f"got {self.treatment_shock_level_range!r}"
            )
        _reject_unrepresentable_bounds(
            "treatment_shock_level_range",
            self.treatment_shock_level_range,
            shock_level_lo,
            shock_level_hi,
        )
        if self.n_treatment_shocks > self.n_time_steps:
            raise ValueError(
                f"n_treatment_shocks must be <= n_time_steps ({self.n_time_steps}), "
                f"got {self.n_treatment_shocks}"
            )
        if self.n_treatment_shocks and shock_len_lo < 2:
            raise ValueError(
                "enabled treatment shocks must last at least 2 weeks so the held level "
                "is observable in treatment"
            )
        if self.n_treatment_shocks and self.n_treatment_shocks * shock_len_hi > self.n_time_steps:
            raise ValueError(
                "n_treatment_shocks * max treatment_shock_length must be <= n_time_steps, got "
                f"{self.n_treatment_shocks} * {shock_len_hi} > {self.n_time_steps}"
            )
        self._validate_active_ranges()
        # The smallest cell must accommodate both the direct-null floor and
        # the mandatory direct treatment.
        if self.min_no_direct_effect_treatments:
            active_lo = self.n_treatments_active_range[0]
            if active_lo <= self.min_no_direct_effect_treatments:
                raise ValueError(
                    f"min_no_direct_effect_treatments ({self.min_no_direct_effect_treatments}) must be < "
                    f"n_treatments_active_range[0] ({active_lo}) so every cell can keep at "
                    "least one direct treatment besides the direct-null ones"
                )
        if not isinstance(self.active_count_allocation, str) or (
            self.active_count_allocation not in ACTIVE_COUNT_ALLOCATIONS
        ):
            raise ValueError(
                "active_count_allocation must be 'independent' or 'stratified', "
                f"got {self.active_count_allocation!r}"
            )
        self.active_count_weight_matrix()
        self._validate_trajectories(_finite_range)

    def _validate_trajectories(self, finite_range: Any) -> None:
        """Validate the composable-trajectory knobs (called by :meth:`validate`).

        Types, finiteness and static bounds are always checked. Rules that depend
        on the horizon, or that would otherwise let an included component change
        nothing, apply only while the component's inclusion probability is > 0, so
        a disabled component's priors never constrain a config.
        """
        for input_type in TRAJECTORY_INPUTS:
            for component in TRAJECTORY_COMPONENTS:
                name = f"{input_type}_{component}_inclusion_prob"
                value = getattr(self, name)
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or not 0.0 <= value <= 1.0
                ):
                    raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")

        def _int_pair(name: str, minimum: int) -> tuple[int, int]:
            value = getattr(self, name)
            try:
                lo, hi = value
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be an integer (lo, hi) pair, got {value!r}")
            if (
                isinstance(lo, (bool, np.bool_))
                or isinstance(hi, (bool, np.bool_))
                or not isinstance(lo, (int, np.integer))
                or not isinstance(hi, (int, np.integer))
                or not minimum <= lo <= hi
            ):
                raise ValueError(
                    f"{name} must have integer bounds satisfying {minimum} <= lo <= hi, "
                    f"got {value!r}"
                )
            return int(lo), int(hi)

        T = int(self.n_time_steps)
        support_weeks = min(T - self.n_query, T // 2)
        for x in TRAJECTORY_INPUTS:
            prob = {
                c: float(getattr(self, f"{x}_{c}_inclusion_prob")) for c in TRAJECTORY_COMPONENTS
            }
            onset_lo, onset_hi = finite_range(f"{x}_onset_frac_range", minimum=0.0, maximum=1.0)
            if onset_hi >= 1.0:
                raise ValueError(
                    f"{x}_onset_frac_range has invalid bounds; a launch must fall inside "
                    f"the reported window (hi < 1), got {getattr(self, f'{x}_onset_frac_range')!r}"
                )
            offset_lo, offset_hi = finite_range(
                f"{x}_offset_frac_range", minimum=0.0, minimum_exclusive=True, maximum=1.0
            )
            period_range = _int_pair(f"{x}_flighting_period_weeks_range", 2)
            duty_lo, _ = finite_range(
                f"{x}_flighting_duty_range", minimum=0.0, minimum_exclusive=True, maximum=1.0
            )
            count_name = f"{x}_level_jump_count"
            count = getattr(self, count_name)
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or count < 1
            ):
                raise ValueError(f"{count_name} must be an integer >= 1, got {count!r}")
            if x == "treatment":
                size_name = "treatment_level_jump_factor_range"
                size_lo, size_hi = finite_range(size_name, minimum=0.0, minimum_exclusive=True)
                neutral_size = size_lo == size_hi == 1.0
                max_log_size = max(abs(float(np.log(size_lo))), abs(float(np.log(size_hi))))
                trend_name = "treatment_trend_log_change_range"
            else:
                size_name = "covariate_level_jump_size_range"
                size_lo, size_hi = finite_range(size_name)
                neutral_size = size_lo == size_hi == 0.0
                max_log_size = 0.0
                trend_name = "covariate_trend_change_range"
            _, amplitude_hi = finite_range(f"{x}_seasonal_amplitude_range", minimum=0.0)
            finite_range(
                f"{x}_seasonal_period_weeks_range",
                minimum=2.0,
                minimum_exclusive=True,
                reason="a period of 2 weeks or less aliases to an alternation at weekly sampling",
            )
            trend_lo, trend_hi = finite_range(trend_name)

            # An included component must be able to change something.
            if prob["onset"] > 0.0 and int(np.floor(onset_lo * T)) < 1:
                raise ValueError(
                    f"{x}_onset_frac_range lower bound {onset_lo} launches at week "
                    f"floor({onset_lo}*{T}) = 0 for some draws; an included onset must delay "
                    "the input by at least one reported week (raise the lower bound)"
                )
            if prob["offset"] > 0.0 and offset_hi >= 1.0:
                raise ValueError(
                    f"{x}_offset_frac_range upper bound must be < 1 so an included offset "
                    f"stops the input before the window ends, got {offset_hi}"
                )
            if prob["flighting"] > 0.0 and period_range[1] > T:
                raise ValueError(
                    f"{x}_flighting_period_weeks_range upper bound ({period_range[1]}) must "
                    f"be <= n_time_steps ({T}): any {T} reported weeks then hold a full "
                    "period, so every included flighting input switches off in the window"
                )
            if prob["level_jump"] > 0.0:
                if count > T - 1:
                    raise ValueError(
                        f"{count_name} ({count}) must be <= n_time_steps - 1 ({T - 1}): "
                        "each level jump needs its own week in [1, n_time_steps)"
                    )
                if neutral_size:
                    raise ValueError(
                        f"{x}_level_jump_inclusion_prob > 0 but {size_name} makes every "
                        "jump a no-op — disable jumps via the inclusion probability instead"
                    )
            if prob["seasonal"] > 0.0 and amplitude_hi <= 0.0:
                raise ValueError(
                    f"{x}_seasonal_inclusion_prob > 0 but {x}_seasonal_amplitude_range has "
                    "zero amplitude — disable seasonality via the inclusion probability instead"
                )
            if prob["trend"] > 0.0 and trend_lo == trend_hi == 0.0:
                raise ValueError(
                    f"{x}_trend_inclusion_prob > 0 but {trend_name} is identically zero — "
                    "disable trends via the inclusion probability instead"
                )

            # Every gated input keeps >= 2 on-weeks inside the support prefix that
            # both split types (and the validation-split repair) leave as context,
            # whichever gate forms it ends up carrying.
            if any(prob[c] > 0.0 for c in GATE_COMPONENTS):
                start_hi = int(np.floor(onset_hi * T)) if prob["onset"] > 0.0 else 0
                stop_lo = int(np.floor(offset_lo * T)) if prob["offset"] > 0.0 else T
                window = min(stop_lo, support_weeks) - start_hi
                worst = min_on_weeks(
                    window, period_range if prob["flighting"] > 0.0 else None, duty_lo
                )
                if worst < 2:
                    raise ValueError(
                        f"{x} gates can leave only {max(worst, 0)} on-week(s) inside the "
                        f"{support_weeks}-week support prefix (min(n_time_steps - n_query, "
                        f"n_time_steps // 2)); need >= 2. The onset upper bound, offset lower "
                        "bound, flighting periods and duty lower bound jointly decide this — "
                        "launch earlier, stop later, shorten the flighting period, raise the "
                        "duty or lengthen n_time_steps"
                    )

            if x == "treatment":
                swing = 0.0
                if prob["seasonal"] > 0.0:
                    swing += amplitude_hi
                if prob["trend"] > 0.0:
                    swing += max(abs(trend_lo), abs(trend_hi))
                if prob["level_jump"] > 0.0:
                    swing += int(count) * max_log_size
                # The bound is inclusive; the tolerance absorbs the rounding of a
                # sum (e.g. 15 * 0.1) that is exactly the bound on paper.
                if swing > TRAJECTORY_MAX_LOG_SHIFT * (1.0 + 1e-12):
                    raise ValueError(
                        f"treatment level components can swing the log-level by {swing!r} "
                        f"(seasonal amplitude + |trend| + jump count * max|log factor|); the "
                        f"bound is {TRAJECTORY_MAX_LOG_SHIFT} (a x{np.exp(TRAJECTORY_MAX_LOG_SHIFT):.0f} "
                        "level multiplier)"
                    )
                # Every range endpoint is float32-representable (finite_range);
                # the scheduled level is a product, so check its reach as well.
                reach = float(self.rw_positive_mean_range[1]) * float(np.exp(swing))
            else:
                reach = float(np.max(np.abs(self.rw_covariate_mean_range)))
                if prob["seasonal"] > 0.0:
                    reach += amplitude_hi
                if prob["trend"] > 0.0:
                    reach += max(abs(trend_lo), abs(trend_hi))
                if prob["level_jump"] > 0.0:
                    reach += int(count) * max(abs(size_lo), abs(size_hi))
            if reach > CORPUS_STORAGE_MAX:
                raise ValueError(
                    f"{x} level components can reach {reach!r}, beyond the float32 corpus "
                    f"storage limit {CORPUS_STORAGE_MAX:.6g}; shrink the level ranges"
                )


def _resolve_budget(rng: np.random.Generator, spec: int | tuple[int, int], n_eligible: int) -> int:
    """Resolve an edge-budget spec to a concrete arrow count, clamped to eligible pairs.

    An ``int`` n is an "up to n" cap: the per-task count is drawn uniformly in
    ``{0, ..., n}`` (never more than n). A ``(lo, hi)`` tuple draws uniformly in
    the inclusive range ``{lo, ..., hi}`` — use ``(n, n)`` for exactly n. Both
    forms are capped at ``n_eligible`` (the number of eligible source->dest
    pairs for the edge type).
    """
    if isinstance(spec, (tuple, list)):
        lo = max(0, min(int(spec[0]), n_eligible))
        hi = max(lo, min(int(spec[1]), n_eligible))
    else:
        lo = 0
        hi = max(0, min(int(spec), n_eligible))
    return int(rng.integers(lo, hi + 1))


def _scatter(rng: np.random.Generator, count: int, n_eligible: int) -> np.ndarray:
    """0/1 vector of length ``n_eligible`` with ``count`` ones at uniform-random slots.

    The "pot" mechanic: ``count`` arrows scattered over the eligible pairs.
    How they clump (one source fanning out vs. one arrow each) is emergent from
    the uniform draw.
    """
    out = np.zeros(n_eligible, dtype="float64")
    if count > 0 and n_eligible > 0:
        out[rng.choice(n_eligible, size=count, replace=False)] = 1.0
    return out


def _scatter_triu(rng: np.random.Generator, n_act: int, count: int, out_full: np.ndarray) -> None:
    """Place ``count`` arrows in the strict upper triangle of ``out_full[:n_act, :n_act]``.

    Strict upper triangle (src index < dst index) guarantees acyclicity for the
    C->C and Z->Z edge types.
    """
    if n_act < 2 or count <= 0:
        return
    iu = np.triu_indices(n_act, k=1)
    count = min(count, iu[0].size)
    sel = rng.choice(iu[0].size, size=count, replace=False)
    out_full[iu[0][sel], iu[1][sel]] = 1.0


def _sample_g(
    rng: np.random.Generator,
    layout: SlotLayout,
    n_treatments_active: int | None = None,
    n_covariates_active: int | None = None,
    n_latent_active: int | None = None,
    rates: dict[str, float] | None = None,
    budget: dict[str, int | tuple[int, int]] | None = None,
    min_no_direct: int = 0,
) -> dict[str, Any]:
    """Draw one DAG cell from the slot base rates (0/1 numpy arrays).

    For variable-size DAGs, pass n_treatments_active, n_covariates_active,
    n_latent_active to restrict edges to active nodes only. Inactive nodes get
    zero-padded g-vectors and active masks.

    Parameters
    ----------
    rng : numpy random generator
    layout : SlotLayout with max sizes
    n_treatments_active, n_covariates_active, n_latent_active : optional int
        Number of active nodes. If None, uses
        ``layout.n_treatments`` / ``layout.n_covariates`` / ``layout.n_latent``
        (all active).
    rates : optional dict
        Per-edge-type base-rate overrides for the legacy types
        ("cy", "dc", "dy", "zy"); missing keys fall back to
        ``EDGE_BASE_RATES``. None (default) is byte-identical to the
        module constants (prior-shift eval support).
    budget : optional dict
        Per-edge-type arrow budget ("pot") for the legacy types: an "up to n"
        cap. When a type is present, its (uniformly drawn) count of arrows is
        scattered uniformly over eligible node pairs instead of drawn per-pair
        Bernoulli (see ``_resolve_budget`` / ``_scatter``). Types absent from
        the dict keep their Bernoulli rate; with ``budget=None`` (or ``{}``)
        the RNG stream is byte-identical to the legacy path.
    min_no_direct : int
        Minimum active treatments without a direct C->Y edge (see
        ``SCMPrior.min_no_direct_effect_treatments``). Caps the direct count at
        ``max(1, n_treatments_active - min_no_direct)``; 0 is inert and
        consumes no extra RNG.

    Returns
    -------
    dict with g-vectors, active masks, and active counts.
    """
    _rates = {**EDGE_BASE_RATES, **(rates or {})}
    _budget = budget or {}
    n_treatments_max = layout.n_treatments
    n_covariates_max = layout.n_covariates
    n_latent_max = layout.n_latent

    # Default: all nodes active (backward compat)
    if n_treatments_active is None:
        n_treatments_active = n_treatments_max
    if n_covariates_active is None:
        n_covariates_active = n_covariates_max
    if n_latent_active is None:
        n_latent_active = n_latent_max

    # Clamp to valid range (ensure non-negative)
    n_treatments_active = max(0, min(n_treatments_active, n_treatments_max))
    n_covariates_active = max(0, min(n_covariates_active, n_covariates_max))
    n_latent_active = max(0, min(n_latent_active, n_latent_max))

    # Generate edges only for active nodes; pad rest with zeros
    g_cy = np.zeros(n_treatments_max)
    if n_treatments_active > 0:
        # Reserve `min_no_direct` active treatments without a direct C->Y edge.
        # Direct treatments are still
        # scattered over ALL active slots — reserving the tail slots instead
        # would make slot index predict the label.
        n_live_max = max(1, n_treatments_active - max(0, min_no_direct))
        if _budget.get("cy") is not None:
            n = _resolve_budget(rng, _budget["cy"], n_live_max)
            g_cy[:n_treatments_active] = _scatter(rng, n, n_treatments_active)
        else:
            g_cy[:n_treatments_active] = rng.binomial(1, _rates["cy"], size=n_treatments_active)
            live = np.flatnonzero(g_cy[:n_treatments_active])
            if live.size > n_live_max:  # unreachable when min_no_direct == 0
                g_cy[rng.choice(live, size=live.size - n_live_max, replace=False)] = 0.0
        # Degenerate guard: ensure at least one C->Y edge in active range
        if not g_cy[:n_treatments_active].any():
            g_cy[rng.integers(0, n_treatments_active)] = 1.0

    g_dc = np.zeros((n_latent_max, n_treatments_max))
    if n_latent_active > 0 and n_treatments_active > 0:
        if _budget.get("dc") is not None:
            n = _resolve_budget(rng, _budget["dc"], n_latent_active * n_treatments_active)
            g_dc[:n_latent_active, :n_treatments_active] = _scatter(
                rng, n, n_latent_active * n_treatments_active
            ).reshape(n_latent_active, n_treatments_active)
        else:
            g_dc[:n_latent_active, :n_treatments_active] = rng.binomial(
                1, _rates["dc"], size=(n_latent_active, n_treatments_active)
            )

    g_dy = np.zeros(n_latent_max)
    if n_latent_active > 0:
        if _budget.get("dy") is not None:
            n = _resolve_budget(rng, _budget["dy"], n_latent_active)
            g_dy[:n_latent_active] = _scatter(rng, n, n_latent_active)
        else:
            g_dy[:n_latent_active] = rng.binomial(1, _rates["dy"], size=n_latent_active)

    g_zy = np.zeros(n_covariates_max)
    if n_covariates_active > 0:
        if _budget.get("zy") is not None:
            n = _resolve_budget(rng, _budget["zy"], n_covariates_active)
            g_zy[:n_covariates_active] = _scatter(rng, n, n_covariates_active)
        else:
            g_zy[:n_covariates_active] = rng.binomial(1, _rates["zy"], size=n_covariates_active)

    # Active-node masks (1 = node exists, 0 = padding)
    active_treatment = np.zeros(n_treatments_max)
    active_treatment[:n_treatments_active] = 1.0
    active_covariate = np.zeros(n_covariates_max)
    active_covariate[:n_covariates_active] = 1.0
    active_latent = np.zeros(n_latent_max)
    active_latent[:n_latent_active] = 1.0

    return {
        "g_cy": g_cy,
        "g_dc": g_dc,
        "g_dy": g_dy,
        "g_zy": g_zy,
        "active_treatment": active_treatment,
        "active_covariate": active_covariate,
        "active_latent": active_latent,
        "n_treatments_active": n_treatments_active,
        "n_covariates_active": n_covariates_active,
        "n_latent_active": n_latent_active,
    }


def sample_g_additive(
    rng: np.random.Generator,
    cfg: SCMPrior,
    layout: SlotLayout,
    n_treatments_active: int | None = None,
    n_covariates_active: int | None = None,
    n_latent_active: int | None = None,
) -> dict[str, Any]:
    """Draw one DAG cell for the additive SCM.

    Extends :func:`_sample_g` with the four new edge types. C->C and Z->Z
    edges are restricted to the strict upper triangle (src index < dst
    index) which guarantees acyclicity. Base rates for dz/zc/cc/zz come
    from ``cfg``; the remaining types use the slots.py base rates.
    When ``cfg.edge_budget`` names a type, that type's arrows are
    placed by budget (uniform scatter over eligible pairs) instead of by
    Bernoulli rate — see :func:`_resolve_budget` and :func:`_scatter`.

    ``treatment_active`` marks a direct C->Y edge OR an outgoing C->C edge.
    This local structural flag does not guarantee a path to Y through a
    downstream treatment. All treatments remain observed; the degenerate guard
    only forces at least one C->Y edge.

    ``cfg.min_no_direct_effect_treatments`` caps the direct-treatment count, so a
    cell can be made to always carry both classes of the direct-effect signal.

    Returns
    -------
    dict with the 8 g-blocks at max (padded) sizes — ``g_cc``/``g_zz`` as
    FULL square matrices with zero diagonals — plus active masks, active
    counts, and ``treatment_active``.
    """
    base = _sample_g(
        rng,
        layout,
        n_treatments_active=n_treatments_active,
        n_covariates_active=n_covariates_active,
        n_latent_active=n_latent_active,
        rates=cfg.edge_rate_overrides,
        budget=cfg.edge_budget,
        min_no_direct=cfg.min_no_direct_effect_treatments,
    )
    n_treatments_max, n_covariates_max, n_latent_max = (
        layout.n_treatments,
        layout.n_covariates,
        layout.n_latent,
    )
    n_treatments_active = base["n_treatments_active"]
    n_covariates_active = base["n_covariates_active"]
    n_latent_active = base["n_latent_active"]
    _budget = cfg.edge_budget or {}

    g_dz = np.zeros((n_latent_max, n_covariates_max))
    if n_latent_active > 0 and n_covariates_active > 0:
        if _budget.get("dz") is not None:
            n = _resolve_budget(rng, _budget["dz"], n_latent_active * n_covariates_active)
            g_dz[:n_latent_active, :n_covariates_active] = _scatter(
                rng, n, n_latent_active * n_covariates_active
            ).reshape(n_latent_active, n_covariates_active)
        else:
            g_dz[:n_latent_active, :n_covariates_active] = rng.binomial(
                1, cfg.dz_base_rate, size=(n_latent_active, n_covariates_active)
            )

    g_zc = np.zeros((n_covariates_max, n_treatments_max))
    if n_covariates_active > 0 and n_treatments_active > 0:
        if _budget.get("zc") is not None:
            n = _resolve_budget(rng, _budget["zc"], n_covariates_active * n_treatments_active)
            g_zc[:n_covariates_active, :n_treatments_active] = _scatter(
                rng, n, n_covariates_active * n_treatments_active
            ).reshape(n_covariates_active, n_treatments_active)
        else:
            g_zc[:n_covariates_active, :n_treatments_active] = rng.binomial(
                1, cfg.zc_base_rate, size=(n_covariates_active, n_treatments_active)
            )

    g_cc = np.zeros((n_treatments_max, n_treatments_max))
    if n_treatments_active > 1:
        if _budget.get("cc") is not None:
            n = _resolve_budget(
                rng, _budget["cc"], n_treatments_active * (n_treatments_active - 1) // 2
            )
            _scatter_triu(rng, n_treatments_active, n, g_cc)  # strict upper: src < dst
        else:
            draws = rng.binomial(
                1, cfg.cc_base_rate, size=(n_treatments_active, n_treatments_active)
            )
            g_cc[:n_treatments_active, :n_treatments_active] = np.triu(
                draws, k=1
            )  # strict upper: src < dst

    g_zz = np.zeros((n_covariates_max, n_covariates_max))
    if n_covariates_active > 1:
        if _budget.get("zz") is not None:
            n = _resolve_budget(
                rng, _budget["zz"], n_covariates_active * (n_covariates_active - 1) // 2
            )
            _scatter_triu(rng, n_covariates_active, n, g_zz)
        else:
            draws = rng.binomial(
                1, cfg.zz_base_rate, size=(n_covariates_active, n_covariates_active)
            )
            g_zz[:n_covariates_active, :n_covariates_active] = np.triu(draws, k=1)

    # Local structural activity: direct C->Y OR outgoing C->C.
    treatment_active = ((base["g_cy"] == 1) | (g_cc.sum(axis=1) > 0)).astype("float64")
    treatment_active *= base["active_treatment"]  # padding nodes are never active

    return {
        **base,
        "g_dz": g_dz,
        "g_zc": g_zc,
        "g_cc": g_cc,
        "g_zz": g_zz,
        "treatment_active": treatment_active,
    }


def _signal_block(
    cfg: SCMPrior,
    layout: SlotLayout,
    outcome_raw: np.ndarray,
    g_tasks: np.ndarray,
    treatment_active_mask: np.ndarray,
    outcome_scale: np.ndarray,
    carryover_family: np.ndarray,
    carryover_alpha: np.ndarray,
    weibull_lam: np.ndarray,
    weibull_k: np.ndarray,
    signal_metrics: np.ndarray,
    signal_metric_valid: np.ndarray,
) -> dict:
    """``diagnostics["signal"]`` for a corpus.

    "Direct active treatment" = cy edge present AND treatment not padding.
    """
    cy_mask = (g_tasks[:, layout.slices["cy"]] == 1) & (treatment_active_mask == 1)
    out = summarize_signal_metrics(
        signal_metrics,
        signal_metric_valid,
        outcome_raw,
        cy_mask,
        outcome_scale=outcome_scale,
        l_max=cfg.l_max,
        carryover_burn_in=cfg.carryover_burn_in,
        carryover_family=carryover_family,
        carryover_alpha=carryover_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
    )
    out["metric_version"] = SIGNAL_METRIC_VERSION
    out["metric_layout"] = list(SIGNAL_METRIC_LAYOUT)
    out["l_max"] = int(cfg.l_max)
    out["carryover_burn_in"] = int(cfg.carryover_burn_in)
    # pymc-marketing min-max rescales the density before sum-normalizing, so one lag
    # has zero weight; a true normalized PDF would not have an exactly zero lag.
    out["carryover_kernel_semantics"] = "normalized-causal-minmax-weibull-density"
    out["carryover_kernel_version"] = 3
    out["outcome_noise_semantics"] = OUTCOME_NOISE_SEMANTICS
    out["outcome_noise_version"] = OUTCOME_NOISE_VERSION
    out["outcome_std_mode"] = cfg.outcome_std_mode
    return out


def _finalize_corpus(corpus: dict[str, Any], cfg: SCMPrior) -> dict[str, Any]:
    """Derive retained-corpus diagnostics and signal features exactly once.

    Generation deliberately leaves task-leading arrays unsummarized so callers
    can truncate them first.  In particular, signal features must be based on
    the exact float32 arrays persisted by the corpus rather than a superseded
    pre-truncation population.
    """
    layout = cfg.layout
    n_tasks = corpus["treatment_raw"].shape[0]
    diagnostics = corpus["diagnostics"]
    elapsed = diagnostics["timing"]["elapsed_s"]

    g_tasks = corpus["g"]
    treatment_active_mask = corpus["treatment_active_mask"]
    cy_mask = (g_tasks[:, layout.slices["cy"]] == 1) & (treatment_active_mask == 1)
    signal_metrics, signal_metric_valid = dense_signal_metrics(
        corpus["treatment_raw"],
        corpus["treatment_contribution_raw"],
        corpus["outcome_raw"],
        corpus["baseline_raw"],
        cy_mask,
        outcome_scale=corpus["outcome_scale"],
        carryover_family=corpus["carryover_family"],
        carryover_alpha=corpus["carryover_alpha"],
        weibull_lam=corpus["weibull_lam"],
        weibull_k=corpus["weibull_k"],
        l_max=cfg.l_max,
        carryover_burn_in=cfg.carryover_burn_in,
    )
    if cfg.include_identifiability_labels:
        corpus["identifiability"] = {
            "signal_metrics": signal_metrics,
            "signal_metric_valid": signal_metric_valid,
        }
    diagnostics["signal"] = _signal_block(
        cfg,
        layout,
        corpus["outcome_raw"],
        g_tasks,
        treatment_active_mask,
        corpus["outcome_scale"],
        corpus["carryover_family"],
        corpus["carryover_alpha"],
        corpus["weibull_lam"],
        corpus["weibull_k"],
        signal_metrics,
        signal_metric_valid,
    )

    # These are intentionally calculated from the retained, persisted arrays.
    # Do not regenerate per-task normalizers here: slicing their already-cast
    # values preserves the serialized-array compatibility contract.
    treatment = corpus["treatment_raw"].astype(np.float64)
    contributions = corpus["treatment_contribution_raw"].astype(np.float64)
    baseline = corpus["baseline_raw"].astype(np.float64)
    qs = (0.1, 0.5, 0.9)
    treatment_mean = treatment.mean(axis=1)
    cv_all = np.divide(
        treatment.std(axis=1),
        treatment_mean,
        out=np.zeros_like(treatment_mean),
        where=treatment_mean != 0.0,
    ).ravel()
    contrib_tot = contributions.sum(axis=(1, 2))
    treatment_denominator = contrib_tot + baseline.sum(axis=1)
    treatment_share = np.divide(
        contrib_tot,
        treatment_denominator,
        out=np.zeros_like(contrib_tot),
        where=treatment_denominator != 0.0,
    )

    edge_marginals = {}
    for edge_type in layout.edge_types:
        block = g_tasks[:, layout.slices[edge_type]]
        edge_marginals[edge_type] = float(block.mean()) if block.size else 0.0

    diagnostics.update(
        {
            "n_tasks": int(n_tasks),
            "n_cells": int(np.unique(corpus["cell_id"]).size),
            "edge_marginals": edge_marginals,
            "treatment_share_quantiles": {
                f"q{int(q * 100)}": float(np.quantile(treatment_share, q)) for q in qs
            },
            "treatment_cv_quantiles": {
                f"q{int(q * 100)}": float(np.quantile(cv_all, q)) for q in qs
            },
        }
    )
    if cfg.trajectory_metadata_enabled:
        # Realised prevalence is computed from the stored (post-truncation)
        # flags over active input slots — the same helper validate_corpus uses.
        diagnostics["trajectory"] = {
            "components": list(TRAJECTORY_COMPONENTS),
            "inclusion_probs": cfg.trajectory_inclusion_probs(),
            "parameter_fields": list(TRAJECTORY_PARAM_FIELDS),
            "jump_counts": {
                input_type: int(getattr(cfg, f"{input_type}_level_jump_count"))
                for input_type in TRAJECTORY_INPUTS
            },
            **summarize_component_prevalence(
                corpus["treatment_components"],
                corpus["covariate_components"],
                corpus["treatment_active_mask"],
                corpus["covariate_active_mask"],
            ),
        }
    if cfg.active_count_allocation == "stratified":
        # Realised coverage of the grid, counted from the stored
        # (post-truncation) masks — the same helper validate_corpus uses.
        treatment_counts, covariate_counts = cfg.active_count_grid
        diagnostics["active_count_coverage"] = {
            "allocation": "stratified",
            "n_treatments_active": list(treatment_counts),
            "n_covariates_active": list(covariate_counts),
            "weights": cfg.active_count_weight_matrix().tolist(),
            **summarize_active_count_coverage(
                corpus["treatment_active_mask"],
                corpus["covariate_active_mask"],
                corpus["cell_id"],
                treatment_counts,
                covariate_counts,
            ),
        }
    # Wall-clock telemetry, kept apart from every other diagnostic because it is
    # the ONLY nondeterministic entry: two same-seed generations agree on every
    # array and every other key, so quarantining the clock here is what lets
    # ``save_corpus`` drop it and persist byte-identical shards.
    diagnostics["timing"] = {
        "elapsed_s": float(elapsed),
        "tasks_per_sec": float(n_tasks / elapsed),
    }
    # These errors must describe the persisted float32 arrays rather than the
    # pre-storage calculations used to produce them.
    diagnostics.update(
        {
            "decomposition_max_abs_error": float(
                np.abs(
                    corpus["baseline_raw"]
                    + corpus["treatment_contribution_raw"].sum(axis=-1)
                    + corpus["indirect_effects"]
                    - corpus["outcome_raw"]
                ).max()
            ),
            "telescoping_split_max_abs_error": float(
                np.abs(
                    corpus["indirect_effects_by_source"].sum(axis=-1) - corpus["indirect_effects"]
                ).max()
            ),
            "full_decomposition_max_abs_error": float(
                np.abs(
                    corpus["baseline_intrinsic"]
                    + corpus["outcome_noise"]
                    + corpus["latent_unobserved_contribution"].sum(axis=-1)
                    + corpus["covariate_contribution"].sum(axis=-1)
                    + corpus["treatment_contribution_raw"].sum(axis=-1)
                    + corpus["indirect_effects_by_source"].sum(axis=-1)
                    - corpus["outcome_raw"]
                ).max()
            ),
            "baseline_decomposition_max_abs_error": float(
                np.abs(
                    corpus["baseline_intrinsic"]
                    + corpus["outcome_noise"]
                    + corpus["latent_unobserved_contribution"].sum(axis=-1)
                    + corpus["covariate_contribution"].sum(axis=-1)
                    - corpus["baseline_raw"]
                ).max()
            ),
        }
    )
    return corpus


def _make_support_mask(
    rng: np.random.Generator, n_time_steps: int, n_query: int, p_long_horizon: float
) -> tuple[np.ndarray, int]:
    """Per-task support mask (u8, 1=support) and the split type.

    All splits are temporal (contiguous suffix) — no random week masking,
    which would leak future information in time series.

    Split types:
        0 = short_horizon: last n_query weeks are query (25% default)
        1 = long_horizon: last n_time_steps//2 weeks are query (50%)
    """
    split_type = int(rng.random() < p_long_horizon)
    support = np.ones(n_time_steps, dtype=np.uint8)
    if split_type == 1:
        # Long horizon: predict the second half
        query_start = n_time_steps // 2
    else:
        # Short horizon: predict the last n_query weeks
        query_start = n_time_steps - n_query
    support[query_start:] = 0
    return support, split_type


def _warn_flat_texture(cfg: SCMPrior) -> None:
    """Steer every caller to the ONE supported world prior.

    The default ``make_scm_prior`` configuration supplies
    additive SCM with high-frequency treatment texture and carryover burn-in.
    A config with the flat (smooth-walk-only) treatment prior still generates
    but warns: its contribution targets degenerate to near-flat lines
    (measured on the reference config: ~49% of direct-treatment targets without
    week-to-week variation; see ``signal_diagnostics``). A treatment is flat when
    it can carry no trajectory component at all: no hf, no pulse (by range or by
    inclusion probability) and no schedule component.
    """
    try:
        treatment_probs = cfg.trajectory_inclusion_probs()["treatment"]
    except (TypeError, ValueError, IndexError):
        return  # an invalid config; validate() reports it with the precise message
    if all(p == 0.0 for p in treatment_probs.values()):
        warnings.warn(
            "The flat (smooth-walk-only) treatment texture is "
            "deprecated: it produces near-flat contribution targets the model cannot "
            "learn attribution from. Build configs with "
            "make_scm_prior.",
            FutureWarning,
            stacklevel=3,
        )


def _recompute_retained_cell_split(corpus: dict) -> None:
    """Repair the train/validation split from the retained cell rows."""
    cell_id = corpus["cell_id"]
    retained_cells = np.unique(cell_id)
    if retained_cells.size < 2:
        raise ValueError(
            "The retained corpus spans fewer than two cells and cannot form a "
            "cell-level train/validation split"
        )

    # Preserve sampled assignments where possible, but only entire cells can
    # move between splits or a graph would leak across train and validation.
    original_val_cells = np.unique(cell_id[corpus["is_val"] == 1])
    val_cells = np.intersect1d(retained_cells, original_val_cells, assume_unique=True)
    if val_cells.size == 0:
        val_cells = retained_cells[:1]
    elif val_cells.size == retained_cells.size:
        val_cells = val_cells[:-1]
    corpus["is_val"] = np.isin(cell_id, val_cells).astype(np.uint8)


def sample_prior_predictive(prior: SCMPrior, n: int | None = None) -> dict[str, Any]:
    """Draw a corpus of n_tasks SCMs: numerical arrays and nested diagnostic metadata.

    Each world routes through :func:`_generate_corpus_additive` — the additive
    causal SCM with the extended g-vector layout and exact interventional
    decomposition targets (``indirect_effects``, ``indirect_effects_by_source``,
    ``covariate_contribution``, ``latent_unobserved_contribution``, ``baseline_intrinsic``)
    plus ``treatment_active``.

    Parameters
    ----------
    prior : SCMPrior
        The prior over SCMs (see :func:`pymc_generator.make_scm_prior`).
    n : int, optional
        Number of worlds to return. If ``None`` (default), returns
        ``prior.n_cells * prior.draws_per_cell`` worlds. For ``n >= 2``, the
        retained corpus always spans at least two cells. When
        ``2 <= n <= prior.draws_per_cell``, generation temporarily uses
        ``max(1, n // 2)`` draws per cell and enough cells for the first ``n``
        rows to span that grid. This observable effective grid is reported by
        ``diagnostics["draws_per_cell"]`` and ``cell_id``. ``n=1`` raises
        because one world cannot carry a cell-level train/validation split.

    Notes
    -----
    Priors with the flat (texture-free) treatment prior emit a ``FutureWarning``
    — build with ``make_scm_prior`` instead.
    """
    _warn_flat_texture(prior)
    prior.validate()
    if n is not None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n == 1:
            raise ValueError("n=1 cannot form a cell-level train/validation split; n must be >= 2")
        draws_per_cell = prior.draws_per_cell
        if n <= draws_per_cell:
            effective_draws_per_cell = max(1, n // 2)
            n_cells = max(2, (n + effective_draws_per_cell - 1) // effective_draws_per_cell)
            prior = replace(
                prior,
                n_cells=n_cells,
                draws_per_cell=effective_draws_per_cell,
            )
        else:
            prior = replace(prior, n_cells=(n + draws_per_cell - 1) // draws_per_cell)
    corpus = _generate_corpus_additive(prior)
    if n is not None and corpus["treatment_raw"].shape[0] > n:
        actual = corpus["treatment_raw"].shape[0]
        for key, val in list(corpus.items()):
            if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == actual:
                corpus[key] = val[:n]
    _recompute_retained_cell_split(corpus)
    return _finalize_corpus(corpus, prior)


# --------------------------------------------------------------------------
# Additive causal graph corpus
# --------------------------------------------------------------------------

_ADDITIVE_OUT_NAMES = (
    "latent_unobserved",
    "covariates",
    "treatments",
    "baseline",
    "baseline_intrinsic",
    "covariate_contribution",
    "latent_unobserved_contribution",
    "contributions",
    "indirect_effects",
    "indirect_effects_by_source",
    "outcome",
    "confounding_strength",
    # Appended last: this tuple is the draw-name order, which fixes PyTensor's
    # RNG traversal, so a new name must not displace an existing one.
    "outcome_noise",
)

# Corpus audit metadata.  These are already deterministics in each cell model;
# requesting just these values preserves the one-model-per-cell sampling path
# while avoiding persistence of natural-path realism outputs or burn-in masks.
_CORPUS_SHOCK_NAMES = (
    "treatment_shock_mask",
    "treatment_shock_index",
    "treatment_shock_start",
    "treatment_shock_length",
    "treatment_shock_level_multiplier",
    "treatment_shock_level",
)
_CORPUS_PARAM_NAMES = (
    "saturation_scale",
    "param_treatment_level",
    "param_carryover_alpha",
    "param_weibull_lam",
    "param_weibull_k",
)
# Trajectory outputs, drawn only when schedule components are enabled. They are
# PREPENDED to the draw names: PyTensor walks the last requested output first,
# so leading names never move the legacy RNG order (and in a cell where nothing
# is wired they are constants or aliases of legacy outputs).
_CORPUS_TRAJECTORY_NAMES = (
    "treatment_activity",
    "covariate_activity",
    "treatment_log_level_shift",
    "covariate_level_shift",
)
# Schedule-free realism references. With shocks they take the trailing slot the
# unshocked pair used (they alias it when nothing is wired); without shocks they
# lead, aliasing ``treatments`` / ``outcome`` when nothing is wired.
_CORPUS_NATURAL_NAMES = ("treatments_natural", "outcome_natural")


def _pack_prior_cond(prior_cond: dict[str, tuple[float, float]]) -> np.ndarray:
    """Pack a cell's ``{quantity: (low, width)}`` draw into a ``(len(PRIOR_COND_LAYOUT),)`` row.

    Columns follow the LOCKED ``PRIOR_COND_LAYOUT`` order — consumers index
    by name via the layout, never by position literals.
    """
    vals: list[float] = []
    for name in PRIOR_COND_LAYOUT:
        q, part = name.rsplit("_", 1)
        lo, width = prior_cond[q]
        vals.append(lo if part == "low" else width)
    return np.asarray(vals, dtype="float64")


def _slice_g_active(
    g: dict[str, np.ndarray],
    n_treatments_active: int,
    n_covariates_active: int,
    n_latent_active: int,
) -> dict[str, np.ndarray]:
    """Restrict padded g-blocks to the active node ranges."""
    return {
        "g_cy": g["g_cy"][:n_treatments_active],
        "g_dc": g["g_dc"][:n_latent_active, :n_treatments_active],
        "g_dz": g["g_dz"][:n_latent_active, :n_covariates_active],
        "g_dy": g["g_dy"][:n_latent_active],
        "g_zy": g["g_zy"][:n_covariates_active],
        "g_zc": g["g_zc"][:n_covariates_active, :n_treatments_active],
        "g_cc": g["g_cc"][:n_treatments_active, :n_treatments_active],
        "g_zz": g["g_zz"][:n_covariates_active, :n_covariates_active],
    }


def _additive_task_ok(
    treatment: np.ndarray,
    outcome: np.ndarray,
    arrays: dict[str, np.ndarray],
    g_cy_active: np.ndarray,
    cv_floor: float,
    outcome_spike_ratio: float = 8.0,
    treatment_spike_ratio: float = 50.0,
    realism_treatment: np.ndarray | None = None,
    realism_outcome: np.ndarray | None = None,
    cv_alternative_treatment: np.ndarray | None = None,
    storage_max: float | None = None,
) -> bool:
    """Single-task realism filter for the additive SCM.

    Realism checks on the additive scale: finite
    arrays and non-negative actual outcome are always required.  CV and spike
    checks can use natural (unshocked) audit paths so a deliberate intervention
    is not rejected for looking unlike organic treatment.

    ``cv_alternative_treatment`` (same shape as the realism treatment) offers a
    second series per treatment for the CV floor only: a direct treatment passes
    when either its realism series or its alternative reaches ``cv_floor``. The
    corpus passes each scheduled treatment's own schedule applied to its natural
    path, so an on/off gate or level component can carry the variation a
    texture-free walk lacks, while shocks never can.

    ``storage_max`` additionally rejects a draw whose arrays exceed that
    magnitude, i.e. that would not survive the float32 corpus cast.
    """
    for a in arrays.values():
        if not np.isfinite(a).all():
            return False
        if storage_max is not None and np.abs(a).max(initial=0.0) > storage_max:
            return False
    if (outcome < 0).any():
        return False
    realism_treatment = treatment if realism_treatment is None else realism_treatment
    realism_outcome = outcome if realism_outcome is None else realism_outcome
    active = np.asarray(g_cy_active) == 1
    if active.any():
        cv = realism_treatment.std(axis=0) / (
            realism_treatment.mean(axis=0) + 1e-12
        )  # (n_treatments,)
        if cv_alternative_treatment is not None:
            alternative = cv_alternative_treatment.std(axis=0) / (
                cv_alternative_treatment.mean(axis=0) + 1e-12
            )
            cv = np.maximum(cv, alternative)
        if cv[active].min() < cv_floor:
            return False
    outcome_med = np.median(realism_outcome)
    safe_med = outcome_med if outcome_med > 0 else 1.0
    if realism_outcome.max() / safe_med >= outcome_spike_ratio:
        return False
    treatment_med = np.median(realism_treatment, axis=0)  # (n_treatments,)
    safe_treatment_med = np.where(treatment_med > 0, treatment_med, 1.0)
    if (realism_treatment.max(axis=0) / safe_treatment_med).max() >= treatment_spike_ratio:
        return False
    return True


def _outcome_norm_overflows(outcome: np.ndarray, support_ends: tuple[int, ...]) -> bool:
    """Whether ``outcome / std(outcome[:end])`` overflows float32 for any support end.

    A task's stored ``outcome_scale`` is the std over its support prefix, and the
    validation-split repair can move a task to the other split type after
    acceptance, so both prefixes must keep ``outcome_norm`` representable. The
    check runs on the float32-rounded outcome, because storage divides the
    stored float32 ``outcome_raw`` by its own std. A zero-std prefix is skipped:
    storage then falls back to the full-series std.
    """
    stored = np.asarray(outcome).astype(np.float32).astype(np.float64)
    peak = float(np.abs(stored).max())
    for end in support_ends:
        scale = float(np.std(stored[:end]))
        if np.isfinite(scale) and scale > 0.0 and peak / scale > CORPUS_STORAGE_MAX:
            return True
    return False


def _generate_corpus_additive(cfg: SCMPrior) -> dict[str, Any]:
    """Generate a padded additive-SCM corpus.

    The public schema is documented by :func:`sample_prior_predictive`.
    Each cell shares a discrete structure; each task draws fresh continuous
    parameters and innovations. Evaluate at active sizes, then copy accepted
    values into the preallocated maximum-size arrays. Base float32 storage
    follows float64 acceptance checks; realised rich truth remains float64.
    """
    from .world_model import build_world_model, draw_worlds, sample_prior_cond, sample_structure

    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    layout = cfg.layout  # extended edge types
    n_query = cfg.n_query
    n_treatments_max, n_covariates_max, n_latent_max = (
        layout.n_treatments,
        layout.n_covariates,
        layout.n_latent,
    )
    n_time_steps = cfg.n_time_steps

    n_tasks = cfg.n_cells * cfg.draws_per_cell
    dimensions = {
        "task": n_tasks,
        "time": n_time_steps,
        "treatment": n_treatments_max,
        "covariate": n_covariates_max,
        "latent": n_latent_max,
        "edge": layout.n_slots,
        "shock": cfg.n_treatment_shocks,
        "indirect_source": 3,
        "component": len(TRAJECTORY_COMPONENTS),
        "treatment_jump": cfg.treatment_level_jump_count,
        "covariate_jump": cfg.covariate_level_jump_count,
    }
    corpus: dict[str, Any] = {
        key: np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
        for key, (axes, dtype) in CORPUS_ARRAY_FIELDS.items()
    }
    if cfg.prior_conditioning:
        corpus["prior_cond"] = np.empty((n_tasks, len(PRIOR_COND_LAYOUT)), dtype=np.float32)
    # Optional composable-trajectory block (like prior_cond: present iff
    # configured). Schedule outputs exist in the cell models only when a schedule
    # component is enabled; otherwise activity is 1 and shifts are 0.
    trajectory_metadata = cfg.trajectory_metadata_enabled
    trajectory_schedules = cfg.trajectory_components_enabled
    if trajectory_metadata:
        for key, (axes, dtype) in TRAJECTORY_ARRAY_FIELDS.items():
            corpus[key] = np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
    mechanism_fields = dict(MECHANISM_ARRAY_FIELDS) if cfg.mechanism_priors_enabled else {}
    if mechanism_fields:
        for input_type in TRAJECTORY_INPUTS:
            if getattr(cfg, f"{input_type}_reference_contribution_range") is not None:
                mechanism_fields.update(
                    {
                        key: spec
                        for key, spec in MECHANISM_REFERENCE_FIELDS.items()
                        if key.startswith(f"{input_type}_")
                    }
                )
        for key, (axes, dtype) in mechanism_fields.items():
            corpus[key] = np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
    # Treatment normalization retains the original float64 reduction order.
    # Other accepted outputs can be cast directly into their final storage.
    treatment_raw = np.zeros((n_tasks, n_time_steps, n_treatments_max), dtype=np.float64)
    draw_fields = {
        "covariates": "covariates",
        "outcome_raw": "outcome",
        "treatment_contribution_raw": "contributions",
        "baseline_raw": "baseline",
        "latent_unobserved": "latent_unobserved",
        "indirect_effects": "indirect_effects",
        "baseline_intrinsic": "baseline_intrinsic",
        "outcome_noise": "outcome_noise",
        "covariate_contribution": "covariate_contribution",
        "latent_unobserved_contribution": "latent_unobserved_contribution",
        "indirect_effects_by_source": "indirect_effects_by_source",
        "confounding_strength": "confounding_strength",
        "treatment_shock_mask": "treatment_shock_mask",
        "treatment_shock_index": "treatment_shock_index",
        "treatment_shock_start": "treatment_shock_start",
        "treatment_shock_length": "treatment_shock_length",
        "treatment_shock_level_multiplier": "treatment_shock_level_multiplier",
        "treatment_shock_level": "treatment_shock_level",
        "treatment_level": "param_treatment_level",
        "saturation_scale": "saturation_scale",
        "carryover_alpha": "param_carryover_alpha",
        "weibull_lam": "param_weibull_lam",
        "weibull_k": "param_weibull_k",
    }
    for key in mechanism_fields:
        if key != "sat_family":
            draw_fields[key] = (
                "saturation_scale" if key == "mechanism_saturation_scale" else f"param_{key}"
            )
    n_evaluated = 0
    n_rejected = 0
    n_draw_failures = 0
    # Stratified allocation fixes every cell's treatment × covariate counts up
    # front on its own stream, seeded by one draw from ``rng``; independent
    # allocation returns None and draws the counts per cell below, exactly as
    # before.
    active_count_plan = plan_active_counts(cfg, rng)

    for cell in range(cfg.n_cells):
        n_treatments_active, n_covariates_active, n_latent_active = draw_active_counts(
            cfg, rng, active_count_plan, cell
        )
        g = sample_g_additive(
            rng,
            cfg,
            layout,
            n_treatments_active=n_treatments_active,
            n_covariates_active=n_covariates_active,
            n_latent_active=n_latent_active,
        )
        rows = slice(cell * cfg.draws_per_cell, (cell + 1) * cfg.draws_per_cell)
        corpus["cell_id"][rows] = cell
        corpus["g"][rows] = layout.pack(
            g_cy=g["g_cy"],
            g_dc=g["g_dc"],
            g_dy=g["g_dy"],
            g_zy=g["g_zy"],
            g_dz=g["g_dz"],
            g_zc=g["g_zc"],
            g_cc=g["g_cc"],
            g_zz=g["g_zz"],
        )
        for key, source in (
            ("treatment_active_mask", "active_treatment"),
            ("covariate_active_mask", "active_covariate"),
            ("latent_active_mask", "active_latent"),
            ("treatment_active", "treatment_active"),
            ("n_treatments_active", "n_treatments_active"),
            ("n_covariates_active", "n_covariates_active"),
            ("n_latent_active", "n_latent_active"),
        ):
            corpus[key][rows] = g[source]
        g_act = _slice_g_active(g, n_treatments_active, n_covariates_active, n_latent_active)

        # One pm.Model per cell (fixed structure: DAG + families + smoothness);
        # the continuous params and noise are the model's RVs, drawn in batches
        # and filtered by the realism gate. Building/compiling the graph
        # dominates at large (n_treatments, n_covariates), so a model per cell
        # (not per draw) is key; FAST_COMPILE (the draw_worlds default) keeps
        # the one-off compile cheap.
        structural = sample_structure(g_act, cfg, rng)
        # Prior-conditioning interval draw (per cell — one model build). False
        # returns None and consumes no RNG, preserving same-environment
        # unconditioned numerical draws, not cross-schema archive-byte identity.
        prior_cond = sample_prior_cond(cfg, rng)
        if prior_cond is not None:
            corpus["prior_cond"][rows] = _pack_prior_cond(prior_cond)
        model, out_names, _param_names = build_world_model(
            g_act, cfg, structural, n_time_steps, prior_cond=prior_cond
        )
        trajectory_reports = (
            {
                key: report
                for key, report in TRAJECTORY_PARAM_REPORTS.items()
                if report in _param_names
            }
            if trajectory_metadata
            else {}
        )
        requested_reports = set(trajectory_reports.values()) | {
            draw_fields[key] for key in mechanism_fields if key != "sat_family"
        }
        # Leading reports join the same accepted draw without displacing the
        # legacy output traversal that assigns RNG streams.
        audit_names = (
            tuple(
                name
                for name in _param_names
                if name in requested_reports and name not in _CORPUS_PARAM_NAMES
            )
            if requested_reports
            else ()
        )
        # Per-input component flags (cell structure, stored per row like the
        # carryover family). Schedule keys exist only when schedules are enabled.
        component_flags: dict[str, np.ndarray] = {}
        scheduled_treatment: np.ndarray = np.zeros(n_treatments_active, dtype=bool)
        if trajectory_metadata:
            for input_type, n_active in (
                ("treatment", n_treatments_active),
                ("covariate", n_covariates_active),
            ):
                component_flags[input_type] = np.stack(
                    [
                        np.asarray(
                            structural.get(
                                structural_key(input_type, component),
                                np.zeros(n_active, dtype=bool),
                            ),
                            dtype=np.uint8,
                        )
                        for component in TRAJECTORY_COMPONENTS
                    ],
                    axis=1,
                )
            schedule_columns = [TRAJECTORY_COMPONENTS.index(c) for c in SCHEDULE_COMPONENTS]
            scheduled_treatment = np.asarray(
                component_flags["treatment"][:, schedule_columns].any(axis=1), dtype=bool
            )
        trajectory_masks = {}
        for key in trajectory_reports:
            input_type, component_leaf = key.removeprefix("trajectory_").split("_", 1)
            component = next(
                name for name in TRAJECTORY_COMPONENTS if component_leaf.startswith(f"{name}_")
            )
            trajectory_masks[key] = (
                component_flags[input_type][:, TRAJECTORY_COMPONENTS.index(component)] != 0
            )
        accepted = 0
        cell_evaluated = 0
        cell_rejected = 0
        cell_draw_failures = 0
        last_draw_error: str | None = None
        for _round in range(MAX_TOPUPS_PER_CELL):
            if accepted == cfg.draws_per_cell:
                break
            n_missing = cfg.draws_per_cell - accepted
            n_req = n_missing + max(2, int(np.ceil(0.5 * n_missing)))
            draw_seed = int(rng.integers(2**31 - 1))
            try:
                # Output order covariates PyTensor's RNG traversal. Metadata
                # deterministics must precede the legacy outputs: appending
                # them changes legacy draws, while a separate same-seed call
                # produces parameters from a different joint world.
                draw_names: tuple[str, ...]
                if trajectory_schedules and cfg.n_treatment_shocks:
                    draw_names = (
                        _CORPUS_TRAJECTORY_NAMES
                        + _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                        + _CORPUS_NATURAL_NAMES
                    )
                elif trajectory_schedules:
                    draw_names = (
                        _CORPUS_TRAJECTORY_NAMES
                        + _CORPUS_NATURAL_NAMES
                        + _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                    )
                elif cfg.n_treatment_shocks:
                    draw_names = (
                        _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                        + ("treatments_unshocked", "outcome_unshocked")
                    )
                else:
                    draw_names = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES
                rng_reference_names = draw_names
                draw_names = audit_names + draw_names
                drawn_b = draw_worlds(
                    model,
                    draw_names,
                    draw_seed,
                    draws=n_req,
                    rng_reference_names=rng_reference_names if audit_names else None,
                )
            except _RETRYABLE_DRAW_ERRORS as exc:
                if not _is_retryable_draw_failure(exc):
                    # Not a numeric failure from inside a PyTensor node
                    # evaluation, or a reference-domain failure that resampling
                    # would hide by truncating the target prior. Retrying only
                    # buries it under MAX_TOPUPS_PER_CELL empty rounds and then
                    # blames the realism filter. Let it out untouched.
                    raise
                # A sporadic numeric draw failure — a hierarchical draw handed a
                # downstream distribution an out-of-domain parameter. Retry with
                # a new seed. This batch produced NO candidates, so it must not
                # enter the evaluated/rejected accounting: those two describe the
                # realism filter, and inflating them makes a broken draw look
                # like a strict gate. Keep the error so a repeatable failure
                # still surfaces in the RuntimeError below.
                last_draw_error = repr(exc)
                n_draw_failures += 1
                cell_draw_failures += 1
                continue

            for b in range(n_req):
                if accepted == cfg.draws_per_cell:
                    break
                drawn = {name: drawn_b[name][b] for name in draw_names}
                n_evaluated += 1
                cell_evaluated += 1
                audit_finite = all(np.isfinite(drawn[name]).all() for name in audit_names)
                if not audit_finite:
                    n_rejected += 1
                    cell_rejected += 1
                    continue
                realism_arrays = (
                    {name: drawn[name] for name in rng_reference_names} if audit_names else drawn
                )

                if trajectory_schedules:
                    # Realism reads the schedule-free natural path; only a
                    # scheduled treatment's own schedule may lift its CV.
                    natural = drawn["treatments_natural"]
                    cv_alternative = None
                    if scheduled_treatment.any():
                        cv_alternative = np.where(
                            scheduled_treatment[None, :],
                            drawn["treatment_activity"]
                            * natural
                            * np.exp(drawn["treatment_log_level_shift"]),
                            natural,
                        )
                    task_ok = _additive_task_ok(
                        treatment=drawn["treatments"],
                        outcome=drawn["outcome"],
                        arrays=realism_arrays,
                        g_cy_active=g_act["g_cy"],
                        cv_floor=cfg.treatment_cv_floor,
                        realism_treatment=natural,
                        realism_outcome=drawn["outcome_natural"],
                        cv_alternative_treatment=cv_alternative,
                        # Scheduled levels multiply texture and parent terms, so
                        # extreme (validated) level priors can still overflow the
                        # float32 storage; such a draw is rejected, not stored.
                        storage_max=CORPUS_STORAGE_MAX,
                    )
                else:
                    task_ok = _additive_task_ok(
                        treatment=drawn["treatments"],
                        outcome=drawn["outcome"],
                        arrays=realism_arrays,
                        g_cy_active=g_act["g_cy"],
                        cv_floor=cfg.treatment_cv_floor,
                        realism_treatment=drawn.get("treatments_unshocked"),
                        realism_outcome=drawn.get("outcome_unshocked"),
                    )
                if not task_ok:
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                support, task_split_type = _make_support_mask(
                    rng, n_time_steps, n_query, cfg.p_long_horizon
                )
                candidate_outcome_scale = float(np.std(drawn["outcome"][support == 1]))
                if not (np.isfinite(candidate_outcome_scale) and candidate_outcome_scale > 0.0):
                    n_rejected += 1
                    cell_rejected += 1
                    continue
                if trajectory_schedules and _outcome_norm_overflows(
                    drawn["outcome"], (n_time_steps - n_query, n_time_steps // 2)
                ):
                    # outcome_norm = outcome / scale would not be representable
                    # under either split (the validation-split repair may flip it).
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                row = cell * cfg.draws_per_cell + accepted
                treatment_raw[row, :, :n_treatments_active] = drawn["treatments"]
                for key, source in draw_fields.items():
                    value = drawn[source]
                    prefix: tuple[int | slice, ...] = (
                        row,
                        *(slice(size) for size in np.shape(value)),
                    )
                    corpus[key][prefix] = value
                corpus["carryover_family"][row, :n_treatments_active] = structural[
                    "carryover_family"
                ]
                if mechanism_fields:
                    corpus["sat_family"][row, :n_treatments_active] = structural["sat_family"]
                if trajectory_metadata:
                    corpus["treatment_components"][row, :n_treatments_active] = component_flags[
                        "treatment"
                    ]
                    corpus["covariate_components"][row, :n_covariates_active] = component_flags[
                        "covariate"
                    ]
                    for key, report in trajectory_reports.items():
                        value = np.where(trajectory_masks[key], drawn[report], 0)
                        prefix = (row, *(slice(size) for size in value.shape))
                        corpus[key][prefix] = value
                    if trajectory_schedules:
                        for key, n_active in (
                            ("treatment_activity", n_treatments_active),
                            ("covariate_activity", n_covariates_active),
                            ("treatment_log_level_shift", n_treatments_active),
                            ("covariate_level_shift", n_covariates_active),
                        ):
                            corpus[key][row, :, :n_active] = drawn[key]
                    else:
                        corpus["treatment_activity"][row, :, :n_treatments_active] = 1
                        corpus["covariate_activity"][row, :, :n_covariates_active] = 1
                corpus["support_mask"][row] = support
                corpus["is_future"][row] = task_split_type
                accepted += 1
            del drawn_b, drawn
        if accepted < cfg.draws_per_cell:
            # Name the actual culprit: a strict realism gate and a draw that
            # keeps blowing up look identical from the accepted count alone, and
            # they call for opposite remedies (loosen the prior vs fix the
            # graph). Report both tallies so the reader can tell which happened.
            raise RuntimeError(
                f"cell {cell} (n_treatments_active={n_treatments_active}, "
                f"n_covariates_active={n_covariates_active}, n_latent_active={n_latent_active}): "
                f"only {accepted}/{cfg.draws_per_cell} tasks "
                f"accepted after {MAX_TOPUPS_PER_CELL} rounds — the realism filter "
                f"rejected {cell_rejected}/{cell_evaluated} evaluated candidates, and "
                f"{cell_draw_failures}/{MAX_TOPUPS_PER_CELL} rounds produced no "
                f"candidates at all because the draw itself failed"
                + (f"; last draw error: {last_draw_error}" if last_draw_error else "")
            )

    support_mask = corpus["support_mask"]
    split_type = corpus["is_future"]
    cell_id = corpus["cell_id"]
    treatment_active_mask = corpus["treatment_active_mask"]

    treatment_means = treatment_raw.mean(axis=1)  # (n_tasks, n_treatments)
    treatment_norm = np.divide(
        treatment_raw,
        treatment_means[:, None, :],
        out=np.zeros_like(treatment_raw),
        where=treatment_means[:, None, :] != 0.0,
    )
    active_treatment_sum = (treatment_raw * treatment_active_mask[:, None, :]).sum(
        axis=-1, keepdims=True
    )
    treatment_share = (
        np.divide(
            treatment_raw,
            active_treatment_sum,
            out=np.zeros_like(treatment_raw),
            where=active_treatment_sum != 0.0,
        )
        * treatment_active_mask[:, None, :]
    )

    corpus["treatment_raw"][:] = treatment_raw
    corpus["treatment_norm"][:] = treatment_norm
    corpus["treatment_share"][:] = treatment_share
    corpus["treatment_means"][:] = treatment_means
    del treatment_raw, treatment_norm, treatment_share, treatment_means

    # -- cell-level validation split (same logic as the legacy path) --------
    n_val_cells = max(1, int(round(cfg.val_cell_frac * cfg.n_cells)))
    if n_val_cells >= cfg.n_cells:
        n_val_cells = cfg.n_cells - 1
    val_cells = rng.permutation(cfg.n_cells)[:n_val_cells]
    corpus["is_val"][:] = np.isin(cell_id, val_cells)
    is_val = corpus["is_val"]

    val_mask = is_val == 1
    val_split_types = split_type[val_mask]
    if val_mask.sum() >= 2:
        if val_split_types.sum() == 0:
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, n_time_steps, n_query, 1.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split
        elif val_split_types.sum() == val_mask.sum():
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, n_time_steps, n_query, 0.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split

    # Derived from the FLOAT32 array that is actually persisted, not from the
    # float64 draw. `outcome_norm = outcome_raw / outcome_scale` is persisted too, so a
    # consumer must be able to reproduce both from the corpus alone; computing
    # the scale at draw precision made that impossible whenever float32 rounding
    # dominated the standard deviation. That is reachable: a short support window
    # over a very smooth walk gives a near-constant slice, where the writer and a
    # float32 recomputation disagreed by 2.5e-5 relative — past the validator's
    # 1e-5 tolerance.
    outcome_stored = corpus["outcome_raw"]
    outcome_scale = np.array(
        [
            float(np.std(outcome_stored[i][support_mask[i] == 1].astype(np.float64)))
            for i in range(n_tasks)
        ],
        dtype=np.float64,
    )
    bad_scale = ~(np.isfinite(outcome_scale) & (outcome_scale > 0.0))
    if bad_scale.any():
        full_std = np.std(outcome_stored[bad_scale].astype(np.float64), axis=1)
        outcome_scale[bad_scale] = np.where(np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0)
    outcome_norm = outcome_stored.astype(np.float64) / outcome_scale[:, None]

    effective_legacy_edge_rates = {**EDGE_BASE_RATES, **(cfg.edge_rate_overrides or {})}

    # -- static diagnostics --------------------------------------------------
    # Every entry here is a deterministic function of (config, seed) EXCEPT the
    # "timing" sub-block, which quarantines the wall clock so ``save_corpus``
    # can drop it and persist byte-identical shards for a same-seed rerun.
    diagnostics = {
        "edge_types": list(layout.edge_types),
        "draws_per_cell": int(cfg.draws_per_cell),
        "short_horizon_n_query": int(n_query),
        "n_draws_evaluated": int(n_evaluated),
        "rejection_rate": float(n_rejected / max(n_evaluated, 1)),
        # Draw batches that raised a retryable numeric failure and were
        # resampled. These produced no candidates, so they are deliberately
        # absent from n_draws_evaluated / rejection_rate: a nonzero value here
        # means the GRAPH misbehaved, not that the realism filter is strict.
        "n_draw_failures": int(n_draw_failures),
        "edge_base_rates": {
            "cy": float(effective_legacy_edge_rates["cy"]),
            "dc": float(effective_legacy_edge_rates["dc"]),
            "dz": float(cfg.dz_base_rate),
            "dy": float(effective_legacy_edge_rates["dy"]),
            "zy": float(effective_legacy_edge_rates["zy"]),
            "zc": float(cfg.zc_base_rate),
            "cc": float(cfg.cc_base_rate),
            "zz": float(cfg.zz_base_rate),
        },
        "edge_budget": (
            {
                et: list(v) if isinstance(v, (tuple, list)) else int(v)
                for et, v in cfg.edge_budget.items()
            }
            if cfg.edge_budget
            else None
        ),
        "min_no_direct_effect_treatments": int(cfg.min_no_direct_effect_treatments),
        "schema_version": CORPUS_SCHEMA_VERSION,
        # Wall clock lives under its own key so the rest of the block stays a
        # pure function of (config, seed); _finalize_corpus adds tasks_per_sec
        # next to it and save_corpus drops the whole block.
        "timing": {"elapsed_s": float(time.perf_counter() - t_start)},
    }
    if cfg.mechanism_priors_enabled:
        diagnostics["mechanism_priors"] = {
            "parameter_fields": list(mechanism_fields),
            "saturation_prior_ranges": {
                family: {name: [float(lo), float(hi)] for name, (lo, hi) in parameters.items()}
                for family, parameters in cfg.saturation_prior_ranges.items()
            },
            "mm_scale_prior": cfg.mm_scale_prior,
            "treatment_reference_contribution_range": (
                [float(x) for x in cfg.treatment_reference_contribution_range]
                if cfg.treatment_reference_contribution_range is not None
                else None
            ),
            "treatment_reference_multiplier": float(cfg.treatment_reference_multiplier),
            "covariate_reference_contribution_range": (
                [float(x) for x in cfg.covariate_reference_contribution_range]
                if cfg.covariate_reference_contribution_range is not None
                else None
            ),
            "covariate_reference_scale": float(cfg.covariate_reference_scale),
        }
    if cfg.prior_conditioning:
        # Self-describing .npz (as with the signal block): echo the layout,
        # the supports, and the width ranges so consumers derive feature
        # scaling from the corpus instead of duplicating constants.
        spec = cfg.prior_cond_spec()
        diagnostics["prior_cond"] = {
            "layout": list(PRIOR_COND_LAYOUT),
            "supports": {q: list(s["support"]) for q, s in spec.items()},
            "width_ranges": {q: list(s["width_range"]) for q, s in spec.items()},
        }

    corpus["outcome_norm"][:] = outcome_norm
    corpus["outcome_scale"][:] = outcome_scale
    corpus["diagnostics"] = diagnostics
    return corpus
