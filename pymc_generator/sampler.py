"""Stratified corpus sampler for the additive causal SCM.

Each world (task) is an additive structural causal model over latent demand
factors D, observed controls Z, media channels C, a baseline B and sales Y.
The DAG is drawn per cell by :func:`sample_g_additive` (per-edge-type
Bernoulli rates or "pot" budgets over the extended 8-block layout). Each world
is then a PyMC model (:mod:`pymc_generator.world_model`) whose continuous
priors are pm distributions and whose noise is pm RVs; drawing it yields the
series with exact interventional decomposition targets — direct contributions,
per-control / per-confounder contributions, ``baseline_intrinsic`` and the
telescoping 3-source indirect split ``(cc, zc, dc)``.

The corpus schema is the dict-of-ndarrays documented in
:func:`sample_prior_predictive` — the same format the structural-pfn training
pipeline consumes (persist with ``pymc_generator.save_corpus``).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

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
    N_COVARIATES_DEMO,
    N_LATENT_DEMO,
    N_TIME_STEPS_DEMO,
    N_TREATMENTS_DEMO,
    PRIOR_COND_LAYOUT,
    PRIOR_COND_QUANTITIES,
    SlotLayout,
)

#: Canonical categorical orders for media-response mechanism family ids.
ADSTOCK_FAMILY_KEYS: tuple[str, ...] = ("none", "geometric", "weibull")
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
OUTCOME_NOISE_SEMANTICS = "baseline-walk-iid-sales-noise"
OUTCOME_NOISE_VERSION = 1


def _default_adstock_family_probs() -> dict[str, float]:
    """Return a fresh default categorical distribution over adstock families."""
    return dict(zip(ADSTOCK_FAMILY_KEYS, (0.15, 0.425, 0.425), strict=True))


def _default_saturation_family_probs() -> dict[str, float]:
    """Return a fresh default categorical distribution over saturation families."""
    return dict(
        zip(
            SATURATION_FAMILY_KEYS,
            (0.15, 0.17, 0.17, 0.17, 0.17, 0.17),
            strict=True,
        )
    )


#: Maximum draw rounds per cell before giving up (post-filter top-up loop).
MAX_TOPUPS_PER_CELL = 8

#: Default per-quantity width ranges for the prior-conditioning hyperprior
#: (ACE). Mirrors the pymc-pfn precedent (``MMMv0Config``:
#: ``adstock_alpha_width_range=(0.05, 0.45)``, ``hill_n_width_range=(0.2, 1.6)``).
PRIOR_COND_DEFAULT_WIDTH_RANGES: dict[str, tuple[float, float]] = {
    "adstock_alpha": (0.05, 0.45),
    "hill_shape": (0.2, 1.6),
}

#: Bound the diagnostic search so extreme ``query_frac`` values cannot make
#: configuration validation unbounded.
MAX_QUERY_HORIZON_SEARCH_STEPS = 1_000_000

#: Every series and parameter array in the corpus is persisted as float32 (see
#: the cast block at the end of ``_generate_corpus_additive``), so this is the
#: largest magnitude the schema can actually hold.
CORPUS_STORAGE_DTYPE = np.float32
CORPUS_STORAGE_MAX = float(np.finfo(CORPUS_STORAGE_DTYPE).max)


def _n_query(n_time_steps: int, query_frac: float) -> int:
    """Return query weeks from the canonical rounded query-fraction rule."""
    return int(round(query_frac * n_time_steps))


def _minimum_valid_query_horizon(
    n_time_steps: int, query_frac: float, response_support: int
) -> int | None:
    """Return the first valid candidate horizon at or above ``n_time_steps``, if bounded.

    ``response_support`` is the largest lag the admitted adstock kernels can
    reach (see :func:`~pymc_generator.signal_diagnostics.
    admitted_response_support_weeks`), i.e. the number of leading reported
    weeks whose media response depends on pre-window spend.
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
    predicate demands BOTH that the failure is numeric AND that it escaped a
    node PyTensor was evaluating (rather than our own call frames around it).
    """
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

    Supports variable-size DAGs via padding to max sizes. Each cell draws
    random active counts (n_treatments_active, n_covariates_active,
    n_latent_active) from configured ranges. Inactive nodes are zero-padded and
    masked via treatment_active_mask / covariate_active_mask /
    latent_active_mask.

    Key fields:
        n_treatments / n_covariates / n_latent: padded graph sizes (media
            channels / observed controls / hidden confounders).
        n_treatments_active_range / ...: per-cell active-count ranges; inactive
            nodes are zero-padded to the sizes above and masked.

    Prefer building configs through
    :func:`pymc_generator.presets.make_scm_prior`, which pins the
    layout and enables the supported "diverse" channel texture.
    """

    n_time_steps: int = N_TIME_STEPS_DEMO
    n_treatments: int = N_TREATMENTS_DEMO  # media channels (the interventions / treatments)
    n_covariates: int = N_COVARIATES_DEMO  # observed controls
    n_latent: int = N_LATENT_DEMO  # hidden confounders (latent demand factors)
    n_cells: int = 50
    draws_per_cell: int = 20
    val_cell_frac: float = 0.2
    query_frac: float = 0.25
    p_long_horizon: float = 0.5  # fraction of tasks with 50% horizon (vs 25%)
    spend_cv_floor: float = 0.08
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

    # -- prior-range constants -------------------------------------------
    # Per-channel media-response mechanism priors (realized as PyMC
    # distributions in pymc_generator.world_model.build_world_model):
    adstock_alpha_range: tuple[float, float] = (0.2, 0.8)
    # Adstock family probabilities by ``ADSTOCK_FAMILY_KEYS``.
    adstock_family_probs: dict[str, float] = field(default_factory=_default_adstock_family_probs)
    # Saturation family probabilities by ``SATURATION_FAMILY_KEYS``.
    saturation_family_probs: dict[str, float] = field(
        default_factory=_default_saturation_family_probs
    )
    # Weibull adstock prior ranges
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
    beta_additive_range: tuple[float, float] = (0.5, 2.0)  # channel effects
    # Random-walk and outcome-noise priors.
    rw_control_mean_range: tuple[float, float] = (-1.0, 1.0)  # control drive (Z)
    rw_positive_mean_range: tuple[float, float] = (0.5, 3.0)  # channels
    rw_baseline_mean_range: tuple[float, float] = (3.0, 8.0)  # baseline level
    rw_std_sigma: float = 1.0  # HalfNormal prior for Z and absolute-mode RW_B std
    rw_baseline_std_sigma: float | None = None  # absolute-mode RW_B; None follows rw_std_sigma
    rw_sales_std_sigma: float = 0.25  # absolute-mode iid sales-noise std
    outcome_std_mode: Literal["relative", "absolute"] = "relative"
    # Relative-mode outcome amplitudes multiply
    # sqrt(sum_k((g_cy[k] * beta[k]) ** 2)), never a realized series.
    rw_baseline_std_range: tuple[float, float] = (0.000, 0.093)
    rw_sales_std_range: tuple[float, float] = (0.010, 0.028)
    rw_smoothness_alpha: float = 2.0  # Beta prior alpha for smoothness
    rw_smoothness_beta: float = 2.0  # Beta prior beta for smoothness
    # 26 weeks (half a year) reproduces the CURRENT n_time_steps=104 reference
    # exactly at smoothness=1.0 (round(1.0 * 104 / 4) == 26), preserving its
    # drift character while making every other horizon consistent. This is a
    # config knob, not a constant, so drift timescale stays tunable
    # independently of n_time_steps — that flexibility is the point.
    rw_smoothness_max_weeks: int = 26
    # Optional shared innovation between the baseline and every channel. When
    # enabled, rho is resolved per world and mixes their already-standardized
    # innovations without changing either marginal innovation variance.
    confounding_strength_range: tuple[float, float] | None = None

    # -- Channel own-drive variation --------------------------------------
    # ``rw_channel_std_range`` is the channel random-walk standard-deviation
    # prior, relative to each channel's own level (softplus of its walk mean).
    # That keeps channel variation scale-free across small and large channels.
    # High-frequency exogenous drive on the channel's own pre-softplus input
    # comes from iid weekly noise (sigma ~ U(range)) and campaign pulses
    # (per-week probability ~ U(prob_range), amplitude ~ U(amp_range)).
    # Defaults retain the deprecated smooth-walk-only texture; make_scm_prior
    # enables the diverse texture, which is the supported prior.
    rw_channel_std_range: tuple[float, float] = (0.15, 0.8)
    channel_hf_sigma_range: tuple[float, float] = (0.0, 0.0)
    channel_pulse_prob_range: tuple[float, float] = (0.0, 0.0)
    channel_pulse_amp_range: tuple[float, float] = (0.5, 1.5)

    # -- Control texture ---------------------------------------------------
    # A control's own drive is otherwise a smoothed random walk, i.e. exactly
    # the function class the equally smooth baseline walk (RW_B) spans, so the
    # Z->Y coefficient trades off against baseline drift and is only weakly
    # identified. These knobs add the high-frequency content a smooth walk
    # cannot mimic — iid weekly noise (sigma ~ U(range)) and calendar pulses
    # (per-week probability ~ U(prob_range), amplitude ~ U(amp_range)) — which
    # is also what real controls (promos, holidays, price steps) look like.
    # Both magnitudes are drawn RELATIVE to the control's own walk std, so they
    # are scale-free like the channel factors; unlike the channel pulse, the
    # control pulse is CENTRED (``amp * (fire - prob)``), because a control is
    # signed and its level is identified by ``rw_z_mean`` alone.
    # Defaults are inert: no graph term, the magnitudes degenerate to constants
    # (no parameter RV, no RNG consumed) and corpora stay byte-identical to the
    # pre-texture format. make_scm_prior enables them.
    control_hf_sigma_range: tuple[float, float] = (0.0, 0.0)
    control_pulse_prob_range: tuple[float, float] = (0.0, 0.0)
    control_pulse_amp_range: tuple[float, float] = (0.0, 0.0)

    # -- Intercept floor ---------------------------------------------------
    # The intercept walk RW_B is signed, so the baseline — and, once the signed
    # D->Y / Z->Y terms are added, sales — can dip below zero. Rare at the
    # default relative-mode scale (mean 3-8, amplitude <= ~0.12 x the media
    # amplitude; measured 0.08% of weeks) but routine in absolute mode with a
    # low mean. A floor CENSORS the intercept: ``B = max(RW_B, baseline_floor)``,
    # so B is >= floor by construction and may sit exactly AT it (a censored
    # walk, not a softplus — zeros are a legitimate baseline).
    #
    # This is a PRIOR-level constraint on one additive term, so it stays inside
    # the function class a consuming MMM can represent. Sales is deliberately
    # NOT clamped: that would censor the OBSERVATION (a likelihood-level
    # change) and make every additive-Gaussian estimator — including this
    # package's own oracle — misspecified. Non-negative sales remains enforced
    # exactly by the acceptance filter. Note the oracle's analytic
    # ``latent="marginal"`` mode is unavailable with a floor, because a
    # censored walk is not Gaussian.
    # None (default) keeps the signed walk and byte-identical draws.
    baseline_floor: float | None = None
    # WHAT the floor clips, when one is configured.
    #   "intercept" (default): the intercept walk only. Every other term stays
    #       exactly linear in its node, so ``control_contribution[:, m]`` remains
    #       ``g_zy[m]·ρ[m]·Z[:, m]`` — but a large negative ρ·Z can still drag
    #       the non-media total, and sales, below zero.
    #   "non_media": the running non-media TOTAL, clipped as each parent joins in
    #       the locked order intercept -> confounders -> controls. A negative
    #       control effect is then credited only down to the floor and the excess
    #       is absorbed, so the whole mean function of sales is >= 0 by
    #       construction (media contributions are already >= 0). Per-node columns
    #       become the telescoping differences each node caused — exact, and
    #       identical to the linear split wherever the floor does not bind, but
    #       no longer linear in the node where it does.
    baseline_floor_scope: Literal["intercept", "non_media"] = "intercept"
    adstock_burn_in: int = 0

    # Symbolic, per-draw held-level channel shocks. A shock clamps observed
    # spend for its window; it never touches the channel's response state, so
    # the clamped path adstocks with the ordinary normalized causal kernel.
    n_channel_shocks: int = 0
    channel_shock_length_range: tuple[int, int] = (2, 2)
    channel_shock_level_range: tuple[float, float] = (0.0, 0.0)

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
    # expose them as conditioning features. False (default) => byte-identical
    # unconditioned corpora (the interval draws consume no RNG when disabled).
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
    # Example: {"zc": 5} places up to 5 control->channel arrows however they
    # land (one control fanning out, or spread across controls — capped at 5),
    # while "zy" (controls' effect on the outcome) is untouched.
    # Note: "cy" keeps its >=1 floor from the degenerate C->Y guard, so a cy
    # budget that draws 0 still yields exactly one C->Y edge.
    edge_budget: dict[str, int | tuple[int, int]] | None = None

    # Reserve active channels with no direct C->Y edge and zero direct
    # contribution. Unlike a fixed cy budget, the cap follows each cell's
    # active count. Such channels can still affect Y through C->C paths.
    # Zero leaves the graph-sampling RNG schedule unchanged.
    min_no_direct_effect_channels: int = 0

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
    def rw_baseline_std_sigma_effective(self) -> float:
        """Baseline walk scale, following the shared walk scale unless overridden."""
        if self.rw_baseline_std_sigma is None:
            return self.rw_std_sigma
        return self.rw_baseline_std_sigma

    @property
    def n_query(self) -> int:
        return _n_query(self.n_time_steps, self.query_frac)

    def prior_cond_spec(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Effective ``{quantity: {"support": (lo, hi), "width_range": (w_lo, w_hi)}}``.

        Quantities iterate in the LOCKED ``PRIOR_COND_QUANTITIES`` order (the
        order both the interval RNG draws and the ``prior_cond`` columns
        follow). Supports come from the same constants the unconditioned
        priors use — ``adstock_alpha_range`` for the geometric adstock decay,
        ``SATURATION_PRIOR_RANGES["hill"]["slope"]`` for the Hill shape — so
        the conditioned interval is nested in the exact global prior by
        construction. Width ranges default to
        :data:`PRIOR_COND_DEFAULT_WIDTH_RANGES`, overridable per quantity via
        ``prior_cond_width_ranges``.
        """
        # Deferred: mechanisms pulls the pytensor / pymc-marketing stack.
        from .mechanisms import SATURATION_PRIOR_RANGES

        supports: dict[str, tuple[float, float]] = {
            "adstock_alpha": (
                float(self.adstock_alpha_range[0]),
                float(self.adstock_alpha_range[1]),
            ),
            "hill_shape": (
                float(SATURATION_PRIOR_RANGES["hill"]["slope"][0]),
                float(SATURATION_PRIOR_RANGES["hill"]["slope"][1]),
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

    def validate(self) -> None:
        def _integer(name: str, value, *, minimum: int) -> None:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")

        def _finite_real(name: str, value, *, positive: bool = False, nonnegative: bool = False):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or (positive and value <= 0)
                or (nonnegative and value < 0)
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
            ("min_no_direct_effect_channels", 0),
        ):
            _integer(name, getattr(self, name), minimum=minimum)
        _finite_real("query_frac", self.query_frac, positive=True)
        _finite_real("val_cell_frac", self.val_cell_frac, positive=True)
        if self.val_cell_frac >= 1.0:
            raise ValueError(f"val_cell_frac must be < 1, got {self.val_cell_frac}")
        _finite_real("spend_cv_floor", self.spend_cv_floor, nonnegative=True)
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
            "adstock_family_probs", self.adstock_family_probs, ADSTOCK_FAMILY_KEYS
        )
        _family_probabilities(
            "saturation_family_probs", self.saturation_family_probs, SATURATION_FAMILY_KEYS
        )

        def _finite_range(
            name: str,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            minimum_exclusive: bool = False,
            reason: str | None = None,
        ) -> tuple[float, float]:
            value = getattr(self, name)
            try:
                lo, hi = value
                if isinstance(lo, (bool, np.bool_)) or isinstance(hi, (bool, np.bool_)):
                    raise TypeError
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError):
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

        _finite_range("adstock_alpha_range", minimum=0.0, maximum=1.0)
        _finite_range("weibull_lam_range", minimum=0.0, minimum_exclusive=True)
        _finite_range("weibull_k_range", minimum=0.0, minimum_exclusive=True)
        for name in (
            "dc_coeff_range",
            "dz_coeff_range",
            "zc_coeff_range",
            "zz_coeff_range",
            "dy_coeff_range",
            "zy_coeff_range",
            "rw_control_mean_range",
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
        for name in ("rw_baseline_std_range", "rw_sales_std_range"):
            _finite_range(name, minimum=0.0)
        _finite_range(
            "cc_coeff_range",
            minimum=0.0,
            reason="C->C coefficients amplify, not cannibalize",
        )
        _, beta_additive_hi = _finite_range(
            "beta_additive_range",
            minimum=0.0,
            reason="channel effect amplitudes must be nonnegative",
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
            reason="channel walks stay positive after softplus",
        )
        _, channel_std_hi = _finite_range(
            "rw_channel_std_range",
            minimum=0.0,
            reason="channel walk amplitudes must be nonnegative",
        )
        if channel_std_hi <= 0.0:
            raise ValueError(
                "rw_channel_std_range must have an upper bound > 0 because a zero-only "
                "channel-walk amplitude produces flat channel paths"
            )
        _finite_range("channel_hf_sigma_range", minimum=0.0)
        _, channel_pulse_prob_hi = _finite_range(
            "channel_pulse_prob_range", minimum=0.0, maximum=0.5
        )
        _finite_range("channel_pulse_amp_range", minimum=0.0)
        _finite_range("control_hf_sigma_range", minimum=0.0)
        _finite_range("control_pulse_prob_range", minimum=0.0, maximum=0.5)
        _finite_range("control_pulse_amp_range", minimum=0.0)
        if self.baseline_floor is not None:
            floor = self.baseline_floor
            if (
                isinstance(floor, (bool, np.bool_))
                or not isinstance(floor, (int, float, np.integer, np.floating))
                or not np.isfinite(floor)
            ):
                raise ValueError(f"baseline_floor must be None or a finite number, got {floor!r}")
        if self.baseline_floor_scope not in ("intercept", "non_media"):
            raise ValueError(
                "baseline_floor_scope must be 'intercept' or 'non_media', got "
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
            for q, cond_spec in self.prior_cond_spec().items():
                s_lo, s_hi = cond_spec["support"]
                w_lo, w_hi = cond_spec["width_range"]
                if not 0.0 < w_lo <= w_hi <= s_hi - s_lo:
                    raise ValueError(
                        f"prior conditioning for {q!r} needs "
                        f"0 < w_lo <= w_hi <= support width; got width range "
                        f"({w_lo}, {w_hi}) against support ({s_lo}, {s_hi})"
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
        for name in ("rw_std_sigma", "rw_sales_std_sigma"):
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
        # Channel texture
        if isinstance(self.adstock_burn_in, bool) or not isinstance(
            self.adstock_burn_in, (int, np.integer)
        ):
            raise ValueError("adstock_burn_in must be an int (not bool)")
        if self.adstock_burn_in < 0:
            raise ValueError(f"adstock_burn_in must be >= 0, got {self.adstock_burn_in}")
        if 0 < self.adstock_burn_in < self.l_max:
            # A partial burn-in still convolves the first reported weeks into
            # zero padding — the warmup artifact the knob exists to remove.
            # Also catches dataclasses.replace(cfg, l_max=...) desyncing a
            # preset-built config (presets pin burn_in = l_max).
            raise ValueError(
                f"adstock_burn_in={self.adstock_burn_in} must be 0 (off) or >= "
                f"l_max ({self.l_max}) — a partial burn-in leaves adstock warmup "
                f"in the reported window"
            )
        if self.adstock_burn_in > 0:
            # The boundary is the response's REACH, not the kernel length: an
            # identity-only family mix (nonlinearity="linear") convolves nothing,
            # so no reported week depends on pre-window spend and a short horizon
            # is perfectly scorable even though the presets still pin
            # adstock_burn_in = l_max. Take the upper bound over every family the
            # config admits (probability > 0) — the families are drawn per world,
            # so validation cannot know which one a given world gets.
            admitted_families = [
                index
                for index, key in enumerate(ADSTOCK_FAMILY_KEYS)
                if self.adstock_family_probs[key] > 0.0
            ]
            warmup_boundary = admitted_response_support_weeks(
                admitted_families, self.l_max, adstock_alpha_range=self.adstock_alpha_range
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
                    "adstock burn-in query overlap: "
                    f"n_time_steps={self.n_time_steps}, l_max={self.l_max}, "
                    f"n_query={self.n_query}; "
                    f"short-horizon query start n_time_steps - n_query={short_query_start}, "
                    f"long-horizon query start n_time_steps // 2={long_query_start}. With "
                    f"burn-in, the first admitted_response_support_weeks = "
                    f"{warmup_boundary} reported weeks carry a "
                    "media response that depends on unpersisted pre-window spend. Both the "
                    "short-horizon (n_time_steps - n_query) and long-horizon "
                    "(n_time_steps // 2) query windows must "
                    "start at or after that boundary, otherwise tasks are scored on targets that "
                    "are not a function of the persisted inputs. To reach this world anyway, "
                    "either set adstock_burn_in=0 (the convolution then zero-pads, which is "
                    "reproducible from persisted spend at every week), restrict "
                    f"adstock_family_probs to the identity family, {horizon_remedy}or lower "
                    "query_frac / l_max."
                )
        if channel_pulse_prob_hi > 0.0 and float(self.channel_pulse_amp_range[1]) <= 0.0:
            raise ValueError(
                "channel_pulse_prob_range enables pulses but channel_pulse_amp_range "
                "has zero amplitude — disable pulses via the prob range instead"
            )
        control_p_hi = float(self.control_pulse_prob_range[1])
        if control_p_hi > 0.0 and float(self.control_pulse_amp_range[1]) <= 0.0:
            raise ValueError(
                "control_pulse_prob_range enables pulses but control_pulse_amp_range "
                "has zero amplitude — disable pulses via the prob range instead"
            )
        # Symbolic channel-shock schedule. Keep this validation explicit rather
        # than relying on PyMC's distribution errors, so invalid schedules fail
        # before a model is built.
        if isinstance(self.n_channel_shocks, bool) or not isinstance(
            self.n_channel_shocks, (int, np.integer)
        ):
            raise ValueError("n_channel_shocks must be an int (not bool)")
        if self.n_channel_shocks < 0:
            raise ValueError(f"n_channel_shocks must be >= 0, got {self.n_channel_shocks}")
        if not isinstance(self.include_identifiability_labels, bool):
            raise ValueError("include_identifiability_labels must be a bool")
        try:
            shock_len_lo, shock_len_hi = self.channel_shock_length_range
        except (TypeError, ValueError):
            raise ValueError("channel_shock_length_range must be an (lo, hi) integer pair")
        if (
            isinstance(shock_len_lo, bool)
            or isinstance(shock_len_hi, bool)
            or not isinstance(shock_len_lo, (int, np.integer))
            or not isinstance(shock_len_hi, (int, np.integer))
            or not 1 <= shock_len_lo <= shock_len_hi <= self.n_time_steps
        ):
            raise ValueError(
                "channel_shock_length_range must have integer bounds satisfying "
                f"1 <= lo <= hi <= n_time_steps, got {self.channel_shock_length_range!r}"
            )
        try:
            shock_level_lo, shock_level_hi = self.channel_shock_level_range
            shock_level_lo, shock_level_hi = float(shock_level_lo), float(shock_level_hi)
        except (TypeError, ValueError):
            raise ValueError("channel_shock_level_range must be a finite (lo, hi) pair")
        if not (
            np.isfinite(shock_level_lo)
            and np.isfinite(shock_level_hi)
            and 0.0 <= shock_level_lo <= shock_level_hi
        ):
            raise ValueError(
                "channel_shock_level_range must satisfy finite 0 <= lo <= hi, "
                f"got {self.channel_shock_level_range!r}"
            )
        _reject_unrepresentable_bounds(
            "channel_shock_level_range",
            self.channel_shock_level_range,
            shock_level_lo,
            shock_level_hi,
        )
        if self.n_channel_shocks > self.n_time_steps:
            raise ValueError(
                f"n_channel_shocks must be <= n_time_steps ({self.n_time_steps}), "
                f"got {self.n_channel_shocks}"
            )
        if self.n_channel_shocks and shock_len_lo < 2:
            raise ValueError(
                "enabled channel shocks must last at least 2 weeks so the held level "
                "is observable in spend"
            )
        if self.n_channel_shocks and self.n_channel_shocks * shock_len_hi > self.n_time_steps:
            raise ValueError(
                "n_channel_shocks * max channel_shock_length must be <= n_time_steps, got "
                f"{self.n_channel_shocks} * {shock_len_hi} > {self.n_time_steps}"
            )
        # Validate variable-size DAG ranges. The upper bound may exceed the
        # padded size and is intentionally clamped, but both declared bounds
        # must still be ordered integers.
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
        # The smallest cell must accommodate both the direct-null floor and
        # the mandatory direct channel.
        if self.min_no_direct_effect_channels:
            active_lo = self.n_treatments_active_range[0]
            if active_lo <= self.min_no_direct_effect_channels:
                raise ValueError(
                    f"min_no_direct_effect_channels ({self.min_no_direct_effect_channels}) must be < "
                    f"n_treatments_active_range[0] ({active_lo}) so every cell can keep at "
                    "least one direct channel besides the direct-null ones"
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
        Minimum active channels without a direct C->Y edge (see
        ``SCMPrior.min_no_direct_effect_channels``). Caps the direct count at
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
        # Reserve `min_no_direct` active channels without a direct C->Y edge.
        # Direct channels are still
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

    ``channel_active`` marks a direct C->Y edge OR an outgoing C->C edge.
    This local structural flag does not guarantee a path to Y through a
    downstream channel. All channels remain observed; the degenerate guard
    only forces at least one C->Y edge.

    ``cfg.min_no_direct_effect_channels`` caps the direct-channel count, so a
    cell can be made to always carry both classes of the direct-effect signal.

    Returns
    -------
    dict with the 8 g-blocks at max (padded) sizes — ``g_cc``/``g_zz`` as
    FULL square matrices with zero diagonals — plus active masks, active
    counts, and ``channel_active``.
    """
    base = _sample_g(
        rng,
        layout,
        n_treatments_active=n_treatments_active,
        n_covariates_active=n_covariates_active,
        n_latent_active=n_latent_active,
        rates=cfg.edge_rate_overrides,
        budget=cfg.edge_budget,
        min_no_direct=cfg.min_no_direct_effect_channels,
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
    channel_active = ((base["g_cy"] == 1) | (g_cc.sum(axis=1) > 0)).astype("float64")
    channel_active *= base["active_treatment"]  # padding nodes are never active

    return {
        **base,
        "g_dz": g_dz,
        "g_zc": g_zc,
        "g_cc": g_cc,
        "g_zz": g_zz,
        "channel_active": channel_active,
    }


def _signal_block(
    cfg: SCMPrior,
    layout: SlotLayout,
    sales_raw: np.ndarray,
    g_tasks: np.ndarray,
    treatment_active_mask: np.ndarray,
    sales_scale: np.ndarray,
    adstock_family: np.ndarray,
    adstock_alpha: np.ndarray,
    weibull_lam: np.ndarray,
    weibull_k: np.ndarray,
    signal_metrics: np.ndarray,
    signal_metric_valid: np.ndarray,
) -> dict:
    """``diagnostics["signal"]`` for a corpus.

    "Direct active channel" = cy edge present AND channel not padding.
    """
    cy_mask = (g_tasks[:, layout.slices["cy"]] == 1) & (treatment_active_mask == 1)
    out = summarize_signal_metrics(
        signal_metrics,
        signal_metric_valid,
        sales_raw,
        cy_mask,
        sales_scale=sales_scale,
        l_max=cfg.l_max,
        adstock_burn_in=cfg.adstock_burn_in,
        adstock_family=adstock_family,
        adstock_alpha=adstock_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
    )
    out["metric_version"] = SIGNAL_METRIC_VERSION
    out["metric_layout"] = list(SIGNAL_METRIC_LAYOUT)
    out["l_max"] = int(cfg.l_max)
    out["adstock_burn_in"] = int(cfg.adstock_burn_in)
    # pymc-marketing min-max rescales the density before sum-normalizing, so one lag
    # has zero weight; a true normalized PDF would not have an exactly zero lag.
    out["adstock_kernel_semantics"] = "normalized-causal-minmax-weibull-density"
    out["adstock_kernel_version"] = 3
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
    n_tasks = corpus["spend_raw"].shape[0]
    diagnostics = corpus["diagnostics"]
    elapsed = diagnostics["timing"]["elapsed_s"]

    g_tasks = corpus["g"]
    treatment_active_mask = corpus["treatment_active_mask"]
    cy_mask = (g_tasks[:, layout.slices["cy"]] == 1) & (treatment_active_mask == 1)
    signal_metrics, signal_metric_valid = dense_signal_metrics(
        corpus["spend_raw"],
        corpus["contributions_raw"],
        corpus["sales_raw"],
        corpus["baseline_raw"],
        cy_mask,
        sales_scale=corpus["sales_scale"],
        adstock_family=corpus["adstock_family"],
        adstock_alpha=corpus["adstock_alpha"],
        weibull_lam=corpus["weibull_lam"],
        weibull_k=corpus["weibull_k"],
        l_max=cfg.l_max,
        adstock_burn_in=cfg.adstock_burn_in,
    )
    if cfg.include_identifiability_labels:
        corpus["identifiability"] = {
            "signal_metrics": signal_metrics,
            "signal_metric_valid": signal_metric_valid,
        }
    diagnostics["signal"] = _signal_block(
        cfg,
        layout,
        corpus["sales_raw"],
        g_tasks,
        treatment_active_mask,
        corpus["sales_scale"],
        corpus["adstock_family"],
        corpus["adstock_alpha"],
        corpus["weibull_lam"],
        corpus["weibull_k"],
        signal_metrics,
        signal_metric_valid,
    )

    # These are intentionally calculated from the retained, persisted arrays.
    # Do not regenerate per-task normalizers here: slicing their already-cast
    # values preserves the serialized-array compatibility contract.
    spend = corpus["spend_raw"].astype(np.float64)
    contributions = corpus["contributions_raw"].astype(np.float64)
    baseline = corpus["baseline_raw"].astype(np.float64)
    qs = (0.1, 0.5, 0.9)
    spend_mean = spend.mean(axis=1)
    cv_all = np.divide(
        spend.std(axis=1), spend_mean, out=np.zeros_like(spend_mean), where=spend_mean != 0.0
    ).ravel()
    contrib_tot = contributions.sum(axis=(1, 2))
    media_denominator = contrib_tot + baseline.sum(axis=1)
    media_share = np.divide(
        contrib_tot,
        media_denominator,
        out=np.zeros_like(contrib_tot),
        where=media_denominator != 0.0,
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
            "media_share_quantiles": {
                f"q{int(q * 100)}": float(np.quantile(media_share, q)) for q in qs
            },
            "spend_cv_quantiles": {f"q{int(q * 100)}": float(np.quantile(cv_all, q)) for q in qs},
        }
    )
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
                    + corpus["contributions_raw"].sum(axis=-1)
                    + corpus["indirect_effects"]
                    - corpus["sales_raw"]
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
                    + corpus["sales_noise"]
                    + corpus["confounder_contribution"].sum(axis=-1)
                    + corpus["control_contribution"].sum(axis=-1)
                    + corpus["contributions_raw"].sum(axis=-1)
                    + corpus["indirect_effects_by_source"].sum(axis=-1)
                    - corpus["sales_raw"]
                ).max()
            ),
            "baseline_decomposition_max_abs_error": float(
                np.abs(
                    corpus["baseline_intrinsic"]
                    + corpus["sales_noise"]
                    + corpus["confounder_contribution"].sum(axis=-1)
                    + corpus["control_contribution"].sum(axis=-1)
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
    additive SCM with high-frequency channel texture and adstock burn-in.
    A config with the flat (smooth-walk-only) channel prior still generates
    but warns: its contribution targets degenerate to near-flat lines
    (measured on the reference config: ~49% of direct-channel targets without
    week-to-week variation; see ``signal_diagnostics``).
    """
    if (
        float(cfg.channel_hf_sigma_range[1]) == 0.0
        and float(cfg.channel_pulse_prob_range[1]) == 0.0
    ):
        warnings.warn(
            "The flat (smooth-walk-only) channel texture is "
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
    ``control_contribution``, ``confounder_contribution``, ``baseline_intrinsic``)
    plus ``channel_active``.

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
    Priors with the flat (texture-free) channel prior emit a ``FutureWarning``
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
    if n is not None and corpus["spend_raw"].shape[0] > n:
        actual = corpus["spend_raw"].shape[0]
        for key, val in list(corpus.items()):
            if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == actual:
                corpus[key] = val[:n]
    _recompute_retained_cell_split(corpus)
    return _finalize_corpus(corpus, prior)


# --------------------------------------------------------------------------
# Additive causal graph corpus
# --------------------------------------------------------------------------

_ADDITIVE_OUT_NAMES = (
    "demand",
    "controls",
    "channels",
    "baseline",
    "baseline_intrinsic",
    "control_contribution",
    "confounder_contribution",
    "contributions",
    "indirect_effects",
    "indirect_effects_by_source",
    "sales",
    "confounding_strength",
    # Appended last: this tuple is the draw-name order, which fixes PyTensor's
    # RNG traversal, so a new name must not displace an existing one.
    "sales_noise",
)

# Corpus audit metadata.  These are already deterministics in each cell model;
# requesting just these values preserves the one-model-per-cell sampling path
# while avoiding persistence of natural-path realism outputs or burn-in masks.
_CORPUS_SHOCK_NAMES = (
    "channel_shock_mask",
    "channel_shock_channel",
    "channel_shock_start",
    "channel_shock_length",
    "channel_shock_level_multiplier",
    "channel_shock_level",
)
_CORPUS_PARAM_NAMES = (
    "saturation_scale",
    "param_channel_level",
    "param_adstock_alpha",
    "param_weibull_lam",
    "param_weibull_k",
)


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
    spend: np.ndarray,
    sales: np.ndarray,
    arrays: dict[str, np.ndarray],
    g_cy_active: np.ndarray,
    cv_floor: float,
    sales_spike_ratio: float = 8.0,
    spend_spike_ratio: float = 50.0,
    realism_spend: np.ndarray | None = None,
    realism_sales: np.ndarray | None = None,
) -> bool:
    """Single-task realism filter for the additive SCM.

    Realism checks on the additive scale: finite
    arrays and non-negative actual sales are always required.  CV and spike
    checks can use natural (unshocked) audit paths so a deliberate intervention
    is not rejected for looking unlike organic spend.
    """
    for a in arrays.values():
        if not np.isfinite(a).all():
            return False
    if (sales < 0).any():
        return False
    realism_spend = spend if realism_spend is None else realism_spend
    realism_sales = sales if realism_sales is None else realism_sales
    active = np.asarray(g_cy_active) == 1
    if active.any():
        cv = realism_spend.std(axis=0) / (realism_spend.mean(axis=0) + 1e-12)  # (n_treatments,)
        if cv[active].min() < cv_floor:
            return False
    sales_med = np.median(realism_sales)
    safe_med = sales_med if sales_med > 0 else 1.0
    if realism_sales.max() / safe_med >= sales_spike_ratio:
        return False
    spend_med = np.median(realism_spend, axis=0)  # (n_treatments,)
    safe_spend_med = np.where(spend_med > 0, spend_med, 1.0)
    if (realism_spend.max(axis=0) / safe_spend_med).max() >= spend_spike_ratio:
        return False
    return True


def _generate_corpus_additive(cfg: SCMPrior) -> dict[str, Any]:
    """Generate a padded additive-SCM corpus.

    The public schema is documented by :func:`sample_prior_predictive`.
    Each cell shares a discrete structure; each task draws fresh continuous
    parameters and innovations. Evaluate at active sizes, then copy accepted
    values into the preallocated maximum-size arrays. Float32 storage follows
    the float64 generation and acceptance checks.
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
        "shock": cfg.n_channel_shocks,
        "indirect_source": 3,
    }
    corpus: dict[str, Any] = {
        key: np.zeros(tuple(dimensions[axis] for axis in axes), dtype=dtype)
        for key, (axes, dtype) in CORPUS_ARRAY_FIELDS.items()
    }
    if cfg.prior_conditioning:
        corpus["prior_cond"] = np.empty((n_tasks, len(PRIOR_COND_LAYOUT)), dtype=np.float32)
    # Spend normalization retains the original float64 reduction order.
    # Other accepted outputs can be cast directly into their final storage.
    spend_raw = np.zeros((n_tasks, n_time_steps, n_treatments_max), dtype=np.float64)
    draw_fields = {
        "controls": "controls",
        "sales_raw": "sales",
        "contributions_raw": "contributions",
        "baseline_raw": "baseline",
        "demand": "demand",
        "indirect_effects": "indirect_effects",
        "baseline_intrinsic": "baseline_intrinsic",
        "sales_noise": "sales_noise",
        "control_contribution": "control_contribution",
        "confounder_contribution": "confounder_contribution",
        "indirect_effects_by_source": "indirect_effects_by_source",
        "confounding_strength": "confounding_strength",
        "channel_shock_mask": "channel_shock_mask",
        "channel_shock_channel": "channel_shock_channel",
        "channel_shock_start": "channel_shock_start",
        "channel_shock_length": "channel_shock_length",
        "channel_shock_level_multiplier": "channel_shock_level_multiplier",
        "channel_shock_level": "channel_shock_level",
        "channel_level": "param_channel_level",
        "saturation_scale": "saturation_scale",
        "adstock_alpha": "param_adstock_alpha",
        "weibull_lam": "param_weibull_lam",
        "weibull_k": "param_weibull_k",
    }
    n_evaluated = 0
    n_rejected = 0
    n_draw_failures = 0

    for cell in range(cfg.n_cells):
        tr = cfg.n_treatments_active_range_effective
        cv = cfg.n_covariates_active_range_effective
        lt = cfg.n_latent_active_range_effective
        n_treatments_active = int(rng.integers(tr[0], tr[1] + 1))
        n_covariates_active = int(rng.integers(cv[0], cv[1] + 1))
        n_latent_active = int(rng.integers(lt[0], lt[1] + 1))
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
            ("channel_active", "channel_active"),
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
        # Prior-conditioning interval draw (per cell — one model build). Returns
        # None and consumes NO RNG when cfg.prior_conditioning is False, so the
        # disabled path reproduces unconditioned corpora bit-for-bit.
        prior_cond = sample_prior_cond(cfg, rng)
        if prior_cond is not None:
            corpus["prior_cond"][rows] = _pack_prior_cond(prior_cond)
        model, out_names, _param_names = build_world_model(
            g_act, cfg, structural, n_time_steps, prior_cond=prior_cond
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
                # Output order controls PyTensor's RNG traversal. Metadata
                # deterministics must precede the legacy outputs: appending
                # them changes legacy draws, while a separate same-seed call
                # produces parameters from a different joint world.
                draw_names: tuple[str, ...]
                if cfg.n_channel_shocks:
                    draw_names = (
                        _CORPUS_PARAM_NAMES
                        + _CORPUS_SHOCK_NAMES
                        + _ADDITIVE_OUT_NAMES
                        + ("channels_unshocked", "sales_unshocked")
                    )
                else:
                    draw_names = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES
                drawn_b = draw_worlds(model, draw_names, draw_seed, draws=n_req)
            except _RETRYABLE_DRAW_ERRORS as exc:
                if not _is_retryable_draw_failure(exc):
                    # Not a numeric failure from inside a PyTensor node
                    # evaluation: a bug here recurs at every seed, so retrying
                    # only buries it under MAX_TOPUPS_PER_CELL empty rounds and
                    # then blames the realism filter. Let it out untouched.
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

                if not _additive_task_ok(
                    spend=drawn["channels"],
                    sales=drawn["sales"],
                    arrays=drawn,
                    g_cy_active=g_act["g_cy"],
                    cv_floor=cfg.spend_cv_floor,
                    realism_spend=drawn.get("channels_unshocked"),
                    realism_sales=drawn.get("sales_unshocked"),
                ):
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                support, task_split_type = _make_support_mask(
                    rng, n_time_steps, n_query, cfg.p_long_horizon
                )
                candidate_sales_scale = float(np.std(drawn["sales"][support == 1]))
                if not (np.isfinite(candidate_sales_scale) and candidate_sales_scale > 0.0):
                    n_rejected += 1
                    cell_rejected += 1
                    continue

                row = cell * cfg.draws_per_cell + accepted
                spend_raw[row, :, :n_treatments_active] = drawn["channels"]
                for key, source in draw_fields.items():
                    value = drawn[source]
                    prefix: tuple[int | slice, ...] = (
                        row,
                        *(slice(size) for size in np.shape(value)),
                    )
                    corpus[key][prefix] = value
                corpus["adstock_family"][row, :n_treatments_active] = structural["adstock_family"]
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
                f"cell {cell}: only {accepted}/{cfg.draws_per_cell} tasks "
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

    spend_means = spend_raw.mean(axis=1)  # (n_tasks, n_treatments)
    spend_norm = np.divide(
        spend_raw,
        spend_means[:, None, :],
        out=np.zeros_like(spend_raw),
        where=spend_means[:, None, :] != 0.0,
    )
    active_spend_sum = (spend_raw * treatment_active_mask[:, None, :]).sum(axis=-1, keepdims=True)
    spend_share = (
        np.divide(
            spend_raw,
            active_spend_sum,
            out=np.zeros_like(spend_raw),
            where=active_spend_sum != 0.0,
        )
        * treatment_active_mask[:, None, :]
    )

    corpus["spend_raw"][:] = spend_raw
    corpus["spend_norm"][:] = spend_norm
    corpus["spend_share"][:] = spend_share
    corpus["spend_means"][:] = spend_means
    del spend_raw, spend_norm, spend_share, spend_means

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
    # float64 draw. `sales_norm = sales_raw / sales_scale` is persisted too, so a
    # consumer must be able to reproduce both from the corpus alone; computing
    # the scale at draw precision made that impossible whenever float32 rounding
    # dominated the standard deviation. That is reachable: a short support window
    # over a very smooth walk gives a near-constant slice, where the writer and a
    # float32 recomputation disagreed by 2.5e-5 relative — past the validator's
    # 1e-5 tolerance.
    sales_stored = corpus["sales_raw"]
    sales_scale = np.array(
        [
            float(np.std(sales_stored[i][support_mask[i] == 1].astype(np.float64)))
            for i in range(n_tasks)
        ],
        dtype=np.float64,
    )
    bad_scale = ~(np.isfinite(sales_scale) & (sales_scale > 0.0))
    if bad_scale.any():
        full_std = np.std(sales_stored[bad_scale].astype(np.float64), axis=1)
        sales_scale[bad_scale] = np.where(np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0)
    sales_norm = sales_stored.astype(np.float64) / sales_scale[:, None]

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
        "min_no_direct_effect_channels": int(cfg.min_no_direct_effect_channels),
        "schema_version": CORPUS_SCHEMA_VERSION,
        # Wall clock lives under its own key so the rest of the block stays a
        # pure function of (config, seed); _finalize_corpus adds tasks_per_sec
        # next to it and save_corpus drops the whole block.
        "timing": {"elapsed_s": float(time.perf_counter() - t_start)},
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

    corpus["sales_norm"][:] = sales_norm
    corpus["sales_scale"][:] = sales_scale
    corpus["diagnostics"] = diagnostics
    return corpus
