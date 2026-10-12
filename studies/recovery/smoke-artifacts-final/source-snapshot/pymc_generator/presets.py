"""Complexity presets for the additive-SCM corpus.

The additive SCM fixes the *structure* — the 8-block extended edge layout
(cy, dc, dz, dy, zy, zc, cc, zz), the additive structural equations, and the
``indirect_effects`` outputs. This module dials *complexity* within that fixed
structure along orthogonal axes, keeping the schema (tensor shapes) identical
so one model / eval harness serves every complexity level:

* **graph size**    — treatment/covariate/latent active-count ranges (``*_active_range``)
* **interactions**  — per-edge-type arrow budgets (``edge_budget``), the "pot",
  plus the direct-null floor (``min_no_direct_effect_treatments``)
* **nonlinearity**  — treatment response family mix (``nonlinearity``)
* **signal / noise** — coefficient and noise ranges (via ``**overrides``)

The schema is pinned by ``n_treatments / n_covariates / n_latent``: inactive nodes are
zero-padded and masked, so a model trained at ``n_treatments=20`` sees the same slot
layout whether a task has 3 or 20 live treatments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior, _n_query
from .slots import TRAJECTORY_INPUTS
from .trajectories import min_on_weeks

#: Named input-trajectory archetypes accepted by ``make_scm_prior(trajectories=...)``.
#: Each makes every treatment and covariate carry the archetype's components.
TRAJECTORY_ARCHETYPES: tuple[str, ...] = (
    "always_on_spikes",
    "periodic_on_off",
    "delayed_start",
    "ramp_up",
    "decay_to_zero",
    "level_doubling",
    "seasonal",
    "trend",
)

#: Every accepted ``make_scm_prior(trajectories=...)`` value: the legacy
#: ``"texture"`` (default), the ``"composable"`` mixture, and each archetype.
TRAJECTORY_PRESETS: tuple[str, ...] = ("texture", "composable", *TRAJECTORY_ARCHETYPES)


def _week_frac(week: int, n_time_steps: int) -> float:
    """A fraction ``u`` with ``floor(u * n_time_steps) == week`` despite rounding."""
    return (week + 0.5) / n_time_steps


def _for_inputs(**fields: Any) -> dict[str, Any]:
    """Expand ``name=value`` to ``treatment_name`` and ``covariate_name``."""
    return {f"{x}_{name}": value for x in TRAJECTORY_INPUTS for name, value in fields.items()}


def _flighting_periods(window: int, duty_lo: float) -> tuple[int, int] | None:
    """The widest preset period range (within 4..13) keeping >= 2 on-weeks in ``window``."""
    p_hi = min(13, window // 2)
    while p_hi >= 2:
        p_lo = min(4, p_hi)
        if min_on_weeks(window, (p_lo, p_hi), duty_lo) >= 2:
            return (p_lo, p_hi)
        p_hi -= 1
    return None


def _trajectory_overrides_at(
    name: str, n_time_steps: Any, query_frac: Any
) -> dict[str, Any] | None:
    """The ``trajectories=name`` overrides at one horizon, or ``None`` if infeasible there.

    Gate windows are placed relative to the support prefix
    ``S = min(T - n_query, T // 2)`` that every split leaves as context, so each
    gated input keeps >= 2 on-weeks inside it (the rule ``SCMPrior.validate``
    enforces). ``n_query`` comes from the values exactly as given (numpy scalars
    included), as in ``SCMPrior.n_query``: their product can round differently
    from a Python-float one. Fractions are built as ``(week + 0.5) / T`` so the
    drawn weeks are exactly the intended ones.
    """
    T = int(n_time_steps)
    S = min(T - _n_query(n_time_steps, query_frac), T // 2)
    seasonal_period = float(max(4, min(52, T // 2)))

    def onset(lo_week: int, hi_week: int) -> dict[str, Any] | None:
        if lo_week > hi_week or hi_week > S - 2:
            return None
        return _for_inputs(
            onset_inclusion_prob=1.0,
            onset_frac_range=(_week_frac(lo_week, T), _week_frac(hi_week, T)),
        )

    def offset() -> dict[str, Any] | None:
        lo_week = max(S, int(round(0.6 * T)))
        hi_week = max(lo_week, min(T - 1, int(round(0.9 * T))))
        if lo_week > T - 1:
            return None
        return _for_inputs(offset_frac_range=(_week_frac(lo_week, T), _week_frac(hi_week, T)))

    out: dict[str, Any]
    if name == "texture":
        return {}
    if name == "always_on_spikes":
        return {
            **_for_inputs(hf_inclusion_prob=1.0, pulse_inclusion_prob=1.0),
            **_for_inputs(pulse_prob_range=(0.05, 0.25)),
        }
    if name == "periodic_on_off":
        periods = _flighting_periods(S, 0.3)
        if periods is None:
            return None
        return _for_inputs(
            flighting_inclusion_prob=1.0,
            flighting_period_weeks_range=periods,
            flighting_duty_range=(0.3, 0.7),
        )
    if name == "delayed_start":
        lo_week = max(1, min(int(round(0.1 * T)), S - 2))
        return onset(lo_week, max(lo_week, min(int(round(0.4 * T)), S - 2)))
    if name == "ramp_up":
        start = onset(1, max(1, min(int(round(0.15 * T)), S - 2)))
        if start is None:
            return None
        return {
            **start,
            **_for_inputs(trend_inclusion_prob=1.0),
            "treatment_trend_log_change_range": (1.0, 2.0),
            "covariate_trend_change_range": (1.0, 2.0),
        }
    if name == "decay_to_zero":
        stop = offset()
        if stop is None or S < 2:
            return None
        return {
            **stop,
            **_for_inputs(offset_inclusion_prob=1.0, trend_inclusion_prob=1.0),
            "treatment_trend_log_change_range": (-1.5, -0.5),
            "covariate_trend_change_range": (-1.5, -0.5),
        }
    if name == "level_doubling":
        # A treatment doubles exactly (factor 2). A signed covariate has no
        # multiplicative level, so it steps by +1 from a pinned level of 1, with
        # texture, walk scale and covariate parents quieted so the step reads
        # as a doubling of the level.
        return {
            **_for_inputs(level_jump_inclusion_prob=1.0, level_jump_count=1),
            "treatment_level_jump_factor_range": (2.0, 2.0),
            "covariate_level_jump_size_range": (1.0, 1.0),
            "rw_covariate_mean_range": (1.0, 1.0),
            "covariate_hf_inclusion_prob": 0.0,
            "covariate_pulse_inclusion_prob": 0.0,
            "rw_std_sigma": 0.05,
            "rw_baseline_std_sigma": 1.0,
            "dz_base_rate": 0.0,
            "zz_base_rate": 0.0,
        }
    if name == "seasonal":
        return {
            **_for_inputs(
                seasonal_inclusion_prob=1.0,
                seasonal_period_weeks_range=(seasonal_period, seasonal_period),
            ),
            "treatment_seasonal_amplitude_range": (0.3, 0.6),
            "covariate_seasonal_amplitude_range": (0.5, 1.0),
        }
    if name == "trend":
        return {
            **_for_inputs(trend_inclusion_prob=1.0),
            "treatment_trend_log_change_range": (-1.0, 1.0),
            "covariate_trend_change_range": (-1.0, 1.0),
        }
    if name == "composable":
        lo_week = max(1, int(round(0.05 * T)))
        hi_week = max(lo_week, S // 4)
        start, stop = onset(lo_week, hi_week), offset()
        periods = _flighting_periods(S - hi_week, 0.3)
        if start is None or stop is None or periods is None:
            return None
        start = {key: value for key, value in start.items() if "inclusion" not in key}
        out = {
            # Wider input level and variation priors than the texture preset.
            "rw_positive_mean_range": (0.1, 8.0),
            "rw_treatment_std_range": (0.1, 1.2),
            "rw_covariate_mean_range": (-3.0, 3.0),
            "rw_std_sigma": 1.5,
            # Pin the absolute-mode baseline walk to the legacy scale, which
            # rw_std_sigma would otherwise also widen.
            "rw_baseline_std_sigma": 1.0,
            **start,
            **stop,
            **_for_inputs(
                flighting_period_weeks_range=periods,
                flighting_duty_range=(0.3, 0.8),
                seasonal_period_weeks_range=(seasonal_period, seasonal_period),
            ),
        }
        for input_type, probs in (
            (
                "treatment",
                (0.9, 0.7, 0.15, 0.1, 0.2, 0.2, 0.3, 0.3),
            ),
            (
                "covariate",
                (0.9, 0.6, 0.1, 0.1, 0.15, 0.3, 0.35, 0.3),
            ),
        ):
            for component, p in zip(
                ("hf", "pulse", "onset", "offset", "flighting", "level_jump", "seasonal", "trend"),
                probs,
                strict=True,
            ):
                out[f"{input_type}_{component}_inclusion_prob"] = p
        return out
    raise ValueError(f"trajectories must be one of {TRAJECTORY_PRESETS}, got {name!r}")


#: How many longer horizons an infeasible preset searches for its "next fit" hint,
#: and how many complete-config validations that search may run.
_HINT_SEARCH_WEEKS = 100_000
_HINT_MAX_VALIDATIONS = 1_000


def _valid_horizon(n_time_steps: Any, query_frac: Any) -> bool:
    """Whether ``SCMPrior.validate`` accepts this horizon / query fraction pair."""
    n_query = _n_query(n_time_steps, query_frac)
    return int(n_time_steps) >= 4 and 0 < n_query <= int(n_time_steps) - 2


def _trajectory_overrides(name: Any, n_time_steps: Any, query_frac: Any) -> dict[str, Any] | None:
    """The ``trajectories=name`` overrides, or ``None`` when the preset cannot fit the horizon.

    An invalid horizon or query fraction derives nothing (``{}``): the config is
    built without the preset and :meth:`SCMPrior.validate` reports the problem.
    """
    if not isinstance(name, str) or name not in TRAJECTORY_PRESETS:
        raise ValueError(f"trajectories must be one of {TRAJECTORY_PRESETS}, got {name!r}")
    if name == "texture":
        return {}
    if (
        isinstance(n_time_steps, (bool, np.bool_))
        or not isinstance(n_time_steps, (int, np.integer))
        or isinstance(query_frac, (bool, np.bool_))
        or not isinstance(query_frac, (int, float, np.integer, np.floating))
        or not np.isfinite(query_frac)
        or not _valid_horizon(n_time_steps, query_frac)
    ):
        return {}
    # Both values stay as given (e.g. np.int64 / np.float32): SCMPrior computes
    # n_query from the raw values, and the gate windows must use the same n_query.
    return _trajectory_overrides_at(name, n_time_steps, query_frac)


def _infeasible_preset_error(
    name: str, base: dict[str, Any], overrides: dict[str, Any]
) -> ValueError:
    """The error for a preset that cannot fit the horizon, naming the next one that does.

    A candidate horizon counts only if the complete config — base fields, the
    preset at that horizon and the caller's overrides — passes validation.
    Candidates the preset itself cannot fit are rejected in microseconds; full
    validations are capped, so an override that is invalid at every horizon
    costs at most ``_HINT_MAX_VALIDATIONS`` of them and simply drops the hint.
    Candidates keep the caller's horizon type (e.g. ``np.int64``), because
    ``n_query`` rounds the product with ``query_frac`` in that type; a candidate
    the type cannot hold, or whose product overflows, ends the search.
    """
    raw_horizon = overrides.get("n_time_steps", SCMPrior.n_time_steps)
    T = int(raw_horizon)
    horizon_type = type(raw_horizon) if isinstance(raw_horizon, np.integer) else int
    query_frac = overrides.get("query_frac", SCMPrior.query_frac)
    validations = 0
    next_fit = None
    for t in range(T + 1, T + 1 + _HINT_SEARCH_WEEKS):
        try:
            candidate = horizon_type(t)
            # A narrow query_frac type (e.g. float16) can overflow on a long
            # candidate; that ends the probe, so its cast warning is noise.
            with np.errstate(over="ignore"):
                if not _valid_horizon(candidate, query_frac):
                    continue
        except OverflowError:
            break
        preset = _trajectory_overrides_at(name, candidate, query_frac)
        if preset is None:
            continue
        if validations == _HINT_MAX_VALIDATIONS:
            break
        validations += 1
        try:
            SCMPrior(**{**base, **preset, **overrides, "n_time_steps": candidate}).validate()
        except (TypeError, ValueError, OverflowError):
            continue
        next_fit = t
        break
    hint = f"; the next longer horizon that fits is n_time_steps={next_fit}" if next_fit else ""
    return ValueError(
        f"trajectories={name!r} cannot keep two on-weeks inside the support prefix at "
        f"n_time_steps={T}, query_frac={query_frac}{hint}"
    )


def _linear_family_probs(family_keys: tuple[str, ...]) -> dict[str, float]:
    """Return a fresh categorical distribution that selects family id zero."""
    return {family: 1.0 if index == 0 else 0.0 for index, family in enumerate(family_keys)}


#: Default treatment and covariate texture applied by ``make_scm_prior``.
#:
#: Treatments get iid weekly execution noise, campaign pulses, a floored uniform
#: walk-std (the legacy HalfNormal piles mass at 0 -> flat contribution
#: targets), a widened treatment-level range, and an carryover burn-in equal to
#: ``l_max`` so the zero-padding warmup never reaches the reported window. The
#: std/sigma/amp ranges are RELATIVE to each treatment's own level (scale-free,
#: like L1's log-space treatment noise); ranges are deliberately WIDE — the goal is
#: many different plausible worlds (near-smooth treatments through heavily pulsed
#: ones), not uniformly jagged series. Sized so the post-mechanism signal
#: survives: carryover low-passes the weekly noise (~2-3x std reduction) and
#: κ-relative saturation roughly halves relative variation at the knee, so
#: treatment CV must reach L1-like territory (~0.3-0.8) for contribution targets
#: to carry signal. Validate any retuning against
#: ``pymc_generator.signal_diagnostics.check_signal_gate`` on a freshly
#: generated corpus's ``diagnostics["signal"]`` block.
#:
#: Covariates get the same two high-frequency terms, RELATIVE to each covariate's
#: own walk std and with a CENTRED pulse. Without them a covariate is a smoothed
#: walk drawn from the same function class as the baseline walk, so ``Z->Y`` is
#: only weakly separable from baseline drift; the added high-frequency content
#: is what a smooth baseline cannot mimic (and what real promo / holiday /
#: price-step regressors look like). The ranges keep the diversity spread:
#: near-smooth seasonality at the low end through spiky calendars at the top.
_DIVERSE_TEXTURE: dict[str, Any] = {
    "rw_treatment_std_range": (0.15, 0.8),
    "rw_positive_mean_range": (0.3, 4.0),
    "treatment_hf_sigma_range": (0.08, 0.6),
    "treatment_pulse_prob_range": (0.0, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "covariate_hf_sigma_range": (0.1, 0.8),
    "covariate_pulse_prob_range": (0.0, 0.25),
    "covariate_pulse_amp_range": (0.5, 3.0),
}


def make_scm_prior(
    *,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    edge_budget: dict[str, int | tuple[int, int]] | None = None,
    n_treatments_active_range: tuple[int, int] | None = None,
    n_covariates_active_range: tuple[int, int] | None = None,
    n_latent_active_range: tuple[int, int] | None = None,
    nonlinearity: str = "diverse",
    trajectories: str = "texture",
    **overrides: Any,
) -> SCMPrior:
    """Build a validated additive-SCM :class:`SCMPrior` with a pinned max-layout.

    Parameters
    ----------
    n_treatments, n_covariates, n_latent : int
        Padded graph sizes — treatment treatments (the treatments/interventions),
        observed covariates, and hidden confounders. Pin these to hold the
        schema (and tensor shapes) fixed across complexity levels.
    edge_budget : dict, optional
        Per-edge-type arrow budget ("pot"), an "up to" cap: ``{"zc": 5}``
        places up to 5 covariate->treatment arrows over the eligible pairs (count
        drawn uniformly in ``{0..5}``, however they land); use ``{"zc": (5, 5)}``
        for exactly 5, or ``{"zc": (2, 5)}`` for a custom range. Each type's pot
        is independent — budgeting ``zc`` leaves ``zy`` (covariates' effect on the
        outcome) alone. Types omitted from the dict keep their Bernoulli base
        rate. See :class:`SCMPrior.edge_budget`.
        A ``cy`` budget alone cannot reserve an active treatment without a direct
        edge: its count is clamped to eligible slots. Pass
        ``min_no_direct_effect_treatments=1`` to cap the direct count at
        ``n_treatments_active - 1``. A reserved treatment can still affect outcome
        indirectly through another treatment.
    n_treatments_active_range, n_covariates_active_range, n_latent_active_range : tuple, optional
        Per-cell active-count ranges (the graph-size axis). Default to
        ``(size, size)`` (every node always active) so size is fixed unless you
        widen it.
    nonlinearity : {"diverse", "linear"}
        ``"linear"`` forces a purely linear treatment response (no carryover, no
        saturation) for the simplest additive graph; ``"diverse"`` keeps the
        full family mix from ``SCMPrior`` defaults.
        Treatment and covariate texture defaults are applied regardless of this
        choice. Configure their ranges through ``**overrides``.
    trajectories : str
        Input-trajectory preset, one of :data:`TRAJECTORY_PRESETS`.
        ``"texture"`` (default) keeps the legacy texture: every input carries
        hf noise and pulses, nothing else, and worlds are unchanged.
        ``"composable"`` lets every input carry every component of
        :data:`pymc_generator.slots.TRAJECTORY_COMPONENTS` independently, at
        moderate inclusion probabilities, and widens the input level and
        variation priors (``rw_positive_mean_range``, ``rw_treatment_std_range``,
        ``rw_covariate_mean_range``, ``rw_std_sigma``). Each name in
        :data:`TRAJECTORY_ARCHETYPES` makes every input follow that archetype.
        Gate windows and flighting periods are derived from the effective
        ``n_time_steps`` / ``query_frac`` so every gated input keeps two
        on-weeks of context. A horizon too short for that raises ``ValueError``
        before ``**overrides`` apply (they cannot rescue it — set the trajectory
        knobs directly instead), naming the next longer horizon at which the
        complete config validates (searching at most 100,000 longer horizons and
        1,000 full validations). Any error the config raises at the requested
        horizon without the preset is attached as the exception's ``__cause__``.
    **overrides
        Any other :class:`SCMPrior` field, passed straight to its constructor.
        For example, ``n_time_steps``, ``n_cells``, ``seed``,
        ``l_max``, any ``*_coeff_range``, ``rw_baseline_std_range`` /
        ``rw_outcome_std_range`` for the default relative outcome-noise axis,
        ``outcome_std_mode="absolute"`` with ``rw_outcome_std_sigma`` for the
        legacy absolute scale axis, or ``prior_conditioning=True`` to enable
        the ACE prior-conditioning hyperprior (per-cell narrowed prior
        intervals, recorded in the corpus ``prior_cond`` key).

        Precedence, in application order: this function's own defaults (the
        pinned ``*_active_range`` values and ``edge_budget``), then the
        ``nonlinearity="linear"`` family probabilities, then the default texture
        ranges (:data:`_DIVERSE_TEXTURE`), then ``carryover_burn_in``, then the
        ``trajectories`` preset, then ``**overrides``. So an override wins over
        every one of them — including
        the texture ranges (``treatment_hf_sigma_range=(0.0, 0.0)`` disables the
        treatment jitter the preset just enabled) and the burn-in
        (``carryover_burn_in=0`` turns it off even though the preset pinned
        ``l_max``). ``nonlinearity="linear"`` sets only
        ``carryover_family_probs`` / ``saturation_family_probs``, so overriding
        one of those two leaves the OTHER forced to its identity family —
        pass ``nonlinearity="diverse"`` rather than fighting the flag.

        An unrecognized key is a hard error, not a silent no-op:
        ``SCMPrior.__init__`` raises ``TypeError: SCMPrior.__init__() got an
        unexpected keyword argument '<key>'``. A recognized key with an invalid
        value raises ``ValueError`` from :meth:`SCMPrior.validate`, which this
        function always calls before returning.

    Returns
    -------
    SCMPrior
        A validated additive-SCM config.
    """
    if nonlinearity not in ("diverse", "linear"):
        raise ValueError(f"nonlinearity must be 'diverse' or 'linear', got {nonlinearity!r}")

    kwargs: dict[str, Any] = {
        "n_treatments": n_treatments,
        "n_covariates": n_covariates,
        "n_latent": n_latent,
        "n_treatments_active_range": (
            n_treatments_active_range
            if n_treatments_active_range is not None
            else (n_treatments, n_treatments)
        ),
        "n_covariates_active_range": (
            n_covariates_active_range
            if n_covariates_active_range is not None
            else (n_covariates, n_covariates)
        ),
        "n_latent_active_range": (
            n_latent_active_range if n_latent_active_range is not None else (n_latent, n_latent)
        ),
        "edge_budget": edge_budget,
    }
    if nonlinearity == "linear":
        kwargs["carryover_family_probs"] = _linear_family_probs(CARRYOVER_FAMILY_KEYS)
        kwargs["saturation_family_probs"] = _linear_family_probs(SATURATION_FAMILY_KEYS)
    kwargs.update(_DIVERSE_TEXTURE)
    # burn-in follows the (possibly overridden) carryover length
    kwargs["carryover_burn_in"] = int(overrides.get("l_max", SCMPrior.l_max))
    preset = _trajectory_overrides(
        trajectories,
        overrides.get("n_time_steps", SCMPrior.n_time_steps),
        overrides.get("query_frac", SCMPrior.query_frac),
    )
    if preset is None:
        # Raised before the overrides apply: they cannot rescue a preset that has
        # no gate windows at this horizon. Set the trajectory knobs directly instead.
        error = _infeasible_preset_error(trajectories, kwargs, overrides)
        # Surface any problem the caller's own fields have at this horizon as the
        # cause, so an invalid override is not hidden behind the preset error.
        try:
            SCMPrior(**{**kwargs, **overrides}).validate()
        except (TypeError, ValueError) as cause:
            raise error from cause
        raise error
    kwargs.update(preset)

    kwargs.update(overrides)  # caller's explicit fields win
    cfg = SCMPrior(**kwargs)
    cfg.validate()
    return cfg
