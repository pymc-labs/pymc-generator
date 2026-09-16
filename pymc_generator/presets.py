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

from .sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior


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
        ranges (:data:`_DIVERSE_TEXTURE`), then ``carryover_burn_in``, then
        ``**overrides``. So an override wins over every one of them — including
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

    kwargs.update(overrides)  # caller's explicit fields win
    cfg = SCMPrior(**kwargs)
    cfg.validate()
    return cfg
