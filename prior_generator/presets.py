"""Complexity presets for the additive-SCM corpus.

The additive SCM fixes the *structure* — the 8-block extended edge layout
(cy, dc, dz, db, zb, zc, cc, zz), the additive structural equations, and the
``indirect_effects`` outputs. This module dials *complexity* within that fixed
structure along orthogonal axes, keeping the schema (tensor shapes) identical
so one model / eval harness serves every complexity level:

* **graph size**    — active K/M/J ranges (``*_active_range``)
* **interactions**  — per-edge-type arrow budgets (``edge_budget``), the "pot"
* **nonlinearity**  — media response family mix (``nonlinearity``)
* **signal / noise** — coefficient and noise ranges (via ``**overrides``)

The schema is pinned by ``K_max / M_max / J_max``: inactive nodes are
zero-padded and masked, so a model trained at ``K_max=20`` sees the same slot
layout whether a task has 3 or 20 live channels.
"""

from __future__ import annotations

from typing import Any

from .sampler import CorpusConfig

# Media-response family mixes for the ``nonlinearity`` axis.
# Adstock:    (none, geometric, weibull)
# Saturation: (none/linear, hill, logistic, michaelis_menten, tanh, root)
_LINEAR_ADSTOCK: tuple[float, ...] = (1.0, 0.0, 0.0)
_LINEAR_SATURATION: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


#: Channel-texture prior for ``texture="diverse"``: iid weekly execution noise,
#: campaign pulses, a floored uniform walk-std (the legacy HalfNormal piles
#: mass at 0 -> flat contribution targets), a widened channel-level range, and
#: an adstock burn-in equal to ``l_max`` so the zero-padding warmup never
#: reaches the reported window. The std/sigma/amp ranges are RELATIVE to each
#: channel's own level (scale-free, like L1's log-space spend noise); ranges
#: are deliberately WIDE — the goal is many different plausible worlds
#: (near-smooth channels through heavily pulsed ones), not uniformly jagged
#: series. Sized so the post-mechanism signal survives: adstock low-passes the
#: weekly noise (~2-3x std reduction) and κ-relative saturation roughly halves
#: relative variation at the knee, so channel CV must reach L1-like territory
#: (~0.3-0.8) for contribution targets to carry signal. Validate any retuning
#: against ``prior_generator.signal_diagnostics.check_signal_gate`` on a
#: freshly generated corpus's ``diagnostics["signal"]`` block.
_DIVERSE_TEXTURE: dict[str, Any] = {
    "rw_channel_std_range": (0.15, 0.8),
    "rw_positive_mean_range": (0.3, 4.0),
    "channel_hf_sigma_range": (0.08, 0.6),
    "channel_pulse_prob_range": (0.0, 0.25),
    "channel_pulse_amp_range": (0.4, 2.5),
}


def make_world_config(
    *,
    K_max: int,
    M_max: int,
    J_max: int,
    edge_budget: dict[str, int | tuple[int, int]] | None = None,
    K_active_range: tuple[int, int] | None = None,
    M_active_range: tuple[int, int] | None = None,
    J_active_range: tuple[int, int] | None = None,
    nonlinearity: str = "diverse",
    texture: str = "diverse",
    **overrides: Any,
) -> CorpusConfig:
    """Build a validated additive-SCM :class:`CorpusConfig` with a pinned max-layout.

    Parameters
    ----------
    K_max, M_max, J_max : int
        Padded layout sizes — pin these to hold the schema (and tensor shapes)
        fixed across complexity levels.
    edge_budget : dict, optional
        Per-edge-type arrow budget ("pot"), an "up to N" cap: ``{"zc": 5}``
        places up to 5 control->channel arrows over the eligible pairs (count
        drawn uniformly in ``{0..5}``, however they land); use ``{"zc": (5, 5)}``
        for exactly 5, or ``{"zc": (2, 5)}`` for a custom range. Each type's pot
        is independent — budgeting ``zc`` leaves ``zb`` (controls' effect on the
        outcome) alone. Types omitted from the dict keep their Bernoulli base
        rate. See :class:`CorpusConfig.edge_budget`.
    K_active_range, M_active_range, J_active_range : tuple, optional
        Active-count ranges (the graph-size axis). Default to ``(max, max)``
        (every node always active) so size is fixed unless you widen it.
    nonlinearity : {"diverse", "linear"}
        ``"linear"`` forces a purely linear media response (no adstock, no
        saturation) for the simplest additive graph; ``"diverse"`` keeps the
        full family mix from ``CorpusConfig`` defaults.
    texture : {"diverse"}
        Channel-texture axis. ``"diverse"`` (the only supported value) gives
        channels high-frequency exogenous drive — iid weekly noise, campaign
        pulses, a floored RELATIVE walk std, and a widened channel-level range
        (``rw_positive_mean_range``) — plus an ``l_max`` adstock burn-in, so
        spend sweeps its response curve and contribution targets carry signal
        (:data:`_DIVERSE_TEXTURE`). The relative factors anchor on
        ``softplus(walk mean)``; heavy pulses raise the realized channel level
        above that anchor (up to ~1.7x at the range top), so realized CVs run
        somewhat below the drawn factors — retune against the
        :mod:`prior_generator.signal_diagnostics` gate, not the raw ranges.
        The deprecated ``"legacy"`` (smooth-walk-only) texture from
        structural-pfn was not migrated; reproducing pre-fix corpora requires
        structural-pfn itself.
    **overrides
        Any other :class:`CorpusConfig` field (e.g. ``T``, ``n_cells``,
        ``rw_sales_std_sigma`` for the noise axis, coefficient ranges).
        Explicit overrides win over the values set here.

    Returns
    -------
    CorpusConfig
        A validated additive-SCM config.
    """
    if nonlinearity not in ("diverse", "linear"):
        raise ValueError(f"nonlinearity must be 'diverse' or 'linear', got {nonlinearity!r}")
    if texture != "diverse":
        raise ValueError(
            f"texture must be 'diverse', got {texture!r}. The deprecated 'legacy' "
            f"texture was not migrated from structural-pfn."
        )

    kwargs: dict[str, Any] = {
        "K": K_max,
        "M": M_max,
        "J": J_max,
        "K_max": K_max,
        "M_max": M_max,
        "J_max": J_max,
        "K_active_range": K_active_range if K_active_range is not None else (K_max, K_max),
        "M_active_range": M_active_range if M_active_range is not None else (M_max, M_max),
        "J_active_range": J_active_range if J_active_range is not None else (J_max, J_max),
        "edge_budget": edge_budget,
    }
    if nonlinearity == "linear":
        kwargs["adstock_family_probs"] = _LINEAR_ADSTOCK
        kwargs["saturation_family_probs"] = _LINEAR_SATURATION
    kwargs.update(_DIVERSE_TEXTURE)
    # burn-in follows the (possibly overridden) adstock length
    kwargs["adstock_burn_in"] = int(overrides.get("l_max", CorpusConfig.l_max))

    kwargs.update(overrides)  # caller's explicit fields win
    cfg = CorpusConfig(**kwargs)
    cfg.validate()
    return cfg
