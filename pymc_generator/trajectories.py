"""Composable per-input trajectory components.

Every input series — each treatment ``C_k`` and each observed covariate ``Z_m`` —
can carry, independently per input, any subset of eight components on top of its
smoothed random-walk own drive (:data:`pymc_generator.slots.TRAJECTORY_COMPONENTS`):

============== ==============================================================
``hf``         iid weekly noise (the existing high-frequency texture)
``pulse``      Bernoulli pulses (the existing pulse texture)
``onset``      delayed start: off before a launch week, burn-in included
``offset``     go to zero: off from a stop week on
``flighting``  periodic on/off: on for ``W`` of every ``P`` weeks
``level_jump`` held level steps at stratified weeks
``seasonal``   a sinusoid with its own period and phase
``trend``      a linear drift through zero at reported week 0
============== ==============================================================

The three gate forms (``onset``, ``offset``, ``flighting``) multiply into one
activity ``a_i(t) ∈ {0, 1}``; the three level forms add into one level shift.
A covariate is signed and linear, so its components add on its own scale::

    Z_m(t) = a_m(t) · (D→Z + Z→Z + walk + [hf] + [pulse] + [seasonal] + [trend] + [jumps])

A treatment is non-negative, so its level components compose in log-level — a
sum of components on the log scale, a product on the series — and its gate
multiplies the whole activated input::

    C_k(t) = a_k(t) · softplus(D→C + Z→C + C→C + walk + [hf] + [pulse])
                    · exp([seasonal] + [trend]) · Π_e [jump factor f_e after τ_e]

so ``C_k ≥ 0`` always, ``C_k == 0.0`` exactly on off-weeks (a treatment shock,
applied after the gate, holds its own level through them), and a jump factor of
2 doubles the series exactly.

Which components an input carries is concrete per-cell structure, drawn by
:func:`sample_component_flags` like the carryover and saturation families. A
component an input does not carry is never wired for it: it contributes exactly
nothing to that input. Gate and level parameters (launch weeks, periods,
phases, amplitudes, factors) are PyMC random variables, one entry per input,
created by :func:`trajectory_params` only when at least one input of the cell
carries the component, and only for non-degenerate ranges. The ``hf`` / ``pulse``
texture variables exist whenever their range is live, as before, and reach only
the inputs that carry them.

This module imports numpy only; PyTensor and PyMC load on first graph use, so
:mod:`pymc_generator.sampler` can import it without pulling either in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS

if TYPE_CHECKING:
    from pytensor.tensor import TensorVariable

    from .sampler import SCMPrior

__all__ = [
    "GATE_COMPONENTS",
    "LEVEL_COMPONENTS",
    "SCHEDULE_COMPONENTS",
    "TEXTURE_COMPONENTS",
    "activity_column",
    "flighting_on_weeks",
    "level_shift_column",
    "min_on_weeks",
    "sample_component_flags",
    "structural_key",
    "summarize_component_prevalence",
    "time_index",
    "trajectory_params",
    "treatment_multiplier_column",
]

#: The existing texture terms, now included per input with a probability.
TEXTURE_COMPONENTS: tuple[str, ...] = ("hf", "pulse")
#: Gate forms; their product is the input's activity ``a_i(t)``.
GATE_COMPONENTS: tuple[str, ...] = ("onset", "offset", "flighting")
#: Level forms; their sum is the input's level (treatments: log-level) shift.
LEVEL_COMPONENTS: tuple[str, ...] = ("level_jump", "seasonal", "trend")
#: Components that change the structural graph beyond the legacy texture.
SCHEDULE_COMPONENTS: tuple[str, ...] = GATE_COMPONENTS + LEVEL_COMPONENTS


def structural_key(input_type: str, component: str) -> str:
    """The :func:`~pymc_generator.world_model.sample_structure` key of one flag vector.

    Treatment flags keep the legacy ``use_hf`` / ``use_pulse`` spelling, covariate
    flags the legacy ``use_covariate_*`` one.
    """
    if input_type == "treatment":
        return f"use_{component}"
    if input_type == "covariate":
        return f"use_covariate_{component}"
    raise ValueError(f"input_type must be one of {TRAJECTORY_INPUTS}, got {input_type!r}")


def sample_component_flags(
    cfg: SCMPrior, n_treatments: int, n_covariates: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw which inputs carry which components, as boolean vectors.

    Each (input type, component) pair is an independent Bernoulli draw per input
    with the config's EFFECTIVE inclusion probability
    (:meth:`~pymc_generator.sampler.SCMPrior.trajectory_inclusion_probs`).
    Probabilities of exactly 0 or 1 draw nothing; the legacy texture flags
    therefore come out exactly as before (``np.full(n, enabled)``).

    Fractional probabilities draw from child streams spawned off ``rng``, one per
    (input type, component) in the fixed ``TRAJECTORY_INPUTS ×
    TRAJECTORY_COMPONENTS`` order. Spawning leaves ``rng``'s bit-generator state
    untouched (it only advances its seed sequence's child counter), so no other
    draw made from ``rng`` moves, and each component's flags are independent of
    every other component's probability. Schedule components (gates and level
    forms) only get keys when the config enables any of them.
    """
    probs = cfg.trajectory_inclusion_probs()
    components = TRAJECTORY_COMPONENTS if cfg.trajectory_components_enabled else TEXTURE_COMPONENTS
    sizes = {"treatment": int(n_treatments), "covariate": int(n_covariates)}
    fractional = any(
        0.0 < probs[input_type][component] < 1.0
        for input_type in TRAJECTORY_INPUTS
        for component in components
    )
    streams = (
        rng.spawn(1)[0].spawn(len(TRAJECTORY_INPUTS) * len(TRAJECTORY_COMPONENTS))
        if fractional
        else None
    )
    flags: dict[str, np.ndarray] = {}
    for a, input_type in enumerate(TRAJECTORY_INPUTS):
        n = sizes[input_type]
        for b, component in enumerate(TRAJECTORY_COMPONENTS):
            if component not in components:
                continue
            p = probs[input_type][component]
            if p >= 1.0:
                value = np.full(n, True)
            elif p <= 0.0:
                value = np.full(n, False)
            else:
                assert streams is not None
                value = streams[a * len(TRAJECTORY_COMPONENTS) + b].random(n) < p
            flags[structural_key(input_type, component)] = value
    return flags


def flighting_on_weeks(duty: Any, period: Any) -> np.ndarray:
    """On-run length ``W = clip(floor(duty·P + 0.5), 1, P − 1)`` of a flighting gate.

    Round half up keeps ``W`` monotone in ``duty``; the ``P − 1`` cap means every
    flighting input switches off at least once per period, so a carried
    flighting component is never an always-on gate in disguise. This is the
    numpy twin of the graph formula in :func:`_input_params`, shared by
    validation and the presets.
    """
    period = np.asarray(period, dtype="int64")
    raw = np.floor(np.asarray(duty, dtype="float64") * period + 0.5).astype("int64")
    return np.asarray(np.clip(raw, 1, period - 1), dtype="int64")


def min_on_weeks(window: int, period_range: tuple[int, int] | None, duty_lo: float) -> int:
    """Fewest on-weeks any draw can leave inside ``window`` consecutive open weeks.

    ``window`` is what an onset/offset pair leaves open. Without flighting
    (``period_range=None``) every week of it is on. With flighting, the worst case
    over every phase of a period ``P`` with on-run ``W`` is
    ``floor(n/P)·W + max(0, n mod P − (P − W))``; ``W`` is smallest at
    ``duty_lo`` and the count is not monotone in ``P``, so every admitted period
    is checked.
    """
    n = int(window)
    if n <= 0:
        return 0
    if period_range is None:
        return n
    p_lo, p_hi = (int(v) for v in period_range)
    worst = n
    for period in range(p_lo, p_hi + 1):
        w = int(flighting_on_weeks(duty_lo, period))
        worst = min(worst, (n // period) * w + max(0, n % period - (period - w)))
    return worst


def _floor_int(x: Any) -> TensorVariable:
    import pytensor.tensor as pt

    return pt.cast(pt.floor(x), "int64")


def _constant(value: float | int, shape, dtype: str) -> TensorVariable:
    import pytensor.tensor as pt

    return pt.as_tensor_variable(np.full(shape, value, dtype=dtype))


def _input_params(
    cfg: SCMPrior,
    structural: dict,
    input_type: str,
    n: int,
    n_time_steps: int,
    *,
    dynamic_flags: bool,
) -> dict[str, Any]:
    """One input type's trajectory spec, with only admitted components wired."""
    import pymc as pm

    from .world_model import _uniform

    prefix = input_type
    if dynamic_flags:
        import pytensor.tensor as pt

        use: dict[str, Any] = {
            component: pt.as_tensor_variable(
                structural[structural_key(input_type, component)]
            ).reshape((n,))
            for component in TRAJECTORY_COMPONENTS
        }
        probs = cfg.trajectory_inclusion_probs()[input_type]
        enabled = {component: n > 0 and probs[component] > 0.0 for component in SCHEDULE_COMPONENTS}
    else:
        use = {
            component: np.asarray(
                structural[structural_key(input_type, component)], dtype=bool
            ).reshape(n)
            for component in TRAJECTORY_COMPONENTS
        }
        enabled = {component: use[component].any() for component in SCHEDULE_COMPONENTS}
    spec: dict[str, Any] = {"use": use}
    T = int(n_time_steps)

    if enabled["onset"]:
        lo, hi = getattr(cfg, f"{prefix}_onset_frac_range")
        spec["onset"] = {"start": _floor_int(_uniform(f"{prefix}_onset_frac", lo, hi, n) * T)}
    if enabled["offset"]:
        lo, hi = getattr(cfg, f"{prefix}_offset_frac_range")
        spec["offset"] = {"stop": _floor_int(_uniform(f"{prefix}_offset_frac", lo, hi, n) * T)}
    if enabled["flighting"]:
        import pytensor.tensor as pt

        p_lo, p_hi = (int(v) for v in getattr(cfg, f"{prefix}_flighting_period_weeks_range"))
        if p_lo == p_hi:
            period = _constant(p_lo, n, "int64")
        else:
            period = pm.DiscreteUniform(f"{prefix}_flighting_period", p_lo, p_hi, shape=n)
        d_lo, d_hi = getattr(cfg, f"{prefix}_flighting_duty_range")
        duty = _uniform(f"{prefix}_flighting_duty", d_lo, d_hi, n)
        # Round half up, capped at P - 1: the graph twin of flighting_on_weeks.
        on_weeks = pt.clip(_floor_int(duty * period + 0.5), 1, period - 1)
        phase_u = pm.Uniform(f"{prefix}_flighting_phase_u", 0.0, 1.0, shape=n)
        phase = pt.minimum(_floor_int(phase_u * period), period - 1)
        spec["flighting"] = {"period": period, "on_weeks": on_weeks, "phase": phase}
    if enabled["level_jump"]:
        import pytensor.tensor as pt

        count = int(getattr(cfg, f"{prefix}_level_jump_count"))
        # Event e lies in the e-th of `count` disjoint slots of weeks [1, T).
        edges = np.asarray([1 + e * (T - 1) // count for e in range(count + 1)], dtype="int64")
        slot_lo, width = edges[:-1], np.diff(edges)
        if (width == 1).all():
            week = _constant(0, (count, n), "int64") + slot_lo[:, None]
        else:
            u = pm.Uniform(f"{prefix}_level_jump_u", 0.0, 1.0, shape=(count, n))
            week = pt.minimum(
                slot_lo[:, None] + _floor_int(u * width[:, None]), (slot_lo + width - 1)[:, None]
            )
        jump: dict[str, Any] = {"count": count, "week": week}
        if input_type == "treatment":
            f_lo, f_hi = (float(v) for v in cfg.treatment_level_jump_factor_range)
            if f_lo == f_hi:
                # Multiply by the factor itself: an exact x2 must not depend on
                # exp(log(2)) rounding back to 2.
                jump["factor"] = _constant(f_lo, (count, n), "float64")
                jump["log_factor"] = _constant(float(np.log(f_lo)), (count, n), "float64")
            else:
                log_factor = pm.Uniform(
                    "treatment_level_jump_log_factor",
                    float(np.log(f_lo)),
                    float(np.log(f_hi)),
                    shape=(count, n),
                )
                jump["factor"] = pt.exp(log_factor)
                jump["log_factor"] = log_factor
        else:
            lo, hi = cfg.covariate_level_jump_size_range
            jump["size"] = _uniform("covariate_level_jump_size", lo, hi, (count, n))
        spec["level_jump"] = jump
    if enabled["seasonal"]:
        a_lo, a_hi = getattr(cfg, f"{prefix}_seasonal_amplitude_range")
        p_lo, p_hi = getattr(cfg, f"{prefix}_seasonal_period_weeks_range")
        spec["seasonal"] = {
            "amplitude": _uniform(f"{prefix}_seasonal_amplitude", a_lo, a_hi, n),
            "period": _uniform(f"{prefix}_seasonal_period", p_lo, p_hi, n),
            "phase": pm.Uniform(f"{prefix}_seasonal_phase", 0.0, 2.0 * np.pi, shape=n),
        }
    if enabled["trend"]:
        lo, hi = (
            cfg.treatment_trend_log_change_range
            if input_type == "treatment"
            else cfg.covariate_trend_change_range
        )
        spec["trend"] = {"change": _uniform(f"{prefix}_trend_change", lo, hi, n)}
    return spec


def trajectory_params(
    cfg: SCMPrior,
    structural: dict,
    n_treatments: int,
    n_covariates: int,
    n_time_steps: int,
    *,
    dynamic_flags: bool = False,
) -> dict[str, dict[str, Any]] | None:
    """The cell's trajectory specs, or ``None`` when no schedule component is enabled.

    Must be called inside a ``pm.Model``. Returns ``{"treatment": spec,
    "covariate": spec}`` where each spec holds the per-input ``"use"`` flags and,
    for every component at least one input carries, its realised parameters:

    * ``"onset"``: ``{"start"}`` — int launch weeks;
    * ``"offset"``: ``{"stop"}`` — int stop weeks;
    * ``"flighting"``: ``{"period", "on_weeks", "phase"}`` — int weeks;
    * ``"level_jump"``: ``{"count", "week"}`` plus ``{"factor", "log_factor"}``
      (treatments) or ``{"size"}`` (covariates), each ``(count, n)``;
    * ``"seasonal"``: ``{"amplitude", "period", "phase"}``;
    * ``"trend"``: ``{"change"}`` — total change across the reported window
      (treatments: in log-level).

    ``dynamic_flags=True`` admits components from the config rather than one
    cell's realised flags. The ``"use"`` vectors may then be symbolic inputs,
    letting a padded template change inclusion without rebuilding its priors.

    :func:`pymc_generator.symbolic_graph.build_symbolic_graph` also accepts
    the same layout with concrete numpy values.
    """
    if not cfg.trajectory_components_enabled:
        return None
    return {
        "treatment": _input_params(
            cfg, structural, "treatment", n_treatments, n_time_steps, dynamic_flags=dynamic_flags
        ),
        "covariate": _input_params(
            cfg, structural, "covariate", n_covariates, n_time_steps, dynamic_flags=dynamic_flags
        ),
    }


def time_index(n_time_steps: int, burn_in: int) -> np.ndarray:
    """Week index over the simulated horizon: reported week 0 is 0, burn-in is negative."""
    return np.arange(-int(burn_in), int(n_time_steps), dtype="int64")


def _uses_component(spec: dict[str, Any], component: str, i: int) -> bool:
    """Whether this component must be built for the input, before runtime selection."""
    from pytensor.tensor import TensorVariable

    flags = spec["use"][component]
    return component in spec if isinstance(flags, TensorVariable) else bool(flags[i])


def _selected_component(spec: dict[str, Any], component: str, i: int, value, neutral):
    """Select a symbolic component flag without adding switches to static worlds."""
    import pytensor.tensor as pt
    from pytensor.tensor import TensorVariable

    flags = spec["use"][component]
    return (
        pt.switch(pt.neq(flags[i], 0), value, np.asarray(neutral))
        if isinstance(flags, TensorVariable)
        else value
    )


def activity_column(spec: dict[str, Any], i: int, t: np.ndarray) -> TensorVariable | None:
    """Input ``i``'s activity over the full horizon, or ``None`` when it carries no gate.

    The product of the gate forms the input carries: ``onset`` is off for every
    week before its launch week (burn-in included), ``offset`` from its stop week
    on, and ``flighting`` outside the first ``on_weeks`` of every ``period``
    weeks, shifted by ``phase``.
    """
    import pytensor.tensor as pt

    factors = []
    if _uses_component(spec, "onset", i):
        gate = pt.ge(t, spec["onset"]["start"][i])
        factors.append(_selected_component(spec, "onset", i, gate, True))
    if _uses_component(spec, "offset", i):
        gate = pt.lt(t, spec["offset"]["stop"][i])
        factors.append(_selected_component(spec, "offset", i, gate, True))
    if _uses_component(spec, "flighting", i):
        flighting = spec["flighting"]
        cycle = pt.mod(t + flighting["phase"][i], flighting["period"][i])
        gate = pt.lt(cycle, flighting["on_weeks"][i])
        factors.append(_selected_component(spec, "flighting", i, gate, True))
    if not factors:
        return None
    activity = factors[0]
    for factor in factors[1:]:
        activity = pt.and_(activity, factor)
    return cast("TensorVariable", activity)


def _smooth_terms(spec: dict[str, Any], i: int, t: np.ndarray, n_time_steps: int) -> list:
    import pytensor.tensor as pt

    weeks = t.astype("float64")
    terms = []
    if _uses_component(spec, "seasonal", i):
        seasonal = spec["seasonal"]
        value = seasonal["amplitude"][i] * pt.sin(
            2.0 * np.pi * weeks / seasonal["period"][i] + seasonal["phase"][i]
        )
        terms.append(_selected_component(spec, "seasonal", i, value, 0.0))
    if _uses_component(spec, "trend", i):
        # Zero through burn-in and week 0, the full change at the last week.
        ramp = np.maximum(t, 0).astype("float64") / float(n_time_steps - 1)
        value = spec["trend"]["change"][i] * ramp
        terms.append(_selected_component(spec, "trend", i, value, 0.0))
    return terms


def _sum(terms: list) -> TensorVariable:
    import pytensor.tensor as pt

    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return pt.as_tensor_variable(total)


def _jump_steps(jump: dict[str, Any], i: int, t: np.ndarray) -> TensorVariable:
    """``(T_full, K)`` booleans: whether each week is at or after each of input ``i``'s jumps.

    One matrix rather than K separate terms: PyTensor would fuse K terms into a
    single elementwise op, and the FAST_COMPILE Python backend refuses more than
    32 operands (the reason :func:`pymc_generator.symbolic_graph._dot_terms`
    uses a dot product too).
    """
    import pytensor.tensor as pt

    weeks = pt.as_tensor_variable(jump["week"])[:, i]
    return cast("TensorVariable", pt.ge(pt.as_tensor_variable(t)[:, None], weeks[None, :]))


def level_shift_column(
    spec: dict[str, Any], i: int, t: np.ndarray, n_time_steps: int, *, log_level: bool
) -> TensorVariable | None:
    """Input ``i``'s summed level shift over the full horizon, or ``None`` when it has none.

    ``seasonal + trend + Σ_e size_e · 1[t ≥ week_e]``. For treatments
    (``log_level=True``) the jump size is ``log(factor)``, so the result is the
    log of the treatment's level multiplier.
    """
    import pytensor.tensor as pt

    terms = _smooth_terms(spec, i, t, n_time_steps)
    if _uses_component(spec, "level_jump", i):
        jump = spec["level_jump"]
        size = pt.as_tensor_variable(jump["log_factor"] if log_level else jump["size"])
        value = pt.dot(pt.cast(_jump_steps(jump, i, t), "float64"), size[:, i])
        terms.append(_selected_component(spec, "level_jump", i, value, 0.0))
    return _sum(terms) if terms else None


def treatment_multiplier_column(
    spec: dict[str, Any], i: int, t: np.ndarray, n_time_steps: int
) -> TensorVariable | None:
    """Treatment ``i``'s level multiplier, ``exp(seasonal + trend) · Π_e f_e^{1[t ≥ week_e]}``.

    Jump factors multiply directly rather than through ``exp(log f)``, so a
    factor of 2 doubles the series exactly. ``None`` when the treatment carries
    no level component.
    """
    import pytensor.tensor as pt

    parts = []
    smooth = _smooth_terms(spec, i, t, n_time_steps)
    if smooth:
        parts.append(pt.exp(_sum(smooth)))
    if _uses_component(spec, "level_jump", i):
        jump = spec["level_jump"]
        factor = pt.as_tensor_variable(jump["factor"])[:, i]
        value = pt.prod(pt.switch(_jump_steps(jump, i, t), factor[None, :], 1.0), axis=1)
        parts.append(_selected_component(spec, "level_jump", i, value, 1.0))
    if not parts:
        return None
    multiplier = parts[0]
    for part in parts[1:]:
        multiplier = multiplier * part
    return pt.as_tensor_variable(multiplier)


def summarize_component_prevalence(
    treatment_components: np.ndarray,
    covariate_components: np.ndarray,
    treatment_active_mask: np.ndarray,
    covariate_active_mask: np.ndarray,
) -> dict[str, Any]:
    """Realised prevalence of every component over active (task, input) pairs.

    ``prevalence[input][component]`` is the fraction of active input slots
    (``*_active_mask == 1``) whose stored flag is 1; ``n_inputs[input]`` is that
    denominator. Padded slots never count. Values are plain Python ``float`` /
    ``int`` so the block round-trips through the corpus JSON metadata exactly.
    """
    prevalence: dict[str, dict[str, float]] = {}
    n_inputs: dict[str, int] = {}
    for input_type, flags, active in (
        ("treatment", treatment_components, treatment_active_mask),
        ("covariate", covariate_components, covariate_active_mask),
    ):
        present = np.asarray(active) == 1
        count = int(present.sum())
        carried = np.asarray(flags)[present]
        n_inputs[input_type] = count
        prevalence[input_type] = {
            component: (float(carried[:, c].sum()) / count if count else 0.0)
            for c, component in enumerate(TRAJECTORY_COMPONENTS)
        }
    return {"prevalence": prevalence, "n_inputs": n_inputs}
