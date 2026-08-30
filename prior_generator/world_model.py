"""The world as a PyMC model: priors are distributions, worlds are draws.

``build_world_model`` assembles one :class:`pymc.Model` for a fixed causal
structure (a DAG ``g`` plus the concrete per-channel mechanism families and
per-node random-walk smoothness) in which every *continuous* SCM parameter is a
PyMC distribution and every noise term is an RV — ``pm.Normal`` walk innovations
and iid outcome/channel jitter, with campaign pulses as ``pm.Bernoulli``. The
structural equations and the exact interventional decomposition are the SAME
ones :func:`prior_generator.symbolic_graph.build_symbolic_graph` builds; this
module only supplies the priors + noise and exposes the outputs as
``pm.Deterministic`` so a single ``pm.draw`` yields params, series, and the full
decomposition jointly (and reproducibly from a seed).

Discrete/structural choices — which edges exist, each channel's adstock and
saturation family, and each random-walk node's smoothness — are drawn concretely
per world by :func:`sample_structure` (they set the graph's shape), matching the
design: continuous priors are distributions; structure is drawn per world.

:func:`build_oracle_model` is the observed-data variant: the same structure
and the same prior definitions with the world's dataset attached, so
``pm.sample`` yields the structure-known posterior on any drawn world — the
identification floor amortized models (PFNs) are judged against. The prior
definitions (:func:`_uniform_prior_specs`, :func:`_walk_priors`) are shared
between the generative and oracle builders so they cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.sharedvar import SharedVariable

from . import mechanisms
from .random_walk import _kernel_width, _walk_basis
from .sampler import ADSTOCK_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior
from .symbolic_graph import (
    _adstock_col,
    _saturate_col,
    _walk_column,
    build_symbolic_graph,
)

# Compiling a draw function costs seconds, so repeated batches from the SAME
# model (realism-filter top-up rounds) reuse one. The cache lives on the model
# object rather than in a module-level dict keyed by ``id(model)``: CPython
# recycles ids after garbage collection, so a global would eventually hand a
# stale function to a brand-new model. Attaching it also ties the cache's
# lifetime to the model's, which is exactly the intended scope.
_DRAW_FN_CACHE_ATTR = "_prior_generator_draw_fn_cache"

DRAW_FN_CACHE_HITS = 0
DRAW_FN_CACHE_MISSES = 0
USE_COMPILE_CACHE = True


def set_compile_cache_enabled(enabled: bool) -> None:
    """Toggle the per-model compiled-draw-function cache (for benchmarks)."""
    global USE_COMPILE_CACHE
    USE_COMPILE_CACHE = bool(enabled)


def reset_world_model_caches() -> None:
    """Reset the compile-cache counters (for tests and benchmarks).

    The cache itself is per-model, so it is discarded with the model and needs no
    explicit clearing.
    """
    global DRAW_FN_CACHE_HITS, DRAW_FN_CACHE_MISSES
    DRAW_FN_CACHE_HITS = 0
    DRAW_FN_CACHE_MISSES = 0


def _structure_compile_inputs(input_vars: list) -> tuple[list, dict]:
    """Map ``pm.Data`` shared variables to explicit tensor inputs via ``givens``."""
    givens: dict = {}
    inputs: list = []
    for var in input_vars:
        if isinstance(var, SharedVariable):
            placeholder = pt.tensor(
                name=f"{var.name}_input",
                dtype=var.type.dtype,
                shape=var.type.shape,
            )
            givens[var] = placeholder
            inputs.append(placeholder)
        else:
            inputs.append(var)
    return inputs, givens


def _get_cached_draw_fn(
    model: pm.Model,
    out_vars: list,
    *,
    mode: str,
    reference_vars: list | None,
    input_vars: list | None = None,
) -> tuple[Callable[..., Any], list]:
    """Compile a draw function for ``model``, reusing this model's earlier one.

    The RNG list is ordered so that every RNG already reachable from
    ``reference_vars`` keeps its position; newly reachable ones are appended.
    That ordering is what makes ``reseed_rngs`` reproduce the same streams when a
    caller asks for additional outputs.
    """
    from pymc.pytensorf import collect_default_updates
    from pymc.pytensorf import compile as compile_pymc

    global DRAW_FN_CACHE_HITS, DRAW_FN_CACHE_MISSES

    input_vars = input_vars or []
    compile_inputs, givens = _structure_compile_inputs(input_vars)
    cache_key = (
        tuple(v.name for v in out_vars),
        tuple(v.name for v in input_vars),
        mode,
        tuple(v.name for v in reference_vars) if reference_vars else None,
    )
    cache: dict[tuple[Any, ...], tuple[Callable[..., Any], list]] | None = None
    if USE_COMPILE_CACHE:
        cache = getattr(model, _DRAW_FN_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            object.__setattr__(model, _DRAW_FN_CACHE_ATTR, cache)
        cached = cache.get(cache_key)
        if cached is not None:
            DRAW_FN_CACHE_HITS += 1
            return cached
        DRAW_FN_CACHE_MISSES += 1

    with model:
        draw_fn = compile_pymc(
            inputs=compile_inputs,
            outputs=out_vars,
            mode=mode,
            **({"givens": givens} if givens else {}),
        )
    output_rngs = list(collect_default_updates(inputs=compile_inputs, outputs=out_vars))
    if reference_vars is None:
        ordered_rngs = output_rngs
    else:
        reference_rngs = list(
            collect_default_updates(inputs=compile_inputs, outputs=reference_vars)
        )
        ordered_rngs = reference_rngs + [rng for rng in output_rngs if rng not in reference_rngs]
    if cache is not None:
        cache[cache_key] = (draw_fn, ordered_rngs)
    return draw_fn, ordered_rngs


def _execute_draws(
    draw_fn: Callable[..., Any],
    ordered_rngs: list,
    *,
    seed: int,
    draws: int,
    input_args: tuple[Any, ...] = (),
) -> list | tuple:
    """Reseed and run ``draws`` forward passes without recompiling.

    ``draw_fn`` is always compiled with a *list* of outputs, so it returns one
    element per output even when a single name was requested.
    """
    from pymc.pytensorf import reseed_rngs
    from pymc.util import _get_seeds_per_chain

    (compile_seed,) = _get_seeds_per_chain(np.random.default_rng(seed), 1)
    reseed_rngs(ordered_rngs, compile_seed)
    if draws == 1:
        return cast("list | tuple", draw_fn(*input_args))
    return [np.stack(values) for values in zip(*(draw_fn(*input_args) for _ in range(draws)))]


def sample_structure(g_active: dict, cfg: SCMPrior, rng: np.random.Generator) -> dict:
    """Draw the concrete per-world structure (families, walk smoothness, texture flags).

    These are the discrete/structural choices that set the graph's shape, drawn
    with numpy from ``cfg``'s family probabilities and random-walk smoothness
    Beta prior. ``RW_Y`` is iid, so it has no structural smoothness. The
    continuous parameters are NOT drawn here — they are PyMC distributions
    inside :func:`build_world_model`.
    """
    n_treatments = len(g_active["g_cy"])
    n_covariates = len(g_active["g_zb"])
    n_latent = len(g_active["g_db"])
    ad_fam = rng.choice(
        len(ADSTOCK_FAMILY_KEYS),
        size=n_treatments,
        p=np.asarray([cfg.adstock_family_probs[key] for key in ADSTOCK_FAMILY_KEYS]),
    )
    sat_fam = rng.choice(
        len(SATURATION_FAMILY_KEYS),
        size=n_treatments,
        p=np.asarray([cfg.saturation_family_probs[key] for key in SATURATION_FAMILY_KEYS]),
    )

    def _smooth(n: int) -> np.ndarray:
        return rng.beta(cfg.rw_smoothness_alpha, cfg.rw_smoothness_beta, size=n)

    hf_on = float(cfg.channel_hf_sigma_range[1]) > 0.0
    pulse_on = float(cfg.channel_pulse_prob_range[1]) > 0.0
    control_hf_on = float(cfg.control_hf_sigma_range[1]) > 0.0
    control_pulse_on = float(cfg.control_pulse_prob_range[1]) > 0.0
    return {
        "adstock_family": ad_fam.astype(int),
        "sat_family": sat_fam.astype(int),
        "smoothness_d": _smooth(n_latent),
        "smoothness_z": _smooth(n_covariates),
        "smoothness_c": _smooth(n_treatments),
        "smoothness_b": _smooth(1),
        "use_hf": np.full(n_treatments, hf_on),
        "use_pulse": np.full(n_treatments, pulse_on),
        "use_control_hf": np.full(n_covariates, control_hf_on),
        "use_control_pulse": np.full(n_covariates, control_pulse_on),
    }


def sample_prior_cond(
    cfg: SCMPrior, rng: np.random.Generator
) -> dict[str, tuple[float, float]] | None:
    """Draw one cell's prior-conditioning intervals (stage-1 concrete numpy).

    For each conditioned quantity ``q`` with support ``[S_lo, S_hi]`` and
    width range ``[w_lo, w_hi]`` (see :meth:`SCMPrior.prior_cond_spec`):

    .. code-block:: text

        w  ~ U(w_lo, w_hi)
        lo ~ U(S_lo, S_hi - w)
        I_q = [lo, lo + w]            (subset of [S_lo, S_hi] by construction)

    Stage 2 (:func:`build_world_model`) then draws that cell's parameter as
    ``pm.Uniform(lo, lo + w)`` instead of the global support. Interval bounds
    become ``pm.Uniform`` constants, so a cell is exactly one model build —
    worlds within a cell share their intervals; cells differ.

    Returns ``None`` — WITHOUT consuming any RNG state — when
    ``cfg.prior_conditioning`` is False, so the disabled path reproduces
    unconditioned corpora bit-for-bit.
    """
    if not cfg.prior_conditioning:
        return None
    out: dict[str, tuple[float, float]] = {}
    for q, spec in cfg.prior_cond_spec().items():
        s_lo, s_hi = spec["support"]
        w_lo, w_hi = spec["width_range"]
        w = float(rng.uniform(w_lo, w_hi))
        lo = float(rng.uniform(s_lo, s_hi - w))
        out[q] = (lo, w)
    return out


def _uniform(name: str, lo: float, hi: float, shape):
    """A ``pm.Uniform`` prior, or a constant when the range is degenerate (lo == hi)."""
    lo, hi = float(lo), float(hi)
    if lo == hi:
        return pt.as_tensor_variable(np.full(shape, lo, dtype="float64"))
    return pm.Uniform(name, lo, hi, shape=shape)


def _channel_shock_schedule(
    cfg: SCMPrior, g_cy: np.ndarray, n_time_steps: int, burn_in: int, c_level
) -> dict[str, Any]:
    """Build the symbolic intervention schedule for configured channel shocks.

    The disjoint chronological slots make overlap impossible even when the
    same direct channel is selected repeatedly. The full-horizon mask and
    level matrix are internal inputs to the channel equations.
    """
    n_shocks = int(cfg.n_channel_shocks)
    n_treatments = len(g_cy)
    if n_shocks == 0:
        return {}

    direct = np.flatnonzero(np.asarray(g_cy) == 1).astype("int64")
    if not len(direct):
        raise ValueError("enabled channel shocks require at least one direct g_cy channel")
    len_lo, len_hi = cfg.channel_shock_length_range
    level_lo, level_hi = cfg.channel_shock_level_range
    if len_lo == len_hi:
        lengths = pt.as_tensor_variable(np.full(n_shocks, len_lo, dtype="int64"))
    else:
        lengths = pm.DiscreteUniform("channel_shock_length", len_lo, len_hi, shape=n_shocks)
    if len(direct) == 1:
        ranks = pt.zeros((n_shocks,), dtype="int64")
    else:
        ranks = pm.DiscreteUniform("channel_shock_channel_rank", 0, len(direct) - 1, shape=n_shocks)
    channels = pt.cast(pt.as_tensor_variable(direct)[ranks], "int64")
    if level_lo == level_hi:
        multipliers = pt.as_tensor_variable(np.full(n_shocks, level_lo, dtype="float64"))
    else:
        multipliers = pm.Uniform(
            "channel_shock_level_multiplier", level_lo, level_hi, shape=n_shocks
        )

    starts = []
    for s in range(n_shocks):
        slot_lo = s * n_time_steps // n_shocks
        slot_hi = (s + 1) * n_time_steps // n_shocks
        # A fixed length that fills the slot has exactly one feasible start.
        if len_lo == len_hi and slot_hi - slot_lo == len_lo:
            starts.append(pt.as_tensor_variable(np.asarray(slot_lo, dtype="int64")))
        else:
            starts.append(
                pm.DiscreteUniform(f"channel_shock_start_{s}", slot_lo, slot_hi - lengths[s])
            )
    starts_t = pt.stack(starts)
    levels = multipliers * c_level[channels]

    def _event_mask(n_time: int, offset: int):
        time = pt.arange(n_time)[:, None]
        active = (time >= (starts_t + offset)[None, :]) & (
            time < (starts_t + lengths + offset)[None, :]
        )
        selected = pt.eq(pt.arange(n_treatments)[:, None], channels[None, :]).T
        return active[:, :, None] & selected[None, :, :]

    event_mask = _event_mask(n_time_steps, 0)
    mask = pt.cast(pt.any(event_mask, axis=1), "int8")
    event_mask_full = _event_mask(n_time_steps + burn_in, burn_in)
    mask_full = pt.cast(pt.any(event_mask_full, axis=1), "int8")
    level_full = pt.sum(
        pt.cast(event_mask_full, "float64") * levels[None, :, None],
        axis=1,
    )
    return {
        "level_full": level_full,
        "channel_shock_mask": mask,
        "channel_shock_mask_full": mask_full,
        "channel_shock_channel": channels,
        "channel_shock_start": starts_t,
        "channel_shock_length": lengths,
        "channel_shock_level_multiplier": multipliers,
        "channel_shock_level": levels,
    }


def _validate_oracle_channel_shocks(
    cfg: SCMPrior, g_cy: np.ndarray, data: dict[str, np.ndarray], n_time_steps: int
) -> None:
    """Validate a world's reported-window held-level schedule for the oracle.

    Channel shocks are known intervention-design state, not latent variables in
    the observed-data model. They clamp observed spend and nothing else, so the
    oracle needs no schedule tensors — it reads the already-clamped channels as
    data. This still checks the complete reported schedule and its held levels
    against that observed spend so corrupt metadata fails loudly.
    """
    n_shocks = int(cfg.n_channel_shocks)
    if n_shocks == 0:
        return

    def _event_int(name: str) -> np.ndarray:
        if name not in data:
            raise ValueError(f"enabled channel shocks require data[{name!r}] metadata")
        value = np.asarray(data[name])
        if value.shape != (n_shocks,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(
                f"data[{name!r}] must be an integer array with shape "
                f"{(n_shocks,)}, got {value.shape}"
            )
        return value.astype("int64", copy=False)

    channel = _event_int("channel_shock_channel")
    start = _event_int("channel_shock_start")
    length = _event_int("channel_shock_length")
    if "channel_shock_level_multiplier" not in data:
        raise ValueError(
            "enabled channel shocks require data['channel_shock_level_multiplier'] metadata"
        )
    multiplier = np.asarray(data["channel_shock_level_multiplier"], dtype="float64")
    if multiplier.shape != (n_shocks,) or not np.isfinite(multiplier).all():
        raise ValueError(
            "data['channel_shock_level_multiplier'] must be a finite array "
            f"with shape {(n_shocks,)}, got {multiplier.shape}"
        )
    if "channel_shock_level" not in data:
        raise ValueError("enabled channel shocks require data['channel_shock_level'] metadata")
    level = np.asarray(data["channel_shock_level"], dtype="float64")
    if level.shape != (n_shocks,) or not np.isfinite(level).all():
        raise ValueError(
            f"data['channel_shock_level'] must be a finite array with shape {(n_shocks,)}, "
            f"got {level.shape}"
        )
    if "channel_level" not in data:
        raise ValueError("enabled channel shocks require data['channel_level'] metadata")
    channel_level = np.asarray(data["channel_level"], dtype="float64")
    if channel_level.shape != g_cy.shape or not (
        np.isfinite(channel_level).all() and (channel_level > 0.0).all()
    ):
        raise ValueError(
            "data['channel_level'] must be finite and positive with shape "
            f"{g_cy.shape}, got {channel_level.shape}"
        )

    direct = np.flatnonzero(np.asarray(g_cy) == 1)
    if not np.isin(channel, direct).all():
        raise ValueError("channel shock metadata must select only direct g_cy channels")
    len_lo, len_hi = cfg.channel_shock_length_range
    if ((length < len_lo) | (length > len_hi)).any():
        raise ValueError("channel shock metadata length is outside the configured range")
    level_lo, level_hi = cfg.channel_shock_level_range
    if ((multiplier < level_lo) | (multiplier > level_hi)).any():
        raise ValueError("channel shock metadata level multiplier is outside the configured range")
    if not np.allclose(level, multiplier * channel_level[channel], rtol=1e-6, atol=1e-7):
        raise ValueError("channel shock level does not match multiplier * channel_level")
    for s in range(n_shocks):
        slot_lo = s * n_time_steps // n_shocks
        slot_hi = (s + 1) * n_time_steps // n_shocks
        if not slot_lo <= start[s] <= slot_hi - length[s]:
            raise ValueError(
                f"channel shock metadata start {start[s]} is infeasible for "
                f"slot {s} and length {length[s]}"
            )
        observed = np.asarray(data["channels"])[start[s] : start[s] + length[s], channel[s]]
        if not np.allclose(observed, level[s], rtol=1e-6, atol=1e-7):
            raise ValueError("channel shock held level does not match observed spend")


def _rw_prior_group(
    name,
    n,
    positive,
    mean_range,
    smoothness=None,
    *,
    rw_smoothness_max_weeks: int | None = None,
    std_sigma=None,
    std_range=None,
    relative=False,
    std_name: str | None = None,
    width_index=None,
):
    """One node group's scale priors and optional concrete smoothing timescale.

    Must be called inside a ``pm.Model`` context. This is THE single
    definition of the walk and iid-noise priors — the generative draw and the
    posterior oracle both build their parameters here, so they cannot drift.
    ``std_range`` and ``std_sigma`` are mutually exclusive scale definitions.
    A group is a random walk when it carries a smoothing timescale, given either
    as a concrete ``smoothness`` or as already-resolved ``width_index`` kernel
    widths (see :func:`prior_generator.random_walk.walk_width_index`). Width
    indices may be tensors, which is what lets the width stay a run-time value
    instead of a compile-time one. With neither, the group is iid noise and every
    random-walk-only field is deliberately omitted.
    """
    if std_range is not None and std_sigma is not None:
        raise ValueError("std_range and std_sigma are mutually exclusive")
    mean = _uniform(f"{name}_mean", mean_range[0], mean_range[1], n)
    if std_range is None:
        if std_sigma is None:
            raise ValueError("std_sigma is required when std_range is absent")
        std = pm.HalfNormal(f"{name}_std", sigma=std_sigma, shape=n)
    else:
        std = _uniform(std_name or f"{name}_std", std_range[0], std_range[1], n)
        if relative:  # scale-free: amplitude relative to the walk's level
            std = std * pt.softplus(mean)
    out = {
        "mean": mean,
        "std": std,
        "positive_only": positive,
    }
    if smoothness is None and width_index is None:
        if rw_smoothness_max_weeks is not None:
            raise ValueError("iid noise must not carry rw_smoothness_max_weeks")
    else:
        if rw_smoothness_max_weeks is None:
            raise ValueError("random walks require rw_smoothness_max_weeks")
        out["rw_smoothness_max_weeks"] = rw_smoothness_max_weeks
        if smoothness is not None:
            out["smoothness"] = smoothness
        if width_index is not None:
            out["width_index"] = width_index
    return out


def _walk_priors(
    cfg: SCMPrior,
    structural: dict,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    include: tuple[str, ...] = ("d", "z", "c", "b", "y"),
    width_indices: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Walk-prior groups per node type, registered in the LOCKED d/z/c/b/y order.

    Sizes follow the :class:`SCMPrior` vocabulary — ``n_treatments`` media
    channels, ``n_covariates`` observed controls, ``n_latent`` hidden
    confounders. ``include`` selects the groups a model needs (the oracle
    skips the ones replaced by observed data); relative order is always
    preserved.

    ``width_indices`` optionally maps a group name to its resolved kernel-width
    indices, which keeps random-walk smoothness a run-time value (see
    :func:`_rw_prior_group`).
    """
    out: dict[str, dict] = {}
    widths = width_indices or {}
    rw_smoothness_max_weeks = cfg.rw_smoothness_max_weeks
    if "d" in include:
        # A latent factor carries no scale or level of its own: both belong to
        # its loadings. Leaving rw_d_mean / rw_d_std free made
        # (sigma_d, w_dc, u_dz, delta_db) -> (lam*sigma_d, w/lam, u/lam, delta/lam)
        # an exact symmetry of every observable, and delta_db*mu_d was
        # absorbed by rw_b_mean, so neither factor was recoverable. Pinning the
        # factor to mean 0 / scale 1 puts the whole D->* magnitude in the
        # loadings, which is what the data identifies.
        # The graph also admits a sign flip: negating eps_d together with w_dc,
        # u_dz, and delta_db flips demand's sign while leaving every other output
        # unchanged. The supported prior
        # assigns all three loading families strictly positive ranges, so the
        # reflection is outside support and demand's sign is identified. If a
        # user widens db_coeff_range / dc_coeff_range / dz_coeff_range to admit
        # negatives, only |D| and the loading magnitudes are recoverable.
        out["rw_d"] = _rw_prior_group(
            "rw_d",
            n_latent,
            False,
            (0.0, 0.0),
            structural.get("smoothness_d"),
            rw_smoothness_max_weeks=rw_smoothness_max_weeks,
            std_range=(1.0, 1.0),
            width_index=widths.get("rw_d"),
        )
    if "z" in include:
        out["rw_z"] = _rw_prior_group(
            "rw_z",
            n_covariates,
            False,
            cfg.rw_mean_range,
            structural.get("smoothness_z"),
            std_sigma=cfg.rw_std_sigma,
            rw_smoothness_max_weeks=rw_smoothness_max_weeks,
            width_index=widths.get("rw_z"),
        )
    if "c" in include:
        out["rw_c"] = _rw_prior_group(
            "rw_c",
            n_treatments,
            True,
            cfg.rw_positive_mean_range,
            structural.get("smoothness_c"),
            rw_smoothness_max_weeks=rw_smoothness_max_weeks,
            std_range=cfg.rw_channel_std_range,
            relative=True,
            width_index=widths.get("rw_c"),
        )
    if "b" in include:
        if cfg.outcome_std_mode == "relative":
            out["rw_b"] = _rw_prior_group(
                "rw_b",
                1,
                False,
                cfg.rw_baseline_mean_range,
                structural.get("smoothness_b"),
                rw_smoothness_max_weeks=rw_smoothness_max_weeks,
                std_range=cfg.rw_baseline_std_range,
                std_name="rw_b_std_rel",
                width_index=widths.get("rw_b"),
            )
        else:
            out["rw_b"] = _rw_prior_group(
                "rw_b",
                1,
                False,
                cfg.rw_baseline_mean_range,
                structural.get("smoothness_b"),
                rw_smoothness_max_weeks=rw_smoothness_max_weeks,
                std_sigma=cfg.rw_baseline_std_sigma_effective,
                width_index=widths.get("rw_b"),
            )
    if "y" in include:
        if cfg.outcome_std_mode == "relative":
            out["rw_y"] = _rw_prior_group(
                "rw_y",
                1,
                False,
                (0.0, 0.0),
                std_range=cfg.rw_sales_std_range,
                std_name="rw_y_std_rel",
            )
        else:
            out["rw_y"] = _rw_prior_group(
                "rw_y",
                1,
                False,
                (0.0, 0.0),
                std_sigma=cfg.rw_sales_std_sigma,
            )
    return out


def _scm_params(
    cfg: SCMPrior,
    specs: dict[str, tuple[str, float, float, Any]],
    rw: dict[str, dict],
    c_level,
    *,
    adstock_family,
    sat_family,
    use_hf,
    use_pulse,
    use_control_hf,
    use_control_pulse,
) -> dict[str, Any]:
    """Every continuous SCM parameter, in the LOCKED RV creation order.

    Shared by :func:`build_world_model` and
    :func:`prior_generator.world_model_batched.build_world_model_template` so the
    per-world and one-compile-per-shard paths cannot drift apart.

    The order is load-bearing: ``reseed_rngs`` hands out random streams by
    position in the collected RNG list, so inserting, removing, or reordering a
    draw here changes every generated world.
    """
    pulse_prob = _uniform(*specs["pulse_prob"])
    return {
        "l_max": cfg.l_max,
        # Concrete structural constant, not a draw: it clips the intercept walk.
        "baseline_floor": cfg.baseline_floor,
        "baseline_floor_scope": cfg.baseline_floor_scope,
        # linear edge coefficients
        "w_dc": _uniform(*specs["w_dc"]),
        "u_dz": _uniform(*specs["u_dz"]),
        "v_zc": _uniform(*specs["v_zc"]),
        "alpha_cc": _uniform(*specs["alpha_cc"]),
        "gamma_zz": _uniform(*specs["gamma_zz"]),
        "delta_db": _uniform(*specs["delta_db"]),
        "rho_zb": _uniform(*specs["rho_zb"]),
        "beta": _uniform(*specs["beta"]),
        # per-node random walks and iid outcome noise
        "rw_d": rw["rw_d"],
        "rw_z": rw["rw_z"],
        "rw_c": rw["rw_c"],
        "rw_b": rw["rw_b"],
        "rw_y": rw["rw_y"],
        "adstock_family": adstock_family,
        "sat_family": sat_family,
        **{name: _uniform(*specs[name]) for name in _MECHANISM_PARAM_NAMES},
        # channel texture: magnitudes relative to the channel level; fires
        # are Bernoulli(pulse_prob)
        "hf_sigma": _uniform(*specs["hf_sigma"]) * c_level,
        "pulse_amp": _uniform(*specs["pulse_amp"]) * c_level,
        "pulse_prob": pulse_prob,
        "use_hf": use_hf,
        "use_pulse": use_pulse,
        # control texture: magnitudes relative to the control's OWN walk std
        # (a signed control has no positive level anchor to scale by); fires
        # are Bernoulli(control_pulse_prob) and are centred in the structural
        # equation, so the control's expected level stays rw_z_mean.
        # These draws leave seeded corpora alone ONLY while their ranges are
        # degenerate: _uniform then emits a constant and no RNG node exists.
        # There is no stream-stable position for a live draw — reseed_rngs
        # walks collect_default_updates' graph-traversal order, not this
        # creation order — so enabling the texture deliberately reseeds every
        # world (see tests/test_identifiability.py's two hash contracts).
        "control_hf_sigma": _uniform(*specs["control_hf_sigma"]) * rw["rw_z"]["std"],
        "control_pulse_amp": _uniform(*specs["control_pulse_amp"]) * rw["rw_z"]["std"],
        "control_pulse_prob": _uniform(*specs["control_pulse_prob"]),
        "use_control_hf": use_control_hf,
        "use_control_pulse": use_control_pulse,
    }


def _scm_eps(
    n_time_steps_full: int,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    pulse_prob,
    control_pulse_prob,
) -> dict[str, Any]:
    """The graph's noise RVs, in the LOCKED creation order (see :func:`_scm_params`)."""
    return {
        "eps_d": pm.Normal("eps_d", 0.0, 1.0, shape=(n_time_steps_full, n_latent)),
        "eps_z": pm.Normal("eps_z", 0.0, 1.0, shape=(n_time_steps_full, n_covariates)),
        "eps_c": pm.Normal("eps_c", 0.0, 1.0, shape=(n_time_steps_full, n_treatments)),
        "eps_b": pm.Normal("eps_b", 0.0, 1.0, shape=(n_time_steps_full,)),
        "eps_y": pm.Normal("eps_y", 0.0, 1.0, shape=(n_time_steps_full,)),
        "eps_c_hf": pm.Normal("eps_c_hf", 0.0, 1.0, shape=(n_time_steps_full, n_treatments)),
        "eps_c_pulse": pm.Bernoulli(
            "eps_c_pulse",
            p=pt.broadcast_to(pulse_prob, (n_time_steps_full, n_treatments)),
            shape=(n_time_steps_full, n_treatments),
        ).astype("float64"),
        # Control texture noise. A disabled config still creates these RVs (the
        # channel texture noise behaves identically): the graph then references
        # neither, so they reach no output, collect no RNG stream, and leave
        # every seeded draw byte-identical — while keeping the audit schema
        # (SCM.exogenous) the same shape for every config.
        "eps_z_hf": pm.Normal("eps_z_hf", 0.0, 1.0, shape=(n_time_steps_full, n_covariates)),
        "eps_z_pulse": pm.Bernoulli(
            "eps_z_pulse",
            p=pt.broadcast_to(control_pulse_prob, (n_time_steps_full, n_covariates)),
            shape=(n_time_steps_full, n_covariates),
        ).astype("float64"),
    }


def _register_param_reports(
    params: dict[str, Any],
    rw: dict[str, dict],
    c_level,
    confounding_strength,
) -> tuple[str, ...]:
    """Register every continuous parameter as a ``param_*`` deterministic.

    A sampled world then carries all the concrete inputs needed to replay its
    structural graph. Families and smoothness are concrete structure already and
    are reported with their corresponding groups.
    """
    report_specs: dict[str, Any] = {
        # linear edge coefficients
        "beta": params["beta"],
        "w_dc": params["w_dc"],
        "u_dz": params["u_dz"],
        "v_zc": params["v_zc"],
        "alpha_cc": params["alpha_cc"],
        "gamma_zz": params["gamma_zz"],
        "delta_db": params["delta_db"],
        "rho_zb": params["rho_zb"],
        # every per-channel mechanism shape parameter
        **{name: params[name] for name in _MECHANISM_PARAM_NAMES},
        # channel texture magnitudes and fire probability
        "hf_sigma": params["hf_sigma"],
        "pulse_amp": params["pulse_amp"],
        "pulse_prob": params["pulse_prob"],
        # control texture magnitudes (already scaled by rw_z_std) and fire
        # probability, so a world replays without re-deriving the scale
        "control_hf_sigma": params["control_hf_sigma"],
        "control_pulse_amp": params["control_pulse_amp"],
        "control_pulse_prob": params["control_pulse_prob"],
    }
    for group_name in ("rw_d", "rw_z", "rw_c", "rw_b", "rw_y"):
        report_specs[f"{group_name}_mean"] = rw[group_name]["mean"]
        report_specs[f"{group_name}_std"] = rw[group_name]["std"]
    report_specs["channel_level"] = c_level
    report_specs["confounding_strength"] = confounding_strength
    for key, tensor in report_specs.items():
        pm.Deterministic(f"param_{key}", tensor)
    return tuple(f"param_{k}" for k in report_specs)


def _disabled_shock_outputs(
    n_time_steps: int, n_time_steps_full: int, n_treatments: int
) -> dict[str, Any]:
    """Zero-sized stand-ins for the channel-shock outputs, for configs without shocks.

    Corpora always carry the shock columns, but a disabled schedule must stay out
    of the structural graph and the RV stream entirely, so these are constants
    rather than a degenerate schedule.
    """
    return {
        "channel_shock_mask": pt.zeros((n_time_steps, n_treatments), dtype="int8"),
        "channel_shock_mask_full": pt.zeros((n_time_steps_full, n_treatments), dtype="int8"),
        "channel_shock_channel": pt.zeros((0,), dtype="int64"),
        "channel_shock_start": pt.zeros((0,), dtype="int64"),
        "channel_shock_length": pt.zeros((0,), dtype="int64"),
        "channel_shock_level_multiplier": pt.zeros((0,), dtype="float64"),
        "channel_shock_level": pt.zeros((0,), dtype="float64"),
    }


def _confounded_channel_eps(cfg: SCMPrior, eps: dict[str, Any]):
    """Resolve the per-world confounding strength and mix it into ``eps_c``.

    Preserves the legacy innovation dictionary and graph path exactly when
    confounding is disabled, including its explicit ``(0.0, 0.0)`` spelling.
    When enabled, rho is a single per-world value and only the channel
    innovation supplied to the graph changes. The orthonormal mixture leaves
    every channel innovation's marginal variance at one while correlating it
    with the baseline innovation.
    """
    if cfg.confounding_strength_range is None:
        return pt.as_tensor_variable(np.asarray(0.0, dtype="float64"))
    lo, hi = cfg.confounding_strength_range
    if float(lo) == float(hi):
        confounding_strength = pt.as_tensor_variable(np.asarray(lo, dtype="float64"))
    else:
        confounding_strength = pm.Uniform("confounding_strength", lo, hi)
    if float(hi) > 0.0:
        eps["eps_c"] = (
            pt.sqrt(1.0 - confounding_strength**2) * eps["eps_c"]
            + confounding_strength * eps["eps_b"][:, None]
        )
    return confounding_strength


def _apply_outcome_std_scale(
    cfg: SCMPrior, rw: dict[str, dict], g_cy: np.ndarray, beta, *, prefix: str = ""
) -> None:
    """Convert relative outcome scales to their parameter-only absolute amplitudes.

    ``g_cy`` is concrete per world, so the anchor
    ``sqrt(sum((g_cy * beta)**2))`` depends only on structural and continuous
    parameters. It cannot depend on innovations without making the prior
    undefined independently of the noise it generates.
    """
    if cfg.outcome_std_mode == "absolute":
        return
    if cfg.outcome_std_mode != "relative":
        raise ValueError(
            f"outcome_std_mode must be 'relative' or 'absolute', got {cfg.outcome_std_mode!r}"
        )
    g_cy_t = pt.as_tensor_variable(g_cy)
    media_amplitude = pt.sqrt(pt.sum((g_cy_t * beta) ** 2))
    for group_name in ("rw_b", "rw_y"):
        rw[group_name]["std"] = pm.Deterministic(
            f"{prefix}{group_name}_std",
            rw[group_name]["std"] * media_amplitude,
        )


#: The per-channel media-response shape params (spec keys), canonical order.
_MECHANISM_PARAM_NAMES: tuple[str, ...] = (
    "adstock_alpha",
    "weibull_lam",
    "weibull_k",
    "hill_slope",
    "hill_kappa_mult",
    "logistic_lam",
    "mm_kappa_mult",
    "tanh_c",
    "root_alpha",
)


#: Shape-parameter spec keys consumed by each canonical adstock family.
#: ``"weibull"`` is the canonical family key for the Weibull-PDF transform.
ADSTOCK_FAMILY_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "none": (),
    "geometric": ("adstock_alpha",),
    "weibull": ("weibull_lam", "weibull_k"),
}

#: Shape-parameter spec keys consumed by each canonical saturation family.
SATURATION_FAMILY_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "linear": (),
    "hill": ("hill_slope", "hill_kappa_mult"),
    "logistic": ("logistic_lam",),
    "michaelis_menten": ("mm_kappa_mult",),
    "tanh": ("tanh_c",),
    "root": ("root_alpha",),
}


def _live_mechanism_param_names(structural: dict) -> tuple[str, ...]:
    """Return the canonical-order shape params consumed by this structure."""
    live: set[str] = set()
    for family_id in np.asarray(structural["adstock_family"]):
        live.update(ADSTOCK_FAMILY_PARAM_NAMES[ADSTOCK_FAMILY_KEYS[int(family_id)]])
    for family_id in np.asarray(structural["sat_family"]):
        live.update(SATURATION_FAMILY_PARAM_NAMES[SATURATION_FAMILY_KEYS[int(family_id)]])
    return tuple(name for name in _MECHANISM_PARAM_NAMES if name in live)


def _uniform_prior_specs(
    cfg: SCMPrior,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    prior_cond: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[str, float, float, Any]]:
    """``{param: (name, lo, hi, shape)}`` for every ``pm.Uniform`` prior.

    THE single definition of the uniform prior ranges — both the generative
    draw (:func:`build_world_model`) and the posterior oracle
    (:func:`build_oracle_model`) create their RVs as ``_uniform(*spec)`` from
    this table, so the priors cannot drift between the two. Also resolves the
    prior-conditioning narrowing (``prior_cond``) for the conditioned set.
    Sizes follow the :class:`SCMPrior` vocabulary (``n_treatments`` media
    channels / ``n_covariates`` controls / ``n_latent`` hidden confounders).
    """
    spr = mechanisms.SATURATION_PRIOR_RANGES
    n_t, n_c, n_l = n_treatments, n_covariates, n_latent
    adstock_alpha_range = cfg.adstock_alpha_range
    hill_shape_range = spr["hill"]["slope"]
    if prior_cond is not None:
        if "adstock_alpha" in prior_cond:
            lo, width = prior_cond["adstock_alpha"]
            adstock_alpha_range = (lo, lo + width)
        if "hill_shape" in prior_cond:
            lo, width = prior_cond["hill_shape"]
            hill_shape_range = (lo, lo + width)
    return {
        # linear edge coefficients
        "w_dc": ("w_dc", cfg.dc_coeff_range[0], cfg.dc_coeff_range[1], (n_l, n_t)),
        "u_dz": ("u_dz", cfg.dz_coeff_range[0], cfg.dz_coeff_range[1], (n_l, n_c)),
        "v_zc": ("v_zc", cfg.zc_coeff_range[0], cfg.zc_coeff_range[1], (n_c, n_t)),
        "alpha_cc": ("alpha_cc", cfg.cc_coeff_range[0], cfg.cc_coeff_range[1], (n_t, n_t)),
        "gamma_zz": ("gamma_zz", cfg.zz_coeff_range[0], cfg.zz_coeff_range[1], (n_c, n_c)),
        "delta_db": ("delta_db", cfg.db_coeff_range[0], cfg.db_coeff_range[1], n_l),
        "rho_zb": ("rho_zb", cfg.zb_coeff_range[0], cfg.zb_coeff_range[1], n_c),
        "beta": ("beta", cfg.beta_additive_range[0], cfg.beta_additive_range[1], n_t),
        # per-channel mechanism shape priors (adstock + saturation families)
        "adstock_alpha": ("adstock_alpha", adstock_alpha_range[0], adstock_alpha_range[1], n_t),
        "weibull_lam": ("weibull_lam", cfg.weibull_lam_range[0], cfg.weibull_lam_range[1], n_t),
        "weibull_k": ("weibull_k", cfg.weibull_k_range[0], cfg.weibull_k_range[1], n_t),
        "hill_slope": ("hill_slope", hill_shape_range[0], hill_shape_range[1], n_t),
        "hill_kappa_mult": (
            "hill_kappa_mult",
            spr["hill"]["kappa_mult"][0],
            spr["hill"]["kappa_mult"][1],
            n_t,
        ),
        "logistic_lam": (
            "logistic_lam",
            spr["logistic"]["lam"][0],
            spr["logistic"]["lam"][1],
            n_t,
        ),
        "mm_kappa_mult": (
            "mm_kappa_mult",
            spr["michaelis_menten"]["kappa_mult"][0],
            spr["michaelis_menten"]["kappa_mult"][1],
            n_t,
        ),
        "tanh_c": ("tanh_c", spr["tanh"]["c"][0], spr["tanh"]["c"][1], n_t),
        "root_alpha": ("root_alpha", spr["root"]["alpha"][0], spr["root"]["alpha"][1], n_t),
        # channel texture factors (relative to the channel level)
        "hf_sigma": (
            "hf_sigma",
            cfg.channel_hf_sigma_range[0],
            cfg.channel_hf_sigma_range[1],
            n_t,
        ),
        "pulse_amp": (
            "pulse_amp",
            cfg.channel_pulse_amp_range[0],
            cfg.channel_pulse_amp_range[1],
            n_t,
        ),
        "pulse_prob": (
            "pulse_prob",
            cfg.channel_pulse_prob_range[0],
            cfg.channel_pulse_prob_range[1],
            n_t,
        ),
        # control texture factors (relative to the control's own walk std)
        "control_hf_sigma": (
            "control_hf_sigma",
            cfg.control_hf_sigma_range[0],
            cfg.control_hf_sigma_range[1],
            n_c,
        ),
        "control_pulse_amp": (
            "control_pulse_amp",
            cfg.control_pulse_amp_range[0],
            cfg.control_pulse_amp_range[1],
            n_c,
        ),
        "control_pulse_prob": (
            "control_pulse_prob",
            cfg.control_pulse_prob_range[0],
            cfg.control_pulse_prob_range[1],
            n_c,
        ),
    }


def build_world_model(
    g_active: dict,
    cfg: SCMPrior,
    structural: dict,
    n_time_steps: int,
    prior_cond: dict[str, tuple[float, float]] | None = None,
) -> tuple[pm.Model, tuple[str, ...], tuple[str, ...]]:
    """Build the ``pm.Model`` for one world structure.

    Returns ``(model, output_names, param_names)`` — the graph outputs and the
    ``param_*`` reporting deterministics, both drawable by name via
    :func:`draw_worlds`.

    Parameters
    ----------
    g_active : dict
        Active-size DAG blocks (from ``sample_g_additive`` + ``_slice_g_active``).
    cfg : SCMPrior
        Supplies every prior range.
    structural : dict
        Output of :func:`sample_structure` (concrete families / smoothness /
        texture-enable flags).
    n_time_steps : int
        Reported weeks (the graph simulates ``n_time_steps + cfg.adstock_burn_in``).
    prior_cond : dict, optional
        Output of :func:`sample_prior_cond` — per-cell narrowed prior
        intervals ``{quantity: (low, width)}``. When given, the conditioned
        quantities (``adstock_alpha`` → the geometric decay,
        ``hill_shape`` → the Hill slope) are drawn as
        ``pm.Uniform(low, low + width)`` instead of their global supports.
        ``None`` (default) keeps the unconditioned priors.

    Returns
    -------
    (pm.Model, tuple[str, ...])
        The model (with all graph outputs registered as ``pm.Deterministic``)
        and the output names, in graph order.
    """
    n_treatments = len(g_active["g_cy"])
    n_covariates = len(g_active["g_zb"])
    n_latent = len(g_active["g_db"])
    burn_in = cfg.adstock_burn_in
    n_time_steps_full = n_time_steps + burn_in
    specs = _uniform_prior_specs(cfg, n_treatments, n_covariates, n_latent, prior_cond)

    with pm.Model() as model:
        rw = _walk_priors(cfg, structural, n_treatments, n_covariates, n_latent)
        rw_c = rw["rw_c"]

        c_level = pt.softplus(rw_c["mean"])  # per-channel level anchor for texture
        shock_outputs = _channel_shock_schedule(
            cfg, g_active["g_cy"], n_time_steps, burn_in, c_level
        )
        params = _scm_params(
            cfg,
            specs,
            rw,
            c_level,
            adstock_family=structural["adstock_family"],
            sat_family=structural["sat_family"],
            use_hf=structural["use_hf"],
            use_pulse=structural["use_pulse"],
            use_control_hf=structural["use_control_hf"],
            use_control_pulse=structural["use_control_pulse"],
        )
        _apply_outcome_std_scale(cfg, rw, g_active["g_cy"], params["beta"])
        if cfg.n_channel_shocks:
            params["channel_shock"] = {
                "mask_full": shock_outputs["channel_shock_mask_full"],
                "level_full": shock_outputs["level_full"],
            }

        eps = _scm_eps(
            n_time_steps_full,
            n_treatments,
            n_covariates,
            n_latent,
            params["pulse_prob"],
            params["control_pulse_prob"],
        )
        confounding_strength = _confounded_channel_eps(cfg, eps)

        graph = build_symbolic_graph(
            g_active,
            params,
            n_time_steps,
            n_treatments,
            n_covariates,
            n_latent,
            burn_in=burn_in,
            eps=eps,
        )
        graph["outputs"]["confounding_strength"] = confounding_strength
        if cfg.n_channel_shocks:
            graph["outputs"].update(
                {
                    key: shock_outputs[key]
                    for key in (
                        "channel_shock_mask",
                        "channel_shock_mask_full",
                        "channel_shock_channel",
                        "channel_shock_start",
                        "channel_shock_length",
                        "channel_shock_level_multiplier",
                        "channel_shock_level",
                    )
                }
            )
        else:
            graph["outputs"].update(
                _disabled_shock_outputs(n_time_steps, n_time_steps_full, n_treatments)
            )
        out_names = tuple(graph["outputs"].keys())
        for name in out_names:
            # Source RVs may already own an output name. Any other collision
            # would make ``model[name]`` silently resolve to the wrong tensor.
            output = graph["outputs"][name]
            existing = model.named_vars.get(name)
            if existing is None:
                pm.Deterministic(name, output)
            elif existing is not output:
                raise ValueError(f"graph output {name!r} collides with a different model variable")

        param_names = _register_param_reports(params, rw, c_level, confounding_strength)

    return model, out_names, param_names


def build_oracle_model(
    g_active: dict,
    cfg: SCMPrior,
    structural: dict,
    data: dict[str, np.ndarray],
    prior_cond: dict[str, tuple[float, float]] | None = None,
    *,
    latent: Literal["marginal", "sampled"] = "marginal",
) -> pm.Model:
    """The observed-data variant of :func:`build_world_model` — the posterior oracle.

    Builds a ``pm.Model`` for the SAME world structure with the world's
    dataset attached, so ``pm.sample`` yields the posterior structural
    parameters that form the identification floor an amortized model (e.g. a
    PFN) is judged against. The priors and the media response transforms are
    the same definitions generation uses (:func:`_uniform_prior_specs`,
    :func:`_walk_priors`, and the adstock/saturation code from
    :mod:`prior_generator.symbolic_graph`), so draw and oracle cannot drift.

    Parameters
    ----------
    g_active : dict
        Active-size DAG blocks — the TRUE structure (see the caveat below).
    cfg : SCMPrior
        Supplies every prior range (one prior definition for draw + oracle).
    structural : dict
        Output of :func:`sample_structure` for the drawn world (concrete
        mechanism families and walk smoothness). ``sample_scm`` records it in
        ``SCM.extras["structural"]``.
    data : dict
        The world's observables — ``"channels"`` (n_time_steps, n_treatments),
        ``"controls"`` (n_time_steps, n_covariates) and ``"sales"``
        (n_time_steps,) — e.g. straight from ``SCM.data``. The required
        ``"saturation_scale"`` (n_treatments,) pins the exact generation-time
        nonlinear response anchor. When channel shocks are enabled, it must
        also carry the world's known design metadata:
        ``channel_shock_channel``, ``channel_shock_start``,
        ``channel_shock_length``, ``channel_shock_level_multiplier``, and
        ``channel_shock_level``, each with one entry per configured shock, plus
        per-channel ``channel_level``. The absolute level must match both
        observed spend throughout its event window and the multiplier-relative
        channel level.
    prior_cond : dict, optional
        The world's prior-conditioning intervals (``SCM.extras["prior_cond"]``)
        so the oracle runs under the SAME narrowed prior the world was drawn
        from.
    latent : {"marginal", "sampled"}, default "marginal"
        ``"marginal"`` integrates the Gaussian ``RW_D``, ``RW_B``, and
        ``RW_Y`` paths analytically into the exact observed-window covariance.
        ``"sampled"`` preserves the previous latent-innovation representation
        and exposes posterior ``demand`` and ``baseline`` series.

    Returns
    -------
    pm.Model
        In marginal mode, free RVs are the outcome-side priors (``beta``,
        live mechanism shapes, ``delta_db``, ``rho_zb``, and walk parameters)
        without latent walk innovations. Deterministics ``contributions``
        (n_time_steps, n_treatments) and ``sales_mu`` (n_time_steps,) remain;
        ``sales_mu`` is ``E[sales | theta]`` and excludes latent walk
        realizations. Sampled mode additionally has ``eps_d`` / ``eps_b`` and
        deterministic ``demand`` (n_time_steps, n_latent) / ``baseline``
        (n_time_steps,), with the pre-existing ``sales_mu`` meaning.

    Notes
    -----
    **What is exact, and what is not.** The oracle keeps everything *upstream*
    of the observation exact and documents six explicit, mode-dependent
    qualifications:

    1. **Structure-known**: the true DAG, mechanism families and walk
       smoothness are given. This is the structure-known oracle — an upper
       bound for any method that must also infer structure; a
       structure-unknown oracle would marginalize over graphs and is out of
       scope.
    2. **Plug-in conditioning on the observed inputs**: ``channels`` and
       ``controls`` enter as data (constants). The information they carry
       about latent demand through ``p(C | D)`` / ``p(Z | D)`` is not modeled
       — including baseline information encoded through the channel–baseline
       correlation (rho, configured here as confounding strength). In sampled
       mode demand is inferred from the sales residual via ``D -> B`` only; in
       marginal mode that same demand path is integrated through the residual
       covariance. Neither mode posits ``p(C | eps_b)`` or claims exact
       conditioning.
    3. **Outcome-side Gaussian representation**: ``RW_Y`` is iid
       ``Normal(0, rw_y_std)`` in generation, so both oracle modes use its
       exact process rather than an iid-noise approximation. In the default
       ``latent="marginal"`` mode, ``RW_D`` and ``RW_B`` are integrated with
       their exact observed-window covariances and the iid ``RW_Y`` variance is
       added to the diagonal. ``latent="sampled"`` retains the exact
       full-horizon demand/baseline walk transforms and the same exact iid
       ``RW_Y`` likelihood. The ``1e-12 I`` covariance floor in marginal mode
       is only a factorization guard: its implied ``1e-6`` standard deviation
       is roughly ``1e-6`` of any realistic sales sd and cannot carry inference.
    4. **Posterior-series labels**: marginal mode has no ``demand`` or
       ``baseline`` deterministic. Its full-length ``sales_mu`` is
       ``E[sales | theta]`` and excludes every latent walk realization.
       Sampled mode's ``baseline`` is ``B`` (including its ``D -> B`` and
       ``Z -> B`` parent terms) without ``RW_Y``, while persisted
       ``data["baseline"]`` is ``B + RW_Y``; it is therefore not directly
       comparable. ``contributions`` is exactly comparable with
       ``world.data["contributions_observed"]`` in both modes; compare
       ``sales_mu`` with observed ``sales`` for total fit.
    5. **Reproducible likelihood window**: the oracle convolves only reported
       spend with a zero-padded start, whereas generation used real burn-in
       history. When burn-in is enabled and at least one direct channel has a
       non-identity adstock kernel, it therefore observes only
       ``sales[l_max - 1:]``. If every direct channel has identity adstock,
       persisted spend reproduces the full response and it observes all sales.
       ``contributions`` and ``sales_mu`` remain full-length deterministics in
       both modes; ``baseline`` is full-length in sampled mode only. For
       non-identity kernels at the truth, residual sd was 0.238 for weeks
       before ``l_max`` versus 0.0093 after, compared with
       ``rw_y_std=0.0126``; ``|z|`` reached 57 sigma and full-window sigma
       MLEs were inflated 1.65x–8.4x across five seeds. Held-level shocks are
       no exception: they clamp observed spend before the convolution and
       never touch response state, so ordinary carryover decays across a shock
       boundary exactly as it does anywhere else.
    6. **Weibull sampler downgrade**: pymc-marketing's ``weibull_adstock``
       min-max normalization contains a ``Min`` operation without a PyTensor
       pullback. When any channel uses Weibull adstock, ``pm.sample``
       therefore assigns Metropolis—not NUTS—to ``weibull_lam`` and
       ``weibull_k``. Identity and geometric adstock channels remain
       differentiable. This is upstream rather than from this package's
       analytic Weibull guard, whose reductions differentiate cleanly; replacing
       the library normalization would abandon load-bearing library parity.
       Metropolis mixing on those two carryover parameters makes their ESS less
       trustworthy, so prefer geometric-adstock worlds when using the oracle as
       a reference posterior.

    **Marginal-mode cost.** Each gradient evaluation factors an
    ``n × n`` covariance, ``n = n_time_steps - warmup``, so it has an
    ``O(n**3)`` Cholesky cost. That is cheap at weekly horizons and expensive
    for very long ``n_time_steps``; use ``latent="sampled"`` when posterior
    ``demand`` or ``baseline`` paths are needed.
    """
    if latent not in ("marginal", "sampled"):
        raise ValueError(f"latent must be 'marginal' or 'sampled', got {latent!r}")

    n_treatments = len(g_active["g_cy"])  # media channels (the interventions)
    n_covariates = len(g_active["g_zb"])  # observed controls
    n_latent = len(g_active["g_db"])  # hidden confounders
    channels = np.asarray(data["channels"], dtype="float64")
    controls = np.asarray(data["controls"], dtype="float64")
    sales = np.asarray(data["sales"], dtype="float64")
    if sales.ndim != 1:
        raise ValueError(f"data sales must have shape (n_time_steps,), got {sales.shape}")
    n_time_steps = int(sales.shape[0])
    if n_time_steps < 1:
        raise ValueError("data sales must contain at least one observation")
    if channels.shape != (n_time_steps, n_treatments) or controls.shape != (
        n_time_steps,
        n_covariates,
    ):
        raise ValueError(
            f"data shapes must be channels (n_time_steps, n_treatments)="
            f"{n_time_steps, n_treatments}, "
            f"controls (n_time_steps, n_covariates)={n_time_steps, n_covariates}, "
            f"sales (n_time_steps,)={(n_time_steps,)}; "
            f"got channels {channels.shape}, controls {controls.shape}"
        )
    if not all(np.isfinite(value).all() for value in (channels, controls, sales)):
        raise ValueError("data channels, controls, and sales must contain only finite values")
    if "saturation_scale" not in data:
        raise ValueError("data requires a saturation_scale")
    saturation_scale = np.asarray(data["saturation_scale"], dtype="float64")
    if saturation_scale.shape != (n_treatments,) or not (
        np.isfinite(saturation_scale).all() and (saturation_scale > 0.0).all()
    ):
        raise ValueError(
            "saturation_scale must be finite and positive with shape "
            f"{(n_treatments,)}, got {saturation_scale!r}"
        )
    g_cy = np.asarray(g_active["g_cy"], dtype="float64")
    adstock_family = np.asarray(structural["adstock_family"])
    burn_in = cfg.adstock_burn_in
    warmup = cfg.l_max - 1 if burn_in > 0 and np.any((g_cy != 0.0) & (adstock_family != 0)) else 0
    # K2 makes this unreachable after SCMPrior.validate(); retain it for callers
    # that invoke build_oracle_model directly with an unvalidated config.
    if warmup >= n_time_steps:
        raise ValueError(
            "oracle likelihood has no reproducible observations: "
            f"n_time_steps={n_time_steps} must exceed warmup={warmup} "
            f"(l_max={cfg.l_max}, adstock_burn_in={burn_in})"
        )
    n_time_steps_full = n_time_steps + burn_in
    window = slice(burn_in, None)
    rows = np.arange(burn_in + warmup, n_time_steps_full)
    g_db = np.asarray(g_active["g_db"], dtype="float64")
    g_zb = np.asarray(g_active["g_zb"], dtype="float64")
    specs = _uniform_prior_specs(cfg, n_treatments, n_covariates, n_latent, prior_cond)
    _validate_oracle_channel_shocks(cfg, g_cy, data, n_time_steps)
    mech_names = _live_mechanism_param_names(structural)

    def _walk_gram(smoothness: float) -> np.ndarray:
        width = _kernel_width(
            float(smoothness),
            n_time_steps_full,
            rw_smoothness_max_weeks=cfg.rw_smoothness_max_weeks,
        )
        basis = _walk_basis(n_time_steps_full, width)[rows]
        return np.asarray(basis @ basis.T)

    with pm.Model() as model:
        # Shared prior definitions — identical names, ranges and shapes to the
        # generative model (the drift-guard tests compare them one by one).
        rw = _walk_priors(
            cfg, structural, n_treatments, n_covariates, n_latent, include=("d", "b", "y")
        )
        beta = _uniform(*specs["beta"])
        _apply_outcome_std_scale(cfg, rw, g_cy, beta)
        delta_db = _uniform(*specs["delta_db"])
        rho_zb = _uniform(*specs["rho_zb"])
        mech: dict[str, Any] = {name: _uniform(*specs[name]) for name in mech_names}
        mech_params: dict[str, Any] = {
            "l_max": cfg.l_max,
            "adstock_family": structural["adstock_family"],
            "sat_family": structural["sat_family"],
            **{name: np.zeros(n_treatments) for name in _MECHANISM_PARAM_NAMES},
            **mech,
        }

        # Observed inputs enter as constants (static shapes — the adstock
        # convolution indexes by the static time length).
        channels_t = pt.as_tensor_variable(channels)

        # Media response on the OBSERVED spend: the same adstock / κ-relative
        # saturation code as generation. Held-level windows are already baked
        # into the observed channel matrix, so no schedule tensors are needed.
        contrib_cols = []
        for k in range(n_treatments):
            ad_obs = _adstock_col(channels_t[:, k], mech_params, k)
            scale_k = pt.as_tensor_variable(saturation_scale[k])
            f_obs = _saturate_col(ad_obs, scale_k, mech_params, k)
            contrib_cols.append((g_cy[k] * beta[k]) * f_obs)
        contributions = pm.Deterministic("contributions", pt.stack(contrib_cols, axis=1))
        term_bz = pt.dot(pt.as_tensor_variable(controls), g_zb * rho_zb)  # (n_time_steps,)

        if latent == "marginal":
            if cfg.baseline_floor is not None:
                raise ValueError(
                    "latent='marginal' cannot represent a floored intercept: "
                    "max(RW_B, baseline_floor) is not Gaussian, so marginalising "
                    "the baseline walk analytically would use the wrong "
                    "covariance. Use latent='sampled', which applies the same "
                    "floor as generation."
                )
            sales_mu = pm.Deterministic(
                "sales_mu",
                rw["rw_b"]["mean"][0] + term_bz + contributions.sum(axis=1),
            )
            covariance = (rw["rw_b"]["std"][0] ** 2) * pt.as_tensor_variable(
                _walk_gram(structural["smoothness_b"][0])
            )
            covariance = covariance + (rw["rw_y"]["std"][0] ** 2) * pt.eye(
                rows.size, dtype="float64"
            )
            for j in range(n_latent):
                if g_db[j] == 0.0:
                    continue
                loading = g_db[j] * delta_db[j]
                covariance = covariance + (loading**2) * pt.as_tensor_variable(
                    _walk_gram(structural["smoothness_d"][j])
                )
            # The floor only guards the float64 covariance factorization; it
            # is too small to provide material likelihood information.
            covariance = covariance + 1e-12 * pt.eye(rows.size, dtype="float64")
            pm.MvNormal(
                "sales",
                mu=sales_mu[warmup:],
                cov=covariance,
                observed=sales[warmup:],
            )
        else:
            # Latent demand + baseline walks: the SAME transform generation
            # uses, simulated over n_time_steps_full and sliced to the reported window.
            eps_d = pm.Normal("eps_d", 0.0, 1.0, shape=(n_time_steps_full, n_latent))
            eps_b = pm.Normal("eps_b", 0.0, 1.0, shape=(n_time_steps_full,))
            d_cols = [
                _walk_column(eps_d[:, j], rw["rw_d"], j, n_time_steps_full) for j in range(n_latent)
            ]
            D_full = pt.stack(d_cols, axis=1)  # (n_time_steps_full, n_latent)
            walk_b = _walk_column(eps_b, rw["rw_b"], 0, n_time_steps_full)
            pm.Deterministic("demand", D_full[window])

            # The SAME clip generation applies, so the oracle stays exactly the
            # generative model rather than an approximation of it.
            floor = cfg.baseline_floor

            def _clip(expr, _floor=floor):
                return expr if _floor is None else pt.maximum(expr, float(_floor))

            if floor is not None and cfg.baseline_floor_scope == "non_media":
                # Absorbing scope: clip the running total as each parent joins,
                # in the same locked order (confounders, then controls).
                running = _clip(walk_b[window])
                for j in range(n_latent):
                    running = _clip(running + (g_db[j] * delta_db[j]) * D_full[window][:, j])
                for m in range(n_covariates):
                    running = _clip(
                        running + (g_zb[m] * rho_zb[m]) * pt.as_tensor_variable(controls)[:, m]
                    )
                baseline = pm.Deterministic("baseline", running)
            else:
                term_bd = pt.dot(D_full[window], g_db * delta_db)  # (n_time_steps,)
                baseline = pm.Deterministic("baseline", term_bd + term_bz + _clip(walk_b)[window])
            sales_mu = pm.Deterministic("sales_mu", baseline + contributions.sum(axis=1))
            pm.Normal(
                "sales",
                mu=sales_mu[warmup:],
                sigma=rw["rw_y"]["std"][0],
                observed=sales[warmup:],
            )

    return model


def draw_worlds(
    model: pm.Model,
    out_names: tuple[str, ...],
    seed: int,
    draws: int = 1,
    mode: str = "FAST_COMPILE",
    *,
    rng_reference_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Draw ``draws`` worlds from a built model, seeded for reproducibility.

    Returns ``{name: array}`` where each array ALWAYS has a leading ``draws``
    axis — even at ``draws == 1`` (``pm.draw`` drops it, which we restore) — so
    callers can index world ``i`` as ``arr[i]`` regardless of ``draws``. ``mode``
    defaults to the python-backend ``FAST_COMPILE``: each world is a small
    one-off graph, so the C-backend compile cost of ``FAST_RUN`` dominates.

    ``rng_reference_names`` pins random-variable stream assignment to an older
    output contract while drawing an expanded set of audit outputs. Every RNG
    already reachable from the reference keeps its prior seeded stream; newly
    reachable RNGs are appended. This lets APIs expose additional realized
    inputs without changing existing seeded worlds.

    Compiled draw functions are cached per ``(model, out_names, mode, reference)``
    so repeated batches from the same cell avoid PyTensor recompilation.
    """
    with model:
        out_vars = [model[name] for name in out_names]
        reference_vars = (
            [model[name] for name in rng_reference_names] if rng_reference_names else None
        )
    draw_fn, ordered_rngs = _get_cached_draw_fn(
        model,
        out_vars,
        mode=mode,
        reference_vars=reference_vars,
    )
    vals = _execute_draws(draw_fn, ordered_rngs, seed=seed, draws=draws)

    # pm.draw drops the leading axis when draws == 1; restore it for a uniform
    # (draws, *shape) contract.
    return {
        name: (np.asarray(value)[None] if draws == 1 else np.asarray(value))
        for name, value in zip(out_names, vals)
    }


def draw_worlds_with_inputs(
    model: pm.Model,
    out_names: tuple[str, ...],
    input_names: tuple[str, ...],
    structure: dict[str, np.ndarray],
    seed: int,
    draws: int = 1,
    mode: str = "FAST_COMPILE",
    *,
    rng_reference_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Draw worlds with structure passed as compiled-function arguments.

    Same semantics as :func:`draw_worlds` after ``pm.set_data(structure)``, but
    structure arrays are explicit positional inputs to the compiled function.
    """
    with model:
        out_vars = [model[name] for name in out_names]
        input_vars = [model[name] for name in input_names]
        reference_vars = (
            [model[name] for name in rng_reference_names] if rng_reference_names else None
        )
    draw_fn, ordered_rngs = _get_cached_draw_fn(
        model,
        out_vars,
        mode=mode,
        reference_vars=reference_vars,
        input_vars=input_vars,
    )
    input_args = tuple(np.asarray(structure[name]) for name in input_names)
    vals = _execute_draws(draw_fn, ordered_rngs, seed=seed, draws=draws, input_args=input_args)
    return {
        name: (np.asarray(value)[None] if draws == 1 else np.asarray(value))
        for name, value in zip(out_names, vals)
    }
