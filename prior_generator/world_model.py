"""The world as a PyMC model: priors are distributions, worlds are draws.

``build_world_model`` assembles one :class:`pymc.Model` for a fixed causal
structure (a DAG ``g`` plus the concrete per-channel mechanism families and
per-node walk smoothness) in which every *continuous* SCM parameter is a PyMC
distribution and every noise term is an RV — ``pm.Normal`` walk innovations and
``pm.Normal`` weekly jitter, with campaign pulses as ``pm.Bernoulli``. The
structural equations and the exact interventional decomposition are the SAME
ones :func:`prior_generator.symbolic_graph.build_symbolic_graph` builds; this
module only supplies the priors + noise and exposes the outputs as
``pm.Deterministic`` so a single ``pm.draw`` yields params, series, and the full
decomposition jointly (and reproducibly from a seed).

Discrete/structural choices — which edges exist, each channel's adstock and
saturation family, and each node's walk smoothness — are drawn concretely per
world by :func:`sample_structure` (they set the graph's shape), matching the
design: continuous priors are distributions; structure is drawn per world.

:func:`build_oracle_model` is the observed-data variant: the same structure
and the same prior definitions with the world's dataset attached, so
``pm.sample`` yields the structure-known posterior on any drawn world — the
identification floor amortized models (PFNs) are judged against. The prior
definitions (:func:`_uniform_prior_specs`, :func:`_walk_priors`) are shared
between the generative and oracle builders so they cannot drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from . import mechanisms
from .sampler import SCMPrior
from .symbolic_graph import _adstock_col, _saturate_col, _walk_column, build_symbolic_graph


def sample_structure(g_active: dict, cfg: SCMPrior, rng: np.random.Generator) -> dict:
    """Draw the concrete per-world structure (families, smoothness, texture flags).

    These are the discrete/structural choices that set the graph's shape, drawn
    with numpy from ``cfg``'s family probabilities and smoothness Beta prior.
    The continuous parameters are NOT drawn here — they are PyMC distributions
    inside :func:`build_world_model`.
    """
    K = len(g_active["g_cy"])
    M = len(g_active["g_zb"])
    J = len(g_active["g_db"])
    ad_fam = rng.choice(
        len(cfg.adstock_family_probs), size=K, p=np.asarray(cfg.adstock_family_probs)
    )
    sat_fam = rng.choice(
        len(cfg.saturation_family_probs), size=K, p=np.asarray(cfg.saturation_family_probs)
    )

    def _smooth(n: int) -> np.ndarray:
        return rng.beta(cfg.rw_smoothness_alpha, cfg.rw_smoothness_beta, size=n)

    hf_on = float(cfg.channel_hf_sigma_range[1]) > 0.0
    pulse_on = float(cfg.channel_pulse_prob_range[1]) > 0.0
    return {
        "adstock_family": ad_fam.astype(int),
        "sat_family": sat_fam.astype(int),
        "smoothness_d": _smooth(J),
        "smoothness_z": _smooth(M),
        "smoothness_c": _smooth(K),
        "smoothness_b": _smooth(1),
        "smoothness_y": _smooth(1),
        "use_hf": np.full(K, hf_on),
        "use_pulse": np.full(K, pulse_on),
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
    cfg: SCMPrior, g_cy: np.ndarray, T: int, burn_in: int, c_level
) -> dict[str, Any]:
    """Build the symbolic audit schedule for configured channel shocks.

    The disjoint chronological slots make overlap impossible even when the
    same direct channel is selected repeatedly. This deliberately does not
    feed the schedule into the channel equations yet.
    """
    S = int(cfg.n_channel_shocks)
    K = len(g_cy)
    if S == 0:
        return {
            "channel_shock_mask": pt.zeros((T, K), dtype="int8"),
            "channel_shock_mask_full": pt.zeros((T + burn_in, K), dtype="int8"),
            "channel_shock_channel": pt.zeros((0,), dtype="int64"),
            "channel_shock_start": pt.zeros((0,), dtype="int64"),
            "channel_shock_length": pt.zeros((0,), dtype="int64"),
            "channel_shock_level_multiplier": pt.zeros((0,), dtype="float64"),
            "channel_shock_level": pt.zeros((0,), dtype="float64"),
        }

    direct = np.flatnonzero(np.asarray(g_cy) == 1).astype("int64")
    if not len(direct):
        raise ValueError("enabled channel shocks require at least one direct g_cy channel")
    len_lo, len_hi = cfg.channel_shock_length_range
    level_lo, level_hi = cfg.channel_shock_level_range
    if len_lo == len_hi:
        lengths = pt.as_tensor_variable(np.full(S, len_lo, dtype="int64"))
    else:
        lengths = pm.DiscreteUniform("channel_shock_length", len_lo, len_hi, shape=S)
    if len(direct) == 1:
        ranks = pt.zeros((S,), dtype="int64")
    else:
        ranks = pm.DiscreteUniform("channel_shock_channel_rank", 0, len(direct) - 1, shape=S)
    channels = pt.cast(pt.as_tensor_variable(direct)[ranks], "int64")
    if level_lo == level_hi:
        multipliers = pt.as_tensor_variable(np.full(S, level_lo, dtype="float64"))
    else:
        multipliers = pm.Uniform("channel_shock_level_multiplier", level_lo, level_hi, shape=S)

    starts = []
    for s in range(S):
        slot_lo = s * T // S
        slot_hi = (s + 1) * T // S
        # A fixed length that fills the slot has exactly one feasible start.
        if len_lo == len_hi and slot_hi - slot_lo == len_lo:
            starts.append(pt.as_tensor_variable(np.asarray(slot_lo, dtype="int64")))
        else:
            starts.append(
                pm.DiscreteUniform(f"channel_shock_start_{s}", slot_lo, slot_hi - lengths[s])
            )
    starts_t = pt.stack(starts)
    levels = multipliers * c_level[channels]

    def _mask(n_time: int, offset: int):
        time = pt.arange(n_time)[:, None]
        active = (time >= (starts_t + offset)[None, :]) & (
            time < (starts_t + lengths + offset)[None, :]
        )
        selected = pt.eq(pt.arange(K)[:, None], channels[None, :]).T
        return pt.cast(pt.any(active[:, :, None] & selected[None, :, :], axis=1), "int8")

    return {
        "channel_shock_mask": _mask(T, 0),
        "channel_shock_mask_full": _mask(T + burn_in, burn_in),
        "channel_shock_channel": channels,
        "channel_shock_start": starts_t,
        "channel_shock_length": lengths,
        "channel_shock_level_multiplier": multipliers,
        "channel_shock_level": levels,
    }


def _rw_prior_group(
    name, n, positive, mean_range, std_sigma, smoothness, std_range=None, relative=False
):
    """One node group's random-walk priors (mean + std RVs, concrete smoothness).

    Must be called inside a ``pm.Model`` context. This is THE single
    definition of the walk priors — the generative draw and the posterior
    oracle both build their walk parameters here, so they cannot drift.
    """
    mean = _uniform(f"{name}_mean", mean_range[0], mean_range[1], n)
    if std_range is not None:
        std = _uniform(f"{name}_std", std_range[0], std_range[1], n)
        if relative:  # scale-free: amplitude relative to the walk's level
            std = std * pt.softplus(mean)
    else:
        std = pm.HalfNormal(f"{name}_std", sigma=std_sigma, shape=n)
    return {"mean": mean, "std": std, "smoothness": smoothness, "positive_only": positive}


def _walk_priors(
    cfg: SCMPrior,
    structural: dict,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    include: tuple[str, ...] = ("d", "z", "c", "b", "y"),
) -> dict[str, dict]:
    """Walk-prior groups per node type, registered in the LOCKED d/z/c/b/y order.

    Sizes follow the :class:`SCMPrior` vocabulary — ``n_treatments`` media
    channels, ``n_covariates`` observed controls, ``n_latent`` hidden
    confounders. ``include`` selects the groups a model needs (the oracle
    skips the ones replaced by observed data); relative order is always
    preserved.
    """
    out: dict[str, dict] = {}
    if "d" in include:
        out["rw_d"] = _rw_prior_group(
            "rw_d",
            n_latent,
            False,
            cfg.rw_mean_range,
            cfg.rw_std_sigma,
            structural["smoothness_d"],
        )
    if "z" in include:
        out["rw_z"] = _rw_prior_group(
            "rw_z",
            n_covariates,
            False,
            cfg.rw_mean_range,
            cfg.rw_std_sigma,
            structural["smoothness_z"],
        )
    if "c" in include:
        out["rw_c"] = _rw_prior_group(
            "rw_c",
            n_treatments,
            True,
            cfg.rw_positive_mean_range,
            cfg.rw_channel_std_sigma,
            structural["smoothness_c"],
            std_range=cfg.rw_channel_std_range,
            relative=cfg.rw_channel_std_range is not None,
        )
    if "b" in include:
        out["rw_b"] = _rw_prior_group(
            "rw_b",
            1,
            False,
            cfg.rw_baseline_mean_range,
            cfg.rw_baseline_std_sigma_effective,
            structural["smoothness_b"],
        )
    if "y" in include:
        out["rw_y"] = _rw_prior_group(
            "rw_y", 1, False, (0.0, 0.0), cfg.rw_sales_std_sigma, structural["smoothness_y"]
        )
    return out


#: The per-channel media-response shape params (spec keys), canonical order.
_MECHANISM_PARAM_NAMES: tuple[str, ...] = (
    "adstock_alpha",
    "weibull_lam",
    "weibull_k",
    "hill_slope",
    "hill_kappa_mult",
    "logistic_lam",
    "mm_alpha",
    "mm_kappa_mult",
    "tanh_b",
    "tanh_c",
    "root_alpha",
)


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
        "mm_alpha": (
            "mm_alpha",
            spr["michaelis_menten"]["alpha"][0],
            spr["michaelis_menten"]["alpha"][1],
            n_t,
        ),
        "mm_kappa_mult": (
            "mm_kappa_mult",
            spr["michaelis_menten"]["kappa_mult"][0],
            spr["michaelis_menten"]["kappa_mult"][1],
            n_t,
        ),
        "tanh_b": ("tanh_b", spr["tanh"]["b"][0], spr["tanh"]["b"][1], n_t),
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
    }


def build_world_model(
    g_active: dict,
    cfg: SCMPrior,
    structural: dict,
    T: int,
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
    T : int
        Reported weeks (the graph simulates ``T + cfg.adstock_burn_in``).
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
    K = len(g_active["g_cy"])
    M = len(g_active["g_zb"])
    J = len(g_active["g_db"])
    burn_in = cfg.adstock_burn_in
    T_full = T + burn_in
    specs = _uniform_prior_specs(cfg, K, M, J, prior_cond)

    with pm.Model() as model:
        rw = _walk_priors(cfg, structural, K, M, J)
        rw_c = rw["rw_c"]

        c_level = pt.softplus(rw_c["mean"])  # per-channel level anchor for texture
        shock_outputs = _channel_shock_schedule(cfg, g_active["g_cy"], T, burn_in, c_level)
        pulse_prob = _uniform(*specs["pulse_prob"])

        params: dict[str, Any] = {
            "l_max": cfg.l_max,
            # linear edge coefficients
            "w_dc": _uniform(*specs["w_dc"]),
            "u_dz": _uniform(*specs["u_dz"]),
            "v_zc": _uniform(*specs["v_zc"]),
            "alpha_cc": _uniform(*specs["alpha_cc"]),
            "gamma_zz": _uniform(*specs["gamma_zz"]),
            "delta_db": _uniform(*specs["delta_db"]),
            "rho_zb": _uniform(*specs["rho_zb"]),
            "beta": _uniform(*specs["beta"]),
            # per-node random walks
            "rw_d": rw["rw_d"],
            "rw_z": rw["rw_z"],
            "rw_c": rw_c,
            "rw_b": rw["rw_b"],
            "rw_y": rw["rw_y"],
            # per-channel mechanism families (concrete) + shape priors
            "adstock_family": structural["adstock_family"],
            "sat_family": structural["sat_family"],
            **{name: _uniform(*specs[name]) for name in _MECHANISM_PARAM_NAMES},
            # channel texture: magnitudes relative to the channel level; fires
            # are Bernoulli(pulse_prob)
            "hf_sigma": _uniform(*specs["hf_sigma"]) * c_level,
            "pulse_amp": _uniform(*specs["pulse_amp"]) * c_level,
            "pulse_prob": pulse_prob,
            "use_hf": structural["use_hf"],
            "use_pulse": structural["use_pulse"],
        }

        eps = {
            "eps_d": pm.Normal("eps_d", 0.0, 1.0, shape=(T_full, J)),
            "eps_z": pm.Normal("eps_z", 0.0, 1.0, shape=(T_full, M)),
            "eps_c": pm.Normal("eps_c", 0.0, 1.0, shape=(T_full, K)),
            "eps_b": pm.Normal("eps_b", 0.0, 1.0, shape=(T_full,)),
            "eps_y": pm.Normal("eps_y", 0.0, 1.0, shape=(T_full,)),
            "eps_c_hf": pm.Normal("eps_c_hf", 0.0, 1.0, shape=(T_full, K)),
            "eps_c_pulse": pm.Bernoulli(
                "eps_c_pulse", p=pt.broadcast_to(pulse_prob, (T_full, K)), shape=(T_full, K)
            ).astype("float64"),
        }

        # Preserve the legacy innovation dictionary and graph path exactly when
        # the feature is disabled. When enabled, rho is a single per-world
        # value and only the channel innovation supplied to the graph changes.
        # The orthonormal mixture leaves every channel innovation's marginal
        # variance at one while correlating it with the baseline innovation.
        if cfg.confounding_strength_range is None:
            confounding_strength = pt.as_tensor_variable(np.asarray(0.0, dtype="float64"))
        else:
            lo, hi = cfg.confounding_strength_range
            if float(lo) == float(hi):
                confounding_strength = pt.as_tensor_variable(np.asarray(lo, dtype="float64"))
            else:
                confounding_strength = pm.Uniform("confounding_strength", lo, hi)
            eps["eps_c"] = (
                pt.sqrt(1.0 - confounding_strength**2) * eps["eps_c"]
                + confounding_strength * eps["eps_b"][:, None]
            )

        graph = build_symbolic_graph(g_active, params, T, K, M, J, burn_in=burn_in, eps=eps)
        graph["outputs"]["confounding_strength"] = confounding_strength
        graph["outputs"].update(shock_outputs)
        out_names = tuple(graph["outputs"].keys())
        for name in out_names:
            # A nondegenerate confounding strength is itself the named Uniform
            # RV. It is already drawable by this output name; wrapping it in a
            # Deterministic would register the same name twice.
            if name not in model.named_vars:
                pm.Deterministic(name, graph["outputs"][name])

        # Register the continuous params that world descriptions / bundles
        # report (edge coefficients + per-channel mechanism/texture params) as
        # "param_*" deterministics, so a single draw yields their concrete
        # per-world values alongside the series. Families / smoothness are
        # concrete already (from `structural`) and are not drawn here.
        report_specs = {
            "beta": params["beta"],
            "w_dc": params["w_dc"],
            "u_dz": params["u_dz"],
            "v_zc": params["v_zc"],
            "alpha_cc": params["alpha_cc"],
            "gamma_zz": params["gamma_zz"],
            "delta_db": params["delta_db"],
            "rho_zb": params["rho_zb"],
            "adstock_alpha": params["adstock_alpha"],
            "weibull_lam": params["weibull_lam"],
            "weibull_k": params["weibull_k"],
            "hf_sigma": params["hf_sigma"],
            "pulse_amp": params["pulse_amp"],
            "pulse_prob": params["pulse_prob"],
            "rw_c_mean": rw_c["mean"],
            "channel_level": c_level,
            "rw_c_std": rw_c["std"],
            "confounding_strength": confounding_strength,
        }
        param_names = tuple(f"param_{k}" for k in report_specs)
        for key, tensor in report_specs.items():
            pm.Deterministic(f"param_{key}", tensor)

    return model, out_names, param_names


def build_oracle_model(
    g_active: dict,
    cfg: SCMPrior,
    structural: dict,
    data: dict[str, np.ndarray],
    prior_cond: dict[str, tuple[float, float]] | None = None,
) -> pm.Model:
    """The observed-data variant of :func:`build_world_model` — the NUTS oracle.

    Builds a ``pm.Model`` for the SAME world structure with the world's
    dataset attached, so ``pm.sample`` yields the posterior over the
    structural parameters and the latent demand — the identification floor an
    amortized model (e.g. a PFN) is judged against. The priors and the media
    response transforms are the same definitions generation uses
    (:func:`_uniform_prior_specs`, :func:`_walk_priors`, and the
    adstock/saturation code from :mod:`prior_generator.symbolic_graph`), so
    draw and oracle cannot drift.

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
        The world's observables — ``"channels"`` (T, K), ``"controls"``
        (T, M) and ``"sales"`` (T,) — e.g. straight from ``SCM.data``.
    prior_cond : dict, optional
        The world's prior-conditioning intervals (``SCM.extras["prior_cond"]``)
        so the oracle runs under the SAME narrowed prior the world was drawn
        from.

    Returns
    -------
    pm.Model
        Free RVs: the outcome-side priors (``beta``, mechanism shapes,
        ``delta_db``, ``rho_zb``, walk params) and the latent demand / baseline
        walk innovations. Deterministics ``contributions`` (T, K),
        ``baseline`` (T,), ``sales_mu`` (T,) and ``demand`` (T, J) expose the
        posterior series; compare ``contributions`` against the world's
        ``contributions_observed`` truth.

    Notes
    -----
    **What is exact, and what is not.** Conditioning this generative process
    exactly is not possible: every random walk is normalized in-place
    (``walk * std / walk.std()``), so the sales-noise walk has no closed-form
    density to invert. The oracle keeps everything *upstream* of the
    observation exact and makes three explicit, documented concessions:

    1. **Structure-known**: the true DAG, mechanism families and walk
       smoothness are given. This is the structure-known oracle — an upper
       bound for any method that must also infer structure; a
       structure-unknown oracle would marginalize over graphs and is out of
       scope.
    2. **Plug-in conditioning on the observed inputs**: ``channels`` and
       ``controls`` enter as data (constants). The information they carry
       about latent demand through ``p(C | D)`` / ``p(Z | D)`` is not modeled
       — demand is inferred from the sales residual via ``D -> B`` only.
    3. **iid sales-noise representation**: the generative sales noise
       ``RW_Y`` (a smoothed, normalized walk with marginal sd exactly
       ``rw_y_std``) is represented as iid ``Normal(0, rw_y_std)`` with the
       SAME HalfNormal prior on the scale. The latent demand and baseline
       walks stay exact (same ``T_full`` simulation, same transform, sliced
       to the reported window).

    Additionally the adstock convolution sees only the reported window
    (zero-padded start) while generation used ``adstock_burn_in`` weeks of
    real history — drop the first ``l_max`` weeks from comparisons.
    """
    n_treatments = len(g_active["g_cy"])  # media channels (the interventions)
    n_covariates = len(g_active["g_zb"])  # observed controls
    n_latent = len(g_active["g_db"])  # hidden confounders
    channels = np.asarray(data["channels"], dtype="float64")
    controls = np.asarray(data["controls"], dtype="float64")
    sales = np.asarray(data["sales"], dtype="float64")
    T = int(sales.shape[0])
    if channels.shape != (T, n_treatments) or controls.shape != (T, n_covariates):
        raise ValueError(
            f"data shapes must be channels (T, n_treatments)={T, n_treatments}, "
            f"controls (T, n_covariates)={T, n_covariates}, sales (T,)={(T,)}; "
            f"got channels {channels.shape}, controls {controls.shape}"
        )
    burn_in = cfg.adstock_burn_in
    T_full = T + burn_in
    W = slice(burn_in, None)
    g_cy = np.asarray(g_active["g_cy"], dtype="float64")
    g_db = np.asarray(g_active["g_db"], dtype="float64")
    g_zb = np.asarray(g_active["g_zb"], dtype="float64")
    specs = _uniform_prior_specs(cfg, n_treatments, n_covariates, n_latent, prior_cond)

    with pm.Model() as model:
        # Shared prior definitions — identical names, ranges and shapes to the
        # generative model (the drift-guard tests compare them one by one).
        rw = _walk_priors(
            cfg, structural, n_treatments, n_covariates, n_latent, include=("d", "b", "y")
        )
        beta = _uniform(*specs["beta"])
        delta_db = _uniform(*specs["delta_db"])
        rho_zb = _uniform(*specs["rho_zb"])
        mech: dict[str, Any] = {name: _uniform(*specs[name]) for name in _MECHANISM_PARAM_NAMES}
        mech_params: dict[str, Any] = {
            "l_max": cfg.l_max,
            "adstock_family": structural["adstock_family"],
            "sat_family": structural["sat_family"],
            **mech,
        }

        # Observed inputs enter as constants (static shapes — the adstock
        # convolution indexes by the static time length).
        channels_t = pt.as_tensor_variable(channels)

        # Latent demand + baseline walks: the SAME transform generation uses,
        # simulated over T_full and sliced to the reported window.
        eps_d = pm.Normal("eps_d", 0.0, 1.0, shape=(T_full, n_latent))
        eps_b = pm.Normal("eps_b", 0.0, 1.0, shape=(T_full,))
        d_cols = [_walk_column(eps_d[:, j], rw["rw_d"], j, T_full) for j in range(n_latent)]
        D_full = pt.stack(d_cols, axis=1)  # (T_full, n_latent)
        walk_b = _walk_column(eps_b, rw["rw_b"], 0, T_full)
        pm.Deterministic("demand", D_full[W])

        term_bd = pt.dot(D_full[W], g_db * delta_db)  # (T,)
        term_bz = pt.dot(pt.as_tensor_variable(controls), g_zb * rho_zb)  # (T,)
        baseline = pm.Deterministic("baseline", term_bd + term_bz + walk_b[W])

        # Media response on the OBSERVED spend: same adstock / κ-relative
        # saturation code as generation (window-only history — see Notes).
        contrib_cols = []
        for k in range(n_treatments):
            ad_obs = _adstock_col(channels_t[:, k], mech_params, k)
            scale_k = pt.maximum(ad_obs.mean(), 1e-8)
            f_obs = _saturate_col(ad_obs, scale_k, mech_params, k)
            contrib_cols.append((g_cy[k] * beta[k]) * f_obs)
        contributions = pm.Deterministic("contributions", pt.stack(contrib_cols, axis=1))

        sales_mu = pm.Deterministic("sales_mu", baseline + contributions.sum(axis=1))
        # iid representation of the RW_Y sales noise (same HalfNormal scale
        # prior; see Notes on why the normalized walk cannot be inverted).
        pm.Normal("sales", mu=sales_mu, sigma=rw["rw_y"]["std"][0], observed=sales)

    return model


def draw_worlds(
    model: pm.Model,
    out_names: tuple[str, ...],
    seed: int,
    draws: int = 1,
    mode: str = "FAST_COMPILE",
) -> dict[str, np.ndarray]:
    """Draw ``draws`` worlds from a built model, seeded for reproducibility.

    Returns ``{name: array}`` where each array ALWAYS has a leading ``draws``
    axis — even at ``draws == 1`` (``pm.draw`` drops it, which we restore) — so
    callers can index world ``i`` as ``arr[i]`` regardless of ``draws``. ``mode``
    defaults to the python-backend ``FAST_COMPILE``: each world is a small
    one-off graph, so the C-backend compile cost of ``FAST_RUN`` dominates.
    """
    with model:
        vals = pm.draw(
            [model[name] for name in out_names],
            draws=draws,
            random_seed=np.random.default_rng(seed),
            mode=mode,
        )
    # pm.draw drops the leading axis when draws == 1; restore it for a uniform
    # (draws, *shape) contract.
    return {
        name: (np.asarray(v)[None] if draws == 1 else np.asarray(v))
        for name, v in zip(out_names, vals)
    }
