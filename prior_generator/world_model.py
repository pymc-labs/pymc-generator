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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from . import mechanisms
from .sampler import CorpusConfig
from .symbolic_graph import build_symbolic_graph


def sample_structure(g_active: dict, cfg: CorpusConfig, rng: np.random.Generator) -> dict:
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


def _uniform(name: str, lo: float, hi: float, shape):
    """A ``pm.Uniform`` prior, or a constant when the range is degenerate (lo == hi)."""
    lo, hi = float(lo), float(hi)
    if lo == hi:
        return pt.as_tensor_variable(np.full(shape, lo, dtype="float64"))
    return pm.Uniform(name, lo, hi, shape=shape)


def build_world_model(
    g_active: dict, cfg: CorpusConfig, structural: dict, T: int
) -> tuple[pm.Model, tuple[str, ...]]:
    """Build the ``pm.Model`` for one world structure; return (model, output_names).

    Parameters
    ----------
    g_active : dict
        Active-size DAG blocks (from ``sample_g_additive`` + ``_slice_g_active``).
    cfg : CorpusConfig
        Supplies every prior range.
    structural : dict
        Output of :func:`sample_structure` (concrete families / smoothness /
        texture-enable flags).
    T : int
        Reported weeks (the graph simulates ``T + cfg.adstock_burn_in``).

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
    spr = mechanisms.SATURATION_PRIOR_RANGES

    with pm.Model() as model:

        def _rw(
            name, n, positive, mean_range, std_sigma, smoothness, std_range=None, relative=False
        ):
            mean = _uniform(f"{name}_mean", mean_range[0], mean_range[1], n)
            if std_range is not None:
                std = _uniform(f"{name}_std", std_range[0], std_range[1], n)
                if relative:  # scale-free: amplitude relative to the walk's level
                    std = std * pt.softplus(mean)
            else:
                std = pm.HalfNormal(f"{name}_std", sigma=std_sigma, shape=n)
            return {"mean": mean, "std": std, "smoothness": smoothness, "positive_only": positive}

        rw_d = _rw(
            "rw_d", J, False, cfg.rw_mean_range, cfg.rw_std_sigma, structural["smoothness_d"]
        )
        rw_z = _rw(
            "rw_z", M, False, cfg.rw_mean_range, cfg.rw_std_sigma, structural["smoothness_z"]
        )
        rw_c = _rw(
            "rw_c",
            K,
            True,
            cfg.rw_positive_mean_range,
            cfg.rw_channel_std_sigma,
            structural["smoothness_c"],
            std_range=cfg.rw_channel_std_range,
            relative=cfg.rw_channel_std_range is not None,
        )
        rw_b = _rw(
            "rw_b",
            1,
            False,
            cfg.rw_baseline_mean_range,
            cfg.rw_std_sigma,
            structural["smoothness_b"],
        )
        rw_y = _rw("rw_y", 1, False, (0.0, 0.0), cfg.rw_sales_std_sigma, structural["smoothness_y"])

        c_level = pt.softplus(rw_c["mean"])  # per-channel level anchor for texture
        pulse_prob = _uniform(
            "pulse_prob", cfg.channel_pulse_prob_range[0], cfg.channel_pulse_prob_range[1], K
        )

        params: dict[str, Any] = {
            "l_max": cfg.l_max,
            # linear edge coefficients
            "w_dc": _uniform("w_dc", cfg.dc_coeff_range[0], cfg.dc_coeff_range[1], (J, K)),
            "u_dz": _uniform("u_dz", cfg.dz_coeff_range[0], cfg.dz_coeff_range[1], (J, M)),
            "v_zc": _uniform("v_zc", cfg.zc_coeff_range[0], cfg.zc_coeff_range[1], (M, K)),
            "alpha_cc": _uniform("alpha_cc", cfg.cc_coeff_range[0], cfg.cc_coeff_range[1], (K, K)),
            "gamma_zz": _uniform("gamma_zz", cfg.zz_coeff_range[0], cfg.zz_coeff_range[1], (M, M)),
            "delta_db": _uniform("delta_db", cfg.db_coeff_range[0], cfg.db_coeff_range[1], J),
            "rho_zb": _uniform("rho_zb", cfg.zb_coeff_range[0], cfg.zb_coeff_range[1], M),
            "beta": _uniform("beta", cfg.beta_additive_range[0], cfg.beta_additive_range[1], K),
            # per-node random walks
            "rw_d": rw_d,
            "rw_z": rw_z,
            "rw_c": rw_c,
            "rw_b": rw_b,
            "rw_y": rw_y,
            # per-channel mechanism families (concrete) + shape priors
            "adstock_family": structural["adstock_family"],
            "sat_family": structural["sat_family"],
            "adstock_alpha": _uniform(
                "adstock_alpha", cfg.adstock_alpha_range[0], cfg.adstock_alpha_range[1], K
            ),
            "weibull_lam": _uniform(
                "weibull_lam", cfg.weibull_lam_range[0], cfg.weibull_lam_range[1], K
            ),
            "weibull_k": _uniform("weibull_k", cfg.weibull_k_range[0], cfg.weibull_k_range[1], K),
            "hill_slope": _uniform(
                "hill_slope", spr["hill"]["slope"][0], spr["hill"]["slope"][1], K
            ),
            "hill_kappa_mult": _uniform(
                "hill_kappa_mult", spr["hill"]["kappa_mult"][0], spr["hill"]["kappa_mult"][1], K
            ),
            "logistic_lam": _uniform(
                "logistic_lam", spr["logistic"]["lam"][0], spr["logistic"]["lam"][1], K
            ),
            "mm_alpha": _uniform(
                "mm_alpha",
                spr["michaelis_menten"]["alpha"][0],
                spr["michaelis_menten"]["alpha"][1],
                K,
            ),
            "mm_kappa_mult": _uniform(
                "mm_kappa_mult",
                spr["michaelis_menten"]["kappa_mult"][0],
                spr["michaelis_menten"]["kappa_mult"][1],
                K,
            ),
            "tanh_b": _uniform("tanh_b", spr["tanh"]["b"][0], spr["tanh"]["b"][1], K),
            "tanh_c": _uniform("tanh_c", spr["tanh"]["c"][0], spr["tanh"]["c"][1], K),
            "root_alpha": _uniform(
                "root_alpha", spr["root"]["alpha"][0], spr["root"]["alpha"][1], K
            ),
            # channel texture: magnitudes relative to the channel level; fires
            # are Bernoulli(pulse_prob)
            "hf_sigma": _uniform(
                "hf_sigma", cfg.channel_hf_sigma_range[0], cfg.channel_hf_sigma_range[1], K
            )
            * c_level,
            "pulse_amp": _uniform(
                "pulse_amp", cfg.channel_pulse_amp_range[0], cfg.channel_pulse_amp_range[1], K
            )
            * c_level,
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

        graph = build_symbolic_graph(g_active, params, T, K, M, J, burn_in=burn_in, eps=eps)
        out_names = tuple(graph["outputs"].keys())
        for name in out_names:
            pm.Deterministic(name, graph["outputs"][name])

    return model, out_names


def draw_worlds(
    model: pm.Model, out_names: tuple[str, ...], seed: int, draws: int = 1
) -> dict[str, np.ndarray]:
    """Draw ``draws`` worlds from a built model, seeded for reproducibility.

    Returns ``{name: array}`` where each array has a leading ``draws`` axis when
    ``draws > 1`` (``pm.draw`` drops it when ``draws == 1``).
    """
    with model:
        vals = pm.draw(
            [model[name] for name in out_names],
            draws=draws,
            random_seed=np.random.default_rng(seed),
        )
    return {name: np.asarray(v) for name, v in zip(out_names, vals)}
