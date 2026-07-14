"""PyTensor symbolic causal-graph builder for the Phase-4 additive rung.

Builds a Pearlian additive SCM (plan doc 03, D1/D5) with inter-variable
interactions as one symbolic PyTensor graph:

.. code-block:: text

    D_j = RW_j                                        (confounder, signed)
    Z_m = Σ_j u_jm·D_j + Σ_{m'<m} γ_{m'm}·Z_{m'} + RW_m   (control, signed)
    E_k = RW_k + σ_k·ε_tk + a_k·b_tk,  b_tk ~ Bernoulli(p_k)  (channel own drive)
    C_k = softplus( Σ_j w_jk·D_j + Σ_m v_mk·Z_m
                    + Σ_{k'<k} α_{k'k}·C_{k'} + E_k )     (channel, positive)
    B   = Σ_j δ_j·D_j + Σ_m ρ_m·Z_m + RW_B                (baseline, signed)
    Y   = B + Σ_k g_cy·β_k·f_k(C_k) + RW_Y                (sales)

Every node carries an independent random-walk noise term (``random_walk``
module); node means are folded into the walks. The channel own drive
``E_k`` additionally carries iid weekly execution noise (σ_k) and
campaign pulses (amplitude a_k, per-week probability P(ε' > z_k)) — the
high-frequency exogenous variation that lets spend sweep its response
curve (without it, contribution targets degenerate to flat lines; the
neutral defaults σ_k = 0, p_k = 0 disable both). C→C and Z→Z edges
are restricted to the strict upper triangle (src index < dst index) which
guarantees acyclicity. The nonlinear transform ``f_k`` (adstock +
saturation, from ``mechanisms``) applies only on the direct C→Y path;
all inter-variable effects are linear (plan doc D2), with a softplus
positivity guard on channels (risk table: spend must be non-negative).

Exact intervention-based decomposition (plan doc D3/D7)
-------------------------------------------------------
``C_base`` is the channel system under the intervention *zero all incoming
channel interactions* (D→C, Z→C, C→C) — i.e. each channel driven only by
its own random walk. Direct contributions and indirect effects are:

.. code-block:: text

    contributions_k  = g_cy·β_k·f_k(C_base_k)                (direct)
    indirect_effects = Σ_k g_cy·β_k·(f_k(C_k) - f_k(C_base_k))
    baseline_out     = B + RW_Y
    sales           == baseline_out + Σ_k contributions_k + indirect_effects

The identity holds *exactly* (not a Taylor approximation) because both Y
and the decomposition are built from the same symbolic quantities. So that
``f_k`` is one fixed function evaluated on two inputs, the κ-relative
saturation scale is computed from the **observed** adstocked channel and
reused for the base channel.

:func:`compute_indirect_effects` additionally supports arbitrary
intervention sets via ``pytensor.graph.replace.clone_replace`` — zero any
named hook group, re-evaluate Y on the same noise, and measure ΔY.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.tensor import TensorVariable

from . import mechanisms
from .random_walk import sample_rw_params, softplus, symbolic_random_walk

__all__ = [
    "sample_scm_params",
    "build_symbolic_graph",
    "compile_graph",
    "draw_from_graph",
    "compute_indirect_effects",
    "INTERVENTION_HOOK_GROUPS",
]

#: Hook groups available for interventions: parent-term variables that
#: :func:`compute_indirect_effects` can replace with zeros.
INTERVENTION_HOOK_GROUPS: tuple[str, ...] = (
    "z_from_d",  # D -> Z terms
    "z_from_z",  # Z -> Z terms
    "c_from_d",  # D -> C terms
    "c_from_z",  # Z -> C terms
    "c_from_c",  # C -> C terms
    "b_from_d",  # D -> B terms
    "b_from_z",  # Z -> B terms
)

# Adstock family ids (match sampler L1 convention): 0=none, 1=geometric, 2=weibull
# Saturation family ids: 0=none(linear), 1=hill, 2=logistic, 3=michaelis_menten,
# 4=tanh, 5=root
_SAT_FAMILY_NAMES = ("none", "hill", "logistic", "michaelis_menten", "tanh", "root")


def _check_strict_upper(mat: np.ndarray, name: str) -> None:
    """Raise if any entry on/below the diagonal is nonzero (acyclicity)."""
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {mat.shape}")
    if np.tril(mat).any():
        raise ValueError(
            f"{name} must be strictly upper-triangular (src index < dst index) "
            "to guarantee acyclicity; found nonzero entries on/below the diagonal"
        )


def sample_scm_params(
    g: dict[str, np.ndarray],
    T: int,
    K: int,
    M: int,
    J: int,
    rng: np.random.Generator,
    *,
    l_max: int = 8,
    # -- linear coefficient ranges (plan doc 4.5 defaults) -------------------
    dc_coeff_range: tuple[float, float] = (0.1, 0.5),
    dz_coeff_range: tuple[float, float] = (0.1, 0.5),
    zc_coeff_range: tuple[float, float] = (0.05, 0.3),
    cc_coeff_range: tuple[float, float] = (0.05, 0.3),
    zz_coeff_range: tuple[float, float] = (-0.2, 0.2),
    db_coeff_range: tuple[float, float] = (0.15, 0.45),
    zb_coeff_range: tuple[float, float] = (0.1, 0.4),
    beta_range: tuple[float, float] = (0.5, 2.0),
    # -- random-walk priors ---------------------------------------------------
    rw_mean_range: tuple[float, float] = (-1.0, 1.0),
    rw_positive_mean_range: tuple[float, float] = (0.5, 3.0),
    rw_baseline_mean_range: tuple[float, float] = (3.0, 8.0),
    rw_std_sigma: float = 1.0,
    rw_channel_std_sigma: float = 0.6,
    rw_sales_std_sigma: float = 0.25,
    rw_smoothness_alpha: float = 2.0,
    rw_smoothness_beta: float = 2.0,
    # -- channel texture: walk-std floor, iid weekly noise, campaign pulses ---
    # Defaults are legacy-neutral: None / (0, 0) draw nothing and consume no
    # RNG, so corpora generated before this knob existed reproduce exactly.
    rw_channel_std_range: tuple[float, float] | None = None,
    channel_hf_sigma_range: tuple[float, float] = (0.0, 0.0),
    channel_pulse_prob_range: tuple[float, float] = (0.0, 0.0),
    channel_pulse_amp_range: tuple[float, float] = (0.5, 1.5),
    # -- mechanism family priors (same convention as sampler L1) -------------
    adstock_family_probs: tuple[float, ...] = (0.15, 0.425, 0.425),
    saturation_family_probs: tuple[float, ...] = (0.15, 0.17, 0.17, 0.17, 0.17, 0.17),
    adstock_alpha_range: tuple[float, float] = (0.2, 0.8),
    weibull_lam_range: tuple[float, float] = (2.0, 8.0),
    weibull_k_range: tuple[float, float] = (1.5, 4.0),
) -> dict[str, Any]:
    """Sample all concrete SCM parameters for one task (plan doc 4.1).

    The g-vector fixes *which* edges exist; this draws the edge
    coefficients, per-node random-walk parameters, and per-channel
    mechanism (adstock + saturation) families and shapes.

    Returns a plain dict of numpy arrays / scalars consumed by
    :func:`build_symbolic_graph`.
    """
    _check_strict_upper(g["g_cc"], "g_cc")
    _check_strict_upper(g["g_zz"], "g_zz")

    def _unif(rng_range: tuple[float, float], size) -> np.ndarray:
        return np.asarray(rng.uniform(rng_range[0], rng_range[1], size=size))

    def _rw_group(
        n: int, positive_only: bool, mean_range, std_sigma, std_range=None, std_relative=False
    ) -> dict[str, Any]:
        params = [
            sample_rw_params(
                positive_only=positive_only,
                rng=rng,
                mean_range=mean_range,
                positive_mean_range=mean_range,
                std_sigma=std_sigma,
                smoothness_alpha=rw_smoothness_alpha,
                smoothness_beta=rw_smoothness_beta,
                std_range=std_range,
                std_relative=std_relative,
            )
            for _ in range(n)
        ]
        return {
            "mean": np.array([p["mean"] for p in params]),
            "std": np.array([p["std"] for p in params]),
            "smoothness": np.array([p["smoothness"] for p in params]),
            "positive_only": positive_only,
        }

    sat_fam = rng.choice(
        len(saturation_family_probs), size=K, p=np.asarray(saturation_family_probs)
    )
    ad_fam = rng.choice(len(adstock_family_probs), size=K, p=np.asarray(adstock_family_probs))
    spr = mechanisms.SATURATION_PRIOR_RANGES

    params: dict[str, Any] = {
        "l_max": int(l_max),
        # linear edge coefficients (gated later by the g masks)
        "w_dc": _unif(dc_coeff_range, (J, K)),
        "u_dz": _unif(dz_coeff_range, (J, M)),
        "v_zc": _unif(zc_coeff_range, (M, K)),
        "alpha_cc": _unif(cc_coeff_range, (K, K)),
        "gamma_zz": _unif(zz_coeff_range, (M, M)),
        "delta_db": _unif(db_coeff_range, J),
        "rho_zb": _unif(zb_coeff_range, M),
        "beta": _unif(beta_range, K),
        # per-node random walks
        "rw_d": _rw_group(J, False, rw_mean_range, rw_std_sigma),
        "rw_z": _rw_group(M, False, rw_mean_range, rw_std_sigma),
        "rw_c": _rw_group(
            K,
            True,
            rw_positive_mean_range,
            rw_channel_std_sigma,
            std_range=rw_channel_std_range,
            std_relative=rw_channel_std_range is not None,  # range reads as a CV range
        ),
        "rw_b": _rw_group(1, False, rw_baseline_mean_range, rw_std_sigma),
        "rw_y": _rw_group(1, False, (0.0, 0.0), rw_sales_std_sigma),
        # per-channel mechanism families + shapes
        "adstock_family": ad_fam.astype(int),
        "sat_family": sat_fam.astype(int),
        "adstock_alpha": _unif(adstock_alpha_range, K),
        "weibull_lam": _unif(weibull_lam_range, K),
        "weibull_k": _unif(weibull_k_range, K),
        "hill_slope": _unif(spr["hill"]["slope"], K),
        "hill_kappa_mult": _unif(spr["hill"]["kappa_mult"], K),
        "logistic_lam": _unif(spr["logistic"]["lam"], K),
        "mm_alpha": _unif(spr["michaelis_menten"]["alpha"], K),
        "mm_kappa_mult": _unif(spr["michaelis_menten"]["kappa_mult"], K),
        "tanh_b": _unif(spr["tanh"]["b"], K),
        "tanh_c": _unif(spr["tanh"]["c"], K),
        "root_alpha": _unif(spr["root"]["alpha"], K),
    }

    # Channel texture: per-channel iid weekly noise and campaign pulses on the
    # pre-softplus own drive. The drawn factors are RELATIVE — multiplied by
    # the channel's own level (softplus of its walk mean, the same anchor
    # sample_rw_params uses for std_relative) so variation is scale-free
    # across small and large channels, like L1's log-space spend noise.
    # Sampled AFTER the dict above so (a) the walk means exist and (b)
    # degenerate ranges draw NOTHING, keeping the legacy RNG stream (and thus
    # existing corpora) byte-identical. Gates compare NUMERICALLY (a config
    # that round-trips through JSON arrives with lists, and a list never
    # equals a tuple — that must not silently enable the texture draws).
    hf_lo, hf_hi = (float(v) for v in channel_hf_sigma_range)
    p_lo, p_hi = (float(v) for v in channel_pulse_prob_range)
    if not 0.0 <= hf_lo <= hf_hi:
        raise ValueError(f"channel_hf_sigma_range must satisfy 0 <= lo <= hi, got {(hf_lo, hf_hi)}")
    if not 0.0 <= p_lo <= p_hi < 1.0:
        raise ValueError(
            f"channel_pulse_prob_range must satisfy 0 <= lo <= hi < 1, got {(p_lo, p_hi)}"
        )
    c_level = softplus(np.asarray(params["rw_c"]["mean"], dtype="float64"))
    if hf_hi > 0.0:
        hf_sigma = _unif((hf_lo, hf_hi), K) * c_level
    else:
        hf_sigma = np.zeros(K)
    if p_hi > 0.0:
        # The pulse fires each week as Bernoulli(pulse_prob); build_symbolic_graph
        # adds pulse_amp * fire (fires drawn by draw_eps / a pm.Bernoulli RV).
        pulse_prob = _unif((p_lo, p_hi), K)
        pulse_amp = _unif(channel_pulse_amp_range, K) * c_level
    else:
        pulse_prob = np.zeros(K)
        pulse_amp = np.zeros(K)
    params.update({"hf_sigma": hf_sigma, "pulse_prob": pulse_prob, "pulse_amp": pulse_amp})
    return params


def _arr(x, shape):
    """Reshape a param to ``shape``, preserving symbolic (RV) params.

    Concrete numpy params are cast to float64; symbolic tensors (params as
    PyMC distributions) are reshaped via pytensor so they compose into the
    graph. This is what lets :func:`build_symbolic_graph` serve both the
    concrete-draw path and the RV / ``pm.Model`` path from one code path.
    """
    if isinstance(x, TensorVariable):
        return cast(TensorVariable, pt.as_tensor_variable(x).reshape(shape))
    return np.asarray(x, dtype="float64").reshape(shape)


def _dot_terms(cols: list, g_mask, coeff, T: int) -> TensorVariable:
    """``Σ_i coeff[i]·cols[i]`` over structurally-present parents (``g_mask[i] != 0``).

    ``g_mask`` is the CONCRETE 0/1 edge indicator (graph structure); ``coeff``
    is the per-parent coefficient, either a numpy array (concrete params) or a
    symbolic RV vector (params-as-distributions). Only parents with an edge are
    wired, so absent edges add nothing and the graph stays sparse. Because
    ``g_mask ∈ {0, 1}``, filtering on it and multiplying by ``coeff`` equals the
    old ``Σ (g·coeff)·cols`` exactly.

    A single ``Dot((T, n), (n,))`` is used rather than a python ``sum()`` of
    scaled columns: the latter builds nested Adds the canonicalizer flattens
    into one wide Add, and past ~32 inputs the py-backend crashes building the
    ufunc. Dot is one BLAS op the rewriter never flattens (and is faster).
    """
    g_mask = np.asarray(g_mask, dtype="float64").ravel()
    nz = [i for i in range(len(cols)) if g_mask[i] != 0.0]
    if not nz:
        return pt.zeros(T)
    if len(nz) == 1:
        return cast(TensorVariable, coeff[nz[0]] * cols[nz[0]])
    mat = pt.stack([cols[i] for i in nz], axis=1)  # (T, n)
    w = pt.stack([coeff[i] for i in nz])  # (n,) — numpy scalars or symbolic
    return cast(TensorVariable, pt.dot(mat, w))


def _walk_column(eps_col, rw_group: dict, i: int, T: int) -> TensorVariable:
    """Symbolic random walk for node i of a group, from its eps column."""
    return cast(
        TensorVariable,
        symbolic_random_walk(
            T,
            # mean / std may be symbolic (RV) params; smoothness must stay a
            # concrete float — it sets the moving-average kernel width, a
            # structural property of the graph.
            mean=rw_group["mean"][i],
            std=rw_group["std"][i],
            smoothness=float(rw_group["smoothness"][i]),
            positive_only=bool(rw_group["positive_only"]),
            eps=eps_col,
        ),
    )


def _adstock_col(c_col: TensorVariable, params: dict, k: int) -> TensorVariable:
    """Adstock transform of a single (T,) channel column for channel k."""
    l_max = params["l_max"]
    ad_fam = int(params["adstock_family"][k])  # family is concrete/structural
    x2d = c_col[:, None]
    if ad_fam == 1:
        out = mechanisms.apply_geometric_adstock(x2d, params["adstock_alpha"][k], l_max)
    elif ad_fam == 2:
        out = mechanisms.apply_weibull_pdf_adstock(
            x2d, params["weibull_lam"][k], params["weibull_k"][k], l_max
        )
    else:
        out = x2d
    return cast(TensorVariable, out[:, 0])


def _saturate_col(
    ad_col: TensorVariable, mean_ad: TensorVariable, params: dict, k: int
) -> TensorVariable:
    """κ-relative saturation of an adstocked column using a *given* scale.

    ``mean_ad`` is passed in (rather than recomputed) so the SAME structural
    response function f_k can be evaluated on several channel variants — the
    observed channel, the base channel, and the telescoping intervention
    variants — with an identical, pinned saturation scale. This is what makes
    the decomposition and the per-source indirect split exact.
    """
    name = _SAT_FAMILY_NAMES[int(params["sat_family"][k])]  # family is concrete/structural
    # Shape params may be symbolic (RV) or concrete — passed straight through
    # to the pymc-marketing-backed wrappers, which accept either.
    if name == "none":
        return cast(TensorVariable, ad_col / mean_ad)
    if name == "hill":
        return mechanisms.hill_kappa_relative(
            ad_col,
            mean_ad,
            slope=params["hill_slope"][k],
            kappa_mult=params["hill_kappa_mult"][k],
        )
    if name == "logistic":
        return mechanisms.logistic_kappa_relative(ad_col, mean_ad, lam=params["logistic_lam"][k])
    if name == "michaelis_menten":
        return mechanisms.michaelis_menten_kappa_relative(
            ad_col,
            mean_ad,
            alpha=params["mm_alpha"][k],
            kappa_mult=params["mm_kappa_mult"][k],
        )
    if name == "tanh":
        return mechanisms.tanh_kappa_relative(
            ad_col,
            mean_ad,
            b=params["tanh_b"][k],
            c=params["tanh_c"][k],
        )
    return mechanisms.root_kappa_relative(ad_col, mean_ad, alpha=params["root_alpha"][k])


def build_symbolic_graph(
    g: dict[str, np.ndarray],
    params: dict[str, Any],
    T: int,
    K: int,
    M: int,
    J: int,
    *,
    burn_in: int = 0,
    eps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the symbolic PyTensor graph for the additive causal DAG.

    Parameters
    ----------
    g : dict
        Edge indicators: ``g_cy`` (K,), ``g_dc`` (J,K), ``g_dz`` (J,M),
        ``g_db`` (J,), ``g_zb`` (M,), ``g_zc`` (M,K), ``g_cc`` (K,K,
        strictly upper-triangular), ``g_zz`` (M,M, strictly upper-triangular).
    params : dict
        Output of :func:`sample_scm_params` (or hand-crafted, same keys).
    T, K, M, J : int
        Time steps and node counts.
    burn_in : int
        Extra leading weeks simulated then dropped from every output. The
        adstock convolution left-pads with zeros, so without burn-in each
        contribution ramps 0 -> level over the first ``l_max`` weeks — an
        artifact that dominates smooth additive targets (measured 3–13x the
        steady-state std). With ``burn_in >= l_max`` the reported window sees
        real history instead of zeros. The κ-relative saturation scale is
        computed over the reported window only. Default 0 preserves legacy
        outputs exactly.

    Returns
    -------
    dict with keys:
        ``inputs`` — dict of symbolic eps inputs, each with leading dim
        ``T_full = T + burn_in``: ``eps_d`` (T_full,J), ``eps_z`` (T_full,M),
        ``eps_c`` (T_full,K), ``eps_b`` (T_full,), ``eps_y`` (T_full,), plus
        the channel-texture noise ``eps_c_hf`` / ``eps_c_pulse`` (T_full,K)
        (zero-fill when the texture is off — see :func:`draw_eps`);
        ``outputs`` — dict of symbolic node outputs: ``demand`` (T,J),
        ``controls`` (T,M), ``channels`` (T,K), ``channels_base`` (T,K),
        ``baseline`` (T,), ``baseline_intrinsic`` (T,),
        ``control_contribution`` (T,M), ``confounder_contribution`` (T,J),
        ``contributions`` (T,K, direct), ``contributions_observed`` (T,K),
        ``indirect_effects`` (T,), ``indirect_effects_by_source`` (T,3),
        ``sales`` (T,);
        ``hooks`` — dict mapping :data:`INTERVENTION_HOOK_GROUPS` names to
        lists of parent-term variables (for graph-surgery interventions).

    Notes
    -----
    Per-node baseline terms split the aggregated D->B / Z->B baseline terms:
    ``control_contribution[:, m] = g_zb[m]·ρ[m]·Z[:, m]`` and
    ``confounder_contribution[:, j] = g_db[j]·δ[j]·D[:, j]``, with
    ``baseline_intrinsic = RW_B + RW_Y`` (baseline minus all parent terms), so
    ``baseline_intrinsic + Σ_j confounder_contribution + Σ_m control_contribution
    == baseline``.

    ``indirect_effects_by_source`` (T, 3) is the telescoping 3-way indirect
    split in the LOCKED order ``(cc, zc, dc)`` — channel->channel,
    control->channel, hidden-confounder->channel — defined by sequential
    graph-surgery interventions (see the inline derivation). The three columns
    sum exactly to ``indirect_effects``.
    """
    _check_strict_upper(g["g_cc"], "g_cc")
    _check_strict_upper(g["g_zz"], "g_zz")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    g_cy = np.asarray(g["g_cy"], dtype="float64")
    g_dc = np.asarray(g["g_dc"], dtype="float64").reshape(J, K)
    g_dz = np.asarray(g["g_dz"], dtype="float64").reshape(J, M)
    g_db = np.asarray(g["g_db"], dtype="float64").reshape(J)
    g_zb = np.asarray(g["g_zb"], dtype="float64").reshape(M)
    g_zc = np.asarray(g["g_zc"], dtype="float64").reshape(M, K)
    g_cc = np.asarray(g["g_cc"], dtype="float64").reshape(K, K)
    g_zz = np.asarray(g["g_zz"], dtype="float64").reshape(M, M)

    # Channel texture. Magnitudes (hf_sigma, pulse_amp) may be symbolic (RV)
    # params; the per-channel ENABLE flags are concrete structure. On the
    # concrete-draw path the flags are derived from the magnitudes; the
    # RV / pm.Model path passes them explicitly (a symbolic magnitude has no
    # concrete truth value). The pulse enters as a 0/1 FIRE indicator
    # (eps_c_pulse), so no threshold lives in the graph — the fires are drawn
    # as a Bernoulli(pulse_prob) upstream (numpy in draw_eps, pm.Bernoulli on
    # the RV path). pulse_prob is carried on the graph dict for draw_eps.
    hf_sigma = _arr(params.get("hf_sigma", np.zeros(K)), (K,))
    pulse_amp = _arr(params.get("pulse_amp", np.zeros(K)), (K,))
    use_hf = params.get("use_hf")
    if use_hf is None:
        use_hf = np.asarray(params.get("hf_sigma", np.zeros(K)), dtype="float64").reshape(K) > 0.0
    use_hf = np.asarray(use_hf).reshape(K)
    use_pulse = params.get("use_pulse")
    if use_pulse is None:
        _pa = np.asarray(params.get("pulse_amp", np.zeros(K)), dtype="float64").reshape(K)
        _pp = np.asarray(params.get("pulse_prob", np.zeros(K)), dtype="float64").reshape(K)
        use_pulse = (_pa != 0.0) & (_pp > 0.0)
    use_pulse = np.asarray(use_pulse).reshape(K)
    _pp_raw = params.get("pulse_prob")
    pulse_prob_concrete = (
        np.asarray(_pp_raw, dtype="float64").reshape(K)
        if isinstance(_pp_raw, np.ndarray)
        else np.zeros(K)
    )

    # Nodes are simulated over T_full = burn_in + T weeks; every output is
    # sliced to the last T (the reported window).
    T_full = T + burn_in
    W = slice(burn_in, None)

    # Noise inputs. When ``eps`` is None (concrete-draw path) they are free
    # pt.tensor inputs the compiled function is fed via draw_eps. When provided
    # (RV / pm.Model path) they are the caller's noise RVs — pm.Normal walks and
    # a pm.Bernoulli 0/1 pulse — so the whole graph is drawn with no free inputs.
    # Static shapes are required: the adstock convolution builds its
    # sliding-window index from the static time length.
    if eps is None:
        eps_d = pt.tensor("eps_d", shape=(T_full, J), dtype="float64")
        eps_z = pt.tensor("eps_z", shape=(T_full, M), dtype="float64")
        eps_c = pt.tensor("eps_c", shape=(T_full, K), dtype="float64")
        eps_b = pt.tensor("eps_b", shape=(T_full,), dtype="float64")
        eps_y = pt.tensor("eps_y", shape=(T_full,), dtype="float64")
        eps_c_hf = pt.tensor("eps_c_hf", shape=(T_full, K), dtype="float64")
        eps_c_pulse = pt.tensor("eps_c_pulse", shape=(T_full, K), dtype="float64")
        inputs = {
            "eps_d": eps_d,
            "eps_z": eps_z,
            "eps_c": eps_c,
            "eps_b": eps_b,
            "eps_y": eps_y,
            "eps_c_hf": eps_c_hf,
            "eps_c_pulse": eps_c_pulse,
        }
    else:
        eps_d, eps_z, eps_c = eps["eps_d"], eps["eps_z"], eps["eps_c"]
        eps_b, eps_y = eps["eps_b"], eps["eps_y"]
        eps_c_hf, eps_c_pulse = eps["eps_c_hf"], eps["eps_c_pulse"]
        inputs = eps

    hooks: dict[str, list[TensorVariable]] = {name: [] for name in INTERVENTION_HOOK_GROUPS}

    def _hook(group: str, expr: TensorVariable, name: str) -> TensorVariable:
        var: TensorVariable = expr.copy(name=name)
        hooks[group].append(var)
        return var

    # -- confounders D (T_full, J): pure random walks -------------------------
    d_cols = [_walk_column(eps_d[:, j], params["rw_d"], j, T_full) for j in range(J)]
    D = pt.stack(d_cols, axis=1) if J > 0 else pt.zeros((T_full, 0))

    # -- controls Z (T_full, M): D->Z + upstream Z->Z + own walk --------------
    u_dz = _arr(params["u_dz"], (J, M))
    gamma_zz = _arr(params["gamma_zz"], (M, M))
    z_cols: list[TensorVariable] = []
    for m in range(M):
        walk = _walk_column(eps_z[:, m], params["rw_z"], m, T_full)
        term_d = _dot_terms(d_cols, g_dz[:, m], u_dz[:, m], T_full)
        term_z = _dot_terms(z_cols[:m], g_zz[:m, m], gamma_zz[:m, m], T_full)
        term_d = _hook("z_from_d", term_d, f"z{m}_from_d")
        term_z = _hook("z_from_z", term_z, f"z{m}_from_z")
        z_cols.append(term_d + term_z + walk)
    Z = pt.stack(z_cols, axis=1) if M > 0 else pt.zeros((T_full, 0))

    # -- channels C (T_full, K): D->C + Z->C + upstream C->C + own drive -----
    # C_base: same walks, all incoming interaction terms zeroed (the
    # "no upstream" intervention used for the exact decomposition).
    w_dc = _arr(params["w_dc"], (J, K))
    v_zc = _arr(params["v_zc"], (M, K))
    alpha_cc = _arr(params["alpha_cc"], (K, K))
    c_cols: list[TensorVariable] = []
    c_base_cols: list[TensorVariable] = []
    # Telescoping intervention variants (see indirect_effects_by_source, below):
    #   c_no_cc      = channel with the C->C term dropped (zero c_from_c)
    #   c_no_cc_zc   = channel with C->C and Z->C dropped (zero c_from_c, c_from_z)
    # These reuse the SAME term_d/term_z/walk as the observed channel and mirror
    # exactly what ``compute_indirect_effects`` produces by zeroing those hook
    # groups (the C->C term references upstream OBSERVED channels, so dropping it
    # is equivalent to zeroing the c_from_c hook variable).
    c_no_cc_cols: list[TensorVariable] = []
    c_no_cc_zc_cols: list[TensorVariable] = []
    for k in range(K):
        walk = _walk_column(eps_c[:, k], params["rw_c"], k, T_full)
        # Own exogenous drive = slow walk + iid weekly execution noise +
        # campaign pulses (plan doc 05 fix: without the high-frequency terms
        # the channel never sweeps its response curve and the contribution
        # target degenerates to a flat line). eps_c_pulse[:, k] is a 0/1 fire
        # indicator (Bernoulli(pulse_prob[k])); magnitudes may be symbolic.
        own = walk
        if use_hf[k]:
            own = own + hf_sigma[k] * eps_c_hf[:, k]
        if use_pulse[k]:
            own = own + pulse_amp[k] * eps_c_pulse[:, k]
        term_d = _dot_terms(d_cols, g_dc[:, k], w_dc[:, k], T_full)
        term_z = _dot_terms(z_cols, g_zc[:, k], v_zc[:, k], T_full)
        term_c = _dot_terms(c_cols[:k], g_cc[:k, k], alpha_cc[:k, k], T_full)
        term_d = _hook("c_from_d", term_d, f"c{k}_from_d")
        term_z = _hook("c_from_z", term_z, f"c{k}_from_z")
        term_c = _hook("c_from_c", term_c, f"c{k}_from_c")
        # softplus guard: spend-like channels must stay non-negative even
        # when signed upstream contributions push the pre-activation down
        c_cols.append(pt.softplus(term_d + term_z + term_c + own))
        c_base_cols.append(pt.softplus(own))
        c_no_cc_cols.append(pt.softplus(term_d + term_z + own))
        c_no_cc_zc_cols.append(pt.softplus(term_d + own))
    C = pt.stack(c_cols, axis=1)
    C_base = pt.stack(c_base_cols, axis=1)

    # -- baseline B (T_full,): D->B + Z->B + own walk --------------------------
    delta_db = _arr(params["delta_db"], (J,))
    rho_zb = _arr(params["rho_zb"], (M,))
    walk_b = _walk_column(eps_b, params["rw_b"], 0, T_full)
    term_bd = _dot_terms(d_cols, g_db, delta_db, T_full)
    term_bz = _dot_terms(z_cols, g_zb, rho_zb, T_full)
    term_bd = _hook("b_from_d", term_bd, "b_from_d")
    term_bz = _hook("b_from_z", term_bz, "b_from_z")
    B = term_bd + term_bz + walk_b

    # -- per-node direct baseline terms (exact split of term_bd / term_bz) ----
    # column m of control_contribution   = g_zb[m]·ρ[m]·Z[:,m]  (sums to term_bz)
    # column j of confounder_contribution = g_db[j]·δ[j]·D[:,j] (sums to term_bd)
    control_contrib_cols = [(g_zb[m] * rho_zb[m]) * z_cols[m] for m in range(M)]
    control_contribution = (
        pt.stack(control_contrib_cols, axis=1) if M > 0 else pt.zeros((T_full, 0))
    )  # (T_full, M)
    confounder_contrib_cols = [(g_db[j] * delta_db[j]) * d_cols[j] for j in range(J)]
    confounder_contribution = (
        pt.stack(confounder_contrib_cols, axis=1) if J > 0 else pt.zeros((T_full, 0))
    )  # (T_full, J)

    # -- direct nonlinear responses + exact decomposition --------------------
    beta = _arr(params["beta"], (K,))
    contrib_obs_cols, contrib_base_cols, sat_scales = [], [], []
    # Telescoping 3-way indirect split (LOCKED order cc -> zc -> dc). Because the
    # direct response f_k is nonlinear, naive one-at-a-time interventions do not
    # sum to the total; the split is defined by a FIXED sequential zeroing order
    # using the SAME pinned saturation scale (mirrors compute_indirect_effects):
    #   ie_cc = Y(all)                          - Y(zero c_from_c)
    #   ie_zc = Y(zero c_from_c)                - Y(zero c_from_c, c_from_z)
    #   ie_dc = Y(zero c_from_c, c_from_z)      - Y(zero c_from_c, c_from_z, c_from_d)
    # These telescope exactly to indirect_effects because Y(zero all three) is
    # baseline + the direct (base-channel) contributions.
    ie_cc_cols, ie_zc_cols, ie_dc_cols = [], [], []
    for k in range(K):
        # Adstock over the full simulated horizon, then slice to the reported
        # window: with burn_in >= l_max the window's convolution sees real
        # pre-window history instead of the zero padding (warmup artifact).
        # The κ scale is a mean over the REPORTED window so f_k's operating
        # point matches what the model observes.
        ad_obs = _adstock_col(c_cols[k], params, k)[W]
        scale_k = pt.maximum(ad_obs.mean(), 1e-8).copy(name=f"sat_scale_{k}")

        # ONE response function per channel, applied to every variant: the
        # telescoping split cancels the middle variants, so a divergence here
        # would pass the identity tests while corrupting the per-source split.
        def _f(ad_col, *, _scale=scale_k, _k=k):
            return _saturate_col(ad_col, _scale, params, _k)

        f_obs = _f(ad_obs)
        f_base = _f(_adstock_col(c_base_cols[k], params, k)[W])
        f_no_cc = _f(_adstock_col(c_no_cc_cols[k], params, k)[W])
        f_no_cc_zc = _f(_adstock_col(c_no_cc_zc_cols[k], params, k)[W])
        gate = g_cy[k] * beta[k]  # g concrete, beta possibly symbolic
        contrib_obs_cols.append(gate * f_obs)
        contrib_base_cols.append(gate * f_base)
        ie_cc_cols.append(gate * (f_obs - f_no_cc))
        ie_zc_cols.append(gate * (f_no_cc - f_no_cc_zc))
        ie_dc_cols.append(gate * (f_no_cc_zc - f_base))
        sat_scales.append(scale_k)
    contributions_observed = pt.stack(contrib_obs_cols, axis=1)  # (T, K)
    contributions = pt.stack(contrib_base_cols, axis=1)  # (T, K) direct
    indirect_effects = (contributions_observed - contributions).sum(axis=1)  # (T,)

    # K == 0 is unsupported (the contributions stack above already requires
    # K >= 1), so the ie stacks need no separate guard.
    ie_cc = pt.stack(ie_cc_cols, axis=1).sum(axis=1)
    ie_zc = pt.stack(ie_zc_cols, axis=1).sum(axis=1)
    ie_dc = pt.stack(ie_dc_cols, axis=1).sum(axis=1)
    indirect_effects_by_source = pt.stack([ie_cc, ie_zc, ie_dc], axis=1)  # (T, 3): cc, zc, dc

    walk_y = _walk_column(eps_y, params["rw_y"], 0, T_full)
    baseline = (B + walk_y)[W]  # sales noise folded into the baseline component
    # "Y independent of everything": baseline minus all parent (D/Z) terms.
    baseline_intrinsic = (walk_b + walk_y)[W]  # (T,)
    sales = baseline + contributions_observed.sum(axis=1)
    # identity: sales == baseline + contributions.sum(1) + indirect_effects
    #        == baseline_intrinsic + Σ confounder_contribution + Σ control_contribution
    #           + contributions.sum(1) + indirect_effects_by_source.sum(1)

    return {
        "inputs": inputs,
        "outputs": {
            "demand": D[W],
            "controls": Z[W],
            "channels": C[W],
            "channels_base": C_base[W],
            "baseline": baseline,
            "baseline_intrinsic": baseline_intrinsic,
            "control_contribution": control_contribution[W],
            "confounder_contribution": confounder_contribution[W],
            "contributions": contributions,
            "contributions_observed": contributions_observed,
            "indirect_effects": indirect_effects,
            "indirect_effects_by_source": indirect_effects_by_source,
            "sales": sales,
        },
        "hooks": hooks,
        "sat_scales": sat_scales,
        # Per-array texture flags (same predicates as the channel loop above):
        # draw_eps draws an eps array iff some graph node references it, and
        # zero-fills it otherwise, so a config never consumes RNG for noise it
        # does not use. extended_noise is the any-texture OR, kept for
        # info/compat with older graph dicts.
        "texture_noise": {"hf": bool(use_hf.any()), "pulse": bool(use_pulse.any())},
        "extended_noise": bool(use_hf.any() or use_pulse.any()),
        # Per-channel Bernoulli fire probability for draw_eps (concrete path);
        # zeros on the RV path, where pulses are drawn as pm.Bernoulli RVs.
        "pulse_prob": pulse_prob_concrete,
        "sizes": {"T": T, "K": K, "M": M, "J": J, "burn_in": burn_in},
    }


_INPUT_ORDER = ("eps_d", "eps_z", "eps_c", "eps_b", "eps_y", "eps_c_hf", "eps_c_pulse")


#: Default compilation mode. Each task gets its own small graph, so the
#: C-backend compile cost of FAST_RUN dominates; the python-backend
#: FAST_COMPILE is ~250x faster end-to-end at these sizes.
_DEFAULT_MODE = "FAST_COMPILE"


def compile_graph(
    graph: dict[str, Any],
    output_names: tuple[str, ...] | None = None,
    mode: str = _DEFAULT_MODE,
) -> Callable[..., list[np.ndarray]]:
    """Compile the graph outputs into one pytensor function of the eps inputs.

    The returned callable takes the eps arrays positionally in the canonical
    :data:`_INPUT_ORDER` (``eps_d, eps_z, eps_c, eps_b, eps_y, eps_c_hf,
    eps_c_pulse`` — call it as ``fn(*[eps[n] for n in _INPUT_ORDER])`` with a
    :func:`draw_eps` dict) and returns the outputs in ``output_names`` order
    (default: all outputs, dict order).
    """
    if output_names is None:
        output_names = tuple(graph["outputs"].keys())
    inputs = [graph["inputs"][name] for name in _INPUT_ORDER]
    outputs = [graph["outputs"][name] for name in output_names]
    return cast(
        "Callable[..., list[np.ndarray]]",
        pytensor.function(inputs, outputs, on_unused_input="ignore", mode=mode),
    )


def draw_eps(graph: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw concrete white-noise inputs for one task.

    Each channel-texture array (``eps_c_hf`` / ``eps_c_pulse``) is drawn only
    when some graph node references it (per-array ``texture_noise`` flags);
    otherwise it is zero-filled WITHOUT consuming RNG. So corpora generated
    before the texture knobs existed reproduce byte-identically from the same
    seed, and an hf-only (or pulse-only) config does not burn RNG on the
    mechanism it disabled.
    """
    sizes = graph["sizes"]
    T, K, M, J = sizes["T"], sizes["K"], sizes["M"], sizes["J"]
    T_full = T + int(sizes.get("burn_in", 0))
    eps = {
        "eps_d": rng.standard_normal((T_full, J)),
        "eps_z": rng.standard_normal((T_full, M)),
        "eps_c": rng.standard_normal((T_full, K)),
        "eps_b": rng.standard_normal(T_full),
        "eps_y": rng.standard_normal(T_full),
    }
    tex = graph.get("texture_noise")
    if tex is None:  # older graph dicts predate the per-array flags
        any_tex = bool(graph.get("extended_noise", False))
        tex = {"hf": any_tex, "pulse": any_tex}
    # hf jitter is iid normal; the pulse is a 0/1 Bernoulli(pulse_prob) fire
    # indicator (the graph adds pulse_amp * fire — no threshold in the graph).
    eps["eps_c_hf"] = rng.standard_normal((T_full, K)) if tex["hf"] else np.zeros((T_full, K))
    if tex["pulse"]:
        pulse_prob = np.asarray(graph.get("pulse_prob", np.zeros(K)), dtype="float64").reshape(K)
        eps["eps_c_pulse"] = (rng.random((T_full, K)) < pulse_prob[None, :]).astype("float64")
    else:
        eps["eps_c_pulse"] = np.zeros((T_full, K))
    return eps


def draw_from_graph(
    graph: dict[str, Any],
    rng: np.random.Generator,
    compiled: Callable[..., list[np.ndarray]] | None = None,
    output_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Draw one task: sample eps, evaluate the graph, return named outputs."""
    if output_names is None:
        output_names = tuple(graph["outputs"].keys())
    if compiled is None:
        compiled = compile_graph(graph, output_names)
    eps = draw_eps(graph, rng)
    values = compiled(*[eps[name] for name in _INPUT_ORDER])
    out: dict[str, Any] = dict(zip(output_names, [np.asarray(v) for v in values]))
    out["eps"] = eps
    return out


def compute_indirect_effects(
    graph: dict[str, Any],
    inputs: dict[str, np.ndarray],
    intervention_sets: dict[str, list[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute indirect effects via graph-surgery interventions (plan doc 4.2).

    For each named intervention set (a list of hook groups from
    :data:`INTERVENTION_HOOK_GROUPS`), replace those parent-term variables
    with zeros, re-evaluate sales on the *same* noise draws, and return
    ``ΔY = Y_original - Y_zeroed``.

    Parameters
    ----------
    graph : dict
        Output of :func:`build_symbolic_graph`.
    inputs : dict
        Concrete eps arrays (e.g. from :func:`draw_eps`, or the ``"eps"``
        entry of :func:`draw_from_graph`). Eps keys the dict omits — e.g. a
        saved legacy 5-key dict predating the channel-texture inputs — are
        zero-filled, which reproduces the pre-texture behaviour exactly.
    intervention_sets : dict[str, list[str]] | None
        Maps an intervention name to the hook groups to zero. Default:
        ``{"all_channel_upstream": ["c_from_d", "c_from_z", "c_from_c"],
        "cc_only": ["c_from_c"]}``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{name: ΔY (T,)}`` per intervention, plus ``"sales_original"``.
    """
    if intervention_sets is None:
        intervention_sets = {
            "all_channel_upstream": ["c_from_d", "c_from_z", "c_from_c"],
            "cc_only": ["c_from_c"],
        }
    sales = graph["outputs"]["sales"]
    input_vars = [graph["inputs"][name] for name in _INPUT_ORDER]
    input_vals = [
        (
            inputs[name]
            if name in inputs
            else np.zeros(tuple(int(s) for s in graph["inputs"][name].type.shape))
        )
        for name in _INPUT_ORDER
    ]

    f_orig = pytensor.function(input_vars, sales, on_unused_input="ignore", mode=_DEFAULT_MODE)
    y_orig = np.asarray(f_orig(*input_vals))

    # Pin the κ-relative saturation scales to their observational values so
    # f_k stays the SAME structural function in the zeroed graph (otherwise
    # the intervention would silently change the response curve itself).
    scale_vars = graph.get("sat_scales", [])
    scale_pins: dict[TensorVariable, TensorVariable] = {}
    if scale_vars:
        f_scales = pytensor.function(
            input_vars, scale_vars, on_unused_input="ignore", mode=_DEFAULT_MODE
        )
        scale_vals = f_scales(*input_vals)
        scale_pins = {
            var: pt.constant(np.asarray(val), name=f"{var.name}_pinned")
            for var, val in zip(scale_vars, scale_vals)
        }

    out: dict[str, np.ndarray] = {"sales_original": y_orig}
    for name, groups in intervention_sets.items():
        # dict[Any, Any]: clone_replace wants dict[Variable, Variable] and
        # dict is invariant, so the TensorVariable-keyed dict won't unify.
        replacements: dict[Any, Any] = dict(scale_pins)
        for group in groups:
            if group not in graph["hooks"]:
                raise KeyError(f"unknown hook group {group!r}; valid: {INTERVENTION_HOOK_GROUPS}")
            for var in graph["hooks"][group]:
                replacements[var] = pt.zeros_like(var)
        # clone_replace (memo-based rebuild), NOT graph_replace: the scale
        # pins are descendants of the hook vars, and graph_replace's
        # truncated-input bookkeeping crashes on such overlapping-ancestry
        # replacement sets under pytensor 2.38.x ("i-N is not a part of
        # graph"; fixed upstream in pytensor 3.x, which filters redundant
        # keys). clone_replace substitutes each key wherever it occurs and
        # silently ignores keys made unreachable by other replacements —
        # the same semantics graph_replace has on pytensor >= 3.
        sales_zeroed = clone_replace([sales], replacements)[0]
        f_zeroed = pytensor.function(
            input_vars, sales_zeroed, on_unused_input="ignore", mode=_DEFAULT_MODE
        )
        y_zeroed = np.asarray(f_zeroed(*input_vals))
        out[name] = y_orig - y_zeroed
    return out
