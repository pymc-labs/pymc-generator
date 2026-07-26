"""PyTensor symbolic causal-graph builder for the additive causal SCM.

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
campaign pulses (amplitude a_k, per-week fire probability p_k) — the
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
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable

from . import mechanisms
from .random_walk import symbolic_random_walk

__all__ = ["build_symbolic_graph"]

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


def _adstock_col_with_resets(c_col: TensorVariable, params: dict, k: int) -> TensorVariable:
    """Adstock a channel, resetting its response history at its shock starts.

    A reset is response-state surgery rather than a change to the natural
    channel path: after a shock starts, its response is the ordinary adstock
    of the input suffix beginning at that start.  Schedule slots are ordered,
    so applying the corresponding switches in schedule order makes a later
    shock on the same channel override an earlier reset.
    """
    schedule = params.get("channel_shock")
    if schedule is None:
        return _adstock_col(c_col, params, k)

    out = _adstock_col(c_col, params, k)
    time = pt.arange(c_col.shape[0])
    for s in range(int(schedule["n_shocks"])):
        start = schedule["start_full"][s]
        applies = pt.eq(schedule["channel"][s], k)
        suffix = c_col * pt.cast(time >= start, c_col.dtype)
        reset_out = _adstock_col(suffix, params, k)
        out = pt.switch(applies & (time >= start), reset_out, out)
    return out


def _clamp_channel(c_col: TensorVariable, params: dict, k: int) -> TensorVariable:
    """Apply this channel's absolute held-level shock windows, when enabled."""
    schedule = params.get("channel_shock")
    if schedule is None:
        return c_col
    return cast(
        TensorVariable,
        pt.switch(
            pt.neq(schedule["mask_full"][:, k], 0),
            schedule["level_full"][:, k],
            c_col,
        ),
    )


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
    eps: dict[str, Any],
) -> dict[str, Any]:
    """Build the symbolic PyTensor graph for the additive causal DAG.

    Parameters
    ----------
    g : dict
        Edge indicators: ``g_cy`` (K,), ``g_dc`` (J,K), ``g_dz`` (J,M),
        ``g_db`` (J,), ``g_zb`` (M,), ``g_zc`` (M,K), ``g_cc`` (K,K,
        strictly upper-triangular), ``g_zz`` (M,M, strictly upper-triangular).
    params : dict
        SCM parameters — the continuous ones may be symbolic (PyMC RV)
        tensors or concrete numpy; families/smoothness are concrete. Assembled
        by :func:`prior_generator.world_model.build_world_model`.
    eps : dict
        The caller's noise RVs, each with leading dim ``T_full = T + burn_in``:
        ``eps_d`` (T_full,J), ``eps_z`` (T_full,M), ``eps_c`` (T_full,K),
        ``eps_b`` (T_full,), ``eps_y`` (T_full,), and the channel-texture noise
        ``eps_c_hf`` (weekly jitter) / ``eps_c_pulse`` (a 0/1 Bernoulli fire).
    T, K, M, J : int
        Time steps and node counts.
    burn_in : int
        Extra leading weeks simulated then dropped from every output. The
        adstock convolution left-pads with zeros, so without burn-in each
        contribution ramps 0 -> level over the first ``l_max`` weeks — an
        artifact that dominates smooth additive targets (measured 3–13x the
        steady-state std). With ``burn_in >= l_max`` the reported window sees
        real history instead of zeros. The κ-relative saturation scale is
        computed over the reported window only.

    Returns
    -------
    dict with a single key ``outputs`` — the symbolic node outputs (sliced to
    the reported T-window):
        ``demand`` (T,J), ``controls`` (T,M), ``channels`` (T,K),
        ``channels_base`` (T,K), ``saturation_scale`` (K,), ``baseline`` (T,),
        ``baseline_intrinsic`` (T,),
        ``control_contribution`` (T,M), ``confounder_contribution`` (T,J),
        ``contributions`` (T,K, direct), ``contributions_observed`` (T,K),
        ``indirect_effects`` (T,), ``indirect_effects_by_source`` (T,3),
        ``sales`` (T,).

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
    # params; the per-channel ENABLE flags are concrete structure, normally
    # passed in as ``use_hf`` / ``use_pulse`` (build_world_model sets them) and
    # derived from the magnitudes as a fallback when absent (a symbolic
    # magnitude has no concrete truth value). The pulse enters as a 0/1 FIRE
    # indicator ``eps_c_pulse`` ~ Bernoulli(pulse_prob), so no threshold lives
    # in the graph.
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

    # Nodes are simulated over T_full = burn_in + T weeks; every output is
    # sliced to the last T (the reported window).
    T_full = T + burn_in
    W = slice(burn_in, None)

    # Noise inputs are the caller's RVs — pm.Normal walks + weekly jitter and a
    # pm.Bernoulli 0/1 pulse — so the whole graph is drawn with no free inputs.
    # Static shapes are required: the adstock convolution builds its
    # sliding-window index from the static time length.
    eps_d, eps_z, eps_c = eps["eps_d"], eps["eps_z"], eps["eps_c"]
    eps_b, eps_y = eps["eps_b"], eps["eps_y"]
    eps_c_hf, eps_c_pulse = eps["eps_c_hf"], eps["eps_c_pulse"]

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
        z_cols.append(term_d + term_z + walk)
    Z = pt.stack(z_cols, axis=1) if M > 0 else pt.zeros((T_full, 0))

    # -- channels C (T_full, K): D->C + Z->C + upstream C->C + own drive -----
    # C_base: same walks, all incoming interaction terms zeroed (the
    # "no upstream" intervention used for the exact decomposition).
    w_dc = _arr(params["w_dc"], (J, K))
    v_zc = _arr(params["v_zc"], (M, K))
    alpha_cc = _arr(params["alpha_cc"], (K, K))
    c_cols: list[TensorVariable] = []
    c_unshocked_cols: list[TensorVariable] = []
    c_base_cols: list[TensorVariable] = []
    # Telescoping intervention variants (see indirect_effects_by_source, below):
    #   c_no_cc      = channel with the C->C term dropped
    #   c_no_cc_zc   = channel with C->C and Z->C dropped
    # These reuse the SAME term_d/term_z/walk as the observed channel; each just
    # omits the named upstream term (the C->C term references upstream OBSERVED
    # channels, so dropping it is the "zero that interaction" intervention).
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
        # The natural recursion is retained solely for the realism reference.
        # The observed recursion instead sees already-clamped upstream parents,
        # which is the SCM meaning of a channel intervention.
        term_c_unshocked = _dot_terms(c_unshocked_cols[:k], g_cc[:k, k], alpha_cc[:k, k], T_full)
        term_c = _dot_terms(c_cols[:k], g_cc[:k, k], alpha_cc[:k, k], T_full)
        # softplus guard: spend-like channels must stay non-negative even
        # when signed upstream contributions push the pre-activation down
        c_unshocked_cols.append(pt.softplus(term_d + term_z + term_c_unshocked + own))
        c_cols.append(_clamp_channel(pt.softplus(term_d + term_z + term_c + own), params, k))
        c_base_cols.append(_clamp_channel(pt.softplus(own), params, k))
        c_no_cc_cols.append(_clamp_channel(pt.softplus(term_d + term_z + own), params, k))
        c_no_cc_zc_cols.append(_clamp_channel(pt.softplus(term_d + own), params, k))
    C = pt.stack(c_cols, axis=1)
    C_base = pt.stack(c_base_cols, axis=1)

    # -- baseline B (T_full,): D->B + Z->B + own walk --------------------------
    delta_db = _arr(params["delta_db"], (J,))
    rho_zb = _arr(params["rho_zb"], (M,))
    walk_b = _walk_column(eps_b, params["rw_b"], 0, T_full)
    term_bd = _dot_terms(d_cols, g_db, delta_db, T_full)
    term_bz = _dot_terms(z_cols, g_zb, rho_zb, T_full)
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
    contrib_obs_cols, contrib_base_cols, sat_scale_cols = [], [], []
    # Telescoping 3-way indirect split (LOCKED order cc -> zc -> dc). Because the
    # direct response f_k is nonlinear, naive one-at-a-time interventions do not
    # sum to the total; the split is defined by a FIXED sequential zeroing order
    # using the SAME pinned saturation scale for every variant:
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
        ad_obs = _adstock_col_with_resets(c_cols[k], params, k)[W]
        scale_k = pt.maximum(ad_obs.mean(), 1e-8).copy(name=f"sat_scale_{k}")
        sat_scale_cols.append(scale_k)

        # ONE response function per channel, applied to every variant: the
        # telescoping split cancels the middle variants, so a divergence here
        # would pass the identity tests while corrupting the per-source split.
        def _f(ad_col, *, _scale=scale_k, _k=k):
            return _saturate_col(ad_col, _scale, params, _k)

        f_obs = _f(ad_obs)
        f_base = _f(_adstock_col_with_resets(c_base_cols[k], params, k)[W])
        f_no_cc = _f(_adstock_col_with_resets(c_no_cc_cols[k], params, k)[W])
        f_no_cc_zc = _f(_adstock_col_with_resets(c_no_cc_zc_cols[k], params, k)[W])
        gate = g_cy[k] * beta[k]  # g concrete, beta possibly symbolic
        contrib_obs_cols.append(gate * f_obs)
        contrib_base_cols.append(gate * f_base)
        ie_cc_cols.append(gate * (f_obs - f_no_cc))
        ie_zc_cols.append(gate * (f_no_cc - f_no_cc_zc))
        ie_dc_cols.append(gate * (f_no_cc_zc - f_base))
    contributions_observed = pt.stack(contrib_obs_cols, axis=1)  # (T, K)
    contributions = pt.stack(contrib_base_cols, axis=1)  # (T, K) direct
    indirect_effects = (contributions_observed - contributions).sum(axis=1)  # (T,)

    # K == 0 is unsupported (the contributions stack above already requires
    # K >= 1), so the ie stacks need no separate guard.
    # Graph surgery against an absent edge family is exactly zero. Returning a
    # literal zero avoids machine-epsilon subtraction residue in persisted truth
    # labels, especially for the structurally edge-free K=1 C->C block.
    ie_cc = pt.stack(ie_cc_cols, axis=1).sum(axis=1) if g_cc.any() else pt.zeros(T)
    ie_zc = pt.stack(ie_zc_cols, axis=1).sum(axis=1) if g_zc.any() else pt.zeros(T)
    ie_dc = pt.stack(ie_dc_cols, axis=1).sum(axis=1) if g_dc.any() else pt.zeros(T)
    indirect_effects_by_source = pt.stack([ie_cc, ie_zc, ie_dc], axis=1)  # (T, 3): cc, zc, dc

    walk_y = _walk_column(eps_y, params["rw_y"], 0, T_full)
    baseline = (B + walk_y)[W]  # sales noise folded into the baseline component
    # "Y independent of everything": baseline minus all parent (D/Z) terms.
    baseline_intrinsic = (walk_b + walk_y)[W]  # (T,)
    sales = baseline + contributions_observed.sum(axis=1)
    # identity: sales == baseline + contributions.sum(1) + indirect_effects
    #        == baseline_intrinsic + Σ confounder_contribution + Σ control_contribution
    #           + contributions.sum(1) + indirect_effects_by_source.sum(1)

    outputs = {
        "demand": D[W],
        "controls": Z[W],
        "channels": C[W],
        "channels_base": C_base[W],
        "saturation_scale": pt.stack(sat_scale_cols),
        "baseline": baseline,
        "baseline_intrinsic": baseline_intrinsic,
        "control_contribution": control_contribution[W],
        "confounder_contribution": confounder_contribution[W],
        "contributions": contributions,
        "contributions_observed": contributions_observed,
        "indirect_effects": indirect_effects,
        "indirect_effects_by_source": indirect_effects_by_source,
        "sales": sales,
    }
    # These are intentionally audit-only paths.  They are drawn to decide
    # whether the *natural* world is realistic, never persisted in corpora.
    if params.get("channel_shock") is not None:
        unshocked_contribs = []
        for k in range(K):
            ad_unshocked = _adstock_col(c_unshocked_cols[k], params, k)[W]
            scale_k = pt.maximum(ad_unshocked.mean(), 1e-8)
            unshocked_contribs.append(
                g_cy[k] * beta[k] * _saturate_col(ad_unshocked, scale_k, params, k)
            )
        outputs["channels_unshocked"] = pt.stack(c_unshocked_cols, axis=1)[W]
        outputs["sales_unshocked"] = baseline + pt.stack(unshocked_contribs, axis=1).sum(axis=1)
    return {"outputs": outputs}
