"""PyTensor causal graph and exact additive decomposition.

Latent latent_unobserved D and observed covariates Z are signed. Treatments C apply softplus
to their own exogenous drive plus additive parent loadings. C->C and Z->Z
edges run from lower to higher indices, so the graph is acyclic. Carryover and
saturation apply on direct C->Y paths; other loadings are linear on the
child's pre-activation scale.

The intercept B has no parents. Latent_unobserved and covariates enter outcome Y directly,
alongside treatment response and iid observation noise. The default floor scope
clips B alone. Optional non-treatment flooring clips the aggregate incrementally,
so its reported latent_unobserved/covariate columns are clipped increments rather than
unmodified linear loadings. Outcome itself is not clamped.

Each non-outcome node has a random-walk own drive. Treatments can additionally
have iid execution noise and campaign pulses; covariates can have iid shocks
and centered pulses. These terms add higher-frequency variation but do not
guarantee informative response curves or identification. Supplied innovations
may be correlated by the configured baseline/treatment confounding mechanism.

``C_base`` removes all incoming treatment interactions while retaining the same
own-drive realizations, including any execution noise and pulses. With each
response function fixed across counterfactual paths:

.. code-block:: text

    contributions_k = g_cy[k] * beta[k] * f_k(C_base_k)
    indirect_effects = sum_k g_cy[k] * beta[k] * (f_k(C_k) - f_k(C_base_k))
    baseline_out = baseline_intrinsic + outcome_noise
                   + sum(latent_unobserved_contribution) + sum(covariate_contribution)
    outcome = baseline_out + sum(contributions) + indirect_effects

The identity uses shared symbolic quantities, not a Taylor approximation,
and holds up to floating-point error. The ordered telescoping indirect split
attributes incoming treatment interactions in ``(cc, zc, dc)`` order.

Saturation scales come from :func:`_reference_levels`. These parameter-only
anchors are not expected or realized mean treatment: nonlinear positivity
transforms and stochastic variation generally separate those quantities.
Anchors do not read realized series or future time windows.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable

from . import mechanisms
from .random_walk import symbolic_random_walk, symbolic_random_walk_by_width
from .sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS

__all__ = ["build_symbolic_graph", "inactive_zero"]

# Mechanism family ids are projected in sampler canonical order.


def inactive_zero(active_flag, expr: TensorVariable) -> TensorVariable:
    """Zero ``expr`` wherever ``active_flag`` is false.

    The ``dynamic_g`` graph is built at the layout's MAXIMUM node counts so one
    compiled function serves every cell. Cells that use fewer nodes keep the
    padded slots in the tensor layout and switch them off here, which is what
    makes the node count a run-time value rather than a compile-time one.
    """
    flag = (
        active_flag
        if isinstance(active_flag, TensorVariable)
        else pt.as_tensor_variable(np.asarray(active_flag, dtype="float64"))
    )
    return cast(TensorVariable, pt.switch(pt.neq(flag, 0.0), expr, pt.zeros_like(expr)))


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


def _required_eps(eps: dict[str, Any], name: str, enabled: bool) -> Any:
    """One optional texture innovation, demanded only when its term is enabled.

    A caller building the graph directly (no ``pm.Model``) may leave a disabled
    term's noise out entirely. Enabling the term without it is a caller bug, so
    it fails here instead of silently degenerating to a zero drive.
    """
    value = eps.get(name)
    if enabled and value is None:
        raise ValueError(f"eps[{name!r}] is required while its texture term is enabled")
    return value


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


def _dot_terms(
    cols: list, g_mask, coeff, n_time_steps: int, *, dynamic_g: bool = False
) -> TensorVariable:
    """``Σ_i coeff[i]·cols[i]`` over structurally-present parents (``g_mask[i] != 0``).

    ``g_mask`` is the 0/1 edge indicator (graph structure); ``coeff`` is the
    per-parent coefficient, either a numpy array (concrete params) or a symbolic
    RV vector (params-as-distributions). Only parents with an edge are wired, so
    absent edges add nothing and the graph stays sparse. Because
    ``g_mask ∈ {0, 1}``, filtering on it and multiplying by ``coeff`` equals the
    old ``Σ (g·coeff)·cols`` exactly.

    A single ``Dot((n_time_steps, n), (n,))`` over the ``n`` wired parents is used
    rather than a python ``sum()`` of scaled columns: the latter builds nested
    Adds the canonicalizer flattens into one wide Add, and past ~32 inputs the
    py-backend crashes building the ufunc. Dot is one BLAS op the rewriter never
    flattens (and is faster).

    Under ``dynamic_g`` the mask is a tensor whose value is not known while the
    graph is built, so the sparsity shortcut is unavailable: every candidate
    parent is wired as ``g_mask[i]·coeff[i]·cols[i]`` and absent edges are
    zeroed numerically instead of structurally. That is the cost of swapping
    DAGs without recompiling.
    """
    if dynamic_g:
        g_dyn = pt.as_tensor_variable(g_mask).reshape((-1,))
        if not cols:
            return pt.zeros(n_time_steps)
        terms = [g_dyn[i] * coeff[i] * cols[i] for i in range(len(cols))]
        return cast(TensorVariable, pt.add(*terms))

    g_mask = np.asarray(g_mask, dtype="float64").ravel()
    nz = [i for i in range(len(cols)) if g_mask[i] != 0.0]
    if not nz:
        return pt.zeros(n_time_steps)
    if len(nz) == 1:
        return cast(TensorVariable, coeff[nz[0]] * cols[nz[0]])
    mat = pt.stack([cols[i] for i in nz], axis=1)  # (n_time_steps, n)
    w = pt.stack([coeff[i] for i in nz])  # (n,) — numpy scalars or symbolic
    return cast(TensorVariable, pt.dot(mat, w))


def _walk_column(eps_col, rw_group: dict, i: int, n_time_steps: int) -> TensorVariable:
    """Symbolic random walk for node i of a group, from its eps column.

    A ``width_index`` entry on the group switches to the kernel-by-index walk,
    which keeps smoothness a run-time value. Groups without it (every
    :func:`pymc_generator.world_model._walk_priors` group) resolve their kernel
    width while the graph is built, as before.
    """
    width_index = rw_group.get("width_index")
    if width_index is not None:
        return cast(
            TensorVariable,
            symbolic_random_walk_by_width(
                n_time_steps,
                mean=rw_group["mean"][i],
                std=rw_group["std"][i],
                width_index=width_index[i],
                positive_only=bool(rw_group["positive_only"]),
                rw_smoothness_max_weeks=int(rw_group["rw_smoothness_max_weeks"]),
                eps=eps_col,
            ),
        )
    return cast(
        TensorVariable,
        symbolic_random_walk(
            n_time_steps,
            # mean / std may be symbolic (RV) params; smoothness and its
            # absolute-week cap must stay concrete — they set the
            # moving-average kernel width, a structural graph property.
            mean=rw_group["mean"][i],
            std=rw_group["std"][i],
            smoothness=float(rw_group["smoothness"][i]),
            positive_only=bool(rw_group["positive_only"]),
            rw_smoothness_max_weeks=int(rw_group["rw_smoothness_max_weeks"]),
            eps=eps_col,
        ),
    )


def _carryover_col(
    c_col: TensorVariable, params: dict, k: int, *, dynamic_family: bool = False
) -> TensorVariable:
    """Carryover transform of a single (n_time_steps,) treatment column for treatment k.

    With a concrete family only the selected transform is built, so the unused
    families' shape parameters stay out of the graph. Under ``dynamic_family``
    the family id is a tensor, so every transform is built and selected by
    ``pt.switch`` — which necessarily pulls all of their parameters in.
    """
    l_max = params["l_max"]
    x2d = c_col[:, None]
    if dynamic_family:
        ad_fam = pt.as_tensor_variable(params["carryover_family"])[k]
        geometric = mechanisms.apply_geometric_carryover(x2d, params["carryover_alpha"][k], l_max)
        weibull = mechanisms.apply_weibull_pdf_carryover(
            x2d, params["weibull_lam"][k], params["weibull_k"][k], l_max
        )
        out = pt.switch(
            pt.eq(ad_fam, CARRYOVER_FAMILY_KEYS.index("geometric")),
            geometric,
            pt.switch(pt.eq(ad_fam, CARRYOVER_FAMILY_KEYS.index("weibull")), weibull, x2d),
        )
        return cast(TensorVariable, out[:, 0])
    ad_fam = int(params["carryover_family"][k])  # family is concrete/structural
    if ad_fam == 1:
        out = mechanisms.apply_geometric_carryover(x2d, params["carryover_alpha"][k], l_max)
    elif ad_fam == 2:
        out = mechanisms.apply_weibull_pdf_carryover(
            x2d, params["weibull_lam"][k], params["weibull_k"][k], l_max
        )
    else:
        out = x2d
    return cast(TensorVariable, out[:, 0])


def _clamp_treatment(c_col: TensorVariable, params: dict, k: int) -> TensorVariable:
    """Apply this treatment's absolute held-level shock windows, when enabled."""
    schedule = params.get("treatment_shock")
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


def _reference_levels(
    params: dict,
    g_zc: np.ndarray,
    g_cc: np.ndarray,
    g_zz: np.ndarray,
    n_treatments: int,
    n_covariates: int,
    use_pulse: np.ndarray,
    *,
    dynamic_g: bool = False,
) -> list[TensorVariable]:
    """Per-treatment saturation anchors from parameters alone.

    The κ-relative response needs one fixed operating point per treatment. This
    returns it: a parameter-only REFERENCE level, built by applying the treatment
    softplus to

    * its own ``softplus(rw_c_mean)``;
    * ``pulse_amp * pulse_prob`` when pulses are enabled;
    * the weighted reference levels of the ``Z -> C`` and earlier ``C -> C``
      parents, accumulated in topological order.

    It is NOT ``E[C_k]``, and the gap has a definite sign. The treatment equation
    is ``C_k = softplus(pre-activation)`` and the treatment walk is itself
    ``softplus(centred path + rw_c_mean)``, so this construction takes
    ``softplus`` of a MEAN where the world takes the MEAN of a ``softplus``,
    twice. Softplus is strictly convex, so Jensen gives ``E[C_k] > anchor_k``
    strictly (measured ``E[C_k]/anchor_k`` in [1.004, 1.099] over 36
    (θ, treatment) cells at 600 noise draws each, every cell above 1; for a
    parentless, texture-free treatment at relative walk std 0.8 the excess is
    +6.0% to +7.8% across ``rw_c_mean`` in [0.3, 4.0]). Callers that need an
    expected level must average a realized series; this is a κ scale, not a
    moment.

    Reading no moment is the POINT. ``D -> C`` drops out because the latent
    factor is normalized to mean zero, and weekly jitter is mean-zero, so
    nothing here depends on the innovations: ``p(theta)`` is defined
    independently of the noise and the response at week ``t`` cannot depend on
    treatment at any ``t' > t``. A window statistic (the historical anchor) had both
    defects.

    The covariate texture needs no term here either: both of its terms are
    mean-zero (its pulse is centred on its own fire probability). A covariate
    applies NO activation, so unlike a treatment its level claim is exact —
    ``E[Z_m]`` is ``rw_z_mean`` plus its upstream ``Z -> Z`` terms, which is
    exactly the ``z_levels`` recursion below.
    """
    dot_kw = {"dynamic_g": dynamic_g}
    z_levels: list[TensorVariable] = []
    gamma_zz = _arr(params["gamma_zz"], (n_covariates, n_covariates))
    for m in range(n_covariates):
        level = pt.as_tensor_variable(params["rw_z"]["mean"][m])
        upstream = _dot_terms(
            [lvl[None] for lvl in z_levels[:m]], g_zz[:m, m], gamma_zz[:m, m], 1, **dot_kw
        )
        z_levels.append(level + upstream.reshape(()))

    rw_c_mean = params["rw_c"]["mean"]
    pulse_amp = _arr(params.get("pulse_amp", np.zeros(n_treatments)), (n_treatments,))
    pulse_prob = _arr(params.get("pulse_prob", np.zeros(n_treatments)), (n_treatments,))
    v_zc = _arr(params["v_zc"], (n_covariates, n_treatments))
    alpha_cc = _arr(params["alpha_cc"], (n_treatments, n_treatments))
    c_levels: list[TensorVariable] = []
    for k in range(n_treatments):
        own = pt.softplus(rw_c_mean[k])
        if use_pulse[k]:
            own = own + pulse_amp[k] * pulse_prob[k]
        term_z = _dot_terms([lvl[None] for lvl in z_levels], g_zc[:, k], v_zc[:, k], 1, **dot_kw)
        term_c = _dot_terms(
            [lvl[None] for lvl in c_levels[:k]], g_cc[:k, k], alpha_cc[:k, k], 1, **dot_kw
        )
        c_levels.append(pt.softplus(own + term_z.reshape(()) + term_c.reshape(())))
    return c_levels


def _saturate_family(
    name: str, ad_col: TensorVariable, saturation_scale: TensorVariable, params: dict, k: int
) -> TensorVariable:
    """One named κ-relative saturation family evaluated on ``ad_col``.

    ``linear`` is the only family without a κ-relative wrapper. The
    name-to-wrapper dispatch lives in mechanisms; these branches only bind each
    wrapper's distinct shape parameters.
    """
    if name == "linear":
        return cast(TensorVariable, ad_col / saturation_scale)
    family = mechanisms.SATURATION_FAMILIES[name]
    if name == "hill":
        return family(
            ad_col,
            saturation_scale,
            slope=params["hill_slope"][k],
            kappa_mult=params["hill_kappa_mult"][k],
        )
    if name == "logistic":
        return family(ad_col, saturation_scale, lam=params["logistic_lam"][k])
    if name == "michaelis_menten":
        return family(ad_col, saturation_scale, kappa_mult=params["mm_kappa_mult"][k])
    if name == "tanh":
        return family(ad_col, saturation_scale, c=params["tanh_c"][k])
    return family(ad_col, saturation_scale, alpha=params["root_alpha"][k])


def _saturate_col(
    ad_col: TensorVariable,
    saturation_scale: TensorVariable,
    params: dict,
    k: int,
    *,
    dynamic_family: bool = False,
) -> TensorVariable:
    """κ-relative saturation of an carryovered column using a *given* scale.

    ``saturation_scale`` is passed in so the same structural
    response function f_k can be evaluated on several treatment variants — the
    observed treatment, the base treatment, and the telescoping intervention
    variants — with an identical, pinned saturation scale. This is what makes
    the decomposition and the per-source indirect split exact.

    With a concrete family only that family is built. Under ``dynamic_family``
    the family id is a tensor, so all six are built and selected by
    ``pt.switch``.
    """
    if dynamic_family:
        sat_fam = pt.as_tensor_variable(params["sat_family"])[k]
        out = _saturate_family(SATURATION_FAMILY_KEYS[-1], ad_col, saturation_scale, params, k)
        for idx in range(len(SATURATION_FAMILY_KEYS) - 2, -1, -1):
            out = pt.switch(
                pt.eq(sat_fam, idx),
                _saturate_family(SATURATION_FAMILY_KEYS[idx], ad_col, saturation_scale, params, k),
                out,
            )
        return out
    name = SATURATION_FAMILY_KEYS[int(params["sat_family"][k])]  # concrete structural family
    return _saturate_family(name, ad_col, saturation_scale, params, k)


def build_symbolic_graph(
    g: dict[str, np.ndarray],
    params: dict[str, Any],
    n_time_steps: int,
    n_treatments: int,
    n_covariates: int,
    n_latent: int,
    *,
    burn_in: int = 0,
    eps: dict[str, Any],
    active: dict[str, Any] | None = None,
    dynamic_g: bool = False,
) -> dict[str, Any]:
    """Build the symbolic PyTensor graph for the additive causal DAG.

    Parameters
    ----------
    g : dict
        Edge indicators: ``g_cy`` (n_treatments,),
        ``g_dc`` (n_latent, n_treatments), ``g_dz`` (n_latent, n_covariates),
        ``g_dy`` (n_latent,), ``g_zy`` (n_covariates,),
        ``g_zc`` (n_covariates, n_treatments),
        ``g_cc`` (n_treatments, n_treatments, strictly upper-triangular),
        ``g_zz`` (n_covariates, n_covariates, strictly upper-triangular).
    params : dict
        SCM parameters — the continuous ones may be symbolic (PyMC RV)
        tensors or concrete numpy; families/smoothness are concrete. Assembled
        by :func:`pymc_generator.world_model.build_world_model`.
    eps : dict
        The caller's noise RVs, each with leading dim
        ``n_time_steps_full = n_time_steps + burn_in``:
        ``eps_d`` (n_time_steps_full, n_latent), ``eps_z`` (n_time_steps_full, n_covariates),
        ``eps_c`` (n_time_steps_full, n_treatments), ``eps_b`` (n_time_steps_full,),
        ``eps_y`` (n_time_steps_full,), the treatment-texture noise
        ``eps_c_hf`` (weekly jitter) / ``eps_c_pulse`` (a 0/1 Bernoulli fire),
        and the covariate-texture noise ``eps_z_hf`` (n_time_steps_full,
        n_covariates) / ``eps_z_pulse`` (a 0/1 Bernoulli fire, centred by the
        covariate equation). The covariate pair is required only when the covariate
        texture is enabled.
    n_time_steps, n_treatments, n_covariates, n_latent : int
        Time steps and node counts.
    burn_in : int
        Extra leading weeks simulated then dropped from every output. The
        carryover convolution left-pads with zeros, so without burn-in each
        contribution ramps 0 -> level over the first ``l_max`` weeks — an
        artifact that dominates smooth additive targets (measured 3–13x the
        steady-state std). Legal values are ``0`` (off) or ``>= l_max``; with
        ``burn_in >= l_max`` the reported window sees real history instead of
        zeros. The κ-relative saturation scale comes from
        ``_reference_levels(...)``, so it uses drawn parameters alone and is
        independent of this window and every realized series.
    active : dict, optional
        Per-node 0/1 activity flags ``active_treatment`` (n_treatments,),
        ``active_covariate`` (n_covariates,), ``active_latent`` (n_latent,).
        Inactive nodes are zeroed in place (see :func:`inactive_zero`) instead of
        being left out of the graph, so one max-size graph can serve cells with
        fewer nodes. Omit it to build at exactly the given node counts.
    dynamic_g : bool
        Treat ``g``, the mechanism families, and ``active`` as tensors whose
        values arrive at call time (typically ``pm.Data``) rather than as
        concrete structure. This trades a denser graph — every candidate edge is
        wired and every mechanism family is built behind a ``pt.switch`` — for
        the ability to swap DAGs without recompiling. Leave it False for the
        per-world path, which stays exactly as sparse as its structure.

    Returns
    -------
    dict with a single key ``outputs`` — the symbolic node outputs (sliced to
    the reported ``n_time_steps`` window):
        ``latent_unobserved`` (n_time_steps, n_latent),
        ``covariates`` (n_time_steps, n_covariates),
        ``treatments`` (n_time_steps, n_treatments),
        ``treatments_base`` (n_time_steps, n_treatments),
        ``saturation_scale`` (n_treatments,), ``baseline`` (n_time_steps,),
        ``baseline_intrinsic`` (n_time_steps,),
        ``covariate_contribution`` (n_time_steps, n_covariates),
        ``latent_unobserved_contribution`` (n_time_steps, n_latent),
        ``contributions`` (n_time_steps, n_treatments) — direct,
        ``contributions_observed`` (n_time_steps, n_treatments),
        ``indirect_effects`` (n_time_steps,),
        ``indirect_effects_by_source`` (n_time_steps, 3),
        ``outcome`` (n_time_steps,), ``outcome_noise`` (n_time_steps,).

    Notes
    -----
    Per-node terms split the aggregated D->Y / Z->Y baseline-side terms:
    ``covariate_contribution[:, m] = g_zy[m]·ρ[m]·Z[:, m]`` and
    ``latent_unobserved_contribution[:, j] = g_dy[j]·δ[j]·D[:, j]``, with
    ``baseline_intrinsic = B`` (the floored intercept alone) and
    ``outcome_noise = RW_Y``, so
    ``baseline_intrinsic + outcome_noise + Σ_j latent_unobserved_contribution
    + Σ_m covariate_contribution == baseline``.

    ``indirect_effects_by_source`` (n_time_steps, 3) is the telescoping 3-way
    indirect split in the LOCKED order ``(cc, zc, dc)`` — treatment->treatment,
    covariate->treatment, hidden-confounder->treatment — defined by sequential
    graph-surgery interventions (see the inline derivation). The three columns
    sum exactly to ``indirect_effects``.
    """
    if dynamic_g:
        # The masks are tensors here, so acyclicity cannot be checked while
        # building. Callers own that guarantee; the sampler only ever emits
        # strictly-upper-triangular C->C and Z->Z blocks.
        pass
    else:
        _check_strict_upper(g["g_cc"], "g_cc")
        _check_strict_upper(g["g_zz"], "g_zz")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    if dynamic_g:
        g_cy = _arr(g["g_cy"], (n_treatments,))
        g_dc = _arr(g["g_dc"], (n_latent, n_treatments))
        g_dz = _arr(g["g_dz"], (n_latent, n_covariates))
        g_dy = _arr(g["g_dy"], (n_latent,))
        g_zy = _arr(g["g_zy"], (n_covariates,))
        g_zc = _arr(g["g_zc"], (n_covariates, n_treatments))
        g_cc = _arr(g["g_cc"], (n_treatments, n_treatments))
        g_zz = _arr(g["g_zz"], (n_covariates, n_covariates))
    else:
        g_cy = np.asarray(g["g_cy"], dtype="float64")
        g_dc = np.asarray(g["g_dc"], dtype="float64").reshape(n_latent, n_treatments)
        g_dz = np.asarray(g["g_dz"], dtype="float64").reshape(n_latent, n_covariates)
        g_dy = np.asarray(g["g_dy"], dtype="float64").reshape(n_latent)
        g_zy = np.asarray(g["g_zy"], dtype="float64").reshape(n_covariates)
        g_zc = np.asarray(g["g_zc"], dtype="float64").reshape(n_covariates, n_treatments)
        g_cc = np.asarray(g["g_cc"], dtype="float64").reshape(n_treatments, n_treatments)
        g_zz = np.asarray(g["g_zz"], dtype="float64").reshape(n_covariates, n_covariates)

    # Treatment texture. Magnitudes (hf_sigma, pulse_amp) may be symbolic (RV)
    # params; the per-treatment ENABLE flags are concrete structure, normally
    # passed in as ``use_hf`` / ``use_pulse`` (build_world_model sets them) and
    # derived from the magnitudes as a fallback when absent (a symbolic
    # magnitude has no concrete truth value). The pulse enters as a 0/1 FIRE
    # indicator ``eps_c_pulse`` ~ Bernoulli(pulse_prob), so no threshold lives
    # in the graph.
    hf_sigma = _arr(params.get("hf_sigma", np.zeros(n_treatments)), (n_treatments,))
    pulse_amp = _arr(params.get("pulse_amp", np.zeros(n_treatments)), (n_treatments,))
    use_hf = params.get("use_hf")
    if use_hf is None:
        use_hf = (
            np.asarray(params.get("hf_sigma", np.zeros(n_treatments)), dtype="float64").reshape(
                n_treatments
            )
            > 0.0
        )
    use_hf = np.asarray(use_hf).reshape(n_treatments)
    use_pulse = params.get("use_pulse")
    if use_pulse is None:
        _pa = np.asarray(params.get("pulse_amp", np.zeros(n_treatments)), dtype="float64").reshape(
            n_treatments
        )
        _pp = np.asarray(params.get("pulse_prob", np.zeros(n_treatments)), dtype="float64").reshape(
            n_treatments
        )
        use_pulse = (_pa != 0.0) & (_pp > 0.0)
    use_pulse = np.asarray(use_pulse).reshape(n_treatments)

    # Covariate texture, the signed counterpart of the treatment texture above:
    # magnitudes (covariate_hf_sigma, covariate_pulse_amp) may be symbolic (RV)
    # params, the per-covariate ENABLE flags are concrete structure (normally
    # ``use_covariate_hf`` / ``use_covariate_pulse`` from build_world_model, derived
    # from concrete magnitudes as a fallback when absent). The pulse enters
    # CENTRED — ``eps_z_pulse - covariate_pulse_prob`` with ``eps_z_pulse ~
    # Bernoulli(covariate_pulse_prob)`` — so a covariate's expected level is still
    # its walk mean (EXACTLY: a covariate applies no activation) and no
    # parameter-only reference level moves.
    covariate_hf_sigma = _arr(
        params.get("covariate_hf_sigma", np.zeros(n_covariates)), (n_covariates,)
    )
    covariate_pulse_amp = _arr(
        params.get("covariate_pulse_amp", np.zeros(n_covariates)), (n_covariates,)
    )
    covariate_pulse_prob = _arr(
        params.get("covariate_pulse_prob", np.zeros(n_covariates)), (n_covariates,)
    )
    use_covariate_hf = params.get("use_covariate_hf")
    if use_covariate_hf is None:
        use_covariate_hf = (
            np.asarray(
                params.get("covariate_hf_sigma", np.zeros(n_covariates)), dtype="float64"
            ).reshape(n_covariates)
            > 0.0
        )
    use_covariate_hf = np.asarray(use_covariate_hf).reshape(n_covariates)
    use_covariate_pulse = params.get("use_covariate_pulse")
    if use_covariate_pulse is None:
        _ca = np.asarray(
            params.get("covariate_pulse_amp", np.zeros(n_covariates)), dtype="float64"
        ).reshape(n_covariates)
        _cp = np.asarray(
            params.get("covariate_pulse_prob", np.zeros(n_covariates)), dtype="float64"
        ).reshape(n_covariates)
        use_covariate_pulse = (_ca != 0.0) & (_cp > 0.0)
    use_covariate_pulse = np.asarray(use_covariate_pulse).reshape(n_covariates)

    # Per-node activity. Absent (the per-world path) every node is present and
    # `_mask` is the identity, so that path builds exactly the graph it did
    # before this switch existed.
    if active is None:
        active_c: Any = None
        active_m: Any = None
        active_j: Any = None
    elif isinstance(active["active_treatment"], TensorVariable):
        active_c = active["active_treatment"]
        active_m = active["active_covariate"]
        active_j = active["active_latent"]
    else:
        active_c = np.asarray(active["active_treatment"], dtype="float64").reshape(n_treatments)
        active_m = np.asarray(active["active_covariate"], dtype="float64").reshape(n_covariates)
        active_j = np.asarray(active["active_latent"], dtype="float64").reshape(n_latent)

    def _mask(flags, i: int, expr: TensorVariable) -> TensorVariable:
        return expr if flags is None else inactive_zero(flags[i], expr)

    dot_kw = {"dynamic_g": dynamic_g}
    family_kw = {"dynamic_family": True} if dynamic_g else {}

    # Nodes are simulated over n_time_steps_full = burn_in + n_time_steps weeks; every
    # output is sliced to the last n_time_steps (the reported window).
    n_time_steps_full = n_time_steps + burn_in
    window = slice(burn_in, None)

    # Noise inputs are the caller's RVs — pm.Normal walks + weekly jitter and a
    # pm.Bernoulli 0/1 pulse — so the whole graph is drawn with no free inputs.
    # Static shapes are required: the carryover convolution builds its
    # sliding-window index from the static time length.
    eps_d, eps_z, eps_c = eps["eps_d"], eps["eps_z"], eps["eps_c"]
    eps_b, eps_y = eps["eps_b"], eps["eps_y"]
    eps_c_hf, eps_c_pulse = eps["eps_c_hf"], eps["eps_c_pulse"]
    # Covariate texture noise is optional for direct concrete callers: a config
    # with the texture disabled builds the pre-texture covariate equation exactly
    # and never reads these. An ENABLED term with no innovation is a caller bug,
    # not a silent fallback to zero.
    eps_z_hf = _required_eps(eps, "eps_z_hf", bool(use_covariate_hf.any()))
    eps_z_pulse = _required_eps(eps, "eps_z_pulse", bool(use_covariate_pulse.any()))

    # -- confounders D (n_time_steps_full, n_latent): pure random walks -------------------------
    d_cols = [
        _mask(active_j, j, _walk_column(eps_d[:, j], params["rw_d"], j, n_time_steps_full))
        for j in range(n_latent)
    ]
    D = pt.stack(d_cols, axis=1) if n_latent > 0 else pt.zeros((n_time_steps_full, 0))

    # -- covariates Z (n_time_steps_full, n_covariates): D->Z + upstream Z->Z + own drive ---
    u_dz = _arr(params["u_dz"], (n_latent, n_covariates))
    gamma_zz = _arr(params["gamma_zz"], (n_covariates, n_covariates))
    z_cols: list[TensorVariable] = []
    for m in range(n_covariates):
        # Own exogenous drive = smoothed walk + iid weekly noise + centred
        # calendar pulses. Without the high-frequency terms a covariate is a
        # smoothed walk over the SAME function space as the baseline walk, so
        # rho_zy trades off against baseline drift and Z->Y is only weakly
        # identified. Both terms are mean-zero (the pulse subtracts its own fire
        # probability), so E[Z_m] is unchanged and _reference_levels stays exact.
        own = _walk_column(eps_z[:, m], params["rw_z"], m, n_time_steps_full)
        if use_covariate_hf[m]:
            own = own + covariate_hf_sigma[m] * eps_z_hf[:, m]
        if use_covariate_pulse[m]:
            own = own + covariate_pulse_amp[m] * (eps_z_pulse[:, m] - covariate_pulse_prob[m])
        term_d = _dot_terms(d_cols, g_dz[:, m], u_dz[:, m], n_time_steps_full, **dot_kw)
        term_z = _dot_terms(z_cols[:m], g_zz[:m, m], gamma_zz[:m, m], n_time_steps_full, **dot_kw)
        z_cols.append(_mask(active_m, m, term_d + term_z + own))
    Z = pt.stack(z_cols, axis=1) if n_covariates > 0 else pt.zeros((n_time_steps_full, 0))

    # -- treatments C (n_time_steps_full, n_treatments): D->C + Z->C + upstream C->C + own drive -----
    # C_base: same walks, all incoming interaction terms zeroed (the
    # "no upstream" intervention used for the exact decomposition).
    w_dc = _arr(params["w_dc"], (n_latent, n_treatments))
    v_zc = _arr(params["v_zc"], (n_covariates, n_treatments))
    alpha_cc = _arr(params["alpha_cc"], (n_treatments, n_treatments))
    c_cols: list[TensorVariable] = []
    c_unshocked_cols: list[TensorVariable] = []
    c_base_cols: list[TensorVariable] = []
    # Telescoping intervention variants (see indirect_effects_by_source, below):
    #   c_no_cc      = treatment with the C->C term dropped
    #   c_no_cc_zc   = treatment with C->C and Z->C dropped
    # These reuse the SAME term_d/term_z/walk as the observed treatment; each just
    # omits the named upstream term (the C->C term references upstream OBSERVED
    # treatments, so dropping it is the "zero that interaction" intervention).
    c_no_cc_cols: list[TensorVariable] = []
    c_no_cc_zc_cols: list[TensorVariable] = []
    for k in range(n_treatments):
        walk = _walk_column(eps_c[:, k], params["rw_c"], k, n_time_steps_full)
        # Own drive includes the same walk, execution noise, and pulses in
        # every intervention. eps_c_pulse[:, k] is a Bernoulli fire indicator;
        # pulse magnitudes may be symbolic.
        own = walk
        if use_hf[k]:
            own = own + hf_sigma[k] * eps_c_hf[:, k]
        if use_pulse[k]:
            own = own + pulse_amp[k] * eps_c_pulse[:, k]
        term_d = _dot_terms(d_cols, g_dc[:, k], w_dc[:, k], n_time_steps_full, **dot_kw)
        term_z = _dot_terms(z_cols, g_zc[:, k], v_zc[:, k], n_time_steps_full, **dot_kw)
        # The natural recursion is retained solely for the realism reference.
        # The observed recursion instead sees already-clamped upstream parents,
        # which is the SCM meaning of a treatment intervention.
        term_c_unshocked = _dot_terms(
            c_unshocked_cols[:k], g_cc[:k, k], alpha_cc[:k, k], n_time_steps_full, **dot_kw
        )
        term_c = _dot_terms(c_cols[:k], g_cc[:k, k], alpha_cc[:k, k], n_time_steps_full, **dot_kw)
        # softplus guard: treatment-like treatments must stay non-negative even
        # when signed upstream contributions push the pre-activation down
        c_unshocked_cols.append(
            _mask(active_c, k, pt.softplus(term_d + term_z + term_c_unshocked + own))
        )
        c_cols.append(
            _mask(
                active_c,
                k,
                _clamp_treatment(pt.softplus(term_d + term_z + term_c + own), params, k),
            )
        )
        c_base_cols.append(_mask(active_c, k, _clamp_treatment(pt.softplus(own), params, k)))
        c_no_cc_cols.append(
            _mask(active_c, k, _clamp_treatment(pt.softplus(term_d + term_z + own), params, k))
        )
        c_no_cc_zc_cols.append(
            _mask(active_c, k, _clamp_treatment(pt.softplus(term_d + own), params, k))
        )
    C = pt.stack(c_cols, axis=1)
    C_base = pt.stack(c_base_cols, axis=1)

    # -- intercept B and the non-treatment aggregate, optionally floored -----------
    # D and Z do NOT enter the intercept: they attach directly to Y below, so
    # the intercept is a pure, separately reported level and Y reads as the
    # equation a standard MMM assumes.
    #
    # ``baseline_floor_scope`` decides WHAT the floor clips.
    #
    # "intercept": clip the intercept walk only. Every other term stays exactly
    #     linear in its node (``covariate_contribution[:, m] == g_zy·ρ·Z``), which
    #     is the cheapest, most estimator-friendly option — but a large negative
    #     ρ·Z can still drag the non-treatment total (and outcome) below zero.
    # "non_treatment": clip the RUNNING TOTAL as each parent is added, in the LOCKED
    #     order intercept -> confounders (j ascending) -> covariates (m ascending).
    #     Each per-node column is then the telescoping difference it caused,
    #     ``A_i - A_{i-1}``, exactly as ``indirect_effects_by_source`` is defined
    #     for treatments. Three consequences, all of them the point:
    #       * the non-treatment total is >= floor by construction, so a negative
    #         covariate effect is credited only down to the floor and the excess is
    #         absorbed rather than pushing outcome negative;
    #       * the columns still telescope EXACTLY, so the decomposition identity
    #         is untouched;
    #       * where the floor does not bind, every column is bit-identical to the
    #         linear split, so this is a clip and never a re-parameterisation.
    delta_dy = _arr(params["delta_dy"], (n_latent,))
    rho_zy = _arr(params["rho_zy"], (n_covariates,))
    walk_b = _walk_column(eps_b, params["rw_b"], 0, n_time_steps_full)
    baseline_floor = params.get("baseline_floor")
    floor_scope = params.get("baseline_floor_scope", "intercept")
    absorb = baseline_floor is not None and floor_scope == "non_treatment"

    def _clip(expr):
        return expr if baseline_floor is None else pt.maximum(expr, float(baseline_floor))

    intercept = _clip(walk_b)
    term_dy = _dot_terms(d_cols, g_dy, delta_dy, n_time_steps_full, **dot_kw)
    term_zy = _dot_terms(z_cols, g_zy, rho_zy, n_time_steps_full, **dot_kw)

    # -- per-node direct baseline terms (an exact split of term_dy / term_zy) --
    if absorb:
        # Sequential graph surgery on the running non-treatment total. ``running`` is
        # the clipped total after each node joins; the column a node contributes
        # is the change it caused. Nodes with no edge are skipped outright rather
        # than added with a zero coefficient: a zero-gated term would still make
        # that node's innovation an ancestor of ``baseline``, which changes which
        # RNGs the compiled graph reaches and therefore every seeded draw. Under
        # ``dynamic_g`` the masks are tensors, so the full chain is wired (the
        # template graph is dense by design).
        def _edge_live(mask, i) -> bool:
            return dynamic_g or bool(np.asarray(mask)[i])

        zero_col = pt.zeros(n_time_steps_full)
        running = intercept
        confounder_contrib_cols = []
        for j in range(n_latent):
            if not _edge_live(g_dy, j):
                confounder_contrib_cols.append(zero_col)
                continue
            nxt = _clip(running + (g_dy[j] * delta_dy[j]) * d_cols[j])
            confounder_contrib_cols.append(nxt - running)
            running = nxt
        covariate_contrib_cols = []
        for m in range(n_covariates):
            if not _edge_live(g_zy, m):
                covariate_contrib_cols.append(zero_col)
                continue
            nxt = _clip(running + (g_zy[m] * rho_zy[m]) * z_cols[m])
            covariate_contrib_cols.append(nxt - running)
            running = nxt
        non_treatment = running
    else:
        # column m of covariate_contribution   = g_zy[m]·ρ[m]·Z[:,m]  (sums to term_zy)
        # column j of latent_unobserved_contribution = g_dy[j]·δ[j]·D[:,j] (sums to term_dy)
        covariate_contrib_cols = [(g_zy[m] * rho_zy[m]) * z_cols[m] for m in range(n_covariates)]
        confounder_contrib_cols = [(g_dy[j] * delta_dy[j]) * d_cols[j] for j in range(n_latent)]
        non_treatment = intercept + term_dy + term_zy
    covariate_contribution = (
        pt.stack(covariate_contrib_cols, axis=1)
        if n_covariates > 0
        else pt.zeros((n_time_steps_full, 0))
    )  # (n_time_steps_full, n_covariates)
    latent_unobserved_contribution = (
        pt.stack(confounder_contrib_cols, axis=1)
        if n_latent > 0
        else pt.zeros((n_time_steps_full, 0))
    )  # (n_time_steps_full, n_latent)

    # -- direct nonlinear responses + exact decomposition --------------------
    beta = _arr(params["beta"], (n_treatments,))
    contrib_obs_cols, contrib_base_cols, sat_scale_cols = [], [], []
    # Telescoping 3-way indirect split (LOCKED order cc -> zc -> dc). Because the
    # direct response f_k is nonlinear, naive one-at-a-time interventions do not
    # sum to the total; the split is defined by a FIXED sequential zeroing order
    # using the SAME pinned saturation scale for every variant:
    #   ie_cc = Y(all)                          - Y(zero c_from_c)
    #   ie_zc = Y(zero c_from_c)                - Y(zero c_from_c, c_from_z)
    #   ie_dc = Y(zero c_from_c, c_from_z)      - Y(zero c_from_c, c_from_z, c_from_d)
    # These telescope exactly to indirect_effects because Y(zero all three) is
    # baseline + the direct (base-treatment) contributions.
    ie_cc_cols, ie_zc_cols, ie_dc_cols = [], [], []
    treatment_levels = _reference_levels(
        params, g_zc, g_cc, g_zz, n_treatments, n_covariates, use_pulse, dynamic_g=dynamic_g
    )
    for k in range(n_treatments):
        # Carryover over the full simulated horizon, then slice to the reported
        # window: with burn_in >= l_max the window's convolution sees real
        # pre-window history instead of the zero padding (warmup artifact).
        # Held-level shocks clamp the treatment BEFORE this convolution and never
        # touch its response state, so the same normalized causal kernel a
        # standard MMM applies reproduces this response exactly.
        # The κ scale is the treatment's PARAMETER-ONLY reference level, never a
        # statistic of the drawn series and never E[C_k] (softplus makes E[C_k]
        # strictly larger): that keeps theta independent of the noise and keeps
        # the response at week t free of treatment at t' > t.
        ad_obs = _carryover_col(c_cols[k], params, k, **family_kw)[window]
        scale_k = pt.maximum(treatment_levels[k], 1e-8).copy(name=f"sat_scale_{k}")
        sat_scale_cols.append(scale_k)

        # ONE response function per treatment, applied to every variant: the
        # telescoping split cancels the middle variants, so a divergence here
        # would pass the identity tests while corrupting the per-source split.
        def _f(ad_col, *, _scale=scale_k, _k=k):
            return _saturate_col(ad_col, _scale, params, _k, **family_kw)

        f_obs = _f(ad_obs)
        f_base = _f(_carryover_col(c_base_cols[k], params, k, **family_kw)[window])
        f_no_cc = _f(_carryover_col(c_no_cc_cols[k], params, k, **family_kw)[window])
        f_no_cc_zc = _f(_carryover_col(c_no_cc_zc_cols[k], params, k, **family_kw)[window])
        gate = g_cy[k] * beta[k]  # g concrete, beta possibly symbolic
        contrib_obs_cols.append(_mask(active_c, k, gate * f_obs))
        contrib_base_cols.append(_mask(active_c, k, gate * f_base))
        ie_cc_cols.append(_mask(active_c, k, gate * (f_obs - f_no_cc)))
        ie_zc_cols.append(_mask(active_c, k, gate * (f_no_cc - f_no_cc_zc)))
        ie_dc_cols.append(_mask(active_c, k, gate * (f_no_cc_zc - f_base)))
    contributions_observed = pt.stack(contrib_obs_cols, axis=1)  # (n_time_steps, n_treatments)
    contributions = pt.stack(contrib_base_cols, axis=1)  # (n_time_steps, n_treatments) direct
    indirect_effects = (contributions_observed - contributions).sum(axis=1)  # (n_time_steps,)

    # n_treatments == 0 is unsupported (the contributions stack above already
    # requires n_treatments >= 1), so the ie stacks need no separate guard.
    # Graph surgery against an absent edge family is exactly zero. Returning a
    # literal zero avoids machine-epsilon subtraction residue in persisted truth
    # labels, especially for the structurally edge-free n_treatments=1 C->C block.
    # Under dynamic_g the masks are tensors, so the edge-free shortcut cannot be
    # taken; the surgery difference is exactly zero in that case anyway, up to
    # the epsilon residue the shortcut exists to avoid.
    def _ie(cols: list, mask) -> TensorVariable:
        if not dynamic_g and not np.asarray(mask).any():
            return pt.zeros(n_time_steps)
        return cast(TensorVariable, pt.stack(cols, axis=1).sum(axis=1))

    ie_cc = _ie(ie_cc_cols, g_cc)
    ie_zc = _ie(ie_zc_cols, g_zc)
    ie_dc = _ie(ie_dc_cols, g_dc)
    # (n_time_steps, 3), columns in the locked order cc, zc, dc
    indirect_effects_by_source = pt.stack([ie_cc, ie_zc, ie_dc], axis=1)

    # ``as_tensor_variable`` matters on the concrete path: with numpy params AND
    # numpy eps this product is a plain ndarray, and ``outcome_noise`` is now an
    # OUTPUT in its own right, so every output must be a tensor.
    walk_y = pt.as_tensor_variable(params["rw_y"]["std"][0] * eps_y)
    # ``baseline`` stays "everything that is not treatment": the intercept, the
    # direct D->Y / Z->Y terms, and the iid outcome noise.
    baseline = (non_treatment + walk_y)[window]
    # The intercept ALONE — floored when a floor is configured, so this target
    # is >= floor by construction. The iid outcome noise is reported separately
    # (``outcome_noise``) instead of being folded in here, which is what keeps
    # this a clean level rather than a level plus observation error.
    baseline_intrinsic = intercept[window]  # (n_time_steps,)
    outcome_noise = walk_y[window]  # (n_time_steps,)
    outcome = baseline + contributions_observed.sum(axis=1)
    # identity: outcome == baseline + contributions.sum(1) + indirect_effects
    #        == baseline_intrinsic + outcome_noise
    #           + Σ latent_unobserved_contribution + Σ covariate_contribution
    #           + contributions.sum(1) + indirect_effects_by_source.sum(1)
    #
    # Outcome is deliberately NOT floored. A clamp on Y is a LIKELIHOOD-level
    # change (censored observations), which would put every world outside the
    # additive-Gaussian class a standard MMM — and this package's own oracle —
    # can represent. Non-negative outcome stays enforced exactly by the
    # acceptance filter (``_additive_task_ok``).

    outputs = {
        "latent_unobserved": D[window],
        "covariates": Z[window],
        "treatments": C[window],
        "treatments_base": C_base[window],
        "saturation_scale": pt.stack(sat_scale_cols),
        "baseline": baseline,
        "baseline_intrinsic": baseline_intrinsic,
        "covariate_contribution": covariate_contribution[window],
        "latent_unobserved_contribution": latent_unobserved_contribution[window],
        "contributions": contributions,
        "contributions_observed": contributions_observed,
        "indirect_effects": indirect_effects,
        "indirect_effects_by_source": indirect_effects_by_source,
        "outcome": outcome,
        # Appended last: output order fixes PyTensor's RNG traversal, so a new
        # output must not displace an existing one.
        "outcome_noise": outcome_noise,
    }
    # These are intentionally audit-only paths.  They are drawn to decide
    # whether the *natural* world is realistic, never persisted in corpora.
    if params.get("treatment_shock") is not None:
        # Reuse the pinned parameter-only anchors. This leaves every persisted
        # response array unchanged; only these audit-only outputs and
        # realism-filter acceptance can move.
        unshocked_contribs = []
        for k in range(n_treatments):
            ad_unshocked = _carryover_col(c_unshocked_cols[k], params, k)[window]
            scale_k = sat_scale_cols[k]
            unshocked_contribs.append(
                g_cy[k] * beta[k] * _saturate_col(ad_unshocked, scale_k, params, k)
            )
        outputs["treatments_unshocked"] = pt.stack(c_unshocked_cols, axis=1)[window]
        outputs["outcome_unshocked"] = baseline + pt.stack(unshocked_contribs, axis=1).sum(axis=1)
    return {"outputs": outputs}
