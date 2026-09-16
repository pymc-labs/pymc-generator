"""Plain-text world descriptions — "get a description of the world".

:func:`describe_scm` renders everything a human needs to audit one world:
the vector-valued structural equations, DAG edges with drawn coefficients,
node connectivity, realized mechanism and texture parameters, prior ranges,
exact decomposition-identity check, and per-treatment signal metrics.
"""

from __future__ import annotations

import io
import json

import numpy as np

from .mechanisms import SATURATION_PRIOR_RANGES
from .worlds import (
    CARRYOVER_NAMES,
    SATURATION_NAMES,
    SCM,
    edges_with_coeffs,
    mechanism_label,
    node_status,
    treatment_role,
)


def _saturation_description(params: dict, k: int, family: str) -> str:
    """Render only the shape parameters used by a saturation family."""
    if family == "hill":
        return (
            f"hill(slope={params['hill_slope'][k]:.2f},"
            f"kappa_mult={params['hill_kappa_mult'][k]:.2f})"
        )
    if family == "logistic":
        return f"logistic(lam={params['logistic_lam'][k]:.2f})"
    if family == "michaelis_menten":
        return f"michaelis_menten(kappa_mult={params['mm_kappa_mult'][k]:.2f})"
    if family == "tanh":
        return f"tanh(c={params['tanh_c'][k]:.2f})"
    if family == "root":
        return f"root(alpha={params['root_alpha'][k]:.2f})"
    return family


def _treatment_lines(g: dict, params: dict) -> list[str]:
    n_treatments = len(g["g_cy"])
    lines = []
    for k in range(n_treatments):
        sat = SATURATION_NAMES[int(params["sat_family"][k])]
        ad = CARRYOVER_NAMES[int(params["carryover_family"][k])]
        bits = [
            f"C{k + 1}: role={treatment_role(g, k)}",
            f"beta={params['beta'][k]:.2f}",
            f"carryover={ad}"
            + (f"(alpha={params['carryover_alpha'][k]:.2f})" if ad == "geometric" else "")
            + (
                f"(lam={params['weibull_lam'][k]:.1f},k={params['weibull_k'][k]:.1f})"
                if ad == "weibull"
                else ""
            ),
            f"saturation={_saturation_description(params, k, sat)}",
            f"walk(mean={params['rw_c']['mean'][k]:.2f},std={params['rw_c']['std'][k]:.2f},"
            f"smooth={params['rw_c']['smoothness'][k]:.2f})",
            f"hf_sigma={params['hf_sigma'][k]:.2f}",
            f"pulse(p={params['pulse_prob'][k]:.2f},amp={params['pulse_amp'][k]:.2f})",
        ]
        lines.append("  " + "  ".join(bits))
    return lines


def _covariate_lines(g: dict, params: dict) -> list[str]:
    n_covariates = len(g["g_zy"])
    lines = []
    for m in range(n_covariates):
        bits = [
            f"Z{m + 1}: zy={int(g['g_zy'][m])}",
            f"walk(mean={params['rw_z']['mean'][m]:.2f},std={params['rw_z']['std'][m]:.2f},"
            f"smooth={params['rw_z']['smoothness'][m]:.2f})",
            f"covariate_hf_sigma={params['covariate_hf_sigma'][m]:.2f}",
            f"pulse(p={params['covariate_pulse_prob'][m]:.2f},"
            f"amp={params['covariate_pulse_amp'][m]:.2f},centred)",
        ]
        lines.append("  " + "  ".join(bits))
    return lines


def _format_signal_metric(signal: dict, key: str, index: int) -> str:
    """Render an estimable signal metric, or ``n/a`` when it is unavailable."""
    if not signal[f"{key}_valid"][index]:
        return "n/a"
    return f"{signal[key][index]:.2f}"


def _texture_label(params: dict) -> str:
    """Summarize the REALIZED texture: how many nodes drew each drive term.

    Never a preset name. The texture preset is not recorded on the world, and
    a bare :class:`~pymc_generator.sampler.SCMPrior` defaults every
    high-frequency range to ``(0.0, 0.0)`` — so a hardcoded "diverse" label
    describes a world that may have none of it. The drawn ``use_*`` flags are
    the ground truth: they are what the structural equations branch on.
    """
    parts = []
    for label, key in (
        ("treatment hf", "use_hf"),
        ("treatment pulse", "use_pulse"),
        ("covariate hf", "use_covariate_hf"),
        ("covariate pulse", "use_covariate_pulse"),
    ):
        flags = np.asarray(params[key])
        parts.append(f"{label} {int(np.count_nonzero(flags))}/{flags.size}")
    return "drawn: " + ", ".join(parts)


def describe_scm(world: SCM) -> str:
    """Render the full plain-text description of one :class:`SCM`.

    Sections: title + purpose, sizes, edge budget, edge census, active edges
    with drawn coefficients, node connectivity, per-treatment mechanism +
    texture, prior ranges, vector-valued structural equations, exact replay
    inputs, decomposition-identity error, and per-direct-treatment signal metrics.
    """
    g, params, cfg = world.g, world.params, world.cfg
    edges = edges_with_coeffs(g, params)
    ident_err = world.identity_error()
    signal = world.signal()

    f = io.StringIO()
    f.write(f"Dataset: {world.name} (additive SCM)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Purpose:\n  {world.purpose}\n\n")
    f.write(
        f"Sizes: n_time_steps={cfg.n_time_steps}, n_treatments={world.n_treatments}, "
        f"n_covariates={world.n_covariates}, n_latent={world.n_latent}, "
    )
    f.write(f"l_max={cfg.l_max}, carryover_burn_in={cfg.carryover_burn_in}\n")
    f.write(f"Edge budget: {cfg.edge_budget}\n\n")
    counts: dict[str, int] = {}
    for et, _src, _dst, _c in edges:
        counts[et] = counts.get(et, 0) + 1
    f.write("Edge census: " + "  ".join(f"{et}={n}" for et, n in sorted(counts.items())) + "\n\n")
    f.write("Active edges (with drawn coefficients):\n")
    for et, src, dst, coef in edges:
        f.write(f"  [{et}] {src} -> {dst}   coef={coef:+.3f}\n")
    status = node_status(g)
    n_iso = sum(s == "isolated" for s in status.values())
    f.write("\nNode connectivity (connected = directed path to Y; isolated = no\n")
    f.write("  edges at all, a deliberate zero-attribution null; dead-ends are\n")
    f.write("  never generated):\n")
    f.write("  " + "  ".join(f"{n}={s}" for n, s in status.items()) + "\n")
    if n_iso:
        f.write(
            f"  -> {n_iso} isolated null node(s): attribution traps, the model "
            "must credit them zero.\n"
        )
    f.write("\nChannels (mechanism + own-drive texture):\n")
    f.write("\n".join(_treatment_lines(g, params)) + "\n\n")
    f.write("Covariates (walk + own-drive texture):\n")
    f.write("\n".join(_covariate_lines(g, params)) + "\n\n")
    prior_cond = world.extras.get("prior_cond")
    if prior_cond:
        spec = cfg.prior_cond_spec()
        f.write("Prior conditioning (ACE): narrowed per-cell prior intervals\n")
        f.write("  (mechanism shape params above were drawn from these):\n")
        for q, (lo, width) in prior_cond.items():
            s_lo, s_hi = spec[q]["support"]
            f.write(
                f"  {q}: U({lo:.3f}, {lo + width:.3f})  width={width:.3f}  "
                f"support=({s_lo}, {s_hi})\n"
            )
        f.write("\n")
    f.write(f"Texture prior ({_texture_label(params)}):\n")
    f.write(f"  rw_treatment_std_range={cfg.rw_treatment_std_range} (relative)\n")
    f.write(f"  treatment_hf_sigma_range={cfg.treatment_hf_sigma_range} (relative)\n")
    f.write(f"  treatment_pulse_prob_range={cfg.treatment_pulse_prob_range}\n")
    f.write(f"  treatment_pulse_amp_range={cfg.treatment_pulse_amp_range} (relative)\n")
    f.write(f"  rw_positive_mean_range={cfg.rw_positive_mean_range}\n")
    f.write(f"  covariate_hf_sigma_range={cfg.covariate_hf_sigma_range} (relative to rw_z_std)\n")
    f.write(f"  covariate_pulse_prob_range={cfg.covariate_pulse_prob_range}\n")
    f.write(
        f"  covariate_pulse_amp_range={cfg.covariate_pulse_amp_range} (relative to rw_z_std, centred)\n"
    )
    f.write(f"  beta_additive_range={cfg.beta_additive_range}\n")
    f.write("  saturation prior ranges: ")
    f.write(json.dumps(SATURATION_PRIOR_RANGES) + "\n\n")
    f.write("Structural equations (vector-valued; active parents only):\n")
    for symbol, equation in world.equations.items():
        f.write(f"  {symbol}: {equation}\n")
    f.write("\nExact replay audit:\n")
    f.write(
        "  world.params is build_symbolic_graph-ready, including any full-horizon shock schedule.\n"
    )
    f.write(
        "  world.equation_parameters is the sparse executed-parameter audit; "
        "world.exogenous holds defensive copies of raw full-horizon innovations.\n\n"
    )
    f.write("Decomposition identity (baseline_intrinsic + outcome_noise + confounder\n")
    f.write("  + covariate + direct contributions + indirect_by_source == outcome,\n")
    f.write("  exactly the sum SCM.reconstruction() forms):\n")
    f.write(f"  max |error| = {ident_err:.2e}\n\n")
    f.write("Signal metrics (per direct treatment):\n")
    for i in range(len(signal["treatment_cv"])):
        f.write(
            f"  C{int(signal['treatment'][i]) + 1}: "
            f"treatment_cv={_format_signal_metric(signal, 'treatment_cv', i)} "
            f"treatment_hf={_format_signal_metric(signal, 'treatment_hf', i)} "
            f"contrib_cv={_format_signal_metric(signal, 'contrib_cv', i)} "
            f"contrib_hf={_format_signal_metric(signal, 'contrib_hf', i)} "
            f"spearman={_format_signal_metric(signal, 'spearman', i)} "
            f"rel_std={_format_signal_metric(signal, 'contrib_rel_std', i)}\n"
        )
    return f.getvalue()


def world_to_dot(world: SCM) -> str:
    """Graphviz DOT source for the world's DAG (text only — the ``graphviz``
    library/binary is never invoked; renderable with any external tool)."""
    g, params = world.g, world.params
    n_treatments = len(g["g_cy"])
    n_covariates = len(g["g_zy"])
    n_latent = len(g["g_dy"])
    f = io.StringIO()
    f.write("digraph CDAG {\n  rankdir=LR;\n  node [shape=ellipse];\n")
    for k in range(n_treatments):
        f.write(f'  C{k + 1} [label="C{k + 1}\\n({mechanism_label(params, k)})"];\n')
    for m in range(n_covariates):
        f.write(f'  Z{m + 1} [label="Z{m + 1}\\n(covariate)"];\n')
    for j in range(n_latent):
        f.write(f'  D{j + 1} [label="D{j + 1}\\n(latent_unobserved)"];\n')
    f.write('  B [label="B\\n(baseline)"];\n  Y [label="Y\\n(outcome)"];\n')
    for et, src, dst, coef in edges_with_coeffs(g, params):
        f.write(f'  {src} -> {dst} [label="{coef:+.2f}", comment="{et}"];\n')
    f.write("  B -> Y;\n}\n")
    return f.getvalue()
