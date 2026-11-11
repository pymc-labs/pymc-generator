"""Plain-text world descriptions — "get a description of the world".

:func:`describe_scm` renders everything a human needs to audit one world:
the vector-valued structural equations, DAG edges with drawn coefficients,
node connectivity, realized mechanism and texture parameters, prior ranges,
exact decomposition-identity check, and per-treatment signal metrics.
"""

from __future__ import annotations

import io
import json
from typing import NamedTuple

import numpy as np

from .worlds import (
    CARRYOVER_NAMES,
    SATURATION_NAMES,
    SCM,
    edges_with_coeffs,
    mechanism_label,
    node_status,
    parameter_text,
    treatment_role,
)

#: Rendered shape parameters of each nonlinear family: ``(label, params key)``.
_SATURATION_SHAPES: dict[str, tuple[tuple[str, str], ...]] = {
    "hill": (("slope", "hill_slope"), ("kappa_mult", "hill_kappa_mult")),
    "logistic": (("lam", "logistic_lam"),),
    "michaelis_menten": (("kappa_mult", "mm_kappa_mult"),),
    "tanh": (("c", "tanh_c"),),
    "root": (("alpha", "root_alpha"),),
}

#: How each node family reads in a graph label, ``name ~ type``, after the
#: ``name ~ Distribution`` labels of PyMC's ``model_to_graphviz``.
NODE_TYPES = {"D": "Latent", "Z": "Control", "C": "Channel", "B": "Baseline", "Y": "Outcome"}
#: Graph fills, as Graphviz X11 greys: what a modeller never observes (latent
#: factors, the baseline) stays white; observed nodes darken from controls to
#: channels to the outcome.
NODE_FILLS = {"D": "white", "B": "white", "Z": "gray90", "C": "gray80", "Y": "gray70"}


class DagNode(NamedTuple):
    """One node as the DAG figure and the DOT export draw it."""

    name: str
    kind: str
    """The family type shown after ``~`` (see :data:`NODE_TYPES`)."""
    notes: tuple[str, ...]
    """Small print under the type, one entry per line: a channel's mechanism,
    its role when that is not a direct effect, and ``isolated null`` for nodes
    without a path to the outcome."""
    fill: str
    """Graphviz X11 grey (see :data:`NODE_FILLS`)."""
    isolated: bool


def dag_nodes(world: SCM) -> list[DagNode]:
    """Every node of the world's DAG in family order: D, Z, C, then B and Y."""
    g, params = world.g, world.params
    status = node_status(g)
    names = (
        [f"D{j + 1}" for j in range(len(g["g_dy"]))]
        + [f"Z{m + 1}" for m in range(len(g["g_zy"]))]
        + [f"C{k + 1}" for k in range(len(g["g_cy"]))]
        + ["B", "Y"]
    )
    nodes = []
    for name in names:
        family = name[0]
        isolated = status.get(name, "connected") != "connected"  # B and Y always reach
        notes: list[str] = []
        if family == "C":
            k = int(name[1:]) - 1
            notes.append(mechanism_label(params, k))
            role = treatment_role(g, k)
            if role not in ("direct", "null"):
                notes.append(role)
        if isolated:
            notes.append("isolated null")
        nodes.append(DagNode(name, NODE_TYPES[family], tuple(notes), NODE_FILLS[family], isolated))
    return nodes


def _saturation_description(params: dict, k: int, family: str) -> str:
    """Render only the shape parameters used by a saturation family."""
    if family not in _SATURATION_SHAPES:
        return family
    fields = ",".join(
        f"{label}={parameter_text(params, params[key][k])}"
        for label, key in _SATURATION_SHAPES[family]
    )
    return f"{family}({fields})"


def _treatment_lines(g: dict, params: dict) -> list[str]:
    n_treatments = len(g["g_cy"])
    lines = []
    for k in range(n_treatments):
        sat = SATURATION_NAMES[int(params["sat_family"][k])]
        ad = CARRYOVER_NAMES[int(params["carryover_family"][k])]
        bits = [
            f"C{k + 1}: role={treatment_role(g, k)}",
            f"beta={parameter_text(params, params['beta'][k])}",
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
        if "treatment_reference_contribution" in params:
            bits.append(
                f"reference(contribution={float(params['treatment_reference_contribution'][k])!r},"
                f"input={float(params['treatment_reference_input'][k])!r},"
                "post-carryover,pre-gate)"
            )
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
        if "covariate_reference_contribution" in params:
            bits.append(
                f"reference(contribution={float(params['covariate_reference_contribution'][m])!r},"
                f"input={float(params['covariate_reference_input'][m])!r},"
                f"rho_zy={float(params['rho_zy'][m])!r},pre-gate,pre-floor)"
            )
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
    texture, prior ranges, vector-valued structural equations, primitive-bound
    replay provenance, decomposition-identity error, and per-direct-treatment signal metrics.
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
        f.write(f"  [{et}] {src} -> {dst}   coef={parameter_text(params, coef, '+.3f')}\n")
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
        # Validated opt-in widths can sit seven digits below their support, so
        # opt-in intervals print exactly, like the reference records.
        exact = params.get("mechanism_priors_enabled", False)

        def _interval(value: float) -> str:
            return repr(float(value)) if exact else f"{value:.3f}"

        for q, (lo, width) in prior_cond.items():
            s_lo, s_hi = spec[q]["support"]
            f.write(
                f"  {q}: U({_interval(lo)}, {_interval(lo + width)})  "
                f"width={_interval(width)}  support=({s_lo}, {s_hi})\n"
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
    f.write(
        json.dumps(
            {
                family: {name: [float(x) for x in bounds] for name, bounds in parameters.items()}
                for family, parameters in cfg.saturation_prior_ranges.items()
            }
        )
        + "\n"
    )
    f.write(f"  mm_scale_prior={cfg.mm_scale_prior}\n")
    if cfg.treatment_reference_contribution_range is not None:
        f.write(
            f"  treatment_reference_contribution_range={cfg.treatment_reference_contribution_range}"
            f"  treatment_reference_multiplier={cfg.treatment_reference_multiplier}\n"
        )
    if cfg.covariate_reference_contribution_range is not None:
        f.write(
            f"  covariate_reference_contribution_range={cfg.covariate_reference_contribution_range}"
            f"  covariate_reference_scale={cfg.covariate_reference_scale}\n"
        )
    f.write("\n")
    f.write("Structural equations (vector-valued; active parents only):\n")
    for symbol, equation in world.equations.items():
        f.write(f"  {symbol}: {equation}\n")
    f.write("\nExact replay audit:\n")
    f.write(
        "  world.replay() recomputes every forward array from world.primitive_parameters.\n"
        "  Exact replay requires accepted provenance from sample_scm in the same numerical environment.\n"
        "  world.params holds inspection/intervention inputs, not exact replay provenance.\n"
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
    """Graphviz DOT source for the world's DAG, styled as ``viz.plot_dag`` draws it.

    Text only: the ``graphviz`` library/binary is never invoked; render it with
    any external tool. Nodes read ``name ~ type`` on grey fills (see
    :func:`dag_nodes`); edge labels carry the drawn coefficients, and the
    baseline's edge to the outcome is dashed.
    """
    g, params = world.g, world.params
    f = io.StringIO()
    f.write('digraph CDAG {\n  node [shape=ellipse, fontname="Times-Roman"];\n')
    for node in dag_nodes(world):
        label = f"{node.name}<BR/>~<BR/>{node.kind}"
        for note in node.notes:
            label += f'<BR/><FONT POINT-SIZE="10" COLOR="gray35">{note}</FONT>'
        style = "filled,dashed" if node.isolated else "filled"
        f.write(f'  {node.name} [label=<{label}>, style="{style}", fillcolor={node.fill}];\n')
    for et, src, dst, coef in edges_with_coeffs(g, params):
        label = parameter_text(params, coef, "+.2f")
        f.write(f'  {src} -> {dst} [label="{label}", comment="{et}"];\n')
    f.write("  B -> Y [style=dashed];\n}\n")
    return f.getvalue()
