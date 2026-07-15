"""Plain-text world descriptions — "get a description of the world".

:func:`describe_scm` renders everything a human needs to audit one world:
the DAG edges with their drawn coefficients, node connectivity, per-channel
mechanism and texture parameters, the prior ranges the draw came from, the
exact decomposition-identity check, and per-channel signal metrics.

The format is byte-compatible with structural-pfn's inspection
``description.txt`` files.
"""

from __future__ import annotations

import io
import json

from .mechanisms import SATURATION_PRIOR_RANGES
from .worlds import (
    ADSTOCK_NAMES,
    SATURATION_NAMES,
    SCM,
    channel_role,
    edges_with_coeffs,
    mechanism_label,
    node_status,
)


def _channel_lines(g: dict, params: dict) -> list[str]:
    K = len(g["g_cy"])
    lines = []
    for k in range(K):
        sat = SATURATION_NAMES[int(params["sat_family"][k])]
        ad = ADSTOCK_NAMES[int(params["adstock_family"][k])]
        bits = [
            f"C{k + 1}: role={channel_role(g, k)}",
            f"beta={params['beta'][k]:.2f}",
            f"adstock={ad}"
            + (f"(alpha={params['adstock_alpha'][k]:.2f})" if ad == "geometric" else "")
            + (
                f"(lam={params['weibull_lam'][k]:.1f},k={params['weibull_k'][k]:.1f})"
                if ad == "weibull"
                else ""
            ),
            f"saturation={sat}",
            f"walk(mean={params['rw_c']['mean'][k]:.2f},std={params['rw_c']['std'][k]:.2f},"
            f"smooth={params['rw_c']['smoothness'][k]:.2f})",
            f"hf_sigma={params['hf_sigma'][k]:.2f}",
            f"pulse(p={params['pulse_prob'][k]:.2f},amp={params['pulse_amp'][k]:.2f})",
        ]
        lines.append("  " + "  ".join(bits))
    return lines


def describe_scm(world: SCM) -> str:
    """Render the full plain-text description of one :class:`SCM`.

    Sections: title + purpose, sizes, edge budget, edge census, active edges
    with drawn coefficients, node connectivity, per-channel mechanism +
    texture, the texture prior ranges, the decomposition-identity error, and
    per-direct-channel signal metrics.
    """
    g, params, cfg = world.g, world.params, world.cfg
    edges = edges_with_coeffs(g, params)
    ident_err = world.identity_error()
    signal = world.signal()

    f = io.StringIO()
    f.write(f"Dataset: {world.name} (additive SCM, texture=diverse)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Purpose:\n  {world.purpose}\n\n")
    f.write(f"Sizes: T={cfg.T}, K={world.K}, M={world.M}, J={world.J}, ")
    f.write(f"l_max={cfg.l_max}, adstock_burn_in={cfg.adstock_burn_in}\n")
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
    f.write("\n".join(_channel_lines(g, params)) + "\n\n")
    f.write("Texture prior (diverse):\n")
    f.write(f"  rw_channel_std_range={cfg.rw_channel_std_range} (relative)\n")
    f.write(f"  channel_hf_sigma_range={cfg.channel_hf_sigma_range} (relative)\n")
    f.write(f"  channel_pulse_prob_range={cfg.channel_pulse_prob_range}\n")
    f.write(f"  channel_pulse_amp_range={cfg.channel_pulse_amp_range} (relative)\n")
    f.write(f"  rw_positive_mean_range={cfg.rw_positive_mean_range}\n")
    f.write(f"  beta_additive_range={cfg.beta_additive_range}\n")
    f.write("  saturation prior ranges: ")
    f.write(json.dumps(SATURATION_PRIOR_RANGES) + "\n\n")
    f.write("Decomposition identity (baseline_intrinsic + confounder + control\n")
    f.write("  + direct contributions + indirect_by_source == sales):\n")
    f.write(f"  max |error| = {ident_err:.2e}\n\n")
    f.write("Signal metrics (per direct channel):\n")
    for i in range(len(signal["spend_cv"])):
        f.write(
            f"  C{int(signal['channel'][i]) + 1}: spend_cv={signal['spend_cv'][i]:.2f} "
            f"spend_hf={signal['spend_hf'][i]:.2f} contrib_cv={signal['contrib_cv'][i]:.2f} "
            f"contrib_hf={signal['contrib_hf'][i]:.2f} spearman={signal['spearman'][i]:.2f} "
            f"rel_std={signal['contrib_rel_std'][i]:.2f}\n"
        )
    return f.getvalue()


def world_to_dot(world: SCM) -> str:
    """Graphviz DOT source for the world's DAG (text only — the ``graphviz``
    library/binary is never invoked; renderable with any external tool)."""
    g, params = world.g, world.params
    K = len(g["g_cy"])
    M = len(g["g_zb"])
    J = len(g["g_db"])
    f = io.StringIO()
    f.write("digraph CDAG {\n  rankdir=LR;\n  node [shape=ellipse];\n")
    for k in range(K):
        f.write(f'  C{k + 1} [label="C{k + 1}\\n({mechanism_label(params, k)})"];\n')
    for m in range(M):
        f.write(f'  Z{m + 1} [label="Z{m + 1}\\n(control)"];\n')
    for j in range(J):
        f.write(f'  D{j + 1} [label="D{j + 1}\\n(demand)"];\n')
    f.write('  B [label="B\\n(baseline)"];\n  Y [label="Y\\n(sales)"];\n')
    for et, src, dst, coef in edges_with_coeffs(g, params):
        f.write(f'  {src} -> {dst} [label="{coef:+.2f}", comment="{et}"];\n')
    f.write("  B -> Y;\n}\n")
    return f.getvalue()
