"""Matplotlib renders for world bundles (DAG, timeseries, decomposition, channels).

matplotlib is imported lazily inside each function so importing the package —
or generating corpora — never touches it. Headless callers (the CLI does this)
should select the Agg backend before calling in, e.g.
``matplotlib.use("Agg")``.
"""

from __future__ import annotations

import numpy as np

from .worlds import SCM, channel_role, edges_with_coeffs, mechanism_label, node_status

# Validated categorical palette — slots are assigned in FIXED order per entity
# and reused consistently across every figure of a bundle.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, MUTED, GRID = "#1f1f1f", "#666666", "#d9d9d9"

#: Edge-type colors, shared between the DAG figure and the decomposition
#: figure (an indirect-effect source IS its edge type).
EDGE_STYLE: dict[str, tuple[str, str]] = {
    "cy": ("#2a78d6", "C→Y direct"),
    "dc": ("#e34948", "D→C (confounding)"),
    "zc": ("#eda100", "Z→C"),
    "cc": ("#4a3aa7", "C→C (halo)"),
    "db": ("#eb6834", "D→B"),
    "zb": ("#1baf7a", "Z→B"),
    "dz": ("#e87ba4", "D→Z"),
    "zz": ("#008300", "Z→Z"),
}


def _style_ax(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_dag(world: SCM, path: str, title: str | None = None) -> None:
    """Column-layout DAG render (pure matplotlib — no graphviz required)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch

    g, params = world.g, world.params
    title = world.name if title is None else title
    K, M, J = len(g["g_cy"]), len(g["g_zb"]), len(g["g_db"])

    def _col(n: int, x: float) -> dict[int, tuple[float, float]]:
        ys = np.linspace(0.9, 0.1, n) if n > 1 else [0.5]
        return {i: (x, float(y)) for i, y in enumerate(ys)}

    pos: dict[str, tuple[float, float]] = {}
    pos.update({f"D{j + 1}": p for j, p in _col(J, 0.05).items()})
    pos.update({f"Z{m + 1}": p for m, p in _col(M, 0.30).items()})
    pos.update({f"C{k + 1}": p for k, p in _col(K, 0.58).items()})
    pos["B"] = (0.93, 0.80)
    pos["Y"] = (0.93, 0.28)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left")

    edges = edges_with_coeffs(g, params) + [("_by", "B", "Y", 0.0)]
    present: dict[str, str] = {}
    for et, src, dst, _ in edges:
        if et == "_by":
            color, ls, rad = "#999999", "--", 0.0
        else:
            color, ls = EDGE_STYLE[et][0], "-"
            rad = 0.35 if src[0] == dst[0] else 0.12  # within-column arcs curve more
            present[et] = EDGE_STYLE[et][1]
        arrow = FancyArrowPatch(
            pos[src],
            pos[dst],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            linestyle=ls,
            color=color,
            shrinkA=31,
            shrinkB=31,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(arrow)

    status = node_status(g)
    for name, (x, y) in pos.items():
        reaches = status.get(name, "connected") == "connected"  # B / Y always reach
        ax.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor="none", zorder=2.5))
        ax.add_patch(
            Circle(
                (x, y),
                0.036,
                facecolor="#f2f2f0" if reaches else "#ffffff",
                edgecolor=MUTED,
                linestyle="-" if reaches else (0, (3, 2)),
                zorder=3,
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=9, color=INK, zorder=4)
        sub = ""
        if name.startswith("C"):
            k = int(name[1:]) - 1
            role = channel_role(g, k)
            sub = mechanism_label(params, k) + ("" if role in ("direct", "null") else f" · {role}")
        if not reaches:
            sub = (sub + " · " if sub else "") + "isolated null"
        if sub:
            ax.text(x, y - 0.055, sub, ha="center", va="top", fontsize=6.5, color=MUTED, zorder=4)

    handles = [
        Line2D([0], [0], color=EDGE_STYLE[et][0], lw=1.8, label=lab) for et, lab in present.items()
    ]
    handles.append(Line2D([0], [0], color="#999999", lw=1.5, ls="--", label="B→Y (baseline)"))
    # legend BELOW the axes so it can never collide with lower-column nodes
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=7.5,
        frameon=False,
        ncol=min(len(handles), 5),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(world: SCM, path: str, title: str | None = None) -> None:
    """Model-input series: per-channel spend, controls + latent demand, sales."""
    import matplotlib.pyplot as plt

    d = world.data
    title = world.name if title is None else title
    T, K = d["channels"].shape
    M = d["controls"].shape[1]
    J = d["demand"].shape[1]
    weeks = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    for k in range(K):
        ax.plot(weeks, d["channels"][:, k], color=PALETTE[k], lw=1.3, label=f"C{k + 1}")
    ax.set_title(f"{title} — model inputs", fontsize=11, color=INK, loc="left")
    ax.set_ylabel("spend", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(K, 8), loc="upper left")

    ax = axes[1]
    for m in range(M):
        ax.plot(weeks, d["controls"][:, m], color=PALETTE[m], lw=1.3, label=f"Z{m + 1}")
    for j in range(J):
        ax.plot(
            weeks,
            d["demand"][:, j],
            color=MUTED,
            lw=1.2,
            ls="--",
            label=f"D{j + 1} (latent)",
        )
    ax.set_ylabel("controls / demand", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(M + J, 8), loc="upper left")

    ax = axes[2]
    ax.plot(weeks, d["sales"], color=INK, lw=1.5)
    ax.set_ylabel("sales Y", fontsize=9, color=MUTED)
    ax.set_xlabel("week", fontsize=9, color=MUTED)

    for ax in axes:
        _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_decomposition(world: SCM, path: str, title: str | None = None) -> None:
    """Every true effect on Y + the reconstruction identity check."""
    import matplotlib.pyplot as plt

    d, g = world.data, world.g
    title = world.name if title is None else title
    ident_err = world.identity_error()
    T, K = d["contributions"].shape
    M = d["control_contribution"].shape[1]
    J = d["confounder_contribution"].shape[1]
    weeks = np.arange(T)
    recon = world.reconstruction()

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    ax.plot(weeks, d["sales"], color=INK, lw=1.6, label="sales Y")
    ax.plot(weeks, recon, color="#999999", lw=1.2, ls="--", label="Σ true components")
    ax.set_title(
        f"{title} — true decomposition (identity max err {ident_err:.1e})",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    for k in range(K):
        if g["g_cy"][k]:  # only channels with a direct edge have a nonzero target
            ax.plot(weeks, d["contributions"][:, k], color=PALETTE[k], lw=1.3, label=f"C{k + 1}")
    ax.set_ylabel("direct contributions", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(K, 8), loc="upper left")

    ax = axes[2]
    ax.plot(weeks, d["baseline_intrinsic"], color=MUTED, lw=1.4, label="baseline intrinsic")
    for j in range(J):
        if g["g_db"][j]:  # absent edges are identically zero — skip the clutter
            ax.plot(
                weeks,
                d["confounder_contribution"][:, j],
                color=EDGE_STYLE["db"][0],
                lw=1.2,
                ls=["-", "--"][j % 2],
                label=f"D{j + 1}→B",
            )
    for m in range(M):
        if g["g_zb"][m]:
            ax.plot(
                weeks,
                d["control_contribution"][:, m],
                color=EDGE_STYLE["zb"][0],
                lw=1.2,
                ls=["-", "--", ":", "-."][m % 4],
                label=f"Z{m + 1}→B",
            )
    ax.set_ylabel("baseline components", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")

    ax = axes[3]
    for i, src in enumerate(("cc", "zc", "dc")):
        ax.plot(
            weeks,
            d["indirect_effects_by_source"][:, i],
            color=EDGE_STYLE[src][0],
            lw=1.3,
            label=f"indirect via {src}",
        )
    ax.axhline(0.0, color=GRID, lw=0.8)
    ax.set_ylabel("indirect effects", fontsize=9, color=MUTED)
    ax.set_xlabel("week", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")

    for ax in axes:
        _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_channels(world: SCM, path: str, title: str | None = None) -> None:
    """Per-channel spend vs true contribution, both indexed to mean 1."""
    import matplotlib.pyplot as plt

    d, g, params = world.data, world.g, world.params
    title = world.name if title is None else title
    T, K = d["channels"].shape
    weeks = np.arange(T)
    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(K):
        ax = axes[k]
        spend = d["channels"][:, k]
        ax.plot(weeks, spend / spend.mean(), color=MUTED, lw=1.0, label="spend (indexed)")
        if g["g_cy"][k]:
            contrib = d["contributions"][:, k]
            ax.plot(
                weeks,
                contrib / max(abs(contrib.mean()), 1e-9),
                color=PALETTE[k],
                lw=1.5,
                label="true contribution (indexed)",
            )
            tag = f"β={params['beta'][k]:.2f}"
        else:
            tag = f"no direct edge ({channel_role(g, k)})"
        ax.set_title(f"C{k + 1} — {mechanism_label(params, k)} — {tag}", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style_ax(ax)
    for ax in axes[K:]:
        ax.axis("off")
    fig.suptitle(
        f"{title} — per-channel spend vs true contribution (mean=1)", fontsize=11, color=INK
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
