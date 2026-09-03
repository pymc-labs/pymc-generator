"""Matplotlib renders for world bundles and for generated-data diagnostics.

Two families live here. The bundle figures (DAG, timeseries, decomposition,
channels) render ONE world; the diagnostics figures render a
:class:`~prior_generator.diagnostics.DataDiagnostics` report over many worlds
(dependence, series distributions, temporal structure, VIF, contributions).

matplotlib is imported lazily inside each function so importing the package —
or generating corpora — never touches it. Headless callers (the CLI does this)
should select the Agg backend before calling in, e.g.
``matplotlib.use("Agg")``.

Every figure is descriptive: no p-values, no significance, no confidence
bands, no causal or forecast claims. The retained latent series (D, B) are
ground truth kept for auditing and are labelled as such, never model inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .diagnostics import (
    DEPENDENCE_METRICS,
    SERIES_SLOTS,
    VIEWS,
    VIF_SCOPES,
    DataDiagnostics,
)
from .outcomes import OutcomeDistributions, QuantityDistribution, StatName, _q_key
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
    n_treatments, n_covariates, n_latent = len(g["g_cy"]), len(g["g_zb"]), len(g["g_db"])

    def _col(n: int, x: float) -> dict[int, tuple[float, float]]:
        ys = np.linspace(0.9, 0.1, n) if n > 1 else [0.5]
        return {i: (x, float(y)) for i, y in enumerate(ys)}

    pos: dict[str, tuple[float, float]] = {}
    pos.update({f"D{j + 1}": p for j, p in _col(n_latent, 0.05).items()})
    pos.update({f"Z{m + 1}": p for m, p in _col(n_covariates, 0.30).items()})
    pos.update({f"C{k + 1}": p for k, p in _col(n_treatments, 0.58).items()})
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
    n_time_steps, n_treatments = d["channels"].shape
    n_covariates = d["controls"].shape[1]
    n_latent = d["demand"].shape[1]
    weeks = np.arange(n_time_steps)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    for k in range(n_treatments):
        ax.plot(weeks, d["channels"][:, k], color=PALETTE[k], lw=1.3, label=f"C{k + 1}")
    ax.set_title(f"{title} — model inputs", fontsize=11, color=INK, loc="left")
    ax.set_ylabel("spend", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(n_treatments, 8), loc="upper left")

    ax = axes[1]
    for m in range(n_covariates):
        ax.plot(weeks, d["controls"][:, m], color=PALETTE[m], lw=1.3, label=f"Z{m + 1}")
    for j in range(n_latent):
        ax.plot(
            weeks,
            d["demand"][:, j],
            color=MUTED,
            lw=1.2,
            ls="--",
            label=f"D{j + 1} (latent)",
        )
    ax.set_ylabel("controls / demand", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(n_covariates + n_latent, 8), loc="upper left")

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
    n_time_steps, n_treatments = d["contributions"].shape
    n_covariates = d["control_contribution"].shape[1]
    n_latent = d["confounder_contribution"].shape[1]
    weeks = np.arange(n_time_steps)
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
    for k in range(n_treatments):
        if g["g_cy"][k]:  # only channels with a direct edge have a nonzero target
            ax.plot(weeks, d["contributions"][:, k], color=PALETTE[k], lw=1.3, label=f"C{k + 1}")
    ax.set_ylabel("direct contributions", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(n_treatments, 8), loc="upper left")

    ax = axes[2]
    ax.plot(weeks, d["baseline_intrinsic"], color=MUTED, lw=1.4, label="baseline intrinsic")
    for j in range(n_latent):
        if g["g_db"][j]:  # absent edges are identically zero — skip the clutter
            ax.plot(
                weeks,
                d["confounder_contribution"][:, j],
                color=EDGE_STYLE["db"][0],
                lw=1.2,
                ls=["-", "--"][j % 2],
                label=f"D{j + 1}→B",
            )
    for m in range(n_covariates):
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
    n_time_steps, n_treatments = d["channels"].shape
    weeks = np.arange(n_time_steps)
    ncols = min(3, n_treatments)
    nrows = int(np.ceil(n_treatments / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n_treatments):
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
    for ax in axes[n_treatments:]:
        ax.axis("off")
    fig.suptitle(
        f"{title} — per-channel spend vs true contribution (mean=1)", fontsize=11, color=INK
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_outcome_distributions(
    dist: OutcomeDistributions,
    path: str,
    *,
    of: StatName = "value",
    bins: int = 60,
    title: str | None = None,
) -> None:
    """Histogram grid over the QUANTITY axis: how large every outcome gets.

    One panel per quantity of an :func:`~prior_generator.outcomes.outcome_distributions`
    result. ``of="value"`` pools every drawn value across worlds and time;
    ``of="mean"`` shows the across-world spread of per-world levels;
    ``of="share"`` shows the share-of-sales budget (media/baseline/noise).

    Quantities with no spread under the chosen statistic are skipped — e.g.
    ``sales`` is identically 1.0 in the share budget and would waste a panel.
    """
    import matplotlib.pyplot as plt

    panels: list[tuple[QuantityDistribution, np.ndarray]] = []
    for d in dist.quantities.values():
        if of == "share" and not d.on_y_scale:
            continue
        x = np.asarray(d.stat(of), dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size and x.min() < x.max():
            panels.append((d, x))
    if not panels:
        raise ValueError(f"no quantity has any spread for of={of!r}")
    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (d, x) in enumerate(panels):
        ax = axes[i]
        median = float(np.median(x))
        ax.hist(x, bins=bins, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="none")
        ax.axvline(median, color=INK, lw=1.0, ls="--")
        ax.set_title(
            f"{d.name} — n={x.size:,} — median {median:,.3g}",
            fontsize=9,
            color=INK,
            loc="left",
        )
        ax.set_ylabel("count", fontsize=8, color=MUTED)
        _style_ax(ax)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    scale = "" if dist.normalize == "none" else f" (normalize={dist.normalize})"
    head = title or (
        f"outcome distributions — {of} — {dist.n_worlds} worlds × {dist.n_time_steps} steps{scale}"
    )
    fig.suptitle(head, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generated-data diagnostics figures
# ---------------------------------------------------------------------------

#: Most series one dependence / temporal / VIF figure will draw. Past this a
#: heatmap cell is thinner than a printed rule and the labels collide.
MAX_HEATMAP_KEYS = 64

#: Most panels one grid figure will draw (also the row cap of the
#: contribution figure). The grid is at most 4 columns x 6 rows.
MAX_SERIES_PANELS = 24

# Figure caps in inches, the panel grid shape, and the tick-thinning
# threshold. Above _TICK_STRIDE_ABOVE keys, every k-th tick is labelled with
# a deterministic stride so the same report always renders the same labels.
_MAX_HEATMAP_INCHES = 24.0
_MAX_PANEL_INCHES = 18.0
_MAX_PANEL_COLS = 4
_MAX_PANEL_ROWS = MAX_SERIES_PANELS // _MAX_PANEL_COLS
_TICK_STRIDE_ABOVE = 32

#: Drawn on any figure the caller restricted to the levels view alone.
LEVELS_ONLY_NOTE = (
    "levels only: weekly series are random-walk-like, so level dependence between "
    "series is large even when the series are independent — the differences view is "
    "the companion reading, not a stationarity proof"
)

#: Drawn on any figure the caller restricted to the differences view alone.
DIFFERENCES_ONLY_NOTE = "view: differences — first differences of every series"

_MASK_NOTE = "grey = masked; N/A = no valid worlds, never drawn as a zero"

_DEPENDENCE_MASK_NOTE = (
    "grey = masked (the diagonal is never reported); N/A = no valid worlds, never drawn as a zero"
)


def _plot_names(values: object, allowed: Sequence[str], what: str) -> tuple[str, ...]:
    """Exact-name selector: non-empty, unique, known, in caller order."""
    if isinstance(values, str):
        raise TypeError(
            f"{what} must be a non-string sequence of names, not the bare string {values!r}"
        )
    if not isinstance(values, Sequence):
        raise TypeError(f"{what} must be a non-string sequence, got {type(values).__name__}")
    names = tuple(str(v) for v in values)
    if not names:
        raise ValueError(f"{what} is empty; choose from {list(allowed)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} repeats an entry: {list(names)}")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected {list(allowed)}")
    return names


def _plot_views(report: DataDiagnostics, views: object) -> tuple[str, ...]:
    """Validated view names that the report actually carries."""
    names = _plot_names(views, VIEWS, "views")
    missing = [name for name in names if name not in report.views]
    if missing:
        raise ValueError(
            f"view(s) {missing} were not computed in this report; available: "
            f"{sorted(report.views)} — rebuild with data_diagnostics(..., views={list(names)})"
        )
    return names


def _plot_keys(
    available: Sequence[str], keys: object, limit: int, *, what: str = "keys"
) -> tuple[str, ...]:
    """Validated key selection, bounded so a figure stays readable."""
    chosen = tuple(available) if keys is None else _plot_names(keys, available, what)
    if not chosen:
        raise ValueError(f"there are no {what} to plot")
    if len(chosen) > limit:
        raise ValueError(
            f"{len(chosen)} {what} exceed this figure's bound of {limit}; narrow "
            f"{what}=... to at most {limit} entries"
        )
    return chosen


def _plot_quantile(quantile: object) -> float:
    level = float(quantile)  # type: ignore[arg-type]
    if not np.isfinite(level) or level < 0.0 or level > 1.0:
        raise ValueError(f"quantile must be finite and within [0, 1], got {quantile!r}")
    return level


def _tick_stride(n: int) -> np.ndarray:
    """Tick positions, thinned deterministically above the label budget."""
    stride = 1 if n <= _TICK_STRIDE_ABOVE else int(np.ceil(n / _TICK_STRIDE_ABOVE))
    return np.arange(0, n, stride)


def _heatmap_figsize(n_cols: int, n_rows: int, n_x: int, n_y: int) -> tuple[float, float]:
    width = min(_MAX_HEATMAP_INCHES, 1.6 + n_cols * (2.4 + 0.14 * n_x))
    height = min(_MAX_HEATMAP_INCHES, 1.4 + n_rows * (2.0 + 0.14 * n_y))
    return width, height


# Per-panel inches, derived so even a full 4 x 6 grid stays inside the cap.
_PANEL_WIDTH_INCHES = min(4.2, _MAX_PANEL_INCHES / _MAX_PANEL_COLS)
_PANEL_HEIGHT_INCHES = min(2.9, _MAX_PANEL_INCHES / _MAX_PANEL_ROWS)


def _panel_grid(n_panels: int) -> tuple[int, int, tuple[float, float]]:
    """Grid shape and figure size for at most ``MAX_SERIES_PANELS`` panels."""
    ncols = min(_MAX_PANEL_COLS, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    return ncols, nrows, (ncols * _PANEL_WIDTH_INCHES, nrows * _PANEL_HEIGHT_INCHES)


def _empty_panel(ax, note: str) -> None:
    """A truthful empty artist: say what is missing instead of drawing a zero."""
    ax.text(
        0.5,
        0.5,
        note,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _value_label(ax, x: float, y: float, text: str, *, color: str) -> None:
    """Label a horizontal bar just past its end, in offset points."""
    ax.annotate(
        text,
        (x, y),
        xytext=(4.0 if x >= 0.0 else -4.0, 0.0),
        textcoords="offset points",
        va="center",
        ha="left" if x >= 0.0 else "right",
        fontsize=7.5,
        color=color,
    )


def _view_banner(fig, views: tuple[str, ...]) -> bool:
    """Flag a single-view figure; returns True when a banner was drawn."""
    if views == ("levels",):
        fig.text(
            0.5,
            0.94,
            LEVELS_ONLY_NOTE,
            ha="center",
            va="top",
            fontsize=9,
            color="#7a1616",
            wrap=True,
            bbox={"facecolor": "#fdf0ef", "edgecolor": "#e34948", "boxstyle": "round,pad=0.4"},
        )
        return True
    if views == ("differences",):
        fig.text(
            0.5,
            0.94,
            DIFFERENCES_ONLY_NOTE,
            ha="center",
            va="top",
            fontsize=9,
            color=INK,
            bbox={"facecolor": "#f2f2f0", "edgecolor": GRID, "boxstyle": "round,pad=0.4"},
        )
        return True
    return False


def _heatmap(
    ax,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    diverging: bool,
    mask_diagonal: bool = False,
):
    """Masked heatmap of an across-world quantile matrix.

    ``diverging`` centres a red/blue map at 0 with symmetric limits (for the
    signed correlations). Otherwise the limits span the data and always
    include 0, so a negative xi stays visible instead of being clamped into
    ``[0, 1]``. Non-finite cells are masked grey and, off the diagonal,
    annotated ``N/A``.
    """
    import matplotlib as mpl

    values = np.asarray(matrix, dtype=np.float64)
    invalid = np.ma.getmaskarray(np.ma.masked_invalid(values))
    diagonal = (
        np.eye(values.shape[0], values.shape[1], dtype=bool)
        if mask_diagonal
        else np.zeros(values.shape, dtype=bool)
    )
    data = np.ma.masked_array(values, mask=invalid | diagonal)
    finite = np.asarray(data.compressed(), dtype=np.float64)
    if diverging:
        limit = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-9)
        vmin, vmax, name = -limit, limit, "RdBu_r"
    else:
        vmin = min(float(finite.min()), 0.0) if finite.size else 0.0
        vmax = max(float(finite.max()), 0.0) if finite.size else 1.0
        if vmax - vmin < 1e-12:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        name = "viridis"
    im = ax.imshow(
        data,
        cmap=mpl.colormaps[name].with_extremes(bad=GRID),
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    n_rows, n_cols = values.shape
    fontsize = max(4.0, 8.5 - 0.09 * float(max(n_rows, n_cols)))
    for i, j in zip(*np.nonzero(invalid & ~diagonal)):
        ax.text(int(j), int(i), "N/A", ha="center", va="center", fontsize=fontsize, color=MUTED)
    xticks = _tick_stride(n_cols)
    ax.set_xticks(xticks)
    ax.set_xticklabels([col_labels[i] for i in xticks], rotation=90, fontsize=fontsize)
    yticks = _tick_stride(n_rows)
    ax.set_yticks(yticks)
    ax.set_yticklabels([row_labels[i] for i in yticks], fontsize=fontsize)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return im


def _colorbar(fig, im, ax) -> None:
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.outline.set_visible(False)


def _finish(fig, path: str, *, banner: bool, footnote: str | None) -> None:
    """Reserve room for the figure-level texts, save, and close."""
    import matplotlib.pyplot as plt

    if footnote is not None:
        fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(
        rect=(0.0, 0.045 if footnote is not None else 0.0, 1.0, 0.9 if banner else 0.95)
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dependence_matrices(
    report: DataDiagnostics,
    path: str,
    *,
    views: Sequence[str] = ("levels", "differences"),
    keys: Sequence[str] | None = None,
    metrics: Sequence[str] = ("pearson", "spearman", "xi_max", "xi"),
    quantile: float = 0.5,
) -> None:
    """One dependence heatmap per (view, metric) of a diagnostics report.

    Rows are views and columns are metrics, so the default figure pairs the
    level reading with its difference companion: level dependence between
    random-walk-like weekly series is large even when the series are
    independent. Ask for one view only and the figure says so — the
    levels-only variant carries that warning as a figure-level banner.

    ``pearson`` and ``spearman`` use a diverging map centred at 0 with
    symmetric limits. ``xi`` and ``xi_max`` span the data including any
    negative values, which are ordinary small-sample noise around
    independence and are never clamped away. The directional ``xi`` panel is
    read row = predictor X, column = target Y.

    Cells are the across-world ``quantile`` of the per-world matrices. The
    diagonal is masked (never reported), and a pair with no valid worlds is
    annotated ``N/A`` instead of being drawn as a zero. At most
    ``MAX_HEATMAP_KEYS`` keys, in a figure at most 24x24 inches, with tick
    labels thinned by a deterministic stride above 32 keys.
    """
    import matplotlib.pyplot as plt

    view_names = _plot_views(report, views)
    metric_names = _plot_names(metrics, DEPENDENCE_METRICS, "metrics")
    chosen = _plot_keys(report.keys, keys, MAX_HEATMAP_KEYS)
    level = _plot_quantile(quantile)
    index = np.array([report.keys.index(key) for key in chosen], dtype=np.int64)
    labels = list(chosen)
    n_keys = len(labels)

    fig, axes = plt.subplots(
        len(view_names),
        len(metric_names),
        figsize=_heatmap_figsize(len(metric_names), len(view_names), n_keys, n_keys),
        squeeze=False,
    )
    for row, view in enumerate(view_names):
        dependence = report[view].dependence
        for col, metric in enumerate(metric_names):
            ax = axes[row][col]
            matrix = dependence.matrix(metric, quantile=level)[np.ix_(index, index)]
            im = _heatmap(
                ax,
                matrix,
                row_labels=labels,
                col_labels=labels,
                diverging=metric in ("pearson", "spearman"),
                mask_diagonal=True,
            )
            orientation = (
                " — row = predictor X → column = target Y"
                if metric == "xi"
                else " — symmetric in the pair"
            )
            ax.set_title(
                f"{metric} — {view}{orientation}",
                fontsize=9,
                color=INK,
                loc="left",
            )
            if metric == "xi":
                ax.set_ylabel("predictor X (row)", fontsize=8, color=MUTED)
                ax.set_xlabel("target Y (column)", fontsize=8, color=MUTED)
            _colorbar(fig, im, ax)

    banner = _view_banner(fig, view_names)
    fig.suptitle(
        f"pairwise dependence — {_q_key(level)} over {report.n_worlds} worlds × "
        f"{report.n_time_steps} steps — {n_keys} of {len(report.keys)} keys",
        fontsize=11,
        color=INK,
    )
    _finish(fig, path, banner=banner, footnote=_DEPENDENCE_MASK_NOTE)


def plot_series_distributions(
    report: DataDiagnostics,
    path: str,
    *,
    views: Sequence[str] = ("levels", "differences"),
    keys: Sequence[str] | None = None,
    of: str = "value",
    kind: str = "hist",
    bins: int = 40,
) -> None:
    """A distribution panel per (view, series) of a diagnostics report.

    ``of="value"`` pools every retained value of a series over the worlds it
    is active in — a descriptive pool of realized values, not an iid sample,
    since consecutive weeks are dependent and worlds have different scales.
    It needs the raw series, so a report built with ``keep_series=False``
    raises. ``of`` may instead name one of the eight per-world slots
    (:data:`~prior_generator.diagnostics.SERIES_SLOTS`), in which case the
    panel shows the across-world distribution of that slot over its valid
    worlds.

    ``kind`` is ``"hist"`` or ``"ecdf"``. A series that is genuinely and
    identically zero renders as a real zero; a series with nothing to show is
    annotated instead. At most ``MAX_SERIES_PANELS`` panels (keys × views) in
    a grid at most 4 columns x 6 rows and at most 18x18 inches.
    """
    import matplotlib.pyplot as plt

    view_names = _plot_views(report, views)
    chosen = _plot_keys(report.keys, keys, MAX_SERIES_PANELS)
    statistic = str(of)
    if statistic == "value":
        if not report.keep_series:
            raise ValueError(
                "of='value' needs the retained series, and this report was built with "
                "keep_series=False; rebuild with data_diagnostics(..., keep_series=True) "
                f"or pass of=<slot> for a per-world slot, one of {list(SERIES_SLOTS)}"
            )
    elif statistic not in SERIES_SLOTS:
        raise ValueError(f"unknown of {of!r}; expected 'value' or one of {list(SERIES_SLOTS)}")
    if kind not in ("hist", "ecdf"):
        raise ValueError(f"unknown kind {kind!r}; expected 'hist' or 'ecdf'")

    panels = [(view, key) for view in view_names for key in chosen]
    if len(panels) > MAX_SERIES_PANELS:
        raise ValueError(
            f"{len(chosen)} keys × {len(view_names)} views = {len(panels)} panels exceed "
            f"this figure's bound of {MAX_SERIES_PANELS}; narrow keys=... (or views=...)"
        )
    ncols, nrows, figsize = _panel_grid(len(panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = np.asarray(axes).ravel()
    for position, (view, key) in enumerate(panels):
        ax = flat[position]
        series = report[view].series
        column = series.keys.index(key)
        eligible = series.eligible[:, column]
        if statistic == "value":
            values = series.values(key)
            worlds = int(eligible.sum())
        else:
            slot_valid = series.slot_valid[:, column, SERIES_SLOTS.index(statistic)]
            values = series.slot(statistic)[:, column][slot_valid]
            worlds = int(slot_valid.sum())
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        descriptor = report.descriptor(key)
        head = key if descriptor.role != "latent" else f"{key} (retained latent truth)"
        if values.size == 0:
            if not eligible.any():
                note = "No eligible observations"
            elif view == "differences" and series.n_time_steps == 0:
                note = "No difference observations"
            else:
                note = "No valid observations"
            _empty_panel(ax, note)
            ax.set_title(f"{head} — {view} — {statistic}", fontsize=9, color=INK, loc="left")
            continue
        color = PALETTE[position % len(PALETTE)]
        median = float(np.median(values))
        if kind == "hist":
            ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="none")
            ax.set_ylabel("count", fontsize=8, color=MUTED)
        else:
            ordered = np.sort(values)
            ax.step(
                ordered,
                np.arange(1, ordered.size + 1) / float(ordered.size),
                where="post",
                color=color,
                lw=1.4,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("fraction ≤ x", fontsize=8, color=MUTED)
        ax.axvline(median, color=INK, lw=1.0, ls="--")
        detail = (
            f"n={values.size:,} values ({worlds} worlds)"
            if statistic == "value"
            else f"n={values.size:,} worlds"
        )
        ax.set_title(
            f"{head} — {view}\n{statistic} — {detail} — median {median:,.3g}",
            fontsize=8,
            color=INK,
            loc="left",
        )
        _style_ax(ax)
    for ax in flat[len(panels) :]:
        ax.axis("off")

    banner = _view_banner(fig, view_names)
    fig.suptitle(
        f"series distributions — {statistic} — {kind} — {report.n_worlds} worlds × "
        f"{report.n_time_steps} steps",
        fontsize=11,
        color=INK,
    )
    _finish(fig, path, banner=banner, footnote=None)


def plot_temporal_diagnostics(
    report: DataDiagnostics,
    path: str,
    *,
    views: Sequence[str] = ("levels", "differences"),
    keys: Sequence[str] | None = None,
    quantile: float = 0.5,
) -> None:
    """Per view: an ACF heatmap, a forward lag-xi heatmap, and roughness/spike.

    Both heatmaps are series × lag, holding the across-world ``quantile`` of
    the per-world values; the third panel holds the per-series median
    roughness and median spike ratio over the worlds where those slots are
    valid. Lag-xi is a forward association between ``z_t`` and ``z_{t+h}``,
    never an impact or a forecast.

    Levels-only figures carry the random-walk warning as a banner. A series
    with no valid lag is annotated ``N/A``, a panel with none at all says
    "No valid lags", and a slot with no valid world is annotated rather than
    drawn as a zero — a real roughness of exactly zero (a genuinely flat
    series) is drawn as the zero it is. At most ``MAX_HEATMAP_KEYS`` keys, in
    a figure at most 24x24 inches, ticks thinned above 32 keys.
    """
    import matplotlib.pyplot as plt

    view_names = _plot_views(report, views)
    chosen = _plot_keys(report.keys, keys, MAX_HEATMAP_KEYS)
    level = _plot_quantile(quantile)
    index = np.array([report.keys.index(key) for key in chosen], dtype=np.int64)
    labels = list(chosen)
    n_keys = len(labels)
    n_lags = len(report.lags)
    width = min(_MAX_HEATMAP_INCHES, 1.8 + 3.0 * (2.6 + 0.10 * max(n_lags, n_keys)))
    height = min(_MAX_HEATMAP_INCHES, 1.4 + len(view_names) * (2.2 + 0.14 * n_keys))

    fig, axes = plt.subplots(len(view_names), 3, figsize=(width, height), squeeze=False)
    for row, view in enumerate(view_names):
        temporal = report[view].temporal
        series = report[view].series
        lag_labels = [f"h{lag}" for lag in temporal.lags]
        for col, metric in enumerate(("acf", "lag_xi")):
            ax = axes[row][col]
            matrix = temporal.matrix(metric, quantile=level)[index, :]
            ax.set_title(
                f"{metric} — {view} — {_q_key(level)} over {temporal.n_worlds} worlds",
                fontsize=9,
                color=INK,
                loc="left",
            )
            if not np.isfinite(matrix).any():
                _empty_panel(ax, "No valid lags")
                continue
            im = _heatmap(
                ax,
                matrix,
                row_labels=labels,
                col_labels=lag_labels,
                diverging=metric == "acf",
            )
            ax.set_xlabel("lag h", fontsize=8, color=MUTED)
            _colorbar(fig, im, ax)

        ax = axes[row][2]
        positions = np.arange(n_keys, dtype=np.float64)
        bar_width = 0.38
        drawn_any = False
        for offset, slot, color in (
            (-bar_width / 2.0, "roughness", PALETTE[0]),
            (bar_width / 2.0, "spike", PALETTE[5 % len(PALETTE)]),
        ):
            slot_values = series.slot(slot)[:, index]
            slot_valid = series.slot_valid[:, :, SERIES_SLOTS.index(slot)][:, index]
            medians = np.full(n_keys, np.nan)
            for position in range(n_keys):
                usable = slot_values[:, position][
                    slot_valid[:, position] & np.isfinite(slot_values[:, position])
                ]
                if usable.size:
                    medians[position] = float(np.median(usable))
            finite = np.isfinite(medians)
            if finite.any():
                drawn_any = True
                ax.bar(
                    positions[finite] + offset,
                    medians[finite],
                    width=bar_width,
                    color=color,
                    label=f"{slot} (median)",
                )
            for missing in np.nonzero(~finite)[0]:
                ax.text(
                    float(missing) + offset,
                    0.0,
                    "N/A",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=MUTED,
                )
        ax.set_title(f"roughness / spike — {view}", fontsize=9, color=INK, loc="left")
        if not drawn_any:
            ax.text(
                0.5,
                0.9,
                "No eligible observations"
                if not series.eligible[:, index].any()
                else "No valid observations",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color=MUTED,
            )
        else:
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        ticks = _tick_stride(n_keys)
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_ylabel("median over valid worlds", fontsize=8, color=MUTED)
        _style_ax(ax)

    banner = _view_banner(fig, view_names)
    fig.suptitle(
        f"temporal structure — {_q_key(level)} over {report.n_worlds} worlds × "
        f"{report.n_time_steps} steps — {n_keys} of {len(report.keys)} keys — "
        f"{n_lags} lags",
        fontsize=11,
        color=INK,
    )
    _finish(fig, path, banner=banner, footnote=_MASK_NOTE)


def plot_vif_diagnostics(
    report: DataDiagnostics,
    path: str,
    *,
    views: Sequence[str] = ("levels", "differences"),
    keys: Sequence[str] | None = None,
    quantile: float = 0.5,
) -> None:
    """Per view and design scope: the across-world quantile of VIF per predictor.

    Columns are the ``observed`` (active C+Z) and ``oracle`` (active C+Z+D)
    designs; rows are views, so the default figure pairs levels with
    differences. A predictor absent from a scope simply does not appear
    there, and a scope with no predictor at all is annotated "No predictors
    in scope".

    A finite quantile is drawn as a bar. A predictor whose VIF is a VALID
    ``+inf`` (exactly collinear with the rest of the design) is drawn as a
    labelled marker on the top edge carrying its world count, because a bar
    of height infinity would autoscale the axis and render as nothing:
    all-infinite is a result, not missing data. A predictor with no valid
    world — constant in every world, or too few observations — gets an
    ``N/A`` annotation instead. At most ``MAX_HEATMAP_KEYS`` keys, in a
    figure at most 24x24 inches.
    """
    import matplotlib.pyplot as plt

    view_names = _plot_views(report, views)
    chosen = _plot_keys(report.keys, keys, MAX_HEATMAP_KEYS)
    level = _plot_quantile(quantile)
    width = min(_MAX_HEATMAP_INCHES, 2.0 + len(VIF_SCOPES) * (3.0 + 0.22 * len(chosen)))
    height = min(_MAX_HEATMAP_INCHES, 1.2 + len(view_names) * 3.0)

    fig, axes = plt.subplots(
        len(view_names), len(VIF_SCOPES), figsize=(width, height), squeeze=False
    )
    for row, view in enumerate(view_names):
        for col, scope in enumerate(VIF_SCOPES):
            ax = axes[row][col]
            diagnostics = report[view].vif[scope]
            in_scope = [key for key in chosen if key in diagnostics.keys]
            ax.set_title(
                f"VIF — {scope} design — {view} — {_q_key(level)} over "
                f"{diagnostics.n_worlds} worlds",
                fontsize=9,
                color=INK,
                loc="left",
            )
            if not in_scope:
                _empty_panel(ax, "No predictors in scope")
                continue
            bars: list[tuple[int, float]] = []
            infinite: list[tuple[int, int]] = []
            unavailable: list[tuple[int, int]] = []
            for position, key in enumerate(in_scope):
                column = diagnostics.keys.index(key)
                values = diagnostics.values(key)
                valid = diagnostics.valid[:, column]
                finite = valid & np.isfinite(values)
                inf_worlds = int(np.count_nonzero(valid & np.isposinf(values)))
                if finite.any():
                    bars.append((position, float(np.quantile(values[finite], level))))
                if inf_worlds:
                    infinite.append((position, inf_worlds))
                if not finite.any() and not inf_worlds:
                    unavailable.append(
                        (position, int(np.count_nonzero(diagnostics.constant[:, column])))
                    )
            if bars:
                ax.bar(
                    [position for position, _ in bars],
                    [value for _, value in bars],
                    width=0.7,
                    color=PALETTE[col % len(PALETTE)],
                )
            top = max((value for _, value in bars), default=0.0)
            top = top * 1.18 if top > 0.0 else 1.0
            ax.set_ylim(0.0, top)
            ax.axhline(1.0, color=GRID, lw=0.8)
            for position, count in infinite:
                ax.plot(
                    [position],
                    [top],
                    marker="^",
                    markersize=9,
                    color=PALETTE[5 % len(PALETTE)],
                    clip_on=False,
                    zorder=5,
                )
                ax.text(
                    float(position),
                    top * 0.97,
                    f"∞ in {count} worlds",
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=PALETTE[5 % len(PALETTE)],
                )
            for position, constant_worlds in unavailable:
                label = "N/A (constant)" if constant_worlds else "N/A (no valid world)"
                ax.text(
                    float(position),
                    0.0,
                    label,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=MUTED,
                )
            if not bars and infinite:
                ax.text(
                    0.5,
                    0.5,
                    "no finite VIF: every valid world is ∞",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=MUTED,
                )
            ax.set_xticks(np.arange(len(in_scope)))
            ax.set_xticklabels(in_scope, rotation=90, fontsize=7)
            ax.set_ylabel(f"VIF ({_q_key(level)} across worlds)", fontsize=8, color=MUTED)
            _style_ax(ax)

    banner = _view_banner(fig, view_names)
    fig.suptitle(
        f"variance inflation — {_q_key(level)} over {report.n_worlds} worlds × "
        f"{report.n_time_steps} steps",
        fontsize=11,
        color=INK,
    )
    _finish(
        fig,
        path,
        banner=banner,
        footnote="▲ on the top edge = valid +inf (exact collinearity); N/A = no valid world",
    )


def plot_contribution_diagnostics(
    report: DataDiagnostics,
    path: str,
    *,
    keys: Sequence[str] | None = None,
    measure: str = "share",
    mode: str = "net",
    weighting: str = "macro",
    sibling_set: str = "top",
    basis: str = "base_direct_plus_indirect",
    population: str = "unconditional",
) -> None:
    """Horizontal bars of one sibling cut of the sales contribution hierarchy.

    ``basis`` selects one reading from ``report.contribution_bases`` and
    raises when that reading is not in the report; the two bases are
    alternative readings of the same media effect and never mix.
    ``sibling_set`` picks the cut (``top``, ``media_children``, ...) and
    ``keys`` narrows the rows within it, in caller order.

    Bars hold the across-world mean (``weighting="macro"``) or the pooled
    value (``weighting="micro"``) of the selected measure, and each value
    label carries the number of worlds behind it. Selecting a subset makes
    the budget a partial projection: the figure then states how many sibling
    rows are omitted and makes no claim that the bars add up. A row with no
    value under the chosen population is annotated ``N/A``; a row that is
    genuinely zero is drawn as the zero it is. At most
    ``MAX_SERIES_PANELS`` rows, in a figure at most 18x18 inches.
    """
    import matplotlib.pyplot as plt

    if not isinstance(basis, str):
        raise TypeError(f"basis must be a string, got {type(basis).__name__}")
    if basis not in report.contribution_bases:
        raise ValueError(
            f"contribution basis {basis!r} is not available in this report; available: "
            f"{sorted(report.contribution_bases)}"
        )
    budget = report.contribution_bases[basis]
    cuts = budget.sibling_sets()
    if sibling_set not in cuts:
        raise ValueError(
            f"unknown sibling_set {sibling_set!r} for basis {basis!r}; expected one of {list(cuts)}"
        )
    cut_keys = tuple(d.key for d in budget.descriptors if d.sibling_set == sibling_set)
    chosen = _plot_keys(cut_keys, keys, MAX_SERIES_PANELS)
    projection = budget.select(list(chosen))
    closure = projection.closure(sibling_set)

    rows = []
    for key in chosen:
        stats, ledger = projection.stats(
            key, measure=measure, mode=mode, weighting=weighting, population=population
        )
        value = stats["value"] if weighting == "micro" else stats["mean"]
        rows.append((key, value, ledger.valid, int(stats["active_worlds"])))

    height = min(_MAX_PANEL_INCHES, 1.9 + 0.46 * len(rows))
    fig, ax = plt.subplots(figsize=(min(_MAX_PANEL_INCHES, 9.5), height))
    offsets = np.arange(len(rows))[::-1]
    for position, (_, value, worlds, _active) in enumerate(rows):
        y = float(offsets[position])
        if value is None:
            _value_label(ax, 0.0, y, f"N/A ({worlds} worlds)", color=MUTED)
            continue
        number = float(value)
        ax.barh(y, number, height=0.62, color=PALETTE[position % len(PALETTE)])
        _value_label(ax, number, y, f"{number:,.3g} ({worlds} worlds)", color=INK)
    drawn = [float(value) for _, value, _, _ in rows if value is not None]
    if drawn:
        low, high = min(min(drawn), 0.0), max(max(drawn), 0.0)
        span = (high - low) or 1.0
        ax.set_xlim(low - 0.3 * span, high + 0.3 * span)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if all(active == 0 for _, _, _, active in rows):
        ax.text(
            0.5,
            0.02,
            "No active worlds for selected key(s)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
    ax.axvline(0.0, color=GRID, lw=0.9)
    ax.set_yticks(offsets)
    ax.set_yticklabels([key for key, _, _, _ in rows], fontsize=8)
    ax.set_xlabel(f"{mode} {measure} ({weighting}, {population})", fontsize=9, color=MUTED)
    ax.set_title(
        f"contributions to sales — {sibling_set} cut of {basis} — "
        f"{report.n_worlds} worlds × {report.n_time_steps} steps",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    if projection.is_partial_projection and closure.omitted_keys:
        footnote = (
            f"partial projection: {len(closure.omitted_keys)} sibling row(s) omitted from "
            f"the {sibling_set!r} cut, so these bars do not add up to {closure.parent}"
        )
    else:
        footnote = f"complete {sibling_set!r} cut against {closure.parent}"
    _finish(fig, path, banner=False, footnote=footnote)
