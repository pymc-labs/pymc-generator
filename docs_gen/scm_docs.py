"""Shared helpers for the executable documentation blocks.

Imported by ``markdown-exec`` code blocks (``docs_gen`` is on ``sys.path`` via
``mkdocs_hooks.py``). Two jobs:

* ``world`` / ``corpus`` — sample ONE world (or a small corpus) and memoize it,
  so a page can plot, describe, and validate the *same* draw across many blocks
  and the build only pays for the (few-second) PyTensor work once.
* ``viz_html`` — turn a package plot into inline PNG HTML.
  A block does ``print(viz_html(...))``: returning the string (rather than
  printing here) is what lets markdown-exec capture it into the page instead of
  onto stdout. The figure is wrapped in a light card that reads in both the
  light and dark site themes.
"""

from __future__ import annotations

import base64
import functools
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import pymc_generator as pg  # noqa: E402

# Importing the submodule registers ``pymc_generator.viz`` as an attribute of the
# package, so the doc blocks (which all import this module) can reference
# ``pg.viz.plot_dag`` — the package's public symbols are lazy-loaded and ``viz``
# is not one of them, so ``pg.viz`` only resolves once the submodule is imported.
import pymc_generator.viz  # noqa: E402,F401

INK, MUTED, GRID = "#1f2937", "#64748b", "#e2e8f0"

matplotlib.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
    }
)


@functools.cache
def world(scenario: int = 1, seed: int = 0, n_time_steps: int = 104) -> pg.SCM:
    """Sample (and cache) one accepted world for a named scenario.

    Deterministic in ``(scenario, seed, n_time_steps)``; memoized for the whole build.
    """
    sc = pg.SCENARIOS[scenario]
    return pg.sample_scm(
        sc.prior(n_time_steps=n_time_steps, seed=seed),
        seed=seed,
        connect_all=sc.connect_all,
        name=sc.name,
        purpose=sc.purpose,
    )


@functools.cache
def corpus(n_treatments: int = 5, n_covariates: int = 3, n_latent: int = 2, seed: int = 42):
    """Generate (and cache) a small demo corpus for the corpus guide."""
    cfg = pg.make_scm_prior(
        n_treatments=n_treatments,
        n_covariates=n_covariates,
        n_latent=n_latent,
        edge_budget={"cy": (4, 4), "dc": (2, 2), "zc": (1, 2), "cc": (0, 1)},
        n_cells=2,
        draws_per_cell=2,
        seed=seed,
    )
    return pg.sample_prior_predictive(cfg)


def _wrap(inner: str, caption: str | None = None) -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="scm-plot">{inner}{cap}</figure>'


def viz_html(plot_fn, w: pg.SCM, *, title: str | None = None, caption: str | None = None) -> str:
    """Return one of the package's ``viz.plot_*`` figures as inline-PNG HTML.

    The ``viz`` helpers write a file, so we render to a temp PNG and embed it as a
    data URI — the figure is exactly what ``write_scm_bundle`` puts on disk. Use
    as ``print(viz_html(pg.viz.plot_dag, world))`` inside a ``markdown-exec`` block.
    """
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        plot_fn(w, path, title or w.name)
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
    finally:
        os.unlink(path)
    img = f'<img src="data:image/png;base64,{b64}" alt="{title or w.name}">'
    return _wrap(img, caption)
