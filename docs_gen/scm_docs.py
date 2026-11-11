"""Shared helpers for the executable documentation blocks.

Imported by ``markdown-exec`` code blocks (``docs_gen`` is on ``sys.path`` via
``mkdocs_hooks.py``). Two jobs:

* ``world`` / ``corpus`` — sample ONE world (or a small corpus) and memoize it,
  so a page can plot, describe, and validate the *same* draw across many blocks
  and the build only pays for the (few-second) PyTensor work once.
* ``viz_html`` — turn a package plot into an inline SVG figure.
  A block does ``print(viz_html(...))``: returning the string (rather than
  printing here) is what lets markdown-exec capture it into the page instead of
  onto stdout. Figures render in the package's house style
  (``pymc_generator.viz.use_style``), the same one bundles and the executed
  notebooks use. Text stays text, so the site's own copy of the figure face
  renders it and the reading loupe magnifies vectors rather than pixels.
"""

from __future__ import annotations

import functools
import os
import re
import tempfile

import matplotlib

matplotlib.use("Agg")

import pymc_generator as pg  # noqa: E402

# Importing the submodule registers ``pymc_generator.viz`` as an attribute of the
# package, so the doc blocks (which all import this module) can reference
# ``pg.viz.plot_dag`` — the package's public symbols are lazy-loaded and ``viz``
# is not one of them, so ``pg.viz`` only resolves once the submodule is imported.
import pymc_generator.viz  # noqa: E402

pymc_generator.viz.use_style()
# Keep SVG text as <text> in the figure face; the site serves that face.
matplotlib.rcParams["svg.fonttype"] = "none"

# Matplotlib's SVG prologue: the XML declaration, doctype, and metadata block.
_SVG_PROLOGUE = re.compile(r"\A.*?(?=<svg\b)", re.S)
_SVG_METADATA = re.compile(r"<metadata>.*?</metadata>\s*", re.S)
# A document-level ``*`` rule in an inline SVG would restyle every page element.
_SVG_GLOBAL_STYLE = re.compile(r"<style[^>]*>.*?</style>\s*", re.S)


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


def _rounded(values, digits: int = 4) -> list:
    return [round(float(v), digits) for v in values]


def world_json(w: pg.SCM, *, seed: int) -> str:
    """Serialize one world's graph, series, and decomposition for the site's
    interactive home page, as a hidden element's ``data-world`` attribute.

    An attribute rather than a ``<script>``: instant navigation re-executes
    scripts it swaps in, and search does not index attributes. Everything is
    read from the sampled world itself; nothing is illustrative.
    """
    import html
    import json

    from pymc_generator.worlds import edges_with_coeffs, mechanism_label

    d = w.data
    nodes = (
        [f"D{j + 1}" for j in range(w.n_latent)]
        + [f"Z{m + 1}" for m in range(w.n_covariates)]
        + [f"C{k + 1}" for k in range(w.n_treatments)]
        + ["B", "Y"]
    )
    payload = {
        "name": w.name,
        "seed": seed,
        "weeks": int(w.n_time_steps),
        "identity_error": float(w.identity_error()),
        "nodes": nodes,
        "edges": [
            {"type": et, "src": src, "dst": dst, "coef": round(float(c), 3)}
            for et, src, dst, c in edges_with_coeffs(w.g, w.params)
        ]
        + [{"type": "by", "src": "B", "dst": "Y", "coef": None}],
        "mechanisms": {f"C{k + 1}": mechanism_label(w.params, k) for k in range(w.n_treatments)},
        "series": {
            **{f"C{k + 1}": _rounded(d["treatments"][:, k]) for k in range(w.n_treatments)},
            **{f"Z{m + 1}": _rounded(d["covariates"][:, m]) for m in range(w.n_covariates)},
            **{f"D{j + 1}": _rounded(d["latent_unobserved"][:, j]) for j in range(w.n_latent)},
            "Y": _rounded(d["outcome"]),
        },
        "decomposition": {
            "baseline": _rounded(d["baseline"]),
            **{f"C{k + 1}": _rounded(d["contributions"][:, k]) for k in range(w.n_treatments)},
            "indirect": _rounded(d["indirect_effects"]),
        },
    }
    text = html.escape(json.dumps(payload, separators=(",", ":")), quote=True)
    return f'<div id="pg-world" hidden data-world="{text}"></div>'


def _wrap(inner: str, caption: str | None = None) -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="scm-plot">{inner}{cap}</figure>'


def viz_html(plot_fn, w: pg.SCM, *, title: str | None = None, caption: str | None = None) -> str:
    """Return one of the package's ``viz.plot_*`` figures as inline SVG HTML.

    The ``viz`` helpers write a file, so we render to a temp SVG and inline its
    markup — the figure is the one ``write_scm_bundle`` draws, as vectors. Use as
    ``print(viz_html(pg.viz.plot_dag, world))`` inside a ``markdown-exec`` block.
    """
    fd, path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        plot_fn(w, path, title or w.name)
        with open(path, encoding="utf-8") as fh:
            svg = fh.read()
    finally:
        os.unlink(path)
    svg = _SVG_GLOBAL_STYLE.sub("", _SVG_METADATA.sub("", _SVG_PROLOGUE.sub("", svg)))
    label = (title or w.name).replace('"', "&quot;")
    svg = svg.replace("<svg ", f'<svg role="img" aria-label="{label}" ', 1)
    return _wrap(svg, caption)
