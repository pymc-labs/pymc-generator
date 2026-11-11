"""Shared helpers for the executable documentation blocks.

Imported by ``markdown-exec`` code blocks (``docs_gen`` is on ``sys.path`` via
``mkdocs_hooks.py``). Three jobs:

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
* ``editor_demo`` — the home page's editor window, filled from a real run.
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


_DEMO_ICONS = {
    "folder": '<path d="M1.5 4.5v8h13v-7h-7L6 3.5H1.5Z"/>',
    "chevron": '<path d="M6 4.5 9.5 8 6 11.5"/>',
    "doc": '<path d="M3.5 1.5h6l3 3v10h-9Z"/><path d="M9.5 1.5v3h3M5.5 8h5M5.5 10.5h5"/>',
    "sheet": '<path d="M2.5 3.5h11v9h-11ZM2.5 6.5h11M2.5 9.5h11M6.5 3.5v9"/>',
    "image": '<path d="M2.5 3.5h11v9h-11Z"/><path d="m2.5 11 3.5-3.5 2.5 2.5 2-2 3 3"/><circle cx="10.5" cy="6" r="1"/>',
    "archive": '<path d="M2.5 2.5h11v11h-11Z"/><path d="M8 2.5v2M8 5.5v1M8 7.5v1"/><path d="M6.5 9.5h3v2.5h-3Z"/>',
}


def _icon(name: str) -> str:
    return f'<svg class="pg-demo__icon" viewBox="0 0 16 16" aria-hidden="true">{_DEMO_ICONS[name]}</svg>'


def _file_icon(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1]
    return _icon({"csv": "sheet", "npy": "sheet", "png": "image", "npz": "archive"}.get(ext, "doc"))


def _code_lines(code: str) -> str:
    """Highlight a cell, one span per line carrying its length for the typing reveal."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    marked = highlight(code, PythonLexer(), HtmlFormatter(nowrap=True)).rstrip("\n").split("\n")
    return "".join(
        f'<span class="pg-demo__line" style="--chars: {len(raw)}">{line}</span>'
        for raw, line in zip(code.split("\n"), marked, strict=True)
    )


def _run_cells(cells: list[str], cwd: str) -> list[str | None]:
    """Run notebook cells in order inside ``cwd``, sharing one namespace.

    Returns what each cell displays, as Jupyter does: the ``repr`` of a final
    bare expression unless it is ``None``. The working directory is restored
    afterwards.
    """
    import ast

    namespace: dict = {}
    shown: list[str | None] = []
    here = os.getcwd()
    os.chdir(cwd)
    try:
        for code in cells:
            tree = ast.parse(code)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), namespace)  # noqa: S102 - the page's own cells
            value = (
                None
                if last is None
                else eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
            )  # noqa: S307
            shown.append(None if value is None else repr(value))
    finally:
        os.chdir(here)
    return shown


def _size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB"


def editor_demo(*, scenario: int = 1, seed: int = 0, rows: int = 24, weeks: int = 16) -> str:
    """Return the home page's editor window as HTML, filled from a real run.

    The window is illustrative; its contents are not. Its four notebook cells
    are executed here, in order, in a temporary directory: the first samples a
    world and shows its identity error, the second writes that world's bundle,
    the third draws a twelve-world corpus and saves it as one compressed
    ``.npz``, and the fourth loads it back. The explorer lists the files those
    cells wrote; the sheets show the first ``rows`` rows of the bundle's
    ``dataset.csv`` and the first ``weeks`` weeks of every world in the archive's
    ``outcome_raw``; the archive view lists the arrays inside the ``.npz``. The
    page's script plays the window from an empty notebook to that last sheet;
    without it, the window shows the final state.
    """
    import html
    import zipfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    folder = "my_world"
    archive = "worlds.npz"
    member = "outcome_raw"
    cells = [
        "import pymc_generator as pg\n"
        "\n"
        f"sc = pg.SCENARIOS[{scenario}]\n"
        f"prior = sc.prior(n_time_steps=104, seed={seed})\n"
        f"world = pg.sample_scm(prior, seed={seed},\n"
        "                      connect_all=sc.connect_all,\n"
        "                      name=sc.name)\n"
        "world.identity_error()",
        f'pg.write_scm_bundle(world, "{folder}/")',
        "cfg = pg.make_scm_prior(\n"
        "    n_treatments=3, n_covariates=2, n_latent=1,\n"
        "    n_cells=4, draws_per_cell=3, seed=42)\n"
        "corpus = pg.sample_prior_predictive(cfg)\n"
        f'pg.save_corpus(corpus, "{archive}")',
        f'c = pg.load_corpus("{archive}")\nc["{member}"].shape',
    ]

    with tempfile.TemporaryDirectory() as tmp:
        shown = _run_cells(cells, tmp)
        bundle = sorted(os.listdir(Path(tmp) / folder))
        data = pd.read_csv(Path(tmp) / folder / "dataset.csv")
        npz = Path(tmp) / archive
        npz_size = npz.stat().st_size
        with zipfile.ZipFile(npz) as z:
            packed = {i.filename: i.compress_size for i in z.infolist()}
        with np.load(npz, allow_pickle=False) as arrays:
            members = [(k, arrays[k].shape, arrays[k].dtype) for k in arrays.files]
            outcome = arrays[member]

    def cell(n: int, code: str, result: str | None) -> str:
        out = (
            ""
            if result is None
            else f'<output class="pg-demo__output">{html.escape(result)}</output>'
        )
        return (
            f'<div class="pg-demo__cell" data-cell="{n}">'
            f'<span class="pg-demo__count">[{n}]</span>'
            f'<div class="pg-demo__code highlight">{_code_lines(code)}</div>'
            f"{out}</div>"
        )

    def item(name: str, made: str, extra: str = "") -> str:
        return (
            f'<li class="pg-demo__file" data-file="{html.escape(name)}" data-made="{made}">'
            f"{_file_icon(name)}<span>{html.escape(name)}</span>{extra}</li>"
        )

    def tab(view: str, icon: str, label: str, active: bool = False) -> str:
        state = " is-active" if active else ""
        return f'<span class="pg-demo__tab{state}" data-tab="{view}">{_icon(icon)}{html.escape(label)}</span>'

    # The bundle's dataset.csv, as a sheet: lettered columns, numbered rows.
    n_rows, n_cols = data.shape
    letters = [chr(ord("A") + j) for j in range(n_cols)]
    sheet_head = "".join(f'<th scope="col">{c}</th>' for c in letters)
    sheet_body = (
        '<tr><th scope="row">1</th>'
        + "".join(f"<td>{html.escape(c)}</td>" for c in data.columns)
        + "</tr>"
    )
    for i, (_, row) in enumerate(data.head(rows).iterrows()):
        values = "".join(
            f"<td>{int(v)}</td>" if col == "week" else f"<td>{v:.4f}</td>" for col, v in row.items()
        )
        sheet_body += f'<tr><th scope="row">{i + 2}</th>{values}</tr>'

    # The archive's members, as an archive tool lists them.
    def kind(dtype: np.dtype) -> str:
        return "str" if dtype.kind == "U" else str(dtype)

    listing = "".join(
        f'<tr data-member="{name}.npy"><td>{_icon("sheet")}{name}.npy</td>'
        f"<td>{html.escape(str(shape))}</td><td>{kind(dtype)}</td><td>{_size(packed[name + '.npy'])}</td></tr>"
        for name, shape, dtype in members
    )

    # Every world in the archive: one row per world, one column per week.
    n_worlds, n_weeks = outcome.shape
    world_head = (
        "".join(f'<th scope="col">{t}</th>' for t in range(weeks)) + '<th scope="col">…</th>'
    )
    world_body = "".join(
        f'<tr><th scope="row">{i}</th>'
        + "".join(f"<td>{v:.3f}</td>" for v in outcome[i, :weeks])
        + "<td>…</td></tr>"
        for i in range(n_worlds)
    )

    return (
        '<div class="pg-demo__window" data-pg-window role="img" '
        'aria-label="Illustrative editor: a notebook samples one world and writes its bundle, opens its '
        f"dataset.csv as a sheet, then draws {n_worlds} worlds into {archive}, lists the arrays inside it, "
        f'and opens {member} with one row per world.">'
        '<div class="pg-demo__bar">'
        '<span class="pg-demo__lights" aria-hidden="true"><i></i><i></i><i></i></span>'
        '<span class="pg-demo__tabs">'
        f"{tab('notebook', 'doc', 'explore.ipynb')}"
        f"{tab('dataset', 'sheet', 'dataset.csv')}"
        f"{tab('archive', 'archive', archive)}"
        f"{tab('outcome', 'sheet', member + '.npy', active=True)}"
        "</span>"
        '<span class="pg-demo__kernel" data-pg-kernel>pymc-generator</span>'
        "</div>"
        '<div class="pg-demo__body is-wide">'
        '<ul class="pg-demo__files">'
        f'<li class="pg-demo__file">{_icon("doc")}<span>explore.ipynb</span></li>'
        f'<li class="pg-demo__folder" data-made="bundle">{_icon("chevron")}{_icon("folder")}<span>{folder}</span></li>'
        + "".join(item(f, "bundle") for f in bundle)
        + item(archive, "corpus", f'<span class="pg-demo__size">{_size(npz_size)}</span>')
        + "</ul>"
        '<div class="pg-demo__pane">'
        '<div class="pg-demo__notebook" data-view="notebook">'
        + "".join(
            cell(n + 1, code, result)
            for n, (code, result) in enumerate(zip(cells, shown, strict=True))
        )
        + "</div>"
        '<div class="pg-demo__sheet" data-view="dataset">'
        f'<div class="pg-demo__formula"><span>{letters[-1]}2</span><span>{data.iloc[0, -1]:.4f}</span></div>'
        '<div class="pg-demo__grid"><table class="pg-demo__table">'
        f'<thead><tr><th scope="col"></th>{sheet_head}</tr></thead><tbody>{sheet_body}</tbody></table></div>'
        f'<div class="pg-demo__status">{n_rows} rows × {n_cols} columns</div>'
        "</div>"
        '<div class="pg-demo__sheet" data-view="archive">'
        '<div class="pg-demo__grid"><table class="pg-demo__table pg-demo__table--list">'
        '<thead><tr><th scope="col">Name</th><th scope="col">Shape</th><th scope="col">Type</th>'
        f'<th scope="col">Packed</th></tr></thead><tbody>{listing}</tbody></table></div>'
        f'<div class="pg-demo__status">{len(members)} arrays · {_size(npz_size)} compressed</div>'
        "</div>"
        '<div class="pg-demo__sheet is-open" data-view="outcome">'
        '<div class="pg-demo__grid"><table class="pg-demo__table pg-demo__table--worlds">'
        f'<thead><tr><th scope="col">world</th>{world_head}</tr></thead><tbody>{world_body}</tbody></table></div>'
        f'<div class="pg-demo__status">{member}.npy · {outcome.dtype} · {n_worlds} worlds × {n_weeks} weeks</div>'
        "</div>"
        "</div>"
        "</div>"
        '<svg class="pg-demo__cursor" viewBox="0 0 16 20" aria-hidden="true">'
        '<path d="M1.5 1.5v14.6l3.9-3.7 2.6 5.9 2.6-1.1-2.6-5.8h5.4Z"/></svg>'
        '<div class="pg-demo__progress" aria-hidden="true"><i></i></div>'
        "</div>"
    )


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
