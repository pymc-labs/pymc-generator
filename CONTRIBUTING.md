# Contributing

## Development setup

```bash
git clone https://github.com/pymc-labs/prior-generator.git
cd prior-generator
python -m venv .venv && source .venv/bin/activate   # or: uv venv
pip install -e .[dev]                               # or: uv pip install -e .[dev]
pre-commit install
```

## Gates (same as CI)

```bash
ruff check . && ruff format --check .
mypy prior_generator
pytest tests/
```

MyPy is blocking. A small set of modules inherited a type-error backlog from
the structural-pfn extraction and are grandfathered in `pyproject.toml` — do
not add new modules to that list.

## Documentation

The docs are a MkDocs Material site under `docs/`. The API pages are generated
from docstrings (mkdocstrings), the guide pages execute real Python and render
plots inline at build time (markdown-exec), and `docs/examples.ipynb` is executed
by mkdocs-jupyter — so the docs need the full modeling stack **and** a Jupyter
kernel named `prior-generator`:

```bash
pip install -e ".[docs]"                              # or: uv pip install -e ".[docs]"
python -m ipykernel install --user --name prior-generator   # once
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs (fails on any warning)
```

Because the pages run code, a build samples several worlds and a small corpus
(cached across pages via `docs_gen/scm_docs.py`), so expect a build to take a
minute or two. To edit the examples notebook, change `docs_gen/build_examples_nb.py`
and regenerate it with `python docs_gen/build_examples_nb.py` (keeping it in sync
rather than hand-editing the `.ipynb`). The site auto-deploys to GitHub Pages from
`main` via `.github/workflows/docs.yml`.

## Reproducibility contract

Same seed + same config must produce an identical corpus. RNG draw order is
deliberately locked (disabled features consume no RNG). Any change that
reorders or adds RNG draws silently forks every stored corpus — call it out
explicitly in your PR if intended.

## Changelog & releases

- User-visible changes get an entry under `## [Unreleased]` in `CHANGELOG.md`
  (Keep a Changelog format).
- To release: move the `[Unreleased]` items under `## [X.Y.Z] - YYYY-MM-DD`,
  bump `version` in `pyproject.toml`, and push tag `vX.Y.Z`. The release
  workflow tests, builds, and publishes to PyPI via trusted publishing.
