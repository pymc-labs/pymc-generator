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
plots inline at build time (markdown-exec), and both notebooks under
`docs/examples/` are executed by mkdocs-jupyter. The docs require the full
modeling stack and a Jupyter kernel named `prior-generator`:

```bash
pip install -e ".[docs]"                              # or: uv pip install -e ".[docs]"
python -m ipykernel install --user --name prior-generator   # once
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs (fails on any warning)
```

Edit the `.ipynb` files directly: they are the canonical sources, not generated
copies of Python string literals. Commit source cells without outputs, execution
counters, or widget state. Clear them before committing:

```bash
uv run --frozen --extra docs jupyter nbconvert --to notebook --inplace \
  --ClearOutputPreprocessor.enabled=True \
  --ClearMetadataPreprocessor.enabled=True \
  --ClearMetadataPreprocessor.preserve_nb_metadata_mask='{("language_info", "name"), "kernelspec"}' \
  docs/examples/*.ipynb
```

A full build samples worlds and executes the recovery case study's three
four-chain MCMC fits. Allow for their runtime and inspect their actual warnings
and diagnostics; successful execution does not imply successful statistical
recovery. Site deployment is controlled by `.github/workflows/docs.yml`.

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
