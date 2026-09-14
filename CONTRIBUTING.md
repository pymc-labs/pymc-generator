# Contributing

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.9.10 or
newer), then use the committed lock rather than resolving dependencies afresh:

```bash
git clone https://github.com/pymc-labs/prior-generator.git
cd prior-generator
uv sync --locked --extra dev
uv run --no-sync pre-commit install
```

The `dev` extra provides the same Ruff, mypy, pytest, and pre-commit versions
used by CI. `uv run --no-sync` retains the explicitly selected profile.
Use `uv lock --check` to check metadata/lock consistency. Update dependencies
deliberately with evidence; do not use `uv lock --upgrade` as a routine setup step.

## Gates (same as CI)

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy prior_generator
uv run --no-sync pytest tests/
uv run --no-sync pytest tests/ --runslow -m slow
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
uv sync --locked --extra dev --extra docs
uv run --no-sync python -m ipykernel install --user --name prior-generator
uv run --no-sync mkdocs serve            # http://127.0.0.1:8000
uv run --no-sync mkdocs build --strict   # fails on documentation build warnings
```

Edit the `.ipynb` files directly: they are the canonical sources, not generated
copies of Python string literals. Commit source cells without outputs, execution
counters, or widget state. Clear them before committing:

```bash
uv run --no-sync jupyter nbconvert --to notebook --inplace \
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

The same seed and configuration must reproduce numerical arrays in the same
locked environment. Timing diagnostics are excluded from persisted shards.
RNG draw order is deliberate: disabled features consume no RNG. Changes to
draw order, numerical dependencies, or generation semantics require explicit
evidence and migration notes; cross-platform bitwise equality is not promised.

## Build artifacts

Build with locked tools rather than an isolated resolver selecting new versions:

```bash
uv sync --locked --all-extras --group build
uv run --no-sync python -m build --no-isolation
uv run --no-sync twine check dist/*
uv run --no-sync python -I scripts/smoke_install.py
```

The smoke command exercises generation, reconstruction, and persisted-corpus
validation. CI also runs it against a wheel installed into a clean environment
with runtime dependencies taken from `uv.lock`.

Source archives also include the complete tests, benchmarks, documentation and
its hooks/helpers, lockfile, and developer configuration. CI installs the archive
into a fresh locked environment, collects the full suite (including slow cases),
and executes both the installed-package smoke and a real documentation example.

## Conda artifacts

The native build has its own conda environment; it does not change your base
environment or the locked uv modeling stack:

```bash
conda env create --file conda/build-environment.yml
conda activate prior-generator-build
uv sync --locked --group build
uv run --no-sync python -m build --sdist --no-isolation
uv run --no-sync python scripts/build_conda.py dist/prior_generator-0.0.1.tar.gz \
  --conda "$CONDA_PREFIX/bin/conda"
```

Supply the exact source archive for the version being built. The script reads
its version and SHA-256, builds and tests the two exact-commit upstream companion
packages, then builds and tests `prior-generator`. All runtime packages are
native conda packages. Pip is used only as the build backend's installation
frontend with dependency resolution and build isolation disabled.

The builder uses libmamba with explicit channels, never uploads to Anaconda.org,
and publishes `dist/conda-channel` only after all package tests pass. Its output
directory must not already exist; choose a fresh `--output-folder` when repeating
a build. Recipes retain upstream license files and immutable source checksums.
When deliberately changing a modeling dependency, update both `uv.lock` and its
native recipe, including the source checksum and commit-specific build string.

Follow the [conda installation instructions](docs/getting-started.md#install-with-conda)
to create an environment from the resulting channel. From the checkout, verify it
with that environment's `python -I scripts/smoke_install.py`. Preserve a
platform-specific `conda list --explicit --sha256` export for native-run replay;
it is not interchangeable with `uv.lock`.

## Changelog & releases

- User-visible changes get an entry under `## [Unreleased]` in `CHANGELOG.md`
  (Keep a Changelog format).
- Move the release's `[Unreleased]` items under `## [X.Y.Z] - YYYY-MM-DD`,
  update `prior_generator/_version.py`, run `uv sync --locked` to refresh installed
  metadata, and push tag `vX.Y.Z` from a reviewed commit. Build metadata and the
  public `__version__` both use this one version value.
- The tag workflow reuses CI's locked tests, lint/type checks, wheel smoke test,
  and native conda build. Conda packages use that same verified source archive.
  It creates a **draft GitHub Release** containing the wheel, source archive,
  `conda-channel.tar.gz`, and `SHA256SUMS`; it does not upload to PyPI.
- Review the draft, documentation checks, and installation evidence before
  publishing it. uv and native conda are the primary installation paths.
  Repository visibility is a separate final readiness decision, never an
  automatic build or release step.
