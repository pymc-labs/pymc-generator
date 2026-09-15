# Contributing

## Working with maintainers

This is a PyMC Labs project. Maintainers listed in [CODEOWNERS](.github/CODEOWNERS)
review contributions and make merge/release decisions. Follow the
[Code of Conduct](CODE_OF_CONDUCT.md); report sensitive vulnerabilities through
[the security policy](SECURITY.md), not a public issue.

Open an issue before a substantial API, schema, or scientific-model change.
Keep pull requests focused and explain the problem, intended behavior, evidence,
and compatibility impact. Submit only material you have the right to contribute,
under the project's [MIT license](LICENSE), and preserve upstream attribution.

Tests should protect observable behavior or a plausible regression. For changes
that do not warrant a permanent test, record an executable smoke check or other
appropriate evidence. Avoid opaque hashes, implementation-only assertions, and
loosening numerical tolerances merely to obtain a passing result.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.9.10 or
newer), then use the committed lock rather than resolving dependencies afresh:

```bash
git clone https://github.com/pymc-labs/pymc-generator.git
cd pymc-generator
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
uv run --no-sync mypy pymc_generator
uv run --no-sync pytest tests/
uv run --no-sync pytest tests/ --runslow -m slow
```

MyPy is blocking for every package module; blanket `ignore_errors` exemptions
are not allowed. The distribution ships `py.typed`, and lazy public exports have
explicit type-checking imports. Corpus mappings contain arrays plus nested
metadata, so their values are heterogeneous rather than uniformly ndarrays.

## Documentation

The docs are a MkDocs Material site under `docs/`. The API pages are generated
from docstrings (mkdocstrings), the guide pages execute real Python and render
plots inline at build time (markdown-exec), and both notebooks under
`docs/examples/` are executed by mkdocs-jupyter. The docs require the full
modeling stack and a Jupyter kernel named `pymc-generator`:

```bash
uv sync --locked --extra dev --extra docs
uv run --no-sync python -m ipykernel install --user --name pymc-generator
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

For scientific or numerical changes, compare representative generated arrays
before and after, record dependency versions and seeds, and explain any intended
differences. Recovery examples must report divergences, effective sample sizes,
convergence diagnostics, and undefined metrics honestly. Passing a build or one
fit is not evidence of universal identifiability, calibration, or recovery.

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
conda activate pymc-generator-build
uv sync --locked --group build
uv run --no-sync python -m build --sdist --no-isolation
uv run --no-sync python scripts/build_conda.py dist/pymc_generator-0.0.1.tar.gz \
  --conda "$CONDA_PREFIX/bin/conda"
```

Supply the exact source archive for the version being built. The script reads
its version and SHA-256, builds and tests the two exact-commit upstream companion
packages, then builds and tests `pymc-generator`. All runtime packages are
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

## Repository protections

Keep `main` protected for administrators too: require an up-to-date branch,
resolved review conversations, and one approval from someone other than the
latest pusher; dismiss stale approvals and prohibit force pushes and deletion.
Required checks are `lint`, `test (3.12)`, `test (3.13)`, `slow-tests`, `build`,
`native-conda`, `docs-build`, `secrets`, and `codeql`. CodeQL intentionally skips
private repositories; GitHub accepts skipped required checks. Do not require
the sole default code owner to approve their own contribution.

Workflow tokens default to read-only and cannot approve pull requests. Actions
are pinned to immutable revisions and checkouts do not retain credentials.
Weekly Dependabot action-update PRs and dependency security alerts/PRs still
require human review; numerical dependency updates need the preservation
evidence above and matching uv/conda changes, not automatic merging.

The free Gitleaks CLI scans all fetched Git history on pull requests, main-branch
pushes, and a weekly schedule, with redacted output. CodeQL runs only when the
event explicitly identifies a public repository. Neither workflow enables paid
private security products. GitHub's private reporting, secret scanning, and push
protection depend on repository eligibility; verify them separately before
public release. Use the [security policy](SECURITY.md) reporting fallback when
private reporting is unavailable. A clean scan is not proof that history contains
no sensitive material.

## Changelog & releases

- User-visible changes get an entry under `## [Unreleased]` in `CHANGELOG.md`
  (Keep a Changelog format).
- Move the release's `[Unreleased]` items under `## [X.Y.Z] - YYYY-MM-DD`,
  update `pymc_generator/_version.py`, run `uv sync --locked` to refresh installed
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

Before publishing a draft, maintainers must check that the release commit has
passing CI, documentation, and applicable security results, review migration notes and scientific
limitations, and verify installation from the actual release assets. Preserve
artifact checksums and platform-specific replay information where applicable.
Do not overwrite a published tag or replace its artifacts to conceal a defect;
issue a new version with an explanation.

Before changing repository visibility, separately review source/history for
secrets and private data, attribution and redistribution rights, available
repository protections, and all outstanding release-readiness failures. A
successful local check is not a substitute for a passing remote workflow.
