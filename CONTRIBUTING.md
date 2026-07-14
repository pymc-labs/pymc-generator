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
