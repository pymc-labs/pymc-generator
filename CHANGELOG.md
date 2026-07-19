# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on 0.x, minor versions may contain breaking changes.

## [Unreleased]

### Added

- Documentation site (`docs/`, MkDocs Material) at
  <https://pymc-labs.github.io/prior-generator/>: docstring-driven API reference
  (mkdocstrings), guide pages whose figures are produced by executing real code
  at build time (markdown-exec), and a runnable examples notebook executed by
  mkdocs-jupyter. Added a `docs` extra, a GitHub Pages workflow
  (`.github/workflows/docs.yml`), and build instructions in `CONTRIBUTING.md`.

## [0.0.1] - 2026-07-14

### Added

- Initial extraction of the synthetic-MMM world generator from
  `pymc-labs/structural-pfn` at commit `d5fd09f` (plan-05 merge).
- Core generation: `SCMPrior` + `sample_prior_predictive` (additive structural
  causal models with exact interventional decompositions) and the
  `make_scm_prior` preset, with DAG sampling and per-edge-type budgets.
- The world is a drawable **PyMC model** (`build_world_model`): continuous
  priors are PyMC distributions, noise is `pm.Normal` / `pm.Bernoulli` RVs, and
  the media response uses **pymc-marketing**'s adstock/saturation transforms;
  one `pm.draw` yields parameters, series, and the full decomposition. Both
  `sample_prior_predictive` and `sample_scm` draw from this model. Discrete structure
  (DAG, mechanism family, walk smoothness) is drawn concretely per world.
- Signal diagnostics: per-channel signal metrics, corpus-level summary, and
  quality gate (`prior_generator.signal_diagnostics`).
- `DataGenerator` facade and `save_corpus`/`load_corpus` (compressed `.npz`,
  the format PFN training pipelines consume).
- SCM-level API: `sample_scm` (single accepted world), `describe_scm`
  (plain-text world descriptions), inspection bundles (CSV datasets,
  decomposition truth, DAG `.dot`, matplotlib figures) and five named
  audit scenarios, plus the `prior-generator` CLI.

### Excluded (deliberately, relative to structural-pfn)

- The deprecated L0/L1 PyMC generation path, legacy channel texture, and the
  deprecated `create_generator`/`create_variable_size_generator` constructors.
  The one supported prior is `make_scm_prior(texture="diverse")`.
- Dead modules with zero importers (`cdag_catalog`, `spec_to_adjacency`,
  `corpus_provider`).
