# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on 0.x, minor versions may contain breaking changes.

## [Unreleased]

## [0.0.1] - 2026-07-14

### Added

- Initial extraction of the synthetic-MMM world generator from
  `pymc-labs/structural-pfn` at commit `d5fd09f` (plan-05 merge).
- Core generation: `CorpusConfig` + `generate_corpus` (additive structural
  causal models with exact interventional decompositions), `make_l1_additive_cfg`
  preset, PyTensor mechanism library (adstock + saturation families),
  smoothed random-walk latents, DAG sampling with per-edge-type budgets.
- Signal diagnostics: per-channel signal metrics, corpus-level summary, and
  quality gate (`prior_generator.signal_diagnostics`).
- `DataGenerator` facade and `save_corpus`/`load_corpus` (compressed `.npz`,
  the format PFN training pipelines consume).
- World-level API: `sample_world` (single accepted world), `describe_world`
  (plain-text world descriptions), inspection bundles (CSV datasets,
  decomposition truth, DAG `.dot`, matplotlib figures) and five named
  audit scenarios, plus the `prior-generator` CLI.

### Excluded (deliberately, relative to structural-pfn)

- The deprecated L0/L1 PyMC generation path, legacy channel texture, and the
  deprecated `create_generator`/`create_variable_size_generator` constructors.
  The one supported prior is `make_l1_additive_cfg(texture="diverse")`.
- Dead modules with zero importers (`cdag_catalog`, `spec_to_adjacency`,
  `corpus_provider`).
