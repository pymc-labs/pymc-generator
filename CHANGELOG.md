# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on 0.x, minor versions may contain breaking changes.

## [Unreleased]

### Added

- **Auditable single-world structural equations**: `SCM.equations` renders the
  executed vector-valued Pearl SCM for every `D`, `Z`, `C`, `B`, and `Y` node;
  `SCM.equation_parameters` exposes the realized active coefficients and
  family-specific response parameters; and `SCM.exogenous` returns defensive
  copies of the accepted full-horizon innovations. `SCM.params` now contains
  every continuous mechanism and random-walk parameter plus realized shock
  state, making exact graph replay possible without changing the corpus schema.
- **Executable PyMC-Marketing recovery audit**: the simple-model notebook now
  builds a dated `C1`–`C3`, `Z1`–`Z2`, `Y` dataset, fits the installed
  PyMC-Marketing MMM with four chains and 500 posterior draws, and compares
  posterior channel/control contributions with the known SCM effects. The audit
  reports convergence, interval coverage, curve correlation, and RMSE instead
  of assuming that a graph-identifiable world is recoverable under default
  finite-sample model settings.

- **Baseline, confounding, and intervention metadata**: `rw_baseline_std_sigma`
  now defaults to `None`, dynamically following `rw_std_sigma` for the `RW_B`
  baseline walk only; it can be set explicitly without changing other walks.
  `confounding_strength_range` optionally draws a per-world `rho` in `[0, 0.95]`
  that mixes baseline innovations into channel innovations with preserved marginal
  variance, and persists the resulting scalar. Corpus worlds can also include
  stratified, non-overlapping random channel-shock schedules with reset-aware
  adstock responses and typed audit metadata.
- **Persisted signal features v2**: corpora now store dense per-direct-channel
  signal metrics and validity masks under the `identifiability` metadata block,
  together with a versioned layout in `diagnostics["signal"]`. Metrics are
  recomputed from the final float32, post-truncation corpus arrays so persisted
  data and diagnostics agree. V2 adds estimability-aware R² validity and
  self-contained adstock recomputation settings.

- **ACE prior-conditioning hyperprior** (`prior_conditioning=True`): each cell
  draws a narrowed prior interval per conditioned quantity (`adstock_alpha`,
  `hill_shape`) — `w ~ U(w_lo, w_hi)`, `lo ~ U(S_lo, S_hi − w)` — and its
  parameters are drawn as `pm.Uniform(lo, lo + w)` instead of the global
  support. Intervals are recorded in the corpus under the new `prior_cond`
  `(N, P)` key (packed `(low, width)` pairs in the locked, append-only
  `PRIOR_COND_LAYOUT` order; present iff enabled) with a self-describing
  `diagnostics["prior_cond"]` echo of layout, supports, and width ranges.
  `sample_scm` honors the same draw (`SCM.extras["prior_cond"]`) and
  `describe_scm` prints the intervals. Unconditioned corpora stay
  byte-identical (the interval draws consume no RNG when disabled).
- **World model promoted to public API**: `build_world_model`,
  `sample_structure`, `sample_prior_cond`, and `draw_worlds` are now exported
  and documented (new [World model](docs/reference/world-model.md) reference
  page).
- **Posterior oracle** (`build_oracle_model` / `SCM.oracle_model()`): the
  observed-data variant of the world model — same structure, same prior
  definitions (shared spec helpers, so draw and oracle cannot drift), with the
  world's spend/controls/sales attached — so `pm.sample` yields the
  structure-known posterior on any drawn world. The generator's normalized
  walks (`walk * std / walk.std()`) admit no closed-form sales-noise density,
  so the oracle keeps the latent demand/baseline walks exact and represents
  the `RW_Y` sales noise as iid Normal with the same HalfNormal scale prior;
  all concessions are documented in the docstring and the new
  [posterior-oracle guide](docs/guide/oracle.md).

- Documentation site (`docs/`, MkDocs Material) at
  <https://pymc-labs.github.io/prior-generator/>: docstring-driven API reference
  (mkdocstrings), guide pages whose figures are produced by executing real code
  at build time (markdown-exec), and a runnable examples notebook executed by
  mkdocs-jupyter. Added a `docs` extra, a GitHub Pages workflow
  (`.github/workflows/docs.yml`), and build instructions in `CONTRIBUTING.md`.

### Changed

- **Named mechanism-family probabilities**: `adstock_family_probs` and
  `saturation_family_probs` are now dictionaries keyed by canonical family
  names instead of positional tuples. Exact key validation prevents silent
  family-order mistakes; legacy tuple inputs are intentionally rejected.

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
