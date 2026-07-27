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
  posterior channel/control contributions with the known SCM effects. Its
  explicit low-noise, intervention-rich easy design checks convergence,
  channel-total error, curve correlation, and normalized RMSE before reporting
  a passing recovery verdict.

- **Baseline, confounding, and intervention metadata**: `rw_baseline_std_sigma`
  now defaults to `None`, dynamically following `rw_std_sigma` for the `RW_B`
  baseline walk only; it can be set explicitly without changing other walks.
  `confounding_strength_range` optionally draws a per-world `rho` in `[0, 0.95]`
  that mixes baseline innovations into channel innovations with preserved marginal
  variance, and persists the resulting scalar. Corpus worlds can also include
  stratified, non-overlapping random channel-shock schedules with typed audit
  metadata.
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
- **Held-level shocks no longer reset adstock state** (breaking): a channel
  shock now only clamps observed spend for its window. The clamped path feeds
  the ordinary normalized causal adstock kernel, so carryover from pre-shock
  spend decays into a held window instead of being discarded, and a zero held
  level reaches an exactly zero direct response only once the full kernel span
  lies inside the window. The previous response-state surgery put generated
  worlds outside the function class a stock `GeometricAdstock`/`WeibullAdstock`
  can represent, which broke carryover recovery for any downstream MMM fit;
  with `adstock_burn_in=0` the generated contributions are now reproducible
  from observed spend to machine precision. `dense_signal_metrics`,
  `per_channel_signal`, and `signal_summary` drop their now-meaningless
  `channel_shock_channel` / `channel_shock_start` arguments, the oracle builds
  no shock tensors (it still validates the schedule against observed spend),
  and `diagnostics["signal"]` reports
  `adstock_kernel_semantics="normalized-causal-weibull-pdf"` at
  `adstock_kernel_version=2`. Corpora written with kernel version 1 no longer
  validate.
- **One amplitude per channel** (breaking): the `michaelis_menten` and `tanh`
  saturation families no longer draw their own asymptote (`mm_alpha`,
  `tanh_b`). Both are pinned to 1, leaving the structural `beta` gate as each
  channel's single amplitude — the convention pymc-marketing itself follows,
  adding a `beta` scale only to families whose transformer is bounded and
  omitting it for `michaelis_menten` / `tanh` / `hill_saturation_sigmoid`,
  whose transformers already expose an asymptote. Carrying both made the
  contribution identify only the product: on a fitted oracle the posterior
  correlation between `beta` and `mm_alpha` was -0.98, each factor was ~35%
  off, and their product was recovered to 0.1-0.3%. `SATURATION_PRIOR_RANGES`
  drops `michaelis_menten.alpha` and `tanh.b`; the oracle and generative
  models drop the matching RVs; `SCM.equation_parameters` and
  `describe_scm` no longer report them. The shape space of both families is
  unchanged — `(b, c) -> (λb, c/λ)` was a pure rescaling — but generated
  worlds change: the effective amplitude of `michaelis_menten` and `tanh`
  channels is now `beta_additive_range` alone, matching every other family
  instead of being multiplied by a second, family-specific factor. Graph,
  spend, controls, demand, active masks and shock schedules are byte-identical;
  the response-path arrays and the frozen corpus hashes move.
- **The latent factor is pinned to mean 0 / scale 1** (breaking): `rw_d_mean`
  and `rw_d_std` are no longer drawn. A latent factor carries no scale of its
  own — `(sigma_d, w_dc, u_dz, delta_db) -> (lam*sigma_d, w/lam, u/lam,
  delta/lam)` left every observable identical, and `delta_db * mu_d` was
  absorbed by `rw_b_mean`, so none of the loadings were recoverable. The whole
  `D -> *` magnitude now lives in the loadings, which is what the data
  identifies. Latent influence is consequently less dispersed across worlds
  (the `HalfNormal` factor is gone); widen `db_coeff_range` / `dc_coeff_range`
  / `dz_coeff_range` to restore spread — those knobs are now meaningful rather
  than confounded.
- **The saturation anchor is a function of parameters alone** (breaking):
  `saturation_scale` was the mean of the realized adstocked series over the
  reported window. Two costs: the "prior" was a statistic of the noise it
  generates, so no `p(theta)` existed independently of `U`; and because the
  mean spans the whole window, `do(C[t'])` for a late `t'` moved the response
  at an early `t` — the model was anti-causal in time. The anchor is now the
  expected channel level built from `softplus(rw_c_mean)`, `pulse_amp *
  pulse_prob`, and the expected levels of `Z -> C` / `C -> C` parents in
  topological order (`D -> C` drops out because the factor is now mean-zero).
  Measured over 36 channels of the supported texture, realized/anchor has
  median 1.11 with a 5-95% range of 0.77-1.54; the operating point is now at
  the knee in expectation rather than pinned per world, which widens the
  spread of channel curvature across the corpus.
- **`indirect_effects_by_source` is documented as a convention**, not an
  estimand: the total `indirect_effects` is order-free, its 3-way split is
  defined by a fixed sequential zeroing order and a different order gives
  different numbers.

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
