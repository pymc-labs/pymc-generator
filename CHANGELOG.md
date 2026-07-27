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

- **Signal diagnostics v3**: `diagnostics["signal"]` now reports the
  machine-readable `response_warmup_weeks` and diagnostic-only
  `frac_zero_contemporaneous_weight` fields. The former identifies reported
  response weeks that depend on unpersisted burn-in spend when an eligible
  direct channel has a nonidentity adstock kernel; the latter surfaces direct
  channels whose current-week adstock weight is zero.

- **Baseline, confounding, and intervention metadata**: `rw_baseline_std_sigma`
  now defaults to `None`, dynamically following `rw_std_sigma` for the `RW_B`
  baseline walk only; it can be set explicitly without changing other walks.
  `confounding_strength_range` optionally draws a per-world `rho` in `[0, 0.95]`
  that mixes baseline innovations into channel innovations with preserved marginal
  variance, and persists the resulting scalar. Corpus worlds can also include
  stratified, non-overlapping random channel-shock schedules with typed audit
  metadata.
- **Persisted signal features**: corpora store dense per-direct-channel signal
  metrics and validity masks under the `identifiability` metadata block,
  together with a versioned layout in `diagnostics["signal"]`. Metrics are
  recomputed from the final float32, post-truncation corpus arrays so persisted
  data and diagnostics agree. The initial layout added estimability-aware R²
  validity and self-contained adstock recomputation settings.

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
  structure-known posterior on any drawn world. Fixed-scale walks are ordinary
  multivariate normals, so the oracle keeps latent demand/baseline walks exact
  and currently represents `RW_Y` as iid Normal with the same HalfNormal scale
  prior; all concessions are documented in the docstring and the new
  [posterior-oracle guide](docs/guide/oracle.md).

- Documentation site (`docs/`, MkDocs Material) at
  <https://pymc-labs.github.io/prior-generator/>: docstring-driven API reference
  (mkdocstrings), guide pages whose figures are produced by executing real code
  at build time (markdown-exec), and a runnable examples notebook executed by
  mkdocs-jupyter. Added a `docs` extra, a GitHub Pages workflow
  (`.github/workflows/docs.yml`), and build instructions in `CONTRIBUTING.md`.

### Changed
- **Adstock kernel metadata corrected** (breaking): the persisted semantics
  label is now `normalized-causal-minmax-weibull-density` and its version is
  `3`. pymc-marketing min-max rescales the Weibull density before
  sum-normalizing, so the old `normalized-causal-weibull-pdf` label was false;
  corpora carrying the prior label or version `2` no longer validate.

- **Named mechanism-family probabilities**: `adstock_family_probs` and
  `saturation_family_probs` are now dictionaries keyed by canonical family
  names instead of positional tuples. Exact key validation prevents silent
  family-order mistakes; legacy tuple inputs are intentionally rejected.
- **Channel-walk amplitudes are relative by default** (breaking):
  `rw_channel_std_range` changes from `None` to `(0.15, 0.8)`, and `rw_c`
  amplitudes always scale by `softplus(rw_c_mean)`. The default configuration
  therefore generates different worlds at a fixed seed.
- **`SlotLayout` is extended-only** (breaking): `SlotLayout(K, M, J)` now uses
  the canonical eight-block layout by default, changing `n_slots`; supplying
  any other `edge_types` sequence now raises instead of constructing a custom
  layout.
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
  `channel_shock_channel` / `channel_shock_start` arguments; the oracle builds
  no shock tensors (it still validates the schedule against observed spend).
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
  response-path arrays, `weibull_lam`, `weibull_k`, `baseline_raw`,
  `baseline_intrinsic`, `control_contribution`, `confounder_contribution`, and
  the frozen corpus hashes move because removing two RVs shifts later seeded draws.
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
  expected channel level built as `softplus(softplus(rw_c_mean) + pulse_amp *
  pulse_prob + weighted expected Z -> C / C -> C parent terms)`, accumulated
  in topological order (`D -> C` drops out because the factor is now mean-zero).
  Measured over 36 channels of the supported texture, realized/anchor has
  median 1.11 with a 5-95% range of 0.77-1.54; the operating point is now at
  the knee in expectation rather than pinned per world, which widens the
  spread of channel curvature across the corpus.
- **`indirect_effects_by_source` is documented as a convention**, not an
  estimand: the total `indirect_effects` is order-free, its 3-way split is
  defined by a fixed sequential zeroing order and a different order gives
  different numbers.
- **Random walks are normalized by a constant, not by their own realized
  standard deviation** (breaking): `symbolic_random_walk` and its numpy twin
  divided the centred, smoothed Brownian path by `walk.std()`, which made
  `eps -> walk` invariant to `eps -> k * eps`. The likelihood was therefore
  exactly flat along the radial direction of every `T_full`-dimensional
  innovation vector — a degeneracy no step size can fix, and the dominant
  remaining obstacle to sampling the oracle. The divisor is now the constant
  `sqrt(tr(A A^T) / T)` for the fixed linear operator
  `A = centre . movavg . cumsum`, computed exactly and cached per
  `(T, width)`. Consequences: `std` is now the walk's EXPECTED amplitude
  (`E[var(walk)] == std ** 2`) rather than its exact realized one, so
  `rw_*_std_sigma` knobs set an expectation and the realized value scatters
  around it; and the walk becomes an ordinary multivariate normal with
  covariance `(std / c) ** 2 * A A^T`, which means the oracle's iid
  sales-noise concession is now removable in principle. Measured on the
  oracle: floor world r-hat 1.295 -> 1.048, ESS 11 -> 67, coverage 2/3 -> 3/3;
  overlap world contribution error 1.3% -> 0.4%, r-hat 2.85 -> 2.13;
  no-overlap contribution error 20.2% -> 10.3%.


- **Signal-summary/gate contract version 2 → 3** (breaking):
  `SIGNAL_METRIC_LAYOUT` and `METRIC_KEYS` remain the same nine entries in the
  same order. `SIGNAL_METRIC_VERSION` changes because summaries add the
  response-warmup/contemporaneous-weight diagnostics and the gate adds
  `frac_contrib_rel_std_lt_001` and `frac_contrib_r2_gt_095` limits of `0.10`.
  Existing corpora must be regenerated for the new summary and gate contract.
- **`build_oracle_model` requires `saturation_scale`** (breaking): the
  observed-channel mean fallback was removed so every oracle uses the same
  parameter-only response anchor as generation.
- **Burn-in configurations now require one reproducible week** (breaking):
  `SCMPrior.validate()` requires `T >= l_max` when `adstock_burn_in > 0`.
  `response_warmup_weeks` and the oracle likelihood tail are kernel-aware:
  they use `l_max - 1` only with burn-in and an eligible direct nonidentity
  adstock kernel; otherwise they use the full reported window. Summary calls
  without `adstock_family` conservatively report `l_max - 1` with burn-in.
- **Oracle likelihood uses only a kernel-required reproducible tail**
  (breaking): with burn-in and an eligible direct nonidentity adstock kernel,
  it observes `sales[l_max - 1:]`; identity-only direct paths retain the full
  likelihood window, while full-length deterministics remain available.
- **Zero confounding is a true no-op** (breaking): setting
  `confounding_strength_range=(0, 0)` now matches disabled confounding at a
  fixed seed. Worlds generated with that configuration change because the
  previously unnecessary RNG dependency is gone.
- **`beta_additive_range` requires a strictly positive upper bound** (breaking),
  preventing direct graph edges with identically zero contribution targets.
- **Removed `rw_channel_std_sigma`** (breaking): the configuration knob was
  unreachable whenever the supported channel-walk range was present.
- **Small corpus requests preserve cell splits** (breaking):
  `sample_prior_predictive(n=1)` and `DataGenerator` requests with
  `n_tasks=1` now raise; `generate_batches` also rejects `batch_size=1`. For
  `2 <= n <= draws_per_cell`, the effective grid uses
  `max(1, n // 2)` draws per cell, observable through `cell_id` and
  `diagnostics["draws_per_cell"]`; a one-world batch remainder is folded into
  the preceding batch.
- **Removed dead public and internal code** (breaking): unused `slots.py`
  symbols, the numpy random-walk API, and `SCM_OUT_NAMES` no longer ship;
  public `sample_g` is now private `_sample_g`. The duplicate saturation
  if/elif dispatch was removed in favor of `SATURATION_FAMILIES`, the single
  name-to-wrapper mapping used by `symbolic_graph._saturate_col`; unused-local
  rule `F841` is enabled again.

### Fixed
- **Rendered random-walk equations match execution**: `SCM.equations["RW"]`
  now names the fixed `centred_walk_scale(T_full, width)` divisor and the
  world's concrete scale values instead of publishing the removed
  realized-standard-deviation normalization.
- **Decomposition diagnostics use persisted precision**: all four
  decomposition-error diagnostics are computed from the retained float32
  corpus arrays, rather than pre-storage draw values.
- **Edge configuration is auditable and validated**: diagnostics
  `edge_base_rates` now honors `edge_rate_overrides`, and corpus validation
  requires `diagnostics["edge_types"]` to match the canonical layout.
- **Signal reports distinguish unavailable metrics**: `describe_scm` renders
  `n/a` rather than a numeric value when a signal metric is invalid.
- **Weibull degeneracy is consistent across symbolic and numpy paths**:
  non-finite or zero min-max spans/totals now produce a zero kernel instead of
  symbolic NaNs.
- **Oracle rejects empty likelihoods**: `build_oracle_model` raises when its
  active-kernel warmup would consume every reported observation.
- **Output and replay audits fail safely**: named graph-output collisions now
  raise; accepted `SCM.data` arrays own their storage; and concrete shock
  replay rejects overlapping windows.
- **Zero-denominator feature normalization is exact**: zero-mean spend and
  zero active-spend totals now normalize to zero instead of relying on `1e-8`
  offsets.

- **Response anchors are parameter-only on every path**: audit-only unshocked
  channel and sales outputs, as well as the oracle, now use the same pinned
  per-channel anchor as generated responses rather than any realized-series
  statistic.
- **Oracle comparison semantics are explicit**: oracle `baseline` is `B`
  without `RW_Y`, while persisted generator baseline is `B + RW_Y`; because
  `RW_Y` is not stored separately, baseline is not directly comparable.
  Oracle `contributions` exactly matches `contributions_observed`; compare
  `sales_mu` with observed sales and treat baseline recovery as one-walk-limited.

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
