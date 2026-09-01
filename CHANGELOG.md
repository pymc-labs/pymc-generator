# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on 0.x, minor versions may contain breaking changes.

## [Unreleased]

### Added

- **Outcome-space distributions** (`outcome_distributions`). Every existing
  diagnostic describes a corpus in PARAMETER space (edge marginals, drawn
  coefficients) or SIGNAL space (per-channel CV, Spearman, warmup ratio).
  Neither answers the magnitude question: how large are the outcomes, and how
  large are the pieces that add up to them? `outcome_distributions(corpus)` —
  or a list of `SCM` worlds — pools every world along the QUANTITY axis and
  returns, per quantity (sales, baseline, per-node baseline components,
  per-channel contribution, indirect effects by source, total media, spend,
  controls, latent demand): every drawn value (`series` / `.values`), per-unit
  stats over time (`unit_mean/std/min/max/total`, whose spread IS the
  across-world spread), and `unit_share = Σ_t value / Σ_t sales`. A unit is one
  world for a scalar quantity, one `(world, column)` pair for a column
  quantity; padded inactive columns are dropped, and structurally-null channels
  are kept as exact zeros and reported via `zero_unit_fraction` instead of
  being silently filtered. Because the decomposition is exact, the additive
  quantities' shares are a real budget: `additive_share_total()` is 1.0 per
  world. Conditioning stays a row mask (`worlds=corpus["cell_id"] == 3`), unit
  filtering is `QuantityDistribution.select`, and `normalize="sales_scale"` /
  `"sales_mean"` makes pooled magnitudes comparable across worlds without
  touching shares. Reports: `table(of=...)` (text), `summary()` (JSON-ready),
  `to_frame()` (long-form pandas), and `viz.plot_outcome_distributions` (a
  histogram grid, constant panels skipped). Purely an analysis API — nothing
  in the persisted corpus or its `diagnostics` changed.

- **The intercept is its own node, and can be floored** (`baseline_floor`).
  `B` no longer absorbs its parents: latent demand and the controls attach
  DIRECTLY to `Y`, so the sales equation is literally the one a standard MMM
  assumes — `B + Σ δ_j D_j + Σ ρ_m Z_m + Σ g_cy β_k f_k(C_k) + RW_Y` — and the
  intercept is a level that can be reported on its own. With no floor this is
  pure associativity, but it is what makes the floor safe: `baseline_floor`
  censors ONE additive term (`B = max(RW_B, floor)`, a censored walk, so `B`
  may sit exactly AT the floor), leaving `control_contribution[:, m]` exactly
  `g_zb[m]·ρ[m]·Z[:, m]` and the decomposition exact. Flooring a sum that still
  contained the parents would break both. `None` (default) keeps the signed
  walk; the floor adds no RV and consumes no RNG, so a floor that never binds
  reproduces the unfloored corpus byte for byte.

  `baseline_floor_scope` (`"intercept"` default, or `"non_media"`) selects WHAT
  the floor clips. A floored intercept alone does not stop a large negative
  `ρ·Z` from dragging the non-media total under (measured: 112 negative weeks on
  a stress fixture). `"non_media"` clips the running total as each parent joins,
  in the locked order intercept → confounders → controls, and each per-node
  column becomes the telescoping difference that node caused — the same
  construction `indirect_effects_by_source` uses for channels. The non-media
  total is then `>= floor` by construction (same fixture: 0 negative weeks, 111
  weeks exactly at the floor, identity error 1.8e-15), the columns still sum
  exactly, and a non-binding floor leaves the persisted corpus byte-identical in
  either scope. The cost: a column is no longer linear in its node where the
  floor binds. Nodes with no edge are skipped rather than added with a zero
  coefficient, so they neither join the clipping order nor change which
  innovations the compiled graph reaches.

  Even so, sales cannot be *strictly* guaranteed non-negative: the additive
  observation noise is symmetric and unbounded, so `P(Y<0)>0` holds for any
  additive-Gaussian outcome. The absorbing scope makes every term of the sales
  MEAN non-negative and leaves the residual to the acceptance filter; measured
  headroom is 88 observation-noise sigma at the worst week over 12 worlds.

  Motivation: the intercept walk is signed, so the baseline could dip below
  zero — measured 0.08% of weeks under the shipped relative-mode prior, and
  19/50 tasks in a low-mean absolute-mode recipe. **Sales is deliberately NOT
  censored**: a clamp on `Y` censors the *observation*, which would put every
  world outside the additive-Gaussian class an MMM likelihood — including this
  package's own oracle — can represent, and measurement shows sales already
  sits 28–83 observation-noise σ above zero. Non-negative sales stays enforced
  exactly by the acceptance filter.

  `baseline_intrinsic` now carries the intercept ALONE (previously
  `RW_B + RW_Y`) and the iid observation noise becomes its own `sales_noise`
  corpus column, so the persisted identity is `baseline_intrinsic + sales_noise
  + Σ control + Σ confounder + Σ contributions + Σ indirect_by_source ==
  sales`. The oracle's analytic `latent="marginal"` mode raises for a floored
  config (a censored walk is not Gaussian); `latent="sampled"` applies the
  identical clip and stays exactly the generative model.

  BREAKING: the new `sales_noise` column and the narrowed `baseline_intrinsic`
  change the corpus schema, and because `baseline_intrinsic` no longer touches
  `eps_y` the graph's RNG traversal is re-partitioned — every non-structural
  array moves at a fixed seed. Structure/mask/split arrays and
  `adstock_family` are unchanged.

- **Control texture** (`control_hf_sigma_range`, `control_pulse_prob_range`,
  `control_pulse_amp_range`): controls now carry high-frequency own drive —
  iid weekly noise plus a **centred** calendar pulse — instead of being smooth
  random walks only. A smooth-walk control lives in the same function space as
  the smooth baseline walk `RW_B`, so the `Z→B` loading `rho_zb` trades off
  against baseline drift and is only weakly identified; the added
  high-frequency content is what a smooth baseline cannot mimic (and what real
  promo / holiday / price-step regressors look like). For control `m` with
  `s = rw_z_std[m]`, the own drive gains
  `q*s * eps_z_hf[:, m] + r*s * (eps_z_pulse[:, m] - p)` with
  `eps_z_pulse ~ Bernoulli(p)`; both magnitudes are drawn RELATIVE to the
  control's own walk std (a signed control has no positive level anchor), and
  the pulse is centred so `E[Z_m]` — hence every parameter-only saturation
  anchor — is unchanged. Measured on 18 controls over six worlds at
  `n_time_steps=78`, R² against a 5-term smooth cosine basis drops from a
  median 0.94 (0.54–0.995) to 0.64 (0.22–0.93): ~6.5x more of the variation
  that identifies `rho_zb`. New reports `param_control_hf_sigma` /
  `param_control_pulse_amp` / `param_control_pulse_prob`, new innovations
  `eps_z_hf` / `eps_z_pulse` in `SCM.exogenous`, per-control `texture` records
  in `SCM.equation_parameters`, and the executed term in `SCM.equations`.
  `make_scm_prior(texture="diverse")` enables it, which **changes generated
  worlds at a fixed seed**: the new draws re-partition the seeded RNG list
  (`reseed_rngs` assigns streams by the compiled graph's traversal order), so
  every non-structural array moves — not only the controls. Provably unchanged:
  the structure/mask/split arrays (`g`, `*_active_mask`, `support_mask`,
  `is_val`, `cell_id`), `adstock_family`, and `channel_level`. A raw
  `SCMPrior` — or the preset with all three ranges set to `(0.0, 0.0)` — is
  inert: no graph term, the magnitudes degenerate to constants (so no
  parameter RV and no RNG consumed; the two noise RVs are created but reach no
  output), and corpora stay byte-identical to the pre-feature format (pinned by
  `EXPECTED_UNTEXTURED_CONTROL_HASHES`).

- **Dead-channel floor** (`min_dead_channels`): the minimum number of *active*
  channels a cell must leave with no direct `C→Y` arrow — spend observed, true
  contribution exactly zero, i.e. the negative class for any direct-effect
  signal. `edge_budget["cy"]` could not express it: a budget is an absolute
  arrow count clamped to the active channels, so a cell drawing 2 active
  channels under `cy=(2, 10)` has both live (measured: 18 of 40 cells under a
  `(2, 10)`-active / `(2, 10)`-cy recipe carried no dead channel), which forced
  corpora to be partitioned by active count to keep both classes. The floor caps
  the per-cell live count at `max(1, K_active − min_dead_channels)` on both the
  budget and the Bernoulli path, keeps the degenerate `cy ≥ 1` guard, still
  scatters live channels over *all* active slots so slot index stays
  uninformative, rejects a floor the smallest drawable cell could not honour
  (`min_dead_channels < n_treatments_active_range[0]`), and echoes the resolved
  value in `diagnostics["min_dead_channels"]`. One config with
  `n_treatments_active_range=(2, 10)`, `cy=(1, 10)` and `min_dead_channels=1`
  now spans 1–9 live channels and always carries a dead one. `0` (the default)
  is inert: identical draws, no extra RNG.
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
  multivariate normals; `latent="sampled"` keeps latent demand/baseline walks
  exact and represents `RW_Y` as iid Normal with the same outcome-scale prior;
  all concessions are documented in the docstring and the new
  [posterior-oracle guide](docs/guide/oracle.md).

- Documentation site (`docs/`, MkDocs Material) at
  <https://pymc-labs.github.io/prior-generator/>: docstring-driven API reference
  (mkdocstrings), guide pages whose figures are produced by executing real code
  at build time (markdown-exec), and a runnable examples notebook executed by
  mkdocs-jupyter. Added a `docs` extra, a GitHub Pages workflow
  (`.github/workflows/docs.yml`), and build instructions in `CONTRIBUTING.md`.

### Changed
- **Descriptive dimension names everywhere** (breaking, plan doc 04): the
  symbolic dimension vocabulary is gone from identifiers, public signatures,
  persisted corpus keys, diagnostics, docstrings, docs, and the example
  notebooks. `N → n_tasks`, `T → n_time_steps`, `K → n_treatments`,
  `M → n_covariates`, `J → n_latent`, `K_max/M_max/J_max →
  n_treatments_max/n_covariates_max/n_latent_max`, the shock count `S →
  n_shocks`, the reported-window slice `W → window`, `T_full →
  n_time_steps_full`, and `T_DEMO/K_DEMO/M_DEMO/J_DEMO →
  N_TIME_STEPS_DEMO/N_TREATMENTS_DEMO/N_COVARIATES_DEMO/N_LATENT_DEMO`.
  Renamed public surface: `SCMPrior.T → SCMPrior.n_time_steps`,
  `SlotLayout.K/.M/.J → .n_treatments/.n_covariates/.n_latent`,
  `SCM.T/.K/.M/.J → .n_time_steps/.n_treatments/.n_covariates/.n_latent`,
  `Scenario.prior(T=...) → Scenario.prior(n_time_steps=...)`, and the
  `build_world_model` / `build_oracle_model` / `write_scenario_bundles`
  horizon parameter. Shape documentation now reads
  `(n_tasks, n_time_steps, n_treatments)` instead of `(N, T, K)`.
  Deliberately unchanged: the edge-type axis (`cy, dc, dz, db, zb, zc, cc, zz`,
  `g_cy`…`g_zz`, the packed `g` key, `edge_budget` keys), node and equation
  symbols (`C_k`, `Z_m`, `D_j`, `B`, `Y`, `RW_*`), loop indices, and `l_max`.
  Generation is byte-identical: the corpus hash contract in
  `tests/test_identifiability.py` reproduces every pre-rename digest for the 35
  untouched keys, and the 6 renamed keys reproduce theirs when hashed under
  their old names (that test seeds the digest with the key name).
- **Persisted corpus schema is versioned at 2** (breaking): the corpus keys
  `K_active`, `M_active`, `J_active`, `active_c_mask`, `active_m_mask`, and
  `active_j_mask` are now `n_treatments_active`, `n_covariates_active`,
  `n_latent_active`, `treatment_active_mask`, `covariate_active_mask`, and
  `latent_active_mask`. New corpora carry
  `diagnostics["schema_version"] == CORPUS_SCHEMA_VERSION` (`2`);
  `load_corpus` migrates a v1 `.npz` on read via
  `slots.LEGACY_CORPUS_KEYS_V1` (raising if a shard mixes both vocabularies)
  and stamps the version, so previously saved shards stay loadable;
  `save_corpus` refuses to write v1 keys, so a new shard can only carry
  canonical names.
- **Outcome-side noise is parameter-scale-aware and non-aliased** (breaking):
  `RW_Y` is now iid observation noise, while `RW_B` remains the sole latent
  baseline walk; this removes the exact `RW_B`/`RW_Y` same-width variance ridge
  (collision probability `0.046` → `0`). Default
  `outcome_std_mode="relative"` draws dimensionless
  `rw_baseline_std_range=(0.000, 0.093)` and
  `rw_sales_std_range=(0.010, 0.028)` scales, then multiplies both by the
  parameter-only media anchor `sqrt(sum((g_cy * beta)**2))`; the legacy
  HalfNormal scales remain available with `outcome_std_mode="absolute"`. In the
  prescribed 150-world calibration, residual/media ratio changed from `3.9` to
  `p5=0.160`, `p50=0.470`, and `p95=1.482`; `RW_Y` carries a median `0.112`
  of outcome-noise variance. A separate 120-world fixed-seed audit has median
  ratio `0.732` and changes `shapes_fitted` median error from `35.7%` to
  `14.6%` at rank 10 and from `29.2%` to `17.2%` at rank 20. The rank-40
  `31.0%` result is expected overfitting by an unpenalized 43-column nuisance
  basis, not a new data-generating defect. The ratio is now centered in the
  real weekly-MMM band and difficulty is a declared axis rather than an
  accidental product of unrelated absolute priors. This changes every generated
  world; corpus diagnostics now carry outcome-noise semantics version `1`, so
  pre-change corpora no longer validate as new semantics.
- **Posterior oracle marginalization**: `build_oracle_model` and
  `SCM.oracle_model()` now integrate the outcome-side Gaussian latents
  analytically by default, using their exact multivariate-normal density rather
  than a sampled latent representation and without latent-walk dimensions. The
  oracle also registers shape parameters only for response families used by the
  world's channels. On the simple fully identified worlds, this changed maximum
  r-hat from 2.2-2.7 to at most 1.01 (unrounded `az.rhat`; `az.summary`'s
  2-significant-figure display shows 1.0) and minimum bulk ESS from 5 to
  480-1500.
  Callers that need posterior `demand` / `baseline` deterministics must request
  `latent="sampled"`; default `sales_mu` is now the latent-marginal posterior
  mean.
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
  absorbed by `rw_b_mean`, so none of the loading magnitudes were recoverable.
  The whole `D -> *` magnitude now lives in the loadings. The graph admits an
  exact sign flip when `eps_d`, `w_dc`, `u_dz`, and `delta_db` are all negated,
  but the supported prior's strictly positive `db_coeff_range`,
  `dc_coeff_range`, and `dz_coeff_range` exclude the reflected parameters, so
  demand's sign is identified. If a user widens any of those ranges to admit
  negatives, only `|D|` and the loading magnitudes are recoverable. Latent
  influence is consequently less dispersed across worlds (the `HalfNormal`
  factor is gone); widen `db_coeff_range` / `dc_coeff_range` / `dz_coeff_range`
  to restore spread — those knobs are now meaningful rather than confounded.
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

- **Published `T_full` walk-scale semantics**: the corpus guide now documents
  `param_rw_*_std` as an expected full-horizon scale under fixed normalization,
  rather than a realized per-path scale.
- **Oracle Weibull sampler downgrade is documented and tested**: a
  Weibull-adstock channel assigns `weibull_lam` and `weibull_k` to Metropolis
  rather than NUTS, and a test pins that sampler selection.

- **Persisted ratios are unit-invariant** (breaking): `spend_norm`,
  `spend_share` and the `spend_cv` / `media_share` diagnostic quantiles divided
  by a denominator carrying a raw-unit `+ 1e-8` (or `+ 1e-12`) epsilon, so a
  pure change of monetary unit changed persisted features — rescaling an
  otherwise identical corpus moved `spend_cv_quantiles.q50` from 0.586 to 0.149
  while the scale-free `spend_cv` signal metric stayed at 0.586, i.e. one
  diagnostics block reported two contradicting CVs. All four now use
  exact-zero-guarded division, so padded and degenerate slots stay exactly zero
  and the ratios are true ratios. `spend_norm` values move by 1.17e-7 relative
  (the last float32 bit of 38 of 96 cells in the frozen contract); every other
  persisted array is byte-identical.

- **Signal-summary/gate contract version 2 → 3** (breaking):
  `SIGNAL_METRIC_LAYOUT` and `METRIC_KEYS` remain the same nine entries in the
  same order. `SIGNAL_METRIC_VERSION` changes because summaries add the
  response-warmup/contemporaneous-weight diagnostics and the gate adds
  `frac_contrib_rel_std_lt_001` and `frac_contrib_r2_gt_095` limits of `0.10`.
  Existing corpora must be regenerated for the new summary and gate contract.
- **`build_oracle_model` requires `saturation_scale`** (breaking): the
  observed-channel mean fallback was removed so every oracle uses the same
  parameter-only response anchor as generation.
- **Query windows cannot overlap non-reproducible burn-in responses**
  (breaking): `SCMPrior.validate()` now rejects a burn-in configuration whose
  query window overlaps the response prefix that depends on unpersisted spend.
  It requires `min(T - n_query, T // 2) >= l_max - 1`, covering both the
  configured short-horizon query split and the `is_future=1` half-series split.
  This replaces — rather than supplements — the prior `T >= l_max` check and
  is strictly stronger. `response_warmup_weeks` and the oracle likelihood tail
  remain kernel-aware: they use `l_max - 1` only with burn-in and an eligible
  direct nonidentity adstock kernel; otherwise they use the full reported
  window. Summary calls without `adstock_family` conservatively report
  `l_max - 1` with burn-in.
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
- **Weibull degeneracy guard is backend-stable**: the analytic
  `raw_weights.max() > 1e-300` floor replaces a compile-mode-dependent check on
  library output and is mirrored in `_adstock_weights`; degenerate weights
  produce a zero kernel rather than symbolic NaNs.
- **Walk-scale and oracle-noise documentation corrected**: published
  reported-window sd / declared-`std` ranges now identify the declared
  `T_full` expected scale rather than full-window realized sd, and the oracle's
  sales-noise heteroscedasticity is correctly described as largest at the
  window edges and smallest mid-window.
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
