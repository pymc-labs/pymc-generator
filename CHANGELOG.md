# Changelog

Notable user-facing changes, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During the alpha series, releases may change public APIs and persistence contracts;
read the migration notes before upgrading.

## [Unreleased]

### Added

- Native conda packaging initially used exact-commit pymc-marketing/pymc-extras
  companions, matching core numerical versions, isolated build tooling, and no
  automatic uploads. Package tests exercise generation, persistence, and the
  installed CLI. Historical evidence for that pre-compatibility stack on macOS
  arm64: all 353 preservation arrays matched the original uv baseline and an
  environment recreated from an explicit 181-package conda lock. This does not
  establish numerical equivalence for the released stack below.
- Tag releases reuse CI's locked verification and native package build, then
  prepare draft GitHub Releases with wheel, source archive, conda channel, and
  SHA-256 checksums. PyPI/TestPyPI publishing and its unused permissions are removed.
- Source archives now include the complete test suite and developer/documentation
  support files. CI exercises the unpacked archive independently of the checkout.
- Package metadata uses an SPDX MIT license expression and retains both copyright
  notices. Runtime and distribution versions share `_version.py`; uv refreshes
  editable metadata when that file changes.
- Distributions ship inline typing information, with typed lazy public exports
  and no module-wide type-error exemptions. Corpus and graph mappings describe
  their heterogeneous values; generation remains numerically unchanged.
- Contribution, conduct, security-reporting, and release-review policies now
  document maintainer responsibilities, scientific evidence requirements, and
  a reporting fallback when private GitHub vulnerability reporting is unavailable.
- Pin workflow actions, remove retained checkout credentials, and add free
  full-history Gitleaks checks plus public-only CodeQL analysis. Repository
  protections require independent review and named checks, with read-only
  workflow defaults and reviewable dependency/action security maintenance.
- Outcome distributions and generated-data diagnostics, with selectable quantities,
  pooled or per-world summaries, explicit validity masks, and optional signal gates.
- Auditable world equations, realized parameters, and exogenous innovations;
  public `build_world_model`, `sample_structure`, `sample_prior_cond`, and `draw_worlds`.
- A plug-in posterior oracle with sampled or analytically marginalized Gaussian
  outcome-side latents. It is a reference model, not a universal recovery bound.
- Configurable intercept/non-media floors, control shocks and centered pulses,
  direct-null channel floors, baseline/channel innovation confounding, and held-spend
  interventions that retain ordinary adstock carryover.
- Optional per-cell narrowed prior intervals and persisted conditioning metadata.
- Dense per-direct-channel signal features, response-support diagnostics, and
  versioned identifiability metadata.
- Complete bundle recipes containing effective priors, seeds, connectivity,
  environment versions, and replay instructions.

### Changed — migration notes

- **Released stack / Python floor (0.0.2 candidate):** require Python 3.13+ and
  test Python 3.13/3.14 in CI. Replace Git development dependencies with registry
  releases: PyMC 6.2.0, pymc-marketing 1.1.0, pymc-extras 0.14.0, PyTensor 3.2.4,
  and PreliZ 0.27.1 in the lock. The native channel now contains only this
  project, with dependencies from conda-forge. On macOS arm64, all 353 arrays
  across eight preservation cases matched the original baseline exactly on both
  Python 3.13.9 and 3.14.0. Generation code, seeds, sampler settings, and
  numerical tolerances were unchanged. This measured result is not a general
  numerical-equivalence guarantee across versions or platforms. The 3.13 floor
  is a deliberate project policy, not an upstream constraint: the pinned
  modeling stack still supports 3.12.
- **Declared `xarray` explicitly:** `mechanisms` imports `pytensor.xtensor`,
  whose type module imports xarray at module level. xarray is not a core
  PyTensor requirement, so it is now a direct dependency instead of relying on
  pymc-marketing to supply it transitively.
- **Conda run constraints mirror `pyproject.toml`:** the recipe no longer pins
  runtime dependencies with exact `==` versions or re-declares transitive
  packages. Exact pins made the package uninstallable beside other conda-forge
  content; reproduction of the validated stack belongs to `uv.lock` and the
  explicit environment export.
- **Oracle sampler eligibility:** the released stack supports gradients through
  Weibull carryover, removing the previous forced Metropolis restriction. Update
  the oracle regression test to require finite, nontrivial gradients for all
  three adstock families. Automatic sampler selection and posterior draws can
  change; NUTS eligibility is not a convergence guarantee.
- **Project rename:** `pymc-generator` replaces `prior-generator`; update imports
  from `prior_generator` to `pymc_generator` and invoke the `pymc-generator` CLI.
  Reinstall from the renamed repository or native conda artifacts and select the
  `pymc-generator` Jupyter kernel. No old import or CLI aliases are provided;
  numerical behavior and persistence schemas are unchanged.
- **Corpus schema v3:** readers migrate recognized v1/v2 metadata without changing
  numerical arrays or packed edge positions. Outcome edges are `dy`/`zy`, not
  `db`/`zb`, throughout graph keys, loadings, configuration, audits, and examples.
  The canonical edge order is `cy, dc, dz, dy, zy, zc, cc, zz`.
- Public dimension counts use descriptive names: `n_tasks`, `n_time_steps`,
  `n_treatments`, `n_covariates`, and `n_latent`. Legacy `K_active`, `M_active`,
  `J_active`, `active_c_mask`, `active_m_mask`, and `active_j_mask` are migrated to
  their descriptive active-count and active-mask equivalents on load.
- `min_dead_channels` is now `min_no_direct_effect_channels`; channels without a
  direct sales edge may still have indirect effects. `rw_mean_range` is now
  `rw_control_mean_range`, since it controls the control drive, not latent demand.
- Saturation wrappers accept `reference_level`, not `mean_x`; internal
  `saturation_scale` replaces `mean_ad`. The anchor is parameter-only, not an
  expected or realized channel mean. It is required by the oracle and shared by
  generated, intervention, and audit paths.
- `DataGenerator.iter_batches` replaces eager `generate_batches`. It uses the
  configuration seed unless overridden and releases earlier batches as iteration
  advances. Requests require at least two tasks; a remainder of one joins the
  preceding batch to preserve cell splits.
- `world_model_template` replaces `world_model_batched`. This remains an
  experimental compilation-reuse path, not a production batching backend.
- Remove the single-choice `texture` argument, unreachable `rw_channel_std_sigma`,
  unused explicit-input draw wrapper, unused NumPy walk API, and obsolete symbols.
  Mechanism probabilities use named dictionaries; `SlotLayout` supports only the
  canonical eight-block layout. There are no compatibility aliases for removed APIs.
- Replace the partial `true_contribution.csv` bundle export with the complete
  `true_components.csv`, including `sales_noise`. Nonempty destinations are refused.
- The pre-review modeling changes use a parentless intercept, iid scale-aware sales
  noise, relative channel-walk amplitudes, fixed walk normalization, mean-zero/unit-scale
  latent demand, and one structural amplitude per channel. These changes can alter
  draws relative to the initial extraction; they are not additional retuning during
  release hardening.
- Kernel and noise metadata describe the current numerical semantics. Incompatible
  semantic labels are rejected rather than silently reinterpreted by schema migration.
  Burn-in/query validation and oracle likelihood windows use admitted response support;
  realized-support diagnostics describe the drawn kernels instead.
- The indirect source split is an ordered attribution convention (`cc, zc, dc`),
  not three order-invariant causal estimands. Non-media flooring reports clipped
  increments in the configured accumulation order.
- Runtime timing stays under `diagnostics["timing"]` and is excluded from saved
  shards. Configuration ranges must be representable in float32.
- Notebook files are the canonical example sources; remove duplicate builders,
  committed outputs, and execution metadata. Full documentation builds execute them.

### Fixed

- Initial release hardening used locked uv environments for development, CI,
  documentation, and build tooling, with a tested Python 3.13 default and a
  3.12/3.13 CI matrix. At that stage all 153 previously locked dependency versions
  and sources were unchanged; the compatibility upgrade above supersedes that
  stack and matrix. Installed wheels are exercised through real generation and
  persistence.
- Preserve the original 2024 Carlos Trujillo copyright from the upstream
  extraction alongside the 2026 PyMC Labs notice.
- Enforce persistence versions before legacy migration. Reject partial legacy
  vocabularies, malformed scalar metadata, unsupported versions, inconsistent
  layouts, and every nonfinite or nonnumeric nested identifiability array.
- Sampled worlds own their configuration and data snapshots. Exported DAGs attach
  demand/control outcome edges to sales, matching the executed equations and audits.
- Normalize selectors without lossy coercion; reject ambiguous masks and invalid
  shapes, preserve scalar unit axes, and validate quantile levels consistently,
  including cached reports. Zero-sales shares remain undefined rather than zero.
- Support arbitrary plot palette sizes and zero-mean series without division by
  zero or artificial contribution scaling. Render unavailable signal metrics explicitly.
- Include observation noise in printed and exported decomposition identities.
  Separate the structural intercept from the full non-media baseline in documentation.
- Preserve same-seed numerical output while preallocating corpus storage and reusing
  walk operators, diagnostic calculations, and requested outcome arrays. The
  historical pre-compatibility release-hardening baseline covers **353 arrays
  across eight configurations**, all unchanged at that stage.
- Keep disabled pulses/confounding and nonbinding floors RNG-inert. Account for
  retried evaluations separately from rejected worlds, reject empty oracle likelihoods,
  and validate replay collisions, overlapping shocks, and feasible scenario connectivity.
- Use unit-invariant ratios, backend-stable Weibull degeneracy handling, and persisted
  float32 values for decomposition diagnostics. Numerical test bounds now cover
  subnormal storage rounding without underflowing to zero.
- Benchmark compilation paths on shared concrete workloads and seeds; remove
  unsupported equivalence and extrapolated production-speed claims.
- Replace opaque digests, sleeps, fixture-plumbing assertions, and prose snapshots
  with observable replay, isolation, storage, and numerical-boundary contracts.
- Make installed-package examples executable. Correct walk-rank, anchor, oracle,
  and identification claims; retain warnings and unrounded convergence diagnostics.
  The MMM case study reports actual divergences, coverage, and error, including null
  channels, rather than asserting a predetermined successful recovery.
- Consolidate duplicated README/schema prose and remove completed implementation
  plans and obsolete phase references from maintained sources.

## [0.0.1] - 2026-07-14

### Added

- Initial extraction from `pymc-labs/structural-pfn` at commit `d5fd09f`.
- Additive SCM generation through `SCMPrior`, `make_scm_prior`, and a drawable
  PyMC world model using pymc-marketing response transforms.
- Corpus generation and compressed persistence, a `DataGenerator` facade,
  signal diagnostics, and intervention-based decomposition targets.
- Single-world inspection, descriptions, CSV/figure bundles, five audit scenarios,
  and the `prior-generator` command-line interface.

### Excluded

- Deprecated upstream generation rungs, legacy texture, obsolete constructors,
  and modules without consumers were deliberately not ported.
