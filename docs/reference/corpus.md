# Corpus generation

Generate an `n_tasks`-world corpus — a dict of stacked numpy arrays over tasks
and weeks — either functionally with
[`sample_prior_predictive`](#pymc_generator.sampler.sample_prior_predictive) or
through the [`DataGenerator`](#pymc_generator.data_generator.DataGenerator)
facade (which adds batching, schema validation, and save-on-generate).

::: pymc_generator.sampler.sample_prior_predictive

::: pymc_generator.data_generator.DataGenerator

## Active-count coverage

::: pymc_generator.active_counts.active_count_coverage

## Persistence

::: pymc_generator.data_generator.save_corpus

::: pymc_generator.data_generator.load_corpus

## Schema versions and migration

New corpora use integer `diagnostics["schema_version"] = 5`. Saving and loading
require this version. Versionless archives and versions 1–4 are rejected with a
clear schema-version error, as are malformed and future versions. There is no
automatic migration: pre-v5 archives cannot reconstruct realised rich-world
truth. Renaming keys or changing a version stamp is not a migration.

Default draw stability concerns numerical/model arrays, not cross-version
metadata or archive-byte identity: the v5 stamp changes every saved corpus.

Base arrays retain their existing storage dtypes and packed graph positions.
The optional trajectory and mechanism blocks add the realised truth below.
Call `DataGenerator.validate_corpus` after loading for full shape, numerical,
padding, schedule and decomposition validation.

### Realised trajectory fields

When `SCMPrior.trajectory_metadata_enabled` is true, all fields in
`TRAJECTORY_ARRAY_FIELDS` accompany `diagnostics["trajectory"]`. The six existing
flag/activity/level-envelope arrays retain `uint8` / `float32` storage.
Additional keys follow `trajectory_{input}_{component}_{leaf}`, where `input`
is `treatment` or `covariate`:

| Component | Leaves | dtype |
| --- | --- | --- |
| `hf` | `sigma` | `float64` |
| `pulse` | `amp`, `prob` | `float64` |
| `onset` / `offset` | `start` / `stop` | `int64` |
| `flighting` | `period`, `on_weeks`, `phase` | `int64` |
| `level_jump` | `week` | `int64` |
| treatment `level_jump` | `factor`, `log_factor` | `float64` |
| covariate `level_jump` | `size` | `float64` |
| `seasonal` | `amplitude`, `period`, `phase` | `float64` |
| `trend` | `change` | `float64` |

Jump leaves have `(task, input_jump, input)` axes; other leaves have
`(task, input)` axes. `diagnostics["trajectory"]["jump_counts"]` records
role-specific jump sizes, and `parameter_fields` records the canonical field
inventory. Unselected components and inactive inputs have zero padding.
The validator checks event domains and reconstructs the stored gate and
level-envelope arrays from these leaves.
Selected seasonal periods/amplitudes, signed trend changes and covariate jump
sizes retain their existing `CORPUS_STORAGE_MAX` primitive magnitude bound despite
float64 truth storage; the bound does not cap conditionally scaled texture or
derived anchors.

### Realised mechanism fields

Opt-in mechanism priors add `MECHANISM_ARRAY_FIELDS` and
`diagnostics["mechanism_priors"]["parameter_fields"]`: `sat_family` (`uint8`),
`mechanism_saturation_scale`, `beta`, `rho_zy`, `hill_slope`, `hill_kappa_mult`,
`logistic_lam`, `mm_kappa_mult`, `tanh_c`, and `root_alpha` (all `float64`).
Treatment fields have `(task, treatment)` axes; `rho_zy` has `(task, covariate)`.
All active treatment slots retain shape draws, including unused response families.

Enabled reference priors add `{treatment,covariate}_reference_contribution`
and `_reference_input`; treatment references also add
`treatment_reference_response`, the actual response used to derive `beta`.
These are `float64` on their role's task/node axes. Inactive padding is zero.
Reference responses and targets retain the generator's numerical values rather
than recomputing steep response curves from rounded feature arrays.
Treatment coefficient consistency is checked against the target divided by that response,
with bounded float64 rounding and no erasure of nonzero values; a subnormal
coefficient's rounded product need not recover its target to the same precision.
Reference-derived coefficients retain the generator's finite float32-magnitude
limit despite float64 truth storage. A zero coefficient is allowed only for
an original zero target, never because a nonzero target's quotient underflowed.
Default recipes omit the entire mechanism block. Keeping new continuous truth
in float64 preserves supported positive values below float32's nonzero range.

## Timing telemetry is not persisted

`sample_prior_predictive` reports wall-clock telemetry under
`diagnostics["timing"] = {"elapsed_s": float, "tasks_per_sec": float}`. Those
are the only nondeterministic values in a generation, so `save_corpus` writes
the diagnostics block without the `timing` key — on a copy, leaving the
caller's dict untouched — which is what makes two independent same-seed
generations produce byte-identical `.npz` files. `validate_corpus` accepts
diagnostics with or without the block.
