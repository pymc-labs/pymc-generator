# Corpus generation

Generate an `n_tasks`-world corpus — a dict of stacked numpy arrays over tasks
and weeks — either functionally with
[`sample_prior_predictive`](#prior_generator.sampler.sample_prior_predictive) or
through the [`DataGenerator`](#prior_generator.data_generator.DataGenerator)
facade (which adds batching, schema validation, and save-on-generate).

::: prior_generator.sampler.sample_prior_predictive

::: prior_generator.data_generator.DataGenerator

## Persistence

::: prior_generator.data_generator.save_corpus

::: prior_generator.data_generator.load_corpus

## Schema version and the v1 migration

`corpus["diagnostics"]["schema_version"]` records the persisted schema version;
it is `2` for every corpus generated today. Version 1 shards used the old
symbolic dimension keys. `load_corpus` renames them on read **first** and then
requires the version to equal `CORPUS_SCHEMA_VERSION` (`2`) as a non-bool
integer, so `.npz` files written before the rename stay loadable unchanged
while a missing, malformed (`"2"`) or future (`99`) version raises instead of
being half-read. `validate_corpus` reports the same condition as an error
string, and `save_corpus` raises — as it also does when handed a corpus that
still carries v1 keys, so new shards can only contain the canonical names. The
rename map is `prior_generator.slots.LEGACY_CORPUS_KEYS_V1`:

| v1 key | v2 key |
| --- | --- |
| `K_active` | `n_treatments_active` |
| `M_active` | `n_covariates_active` |
| `J_active` | `n_latent_active` |
| `active_c_mask` | `treatment_active_mask` |
| `active_m_mask` | `covariate_active_mask` |
| `active_j_mask` | `latent_active_mask` |

## Timing telemetry is not persisted

`sample_prior_predictive` reports wall-clock telemetry under
`diagnostics["timing"] = {"elapsed_s": float, "tasks_per_sec": float}`. Those
are the only nondeterministic values in a generation, so `save_corpus` writes
the diagnostics block without the `timing` key — on a copy, leaving the
caller's dict untouched — which is what makes two independent same-seed
generations produce byte-identical `.npz` files. `validate_corpus` accepts
diagnostics with or without the block.
