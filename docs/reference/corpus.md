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
symbolic dimension keys. `load_corpus` renames them on read and stamps the
version, so `.npz` files written before the rename stay loadable unchanged;
`save_corpus` raises `ValueError` when handed a corpus that still carries v1
keys, so new shards can only contain the canonical names. The rename map is
`prior_generator.slots.LEGACY_CORPUS_KEYS_V1`:

| v1 key | v2 key |
| --- | --- |
| `K_active` | `n_treatments_active` |
| `M_active` | `n_covariates_active` |
| `J_active` | `n_latent_active` |
| `active_c_mask` | `treatment_active_mask` |
| `active_m_mask` | `covariate_active_mask` |
| `active_j_mask` | `latent_active_mask` |
