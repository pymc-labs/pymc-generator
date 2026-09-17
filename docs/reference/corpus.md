# Corpus generation

Generate an `n_tasks`-world corpus — a dict of stacked numpy arrays over tasks
and weeks — either functionally with
[`sample_prior_predictive`](#pymc_generator.sampler.sample_prior_predictive) or
through the [`DataGenerator`](#pymc_generator.data_generator.DataGenerator)
facade (which adds batching, schema validation, and save-on-generate).

::: pymc_generator.sampler.sample_prior_predictive

::: pymc_generator.data_generator.DataGenerator

## Persistence

::: pymc_generator.data_generator.save_corpus

::: pymc_generator.data_generator.load_corpus

## Schema versions and migration

New corpora use integer `diagnostics["schema_version"] = 3`. `save_corpus`
accepts only the current version and vocabulary. `load_corpus` supports:

- **v1:** unstamped shards with all six historical dimension keys below.
- **v2:** explicitly stamped shards with the historical edge order
  `cy, dc, dz, db, zb, zc, cc, zz`.
- **v3:** the current vocabulary.

Loading v1/v2 migrates metadata in memory; it does not rewrite the file,
reorder graph slots, cast arrays, or regenerate values. Missing or malformed
versions, future versions, partial/mixed dimension vocabularies, conflicting
edge names, and unrecognized v2 edge orders are rejected.

Call `DataGenerator.validate_corpus` after loading when full shape, numerical,
and decomposition validation is required.

### Historical dimension names

| v1 key | Current key |
| --- | --- |
| `K_active` | `n_treatments_active` |
| `M_active` | `n_covariates_active` |
| `J_active` | `n_latent_active` |
| `active_c_mask` | `treatment_active_mask` |
| `active_m_mask` | `covariate_active_mask` |
| `active_j_mask` | `latent_active_mask` |

### Historical outcome-edge names

The two outcome blocks keep their original positions in packed `g`.
Their names change from `db` to `dy` and `zb` to `zy` in `edge_types`,
`edge_base_rates`, `edge_marginals`, and `edge_budget`. The diagnostic
`min_dead_channels` becomes `min_no_direct_effect_treatments`.

The Python API uses only the new names: `g_dy`, `g_zy`, `delta_dy`, `rho_zy`,
`dy_coeff_range`, and `zy_coeff_range`. Old keyword arguments and graph-dict
aliases are not supported. These edges enter outcome `Y`, not the parentless
intercept `B`.

## Timing telemetry is not persisted

`sample_prior_predictive` reports wall-clock telemetry under
`diagnostics["timing"] = {"elapsed_s": float, "tasks_per_sec": float}`. Those
are the only nondeterministic values in a generation, so `save_corpus` writes
the diagnostics block without the `timing` key — on a copy, leaving the
caller's dict untouched — which is what makes two independent same-seed
generations produce byte-identical `.npz` files. `validate_corpus` accepts
diagnostics with or without the block.
