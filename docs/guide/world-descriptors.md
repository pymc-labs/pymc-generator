# Comparing worlds

[`data_diagnostics`](data-diagnostics.md) looks *inside* a set of worlds. The
questions here are *between* worlds: what kinds of worlds did a prior produce,
which of them resemble each other, which regions are thinly populated, and
where does another collection — a second prior, or observed datasets — sit
relative to the first.

```text
corpus | [SCM, ...] | ObservedWorlds
                │
        world_descriptors()          one row of named statistics per world
                │
         WorldDescriptors
          ├── summarize_descriptors()   distributions, optionally per stratum
          ├── bin_counts()              worlds per explicit region
          ├── nearest_worlds()          most similar worlds, and how far
          └── compare_descriptors()     one collection against a reference
```

The four analyses are independent: call any of them, in any order, on any
descriptor table. None generates data or changes a prior; improving a prior
stays an explicit loop you run — change a setting, generate, describe, compare.

## Describe two collections

Collection names are arbitrary labels. Here two priors differ only in the range
of covariate levels, which moves how level-dominated the covariates are.

```python exec="1" source="block" result="text" session="descriptors"
import pymc_generator as pg

common = dict(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=40,
              n_cells=4, draws_per_cell=3)
narrow = pg.sample_prior_predictive(pg.make_scm_prior(**common, seed=1))
wide = pg.sample_prior_predictive(
    pg.make_scm_prior(**common, seed=2, rw_covariate_mean_range=(-20.0, 20.0))
)

a = pg.world_descriptors(narrow, source_id="narrow", lags=(1,))
b = pg.world_descriptors(wide, source_id="wide", lags=(1,))
print(a.table(["covariate_level_to_variation_median", "treatment_cv_median",
               "treatment_acf_lag_1_median", "treatment_pair_diff_abs_pearson_max"]))
```

Each statistic is computed inside one world over its active series and then
reduced across the active nodes of a role; the name says how
(`covariate_level_to_variation_median` is the median over active covariates of
`|mean| / std`). Status counts keep an infinite value (a constant series), an
undefined value (`0 / 0`) and an ineligible one (no active covariates) apart.

## Distributions

```python exec="1" source="block" result="text" session="descriptors"
both = pg.WorldDescriptors.concat([a, b])
summary = pg.summarize_descriptors(
    both, features=["covariate_level_to_variation_median"], by=("source_id",)
)
print(summary.table("covariate_level_to_variation_median"))
```

`by=` splits the summary into strata — here by collection; active counts
(`n_treatments_active`, `n_covariates_active`) and `n_time_steps` work the same
way, which is how you check a richer prior kept the same count mix.

## Regions

```python exec="1" source="block" result="text" session="descriptors"
edges = {"covariate_level_to_variation_median": [0, 0.5, 1, 3, 10, 100]}
print(pg.bin_counts(both, edges, by=("source_id",)).table())
```

Bins are explicit and reusable, so the same regions can be counted in any
collection. Values outside the edges, undefined and ineligible values are
reported separately instead of being folded into the end bins, and each cell
also counts distinct generation groups: ten worlds from one cell are one
structure seen ten times.

## Similarity

```python exec="1" source="block" result="text" session="descriptors"
features = ("covariate_level_to_variation_median", "treatment_cv_median",
            "treatment_acf_lag_1_median")
within = pg.nearest_worlds(a, features=features, exclude_same_group=True)
print(within.table())
```

Without a query, every world is matched against the others in its own
collection; `exclude_same_group=True` asks for the nearest world from a
*different* generation cell. The scale is fitted on the reference and reused, so
distances from different queries are on one ruler:

```python exec="1" source="block" result="text" session="descriptors"
across = pg.nearest_worlds(a, b, features=features, scale=within.scale)
print(across.table())
```

## One collection against another

```python exec="1" source="block" result="text" session="descriptors"
comparison = pg.compare_descriptors(a, b, features=features)
print(comparison.table("covariate_level_to_variation_median"))
```

Each query world gets a midrank percentile among the reference worlds of its
stratum; per stratum you also get the median shift, the fraction of query
values outside the reference range and a descriptive KS distance. Swapping
`reference` and `query` changes these directional numbers, whatever the
collections are called.

## Observed datasets

```python
observed = pg.ObservedWorlds(
    treatment_raw=treatments[None],   # (1, n_time_steps, n_treatments)
    covariates=controls[None],        # (1, n_time_steps, n_covariates)
    outcome_raw=outcome[None],        # (1, n_time_steps)
)
target = pg.world_descriptors(observed, source_id="observed", lags=(1,))
pg.compare_descriptors(a, target, features=features).query_frame()
```

Observed data yield the same observable features as a corpus with the same
series; generating-process features (`include_truth=True`) are refused rather
than invented.

## Joining your own measurements

`to_frame()` returns one row per world with `source_id`, `world_id` and
`group_id`. Anything measured elsewhere per world — a model's error on that
world, say — joins on those columns, and the joined frame can be summarized
per region with `bin_counts(...).locate(rows)`. The descriptor layer itself
knows nothing about models or training.

See the [API reference](../reference/descriptors.md) for exact definitions,
status rules and the distance contract.
