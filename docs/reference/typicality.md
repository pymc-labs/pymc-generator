# Dataset typicality

How typical is one observed dataset — treatments, covariates and outcome over
time — of a generated corpus or of a generator configuration? The
[guide](../guide/dataset-typicality.md) explains the method and how to read the
result; the dataset format (`ObservedWorlds.from_frame`, `from_csv`,
`to_frame`) is documented with [`ObservedWorlds`](descriptors.md#pymc_generator.descriptors.ObservedWorlds).

```python
import pymc_generator as pg

dataset = pg.ObservedWorlds.from_csv(
    "sales.csv", treatments=["tv", "search"], covariates=["price"], outcome="sales", time="week"
)
prior = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=dataset.n_time_steps)
matched = pg.matched_prior(prior, dataset, n_cells=200, draws_per_cell=3, seed=11)
corpus = pg.sample_prior_predictive(matched)
result = pg.typicality(
    pg.world_descriptors(corpus, source_id="prior"),
    pg.world_descriptors(dataset, source_id="sales"),
)
print(result.table())
result.profile(0)
```

| Symbol | Purpose |
| --- | --- |
| `typicality` | Split-conformal typicality p-value and descriptor profile of every query row against the reference worlds of its shape. |
| `Typicality` | The result: per-row status, p-value, `p_min`, violations, distance, counts; per-feature value, percentile, reference interval, `inside`, `violation`, role. |
| `matched_prior` | A prior conditioned on a dataset's shape (active treatment and covariate counts), for generating a matched reference. |

::: pymc_generator.dataset_typicality
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - typicality
        - Typicality
        - matched_prior
        - TYPICALITY_STATUSES
        - FEATURE_ROLES
