# Outcome distributions

The other diagnostics describe a corpus in *parameter* space (edge marginals,
drawn coefficients) or *signal* space (per-channel CV, Spearman, warmup ratio).
Neither answers the magnitude question you ask of a prior before trusting it:
**how large are the outcomes, and how large are the pieces that add up to
them?**

`outcome_distributions` pools every world along the **quantity** axis instead
of the parameter axis. It accepts a corpus (from
[`sample_prior_predictive`](corpus.md), `DataGenerator.generate`, or
`load_corpus`) or a list of [`SCM`](sampling.md#prior_generator.worlds.SCM)
worlds sharing one horizon.

```python
import prior_generator as pg

corpus = pg.sample_prior_predictive(cfg)
dist = pg.outcome_distributions(corpus)

print(dist.table())                    # pooled value quantiles
print(dist.table(of="share"))          # the share-of-sales budget
print(dist.table(of="mean"))           # across-world spread of per-world levels

y = dist["sales"].values               # every Y value, all worlds
media = dist["channel_contribution"]   # one unit per (world, channel)
media.quantiles(of="share")            # how big media effects get
```

## Units

A **unit** is one world for a scalar quantity (`sales`, `baseline`,
`baseline_intrinsic`, `sales_noise`, `media_contribution`, `indirect_effects`)
and one `(world, column)` pair for a column quantity (`channel_contribution`,
`control_contribution`, `confounder_contribution`, `indirect_by_source`,
`spend`, `controls`, `demand`).

Padded inactive columns are dropped, so no zero padding contaminates any
statistic. A structurally-null channel — active, spend observed, no `C→Y` edge
— *stays in* as an exact zero and is reported by `zero_unit_fraction` rather
than silently filtered: it is a real outcome of the prior, and it does pull
every contribution statistic toward zero. Filter it explicitly when you want
direct channels only:

```python
direct = media.select(media.unit_max > 0.0)
```

## The share budget is exact

The quantities in `ADDITIVE_QUANTITIES` — `baseline_intrinsic`, `sales_noise`,
`control_contribution`, `confounder_contribution`, `channel_contribution`,
`indirect_by_source` — are exactly the per-node
[decomposition of sales](../guide/decomposition.md). Their `unit_share` values
therefore form a true budget, not a set of loosely related ratios:

```python
assert np.allclose(dist.additive_share_total(), 1.0)
```

## Conditioning and scale

Conditioning is a world row mask, not a separate API:

```python
pg.outcome_distributions(corpus, worlds=corpus["cell_id"] == 3)
pg.outcome_distributions(corpus, worlds=corpus["n_treatments_active"] > 4)
```

Worlds have arbitrary sales levels, so pooled *raw* values mix scales.
`normalize="sales_scale"` divides every Y-scale quantity by the world's own
`sales_scale`; `normalize="sales_mean"` divides by its mean sales. Exogenous
inputs (`spend`, `controls`, `demand`) are not in sales units and are never
rescaled. Shares are ratios and never move.

`keep_series=False` drops the raw values (roughly corpus-sized) while keeping
per-unit stats and pooled quantiles — use it for very large corpora.

## Reports

`summary()` is JSON-serializable, `to_frame()` is a long-form per-unit pandas
table, and [`viz.plot_outcome_distributions`](viz.md) renders the histogram
grid (`of="value" | "mean" | "share"`).

::: prior_generator.outcomes
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - outcome_distributions
        - OutcomeDistributions
        - QuantityDistribution
        - OUTCOME_QUANTITIES
        - ADDITIVE_QUANTITIES
        - DEFAULT_QUANTILES
