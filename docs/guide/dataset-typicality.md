# How typical is a dataset?

[Comparing worlds](world-descriptors.md) describes collections. This page asks
one inferential question about **one observed dataset** — its treatments,
covariates and outcome over time:

> Could this dataset be a world of this corpus, or of this generator, and if it
> stands out, in what way?

For descriptive statistical comparisons rather than a calibrated inference
question, see [existing-corpus comparisons](corpus-comparison.md). Cropping or
projecting a corpus does not automatically preserve this page's guarantee.

```text
dataset    CSV | DataFrame
  └ ObservedWorlds.from_csv / from_frame
      └ world_descriptors ──────────► query rows ───────┐
reference  a corpus, or an SCMPrior through
           matched_prior → sample_prior_predictive      │
  └ world_descriptors ──────────────► reference rows ───┤
                                                        ▼
             typicality(reference rows, query rows)
                      → p-value + profile
```

The answer has two parts:

* a **typicality p-value**: the share of calibration worlds (reference worlds of
  the dataset's shape, one per calibration cell) that are at least as unusual as
  the dataset, `p = (1 + #at least as unusual) / (n_calibration + 1)`;
* a **profile**: for every descriptor, the dataset's value, its percentile among
  the reference worlds, the reference's central interval, and whether the value
  lies inside it.

## The dataset format

A dataset is one table: **one row per time step**, rows in time order and
equally spaced, **one column per series**. You say which columns are
treatments, which are covariates and which one is the outcome; all three roles
are required, and other columns are ignored.

```python exec="1" source="block" result="text" session="typicality"
import io

import numpy as np
import pandas as pd

import pymc_generator as pg


def weekly_table(seed, n_weeks=52, flighting=False):
    """A simple always-on weekly dataset; `flighting` switches tv off every other month."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    tv = rng.gamma(4.0, 25.0, n_weeks)
    if flighting:
        tv[(t // 4) % 2 == 1] = 0.0
    search = 50.0 + np.cumsum(rng.normal(0.0, 3.0, n_weeks)).clip(-40.0)
    price = 10.0 + np.cumsum(rng.normal(0.0, 0.1, n_weeks))
    temperature = 15.0 + 10.0 * np.sin(2 * np.pi * t / 52) + rng.normal(0.0, 1.0, n_weeks)
    sales = 200 + 0.8 * tv + 1.5 * search - 5 * price + 2 * temperature + rng.normal(0, 10, n_weeks)
    return pd.DataFrame({
        "week": pd.date_range("2024-01-07", periods=n_weeks, freq="W-SUN").strftime("%Y-%m-%d"),
        "tv": tv, "search": search, "price": price, "temperature": temperature, "sales": sales,
    })


csv_text = weekly_table(seed=0).to_csv(index=False)  # stands in for a file on disk
dataset = pg.ObservedWorlds.from_csv(
    io.StringIO(csv_text),
    treatments=["tv", "search"],
    covariates=["price", "temperature"],
    outcome="sales",
    time="week",
)
print(dataset.treatment_raw.shape, dataset.covariates.shape, dataset.outcome_raw.shape)
```

`ObservedWorlds.from_frame` takes the same arguments with a pandas DataFrame or a
mapping of column name to 1-D array; `from_csv` reads with `pandas.read_csv`
(extra keyword arguments such as `sep=";"`, `decimal=","` or `parse_dates=` are
passed through) and keeps float64 values exact with the default parser (options
that need pandas' python engine, such as `sep=None`, are not exact).

| Argument | Meaning |
| --- | --- |
| `treatments` | Treatment columns (at least one): a list, a pandas `Index` or an array of labels. |
| `covariates` | Covariate columns (at least one), likewise. |
| `outcome` | The outcome column. |
| `time` | Optional time column. Numbers, datetimes, periods, timedeltas or ISO date text; it must increase strictly. It only checks the order: rows are never sorted. |

When `treatments`, `covariates` and `outcome` are all omitted, the columns of
an [audit bundle's](bundles.md) `dataset.csv` are used: `treatment_C1, …`,
`covariate_Z1, …` and `outcome_Y`. `ObservedWorlds.to_frame(world)` writes a
world in that layout, so any generated world can be exported as a dataset.

Refused, with a message that names the column (and, for missing, non-finite or
out-of-order values, the first offending row): missing or non-finite values,
text or boolean columns, a column used twice, an all-zero column (an inactive
series: leave it out), a time column that does not increase. Nothing is
imputed, sorted or rescaled.

!!! note "Raw levels only"
    Pass the series as they were measured. Centering, standardizing or taking
    logs changes the descriptors (a centered series has no meaningful
    coefficient of variation), so a transformed dataset looks atypical for
    reasons that have nothing to do with the data. Level descriptors
    (`cv`, `level_to_variation`) also depend on where a series' zero is: a
    temperature in °C, °F or K gives different values. Prefer units with a
    true zero, or leave such descriptors out with `features=`.

## The two references

### An existing corpus

Any generated corpus works, loaded or in memory. Its worlds carry a generation
cell (`cell_id`), which `world_descriptors` records as the group:

```python
corpus = pg.load_corpus("corpus.npz")
reference = pg.world_descriptors(corpus, source_id="corpus", lags=(1, 4))
target = pg.world_descriptors(dataset, source_id="sales", lags=(1, 4))
result = pg.typicality(reference, target)
```

The dataset is compared only with reference worlds of **its shape**: the same
number of active treatments, of active covariates and of time steps. A corpus
with no world of that shape answers `no_reference`. Corpora with several draws
per cell are fine: calibration takes one world (its lowest `world_id`) per
calibration cell; every draw of a ruler cell joins the ruler, and the other
draws of calibration cells enter only the profile and the duplicate check.

### A generator configuration

To ask about a configuration rather than a stored corpus, generate a reference
of the dataset's shape from it. `matched_prior` pins the active counts to the
dataset's, which samples the configuration *conditioned on that shape* (counts
are drawn first in every cell, everything else given them):

```python
prior = pg.make_scm_prior(
    n_treatments=4, n_covariates=4, n_latent=1,
    n_treatments_active_range=(1, 4), n_covariates_active_range=(1, 4),
    n_time_steps=dataset.n_time_steps,
)
matched = pg.matched_prior(prior, dataset, n_cells=200, draws_per_cell=3, seed=11)
corpus = pg.sample_prior_predictive(matched)
reference = pg.world_descriptors(corpus, source_id="prior", lags=(1, 4))
```

`matched_prior` refuses a prior built for another horizon (presets derive
settings from `n_time_steps`, so rebuild it at the dataset's horizon) and a
shape the configuration never makes. `seed` is required: it must differ from
any seed that produced the dataset, or the reference can contain the dataset
itself (`typicality` then reports it as a `duplicate`).

Calibration takes one world per generation cell, so `n_cells` sets the
resolution: with `n_cells=200` and the default split, 100 calibration worlds
and a smallest possible p of 1/101. It also sets the generation time, roughly
one second per cell for small worlds. Extra draws per cell (`draws_per_cell`)
are nearly free to generate and enlarge the ruler, which can add power against
subtle differences; they cost scoring time. At least `max(k, 2)` ruler worlds
are needed: `n_cells ≥ 9` with one draw per cell and the defaults.

Scoring is fast for references of a few thousand worlds, but every call scores
each calibration world against the ruler, so its cost grows with the product
of their sizes (seconds at 10,000 reference worlds, tens of seconds at
40,000). Score a panel of datasets in one call, and for a very large corpus
describe a random subset of whole generation cells: cells are exchangeable, so
the guarantee still holds.

## Reading the result

To show the output without generating anything, the reference below is a
panel of 200 other tables from the same simple recipe, one group per table.
With a corpus or a generator only the reference rows change.

```python exec="1" source="block" result="text" session="typicality"
panel = [weekly_table(seed) for seed in range(1, 201)]
reference = pg.world_descriptors(
    pg.ObservedWorlds(
        treatment_raw=np.stack([t[["tv", "search"]].to_numpy() for t in panel]),
        covariates=np.stack([t[["price", "temperature"]].to_numpy() for t in panel]),
        outcome_raw=np.stack([t["sales"].to_numpy() for t in panel]),
    ),
    source_id="panel",
    group_ids=np.arange(len(panel)),
)

flighted = pg.ObservedWorlds.from_frame(
    weekly_table(seed=0, flighting=True),
    treatments=["tv", "search"], covariates=["price", "temperature"], outcome="sales",
)
target = pg.WorldDescriptors.concat([
    pg.world_descriptors(dataset, source_id="always-on"),
    pg.world_descriptors(flighted, source_id="flighted"),
])
result = pg.typicality(reference, target)
print(result.table())
```

One line per dataset: how many reference worlds share its shape, the status,
the p-value and the smallest p the calibration set allows (`p_min`), the number
of **support violations** and the **distance**, the ruler and calibration sizes,
how many profiled descriptors lie inside the reference's central interval
(`inside`), and how many a world of the reference would have outside by chance
(`exp_outside`). In the flighted dataset one treatment is exactly zero for
weeks at a time, which no reference world shows: its zero-share descriptors
are support violations, no calibration world has one, so its p-value is the
floor.

```python exec="1" source="block" result="text" session="typicality"
profile = result.profile(1)
print(profile[profile["violation"] | ~profile["inside"]][
    ["feature", "value", "percentile", "reference_low", "reference_high", "inside", "violation"]
].to_string(index=False))
```

`profile(row)` has one row per feature: the dataset's value and status, its
percentile among the reference worlds of its shape, the reference's
`reference_low` / `reference_median` / `reference_high` (the central 90% by
default, `central=`), `inside`, `violation` and the feature's role in the score.
`to_frame()` gives one row per dataset and `summary()` a strictly JSON-safe
report.

| Status | Meaning |
| --- | --- |
| `ok` | Scored. |
| `no_reference` | No reference world has the dataset's shape. |
| `too_few` | Too few reference worlds of that shape to calibrate (fewer than `max(k, 2)` ruler worlds or no calibration world). |
| `duplicate` | A reference world matches the dataset on every shared descriptor except the zero and constant fractions — equal statuses, values within a relative 1e-4 (`1e-4 * max(abs(x), abs(r), 1)`): the dataset is in the reference. |

## How the p-value is built

1. **Shape.** Only reference worlds with the dataset's active counts and
   horizon take part.
2. **Split.** Their generation groups are split once, by a seeded hash of the
   group's identity and never by values, into **calibration** groups (one world
   each, `calibration_fraction` of the groups, half by default) and **ruler**
   groups (all their worlds).
3. **Score.** Every calibration world and the dataset are scored against the
   ruler only, in two steps compared in order:
    * **support violations** — descriptors whose state the ruler never shows
      (for example defined versus undefined), plus descriptors that are one
      constant across the ruler (every ruler world has no zero weeks) where the
      world differs;
    * **distance** — the mean distance to the `k` nearest ruler worlds over the
      remaining descriptors, each standardized by the ruler's median and
      interquartile range (its standard deviation where the IQR is zero; as in
      [`nearest_worlds`](world-descriptors.md#similarity)).
4. **Rank.** `p = (1 + #calibration worlds scoring at least as high) /
   (n_calibration + 1)`.

Descriptors that are the same number by construction at the dataset's shape
(median, min and max over a single treatment) are scored once.

**Guarantee.** If the dataset and the reference worlds are independent draws
from one generator, given the dataset's shape, and `features`, `k`,
`calibration_fraction`, `central` and `seed` are chosen before looking at the
result, then `P(p ≤ α) ≤ α` for every α. One world per calibration cell is what
makes the calibration worlds independent: worlds of one cell share their graph
and mechanism families (only continuous parameters and noise are redrawn).

## What the answer is not

* **Not proof of origin.** A large p means these descriptors do not tell the
  dataset apart from the reference; another generator may fit as well. Power
  grows with the number of calibration worlds and with how strongly the
  generators differ: a generator that only shifts covariate levels can pass
  where one that adds on/off flighting cannot.
* **Not a rate for one reference.** The guarantee holds on average over
  references; against one fixed reference, the share of p ≤ α among many
  datasets scatters around α, and p-values of a panel scored against one
  reference are dependent.
* **Not an explanation.** The profile shows which descriptors stand out, but
  about 10% of features fall outside a 90% interval by chance
  (`expected_outside`); read it as description, not as one test per feature.

The [dataset typicality example](../examples/dataset-typicality.ipynb) generates
datasets from the same generator and from other generators and scores them
against one reference. See the [API reference](../reference/typicality.md) for
every field.
