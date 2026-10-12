# World descriptors

`data_diagnostics` explains what happens inside a set of worlds.
`world_descriptors` turns every world into **one row of named statistics** so
that worlds can be compared with each other, and four independent functions
analyse those rows:

| Function | Question it answers |
| --- | --- |
| `summarize_descriptors` | How is each statistic distributed across these worlds (optionally per stratum)? |
| `bin_counts` | How many worlds (and distinct generation groups) fall in each explicit region? |
| `nearest_worlds` | Which worlds are most similar in a chosen feature space, and how far away are they? |
| `compare_descriptors` | Where does one collection sit relative to a reference collection? |

```python
import pymc_generator as pg

rows = pg.world_descriptors(corpus, source_id="smooth", lags=(1, 4))
print(rows.table())

pg.summarize_descriptors(rows, by=("n_treatments_active",)).table("treatment_cv_median")
pg.bin_counts(rows, {"covariate_level_to_variation_median": [0, 1, 3, 10, 30]}).table()
pg.nearest_worlds(rows, features=("treatment_cv_median", "treatment_acf_lag_1_median")).table()
pg.compare_descriptors(rows, other_rows).table("covariate_level_to_variation_median")
```

None of these functions generates data, changes a prior, or ranks a
collection. They share only the descriptor table. Collection names
(`source_id`) are arbitrary labels; in the directional functions the argument
**role** (`reference` versus `query`) is what changes the answer.

## Sources

`world_descriptors` accepts a corpus mapping, a sequence of
[`SCM`](sampling.md#pymc_generator.worlds.SCM) worlds sharing one horizon, or
[`ObservedWorlds`](#pymc_generator.descriptors.ObservedWorlds) — observed
datasets passed with the corpus array names (`treatment_raw`, `covariates`,
`outcome_raw`, optional active masks). One dataset is one world. Observed data
carry no graph, contribution truth or generation group, and none is invented.

## Row identity and groups

Every row is identified by `(source_id, world_id)`, which must be unique.
`world_id` defaults to the source position (a `worlds=` selection keeps the
original positions); pass `world_ids=` for already-sliced shards. `group_id`
defaults to a corpus's `cell_id` — the generation cell that fixes graph, active
nodes and discrete mechanism choices — and to `-1` (unknown) otherwise. Two rows
share a group only when their `source_id` and a known `group_id` agree, so cells
from independently generated batches never collide as long as each batch gets
its own `source_id`.

## Statistics

Each statistic is computed inside one world over its **active** series, then
reduced across the active nodes of a role with `median`, `min` or `max`.
Feature names spell this out: `covariate_level_to_variation_median`,
`treatment_acf_lag_1_max`, `treatment_pair_diff_abs_pearson_median`,
`outcome_cv`.

| Family | Definition | Notes |
| --- | --- | --- |
| `level_to_variation` | `abs(mean) / std` | `+inf` for a nonzero constant, undefined for an all-zero series; invariant to positive rescaling, sensitive to shifts. |
| `cv` | `std / abs(mean)` | Signal-gate convention: `0` for constants and all-zero series, undefined for a varying zero-mean series. |
| `zero_fraction` | fraction of exactly-zero steps | Needs a meaningful zero (on/off flighting). |
| `roughness`, `spike` | the `data_diagnostics` slots | Roughness ≈ 1 for white noise, → 0 for smooth drift. |
| `acf`, `diff_acf` | sample ACF of levels / first differences | One feature per requested lag; too few lag pairs is undefined. |
| `constant_fraction` | share of active series that are constant | Keeps constant-dominated worlds visible behind a reducer. |
| `abs_pearson`, `diff_abs_pearson` | `abs(Pearson)` in levels / differences | Over treatment pairs, covariate pairs, treatment-covariate and treatment-outcome pairs. |
| `truth_*_share` | net top-cut contribution shares | Only with `include_truth=True`, generated sources only, `observable=False`. |

Moments use population `std`, as in `data_diagnostics`. Degenerate series are
detected from exact ranges rather than a computed `std`, whose rounding residue
misses true constants. With `eps` the machine epsilon of the input dtype
(float32 corpora keep float32 resolution), a series is **constant** when
`max - min <= 4 * eps * max|x|`, its **steps are constant** (a linear ramp) when
their range is at most `8 * eps * max|x|`, and its **mean is zero** when
`|mean| <= (min(T, 32) * eps + T * eps64) * max|x|` — input-storage rounding plus
float64 summation error — and it takes both signs. A constant has infinite level-to-variation, CV 0,
roughness and spike 0, and no autocorrelation or correlation; a ramp has
roughness and spike 0 and no differenced statistics. There is no absolute
epsilon, and every series is first rescaled exactly by a power of two, so values
from `1e-300` to `1e300` describe identically. Roughness and spike need at least
three steps (`T >= 4`), like every lag-based statistic. Levels are
random-walk-like, so level-view correlations are large even between independent
series; read the `diff_` companions alongside them.

## Valid, ineligible, undefined

Every cell carries a status. **Valid** values may be `+inf`/`-inf`: an infinite
level-to-variation ratio is a real, ordered fact. **Ineligible** means the
question does not apply (no active covariates, fewer than two treatments for a
pair statistic, no observed outcome). **Undefined** means it applies but has no
answer (`0 / 0`, a lag beyond the usable horizon). Values are NaN exactly when
the status is not valid; `to_frame(with_status=True)` keeps the two NaN reasons
apart, and every summary counts them separately.

Reducers run over valid node values in the extended reals, so the median of
`[1, inf, inf]` is `inf`, not `1`.

## Distances and comparisons

`nearest_worlds` uses a fixed selected feature list and one frozen
[`DescriptorScale`](#pymc_generator.descriptor_analysis.DescriptorScale),
fitted on the **reference** only (median/IQR by default), so a query collection
can never move the ruler it is measured with. Distance is the root-mean-square
of standardized differences over the selected features, computed only for rows
where every selected feature is finite; other rows are reported, not dropped.
A feature that is constant in the reference cannot be scaled and raises: drop
it or build a `DescriptorScale` explicitly. The same identity is never its own
neighbour; `exclude_same_group=True` also excludes worlds from the same
generation cell, and `match=` restricts neighbours to equal strata (for
example equal active counts).

`compare_descriptors` ranks each query value against the reference values of
its own stratum with the midrank convention
`(count(ref < x) + 0.5 * count(ref == x)) / n`, leaving the query world out when
it also belongs to the reference. It also reports per-stratum distribution
summaries, the median shift, the fraction of query values outside the
reference range, and a descriptive Kolmogorov–Smirnov distance.

## What these numbers are not

* **Not information or learnability.** Two worlds with equal descriptors can
  have different trajectories, and descriptor distance says nothing about how
  much a model learns from either.
* **Not duplication.** Worlds from one generation cell share structure but are
  different realizations; a small distance is descriptor similarity, not
  repeated data.
* **Not inference.** No p-values or significance: worlds within a cell are
  dependent and descriptors are noisy functions of finite series.
* **Not a support guarantee.** A target can be ordinary on every marginal and
  still sit far from every reference world jointly; check both.

::: pymc_generator.descriptors
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - world_descriptors
        - WorldDescriptors
        - ObservedWorlds
        - FeatureDefinition
        - STATISTICS
        - REDUCERS
        - PAIR_ROLES
        - METADATA_COLUMNS
        - DESCRIPTOR_STATUSES
        - TRUTH_FEATURES
        - DESCRIPTOR_VERSION

::: pymc_generator.descriptor_analysis
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - summarize_descriptors
        - DescriptorSummary
        - bin_counts
        - BinCounts
        - nearest_worlds
        - NearestWorlds
        - DescriptorScale
        - NEIGHBOR_STATUSES
        - compare_descriptors
        - DescriptorComparison
