# Data diagnostics

`outcome_distributions` answers "how large is each quantity". `data_diagnostics`
answers the questions that need the **series themselves** and the relations
**between** them: how each series is shaped, what depends on what, how much of
the design is redundant, how the series move over time, and exactly what adds
up to outcome.

```python
import pymc_generator as pg
from pymc_generator import viz

corpus = pg.sample_prior_predictive(cfg)
rep = pg.data_diagnostics(corpus, scopes=("nodes", "decomposition"))

print(rep.table())                                   # overview + the top cut
print(rep["levels"].dependence.table("xi_max"))      # strongest pairs
print(rep["levels"].vif["oracle"].table())           # collinearity incl. latent D
print(rep["differences"].series.table("roughness"))  # week-to-week texture
```

It accepts a corpus (from [`sample_prior_predictive`](corpus.md),
`DataGenerator.generate` or `load_corpus`) or a list of
[`SCM`](sampling.md#pymc_generator.worlds.SCM) worlds sharing one horizon.
Everything is strictly post-hoc over retained arrays: no generation, no RNG
draw, no mutation of the source, nothing written.

## The three rules behind every number

1. **World first.** Every statistic is computed inside one world and one view,
   then summarized across worlds. Pooling rows from different worlds
   manufactures dependence out of between-world level differences — two worlds
   with zero within-world correlation can pool to `r = 0.95`.
2. **Levels and differences.** Weekly series are random-walk-like, so level
   dependence is large even between independent series. Both views are
   reported; differencing is a companion view, **not** a proof of stationarity.
3. **Description, not inference.** No p-values, no error bands, no
   significance, no causal or forecast claims. Overlapping lag pairs and
   correlated worlds make iid inference invalid here, and the latent series
   (`D`, `B`) are retained ground truth, never model inputs.

## Series keys

Keys are globally unique ASCII tokens fixed from the FULL source before any
world selection, and they are the only selector raw access and plots accept —
display labels are not unique (`C1`, `C1_direct_y` and `C1_base` all read as
"treatment 1" to a human).

| Scope | Keys |
| --- | --- |
| `nodes` (always on) | `C1..CK` treatment, `Z1..ZM` covariates, `D1..DJ` latent-unobserved factors, `B` retained baseline, `Y` outcome |
| `decomposition` | `B_intrinsic`, `Y_noise`, `Zm_baseline_alloc`, `Dj_baseline_alloc`, `Ck_direct_y`, `indirect_cc/zc/dc`, `indirect_total`, `treatment_total` |
| `counterfactual` (SCM only) | `Ck_base`, `Ck_observed_y`, and — when shocks were drawn — `Ck_unshocked`, `Y_unshocked` |

`Zm_baseline_alloc` and `Dj_baseline_alloc` are the **retained baseline
allocation**, not necessarily the raw linear edge term: under a non-treatment
baseline floor they are locked-order clipped telescoping increments, and the
corpus does not retain the floor metadata needed to undo that.

An optional path a source never carried stays a **known, ineligible** key with
zero eligible worlds (`descriptor(key).available is False`) instead of reading
as zero.

## What each metric says

| Facet | Metric | Reads as |
| --- | --- | --- |
| Series | `mean`, `std`, `median`, `iqr`, `min`, `max` | level and spread of the series, per world |
| | `roughness` = `std(Δz) / (√2·std z)` | 1 ≈ white noise, → 0 smooth drift |
| | `spike` = `max|Δ − median Δ| / IQR(Δ)` | pulse dominance; `+inf` when one jump sits on an otherwise flat series |
| Temporal | `acf(h)` | linear persistence from week `t` to `t+h` |
| | `lag_xi(h)` | serial predictability of any shape, including the nonlinear structure ACF misses |
| Dependence | `pearson` | signed linear co-movement within a world |
| | `spearman` | monotone co-movement, outlier-robust |
| | `xi` | how much the column (target Y) is a **function** of the row (predictor X); asymmetric by construction |
| | `xi_max` | symmetric "is there any functional relation", for heatmaps |
| VIF | `observed` (active C+Z) | redundancy a real model would face among visible predictors |
| | `oracle` (active C+Z+D) | the extra redundancy hidden latent-unobserved factors inject; `+inf` = exact collinearity |
| Contributions | `net_share` | signed fraction of Σ outcome; a complete top cut sums to 1 per world |
| | `net_total`, `net_mean_per_period` | the same in outcome units — equal shares can hide 100× magnitude gaps |
| | `gross_over_net_outcome` | activity magnitude including cancellation; ≥ 1 for a complete valid top cut |

## Chatterjee's xi, precisely

`xi(X → Y) = 1 − n·E[Σ|r_{i+1} − r_i|] / (2·Σ l_i (n − l_i))` with the data
ordered by X, `r_i = #{j: Y_j ≤ Y_i}` and `l_i = #{j: Y_j ≥ Y_i}`.

* **Directional.** For `Y = X²`, `xi(X → Y)` is large and `xi(Y → X)` is not.
  The heatmap row is the predictor, the column is the target.
* **Ties are averaged exactly.** A tie in X leaves the order inside the block
  undefined and the statistic is not invariant to how it is broken. Instead of
  inheriting the sort's accident, the implementation averages analytically over
  every ordering a tie block admits — deterministic, no RNG, still `O(n log n)`.
* **Not clipped.** The sample statistic can be negative. With distinct target
  values it cannot exceed `(n − 2)/(n + 1)`; tied targets can.
* **Constant target ⇒ no answer** (the denominator is zero). A constant
  *predictor* against a varying target is a valid question whose answer is
  exactly `0`.
* xi is a *general* dependence measure, not a "nonlinear-only" one and not
  universally faster or better than distance correlation — fast univariate dCor
  algorithms are also `O(n log n)`. dCor is deferred because its estimand,
  scale and serial calibration are separate decisions, not because it is slow.

## VIF is a projection, never a rank test

For each target the other predictors are centred, scaled to unit norm, SVD'd,
and the target is projected onto the retained left-singular basis;
`VIF = 1 / (RSS/TSS)`, with `+inf` only when the residual falls below an
explicit tolerance. Inferring `+inf` from "the augmented design has the same
numerical rank as the nuisance design" is wrong: a target can sit far outside a
numerically rank-deficient span and still have a perfectly finite VIF.

Design rank and condition number are reported separately, and the condition
number can be infinite while individual target VIFs stay finite. `condition`
is the condition number of the centred, unit-column-normalised design over its
**varying** columns, and it is `+inf` whenever any predictor is constant or the
design is rank deficient — a constant column carries no direction, so the
design it belongs to is degenerate even when every other target is fine. A
column whose variation sits at or below the rounding floor of its own scale is
treated as constant rather than unit-normalised into a noise direction.

Exact OLS says adding regressors cannot decrease a common target's VIF — but
per-scope SVD truncation is not nested, so no numerical `oracle ≥ observed`
invariant is claimed and each scope is verified on its own.

## Report-validity policies (not mathematical limits)

* Pairwise dependence needs `n ≥ 3` observations and a non-constant target.
* A VIF needs `n ≥ 3` observations and a non-constant target column.
* An ACF or lag-xi value needs at least three usable pairs (`T_view − h ≥ 3`).

All three are conservative reporting choices, not statements that the formulas
cannot be evaluated with fewer points.

## Coverage, infinities and closure

Every aggregate carries a `CoverageLedger` with
`selected / eligible / valid / finite / positive_infinite / negative_infinite /
invalid` counts, satisfying `valid = finite + ±inf` and
`eligible = valid + invalid`. Finite means, standard deviations and quantiles
use the finite valid values only — `np.quantile([1.0, inf], 0.5)` is NaN, which
would be indistinguishable from "no data" — and infinities are counted, never
reclassified. A stats dict holds `None`, not NaN, where nothing finite exists,
so `json.dumps(rep.summary(), allow_nan=False)` succeeds.

Closure is a property of a **cut**, not of a table. A complete sibling cut
inside one world closes exactly; common-population linear rollups can close;
marginal quantiles, activity-conditioned populations, partial projections and
mixed bases do not, and are marked incomplete with their omitted keys instead
of making a closure claim.

## The contribution hierarchy

Basis `base_direct_plus_indirect` (default):

```text
treatment_total                          ← top cut
├── treatments_direct_y_total
│   └── Ck_direct_y                  (one per active channel)
├── indirect_cc / indirect_zc / indirect_dc
covariates_baseline_alloc_total        ← top cut
└── Zm_baseline_alloc
latent_unobserved_baseline_alloc_total          ← top cut
└── Dj_baseline_alloc
B_intrinsic                          ← top cut
Y_noise                              ← top cut
```

Parents roll up from their **atomic descendants'** gross totals:
`abs(Σ children)` would erase cancellation (children `[+10, −10]` are 20 units
of activity, not 0). A complete valid top cut therefore has
`Σ gross_over_net_outcome ≥ 1`, with equality exactly when nothing cancels;
an arbitrary subset has no such bound.

Per-treatment rows are the **direct / base-path** effect. The aggregate indirect
terms (`indirect_cc`, `indirect_zc`, `indirect_dc`) cannot be attributed to
individual nodes from a corpus, and this API never pretends otherwise. A
sequence of `SCM` worlds additionally exposes the alternative
`observed_path_treatment` basis (`treatment_total_observed_path` → `Ck_observed_y`),
which is a different *reading* of the same treatment effect and is never additive
alongside the default one.

```python
budget = rep.contributions
budget.table(sibling_set="top")                       # macro, share, net
budget.stats("C1_direct_y", measure="total", weighting="micro")
budget.select(("C1_direct_y", "C3_direct_y")).table(measure="total")
budget.closure("treatment_children").max_abs_residual("net_total")
```

`weighting="macro"` is the equal-world distribution; `"micro"` pools numerator
and denominator (an outcome-weighted ratio, never a mean of ratios).
`population="conditional_on_active"` restricts to the worlds where a row
exists; `"unconditional"` (default) encodes absence as an exact zero. An active
structural zero stays included either way — it is real data.

## Selection, views and lean mode

```python
pg.data_diagnostics(corpus, worlds=corpus["cell_id"] == 3)
pg.data_diagnostics(corpus, worlds=[17])            # one world, same schema
pg.data_diagnostics(corpus, views=("levels",))      # see the warning above
pg.data_diagnostics(corpus, lags=(1, 2, 4, 52))
pg.data_diagnostics(corpus, keep_series=False)      # drop the raw values
```

World selectors are `None`, a slice, an integer position, an integer position
array or a boolean row mask. Duplicated positions are rejected: a repeated
world would be counted twice in every across-world summary. The same 0/1
integer ambiguity `outcome_distributions` refuses is refused here.

The default lag axis is the contiguous `1..min(52, T // 2)`. A sparse axis
silently misses MA(2)-style structure at lag 2 and annual recurrence at lag 52.

`keep_series=False` leaves every non-raw result bit-identical and only makes
`SeriesDiagnostics.values` (and the raw value plots) raise.

## Reports

`summary()` is strictly JSON-safe, `to_frame()` is long-form pandas, `table()`
methods render fixed-width text, and
[`viz.plot_dependence_matrices`](viz.md) and friends render the figures.

::: pymc_generator.diagnostics
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - data_diagnostics
        - DataDiagnostics
        - DiagnosticView
        - SeriesDescriptor
        - SeriesDiagnostics
        - DependenceDiagnostics
        - VIFDiagnostics
        - TemporalDiagnostics
        - ContributionBudget
        - ContributionDescriptor
        - ContributionClosure
        - CoverageLedger
        - SERIES_SLOTS
        - DEPENDENCE_METRICS
        - VIF_SCOPES
        - CONTRIBUTION_FIELDS
        - CONTRIBUTION_BASES
        - DEFAULT_MAX_LAG
