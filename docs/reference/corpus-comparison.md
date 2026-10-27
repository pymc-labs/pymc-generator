# Existing-corpus comparisons

Descriptive statistical comparisons of observed data with an existing corpus.
The [executable guide](../guide/corpus-comparison.md) demonstrates selection,
statistics, matching, held-out score assessment, and all five standalone plots
using independently constructed synthetic arrays. No model or generator is run.

| Function | Input and result |
| --- | --- |
| `sample_reference_windows` | Corpus mapping or `ObservedWorlds` → aligned windows, explicit role projection, and original world/cell/column lineage. |
| `temporal_statistics` | `(world,time,series)` values → original-unit variability, time correlation, overlap-Pearson profiles, and counts. |
| `activity_runs` | `(world,time,series)` values → exact-zero spells, censoring, summaries, and detailed records. |
| `lagged_dependence` | Explicit within-world pairs and signed lags → level/difference Pearson and observation counts. |
| `normalize_series_quantile` | Finite values with an explicit time axis, quantile and mode → divisors, scaled values, source resolution, and availability. |
| `compare_series` | Aligned `(series,time)` arrays or normalization results → per-metric pair matrices. |
| `nearest_series` | A `SeriesComparison` → best signed/lowest-distance scores, candidate indices and finite counts. |
| `assess_series_similarity` | Query, held-out series and a shared bank → empirical score percentiles, tie bounds, and finite counts. |

Keep query and reference roles directional. Preserve normalization metadata, and
pass stored source resolution when extracting arrays from `ObservedWorlds`.
Group labels must be known and share a namespace when disjointness is required.
Window extraction and role projection change the reference population; they do
not automatically preserve an original corpus's conformal guarantee.

`overlap_pearson_rmse` compares overlap-centered Pearson profiles, not the
existing global-mean/full-denominator sample ACF. Signed correlation maxima do
not become absolute maxima. Undefined comparisons remain unavailable; no finite
candidate is represented by `NaN`, index `-1`, and count `0`. Empirical ranks and
reference quantile bands are descriptive, not origin probabilities, confidence
intervals, causal evidence, or new calibrated tests.

## Reference selection

::: pymc_generator.dataset_typicality
    options:
      show_root_heading: false
      members:
        - sample_reference_windows
        - ReferenceWindows

## Statistical profiles

::: pymc_generator.diagnostics
    options:
      show_root_heading: false
      members:
        - temporal_statistics
        - TemporalStatistics
        - activity_runs
        - ActivityRuns
        - ACTIVITY_STATISTICS
        - lagged_dependence
        - LaggedDependence

## Scaling, matching, and empirical scores

::: pymc_generator.series_comparison
    options:
      show_root_heading: false
      members:
        - normalize_series_quantile
        - QuantileNormalization
        - compare_series
        - SeriesComparison
        - nearest_series
        - NearestSeries
        - assess_series_similarity
        - SimilarityAssessment
        - SERIES_METRICS

The [visualization reference](viz.md#existing-corpus-comparison-figures) documents
standalone comparison renderers. Extended world-level feature families are
selected explicitly with `world_descriptors(statistics=..., lags=...,
dependence_lags=...)` and use the existing `WorldDescriptors`,
`DescriptorComparison`, scaling, ranking, and typicality infrastructure.
