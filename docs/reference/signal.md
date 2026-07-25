# Signal diagnostics

A corpus can satisfy every schema contract and still be *unlearnable*: if the
true per-channel contribution barely varies, there is no signal to attribute.
This module quantifies, per direct (C→Y) channel, how much signal the generator
actually produced — embedded in every corpus's `diagnostics["signal"]` block and
checkable at generation time.

## Persisted layout (version 1)

`signal_metrics` is a dense `float32` array of shape `(N, K, 9)` and
`signal_metric_valid` is a same-shaped binary `uint8` array. The locked,
append-only v1 layout in `diagnostics["signal"]["metric_layout"]` is:

1. `spend_cv`
2. `spend_hf`
3. `contrib_cv`
4. `contrib_hf`
5. `contrib_rel_std`
6. `spearman`
7. `warmup_ratio`
8. `contrib_r2_explained_by_rest`
9. `contrib_corr_baseline`

`metric_version` is `1`. Entries for inactive or non-direct channels are zero
and invalid. Validity is per metric, rather than a promise that every metric
can be computed for every eligible pair; invalid metric values are exactly zero
and summaries exclude them from both quantiles and fractions. For example,
Spearman requires at least three observations after dropping its initial
`l_max` prefix, while the
high-frequency and regression/correlation metrics require at least two points.

All metrics use the full reported window except `spearman`: it compares the
contribution with observed spend adstocked **once** using the world's actual
geometric or Weibull parameters and any recorded reset starts, then discards
the first `l_max` observations. It does not adstock a contribution a second
time. `warmup_ratio` is N/A when `adstock_burn_in >= l_max` (its aggregate
artifact fraction is reported as `0.0` with sufficient burn-in); it is also
invalid when the window is too short.

`contrib_r2_explained_by_rest` is full-window R² against an intercept,
baseline, and the other active direct contributions. For a constant target it
is `1` only if that full-window fit reproduces the target within the
scale-aware numerical tolerance; otherwise ordinary R² is clipped to `[0, 1]`.
`contrib_corr_baseline` is the signed Pearson correlation and is encoded as
zero when either series has no variance. Constant-rank Spearman inputs are also
encoded as zero, not NaN.

Metrics are calculated from the final stored float32 arrays, after any task
truncation, so they are exactly recomputable after save/load. The signal summary
is likewise derived from those retained arrays and carries the layout/version.

```python
from prior_generator.signal_diagnostics import SIGNAL_METRIC_LAYOUT

metrics = corpus["signal_metrics"]
valid = corpus["signal_metric_valid"].astype(bool)
spearman = metrics[..., SIGNAL_METRIC_LAYOUT.index("spearman")]
```

::: prior_generator.signal_diagnostics
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - dense_signal_metrics
        - per_channel_signal
        - summarize_signal_metrics
        - signal_summary
        - check_signal_gate
        - SIGNAL_METRIC_VERSION
        - SIGNAL_METRIC_LAYOUT
        - DEFAULT_GATE
        - METRIC_KEYS
        - FRAC_KEYS
