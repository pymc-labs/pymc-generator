# Signal diagnostics

A corpus can satisfy every schema contract and still be *unlearnable*: if the
true per-channel contribution barely varies, there is no signal to attribute.
This module quantifies, per direct (C→Y) channel, how much signal the generator
actually produced — embedded in every corpus's `diagnostics["signal"]` block and
checkable at generation time.

## Persisted layout (version 2)

With `include_identifiability_labels=True` (the default), `signal_metrics` is a
dense `float32` array of shape `(N, K, 9)` and `signal_metric_valid` is a
same-shaped binary `uint8` array. These arrays are truth-derived metadata, not
model features; setting the option to `False` omits both without changing any
observable array. The locked, append-only v2 layout in
`diagnostics["signal"]["metric_layout"]` is:

1. `spend_cv`
2. `spend_hf`
3. `contrib_cv`
4. `contrib_hf`
5. `contrib_rel_std`
6. `spearman`
7. `warmup_ratio`
8. `contrib_r2_explained_by_rest`
9. `contrib_corr_baseline`

`metric_version` is `2`. Entries for inactive or non-direct channels are zero
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
baseline, and the other active direct contributions. It is invalid when the
design rank leaves no residual degrees of freedom. An exactly constant stored
target is `1`; every representably nonconstant target uses ordinary centered
R² clipped to `[0, 1]`, without an amplitude-dependent tolerance floor.
`contrib_corr_baseline` is the signed Pearson correlation and is encoded as
zero when either series has no variance. Constant-rank Spearman inputs are also
encoded as zero, not NaN.

The legacy flattened helpers keep `baseline` optional. When it is omitted,
both baseline-dependent metrics are zero and invalid rather than being inferred
from a fabricated zero baseline. Persisted corpus generation always supplies
the true baseline.

Metrics are calculated from the final stored float32 arrays, after any task
truncation, so they are exactly recomputable after save/load. The signal summary
is likewise derived from those retained arrays and carries the layout/version.
It also records `l_max`, `adstock_burn_in`, `adstock_kernel_semantics`, and
`adstock_kernel_version`, so nondefault shards can be recomputed without their
original Python config object.

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
