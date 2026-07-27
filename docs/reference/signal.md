# Signal diagnostics

A corpus can satisfy every schema contract and still be *unlearnable*: if the
true per-channel contribution barely varies, there is no signal to attribute.
This module quantifies, per direct (C→Y) channel, how much signal the generator
actually produced — embedded in every corpus's `diagnostics["signal"]` block and
checkable at generation time.

## Persisted layout (version 3)

With `include_identifiability_labels=True` (the default),
`corpus["identifiability"]["signal_metrics"]` is a dense `float32` array of
shape `(N, K, 9)` and `signal_metric_valid` in the same block is a same-shaped
binary `uint8` array. This nested block keeps truth-derived metadata outside the
top-level model-feature arrays; setting the option to `False` omits the block
without changing any observable array. The locked, append-only layout in
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

`metric_version` is `3`. Entries for inactive or non-direct channels are zero
and invalid. Validity is per metric, rather than a promise that every metric
can be computed for every eligible pair; invalid metric values are exactly zero
and summaries exclude them from both quantiles and fractions. For example,
Spearman requires at least three observations after dropping its initial
`l_max` prefix, while the high-frequency and regression/correlation metrics
require at least two points.

All metrics use the full reported window except `spearman`: it compares the
contribution with observed spend adstocked **once** using the world's actual
geometric or Weibull parameters, then discards the first `l_max` observations.
It does not adstock a contribution a second time.
`warmup_ratio` is N/A when `adstock_burn_in >= l_max` (its aggregate
artifact fraction is reported as `0.0` with sufficient burn-in); it is also
invalid when the window is too short.

## Response warmup and delayed adstock

`response_warmup_weeks` is `l_max - 1` only when `adstock_burn_in > 0` and at
least one eligible direct channel has a nonidentity kernel
(`adstock_family != 0`); it is `0` otherwise. If `adstock_family` is
unavailable to `summarize_signal_metrics`, it conservatively reports
`l_max - 1` with burn-in. Weeks before a nonzero count have responses that
depend on unpersisted pre-window spend; it is a machine-readable warning that
their targets cannot be reconstructed from persisted inputs. It is unrelated to
`support_mask`, which selects temporal support versus query weeks. Without
burn-in, the zero-padded kernel makes every reported response reproducible.

The Weibull kernel is not a Weibull pdf. pymc-marketing min-max rescales its
density before sum-normalizing, which forces one or more lag weights to zero.
Accordingly, its persisted metadata is
`adstock_kernel_semantics="normalized-causal-minmax-weibull-density"` at
`adstock_kernel_version=3`.
Under the default prior, 45.0% of Weibull channels have zero current-week weight
and 38.5% peak at lag 5 or later; for those channels, `contributions[t]` is
independent of `channels[t]`. `frac_zero_contemporaneous_weight` reports the
fraction of eligible direct channels whose normalized current-week weight is
below `1e-9`; it is `None` when adstock metadata is absent or no channels are
eligible. It is diagnostic-only and is deliberately not part of `DEFAULT_GATE`.

`DEFAULT_GATE` additionally limits
`frac_contrib_rel_std_lt_001 <= 0.10`, the share of targets too small to matter
in loss units, and `frac_contrib_r2_gt_095 <= 0.10`, the share that is a
near-perfect linear combination of the baseline and other channels. A gate PASS
therefore now checks these amplitude and collinearity failure modes as well as
texture.

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
It also records `l_max`, `adstock_burn_in`, `response_warmup_weeks`,
`frac_zero_contemporaneous_weight`, `adstock_kernel_semantics`, and
`adstock_kernel_version`, so nondefault shards can be recomputed or assessed
without their original Python config object.

```python
from prior_generator.signal_diagnostics import SIGNAL_METRIC_LAYOUT

metrics = corpus["identifiability"]["signal_metrics"]
valid = corpus["identifiability"]["signal_metric_valid"].astype(bool)
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
