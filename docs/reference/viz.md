# Visualization

Matplotlib renders in two families. The bundle figures take an
[`SCM`](sampling.md#prior_generator.worlds.SCM) and a save path — the four
figures every bundle contains. The diagnostics figures take a
[`DataDiagnostics`](diagnostics.md) report and a save path. matplotlib is
imported lazily, and headless callers should select the `Agg` backend first
(`matplotlib.use("Agg")`).

Every function has the same shape — `plot_*(result, path, *, ...) -> None`: it
saves one figure to `path`, closes it, and returns nothing.

## Diagnostics figures

Five figures render a [`data_diagnostics`](diagnostics.md) report.

| figure | what it draws |
| --- | --- |
| `plot_dependence_matrices` | one heatmap per (view, metric) of the across-world quantile of the pairwise matrices |
| `plot_series_distributions` | one distribution panel per (view, series): raw retained values, or one of the eight per-world slots |
| `plot_temporal_diagnostics` | per view: an ACF heatmap, a forward lag-xi heatmap, and per-series median roughness / spike |
| `plot_vif_diagnostics` | per view and design scope (`observed`, `oracle`): the across-world quantile of VIF per predictor |
| `plot_contribution_diagnostics` | horizontal bars of one sibling cut of the sales contribution hierarchy |

### Paired views, and the levels-only warning

`views` defaults to **both** `("levels", "differences")`, so the default
figure always shows the level reading next to its difference companion.
Asking for a view the report was not built with raises `ValueError` naming the
missing view.

A figure the caller restricted to `views=("levels",)` carries
`viz.LEVELS_ONLY_NOTE` as a prominent figure-level banner: weekly series are
random-walk-like, so level dependence between series is large even when the
series are independent, and the differences view is the companion reading (not
a stationarity proof). A figure restricted to `views=("differences",)` is
labelled with `viz.DIFFERENCES_ONLY_NOTE` so the view is never guessed from
the numbers.

### Selectors

`views`, `keys` and `metrics` are exact-name, non-empty, unique sequences and
are honoured in caller order. A caller-passed empty sequence raises
`ValueError`, a bare string raises `TypeError`, and an unknown name raises
`ValueError` listing the valid names. `keys=None` (the default) selects every
key the report carries; `metrics` defaults to all four of
`("pearson", "spearman", "xi_max", "xi")`.

In the directional `xi` panel the **row is the predictor X and the column is
the target Y** — the panel title and axis labels say so, because the matrix is
deliberately asymmetric. `pearson` and `spearman` use a diverging map centred
at 0 with symmetric limits; `xi` and `xi_max` span the data including negative
values, which are ordinary small-sample noise around independence and are
never clamped into `[0, 1]`.

### Bounds

| figure | selection bound | figure cap |
| --- | --- | --- |
| `plot_dependence_matrices`, `plot_temporal_diagnostics`, `plot_vif_diagnostics` | `viz.MAX_HEATMAP_KEYS` = 64 keys | at most 24×24 inches |
| `plot_series_distributions` | `viz.MAX_SERIES_PANELS` = 24 panels (keys × views), grid at most 4 columns × 6 rows | at most 18×18 inches |
| `plot_contribution_diagnostics` | `viz.MAX_SERIES_PANELS` = 24 rows | at most 18×18 inches |

Over the bound the call raises `ValueError` telling you to narrow `keys`.
Above 32 keys the heatmap tick labels are thinned by a deterministic stride,
so the same report always renders the same labels.

`plot_contribution_diagnostics` selects its hierarchy reading with `basis`
(from `report.contribution_bases`) and raises `ValueError` naming the
available bases when that reading is not in the report; `sibling_set` picks
the cut and `keys` narrows the rows inside it.

### Truthful empty artists

Nothing missing is ever drawn as a zero.

- A masked heatmap cell is grey: the diagonal is never reported, and a pair
  with no valid world is annotated `N/A`.
- A panel with nothing to show is annotated in place —
  `No eligible observations`, `No difference observations`,
  `No valid observations`, `No valid lags`, `No predictors in scope` or
  `No active worlds for selected key(s)`.
- A **valid infinite VIF** (a predictor exactly collinear with the rest of the
  design) is drawn as a labelled marker on the **top edge of the axes**
  carrying its world count, never as a bar of height infinity — that bar would
  autoscale the axis and render as nothing. All-infinite is a result, not
  missing data, and the panel says so. A predictor with no valid world at all
  — constant in every world, or too few observations — gets a distinct `N/A`
  annotation instead.
- A row of the contribution figure with no value under the chosen population
  is annotated `N/A` with its world count; every value label carries the
  number of valid worlds behind it. Selecting a subset of a sibling cut makes
  the budget a partial projection, and the figure then states how many sibling
  rows are omitted and makes no claim that the bars add up.
- An **active structural zero** — a real series that is identically zero, or a
  contribution row that is genuinely zero — renders as the zero it is.

Every figure is descriptive: no p-values, no significance, no confidence
bands, no causal or forecast claims. The retained latent series (D, B) are
labelled *retained latent truth* — ground truth kept for auditing, never model
inputs.

::: prior_generator.viz
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - plot_dag
        - plot_timeseries
        - plot_decomposition
        - plot_channels
        - plot_outcome_distributions
        - plot_dependence_matrices
        - plot_series_distributions
        - plot_temporal_diagnostics
        - plot_vif_diagnostics
        - plot_contribution_diagnostics
