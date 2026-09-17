# Interrogating the generated data

A corpus can satisfy every schema contract, close its decomposition to machine
precision, and still be the wrong training data: the treatments may be
collinear, the latent-unobserved factors may dominate every series, the treatment may be a
smooth drift with no week-to-week signal, or treatments may account for 2% of outcome.
`data_diagnostics` is the blood panel for that question. One call produces one
report object, and every aspect of the data is a facet of it.

```python
import pymc_generator as pg

corpus = pg.sample_prior_predictive(cfg)
rep = pg.data_diagnostics(corpus, scopes=("nodes", "decomposition"))
```

```text
corpus | [SCM, ...]
      └─ data_diagnostics(...)  ──►  DataDiagnostics
                                       ├─ .outcomes        magnitudes and shares (the outcome-space companion)
                                       ├─ .contributions   who explains sales, in % and in sales units
                                       └─ .views["levels" | "differences"]
                                             ├─ .series      shape, spread and the raw values
                                             ├─ .temporal    ACF, lag-xi, roughness, spike
                                             ├─ .dependence  Pearson / Spearman / xi / xi-max
                                             └─ .vif         observed (C+Z) and oracle (C+Z+D)
```

## A worked pass

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=2,
    n_time_steps=48, n_cells=2, draws_per_cell=2, seed=606,
)
corpus = pg.sample_prior_predictive(cfg)
rep = pg.data_diagnostics(corpus, scopes=("nodes", "decomposition"))
print(rep.table())
```

The top cut is the first thing to read: five non-overlapping groups whose
signed shares sum to exactly 1 in every world. If treatment contribution is 3% you are
training a model to find something that is barely there; if `Y_noise` sits at
40% the target is mostly unlearnable.

### Is anything collinear?

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=2,
    n_time_steps=48, n_cells=2, draws_per_cell=2, seed=606,
)
rep = pg.data_diagnostics(pg.sample_prior_predictive(cfg))
print(rep["differences"].vif["observed"].table())
```

Two scopes are reported. `observed` holds the active `C` and `Z` series — the
redundancy a real model would face. `oracle` adds the latent `D`, so the gap
between them is exactly the redundancy hidden latent-unobserved factors inject into the design a
model cannot see. Read the **differences** view first: independent random walks
routinely produce a level VIF above 10 with no shared structure at all.

### What depends on what?

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=2,
    n_time_steps=48, n_cells=2, draws_per_cell=2, seed=606,
)
rep = pg.data_diagnostics(pg.sample_prior_predictive(cfg))
print(rep["levels"].dependence.table("xi_max", limit=8))
print()
print(rep["differences"].dependence.table("xi_max", limit=8))
```

Four metrics, all signed where a sign exists and all computed inside a world
before anything is aggregated:

* **Pearson** — linear co-movement.
* **Spearman** — monotone co-movement, robust to a single outlier week.
* **xi** — directional general dependence: how much the target is a *function*
  of the predictor, of any shape. For `Y = X²` the Pearson correlation is
  ≈0 while `xi(X → Y)` is ≈0.8.
* **xi-max** — the symmetric maximum of the two directions, for heatmaps.

Comparing the two views is the point. Dependence that survives differencing is
dependence between innovations; dependence that collapses was shared trend.

### How do the series move?

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=2,
    n_time_steps=48, n_cells=2, draws_per_cell=2, seed=606,
)
rep = pg.data_diagnostics(pg.sample_prior_predictive(cfg))
print(rep["levels"].temporal.table("acf", lags=(1, 2, 4, 8, 13, 24)))
print()
print(rep["levels"].series.table("roughness"))
```

`roughness` is 1 for white noise and → 0 for a smooth drift — the same
high-frequency ratio the [signal gate](../reference/signal.md) uses, applied to
every series rather than direct treatments only. `spike` is a robust
`max|Δ − median Δ| / IQR(Δ)`: it is ~1.7 for white noise and ~100 for a series
that is flat apart from one pulse, and it is `+inf` (a real, counted value, not
a missing one) when a single jump sits on an otherwise perfectly flat series.

`lag_xi` is the nonlinear companion to `acf`. A logistic map has ACF ≈ 0.06 at
lag 1 and lag-xi ≈ 0.94: fully deterministic, invisible to correlation. It is
lag *association*, never impact, and the overlapping pairs are not independent
observations.

### Drill into one world or one node

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=2,
    n_time_steps=48, n_cells=2, draws_per_cell=2, seed=606,
)
corpus = pg.sample_prior_predictive(cfg)
one = pg.data_diagnostics(corpus, worlds=[0])
print(one.contributions.table(sibling_set="treatment_children", measure="share"))
print()
print(one.contributions.select(("C1_direct_y", "C2_direct_y")).table(measure="total"))
```

A single world produces the same report shape as the whole corpus — one row per
world axis — and its numbers are exactly the corpus report's row for that
world. `select(...)` projects onto specific rows in caller order, and says so:
a projection is marked partial, lists what it omitted, and stops claiming its
rows add up.

## Plots

```python
from pymc_generator import viz

viz.plot_dependence_matrices(rep, "dependence.png")
viz.plot_series_distributions(rep, "distributions.png", of="value")
viz.plot_temporal_diagnostics(rep, "temporal.png")
viz.plot_vif_diagnostics(rep, "vif.png")
viz.plot_contribution_diagnostics(rep, "contributions.png")
```

Every figure defaults to showing both views side by side. Asking for levels
only is allowed and carries a visible trend warning; an infinite VIF is drawn
as a labelled top-edge marker rather than an invisible bar; a panel with no
eligible data is annotated as such rather than zero-filled — an active series
that is genuinely zero is a real zero and looks like one.

## What the report will not tell you

* **Nothing is inferential.** There are no p-values, no confidence bands and no
  significance marks anywhere, by design: worlds are correlated, lag pairs
  overlap, and the series are random-walk-like.
* **Nothing is causal.** `lag_xi` is not Granger causality and a high `xi` is
  not an effect. The causal answer already exists elsewhere — it is the exact
  [decomposition](decomposition.md).
* **Per-treatment contributions are direct-path.** The aggregate indirect terms
  cannot be split per node from a corpus, and the report never pretends they
  can.
* **`D` and `B` are retained ground truth**, kept so you can audit what a model
  cannot see. They are never model inputs.

## Cost and scale

Raw retention is `n_worlds × n_series × T` float64 values; pass
`keep_series=False` for very large corpora and every non-raw result stays
identical. Narrow the analysis with `worlds=`, `scopes=` and `views=` — the
`keys=` argument on the plots bounds rendering only, not the computation.

See the [API reference](../reference/diagnostics.md) for the exact estimator
definitions, the coverage-ledger invariants and the closure lattice.
