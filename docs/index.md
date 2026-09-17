---
hide:
  - navigation
  - toc
---

<div class="pg-hero" markdown>
<span class="pg-hero__eyebrow">structural causal models · exact ground truth</span>

# Simulate causal worlds. <span class="pg-gradient-text">Know the answer.</span>

Thousands of **worlds**, each a random Pearlian DAG compiled into one
PyTensor / PyMC graph — so a single draw gives you the observable data *and*
the exact causal decomposition of the outcome, reproducibly from a seed.

<div class="pg-hero__cta" markdown>
[Get started :material-arrow-right:](getting-started.md){ .pg-btn .pg-btn--primary }
[Read the model](guide/foundation.md){ .pg-btn .pg-btn--ghost }
[View on GitHub](https://github.com/pymc-labs/pymc-generator){ .pg-btn .pg-btn--ghost }
</div>
</div>

## From one seed, everything

<div class="pg-grid" markdown>

<div class="pg-card" markdown>
<div class="pg-card__icon">🎲</div>
### Draw a world
Every prior is a PyMC distribution; the treatment response mechanisms
(carryover, saturation) are pymc-marketing's own. One `pm.draw` yields the
series and the truth together.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">🧮</div>
### Exact decomposition
The outcome splits into per-treatment direct contributions, baseline drivers,
and a telescoping indirect split — holding to **float precision (~1e-15)**,
not a Taylor approximation.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">📂</div>
### Auditable bundles
Write a folder a human can read end-to-end: CSVs, a plain-text world
description, the DAG, and diagnostic figures.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">🧱</div>
### Stacked corpora
Stack thousands of worlds into one padded tensor `.npz` with explicit active
counts and masks — decomposition targets included.
</div>

</div>

## A world, drawn and audited

Here is one world sampled live during this docs build — its causal graph, its
observable series, and a machine-checked proof that the decomposition adds up.

```python exec="1" source="block" html="1"
from scm_docs import world, viz_html
import pymc_generator as pg

w = world(scenario=1, seed=0)          # "confounded_treatment"
print(viz_html(pg.viz.plot_dag, w, caption="The drawn causal graph (D confounds spend)."))
```

```python exec="1" source="block" html="1"
from scm_docs import world, viz_html
import pymc_generator as pg

w = world(scenario=1, seed=0)
print(viz_html(pg.viz.plot_timeseries, w, caption="What a modeller observes: spend, controls & demand, sales."))
```

```python exec="1" source="block" result="text"
from scm_docs import world
w = world(scenario=1, seed=0)
print("decomposition identity error:", f"{w.identity_error():.2e}")
print("reconstruction == sales:", bool((abs(w.reconstruction() - w.data['outcome']) < 1e-9).all()))
```

!!! tip "Everything on this site is live"
    Each figure and number above was produced by executing the code block right
    above it while the site was built. Nothing is pasted in.

## Where to go next

<div class="pg-grid" markdown>

<div class="pg-card" markdown>
<div class="pg-card__icon">⚡</div>
### [Getting started](getting-started.md)
Install, sample your first world, and write a bundle.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">🧠</div>
### [The causal model](guide/foundation.md)
The five node families, eight edge types, and structural equations.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">📓</div>
### [Examples notebook](examples/index.ipynb)
An end-to-end tour: sample, inspect, plot, validate.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">📖</div>
### [API reference](reference/index.md)
Every public symbol, generated from the source.
</div>

</div>
