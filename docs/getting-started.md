# Getting started

This page takes you from an installed package to an audited world. Run the Python
snippets in order in one session. World summaries and inline figures are generated
during site builds; shell commands and file-writing snippets run locally.

## Install

The modeling stack is pinned to a validated combination — `pymc-marketing` is
not yet on PyPI and is pinned to an exact commit so installs stay reproducible —
so install from source:

```bash
pip install "git+https://github.com/pymc-labs/prior-generator.git"
```

Or, for development:

```bash
git clone https://github.com/pymc-labs/prior-generator.git
cd prior-generator
pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
```

!!! info "Requirements"
    Python **≥ 3.12** (the pinned `pymc` needs it). Core dependencies:
    `pytensor`, `pymc`, `pymc-marketing`, `numpy`, `scipy`, `pandas`,
    `matplotlib`. Public symbols are lazy-loaded, so `import prior_generator`
    stays light until you touch something that needs the heavy stack.

## 1 · Sample one world

A world is drawn from a **prior**. The quickest way to get a well-formed prior is
a named audit scenario, each of which isolates one causal pathway.

```python exec="1" session="quickstart" source="material-block" result="text"
import prior_generator as pg

scenario = pg.SCENARIOS[1]                       # "confounded_spend"
scm = pg.sample_scm(
    scenario.prior(n_time_steps=104, seed=0), seed=0,
    connect_all=scenario.connect_all,
    name=scenario.name, purpose=scenario.purpose,
)
print(f"drew world {scm.name!r}: {scm.n_treatments} channels, "
      f"{scm.n_covariates} controls, {scm.n_latent} demand factors, "
      f"{scm.n_time_steps} weeks")
```

## 2 · Read its story

`describe_scm` renders everything you need to audit a world as plain text — its
DAG with drawn coefficients, node connectivity, per-channel mechanisms, the
decomposition-identity check, and signal metrics.

```python exec="1" session="quickstart" source="material-block" result="text"
text = pg.describe_scm(scm)
print("\n".join(text.splitlines()[:22]))          # first 22 lines
```

## 3 · See it

Every world renders four figures — the same ones a bundle writes to disk. Here is
the causal graph and the observable series:

```python
from prior_generator.viz import plot_dag, plot_timeseries

plot_dag(scm, "dag.png")
plot_timeseries(scm, "timeseries.png")
```

```python exec="1" session="quickstart" html="1"
from scm_docs import viz_html
from prior_generator.viz import plot_dag, plot_timeseries

print(viz_html(plot_dag, scm,
    caption="Nodes: D latent demand · Z controls · C channels · B intercept · Y sales."))
print(viz_html(plot_timeseries, scm,
    caption="Model inputs: spend, controls & latent demand, and sales."))
```

## 4 · Trust it

The reason to *simulate* rather than collect data is that you get the answer —
and it is exact. Sales equals the sum of its true components to float precision:

```python exec="1" session="quickstart" source="material-block" result="text"
print("max |Σ components − sales| =", f"{scm.identity_error():.2e}")
```

## 5 · Write an auditable bundle

Persist a world as a folder a human can inspect end-to-end — CSVs, the plain-text
description, the DAG (`.dot` + `.png`), and diagnostic figures.

```python
import prior_generator as pg

pg.write_scm_bundle(scm, "my_world/")
# my_world/
#   dataset.csv            model inputs (week, spend_C*, control_Z*, sales_Y)
#   recipe.json            configuration, seed, and replay instructions
#   true_components.csv    the full additive decomposition truth
#   description.txt        the world's story
#   dag.dot / dag.png      the causal graph
#   timeseries.png  decomposition.png  channels.png
```

Or generate the full five-scenario inspection set from the command line:

```bash
prior-generator --out inspection-datasets --seed 20260712
```

## 6 · Generate a training corpus

Stack many worlds into the tensor `.npz` format amortized-inference / PFN
pipelines consume — the decomposition targets come baked in.

```python
import prior_generator as pg

cfg = pg.make_scm_prior(n_treatments=8, n_covariates=4, n_latent=2,
                        n_cells=2, draws_per_cell=2, n_time_steps=32, seed=42)
corpus = pg.sample_prior_predictive(cfg)          # arrays plus diagnostic metadata
pg.save_corpus(corpus, "corpus.npz")              # compressed, self-describing
loaded = pg.load_corpus("corpus.npz")
```

## Where to next

<div class="pg-grid" markdown>

<div class="pg-card" markdown>
<div class="pg-card__icon">🧠</div>
### [The causal model](guide/foundation.md)
Understand the five node families, eight edge types, and the structural
equations behind every world.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">🧮</div>
### [The exact decomposition](guide/decomposition.md)
How sales splits — exactly — into direct, baseline, and indirect effects, and
how to verify it yourself.
</div>

<div class="pg-card" markdown>
<div class="pg-card__icon">📓</div>
### [Examples notebook](examples/index.ipynb)
A runnable end-to-end tour, including several time series and a full validation
pass.
</div>

</div>
