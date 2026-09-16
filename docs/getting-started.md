# Getting started

This page takes you from an installed package to an audited world. Run the Python
snippets in order in one session. World summaries and inline figures are generated
during site builds; shell commands and file-writing snippets run locally.

## Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the repository and use its checked-in dependency lock:

```bash
git clone https://github.com/pymc-labs/pymc-generator.git
cd pymc-generator
uv sync --frozen
uv run --no-sync python
```

For a released source archive, unpack it and run the same `uv sync` command
inside its project directory. The supported workflow does not yet depend on a
PyPI publication.

The lock selects released registry packages: PyMC **6.2.0**, PyTensor **3.2.4**,
pymc-marketing **1.1.0**, pymc-extras **0.14.0**, and PreliZ **0.27.1**.
No Git development companions are required. Project requirements allow
`pymc>=6.2,<7`, `pymc-marketing>=1.1,<2`, and `pytensor>=3.2.3,<4`;
pymc-marketing 1.1.0 further requires `pymc>=6.2,<6.3`, so do not force a newer
PyMC outside that intersection. Dependency upgrades can change numerical results
even with identical configurations and seeds. Use `uv lock --check` to check lock
freshness; do not run a blanket dependency upgrade to fix an installation problem.

!!! info "Environment requirements"
    Python **3.13 or newer** and uv **0.9.10 or newer**. CI selects Python 3.13
    and 3.14 explicitly. Core packages include NumPy, SciPy, pandas, Matplotlib,
    and the pinned modeling stack. PyTensor may compile native code; install
    your platform's C/C++ toolchain when needed. Graphviz's `dot` executable is
    needed for DAG image export and full documentation builds, not basic draws.

Development and documentation profiles are described in
[Contributing](https://github.com/pymc-labs/pymc-generator/blob/main/CONTRIBUTING.md).
After selecting a profile with `uv sync`, use `uv run --no-sync` so running a
command does not silently remove optional tools from that environment.

## Install with conda

Use the native channel built from this repository or, when available, unpacked
from a GitHub release's `conda-channel.tar.gz`. The channel contains only
`pymc-generator`; released modeling packages and all other native dependencies
come from conda-forge. Selected modeling, numerical, and diagnostic source
versions are pinned to match the uv lock. This does not require publication
of `pymc-generator` on PyPI or conda-forge.

From a checkout with a channel at `dist/conda-channel`:

```bash
conda create --name pymc-generator --override-channels --strict-channel-priority \
  --channel "file://$PWD/dist/conda-channel" --channel conda-forge \
  python=3.13 pymc-generator=0.0.2
conda activate pymc-generator
python -c "import pymc_generator as pg; print(pg.__version__)"
pymc-generator --help
```

For an unpacked release channel, substitute its absolute directory in the
`file://` URL. While no release channel is available, follow the
[native build instructions](https://github.com/pymc-labs/pymc-generator/blob/main/CONTRIBUTING.md#conda-artifacts).
This is a native conda install: it does not overlay a pip environment or build
private development companions.

Record a solved environment when reproducing a native run:

```bash
conda list --explicit --sha256 > conda-platform.lock
conda create --name pymc-generator-replay --file conda-platform.lock
```

An explicit conda lock is **platform-specific** and may contain absolute local
channel URLs; retain that channel and its packages. Native BLAS/compiler builds
and conda dependency resolution need not match the uv wheel environment, so
cross-environment bitwise equality is not a supported guarantee.

## 1 · Sample one world

A world is drawn from a **prior**. The quickest way to get a well-formed prior is
a named audit scenario, each of which isolates one causal pathway.

```python exec="1" session="quickstart" source="block" result="text"
import pymc_generator as pg

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

```python exec="1" session="quickstart" source="block" result="text"
text = pg.describe_scm(scm)
print("\n".join(text.splitlines()[:22]))          # first 22 lines
```

## 3 · See it

Every world renders four figures — the same ones a bundle writes to disk. Here is
the causal graph and the observable series:

```python
from pymc_generator.viz import plot_dag, plot_timeseries

plot_dag(scm, "dag.png")
plot_timeseries(scm, "timeseries.png")
```

```python exec="1" session="quickstart" html="1"
from scm_docs import viz_html
from pymc_generator.viz import plot_dag, plot_timeseries

print(viz_html(plot_dag, scm,
    caption="Nodes: D latent demand · Z controls · C channels · B intercept · Y sales."))
print(viz_html(plot_timeseries, scm,
    caption="Model inputs: spend, controls & latent demand, and sales."))
```

## 4 · Trust it

The reason to *simulate* rather than collect data is that you get the answer —
and it is exact. Sales equals the sum of its true components to float precision:

```python exec="1" session="quickstart" source="block" result="text"
print("max |Σ components − sales| =", f"{scm.identity_error():.2e}")
```

## 5 · Write an auditable bundle

Persist a world as a folder a human can inspect end-to-end — CSVs, the plain-text
description, the DAG (`.dot` + `.png`), and diagnostic figures.

```python
import pymc_generator as pg

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
pymc-generator --out inspection-datasets --seed 20260712
```

## 6 · Generate a training corpus

Stack many worlds into one padded tensor `.npz` archive — the decomposition
targets come baked in.

```python
import pymc_generator as pg

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
