# prior-generator

Simulate thousands of synthetic marketing-mix-model (MMM) **worlds** — and get
their exact ground truth. Each world is an additive structural causal model over
latent demand, observed controls, media channels, a baseline, and sales, built
as a **PyTensor / PyMC** graph using **pymc-marketing**'s adstock and saturation
transforms. You draw from the graph to get the observable series *and* the exact
causal decomposition of sales (per-channel direct contributions, per-control and
per-confounder effects, and the telescoping indirect split), a plain-text
description of the world, an auditable dataset bundle, or a corpus in the format
PFN training pipelines consume.

> Status: **v0.0.1**, alpha. Extracted from
> [`pymc-labs/structural-pfn`](https://github.com/pymc-labs/structural-pfn) and
> refactored so the SCM is a drawable PyMC model.

## Install

The modeling stack is pinned to the validated combination and pymc-marketing is
tracked on its `v1.0.0` branch (not yet on PyPI), so install from source:

```bash
pip install "git+https://github.com/pymc-labs/prior-generator.git"
# or, for development:
git clone https://github.com/pymc-labs/prior-generator.git
cd prior-generator
pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
```

Requires Python ≥ 3.12 (the pinned `pymc` needs it). Core dependencies:
`pytensor`, `pymc`, `pymc-marketing`, `numpy`, `scipy`, `pandas`, `matplotlib`.

## Quickstart

### Sample one world and read its story

```python
import prior_generator as pg

# A named audit scenario (isolates one causal pathway), or build your own cfg.
scenario = pg.SCENARIOS[1]                       # "confounded_spend"
world = pg.sample_world(scenario.cfg(T=104, seed=0), seed=0,
                        name=scenario.name, purpose=scenario.purpose)

print(pg.describe_world(world))                  # DAG edges + coefficients,
                                                 # mechanisms, decomposition
                                                 # identity, signal metrics
print("decomposition error:", world.identity_error())   # ~1e-15
```

### Write an auditable bundle (CSVs + figures + description)

```python
pg.write_world_bundle(world, "my_world/")
# dataset.csv, true_components.csv, true_contribution.csv, description.txt,
# dag.dot, dag.png, timeseries.png, decomposition.png, channels.png
```

Or generate the full five-scenario inspection set from the command line:

```bash
prior-generator --out inspection-datasets --seed 20260712
```

### Generate a corpus for PFN training

```python
cfg = pg.make_world_config(K_max=8, M_max=4, J_max=2,
                              n_cells=50, draws_per_cell=20, seed=42)
corpus = pg.generate_corpus(cfg)                 # dict of numpy arrays
pg.save_corpus(corpus, "corpus.npz")             # the PFN-consumable format
loaded = pg.load_corpus("corpus.npz")
```

Every corpus satisfies, exactly (float64 pre-storage):

```
sales = baseline + Σ direct_contributions + indirect_effects
      = baseline_intrinsic + Σ confounder + Σ control + Σ direct + Σ indirect_by_source
```

## How it works

For each world, the **structure** is drawn concretely — the DAG (`sample_g_additive`,
with optional per-edge-type "budgets"), each channel's adstock/saturation family,
and each node's walk smoothness. Then `build_world_model` assembles a
`pymc.Model` in which every **continuous** parameter (edge coefficients, walk
means/stds, mechanism shapes, channel texture) is a PyMC distribution and every
noise term is an RV — `pm.Normal` walk innovations and weekly jitter, `pm.Bernoulli`
campaign pulses. The direct media response uses pymc-marketing's `geometric_adstock`
/ `weibull_adstock` and saturation curves. A single `pm.draw` yields the parameters,
the series, and the full interventional decomposition together, reproducibly from a
seed.

Complexity is dialed within a fixed schema via `make_world_config`
(graph size, edge budgets, `nonlinearity`, coefficient/noise ranges), and
`prior_generator.signal_diagnostics` gates whether a corpus carries learnable
signal.

## Public API

| Symbol | Purpose |
| --- | --- |
| `make_world_config` | Build a validated additive-rung `CorpusConfig`. |
| `generate_corpus` / `DataGenerator` | Generate an N-world corpus (dict of arrays). |
| `save_corpus` / `load_corpus` | Compressed `.npz` persistence (PFN-consumable). |
| `sample_world` / `World` | Draw one accepted world with its full ground truth. |
| `describe_world` | Plain-text description of a world. |
| `write_world_bundle` | Write the per-world auditable folder. |
| `SCENARIOS` | Five named audit scenarios, each isolating a pathway. |

## Development

```bash
ruff check . && ruff format --check .
mypy prior_generator
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Same seed + config must reproduce an
identical corpus.

## License

MIT — see [LICENSE](LICENSE).
