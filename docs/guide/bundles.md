# Audit bundles

A **bundle** is a folder a human can audit end-to-end. It is the artifact the
`prior-generator` CLI writes, and the most direct way to hand a world to someone
who wants to *read* it rather than load it.

## Write one

```python
import prior_generator as pg

scm = pg.sample_scm(pg.SCENARIOS[1].prior(n_time_steps=104, seed=0), seed=0,
                    name="confounded_spend")
pg.write_scm_bundle(scm, "my_world/")
```

Here is that call run live against a temporary directory, listing exactly what it
produced:

```python exec="1" source="material-block" result="text"
import tempfile, os
from scm_docs import world
import prior_generator as pg

scm = world(scenario=1, seed=0)
out = pg.write_scm_bundle(scm, os.path.join(tempfile.mkdtemp(), "confounded_spend"))

for f in sorted(os.listdir(out)):
    size = os.path.getsize(os.path.join(out, f))
    print(f"  {f:<24} {size:>7,d} bytes")
```

## What each file is for

| File | What it holds |
| --- | --- |
| `dataset.csv` | What a model eats: `week`, `spend_C*`, `control_Z*`, `sales_Y` |
| `true_components.csv` | The full additive truth — `baseline_intrinsic`, `sales_noise`, the baseline split, per-channel direct contributions, indirect by source, latent demand, base channels |
| `true_contribution.csv` | Legacy-compatible view (`contribution_C*`, `baseline_B`) |
| `description.txt` | DAG + drawn coefficients, mechanisms, decomposition check, signal metrics |
| `dag.dot` / `dag.png` | The causal graph (matplotlib render — no graphviz needed) |
| `timeseries.png` | Model-input series |
| `decomposition.png` | Every true effect on Y + reconstruction check |
| `channels.png` | Per-channel spend vs true contribution (indexed) |

Every column of `true_components.csv` except `week`, `sales_reconstructed` and
the `demand_*` / `channel_base_*` diagnostics is an additive term of
`dataset.csv`'s `sales_Y`, and they sum to it **exactly**. That includes
`sales_noise`, the iid observation-noise column: without it the additive columns
summed to sales minus the noise, so an auditor adding them up saw a residual of
`max|sales_noise|` (0.07–0.18 on the shipped scenarios) while `description.txt`
reported a decomposition error around `1e-15`.

## The description

`description.txt` is the world's story as plain text. The first lines:

```python exec="1" source="material-block" result="text"
from scm_docs import world
import prior_generator as pg

text = pg.describe_scm(world(1, 0))
print("\n".join(text.splitlines()[:28]))
```

## The figures

`write_scm_bundle` renders four figures. Two more (the DAG and time series) appear
on other pages; here are the per-channel and decomposition views a bundle
contains:

```python exec="1" source="material-block" html="1"
from scm_docs import world, viz_html
import prior_generator as pg

print(viz_html(pg.viz.plot_channels, world(1, 0),
         caption="channels.png — each channel's spend (grey) vs its true contribution (color), indexed to mean 1."))
```

## The five-scenario inspection set

`write_scenario_bundles` writes one numbered folder per scenario plus a root
`README.md`. This is what the CLI produces:

```bash
prior-generator --out inspection-datasets --seed 20260712
# README + 5 datasets under inspection-datasets
```

```python
import prior_generator as pg

paths = pg.write_scenario_bundles("inspection-datasets", seed=20260712)
# inspection-datasets/
#   README.md
#   0/  direct_only       1/  confounded_spend    2/  promo_drives_spend
#   3/  channel_halo      4/  kitchen_sink
```

Scenario `idx` is sampled with `seed + idx`, so the whole set is fully
deterministic given `(scenarios, n_time_steps, seed)`.

!!! info "Text/CSV-only bundles"
    Pass `plots=False` to skip the PNG rendering (and the matplotlib import) when
    you only need the data and description.

See the [API reference](../reference/bundles.md) for the full signatures.
