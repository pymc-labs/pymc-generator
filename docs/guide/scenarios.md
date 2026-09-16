# Named scenarios

[`SCENARIOS`](../reference/scenarios.md) are five hand-built recipes, each
isolating one causal pathway so a decomposition failure can be traced to the
mechanism that broke. Each fixes graph sizes and per-edge-type arrow budgets.

| # | Scenario | Isolates |
| --- | --- | --- |
| 0 | `direct_only` | Pure `C→Y`; no interactions — indirect effects are exactly zero |
| 1 | `confounded_treatment` | Latent-unobserved factor drives both treatments (`D→C`) and outcome (`D→Y`) |
| 2 | `promo_drives_treatment` | Covariates push treatments (`Z→C`) and outcome (`Z→Y`); latent-unobserved factor moves covariates (`D→Z`) |
| 3 | `treatment_halo` | Treatment-to-treatment amplification (`C→C`); feeder & isolated-null treatments |
| 4 | `kitchen_sink` | Everything at once at sparse budgets — the hardest decomposition |

## Build a scenario's prior

```python exec="1" source="block" result="text"
import pymc_generator as pg

sc = pg.SCENARIOS[3]                               # treatment_halo
print("name       :", sc.name)
print("sizes      :", f"n_treatments={sc.n_treatments}, "
                      f"n_covariates={sc.n_covariates}, n_latent={sc.n_latent}")
print("connect_all:", sc.connect_all)
print("edge budget:", sc.edge_budget)

scm = pg.sample_scm(sc.prior(n_time_steps=104, seed=0), seed=0,
                    connect_all=sc.connect_all, name=sc.name, purpose=sc.purpose)
print("→ drew:", f"n_treatments={scm.n_treatments}, "
                 f"n_covariates={scm.n_covariates}, n_latent={scm.n_latent}")
```

## The gallery

One graph drawn from each scenario (seed 0). Notice how the live edge types shift
from scenario to scenario — that is the pathway each one isolates.

```python exec="1" source="block" html="1"
from scm_docs import world, viz_html
import pymc_generator as pg

for idx in range(len(pg.SCENARIOS)):
    sc = pg.SCENARIOS[idx]
    print(viz_html(pg.viz.plot_dag, world(scenario=idx, seed=0),
             title=f"{idx}: {sc.name}",
             caption=f"{sc.name} — {sc.purpose.split('.')[0]}."))
```

## Isolated nulls are deliberate

`treatment_halo` and `kitchen_sink` set `connect_all=False`, which permits
**fully-isolated null nodes** (no edges at all) as zero-attribution traps — a
model must credit them exactly nothing. Dead-end nodes (edges that never reach Y)
are *never* generated.

```python exec="1" source="block" result="text"
from scm_docs import world
from pymc_generator.worlds import node_status

scm = world(scenario=3, seed=0)                    # treatment_halo
status = node_status(scm.g)
print("node status:", status)
iso = [n for n, s in status.items() if s == "isolated"]
print("isolated nulls:", iso or "none this draw")
```

## Forcing connectivity substitutes an edge budget

`connect_all=True` rejects any draw with an isolated node, and for a scenario
designed around isolated nulls that can be infeasible: with `zc = zz = dz = 0`,
`treatment_halo`'s only route from a covariate to `Y` is its own `Z→Y` arrow, so its
`zy=(1, 1)` budget left the second covariate permanently isolated — 0.00% of draws
were feasible, and the CLI's `--require-path-to-y` died with
`RuntimeError: world 'treatment_halo': no DAG satisfying the connectivity rule in
2000 draws` after having already written folders `0`, `1`, `2`.

Scenarios therefore carry a second, connectivity-only budget in
`Scenario.connect_all_edge_budget`, and `Scenario.prior(..., connect_all=...)`
substitutes its entries over `edge_budget` **only when connectivity is forced**.
`treatment_halo` supplies `{"zy": (2, 2)}` (feasible fraction 0.00% → 17.07%) and
`kitchen_sink` supplies `{"cc": (3, 4)}` — with `cy = 4` of 6, the two feeder
treatments each need an outgoing halo arrow and a `(1, 2)` budget can draw just
one (0.85% → 5.89%). Default, unforced worlds are bit-identical to before.

`write_scenario_bundles` is also atomic: it pre-flights every scenario's graph
search before creating any directory and raises a single `RuntimeError` naming
all infeasible scenarios, leaving `out_root` nonexistent rather than
half-written.

## The whole set at once

`write_scenario_bundles` (and the CLI) writes every scenario to its own numbered
folder — see [Audit bundles](bundles.md#the-five-scenario-inspection-set).

```bash
pymc-generator --out inspection-datasets --seed 20260712
```

!!! tip "Roll your own"
    Scenarios are just presets. Build any world you like directly with
    [`make_scm_prior`](../reference/config.md#pymc_generator.presets.make_scm_prior)
    and its `edge_budget` — see [Generating a corpus](corpus.md#dialing-complexity).
