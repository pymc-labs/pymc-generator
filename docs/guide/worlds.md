# Sampling a world

[`sample_scm`](../reference/sampling.md#prior_generator.worlds.sample_scm) draws
**one accepted world** at the config's max sizes and returns an
[`SCM`](../reference/sampling.md#prior_generator.worlds.SCM) — the object every
description, bundle, and figure consumes. This page is a tour of what is inside
it and how to reach every piece.

## Draw one

```python exec="1" source="material-block" result="text"
import prior_generator as pg

# Build a prior directly (or use a named scenario — see the Scenarios guide).
cfg = pg.make_scm_prior(n_treatments=5, n_covariates=3, n_latent=2,
                        edge_budget={"cy": (4, 4), "dc": (2, 2), "zc": (1, 2)},
                        n_time_steps=104, seed=7)
scm = pg.sample_scm(cfg, seed=7, name="tour")

print(type(scm).__name__, "→",
      f"n_treatments={scm.n_treatments}, n_covariates={scm.n_covariates}, "
      f"n_latent={scm.n_latent}, n_time_steps={scm.n_time_steps}")
```

`sample_scm` is deterministic in `(cfg, seed)`: the same pair reproduces the same
world bit-for-bit. It resamples the DAG until connectivity holds, then draws
candidate worlds until one passes the realism filter — all off a single RNG.

## What's inside: `.data`

The `.data` dict holds the world's 13 named series at active sizes. Channels,
controls, and demand are 2-D — `(n_time_steps, n_treatments)`,
`(n_time_steps, n_covariates)`, `(n_time_steps, n_latent)` — while
scalar-per-week series are 1-D `(n_time_steps,)`.

```python exec="1" source="material-block" result="text"
from scm_docs import world
scm = world(scenario=4, seed=0)          # kitchen_sink

for name, arr in scm.data.items():
    print(f"{name:<28} shape={str(arr.shape):<10} dtype={arr.dtype}")
```

| Group | Keys |
| --- | --- |
| **Observable** (what a modeller sees) | `channels`, `controls`, `sales` |
| **Direct truth** | `contributions` (per channel), `baseline`, `baseline_intrinsic` (the intercept), `sales_noise` |
| **Baseline split** | `control_contribution` (Z→Y), `confounder_contribution` (D→Y) |
| **Indirect truth** | `indirect_effects`, `indirect_effects_by_source` (cc, zc, dc) |
| **Latent / audit** | `demand`, `channels_base`, `contributions_observed`; with channel shocks enabled: `channels_unshocked`, `sales_unshocked`, and the `channel_shock_*` schedule |

`channels_unshocked` and `sales_unshocked` are single-world audit paths used by
the realism filter. They are not public top-level arrays in generated corpora.

## What's inside: `.g` (the DAG)

`.g` holds the active-size DAG blocks — one array per edge type.

```python exec="1" source="material-block" result="text"
from scm_docs import world
scm = world(4, 0)

for name, block in scm.g.items():
    print(f"{name:<7} shape={str(block.shape):<8} arrows={int(block.sum())}")
```

## What's inside: `.params`

`.params` holds the drawn coefficients, per-channel mechanism families, random-walk
parameters, and texture.

```python exec="1" source="material-block" result="text"
from scm_docs import world
scm = world(4, 0)

print("param keys:", sorted(scm.params)[:12], "…")
print("beta (direct C→Y):", scm.params["beta"].round(2))
print("adstock_family:", scm.params["adstock_family"])
print("sat_family:    ", scm.params["sat_family"])
```

## Introspection helpers

A handful of pure functions turn `.g` / `.params` into human-readable facts —
these are exactly what the description and DAG figure use.

```python exec="1" source="material-block" result="text"
from scm_docs import world
from prior_generator.worlds import (
    node_status, path_to_y, channel_role, mechanism_label, edges_with_coeffs,
)

scm = world(4, 0)
print("node status :", node_status(scm.g))
print("reaches Y   :", path_to_y(scm.g))
for k in range(scm.n_treatments):
    print(f"  C{k+1}: role={channel_role(scm.g, k):<7} mechanism={mechanism_label(scm.params, k)}")
print("active edges:", len(edges_with_coeffs(scm.g, scm.params)))
```

## Built-in checks

The `SCM` object carries three methods for validation and signal — no external
code needed.

```python exec="1" source="material-block" result="text"
from scm_docs import world
scm = world(4, 0)

# 1. reconstruction() — Σ of all true components (should equal sales)
recon = scm.reconstruction()

# 2. identity_error() — max |reconstruction − sales|
print("identity error :", f"{scm.identity_error():.2e}")

# 3. signal() — per-direct-channel signal metrics
sig = scm.signal()
print("direct channels:", (sig["channel"].astype(int) + 1).tolist())
print("spend_cv       :", sig["spend_cv"].round(2).tolist())
print("contrib_cv     :", sig["contrib_cv"].round(2).tolist())
print("spearman       :", sig["spearman"].round(2).tolist())
```

## Straight to a DataFrame

`.data` is plain numpy, so anything you would do with a MMM dataset works. Here we
assemble the *model-input* table — precisely what `dataset.csv` in a bundle
contains.

```python exec="1" source="material-block" result="text"
from scm_docs import world
import numpy as np, pandas as pd

scm = world(1, 0)
d = scm.data
df = pd.DataFrame({"week": np.arange(scm.n_time_steps)})
for k in range(scm.n_treatments):
    df[f"spend_C{k+1}"] = d["channels"][:, k]
for m in range(scm.n_covariates):
    df[f"control_Z{m+1}"] = d["controls"][:, m]
df["sales_Y"] = d["sales"]

print(df.head().to_string(index=False))
```

!!! tip "Next"
    [The exact decomposition](decomposition.md) unpacks the *truth* columns and
    proves they sum to sales.
