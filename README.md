# prior-generator

Simulate thousands of synthetic marketing-mix-model (MMM) **worlds** — and get
their *exact* ground truth.

Each world is an **additive structural causal model (SCM)** over latent demand,
observed controls, media channels, a baseline, and sales. It is built as a single
**PyTensor / PyMC** graph — every prior is a PyMC distribution and every mechanism
(adstock, saturation) comes from **pymc-marketing** — so one `pm.draw` yields the
observable series *and* the exact causal decomposition of sales together,
reproducibly from a seed.

From one world you can get:

- the **observable data** a modeller would see (spend, controls, sales),
- the **exact decomposition** of sales into per-channel direct contributions,
  per-control and per-confounder baseline effects, and the telescoping indirect
  split — holding to float precision, not a Taylor approximation,
- a **plain-text description** of the world (its DAG, coefficients, mechanisms,
  and signal diagnostics),
- an **auditable dataset bundle** (CSVs + figures + description), or
- a **corpus** of many worlds in the tensor format amortized-inference / PFN
  training pipelines consume.

> **Status: v0.0.1, alpha.** Extracted from
> [`pymc-labs/structural-pfn`](https://github.com/pymc-labs/structural-pfn) and
> refactored so the SCM is a drawable PyMC model.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Foundation: the structural causal model](#foundation-the-structural-causal-model)
- [How a world is created](#how-a-world-is-created)
- [The exact decomposition](#the-exact-decomposition)
- [Assumptions and design choices](#assumptions-and-design-choices)
- [Dialing complexity](#dialing-complexity)
- [Outputs](#outputs)
- [Named audit scenarios](#named-audit-scenarios)
- [Public API](#public-api)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Foundation libraries](#foundation-libraries)

## Install

The modeling stack is pinned to a validated combination — pymc-marketing is not
yet on PyPI and is pinned to an exact commit (from its `v1.0.0` line, not the
moving branch) so installs stay reproducible — so install from source:

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

# A named audit scenario (isolates one causal pathway), or build your own prior.
scenario = pg.SCENARIOS[1]                        # "confounded_spend"
scm = pg.sample_scm(scenario.prior(T=104, seed=0), seed=0,
                    connect_all=scenario.connect_all,
                    name=scenario.name, purpose=scenario.purpose)

print(pg.describe_scm(scm))          # DAG edges + coefficients, mechanisms,
                                     # decomposition identity, signal metrics
print("decomposition error:", scm.identity_error())   # ~1e-15
```

### Write an auditable bundle (CSVs + figures + description)

```python
pg.write_scm_bundle(scm, "my_world/")
# dataset.csv, true_components.csv, true_contribution.csv, description.txt,
# dag.dot, dag.png, timeseries.png, decomposition.png, channels.png
```

Or generate the full five-scenario inspection set from the command line:

```bash
prior-generator --out inspection-datasets --seed 20260712
```

### Generate a corpus for training

```python
cfg = pg.make_scm_prior(n_treatments=8, n_covariates=4, n_latent=2,
                        n_cells=50, draws_per_cell=20, seed=42)
corpus = pg.sample_prior_predictive(cfg)          # dict of numpy arrays
pg.save_corpus(corpus, "corpus.npz")              # compressed, self-describing
loaded = pg.load_corpus("corpus.npz")
```

Every corpus satisfies, exactly (float64 pre-storage):

```
sales = baseline + Σ direct_contributions + indirect_effects
      = baseline_intrinsic + Σ confounder + Σ control + Σ direct + Σ indirect_by_source
```

## Foundation: the structural causal model

Every world is a **Pearlian additive SCM** over five families of nodes. The whole
point of the design is the distinction between what a modeller *observes* and what
*confounds* them:

| Node | Symbol | Role | Observed? | Sign |
| --- | --- | --- | --- | --- |
| Latent demand | `D₁…D_J` | Hidden confounders — the classic MMM bias source | **No** | signed |
| Controls | `Z₁…Z_M` | Observed covariates (promo calendar, price, seasonality…) | Yes | signed |
| Media channels | `C₁…C_K` | The treatments/interventions; **spend is observed** | Yes | ≥ 0 |
| Baseline | `B` | Organic demand level; always feeds sales | (latent) | signed |
| Sales | `Y` | The outcome (a sink) | Yes | ≥ 0 |

Nodes are connected by a directed acyclic graph drawn from **eight edge types**.
The type of an edge determines *where it enters* and therefore *what it does*:

```mermaid
flowchart LR
    D["D — latent demand<br/>(hidden confounders)"]
    Z["Z — observed controls"]
    C["C — media channels<br/>(spend, observed)"]
    B["B — baseline"]
    Y["Y — sales"]

    D -- "dc" --> C
    D -- "dz" --> Z
    D -- "db" --> B
    Z -- "zc" --> C
    Z -- "zb" --> B
    Z -. "zz" .-> Z
    C -. "cc" .-> C
    C == "cy — nonlinear:<br/>adstock + saturation" ==> Y
    B --> Y

    classDef latent fill:#eee,stroke:#999,stroke-dasharray:4 3;
    class D,B latent;
```

| Type | Edge | Meaning | Effect on Y |
| --- | --- | --- | --- |
| `cy` | C → Y | **Direct** media response (adstock + saturation), coeff `βₖ` | direct |
| `db` | D → B | Demand lifts the baseline | direct (baseline) |
| `zb` | Z → B | Controls lift the baseline | direct (baseline) |
| `dc` | D → C | Demand drives spend — **confounds** attribution | indirect |
| `zc` | Z → C | Controls drive spend (e.g. promo triggers media) | indirect |
| `cc` | C → C | Channel **halo** (upstream channels amplify downstream) | indirect |
| `dz` | D → Z | Demand moves the controls | via Z (baseline and/or `zc`) |
| `zz` | Z → Z | Control chains | via Z (baseline and/or `zc`) |

The structural equations (`t` indexes weeks; each term is gated by whether the
corresponding edge exists in the drawn DAG):

```text
D_j = RW_j                                                  (confounder — a random walk)
Z_m = Σ_j u_jm·D_j + Σ_{m'<m} γ_{m'm}·Z_{m'} + RW_m         (control)
E_k = RW_k + σ_k·ε_tk + a_k·b_tk,  b_tk ~ Bernoulli(p_k)    (channel's own exogenous drive)
C_k = softplus( Σ_j w_jk·D_j + Σ_m v_mk·Z_m
                + Σ_{k'<k} α_{k'k}·C_{k'} + E_k )            (channel spend — non-negative)
B   = Σ_j δ_j·D_j + Σ_m ρ_m·Z_m + RW_B                      (baseline)
Y   = B + Σ_k g_cy·β_k·f_k(C_k) + RW_Y                      (sales)
```

Two structural facts do the heavy lifting:

1. **Only the direct `C → Y` path is nonlinear.** `f_k` is that channel's
   adstock ⊙ saturation response. **Every other edge is linear** — all the
   input→input interactions (`dc, zc, cc, dz, zz`) and the baseline drivers
   (`db, zb`) are plain linear loadings. This keeps the interaction structure
   interpretable while the media response stays realistically curved.
2. **Every node carries its own random-walk noise term**, and channels
   additionally carry high-frequency drive — iid weekly execution noise `σ_k·ε`
   and campaign pulses `a_k·b` (a Bernoulli fire). That high-frequency variation
   is what makes spend *sweep* its response curve; without it, the contribution
   targets degenerate to flat lines.

## How a world is created

A world is drawn in **two stages** — structure first (concrete, with numpy), then
parameters and noise (as a PyMC model). This mirrors the design principle:
*discrete structure is drawn per world; continuous priors are distributions.*

### Stage 1 — draw the structure (concrete)

`sample_g_additive` + `sample_structure` draw everything that fixes the graph's
*shape*:

- **Active sizes** — how many channels / controls / demand factors are live this
  task (drawn from the `*_active_range`s; inactive slots are zero-padded and
  masked, so tensor shapes stay fixed).
- **The DAG** `g` — which of the 8 edge types connect which nodes. Each type is
  either drawn **per-pair Bernoulli** at its base rate, or scattered from a
  per-type **arrow budget** (a "pot" — see [edge budgets](#interactions-the-edge-budget)).
  `cc` and `zz` are restricted to the **strict upper triangle** (`src < dst`),
  which guarantees acyclicity.
- **Per-channel mechanisms** — each channel's adstock family
  (`none / geometric / weibull`) and saturation family
  (`none / hill / logistic / michaelis_menten / tanh / root`), drawn from the
  family-probability simplexes.
- **Per-node walk smoothness** — a Beta prior. (The channel texture-*enable*
  flags are not drawn; they are fixed by whether the config's texture ranges are
  non-zero.)

In the **single-world** path (`sample_scm`), the DAG is resampled until it
satisfies a connectivity rule: **dead-end nodes (edges that never reach Y) are
never allowed**, and fully-isolated null nodes are permitted (as deliberate
zero-attribution traps) unless `connect_all=True`. The **corpus** path
(`sample_prior_predictive`) draws each cell's structure once, *without* this
filter — training tasks may contain dead-end or isolated nodes as they fall out of
the base rates and budgets.

### Stage 2 — draw parameters and noise (a PyMC model)

`build_world_model` assembles one `pm.Model` for that fixed structure in which
**every continuous quantity is a random variable**:

- **Edge coefficients** — `pm.Uniform` over the per-edge-type coefficient ranges
  (`w_dc`, `v_zc`, `α_cc`, `β`, …).
- **Random walks** — `pm.Uniform`/`pm.HalfNormal` means and stds, `pm.Normal`
  innovations; the walk's smoothness sets a moving-average kernel width.
- **Mechanism shapes** — adstock decay / Weibull shape and the saturation shape
  priors (`SATURATION_PRIOR_RANGES`).
- **Channel texture** — weekly-jitter `pm.Normal` and campaign-pulse
  `pm.Bernoulli`, with magnitudes scaled relative to each channel's own level.

Every graph output is registered as a `pm.Deterministic`, so a single **`pm.draw`**
returns the parameters, the series, and the full decomposition jointly. Candidate
draws are run through the [realism filter](#the-realism-filter); the first accepted
draw is kept.

### Corpus mode: cells

`sample_prior_predictive` scales this up in **cells**. Each cell fixes one
structure and draws `draws_per_cell` worlds from it (so tasks within a cell share
a DAG but vary in coefficients and noise); `n_cells` cells give
`N = n_cells × draws_per_cell` tasks. Active sizes are drawn per cell, and the
train/validation split is made at the **cell** level (`val_cell_frac`) so no
structure leaks across the split.

Everything is driven by **one numpy RNG** seeded from `cfg.seed` — it drives the
structure draws *and* the per-round `pm.draw` seeds — so `(cfg, seed)` reproduces
a world (or an entire corpus) bit-for-bit.

## The exact decomposition

The reason to simulate rather than collect data is that you get the answer. Each
world reports how every dollar of sales was actually produced, and the pieces sum
to sales **exactly** (≈1e-15 in float64):

```text
contributions_k  = g_cy·β_k · f_k(C_base_k)                    (direct)
indirect_effects = Σ_k g_cy·β_k · ( f_k(C_k) − f_k(C_base_k) ) (interaction-routed)

sales = baseline + Σ_k contributions_k + indirect_effects
```

`C_base` is the channel system under the **intervention "zero all incoming channel
interactions"** (`D→C`, `Z→C`, `C→C` removed) — each channel driven by its own
walk and drive alone. So `contributions_k` is what channel *k* would have produced
on its own, and `indirect_effects` is the *extra* sales explained by spend that was
itself moved by demand, controls, or other channels.

The identity is exact — **not** a Taylor approximation — because `Y` and the
decomposition are built from the *same* symbolic quantities: `f_k` is one fixed
function evaluated on two inputs, and the κ-relative saturation scale is computed
once (from the observed channel) and reused for the base channel.

`indirect_effects` is further split, by sequential graph surgery, into a
**telescoping 3-way attribution** in the locked order `(cc, zc, dc)` —
channel→channel, control→channel, demand→channel. The three columns sum exactly
to `indirect_effects`. (`dz` and `zz` change the observed controls, so their
effect on channels is carried by the `zc` column.)

Finally, the baseline itself splits per node, so the corpus also satisfies the
fully-decomposed identity:

```text
sales = baseline_intrinsic                 (organic level, walks only)
      + Σ_j confounder_contribution_j      (D → B)
      + Σ_m control_contribution_m         (Z → B)
      + Σ_k contributions_k                (direct C → Y)
      + Σ_s indirect_effects_by_source_s   (s ∈ {cc, zc, dc})
```

All four invariants are computed at generation time and reported in
`corpus["diagnostics"]` (`decomposition_max_abs_error`, `telescoping_…`,
`full_decomposition_…`, `baseline_decomposition_…`); the test suite asserts them
directly.

## Assumptions and design choices

These are the modeling commitments baked into the generator. They are deliberate,
and they bound what a model trained on this data can be expected to learn.

- **Additive sales.** Sales is a *sum* of a baseline and per-channel media
  contributions (plus noise), not a multiplicative model. Media contributions add;
  they do not scale the baseline.
- **Linear interactions, nonlinear direct response.** All input→input edges and
  baseline drivers are linear loadings; only the direct `C → Y` media path carries
  adstock and saturation. Interaction *structure* is rich; interaction *shape* is
  linear.
- **Spend is non-negative.** Channels pass through `softplus`, so spend stays ≥ 0
  even when signed upstream terms push the pre-activation below zero.
- **Latent demand is never observed.** `D` is the hidden confounder that drives
  both spend (`dc`) and the baseline (`db`) — the exact mechanism that biases
  naive attribution. Controls `Z` *are* observed. Getting attribution right in the
  presence of `D` is the core task.
- **Acyclicity by construction.** `C→C` and `Z→Z` live on the strict upper
  triangle (`src < dst`), so the graph is always a DAG.
- **κ-relative saturation.** Each saturation curve's knee is set relative to the
  channel's own mean adstocked level, so every channel operates in a meaningful
  regime regardless of its scale — and the same pinned scale is reused across the
  decomposition so the identity holds exactly.
- **Adstock burn-in.** The adstock convolution left-pads with zeros, which would
  make early weeks ramp up artificially. Worlds simulate `T + adstock_burn_in`
  weeks and report the last `T`, so the reported window sees real history
  (`adstock_burn_in ≥ l_max`).
- **Realism filter.** A drawn world is only accepted if it *looks like data a
  modeller would actually get* — see below.
- **Learnable signal.** Contribution targets should carry real week-to-week
  variation, not flat lines. Every corpus reports signal metrics in its
  diagnostics, and `signal_diagnostics.check_signal_gate` is an opt-in check a
  caller can run to reject a too-flat corpus (see
  [Reproducibility and validation](#reproducibility-and-validation)).

### The realism filter

Every candidate draw must pass `_additive_task_ok` or it is rejected and redrawn:

1. **Finiteness** — every series is all-finite.
2. **Non-negative sales** — no `sales < 0`.
3. **Spend-CV floor** — each active direct channel's coefficient of variation
   ≥ `spend_cv_floor` (default `0.08`); a channel that never moves teaches nothing.
4. **Sales spike guard** — `max(sales) / median(sales) < 8`.
5. **Spend spike guard** — per channel, `max / median < 50`.

In corpus mode, a draw is additionally rejected if its support-window sales
standard deviation is not finite and positive (it sets the `sales_norm` scale).

## Dialing complexity

The SCM *structure* is fixed (the 8 edge types, additive equations, exact
decomposition). `make_scm_prior` dials *complexity* within that fixed structure
along orthogonal axes, keeping tensor shapes identical so one model / eval harness
serves every level:

| Axis | How | Knobs |
| --- | --- | --- |
| **Graph size** | how many nodes are live | `n_treatments`, `n_covariates`, `n_latent`, and their `*_active_range`s |
| **Interactions** | how many arrows of each type | `edge_budget` (per-type "pot") |
| **Nonlinearity** | media-response family mix | `nonlinearity="diverse"` / `"linear"` |
| **Signal / noise** | coefficient & noise ranges | `**overrides` (e.g. `rw_sales_std_sigma`, coefficient ranges) |
| **Texture** | channels' high-frequency drive | `texture="diverse"` |

### Interactions: the edge budget

`edge_budget` maps an edge type to an **arrow budget** — an "up to N" pot scattered
uniformly over the eligible node pairs, rather than a per-pair coin flip:

```python
cfg = pg.make_scm_prior(
    n_treatments=6, n_covariates=4, n_latent=2,
    edge_budget={
        "cy": (4, 4),   # exactly 4 direct channels
        "dc": (2, 2),   # exactly 2 demand→spend confounding arrows
        "zc": (1, 3),   # 1–3 control→spend arrows
        "cc": (1, 2),   # 1–2 halo arrows
    },
)
```

- An **int** `N` means "up to N" — the per-task count is drawn uniformly in
  `{0..N}`. A **tuple** `(lo, hi)` draws in the inclusive range; use `(N, N)` for
  exactly `N`.
- Counts are capped at the number of **eligible pairs** for that type.
- Edge types **omitted** from the dict keep their per-pair Bernoulli base rate —
  budgeting `zc` leaves `zb` untouched.
- `cy` keeps its `≥ 1` floor (a world always has at least one direct channel).

## Outputs

The generator produces three kinds of artifact.

### 1. A training corpus — `sample_prior_predictive`

A dict of stacked numpy arrays over `N` tasks and `T` weeks, with padded max sizes
`K / M / J`. Internal math is float64; storage is float32 (floats), `uint8`
(masks), `int32` (counts). The most important keys:

| Key | Shape | What |
| --- | --- | --- |
| `spend_raw` | (N, T, K) | media spend (observed input) |
| `controls` | (N, T, M) | observed controls |
| `sales_raw` | (N, T) | sales (observed target) |
| `contributions_raw` | (N, T, K) | per-channel **direct** contributions (truth) |
| `indirect_effects` | (N, T) | total interaction-routed effect (truth) |
| `indirect_effects_by_source` | (N, T, 3) | telescoping split, order `(cc, zc, dc)` |
| `baseline_raw` | (N, T) | full baseline |
| `demand` | (N, T, J) | latent demand series (truth) |
| `g` | (N, S) | the packed DAG (all 8 edge blocks) |
| `active_c_mask` / `active_m_mask` / `active_j_mask` | (N, K/M/J) | which slots are live |
| `diagnostics` | dict | edge marginals, decomposition errors, signal block |

<details>
<summary>Full corpus schema (all keys)</summary>

Observables & normalizations: `spend_raw`, `spend_norm`, `spend_share`,
`spend_means`, `controls`, `sales_raw`, `sales_norm`, `sales_scale`.
Ground truth: `contributions_raw`, `baseline_raw`, `baseline_intrinsic`,
`control_contribution` (N,T,M), `confounder_contribution` (N,T,J), `demand`,
`indirect_effects`, `indirect_effects_by_source` (N,T,3, order `cc,zc,dc`).
Structure & masks: `g` (N,S packed), `active_c_mask`/`active_m_mask`/`active_j_mask`,
`channel_active`, `K_active`/`M_active`/`J_active`.
Splits & bookkeeping: `support_mask`, `is_future`, `is_val`, `cell_id`.
Plus `diagnostics` (a dict, round-tripped through the `.npz` as JSON).

</details>

### 2. A single world — `sample_scm` → `SCM`

An `SCM` object bundles the active-size series (`.data`), the active DAG blocks
(`.g`), the drawn parameters (`.params`), and helpers: `.identity_error()`,
`.reconstruction()`, `.signal()`, and size properties. This is the object
`describe_scm`, `write_scm_bundle`, and the plotting layer consume.

### 3. An audit bundle — `write_scm_bundle` / `write_scenario_bundles`

A folder a human can inspect: `dataset.csv` (the observable inputs),
`true_components.csv` (the full per-node decomposition), `true_contribution.csv`,
`description.txt` (`describe_scm`), `dag.dot`, and — unless disabled — `dag.png`,
`timeseries.png`, `decomposition.png`, `channels.png`.

## Named audit scenarios

`SCENARIOS` are five hand-built recipes, each isolating one causal pathway so a
decomposition failure can be traced to the mechanism that broke:

| # | Scenario | Isolates |
| --- | --- | --- |
| 0 | `direct_only` | Pure `C→Y`; no interactions — indirect effects are exactly zero |
| 1 | `confounded_spend` | Latent demand drives both spend (`D→C`) and baseline (`D→B`) |
| 2 | `promo_drives_spend` | Controls push spend (`Z→C`) and baseline (`Z→B`); demand moves controls (`D→Z`) |
| 3 | `channel_halo` | Channel-to-channel amplification (`C→C`); feeder & isolated-null channels |
| 4 | `kitchen_sink` | Everything at once at sparse budgets — the hardest decomposition |

```python
sc  = pg.SCENARIOS[3]                              # channel_halo
scm = pg.sample_scm(sc.prior(T=104, seed=0), seed=0,
                    connect_all=sc.connect_all, name=sc.name, purpose=sc.purpose)
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `make_scm_prior` | Build a validated additive-SCM `SCMPrior` (pins layout, enables diverse texture). |
| `SCMPrior` | The config: graph sizes, edge budgets, coefficient/noise ranges, prior ranges. |
| `sample_prior_predictive` / `DataGenerator` | Generate an N-world corpus (dict of arrays). |
| `save_corpus` / `load_corpus` | Compressed `.npz` persistence. |
| `sample_scm` / `SCM` | Draw one accepted world with its full ground truth. |
| `describe_scm` | Plain-text description of a world. |
| `write_scm_bundle` | Write one world's auditable folder. |
| `write_scenario_bundles` | Write the full five-scenario inspection set (the CLI's datasets). |
| `SCENARIOS` | Five named audit scenarios, each isolating a pathway. |

The public surface reads mathematically: **treatments** (`n_treatments`, media
channels), **covariates** (`n_covariates`, controls), and **latent** factors
(`n_latent`, hidden confounders) size the graph; `make_scm_prior` builds a prior;
`sample_prior_predictive` / `sample_scm` draw from it.

## Reproducibility and validation

- **Determinism.** Same `(cfg, seed)` produces a byte-identical world or corpus —
  a load-bearing property, covered by the test suite.
- **Signal diagnostics.** `signal_diagnostics.per_channel_signal` reports, per
  direct channel, spend/contribution CV and high-frequency ratios, the
  spend↔contribution rank correlation, and a warmup ratio; the summary is embedded
  in `corpus["diagnostics"]["signal"]` on every corpus. `check_signal_gate` is an
  opt-in gate a caller can run against that summary to reject a too-flat corpus —
  generation itself does not enforce it.
- **Invariants as tests.** The additive identity, the telescoping split, the full
  per-node decomposition, and zero-padding of inactive slots are asserted directly
  in `tests/`.

```bash
ruff check . && ruff format --check .
mypy prior_generator
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Foundation libraries

The generator is built directly on the PyMC stack — no bespoke sampling or
mechanism code:

- **[PyTensor](https://github.com/pymc-devs/pytensor)** — the symbolic tensor
  engine. The entire SCM (all nodes, all interventions, the decomposition) is one
  PyTensor graph, so `Y` and its decomposition share subexpressions and the
  identity is exact by construction.
- **[PyMC](https://github.com/pymc-devs/pymc)** — priors and sampling. Every
  continuous parameter is a PyMC distribution and every noise term an RV
  (`pm.Normal`, `pm.HalfNormal`, `pm.Bernoulli`); a world is a `pm.draw` from the
  prior predictive.
- **[pymc-marketing](https://github.com/pymc-labs/pymc-marketing)** — the MMM
  mechanisms. The adstock (`geometric_adstock`, `weibull_adstock`) and saturation
  (`hill`, `logistic`, `michaelis_menten`, `tanh`, `root`) transforms are the
  library's own, bridged through its xtensor named-dim API.

## License

MIT — see [LICENSE](LICENSE).
