# The exact decomposition

The reason to *simulate* rather than collect data is that you get the answer.
Each world reports how every dollar of sales was actually produced, and the
pieces sum to sales **exactly** — ≈1e-15 in float64, **not** a Taylor
approximation.

## The identity

$$
\text{sales} = \text{baseline} + \sum_k \text{contributions}_k + \text{indirect\_effects}
$$

- **`contributions`** — each channel's *direct* effect, $g^{cy}_k\,\beta_k\,f_k(C^{\text{base}}_k)$,
  where $C^{\text{base}}$ is the channel system under the intervention *"zero all
  incoming channel interactions"* (`D→C`, `Z→C`, `C→C` removed). So it is what
  channel $k$ would have produced on its own.
- **`indirect_effects`** — the *extra* sales explained by spend that was itself
  moved by demand, controls, or other channels:
  $\sum_k g^{cy}_k\,\beta_k\,\big(f_k(C_k) - f_k(C^{\text{base}}_k)\big)$.

The identity is exact because `Y` and the decomposition are built from the *same*
symbolic quantities — $f_k$ is one fixed function evaluated on two inputs, and its
κ-relative saturation anchor is a function of the drawn *parameters* alone. That
anchor choice matters causally: deriving it from the realized series (its window
mean) would make the response at week $t$ depend on spend at every $t' > t$, so
$do(C_{k,t'})$ for a late $t'$ would move an early $Y_t$. It is fixed once per
world and reused by every intervention variant.

!!! warning "The 3-way `indirect_effects_by_source` split is a convention"

    `indirect_effects` above is a genuine causal quantity: it is order-free.
    Its split into `(cc, zc, dc)` is not. Because $f_k$ and the channel
    equation's softplus are both nonlinear, one-at-a-time interventions do not
    sum to the total, so the split is *defined* by a fixed sequential zeroing
    order. A different order, or a Shapley average over orders, gives different
    numbers with equal claim to being "the" indirect effect. Treat the split as
    a labelling choice, not an estimand — in particular, do not train against it
    as though it were identified.

## Held-spend shocks do not break the identity

When channel shocks are enabled, a scheduled reported-window interval clamps a
direct channel to its recorded held level. That is the shock's *only* effect:
the clamped path then feeds the ordinary normalized causal adstock kernel, so
carryover from pre-shock spend decays across the interval exactly as it decays
anywhere else. A zero held level therefore drives the direct contribution to
zero only once the whole kernel span sits inside the interval — which is
precisely what a standard MMM adstock reproduces from the same observed spend.

These interventions change the observed channel path (and can propagate to
downstream channels), but all decomposition terms are evaluated on the same
intervened graph. The additive identity, telescoping indirect split, baseline
split, and full reconstruction therefore hold inside and outside shock masks,
with only dtype-scaled rounding after float32 corpus persistence. Schedule
metadata is observable in the corpus; see
[Generating a corpus](corpus.md#channel-adstock-and-intervention-audit-metadata).

## Verify it yourself

```python exec="1" source="material-block" result="text"
from scm_docs import world
import numpy as np

scm = world(scenario=1, seed=0)
d = scm.data

lhs = d["sales"]
rhs = d["baseline"] + d["contributions"].sum(1) + d["indirect_effects"]
print("sales = baseline + Σ direct + indirect")
print("  max |lhs − rhs| =", f"{np.abs(lhs - rhs).max():.2e}")
```

## The telescoping indirect split

`indirect_effects` is further split, by sequential graph surgery, into a
**telescoping 3-way attribution** in the locked order `(cc, zc, dc)` —
channel→channel, control→channel, demand→channel. The three columns sum exactly
to `indirect_effects`.

```python exec="1" source="material-block" result="text"
from scm_docs import world
import numpy as np

scm = world(1, 0)
by_src = scm.data["indirect_effects_by_source"]        # (n_time_steps, 3) — cc, zc, dc
total = scm.data["indirect_effects"]

for i, src in enumerate(("cc", "zc", "dc")):
    col = by_src[:, i]
    print(f"  indirect via {src}: mean={col.mean():+.3f}")
print("split sums to total? max err =",
      f"{np.abs(by_src.sum(1) - total).max():.2e}")
```

## The fully-decomposed identity

The baseline itself splits per node, so a world also satisfies the
completely-unrolled identity — every term a named series:

$$
\text{sales} = \underbrace{\text{baseline\_intrinsic}}_{\text{organic}}
 + \sum_j \text{confounder\_contribution}_j
 + \sum_m \text{control\_contribution}_m
 + \sum_k \text{contributions}_k
 + \sum_s \text{indirect\_effects\_by\_source}_s
$$

That full sum is exactly what `SCM.reconstruction()` computes:

```python exec="1" source="material-block" result="text"
from scm_docs import world
import numpy as np

scm = world(1, 0)
d = scm.data

manual = (
    d["baseline_intrinsic"]                   # the intercept B
    + d["sales_noise"]                        # iid observation noise
    + d["confounder_contribution"].sum(1)     # D → B
    + d["control_contribution"].sum(1)        # Z → B
    + d["contributions"].sum(1)               # direct C → Y
    + d["indirect_effects_by_source"].sum(1)  # cc + zc + dc
)
print("manual == reconstruction()?  ",
      bool(np.allclose(manual, scm.reconstruction())))
print("reconstruction == sales?     max err =", f"{scm.identity_error():.2e}")
```

## See every effect at once

`plot_decomposition` overlays sales against Σ-components (the identity check),
then the direct contributions, the baseline components, and the indirect sources.

```python exec="1" source="material-block" html="1"
from scm_docs import world, viz_html
import prior_generator as pg

print(viz_html(pg.viz.plot_decomposition, world(1, 0),
         caption="Top: sales vs Σ true components. Below: direct, baseline, and indirect effects."))
```

!!! note "Why isolated null channels credit zero"
    Some scenarios keep fully-isolated channels as deliberate zero-attribution
    traps. Their direct `contributions` column is identically zero — a model must
    learn to credit them nothing. See [Named scenarios](scenarios.md).

## In a corpus, these are diagnostics

When you generate a [corpus](corpus.md), all four invariants — the additive
identity, the telescoping split, the full per-node decomposition, and
zero-padding of inactive slots — are computed at generation time and reported in
`corpus["diagnostics"]`, and asserted directly in the test suite.
