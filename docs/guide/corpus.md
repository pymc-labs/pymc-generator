# Generating a corpus

A **corpus** stacks many worlds into a dict of numpy arrays over `N` tasks and
`T` weeks — the tensor format amortized-inference / PFN pipelines consume, with
the decomposition targets baked in. Generate it functionally with
[`sample_prior_predictive`](../reference/corpus.md#prior_generator.sampler.sample_prior_predictive)
or through the [`DataGenerator`](../reference/corpus.md#prior_generator.data_generator.DataGenerator)
facade.

## Generate one

```python exec="1" source="material-block" result="text"
import prior_generator as pg

cfg = pg.make_scm_prior(n_treatments=5, n_covariates=3, n_latent=2,
                        edge_budget={"cy": (4, 4), "dc": (2, 2), "zc": (1, 2)},
                        n_cells=2, draws_per_cell=2, seed=42)
corpus = pg.sample_prior_predictive(cfg)
print("N tasks:", corpus["spend_raw"].shape[0])
```

Corpus mode scales up in **cells**: each cell fixes one structure and draws
`draws_per_cell` worlds from it, so `N = n_cells × draws_per_cell`. The
train/validation split is made at the *cell* level, so no structure leaks across
the split.

## The schema

The most important keys (padded max sizes `K / M / J`; internal math is float64,
storage is float32/uint8/int32):

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

Inspect the real arrays:

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
c = corpus()

for key in ["spend_raw", "controls", "sales_raw", "contributions_raw",
            "indirect_effects_by_source", "demand", "g", "active_c_mask"]:
    print(f"{key:<28} {str(c[key].shape):<14} {c[key].dtype}")
```

## Validate the output

`DataGenerator.validate_corpus` checks the schema — required keys, shapes,
finiteness, binary masks, and normalization identities — and returns a list of
errors (empty means valid). `DataGenerator.generate` runs it for you unless you
pass `validate=False`.

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
from prior_generator.data_generator import DataGenerator

errors = DataGenerator.validate_corpus(corpus())
print("validation errors:", errors or "none — schema OK ✅")
```

## The decomposition holds across the whole corpus

The generator records the four invariants in `diagnostics`. You can also check
the additive identity directly on the stacked arrays:

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
import numpy as np

c = corpus()
lhs = c["sales_raw"].astype(np.float64)
rhs = (c["baseline_raw"] + c["contributions_raw"].sum(-1)
       + c["indirect_effects"]).astype(np.float64)
print("sales = baseline + Σ direct + indirect")
print("  max abs error over all tasks/weeks:", f"{np.abs(lhs - rhs).max():.2e}")

diag = c["diagnostics"]
print("reported decomposition_max_abs_error:",
      f"{diag.get('decomposition_max_abs_error', float('nan')):.2e}")
```

## Signal diagnostics

A corpus can satisfy every schema contract and still be *unlearnable* if the true
contributions barely move. Every corpus embeds a signal summary; `check_signal_gate`
turns it into PASS/FAIL rows.

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
from prior_generator.signal_diagnostics import check_signal_gate

signal = corpus()["diagnostics"]["signal"]
ok, lines = check_signal_gate(signal)
print("overall:", "PASS ✅" if ok else "see rows below")
for line in lines:
    print(" ", line)
```

!!! note "Tiny demo corpus"
    The corpus on this page is deliberately small (4 tasks) so the docs build
    fast. Real training corpora use `n_cells` in the hundreds and `draws_per_cell`
    in the tens; the signal gate is tuned for those pool sizes, so a 4-task gate
    row may read FAIL purely from small-sample noise.

## Persistence

`save_corpus` writes a compressed, self-describing `.npz` (the `diagnostics` dict
round-trips as JSON); `load_corpus` reads it back.

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
import prior_generator as pg
import tempfile, os, numpy as np

c = corpus()
path = os.path.join(tempfile.mkdtemp(), "demo.npz")
pg.save_corpus(c, path)
loaded = pg.load_corpus(path)

print("saved to:", os.path.basename(path), f"({os.path.getsize(path):,} bytes)")
print("round-trips spend_raw exactly:",
      bool(np.array_equal(c["spend_raw"], loaded["spend_raw"])))
print("diagnostics restored as dict:", isinstance(loaded["diagnostics"], dict))
```

The [`DataGenerator`](../reference/corpus.md#prior_generator.data_generator.DataGenerator)
facade adds batching (`generate_batches`) and generate-and-save
(`generate_and_save`) on top of the same machinery.

## Dialing complexity

The SCM *structure* is fixed; `make_scm_prior` dials *complexity* within it along
orthogonal axes, keeping tensor shapes identical so one model / eval harness
serves every level.

| Axis | How | Knobs |
| --- | --- | --- |
| **Graph size** | how many nodes are live | `n_treatments`, `n_covariates`, `n_latent`, and their `*_active_range`s |
| **Interactions** | how many arrows of each type | `edge_budget` (per-type "pot") |
| **Nonlinearity** | media-response family mix | `nonlinearity="diverse"` / `"linear"` |
| **Signal / noise** | coefficient & noise ranges | `**overrides` |
| **Texture** | channels' high-frequency drive | `texture="diverse"` |

### The edge budget

`edge_budget` maps an edge type to an **arrow budget** — an "up to N" pot
scattered over the eligible node pairs, rather than a per-pair coin flip:

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

- An **int** `N` means "up to N" (drawn uniformly in `{0..N}`); a **tuple**
  `(lo, hi)` draws in the inclusive range; `(N, N)` means exactly `N`.
- Counts are capped at the number of eligible pairs.
- Types **omitted** keep their per-pair Bernoulli base rate; `cy` keeps its `≥ 1`
  floor.

See the full configuration surface in the
[API reference](../reference/config.md).
