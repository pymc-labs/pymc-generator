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
| `diagnostics` | dict | edge marginals, decomposition errors, signal block, and `short_horizon_n_query` split metadata |

`support_mask` is always a contiguous support prefix followed by a nonempty
query suffix. `is_future=0` uses `diagnostics["short_horizon_n_query"]` query
weeks; `is_future=1` uses the second half of the series. `sales_scale` is the
standard deviation of supported sales, with the full-series standard deviation
and then `1.0` as deterministic fallbacks for degenerate support windows.

### Channel, adstock, and intervention audit metadata

The following arrays are persisted for every corpus, including empty `(N, 0)`
schedule arrays when shocks are disabled. `S` is the configured
`n_channel_shocks`, so every world has exactly `S` event records.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `confounding_strength` | `(N,)` | `float32` | drawn per-world rho (`0` when disabled) |
| `channel_level` | `(N, K)` | `float32` | `softplus(rw_c_mean)` reference level |
| `saturation_scale` | `(N, K)` | `float32` | generation-time response anchor (zero-padded for inactive channels) |
| `adstock_family` | `(N, K)` | `uint8` | `0=none`, `1=geometric`, `2=Weibull` |
| `adstock_alpha` | `(N, K)` | `float32` | geometric decay parameter |
| `weibull_lam` | `(N, K)` | `float32` | Weibull scale parameter |
| `weibull_k` | `(N, K)` | `float32` | Weibull shape parameter |
| `channel_shock_mask` | `(N, T, K)` | `uint8` | binary reported-window held-spend mask |
| `channel_shock_channel` | `(N, S)` | `int32` | selected direct-channel index per event |
| `channel_shock_start` | `(N, S)` | `int32` | reported-window event start |
| `channel_shock_length` | `(N, S)` | `int32` | held duration in weeks |
| `channel_shock_level_multiplier` | `(N, S)` | `float32` | sampled relative held-level multiplier |
| `channel_shock_level` | `(N, S)` | `float32` | realized held level |

Together these schedule fields are observable intervention/reset metadata: they
reconstruct each reported held-spend window without storing a natural path or a
full burn-in mask. The realized level equals the multiplier times the selected
`channel_level`. For an enabled schedule, events occupy globally
non-overlapping deterministic time slots; channel selection may repeat because
direct channels are sampled uniformly with replacement.

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

The optional `identifiability` metadata block contains
`signal_metrics: float32 (N, K, 9)` and
`signal_metric_valid: uint8 (N, K, 9)`. Their exact versioned layout and
validity rules are in the [signal diagnostics reference](../reference/signal.md).
They are calculated from the final retained float32 arrays (after truncation),
so a consumer can recompute them after loading the `.npz`. They are labels, not
model inputs. Set `include_identifiability_labels=False` to omit both arrays;
all observable arrays remain byte-identical and the aggregate signal diagnostics
remain available.

The `diagnostics["signal"]` block includes the metric and adstock-kernel
versions plus `l_max` and `adstock_burn_in`; use those stored values rather than
assuming configuration defaults when recomputing a loaded shard.

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

## Prior conditioning (ACE)

A PFN checkpoint distills the prior it was trained on. **ACE-style prior
conditioning** makes the prior an *input*: with
`make_scm_prior(..., prior_conditioning=True)`, each **cell** draws a narrowed
prior interval per conditioned quantity —

```text
w  ~ U(w_lo, w_hi)            # interval width
lo ~ U(S_lo, S_hi − w)        # interval start
I  = [lo, lo + w]             # ⊆ the global support, by construction
```

— and that cell's parameter is drawn as `pm.Uniform(lo, lo + w)` instead of the
global support. The v1 conditioned set is `adstock_alpha` (geometric decay) and
`hill_shape` (Hill slope); width ranges are overridable per quantity via
`prior_cond_width_ranges`.

The draws are recorded in the corpus so consumers can expose them as
conditioning features:

```python exec="1" source="material-block" result="text"
import numpy as np
import prior_generator as pg
from prior_generator.slots import PRIOR_COND_LAYOUT

cfg = pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, T=40,
                        n_cells=2, draws_per_cell=2, seed=7,
                        prior_conditioning=True)
c = pg.sample_prior_predictive(cfg)

print("prior_cond:", c["prior_cond"].shape, "— columns:", list(PRIOR_COND_LAYOUT))
print("first world:", np.round(c["prior_cond"][0], 3))
print("echo:", c["diagnostics"]["prior_cond"]["supports"])
```

- `prior_cond` `(N, P)` holds packed `(low, width)` pairs in the **locked,
  append-only** `PRIOR_COND_LAYOUT` order — index columns by name, never by
  position literals.
- The key is present **iff** `prior_conditioning=True`; an unconditioned corpus
  is byte-identical to before the feature existed.
- `diagnostics["prior_cond"]` echoes the layout, the supports, and the width
  ranges, so feature standardization can be derived from the corpus instead of
  duplicated in the consumer.
- The single-world path honors the same draw: `sample_scm` records the
  intervals in `SCM.extras["prior_cond"]` and `describe_scm` prints them
  alongside the mechanisms they bound.
