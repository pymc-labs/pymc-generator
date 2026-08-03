# Generating a corpus

A **corpus** stacks many worlds into a dict of numpy arrays over `n_tasks` tasks
and `n_time_steps` weeks — the tensor format amortized-inference / PFN pipelines
consume, with the decomposition targets baked in. Generate it functionally with
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
print("n_tasks:", corpus["spend_raw"].shape[0])
```

Corpus mode scales up in **cells**: each cell fixes one structure and draws
`draws_per_cell` worlds from it, so `n_tasks = n_cells × draws_per_cell`. The
train/validation split is made at the *cell* level, so no structure leaks across
the split.

## The schema

The most important keys (padded max sizes
`n_treatments / n_covariates / n_latent`; internal math is float64, storage is
float32/uint8/int32):

| Key | Shape | What |
| --- | --- | --- |
| `spend_raw` | (n_tasks, n_time_steps, n_treatments) | media spend (observed input) |
| `controls` | (n_tasks, n_time_steps, n_covariates) | observed controls |
| `sales_raw` | (n_tasks, n_time_steps) | sales (observed target) |
| `contributions_raw` | (n_tasks, n_time_steps, n_treatments) | per-channel **direct** contributions (truth) |
| `indirect_effects` | (n_tasks, n_time_steps) | total interaction-routed effect (truth) |
| `indirect_effects_by_source` | (n_tasks, n_time_steps, 3) | telescoping split, order `(cc, zc, dc)` |
| `baseline_raw` | (n_tasks, n_time_steps) | full baseline |
| `demand` | (n_tasks, n_time_steps, n_latent) | latent demand series (truth) |
| `g` | (n_tasks, n_slots) | the packed DAG (all 8 edge blocks) |
| `treatment_active_mask` / `covariate_active_mask` / `latent_active_mask` | (n_tasks, n_treatments / n_covariates / n_latent) | which slots are live |
| `diagnostics` | dict | edge marginals, decomposition errors, signal block, and `short_horizon_n_query` split metadata |

`demand` is a latent factor pinned to mean 0 / scale 1, putting its `D→B`,
`D→C`, and `D→Z` magnitude in the loadings. The graph admits an exact sign
flip (`eps_d`, `w_dc`, `u_dz`, and `delta_db` all negated), but the supported
prior's strictly positive `db_coeff_range`, `dc_coeff_range`, and
`dz_coeff_range` exclude the reflected parameters, so demand's sign is
identified unless a user widens one of those ranges to admit negatives; then
only `|D|` and the loading magnitudes are recoverable.

`support_mask` is always a contiguous support prefix followed by a nonempty
query suffix. `is_future=0` uses `diagnostics["short_horizon_n_query"]` query
weeks; `is_future=1` uses the second half of the series. `sales_scale` is the
standard deviation of supported sales, with the full-series standard deviation
and then `1.0` as deterministic fallbacks for degenerate support windows.

`diagnostics["signal"]["response_warmup_weeks"]` identifies a separate
reproducibility boundary. It is `l_max - 1` when `adstock_burn_in > 0` and at
least one eligible direct channel uses nonidentity adstock
(`adstock_family != 0`); otherwise it is `0`. When
`summarize_signal_metrics` lacks adstock metadata, it conservatively reports
`l_max - 1` with burn-in. Reported weeks
`0 .. response_warmup_weeks - 1` have a media response that depends on
pre-window spend the corpus does not persist, so their targets are not functions
of persisted inputs. This is not a train/query mask: `support_mask` remains only
the temporal support/query split. With `adstock_burn_in == 0`, zero padding makes
every reported response reproducible from persisted spend.

### Random-walk parameter labels

Persisted `param_rw_*_std` labels declare the walk's **expected standard
deviation** over the full simulated horizon `n_time_steps + adstock_burn_in`.
Because each path is divided by a fixed constant rather than by its own
realized standard deviation, that scale is realized only in expectation. It is
therefore neither the realized standard deviation of an individual path nor a
standard deviation measured only over the reported window. `smoothness`
likewise maps to an absolute moving-average kernel width in weeks, governed by
`rw_smoothness_max_weeks` (26 by default) and clamped to that full horizon. For
`positive_only` channel walks, `param_rw_c_std` is the pre-softplus amplitude,
so it is excluded from the signed-walk table below rather than reported with a
misleadingly wide range.

Across 40 signed walks from eight worlds at `n_time_steps=52` and
`adstock_burn_in=8`, the reported-window sd / declared `std` was:

| signed group | reported-window sd / declared `std` | median |
| --- | --- | --- |
| `rw_d` | [0.396, 0.927] | 0.690 |
| `rw_z` | [0.252, 1.985] | 0.868 |
| `rw_b` | [0.264, 1.809] | 0.814 |
| `rw_y` | [0.294, 1.335] | 0.757 |
| **all signed (n=40)** | **[0.25, 1.99]** | **0.80** |

This is roughly an 8× spread. `param_rw_*_std` is therefore a weak label for
anything measured on the reported window; consumers should not score it as if
it were the realized reported-window standard deviation.


### Channel, adstock, and intervention audit metadata

The following arrays are persisted for every corpus, including empty
`(n_tasks, 0)` schedule arrays when shocks are disabled. `n_shocks` is the
configured `n_channel_shocks`, so every world has exactly `n_shocks` event
records.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `confounding_strength` | `(n_tasks,)` | `float32` | drawn per-world rho (`0` when disabled) |
| `channel_level` | `(n_tasks, n_treatments)` | `float32` | `softplus(rw_c_mean)` reference level |
| `saturation_scale` | `(n_tasks, n_treatments)` | `float32` | response anchor: expected channel level from parameters alone (zero-padded for inactive channels) |
| `adstock_family` | `(n_tasks, n_treatments)` | `uint8` | `0=none`, `1=geometric`, `2=Weibull` |
| `adstock_alpha` | `(n_tasks, n_treatments)` | `float32` | geometric decay parameter |
| `weibull_lam` | `(n_tasks, n_treatments)` | `float32` | Weibull scale parameter |
| `weibull_k` | `(n_tasks, n_treatments)` | `float32` | Weibull shape parameter |
| `channel_shock_mask` | `(n_tasks, n_time_steps, n_treatments)` | `uint8` | binary reported-window held-spend mask |
| `channel_shock_channel` | `(n_tasks, n_shocks)` | `int32` | selected direct-channel index per event |
| `channel_shock_start` | `(n_tasks, n_shocks)` | `int32` | reported-window event start |
| `channel_shock_length` | `(n_tasks, n_shocks)` | `int32` | held duration in weeks |
| `channel_shock_level_multiplier` | `(n_tasks, n_shocks)` | `float32` | sampled relative held-level multiplier |
| `channel_shock_level` | `(n_tasks, n_shocks)` | `float32` | realized held level |

Together these schedule fields are observable intervention metadata: they
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
            "indirect_effects_by_source", "demand", "g", "treatment_active_mask"]:
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
`signal_metrics: float32 (n_tasks, n_treatments, 9)` and
`signal_metric_valid: uint8 (n_tasks, n_treatments, 9)`. Their exact versioned
layout and validity rules are in the
[signal diagnostics reference](../reference/signal.md).
They are calculated from the final retained float32 arrays (after truncation),
so a consumer can recompute them after loading the `.npz`. They are labels, not
model inputs. Set `include_identifiability_labels=False` to omit both arrays;
all observable arrays remain byte-identical and the aggregate signal diagnostics
remain available.

The `diagnostics["signal"]` block includes the metric and adstock-kernel
versions plus `l_max`, `adstock_burn_in`, and `response_warmup_weeks`; use those
stored values rather than assuming configuration defaults when recomputing a
loaded shard. Its kernel metadata is
`adstock_kernel_semantics="normalized-causal-minmax-weibull-density"` at
`adstock_kernel_version=3`, reflecting pymc-marketing's min-max rescaling before
sum normalization. `frac_zero_contemporaneous_weight` reports the eligible direct
channel share whose current-week normalized adstock weight is effectively zero.
It is reported for inspection, not used to gate a corpus.

```python exec="1" source="material-block" result="text"
from scm_docs import corpus
from prior_generator.signal_diagnostics import check_signal_gate

signal = corpus()["diagnostics"]["signal"]
ok, lines = check_signal_gate(signal)
print("overall:", "PASS ✅" if ok else "see rows below")
for line in lines:
    print(" ", line)
```

The default gate also limits amplitude and collinearity failures:
`frac_contrib_rel_std_lt_001 <= 0.10` caps the share of direct channels whose
true contribution is too small to matter in loss units, and
`frac_contrib_r2_gt_095 <= 0.10` caps the share whose contribution is a
near-perfect linear combination of the baseline and other channels. A prior gate
PASS alone did not certify either property.

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

### Schema version and the v1 migration

Every corpus records its persisted-schema version in
`corpus["diagnostics"]["schema_version"]`, which is `2` for anything generated
today. Version 1 shards used the old symbolic dimension keys; `load_corpus`
renames them on read and stamps the version, so a `.npz` written before the
rename stays loadable with no conversion step. `save_corpus` raises a
`ValueError` if it is handed a corpus that still carries v1 keys, so new shards
can only contain the canonical names. The rename map is
`prior_generator.slots.LEGACY_CORPUS_KEYS_V1`:

| v1 key | v2 key |
| --- | --- |
| `K_active` | `n_treatments_active` |
| `M_active` | `n_covariates_active` |
| `J_active` | `n_latent_active` |
| `active_c_mask` | `treatment_active_mask` |
| `active_m_mask` | `covariate_active_mask` |
| `active_j_mask` | `latent_active_mask` |

## Dialing complexity

The SCM *structure* is fixed; `make_scm_prior` dials *complexity* within it along
orthogonal axes, keeping tensor shapes identical so one model / eval harness
serves every level.

| Axis | How | Knobs |
| --- | --- | --- |
| **Graph size** | how many nodes are live | `n_treatments`, `n_covariates`, `n_latent`, and their `*_active_range`s |
| **Interactions** | how many arrows of each type | `edge_budget` (per-type "pot"), `min_dead_channels` |
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

### The dead-channel floor

A **dead** channel is an *active* channel with no direct `C→Y` arrow: spend
observed, true contribution exactly zero — the negative class for "which channels
move sales". A `cy` budget cannot guarantee one, because it is an absolute arrow
count clamped to the eligible slots: a task drawing 2 active channels under
`cy=(2, 10)` has both of them live. `min_dead_channels` caps the live count at
`n_treatments_active − min_dead_channels` instead (never below `1`):

```python
cfg = pg.make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),   # 2–10 active channels per task
    edge_budget={"cy": (1, 10)},
    min_dead_channels=1,                 # ≥ 1 active channel never reaches Y
)
```

One recipe then spans 1–9 live channels over 2–10 active ones and always carries a
dead one; the live channels stay scattered over all active slots, so slot index
says nothing about the label. The floor must be `< n_treatments_active_range[0]`
or `make_scm_prior` rejects the config. Under a `cc` budget a dead channel can
still reach `Y` as a *feeder* — pin `edge_budget={"cc": 0}` if the floor must mean
"no path to `Y`".

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

cfg = pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=40,
                        n_cells=2, draws_per_cell=2, seed=7,
                        prior_conditioning=True)
c = pg.sample_prior_predictive(cfg)

print("prior_cond:", c["prior_cond"].shape, "— columns:", list(PRIOR_COND_LAYOUT))
print("first world:", np.round(c["prior_cond"][0], 3))
print("echo:", c["diagnostics"]["prior_cond"]["supports"])
```

- `prior_cond` `(n_tasks, P)` holds packed `(low, width)` pairs in the **locked,
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
