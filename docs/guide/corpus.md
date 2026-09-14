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
reproducibility boundary: the number of leading reported weeks whose media
response reaches back before the window. It is the **realized** kernel support
— the largest positive lag that carries nonzero normalized adstock weight —
taken over the eligible direct channels, and `0` when `adstock_burn_in == 0`.
So it is `0` for identity-only direct paths, `0` for a geometric kernel drawn at
`alpha == 0` (an exact identity) or a Weibull kernel whose trailing taps are
annihilated by the min-max normalization, and at most `l_max - 1`. When
`summarize_signal_metrics` lacks the full adstock metadata (family, decay,
Weibull shape) it falls back to the conservative `l_max - 1` with burn-in.
Reported weeks
`0 .. response_warmup_weeks - 1` have a media response that depends on
pre-window spend the corpus does not persist, so their targets are not functions
of persisted inputs. This is not a train/query mask: `support_mask` remains only
the temporal support/query split. With `adstock_burn_in == 0`, zero padding makes
every reported response reproducible from persisted spend.

### Random-walk parameter labels

Persisted `param_rw_*_std` labels declare a walk's **expected standard
deviation** over the full simulated horizon `n_time_steps + adstock_burn_in`.
The guarantee is second-moment, and worth stating precisely:

```text
E[ var_pop(path) ] = std²      i.e.   std = sqrt( E[var_pop(path)] )
```

It is *not* `E[sd(path)] = std`. `sqrt` is concave, so the expected *sd* sits
strictly below the declared `std` even when the expected *variance* is exactly
right. Measured over 20 000 unit-`std` paths at `n_time_steps=60`,
`E[var_pop]/std²` was 0.990–1.004 (target 1.0) while `E[sd]/std` was 0.9220 at
kernel width 1 and 0.8674 at width 26.

Each path is divided by a fixed constant rather than by its own
realized standard deviation, so that scale is realized only in expectation. It is
therefore neither the realized standard deviation of an individual path nor a
standard deviation measured only over the reported window. `smoothness`
likewise maps to an absolute moving-average kernel width in weeks, governed by
`rw_smoothness_max_weeks` (26 by default) and clamped to that full horizon. For
`positive_only` channel walks, `param_rw_c_std` is the pre-softplus amplitude,
so it is excluded from the signed-walk table below rather than reported with a
misleadingly wide range.

The signed `rw_d` / `rw_z` / `rw_b` walks are centred, smoothed Gaussian paths:
cumulative sum, edge-padded moving average, full-path centring, then the fixed
scale divisor. A channel's own drive is the *positive-only* version of the same
construction — the identical signed path, wrapped in `softplus` — which is why
its amplitude label is pre-softplus.

Across 32 signed walks from eight worlds at `n_time_steps=52` and
`adstock_burn_in=8`, the reported-window sd / declared `std` was:

| signed group | reported-window sd / declared `std` | median |
| --- | --- | --- |
| `rw_d` | [0.396, 0.927] | 0.690 |
| `rw_z` | [0.252, 1.985] | 0.868 |
| `rw_b` | [0.264, 1.809] | 0.814 |

Combined, those 32 walks span [0.25, 1.99] — roughly an 8× spread.
`param_rw_*_std` is therefore a weak label for
anything measured on the reported window; consumers should not score it as if
it were the realized reported-window standard deviation.

`param_rw_y_std` is **not** in that table, because `RW_Y` is not a walk. It is
iid observation noise, `RW_Y = rw_y_std * eps_y`, with no cumulative sum, no
smoothing and no centring, so its label is the exact per-week Normal σ:
`sales_noise == param_rw_y_std * eps_y[adstock_burn_in:]` holds to the last bit
(max absolute difference 0.0 over eight worlds). Its realized window sd still
scatters — measured [0.838, 1.314], median 1.001, over the same eight worlds —
but that is the sample sd of 52 standard normals under the acceptance filter,
not a loose label.

#### Walk paths are drawn jointly, not sequentially

Full-path centring plus a symmetric moving average makes a walk
**non-adapted**: week `t` of the path depends on the *whole* innovation vector,
including innovations at `t' > t`. Perturbing only the last innovation of a
60-week unit path moves week 0 by `-0.0231` at kernel width 26 and by `-0.0053`
at width 1 (the centring alone), and moves all 60 weeks. The paths are
consequently drawn jointly, offline, as one multivariate normal; they are not a
filtration you can simulate forward a week at a time.

This does **not** leak future spend into the media response. Causality here is a
property of the *response*: `f_k` applies a causal adstock kernel and a κ scale
computed from parameters alone, so the response at week `t` reads only spend at
`t' ≤ t`. Bumping one late week of observed spend and re-running the generator's
own response code leaves every earlier week bit-identical (18 of 18
(world, channel) checks at exactly zero change). Non-adaptedness describes how
`eps → path` factorizes; `eps` itself is exogenous.


### Channel, adstock, and intervention audit metadata

The following arrays are persisted for every corpus, including empty
`(n_tasks, 0)` schedule arrays when shocks are disabled. `n_shocks` is the
configured `n_channel_shocks`, so every world has exactly `n_shocks` event
records.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `confounding_strength` | `(n_tasks,)` | `float32` | drawn per-world rho (`0` when disabled) |
| `channel_level` | `(n_tasks, n_treatments)` | `float32` | `softplus(rw_c_mean)` reference level |
| `saturation_scale` | `(n_tasks, n_treatments)` | `float32` | κ response anchor: a parameter-only **reference level**, not `E[C]` (zero-padded for inactive channels) |
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

`saturation_scale` is the κ anchor each channel's response curve is normalized
by: `softplus(softplus(rw_c_mean) + pulse_amp * pulse_prob + weighted reference
Z→C / C→C parent terms)`, accumulated in topological order. It is a
**parameter-only reference level and deliberately not `E[C]`** — the channel
equation applies `softplus` and the channel walk is itself a `softplus`, so the
anchor takes `softplus` of a mean where the world takes the mean of a
`softplus`. Softplus being strictly convex, Jensen puts realized expected spend
strictly above it: measured `E[C_k] / saturation_scale` ran 1.004–1.099 over 36
(θ, channel) cells at 600 noise draws each, every cell above 1. Do not score it
as a predicted channel level; it exists precisely because it reads no moment,
which is what keeps `p(θ)` independent of the noise and the week-`t` response
free of spend at `t' > t`.

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
facade adds lazy batching (`iter_batches`) and generate-and-save
(`generate_and_save`) on top of the same machinery.

```python
generator = pg.DataGenerator(cfg)
for index, batch in enumerate(generator.iter_batches(n_tasks=1000, batch_size=100)):
    pg.save_corpus(batch, f"corpus-{index:03d}.npz")
```

Each batch has its own cell-level train/validation split. A one-world remainder
joins the previous batch. Seeds start at `cfg.seed` unless explicitly supplied
and increment per batch; batching is reproducible but is not equivalent to one
larger generation call. Consume the iterator as shown to bound retained data,
or explicitly call `list(...)` if retaining all batches is intentional.

### Schema version and the v1 migration

Every corpus records its persisted-schema version in
`corpus["diagnostics"]["schema_version"]`, which is `2` for anything generated
today. Version 1 shards used the old symbolic dimension keys; `load_corpus`
renames them on read **first**, then requires the version to be exactly
`CORPUS_SCHEMA_VERSION` (`2`) as a non-bool integer — so a `.npz` written
before the rename stays loadable with no conversion step, while a shard whose
version is missing, malformed (the string `"2"`), or from the future (`99`) is
rejected outright rather than half-read. `validate_corpus` reports the same
condition as an error string, and `save_corpus` raises; `save_corpus` also
refuses a corpus that still carries v1 keys, so new shards can only contain the
canonical names. The rename map is
`prior_generator.slots.LEGACY_CORPUS_KEYS_V1`:

| v1 key | v2 key |
| --- | --- |
| `K_active` | `n_treatments_active` |
| `M_active` | `n_covariates_active` |
| `J_active` | `n_latent_active` |
| `active_c_mask` | `treatment_active_mask` |
| `active_m_mask` | `covariate_active_mask` |
| `active_j_mask` | `latent_active_mask` |

### Diagnostics that do not go into the file

`diagnostics` is otherwise a faithful record of the run, with one deliberate
exception: the wall-clock telemetry lives in
`diagnostics["timing"] = {"elapsed_s": float, "tasks_per_sec": float}`, and
`save_corpus` writes the diagnostics block **without** it (on a copy — the
caller's dict is never mutated). Those two numbers are the only nondeterministic
values a generation produces, so excluding them is what makes two independent
same-seed generations save byte-identical `.npz` files. Read them off the
in-memory corpus if you want them; do not expect them back from `load_corpus`,
and note that `validate_corpus` accepts diagnostics with or without the block.

## Dialing complexity

The SCM *structure* is fixed; `make_scm_prior` dials *complexity* within it along
orthogonal axes, keeping tensor shapes identical so one model / eval harness
serves every level.

| Axis | How | Knobs |
| --- | --- | --- |
| **Graph size** | how many nodes are live | `n_treatments`, `n_covariates`, `n_latent`, and their `*_active_range`s |
| **Interactions** | how many arrows of each type | `edge_budget`, `min_no_direct_effect_channels` |
| **Nonlinearity** | media-response family mix | `nonlinearity="diverse"` / `"linear"` |
| **Signal / noise** | coefficient & noise ranges | `**overrides` |
| **Texture** | channels' and controls' high-frequency drive | explicit noise, pulse, and walk ranges |

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

### Direct-null channels

`min_no_direct_effect_channels` reserves active channels without a direct
`C→Y` edge. Their direct contribution is zero. A `cy` budget alone cannot
guarantee this because its arrow count is clamped to the active slots.

```python
cfg = pg.make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),   # 2–10 active channels per task
    edge_budget={"cy": (1, 10)},
    min_no_direct_effect_channels=1,   # at least one active channel has no direct effect
)
```

The floor must be smaller than `n_treatments_active_range[0]` so the smallest
cell still has a direct channel. Direct channels remain scattered over active
slots. A direct-null channel can still reach `Y` as a **feeder** through `C→C`;
set `edge_budget["cc"] = 0` as well if it must have no path to `Y`.

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
