# Generating a corpus

A **corpus** stacks many worlds into a dict of numpy arrays over `n_tasks` tasks
and `n_time_steps` weeks — one padded tensor archive, with the decomposition
targets baked in. Generate it functionally with
[`sample_prior_predictive`](../reference/corpus.md#pymc_generator.sampler.sample_prior_predictive)
or through the [`DataGenerator`](../reference/corpus.md#pymc_generator.data_generator.DataGenerator)
facade.

## Generate one

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(n_treatments=5, n_covariates=3, n_latent=2,
                        edge_budget={"cy": (4, 4), "dc": (2, 2), "zc": (1, 2)},
                        n_cells=2, draws_per_cell=2, seed=42)
corpus = pg.sample_prior_predictive(cfg)
print("n_tasks:", corpus["treatment_raw"].shape[0])
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
| `treatment_raw` | (n_tasks, n_time_steps, n_treatments) | observed treatment input |
| `covariates` | (n_tasks, n_time_steps, n_covariates) | observed covariates |
| `outcome_raw` | (n_tasks, n_time_steps) | outcome (observed target) |
| `treatment_contribution_raw` | (n_tasks, n_time_steps, n_treatments) | per-treatment **direct** contributions (truth) |
| `indirect_effects` | (n_tasks, n_time_steps) | total interaction-routed effect (truth) |
| `indirect_effects_by_source` | (n_tasks, n_time_steps, 3) | telescoping split, order `(cc, zc, dc)` |
| `baseline_raw` | (n_tasks, n_time_steps) | full baseline |
| `latent_unobserved` | (n_tasks, n_time_steps, n_latent) | latent-unobserved series (truth) |
| `g` | (n_tasks, n_slots) | the packed DAG (all 8 edge blocks) |
| `treatment_active_mask` / `covariate_active_mask` / `latent_active_mask` | (n_tasks, n_treatments / n_covariates / n_latent) | which slots are live |
| `diagnostics` | dict | edge marginals, decomposition errors, signal block, and `short_horizon_n_query` split metadata |

The latent-unobserved factor is pinned to mean 0 / scale 1, putting its `D→Y`,
`D→C`, and `D→Z` magnitude in the loadings. The graph admits an exact sign
flip (`eps_d`, `w_dc`, `u_dz`, and `delta_dy` all negated), but the supported
prior's strictly positive `dy_coeff_range`, `dc_coeff_range`, and
`dz_coeff_range` exclude the reflected parameters, so the latent-unobserved factor's sign is
identified unless a user widens one of those ranges to admit negatives; then
only `|D|` and the loading magnitudes are recoverable.

`support_mask` is always a contiguous support prefix followed by a nonempty
query suffix. `is_future=0` uses `diagnostics["short_horizon_n_query"]` query
weeks; `is_future=1` uses the second half of the series. `outcome_scale` is the
standard deviation of supported outcome, with the full-series standard deviation
and then `1.0` as deterministic fallbacks for degenerate support windows.

`diagnostics["signal"]["response_warmup_weeks"]` identifies a separate
reproducibility boundary: the number of leading reported weeks whose treatment
response reaches back before the window. It is the **realized** kernel support
— the largest positive lag that carries nonzero normalized carryover weight —
taken over the eligible direct treatments, and `0` when `carryover_burn_in == 0`.
So it is `0` for identity-only direct paths, `0` for a geometric kernel drawn at
`alpha == 0` (an exact identity) or a Weibull kernel whose trailing taps are
annihilated by the min-max normalization, and at most `l_max - 1`. When
`summarize_signal_metrics` lacks the full carryover metadata (family, decay,
Weibull shape) it falls back to the conservative `l_max - 1` with burn-in.
Reported weeks
`0 .. response_warmup_weeks - 1` have a treatment response that depends on
pre-window treatment the corpus does not persist, so their targets are not functions
of persisted inputs. This is not a train/query mask: `support_mask` remains only
the temporal support/query split. With `carryover_burn_in == 0`, zero padding makes
every reported response reproducible from persisted treatments.

### Random-walk parameter labels

Persisted `param_rw_*_std` labels declare a walk's **expected standard
deviation** over the full simulated horizon `n_time_steps + carryover_burn_in`.
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
`positive_only` treatment walks, `param_rw_c_std` is the pre-softplus amplitude,
so it is excluded from the signed-walk table below rather than reported with a
misleadingly wide range.

The signed `rw_d` / `rw_z` / `rw_b` walks are centred, smoothed Gaussian paths:
cumulative sum, edge-padded moving average, full-path centring, then the fixed
scale divisor. A treatment's own drive is the *positive-only* version of the same
construction — the identical signed path, wrapped in `softplus` — which is why
its amplitude label is pre-softplus.

Across 32 signed walks from eight worlds at `n_time_steps=52` and
`carryover_burn_in=8`, the reported-window sd / declared `std` was:

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
`outcome_noise == param_rw_y_std * eps_y[carryover_burn_in:]` holds to the last bit
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

This does **not** leak future treatment into the response. Causality here is a
property of the *response*: `f_k` applies a causal carryover kernel and a κ scale
computed from parameters alone, so the response at week `t` reads only treatment at
`t' ≤ t`. Bumping one late week of observed treatment and re-running the generator's
own response code leaves every earlier week bit-identical (18 of 18
(world, treatment) checks at exactly zero change). Non-adaptedness describes how
`eps → path` factorizes; `eps` itself is exogenous.


### Treatment, carryover, and intervention audit metadata

The following arrays are persisted for every corpus, including empty
`(n_tasks, 0)` schedule arrays when shocks are disabled. `n_shocks` is the
configured `n_treatment_shocks`, so every world has exactly `n_shocks` event
records.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `confounding_strength` | `(n_tasks,)` | `float32` | drawn per-world rho (`0` when disabled) |
| `treatment_level` | `(n_tasks, n_treatments)` | `float32` | `softplus(rw_c_mean)` reference level |
| `saturation_scale` | `(n_tasks, n_treatments)` | `float32` | κ response anchor: a parameter-only **reference level**, not `E[C]` (zero-padded for inactive treatments) |
| `carryover_family` | `(n_tasks, n_treatments)` | `uint8` | `0=none`, `1=geometric`, `2=Weibull` |
| `carryover_alpha` | `(n_tasks, n_treatments)` | `float32` | geometric decay parameter |
| `weibull_lam` | `(n_tasks, n_treatments)` | `float32` | Weibull scale parameter |
| `weibull_k` | `(n_tasks, n_treatments)` | `float32` | Weibull shape parameter |
| `treatment_shock_mask` | `(n_tasks, n_time_steps, n_treatments)` | `uint8` | binary reported-window held-treatment mask |
| `treatment_shock_index` | `(n_tasks, n_shocks)` | `int32` | selected direct-treatment index per event |
| `treatment_shock_start` | `(n_tasks, n_shocks)` | `int32` | reported-window event start |
| `treatment_shock_length` | `(n_tasks, n_shocks)` | `int32` | held duration in weeks |
| `treatment_shock_level_multiplier` | `(n_tasks, n_shocks)` | `float32` | sampled relative held-level multiplier |
| `treatment_shock_level` | `(n_tasks, n_shocks)` | `float32` | realized held level |

Together these schedule fields are observable intervention metadata: they
reconstruct each reported held-treatment window without storing a natural path or a
full burn-in mask. The realized level equals the multiplier times the selected
`treatment_level`. For an enabled schedule, events occupy globally
non-overlapping deterministic time slots; treatment selection may repeat because
direct treatments are sampled uniformly with replacement.

`saturation_scale` is the κ anchor each treatment's response curve is normalized
by: `softplus(softplus(rw_c_mean) + pulse_amp * pulse_prob + weighted reference
Z→C / C→C parent terms)`, accumulated in topological order. It is a
**parameter-only reference level and deliberately not `E[C]`** — the treatment
equation applies `softplus` and the treatment walk is itself a `softplus`, so the
anchor takes `softplus` of a mean where the world takes the mean of a
`softplus`. Softplus being strictly convex, Jensen puts realized expected treatment
strictly above it: measured `E[C_k] / saturation_scale` ran 1.004–1.099 over 36
(θ, treatment) cells at 600 noise draws each, every cell above 1. Do not score it
as a predicted treatment level; it exists precisely because it reads no moment,
which is what keeps `p(θ)` independent of the noise and the week-`t` response
free of treatment at `t' > t`.

Inspect the real arrays:

```python exec="1" source="block" result="text"
from scm_docs import corpus
c = corpus()

for key in ["treatment_raw", "covariates", "outcome_raw", "treatment_contribution_raw",
            "indirect_effects_by_source", "latent_unobserved", "g", "treatment_active_mask"]:
    print(f"{key:<28} {str(c[key].shape):<14} {c[key].dtype}")
```

## Validate the output

`DataGenerator.validate_corpus` checks the schema — required keys, shapes,
finiteness, binary masks, and normalization identities — and returns a list of
errors (empty means valid). `DataGenerator.generate` runs it for you unless you
pass `validate=False`.

It also requires `treatment_raw >= 0` in every corpus; every corpus the
generator has produced satisfies that by construction, so the rule rejects no
existing shard. When the optional [trajectory block](#composable-input-trajectories) is
present, `validate_corpus` checks it end to end: its flags, envelopes and realised
parameter arrays must occur together with `diagnostics["trajectory"]`, with
the prescribed shapes and dtypes, binary
flags and activity, zero padding, flags constant within a cell, activity `1` on
active inputs without a gate component and shift `0.0` on active inputs without
a level component, and the converse for active gated inputs — any gate flag
means at least one off-week in the window and at least two on-weeks inside the
task's support window, an `onset` flag means week 0 is off, and an `offset` flag
means the last week is off. It also checks `treatment_raw == 0` on activity-off
weeks outside shocks and `covariates == 0` on covariate off-weeks, the canonical
component layout, inclusion probabilities that are floats in `[0, 1]` — an echo
of exactly `0.0` requires prevalence `0`, and one of exactly `1.0` prevalence
`1` wherever that input type has active inputs, both counted from the stored
flags — and `prevalence` / `n_inputs` equal to their exact recomputation from
the stored flags.
It also reconstructs gates and level envelopes from the stored realised parameters,
checks parameter domains and ordered jump weeks, and rejects missing or unexpected
parameter fields.

When a corpus carries `diagnostics["active_count_coverage"]`,
`validate_corpus` recounts it from the stored masks and checks that its cells
round the allocation targets of its weights ([below](#active-count-coverage)).

```python exec="1" source="block" result="text"
from scm_docs import corpus
from pymc_generator.data_generator import DataGenerator

errors = DataGenerator.validate_corpus(corpus())
print("validation errors:", errors or "none — schema OK ✅")
```

## The decomposition holds across the whole corpus

The generator records the four invariants in `diagnostics`. You can also check
the additive identity directly on the stacked arrays:

```python exec="1" source="block" result="text"
from scm_docs import corpus
import numpy as np

c = corpus()
lhs = c["outcome_raw"].astype(np.float64)
rhs = (c["baseline_raw"] + c["treatment_contribution_raw"].sum(-1)
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

The `diagnostics["signal"]` block includes the metric and carryover-kernel
versions plus `l_max`, `carryover_burn_in`, and `response_warmup_weeks`; use those
stored values rather than assuming configuration defaults when recomputing a
loaded shard. Its kernel metadata is
`carryover_kernel_semantics="normalized-causal-minmax-weibull-density"` at
`carryover_kernel_version=3`, reflecting pymc-marketing's min-max rescaling before
sum normalization. `frac_zero_contemporaneous_weight` reports the eligible direct
treatment share whose current-week normalized carryover weight is effectively zero.
It is reported for inspection, not used to gate a corpus.

```python exec="1" source="block" result="text"
from scm_docs import corpus
from pymc_generator.signal_diagnostics import check_signal_gate

signal = corpus()["diagnostics"]["signal"]
ok, lines = check_signal_gate(signal)
print("overall:", "PASS ✅" if ok else "see rows below")
for line in lines:
    print(" ", line)
```

The default gate also limits amplitude and collinearity failures:
`frac_contrib_rel_std_lt_001 <= 0.10` caps the share of direct treatments whose
true contribution is too small to matter in loss units, and
`frac_contrib_r2_gt_095 <= 0.10` caps the share whose contribution is a
near-perfect linear combination of the baseline and other treatments. A prior gate
PASS alone did not certify either property.

!!! note "Tiny demo corpus"
    The corpus on this page is deliberately small (4 tasks) so the docs build
    fast. Real training corpora use `n_cells` in the hundreds and `draws_per_cell`
    in the tens; the signal gate is tuned for those pool sizes, so a 4-task gate
    row may read FAIL purely from small-sample noise.

## Persistence

`save_corpus` writes a compressed, self-describing `.npz` (the `diagnostics` dict
round-trips as JSON); `load_corpus` reads it back.

```python exec="1" source="block" result="text"
from scm_docs import corpus
import pymc_generator as pg
import tempfile, os, numpy as np

c = corpus()
path = os.path.join(tempfile.mkdtemp(), "demo.npz")
pg.save_corpus(c, path)
loaded = pg.load_corpus(path)

print("saved to:", os.path.basename(path), f"({os.path.getsize(path):,} bytes)")
print("round-trips treatment_raw exactly:",
      bool(np.array_equal(c["treatment_raw"], loaded["treatment_raw"])))
print("diagnostics restored as dict:", isinstance(loaded["diagnostics"], dict))
```

The [`DataGenerator`](../reference/corpus.md#pymc_generator.data_generator.DataGenerator)
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

### Schema versions

New corpora use `diagnostics["schema_version"] = 5`. `save_corpus` and
`load_corpus` accept only the current version. Every pre-v5 archive is rejected
with a schema-version error: old archives cannot recover the new realised
component truth, and changing their stamp would not reconstruct it.

See the [schema reference](../reference/corpus.md#schema-versions-and-migration)
for realised field layouts and the breaking persistence contract.

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
| **Graph size** | how many nodes are live, and how cells cover the treatment × covariate counts | `n_treatments`, `n_covariates`, `n_latent`, their `*_active_range`s, and [`active_count_allocation` / `active_count_weights`](#active-count-coverage) |
| **Interactions** | how many arrows of each type | `edge_budget`, `min_no_direct_effect_treatments` |
| **Nonlinearity** | treatment-response family mix | `nonlinearity="diverse"` / `"linear"` |
| **Signal / noise** | coefficient & noise ranges | `**overrides` |
| **Texture** | treatments' and covariates' high-frequency drive | explicit noise, pulse, and walk ranges |
| **Trajectories** | which inputs launch, stop, run in flights, jump, cycle or drift | `trajectories="composable"` or an [archetype](trajectories.md#archetypes); the `*_inclusion_prob` knobs and component priors |

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

### Direct-null treatments

`min_no_direct_effect_treatments` reserves active treatments without a direct
`C→Y` edge. Their direct contribution is zero. A `cy` budget alone cannot
guarantee this because its arrow count is clamped to the active slots.

```python
cfg = pg.make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),   # 2–10 active channels per task
    edge_budget={"cy": (1, 10)},
    min_no_direct_effect_treatments=1,   # at least one active channel has no direct effect
)
```

The floor must be smaller than `n_treatments_active_range[0]` so the smallest
cell still has a direct treatment. Direct treatments remain scattered over active
slots. A direct-null treatment can still reach `Y` as a **feeder** through `C→C`;
set `edge_budget["cc"] = 0` as well if it must have no path to `Y`.

See the full configuration surface in the
[API reference](../reference/config.md).

## Active-count coverage

With active ranges wider than one value, every cell has its own active
treatment and covariate counts. By default each cell draws both counts
independently and uniformly, so a finite corpus covers their joint grid only in
expectation, and a small one can miss combinations entirely.
`active_count_allocation="stratified"` allocates the cells over the grid
instead: under uniform weights the cell counts per combination differ by at
most one, every combination appears once there are at least as many cells as
combinations, and `active_count_weights` reweights the combinations. The
[configuration reference](../reference/config.md#active-count-coverage) states
the exact rule, which cells it covers and what it does to seeds.

```python exec="1" source="block" result="text"
import pymc_generator as pg
from pymc_generator.data_generator import DataGenerator

sizes = dict(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=16,
             n_treatments_active_range=(2, 3), n_covariates_active_range=(1, 2),
             n_cells=4, draws_per_cell=1, seed=0)
independent = pg.make_scm_prior(**sizes)
stratified = pg.make_scm_prior(**sizes, active_count_allocation="stratified")

coverage = pg.active_count_coverage(pg.sample_prior_predictive(independent),
                                    prior=independent)
print("rows: n_treatments_active", coverage["n_treatments_active"],
      "| columns: n_covariates_active", coverage["n_covariates_active"])
print("independent n_cells:", coverage["n_cells"])

c = pg.sample_prior_predictive(stratified)
block = c["diagnostics"]["active_count_coverage"]
print("stratified  n_cells:", block["n_cells"])
print("block keys:", list(block))
print("validation errors:", DataGenerator.validate_corpus(c) or "none")
```

A stratified corpus records the realised coverage in
`diagnostics["active_count_coverage"]`; an independent corpus has no such key.
Keeping allocation at its default preserves same-environment numerical/model
arrays, not cross-schema metadata or archive-byte identity.

| Key | Value |
| --- | --- |
| `allocation` | `"stratified"` |
| `n_treatments_active` | the grid's rows: every count of the effective treatment range, ascending |
| `n_covariates_active` | the grid's columns: every count of the effective covariate range, ascending |
| `weights` | the weights used, as floats; all ones for `active_count_weights=None` |
| `n_cells` | distinct `cell_id`s per combination |
| `n_worlds` | tasks per combination |

- The matrices are nested lists indexed `[treatment row][covariate column]`:
  `n_cells[i][j]` counts the cells with `n_treatments_active[i]` active
  treatments and `n_covariates_active[j]` active covariates.
- Both counts come from the stored `treatment_active_mask`,
  `covariate_active_mask` and `cell_id` after any `n=` truncation, so they
  describe the corpus you hold: a truncated last cell counts as one cell, and
  only its retained worlds count.
- `validate_corpus` recomputes both exactly. It also checks the key set and
  `allocation`, the axes (consecutive counts within the mask widths), the
  weights (the grid's shape, finite, non-negative, some positive), that every
  task lies inside the grid, and that `n_cells` rounds the allocation targets
  of `weights` for that many cells, so a zero-weight combination holds no cell.
- Every value is a plain string, number or list, so the block round-trips
  through `save_corpus` / `load_corpus` unchanged. There is no schema-version
  bump: like the trajectory block, it is optional.

[`pg.active_count_coverage(corpus, prior=None)`](../reference/corpus.md#pymc_generator.active_counts.active_count_coverage)
returns the same grid axes and `n_cells` / `n_worlds` matrices for **any**
corpus, independent ones included; combinations no task reached show up as
zeros. It reads the grid from `prior` when given, else from the stored block,
so a corpus without the block needs `prior=`, the config that generated it. A
task outside the grid raises `ValueError`.

## Mechanism-prior metadata

Non-default effective [mechanism priors](../reference/config.md#mechanism-priors)
add `diagnostics["mechanism_priors"]`: the complete saturation shape supports,
MM scale distribution, and treatment/control reference-target ranges and scales.
The diagnostics block survives corpus save/load. In schema v5, opting into
richer mechanism priors also adds the optional
[realised mechanism arrays](../reference/corpus.md#realised-mechanism-fields)
and any enabled reference fields. Both blocks are absent for legacy defaults.
Targets are nominal responses before edge
gates; treatment references are post-carryover and control references are
pre-floor. The usual contribution arrays remain the executed, gated truth.


## Prior conditioning (ACE)

**ACE-style prior conditioning** makes the prior an *input*: with
`make_scm_prior(..., prior_conditioning=True)`, each **cell** draws a narrowed
prior interval per conditioned quantity —

```text
w  ~ U(w_lo, w_hi)            # interval width
lo ~ U(S_lo, S_hi − w)        # interval start
I  = [lo, lo + w]             # ⊆ the global support, by construction
```

— and that cell's parameter is drawn as `pm.Uniform(lo, lo + w)` instead of the
global support. The v1 conditioned set is `carryover_alpha` (geometric decay) and
`hill_shape` (Hill slope); width ranges are overridable per quantity via
`prior_cond_width_ranges`.
The Hill global support is the configured
`saturation_prior_ranges["hill"]["slope"]`, not a fixed module constant.

The draws are recorded in the corpus so consumers can expose them as
conditioning features:

```python exec="1" source="block" result="text"
import numpy as np
import pymc_generator as pg
from pymc_generator.slots import PRIOR_COND_LAYOUT

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
- The key is present **iff** `prior_conditioning=True`. Leaving conditioning
  disabled preserves same-environment numerical/model arrays, not cross-schema
  metadata or archive-byte identity.
- `diagnostics["prior_cond"]` echoes the layout, the supports, and the width
  ranges, so feature standardization can be derived from the corpus instead of
  duplicated in the consumer.
- The single-world path honors the same draw: `sample_scm` records the
  intervals in `SCM.extras["prior_cond"]` and `describe_scm` prints them
  alongside the mechanisms they bound.

## Composable input trajectories

A config that sets any [trajectory knob](trajectories.md) away from its default
(`SCMPrior.trajectory_metadata_enabled`) adds the six schedule/flag arrays below,
float64 realised component parameters, and `diagnostics["trajectory"]`.
They are present together or not at all; a default corpus omits this block.

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `treatment_components` | `(n_tasks, n_treatments, 8)` | `uint8` | `1` where the treatment carries the component, in `TRAJECTORY_COMPONENTS` order; constant within a cell |
| `covariate_components` | `(n_tasks, n_covariates, 8)` | `uint8` | the same for covariates |
| `treatment_activity` | `(n_tasks, n_time_steps, n_treatments)` | `uint8` | the gate schedule, `1` = on; all ones for an input without a gate |
| `covariate_activity` | `(n_tasks, n_time_steps, n_covariates)` | `uint8` | the same for covariates |
| `treatment_log_level_shift` | `(n_tasks, n_time_steps, n_treatments)` | `float32` | seasonal + trend + `log f` of every passed jump — the log of the level multiplier; `0.0` without a level component |
| `covariate_level_shift` | `(n_tasks, n_time_steps, n_covariates)` | `float32` | seasonal + trend + every passed jump size, in covariate units |

The component axis follows `pymc_generator.slots.TRAJECTORY_COMPONENTS`
(`hf, pulse, onset, offset, flighting, level_jump, seasonal, trend`). Padded
slots are zero in the schedule and parameter arrays. A config that only changes
`hf` / `pulse` probabilities stores activity `1` and shift `0.0` on every active input.
Off-weeks are exact: `treatment_raw == 0.0` there outside shocks, and
`covariates == 0.0`. `treatment_activity` is the schedule **gate**, not the
realised on-state — a [treatment shock](../reference/config.md#treatment-shocks)
holds its level through an off-week — so read it with `treatment_shock_mask`.

```python exec="1" source="block" result="text"
import pymc_generator as pg
from pymc_generator.data_generator import DataGenerator
from pymc_generator.slots import TRAJECTORY_ARRAY_FIELDS

cfg = pg.make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=40,
                        n_cells=2, draws_per_cell=2, seed=7, trajectories="composable")
c = pg.sample_prior_predictive(cfg)

for key in TRAJECTORY_ARRAY_FIELDS:
    print(f"{key:<26} {str(c[key].shape):<12} {c[key].dtype}")
block = c["diagnostics"]["trajectory"]
print("diagnostics['trajectory'] keys:", list(block))
print("n_inputs:", block["n_inputs"])
print("treatment prevalence:", {k: round(v, 2) for k, v in block["prevalence"]["treatment"].items()})
print("validation errors:", DataGenerator.validate_corpus(c) or "none")
```

- `components` is the layout list above.
- `inclusion_probs` echoes the **effective** probabilities,
  `SCMPrior.trajectory_inclusion_probs()`: `hf` / `pulse` read `0.0` wherever
  their texture range is off.
- `prevalence[input][component]` is the fraction of active `(task, input)`
  slots — `*_active_mask == 1` — whose stored flag is `1`, and
  `n_inputs[input]` is that denominator; padded slots never count. Both are
  computed from the stored arrays after any `n=` truncation, so they describe
  the corpus you hold, and `validate_corpus` recomputes them exactly.
- `parameter_fields` names every realised component field in canonical order.
  `jump_counts` records separate treatment and covariate event counts.
  Gate/jump week indexes and flighting periods/phases are `int64`; continuous
  leaves, including seasonal periods/phases, are `float64`. Unselected components
  and inactive nodes have zero padding.
- Metadata uses JSON-serializable lists, mappings and scalar values, so the block
  round-trips through `save_corpus` / `load_corpus` unchanged.

Schema v5 adds `trajectory_{input}_{component}_{leaf}` arrays for texture
magnitudes/probabilities, launch/stop weeks, flighting periods/on-weeks/phases,
jump weeks/factors/log-factors or signed sizes, sinusoid amplitudes/periods/phases,
and trend changes. Jump arrays have `(task, input_jump, input)` axes;
other leaves have `(task, input)` axes. The complete layout is in the
[schema reference](../reference/corpus.md#schema-versions-and-migration).
The validator reconstructs gates and level envelopes from this realised truth.
See the [trajectories guide](trajectories.md) for the archetypes it records.
