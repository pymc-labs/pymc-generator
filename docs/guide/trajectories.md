# Input trajectories

Real inputs are rarely always-on random walks. Channels launch mid-year, go
dark, run in flights, double their budget, follow a season or drift; covariates
step, cycle and trend. Each treatment and each covariate can carry,
**independently per input**, any subset of eight trajectory components on top
of its smoothed random walk, so one corpus can hold all of these shapes — and
record exactly which input carries which.

!!! note "End-to-end support"
    All eight components and fractional per-input inclusion work in corpus
    generation, `sample_scm`, and the reusable template. A sampled `SCM` records
    realised schedule parameters and can replay every forward output through
    `SCM.replay()`. The oracle uses the already-scheduled observed inputs,
    without applying the schedules again.

## Eight components

| Component | Kind | What it does | Priors (`SCMPrior` defaults) |
| --- | --- | --- | --- |
| `hf` | texture | iid weekly noise — the existing high-frequency texture | `*_hf_sigma_range` (set by the texture preset) |
| `pulse` | texture | Bernoulli pulses — the existing pulse texture, centred on covariates | `*_pulse_prob_range`, `*_pulse_amp_range` (texture preset) |
| `onset` | gate | off before a launch week, burn-in included: a cold launch | `*_onset_frac_range = (0.05, 0.25)` |
| `offset` | gate | off from a stop week on | `*_offset_frac_range = (0.6, 0.9)` |
| `flighting` | gate | on for `W` of every `P` weeks, at a random phase | `*_flighting_period_weeks_range = (4, 13)`, `*_flighting_duty_range = (0.3, 0.8)` |
| `level_jump` | level | `K` held steps, one per stratified slot of weeks `[1, T)` | `*_level_jump_count = 1`, `treatment_level_jump_factor_range = (0.5, 2.0)`, `covariate_level_jump_size_range = (-1.0, 1.0)` |
| `seasonal` | level | a sinusoid with its own amplitude, period and phase | `treatment_seasonal_amplitude_range = (0.1, 0.5)`, `covariate_seasonal_amplitude_range = (0.2, 1.0)`, `*_seasonal_period_weeks_range = (52.0, 52.0)` |
| `trend` | level | a straight line from 0 at reported week 0 | `treatment_trend_log_change_range = (-1.0, 1.0)`, `covariate_trend_change_range = (-1.0, 1.0)` |

`*` is `treatment` or `covariate`. The
[configuration reference](../reference/config.md#composable-input-trajectories)
gives every equation, bound and validation rule.

## How they compose

The sign of the input decides:

- A **covariate** is signed and linear. Its level components add on its own
  scale, and its gate multiplies the whole assembled covariate — parents
  included — so an off-week is exactly `0.0` for the covariate and for every
  term it feeds. Covariate trajectories can go negative.
- A **treatment** is non-negative. Its level components add on the **log**
  scale, and its gate multiplies the whole activated input:

    ```text
    C = gate · softplus(pre-activation) · exp(seasonal + trend) · Π jump factors
    ```

    so `C ≥ 0` always, `C == 0.0` exactly on off-weeks (outside treatment
    shocks, which hold their level), and a jump factor of 2 doubles the series
    exactly.

Gates combine by product — an input with `onset` and `flighting` runs in
flights after its launch — and level forms by sum. Every intervention variant
the [decomposition](decomposition.md) compares carries the same schedule, so
`contributions`, `indirect_effects` and the telescoping split stay exact.

## Who carries what

`<input>_<component>_inclusion_prob` is the chance that one input carries a
component. Each cell draws a flag for every input and every component
independently — concrete structure like the carryover family, shared by all of
the cell's draws. `1.0` puts the component on every input and `0.0` on none;
`hf` and `pulse` default to `1.0` and every other component to `0.0`, so the
default is today's texture. A component an input does not carry contributes
exactly nothing.

Ordinary active-size cell construction omits gate/level component priors when
no input of the cell carries them. Reusable templates instead keep cfg-admitted
gate/level priors wired and drawn even for all-off cells; runtime flags
neutralize their effects, not those prior draws. Legacy `hf` / `pulse` variables
retain their existing random-draw contract: switching their flags off does not
promise to eliminate those prior draws.

The realised **prevalence** — the fraction of active inputs carrying each
component — lands in `diagnostics["trajectory"]` next to the effective inclusion
probabilities. It is computed from the stored flags and recomputed by
`validate_corpus`; see the [corpus guide](corpus.md#composable-input-trajectories).

## Archetypes

Each archetype requested in
[#24](https://github.com/pymc-labs/pymc-generator/issues/24) is one
`make_scm_prior(trajectories=...)` value. An archetype puts its components on
every treatment **and** every covariate (inclusion `1.0`) and keeps the default
`hf` + `pulse` texture unless the table says otherwise:

| Archetype | `trajectories=` | Components | What it sets (at `n_time_steps=104`) |
| --- | --- | --- | --- |
| always-on with spikes (today) | `"always_on_spikes"` | `hf`, `pulse` | `*_pulse_prob_range=(0.05, 0.25)`, so every input fires with probability at least 5% per week; it leaves every trajectory knob at its default, so its corpora carry no trajectory block and `sample_scm` accepts it |
| periodic on/off | `"periodic_on_off"` | `flighting` | period `(4, 13)` weeks, duty `(0.3, 0.7)` |
| delayed start | `"delayed_start"` | `onset` | launch in weeks 10–42 |
| ramp-up | `"ramp_up"` | `onset`, `trend` | launch in weeks 1–16; trend `(1.0, 2.0)` — log-level on treatments, a ×2.7 to ×7.4 rise |
| decay / go to zero | `"decay_to_zero"` | `offset`, `trend` | stop in weeks 62–94; trend `(-1.5, -0.5)` |
| level doubling | `"level_doubling"` | `level_jump` (one step) | treatment factor `(2.0, 2.0)`; covariates step `+1` from a level pinned at `1` (`rw_covariate_mean_range=(1.0, 1.0)`), with covariate `hf` / `pulse` inclusion `0`, `rw_std_sigma=0.05`, `rw_baseline_std_sigma=1.0` and `dz_base_rate = zz_base_rate = 0` |
| seasonal | `"seasonal"` | `seasonal` | period 52 weeks; amplitude `(0.3, 0.6)` log-level on treatments, `(0.5, 1.0)` on covariates |
| trend | `"trend"` | `trend` | change `(-1.0, 1.0)` — log-level on treatments |

A signed covariate has no multiplicative level, so `level_doubling` writes its
doubling as an additive step equal to its (pinned, quieted, parentless) level.
`"texture"`, the default, leaves worlds unchanged, and `"composable"` mixes
every component ([below](#the-composable-mixture)).

Gate weeks, flighting periods and the seasonal period are **derived from the
effective `n_time_steps` and `query_frac`**, so every gated input keeps at least
two on-weeks inside the support prefix
`S = min(n_time_steps - n_query, n_time_steps // 2)` that every task keeps:

| Derived range | Rule (`T = n_time_steps`) | `T = 104` | `T = 52` |
| --- | --- | --- | --- |
| `delayed_start` launch weeks | `[lo, max(lo, min(round(0.4 T), S - 2))]`, `lo = max(1, min(round(0.1 T), S - 2))` | 10–42 | 5–21 |
| `ramp_up` launch weeks | `[1, max(1, min(round(0.15 T), S - 2))]` | 1–16 | 1–8 |
| `decay_to_zero` stop weeks | `[lo, max(lo, min(T - 1, round(0.9 T)))]`, `lo = max(S, round(0.6 T))` | 62–94 | 31–47 |
| `periodic_on_off` periods | the widest `(min(4, hi), hi)` with `hi <= min(13, S // 2)` that keeps two on-weeks in `S` weeks at duty `0.3` | 4–13 | 4–13 |
| `seasonal` period | `max(4, min(52, T // 2))` | 52 | 26 |

Each week is written as the fraction `(week + 0.5) / T`, so the drawn weeks are
exactly the intended ones. At a horizon where a preset cannot keep two on-weeks
of context, `make_scm_prior` raises a `ValueError` before `**overrides` apply —
overrides cannot rescue the preset there, so set the trajectory knobs directly
instead. The message names the next longer `n_time_steps` at which the complete
config, your overrides included, validates — `"composable"` at `n_time_steps=8`
names `14` with the default `l_max=8` and `102` with `l_max=52`. That search is
bounded (100,000 longer horizons, at most 1,000 full validations) and names no
horizon if it finds no fit, and any error your own fields raise at the requested
horizon is attached as the exception's `__cause__`, so an invalid override is
not hidden. Feasibility is not monotone in the horizon, so a longer horizon is
not guaranteed to fit ([example](../reference/config.md#presets)).

### Reproduced from a corpus

Every block below runs during the docs build. It generates one small corpus per
archetype — two treatments, one covariate, `n_time_steps=52`, two cells of one
draw each — and reads the archetype back from the **stored arrays**. First, the
realised prevalence each corpus reports:

```python exec="1" source="block" result="text" session="trajectories"
import numpy as np
import pymc_generator as pg
from pymc_generator.presets import TRAJECTORY_ARCHETYPES


def archetype_corpus(name):
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=52,
                            n_cells=2, draws_per_cell=1, seed=3, trajectories=name)
    return pg.sample_prior_predictive(cfg)


corpora = {name: archetype_corpus(name) for name in TRAJECTORY_ARCHETYPES}

for name, c in corpora.items():
    block = c["diagnostics"].get("trajectory")
    if block is None:
        print(f"{name:<17} no trajectory block: only texture ranges changed")
        continue
    for x in ("treatment", "covariate"):
        carried = ", ".join(f"{k} {v:.2f}" for k, v in block["prevalence"][x].items() if v)
        print(f"{name if x == 'treatment' else '':<17} {x:<10} {carried}")
```

`always_on_spikes` changes texture ranges only, so — like a default corpus — it
carries no trajectory block. Its spikes show in `treatment_raw` itself, which
never switches off:

```python exec="1" source="block" result="text" session="trajectories"
x = corpora["always_on_spikes"]["treatment_raw"].astype(np.float64)  # (task, week, treatment)
ratio = x.max(axis=1) / np.median(x, axis=1)
print("weeks with treatment > 0:", f"{(x > 0).mean():.0%}")
print("peak / median per treatment:", ", ".join(f"{r:.1f}" for r in ratio.ravel()))
```

The gate archetypes in `treatment_activity` / `covariate_activity` (first task
of each corpus, `█` = on): flights at each input's own period, duty and phase,
late launches, and stops.

```python exec="1" source="block" result="text" session="trajectories"
def strip(column):
    return "".join("█" if on else "·" for on in column)


print(" " * 23 + "".join(f"{week:<10}" for week in range(0, 52, 10)))
for name in ("periodic_on_off", "delayed_start", "ramp_up", "decay_to_zero"):
    c = corpora[name]
    rows = [(f"C{k + 1}", c["treatment_activity"][0, :, k]) for k in range(2)]
    rows.append(("Z1", c["covariate_activity"][0, :, 0]))
    for i, (label, column) in enumerate(rows):
        print(f"{name if i == 0 else '':<17} {label:<5}{strip(column)}")
```

Off-weeks are exact zeros, on-weeks stay positive, and a launch is cold —
nothing carries over from before it:

```python exec="1" source="block" result="text" session="trajectories"
gated = [corpora[name] for name in ("periodic_on_off", "delayed_start", "ramp_up", "decay_to_zero")]
print("treatment_raw == 0.0 on every off-week:",
      all((c["treatment_raw"][c["treatment_activity"] == 0] == 0.0).all() for c in gated))
print("treatment_raw  > 0.0 on every on-week: ",
      all((c["treatment_raw"][c["treatment_activity"] == 1] > 0.0).all() for c in gated))
print("covariates  == 0.0 on every off-week:  ",
      all((c["covariates"][c["covariate_activity"] == 0] == 0.0).all() for c in gated))

c = corpora["delayed_start"]
launch = c["treatment_activity"].argmax(axis=1)  # first on-week, per (task, treatment)
print("direct contribution == 0.0 before launch:",
      all((c["treatment_contribution_raw"][t, :launch[t, k], k] == 0.0).all()
          for t in range(2) for k in range(2)))
```

The level archetypes in `treatment_log_level_shift` (a log-level) and
`covariate_level_shift` (covariate units), first task of each corpus:

```python exec="1" source="block" result="text" session="trajectories"
c = corpora["level_doubling"]
for k in range(2):
    shift = c["treatment_log_level_shift"][0, :, k]
    week = int(np.flatnonzero(shift)[0])
    print(f"level_doubling C{k + 1}: log-shift 0 until week {week}, then {shift[week]:.7f} = log 2")
z, step = c["covariates"][0, :, 0], c["covariate_level_shift"][0, :, 0]
week = int(np.flatnonzero(step)[0])
print(f"level_doubling Z1: step +{step[week]:.1f} at week {week}; mean level "
      f"{z[:week].mean():.2f} before, {z[week:].mean():.2f} after")

c = corpora["seasonal"]
for k in range(2):
    shift = c["treatment_log_level_shift"][0, :, k].astype(np.float64)
    period = 52 / (1 + int(np.abs(np.fft.rfft(shift))[1:].argmax()))
    print(f"seasonal       C{k + 1}: amplitude {np.abs(shift).max():.2f}, "
          f"dominant period {period:.0f} weeks")

c = corpora["trend"]
weeks = np.arange(52)
for k in range(2):
    shift = c["treatment_log_level_shift"][0, :, k].astype(np.float64)
    gap = np.abs(shift - shift[-1] * weeks / 51).max()
    print(f"trend          C{k + 1}: exactly 0 at week 0: {shift[0] == 0}; {shift[-1]:+.2f} "
          f"at week 51, off a straight line by {gap:.0e}")

c = corpora["ramp_up"]
for k in range(2):
    launch = int(np.flatnonzero(c["treatment_activity"][0, :, k])[0])
    shift = c["treatment_log_level_shift"][0, :, k]
    print(f"ramp_up        C{k + 1}: launches at week {launch} at log-shift "
          f"{shift[launch]:+.2f}, rising to {shift[-1]:+.2f} by week 51")

c = corpora["decay_to_zero"]
for k in range(2):
    stop = int(np.flatnonzero(c["treatment_activity"][0, :, k] == 0)[0])
    shift = c["treatment_log_level_shift"][0, :, k]
    print(f"decay_to_zero  C{k + 1}: log-shift down to {shift[stop - 1]:+.2f} by week "
          f"{stop - 1}, off from week {stop}")
```

## The composable mixture

`trajectories="composable"` lets every input carry every component at
moderate prevalence and widens the input level and variation priors
([values](../reference/config.md#presets)). A small corpus reports each
effective inclusion probability next to its realised prevalence:

```python exec="1" source="block" result="text" session="trajectories"
cfg = pg.make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=52,
                        n_cells=6, draws_per_cell=1, seed=11, trajectories="composable")
mixed = pg.sample_prior_predictive(cfg)
block = mixed["diagnostics"]["trajectory"]
p, realised = block["inclusion_probs"], block["prevalence"]

print("active inputs:", block["n_inputs"])
print(f"{'component':<11}{'treatment p':>12}{'realised':>10}{'covariate p':>13}{'realised':>10}")
for comp in block["components"]:
    print(f"{comp:<11}{p['treatment'][comp]:>12.2f}{realised['treatment'][comp]:>10.2f}"
          f"{p['covariate'][comp]:>13.2f}{realised['covariate'][comp]:>10.2f}")
```

Prevalence counts active `(task, input)` pairs, and every draw of a cell
repeats that cell's flags, so six cells only roughly track the probabilities;
the two converge as the number of cells grows. Every corpus on this page passes
`validate_corpus`, trajectory checks included:

```python exec="1" source="block" result="text" session="trajectories"
from pymc_generator.data_generator import DataGenerator

errors = {name: DataGenerator.validate_corpus(c) for name, c in {**corpora, "composable": mixed}.items()}
print("validation errors:", {name: e for name, e in errors.items() if e} or "none")
```

## Realism with schedules

A deliberate schedule must not be rejected for looking unlike organic texture:
a launch or a flight makes a treatment's peak-to-median ratio arbitrary. So
once any schedule component is enabled, the
[realism filter](foundation.md#the-realism-filter) splits its checks:

- **Actual series:** every drawn array must be finite, and outcome
  non-negative.
- **Natural path:** the treatment and outcome spike guards read
  `treatments_natural` / `outcome_natural` — the treatment recursion with no
  treatment schedule and no shock (natural parents), and the outcome built from
  it.
- **Treatment CV floor:** a direct treatment passes when its natural series
  reaches the floor or — only if it carries a gate or level component — when
  its own schedule applied to that natural series does
  (`activity × natural × exp(log_shift)`). A schedule can rescue a texture-free
  walk; a shock never counts.

Covariate components are **not** filtered out: covariates enter the natural
path as realised — as the natural treatments' `Z → C` parents and inside the
natural outcome — so the outcome spike guard still screens them. When a cell
wires no treatment schedule and has no shocks, the natural pair is simply the
actual pair; with shocks it is the existing unshocked pair.

Treatment shocks hold `treatment_shock_level_multiplier * softplus(rw_c_mean)`
regardless of the envelope and override the gate: during an event the
treatment is the held level even on an off-week. `treatment_activity` is
therefore the schedule gate, not the realised on-state — read it together with
`treatment_shock_mask`.

## Seeds

With every trajectory knob at its default nothing new is drawn: the controls
leave same-seed numerical/model arrays and worlds unchanged. This is not a
cross-version archive-identity guarantee; schema v5 changes the version stamp
and saved corpus bytes. Inclusion flags draw from child streams spawned off the
corpus's numpy generator and consume none of its state. A cell or world that
carries no schedule component and keeps `hf` / `pulse` on every input whose
range is live keeps the default draws. But once a cell wires a component, the
random variables its compiled draw reaches can change, so that cell's *other*
draws generally change at the same seed too: PyMC assigns streams by discovery
order. Inside one compiled template, per-cell flags are data and move no draw.
Corpora of two configs line up cell by cell only while their acceptance counts
match. Details: [randomness and seeds](../reference/config.md#randomness-and-seeds).

## Limits

- **Template scope.** Trajectory flags are per-cell runtime inputs. Treatment
  shocks, prior conditioning and non-fixed confounding-strength ranges remain
  outside the experimental template's supported configurations. Template
  padding changes RV shapes, so matching seeds alone do not align its draws
  with ordinary generation; forward parity requires the same realised inputs.
- **κ anchors exclude the components.** `saturation_scale` stays
  parameter-only, so a doubled treatment runs at twice its κ-relative level.
- **Exact covariate levels need component-free covariates.** A covariate's
  expected level is exactly `rw_z_mean` plus its upstream `Z → Z` terms only
  while neither it nor any upstream `Z → Z` ancestor carries a gate or level
  component.
- **Inputs only.** The baseline process is unchanged (a smoothed walk with no
  explicit seasonal term), and no Fourier series are added as covariates.

!!! tip "Next"
    [The posterior oracle](oracle.md), or the
    [corpus guide](corpus.md#composable-input-trajectories) for the stored
    trajectory arrays.
