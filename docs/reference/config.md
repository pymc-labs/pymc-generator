# Configuration

Build configs through [`make_scm_prior`](#prior_generator.presets.make_scm_prior)
— it pins the layout and enables the supported *diverse* channel **and control**
texture — and reach for [`SCMPrior`](#prior_generator.sampler.SCMPrior) directly
only when you need a field the preset does not surface.

## Control texture

A control's own drive is otherwise a smoothed random walk, i.e. the same
function class as the smooth baseline walk `RW_B`. The two are then nearly
collinear over a typical horizon, so the `Z → B` loading `rho_zb` trades off
against baseline drift and is only weakly identified. These three knobs add the
high-frequency content a smooth baseline cannot mimic:

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(            # the diverse preset already sets these
    control_hf_sigma_range=(0.1, 0.8),
    control_pulse_prob_range=(0.0, 0.25),
    control_pulse_amp_range=(0.5, 3.0),
)
smooth = make_scm_prior(         # pre-texture controls, byte-identical corpora
    control_hf_sigma_range=(0.0, 0.0),
    control_pulse_prob_range=(0.0, 0.0),
    control_pulse_amp_range=(0.0, 0.0),
)
```

For control `m`, with `s = rw_z_std[m]`:

$$
F_m = \mathrm{RW}_m + \underbrace{q\,s}_{\texttt{control\_hf\_sigma}}\eta_{tm}
    + \underbrace{r\,s}_{\texttt{control\_pulse\_amp}}\,(h_{tm} - p),\qquad
h_{tm}\sim\mathrm{Bernoulli}(p).
$$

- `control_hf_sigma_range: tuple[float, float] = (0.0, 0.0)` — the iid weekly
  factor `q`, **relative to the control's own walk std** (a signed control has
  no positive level to anchor on). Bounds must be finite with `0 <= lo <= hi`.
- `control_pulse_prob_range: tuple[float, float] = (0.0, 0.0)` — the per-week
  fire probability `p`, with `0 <= lo <= hi <= 0.5`. Its upper bound is the
  on/off switch for the pulse term.
- `control_pulse_amp_range: tuple[float, float] = (0.0, 0.0)` — the pulse
  amplitude factor `r`, also relative to the walk std. It is an *amplitude*,
  not a standard deviation: the term's variance is
  `(r*s)**2 * p * (1 - p)`. Enabling `control_pulse_prob_range` with a
  zero-only amplitude is a validation error.

The pulse is **centred**: subtracting `p` makes both added terms mean-zero
(`E[F_m - RW_m] = 0`), so a control's expected level stays `rw_z_mean` and the
parameter-only saturation anchors are unchanged. (Channel pulses are
deliberately uncentred — a channel is positive
and its realized level may rise above the walk anchor.) Centring holds in
expectation conditional on the drawn parameters, not as the temporal mean of
every finite path.

All three default to `(0.0, 0.0)` on a raw `SCMPrior`: no term enters the graph,
the parameters degenerate to constants, and the corpus is byte-identical to one
generated before the mechanism existed.

## Baseline walk scale

`rw_baseline_std_sigma: float | None = None` controls **only** the `RW_B`
baseline-walk HalfNormal scale. Its effective value is dynamic: when it is
`None`, it follows the current `rw_std_sigma`; when supplied, the explicit
positive finite value overrides that baseline scale. It does not alter the
demand, control, channel, or sales-noise walk settings.

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(rw_std_sigma=1.2)  # RW_B also uses 1.2
cfg = make_scm_prior(rw_std_sigma=1.2, rw_baseline_std_sigma=0.35)
```

## Shared baseline/channel innovations

`confounding_strength_range: tuple[float, float] | None = None` enables an
optional per-world scalar `rho`. Bounds must be finite and satisfy
`0 <= lo <= hi <= 0.95`. For each channel/week, the generator replaces its
standard innovation with

$$
\epsilon_c' = \sqrt{1 - \rho^2}\,\epsilon_c + \rho\,\epsilon_b.
$$

This preserves each channel innovation's unit marginal variance while inducing
shared correlation with the baseline innovation. `None` retains the legacy
independent path (and reports `rho = 0`); a degenerate range fixes the value.
The drawn scalar is persisted as `confounding_strength` with shape `(n_tasks,)` and
dtype `float32` (and is available in single-world data/parameters).

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(confounding_strength_range=(0.2, 0.5))
```

The posterior oracle is a plug-in model for observed channels and controls. It
does **not** model their likelihood conditional on latent demand or
`epsilon_b`; consequently, it does not use channel information about this
rho-induced baseline correlation to infer demand. It remains a
structure-known upper-bound oracle, not exact joint conditioning.

## Channel shocks

The optional held-level intervention API is:

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(
    n_channel_shocks=2,
    channel_shock_length_range=(2, 4),
    channel_shock_level_range=(0.0, 0.5),
)
```

- `n_channel_shocks: int = 0` is the exact number `n_shocks` of shocks in
  **each** world. It must be an integer in `[0, n_time_steps]` and must fit at
  maximum length:
  `n_shocks * channel_shock_length_range[1] <= n_time_steps`.
- `channel_shock_length_range: tuple[int, int] = (2, 2)` has integral bounds
  `1 <= lo <= hi <= n_time_steps`. When `n_shocks > 0`, `lo` must be at least 2
  so every intervention creates a spend-visible held-level plateau. One-week
  ranges remain valid only while shocks are disabled.
- `channel_shock_level_range: tuple[float, float] = (0.0, 0.0)` has finite
  bounds `0 <= lo <= hi`.

Schedule event `s` is placed in deterministic stratified reported-window slot
`[floor(s*n_time_steps/n_shocks), floor((s+1)*n_time_steps/n_shocks))`; its
feasible start and sampled length stay inside that slot. Thus events are
globally non-overlapping, including when a channel is selected more than once.
Channels are sampled uniformly **with
replacement** from direct (`g_cy`) channels. The held level is sampled ex ante
as `channel_shock_level_multiplier * softplus(rw_c_mean)` for the selected
channel. Starts are reported-window indices (not burn-in-offset full-horizon
indices).

During an event, observed spend is clamped to that level. Clamping is the whole
intervention: the clamped path then feeds the ordinary normalized causal
adstock kernel, so carryover from pre-event spend decays into the window rather
than being discarded, and a zero held level reaches an exactly zero direct
response only once the full kernel span lies inside the window. Outside an
event the natural spend process resumes, and downstream channel recursion still
observes the clamped parent. This keeps every generated response inside the
function class a standard MMM adstock can represent from the same observed
spend. It is a known held-level intervention design, **not** a conventional
spend-only lift test.

## Dead channels

`min_dead_channels: int = 0` is the minimum number of **active** channels each
cell must leave without a direct `C→Y` arrow — spend observed, true contribution
exactly zero. It exists because `edge_budget["cy"]` cannot express it: a budget is
an absolute arrow count clamped to the eligible slots, so a cell drawing
`n_treatments_active = 2` under `cy=(2, 10)` has both channels live and no
negative class at all.

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),
    edge_budget={"cy": (1, 10)},
    min_dead_channels=1,
)
```

- The per-cell live count becomes `min(cy draw, max(1, n_treatments_active −
  min_dead_channels))`; the degenerate `cy ≥ 1` guard always wins, so a cell is
  never left without a direct channel. The example spans 1–9 live channels over
  2–10 active ones.
- It applies to the Bernoulli path too: without a `cy` budget, surplus live
  channels are demoted uniformly at random after the per-slot draw.
- Live channels are scattered over **all** active slots, so slot index carries no
  information about the label.
- `min_dead_channels` must be `< n_treatments_active_range[0]`; a floor the
  smallest drawable cell could not honour is a validation error, not a silently
  dropped constraint.
- "Dead" means no **direct** arrow. Under a `cc` budget a dead channel can still
  reach `Y` through another channel (`worlds.channel_role` reports that as a
  `feeder`); pin `edge_budget={"cc": 0}` for "no path to `Y`".
- `0` (the default) is inert: identical draws, no extra RNG, and
  `diagnostics["min_dead_channels"]` echoes the resolved value in every corpus.

## Identifiability labels

`include_identifiability_labels: bool = True` controls the nested
`identifiability` block containing the dense, truth-derived `signal_metrics`
and `signal_metric_valid` arrays. Setting it to `False` omits that block while retaining the aggregate
`diagnostics["signal"]` quality report. It changes no model input, random draw,
or generated observable array; use the disabled form for feature-only shards.

::: prior_generator.presets.make_scm_prior

::: prior_generator.sampler.SCMPrior
    options:
      show_source: false
      members: false
