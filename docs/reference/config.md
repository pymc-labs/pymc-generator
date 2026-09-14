# Configuration

Build configs through [`make_scm_prior`](#prior_generator.presets.make_scm_prior)
— it pins the layout and enables the supported *diverse* channel **and control**
texture — and reach for [`SCMPrior`](#prior_generator.sampler.SCMPrior) directly
only when you need a field the preset does not surface.

## Control texture

`rw_control_mean_range` sets the signed control drive's walk mean. It does not
set latent demand's mean: demand is anchored to mean zero and unit scale, with
its magnitude represented by the outgoing loadings.

A control's own drive is otherwise a smoothed random walk, i.e. the same
function class as the smooth baseline walk `RW_B`. The two are then nearly
collinear over a typical horizon, so the `Z → Y` loading `rho_zy` trades off
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
(`E[F_m - RW_m] = 0`), so a control's expected level stays `rw_z_mean` — and
that claim is EXACT, because a control's equation applies no activation — which
is why the parameter-only saturation reference levels are unchanged. (Channel
pulses are
deliberately uncentred — a channel is positive
and its realized level may rise above the walk anchor; a channel's equation
also applies `softplus`, so its own expected level sits strictly above its
parameter-only anchor regardless of pulses.) Centring holds in
expectation conditional on the drawn parameters, not as the temporal mean of
every finite path.

All three default to `(0.0, 0.0)` on a raw `SCMPrior`: no term enters the graph,
the parameters degenerate to constants, and the corpus is byte-identical to one
generated before the mechanism existed.

## Intercept floor

`baseline_floor: float | None = None` censors the intercept walk:

$$B_t = \max(\mathrm{RW}_{B,t},\ \texttt{baseline\_floor}).$$

A **censored** walk, not a softplus, so `B` can sit exactly *at* the floor —
"zero or above, never below". `None` keeps the signed walk.

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(baseline_floor=0.0)   # baseline >= 0 by construction
```

Why this is safe: the intercept carries **no parents**. Latent demand and the
controls enter `Y` directly, so the floor clips one additive term and
`control_contribution[:, m]` stays exactly `g_zy[m] * rho_zy[m] * Z[:, m]`.
Flooring a sum that contained the parents would break that identity.

The floor is a pure clip: it adds no random variable and consumes no RNG, so a
floor that never binds reproduces the unfloored corpus byte for byte (pinned by
`test_the_intercept_floor_is_a_pure_clip_and_consumes_no_rng`). Two consequences
to know about:

- **Sales is not censored.** A clamp on `Y` would censor the *observation*,
  putting every world outside the additive-Gaussian class an MMM likelihood —
  including this package's own oracle — can represent. Non-negative sales stays
  enforced by the acceptance filter. Measured on the shipped prior, sales sits
  28–83 observation-noise σ above zero and a negative intercept week occurs in
  ~0.08% of weeks, so the floor is a guarantee rather than a frequent
  intervention.
- **The oracle's `latent="marginal"` mode raises** for a floored config, because
  `max(RW_B, floor)` is not Gaussian and the analytic covariance would be wrong.
  Use `latent="sampled"`, which applies the identical clip.

### What the floor clips: `baseline_floor_scope`

`baseline_floor_scope: Literal["intercept", "non_media"] = "intercept"`.

A floored *intercept* alone does not make sales non-negative: a large negative
$\rho_m Z_m$ can still drag the non-media total under (measured on a stress
fixture: 112 negative weeks with `scope="intercept"`). `scope="non_media"`
clips the **running total** as each parent joins, in the locked order
intercept → confounders ($j$ ascending) → controls ($m$ ascending):

$$A^{(0)} = B,\qquad A^{(i)} = \max\!\big(A^{(i-1)} + \text{node}_i,\ \texttt{floor}\big),\qquad A = A^{(\text{last})}.$$

Each per-node column is the telescoping difference that node caused,
$A^{(i)} - A^{(i-1)}$ — the same construction `indirect_effects_by_source` uses
for channels. So:

- the non-media total is `≥ floor` **by construction**: a negative control
  effect is credited only down to the floor and the excess is absorbed instead
  of pushing sales negative (same fixture: 0 negative weeks, 111 weeks sitting
  exactly at the floor);
- the columns still sum **exactly** to the total, so the decomposition identity
  is untouched (measured identity error 1.8e-15);
- where the floor does not bind, every column equals the linear split
  ($g^{zy}_m \rho_m Z_m$) and the persisted corpus is byte-identical to the
  unfloored one.

The cost is that a per-node column is **no longer linear in its node** where the
floor binds: a linear MMM's $\rho_m Z_m$ term cannot reproduce the absorbed
part. Nodes with no edge are skipped entirely rather than added with a zero
coefficient, so they neither take part in the clipping order nor change which
innovations the graph reaches.

#### Why sales still is not strictly guaranteed

With `scope="non_media"` every term of the sales **mean** is non-negative
($A \ge 0$, media $\ge 0$). What remains is the additive observation noise
$\mathrm{RW}_Y$, which is symmetric and unbounded, so
$P(Y<0)>0$ is unavoidable for *any* additive-Gaussian outcome — that is a
property of the likelihood, not of this generator. Strictness would require
censoring the observation or bounding the noise, both of which change the
function class an MMM can represent. Measured margin under the absorbing scope:
the sales mean sits **88 σ** of observation noise above zero at its worst week
over 12 worlds, so the residual probability is ~$10^{-1700}$, and the
acceptance filter remains the exact backstop.

### Decomposition columns

`baseline_intrinsic` is the intercept **alone** (`B`), and the iid observation
noise is its own `sales_noise` column, so the persisted identity is

```text
baseline_intrinsic + sales_noise + Σ control_contribution
  + Σ confounder_contribution + Σ contributions
  + Σ indirect_effects_by_source == sales
```

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

## Direct-null channels

`min_no_direct_effect_channels: int = 0` is the minimum number of **active**
channels each cell leaves without a direct `C→Y` edge. Their **direct**
contribution is zero; their total causal effect need not be. A `cy` budget is
an absolute arrow count clamped to the eligible slots, so a cell drawing
`n_treatments_active = 2` under `cy=(2, 10)` has both channels live and no
negative class at all.

```python
from prior_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),
    edge_budget={"cy": (1, 10)},
    min_no_direct_effect_channels=1,
)
```

- The maximum direct-channel count is
  `max(1, n_treatments_active - min_no_direct_effect_channels)`. The existing
  budget or Bernoulli sampler operates subject to this cap, and the mandatory
  `cy ≥ 1` guard still applies.
- It applies to the Bernoulli path too: without a `cy` budget, surplus live
  channels are demoted uniformly at random after the per-slot draw.
- Live channels are scattered over **all** active slots, so slot index carries no
  information about the label.
- `min_no_direct_effect_channels` must be `< n_treatments_active_range[0]`; a floor the
  smallest drawable cell could not honour is a validation error, not a silently
  dropped constraint.
- A direct-null channel can still
  reach `Y` through another channel (`worlds.channel_role` reports that as a
  `feeder`); pin `edge_budget={"cc": 0}` for "no path to `Y`".
- `0` (the default) is inert: identical draws, no extra RNG, and
  `diagnostics["min_no_direct_effect_channels"]` records the resolved value.

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
