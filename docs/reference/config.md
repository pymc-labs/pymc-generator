# Configuration

Build configs through [`make_scm_prior`](#prior_generator.presets.make_scm_prior)
— it pins the layout and enables the supported *diverse* channel texture — and
reach for [`SCMPrior`](#prior_generator.sampler.SCMPrior) directly only when you
need a field the preset does not surface.

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
The drawn scalar is persisted as `confounding_strength` with shape `(N,)` and
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

- `n_channel_shocks: int = 0` is the exact number `S` of shocks in **each**
  world. It must be an integer in `[0, T]` and must fit at maximum length:
  `S * channel_shock_length_range[1] <= T`.
- `channel_shock_length_range: tuple[int, int] = (2, 2)` has integral bounds
  `1 <= lo <= hi <= T`. When `S > 0`, `lo` must be at least 2 so every
  intervention creates a spend-visible held-level plateau. One-week ranges
  remain valid only while shocks are disabled.
- `channel_shock_level_range: tuple[float, float] = (0.0, 0.0)` has finite
  bounds `0 <= lo <= hi`.

Schedule event `s` is placed in deterministic stratified reported-window slot
`[floor(s*T/S), floor((s+1)*T/S))`; its feasible start and sampled length stay
inside that slot. Thus events are globally non-overlapping, including when a
channel is selected more than once. Channels are sampled uniformly **with
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
