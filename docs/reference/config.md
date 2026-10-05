# Configuration

Build configs through [`make_scm_prior`](#pymc_generator.presets.make_scm_prior)
— it pins the layout and enables the supported *diverse* treatment **and covariate**
texture — and reach for [`SCMPrior`](#pymc_generator.sampler.SCMPrior) directly
only when you need a field the preset does not surface.

## Mechanism priors

`SCMPrior.saturation_prior_ranges` configures the dimensionless shape supports
for each nonlinear family. Supply the complete nested mapping below; there are
no shape parameters for `linear`.

| Family | Parameter | Default range | Domain |
| --- | --- | --- | --- |
| `hill` | `slope` | `(1.0, 3.0)` | `0 < lo <= hi` |
| `hill` | `kappa_mult` | `(0.7, 1.5)` | `0 < lo <= hi` |
| `logistic` | `lam` | `(0.5, 3.0)` | `0 < lo <= hi` |
| `michaelis_menten` | `kappa_mult` | `(0.7, 1.5)` | `0 < lo <= hi` |
| `tanh` | `c` | `(0.3, 1.5)` | `0 < lo <= hi` |
| `root` | `alpha` | `(0.3, 0.9)` | `0 < lo <= hi <= 1` |

Bounds must be finite real numbers, not booleans or numeric strings, and must
fit the existing float32 corpus-storage maximum. Unknown or missing families
and parameters raise `ValueError`. Equal bounds fix a parameter exactly and
consume no random draw. The defaults are the former module-constant ranges;
each configuration owns its mapping.

Schema-v5 validation applies the same domain limits to persisted primitive
shape supports and realised values, even for an unused family. Derived reference
and saturation anchors are not primitive shape endpoints and are not capped by
this rule; their float64 storage preserves legitimate parameter amplification.

Shape parameters are uniform on their configured ranges, except that
`mm_scale_prior="log_uniform"` opts into

$$
\log K_k \sim \mathrm{U}(\log a,\log b),\qquad
\kappa_k = K_k r_k,\qquad
p(K_k)=\frac{1}{K_k\log(b/a)}.
$$

Here `(a, b)` is
`saturation_prior_ranges["michaelis_menten"]["kappa_mult"]` and `r_k` is the
existing parameter-only `saturation_scale`. A broad range such as `(0.01, 100)`
spreads the half-saturation point over four orders of magnitude: small `K`
saturates early; large `K` is nearly linear around the anchor.
`mm_scale_prior="uniform"` is the **default**, retaining the old law and seeded
outputs. Fixed MM bounds remain exactly their original value, including in
log-uniform mode. Distinct bounds that collapse to the same float64 logarithm
are rejected rather than replaced by an out-of-support constant, and
`exp(log K)` is clipped to `[a, b]` so rounded logarithmic bounds never place
`K` an ulp outside the configured support. The lower MM bound times the
minimum anchor `1e-8` must stay positive, so a zero treatment input never
evaluates `0/0`.

Hill [prior conditioning](../guide/corpus.md#prior-conditioning-ace) uses the configured `slope`
support too. Its width ranges must fit that support, and with conditioning
enabled every width must exceed `8 * eps32 * max(|support|)`, the narrowest
interval float32 `prior_cond` labels resolve, so stored intervals still match
the interval actually used. Narrow or very large supports may require
overriding `prior_cond_width_ranges["hill_shape"]`. With
`prior_conditioning=False`, only explicitly configured width ranges are checked
against their supports.

### Reference-contribution priors

These optional priors replace raw outcome coefficients, not input-to-input
edge loadings:

| Field | Default | Meaning |
| --- | --- | --- |
| `treatment_reference_contribution_range` | `None` | Per-treatment `q_k ~ U(lo, hi)`; `0 <= lo <= hi`, `hi > 0` |
| `treatment_reference_multiplier` | `1.0` | Positive reference input relative to `r_k` |
| `covariate_reference_contribution_range` | `None` | Per-control `q_m ~ U(lo, hi)`; signed bounds allowed |
| `covariate_reference_scale` | `1.0` | Fixed positive reference input in covariate units |

With treatment targets enabled, for the drawn saturation shape:

$$
x_{\mathrm{ref},k}=m r_k,\qquad
\beta_k=\frac{q_k}{f_k(x_{\mathrm{ref},k};r_k)},\qquad
\beta_k f_k(x_{\mathrm{ref},k};r_k)=q_k.
$$

This is mathematically `q_k / f_k(m; 1)`. The implementation evaluates the
actual parameter-only input and anchor with the shared response function,
avoiding a separately rounded calibration point for steep curves.

The target is the nominal response at a **post-carryover saturation input**,
before the `g_cy` edge gate. With a valid unit-mass carryover kernel, holding
raw treatment at `x_ref` for the kernel's full span gives this steady-state
response. It is not a claim about finite-window means, cold starts, trajectories,
or an invalid legacy Weibull kernel that produces zero carryover. A treatment
without a direct outcome edge still contributes exactly zero.

For controls, `rho_zy[m] = q_m / covariate_reference_scale` (computed as `q_m`
times the float64 reciprocal of the scale). This is the nominal
linear response at the fixed positive reference input, **before `g_zy` and any
absorbing baseline floor**. Negative targets mean negative loadings; a control
series remains signed. The fixed input avoids division by a zero or signed
realized control mean. Under `baseline_floor_scope="non_treatment"`, credited
control contributions are clipped differences and need not equal this nominal
target.

Both reference scales come from parameters, never a statistic of a realized
series. Each enabled target prior takes precedence over `beta_additive_range`
or `zy_coeff_range`, respectively; those raw ranges remain validated.
`None` retains the corresponding raw-coefficient prior. Relative outcome noise
still scales with `sqrt(sum((g_cy * beta)**2))`, now using the **derived beta**,
not the reference target. Arithmetic rescaling and a final mantissa/exponent
product preserve representable tiny and subnormal absolute standard deviations,
with exact zero for zero amplitudes and no coefficient or noise floor.
The analytic pullback uses the bounded coefficient direction to avoid rounding
or overflowing an intermediate gradient. Reference
scalars are validated even when unused, but changing an unused scalar introduces
no graph or RNG changes.

Impossible reference configurations raise field-specific errors: the
ideal treatment-response support must be normal and representable (actual rounded
positive treatment responses may be subnormal), nonzero targets must not derive
zero float64 coefficients, and coefficient support must fit the
float32 corpus-storage maximum. Target supports containing zero are checked at
their smallest nonzero float64 draw, not only at their endpoints. There is no
silent clamping of targets or coefficients. With treatment targets enabled, the
multiplier must produce a normal (not subnormal) positive input even at the
generator's minimum anchor `r_k=1e-8`. Treatment coefficient bounds keep a
small rounding margin, widened for Hill by its slope and tail depth. Actual
reference-input products, including those from an oracle's supplied
`saturation_scale`, must remain finite and positive. A drawn input or
coefficient that still leaves this domain through rounding raises the same
named `ValueError` in `sample_scm` and in corpus generation; corpus generation
never resamples it. Opt-in mechanism settings use numerically stable,
mathematically equivalent evaluation of every saturation family (Hill, logistic,
Michaelis–Menten, tanh and root). Defaults keep pymc-marketing's library graphs,
their rounding and their random-stream order. The opt-in MM, tanh and root forms
retain their unit forward curves and use analytic pullbacks that avoid
tiny-denominator squares or indeterminate saturated derivatives. The stable
relative-noise norm above applies whenever an opt-in mechanism setting is active;
default recipes keep `std * sqrt(sum((g_cy * beta)**2))` and its rounding, so
raw-coefficient ranges reaching subnormal noise products need such a setting for
extended-range arithmetic. Descriptions and DOT graphs of opt-in worlds
print saturation shape parameters, `beta` and edge coefficients with four
significant digits, and conditioning intervals exactly.

The normal-input requirement is treatment-only. Covariate reference inputs can
be positive subnormal values when their derived coefficient remains representable;
schema-v5 truth preserves those inputs, targets and nonzero subnormal responses.

Subnormal logistic unit responses retain the complete half-argument through final
float64 rounding. Exact binary midpoint comparisons preserve even half-ULP cases,
so a large admitted coefficient cannot amplify a response erased by intermediate
rounding. Outside this smallest-response boundary, the existing `expm1` law and
rounding are unchanged.

```python exec="1" source="block" result="text"
import numpy as np
from pymc_generator import SCMPrior, make_scm_prior, sample_scm

shapes = SCMPrior().saturation_prior_ranges
shapes["michaelis_menten"]["kappa_mult"] = (0.01, 100.0)
families = ("linear", "hill", "logistic", "michaelis_menten", "tanh", "root")
cfg = make_scm_prior(
    n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=32,
    saturation_prior_ranges=shapes,
    saturation_family_probs={f: float(f == "michaelis_menten") for f in families},
    mm_scale_prior="log_uniform",
    treatment_reference_contribution_range=(0.7, 1.3),
    treatment_reference_multiplier=2.0,
    covariate_reference_contribution_range=(-0.2, 0.4),
    covariate_reference_scale=3.0,
    outcome_std_mode="absolute", rw_outcome_std_sigma=0.01,
)
world = sample_scm(cfg, seed=25)
p = world.params
reference_response = 2.0 / (2.0 + p["mm_kappa_mult"])
np.testing.assert_allclose(
    p["beta"] * reference_response, p["treatment_reference_contribution"],
)
np.testing.assert_allclose(
    p["rho_zy"] * 3.0, p["covariate_reference_contribution"],
)
print("Reference contributions:", np.round(p["treatment_reference_contribution"], 6))
```

Enabled targets and actual reference inputs are recorded as
`world.params["{treatment,covariate}_reference_{contribution,input}"]`, in
`world.equation_parameters`, and in `describe_scm` / bundle descriptions.
The oracle uses the same primitive priors and derives the same coefficients;
in log-uniform mode its free MM variable is `mm_kappa_mult_log`, while
`mm_kappa_mult` is deterministic. With target priors enabled, `beta` / `rho_zy`
are deterministic and the corresponding `*_reference_contribution` variables
are the primitive priors.

Non-default effective mechanism settings add `diagnostics["mechanism_priors"]`
and a declared `parameter_fields` inventory to schema-v5 corpora. Alongside the
configured shapes, MM distribution and reference settings, the corpus retains
float64 realised shapes, `beta`, `rho_zy`, and the exact response anchor.
Enabled reference priors also store their targets and actual inputs; treatment
references retain the response used to derive `beta`. These arrays and metadata
survive `save_corpus` / `load_corpus`; default recipes omit the block.
See [schema-v5 fields](corpus.md#schema-versions-and-migration).
Enabling a new prior can change PyMC stream assignment at the same seed;
stream alignment between different configurations is not promised.

### Carryover normalization

Geometric carryover delegates to
`pymc_marketing.mmm.transformers.geometric_adstock(normalize=True)`.
Normalization divides the finite kernel weights by their sum, so they sum to
one; **it does not rescale data into `[0, 1]`**. The output remains in input
units. Zero-padding at a cold start can reduce a constant input's initial
response; normalization does not manufacture pre-window history.

## Covariate texture

`rw_covariate_mean_range` sets the signed covariate drive's walk mean. It does not
set the latent-unobserved factor's mean: it is anchored to mean zero and unit scale, with
its magnitude represented by the outgoing loadings.

A covariate's own drive is otherwise a smoothed random walk, i.e. the same
function class as the smooth baseline walk `RW_B`. The two are then nearly
collinear over a typical horizon, so the `Z → Y` loading `rho_zy` trades off
against baseline drift and is only weakly identified. These three knobs add the
high-frequency content a smooth baseline cannot mimic:

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(            # the diverse preset already sets these
    n_treatments=4, n_covariates=2, n_latent=1,
    covariate_hf_sigma_range=(0.1, 0.8),
    covariate_pulse_prob_range=(0.0, 0.25),
    covariate_pulse_amp_range=(0.5, 3.0),
)
smooth = make_scm_prior(         # pre-texture numerical behavior, not archive-byte identity
    n_treatments=4, n_covariates=2, n_latent=1,
    covariate_hf_sigma_range=(0.0, 0.0),
    covariate_pulse_prob_range=(0.0, 0.0),
    covariate_pulse_amp_range=(0.0, 0.0),
)
```

For covariate `m`, with `s = rw_z_std[m]`:

$$
F_m = \mathrm{RW}_m + \underbrace{q\,s}_{\texttt{covariate\_hf\_sigma}}\eta_{tm}
    + \underbrace{r\,s}_{\texttt{covariate\_pulse\_amp}}\,(h_{tm} - p),\qquad
h_{tm}\sim\mathrm{Bernoulli}(p).
$$

- `covariate_hf_sigma_range: tuple[float, float] = (0.0, 0.0)` — the iid weekly
  factor `q`, **relative to the covariate's own walk std** (a signed covariate has
  no positive level to anchor on). Bounds must be finite with `0 <= lo <= hi`.
- `covariate_pulse_prob_range: tuple[float, float] = (0.0, 0.0)` — the per-week
  fire probability `p`, with `0 <= lo <= hi <= 0.5`. Its upper bound is the
  on/off switch for the pulse term.
- `covariate_pulse_amp_range: tuple[float, float] = (0.0, 0.0)` — the pulse
  amplitude factor `r`, also relative to the walk std. It is an *amplitude*,
  not a standard deviation: the term's variance is
  `(r*s)**2 * p * (1 - p)`. Enabling `covariate_pulse_prob_range` with a
  zero-only amplitude is a validation error.

The pulse is **centred**: subtracting `p` makes both added terms mean-zero
(`E[F_m - RW_m] = 0`), so a covariate's expected level stays `rw_z_mean` — and
that claim is EXACT, because a covariate's equation applies no activation, as
long as neither it nor any upstream `Z → Z` ancestor carries a
[trajectory gate or level component](#composable-input-trajectories) —
which is why the parameter-only saturation reference levels are unchanged. (Treatment
pulses are
deliberately uncentred — a treatment is positive
and its realized level may rise above the walk anchor; a treatment's equation
also applies `softplus`, so its own expected level sits strictly above its
parameter-only anchor regardless of pulses.) Centring holds in
expectation conditional on the drawn parameters, not as the temporal mean of
every finite path.

All three default to `(0.0, 0.0)` on a raw `SCMPrior`: no term enters the graph
and the parameters degenerate to constants. Leaving these controls disabled
preserves the same-environment numerical/model arrays, not cross-schema
metadata or archive-byte identity.

## Composable input trajectories

Every treatment and every covariate can carry, **independently per input**, any
subset of eight trajectory components on top of its walk. Their canonical order —
also the `component` axis of the
[optional corpus arrays](../guide/corpus.md#composable-input-trajectories) — is
`pymc_generator.slots.TRAJECTORY_COMPONENTS`:

| Component | Kind | On a covariate (signed) | On a treatment (non-negative) |
| --- | --- | --- | --- |
| `hf` | texture | iid weekly noise ([above](#covariate-texture)) | iid execution noise inside the `softplus` |
| `pulse` | texture | centred Bernoulli pulses | Bernoulli campaign pulses inside the `softplus` |
| `onset` | gate | off before a launch week | off before a launch week |
| `offset` | gate | off from a stop week on | off from a stop week on |
| `flighting` | gate | on for `W` of every `P` weeks | on for `W` of every `P` weeks |
| `level_jump` | level | `+ J` from each jump week on | `× f` from each jump week on |
| `seasonal` | level | `+ A·sin(2πt/P + φ)` | `× exp(A·sin(2πt/P + φ))` |
| `trend` | level | `+ B·max(t, 0)/(T − 1)` | `× exp(B·max(t, 0)/(T − 1))` |

The [input-trajectories guide](../guide/trajectories.md) draws every archetype
from a live corpus; this section is the field reference.

```python
from pymc_generator import make_scm_prior

mixed = make_scm_prior(          # every component, at moderate prevalence
    n_treatments=4, n_covariates=2, n_latent=1, trajectories="composable",
)
flighted = make_scm_prior(       # one archetype on every input
    n_treatments=4, n_covariates=2, n_latent=1, trajectories="periodic_on_off",
)
custom = make_scm_prior(         # or any field directly
    n_treatments=4, n_covariates=2, n_latent=1,
    treatment_onset_inclusion_prob=0.3,
    covariate_seasonal_inclusion_prob=0.5,
)
```

Outcome-side seasonality is not a component: the baseline process is unchanged
(a smoothed walk with no explicit seasonal term), and no Fourier series are
added as covariates.

### Composition

Reported weeks are $t = 0, \dots, T-1$ ($T$ = `n_time_steps`) and burn-in weeks
$t = -\texttt{carryover\_burn\_in}, \dots, -1$. The gate forms an input carries
multiply into one activity $\chi_i(t) \in \{0, 1\}$ ($\chi_i \equiv 1$ without a
gate); its level forms add into one level shift — in covariate units for a
covariate, in log-level for a treatment. With $F_m$ covariate $m$'s own drive
(its walk plus the texture it carries) and $L_m$ its level shift, and with
$\mathrm{pre}_k$ treatment $k$'s unchanged pre-activation, $S_k$ and $R_k$ its
seasonal and trend terms and $f_{ek}$ its jump factors:

$$
\begin{aligned}
Z_m(t) &= \chi_m(t)\Big(\textstyle\sum_j u_{jm}D_j + \sum_{m'<m}\gamma_{m'm}Z_{m'} + F_m(t) + L_m(t)\Big),\\[2pt]
C_k(t) &= \mathrm{clamp}_k\Big(\chi_k(t)\;\mathrm{softplus}\big(\mathrm{pre}_k(t)\big)\;e^{S_k(t) + R_k(t)}\;\textstyle\prod_e f_{ek}^{\,\mathbf{1}[t \ge \tau_{ek}]}\Big),
\end{aligned}
$$

where $\mathrm{clamp}_k$ is the optional held-level [shock](#treatment-shocks). So:

- **Covariates stay signed and additive.** The gate wraps the assembled
  covariate, parents included, so an off-week is exactly `0.0` for `Z` and for
  every `Z → C`, `Z → Z` and `Z → Y` term it feeds. Under the default intercept
  floor scope the `Z → Y` term stays literally `g_zy[m] * rho_zy[m] * Z[:, m]`.
- **Treatments stay non-negative.** Level components compose in log-level — a
  sum on the log scale, a product on the series — and the gate multiplies the
  whole activated input, so `C ≥ 0` always and `C == 0.0` exactly on off-weeks
  outside shocks. A jump factor multiplies the series directly, never through
  `exp(log f)`, so a constant factor of `2` doubles it exactly from its jump week
  on.
- **Every decomposition variant shares the schedule.** The same gate and
  multiplier apply to the observed treatment and to every intervention variant
  the decomposition compares (`treatments_base` and the telescoping variants
  behind `indirect_effects_by_source`), so its identities stay exact. Only the
  schedule-free natural recursion read by the
  [realism filter](../guide/trajectories.md#realism-with-schedules) omits them.
- **Shocks override the schedule.** The clamp applies after the envelope: an
  event holds `treatment_shock_level_multiplier * softplus(rw_c_mean)` whatever
  the gate and multiplier say, off-weeks included.

### Inclusion probabilities

`<input>_<component>_inclusion_prob` is the probability that one input carries
the component. Each cell draws it once per input — every (input, component)
pair an independent Bernoulli — as concrete structure, like the carryover
family, so all of that cell's draws share it. Every probability must be a finite
number in `[0, 1]` (`bool` is rejected).

| Component | Treatment field | Covariate field | `SCMPrior` default | `"composable"` (treatment / covariate) |
| --- | --- | --- | --- | --- |
| `hf` | `treatment_hf_inclusion_prob` | `covariate_hf_inclusion_prob` | `1.0` | `0.9` / `0.9` |
| `pulse` | `treatment_pulse_inclusion_prob` | `covariate_pulse_inclusion_prob` | `1.0` | `0.7` / `0.6` |
| `onset` | `treatment_onset_inclusion_prob` | `covariate_onset_inclusion_prob` | `0.0` | `0.15` / `0.1` |
| `offset` | `treatment_offset_inclusion_prob` | `covariate_offset_inclusion_prob` | `0.0` | `0.1` / `0.1` |
| `flighting` | `treatment_flighting_inclusion_prob` | `covariate_flighting_inclusion_prob` | `0.0` | `0.2` / `0.15` |
| `level_jump` | `treatment_level_jump_inclusion_prob` | `covariate_level_jump_inclusion_prob` | `0.0` | `0.2` / `0.3` |
| `seasonal` | `treatment_seasonal_inclusion_prob` | `covariate_seasonal_inclusion_prob` | `0.0` | `0.3` / `0.35` |
| `trend` | `treatment_trend_inclusion_prob` | `covariate_trend_inclusion_prob` | `0.0` | `0.3` / `0.3` |

- `1.0` puts the component on every input and `0.0` on none; neither samples
  a random inclusion flag. Components combine freely: one input can launch late, run in
  flights and trend at once.
- A component an input does not carry adds exactly nothing. Ordinary active-size
  cell construction omits gate/level priors when no input carries the component;
  reusable templates keep cfg-admitted priors wired and drawn even for all-off
  cells, with effects neutralized by runtime flags. Legacy `hf` / `pulse`
  magnitude and noise variables retain their existing construction/draw
  contract; their forward effects reach only carrying inputs. Explicit
  provenance/audit requests can still sample unused variables.
- `hf` and `pulse` keep their texture ranges as on/off switches: their
  **effective** probability is `0` whenever `*_hf_sigma_range[1] == 0` /
  `*_pulse_prob_range[1] == 0` (as on a raw `SCMPrior`), whatever the knob says.
  `SCMPrior.trajectory_inclusion_probs()` returns the effective table,
  `{"treatment": {component: p}, "covariate": {component: p}}`, which corpora
  echo.
- `SCMPrior.trajectory_components_enabled` is `True` when any gate or level
  component can be included for either input type; it adds the
  [model outputs](#model-outputs) and the natural realism path.
  `SCMPrior.trajectory_metadata_enabled` is `True` when that holds or any `hf` /
  `pulse` knob differs from `1.0`; it adds the optional corpus block.
- A treatment that can carry nothing — effective `hf`, `pulse` and every
  schedule probability at `0` — is the deprecated flat texture and emits a
  `FutureWarning`.

### Texture components

`hf` and `pulse` are the existing texture terms: the inclusion probability
decides which inputs carry them and their ranges set the magnitudes. Inside a
treatment's `softplus`, the own drive is

$$
E_k = \mathrm{RW}_k + \sigma_k\,\varepsilon_{tk} + a_k\,b_{tk},\qquad
b_{tk}\sim\mathrm{Bernoulli}(p_k),
$$

with each term present only when the treatment carries it, and $\sigma_k$ and
$a_k$ drawn relative to the treatment's level `softplus(rw_c_mean)`. A
covariate's terms are relative to its own walk std, and its pulse is centred
([above](#covariate-texture)). Each input's $\sigma_k$, $p_k$ and $a_k$ are
drawn uniformly from the ranges below.

| Field | `SCMPrior` default | `make_scm_prior` | Bounds |
| --- | --- | --- | --- |
| `treatment_hf_sigma_range` | `(0.0, 0.0)` | `(0.08, 0.6)` | `0 <= lo <= hi` |
| `treatment_pulse_prob_range` | `(0.0, 0.0)` | `(0.0, 0.25)`; `always_on_spikes`: `(0.05, 0.25)` | `0 <= lo <= hi <= 0.5` |
| `treatment_pulse_amp_range` | `(0.5, 1.5)` | `(0.4, 2.5)` | `0 <= lo <= hi`; `hi > 0` while pulses are on |
| `covariate_hf_sigma_range` | `(0.0, 0.0)` | `(0.1, 0.8)` | `0 <= lo <= hi` |
| `covariate_pulse_prob_range` | `(0.0, 0.0)` | `(0.0, 0.25)`; `always_on_spikes`: `(0.05, 0.25)` | `0 <= lo <= hi <= 0.5` |
| `covariate_pulse_amp_range` | `(0.0, 0.0)` | `(0.5, 3.0)` | `0 <= lo <= hi`; `hi > 0` while pulses are on |

`make_scm_prior` applies these texture ranges for every `trajectories=` value.
A `*_hf_sigma_range` or `*_pulse_prob_range` upper bound of `0` switches that
term off for every input, whatever its inclusion probability.

### Gate forms

$$
\begin{aligned}
\text{onset}:&\quad \mathbf{1}[t \ge s_i], & s_i &= \lfloor u_i T \rfloor,\quad u_i \sim \mathrm{U}(\texttt{onset\_frac\_range}),\\
\text{offset}:&\quad \mathbf{1}[t < e_i], & e_i &= \lfloor u_i T \rfloor,\quad u_i \sim \mathrm{U}(\texttt{offset\_frac\_range}),\\
\text{flighting}:&\quad \mathbf{1}\big[(t + \phi_i) \bmod P_i < W_i\big], & W_i &= \mathrm{clip}\big(\lfloor d_i P_i + 0.5 \rfloor,\ 1,\ P_i - 1\big),\quad \phi_i = \min\big(\lfloor v_i P_i \rfloor,\ P_i - 1\big),
\end{aligned}
$$

with the period $P_i$ an integer uniform on `flighting_period_weeks_range`, the
duty $d_i \sim \mathrm{U}(\texttt{flighting\_duty\_range})$ and
$v_i \sim \mathrm{U}(0, 1)$. Each pair of fields below shares one default.

- `treatment_onset_frac_range`, `covariate_onset_frac_range: tuple[float, float] = (0.05, 0.25)`
  — the launch week as a fraction of `n_time_steps`. A carried onset is off for
  **every** earlier week, burn-in included: a cold launch, so carryover has
  nothing to carry and the treatment's direct contribution is exactly `0`
  before it. Bounds `0 <= lo <= hi < 1`; while an onset can be included,
  `floor(lo * n_time_steps) >= 1`, so every onset delays its input by at least
  one reported week.
- `treatment_offset_frac_range`, `covariate_offset_frac_range: tuple[float, float] = (0.6, 0.9)`
  — the stop week as a fraction of `n_time_steps`: off from it on, on through
  burn-in. Carryover from the on-weeks still decays into the off-weeks; the
  direct response reaches exactly zero once the full kernel span lies past the
  stop. Bounds `0 < lo <= hi <= 1`; while an offset can be included, `hi < 1`,
  so every offset stops its input before the window ends.
- `treatment_flighting_period_weeks_range`, `covariate_flighting_period_weeks_range: tuple[int, int] = (4, 13)`
  — the period `P`, uniform over the integers in `[lo, hi]` (fixed when
  `lo == hi`). Integer bounds `2 <= lo <= hi`; while flighting can be included,
  `hi <= n_time_steps`, so any `n_time_steps` reported weeks hold a full period
  and every included flighting input switches off inside the window.
- `treatment_flighting_duty_range`, `covariate_flighting_duty_range: tuple[float, float] = (0.3, 0.8)`
  — the duty `d`: the on-run is `W = clip(floor(d * P + 0.5), 1, P - 1)` weeks,
  rounding half up, so every flighting input switches off at least once per
  period; the phase is uniform. The cycle runs through burn-in. Bounds
  `0 < lo <= hi <= 1`.

### Level forms

$$
\begin{aligned}
\text{level\_jump}:&\quad \textstyle\sum_{e<K} J_{ei}\,\mathbf{1}[t \ge \tau_{ei}], & \tau_{ei} &\in \big[\,1 + \lfloor e(T-1)/K \rfloor,\ 1 + \lfloor (e+1)(T-1)/K \rfloor\big),\\
\text{seasonal}:&\quad A_i \sin(2\pi t / P_i + \varphi_i), & \varphi_i &\sim \mathrm{U}(0, 2\pi),\\
\text{trend}:&\quad B_i \max(t, 0) / (T - 1), & &
\end{aligned}
$$

with each $\tau_{ei}$ uniform within its slot. A covariate adds its level shift
in covariate units, with $J$ the step size. A treatment's level shift is a
log-level with $J = \log f$, applied as the multiplier
$e^{S + R}\prod_e f_e^{\mathbf{1}[t \ge \tau_e]}$.

- `treatment_level_jump_count`, `covariate_level_jump_count: int = 1` — `K`, the
  exact number of steps per carrying input. Step `e` falls uniformly inside the
  `e`-th of `K` disjoint slots of weeks `[1, n_time_steps)`, so steps are
  distinct and ordered and never land on week 0 or in burn-in. An integer
  `>= 1`; while jumps can be included, `K <= n_time_steps - 1`.
- `treatment_level_jump_factor_range: tuple[float, float] = (0.5, 2.0)` — the
  factor `f`, log-uniform on `[lo, hi]`; a degenerate range is used as given, so
  `(2.0, 2.0)` doubles exactly. Bounds `0 < lo <= hi`; while jumps can be
  included, not `(1.0, 1.0)`.
- `covariate_level_jump_size_range: tuple[float, float] = (-1.0, 1.0)` — the
  signed step `J`, uniform, in covariate units. Finite `lo <= hi`; while jumps
  can be included, not `(0.0, 0.0)`.
- `treatment_seasonal_amplitude_range: tuple[float, float] = (0.1, 0.5)` — the
  amplitude `A`, uniform, in **log-level** (`0.5` scales the level between
  `×0.61` and `×1.65`). Bounds `0 <= lo <= hi`; while seasonality can be
  included, `hi > 0`.
- `covariate_seasonal_amplitude_range: tuple[float, float] = (0.2, 1.0)` — the
  same in covariate units.
- `treatment_seasonal_period_weeks_range`, `covariate_seasonal_period_weeks_range: tuple[float, float] = (52.0, 52.0)`
  — the period `P` in weeks, continuous uniform (fixed when `lo == hi`); the
  phase is uniform on `[0, 2π)`, and the sinusoid runs through burn-in. Bounds
  `2 < lo <= hi`: a period of two weeks or less aliases to an alternation at
  weekly sampling.
- `treatment_trend_log_change_range: tuple[float, float] = (-1.0, 1.0)` — `B`,
  the total **log-level** change across the reported window, uniform: the trend
  is `0` through burn-in and at week 0 and reaches `B` at the last week. Finite
  `lo <= hi`; while trends can be included, not `(0.0, 0.0)`.
- `covariate_trend_change_range: tuple[float, float] = (-1.0, 1.0)` — the same in
  covariate units.

Float range bounds must be finite and capped in magnitude by the existing
float32 corpus-storage maximum. Schema-v5 realised continuous trajectory truth
is retained in float64, including valid sub-float32 values; existing feature
dtypes are unchanged. The integer flighting period range is bounded by
`n_time_steps` instead, once flighting can be included.

### Rules across components

Every bound above is checked for any probability. The rules marked "while … can
be included", and the three below, apply only once the component's inclusion
probability is `> 0`, so a disabled component's priors never constrain a
config:

- **Two on-weeks of context.** Every task keeps its first
  `S = min(n_time_steps - n_query, n_time_steps // 2)` weeks as support, under
  either split. With `start_hi = floor(onset_hi * T)` if an onset can be included
  (else `0`) and `stop_lo = floor(offset_lo * T)` if an offset can be (else `T`),
  the window `n = min(stop_lo, S) - start_hi` must hold two on-weeks: `n >= 2`
  without flighting, and with flighting
  `floor(n / P) * W + max(0, n % P - (P - W)) >= 2` — the worst phase — for every
  admitted period `P` at `W = W(duty_lo, P)`. So every gated input has at least
  two on-weeks inside its task's support prefix, for every draw and whatever
  combination of gates it ends up carrying.
- **Bounded treatment swing.** `A_hi + max|B| + K * max|log f|` — each term only
  when its component can be included — must be
  `<= TRAJECTORY_MAX_LOG_SHIFT = 3.0` (`pymc_generator.slots`), so a treatment's
  level multiplier stays inside `[e^-3, e^3]`, about `×20` either way. That keeps
  scheduled treatments at a realism-filter scale and their decomposition
  representable in float32. The bound is inclusive, with a relative tolerance of
  `1e-12` for floating-point rounding, so a swing that is exactly `3.0` on paper
  is accepted: 30 jumps with factors in `[e^-0.1, e^0.1]` compute to
  `3.000000000000002` and pass, while 31 are rejected.
- **Float32 reach.** The summed level reach must fit float32 corpus storage
  (`np.finfo(np.float32).max`, about `3.4e38`):
  `rw_positive_mean_range[1] * exp(swing)` for treatments, with the swing above,
  and `max|rw_covariate_mean_range| + A_hi + max|B| + K * max|J|` for
  covariates, each term only when its component can be included. This static
  rule is coarse — it counts neither texture nor parent terms — so with schedule
  components enabled the corpus also rejects any draw whose arrays would not
  survive the float32 cast, or whose `outcome / outcome_scale` would overflow;
  a prior that only produces such draws exhausts its cell with the usual
  realism-filter `RuntimeError` instead of storing `inf`.

### Model outputs

With `trajectory_components_enabled`, a cell model from
[`build_world_model`](world-model.md#pymc_generator.world_model.build_world_model)
also outputs, over the reported window:

- `treatment_activity`, `covariate_activity` — `int8` `(n_time_steps, n)`, the
  gate schedule (`1` = on; all ones for an input without a gate);
- `treatment_log_level_shift` — `float64`, `S + R + Σ_e log f_e · 1[t ≥ τ_e]`,
  exactly `0.0` without a level component;
- `covariate_level_shift` — `float64`, the covariate's level shift `L_m`;
- `treatments_natural`, `outcome_natural` — the schedule-free realism
  references.

Gate/level component parameters are model variables such as
`treatment_onset_frac` or `covariate_flighting_period`. Ordinary active-size
cell models wire their priors only where some input carries the component;
degenerate ranges are constants, not random variables. Reusable templates
keep cfg-admitted gate/level priors wired even when the current cell's flags
are all off, so flags can change without recompilation. Runtime flags
neutralize the forward effects, not those prior draws.

### Presets

`make_scm_prior(trajectories=...)` takes one of
`pymc_generator.presets.TRAJECTORY_PRESETS`; `**overrides` still win over it.

- `"texture"` (default) — no override: every input carries `hf` and `pulse` with
  the diverse texture ranges and nothing else, so worlds are unchanged.
- `"composable"` — every component on both input types at the probabilities in
  the [table above](#inclusion-probabilities), with wider input level and
  variation priors than the texture preset: `rw_positive_mean_range=(0.1, 8.0)`
  (texture `(0.3, 4.0)`), `rw_treatment_std_range=(0.1, 1.2)` (`(0.15, 0.8)`),
  `rw_covariate_mean_range=(-3.0, 3.0)` (`(-1.0, 1.0)`) and `rw_std_sigma=1.5`
  (`1.0`), with `rw_baseline_std_sigma=1.0` pinned so absolute-mode baselines
  keep the legacy scale. Its gate windows, flighting periods and seasonal period
  follow the horizon (below); every other component prior keeps its `SCMPrior`
  default.
- every name in `pymc_generator.presets.TRAJECTORY_ARCHETYPES` — each input
  follows that archetype; see the
  [archetype table](../guide/trajectories.md#archetypes).

Horizon-derived ranges use the effective `T = n_time_steps`,
`n_query = round(query_frac * T)` and `S = min(T - n_query, T // 2)`, and write
each week as the fraction `(week + 0.5) / T`, so `floor(u * T)` lands on exactly
the intended weeks. For `"composable"`:

| Range | Rule | At `T = 104` (`S = 52`) |
| --- | --- | --- |
| launch weeks (`*_onset_frac_range`) | `[lo, max(lo, S // 4)]`, `lo = max(1, round(0.05 T))` | 5–13: `(5.5/104, 13.5/104)` |
| stop weeks (`*_offset_frac_range`) | `[lo, max(lo, min(T - 1, round(0.9 T)))]`, `lo = max(S, round(0.6 T))` | 62–94: `(62.5/104, 94.5/104)` |
| `*_flighting_period_weeks_range` | the widest `(min(4, hi), hi)` with `hi <= min(13, n // 2)` that keeps two on-weeks in `n = S - launch_hi` weeks at duty `0.3` | `(4, 13)` |
| `*_flighting_duty_range` | fixed | `(0.3, 0.8)` |
| `*_seasonal_period_weeks_range` | `max(4, min(52, T // 2))` for both bounds | `(52.0, 52.0)` |

At a horizon where a preset cannot keep two on-weeks of context,
`make_scm_prior` raises `ValueError` **before** `**overrides` apply: overrides
cannot rescue a preset that has no gate windows at that horizon, so set the
trajectory knobs directly instead (without `trajectories=`). The message names
the preset and the next longer `n_time_steps` at which the complete config —
the base fields, the preset at that horizon and your overrides — validates, so
every other rule counts too: `trajectories="composable"` at `n_time_steps=8`
names `14` with the default `l_max=8` and `102` with `l_max=52`. The search is
bounded — at most 100,000 longer horizons and 1,000 full validations, and the
message names no horizon if it finds no fit within those limits — and any error
the config raises at the requested horizon without the preset (base fields and
overrides) is attached as the exception's `__cause__`, so an invalid or
misspelled override is not hidden (at `n_time_steps=8` with `l_max=8` the cause
is the carryover burn-in / query overlap rule). A named horizon is the next fit,
not a promise that every longer horizon fits; feasibility is not
monotone in the horizon — `"composable"` at `query_frac=0.9` fits at
`n_time_steps=65` to `69`, fails at `70` to `75` (naming `76`) and fits again
at `76`.

### Randomness and seeds

- **Defaults draw nothing new.** With every schedule probability at `0.0` and
  `hf` / `pulse` at `1.0`, no random variable, numpy draw or corpus key is
  added by these controls: same-seed numerical/model arrays, `sample_scm` worlds
  and template draws are unchanged by leaving them at their defaults.
  This does not promise cross-version metadata/archive identity: schema v5
  changes the version stamp and saved corpus bytes, and `recipe.json` lists
  the new `SCMPrior` fields at their defaults.
- **Fractional flags consume no main-RNG state.** Probabilities of exactly `0`
  or `1` draw nothing. Fractional ones draw `stream.random(n) < p` from child
  streams spawned, once per cell, off the corpus's numpy generator — one fixed
  stream per (input type, component) — and spawning never advances the
  generator itself. So flag draws never move the legacy structure, draw seeds or
  support masks drawn from that generator, each component's flags do not depend
  on any other component's probability, and raising one probability at a fixed
  seed only adds inputs to the set that carries it.
- **Ordinary-cell wiring changes other draws.** When changing only inclusion
  probabilities, an ordinary active-size cell whose realised wiring matches
  the default — no schedule component on any input, and `hf` / `pulse` on every
  input whose range is live — retains the default legacy draws. A `sample_scm`
  world whose inputs carry no schedule component likewise keeps the default
  world's draws: its random-stream reference leaves out the schedule audit
  outputs (corpus generation gets the same effect by requesting them ahead of the
  legacy names). Once its wiring
  differs, the random variables its outputs reach, and the order a compiled
  draw visits them (which assigns random streams), can change: at the same
  seed, legacy parameters, innovations and other components generally change
  too. Stream stability under enabling is tracked in
  [#28](https://github.com/pymc-labs/pymc-generator/issues/28).
- **Reusable templates keep admitted priors wired.** Cfg-admitted gate/level
  priors remain reachable and drawn even for an all-off cell. Runtime flags
  neutralize their effects without an ordinary-cell RNG-pruning or same-seed
  draw-identity promise. Legacy `hf` / `pulse` variables retain their existing
  random-draw contract; switching their flags off does not promise to eliminate
  those prior draws.
- **Corpus alignment needs matching acceptance.** Cell `c` of two configs lines
  up only while every earlier cell consumed the same main-RNG draws — one draw
  seed per top-up round and one support-mask draw per candidate that passes the
  realism filter — that is, while their acceptance counts match.

### Limits

- **End-to-end support.** Ordinary generation, `sample_scm`, descriptions,
  bundles, `SCM.replay()` and the fixed-input oracle support every component
  and fractional per-input inclusion. The reusable template takes flags as
  per-cell data inputs; it still excludes unrelated treatment shocks,
  prior-conditioning intervals and non-fixed confounding-strength ranges.
  Equal seeds do not align padded template and active-size ordinary RV draws:
  compare their forward values at the same concrete inputs.
- **κ anchors exclude the components.** `saturation_scale` stays parameter-only:
  a doubled treatment runs at twice its κ-relative level, and gates, seasonality
  and trends move a treatment along its response curve rather than re-centring
  it.
- **The covariate exact-level claim needs component-free covariates.**
  `E[Z_m] = rw_z_mean + upstream Z → Z terms` holds only while neither the
  covariate nor any upstream `Z → Z` ancestor carries a gate or level component.

## Intercept floor

`baseline_floor: float | None = None` censors the intercept walk:

$$B_t = \max(\mathrm{RW}_{B,t},\ \texttt{baseline\_floor}).$$

A **censored** walk, not a softplus, so `B` can sit exactly *at* the floor —
"zero or above, never below". `None` keeps the signed walk.

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=4, n_covariates=2, n_latent=1,
    baseline_floor=0.0,  # intrinsic intercept B >= 0, not the aggregate baseline
)
```

Why this is safe: the intercept carries **no parents**. Latent-unobserved factors and
covariates enter `Y` directly, so the floor clips one additive term and
`covariate_contribution[:, m]` stays exactly `g_zy[m] * rho_zy[m] * Z[:, m]`.
Flooring a sum that contained the parents would break that identity.

The floor is a pure clip: it adds no random variable and consumes no RNG, so a
floor that never binds reproduces the unfloored corpus byte for byte (pinned by
`test_the_intercept_floor_is_a_pure_clip_and_consumes_no_rng`). Two consequences
to know about:

- **Outcome is not censored.** A clamp on `Y` would censor the *observation*,
  putting every world outside the additive-Gaussian class an MMM likelihood —
  including this package's own oracle — can represent. Non-negative outcome stays
  enforced by the acceptance filter. Measured on the shipped prior, outcome sits
  28–83 observation-noise σ above zero and a negative intercept week occurs in
  ~0.08% of weeks, so the floor is a guarantee rather than a frequent
  intervention.
- **The oracle's `latent="marginal"` mode raises** for a floored config, because
  `max(RW_B, floor)` is not Gaussian and the analytic covariance would be wrong.
  Use `latent="sampled"`, which applies the identical clip.

### What the floor clips: `baseline_floor_scope`

`baseline_floor_scope: Literal["intercept", "non_treatment"] = "intercept"`.

A floored *intercept* alone does not make outcome non-negative: a large negative
$\rho_m Z_m$ can still drag the non-treatment total under (measured on a stress
fixture: 112 negative weeks with `scope="intercept"`). `scope="non_treatment"`
clips the **running total** as each parent joins, in the locked order
intercept → latent-unobserved factors ($j$ ascending) → covariates ($m$ ascending):

$$A^{(0)} = B,\qquad A^{(i)} = \max\!\big(A^{(i-1)} + \text{node}_i,\ \texttt{floor}\big),\qquad A = A^{(\text{last})}.$$

Each per-node column is the telescoping difference that node caused,
$A^{(i)} - A^{(i-1)}$ — the same construction `indirect_effects_by_source` uses
for treatments. So:

- the non-treatment total is `≥ floor` **by construction**: a negative covariate
  effect is credited only down to the floor and the excess is absorbed instead
  of pushing outcome negative (same fixture: 0 negative weeks, 111 weeks sitting
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

#### Why outcome still is not strictly guaranteed

With `scope="non_treatment"` every term of the outcome **mean** is non-negative
($A \ge 0$, treatments $\ge 0$). What remains is the additive observation noise
$\mathrm{RW}_Y$, which is symmetric and unbounded, so
$P(Y<0)>0$ is unavoidable for *any* additive-Gaussian outcome — that is a
property of the likelihood, not of this generator. Strictness would require
censoring the observation or bounding the noise, both of which change the
function class an MMM can represent. Measured margin under the absorbing scope:
the outcome mean sits **88 σ** of observation noise above zero at its worst week
over 12 worlds, so the residual probability is ~$10^{-1700}$, and the
acceptance filter remains the exact backstop.

### Decomposition columns

`baseline_intrinsic` is the intercept **alone** (`B`), and the iid observation
noise is its own `outcome_noise` column, so the persisted identity is

```text
baseline_intrinsic + outcome_noise + Σ covariate_contribution
  + Σ latent_unobserved_contribution + Σ contributions
  + Σ indirect_effects_by_source == sales
```

## Baseline walk scale

In `outcome_std_mode="absolute"`, `rw_baseline_std_sigma: float | None = None`
controls the `RW_B` HalfNormal scale. `None` follows `rw_std_sigma`; an explicit
positive finite value overrides that baseline scale. Under the default
`"relative"` mode, `rw_baseline_std_range` instead scales the walk relative to
the parameter-only treatment anchor. Neither option changes the latent-unobserved factor or outcome noise.

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=4, n_covariates=2, n_latent=1,
    outcome_std_mode="absolute", rw_std_sigma=1.2,  # RW_B also uses 1.2
)
cfg = make_scm_prior(
    n_treatments=4, n_covariates=2, n_latent=1,
    outcome_std_mode="absolute", rw_std_sigma=1.2, rw_baseline_std_sigma=0.35,
)
```

## Shared baseline/treatment innovations

`confounding_strength_range: tuple[float, float] | None = None` enables an
optional per-world scalar `rho`. Bounds must be finite and satisfy
`0 <= lo <= hi <= 0.95`. For each treatment/week, the generator replaces its
standard innovation with

$$
\epsilon_c' = \sqrt{1 - \rho^2}\,\epsilon_c + \rho\,\epsilon_b.
$$

This preserves each treatment innovation's unit marginal variance while inducing
shared correlation with the baseline innovation. `None` retains the legacy
independent path (and reports `rho = 0`); a degenerate range fixes the value.
The drawn scalar is persisted as `confounding_strength` with shape `(n_tasks,)` and
dtype `float32` (and is available in single-world data/parameters).

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=4, n_covariates=2, n_latent=1,
    confounding_strength_range=(0.2, 0.5),
)
```

The posterior oracle is a plug-in model for observed treatments and covariates. It
does **not** model their likelihood conditional on latent-unobserved factors or
`epsilon_b`; consequently, it does not use treatment information about this
rho-induced baseline correlation to infer the latent-unobserved factor. It is a structure-known
plug-in reference, not exact joint conditioning or a universal recovery bound.

## Treatment shocks

The optional held-level intervention API is:

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=4, n_covariates=2, n_latent=1,
    n_treatment_shocks=2,
    treatment_shock_length_range=(2, 4),
    treatment_shock_level_range=(0.0, 0.5),
)
```

- `n_treatment_shocks: int = 0` is the exact number `n_shocks` of shocks in
  **each** world. It must be an integer in `[0, n_time_steps]` and must fit at
  maximum length:
  `n_shocks * treatment_shock_length_range[1] <= n_time_steps`.
- `treatment_shock_length_range: tuple[int, int] = (2, 2)` has integral bounds
  `1 <= lo <= hi <= n_time_steps`. When `n_shocks > 0`, `lo` must be at least 2
  so every intervention creates a treatment-visible held-level plateau. One-week
  ranges remain valid only while shocks are disabled.
- `treatment_shock_level_range: tuple[float, float] = (0.0, 0.0)` has finite
  bounds `0 <= lo <= hi`.

Schedule event `s` is placed in deterministic stratified reported-window slot
`[floor(s*n_time_steps/n_shocks), floor((s+1)*n_time_steps/n_shocks))`; its
feasible start and sampled length stay inside that slot. Thus events are
globally non-overlapping, including when a treatment is selected more than once.
Treatments are sampled uniformly **with
replacement** from direct (`g_cy`) treatments. The held level is sampled ex ante
as `treatment_shock_level_multiplier * softplus(rw_c_mean)` for the selected
treatment. Starts are reported-window indices (not burn-in-offset full-horizon
indices).

During an event, observed treatment is clamped to that level. Clamping is the whole
intervention: the clamped path then feeds the ordinary normalized causal
carryover kernel, so carryover from pre-event treatment decays into the window rather
than being discarded, and a zero held level reaches an exactly zero direct
response only once the full kernel span lies inside the window. Outside an
event the natural treatment process resumes, and downstream treatment recursion still
observes the clamped parent. This keeps every generated response inside the
function class a standard MMM carryover transform can represent from the same observed
treatment. It is a known held-level intervention design, **not** a conventional
treatment-only lift test.

## Active-count coverage

Every corpus cell fixes how many treatment and covariate slots are live. Two
fields decide how the cells of a corpus spread over the grid of
(`n_treatments_active`, `n_covariates_active`) combinations spanned by the
**effective** active ranges — each upper bound clamped to its padded size, as
`SCMPrior.active_count_grid` lists them:

| Field | Default | Meaning |
| --- | --- | --- |
| `active_count_allocation` | `"independent"` | `"independent"` draws both counts per cell, uniformly and independently, so a finite corpus covers the grid only in expectation; `"stratified"` allocates the cells over the grid by weight |
| `active_count_weights` | `None` | Weight matrix over the effective grid: row `i` holds `n_treatments_active = n_treatments_active_range[0] + i`, column `j` holds `n_covariates_active = n_covariates_active_range[0] + j`; `None` is uniform |

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=3, n_covariates=3, n_latent=1,
    n_treatments_active_range=(2, 5),   # clamped to (2, 3): rows 2 and 3
    n_covariates_active_range=(1, 3),   # columns 1, 2 and 3
    active_count_allocation="stratified",
    active_count_weights=[[1, 1, 0],    # 2 treatments: never with 3 covariates
                          [1, 2, 4]],   # 3 treatments: favour more covariates
)
```

Weights must form a rectangular matrix of exactly the grid's shape — nested
lists or tuples of `int` / `float` (numpy scalars included); arrays, booleans
and strings are rejected (pass `weights.tolist()` for an array) — with finite,
non-negative entries and at least one positive. They are stored as given; only
their ratios matter. The `ValueError` names the expected shape and axes: `2x3`
here, not the `4x3` the raw `(2, 5)` range suggests. Both fields are validated
in either mode; the weights are used only when stratified. `n_latent_active` is
not part of the grid: it is drawn per cell in both modes.
`write_scenario_bundles` stores priors in `recipe.json` as JSON, so priors it
writes need their weights as plain Python numbers.

### The stratified rule

Write $N$ for `n_cells` and $S$ for the combinations with positive weight. Each
combination gets an exact rational **target** $\tau_{tc}$; the targets sum to
$N$.

- Weight zero: $\tau_{tc} = 0$, so the combination never appears.
- **Coverage floor** ($N \ge |S|$): every positive-weight combination gets at
  least one cell. Each combination whose proportional share of the cells is
  below one is fixed at $\tau_{tc} = 1$, the remaining cells are shared out over
  the remaining combinations, and this repeats until no share is below one.
- **Cap** ($N < |S|$): no combination can hold two cells. Each share of one or
  more is fixed at $\tau_{tc} = 1$, and the remaining cells are shared out again
  until no share reaches one.

A combination that is not fixed gets its share,
$\tau_{tc} = N_F\, w_{tc} / W_F$, where $N_F$ is the number of cells not fixed
and $W_F$ the total weight of the combinations not fixed; with nothing fixed
this is the plain quota $N w_{tc} / \sum w$. Each count is then
$\lfloor \tau_{tc} \rfloor$ or $\lceil \tau_{tc} \rceil$: the fractional parts
are rounded by systematic sampling in a random order, so the counts sum to $N$,
every count's expectation is its target, and a whole-number target is met
exactly. The allocated combinations are finally shuffled over the cells, so
`cell_id` carries no information about the counts. Hence:

- Under uniform weights the counts differ by at most one, and every combination
  appears once `n_cells` is at least the grid size.
- Under any weights every positive-weight combination appears once `n_cells`
  is at least their number, $|S|$.
- Targets are exact fractions, so equal weights tie exactly and a whole-number
  quota is never lost to float rounding.

| `active_count_weights` | `n_cells` | Targets | Counts |
| --- | --- | --- | --- |
| `[[1, 0], [1, 2]]` | 4 | `[[1, 0], [1, 2]]` | always the targets |
| `[[1, 100], [1, 1]]` | 4 | `[[1, 1], [1, 1]]` | always the targets: the floor lifts the three light combinations |
| `None` (2×2 grid) | 6 | `3/2` each | two combinations get 2 cells, the other two get 1 |
| `[[1, 9], [16, 37]]` | 7 | `[[1, 1], [80/53, 185/53]]` | `[[1, 1], [1, 4]]` or `[[1, 1], [2, 3]]` |
| `[[100, 1], [1, 1]]` | 3 | `[[1, 2/3], [2/3, 2/3]]` | the heavy combination plus two of the other three |

In the fourth row the floor takes two passes: the first fixes the weight-1
combination (share $7/63$), the second the weight-9 one (share $54/62$ of the
six cells left); the last five cells split $16 : 37$. A stratified corpus
records its realised counts in
[`diagnostics["active_count_coverage"]`](../guide/corpus.md#active-count-coverage),
and [`pymc_generator.active_count_coverage`](corpus.md#pymc_generator.active_counts.active_count_coverage)
reports them for any corpus (pass `prior=` when the corpus carries no block).

### Which cells are allocated

The allocation covers the cells of one generation call.

- `sample_prior_predictive(cfg, n=...)` and `DataGenerator.generate(n_tasks=...)`
  derive the cell count from `n` before generating — `ceil(n / draws_per_cell)`
  cells, or two or three cells of `max(1, n // 2)` draws when
  `n <= draws_per_cell` — and keep every generated cell; only the last may be
  partial. The allocation is therefore over exactly the cells you get, and a
  partial cell counts as one cell.
- `DataGenerator.iter_batches` generates every batch as its own corpus with its
  own seed: each batch is stratified, but their union is not allocated as a
  whole. Each batch applies the cap or the coverage floor to its own few cells,
  so with non-uniform weights the union follows the weights only when neither
  moves any batch's targets off their plain quotas `n_cells·w/Σw`. That takes
  many more cells per batch than positive-weight combinations, and more still
  for skewed weights. With the
  default `draws_per_cell=20`, `batch_size=100` gives five cells per batch; on
  a grid with more than five positive-weight combinations every batch then
  caps each combination at one cell, however much weight it holds. For
  weighted corpora, generate in one call or use batches with many cells.
  Uniform weights stay balanced in expectation across batches.
- The cell-level validation split stays a random subset of cells. It is not
  stratified, so a combination holding a single cell can sit entirely in
  validation.
- `sample_scm` draws one world with every node active. It has no cells and
  ignores the allocation; the weights are still validated.
- The experimental template path allocates through the same helpers: for a
  stratified config and `rng = np.random.default_rng(cfg.seed)`,
  `world_model_template.sample_cell_structures(cfg, rng)` plans the same
  per-cell treatment and covariate counts as `sample_prior_predictive(cfg)`
  (latent counts and structures follow the template's own RNG schedule). The
  plan follows the generator's state, so any generator in the same state plans
  the same cells. It also checks the allocation, the active ranges and the
  weights itself, since it never calls `validate()`. One compiled template
  serves every combination, because structure arrives as max-size inputs with
  activity masks.

### Seeds and randomness

- **Independent is the legacy draw.** Each cell draws `n_treatments_active`,
  `n_covariates_active` and `n_latent_active` from the corpus RNG in that
  order, exactly as before these fields existed, and nothing is spawned.
  Same-seed active counts, structures, draw seeds and model-array draws are
  unchanged by leaving these allocation controls at their defaults.
  Metadata and saved shards are not promised byte-identical across schema
  versions; `recipe.json` additionally lists the two fields at their defaults.
- **Stratified changes the corpus at the same seed.** Before the first cell,
  the allocation draws one integer seed from the corpus RNG for its own stream
  and draws only from that stream: one permutation of the grid and one uniform
  offset for the rounding, then one permutation of the cells. Each cell then
  skips its treatment and covariate draws and draws only its latent count, so
  every later corpus-RNG draw (graphs, structures, draw seeds, support masks,
  the validation split) shifts. This RNG layout does not allow aligning a
  stratified corpus with the independent one
  ([#28](https://github.com/pymc-labs/pymc-generator/issues/28)).
- A cell that exhausts the realism filter raises `RuntimeError` naming its
  `n_treatments_active`, `n_covariates_active` and `n_latent_active`, so a
  combination the prior cannot realise is identified directly.

## Direct-null treatments

`min_no_direct_effect_treatments: int = 0` is the minimum number of **active**
treatments each cell leaves without a direct `C→Y` edge. Their **direct**
contribution is zero; their total causal effect need not be. A `cy` budget is
an absolute arrow count clamped to the eligible slots, so a cell drawing
`n_treatments_active = 2` under `cy=(2, 10)` has both treatments live and no
negative class at all.

```python
from pymc_generator import make_scm_prior

cfg = make_scm_prior(
    n_treatments=10, n_covariates=6, n_latent=3,
    n_treatments_active_range=(2, 10),
    edge_budget={"cy": (1, 10)},
    min_no_direct_effect_treatments=1,
)
```

- The maximum direct-treatment count is
  `max(1, n_treatments_active - min_no_direct_effect_treatments)`. The existing
  budget or Bernoulli sampler operates subject to this cap, and the mandatory
  `cy ≥ 1` guard still applies.
- It applies to the Bernoulli path too: without a `cy` budget, surplus live
  treatments are demoted uniformly at random after the per-slot draw.
- Live treatments are scattered over **all** active slots, so slot index carries no
  information about the label.
- `min_no_direct_effect_treatments` must be `< n_treatments_active_range[0]`; a floor the
  smallest drawable cell could not honour is a validation error, not a silently
  dropped constraint.
- A direct-null treatment can still
  reach `Y` through another treatment (`worlds.treatment_role` reports that as a
  `feeder`); pin `edge_budget={"cc": 0}` for "no path to `Y`".
- `0` (the default) is inert: identical draws, no extra RNG, and
  `diagnostics["min_no_direct_effect_treatments"]` records the resolved value.

## Identifiability labels

`include_identifiability_labels: bool = True` controls the nested
`identifiability` block containing the dense, truth-derived `signal_metrics`
and `signal_metric_valid` arrays. Setting it to `False` omits that block while retaining the aggregate
`diagnostics["signal"]` quality report. It changes no model input, random draw,
or generated observable array; use the disabled form for feature-only shards.

::: pymc_generator.presets.make_scm_prior

::: pymc_generator.sampler.SCMPrior
    options:
      show_source: false
      members: false
