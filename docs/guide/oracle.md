# The posterior oracle

Ground truth answers *what happened*; the exact posterior on the observed data
answers *what is knowable* — the identification floor. An amortized model
(e.g. a PFN) is judged against the second: "PFN at oracle parity" is only a
meaningful claim if the oracle runs on **the same generative model, priors and
all**, that produced the data.

prior-generator owns that model — every world is one drawable `pm.Model`
([`build_world_model`](../reference/world-model.md#prior_generator.world_model.build_world_model))
— and exposes its observed-data variant
([`build_oracle_model`](../reference/world-model.md#prior_generator.world_model.build_oracle_model)):
the same structure and the *same prior definitions* with the world's dataset
attached, so `pm.sample` yields the posterior a NUTS fit can usually reach.
Weibull-adstock worlds are the exception: their two carryover parameters use
Metropolis rather than NUTS. Consumers should use it for oracle baselines,
never re-implement it.

## The recipe

Draw a world, build its oracle from the world object itself, fit it with
`pm.sample`, and compare posterior contribution bands against the world's
decomposition truth:

```python exec="1" source="material-block" result="text"
import prior_generator as pg

cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=28)
world = pg.sample_scm(cfg, seed=8)

oracle = world.oracle_model()      # pm.Model — same priors, data attached
print("free RVs:", sorted(rv.name for rv in oracle.free_RVs))
print("posterior series:", sorted(d.name for d in oracle.deterministics))
```

Then fit and compare (a short fit on this small world takes ~half a minute):

```python
import pymc as pm
import numpy as np

with oracle:
    idata = pm.sample(draws=500, tune=500, chains=2)

post = idata.posterior["contributions"]        # (chain, draw, n_time_steps, n_treatments)
bands = post.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
# the observed-path truth, shape (n_time_steps, n_treatments)
truth = world.data["contributions_observed"]

covered = (bands.sel(quantile=0.05) <= truth) & (truth <= bands.sel(quantile=0.95))
print("90% band coverage:", float(covered.mean()))
```

Compare the oracle's `contributions` against the world's
**`contributions_observed`** (the response evaluated on the observed spend) —
that is the quantity an MMM fit on observables estimates. The do()-style
`contributions` truth additionally removes upstream influence from spend and is
not identifiable from the observables alone.

### Two latent modes, two different posteriors

`oracle_model()` takes `latent="marginal"` (the **default**) or
`latent="sampled"`, and they expose different variables. Marginal mode
integrates the outcome-side Gaussian latents analytically — no latent-walk
dimensions, an exact multivariate-normal likelihood — which is why it samples
far better; sampled mode keeps the walks as explicit RVs so you can look at
them.

| | `latent="marginal"` (default) | `latent="sampled"` |
| --- | --- | --- |
| likelihood | `pm.MvNormal` on `sales` (exact walk covariance + iid `RW_Y` on the diagonal, plus a `1e-12 I` factorization guard) | `pm.Normal` on `sales`, `sigma=rw_y_std` |
| deterministics | `contributions`, `sales_mu`, plus `rw_b_std` / `rw_y_std` in relative outcome-noise mode | the same, **plus** `baseline` and `demand` |
| extra free RVs | — | `eps_d`, `eps_b` (the walk innovations) |
| floored intercept | raises — `max(RW_B, baseline_floor)` is not Gaussian | supported, applies the identical clip |

So the default oracle has **no `baseline` and no `demand` deterministic**: its
`sales_mu` is `E[sales | θ]` and excludes every latent walk realization. Ask for
`latent="sampled"` when you need posterior baseline or demand paths, and expect
the `O(n³)` Cholesky per gradient in marginal mode to be the cheaper trade at
weekly horizons.

What is comparable, in both modes: `contributions` against
`world.data["contributions_observed"]` (exactly), and `sales_mu` against
`world.data["sales"]` for total mean fit. Sampled mode's `baseline` is the
**non-media aggregate without observation noise**: intrinsic intercept `B`
plus attributed `D→Y` and `Z→Y` effects. These edges terminate at `Y`, not `B`.
Persisted `world.data["baseline"]` also includes `sales_noise`, so subtract that
column before comparing, or sum `baseline_intrinsic`, `confounder_contribution`,
and `control_contribution`. Comparing the two baseline labels without this
adjustment includes the realized observation noise in the recovery error.

Because the oracle runs under the same prior the world was drawn from, the
comparison is apples-to-apples for a PFN trained on corpora from the same
config — including [ACE prior-conditioning](corpus.md#prior-conditioning-ace)
intervals, which `SCM.oracle_model()` picks up automatically when enabled.

## What the oracle is — and is not

`build_oracle_model` keeps everything upstream of the observation exact and
documents six explicit, **mode-dependent** qualifications (the same six, in the
same order, as the API reference):

1. **Structure-known.** The true DAG, mechanism families, and walk smoothness
   are given. This is the *structure-known* oracle — an **upper bound** for any
   method that must also infer structure. A structure-unknown oracle would
   marginalize over graphs and is out of scope.
2. **Plug-in conditioning on the observed inputs.** Spend and controls enter as
    data. The information they carry about latent demand through `p(C | D)` /
    `p(Z | D)` is not modeled. This also deliberately does **not** model
    `p(C | eps_b)` or exploit baseline information encoded through the
    channel–baseline correlation (rho, configured here as confounding strength);
    demand is inferred from the sales residual via `D → Y` only — in sampled
    mode as an explicit latent, in marginal mode by integrating that same path
    through the residual covariance. This is a
    plug-in-channel qualification, not a claim of exact conditioning.
3. **Outcome-side Gaussian representation.** This is *not* an approximation of
   `RW_Y`: generation already draws `RW_Y = rw_y_std * eps_y` as iid Gaussian
   observation noise, so both oracle modes use its exact process. Under the
   default `outcome_std_mode="relative"` the free scale RV is the dimensionless
   `rw_y_std_rel` (`pm.Uniform` over `rw_sales_std_range`) and `rw_y_std` is a
   deterministic — `rel × sqrt(Σ (g_cy·β)²)`, the parameter-only media anchor;
   the `HalfNormal` scale prior appears only under
   `outcome_std_mode="absolute"`, where `rw_y_std` is itself the free RV.
   Marginal mode integrates `RW_D` and `RW_B` with their exact observed-window
   covariances and adds the iid `RW_Y` variance to the diagonal; sampled mode
   keeps the full-horizon walk transforms and the same exact iid `RW_Y`
   likelihood. The `1e-12 I` covariance floor in marginal mode is a
   factorization guard only — its implied `1e-6` standard deviation is ~`1e-6`
   of any realistic sales sd and cannot carry inference.

4. **Posterior-series labels.** Marginal mode has no `demand` and no
   `baseline` deterministic; its full-length `sales_mu` is `E[sales | θ]` and
   excludes every latent walk realization. Sampled mode's `baseline` aggregates
   the intrinsic intercept and attributed `D→Y` / `Z→Y` effects without `RW_Y`.
   The persisted `data["baseline"]` additionally includes `RW_Y` — subtract
   `sales_noise` to compare them. `contributions` is exactly comparable
   with `world.data["contributions_observed"]` in both modes; compare
   `sales_mu` with observed `sales` for total fit.
5. **Reproducible likelihood window.** The adstock convolution sees only the
   reported window (zero-padded start) while generation used
   `adstock_burn_in` weeks of real history. With burn-in enabled the oracle
   therefore observes `sales[warmup:]`, where `warmup` is the response support
   **admitted by the oracle's own inference priors**: the direct channels'
   adstock families, `l_max`, and the geometric decay range *after* any ACE
   prior-conditioning narrowing. Weibull's shape box
   (`weibull_lam_range` / `weibull_k_range`) is deliberately not consulted — a
   Weibull family's admitted reach is `l_max - 1` whatever its shape range.
   Concretely: a family set admitting Weibull, or geometric with a positive
   decay upper bound, drops `l_max - 1` rows; identity-only direct paths drop
   none; and geometric-only with the effective decay range pinned at `(0, 0)`
   also drops none, because a zero-decay kernel is an exact identity and every
   reported week is reproducible from persisted spend. Without burn-in
   `warmup` is `0`. If `warmup` would consume every reported week the builder
   raises rather than fitting an empty likelihood. Full-length `contributions`
   and `sales_mu` deterministics remain
   available for band comparison (`baseline` too, in sampled mode). The
   per-channel `saturation_scale` is a
   parameter-only reference level; it is persisted and supplied by
   `SCM.oracle_model()` so the oracle anchors the nonlinear response exactly
   where generation did.
6. **Weibull sampler downgrade.** pymc-marketing's `weibull_adstock` performs
   min-max normalization with an upstream `Min` operation that has no PyTensor
   pullback. For any Weibull-adstock channel, `pm.sample` consequently
   uses Metropolis—not NUTS—for `weibull_lam` and `weibull_k`. Identity and
   geometric-adstock channels differentiate cleanly and remain NUTS-eligible.
   The package's own analytic Weibull guard is not the cause; its reductions
   differentiate cleanly, and replacing the library normalization would break
   load-bearing parity with a stock `WeibullAdstock`. Metropolis mixing on the
   two carryover parameters makes their ESS suspect, so prefer
   geometric-adstock worlds when using the oracle as a reference posterior.


### Held-level shocks

When channel shocks are enabled, `SCM.oracle_model()` also passes the world's
reported shock channel, start, length, and held-level multiplier to the
oracle, together with the realized absolute held level. The oracle rejects a
schedule whose claimed held level does not match the recorded spend window or
`multiplier * channel_level`.
This metadata is **additional observed design state**, not an inferred
schedule and not a free random variable in the oracle. Because a shock only
clamps observed spend, the clamped window is already baked into the channel
matrix the oracle reads as data; the oracle builds no schedule tensors and
applies the same plain adstock kernel used everywhere else.

These are not conventional spend-only lift tests: a shock holds a channel to an
absolute level rather than perturbing it. It leaves the response state alone,
so ordinary carryover from pre-window spend decays into a held window and the
reported-window zero-padding caveat above applies uniformly.

## Guarantees that cannot drift

One definition of the priors serves draw and oracle: the `pm.Uniform` ranges
and walk priors live in shared helpers consumed by both builders, and the test
suite asserts per-variable log-density equality between the generative and
oracle models at drawn values, plus a finite oracle log-density at the truth.
Generation itself is untouched — building an oracle consumes no RNG, and
corpora remain byte-identical.
