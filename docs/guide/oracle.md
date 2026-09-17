# The posterior oracle

The oracle is a **structure-known, plug-in reference model**, not the exact
posterior of the complete data-generating process and not a universal recovery
bound. It shares the generator's outcome-side prior definitions and response
functions, while treating observed treatments and covariates as fixed inputs.
Their likelihoods given latent-unobserved factors or baseline innovations are omitted.

Use [`build_oracle_model`](../reference/world-model.md#pymc_generator.world_model.build_oracle_model)
to compare a fitted reference posterior with known simulation truth. Interpret
that comparison under the conditioning assumptions below and report convergence
diagnostics. With the locked released stack, identity, geometric, and Weibull
carryover parameters are NUTS-eligible; Weibull no longer requires Metropolis
because of missing upstream gradients.

## The recipe

Draw a world, build its oracle from the world object itself, fit it with
`pm.sample`, and compare posterior contribution bands against the world's
decomposition truth:

```python exec="1" source="block" result="text"
import pymc_generator as pg

cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=28)
world = pg.sample_scm(cfg, seed=8)

oracle = world.oracle_model()      # pm.Model — same priors, data attached
print("free RVs:", sorted(rv.name for rv in oracle.free_RVs))
print("posterior series:", sorted(d.name for d in oracle.deterministics))
```

Then fit and compare; runtime and mixing depend on the realized mechanisms:

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
**`contributions_observed`** (the response evaluated on the observed treatment) —
that is the quantity an MMM fit on observables estimates. The do()-style
`contributions` truth additionally removes upstream influence from treatment.
Recovering that causal target requires assumptions beyond an observational fit.

### Two latent representations

`oracle_model()` takes `latent="marginal"` (the **default**) or
`latent="sampled"`. For unfloored Gaussian walks, marginal mode analytically
integrates the outcome-side latents in the sampled specification, with the
numerical covariance guard described below. The parameter marginals therefore
represent the same plug-in model, apart from that guard, but the exposed latent
variables differ. Marginalization reduces dimension; actual mixing and runtime
still need measurement.

| | `latent="marginal"` (default) | `latent="sampled"` |
| --- | --- | --- |
| likelihood | `pm.MvNormal` on `sales` (exact walk covariance + iid `RW_Y` on the diagonal, plus a `1e-12 I` factorization guard) | `pm.Normal` on `sales`, `sigma=rw_y_std` |
| deterministics | `contributions`, `outcome_mu`, plus `rw_b_std` / `rw_y_std` in relative outcome-noise mode | the same, **plus** `baseline` and `demand` |
| extra free RVs | — | `eps_d`, `eps_b` (the walk innovations) |
| floored intercept | raises — `max(RW_B, baseline_floor)` is not Gaussian | supported, applies the identical clip |

So the default oracle has **no `baseline` and no `demand` deterministic**: its
`outcome_mu` is `E[sales | θ]` and excludes every latent walk realization. Ask for
`latent="sampled"` when you need posterior baseline or latent-unobserved paths. Marginal
mode trades fewer sampled dimensions for an `O(n³)` Cholesky factorization per
likelihood evaluation; neither representation is uniformly cheaper.

What is comparable, in both modes: `contributions` against
`world.data["contributions_observed"]` (exactly), and `outcome_mu` against
`world.data["outcome"]` for total mean fit. Sampled mode's `baseline` is the
**non-treatment aggregate without observation noise**: intrinsic intercept `B`
plus attributed `D→Y` and `Z→Y` effects. These edges terminate at `Y`, not `B`.
Persisted `world.data["baseline"]` also includes `outcome_noise`, so subtract that
column before comparing, or sum `baseline_intrinsic`, `latent_unobserved_contribution`,
and `covariate_contribution`. Comparing the two baseline labels without this
adjustment includes the realized observation noise in the recovery error.

The shared prior definitions make this a useful reference for amortized models
trained with the same configuration, including [ACE prior-conditioning](corpus.md#prior-conditioning-ace)
intervals that `SCM.oracle_model()` picks up automatically. Shared priors alone do
not remove the plug-in conditioning qualification.

## What the oracle is — and is not

The outcome-side response functions and priors are shared with generation.
The API reference documents these mode-dependent qualifications:

1. **Structure-known.** The true DAG, mechanism families, and walk smoothness
   are given. Extra structural information is useful, but does not establish
   a universal upper bound when this fit omits information from the inputs.
   A structure-unknown reference would marginalize over graphs.
2. **Plug-in conditioning on the observed inputs.** Treatments and covariates enter as
    data. The information they carry about latent-unobserved factors through `p(C | D)` /
    `p(Z | D)` is not modeled. This also deliberately does **not** model
    `p(C | eps_b)` or exploit baseline information encoded through the
    treatment–baseline correlation (rho, configured here as confounding strength);
    the latent-unobserved factor is inferred from the outcome residual via `D → Y` only — in sampled
    mode as an explicit latent, in marginal mode by integrating that same path
    through the residual covariance. This is a
    plug-in-treatment qualification, not a claim of exact conditioning.
3. **Outcome-side Gaussian representation.** This is *not* an approximation of
   `RW_Y`: generation already draws `RW_Y = rw_y_std * eps_y` as iid Gaussian
   observation noise, so both oracle modes use its exact process. Under the
   default `outcome_std_mode="relative"` the free scale RV is the dimensionless
   `rw_y_std_rel` (`pm.Uniform` over `rw_outcome_std_range`) and `rw_y_std` is a
   deterministic — `rel × sqrt(Σ (g_cy·β)²)`, the parameter-only treatment anchor;
   the `HalfNormal` scale prior appears only under
   `outcome_std_mode="absolute"`, where `rw_y_std` is itself the free RV.
   Marginal mode integrates `RW_D` and `RW_B` with their exact observed-window
   covariances and adds the iid `RW_Y` variance to the diagonal; sampled mode
   keeps the full-horizon walk transforms and the same exact iid `RW_Y`
   likelihood. Marginal mode also adds `1e-12 I` as a numerical factorization
   guard. Its `1e-6` standard-deviation scale should be assessed against the
   chosen outcome units, rather than assumed negligible at every scale.

4. **Posterior-series labels.** Marginal mode has no `demand` and no
   `baseline` deterministic; its full-length `outcome_mu` is `E[sales | θ]` and
   excludes every latent walk realization. Sampled mode's `baseline` aggregates
   the intrinsic intercept and attributed `D→Y` / `Z→Y` effects without `RW_Y`.
   The persisted `data["baseline"]` additionally includes `RW_Y` — subtract
   `outcome_noise` to compare them. `contributions` is exactly comparable
   with `world.data["contributions_observed"]` in both modes; compare
   `outcome_mu` with observed `sales` for total fit.
5. **Reproducible likelihood window.** The carryover convolution sees only the
   reported window (zero-padded start) while generation used
   `carryover_burn_in` weeks of real history. With burn-in enabled the oracle
   therefore observes `sales[warmup:]`, where `warmup` is the response support
   **admitted by the oracle's own inference priors**: the direct treatments'
   carryover families, `l_max`, and the geometric decay range *after* any ACE
   prior-conditioning narrowing. Weibull's shape box
   (`weibull_lam_range` / `weibull_k_range`) is deliberately not consulted — a
   Weibull family's admitted reach is `l_max - 1` whatever its shape range.
   Concretely: a family set admitting Weibull, or geometric with a positive
   decay upper bound, drops `l_max - 1` rows; identity-only direct paths drop
   none; and geometric-only with the effective decay range pinned at `(0, 0)`
   also drops none, because a zero-decay kernel is an exact identity and every
   reported week is reproducible from persisted treatments. Without burn-in
   `warmup` is `0`. If `warmup` would consume every reported week the builder
   raises rather than fitting an empty likelihood. Full-length `contributions`
   and `outcome_mu` deterministics remain
   available for band comparison (`baseline` too, in sampled mode). The
   per-treatment `saturation_scale` is a
   parameter-only reference level; it is persisted and supplied by
   `SCM.oracle_model()` so the oracle anchors the nonlinear response exactly
   where generation did.
6. **Sampler eligibility.** The locked released stack differentiates
   pymc-marketing's Weibull normalization, so `pm.sample` can use NUTS for
   `weibull_lam` and `weibull_k` as well as identity/geometric-carryover parameters.
   This removes the previous upstream gradient restriction, not the need to
   assess the fit. Check divergences, R-hat, and effective sample sizes for every
   posterior; gradient availability alone establishes neither convergence nor
   reliable recovery. Automatic sampler selection can differ from the previous
   stack, so unchanged generation seeds do not imply identical posterior draws.


### Held-level shocks

When treatment shocks are enabled, `SCM.oracle_model()` also passes the world's
reported shock treatment, start, length, and held-level multiplier to the
oracle, together with the realized absolute held level. The oracle rejects a
schedule whose claimed held level does not match the recorded treatment window or
`multiplier * treatment_level`.
This metadata is **additional observed design state**, not an inferred
schedule and not a free random variable in the oracle. Because a shock only
clamps observed treatment, the clamped window is already baked into the treatment
matrix the oracle reads as data; the oracle builds no schedule tensors and
applies the same plain carryover kernel used everywhere else.

These are not conventional treatment-only lift tests: a shock holds a treatment to an
absolute level rather than perturbing it. It leaves the response state alone,
so ordinary carryover from pre-window treatment decays into a held window and the
reported-window zero-padding caveat above applies uniformly.

## Guarantees that cannot drift

One definition of the priors serves draw and oracle: the `pm.Uniform` ranges
and walk priors live in shared helpers consumed by both builders, and the test
suite asserts per-variable log-density equality between the generative and
oracle models at drawn values, plus a finite oracle log-density at the truth.
Generation itself is untouched — building an oracle consumes no RNG, and
corpora remain byte-identical.
