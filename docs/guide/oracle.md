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

The oracle's `baseline` deterministic is `B`, including its `D→B` and `Z→B`
parent terms, but excluding `RW_Y`; the generator persists `baseline = B +
RW_Y`. Because `RW_Y` is not persisted separately, those baselines are **not**
directly comparable. `contributions` is exactly comparable with
`world.data["contributions_observed"]`; compare `sales_mu` with
`world.data["sales"]` for total mean fit. Baseline recovery carries an
irreducible floor of one sales-noise walk.

Because the oracle runs under the same prior the world was drawn from, the
comparison is apples-to-apples for a PFN trained on corpora from the same
config — including [ACE prior-conditioning](corpus.md#prior-conditioning-ace)
intervals, which `SCM.oracle_model()` picks up automatically when enabled.

## What the oracle is — and is not

`build_oracle_model` keeps everything upstream of the observation exact, with
six explicit concessions (all documented in the API reference):

1. **Structure-known.** The true DAG, mechanism families, and walk smoothness
   are given. This is the *structure-known* oracle — an **upper bound** for any
   method that must also infer structure. A structure-unknown oracle would
   marginalize over graphs and is out of scope.
2. **Plug-in conditioning on the observed inputs.** Spend and controls enter as
    data. The information they carry about latent demand through `p(C | D)` /
    `p(Z | D)` is not modeled. This also deliberately does **not** model
    `p(C | eps_b)` or exploit baseline information encoded through the
    channel–baseline correlation (rho, configured here as confounding strength);
    demand is inferred from the sales residual via `D → B` only. This is a
    plug-in-channel concession, not a claim of exact conditioning.
3. **iid sales-noise representation.** The oracle represents `RW_Y` as iid
   `Normal(0, rw_y_std)` with the **same** `HalfNormal` prior on the scale.
   Since the walk is normalized by a constant rather than by its own realized
   standard deviation, it is an ordinary multivariate normal and an exact
   density does exist — this concession is removable, just not yet removed. The
   latent demand and baseline walks stay exact (same transform, same horizon,
   sliced to the reported window).

4. **Baseline label.** The oracle's `baseline` is `B` (including its `D→B`
   and `Z→B` parent terms) without `RW_Y`, whereas the generator persists
   `B + RW_Y`. `RW_Y` is not stored separately, so baseline is not directly
   comparable. `contributions` is exactly comparable with
   `world.data["contributions_observed"]`; compare `sales_mu` with observed
   `sales` for total fit. Baseline recovery has an irreducible one-walk floor.
5. **Reproducible likelihood window.** The adstock convolution sees only the
   reported window (zero-padded start) while generation used
   `adstock_burn_in` weeks of real history. With positive burn-in and an
   eligible direct nonidentity adstock kernel, the oracle observes only
   `sales[l_max - 1:]`; otherwise every reported week is reproducible and the
   likelihood uses the full window. The full-length deterministics remain
   available for band comparison. The per-channel `saturation_scale` is a
   function of the drawn parameters alone; it is persisted and supplied by
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
