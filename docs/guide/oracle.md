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
attached, so `pm.sample` yields the posterior a NUTS fit can actually reach.
Consumers should use it for oracle baselines, never re-implement it.

## The recipe

Draw a world, build its oracle from the world object itself, fit NUTS, and
compare posterior contribution bands against the world's decomposition truth:

```python exec="1" source="material-block" result="text"
import prior_generator as pg

cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, T=28)
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

post = idata.posterior["contributions"]              # (chain, draw, T, K)
bands = post.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
truth = world.data["contributions_observed"]         # (T, K) — the observed-path truth

covered = (bands.sel(quantile=0.05) <= truth) & (truth <= bands.sel(quantile=0.95))
print("90% band coverage:", float(covered.mean()))
```

Compare the oracle's `contributions` against the world's
**`contributions_observed`** (the response evaluated on the observed spend) —
that is the quantity an MMM fit on observables estimates. The do()-style
`contributions` truth additionally removes upstream influence from spend and is
not identifiable from the observables alone.

Because the oracle runs under the same prior the world was drawn from, the
comparison is apples-to-apples for a PFN trained on corpora from the same
config — including [ACE prior-conditioning](corpus.md#prior-conditioning-ace)
intervals, which `SCM.oracle_model()` picks up automatically when enabled.

## What the oracle is — and is not

`build_oracle_model` keeps everything upstream of the observation exact, with
three explicit concessions (all documented in the API reference):

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
3. **iid sales-noise representation.** Every random walk in the generator is
   normalized in-place (`walk * std / walk.std()`), so the sales-noise walk has
   no closed-form density to invert — exact `pm.observe`-style conditioning on
   this graph is mathematically impossible. The oracle represents `RW_Y` as iid
   `Normal(0, rw_y_std)` with the **same** `HalfNormal` prior on the scale. The
   latent demand and baseline walks stay exact (same transform, same horizon,
   sliced to the reported window).

Additionally, the adstock convolution sees only the reported window
(zero-padded start) while generation used `adstock_burn_in` weeks of real
history, so **drop the first `l_max` weeks** from band comparisons. The
generation-time per-channel `saturation_scale` is persisted and supplied by
`SCM.oracle_model()`, preventing that initial-history difference from shifting
the nonlinear response anchor after carryover itself has converged.

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
