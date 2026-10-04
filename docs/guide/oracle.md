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

## Public sampling API

The maintained public objects include the sampling types below and the explicit
reusable lifecycle `OracleTemplate`, `CompiledOracle`, `build_oracle_template`,
and `compile_oracle`:

- `OracleSamplingConfig`: frozen sampling options. The forward-looking
  maintained API defaults are `draws=800`, `tune=500`, `chains=4`, `cores=4`,
  `target_accept=0.9`, and `random_seed=None`. The default latent representation
  is `"marginal"`; `discard_tuned_samples=True`, `progressbar=False`, and
  `compute_convergence_checks=False`. The explicit `nuts_sampler` config field
  defaults to `"nutpie"` and accepts `"pymc"` for an experiment that needs the
  PyMC sampler. Use `sampler_kwargs` only for additional, non-reserved PyMC
  options. For Nutpie, `adaptation` is validated against the installed
  Nutpie 0.16.11 values (`"diag"`, `"draw_diag"`, `"low_rank"`, and `"flow"`)
  and defaults to `"diag"`. It is passed through `nuts_sampler_kwargs` for
  one-shot PyMC delegation and as `adaptation` for compiled Nutpie sampling.
  A non-default adaptation request is rejected with `nuts_sampler="pymc"`
  rather than silently ignored. These defaults are not retrospective provenance
  for any earlier experiment.

- `OracleHealthCriteria`: thresholds and whether each diagnostic is required.
  Its defaults are zero divergences, maximum rank R-hat `1.01`, minimum bulk
  and tail ESS `400` (all required), maximum tree-depth saturation `0` and
  minimum BFMI `0.3` (both optional).
- `OracleSamplingReceipt`: immutable, JSON-native provenance and diagnostic
  metadata. Call `receipt.to_dict()` or `receipt.to_json()`; posterior values
  are not copied into it.
- `OracleSamplingResult`: the pair `result.idata` and `result.receipt`.
  `result.idata` is the unmodified PyMC 6 `xarray.DataTree`.
- `sample_oracle(world, config=None, *, criteria=None, world_identity=None,
  source_identity=None, configuration_identity=None, observed_indices=None)`:
  a one-shot convenience
  wrapper. It builds through `world.oracle_model(latent=config.latent)`, samples
  once, and returns an `OracleSamplingResult`; it does not cache compilation.
- `build_oracle_template(world, *, latent="marginal", observed_indices=None)`:
  builds one full-horizon graph and records its structural signature. Observation
  selectors use reported-row coordinates (after response-history/carryover warmup),
  defaulting to every reported row. This model-history warmup is distinct from
  MCMC tuning (warmup) draws configured by `tune`.
- `compile_oracle(template_or_world)`: compiles exactly once with Nutpie/Numba.
  Call `compiled.fit(other_world, ...)` for compatible worlds. Each fit binds a
  fresh data view with `with_data`; values are never copied into the receipt.
  Topology, rank, or unsupported-shape changes fail closed with a signature or
  shape error. A compiled owner is process-local and serialized for safe use.

#### `observed_indices` and reported-row coordinates

`SCM.oracle_model`, `build_oracle_template`, and `CompiledOracle.fit` accept
`observed_indices` as a one-dimensional sequence of unique non-negative integer
positions, or as a boolean mask. The mask length is the number of **reported
rows**; an integer selector must be in `[0, reported_rows)`. In
`SCM.oracle_model`, `sample_oracle`, and `build_oracle_template`, omitting
`observed_indices` defaults to all reported rows. In contrast,
`CompiledOracle.fit` reuses the selector bound to the compiled template when
`observed_indices` is omitted. For example, `observed_indices=[0, 2, 5]`
selects the first, third, and sixth reported rows, in that order.

Reported-row coordinates are positions in the caller-supplied reported
window, not absolute latent-horizon indices and not MCMC tuning draws. Let
`warmup` be the model-history response prefix discarded by the Oracle, and let
`burn_in` be the generator's carryover history used to extend the latent
horizon. Then `observed_indices=i` selects caller-supplied row
`warmup + i`; in particular, selector row `0` maps to caller row `warmup`.
The corresponding row in the internally extended latent horizon is
`burn_in + warmup + i`, so selector row `0` maps to latent-horizon row
`burn_in + warmup`. (`burn_in` is zero when the horizon is not internally
extended.) This offset is separate from sampler tuning/warmup controlled by
`config.tune`.

The oracle builds the full-horizon graph. The selector is placed in
`observed_indices_data` and the likelihood reads the selected reported outcome
rows, while full-window deterministic outputs retain their full reported-window
shape. A selected subset must contain at least one row. A reusable compiled
Oracle may change selector values between fits only without changing the
selected-row count; topology, rank, shape, duplicate, and out-of-range changes
fail closed. Selector order is meaningful. Compiled receipts include selector
identity separately from the bound `observed_indices_data` payload identity.
The one-shot `sample_oracle` receipt currently does not record selector
identity; record the selector itself (and this reported-row coordinate
convention) in any external comparison ledger rather than inferring it from a
one-shot receipt or reinterpreting it as a context/query mask over raw
full-horizon arrays.

#### Reusable data/configuration contract

The template carries an explicit data contract. The required legacy inputs are
`channels_data`, `controls_data`, `sales_data`, `saturation_scale_data`,
`g_cy_data`, `g_db_data`, `g_zb_data`, and `observed_indices_data`; generated
Oracle templates additionally require the four runtime numeric inputs
`prior_cond_carryover_alpha_data`, `prior_cond_hill_shape_data`,
`walk_width_d_data`, and `walk_width_b_data`. Manually constructed templates
must pass the exact names they provide in `data_contract`; missing or extra
known Oracle inputs fail closed before binding.

The structural signature contains only topology, rank/shape, distribution-family,
and branch choices. Prior interval endpoints and walk smoothness are compatible
runtime data and are validated through the named `pm.Data` inputs. Other numeric
configuration values consumed while constructing prior bounds are an explicit
fixed compatibility contract (see `ORACLE_CONFIG_SCHEMA`); changing one raises
an incompatibility instead of silently recompiling or binding stale values.
Zero-width prior conditioning is a separate point-mass topology. Direct
one-shot model construction supports it, but reusable templates reject it
before compilation/binding because a point mass cannot be rebound soundly.

Configuration is validated before model construction: draw/tune/chain/core
counts, target acceptance, seed, latent mode, Boolean options, thresholds, and
JSON-native sampler options must satisfy their declared types and ranges. Reserved
sampling controls cannot be smuggled through `sampler_kwargs`; invalid input
raises instead of being coerced, clipped, retried, or silently replaced. A model
or sampling exception propagates unchanged and produces no partial receipt.

The receipt records schema and package/environment versions, caller identities,
the oracle builder and latent mode, requested/effective sampling metadata,
elapsed timing, posterior presence (without posterior contents), diagnostics,
health, and limitation codes. `to_json()` uses canonical JSON without NaN values.

#### Nutpie version policy

Ordinary `pymc-generator` package and Conda installs intentionally declare
`nutpie` without an exact version pin. This is the supported compatibility
contract for the public API: the installed PyMC/Nutpie stack is reported in
each receipt, and unsupported behavior must not be inferred from a requested
sampler. The reproducibility lock (`uv.lock`) currently resolves Nutpie to
`0.16.11`.

For example, an experiment can explicitly override the maintained sampler
request with `pg.OracleSamplingConfig(nuts_sampler="pymc")`; the default remains
`"nutpie"`. For repeated compatible fits, prefer the explicit lifecycle:

```python
compiled = pg.compile_oracle(world)
first = compiled.fit(world, world_identity={"id": "w0"})
second = compiled.fit(other_world, world_identity={"id": "w1"})
```

The compiled receipt records the structural template signature, compile backend,
observed-index identity, and a hash/dtype/shape identity for every shared data
payload. It records no posterior values. There are no retry or fallback paths;
compile, binding, and sampling exceptions propagate unchanged.

```python
import pymc_generator as pg

cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=28)
world = pg.sample_scm(cfg, seed=8)
result = pg.sample_oracle(world)

# PyMC 6 returns a DataTree: use bracket access for groups.
post = result.idata["posterior"]["contributions"]
print("posterior shape:", post.shape)
print("health:", result.receipt.to_dict()["health"]["status"])
```

`result.idata["posterior"]` and `result.idata["sample_stats"]` are the supported
forms for DataTree groups; use bracket access consistently. Compare the oracle's
`contributions` against the world's
**`contributions_observed`** (the response evaluated on the observed treatment)
— that is the quantity an observational fit estimates. The do()-style
`contributions` truth additionally removes upstream influence from treatment;
recovering that causal target requires assumptions beyond an observational fit.

`random_seed=None` intentionally leaves seed selection to the sampling stack.
For a reproducible batch, pass an explicit per-world seed (for example,
`base_seed + world_index`) in each `OracleSamplingConfig`, and record the world
identity alongside the receipt. A fixed seed does not promise identical draws
across PyMC, Nutpie, PyTensor, NumPy, compiler, hardware, or other numerical-stack
versions; exact cross-stack draw reproducibility is not guaranteed.

### Diagnostics and health

Each receipt diagnostic metric has status `available`, `unavailable`, or
`invalid`. The metrics are `divergences`, `rhat`, `ess_bulk`, `ess_tail`,
`tree_depth_max`, `tree_depth_saturation`, and `bfmi`. Missing prerequisites are
`unavailable`; malformed data, failed diagnostic computation, or no finite result
is `invalid`. An available metric includes its value (and, where applicable,
method/reduction metadata). Tree-depth maximum and tree-depth saturation are
separate metrics: an observed maximum does not by itself establish saturation.

Health is tri-state and evaluated in this order. Thresholds and availability
requirements are independent: setting a threshold to `None` disables that
threshold, but does not make a diagnostic optional when its `require_*` flag is
`True`.

1. If **any available metric** crosses its enabled threshold, status is
   `"unhealthy"`.
2. Otherwise, if any **required** metric is `unavailable` or `invalid`—even when
   that metric's threshold is disabled—status is `"unknown"`.
3. Otherwise, status is `"healthy"`.

Optional unavailable diagnostics are retained as missing diagnostics and do not
change the status. A `healthy` result means only that the available, enabled
checks did not fail and required checks were available; it is not a convergence
guarantee, a scientific-validity claim, or evidence of recovery.

### Requested versus effective sampling

The receipt's `sampling.requested` records the requested draws, tuning, chains,
cores, target acceptance, seed, options, explicit `nuts_sampler` value (default
`"nutpie"`, or `"pymc"` when overridden), and Nutpie `adaptation` request.
`sampling.effective` records
what can be read from the returned DataTree, such as
present groups, chain/draw counts, sample-stat names, and the PyMC inference
library/version. The returned sampling result does **not** expose the effective
backend, step-method class, or mass-matrix representation. Accordingly,
`step_methods` and `mass_matrix` are reported as unavailable rather than guessed;
requesting Nutpie or an adaptation mode is not evidence for a particular
effective backend, step, mass matrix, or effective adaptation behavior.
There are no undocumented backend fallbacks, retries, or successful results after
an exception.

### Receipt limits

The receipt contains no posterior draws, posterior values, coordinates, model
graph, initial points, sample-stat arrays, or prior-predictive values. It does not
invent world/source/configuration IDs or hashes. Its health status must not be
used as a convergence, scientific-validity, or universal-recovery claim. The
oracle is a structure-known, plug-in reference and is not an exact joint DGP
posterior or a do()-style contribution estimand. It does not establish exact
cross-stack draw reproducibility, step-method selection, or a diagonal/dense
mass matrix. The corresponding
receipt limitation codes are `health_is_not_a_convergence_guarantee`,
`oracle_is_a_structure_known_plugin_reference`, `posterior_draws_are_not_included`,
and `step_method_and_mass_matrix_are_not_observed`.

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

post = idata["posterior"]["contributions"]  # (chain, draw, n_time_steps, n_treatments)
bands = post.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
# the observed-path truth, shape (n_time_steps, n_treatments)
truth = world.data["contributions_observed"]

covered = (bands.sel(quantile=0.05) <= truth) & (truth <= bands.sel(quantile=0.95))
print("90% band coverage:", float(covered.mean()))
```

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
   after that response-history burn-in. This is a data/model-history offset, not
   MCMC tuning or warmup draws; `tune` controls only the sampler's adaptation
   iterations.
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
