# Oracle with inferred walk smoothness

`pymc-generator` owns model construction, sampling primitives and posterior
prediction. Dataset eligibility/selection, evaluation-world preparation, fitting
job scheduling, scoring, qualification policies and dashboards belong to
`MarketingFoundationalModels/code/mfm_ops`. See [generator #69](https://github.com/pymc-labs/pymc-generator/issues/69)
and [consumer #14](https://github.com/pymc-labs/MarketingFoundationalModels/issues/14).

## Scientific contract

The oracle knows the active input identities, graph gates, response families,
generating priors and generation-time saturation reference anchors. It **does not
receive realized coefficients, baseline/latent paths or walk smoothness**. It
conditions on recorded input trajectories and fits selected context outcomes;
query outcomes and contribution truth are reserved for downstream evaluation.
No `sample_scm` regeneration is required.

The baseline and each latent walk have the generator's Beta-induced probabilities
over rounded/clamped integer smoothing widths. The model exactly sums the **joint
width choice over the entire context**, sharing continuous walk innovations across
widths. It does not mix independently at each observation or substitute mean
smoothness. One baseline plus one latent, each with 26 widths, gives 676 components.
Construction rejects more than 4096 combinations explicitly.

Both intercept and absorbing non-treatment floors preserve the generator's
clipping order. All selected context rows are fitted, including early rows.
Missing pre-window input history uses **zero-initialized adstock**, an approximation
recorded in result metadata. Baseline/latent walk operators retain full burn-in
calibration. Input-generating likelihoods are not modeled, so this remains an
outcome-side reference, not the complete joint posterior.

## Construct directly from persisted data

```python
import pymc as pm
from pymc_generator import build_oracle_model, sample_inferred_walk_predictions

# g_active: active-size graph; cfg: recorded generating priors.
# structural: carryover_family and sat_family; no smoothness fields needed.
# data: treatments, covariates, outcome, saturation_scale, in raw units.
model = build_oracle_model(
    g_active, cfg, structural, data,
    latent="sampled", walk_smoothness="inferred",
    observed_indices=context_indices,
)
trace = pm.sample(model=model, nuts_sampler="nutpie", draws=800, tune=500,
                  chains=4, cores=4, random_seed=42)
paired = sample_inferred_walk_predictions(model, trace["posterior"], random_seed=43)
```

Budgets above illustrate the API, not a convergence guarantee. In inferred mode,
indices address the **full reported window**. Only selected outcomes must be
finite; query values can be NaN and are replaced with zeros in model data.
An SCM wrapper carrying the same metadata can use
`OracleSamplingConfig(latent="sampled", walk_smoothness="inferred")` with
`sample_oracle`; its receipt records the mode.

The legacy default remains `walk_smoothness="known"`, preserving its model,
receipt serialization and discarded response-history prefix. The older
`OracleTemplate`/`CompiledOracle` lifecycle remains a supplied-width interface;
it explicitly rejects inferred-width requests. Use the family worker below for
reusable inferred-width compilation.

## Compile once per family worker

```python
from pymc_generator import FamilyOracle, OracleInputs, OracleSamplingConfig

worker = FamilyOracle(cfg, saturation="hill", carryover="geometric")
inputs = OracleInputs(
    channels=active_channels,              # (reported time, active channels)
    controls=active_controls,              # (reported time, active controls)
    context_indices=context_indices,      # unique full-window row positions
    context_outcome=context_outcomes,      # only those selected outcomes
    saturation_scale=active_anchors,
    channel_ids=original_channel_ids,
    control_ids=original_control_ids,
    channel_gates=direct_channel_gates,
    control_gates=direct_control_gates,
    latent_gates=direct_latent_gates,
)
bound = worker.bind(inputs, cfg=world_cfg)
config = OracleSamplingConfig(
    latent="sampled", walk_smoothness="inferred",
    draws=800, tune=500, chains=4, cores=4, random_seed=42,
)
trace = worker.sample(bound, config)
paired = worker.predict_posterior(bound, trace["posterior"], random_seed=43)
```

The caller removes absent input columns and retains original identities; there
are no padded sampled input parameters. At least one channel is required; zero
controls are supported. `latent_gates` has one entry per configured latent factor
(an empty vector for none). The worker does not select datasets or columns.

A binding snapshots observations and prior data. A later bind cannot mutate an
earlier fit. Fresh adaptation runs for each fit. A worker reuses its numerical
compilation across:

- active channel/control counts and direct gates;
- context lengths and selected row positions;
- regime-specific baseline mean bounds, baseline/noise scales and relative-scale
  bounds, and smoothness Beta hyperparameters.

The first binding materializes the lazy Numba signature. Later bindings must not
add a signature; compilation identity/count are reported in trace attributes.
Input-generation-only trajectory settings do not split workers. Horizon, burn-in,
latent count, maximum smoothing width, family pair, floor semantics, scale mode,
mechanism/coefficient prior supports and point-vs-continuous prior topology remain
fixed. Incompatible bindings fail rather than silently compiling another graph.
World-specific prior conditioning is currently unsupported in `FamilyOracle`;
the one-shot builder supports it. Flow adaptation is unsupported by the dynamic
callback backend. Diagnostic qualification and storage policy remain consumer
responsibilities. Store the sampling configuration, source identity and exact
world configuration alongside outputs.

## Posterior and prediction semantics

`walk_width_probabilities` records the conditional joint posterior probabilities.
Its `walk_combination` coordinate lists widths in baseline-then-latent order.
Averaging probabilities over posterior draws gives the marginal width posterior.
Smoothness within a width bin is drawn from its truncated generating Beta prior;
that within-bin detail is not separately identifiable from observations.

Posterior `baseline`, `outcome_mu` and `latent_unobserved` are **conditional
width-averaged means**, not joint trajectory draws. For uncertainty, coverage and
scoring use the paired prediction helpers. They select one joint width per draw
using the whole context, then return:

- `baseline`: intrinsic baseline plus latent and control effects, excluding noise;
- `contributions` and `control_contributions`: nominal per-input effects (under
  an absorbing floor these nominal terms need not be additive components of baseline);
- `latent_unobserved`, `outcome_mu` and a newly noisy `outcome` over the full window;
- jointly sampled `walk_width` and `walk_smoothness`.

Original chain/draw coordinates and family-worker input IDs are preserved.
`FamilyOracle` stores a private `_state` variable needed by `predict_posterior`;
retain it until predictions are generated. Prediction uses the same immutable
binding as the fit, including that world's smoothness prior. All supplied draws
are used; no thinning, scoring or fit filtering occurs. Consumers can compute
full-draw summaries before choosing a storage retention policy.

A Potential represents the joint context likelihood. Ordinary
`pm.sample_posterior_predictive` does not reconstruct the marginalized widths:
use these helpers. The mixture does not supply independent pointwise likelihoods
for automatic LOO, because its width choice is shared across observations.

## Verification and limits

Tests compare the joint mixture with explicit supplied-width enumeration, check
floor semantics and context/truth isolation, check gradients by finite differences,
and exercise active-count/regime rebinding and real Nutpie execution. Short smoke
fits validate integration only: they do not establish convergence, calibration,
recovery or production throughput. Production fits still require chain-aware
diagnostics and Monte Carlo precision assessment in the evaluation consumer.
