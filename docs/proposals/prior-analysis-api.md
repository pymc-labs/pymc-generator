# Proposal: prior coverage and diversity analysis API

**Status:** design for discussion; no implementation or API commitment.
**Baseline:** `main` at `ce068d5` (domain-neutral scientific API).

## Decision to discuss

Add a post-hoc, between-world analysis layer beside `data_diagnostics`, not a
new generator or an automatic prior optimizer. Separate three questions:

1. **Patterns:** what kinds of worlds did this prior actually produce?
2. **Repetition:** which worlds look similar, and is that similarity within or
   across generation groups?
3. **Coverage:** where do target worlds lie relative to those patterns, and
   which regions have few examples?

A descriptor distance is not information content, identifiability, or expected
learning value. There is no defensible universal percentage of repetition to
recommend from these diagnostics alone.

## What main already provides

- `SCMPrior`, `make_scm_prior`, `sample_prior_predictive`, `DataGenerator`, and
  persisted corpora; generation remains explicit and reproducible.
- `data_diagnostics(corpus | sequence_of_scms)` computes statistics **inside
  each world**, then summarizes across worlds: means/spread, roughness/spikes,
  dependence, VIF, temporal association, and contribution budgets.
- Result objects expose `table()`, strict-JSON `summary()`, and `to_frame()`;
  descriptors carry stable keys and observed/latent/derived roles. Coverage
  ledgers distinguish eligibility, invalidity, finite values, and infinities.
- Both levels and differences are available. Neither differencing nor a large
  collection of worlds establishes stationarity or independent sampling.

Reuse these definitions and output conventions. Do not call the full expensive
`data_diagnostics` report just to extract a few cheap features: share the
relevant calculation kernels internally, preserving existing public behavior.
The existing `CoverageLedger` counts evidence behind a statistic; the new
**target coverage** report answers a different question.

## Proposed user-facing shape

The following is proposed usage, **not executable on main**. Names and defaults
are discussion points. Keep two operations visible: extract descriptors; fit a
reference report and compare targets using that same reference geometry.

```python
import pymc_generator as pg

# Existing API: keep generation separate from analysis.
corpus = pg.DataGenerator(cfg).generate(n_tasks=10_000, seed=42)

# Proposed API: small, inspectable per-world representation.
worlds = pg.world_descriptors(
    corpus,
    worlds=(corpus["is_val"] == 0),  # reference uses training worlds only
    source_id="training-A",
    lags=(1, 4, 13),
)
print(worlds.table())
worlds.to_frame()  # one world per row; named descriptor columns

# Explicit feature selection: no universal similarity score over everything.
report = pg.corpus_diagnostics(
    worlds,
    features=(
        "covariate_level_to_variation_median",
        "treatment_cv_median",
        "treatment_acf_lag_1_median",
        "treatment_pair_abs_pearson_differences_median",
    ),
    strata=("n_treatments_active", "n_covariates_active", "n_time_steps"),
    max_reference_worlds=4096,
    max_query_worlds=4096,
    seed=42,
)
print(report.redundancy.table())

# Same extractor and definitions for a generated target scenario.
targets = pg.world_descriptors(
    paper_worlds, source_id="paper", lags=(1, 4, 13),
)
comparison = report.compare(targets)
print(comparison.table())
comparison.to_frame()  # target ID, percentiles, distances, neighbor IDs, status

# Raw-feature bins are explicit and reusable across candidate priors.
density = report.density(
    bins={"covariate_level_to_variation_median": [0, 1, 3, 10, 30, float("inf")]},
)
print(density.table())
```

`WorldDescriptors`, `CorpusDiagnostics`, and their result facets follow the
existing report style; NumPy is the primary representation and pandas import
stays lazy. No estimator inheritance hierarchy, plugin registry, CLI, or
implicit generation is needed for this proposal.

### Real-data targets without invented ground truth

`world_descriptors` also accepts a proposed `ObservedWorlds` value object:

```python
observed = pg.ObservedWorlds(
    treatment_raw=treatments,       # (world, time, treatment)
    covariates=covariates,          # (world, time, covariate)
    outcome_raw=outcome,            # optional (world, time)
    treatment_active_mask=treatment_mask,
    covariate_active_mask=covariate_mask,
)
targets = pg.world_descriptors(
    observed, source_id="observed-study", lags=(1, 4, 13),
)
comparison = report.compare(targets)
```

Use the scientific corpus array names and explicit axes/masks. This adapter is
not a partially valid training corpus: no synthetic latent variables,
decomposition, graph, or group IDs are fabricated. A single real dataset is one
world, not one world per time point. Initially each input has one common horizon;
separate horizons can be described separately and concatenated. Time sampling
frequency must match for lag-based comparisons; no resampling is implicit.
Missing/nonfinite observed values require explicit preprocessing; padding is
excluded by masks, never inferred from zero-valued data.

### Descriptor and identity contract

- Rows have `(source_id, world_id)` identity. A group identity is
  `(source_id, cell_id)` when actual generation provenance is available; it is
  not inferred from similar features. Namespaces identify original generation
  runs, not merely filenames. Shards of one run retain original row/group IDs;
  independent batch generations need different namespaces.
- The existing `worlds=` selection convention preserves original row IDs.
  Explicit `world_ids=` and `cell_ids=` overrides accept original identities
  for already-sliced/repacked sources, validated against row count.
- `WorldDescriptors.concat(parts)` checks descriptor schema/configuration
  compatibility and identity collisions. Caller-supplied IDs are required when
  slicing/renumbering would otherwise lose original provenance. Identity
  collisions are errors, not a request to deduplicate data.
- Store a numerical values matrix, eligibility/validity state, per-feature
  definitions, and row metadata separately. Numerical NaNs never become zeros.
  Tables/frames expose status, and JSON uses null plus explicit reasons/counts.
- Metadata includes active counts, horizon, source kind, and available grouping
  and structural provenance. Counts/graph labels are not accidentally treated
  as continuous distances.
- Default descriptors are **observable**. Retained truth such as decomposition,
  graph/mechanism labels, or noise diagnostics is explicitly requested with
  `include_truth=True`, marked as such, and unavailable for observed-only data.
- Role summaries use active nodes only, with explicit reducers and denominators;
  e.g. `covariate_level_to_variation_median` is the finite-only median of
  per-active-covariate ratios, not a ratio after pooling nodes or worlds.
  This follows existing finite-only report aggregation: no finite entries
  gives an unavailable reduction, not zero. Return contributing, invalid, and
  infinite node counts; a constant-node fraction keeps constant-dominated
  worlds visible rather than hiding them behind a finite median.
  Existing series keys support drilling back into individual nodes. Role
  reductions are permutation-invariant but lossy; identical summaries do not
  prove identical worlds.

## Initial descriptor vocabulary

| Facet | Examples | Boundary |
|---|---|---|
| Level versus variation | `abs(mean) / std` for covariates | Multiplicative-unit invariant; deliberately sensitive to offsets |
| Treatment variability | `std / abs(mean)`, zero fraction, active run lengths | CV follows the existing all-zero convention below; zero fraction needs a meaningful zero origin |
| Temporal shape | ACF at named lags, roughness, spike ratio | Levels and differences stay separate; insufficient lag pairs are invalid |
| Dependence | Absolute pairwise Pearson summaries in each view | Fewer than two active treatments has no treatment-pair statistic |
| Composition | Active counts, horizon, known graph/family labels | Strata/metadata, not ordered numerical features |
| Truth-only audit | Existing contribution shares, explicit signal/noise definitions | Never estimated from observed data by pretending truth is available |

Start with the metrics needed to expose the reported level-dominated-control
mismatch. Preserve current definitions for existing roughness, ACF, dependence,
and contribution statistics. A future trend/seasonality classifier is not
implied by an ACF or roughness measurement.

For `abs(mean) / std`, nonzero constants yield positive infinity; all-zero
series are undefined (`0/0`). For treatment CV, reuse the existing
`signal_diagnostics._coefficient_of_variation` convention: nonzero constants
and all-zero series have CV zero; zero-mean nonzero series are invalid.
The all-zero case is a package convention, not a mathematical resolution of
`0/0`, and remains identifiable through the zero/constant descriptors.
No new CV definition or changed signal-gate behavior is introduced. No arbitrary
absolute epsilon or clipping should silently turn undefined ratios into finite
examples.
Near-constant detection, if used, needs a relative, documented numerical
policy. Positive rescaling preserves these ratios; centering does not preserve
level-to-variation. “Scale-free” never means invariant to every preprocessing
operation, nor does it prove that magnitudes are irrelevant to every learner.

## Report semantics

### 1. Repetition, not a claim about information

Expose nearest-neighbor distances and the selected neighbor identities in the
chosen descriptor space, with two separate views:

- **Other worlds:** exclude only the same world identity.
- **Other cells:** additionally exclude the same known generation group.
  Missing group provenance makes this view unavailable, not a singleton group.

Report the query/reference/eligible counts, distance quantiles, and selected
feature definitions. Singleton strata, all-same-cell strata, and insufficient
finite data return a reason, not a made-up distance. Known groups provide
realizations-per-group counts; they do not prove that independent groups have
different graphs or that draws within a group contain identical data.

Feature collisions are **descriptor collisions**, not exact duplicate data.
Byte-level reuse or repetition induced by an optimizer's epoch schedule needs
source/training provenance; nearest-neighbor statistics cannot reconstruct it.
Do not introduce an automatic deduplication or sampling gate here.

### 2. Freeze the comparison geometry

Within each requested stratum, compute median/IQR standardization from all
complete finite reference rows; fall back to reference standard deviation for
nonconstant zero-IQR columns. Constant-reference columns are constraints:
a target mismatch is flagged, never silently hidden by dropping the column.
If all selected columns are constant there is no numerical neighbor geometry.
Distance is Euclidean on the retained standardized finite features; show
per-feature standardized differences for the closest neighbor.

Use one fixed feature set and complete finite rows per stratum, with excluded
row counts/reasons. Do not use pair-specific feature deletion: it creates
incomparable distances and can destroy metric properties. Undefined/infinite
features still participate in marginal/status reports but not finite-vector
neighbor calculations. Unsupported target strata are explicitly uncovered.
Do not fit scaling, feature selection, bins, or thresholds on target values.

Comparing candidate priors by raw distance requires a shared fitted geometry,
feature set, and sampling budget; independently standardized reports do not
provide an apples-to-apples ranking. The initial report is explanatory, not an
automatic winner selector. Per-prior percentile and density reports remain
useful with their reference populations clearly named.
Distances from different stratum-specific scalings are also not directly
comparable; do not average them into a single corpus quality ranking.

### 3. Target coverage

`report.compare(targets)` reuses the frozen reference geometry.
It requires identical definitions for the **selected** features and strata:
version, role, view, lag, reducer, numerical/mask policy, and time convention.
Unselected descriptor differences are harmless. A missing selected target
feature yields `missing_feature` with no joint distance; the same key with a
conflicting definition is an error, not an implicit conversion.

Same `(source_id, world_id)` is always excluded from neighbor candidates.
`compare(targets, exclude_same_cell=True)` additionally excludes shared known
cells; missing group identity makes that requested comparison unavailable.
The default excludes self only and reports that policy. These exclusions also
apply to each target's marginal reference population and its denominator.
Namespace aliases cannot establish independent provenance.

It returns:

- Per-feature empirical reference percentiles, with midrank tie convention
  `(count(reference < x) + 0.5 * count(reference == x)) / n_valid`.
  Rank within the target's stratum using eligible, valid reference values,
  including ordered signed infinities; exclude undefined/ineligible values and
  state `n_valid`. Empty reference populations and undefined target values have
  no percentile. Finite-only summary quantiles remain separate from this ECDF.
- Joint nearest-reference distances and neighbor IDs, not just marginals.
- The reference leave-self-out/leave-cell-out distance summaries for context,
  with all exclusions recorded; these are not calibrated p-values or automatic
  out-of-distribution thresholds.
- Missing/undefined features, unsupported strata, and violations of constant
  reference features, even when a numerical distance cannot be computed.

A target can be ordinary on every marginal and still fall in a joint-support
gap. An extreme percentile is evidence about a selected descriptor, not proof
that the prior assigns zero probability to a world or that inference will fail.
No target data is silently inserted into the reference sample.

### 4. Density and representation

`report.density(bins=...)` counts worlds and distinct known groups in explicit
one- or two-dimensional raw-feature bins, optionally crossed with the report's
count/horizon strata. Bin intervals are left-closed/right-open, with the final
right endpoint included. Report empty bins, invalid/missing/infinite states,
underflow/overflow, and denominators. A positive-infinity endpoint closes the
finite tail; positive-infinity feature values remain a separate status bucket.
Changing bin edges changes the question; this is not a density-free coverage
metric. Both world-weighted and group counts are descriptive, not effective
independent sample sizes.

## Bounded cost and reproducibility

Descriptor extraction is chunkable over existing `DataGenerator.iter_batches`
or stored corpora and retains compact feature rows, not all raw time series.
Do not promise batch-size-invariant generation: the current iterator calls
separate seeded generation runs. Retain original run namespaces when combining.

Marginal summaries, fitted scaling, and explicit density counts use all
descriptor rows eligible for each calculation, not just neighbor samples.
Neighbor search uses reproducibly sampled reference and query rows when its
configured caps are exceeded. Caps apply across the report, not per stratum.
Allocate proportionally to eligible stratum size with largest-remainder
rounding and one per nonempty stratum when the cap permits; otherwise mark
unrepresented strata explicitly. Sampling is uniform within allocated strata.
Reports expose IDs, seed, allocation, exclusions, and exact/sampled scope.

One reference sample is reused for target comparisons. The query cap applies
separately to redundancy queries and each target comparison. All target IDs
remain in the result; unselected target rows keep marginal/status results but
their neighbor result is `not_sampled`. Insufficient candidates after identity
or cell exclusion yields `insufficient_reference`, not a fresh adaptive sample.
Sampling draws do not consume a generator's RNG.

Blocking distance calculations avoids an `N x N` allocation, but does not
remove the arithmetic cost. No dense million-world pairwise matrix or
automatic full-corpus eigendecomposition.

[Vendi](https://arxiv.org/abs/2210.02410) is a possible opt-in extension, not the
headline score: `exp(-sum(lambda * log(lambda)))` for eigenvalues of a trace-one
positive-semidefinite similarity matrix (`K / n` for unit-diagonal `K`).
Kernel, bandwidth, selected features, weights, and sample size define its
meaning. It is an effective diversity under
that similarity, not a model-independent count of distinct or informative
worlds. Equal-budget sampling and sensitivity checks would be mandatory.
Group exclusions apply to neighbor lookup, not by zeroing same-cell entries of
a Vendi kernel: such masking need not preserve positive semidefiniteness.

## Enriching the prior is a separate decision

Hold active-count distributions and graph edge budgets fixed while auditing
realized within-world patterns. A valid count range alone does not guarantee
that two generated corpora have the same realized count frequencies; show both.
Do not change graph topology or saturation families incidentally just to widen
one covariate pattern. Current controls already include:

- `n_*_active_range` for count distributions, and `edge_budget` for counts of
  arrows. An integer budget is an upper cap, not an exact count; `(k, k)` is
  exact when feasible, with the documented minimum-one `cy` constraint.
- `rw_treatment_std_range`, `treatment_hf_sigma_range`, and
  `treatment_pulse_*_range` for treatment variation and pulse texture.
- `covariate_hf_sigma_range` and `covariate_pulse_*_range` for covariate texture;
  `make_scm_prior` already enables diverse texture.
- `carryover_family_probs` and `saturation_family_probs` for per-treatment
  family mixtures, already supported within a single world.

Do not propose these as missing features. A mixture of whole `SCMPrior`
configurations is different and is not currently a generation API.
Adding any genuinely missing pattern family or a weighted configuration-mixture
sampler is separately scoped work, not hidden inside a diagnostic call.

**Provenance limit:** a generation cell fixes the graph, active nodes, discrete
mechanism assignments, walk smoothness, and texture-enable flags; draws resample continuous
parameters and realizations. The persisted corpus retains `g`, masks,
`cell_id`, and `carryover_family`, but not the discrete saturation-family IDs
or all texture settings (`slots.py`, `CORPUS_ARRAY_FIELDS`). Those missing
labels must be reported unavailable; they cannot be recovered reliably from
the observed series. Graphs and carryover labels also remain truth-only for
ordinary real-data comparisons, even though the generator knows them.

The analysis should explain gaps, not prescribe “more uniform is better.”
Repeated stochastic realizations can be useful. More fresh worlds at a fixed
training budget may or may not improve accuracy; assess held-out performance
by descriptor slices using cell-disjoint evaluation. Training loss, calibration,
compute budgets, and epoch schedules belong to the downstream learner.
Historical claims about training cost, 95th-percentile mismatch, or a particular
model's gains are motivation, not reverified evidence in this proposal.

## Baseline probe

A temporary script executed the existing API in the branch-local locked
environment: `make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1,
n_time_steps=32, n_cells=3, draws_per_cell=2, seed=606)`, then
`sample_prior_predictive` and `data_diagnostics(views=("levels",),
lags=(1, 4), keep_series=False)`.

Observed: six worlds; keys `C1, C2, C3, Z1, Z2, D1, B, Y`; group IDs
`[0, 0, 1, 1, 2, 2]`; per-world covariate means/stds accessible without raw
series retention; `json.dumps(report.summary(), allow_nan=False)` succeeded.
For the separate vector `[8, 9, 10, 11, 12]`, level-to-variation was
`7.071067811865475`, unchanged after multiplication by 1000, and
`77.78174593052022` after adding 100. This verifies the available primitive and
the scale-versus-offset distinction, not the proposed API or any training gain.

## Acceptance for a later implementation

- Analysis leaves source arrays, generator RNG streams, and corpus schema intact.
- Corpus and equivalent SCM/observed inputs agree on observable descriptors;
  unavailable truth is never fabricated.
- Positive unit changes preserve the relevant descriptors; offsets expose the
  intended change. Padding, zero-active roles, constants, all-zero series,
  short horizons, and unavailable provenance have explicit outcomes.
- Similar descriptors cannot be reported as byte-identical worlds; same-cell
  exclusion and source namespaces prevent identity leakage across shards.
- Targets outside a joint cloud despite ordinary marginals are distinguishable;
  target values cannot change fitted scales, bins, or reference membership.
- Reports retain finite/infinite/invalid denominators, sample provenance, and
  strict-JSON output. Large-corpus work is bounded and sampling is reproducible.

## Questions for discussion

1. Is `world_descriptors` + `corpus_diagnostics` the right separation, keeping
   `data_diagnostics` focused on within-world explanation?
2. Are permutation-invariant role summaries the right default, with named-node
   drill-down, rather than requiring aligned channel labels across datasets?
3. Should the first implementation prioritize observable target coverage and
   grouped repetition, leaving Vendi and prior-mixture generation separate?

Recommendation: yes to all three. Agree on the workflow and metric semantics
before committing to individual descriptor names or implementing it.
