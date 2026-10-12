# Interpretation: media attribution with a varying baseline

**Report:** `studies/recovery/report-artifacts/tv50-profile-v2/`

**Exact report run ID:** `f2dccfb682b95c9e8da4ffdc0d3f8c5f4134098468d54ef4f8c8e974cfbbffb3`

This narrative describes the existing 50-world TV-MCMC study, not the four-world software smoke test. Approval, if granted, applies only to this report identity; it is recorded separately from this text.

## Main finding

**Healthy MCMC did not usually recover every individual media contribution, although total media contribution was recovered much more often.** Observation noise was small relative to aggregate sales, but often comparable to the temporal variation of a single media contribution. The results therefore support using this generator configuration as a challenging attribution benchmark—not treating its attribution targets as uniformly easy because aggregate noise is low.

This is a statement about recovery under this experiment's information, priors, sample length and error allowance. It is not proof of structural non-identifiability, a general limitation of Bayesian inference, or validation that the synthetic inputs resemble real spending.

## What was evaluated

The study contains 50 synthetic worlds, 104 reported weeks and 97 scored weeks (7–103). Earlier inputs supply carryover history. There are 236 nonzero media effects and 148 nonzero control effects. Media inputs combine a smoothed random walk with optional high-frequency variation, pulses, activity gates, level changes, seasonality and trends. Contributions use geometric adstock followed by Michaelis–Menten saturation. The intrinsic baseline varies over time; observation noise is a separate Gaussian component.

This is an **oracle-information benchmark**: fitting knows the zero/nonzero effect graph and generator reference anchors/supports. It does not infer the graph from data, and realized coefficients or baseline paths are not supplied as initialization. Performance should not be advertised as expected performance on an unprivileged real-world modeling task.

A channel passes when the L2 error of its estimated contribution trajectory is no larger than the greater of 10% of that channel's true L2 norm and 1% of the noiseless outcome's L2 norm. Thus this is not a pure 10% relative-error criterion; small effects can be judged against the whole-outcome floor. Joint recovery requires every nonzero media/control effect to pass, excluding the baseline. The total-media sum and baseline path are separate estimands with their own gates.

## Results

All 50 fits passed the prespecified MCMC diagnostic screen: no divergences or maximum-depth hits, maximum recorded R-hat 1.0069, minimum bulk ESS 1,062, minimum tail ESS 1,168 and minimum BFMI 0.624. That makes their recovery scores evaluable; it does not guarantee exploration of every posterior mode or accurate attribution.

| Estimand | Point-recovery passes |
|---|---:|
| Every individual media contribution in a world | 11/50 (22%) |
| Every individual media and control contribution | 9/50 (18%) |
| Total media contribution trajectory | 36/50 (72%) |
| Baseline contribution trajectory | 36/50 (72%) |

At channel level, **144/236 media contributions failed (61.0%)**, compared with **19/148 controls (12.8%)**. Channels from the same world are dependent; these are descriptive pooled counts, not 384 independent experiments. Larger worlds receive more weight in pooled channel summaries, and an all-channel world gate becomes harder to satisfy as the number of effects increases.

The better total-media point recovery is consistent with uncertainty in allocating effects across components rather than uniformly poor estimation of the aggregate media contribution. It does **not** establish narrow posterior uncertainty for that total, nor demonstrate that every individual failure is merely a compensating allocation error. The report's 95% intervals are pointwise intervals, not simultaneous trajectory guarantees or held-out predictive calibration.

## What distinguishes passing and failing media channels?

These descriptors use the **true generated trajectories and parameters**, not estimates from the failed fits. CV is temporal standard deviation divided by the absolute temporal mean; saturation is the realized unit response after adstock, averaged over the scored window.

| Median true descriptor | Passing channels (92) | Failing channels (144) |
|---|---:|---:|
| Mean saturation | 41.7% | 51.3% |
| Adstock decay alpha | 0.451 | 0.547 |
| Asymptotic beta | 2.41 | 2.63 |
| Raw-spend CV | 71.6% | 58.2% |
| Contribution CV | 34.1% | 22.9% |
| Contribution temporal SD / observation-noise SD | 1.57 | 0.78 |

Observation-noise SD divided by mean observed sales has a median of **1.86%**, with a middle 50% of **1.09–3.48%**. This aggregate ratio alone is a poor difficulty indicator: an individual contribution's variation can be smaller than the noise even when total sales are measured relatively cleanly.

The clearest descriptive contrast in this table is the variation remaining in the contribution relative to noise. Failing channels also tend to have less spend variation, higher adstock decay and greater saturation. The two groups' median beta values are close; this does not rule out differences elsewhere in their distributions. These features are related to each other and to world-level conditions; the table does not identify their independent or causal effects. Changing one parameter does not guarantee improved recovery.

## Pattern-specific failures

Each numerator below counts failures of **the channel carrying the pattern**, not failures elsewhere in a world containing that pattern. Only nonzero effects are eligible; all 50 MCMC worlds are evaluable here.

| Assigned media pattern | Failed / eligible | Failure rate |
|---|---:|---:|
| Upward trend | 19/35 | 54% |
| Downward trend | 19/32 | 59% |
| Seasonal | 42/73 | 58% |
| On/off flighting | 14/45 | 31% |
| Delayed start | 18/37 | 49% |
| Early end | 11/17 | 65% |
| Level jump | 19/34 | 56% |
| High-frequency component | 130/214 | 61% |
| Pulses | 93/156 | 60% |

Patterns overlap, so rows are not additive or mutually exclusive comparisons. Assignment does not imply that the component dominates the final observed input. Flighting has the lowest observed failure fraction in this sample, but this is not a controlled test of flighting: amplitude, adstock, saturation, other assigned patterns and world context differ too. The early-end group is small. In particular, **upward-trending media are not uniformly unrecoverable**.

## Implications for generator development

1. Assess generated tasks using transformed contribution variation relative to noise, not only raw-spend CV or noise as a percentage of total sales.
2. Preserve separate channel-level and aggregate estimands: a useful total-media estimate is not a substitute for accurate channel attribution.
3. Evaluate proposed generator changes with matched experiments and fresh recovery results. Descriptive associations do not justify automatically increasing variability or replacing innovation distributions.
4. Retain hard cases and report their prevalence unless the intended task population explicitly excludes them. A difficult synthetic case is not, by itself, unrealistic.

## Limits and provenance

This is one exploratory 50-world configuration, not a calibrated distribution of real datasets. There is no real-data reference here establishing that the noise, schedules, smoothing or response shapes are empirically representative. The signed baseline has no nonnegative floor; its interpretation should follow that generator definition rather than assume a strictly positive autonomous sales component.

Do not directly compare these rates with the constant-baseline bounded-OLS report as an algorithm ranking: the outcome/baseline regime and estimation objective differ. The short four-world smoke fits validate software execution only and are not additional evidence for these scientific conclusions.

The saved study has **legacy manifest/input-hash compatibility only**; original per-output and historical source hashes are unavailable. The new report binds the evidence it reads, but cannot retroactively certify the old fitting process. Conclusions must be reconsidered after a dataset/configuration change rather than copied to a new run.
