# Interpretation: bounded least-squares contribution recovery

**Report:** `studies/recovery/report-artifacts/bounded50-profile-v2/`

**Exact report run ID:** `041a2b4bfc17d53631ad524508fe0a6243c8bdf50df47ec3ee567a7c202562de`

This narrative concerns the existing 50-world constant-baseline study. Approval, if granted, applies only to this report identity; it is recorded separately from this text.

## Main finding

**Optimizer convergence was much more common than successful channel attribution.** Among 48 computationally evaluable worlds, only 13 recovered all individual media contributions, whereas 28 recovered the total media contribution. This distinguishes numerical completion from scientific recovery and motivates keeping both checks in the study runner.

Here “OLS” names the report's method slot; the procedure is **bounded estimated-transform nonlinear least squares**, not ordinary unconstrained linear regression. It estimates adstock and saturation transformations under generator-supported constraints and uses known zero/nonzero effect identities. It returns point estimates, not a posterior distribution.

## Results and denominators

The study contains 50 worlds with constant intrinsic baselines and 97 scored weeks (7–103). Two optimizer runs, zero-based world IDs 6 and 9, did not satisfy the computational acceptance rule and remain unresolved. They are not counted as attribution failures or dropped without disclosure.

The point-recovery allowance is the greater of 10% of the true channel's L2 norm and 1% of the true noiseless outcome's L2 norm. Joint recovery covers nonzero media/control effects but excludes the baseline; total media and baseline are evaluated separately.

| Estimand | Passes / evaluable worlds | Rate |
|---|---:|---:|
| Every individual media contribution | 13/48 | 27.1% |
| Every individual media and control contribution | 13/48 | 27.1% |
| Total media contribution trajectory | 28/48 | 58.3% |
| Baseline contribution | 26/48 | 54.2% |

Relative to all 50 attempted worlds, those are 13, 13, 28 and 26 established passes, respectively, with two unresolved worlds. The table conditions on computational validity; it is not an unconditional performance estimate for all generated worlds.

| Nonzero effects | Failed / evaluable | Failure rate | Unresolved |
|---|---:|---:|---:|
| Media | 122/211 | 57.8% | 25 |
| Controls | 15/135 | 11.1% | 13 |

There are 236 eligible media and 148 eligible control effects overall. Known-zero effects are not part of these denominators. Channels within worlds are dependent, and worlds with more channels contribute more observations to pooled rates.

## Interpretation

The optimizer often finds a solution satisfying its stopping rule without finding contribution trajectories close to truth. The result is compatible with weak separation of component effects and with limitations of a finite optimization procedure; **convergence is not a certificate of a global optimum**. These results alone cannot distinguish structural ambiguity, weak finite-sample information, objective geometry and remaining optimization error.

Total-media point recovery is more frequent than recovery of every component, illustrating why an aggregate contribution estimate should not be advertised as channel-level attribution. Conversely, a failed all-channel gate need not mean that every channel or the aggregate estimate failed.

There are no posterior intervals in this point-estimation report, so it cannot establish uncertainty calibration, posterior concentration or credible-interval coverage. It also does not evaluate held-out predictions or interventions. The generator's known graph and supported transformations make this an oracle-information benchmark rather than a realistic variable-selection task.

The TV-MCMC report is useful complementary evidence but **not a controlled algorithm comparison**: it uses a changed baseline/outcome regime and a different estimation objective. These two headline rates do not support saying that one inference method is better than the other. A method comparison would require the same generated worlds, target, information and scoring conditions, with comparable computational qualification.

## Implications and provenance

Keep optimizer acceptance, channel recovery and aggregate recovery separate. Preserve unresolved denominators when summarizing future studies. Test generator or fitting changes through prespecified matched comparisons rather than interpreting these exploratory failure counts as a causal explanation.

This legacy evidence has manifest/input-hash compatibility only; unavailable original per-output/source hashes cannot be certified retrospectively. The report seals its current evidence without upgrading that historical provenance. The four-world software smoke is not additional scientific evidence, and no new fitting was performed to prepare this interpretation.
