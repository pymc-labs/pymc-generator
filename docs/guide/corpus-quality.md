# Corpus quality and variety

The trajectory showcase notebook, `docs/examples/dev/trajectory-showcase.ipynb`
in the repository, is a tool for one question: **with a given set of parameter
ranges, how varied is a cohort of N generated worlds, and what is that variety
made of?** It generates N worlds (10 by default), saves them with the
configuration that produced them, and then draws every figure from the saved
file alone. It is a development notebook, so this site does not render it: run
it from a checkout as its first cells describe.

| Figure | Question it answers |
| --- | --- |
| Count grid | How many channels and controls do the worlds have, and is every count used equally? |
| Parameter grid | Is every generation parameter spread over its configured range the way its prior says? |
| Trend | How are trend parameters split into flat / up / down, and which direction do the series actually take? |
| Zero weeks | How many weeks of each input are exactly zero, from none to 100? |
| Outcome shares | How much of the outcome comes from channels, controls and the baseline, world by world? |
| Example worlds | What do the extreme and a seeded world look like, input by input and part by part? |
| Quality checks | Component prevalence, count-grid coverage, identifiability screen, descriptors, acceptance and signal checks. |

All figures are plot-first; the notebook prints no large tables.

## The cohort design

- **Slots and counts.** 10 channel slots and 10 control slots; each world has
  1–10 active channels and 1–10 active controls. `active_count_allocation="stratified"`
  with an integer weight matrix `W` and `n_cells = N`, `draws_per_cell = 1`, so no
  two worlds share a structure. `W` contains `N // 100` copies of the full grid
  plus `(N % 100) / 10` disjoint cyclic shifts of one permutation: every channel
  count and every control count appears equally often, and up to N = 100 no pair
  repeats. When `ΣW = n_cells` the allocation targets equal `W` exactly. The
  permutation is searched for a rank correlation between the two counts, over
  the N realized pairs, of at most 0.1 (the closest one found otherwise); the
  notebook prints it. N must be a positive multiple of 10.
- **Edges.** Every active channel and every active control has a direct edge to
  the outcome (`cy` and `zy` budgets equal to the slot counts, which clamp to the
  active counts). All other edge types are zero; the latent factor is
  disconnected.
- **Mechanisms.** Geometric carryover (decay `U(0.05, 0.95)`) and
  Michaelis–Menten saturation (κ multiplier log-uniform on `[0.05, 20]`).
- **Effects.** Contributions are not targeted. Each channel and each control
  draws its contribution at its reference level from `U(0, 3)`; controls use a
  reference input of 1, so their effects are non-negative while their series stay
  signed. The baseline level is `U(0.05, 8)`, floored at 0.
- **Trajectories.** Every input carries a trend (`U(−1, 1)`); seasonality, level
  jumps and pulses each with probability 0.5; weekly noise always; flighting with
  probability 10/11, period `DiscreteUniform(14, 26)` and duty `U(0.03, 0.99)`;
  no onset or offset gates.
- **Walk parameters.** `record_walk_parameters=True` persists the walk means,
  volatilities and smoothness and the outcome-noise scale (see
  [the corpus schema](corpus.md#recorded-walk-parameters)), so every generation
  parameter can be plotted from the file. Smoothness is drawn from `Beta(1, 1)`.

## Reading the references

Each reference shows what the configured law expects **at the realized number of
values**: black dots for the expected count per bin and black whiskers for a
family-wise envelope (Šidák-corrected exact binomial quantiles across the panel's
bins); bars that leave it are outlined in red. When the generator follows its
law, a whole panel stays inside its envelope at least 90% of the time — more at
small n, because the exact quantiles are conservative — so with many panels a
few exceed it by chance. Titles report `k of m bins outside`, and the parameter
grid prints the expected number of panels outside: the sum of each panel's
chance of leaving its envelope (exact for the independent component flags,
otherwise simulated with a fixed seed: one multinomial over the pooled values, or
one per event slot for level-jump weeks and shock starts). References describe
the law **before** the realism filter; the report prints the share of candidates
that filter rejected.
Quantities that are deterministic functions of others (`beta`, absolute
volatilities, effective texture magnitudes) are labelled derived and carry no
reference. Structural flags, smoothness and counts are drawn once per
generation cell; their references count one row per cell, and are omitted when
`draws_per_cell > 1`.

## Trend: parameter versus realized direction

The trend component adds `change · t / (T − 1)` to an input's level (log level for
channels, level for controls). The parameter panel classifies `change` into down
(`< −1/3`), flat and up (`> 1/3`); under `U(−1, 1)` each third is equally likely.
The realized panel fits the slope of each series over its fit weeks — on-weeks
only, and for channels also outside treatment shocks — and classifies
`slope · (T − 1)` with the same thresholds. Random-walk drift, seasonality and
level jumps also move the realized direction; narrower walk volatilities
(`rw_treatment_std_range`, `rw_std_sigma`) bring the two closer.

## Zero weeks

A week is zero when the input is exactly `0.0`. Exact zeros come from schedule
gates and, for channels, from held treatment shocks (level 0 by default). This
design uses only flighting, and every active input satisfies
`zeros = (1 − activity).sum()` with `activity` recomputed from its saved gate
fields (gates are applied only to inputs that carry them). Categories are
`0`, `1–10`, …, `91–100`, `101–104` for 104 weeks. The reference is the exact law
of the configured flighting: period × duty sub-intervals through
`trajectories.flighting_on_weeks` × phase, mixed with the inputs that carry no
flighting. For the default ranges every category from 0 to 100 has 8.7–9.5%;
`101–104` is structurally empty because every flight keeps at least one on-week
per period. The reference is drawn only when the configuration has no onset or
offset gate.

## Outcome shares

For each world, over the reported weeks:

- channels = Σ `treatment_contribution_raw` + Σ `indirect_effects`;
- controls = Σ `covariate_contribution`;
- baseline = Σ `baseline_intrinsic` + Σ `latent_unobserved_contribution` + Σ `outcome_noise`;

each divided by Σ `outcome_raw`. The three add up to 1 up to float32 storage
rounding, which the report asserts. Do not use the library's `baseline`
(`baseline_raw` already contains the controls) or `treatment_share` (the mix
between channels). Because the shares sum to 100%, they cannot all be uniform:
if every composition were equally likely, each single share would follow
`Beta(1, 2)` — falling linearly from 0% to 100%. The report draws that ideal as a
guide, not as the generator's law, together with the minimum and maximum it would
give at this N. The triangle places each world by its three shares; worlds with a
negative part (signed control series) are marked separately.

## Report robustness

| Check | Needs | Without it |
| --- | --- | --- |
| Count grid, realized rank correlation | active counts | always |
| Design assertions | the `.config.json` written by the notebook | skipped |
| Configured-law references | `.config.json` | histogram only, "configured law unavailable" |
| Walk and noise panels | walk-parameter fields | "not recorded" |
| Component prevalence, inclusion panels | trajectory block | "not recorded" |
| Zero identity | trajectory block | observed zeros only |
| Zero reference law | `.config.json` without onset/offset (channels: no shocks) | "configured law unavailable" |
| Trend parameter view | trajectory block (written whenever a schedule component can occur) | "no trend parameters" |
| Realized trend bars, shares, identifiability, descriptors, example worlds, CSV exports | core arrays | always |

## Report on an existing shard

Every figure is a function of the saved file. The `.config.json` holds the
effective configuration (`dataclasses.asdict(cfg)`), the notebook knobs and the
SHA-256 of the `.npz`. Loading checks the hash (`SidecarMismatchError`) and
compares the configuration with what the shard records about itself
(`SidecarConfigMismatchError`). Execute the cells tagged `report-setup`,
`report-definitions` and `report-run`, in order, with
`PYMC_GENERATOR_REPORT_PATH` set to the absolute path of the shard for that one
command (a full run that still sees the variable reports that shard instead of
the new one, and says so). The report also exports the shard's model-ready CSVs
and `manifest.csv` (which names the shard by absolute path and SHA-256) to
`$PYMC_GENERATOR_SHOWCASE_DIR/run-*/datasets/<shard name>/` (under a new
temporary directory when the variable is unset):

```bash
PYMC_GENERATOR_REPORT_PATH=/absolute/path/to/variety-10-worlds.npz \
PYMC_GENERATOR_SHOWCASE_DIR=/absolute/path/to/report-output \
.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient

notebook = nbformat.read("docs/examples/dev/trajectory-showcase.ipynb", as_version=4)
tags = {"report-setup", "report-definitions", "report-run"}
notebook.cells = [c for c in notebook.cells if tags.intersection(c.metadata.get("tags", []))]
NotebookClient(notebook, kernel_name="pymc-generator", timeout=600).execute()
nbformat.write(notebook, "/tmp/corpus-quality-report.ipynb")
PY
```

A given file reproduces the same figures and printed values in the same
numerical environment. Same-seed, same-configuration generation in the locked
environment writes byte-identical shards because timing is excluded; see
[persistence](corpus.md#diagnostics-that-do-not-go-into-the-file).
Cross-platform bitwise results are not promised.

## Empirical identifiability screen

The screen uses `data_diagnostics(...)[view].vif["observed"]`: active raw
treatment and covariate predictors, centered and unit-column-normalized. It
flags a world when **rank < active predictor count** or any valid VIF is
**≥ 10** (including infinity). Constants reduce rank. Ineligible padded
predictors do not count; missing eligible results or too few observations make
the screen unassessed, never an automatic pass. Rank uses the library's SVD
tolerance `max(eps64, eps64 · max(n_observations, n_varying_columns) ·
largest_singular_value)`. Differences are the primary view; independent random
walks can have high level VIF. The cohort passes when the flagged share is
**≤ 0.25** (notebook policy, not a library default; it must stay below 0.5 to
require a minority). This diagnoses finite-sample design redundancy, not causal
identifiability, nonlinear-response recovery or training utility. No world is
filtered to improve the result.

See also [input trajectories](trajectories.md), [generating a corpus](corpus.md),
[within-world diagnostics](data-diagnostics.md) and
[comparing worlds](world-descriptors.md).
