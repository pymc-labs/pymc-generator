# Corpus quality and trajectory coverage

The [trajectory showcase notebook](../examples/trajectory-showcase.ipynb) generates
100 primary mixed simulations after inspecting an eight-world baseline, all
104 weeks long. Its export manifest contains 117 source-qualified datasets:

- **100 primary mixed worlds:** 50 generation cells, two draws per cell, 15
  treatment and 10 covariate slots, 5–15 active treatments and 3–10 active
  covariates. Each world has exactly five treatment → outcome and three
  covariate → outcome edges. All other edges are zero; the retained latent
  is disconnected.
- **Eight frozen baseline worlds:** four cells × two draws with the same priors
  and count mix. Parameter and zero-period diagnostics run on these first,
  before the 100-world cohort is generated.
- **Eight archetype worlds:** always-on spikes, periodic on/off, delayed start,
  ramp-up, decay to zero, level doubling, seasonal, and trend. Each is a small
  `SCM` with its own model-facing CSV and separate truth audit bundle.
- **One all-component world:** exact primitive-bound replay, same-input reusable
  template evaluation, and a fixed-input sampled oracle forward check. No MCMC
  or extra datasets are needed for those comparisons.

The mixed worlds use geometric carryover and Michaelis–Menten saturation,
log-uniform half-saturation scales, parameter-only reference-contribution priors,
and independently included trajectory components. The notebook exports a
schema-v5 `mixed-worlds.npz` with **100 worlds**, a separate
`baseline-8-worlds.npz`, one model-facing CSV for each of those 108 worlds,
the eight atlas and one replay bundles, and a `manifest.csv`. Dataset IDs,
source/world pairs, and CSV paths are unique; `truth_source` points to the
appropriate archive or audit bundle, never a model feature.
Outputs default to a new temporary directory; set
`PYMC_GENERATOR_SHOWCASE_DIR` to retain them in a directory of your choice.

## Run the showcase

Use Python 3.13+ and the checkout containing the notebook. It is stacked on
[PR #38](https://github.com/pymc-labs/pymc-generator/pull/38), whose base revision
is `ccbfa7ad51c90bacfab6e612bbc061024926af48`. An older released wheel can share
version `0.0.2` without these unmerged features; use this checkout's code and
locked dependencies, not the version number alone.

```bash
uv sync --locked --extra docs
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name pymc-generator
.venv/bin/python -m nbconvert --execute --to html \
  --ExecutePreprocessor.timeout=1800 --output-dir /tmp \
  docs/examples/trajectory-showcase.ipynb
```

The setup cell prints the kernel interpreter, imported package location, numerical
versions, and output directory. Choose that local kernel in your notebook editor.
The guide navigation also links directly to the executable notebook. A strict
site build runs it through `mkdocs-jupyter`; the plugin's execution cache can
reuse previous output, so use an uncached build when verifying changed library
code, rather than treating a cache hit as execution evidence.

## Report on an existing shard without generating data

The notebook's `quality_report(path, seed=31, vif_threshold=10.0,
max_bad_share=0.25, design_view="differences")` reloads the file and has no
prior/configuration or generation-state dependency. It returns descriptor rows
and distributions, within-world diagnostics, mask-recomputed component prevalence
and count-grid coverage, a per-world design screen, trajectory measurements and
inventory, a parameter catalog with applicable values/equal-world weights,
zero-series/episode/world/balance tables, and seeded world/input selections.

In an editor, execute all cells tagged `report-setup`, `report-definitions`, or
`report-run`, in notebook order. Set `PYMC_GENERATOR_REPORT_PATH` to the
**absolute path** of the stored shard before starting the kernel. Skip generation
and atlas cells. The tagged path renders the seeded input-series sample and world
decompositions using only the loaded file, including texture-only archives without
schedule truth. Alternatively, this command starts a fresh kernel and writes the
tables and visual checks to an executed report notebook:

```bash
export PYMC_GENERATOR_REPORT_PATH=/absolute/path/to/mixed-worlds.npz
# Optional: compare a second saved cohort without generating either one.
# export PYMC_GENERATOR_BASELINE_PATH=/absolute/path/to/baseline-8-worlds.npz
.venv/bin/python - <<'PY'
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path("docs/examples/trajectory-showcase.ipynb")
notebook = nbformat.read(path, as_version=4)
tags = {"report-setup", "report-definitions", "report-run"}
notebook.cells = [
    cell for cell in notebook.cells
    if tags.intersection(cell.metadata.get("tags", []))
]
NotebookClient(notebook, kernel_name="pymc-generator", timeout=600).execute()
for cell in notebook.cells:
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            print(output.text)
nbformat.write(notebook, "/tmp/corpus-quality-report.ipynb")
PY
```

A given file, seed, and threshold policy reproduces numeric/status tables and
selected IDs in the same numerical environment. Runtime and temporary output
paths are not part of that report-only contract. Separately, same-seed,
same-configuration generation in the locked environment persists byte-identical
NPZ shards because timing is excluded; see
[persistence](corpus.md#diagnostics-that-do-not-go-into-the-file).
Cross-platform bitwise results are not promised.

Optional trajectory truth is unavailable in texture-only archives, not inferred
from observed curves. Without stored count-allocation metadata, the report can
show gaps within observed count bounds but cannot recover the intended grid or
weights. Raw series still plot without trajectory truth; the schedule panel
is explicitly marked unavailable rather than filled with an invented gate.

### Parameter distributions across worlds

`parameter_report` catalogs persisted carryover, response, reference and
trajectory parameters. It excludes inactive padding, unused response/carryover
families and unselected trajectory components, but retains genuine enabled zeros,
negative values, week zero and every jump-event axis. Drawn-but-unused family
values, missing optional truth, absent events and unpersisted walk/noise/edge
primitives are separate statuses, not fabricated zeros.

Every applicable world has equal total numeric weight, split among its eligible
inputs/events. Structural choices repeat within a generation cell; their
descriptive plots report the cell count and do not claim independent-world
prior-fit evidence. The 100-world cohort has 50 structural draws, not 100.

`parameter_bin_report` uses shared bins when comparing cohorts. Recorded uniform
and log-uniform supports get justified nominal mass references. Matching recorded
supports and laws use equal-prior-mass bins: for MM log-uniform scales, equal
**log-prior-mass**, not a flat raw-value target. Different supports or laws use a
shared domain with cohort-specific nominal masses. Reference-derived `beta`,
measured means and other nonlinear anchors have no uniform promise.
Signed `rho_zy` retains a uniform nominal law when it is a uniform recorded
reference target divided by a fixed recorded scale.

High-frequency sigma and pulse amplitude are also effective magnitudes: their
primitive multiplier is scaled by a treatment-drive anchor or covariate-walk
standard deviation. They are reported in raw-input units as derived, nonuniform
quantities; the primitive multiplier is not separately persisted.

The report distinguishes a configured point mass supported by metadata from an
observed constant or singleton. For example, this recipe uses a 52-week seasonal
period, but its configured bounds are not in the archive; the portable report
therefore labels its support unknown and supplies no prior-fit score. Unknown
supports retain the **combined observed domain**, unioned with recorded supports
in mixed-support comparisons; this cannot assess unexplored prior tails. The
catalog also records float32 versus float64 storage precision.

Per-cell conditional low/width labels can be float32 while a Hill parameter is
float64. Support checks account for both precisions independently at each
endpoint. Conditional nominal references use those rounded stored intervals;
the original sampling intervals are unavailable, so these are approximate
references rather than exact reconstruction.

Coverage, empty-bin counts, maximum weighted bin mass and deviation from a
justified nominal reference are descriptive. The worlds have survived the
generator's realism/overflow filters; these plots are not unconditioned iid
goodness-of-fit tests. More worlds do not automatically improve coverage or
balance, and no accepted worlds are removed to make the figures flatter.

### Exact zero periods and population balance

`zero_period_report` measures exact `raw == 0` in each active treatment channel
and signed control over the full reported window. Negative controls and tiny
nonzero values are not zeros; inactive slots are not series. A period is a
maximal contiguous zero run, with inclusive zero-based endpoints and an observed
length in weeks. Runs touching either window edge are marked because their full
duration may extend beyond the observed 104 weeks. This is not the train/query
split or carryover burn-in window.

The report and plots keep four populations separate:

- **Series:** no-zero versus any-zero balance; all-zero is a subset of any-zero.
  Both pooled counts and mean within-world shares are shown so many-input
  worlds cannot silently dominate the latter.
- **Worlds:** none, mixed, or all active inputs affected, plus the world-any
  share. “All affected” does not mean every observation is zero.
- **Weeks:** the fraction exactly zero in each active series.
- **Episodes:** actual run counts per series, including zero, and observed run
  lengths. No-zero series have no invented zero-length episode; duration
  summaries are unavailable when there are no episodes.

Empty active populations have unavailable rates, not fake 0% coverage.
Episode-free populations still retain their series/world denominators.
Eight-versus-100 plots use common normalized fraction, integer-count and week
bins, with boundary-touching length mass distinguished from interior durations.

The **50% line is a diagnostic reference, not a prior guarantee**. Under the
unchanged composable gate probabilities, the nominal chance of carrying none of
onset/offset/flighting is 61.2% for treatments and 68.85% for controls. Those are
flag-law probabilities, not exact realized zero-free rates; schedules and
acceptance matter. World-any prevalence has a different denominator and no 50%
expectation. The notebook measures this imbalance rather than changing priors
or resampling to enforce a favorable split.

The full tour runs these diagnostics on eight worlds **first**, then changes
only `n_cells` from four to fifty. Both saved cohorts keep generation seed 29;
the changed allocation means the 100 worlds are not a prefix extension or a
paired controlled experiment. A file-only comparison sets the optional
`PYMC_GENERATOR_BASELINE_PATH` alongside the primary report path and uses the
same analysis and shared bins, without either generation phase.

## Read the report

### Component flags versus visible trajectories

Component prevalence is the fraction of **active `(world, input)` pairs** whose
stored component flag is one. It excludes padding, includes off-weeks implicitly
as part of an input's schedule, and matches `diagnostics["trajectory"]`. Draws
from one cell share flags: eight baseline worlds provide four structural draws,
and the primary 100 worlds provide fifty—not 100 independent prevalence samples.

The visible-input inventory measures exact-zero fraction, first/last nonzero
week, adjacent off→on/on→off transitions, nonzero-run count, and signed/magnitude
slopes on nonzero weeks. Time uses original reported-week positions scaled to
`[0, 1]`; slopes are normalized by on-week RMS. Signed controls retain their sign;
absolute magnitude distinguishes a negative value approaching zero from one
becoming more negative. Fewer than two nonzero weeks gives an undefined slope.

The example's ramp/decay screen uses magnitude slopes `>= 0.25` after a delayed
start and `<= -0.25` before a terminal shutdown. These are descriptive heuristics,
not monotonicity or component-presence proofs. The exact-doubling row instead
reads treatment jump factors from schedule truth. **Absent in this shard** does
not mean unsupported: the doubling archetype deliberately fills that gap.

### Count-grid coverage

World and distinct-cell counts are recalculated from `treatment_active_mask`,
`covariate_active_mask`, and `cell_id`, then compared with the stored coverage.
The 11×8 grid targets four equally weighted combinations: `(5,3)`, `(8,5)`,
`(11,7)`, `(15,10)`. Each target gets one cell/two baseline worlds, then
12–13 cells/24–26 worlds in the primary cohort. The other 84
combinations have zero weight and are marked as exclusions, not accidental
missing coverage. To target the whole grid uniformly, remove the weights and
increase `n_cells` to at least 88.

### Empirical identifiability screen

The screen uses `data_diagnostics(...)[view].vif["observed"]`: active raw treatment
and covariate predictors, centered and unit-column-normalized. It flags a world
when **rank < active predictor count** or any valid VIF is **>= 10** (including
infinity). Constants reduce rank. Ineligible padded predictors do not count;
otherwise missing eligible results or too few observations make the screen
unassessed, never an automatic pass.

Rank uses the library's SVD tolerance
`max(eps64, eps64 * max(n_observations, n_varying_columns) * largest_singular_value)`.
VIF uses per-target residual projection onto the other predictors; it is not
inferred from rank. Condition numbers are reported without an extra threshold.

Defaults are notebook policy, not library defaults: `vif_threshold=10.0` and
`max_bad_share=0.25`. A nonempty, completely assessed selection passes when the
flagged-world fraction is **<= 0.25**. The bad-share limit must remain below
0.5 to require a minority. Differences are primary; levels are shown separately
because independent random walks can have high level VIF.

This diagnoses **finite-sample design redundancy**, not causal identifiability,
nonlinear-response recovery, or training utility. Read the signal gate and outcome
contribution shares alongside it. No worlds are silently filtered to improve a
PASS/FAIL result.

### Diversity and hand-picked datasets

`world_descriptors` and `summarize_descriptors` retain finite, infinite, undefined,
and ineligible counts. `bin_counts` counts worlds and distinct generation groups.
`nearest_worlds(..., exclude_same_group=True)` excludes sibling draws;
`compare_descriptors` places atlas worlds against the mixed reference. Descriptor
similarity does not establish duplication or learning value.

Edit `MANUAL_WORLD_IDS` and `MANUAL_INPUTS` to inspect existing mixed-shard worlds
and active input slots. Those IDs are independent of the seeded visual sample.
The notebook converts the explicit ID list to a boolean `worlds=` mask, avoiding
integer-mask ambiguity even for a two-world shard. Tables use archive order,
world plots use your list order, and empty input picks permit world-only inspection.
Selectors preserve source positions rather than slicing arrays while leaving
stale diagnostics behind. For a singleton atlas
source use `worlds=None` or a slice, and qualify world ID zero by its source name.
The manifest maps each dataset to its model-facing CSV; privileged truth remains
in the NPZ or the separate audit files.

See also [input trajectories](trajectories.md), [generating a corpus](corpus.md),
[within-world diagnostics](data-diagnostics.md), and
[comparing worlds](world-descriptors.md).
