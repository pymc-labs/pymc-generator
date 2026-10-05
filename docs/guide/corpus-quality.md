# Corpus quality and trajectory coverage

The [trajectory showcase notebook](../examples/trajectory-showcase.ipynb) generates
17 small, 104-week datasets and shows how to inspect them before training:

- **Eight mixed worlds:** four generation cells, two draws per cell, 15 treatment
  and 10 covariate slots, 5–15 active treatments and 3–10 active covariates.
  Each world has exactly five treatment → outcome and three covariate → outcome
  edges. All other edges are zero; the retained latent is disconnected.
- **Eight archetype worlds:** always-on spikes, periodic on/off, delayed start,
  ramp-up, decay to zero, level doubling, seasonal, and trend. Each is a small
  `SCM` with its own model-facing CSV and separate truth audit bundle.
- **One all-component world:** exact primitive-bound replay, same-input reusable
  template evaluation, and a fixed-input sampled oracle forward check. No MCMC
  or extra datasets are needed for those comparisons.

The mixed worlds use geometric carryover and Michaelis–Menten saturation,
log-uniform half-saturation scales, parameter-only reference-contribution priors,
and independently included trajectory components. The notebook exports a
schema-v5 `mixed-worlds.npz`, one CSV per mixed world, the single-world bundles,
and a `manifest.csv`. Outputs default to a new temporary directory; set
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
  --ExecutePreprocessor.timeout=600 --output-dir /tmp \
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
inventory, and seeded world/input selections.

In an editor, execute all cells tagged `report-setup`, `report-definitions`, or
`report-run`, in notebook order. Set `PYMC_GENERATOR_REPORT_PATH` to the
**absolute path** of the stored shard before starting the kernel. Skip generation
and atlas cells. The tagged path renders the seeded input-series sample and world
decompositions using only the loaded file, including texture-only archives without
schedule truth. Alternatively, this command starts a fresh kernel and writes the
tables and visual checks to an executed report notebook:

```bash
export PYMC_GENERATOR_REPORT_PATH=/absolute/path/to/mixed-worlds.npz
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

## Read the report

### Component flags versus visible trajectories

Component prevalence is the fraction of **active `(world, input)` pairs** whose
stored component flag is one. It excludes padding, includes off-weeks implicitly
as part of an input's schedule, and matches `diagnostics["trajectory"]`. Draws
from one cell share flags, so eight worlds here provide four structural draws,
not eight independent prevalence samples.

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
`(11,7)`, `(15,10)`. Every target gets one cell and two worlds. The other 84
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
