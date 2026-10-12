# Configurable native recovery pipeline

This is an **artifact-free, narrow direct-effect study recipe**, not a universal DGP/model plug-in. A fresh user's *full explicit prior* controls the supported prior family. Original 50-world archives, historical notebooks, paired-baseline interventions and smoothing replay are **not shipped or read**. Replaying their exact arrays/posteriors is not promised. No stage runs on import or during report/Run All. Short examples demonstrate plumbing, **not** posterior convergence or scientific identifiability.

## Setup (repository root, Python 3.13.12)

```sh
uv venv studies/recovery/.venv --python 3.13.12
uv pip install --python studies/recovery/.venv/bin/python -r studies/recovery/requirements-report.txt -e .
studies/recovery/.venv/bin/python -m ipykernel install --prefix studies/recovery/.venv --name recovery-pipeline --display-name 'Recovery isolated Python'
export JUPYTER_PATH="$(pwd)/studies/recovery/.venv/share/jupyter"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

The complete `example_{constant,varying}_prior.json` files contain **every SCMPrior field** and are resolved relative to their TOML, then embedded (not just path-referenced) in stage/report identities. Edit/copy one with its TOML to vary dimensions (at least one active treatment AND covariate, even worlds/two generator cells per batch), feasible direct cy/zy effect counts, input textures, walk/noise/reference priors and MM/carryover ranges **without editing pipeline code**. Config preflight rejects unknown/missing fields, indirect/latent graphs, changed transform family, unknown anchors, absolute noise, confounding, floor/censoring, prior conditioning, shocks, unsupported baseline/OLS combinations and incoherent horizon/window/burn-in. `pg.SCMPrior.validate()` is also run before stage writes. Parameter priors used for inference are taken from the exact resolved prior, never realized true parameter values. Some full-prior generator administrative fields are explicit for reproducible generator semantics; do not assume a changed family is supported just because the underlying package accepts it.

## Fresh native workflow

```sh
P=studies/recovery/.venv/bin/python
CFG=studies/recovery/example_constant.toml
ROOT=studies/recovery/run-artifacts/constant-new
$P -m studies.recovery generate --config "$CFG" --out "$ROOT"
$P -m studies.recovery fit --config "$CFG" --out "$ROOT" --method ols
$P -m studies.recovery fit --config "$CFG" --out "$ROOT" --method advi
$P -m studies.recovery fit --config "$CFG" --out "$ROOT" --method mcmc
$P -m studies.recovery analyze --config "$CFG" --saved "$ROOT"
$P -m studies.recovery report --config "$CFG" --saved "$ROOT" --out studies/recovery/run-artifacts/constant-report
$P -m studies.recovery.execute_report --report studies/recovery/run-artifacts/constant-report --out studies/recovery/run-artifacts/constant-report-executed.ipynb
```

For varying baseline use `example_varying.toml`, a distinct NEW root and **only** ADVI/MCMC. This independently generates varying baselines; it does not couple them to the constant sample. CLI `fit --method ols` rejects varying before fitting. `analyze` runs the same **current-identity** reader as report and returns status counts and per-pattern channel denominators; it never fits. `report --report-method ols` (or another nonempty enabled subset) requires only that already-completed method for *every* world; without selection every enabled method and every ADVI seed is mandatory. Fit identities always use the full original configuration, even for selected reports. Missing, corrupted or stale stage receipts reject; generated/fit stages resume complete checkpoints without rerunning inference, but changed config/source requires a NEW root. Report root must be new; the executed `.ipynb` is an external derived copy and cannot overwrite source or sealed report. Report portability requires moving its relative source and report together.

## Supported family, inputs and adaptation boundary

- **Observed data:** per-world, two-cell `dataset.npz` with full reported weekly input paths and observed `outcome_raw`. `score_weeks=[l_max-1, n_time_steps-1]`, `l_max=8`, `burn_in>=8`; fit histories include the full preceding input rows. Generator `outcome_noise` is distinct from the intrinsic baseline. The noiseless target equals baseline + every direct treatment/covariate contribution (within floating tolerance).
- **Oracle side information:** `treatment_active_mask`, `covariate_active_mask`, direct `cy`/`zy` graph masks and parameter-only `mechanism_saturation_scale` reference anchors are supplied to both fits. This is **not** graph discovery or a fit from just raw observed inputs. Shape, mask, effect support and positive scales are checked. The full prior file holds coefficient/reference bounds and widths. Any new corpus adapter must provide these fields without leaking realized coefficients/baselines/noise into fitting.
- **Methods:** `ols` is bounded nonlinear estimated-transform least squares (not unconstrained linear regression), requiring constant intrinsic baseline, two starts by example, generator-support coefficient/alpha/log-MM bounds. Full-rank ADVI uses all configured seeds; `finite_only_not_converged` requires BOTH explicit `loss_finite: true` and `posterior_finite: true` in the sealed fit report plus finite selected component draws. False/missing flags or nonfinite draws are **unresolved**, with null point recovery and unresolved pattern-bearing eligible channels; nonfinite fitted trajectories/intervals are unavailable, never fabricated. Even a finite-only result is **not** posterior convergence or calibrated intervals. Nutpie MCMC uses `draw_diag` and checks divergence/maxdepth, rank R-hat, bulk/tail ESS and BFMI; a failed or short run is **unresolved**, not a point-recovery failure. Varying-baseline PyMC likelihood marginalizes Gaussian smoothed walk width/path and reconstructs posterior baseline paths conditionally. Model assumption: direct additive contributions, geometric carryover, Michaelis–Menten saturation, one unused latent slot, relative Gaussian observation noise, no floors or indirect/confounding paths.
- **Results:** stage `manifest.json` records resolved config, seed, environment/library versions and relevant source/input digests; `completion.json` seals output bytes. Per-world generator `dataset.npz`/`metadata.json`, fit `report.json` plus `components.npz` or `posterior.nc` are preserved before analysis. `evidence.json` contains full per-world/method computational status, gates/uncertainty, per-channel patterns/known zeros, denominators, profiles and saved scored true/fitted trajectories. Point gates compare L2 contribution error with `max(relative × L2(true channel), absolute × L2(noiseless outcome))`. Invalid computation never counts as a scored failure; absent pattern is **not represented**. True baseline/media/control variation derives from the same scored truth used in plots; noise/mechanism parameters derive from the generated source world.
- **Static report:** `report.ipynb` has ordinary Jupyter saved HTML tables and Matplotlib PNG figures; **no widget, callback, Plotly-only MIME or hidden state**. All world/method/channel ledger rows and pattern groups are printed. Example plots explicitly choose every method for the **first two world IDs**, first nonzero media channel in each; raw spend, true/fitted contribution, pointwise intervals where available and baseline are displayed. Other trajectories live in `evidence.json`; they are not silently represented as plotted. Sealed `interpretation.md` is **pending review** for its exact run; no automatic scientific conclusions, inheritance or real review approval.

**Larger changes require adaptation, not configuration sleight of hand:** for non-geometric carryover, other saturations, zero-active inputs, new direct/indirect graphs, floors, confounding, changed latent structure/priors, absolute noise, non-Gaussian observations, another outcome support, windows or baseline process, update *together* the config contract (`config.py`), generator and truth validation (`dgp.py`), inference/likelihood (`methods.py`, `models.py`), score/diagnostics and stage reader (`analysis.py`, `reporting.py`), static presentation (`report_notebook.py`) and focused synthetic and real integration tests. Declare observation units, true decomposition/graph and masks, reference anchors, dimensions, model priors, status/gates, output schema and source/seed receipt migration. Reject unsupported requests before stage output; never label inherited historical fits current, infer new science from tiny budgets or reuse an approved interpretation from another run.

## Focused checks (ship source, not generated output)

```sh
P=studies/recovery/.venv/bin/python
studies/recovery/.venv/bin/ruff check studies/recovery/*.py
studies/recovery/.venv/bin/ruff format --check studies/recovery/*.py
$P -m pytest -p no:cacheprovider studies/recovery/test_pipeline.py -q
```

The `run-artifacts/` tree is ignored, derived and may include notebook JSON; **do not lint the whole `studies/recovery/` directory after runtime**, which would incorrectly lint generated notebook code. The scoped Ruff commands above check every shipped study Python source plus the self-contained tests regardless of generated artifacts. Unit tests perform no generation or inference. Real fitting is an explicit, separately budgeted operator gate, not an ordinary test.
