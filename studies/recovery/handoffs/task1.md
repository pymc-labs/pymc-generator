# Task 1 handoff — 2026-10-12

## Scope and files

Read the complete approved `studies/recovery/PLAN.md`. Sole writer; no commits/pushes, no data/fit/notebook edits, no sampling, no notebook import or execution. New code: `studies/recovery/{__init__,__main__,config,artifacts,analysis}.py`, `studies/recovery/saved_tv_example.toml`, `tests/test_recovery_task1.py`, this handoff. Root `bak_AGENTS.md` and `notebooks/` were already untracked before work and remain untouched. Environment: PI_MODEL=gpt-6-sol, PI_REASONING_LEVEL=medium, HERDR_ENV=1. Used root project's existing Python venv only for read-only validation (the cwd's `uv run --no-sync` created an empty ignored `.venv` before we discovered that the project root venv contains NumPy).

## Extraction inventory (read-only source review)

`notebooks/ols_reporting.py`: `prepare_ols_case` ~578; `channel_gate_metrics` ~1703; `build_bayesian_mmm` ~1903; `fit_generator_bounded` ~2337; `build_tv_baseline_model` ~3937; `reconstruct_tv_posterior` ~4107; `summarize_tv_fit` ~4135; TV comparison ~4301; matched media-only smoothness ~4813. The code here does not copy inference builders or execute notebook cells. Native saved report uses 104 reported rows, 112 full rows with 8 burn-in, 97 scored rows (weeks 7–103). `time_varying_baseline/data_manifest.json` records full_horizon=112, reported_window=[8,112], fit_window=[15,112]. Bounded OLS is estimated-transform nonlinear least squares (not ordinary OLS). TV MCMC is Nutpie draw_diag; ADVI is a different diagnostic category. Sources keep generated values separate from true zero/nonzero graph and saturation anchors; adapters must not initialize from realized truth.

Read existing artifact families: `notebooks/ols_study_50_seed10029/` (25 batches, 50 records and fits), `notebooks/ols_study_50_seed10029_generator_bounds/` (50 records/fits), `notebooks/bayesian_mmm_draw_diag/` (constant posterior and reports), `notebooks/time_varying_baseline/` (paired data, 50 NUTS reports/recovery files, first-five two-seed ADVI). Other artifacts including `notebooks/media_smoothing_comparison/` are untouched. Legacy files are not retroactively stamped with the new manifest identity.

## Contracts implemented

Strict TOML preflight rejects unknown fields, incompatible baseline/method, unsupported mechanism/edge types, malformed windows and budgets, non-Nutpie MCMC, and matched media changes without explicit coupling metadata; it rejects passing shared smoothness as an isolated media change. Config captures explicit stage methods/budgets, score and diagnostic thresholds, horizon and scoring windows. This is a **preflight schema**, not a generation adapter: requested matched coupling still requires implementation and invariant verification in Task 2. Stage manifests include schema/stage, relevant resolved configuration, source file hashes, upstream input hashes, and Python/package versions. `checkpoint()` fails closed on manifest mismatch or orphaned outputs; stage config is scoped so score changes don't invalidate generation/fit. There is no automatic regeneration. CLI has explicit stages: only `analyze --saved ... --format ...` works in Task 1; generate/fit/report fail without side effects. Saved-result import is expressly a legacy read-only parity path, not provenance qualification of all old artifacts.

Saved-result analysis recomputes individual effects, joint world (baseline excluded), total media, and baseline point gates from stored arrays. True-zero terms remain visible but are excluded from *nonzero* channel denominators. It checks saved truth/noiseless and total-media closure, and exposes unresolved fits separately. It does not compute posterior precision from draws, pattern rates, or fresh report interpretation (Task 3). The saved TV gate summaries have been recomputed from paths rather than trusted from JSON. Existing report JSON gates are cross-checked per world in tests. One existing `comparison.json` pattern table counts worlds with any patterned channel and world-wide failure, not channel-specific pattern failure; **do not reuse it** as the mandatory Task 3 pattern-specific channel-rate numerator.

## Evidence / exact commands

Using `/home/teemu/pymc-labs/prior-generator/.venv/bin/python`:

- `python -m pytest tests/test_recovery_task1.py -q` → **4 passed in 0.32s**. Includes all 50 per-world TV and bounded saved gate comparisons, config/capability rejection, stage identity mutation/stale-cache, null denominator and closure.
- `/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py` → **All checks passed**.
- `/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py` → **6 files already formatted**.
- `python -m studies.recovery analyze --config studies/recovery/saved_tv_example.toml --saved notebooks/time_varying_baseline/nuts --format tv_nuts` → 50 valid / 0 unresolved, individual media pass 11, joint pass 9, total-media pass 36, **144 failures / 236 nonzero media**, **19 / 148 nonzero controls**. These agree with path-derived values in the approved contract; counts are tested, not hardcoded in implementation.
- `python -m studies.recovery generate --config studies/recovery/saved_tv_example.toml` → exit 2, explicit unimplemented-stage error; no generation.
- Bounded estimated comparator: **48 optimizer convergences, 2 unresolved, 13/50 media passes** at 10%/1%. TV MCMC: all 50 diagnostic pass per saved reports. These are saved-evidence parity, not new inference or scientific validation.

## Next step / limitations

Independent verifier should check stage schema assumptions and saved-result parity, especially window alignment and legacy provenance caveat. Wait for coordinator release before Task 2. Task 2 must build actual adapters, validate DGP-specific matching by primitive RV invariants (same seed alone is insufficient), hash actual stage inputs/outputs and atomically checkpoint partial work, ensure zero-control/padding support, and use only authorized four-world smoke budgets. Task 3 must implement the Jupyter-only saved-artifact report and freshly reviewed interpretation; no conclusions may carry over to new run IDs. No full-study inference has been performed.
