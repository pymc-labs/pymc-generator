# Approved recovery-study extraction contract

Approved by the operator in the parent interactive session, 2026-10-12. Parent Herdr pane: `w1B:pC`. Worktree: `/home/teemu/pymc-labs/prior-generator/.worktrees/ols-reporting-marimo-restart`, branch `feature/ols-reporting-marimo-restart`.

## Objective

Build a reproducible, extensible config-driven analysis script and Jupyter report for assessing contribution recoverability under the existing oracle-information benchmark. Keep factual evidence generated from each run separate from reviewed, run-specific interpretation. Changing the dataset must NOT inherit old conclusions.

## Boundaries

- No commits, pushes, production generator API/default changes, or full-study sampling.
- Preserve ALL existing data, posterior/fit artifacts and live notebook. `bak_AGENTS.md` is disabled, not policy. Do not revive foreign Issue30/full104 requirements from memory: THIS study scores reported weeks 7–103 of 104, preserving weeks 0–6 for carryover history; generation has 8 burn-in weeks. Make horizon/window validation explicit, not hardcoded unknowingly.
- `notebooks/ols_reporting.py` is live in marimo at port2730. Never edit its file directly. Prefer READ-ONLY extraction from it; no live-cell execution/import that triggers generation or fitting. If a live edit ever becomes necessary, ask coordinator first and use marimo code mode.
- One implementer writes; the verifier independently reviews without changing implementation. No further delegation by either agent.
- Do not change agent models/providers; use requested openai-codex/gpt-6-sol at medium thinking. No global config changes.

## Architecture and public behavior

Config-driven CLI with explicit separate generate, fit, analyze, report stages. Expensive stages require explicit commands; changing config or rendering the report never auto-starts inference. Enabled methods and budgets are explicit. Checkpoint generation and fits separately, resume compatible partial work, fail on incompatible/stale artifacts instead of silent reuse or regeneration. Record resolved configs, seeds, source/environment versions and dataset hashes. Deterministic generation reproducibility is within the recorded environment, not a promise of identical MCMC across hardware.

DGP adapter -> standard dataset bundle (inputs, observed/noiseless outcomes, named true contributions, metadata, capabilities). Separate method-specific inference builders from generic scoring/reporting. Initial DGP: direct media/control effects, no other graph edges, geometric adstock, Michaelis–Menten saturation, constant or varying baseline. Validate and reject unsupported configurations/combinations before expensive execution. Compatible input counts, ranges, patterns, seeds, etc. configurable; extensions must not require rewriting metrics/reporting.

Oracle inputs: true zero/nonzero identities; generator reference anchors/supports where used. NO realized coefficient/shape/baseline/noise values for inference initialization. Clearly label privileges per method.

Both independent studies and matched sensitivity studies are supported concepts. Matched pairing needs DGP-specific coupling and recorded invariants; same seed alone is NOT a matching guarantee. Media-only smoothness override is an explicit paired capability; preserve controls/baseline/other RVs. Do not merely alter shared rw_smoothness_beta and claim media-only matching.

## Methods

Constant baseline: generator-bounded estimated-transform least squares, full-rank ADVI, Nutpie MCMC. Label bounded nonlinear least squares accurately rather than ordinary unconstrained OLS. Optional known-transform OLS diagnostic comparator.

Varying baseline: ADVI and MCMC only initially; reject OLS explicitly. Reuse the existing exact-marginalized walk/width model and conditional baseline reconstruction rather than substituting a new statistical model.

MCMC: Nutpie adaptation="draw_diag", no fallback to native PyMC. Configurable chains, warmup, draws, target_accept and seed. Current full example 4 chains,1000+1000,.95,maxdepth10. ADVI full rank, configurable budgets, two seeds by default; current full example20k steps4000 draws. Separate finite/optimization checks and seed sensitivity from convergence; no Rhat/ESS for ADVI.

## Analysis/report

Primary gate configurable default relative=.10, absolute=.01: channel L2 error <= max(relative*true-channel L2 norm, absolute*noiseless-outcome L2 norm). Separate individual media, controls, whole-world joint (baseline excluded), total media, and baseline results. Separate computational validity, point recovery, posterior precision; unresolved fits remain visible. MCMC gate currently zero divergences/maxdepth terminations, Rhat<1.01, bulk/tailESS>=400,BFMI>=.3 (explicit config).

Mandatory coverage: dataset/pattern coverage, noise, saturation/adstock/amplitudes/variation, diagnostics, recovery, uncertainty where available, trajectory examples, PATTERN-SPECIFIC CHANNEL FAILURE RATES. Pattern failure numerator counts failures of the CHANNELS carrying each pattern, not failures elsewhere in a world. Media/controls separate. Patterns overlap, upward/downward trends separate. Show total, evaluable, failed among evaluable, unresolved denominators. Zero coverage = "not represented", not0%. Disclose small counts.

Optional logistic screening (hold out entire worlds), t-SNE and correlated-pair analysis. Insufficient data gets explicit status. Hide known-zero contribution traces, retain baseline. Plotly separate panels, legends below; raw media spend alongside contribution plots where useful.

Report is Jupyter `.ipynb`, NOT HTML and no marimo server dependency. Run All only reads saved artifacts and renders figures/tables. Relative artifact paths. Save machine-readable tables plus a Markdown interpretation section/template tied to exact run, marked pending review for new dataset/run. Measured facts/caveats automated; scientific conclusions must be freshly reviewed.

## Delivery tasks and gates

### Task 1 — NOW
Configuration and artifact contracts, clean module seams/CLI skeleton, saved-result import/analysis parity. Inspect actual sources and repository conventions, choose a modest package location (avoid a giant script or arbitrary framework). Implement focused fast tests. Reproduce current saved-result summaries WITHOUT regeneration/refitting. Include meaningful config/capability rejection, artifact identity/stale-cache behavior and tests as far as appropriate to this stage. Record extraction inventory and parity evidence. Do NOT implement/execute Tasks2–3 until coordinator release.

### Task 2 — after Task1 independent verification
Generation + supported method adapters + checkpoint/resume. Verify four tiny smoke worlds: two constant, two varying, OLS constant only, ADVI/MCMC both. Authorized SMOKE ONLY: ADVI500 iterations (two seeds), MCMC2 chains x100 warmup+100 draws, configurable small retained ADVI draws. Smoke checks execution/reporting, NOT recovery/convergence; unhealthy diagnostics must be visible. No 50-world inference. Validate no truth initialization, masks, supports, model equivalence and error behavior. Preserve exact original artifact parity and all source files.

### Task 3 — after Task2 independent verification
Jupyter reporting, mandatory pattern rates, interpretation lifecycle, optional explorations as budget permits, example configs/usage docs, end-to-end validation and delivered run evidence. Full scientific example is configuration only, not executed.

At each task: implementer evidence -> independent same-model verifier -> fix/reverify findings -> parent progress update -> durable handoff -> compact agents -> next task. Operator clarification: retain the SAME agent context through fixes and re-review rounds within a task. Compact only AFTER the task is accepted and BEFORE the next task, or earlier when actual context pressure warrants it; finishing a review round is not finishing the task. Do not silently drop scope or declare whole project complete from a single stage.

## Existing sources and checks

Notebook: `notebooks/ols_reporting.py` (read-only). Artifact directories:
- `notebooks/ols_study_50_seed10029/`: canonical50, ordinary oracle OLS, manifest and records.
- `notebooks/ols_study_50_seed10029_generator_bounds/`: boundedOLS100 fits.
- `notebooks/bayesian_mmm_draw_diag/`: constant NUTS50 (46 diagnostic passes, worlds12/39/40/50 unresolved).
- `notebooks/fast_approximation_first5/`: first-five approximations.
- `notebooks/time_varying_baseline/`: matched TV50,50 healthy NUTS,first5x2ADVI;README/comparison/reports/validation.
- `notebooks/media_smoothing_comparison/`: generation-only matched50; no new fits;README,manifest,paired arrays,structures,RV snapshots,validation.
- `notebooks/correlated_media_pair_recovery/`, `total_media_recovery/`, `mcmc_difficulty_profile/`.

TV primary10%/1% reference: media pass11/50, joint9/50,total36/50; media-channel failures144/236; control19/148. All50 diagnostic passes. Constant valid46, unresolved4; bounded estimated13/50 point passes,48 optimizer convergences; bounded known transforms19/50,50 convergences. Verify every value against artifacts, do not merely bake numbers into output.

The source study uses editable local library and shared transformation expressions. Relevant notebook functions include prepare_ols_case, fit_generator_bounded, build_bayesian_mmm, build_tv_baseline_model, reconstruct_tv_posterior, summarize_tv_fit; private helper boundaries must be reviewed rather than blindly copied. Known pitfalls include padded/null dimensions, zero-control worlds, 112full/104reported/97scored windows, explicit xarray.DataArray assignment, lower_xtensor rewrite inside marginalized likelihood, pandas string option around Nutpie compilation. Read current live/disk cell source before extraction; old /tmp scripts are not authoritative.

## Communication and handoffs

Implementer prefix: `[STUDY-IMPLEMENTER][Task N][READY|BLOCKED|QUESTION]`.
Verifier prefix: `[STUDY-VERIFIER][Task N][PASS|FINDINGS|BLOCKED]`.

Notify the coordinator directly with `herdr agent prompt w1B:pC "<prefixed message with evidence summary>"` WITHOUT --wait. This is an explicitly authorized inter-agent handoff, not a new delegation. Never only print the message in your own pane and assume the parent receives it. On READY, stop and wait for coordinator instructions. No inference/task expansion without release.

Persist compact task handoffs under `studies/recovery/handoffs/` (changed files, exact test commands/results, parity checks, limitations and next step). This operator-required durable record supports compaction; also send a succinct identifying notification. Before compaction, ensure handoff captures all unresolved issues. The coordinator sends /compact while idle and verifies completion before assigning the next task.
