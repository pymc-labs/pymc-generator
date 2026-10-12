# Coordinator status

## Current follow-up — scientific interpretation and baseline-profile correction

User requested interpretation for including the report in the PR. Coordinator narratives: `studies/recovery/interpretations/tv50.md` and `bounded50.md`; independent scientific review PASS in `interpretation-science-review.md`. Human approval received for both exact-ID narratives in Plannotator `pn-2a768e`; user explicitly supplied reviewer name **Teemu Säilynoja**. Canonical sibling `tv50-profile-v2-review` and `bounded50-profile-v2-review` attachments created and verified. Both `*-profile-v2-reviewed-executed.ipynb` copies execute seven cells and render the exact approved text/reviewer/run identity, with Plotly output and no top-level/hidden errors. Source reports remain immutable. See `../interpretations/APPROVAL.md`. The user subsequently explicitly authorized PR publication. Publication includes selected verified artifacts/fixtures, not local environments or large legacy posterior archives; stacked on #39. Exported staged-tree validation passes 32 tests, verifies four report identities and two real approvals, and current-source Ruff checks pass. No new scientific generation/fitting.

During interpretation, coordinator discovered a post-acceptance bug: legacy TV profiles derive baseline statistics from the constant reference corpus instead of scored TV truth. World 0 reports SD 0 instead of 0.4554043845587071, mean 3.5764687061309814 instead of 3.5845318436461295. Scoring/trajectory evidence is unaffected. Correction independently ACCEPTED in `interpretation-profile-review.md`: 32 tests pass, Ruff clean; all four NEW `*-profile-v2` reports and executed widget/Plotly state verified. Exactly 100 TV baseline profile fields changed; all other evidence unchanged. Original bundles/receipts stay unchanged and superseded. Scientific narratives name exact verified TV/bounded v2 run IDs; human review and attachment are now complete. Existing three-task PASS records remain historical evidence, not approval of this newly identified error. No sampling/fits or scientific model changes authorized.

Agents retain context through this follow-up's fix/review rounds. Coordinator owns interpretations/PR/status; implementer owns code/tests/README and regeneration; verifier independently checks narrative facts and later the code/report correction.

## Task 1 — ACCEPTED

Independent acceptance: `task1-rereview-3.md`. Twelve focused tests, Ruff checks/format and independently calculated 50-world TV/bounded saved-result parity passed. All four initial findings and subsequent receipt/provenance/input-inventory omissions were fixed and reverified. Prior reviews and fix handoffs remain preserved.

Limitation retained: legacy manifest/input-hash compatibility cannot certify old result-file or source hashes. Never promote imported legacy evidence to a newly provenance-qualified fit checkpoint.

No generation/inference/notebook modifications were part of Task1. Operator clarified that reviewer/implementer context persists THROUGH fix/re-review rounds. Compact at accepted task boundaries (or demonstrated context pressure), not after each review pass.

## Task 2 — ACCEPTED

Independent acceptance: `task2-rereview-2.md`, 23 focused tests and Ruff checks passed. All review findings fixed. Four-world runtime evidence remains frozen-source/historical after guard-only config changes; numerical code and resolved science configs are byte-identical to snapshots, historical output receipts intact, current-source stale rejection correct. Do not claim current-source fit qualification or convergence. Current matched fixed-RV replay is separately qualified under its recorded dependency hashes.

### Authorized Task2 scope and execution history

Follow PLAN.md Task2 only. Build genuine supported DGP and inference adapters, explicit CLI generation/fitting and hash-qualified atomic per-world checkpoint/resume. Preserve study science (oracle graph, supports, reference anchors, score window, Nutpie draw_diag, full-rank ADVI; TV exact width/path marginalization and conditional reconstruction). No import/execution of live notebook; read-only extraction permitted.

Authorized runtime: four small smoke worlds total, two constant and two varying; bounded estimated-transform least squares on constant only; full-rank ADVI500 steps with two seeds per world; Nutpie2 chains x100 warmup+100 retained for each world. Small configurable ADVI posterior draw count. These test execution/persistence, not convergence or recovery. No enlarged retries to force healthy diagnostics; report unhealthy fits honestly. Resume tests reuse these same four worlds and existing fit artifacts; do not add more scientific worlds. If blocked by runtime/dependencies or model equivalence, send evidence/question rather than silently changing models, methods or budgets.

Matched-validation clarification: deterministic replay of TWO EXISTING media-smoothing fixture worlds from their saved primitive RVs/structures is authorized, with zero new random draws and zero additional inference. Exercise the actual reusable matched replay adapter, not just old flags; verify unchanged primitive values/protected outputs and original/modified trajectory parity. Original fixture artifacts remain read-only; derived validation goes under study-local paths. This is separate from the four newly sampled smoke worlds. Do not drop matched replay functionality; distinguish its supported fixed-state capability from unexecuted fresh stochastic matched-population generation.

One-time smoke replay authorization (formatting-induced stale receipts): after all source edits/formatting/lint/tests finish, freeze source hashes/snapshots and repeat the SAME four smoke cases with identical seeds/configs/budgets into a NEW artifact root. This adds generation/inference computation (expected 2 OLS + 8 ADVI seed fits + 4 MCMC fits), not distinct scientific cases. Preserve historical stale artifacts and receipts without mutation. Independently compare generated arrays with the first smoke run. Verify new receipts and zero-fit resume after execution. No diagnostic-driven retries, budget increases or further source-triggered reruns without coordinator approval. Record root paths, full costs/timings and stale event honestly. Subsequent computational source edits must not silently invalidate accepted smoke evidence.

Zero-control clarification: the acceptance case is zero CONTRIBUTING control effects with active control inputs present, matching the original study's oracle masks. Library `_validate_active_ranges` requires at least one active control; literal zero-active-control input datasets are outside the initial capability. Reject these explicitly before generation. Cover no live control coefficients, exact-zero contributions, diagnostics/empty selections and scoring denominators; do not label an empty eligible group as observed successful recovery. No generator changes or new inference repeats authorized by this clarification.

Keep existing source datasets/fits/live notebook and production API/defaults untouched. No global environment changes, commits/pushes, other agents, or Task3 before gate. Isolated reproducible study dependencies may be added under the study if required; do not mutate the live marimo or root environments.

Acceptance evidence: targeted tests of models/transforms, no-truth-initialization, masks/zero controls, configuration/capability rejection, generated identity/provenance, resume/corruption behavior; actual four-world smoke outputs with computational status; commands/timings/configs; preserved existing parity. Matched DGP capability must verify declared primitive-RV invariants, not merely reuse seeds; keep matched generation tests bounded and do not authorize extra scientific inference. Code and evidence handed to same independent verifier before acceptance.

## Task 2 first review — targeted fixes authorized, no further inference

`task2-review.md` verified runtime/scientific smoke evidence but found ignored score-window upper bound, silently ignored/incompatible pairing settings, and missing evaluator source hashes in matched-replay identity. Fix via explicit capability rejection and config/matched provenance guards where possible, without changing numerical model/DGP/fitter code. No extra stochastic generation or fitting. Repeating the same two deterministic existing-world replay fixtures in a new validation root is allowed.

Preserve all prior receipts and frozen source snapshot. Guard changes may legitimately make prior smoke identities stale: do not rewrite or relabel them. Evidence must distinguish frozen-source smoke runtime from current guard-only regression checks, prove unchanged resolved smoke configs and numerical source bytes, and retain correct current-source stale rejection. If a fix needs numerical inference/generation changes, ask coordinator. This is not permission to silently qualify old outputs under new source hashes. Task2 remains unaccepted; Task3 blocked; retain both agents' context through this fix/re-review cycle.

## Task 3 — ACCEPTED; approved implementation plan complete

Independent acceptance: `task3-rereview.md`. Final focused suite: 31 passing tests, Ruff clean. Reviewer independently verified the native/current report path including three unequal ADVI seeds and explicit method subsets on labelled synthetic fixtures; all four corrected notebook outputs, real-data selector changes and hidden-error absence; separate sealed run-bound interpretation approval with test-only reviewer; mandatory channel-pattern rates/parity; preservation of historical receipts and source evidence limits.

At the original Task 3 gate, real reports remained pending scientific interpretation; the approved follow-up is recorded above. No full-budget new study or scientific convergence claim. Optional logistic/t-SNE/pair reporting remains explicitly deferred. Headless notebook widget idle delay is documented (~122–197 seconds depending on execution). Browser-rendered captures of actual persisted notebook Plotly/table outputs are preserved under `studies/recovery/evidence/`, clearly not live Jupyter screenshots. Updated unpublished PR description: `studies/recovery/PR_DESCRIPTION.md`.

Both agents are instructed to compact at this accepted-task boundary and remain idle. No next task, commits, pushes, PR/ticket publication, extra generation or fitting authorized.

### Authorized Task3 scope and execution history

Implement artifact-only analysis and Jupyter reporting, example configs/usage, run-bound interpretation lifecycle and focused end-to-end checks. Read PLAN.md mandatory sections; specifically pattern-bearing CHANNEL failure rates, separate media/control and overlapping pattern labels, trend direction, total/evaluable/failed/unresolved counts, absent patterns "not represented". Preserve healthy diagnostics vs point recovery vs uncertainty distinctions and ADVI finite-only/seed sensitivity caveats.

No new random generation or OLS/ADVI/MCMC fitting. Generate and EXECUTE the new report notebook from saved artifacts to prove Run All only reads/renders, using the study-local environment (local report dependencies may be added reproducibly; no global/root/live env changes). Do not execute/modify the live original marimo notebook. Produce reports for actual available legacy science evidence and explicit historical smoke evidence with appropriate provenance labels. A historical read/report path must be EXPLICIT and validate recorded identities/output receipts while warning current-source reuse for fitting is forbidden; never silently relax normal stage compatibility. Legacy imports retain their limited manifest/input-hash claim. Reports must not recertify old outputs under current fitter source.

Keep numerical DGP/model/method/runner code untouched where possible; new analysis/report modules and CLI wiring only. Scope source fingerprints to real stage dependencies; adding report functionality must not initiate refits or silently bypass staleness. Include relative artifact paths, machine-readable tables and pending-review interpretation keyed to exact run/data/analysis identity. No auto-inherited conclusions, no hardcoded counts/results, no HTML deliverable, no marimo runtime dependency. Refuse unsupported inputs cleanly before writing outputs; new report roots must not overwrite original evidence.

Mandatory notebook sections: configuration/provenance/coverage, noise and true parameter/variation profiles, computational diagnostics, individual/media/control/joint/total/baseline recovery, available posterior uncertainty, pattern-specific channel failures with denominators, selectable trajectory examples (Plotly, separate panels, legends below, raw spend alongside media contribution, nulls hidden but baseline kept). Optional logistic/t-SNE/correlated-pair modules must either be implemented and explicitly enabled or clearly declared deferred/unsupported; do not claim their delivery without evidence. Logistic validation holds out worlds and inadequate data produces explicit status.

Ship example full-study configs without executing them, smoke examples and reproducible dependency/setup instructions, exact CLI usage and capability matrix (including fixed-state-only matched replay and no literal zero active controls). Document provenance/stale history and the distinction between smoke execution, diagnostic validity and scientific findings. End-to-end artifact/report checks must prove no sampling on report/Run All and no changes to original artifacts. Hand off Task3 evidence to independent same-context reviewer; no commits/pushes or whole-project completion claim until gate passes.

## Agents

- Implementer: `study-implementer`, pane `w1B:pD`, tab `w1B:tB`.
- Verifier: `study-verifier`, pane `w1B:pE`, tab `w1B:tC`.
- Both: `openai-codex/gpt-6-sol`, medium.
- Parent: `w1B:pC`.
- Completion/blocker notifications use PLAN.md prefixes and `herdr agent prompt w1B:pC ...` without --wait.
