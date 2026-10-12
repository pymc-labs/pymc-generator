# Recovery study · artifact-only reporting

This study has separate **generate**, **fit**, **analyze**, **report** stages. Nothing here runs implicitly. Task 3 is **reporting from saved artifacts only**; no new generation/fitting is authorized. The original `notebooks/ols_reporting.py`, original datasets/fits, and their receipts remain untouched. `studies/recovery/report-artifacts/` contains derived reports. The publication includes the four verified `*-profile-v2` bundles, the two approved scientific review siblings and their reviewed executed notebooks; other generated reports remain ignored. No marimo runtime is needed for the Jupyter report.

## Reproducible isolated setup (repository root)

Python 3.13; install with `uv` into the **study-local** environment, never the root/live notebook environment:

```sh
uv venv studies/recovery/.venv --python 3.13
uv pip install --python studies/recovery/.venv/bin/python -r studies/recovery/requirements-report.txt -e .
studies/recovery/.venv/bin/python -m ipykernel install --prefix studies/recovery/.venv --name recovery-study --display-name 'Recovery study isolated Python'
export JUPYTER_PATH="$(pwd)/studies/recovery/.venv/share/jupyter"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

`requirements-report.txt` extends pinned `requirements-smoke.txt` with pinned notebook, kernel, Plotly and widget dependencies. The `.venv` and newly generated report artifacts are ignored locally; explicitly selected publication bundles are tracked. No global Jupyter kernel registration. To reproduce numerical environment precisely, use the pinned Python 3.13.12 available from `uv`; do not update a live notebook session.

## Report examples — no sampling or fits

```sh
# Real existing 50-world TV NUTS recovery results, legacy evidence label.
studies/recovery/.venv/bin/python -m studies.recovery report \
  --config studies/recovery/saved_tv_example.toml \
  --saved notebooks/time_varying_baseline/nuts --format tv_nuts \
  --out studies/recovery/report-artifacts/tv50-profile-v2

# Real existing 50-world bounded estimated-transform least-squares results.
studies/recovery/.venv/bin/python -m studies.recovery report \
  --config studies/recovery/saved_bounded_example.toml \
  --saved notebooks/ols_study_50_seed10029_generator_bounds --format bounded_ols \
  --out studies/recovery/report-artifacts/bounded50-profile-v2

# Explicit HISTORICAL smoke: validates recorded identities AND output receipts,
# warns that current-source fitting would reject them as stale. Never --fit.
studies/recovery/.venv/bin/python -m studies.recovery report \
  --config studies/recovery/smoke_constant.toml \
  --saved studies/recovery/smoke-artifacts-final/constant --format smoke --historical \
  --out studies/recovery/report-artifacts/smoke-constant-profile-v2
studies/recovery/.venv/bin/python -m studies.recovery report \
  --config studies/recovery/smoke_varying.toml \
  --saved studies/recovery/smoke-artifacts-final/varying --format smoke --historical \
  --out studies/recovery/report-artifacts/smoke-varying-profile-v2

# Execute the NEW, sealed report notebook (Run All, read-only source).
# The executed notebook is a separate derived copy OUTSIDE the sealed report.
studies/recovery/.venv/bin/python -m studies.recovery.execute_report \
  --report studies/recovery/report-artifacts/tv50-profile-v2 \
  --out studies/recovery/report-artifacts/tv50-profile-v2-executed.ipynb

# Future NATIVE run ONLY after separately authorized generation/fits have completed:
# The exact original fit config is unchanged; select which already-fitted methods to report.
# Omit --report-method to require ALL enabled methods and all configured ADVI seeds.
studies/recovery/.venv/bin/python -m studies.recovery report \
  --config PATH/TO/original-run.toml --saved PATH/TO/run-root --format native \
  --report-method mcmc --out PATH/TO/new-empty-report-root
```

`native` requires even supported world count, current-source generation and selected fit manifest identities/output receipts for EVERY world; missing or stale fits fail before any report output. `--report-method` is a nonempty distinct subset of the **original config's enabled methods**; it never changes source fit identity. ADVI means ALL original configured seeds (e.g. three seeds → `advi_1` through `advi_3`); no method is silently omitted. Legacy formats have only their one recorded method and reject explicit method selection. Historical `smoke` requires `--historical` and never claims current-source qualification. The `native` example above is a **usage template only**, not an executed science run.

Run All manually: launch Jupyter with study-local kernel `recovery-study`; open `report.ipynb`, **set working directory to the report root**, then Run All. `execute_report` does the same and writes a separate executed copy. Input paths are **relative to the report directory**, and generated notebook code resolves project imports from a relative project path; relocate the project and reports together, preserving their layout. The source tree must be available to verify source identity. The notebook checks all current report source/config/environment/input hashes, all report output receipts and the exact interpretation identity **before** rendering. Any changed input, source or config requires a new output root and new interpretation. No silent resume or inherited conclusions. `--out` must not exist and must not contain source data. No report writer touches original inputs.

Previously generated `*-corrected` and `*-final` report roots are preserved but their report-source identities are **superseded/stale** after the scored-truth profile correction; do not use them for new interpretations. Regenerate in fresh `*-profile-v2` roots as above; old original inputs/receipts must not be edited.

`evidence.json` contains machine-readable per-world diagnostics, recovery, uncertainty, oracle profiles, per-channel flags/scores, per-method pattern denominator rows, all true/fitted trajectories and selected raw spend. `manifest.json` + `completion.json` seal the report under its own source and input identity. `interpretation.md` is **pending review**, tied to the exact identity; editing the sealed interpretation invalidates its receipt. For an ACTUAL human review, prepare a nonempty Markdown interpretation file **outside** the sealed report, inspect the exact printed run ID, then explicitly approve it (do not do this for smoke demos without a human scientific review):

```sh
studies/recovery/.venv/bin/python -m studies.recovery.review_interpretation \
  --report PATH/TO/report-root --interpretation PATH/TO/human-reviewed.md \
  --reviewer 'EXPLICIT HUMAN NAME' --approve-run EXACT-64-HEX-RUN-ID
# Creates only PATH/TO/report-root-review/{interpretation.md,review.json,review-receipt.json}.
# Reopen/Run All the original sealed report: a verified attached review is shown.
```

The separate immutable sibling review seals the reviewer, explicit decision, exact run ID, report manifest/evidence/completion bytes, text hash and its own receipt. A missing review remains pending; a present but changed, tampered or cross-run review fails closed. The source `report.ipynb`, `interpretation.md` pending template and report receipt are never altered. Reviewer name and approval are *recorded assertions, not cryptographic authentication*; independent governance must establish actual reviewer identity. A prior report/review never supplies approval to a changed dataset. Archived legacy outputs have **only manifest/input-hash compatibility**, lacking old per-output and old-source certification; newly hashing files for this report does not upgrade the historical evidence. The frozen smoke checkpoint manifests and output receipts verify recorded identity; the later guard edits make them **stale under current fitting identity**. No past MCMC smoke is claimed converged.

## Meaning of tables and plots

Scoring uses 97 weeks 7–103 out of 104 reported, with eight generator burn-in weeks; gate `L2(error) ≤ max(relative × L2(true channel), absolute × L2(true noiseless outcome))` (default 10%, 1%). Baseline is not part of the joint-effect score; media, controls, joint, total-media and baseline results stay distinct. Individual **pattern-bearing-channel** failure counts use the channel's own nonzero effect and own gate, not failures of another channel in its world. Patterns overlap; trend up and down have separate rows. Each group reports `total` (pattern-bearing, including exact known-zero), `eligible` (nonzero), `known_zero`, `evaluable` (computationally valid), `failed` (among evaluable), `unresolved` (eligible but computationally invalid), `failure_rate=failed/evaluable` or null. Zero coverage is **not represented**, not 0%; <10 evaluable is flagged. Baseline is always present; null/padded contribution traces are hidden. Select a world/method and channel to see separate Plotly contribution, raw media spend and baseline panels with legend below.

Numerical profiles show realized noise SD, true media/control contribution variation, baseline temporal SD/mean, carryover alpha, MM multiplier, media and control effect values. Baseline and media/control variation are computed from the **exact scored-window truth trajectories** saved in the report, including the TV-baseline regime rather than the constant source corpus; noise SD and mechanism/effect parameters retain the original source-corpus inputs. The TV observation residual agrees with that original noise up to saved float rounding. They are *descriptive oracle diagnostics*, never fit initialization. MCMC diagnostics are screening thresholds (zero divergences/maxdepth hits, R-hat<1.01, bulk/tail ESS≥400, BFMI≥0.3). Smoke budgets cannot satisfy reliable inference; MCMC point scores remain unresolved if diagnostic gates fail. ADVI reports finite loss and **interseed channel discrepancy relative to the score gate**; variational draws have no meaningful R-hat/ESS and finite 500-step smoke loss is not optimization/convergence certification. Bounded least squares has an optimizer status but no posterior uncertainty. Posterior intervals are 95% *pointwise* intervals where saved (legacy TV and smoke draws), not out-of-sample calibration.

Capability matrix:

| Capability | Supported now | Limitation |
|---|---|---|
| Existing TV NUTS50 / bounded OLS50 | Read-only `report --format tv_nuts` / `bounded_ols` | Legacy hashes limited; score thresholds fixed by provenance where relevant |
| Frozen four-world smoke | Explicit `--format smoke --historical` | Recorded receipts checked; current-source fit resume refused |
| New completed native runs | `--format native` and optional `--report-method` subset | CURRENT identities/receipts required per world, all selected ADVI replicates; no fit triggered |
| Interpretation review | Explicit `review_interpretation --approve-run ...` | Immutable sibling, not authenticated identity; no automatic prior-run inheritance |
| New independent generation/fits | Explicit separate CLI commands only, **not authorized in Task 3** | Constant: OLS/ADVI/Nutpie; varying: ADVI/Nutpie, no OLS |
| Media-smoothing matched | Fixed primitive-RV replay of TWO existing fixtures | Not fresh stochastic matched generation; seed reuse alone not matching |
| Control effects | Zero *contributing* effects supported | Literal zero-active-control inputs rejected |
| Optional logistic/t-SNE/correlated pairs | **Deferred/unsupported** | No claim of held-out-world logistic validation or exploratory pair effects |

Example-only, **not executed**: `full_constant_example.toml`, `full_varying_example.toml` configure 50 worlds with full 4-chain, 1000+1000 `.95` Nutpie and 20k-step/4000-draw two-seed ADVI budgets. Bounded OLS example uses the generator-supported estimated transforms, not ordinary unconstrained OLS. Small execution examples: `smoke_constant.toml`, `smoke_varying.toml`; these also must not be rerun without coordinator authorization. The independent scoring/readers and report have no authorization to generate or fit on their own.
