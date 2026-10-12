<!-- PR body source; repository-relative links are expanded to commit-pinned URLs at publication. -->

## Summary

The recovery study depended on a live notebook and dataset-specific analysis code. This change provides a config-driven runner and artifact-only Jupyter report so supported new datasets can be evaluated without rewriting the analysis or inheriting old conclusions.

```text
config → generate → selected OLS / ADVI / MCMC → analyze/report.ipynb
           └──── explicit commands, checkpointed artifacts ────┘
```

- Separate DGP/model adapters, scoring and reporting. Initially supports direct media/control effects, geometric adstock, Michaelis–Menten saturation, and constant/varying baselines. Bounded estimated-transform least squares is constant-baseline-only; full-rank ADVI and Nutpie draw_diag cover both.
- Validate configuration, source/input identities, complete output inventories and receipts before reuse. Reject unsupported requests before writing artifacts. Fresh qualified runs and explicitly historical evidence have separate reporting routes.
- Report pattern-specific **channel** failures, overlapping/upward/downward patterns, evaluable/unresolved denominators and absent groups—not failures elsewhere in a world.
- Jupyter Run All only loads verified artifacts and renders tables/plots. Reviewed interpretations are separate sealed attachments bound to an exact report identity; new reports start pending review.
- Preserve original datasets, fits, the live notebook and production generator behavior. Optional logistic, t-SNE and correlated-pair report modules remain explicitly deferred.

Stacked on #39 (`agent/public-trajectory-showcase`), with study verification performed on inherited snapshot `3f40ac0`; later parent-branch changes have not been incorporated or qualified. Recheck source identities/tests when restacking.

All three implementation tasks passed independent verification. This is engineering/report acceptance, **not** validation of a newly sampled scientific population or approval of scientific conclusions.

## Evidence

- Independently reproduced saved TV50: 50 diagnostically valid fits, **11/50 media**, **9/50 joint**, **36/50 total-media** passes; **144/236 media** and **19/148 control** channel failures. Bounded OLS: 48 valid fits, 13 media/joint passes; 122/211 media and 15/135 control failures among valid fits.
- Four distinct short smoke worlds exercised generation, bounded OLS, two-seed ADVI and Nutpie MCMC. All four MCMC screens failed; one recorded **68 maximum-depth hits**. ADVI finite outputs are not labelled convergence. These are execution checks, not reliable recovery evidence.
- Two existing matched fixtures replayed through the actual generator at fixed primitive RVs: exact original/changed trajectories and seven protected outputs, with no new random draws or inference.
- **32 focused tests passed**, Ruff check/format clean. Four corrected notebooks have seven executed code cells each, persisted Plotly output, valid selectors and no top-level or hidden widget errors. Independent selector-switch assertions matched actual saved data.
- Normal native reporting independently tested using explicitly synthetic current-identity fixtures, including three ADVI seeds `[101,207,303]`, method subsets, missing-fit rejection and retained full fit configuration. No historical fitted outputs were relabelled current.
- Scientific narratives independently checked against original saved artifacts: [TV-MCMC interpretation](studies/recovery/interpretations/tv50.md) and [bounded least-squares interpretation](studies/recovery/interpretations/bounded50.md). **Approved by Teemu Säilynoja** in Plannotator session `pn-2a768e`, then attached through separate sealed, exact-run-bound reviews. Both reviewed notebooks were re-executed successfully: seven cells each, rendered reviewer/run identity and approved text, persisted Plotly figures, no top-level or hidden widget errors. Cross-run/tamper tests remain explicitly synthetic.

### Visual evidence

Actual executed report output, rendered by Chromium from persisted Plotly/table MIME. These are not live Jupyter screenshots; interactive selection changes were separately exercised in notebook assertions. The examples show historical smoke evidence, not scientific adequacy.

![True/fitted contribution, raw spend and baseline from executed report](studies/recovery/evidence/actual-executed-media-plot.png)

![Pattern-specific channel denominators and unresolved/absent groups](studies/recovery/evidence/actual-executed-pattern-table.png)

Capture details: `studies/recovery/evidence/README.md`.

## Test plan

- [x] Independently recompute saved TV/bounded parity and pattern-bearing channel rates, including overlaps, trends, nulls and unresolved denominators.
- [x] Test configuration/capability rejection and complete input inventories.
- [x] Test stale, partial, missing and tampered checkpoint/receipt behavior.
- [x] Execute four short smoke worlds; verify frozen-source receipts and zero-fit resume at that source version.
- [x] Verify zero-effect controls, supported scoring windows and deterministic fixed-state replay with whole-request preflight.
- [x] Verify native/current reporting, configured ADVI seeds, explicit method subsets and incomplete/stale rejection using labelled synthetic fixtures.
- [x] Execute corrected report notebooks; verify actual initial Plotly output, selector switching and absence of embedded widget errors.
- [x] Verify separate run-bound interpretation approval, wrong-run/tamper rejection and immutable source evidence.
- [x] Preserve original datasets, fits, notebook and numerical generator/model code.
- [x] Independently review and re-verify all three delivery tasks.
- [x] Publication-tree check: 32 tests pass from an export containing only staged files, using the existing pinned study environment; all four sealed reports and both real approvals verify after relocation. Current study-source Ruff checks pass (18 files); immutable archived sources/notebooks are not reformatted.
- [ ] Full repository suite—not run.
- [ ] Full-budget new scientific study—not run or authorized.
- [ ] Live Jupyter browser click-through—not performed; executed callback assertions and persisted-output renders used instead.

Focused commands (isolated study environment):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
JUPYTER_PATH="$PWD/studies/recovery/.venv/share/jupyter" \
studies/recovery/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_recovery_task1.py tests/test_recovery_task2.py \
  tests/test_recovery_task3.py -q

studies/recovery/.venv/bin/ruff check studies/recovery tests/test_recovery_task*.py
studies/recovery/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task*.py
```

## Decisions taken without asking

- The coordinator authorized one additional same-four-world smoke repeat after formatting changed source hashes: approximately **410 seconds** of additional generation/inference, including two OLS, eight ADVI seed fits and four MCMC fits. No additional distinct scientific cases; earlier artifacts were preserved.
- After subsequent guard-only fixes, retained independently verified frozen-source numerical execution evidence instead of repeating inference again. Current-source staleness remains explicit; no old outputs were relabelled or resealed.
- Deferred optional logistic, t-SNE and correlated-pair report modules rather than expanding the mandatory delivery. Their absence is documented.

## Mistakes found during self-testing

Self-testing and independent review found and reverified fixes for:

- Incomplete receipts/hash inventories being accepted as complete.
- Score-window endpoints and pairing settings being silently ignored.
- Invalid public replay requests creating manifests before preflight rejection.
- Missing generator dependencies in matched-replay fingerprints.
- Historical-only reporting and hardcoded ADVI replicate assumptions.
- Hidden widget `ValueError: None is not in list` missed by top-level cell checks; default plots did not render.
- A sealed pending interpretation with no usable reviewed transition.

Execution mistakes and retained limitations:

- The first publication export omitted `notebooks/interactive_batches_legacy/config_29.json`; the exported-tree tests caught it. Included the unchanged configuration, provided the documented study environment, and reran all 32 tests successfully. This check reused installed dependencies; it was not a fresh dependency installation.

- Formatting after smoke execution invalidated source-qualified receipts, prompting the explicitly recorded repeat. The original artifacts remain intact.
- An initial ADVI serialization attempt failed because the isolated environment lacked `h5py`; the local dependency was installed and that attempt retried. The approximately 410-second figure describes the authorized final repeat, not all development/failed-attempt costs.
- Headless widget execution still incurs a substantial idle gap: corrected reports took roughly **122–125 seconds**, and independent runs with extra assertions/approval rendering took about **197 seconds**. This is documented, not claimed fixed.
- Scientific interpretation exposed an additional baseline-profile error after the initial engineering acceptance: the TV report used the constant reference baseline for its mean/SD. All 100 affected fields were corrected from scored TV truth, with regression coverage and separate independent verification. All recovery results and every other evidence field remain unchanged. Earlier report bundles remain preserved and are superseded by `*-profile-v2` bundles.

## Limits and merge danger

**Door:** Two-way. **Blast radius:** Study tooling and reports.

Legacy results provide manifest/input-hash compatibility only; unavailable historical result/source hashes cannot be retroactively certified. The four-world inference smoke is qualified under its recorded frozen source, not current-source fitting after guard changes. The TV50 and bounded50 interpretations are human-approved for their exact `*-profile-v2` identities; historical smoke interpretations remain pending and are not scientific conclusions. Reviewer names are recorded assertions—not cryptographically authenticated identities.

Current reports: `studies/recovery/report-artifacts/{tv50-profile-v2,bounded50-profile-v2,smoke-constant-profile-v2,smoke-varying-profile-v2}/report.ipynb`. These four selected bundles, the two scientific review siblings and their reviewed executed notebooks are included, along with their compact saved input dependencies, two matched replay fixtures and the historical four-world smoke. Large legacy posterior archives, the live marimo notebook, local environments and superseded outputs are excluded. The historical smoke's frozen source snapshot—including its archived notebook copy—is preserved as evidence, not a new live notebook. Readers can reproduce artifact-only reporting with the pinned environment; this does not reproduce or retroactively certify historical fitting. Setup, capabilities and exact commands are in `studies/recovery/README.md`.

Publication is explicitly authorized by the user. No new ticket, generation or fitting is part of publication. Selected frozen-source fixture bytes are retained verbatim, including an existing trailing blank line in the snapshot's `pyproject.toml`; changing it would invalidate the historical inventory.
