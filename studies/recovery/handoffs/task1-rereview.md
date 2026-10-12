# Independent Task 1 re-verification — FINDINGS

Scope: read-only code and saved artifacts; only this new handoff written. Reviewed `studies/recovery/PLAN.md`, both prior handoffs, all current Task 1 modules/config/tests, and actual base/TV/bounded manifests. `HERDR_ENV=1`, `PI_MODEL=gpt-6-sol`, `PI_REASONING_LEVEL=medium`. No notebook import/execution, sampling, production edits, delegation, commit or push. Prior review files preserved.

## Remaining actionable defects (Task 1)

1. **HIGH — forged/incomplete completion receipt bypasses declared inventory.** `studies/recovery/artifacts.py:131-150` compares on-disk files only with `completion.json.outputs`, not with `manifest.json.expected_outputs`. The sealer (`:162-186`) rejects a subset, but the *reader* accepts one if a receipt was truncated/altered: in a temporary stage with identity inventory `['a','b']`, remove `b` and replace completion receipt with `{'schema':1,'outputs':{'a':sha256(a)}}`; `checkpoint(stage, identity).state` returns **`complete`**. That is a false completed stage under the recorded identity; inspection must independently require `set(receipt.outputs) == set(expected['expected_outputs'])` and reject missing/null inventory. Add a regression test for independently modified receipts (not only the `complete_checkpoint` writer's guard). Source files untouched.

2. **MEDIUM — legacy TV compatibility accepts contradictory saved generator/model provenance.** `studies/recovery/analysis.py:54-67,75-109,110-144` checks the embedded TV manifest equals the standalone copy and validates some fields, but does **not** validate recorded `smoothness_beta_prior`, `max_smoothing_width`, `confounding_strength`, `baseline_floor`, TV generation seeds or full NUTS seeds against the supported legacy design / base config. In a disposable layout containing copies of actual manifests and symlinks to the unchanged, digest-verified base `batch_*.npz`, `_legacy_compatible(..., 'tv_nuts', default_varying_config)` accepted the authentic metadata **and** separately accepted both manifests changed together to `[4,4]` smoothness prior, width 3, confounding strength 1 or TV generation seeds `[1]*50`; changing the NUTS seed list to `[92001,123]` also passed. No result-output hashes exist, so this cannot prove old output provenance; still, do not allow *known contradictory manifest claims* through a compatibility guard. Compare recorded model/DGP settings to actual supported legacy configuration and ensure seed inventory is contiguous/complete for requested worlds, or explicitly narrow the claimed compatibility. Preserve the explicit `provenance: legacy manifest/input-hash compatibility only; output and source hashes unavailable` qualification: it correctly prevents certification of outputs or new-style checkpoint completeness.

## Four original failure modes: fixed on independently repeated probes

- Saved windows/format: both TV and bounded reject `score_weeks=[0,103]`; bounded rejects varying baseline; TV rejects wrong study seed, changed NUTS draw budget/diagnostic gate, or unsupported paired claim. Actual CLI mismatch exits 1 with **empty stdout**; supported example succeeds with compatible manifests. The 97 scored rows correspond to 7–103, not foreign full-104 scoring.
- Preflight: default complete eight-edge dictionary is direct-only with `cy` >=1; baseline support resolves `[0,0]` for constant and `[0,0.093]` for varying. Zero media active range, zero/infeasible `cy`, infeasible `zy`, and burn-in 1 are `ContractError`, not accepted.
- Checkpoint: manifest-only is `initialized`, one of two outputs is `partial`, complete only after receipt from sealer; missing or changed completed output rejected. **Exception:** the receipt-subset bug above, so the full checkpoint-inventory fix is not yet complete.
- Nested lists for `methods.enabled` and `methods.advi.seeds` now raise `ContractError` rather than `TypeError`.

## Command evidence

From repo root with `/home/teemu/pymc-labs/prior-generator/.venv/bin/python` (`V` below), no cache/bytecode generation:

```sh
V=/home/teemu/pymc-labs/prior-generator/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$V" -m pytest -p no:cacheprovider tests/test_recovery_task1.py -q
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py
PYTHONDONTWRITEBYTECODE=1 "$V" -m studies.recovery analyze --config studies/recovery/saved_tv_example.toml --saved notebooks/time_varying_baseline/nuts --format tv_nuts
```

Results **8 passed in 0.38s**, Ruff clean and 6 files formatted; example CLI outputs 50 valid, media 11, joint 9, total media 36, failures media 144/236 and controls 19/148 with a limited provenance label. Also invoked all unsupported CLI stages `generate`, `fit`, `report` with valid config and `analyze` missing `--saved`/`--format` via `subprocess.run`: each exit 2, empty stdout; mismatched TV window and TV config against bounded directory exit 1, empty stdout.

Independent one-off read-only parity probe (not `analysis.py`): for each of worlds `0..49`, load `world_{w:02d}_nuts_report.json` + `world_{w:02d}_nuts_recovery.npz`, respectively bounded `world_{w:02d}.json` + `world_{w:02d}_fits.npz` with `allow_pickle=False`; calculate `norm(truth-estimate,axis=0) <= maximum(.1*norm(truth,axis=0),.01*norm(truth.sum(axis=1)))`, total-media sum trajectories with the analogous allowance, apply diagnostic/optimizer validity for aggregates, count failures among nonzero-truth terms only, and compare joint gates against saved JSON. **TV**: 50 valid/0 unresolved, 11 media/9 joint/36 total, media failures 144/236, control failures 19/148; **bounded estimated**: 48 valid/2 unresolved, 13 media/13 joint/28 total, media failures 122/211, control failures 15/135. Every scored shape has 97 rows, 50/50 saved joint gates match for each format. This verifies saved-result *arithmetic*, not scientific recovery validity.

To reproduce false completed receipt without touching source artifacts:

```sh
PYTHONDONTWRITEBYTECODE=1 "$V" - <<'PY'
import json, tempfile
from pathlib import Path
from studies.recovery.config import StudyConfig
from studies.recovery.artifacts import identity, write_manifest, checkpoint, complete_checkpoint, digest
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp); source = base/'source'; source.write_text('x')
    ident = identity('generate', StudyConfig.parse({}), source_files={'src': source}, expected_outputs={'a','b'})
    stage = base/'stage'; write_manifest(stage, ident)
    print(checkpoint(stage,ident).state)  # initialized
    (stage/'a').write_bytes(b'a'); print(checkpoint(stage,ident).state)  # partial
    (stage/'b').write_bytes(b'b'); complete_checkpoint(stage,ident,{'a','b'})
    print(checkpoint(stage,ident).state)  # complete
    (stage/'b').unlink()
    (stage/'completion.json').write_text(json.dumps({'schema':1,'outputs':{'a':digest(stage/'a')}}))
    print(checkpoint(stage,ident).state)  # BUG: complete, despite missing b
PY
```

For provenance negatives, used `tempfile.TemporaryDirectory()`, copied `ols_study_50_seed10029/manifest.json`, symlinked the **actual** base `batch_*.npz` into temporary base directory, copied `time_varying_baseline/data_manifest.json` and `nuts/manifest.json` into a temporary sibling layout, and invoked `_legacy_compatible(temp_nuts,'tv_nuts', StudyConfig.parse({'study':{'baseline':'varying'}}))`; repeated after making *identical* changes to nested `fit_manifest['data_manifest']` and standalone data manifest for the four fields above, plus separate fit-seeds change. All five contradictory records accepted. Symlinks and edited manifests were disposable; saved source manifests/arrays untouched.

## Boundaries

The fixes materially improve Task 1 and saved arithmetic remains sound, but these two guard failures warrant remediation and independent retest before Task 1 PASS. Legacy metadata cannot certify actual result hashes or live source identity; this is disclosed in the output. Do not require Task 2 generator/fit adapters, atomic per-world resume, or Task 3 Jupyter/reporting/pattern rates for this gate. No new generation or inference performed.
