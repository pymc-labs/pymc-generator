# Task 1 independent re-review, pass 2 — FINDINGS

Read `PLAN.md`, `task1-rereview.md`, and `task1-fixes-2.md` before checking current code, tests and real saved manifests. This is **Task 1 only**. Only this handoff was written; previous handoffs/evidence and all source artifacts remain unchanged. No notebook execution/import, generation, fitting, delegation, commit, push or Task 2/3 work.

## Verdict

**The two prior findings are fixed, but Task 1 is not yet a PASS:** a nearby legacy input-hash *inventory omission* remains. The saved-result arithmetic and CLI negative behavior are intact. Do not equate the old manifests with certified output or live source hashes.

## Re-verified previous findings

1. **Receipt reader genuinely fixed:** `studies/recovery/artifacts.py:131-163` checks a nonempty, unique, plain-filename identity inventory, requires exactly those keys in `completion.json.outputs`, requires exactly those files on disk and then compares their bytes' SHA256. Independent reproduction from `task1-rereview.md` prints `initialized partial complete`, then raises `ContractError: completion receipt inventory differs from stage identity inventory` after removing `b` and rewriting the receipt to just the valid hash for `a`. This invokes `checkpoint` directly after receipt tampering, not `complete_checkpoint`. A further disposable-stage probe rejected receipt subset *even with both files present*, extra/missing/null receipt entries, wrong receipt hash, changed/missing/extra on-disk files, and null/empty/duplicate/non-string/missing identity inventory. The new `tests/test_recovery_task1.py:test_receipt_reader_rejects_forged_inventory` covers the main cases.
2. **Contradictory TV metadata genuinely fixed:** `studies/recovery/analysis.py:110-170` validates recorded base recipe, TV data metadata and complete contiguous TV generation/NUTS seed inventories. Repeated the prior disposable-layout probe: authentic original base/TV/fit manifests with symlinks to unchanged original base batch files **accepted**; both standalone and fit-embedded data manifests jointly changed to smoothness `[4,4]`, max width `3`, confounding `1`, floor `2`, generation seeds `[1]*50`, or missing floor each **rejected**. Fit seeds `[92001,123]`, one seed, duplicate final seed each rejected; changed base beta, width, confounding range or floor each rejected. `tests/test_recovery_task1.py:test_legacy_tv_rejects_contradictory_manifest_metadata` adds a focused regression. The provenance string remains `legacy manifest/input-hash compatibility only; output and source hashes unavailable` (`analysis.py:341`): metadata consistency and existing input hashes do not certify old outputs/live source.

## New finding, Task 1 scope

**MEDIUM — legacy reader accepts missing input-hash inventory entries, including a corrupted unlisted input batch.** `studies/recovery/analysis.py:198-212` requires the `source_sha256` (TV) or `data_sha256` (bounded) map be nonempty and checks only its *supplied* entries. Both authentic manifests contain hashes of all 25 base `batch_00.npz`…`batch_24.npz` (base `worlds=50`, `cells_per_batch=2`). In a temporary sibling layout with copied manifests and symlinks to all 25 unchanged base batches, reduce TV `source_sha256` to `{batch_00.npz: original_hash}` in **both** TV manifests: `_legacy_compatible(...,'tv_nuts', ...)` **accepts**. Replacing only the temporary `batch_01.npz` symlink with corrupted bytes still **accepts**, because `batch_01` is no longer inventoried. The identical reduced-map/corrupted-unlisted-batch probe **also accepts** for bounded `data_sha256` with `StudyConfig.parse({})`. This defeats the advertised *input-hash compatibility* check for the rest of the requested 50 worlds, even though output/source certification is correctly disclaimed. Require the recorded hash-key set to cover exactly the expected recorded base batch inventory for the legacy population (and validate expected count/names from the base manifest), then hash all entries. Do not attempt to infer output hashes or execute notebooks. Add a regression for the truncated input inventory, preferably TV and bounded.

Exact minimal TV/bounded reproducer from repo root (uses only disposable manifests and temporary bytes; nothing changes in `notebooks/`):

```sh
PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python - <<'PY'
import json, tempfile
from pathlib import Path
from studies.recovery.analysis import _legacy_compatible
from studies.recovery.config import StudyConfig
original = Path.cwd() / 'notebooks'
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp); base = tmp / 'ols_study_50_seed10029'; base.mkdir()
    (base/'manifest.json').write_bytes((original/base.name/'manifest.json').read_bytes())
    for batch in (original/base.name).glob('batch_*.npz'):
        (base/batch.name).symlink_to(batch)
    tv = tmp/'time_varying_baseline'; nuts = tv/'nuts'; nuts.mkdir(parents=True)
    data = json.loads((original/tv.name/'data_manifest.json').read_text())
    fit = json.loads((original/tv.name/'nuts/manifest.json').read_text())
    data['source_sha256'] = {'batch_00.npz': data['source_sha256']['batch_00.npz']}
    (tv/'data_manifest.json').write_text(json.dumps(data))
    (nuts/'manifest.json').write_text(json.dumps({**fit, 'data_manifest': data}))
    _legacy_compatible(nuts, 'tv_nuts', StudyConfig.parse({'study': {'baseline': 'varying'}}))
    (base/'batch_01.npz').unlink(); (base/'batch_01.npz').write_bytes(b'corrupted')
    _legacy_compatible(nuts, 'tv_nuts', StudyConfig.parse({'study': {'baseline': 'varying'}}))
    print('BUG: TV accepts subset hashes AND corrupted unlisted input batch')
    bounds = tmp/'ols_study_50_seed10029_generator_bounds'; bounds.mkdir()
    record = json.loads((original/bounds.name/'manifest.json').read_text())
    record['data_sha256'] = {'batch_00.npz': record['data_sha256']['batch_00.npz']}
    (bounds/'manifest.json').write_text(json.dumps(record))
    _legacy_compatible(bounds, 'bounded_ols', StudyConfig.parse({}))
    print('BUG: bounded also accepts subset hashes AND corrupted unlisted input batch')
PY
```

## Repeatable checks and unaffected parity

```sh
PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m pytest -p no:cacheprovider tests/test_recovery_task1.py -q
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py
PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m studies.recovery analyze --config studies/recovery/saved_tv_example.toml --saved notebooks/time_varying_baseline/nuts --format tv_nuts
```

Results: **10 passed in 0.39s**, Ruff clean, **6 files formatted**; supported CLI retains explicit legacy provenance and 50 valid/0 unresolved, media 11, joint 9, total-media 36, media failures 144/236, control failures 19/148. Independently recalculated from the raw JSON+NPZ of all 50 worlds without calling new scoring code: L2 gate `norm(truth-estimate,axis=0) <= maximum(.1*norm(truth,axis=0), .01*norm(truth.sum(axis=1)))`; total media scored from summed channel paths, healthy diagnostic/optimizer filters applied, nonzero-truth channel failure denominators. All scores use 97 rows. TV: 50 valid, 11 media, 9 joint, 36 total, 144/236 media and 19/148 control failures; bounded: 48 valid/2 unresolved, 13 media, 13 joint, 28 total, 122/211 media and 15/135 controls. Saved 10%/1% joint gates match 50/50 in each format. This checks arithmetic, not scientific identifiability.

Independently reran original negative config and CLI probes: zero media/`cy`, infeasible `zy`, short burn-in, nested `methods.enabled` and ADVI `seeds` all raise `ContractError`; wrong TV/OLS window, baseline, seed, draws, diagnostics and paired claim rejected; `generate`, `fit`, `report` and incomplete `analyze` CLI return 2/empty stdout, incompatible saved CLI returns 1/empty stdout. These are stage skeleton/compatibility checks, not Task 2 execution.
