# Task 2 targeted-fix independent re-review — FINDINGS

Read `task2-fixes.md`, `task2-review.md`, `coordinator-status.md` and changed source/tests. **One remaining in-scope preflight defect** prevents PASS; the original scientific/config findings are fixed. No source, notebook, original/derived artifacts or environment was changed; only this new review handoff was written. No stochastic generation/inference, live notebook import/execution, commits, pushes, delegation or Task 3 work. Do not repeat smoke inference.

## Remaining actionable finding

**MEDIUM — incompatible fixed-RV replay writes a new artifact manifest before rejecting.** The direct `matched.replay_world` guard now correctly rejects requested 3 treatment slots vs the fixture's 15, but the public `replay` path (`matched.py`, `_replay_identity` → `checkpoint` → `write_manifest` → `replay_world`) creates an output directory with an initialized `manifest.json` **before** performing that compatibility check. Exact reproduction with only disposable output, no evaluation, random draws or fitting:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 studies/recovery/.venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from studies.recovery.config import ContractError, StudyConfig
from studies.recovery.matched import replay, replay_world
source = Path('notebooks/ols_study_50_seed10029')
paired = Path('notebooks/media_smoothing_comparison')
pair = StudyConfig.load(Path('studies/recovery/matched_existing_example.toml')).resolved['pairing']
cfg = StudyConfig.parse({'study': {'worlds': 2, 'seed': 10029},
                         'generator': {'n_treatments': 3}, 'pairing': pair})
try:
    replay_world(cfg, source, paired, 0)
except ContractError as exc:
    print('direct:', exc)
with TemporaryDirectory() as tmp:
    out = Path(tmp) / 'new-output'
    assert not out.exists()
    try:
        replay(cfg, source, paired, out, worlds=(0,))
    except ContractError as exc:
        print('public replay:', exc)
    print('created despite invalid request:', (out / 'world_00/manifest.json').exists())
PY
```

Output: both raise `matched generator n_treatments differs from fixed-state fixture`; the final print is **`True`**. A second invocation into that path will see a compatible *initialized* manifest, not a clean preflight rejection. Validate all requested existing fixture IDs/configs and coupling metadata **before** creating the first checkpoint/output directory; add a regression invoking `replay(...)` (not just `replay_world(...)`) for the invalid config and asserting no destination. This is a narrow side-effect/order fix; it needs no new scientific worlds, numerical-source edits or inference. The valid two-fixture replay remains intact.

## Prior findings re-verified fixed

- `StudyConfig.parse({'study': {'score_weeks':[7,20]}})` on default 104 weeks now raises `ContractError` before execution, as do `[8,103]`, `[0,103]`; both `generate` and `fit` CLI with a temporary invalid TOML reject before creating `--out`. Valid default `[7,103]` and explicitly configured 21-week horizon `[0,20]` with suffix `[7,20]` parse. The numerical 97/104/112 implementation was not changed. Independent pairing `media_smoothness_beta=4` and independent `coupling_manifest='paired.json'` each reject; unsupported generator-level media smoothness also rejects at config preflight.
- Direct `replay_world` with 3 slots vs the 15-slot existing fixture now raises the precise mismatch; source/paired recipe, effect/active ranges, supported latent/likelihood/baseline/smoothness, seed/burn-in/world bounds are checked in `matched.py` before model evaluation. Compatible world IDs 0 and 1 each **actually replayed read-only** against existing fixture RVs with the current adapter: 56 fixed RVs, seven protected outputs and exact original/changed trajectories and paired target. Only the existing two fixture worlds were used. Prior test still perturbs a disposable saved `eps_y` and fails actual replay rather than mocking the evaluator; model/TV/zero-control/fit tests use unchanged numerical code and recorded artifacts.
- `_replay_identity` now contains **30** source hashes, including the five study contract/CLI modules and *all* production generator `.py` (world model, sampler, slots, mechanisms, random walk included), plus six fixture input hashes per world. Independently checked both current `smoke-artifacts-final/matched-replay-fixes/world_{00,01}` validation receipts `complete` under this identity; old matched validation receipts correctly reject as stale. Regression mutates an isolated copy of `world_model.py` and checks the hash changes, without touching production code.

## Evidence split and limitations (important)

Independently compared all **44** frozen `source-snapshot/inventory.json` entries against frozen copy hash/size and current bytes: **only `studies/recovery/config.py` and `studies/recovery/matched.py` differ**. `dgp.py`, `models.py`, `methods.py`, `runner.py`, `artifacts.py`, all production generator `.py`, original notebook, reference config and both smoke TOMLs are byte-identical. Final/old generated corpora of the same four logical smoke worlds remain identical across every 95 NPZ members. Both current smoke configs resolve to the stored `config` of every historical generation and all **14 fit** manifests. All **4 generation + 14 fit** historical receipts and corresponding output hashes still validate under their **recorded** identities; current-source `_world_identity` rejects those checkpoints as **stale**, correctly. Do not recast old fits as current-source-qualified or request another repeat. Current-source evidence is guards/tests, unchanged numerical/input bytes and deterministic replay only; historical diagnostic failures, including varying world1's 68 maxdepth hits, remain historical smoke outcomes, not scientific recovery evidence. Legacy imported saved results still lack source/output certification.

Safe commands repeated:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 studies/recovery/.venv/bin/python -m pytest -p no:cacheprovider tests/test_recovery_task1.py tests/test_recovery_task2.py -q
studies/recovery/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py tests/test_recovery_task2.py
studies/recovery/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py tests/test_recovery_task2.py
```

**22 passed, 2 Numba non-cacheability warnings (30.86s); Ruff clean; 13 files formatted.** Tests do not currently cover the public invalid-replay side effect above; direct `replay_world` rejection alone is insufficient. Fix and re-review in this same context; Task 3 remains unreleased.
