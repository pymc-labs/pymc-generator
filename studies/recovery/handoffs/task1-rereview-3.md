# Task 1 independent re-review, pass 3 — PASS

Read `PLAN.md`, `task1-fixes-3.md`, `task1-rereview-2.md`, current `studies/recovery/{analysis,artifacts}.py` and `tests/test_recovery_task1.py`. **Task 1 accepted at its approved gate**, subject to coordinator release; this does not accept Tasks 2–3 or make a scientific identifiability claim. This handoff is the only file I wrote. Prior reviews and source/saved artifacts are preserved; no notebook execution/import, generation, inference, implementation changes, commits, pushes or delegation.

## Original pass-2 finding: independently verified fixed

`analysis.py:70-75` requires the base's 50 worlds, 2 cells per batch and 1 draw per cell as genuine integers. `analysis.py:201-220` derives exactly 25 expected batch names `batch_00.npz`…`batch_24.npz` from that population, requires equality with TV `source_sha256` / bounded `data_sha256` keys, validates each SHA256 as a lowercase 64-hex string, then digests **every** batch. This is not merely a check of supplied map entries.

Independent `tempfile.TemporaryDirectory()` probe: copy original base, TV data, TV fit, and bounded fit manifests to sibling layout; symlink all 25 *unchanged original* base batch files. Both authentic complete TV and bounded manifests passed `_legacy_compatible`. For **each** format, reducing the hash map to only the genuine `batch_00.npz` entry rejected with `ContractError` **before** and **after** replacing only the temporary `batch_01.npz` symlink with corrupted bytes (exact prior reproduction). Extra `batch_25.npz`, extra `oops`, omitted `batch_24.npz`, empty/null/list inventories rejected. Full map with corrupted `batch_01.npz` rejected on hash mismatch; restoring the authentic input passed again. Short, uppercase, non-hex or non-string hash values rejected. Mutating base worlds to 49 or `true`, cells per batch to 1 or draws per cell to 2 rejected. No original file was modified. `tests/test_recovery_task1.py:test_legacy_input_hash_inventory_is_complete` covers the main TV/bounded cases.

Prior forged-receipt reader, TV metadata/complete contiguous generation-and-fit seed checks, config rejection, saved windows/format, and stale-cache behavior remain exercised in the full focused suite. `artifacts.py:137-159` still compares completed receipt keys against identity and disk inventory and verifies bytes; `analysis.py:110-187` still rejects contradictory recorded legacy TV recipe/seed inventories. Nothing in this pass implies inference or report adapters are implemented.

## Repeated commands/results

```sh
PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m pytest -p no:cacheprovider tests/test_recovery_task1.py -q
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py
/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py
PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m studies.recovery analyze --config studies/recovery/saved_tv_example.toml --saved notebooks/time_varying_baseline/nuts --format tv_nuts
```

**12 passed in 0.42s**, Ruff all checks passed, **6 files already formatted**. Successful CLI output retains `provenance: legacy manifest/input-hash compatibility only; output and source hashes unavailable` and reports TV valid 50, media 11, joint 9, total media 36, media failures 144/236 and control failures 19/148. The pass-2 exact subset/corrupted-unlisted-input script, altered to expect `ContractError` for both TV and bounded, also rejected all three former false accepts.

Independently rescored all 50 worlds from saved raw JSON/NPZ **without** calling `studies.recovery.analysis` scoring: channel L2 error <= `max(.1 * L2(true channel), .01 * L2(sum true components))`, aggregate total-media trajectories analogously, diagnostic/optimizer health gating and failures only among nonzero-truth channels. All paths have 97 scored rows. **TV:** 50 valid/0 unresolved, media 11, joint 9, total media 36, media failures 144/236, controls 19/148. **Bounded estimated:** 48 valid/2 unresolved, media 13, joint 13, total media 28, media failures 122/211, controls 15/135. Saved 10%/1% world joint gates agree **50/50 for each format**. This verifies read-only arithmetic, not successful parameter identification.

## Explicit legacy limit and next step

Original saved manifests and checked input bytes are *compatibility evidence*: they cannot authenticate old per-result outputs, historical source bytes or the relationship between saved TV results and their source. A fabricated matching manifest and fabricated bytes are not a trusted signature. The output label remains appropriately narrow; do **not** certify old runs as new completed checkpoints. Task 1 passes its scoped contract despite this inherent legacy limitation. Parent may release Task 2 separately after the prescribed between-task handoff/compaction; this reviewer pauses now.
