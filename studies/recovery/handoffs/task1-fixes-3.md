# Task 1 fix pass 3 — complete legacy input inventory

Read `task1-rereview-2.md`. Preserved every prior handoff and original notebook/artifact; changed only `studies/recovery/analysis.py`, `tests/test_recovery_task1.py`, and this handoff. No generation, inference, notebook import/execution, production generator edits, commits, push or Task 2.

## Finding and resolution

Before changing the reader, the new parametrized negative regression failed **for both TV and bounded**: a one-entry hash map was accepted. The `_legacy_compatible` import boundary now requires the base manifest's recorded 50 worlds, two cells per batch and one draw per cell (integers, not bools); derives the complete 25-name set `batch_00.npz`…`batch_24.npz` from these counts. TV `source_sha256` or bounded `data_sha256` must be exactly that set: null, empty, subset, extra or differently named inventories are rejected **before** any hashes are trusted. Each value must be a lowercase 64-hex SHA256; all 25 existing inputs must match their recorded digests. No hashes were invented. A changed hash and matching mutated bytes can still be fabricated in untrusted legacy manifests; existing output/source-hash certification remains explicitly unavailable in the report.

The regression uses copied manifests and symlinks to unchanged originals in a temporary sibling layout for **both** formats. It establishes authentic complete inventories pass; subset-map rejection both before and after replacing only temporary `batch_01.npz` with corrupted bytes; extra, malformed and absent inventories reject; full inventory detects corrupted input via hash mismatch; restored genuine batch and manifests pass. Also checks altered base `worlds`, `cells_per_batch`, or `draws_per_cell` reject. Source artifacts stay read-only.

Analogous omission review of this same import boundary: TV embedded and standalone data manifests are compared in full; generation and NUTS seed lists must match the complete recorded population; `saved_summary` reads every requested world by index and cannot silently skip a missing report/array; completed checkpoint reader compares receipt, identity and on-disk output inventories. The TV and bounded batch maps were the remaining pair of optional supplied-entry-only hash loops; both now share the same complete inventory check. This is a metadata/input-hash consistency gate, not legacy output certification or Task 2 resume.

## Exact evidence

- Initial reproduction: `PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m pytest -p no:cacheprovider tests/test_recovery_task1.py -k legacy_input_hash_inventory -q` → **2 failed, 10 deselected** (both truncated maps accepted before fix).
- Replayed disposable-layout subset-map + corrupted-unlisted-`batch_01.npz` probe from `task1-rereview-2.md` for TV and bounded with explicit expected exceptions: TV subset, TV subset + corrupt, and bounded subset + corrupt each raise `ContractError: legacy dataset provenance inventory must contain exactly the recorded base batches`; restored unchanged complete TV and bounded accepted.
- `PYTHONDONTWRITEBYTECODE=1 /home/teemu/pymc-labs/prior-generator/.venv/bin/python -m pytest -p no:cacheprovider tests/test_recovery_task1.py -q` → **12 passed in 0.41s**, including prior regressions and 50-world TV/bounded saved parity.
- `/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff check studies/recovery tests/test_recovery_task1.py` → **All checks passed**; `/home/teemu/pymc-labs/prior-generator/.venv/bin/ruff format --check studies/recovery tests/test_recovery_task1.py` → **6 files already formatted**.

Await independent re-verification and coordinator release. No Task 2.
