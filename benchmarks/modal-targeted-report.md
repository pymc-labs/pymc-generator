# Bounded Modal-targeted comparison — partial evidence

## Decision outcome

The bounded attempt does **not** support selecting a general generation method. Ordinary generation leads the completed sparse screen despite rebuilding across structures. Dense template arms, full replay completion, repetitions and observed variability remain unresolved. No production throughput, Modal cost/scaling, or deployed-image equivalence is established. No additional execution is authorized by this report.

## Provenance and compiler baseline

Checked the local PFN launcher at `a7cecaa83c8832353c4a88b3f69cf3a34924fc5a`: `scripts/modal/pfn_modal.py` declares `GEN_CPU=1.0`, `GEN_MEMORY_MIB=2048`, Python 3.12, PyMC 6.0.1 and PyTensor 3.0.7. It does not explicitly set compiler or threading overrides. The command calls `scripts/generate_corpus.py`, then `corpora/build.py`, then `DataGenerator.generate_and_save`.

The reviewed generator pin is `90773bb8960a9c7570ece93e293bee66b1fd3a94` in prior-generator. Git-object source hashing over its 17 Python files matches the launcher's reviewed digest `c5ed9dc3684af0e2f4af5a938103a530a1fcbed1b12cac01db79a2dcf9ac0e7b`. Its DataGenerator calls `sample_prior_predictive`, the additive sampler calls `draw_worlds` without a mode override, and the pinned `draw_worlds` defaults to FAST_COMPILE. Thus FAST_RUN is not the baseline; no duplicate compiler arms were run.

The local wheel directory is absent. Neither a wheel nor a deployed image was verified or executed. Measurements use the prior isolated pymc-generator benchmark source at `1b45205f3934d7e26fbdc1895cda2689ab7352b2`, with follow-up harness commit `b7a5528`. They are current-candidate FAST_COMPILE measurements, **not** timings of the pinned prior-generator artifact. Each arm records loaded module path and actual Python/package versions; Python 3.13 locally differs from the launcher's Python 3.12. This mismatch limits transfer to Modal.

## Fixed inputs and lifecycle

Use immutable `comparison-preflight-evidence/workloads.json`, workload 0 (5 treatments, 3 controls) and workload 3 (15 treatments, 10 controls), repetition 0, first two structures and their recorded seeds. Both use 104 weeks, one latent, padded maxima 15/10, Michaelis–Menten saturation and geometric carryover. Original priors/config are asserted equal before changing only the number of selected cells and draw grouping. No replenishment, acceptance changes or scientific-model edits.

The final screen requests exactly 1, 10 and 100 draws per structure, as batches, with two structures per group. Ordinary retains only the current compiled model and clears it at each structure transition; templates retain one compatible model. Replay traverses both structures again, including rebuilding ordinary models, and checks exact active-output hashes. These are sequential worker/cache measurements, not concurrent dispatch tests. The caller may transiently retain the preceding uncompiled model graph while a replacement is built; no multi-model compiled cache is retained.

Fresh independent compiler-cache directories are allocated per method/workload/attempt. Caches warm within an arm, including replay, and are never shared between arms. Order is sparse ordinary/dynamic/specialized then dense ordinary/dynamic/specialized, not randomized. No repetitions were completed, so uncertainty intervals and order effects are not estimated. Host page caches are not flushed.

## Resource and budget controls

Preflight began 2026-10-13 20:34:12 UTC. Absolute full deadline: 20:49:12; execution deadline: 20:47:12, reserving two minutes for reporting. Attempts share these same timestamps and the original 900-second reservation; no previous comparison ledger was reused or reset. Arm ledgers include failures and elapsed times. Execution stopped before its deadline; the scope is inactive.

All benchmark subprocesses ran sequentially in a systemd scope with MemoryMax=2 GiB, MemorySwapMax=0, CPUQuota=100%, TasksMax=64, and affinity to logical CPU 0. All five recorded numerical thread settings were 1. This confines execution to one logical CPU on one physical core, not a demonstration of Modal CPU equivalence. A private 256-MiB tmpfs bounds output/cache scratch, with a 64-MiB output stop threshold. Recorded cgroup peaks are cumulative across arms in a scope, not independent per-arm peaks; self peak RSS is also recorded. No memory-limit failure occurred in observed arms. No paid execution occurred.

## Failures and correctness

Attempt 1: all six arms failed before generation because metadata contained a non-JSON Path. Attempt 2: all six arms reached generation but failed corpus validation; the harness's single-cell corpus cannot satisfy the unchanged requirement for at least one training and one validation world. Source inspection identifies that requirement in `DataGenerator.validate_corpus`. The first assertion did not record the validator's error list, so this cause is additionally supported by the successful two-cell rerun, not a preserved detailed validator error from attempt 2. Both failed attempts are retained, not omitted from the budget.

Attempt 3 uses two structures without weakening validation. Completed groups check causal decomposition, inactive padding, exact same-method replay, corpus validation, lossless array persistence/readback and revalidation. Dynamic 100-draw primary output passed validation/persistence, but replay did not finish before its 147-second arm timeout. Dense ordinary timed out after completed 1/10-draw groups. Dense dynamic and specialized were not launched before the absolute execution stop. Unfinished groups are not correctness passes.

Cross-method controlled-input equivalence was checked by the preceding comparison, **not rerun here**. This follow-up reuses its unchanged scientific adapters but does not treat historical checks as fresh proof. No distributional equivalence or worker-scheduling reproducibility proof is claimed.

## Timing interpretation

JSON records build, compile, draw, invariants/hash, acceptance/packing, normalization/diagnostics, validation, persistence and readback separately. Pipeline time includes generation plus validation, writing and readback, but excludes interpreter/import startup and diagnostic replay. Import/startup, failures and replay remain included in arm elapsed times and the aggregate budget. Pipeline time can exceed the sum of named substages because allocation, disposal/GC and other glue are not separately instrumented.

The initial group includes setup. Later template groups reuse the model; ordinary groups rebuild on both structure transitions. These are not interchangeable cold and warm measurements. Sparse draw cost is approximately proportional across observed 1/10/100 group sizes, but setup does not dominate every method/group: padded template sampling dominates at 100 draws. Do not extrapolate a crossover or linear scaling beyond these observations.

## Smallest next comparison

With a separately approved budget, first complete dense ordinary/dynamic/specialized 1/10/100 groups and dynamic sparse 100-draw replay, then repeat only decision-relevant cases to quantify variability. Explicitly decide whether local candidate evidence is sufficient or whether a verified pinned-image-compatible environment is required. Keep independent caches, one compiled compatible model, unchanged scientific checks and all failures. Method selection remains Teemu's decision in the method-selection ticket.

## Measured groups

Generated table follows; `replay=no` is incomplete evidence, not a pass.

| Arm | Draws/structure | Candidates | Accepted | Rejected | Pipeline s | Replay | Build s | Compile s | Draw s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0-dynamic | 1 | 2 | 2 | 0 | 37.870 | yes | 4.876 | 32.456 | 0.505 |
| 0-dynamic | 10 | 20 | 20 | 0 | 5.207 | yes | 0.000 | 0.000 | 5.072 |
| 0-dynamic | 100 | 200 | 200 | 0 | 52.098 | no | 0.000 | 0.000 | 50.948 |
| 0-ordinary | 1 | 2 | 2 | 0 | 3.991 | yes | 1.000 | 2.738 | 0.025 |
| 0-ordinary | 10 | 20 | 20 | 0 | 3.504 | yes | 0.999 | 1.728 | 0.239 |
| 0-ordinary | 100 | 200 | 200 | 0 | 7.002 | yes | 1.087 | 1.724 | 2.426 |
| 0-specialized | 1 | 2 | 2 | 0 | 16.376 | yes | 1.795 | 14.083 | 0.463 |
| 0-specialized | 10 | 20 | 20 | 0 | 4.749 | yes | 0.000 | 0.000 | 4.605 |
| 0-specialized | 100 | 200 | 200 | 0 | 47.338 | yes | 0.000 | 0.000 | 46.158 |
| 3-ordinary | 1 | 2 | 2 | 0 | 10.177 | yes | 2.403 | 7.385 | 0.066 |
| 3-ordinary | 10 | 20 | 20 | 0 | 10.612 | yes | 2.421 | 6.533 | 0.645 |

Retained evidence before final ledger: 8,508,187 bytes. Completed primary groups: 688 candidates, all accepted; replay is incomplete for the dynamic 100-draw group. Partial draws outside these primary groups are recorded in event logs, not counted as persisted usable output.
