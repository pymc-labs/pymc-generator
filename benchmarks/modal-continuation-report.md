# Modal-targeted continuation: completed bounded screen

## Outcome

All seven planned arms and 21 primary groups completed, including exact replay, under the additional 20-minute envelope. The sparse dynamic replay gap and missing dense 1/10/100-draw comparisons are closed. Dense methods were each measured twice with the same inputs in fresh processes and independent empty compiler caches, with the second pass in reversed method order.

**Bounded recommendation for the method decision:** ordinary generation is the stronger candidate for these sparse workloads and for dense instructions producing 100 draws per structure. Family specialization remains a meaningful candidate for dense, low-draw instructions when its initial setup can be amortized across enough structures. Dynamic generation has no observed timing advantage over specialization in this screen. Do not select a universal winner, a switching threshold, or production-optimal resource settings from this evidence. The method decision remains Teemu's.

## Headline dense timings

Times below are seconds for **two structures**, after generation, validation, persistence and readback. Initial one-draw groups include setup. Ten/hundred-draw groups reuse compatible templates; ordinary rebuilds at both structure transitions. Ranges are two observed runs, **not confidence intervals**.

| Draws per structure | Ordinary | Dynamic template | Family-specialized |
|---|---:|---:|---:|
| 1, initial setup included | 10.151–10.162 | 37.654–38.054 | 16.444–17.101 |
| 10, template already built | 10.652–10.778 | 5.419–5.640 | 4.914–4.924 |
| 100, template already built | 19.297–19.831 | 51.311–54.486 | 49.372–49.568 |

Thus specialization is about 2.2× faster at the measured warm 10-draw point, whereas ordinary is about 2.5× faster at 100 draws despite its rebuilds. Ordinary also has lower initial setup for these two structures. The effects exceed observed repeat variation, but two repeats cannot quantify tail latency or broad uncertainty. The dynamic 100-draw range is about 6% of its midpoint; other dense pipeline ranges are smaller.

The [stage table](modal-continuation-tables.md) and [machine-readable summary](modal-continuation-evidence/summary.json) distinguish build/compile/draw and retain validation/persistence timings. In dense ordinary, build+compile stays near 9 seconds for two structures; sampling rises from roughly 0.065 to 0.65 to 6.5 seconds at 1/10/100 draws. Padded methods draw roughly 0.46–0.53, 4.5–5.3 and 46–51 seconds. Approximately linear sampling is consistent with these three points, not imposed. Compilation does **not** dominate padded 100-draw groups. No fitted or measured crossover threshold is claimed.

## Protocol and provenance

This continues the [original partial report](https://github.com/pymc-labs/pymc-generator/blob/3f7c81b8222a6e0eb0bd8760d11e0767e5693431/benchmarks/modal-targeted-report.md), preserving all original failures, artifacts and budget records. New harness commit: `824d4b3`. Scientific modules, the fixed-allocation scientific packing/acceptance code and model adapters are unchanged from the preceding benchmark. Only continuation orchestration and resource-assertion metadata were added.

Inputs remain manifest workloads 0 (sparse 5/3) and 3 (dense 15/10), first two predetermined structures, seed set 0, 104 weeks, one latent, padded maxima 15/10, Michaelis–Menten saturation and geometric carryover. Batch sizes are exactly 1/10/100 per structure. No rejection replenishment, science changes or additional compiler arm. Each arm records exact config, input manifest hash, software versions, source location and compiler/cache/thread settings. Repetition uses identical structures/seeds; it measures timing variability, not additional independent scientific coverage.

The pinned Modal generator source already defaults to FAST_COMPILE. Local measurements remain current benchmark-candidate FAST_COMPILE evidence, not execution of the reviewed prior-generator wheel or deployed image. The wheel is absent, deployed image unverified, and local Python 3.13 differs from the checked launcher Python 3.12. Prior provenance limitations remain in force.

Each worker retains one compatible compiled model. Ordinary discards it on structure changes; templates reuse one compatible model. The two structures recur across groups/replay; ordinary compiler disk caches therefore warm. This is a concrete turnover probe, **not** an unbounded stream of never-before-seen structures. The brief uncompiled graph-overlap caveat from the original report remains. No multi-model compiled cache is retained.

## Correctness and counts

- Seven arms passed; all 21 groups passed exact same-method replay, causal and inactive-padding checks, corpus validation, lossless array persistence/readback and revalidation.
- 1,554 primary candidate evaluations were accepted, zero rejected. These are **evaluations**, not 1,554 unique new worlds: fixed identities repeat across timing repetitions and grouping experiments. Replay is additional diagnostic work, not counted as primary output.
- Nine complete-archive SHA-256 equality checks passed between dense repeats, plus five against overlapping original-envelope archives, including the formerly replay-incomplete sparse dynamic 100-draw archive. Fourteen archive identity checks total.
- Cross-method controlled-input equivalence was not rerun in this follow-up; the preceding comparison checked it with unchanged scientific adapters. This is not fresh proof of distributional equivalence, nor a scheduling/retry integration test.
- No memory failures, numerical failures or timeouts occurred in the continuation. Original-envelope failures remain visible in their own report.

## Envelope and resource evidence

Approved additional envelope began 2026-10-13 20:54:29 UTC; hard full deadline 21:14:29. Execution deadline 21:11:29 reserved three minutes for analysis/publication. All seven arms finished at approximately 21:10:44; the scope was confirmed inactive. The [arm ledger](modal-continuation-evidence/arms.json) includes startup and replay. The [final ledger](modal-continuation-evidence/budget-final.json) includes elapsed preflight, execution and report work to its recording time. The original 15-minute ledger is neither reused nor reset.

Runtime assertions verify MemoryMax=2 GiB, MemorySwapMax=0, CPU quota=100%, affinity to logical CPU 0 and one numerical thread. Execution is sequential. The maximum recorded self peak RSS is 567,196 KiB (~554 MiB); maximum recorded cumulative scope peak is 566,026,240 bytes (~540 MiB). RSS and cgroup accounting differ; scope peak is cumulative, not a per-arm incremental memory measurement. These observations do not justify lowering the production allocation or claim equivalence to Modal CPU performance.

A private 256-MiB tmpfs bounds output/cache scratch; the output stop threshold remains 64 MiB. Retained artifact size is recorded in the final ledger. No production changes, paid execution, PR or merge.

## Handoff boundaries

The approved missing-case and repetition work is complete; stop rather than spend the remainder. The method-selection discussion can now weigh expected draws per structure and structure turnover against the measured setup/runtime trade-off. Other saturation/carryover families have earlier local evidence but were not repeated here. A verified target-environment run, broader unseen-structure stream or additional correctness/performance investigation needs a separately approved envelope. Consumer-pipeline validation and Carlos's contract approval remain separate requirements; these timings are not final usable production throughput or dollars-per-world.
