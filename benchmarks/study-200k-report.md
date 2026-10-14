# Bounded 200k composable-generator comparison

**Research evidence; parent owns method selection. No production integration or RNG migration.**

> 🧠 **From Hindsight memory (Bounded local generation-comparison evidence)** — The comparison is an isolated benchmark prototype and decision gate, not production integration; earlier reuse results are not fresh-structure performance claims. Current measurements below use the new one-draw/fresh-structure contract.

## Contract and scope

Fresh structural cell per world, one draw per cell. Media counts independently uniform 5–15; controls 3–10; capacities 15/10; one latent; 104 reported weeks; adstock `l_max=8`, burn-in 8. Saturation is independently uniform over linear/hill/logistic/Michaelis–Menten/tanh/root and adstock over none/geometric/Weibull; each chosen family is shared across media within a world, with channel-specific parameters. No accepted-family quotas or replenishment.

Carlos's original `composable` preset and ranges are retained. Component order is HF, pulse, onset, offset, flighting, level jump, seasonal, trend. Treatment inclusion probabilities: .9/.7/.15/.1/.2/.2/.3/.3; covariate: .9/.6/.1/.1/.15/.3/.35/.3. Forced off/on/mixed cases are diagnostic oracles, **not** a p=.5 benchmark prior.

Ordinary builds a fresh model per world. Universal and family-specialized research prototypes reuse fixed-capacity graphs, with outside-graph or inside-graph flag sampling and eager/lazy deterministic evaluation. Active masks apply before adjacency propagation, parameter/reference/noise scaling, not only output cropping. Each worker retains one compatible model, not an unbounded bank.

All scientific runs use `study_200k_envelope.sh`: local CPU0, CPU quota one core, numerical threads one, memory 2GiB, swap zero, isolated 512MiB tmpfs scratch/cache, retained new evidence ceiling 2GiB. Absolute deadline 2026-10-14 05:11:27 UTC; scientific cutoff 04:56:27; approach budget 2700s cumulative, including failures.

## Evidence map and reproduction

- `study-200k-evidence/ledger.json`: immutable charged attempts, caps, exits and artifact sizes.
- Each attempt: source snapshots, command/environment, cgroup accounting, process/events logs, raw instructions/input arrays, and results/progress where available.
- Completed primary: full compressed corpora, array/dtype/shape/byte readback equality, corpus validation, actual flag metadata, rejected-candidate reasons, raw candidate hashes, compilation/graph/RNG-work records.
- `rebuild.json`: **separate post-primary** new-model reverse-order raw-byte replay; excluded from primary timing/RSS.
- `study-200k-summary.json`: read-only aggregation, descriptive forecasts and component/family counts; regenerate with `python benchmarks/study_200k_report.py`.
- `study-200k-evidence/artifact-manifest.json`: SHA-256/byte manifest of retained evidence.
- `study-200k-audit.md`: implementation audit and corrections.

Canonical bounded command (use a new name; no overwrite):

```bash
bash benchmarks/study_200k_envelope.sh NEW_NAME 900 universal_internal_lazy \
  --mode cvm_compile --phase primary --candidates 4096 --chunk 128 \
  --cohort-size 2048 --seed 631228 --seed-contract logical_id --rebuild-check
```

The envelope contains the study's immutable timestamps; after this window it intentionally refuses scientific work. A future independent study needs a separately approved envelope, not a silent clock reset. `--pair` is conditional diagnostic only; unconditional primaries draw naturally selected families/counts.

## Correctness and seed contract

> 🧠 **From Hindsight memory (Conventions and patterns)** — Project tests distinguish numerical oracle tolerance (1e-10) from same-environment seeded dtype/shape/raw-byte identity, and prefer executable contracts. The checks below implement that distinction; their pass status comes from the current retained evidence, not memory.

Ordinary agreement uses common primitive values and checks every scientific corpus field at rtol/atol 1e-10. All 18 family combinations receive random/off/on/mixed cases and dormant-edge poisoning. Primaries additionally check finite values, decomposition, active tails, refreshed in-graph flag metadata, acceptance/normalization, complete serialization/readback, demanded compiled RNG progression and exact reseeding. Archive-based oracle completion proves deterministic physics/masks but explicitly reports **no fresh RNG progression**; fresh primary diagnostics cover actual RNG work.

Seed audit: root 631227, logical IDs 0–199999, **zero worlds generated**. Draft uint32 world seeds: 199998 unique, 2 colliding pairs. Actual downstream 30-bit PyMC compile seeds: 199979 unique, 21 colliding pairs, including the first two. These are shared-seed/stream risks, **not duplicate complete-world claims**: differing families/structures can yield distinct complete outputs.

Research `logical_id` uses `reseed_rngs(ordered_rngs,[root_seed,logical_candidate_id,300])`, with no worker/shard index. All 200000 first-PCG64 states were distinct in this audited set; rebuilt/reverse mapping matched the actual API. This is an empirical check, not an infinite-domain collision guarantee. `legacy` remains the default; every new cohort records its policy/root. The already-running 1024-candidate eager cohort remained legacy and is preserved separately. Do not attribute cross-policy acceptance/performance differences solely to execution mode.

> 🧠 **From Hindsight memory (Key decisions and rationale)** — Default seeded-stream compatibility is an established project contract. The full-entropy path is an explicit research opt-in; production RNG code remains unchanged.

Raw-world replay is invariant to the tested reordered scheduling/new-model rebuild. **Normalized corpus bytes and train/validation labels are not promised invariant under changed persistence shard boundaries**: the existing split repair can depend on shards.

## Interpretation boundaries and failure accounting

- `cvm_compile` is CVM with fast-compile rewrites; it is not FAST_RUN. Modes/seed policies/source versions are recorded per attempt.
- Original verified pipeline timers cover generation, acceptance/packing/normalization, validation, compression/write and complete readback. They exclude explicit post-chunk GC/progress logging and diagnostic replay. New runs additionally time cleanup-inclusive steady service and model release; full primary and complete attempt wall time are retained. Pipeline-only rates must not be called end-to-end wall throughput.
- First-of-family cohorts include one-off release/turnover overhead; resident later cohorts separate that effect. Forecasts require amortizing fixed setup/lifecycle and stating persistence granularity. Do not extrapolate the first 1024-world average as if every 1024 worlds paid a new 18-family compile bank.
- tmpfs I/O proves complete archives/round trips, not SSD/object-store/network throughput or crash durability. Parallel workers, paid Modal and production runs are unmeasured.
- This research runner materializes its **bounded** instruction set. A 200k production corpus requires streaming planning/storage and validated worker routing; the runner is not a demonstrated 200k, 2GiB production service.
- Legacy ordinary family-group persistence produced singleton shards and failed the unchanged train/validation corpus gate. Ordinary now pools families; reusable batches merge singleton remainders. Priors/acceptance/rejected-world replacement are unchanged.
- A too-strong ordinary RNG assertion failed after persistence/readback. Exact ID-7 probe: 45 original seed slots, 44 compiled samplers; `delta_dy` (shape 1) is optimized out under constant-zero latent→outcome edges. Diagnostic now records pruned slots and requires **all actually compiled samplers** to advance, plus output progression/replay. Corrected ordinary reference passes.
- Initial universal uncached oracle was stopped because rebuilding/recompiling per family exhausted useful diagnostic time; cached evaluator evidence supersedes it without deleting the partial attempt.
- Specialized diagnostic timeouts retain passed per-family progress; bounded missing-pair completions count toward the same approach budget. Only the union of all 18 completed cases permits a primary.
- Logical eager primary completed but its separate rebuild diagnostic was capped; retain that incompleteness. The earlier legacy eager rebuild passed.
- Universal eager charged wrapper time reached **2700.309620857s**, a **0.310s cap overrun** due to outer timeout teardown/copy beyond worker reservation. Ledger is preserved; no further eager work is permitted. Subsequent launches reserve 5s for outer cleanup. Other limits are not silently relaxed.
- Queue startup lock races/phase configuration errors are preserved as failures, not scientific evidence or successful measurements. No budget/deadline resets.

## Results and recommendation

### Completed CVM evidence

All rows below use CVM with fast-compile rewrites. Rates are accepted worlds/s; “resident” means the final fresh-ID cohort after first-per-family setup. Cleanup is explicit GC/progress logging. RSS is the scientific worker's primary peak, not cgroup total or the later rebuild.

| Method / cohort | Accepted / candidates | Build+compile s | Resident verified pipeline/s | Including cleanup/s | Primary RSS MiB | New-model replay |
|---|---:|---:|---:|---:|---:|---|
| Ordinary fresh model, logical ID | 8/8 | 330.48 (8 fresh builds) | **0.02391 complete service/s** | not separately timed | 424.61 | in-model replay; fresh model each candidate |
| Universal internal eager, legacy | 1015/1024 | 296.80 | 20.19 | not timed | 672.98 | pass |
| Universal internal eager, logical ID | 507/512 | 336.97 | 20.42 | not timed | 648.76 | **incomplete: capped** |
| Universal internal lazy, logical ID | 1016/1024 | 313.84 | 20.51 | not timed | 656.06 | pass |
| Specialized external eager, logical ID, chunk32 | 1016/1024 | 1041.38 | 24.85 | not timed | 591.79 | pass |
| Specialized external lazy, logical ID, chunk32 | 1016/1024 | 930.23 | 28.29 | not timed | 586.49 | pass |
| Specialized internal eager, logical ID, chunk32 | 1017/1024 | 978.45 | 24.57 | 12.80 | 598.20 | pass |
| Specialized internal lazy, logical ID, chunk128 | 4070/4096 | 986.96 | 24.89 | **20.47** | 681.80 | pass |
| Universal internal lazy, independent root631228, chunk128 | 4067/4096 | 309.19 | 20.20 | **16.37** | 756.05 | pass |

Specialized external eager/lazy matched **all 1024 raw candidate hashes and all corpus archive hashes**, with identical accepted IDs. Specialized internal eager/lazy matched the first 1024 raw candidate hashes despite changed persistence granularity. Thus neither requires a physics change for the observed results. External eager/lazy timing comparisons share policy and artifacts; the internal cleanup-rate gain **cannot** be attributed solely to laziness because chunk size/cohort size changed.

Universal lazy's independent-root/process repeat has two2048-ID cohorts: 20.59/20.20 pipeline and16.80/16.37 cleanup-inclusive accepted/s. Specialized internal lazy's corresponding two2048-ID cohorts: 19.22/24.89 pipeline and16.26/20.47 cleanup-inclusive; first-per-family release cost is22.64s (once for18 transitions), explaining most of its first-cohort penalty. Its 4096-candidate actual primary wall time is1232.24s (including setup/diagnostics/logging), versus573.11s for universal's independent repeat. This is the important small-corpus versus amortized-large-corpus tradeoff.

### Conditional 200k accepted-world projection

Under the **measured chunk128 local pipeline**, amortizing setup/release once and using resident cleanup-inclusive rates, specialized internal lazy projects about**3.00h** for200k accepted worlds on one CPU versus about**3.48h** for universal internal lazy (roughly29min difference). These are conditional forecasts, not a demonstrated200k service, and exclude disk/upload/parallelism. Do not substitute the faster pipeline-only rate while silently omitting cleanup. The report/summary retains full wall time so different deployment accounting can be recomputed.

Specialized internal lazy acceptance is4070/4096 (99.365%; Wilson95%99.071–99.566%), giving about201278 candidates for200k accepted at that empirical rate. Corpus-only compressed size projects about**4.60GiB**; sidecars, caches and real storage costs are additional. The8-world ordinary reference's334.64s complete service demonstrates fresh compile domination; its roughly97-day linear extrapolation is **not** a validated long-run ordinary forecast.

Actual specialized-lazy family counts (candidate→accepted; saturation rows in documented order, adstock columns none/geometric/Weibull):

| Saturation | none | geometric | Weibull |
|---|---:|---:|---:|
| linear |247→246|249→247|224→224|
| hill |219→219|234→232|226→223|
| logistic |226→224|240→238|223→221|
| Michaelis–Menten |218→216|238→238|210→210|
| tanh |205→203|239→239|242→240|
| root |183→181|249→246|224→223|

No quota/top-up was used. Channel-weighted component fractions before→after acceptance, in the component order above:

- Treatment: `.9006/.6986/.1488/.1014/.2042/.2029/.2987/.2988` → `.9008/.6987/.1488/.1015/.2042/.2029/.2988/.2986` (40922→40683 active channels).
- Covariate: `.9014/.5998/.0976/.1013/.1499/.3033/.3518/.3011` → `.9014/.5999/.0978/.1012/.1499/.3028/.3521/.3009` (26757→26575 active channels).

### Recommendation boundary

Pending the final explicit-CVM FAST_RUN diagnostic, the **best completely measured cleanup-inclusive large-corpus contender is specialized internal lazy with chunk128**. Universal internal lazy is the simpler, cheaper-startup fallback with independent-root repetition. Specialized external lazy has a faster resident pipeline but lacks a completed chunk128 cleanup-inclusive repeat within its approach budget; calling it the complete end-to-end winner would be inference, not measurement. The parent should select the method and RNG migration policy only after considering these caveats; #58 remains open.

### FAST_RUN dispatch failure and remaining diagnostic

The first named `FAST_RUN` attempt actually recorded **NumbaLinker**, not CVM. Its first runtime JIT hit the hard2GiB OOM limit (systemd journal retained); this is **not evidence that CVM FAST_RUN failed**. Research selection now explicitly constructs CVM with fast-run rewrites. That separate bounded diagnostic is still pending at this report snapshot; any incomplete/failed result must remain explicit, and no primary may run before its full18-case oracle passes.
