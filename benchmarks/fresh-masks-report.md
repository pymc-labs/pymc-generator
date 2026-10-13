# Fresh uniform counts and masks per world — local smoke test

## Result

**PASS:** 24 worlds with independently sampled treatment/control counts reused one fixed-capacity family-specialized compiled sampler. There were 21 distinct active-count pairs; all 24 candidates passed acceptance, corpus validation, lossless save/readback and exact reversed-order replay. No compilation occurred when a count, graph, smoothness input or activity mask changed.

This establishes feasibility for this supported configuration, not a universal performance result or a production implementation. Counts are sampled in NumPy per logical candidate and supplied as runtime inputs; they are **not** PyMC random variables in this test. Full-size arrays are evaluated with masks inside the causal calculation, **not** generated unmasked and then deleted afterward.

## Configuration and count contract

- Maximum capacities: 15 media channels, 10 controls, 1 latent; 104 reported weeks and 4 burn-in weeks.
- Independent discrete-uniform counts: media **5 through 15 inclusive**, controls **3 through 10 inclusive**. Each candidate has its own count stream, graph stream and draw seed derived from its stable candidate ID. Reconstructing its inputs is exact and independent of traversal order.
- Prefix activity masks (`slot_index < count`) preserve the existing slot/topological-order convention. This tests uniform **counts**, not a uniform random subset of slots. Independent Bernoulli masks and subtype quotas are not used.
- One Michaelis–Menten saturation family and one geometric carryover family shared across media channels. Numerical parameters still vary by channel and world.
- Fresh graph sampled against each world's active counts, including nonzero control→media, media→media and control→control edge budgets. Zero latent-edge budgets and the baseline prior's other scientific settings remain fixed. New count draws occur for every candidate rather than once per multi-draw group.
- One candidate per cell in the existing fixed-allocation harness; 24 cells are generated sequentially. Inputs are prepared before drawing, with preparation time recorded. This is not vectorized varying-mask generation in one batched PyMC call.

The exact count API is checked at both endpoints: integer RNG calls are `(5,16)`, `(3,11)` and `(1,2)`, with an exclusive upper bound. A separate 10,000-count diagnostic records the joint/marginal histograms without demanding equal finite-sample bin counts. The distribution follows the integer-uniform sampling rule; a histogram alone is not a proof of uniformity. All 24 tested candidates were accepted; rejection could still distort accepted-world composition in another workload.

## What was verified

1. Masks have the sampled counts and fixed maximum shapes, and all media share the configured family pair.
2. Model build and compile events both occur exactly once in the passing attempt. The exact same compiled function serves every primary world, reversed replay and diagnostic variant.
3. Existing adapter checks verify causal decomposition and zero inactive-slot outputs before cropping.
4. All 24 primary worlds replay byte-exactly in reverse order after intervening draws. Reconstructing each candidate's input payload also agrees exactly.
5. The corpus passes the unchanged validator, and all array dtypes, shapes and bytes survive save/readback. The primary corpus produced before the first diagnostic failure also matches the passing attempt's complete archive SHA-256.
6. Two diagnostic graph representations agree at rtol=atol=1e-10: mask removed nodes and zero their incident edges, versus keep those nodes alive but fully disconnect them. Retained trajectories, contributions and sales agree, and masked node trajectories are exactly zero.

Incident edge masks matter too: activity must be reflected in direct/mediating graph edges so graph-dependent reference levels and outcome scale calculations see the active configuration. Production candidate inputs use the existing active-count-aware graph sampler, which already excludes inactive edges. The synthetic diagnostic explicitly masks edge rows/columns.

## Why posthoc deletion is not equivalent

A diagnostic removes media slot 0 and control slot 0, using the same compiled function, seed, underlying random parameter draws and innovation streams for each variant. Derived baseline/noise scales can change with the graph. These are deliberately constructed counterexamples, not additional prior candidates.

| Diagnostic | Max absolute sales error from subtracting removed contributions | Error after additionally substituting the correctly masked baseline/noise |
|---|---:|---:|
| No causal interactions | 0.03028246 | 3.55e-15 |
| Z0→C1, C0→C1 and Z0→Z1 | 0.11321566 | 0.11021169 |

The first discrepancy exposed an extra dependency: in the current **relative outcome scale** mode, baseline-walk and outcome-noise standard deviations are multiplied by `sqrt(sum((g_cy * beta)**2))`. Removing a media channel changes that anchor. The second discrepancy remains after accounting for this scaling because retained variables still carry the removed nodes' causal effects in the unmasked simulation.

The baseline/noise substitution is diagnostic attribution using the masked reference, **not** a proposed posthoc algorithm. Correct computation evaluates baseline/noise scaling and causal propagation against the masked graph before producing final sales and labels. Merely recomputing sales from selected columns is insufficient for the current DGP.

The first attempt expected the no-interaction case to match without this scale adjustment and failed that assertion. Its evidence is retained. The corrected diagnostic identifies the scale dependency instead of changing priors, disabling relative scaling, loosening tolerances or weakening scientific checks. Both attempts used the same absolute work budget; primary generation and reversed replay already passed in the first attempt.

## Timings and resources

Passing attempt, seconds:

| Stage | Seconds |
|---|---:|
| Prepare 24 fresh world inputs | 0.020 |
| Build model | 1.761 |
| Compile | 13.690 |
| Draw 24 worlds with fresh runtime masks | 4.748 |
| Validation | 0.070 |
| Persistence | 0.028 |
| Readback and revalidation | 0.086 |
| Entire primary pipeline including setup | **20.509** |
| Exact reverse-order replay of all 24 worlds | 4.643 |

Warm sampling is approximately **0.198 seconds/world (5.1 worlds/second)** here; it excludes setup and downstream stages. The primary pipeline includes preparation, generation, validation, writing and readback, but not interpreter startup or diagnostics. Only one passing timing run is reported, with a retained failed diagnostic attempt; this is a smoke test, not a performance comparison or uncertainty estimate. Do not compare directly against the previous direct-effect-only fixed-mask benchmarks as if workloads were identical.

Resource assertions verify one logical CPU, 100% CPU quota, 2-GiB memory maximum, no swap and one numerical thread. Maximum recorded self RSS was 406,372 KiB (~397 MiB), scope peak 470,798,336 bytes (~449 MiB), with no OOM/max-memory events. Scratch output/cache is hard-bounded by a 128-MiB tmpfs. The execution scope is inactive.

The self-imposed 10-minute envelope began 2026-10-13 22:20:09 UTC, with an execution deadline of 22:27:09 and total deadline 22:30:09. Both attempts are included; no previous envelope was reset. Actual elapsed time and artifact size are in [the final ledger](fresh-masks-evidence/budget-final.json). No paid Modal calls, production edits, PR or merge.

## Artifacts and limits

- [Passing results](fresh-masks-evidence/attempt2/result.json), [exact world inputs](fresh-masks-evidence/attempt2/world-inputs.json), [count contract/histograms](fresh-masks-evidence/attempt2/count-contract.json), [environment](fresh-masks-evidence/attempt2/environment.json), [events](fresh-masks-evidence/attempt2/events.jsonl).
- [Saved corpus](fresh-masks-evidence/attempt2/fresh-worlds.npz), diagnostic full/masked/reference/posthoc sales arrays in the same directory, and [cross-attempt archive check](fresh-masks-evidence/archive-check.json).
- [Initial diagnostic failure](fresh-masks-evidence/process.log) is retained separately.
- Passing harness source commit: `45f6db2`; failed diagnostic source commit: `29357d5`. Scientific modules and the existing benchmark model adapters are unchanged.

This does not verify other family pairs, sampled family changes, GPU/Modal execution, production publication/retry handling, or an in-graph discrete-uniform RV implementation. It preserves the established active-prefix convention and tests one supported baseline configuration. The result supports proceeding with per-world runtime masks and active-graph-aware computation; it does not select a universal generation method.
