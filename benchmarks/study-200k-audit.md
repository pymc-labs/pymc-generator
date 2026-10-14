# Composable 200k strategy scaffold audit

Research only; no production module edits. Worktree `/tmp/pymc-generator-200k-study`; agent `rich-200k-study / w1F:p3`.

> 🧠 **From Hindsight memory (Conventions and patterns)** — Repository tests distinguish exact seeded replay from numerical oracle agreement at 1e-10. The handoff and inspected benchmark code use those separate gates.

## Contract checked against source

`git diff e74b2bdec56d62c9f2f18c0403ed331a15deb444 -- pymc_generator/presets.py` shows only a docstring difference. The original composable preset, including parameter ranges, is unchanged. Treatment component probabilities are (.9,.7,.15,.1,.2,.2,.3,.3); covariate probabilities are (.9,.6,.1,.1,.15,.3,.35,.3), in hf/pulse/onset/offset/flighting/level_jump/seasonal/trend order. Flags combine independently; `draws_per_cell=1` gives fresh structural instructions per candidate. There is no p=.5 assumption or one-of-N selector here.

Uniform world-shared saturation families: linear, hill, logistic, michaelis_menten, tanh, root. Uniform world-shared carryover families: none, geometric, weibull. Family selections are independent. Active counts are independently uniform over 5–15 treatments and 3–10 covariates; maximum capacity 15/10, one latent, 104 reported weeks, l_max=8 and eight burn-in weeks. Edge budgets in the runner are study assumptions, not production decisions.

The planner uses stable logical IDs, separate count/graph/family/draw streams, and samples families before scheduling. Conditioning pilots on one family pair is explicitly diagnostic, not an accepted-corpus composition policy. Internal flags replace exterior placeholders before acceptance and truth packing. Support-mask seeds use stable logical IDs, not reordered chunk indices. No rejected candidates are replaced.

## Audit changes

- Added the immutable absolute/cumulative-time envelope, 512MiB tmpfs per attempt, isolated caches, 2GiB retained cap, process-file limit, cgroup accounting, process-tree cleanup and durable ledger before the first scientific execution.
- Added runtime edge masking to the research adapter before parameter construction, outcome scaling and scientific graph/reference calculation. The draft relied on the planner zeroing dormant adjacency; explicit masks make the causal contract robust to dormant-edge pollution. Production source remains untouched.
- Fixed normalization/diagnostics stage accumulation across persistence chunks (the draft retained only the last chunk's normalization stage time).
- Expanded separate oracle diagnostics to all-off/all-on/mixed flags, demanded RNG-state advancement and dormant-edge poisoning. Added source digests and family-name/workload metadata.

Additional audit findings:

- The draft packs ordinary worlds into singleton family shards; the unchanged corpus validator requires both training and validation worlds. Ordinary reference cells are now pooled across families; reusable chunks merge a singleton remainder. Acceptance filters and candidate allocations are unchanged.
- A diagnostic that required every pre-optimization RNG seed slot to advance was too strong. The exact ordinary ID-7 probe observed 45 seed slots but 44 compiled samplers; `delta_dy` was optimized out under constant-zero latent→outcome edges. The diagnostic now records pruned seed slots and requires all actual compiled samplers to progress; exact replay and scientific/corpus gates are unchanged.
- Per-chunk explicit GC/progress writes were outside the original pipeline timer. New runs retain that timer's definition and additionally report cleanup-inclusive steady rates and model-release time; earlier full primary/attempt wall times remain available. Pipeline-only figures must not be presented as complete wall throughput.
- A timed-out eager run reached 2700.309620857s cumulative wrapper wall time: a **0.310s cap overrun** from teardown/copy beyond worker reservation. The immutable ledger is retained, eager runs are exhausted, and subsequent launches reserve 5s for outer cleanup. This is disclosed, not rounded away or reset.
- The draft compresses world identities to uint32 seeds, then the existing library wrapper compresses further to a 30-bit PyMC chain seed. A bounded 200,000-ID mapping-only audit (zero worlds generated) found 2 uint32 colliding pairs and 21 downstream colliding pairs, including those 2. Research-only `--seed-contract logical_id` calls `reseed_rngs(ordered_rngs,[root_seed,logical_candidate_id,300])`; all 200,000 first-PCG64 states were distinct and actual reseeding matched reconstructed/reverse mappings. These are shared-seed/stream risks, **not** duplicate complete-world claims. Default remains `legacy`; production source is untouched.

> 🧠 **From Hindsight memory (Key decisions and rationale)** — Default seeded-stream compatibility is a project contract. The full-entropy path is therefore an explicit research opt-in, not a production RNG migration.

The first external/internal eager pilots were loaded from the draft methods before the explicit-edge-mask change; they remain preserved as pilot evidence only. Later attempts record exact source digests. Their all-zero padded dormant adjacency makes this difference inert on candidate inputs, but final comparisons use audited source.

## Envelope / rerun

```sh
bash benchmarks/study_200k_envelope.sh <unique-attempt-name> <seconds> \
  <approach> --phase <pilot|primary|oracle> --candidates <N> --chunk 32 [--pair 0,0]
```

The wrapper refuses cumulative execution above 2700 seconds for an approach, including failures. It enforces the original deadline 2026-10-14 05:11:27 UTC and scientific cutoff 04:56:27 UTC; this historical envelope must not be silently reset to rerun later. A new user-approved envelope is needed after expiration. CPU0/quota100%, memory2GiB, swap0 and numerical threads1 are asserted inside workers. New evidence is in `benchmarks/study-200k-evidence/`; previous evidence is unchanged. Actual systemd controls, cgroup peaks/events/CPU and tmpfs usage are sampled separately from worker RSS. Compiler caches are cold per attempt but can warm within an attempt. Scope-wide termination follows timeout or supervisor interruption. Output copied from bounded tmpfs preserves partial results.

Storage is tmpfs: write/readback validates complete compressed archives and exact arrays, but does not measure SSD/object-store/network throughput or power-loss durability. Persistence chunks can invoke the library's validation-split repair; **raw world/RNG replay**, not identical corpus labels under changed shard boundaries, is the scheduling-invariance contract tested here. The final report distinguishes seed policies, completed gates, failures and unmeasured claims; passing a pilot is not a method recommendation.
