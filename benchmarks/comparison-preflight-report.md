# Local generation-comparison preflight — partial evidence

## Status

**Preflight passed; the comparison task is not complete. No method recommendation.**

Protocol: [Approve the generation-comparison protocol and bounded work envelope](https://github.com/pymc-labs/pymc-generator/issues/55#issuecomment-6149847169).
Task: [Collect the approved bounded generation-comparison evidence](https://github.com/pymc-labs/pymc-generator/issues/56).

Baseline: `b941d3a222fb91475cc2b59db14281e2b6498c49`.
Executed instrumentation: `463c684` (full hash is in `comparison-preflight-evidence/environment.json`).
The committed scripts are a preliminary invocation, not the finished benchmark harness.

## Reproduce this invocation

From the isolated research worktree, with the locked environment installed:

```sh
uv sync --locked
bash -n benchmarks/comparison_preflight.sh
.venv/bin/python -m py_compile benchmarks/comparison_preflight.py
bash benchmarks/comparison_preflight.sh /tmp/generation-comparison-evidence-NEW
```

The output directory must not exist. **Do not rerun without charging it against the
shared two-hour work envelope.** The script enforces a single invocation only;
it is not an aggregate-budget supervisor and must not be used in an unbounded
loop. No paid execution is authorized.

## Workloads pinned before observing draws

`workloads.json` stores all 12 approved workloads, three seed sets, and five
concrete structures per workload/seed set, including full resolved priors, DAGs,
realized families, runtime template inputs, and draw seeds. Thus 180 structures
are prepared for reuse by all three methods and both draw groupings. The primary
allocation remains 5,940 draws. **None of those primary draws has been executed.**

Seed procedure: workload index follows protocol order; seed-set repetition is
0, 1, or 2; configuration seed is `999000 + 1000 * workload_index + repetition`.
The pinned baseline's `make_workload` generates the five concrete structures and
draw seeds. Groupings 1 and 10 reuse those inputs but remain distinct replay
configurations; the manifest's configuration uses `draws_per_cell=1`, to be
explicitly replaced with 10 for that grouping. No outcomes informed seed choice.

All resolved configurations pass `SCMPrior.validate` and the existing template's
configuration-level support check. This is **not** numerical validation of all
workloads or support evidence for an unbuilt specialized candidate.

## What ran

One dynamic-template model at padded capacity 15 media / 10 controls / 1 latent,
104 weeks, active counts (5,3), geometric carryover and Michaelis–Menten saturation,
workload 0 / seed-set 0. The compiler cache was initially empty and isolated;
compiler mode was `FAST_COMPILE`, float64, using the locked numerical stack.

Diagnostic task order was cell 0, cell 1, cell 0, one draw each. These **three
executions are replay diagnostics, not independent primary samples**.

- All 24 requested scientific array fields reproduce exactly after an intervening
  task: dtype, shape, and raw bytes.
- All fields of the first result are finite.
- Compressed NPZ read-back preserves every field's dtype, shape, and raw bytes.
- A SHA-256 digest binds the archive; see `events.jsonl`.

No acceptance filtering, corpus normalization, corpus validation, controlled-input
cross-method equivalence, or specialized-method evaluation ran. Raw array
read-back is not corpus-schema validation.

## Observations, not throughput claims

- Structure/configuration manifest completed about 0.60 seconds after script initialization.
- Model construction: 4.18 seconds.
- Compilation: 31.40 seconds.
- Diagnostic calls: 0.219, 0.213, 0.216 seconds.
- Entire resource-scoped invocation: 44 seconds at integer-second resolution.

These are a single local preflight, not repeated performance estimates, accepted
output throughput, Modal cold-start timings, or cloud cost evidence. Do not rank
methods using them. Python import/startup is included in the enclosing invocation,
not in the script's stage timer.

## Safety envelope and remaining budget

The benchmark ran under a systemd scope with `MemoryMax=8G`, no swap, a 100% CPU
quota, and one configured compute thread. `timeout` sends TERM at 895 seconds and
KILL five seconds later. An isolated 2-GiB tmpfs bounds **combined** output,
compiler cache, and temporary files (stricter than the retained-output limit).
The namespace owner is outside the memory-limited benchmark scope so it can
preserve files after a killed child. Only outputs are retained; the cache is
removed by unmounting the private filesystem. Peak observed memory was not
recorded for this invocation; the configured ceiling is not a peak measurement.

Environment setup also included an 8.114-second dependency-import probe under an
8-GiB/120-second service cap, and a failed 18-ms launch caused by systemd's default
working directory. The latter executed no Python; the corrected command used an
absolute interpreter path and explicit working directory.

**Conservatively reserve 60 seconds of the two-hour aggregate budget for this
session. At most 7,140 seconds remain for all subsequent comparison/correctness
invocations.** Retained evidence is approximately 5 MiB. Future work must enforce
the aggregate runtime and retained-output caps across this evidence and all new
invocations; this preliminary single-invocation script does not do that bookkeeping.

No Modal calls, spending, production integration, unrelated source modifications,
or changes to scientific distributions occurred.

## Still required before resolving the task

1. Build the isolated family-specialized padded candidate, retaining mixed
   carryover and runtime causal masking; inspect prior/family-selection semantics.
2. Implement the aggregate-budget supervisor with peak-memory accounting.
3. Add controlled-input graph equivalence (`rtol=atol=1e-10`), causal/padding
   invariants, and all-method replay for both logical draw groupings.
4. Apply identical fixed-allocation acceptance checks; normalize, validate, save,
   and reload actual corpus-compatible outputs without replenishment.
5. Run the authorized workload/repetition matrix within the remaining caps,
   preserving unsupported cases, failures, rejections, and incomplete evidence.
6. Report stage timings and spread, with no method selection or cloud claims.

The task stays open. This is a checkpoint, not a resolution or a new approval.
