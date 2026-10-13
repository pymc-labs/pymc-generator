# Lazy branch diagnostic — not the target compositional DGP

## Scope clarification

Teemu initially requested testing a compiled runtime one-of-N shape selector, then clarified that the desired DGP must still sample and **combine independently selected components per input series**. The completed one-of-N test is retained as a branch-execution diagnostic, not a recommendation to replace that distribution. No production change was made.

## What was tested

Two deterministic PyTensor functions, compiled once each, accept a base 108×25 array, a separate integer shape selector and activity flag per slot, and numerical parameters. Fifteen slots have media-style multiplicative level effects, ten have control-style additive effects. The seven choices are plain, seasonal, trend, flighting, onset, offset and level jump. Each slot can choose differently, without recompilation.

The eager version uses nested elementwise `switch`; the lazy version uses `ifelse`. Both use the Python VMLinker with `lazy=True` and fast_compile rewrites. Shape parameters and base series are **inputs**, not PyMC random variables. No random-walk generation, parameter RNG, causal propagation, saturation/adstock, acceptance or corpus pipeline is included.

All 512 input cases match each other and an independent NumPy reference at rtol=atol=1e-12. Three timing passes per case use 128 function calls, reversing method order on alternate passes.

| Input case | Eager median ms/call | Lazy median ms/call | Observed kernel speedup |
|---|---:|---:|---:|
| All seasonal | 10.186 | 5.718 | 1.78× |
| All trend | 10.381 | 4.450 | 2.33× |
| Mixed shapes, all slots active | 10.156 | 4.387 | 2.31× |
| Mixed shapes and active counts | 10.460 | 3.552 | 2.94× |

An additional instrumented two-slot probe (seven branches each, second slot inactive) recorded **14 branch executions with eager switch versus one with lazy ifelse**, specifically the selected active slot's branch. Instrumented probes disable rewrites and are excluded from timing; they demonstrate lazy VM execution semantics, not a full-generator RNG execution trace.

Observed primary compilation was 25.15 s eager and 1.72 s lazy, but functions/probes ran in fixed order and shared one initially empty compiler cache. **These are not independent cold-cache compilation comparisons; do not infer a compilation speedup.** Host page caches were not controlled. Both methods coexist in the process, so its ~118-MiB peak RSS is not a per-method memory comparison or comparable to the full PyMC DGP's footprint. Scope peak was ~191 MiB; no memory-limit events occurred.

## Implication for the clarified design

The supported inference is narrow: a compiled runtime branch can avoid evaluating unselected deterministic component expressions, and this kernel was faster with lazy selection. The same lazy primitive can potentially gate each independently selected component before their contributions are composed; exclusive one-of-N selection is not required by the primitive.

What remains **unmeasured** is the full intended implementation: per-input component flags sampled as random variables inside the graph, compositional lazy gating, and PyMC's RNG update/reseeding behavior. Random parameters or RNG-state updates may still execute even when their deterministic component branch is disabled. Moving a Bernoulli flag into the graph alone does not establish saved work. Nor does a 2–3× isolated-kernel speedup imply that speedup for a DGP whose random walks and causal graph dominate runtime.

The [completed rich-component study](rich-components-report.md) already measured the full common-output generator with runtime component composition, but flags were sampled outside PyTensor. It found ~24% higher candidate-drawing time and ~8% higher primary peak process RSS for random rich flags versus its matched baseline. A next integration experiment must preserve independent component combinations, priors, seeded replay and output correctness rather than turn the DGP into a one-shape-per-series model.

## Evidence and envelope

Harness commit `3a10cfc`; [result](shape-selector-evidence/result.json), [branch probe](shape-selector-evidence/branch-probe.json), [timings](shape-selector-evidence/timings.json), [ledger](shape-selector-evidence/budget-final.json). PyTensor 3.2.4, one logical CPU, 2-GiB hard memory maximum, no swap, one numerical thread. A private 128-MiB tmpfs bounded scratch. This separately recorded self-imposed ten-minute envelope began 2026-10-13 22:53:29 UTC, execution deadline 23:01:29, total deadline 23:03:29. Execution finished before 22:58:09; no further runs were made. No paid execution, production integration, PR or merge.
