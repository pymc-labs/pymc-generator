# Bounded local generation comparison

## Outcome

**The approved primary matrix completed. This report supplies evidence, not a production method selection.**

Protocol: [Approve the generation-comparison protocol and bounded work envelope](https://github.com/pymc-labs/pymc-generator/issues/55#issuecomment-6149847169).
Evidence task: [Collect the approved bounded generation-comparison evidence](https://github.com/pymc-labs/pymc-generator/issues/56).
Next decision: [Choose the generation method and candidate resource settings from comparison evidence](https://github.com/pymc-labs/pymc-generator/issues/58).

- 12 workloads × 3 seed sets × 5 structures × (1 + 10 draws) × 3 methods = **5,940 primary candidates**.
- **5,938 accepted and persisted**; two scientifically rejected, with no replenishment.
- 36 invocations completed; every invocation passed the implemented controlled-input equivalence gate, all-task same-method replay, causal/padding checks, corpus validation, and lossless array read-back.
- 6,634 seconds charged to the 7,200-second envelope, including the conservative earlier preflight reservation. **566 seconds remain, not an authorization to expand the approved matrix.**
- Largest measured invocation cgroup memory peak: **1,318,154,240 bytes (1.23 GiB)**. Full retained comparison evidence is about 121 MiB.
- No paid execution, Modal experiment, production integration, deployment, PR, or merge.

## Main observations

### Warm reuse favored ordinary models in this experiment

For the ten-draw grouping, which reused already compiled models, ordinary generation delivered a median paired advantage of **7.41× over the dynamic template** and **6.35× over the specialized template** across the 36 workload/seed-set pairs. Observed paired ranges were 4.23–20.55× and 3.77–12.77× respectively.

These rates count **accepted, normalized, validated, compressed-NPZ-persisted and read-back outputs**, not just draw calls. They also include the harness's causal/hash checks and bookkeeping. They are local benchmark-format rates, not final consumer-ready or production throughput.

**Important boundary:** ordinary generation retained five compiled cell models, while each template retained one compatible padded model. Reusing the same five structures is not the same workload as an unbounded stream of new structures. This result does not locate the compilation-amortization crossover or show which architecture wins for a large corpus of distinct structures.

### Specialization reduced initial setup relative to the dynamic template

First-use measurements include construction, compilation, five candidate draws, acceptance, normalization and persistence. The specialized candidate was materially cheaper to initialize than the existing dynamic template. It also beat ordinary first-use time in several workloads, including the fully active (15,10) case: median **15.63 s specialized**, **21.19 s ordinary**, **36.36 s dynamic** for five candidates.

At (5,3), the corresponding medians were **17.67 s specialized**, **8.02 s ordinary**, **39.07 s dynamic**. Thus the evidence does not support a simple universal ranking independent of structure count and model reuse.

**Cache qualification:** each invocation is a fresh process with an initially empty isolated compiler cache. Method order is always ordinary → dynamic → specialized, and the methods share that invocation's cache. First-use template timings are **not independent empty-cache measurements**, and none of these observations measures Modal cold starts. No causal attribution of the observed differences solely to specialization is justified by this ordering.

### Rejections were retained and explained

| Method | Evaluated | Accepted | Rejected |
|---|---:|---:|---:|
| Ordinary | 1,980 | 1,980 | 0 |
| Dynamic template | 1,980 | 1,980 | 0 |
| Specialized template | 1,980 | 1,978 | 2 |

Diagnostic replay confirmed the two specialized rejections:

- Hill/geometric, (10,6), seed set 0, cell 0, draw 8: minimum direct-treatment CV **0.05581044336**, below the unchanged **0.08** floor.
- Michaelis–Menten/geometric, (5,10), seed set 2, cell 2, draw 6: **negative outcome**.

The audit replayed the complete ten-draw cell groups, reproduced their acceptance decisions, and checked accepted float32 outcomes against the previously persisted corpora. Raw rejected arrays and detailed metrics are retained in the audit directories. These 20 audit draws are diagnostics, not new primary contributions.

Different rejection counts across methods are not evidence of a distribution change: their RNG schedules differ, and equal seed numbers do not align their independently sampled worlds. Conversely, this small experiment cannot statistically establish equal distributions or precisely estimate rare rejection probabilities.

## Workload and provenance

The unchanged scientific baseline is `b941d3a222fb91475cc2b59db14281e2b6498c49`.
The successful primary runs all used prototype commit `20e266f` (full hash in each environment file). Rejection audits used `18c0795b939fdb325d506a8185f3c82bc2fc6082`; that commit added audit/reporting controls without changing any generation method.

The prior preflight's immutable `comparison-preflight-evidence/workloads.json` pins every resolved configuration, structure, realized family, runtime input and seed. Configurations were rebuilt from the reviewed prior and checked for exact JSON equality against that manifest before execution.

- 104 reported weeks, one latent slot, carryover length/burn-in 4.
- Padded capacity 15 media / 10 controls.
- Michaelis–Menten/geometric at (5,3), (5,10), (15,3), (15,10), (10,6).
- At (10,6), linear, Hill, logistic, tanh and root saturation with geometric carryover; Michaelis–Menten with Weibull-only and equal-probability geometric/Weibull carryover.
- Five structures per configuration/seed set; both logical draw groupings reuse those structures.
- Three fixed seed sets; every other prior/trajectory setting remains the reviewed baseline.
- One configured compute thread and a 100% CPU quota, float64, `FAST_COMPILE`, locked dependency environment. Hardware details are in the earlier preflight environment; installed versions are recorded per invocation.

No model runs used adaptive seed selection, top-ups, automatic numerical retries, or a reduced workload. An initial harness import failed before any generation because `clone_replace` was imported from the wrong PyTensor namespace. It was corrected to the version-pinned `pytensor.graph.replace` import. The one-second failed launch remains in the ledger and artifacts; all primary measurements subsequently used the corrected immutable commit.

## Prototype and correctness surface

`comparison_methods.py` provides three adapters. The ordinary adapter builds the existing ordinary model for each cell. The dynamic adapter builds the existing padded template. The specialized adapter clones the existing template builder's function namespace, rebinding only its parameter and graph collaborators; it does not monkeypatch or edit production modules.

Specialization replaces singleton saturation/carryover IDs with concrete family vectors so unused response branches are not built. Mixed carryover remains the original runtime family vector. All primitive prior constructors and creation order are reused from the existing implementation. Dynamic and specialized model primitive-RV names, operator types, dtypes and ranks are checked against each other. Runtime edge and activity masks and the causal graph implementation remain unchanged. The candidate explicitly rejects broader trajectory/mechanism-prior configurations outside this baseline.

The fixed-allocation formatter is an isolated adaptation of the production additive corpus formatter. It retains scientific acceptance and normalization, but supplies predetermined cells, requests exactly one batch per cell, disables replenishment/retries, compacts rejected rows, and stores logical identities separately from compact physical cell IDs. Candidate-local support-mask RNGs are fixed by `(configuration seed, cell, draw, 4401)` for all methods. These are benchmark-specific identities and packing choices, not a proposal for the final consumer schema.

Implemented checks:

1. **Exact same-mode replay:** all 5,940 primary draws were re-executed after intervening cell tasks; field names, shapes, dtypes and bytes were compared through SHA-256 fingerprints. Replays never contribute to accepted corpus counts.
2. **Controlled-input equivalence:** one shared concrete primitive-parameter/innovation fixture per workload and seed set (36 total) was bound into all three actual model graphs. The 24 requested scientific outputs agreed at `rtol=atol=1e-10`. This is not a same-seed comparison of independent method draws.
3. **Causal/padding invariants:** finite candidates were checked for additive decomposition and indirect-effect telescoping. Padded causal output slots were checked before cropping. Production modules and their masking order were not modified.
4. **Scientific acceptance and persistence:** identical production realism/support-scale checks; the corpus validator passed before saving and after loading. Every nested persisted array retained its dtype, shape and raw bytes.
5. **Sampling semantics inspection:** same prior-construction helpers, settings and supports; only family-dispatch graph construction changes. Mixed-family instructions remain mixed. This structural argument and the controlled tests support the tested envelope, not statistical equivalence certification or arbitrary configuration support.

The primary pass includes 5,940 primary draws, 5,940 replay draws, and 36 diagnostic joint-primitive draws. The two rejection audits add 20 diagnostic draws. The earlier preflight contributes three diagnostic draws. All are accounted for within the same work envelope.

## Timing and resource interpretation

Raw summaries separate build, compilation, draw, acceptance/packing, normalization/diagnostics, validation, serialization and read-back/revalidation. Acceptance and packing are reported jointly rather than pretending to separate them faithfully. Structure preparation was recorded in the preflight; each invocation records input loading and its original preparation cost separately. Method pipeline rates exclude shared preparation, process import/startup, replay and controlled-gate work; the enclosing invocation/aggregate ledger includes that execution time. Do not treat a pipeline rate as inclusive of those excluded stages.

The first grouping builds fresh method models. The second grouping reuses those models after the first grouping and replay. There are three seed-set measurements per row, reported as **median [minimum, maximum]**, not confidence intervals. The host was not reserved exclusively or pinned to a particular CPU; observed spread and fixed method order must be retained when interpreting results.

Memory is an **invocation-wide cumulative cgroup peak**, including all methods, retained models, and diagnostic gates. It is not an isolated per-method memory requirement, nor justification for reducing production memory limits.

The aggregate launcher reserves time and retained-output capacity before each invocation, with an exclusive lock. A supervisor crash leaves reservations charged rather than silently restoring budget. Each compute child has an enforced ≤15-minute timeout, 8-GiB/no-swap cgroup ceiling, and one-core CPU quota. A private size-limited tmpfs bounds output plus compiler cache plus temporary writes; its allowed size shrinks with retained evidence. Cache is discarded between invocations. The namespace owner preserves artifacts after the compute process exits.

There is no cloud billing basis, so **no dollars/world or projected Modal cost is reported**.

## Artifacts and reproduction

- [All descriptive tables](comparison-tables.md): warm rates and first-use pipeline times with spread.
- [Machine-readable summary](comparison-summary.json) and [per-run CSV](comparison-summary.csv).
- [SHA-256 inventory](comparison-inventory.json) for all retained files.
- `comparison-evidence/w<workload>-r<seedset>/`: six corpus NPZs, six timing/count/identity JSONs, raw event log, environment, result, envelope and budget snapshot.
- `comparison-evidence/audit-w6-r0/` and `audit-w1-r2/`: rejection metrics and raw rejected draws.
- `comparison-evidence/comparison-broad-000/`: failed pre-generation import evidence.
- `comparison-evidence/budget-final.json`: final cumulative ledger. Its original `/tmp/` destinations map to the archive directories above; files were moved, not regenerated.

Recompute summaries without executing models:

```sh
python benchmarks/comparison_report.py
```

Original execution pattern (already completed; **do not reset the ledger or blindly rerun**):

```sh
uv sync --locked
python benchmarks/comparison_budget.py --workload W --repeat R --destination /tmp/NEW-DIRECTORY
# Diagnostic audit, only after a recorded rejection:
python benchmarks/comparison_budget.py --workload W --repeat R \
  --audit-source /path/to/original/invocation --destination /tmp/NEW-AUDIT-DIRECTORY
```

Reproduction of the whole experiment now requires a fresh explicit work envelope, since only 566 seconds remain in this one. It does not authorize paid runs or expanded workloads.

## Decision handoff, not further execution

The method-selection ticket now has measured evidence for the approved compact envelope. It should explicitly weigh first-use setup against warm reuse and the expected number of distinct structures per compatible worker. This experiment does not establish a many-structure crossover, `FAST_RUN` behavior, CPU/thread scaling, snapshot benefits, Modal costs, or final publication/recovery throughput. Any such follow-up requires its own bounded scope/approval; none was run here.

Carlos's contract approval and consumer-pipeline validation remain independent, unfinished gates. No final generation method, worker sizing, numeric production acceptance threshold, or deployment decision is made by this evidence task.
