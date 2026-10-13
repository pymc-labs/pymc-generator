# Rich trajectory components: matched local speed and memory screen

## Result

The richer reusable graph remained comfortably within the tested 2-GiB limit. With independent 50% component inclusion, median candidate-drawing time rose **23.6%** and primary process peak RSS rose **8.4%** versus the matched baseline. This is measurable overhead, not a combinatorial compilation explosion. All four arms and their replay/persistence checks passed.

| Arm | Build s | Compile s | Median draw s / 24 worlds | Worlds/s | Primary peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Legacy graph | 1.580 | 13.178 | 4.385 | 5.47 | 392.7 |
| Rich graph, legacy-like flags | 2.195 | 19.121 | 5.457 | 4.40 | 425.3 |
| Rich graph, random 50% flags | 2.186 | 19.124 | 5.421 | 4.43 | 425.8 |
| Rich graph, all flags on | 2.923 | 23.348 | 6.775 | 3.54 | 439.9 |

The three rich arms have the **same 6,650-node compiled graph**, versus 4,497 nodes in the baseline. Their differences are runtime values, not separately specialized flag combinations. Each arm is a separate process for measurement isolation; one primary compilation per arm serves all 24 worlds and two additional replay passes.

Warm pass ranges, seconds per 24 worlds: baseline 4.372–4.386; rich legacy 5.436–5.468; rich random 5.383–5.425; rich all-on 6.714–6.880. These are three same-input sequential passes, not independent fresh-process repetitions or confidence intervals. Compilation was measured once per arm. Fixed arm order and an unreserved host limit causal attribution of differences between the identical rich graphs; all-on is an observed case, not a proven worst-case bound.

## Matched workload and experimental assumptions

The same 24 candidate identities, seeds, active-count pairs, DAGs, family IDs and smoothness values are asserted equal across arms. Counts are fresh per world: uniform 5–15 active media and 3–10 controls at fixed capacities 15/10, one latent, 104 reported weeks, four burn-in weeks. Media use shared Michaelis–Menten saturation and geometric carryover. Causal edge budgets include control→media, media→media and control→control.

The eight independently selectable components for each active media/control input are **high-frequency noise, pulses, onset, offset, flighting, level jump, seasonality and trend**. The three rich arms all admit the entire set using config inclusion probabilities 0.5; the legacy-like/all-on diagnostic arms override only realized runtime flags. The random arm uses actual independent Bernoulli(0.5) flags, different across channels, controls and worlds. This is an experimental interpretation of uniform subset selection, not a finalized DGP policy.

Both baseline and rich configs give covariate high-frequency/pulse texture nonzero priors (sigma 0.05–0.15, pulse probability 0.1–0.3, pulse amplitude 0.1–0.3) so all eight components can have effects. The baseline realizes HF/pulse everywhere and omits the six schedule components. Other component parameters use recorded config defaults, including one level jump and 52-week seasonal periods. Consequently this baseline is matched to the rich arms, but differs from the previous fresh-mask smoke test where covariate texture was disabled.

All arms request the **same 24 output fields**, preserving a common output surface rather than charging richer arms for extra exported metadata. These are candidate-drawing timings, **not accepted-world or full-corpus pipeline throughput**. Raw candidate archives are persisted and round-tripped, but scientific acceptance, normalization/corpus packing, full rich metadata and publication are not exercised. No 200k-world estimate should silently treat these as measured accepted/published throughput.

## Prototype and correctness

The earlier isolated family-specialized benchmark intentionally rejected rich trajectory configs. `rich_components_run.py` extends that prototype's scope guard and reuses the same scientific template builder, priors and graph helpers; it does not edit production scientific modules. Fixed singleton saturation/carryover branches remain specialized, while trajectory flags stay runtime inputs.

- Every candidate is finite and passes causal decomposition and inactive-slot zero checks.
- Two repeat passes (reverse and deterministic shuffled order) reproduce each candidate's output hash exactly through the same compiled function.
- The common-output raw archives preserve array dtype, shape and bytes on readback.
- The rich-random arm additionally passes an independent **ordinary-builder comparison on identical primitive random inputs** for one representative world across all 24 common fields at rtol=atol=1e-10. That diagnostic is timed and measured separately after the primary measurements; its extra compilation is not hidden in primary setup or draw timing. Its peak RSS was approximately 599 MiB and is not the primary generator memory requirement.
- Only one family pair/configuration was tested; this is not distributional equivalence or exhaustive feature coverage.

## Memory accounting and envelope

Each arm ran in its **own** systemd cgroup, with asserted MemoryMax=2 GiB, MemorySwapMax=0, CPUQuota=100%, CPU-0 affinity and one numerical thread. Compiler caches were independent and initially empty. The common host/filesystem page caches were not flushed.

Primary cgroup peaks, which include different cache/shared-page charges, were baseline 697.6 MiB, rich legacy 416.8 MiB, rich random 431.6 MiB and rich all-on 437.2 MiB. These are recorded rather than discarded; they must **not** be interpreted as proof that richer graphs reduce total memory. The table's process peak RSS is the clearer process-footprint comparison. Import/compiler caches, allocation order and retained raw outputs are included; incremental pure-graph memory is not isolated. No OOM or memory-max events occurred.

A private 256-MiB tmpfs bounded scratch; output had a 64-MiB stop threshold. The self-imposed 15-minute envelope began 2026-10-13 22:46:18 UTC, execution deadline 22:58:18, full reporting deadline 23:01:18. All arms finished around 22:53:05. See [arm ledger](rich-components-evidence/arms.json) and [final ledger](rich-components-evidence/budget-final.json). No paid execution, production edit, PR or merge.

## Artifacts

Harness commit `572077a`; each arm's `environment.json` records exact config, software versions and controls. [Raw evidence directory](rich-components-evidence/) contains inputs/flags, three pass timings, primary and final memory, replay results, raw NPZs and the oracle result. The original completed studies are unchanged.

## Implication

Runtime flags eliminate per-combination compilation, but in this implementation disabling components does not recover the baseline graph's speed: rich legacy-like and random flags cost about the same. A lazy deterministic selector is worth examining separately; it cannot be assumed to eliminate upstream random-parameter generation or the rest of the DGP. The follow-up [one-of-N selector experiment](shape-selector-report.md) tests that narrower question and uses a different, mutually exclusive shape distribution.
