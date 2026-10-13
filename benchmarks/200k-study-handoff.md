# Delegated 200k-world execution-strategy investigation

## Authority and callback

Teemu requested implementation/testing of the compositional approaches, allowing up to six hours total and 30–45 minutes per approach. He then explicitly requested a Herdr agent in another tab using `openai-codex/gpt-6.1-sol:medium`, reporting back to the parent with a prefix clearly distinguishing agent messages from user messages. This file is the parent's handoff, not a new user request.

You own implementation and execution in `/tmp/pymc-generator-200k-study`, branch `research/200k-uniform-rich-strategy`. The parent will not write here while you work. No further delegation. Preserve the `.venv` symlink and other pre-existing worktrees.

Read the Herdr skill at `/home/teemu/.pi/agent/skills/herdr/SKILL.md`. Report acceptance, meaningful milestones, blockers and final results to the parent using:

```sh
herdr agent prompt w1F:p2 '[HERDR AGENT REPORT — NOT THE USER] rich-200k-study: ...'
```

Every callback MUST begin with that exact prefix. Do not impersonate the user or present the callback as new authorization. Do not use `--wait` for callbacks: the parent may be working. Include your actual pane/agent identity and distinguish progress from completion. Read Herdr state before retrying failed delivery; no blind duplicates. Parent reviews evidence before making the final recommendation.

## Immutable envelope

- Start: **2026-10-13 23:11:27 UTC**, epoch **1791933087**.
- Total deadline including reporting: **2026-10-14 05:11:27 UTC**, epoch **1791954687**. Delegation does not reset it.
- Stop new scientific execution at least 15 minutes before the total deadline. Cap each approach at 45 minutes, accounting for failed attempts; early-stop uncompetitive paths.
- Local only. No paid Modal runs, production deployment, PR or merge. Research commits/push are authorized.
- Primary numerical workers: one logical CPU (affinity CPU 0), CPUQuota=100%, MemoryMax=2G, MemorySwapMax=0, numerical threads=1. Use isolated compiler caches. Prior `rich_components_envelope.sh` and `rich_components_supervise.py` show working namespace/tmpfs/systemd patterns.
- Before execution, create a durable absolute-time budget ledger, a timeout/process-tree kill envelope, and bounded scratch/artifacts. Suggested cap: 512 MiB scratch per attempt and 2 GiB total new retained evidence. Reserve space/time for partial evidence. Record actual limits, crashes, cache conditions, RSS and cgroup accounting separately. Do not increase resource caps silently.
- No scientific executions under this new envelope have happened yet. Only inspection and draft source writing; the drafts are untested.

## Scientific contract — latest user correction is authoritative

The user rejected the parent's independent 50% component assumption: **use Carlos's original `composable` preset, including component parameter ranges**. Verified against Carlos's introducing commit `e74b2bdec56d62c9f2f18c0403ed331a15deb444`, `pymc_generator/presets.py:173–213`:

| Component | Treatment/media | Covariate/control |
|---|---:|---:|
| hf | .9 | .9 |
| pulse | .7 | .6 |
| onset | .15 | .1 |
| offset | .1 | .1 |
| flighting | .2 | .15 |
| level_jump | .2 | .3 |
| seasonal | .3 | .35 |
| trend | .3 | .3 |

Selections are independent per component per input; multiple components combine. Original code samples flags once per structural cell, shared by its draws. For this target, use **one fresh cell per world** (`draws_per_cell=1`), so each world gets fresh counts, graph, flags and parameter draws. This is not exclusive one-of-N shape selection.

Saturation and adstock families: independently uniform across currently supported families, **one saturation family and one adstock family shared by all media inside a world**; numerical parameters remain channel-specific. Include identity/no-adstock entries; record all family names. Family sampling may remain in the external instruction planner so specialized workers can be routed by family. In-graph component flag sampling is a key requested approach, not an already-proven optimization.

Existing target dimensions: 5–15 media, 3–10 controls sampled independently uniformly within fixed maximum capacities 15/10; active-prefix masks; one latent, 104 reported weeks. Draft uses Carlos's preset/default l_max=8 and eight burn-in weeks (previous mini-study used 4; distinguish workloads). Existing sparse graph budgets are in the draft config; these are study assumptions, not newly adopted production policy. Masks must affect graph edges, propagation and relative/reference scaling, not just delete outputs.

Uniform family draws describe the candidate prior, not exact quotas in the accepted corpus. Report acceptance and component/family composition before and after rejection. Do not silently fix acceptance by altering priors or suppressing filters. If exact accepted-family uniformity needs a different policy, flag it for the parent/user.

## Starting code and work sequence

Three newly written **untested scaffolds** exist under `benchmarks/`:

1. `study_200k_methods.py`: ordinary / family-specialized / universal adapters; external or in-graph Bernoulli flags; eager or per-component lazy `ifelse`; optional FAST_RUN. Uses function cloning/proxies rather than editing production modules. Keeps one compatible model resident and reuses it across changing inputs. Inspect for errors before trusting any numbers.
2. `study_200k_packing.py`: copied `fixed_allocation.py`, with realised in-graph flags refreshed BEFORE acceptance and truth packing, and support seeds based on stable logical candidate IDs. Full corpus packing/validation, no replacement of rejected candidates. Audit the changes against its source.
3. `study_200k_run.py`: config, stable-ID instructions with shared uniform family assignments, grouped scheduling, chunked full-corpus write/readback, candidate/accepted component/family counts, replay and controlled-input ordinary-builder oracles. Uses the existing environment and benchmark helpers.

No envelope or completed tests yet. You may correct or replace these research drafts; preserve the scientific contract and evidence. Do not assume the draft is correct because the parent wrote it.

Suggested progression:

1. Audit/scaffold, validate the corrected config, implement and verify the hard envelope. Do small pilot tests before large cohorts.
2. Compare a modest ordinary-generation reference and reusable specialized external-eager / external-lazy / internal-eager / internal-lazy approaches. Add universal internal (and external if informative) to assess one universal compile vs the finite family bank. Test FAST_RUN/CVM for a promising contender; consider other supported optimizations only when evidence justifies them. This is an adaptive investigation, not an obligation to burn 45 minutes on every obvious loser.
3. Cover all family pairs, active-count variation, all-off/all-on/mixed component cases. Controlled primitives must agree with the ordinary scientific builder; verify mask causality, decomposition identities, in-graph flags, RNG state advancement, seed replay after intervening worlds and rebuild/reordered scheduling. Record actual RNG work: lazy deterministic gates do not automatically skip random draws whose updates are demanded.
4. Measure repeated steady workloads with fresh world instructions, full accepted-and-persisted outputs, cold build/compile costs, process RSS and bounded-cache conditions. Favor reasonably large cohorts and independently repeated winner measurements over tiny-kernel speed claims. The draft defaults to 1024 candidate worlds with 32-world persistence chunks; lower ordinary counts and pilots are sensible. Any diagnostic compilation/oracle process must remain separate from primary memory comparisons.
5. Recommend the best measured method for 200k accepted worlds: estimate candidate count under rejection, amortized compile cost, accepted/persisted throughput, memory and output size, sensible worker/shard lifecycle, and where parallel/Modal extrapolation remains unmeasured. Do not actually create 200k worlds or deploy a corpus without a separate request.

## Evidence and reporting

Preserve previous study evidence unchanged. New reports/results belong under `benchmarks/` with a self-contained rerun recipe and versioned config/IDs, raw timing/acceptance records, correctness results, absolute final ledger, and explicit failed/unmeasured approaches. NPZ files are ignored by Git: force-add only intended evidence archives. Keep ordinary scientific library code unchanged.

Previous reports on this branch:
- `rich-components-report.md`: 50% external flag mini-study, not the corrected target prior.
- `shape-selector-report.md`: one-of-N deterministic lazy-branch diagnostic, not the target distribution or whole-generator speed.
- `fresh-masks-report.md`, `modal-targeted-report.md`: earlier execution/masking context.

Tracking: GitHub map #47; reopened evidence ticket #61; method-selection #58 remains open. Scope authorization/correction comments are in #61. Add a concise evidence handoff to #61 and link #58 when done; do not claim a production method selected or close #58. Hindsight initiative `kp-658cf9f0f07b4f488179b717ec4c9264` already tracks this corrected scope; normal memory attribution rules apply.

If blocked or time/resource limits make further work indefensible, stop execution, retain partial results, report exact attempted paths and blockers to the parent with the required prefix. A bounded negative result is acceptable; fabricated throughput or weakened correctness is not.
