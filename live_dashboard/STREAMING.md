# Reusable-family streaming generation (opt-in)

Use **`scripts/modal_streaming.py`**, not `live_dashboard/launch.py`, for the new
compile-once-per-family workflow. The legacy launcher remains unchanged for existing
runs and scientific presets. This script publishes to the existing dashboard Volume
in Modal environment `pfn`; it does **not** deploy or replace the dashboard app.

## Small verification run

From a checkout of the latest `agent/live-dashboard`, with Modal credentials for the
`pymc-labs` workspace and access to environment `pfn`:

```bash
uv sync --frozen --extra dev
# Dry validation/tests, no Modal execution:
uv run --no-sync pytest tests/test_streaming.py tests/test_stream_store.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest tests/test_streaming.py --runslow -m slow

# Paid execution: obtain operator approval for the quota and budget first.
STREAM_WORKERS=2 uv run --with modal modal run --env pfn --detach \
  scripts/modal_streaming.py \
  --label stream-composable-test --worlds-per-split 100 --batch-size 10 \
  --families hill:geometric,logistic:geometric \
  --seed 631227 --queue-size 2 --max-cost 1
```

The label must be a fresh lowercase slug, at most 40 characters. The immutable run ID
adds a digest. The result is **200 worlds total**, ten ten-world publications per
family. The first publication waits for cold compilation; subsequent publications
reuse the same compiled function. The dashboard polls committed completion records,
not the queue. Select the new label in:

<https://pymc-labs-pfn--pymc-generator-dashboard.us-east.modal.run>

Keep the **local launcher alive** until its summary: `--detach` is not a durable
remote orchestrator for the whole local entrypoint. If it disconnects, inspect the
Modal app before taking over; do not assume every queued family was dispatched.

## The structural-split design (`--design structural-splits`)

The 1,000-world `structural-splits` design of the legacy launcher (10 saturation ×
adstock splits × 5 texture regimes, 20 worlds per split × regime, controls that never go
below zero) runs through the same backend with `--design`:

```bash
STREAM_WORKERS=5 uv run --with modal modal run --env pfn --detach \
  scripts/modal_streaming.py \
  --label stream-structural-splits --design structural-splits --compile-mode python \
  --worlds-per-split 100 --batch-size 10 \
  --seed "$(date -u +%Y%m%d%H%M%S)" --queue-size 2 --max-cost 2.7
```

- **Partitions.** Every design member (split × regime, `<saturation>-<adstock>-<regime>`)
  is a partition: one worker, one compiled model of the member's legacy configuration
  (`generation.DESIGN_PRESETS`). Regimes change priors baked into the graph, so a split
  cannot share one model over its regimes: 50 compilations. Task `i` belongs to the same
  member as in the legacy design (`structural.SplitDesign.member_of`); 1,000 worlds are
  100 ten-world tasks, two per member.
- **Compile mode.** With only 20 worlds per compiled model, compilation dominates. In
  cold 1 CPU / 2 GiB Modal containers (pre-flight, 2026-10-15) C thunks (`cvm`, the
  default) compiled a member in 110–200 s and drew a world in about 10 ms; Python thunks
  (`--compile-mode python`) compiled in 13–34 s and drew a world in 0.3–0.5 s, so a whole
  member took 21–45 s instead of 2–3.5 minutes. The mode changes execution, not the prior.
- **Workers.** The dashboard Volume is a Modal Volume v1, for which Modal recommends at
  most five concurrent writers; each partition commits twice per published batch.
- **Seeds.** `--seed` is the base seed (by convention the UTC launch time). Each split's
  seed is the legacy split seed (first 31 bits of
  `sha256('<base seed>:<saturation>:<adstock>')`), shared by its five regimes, whose
  candidates differ by their global candidate IDs. The manifest records every split seed
  (`generator.config.structural_splits`) and every member configuration
  (`generator.config.presets`); each completion records its split, regime and root seed.
- **Positive controls.** A world whose control value or control contribution goes below
  zero is rejected inside the 2× candidate allowance and counted as `negative_controls`
  (the legacy run redrew the whole task instead; the per-world distribution is the same).
- **Run identity.** The run preset is `stream-structural-splits`, a plain run to the
  dashboard: the legacy design's seed and redraw contract does not apply. The task
  "seed" the dashboard shows is the logical `base seed + task index`.
- **Same worlds, different machinery.** Given the same random draws, a member's compiled
  template reproduces the ordinary per-world model's outputs and mechanism reports
  (a slow test checks this for a calm and a hectic member, in both compile modes). What
  differs from the legacy run: the random-number contract (no same-seed equivalence);
  10-world instead of 2-world tasks (so "task `i`" holds other worlds); the compiled
  template (C or Python thunks) on 1 CPU / 2 GiB instead of the per-world FAST_COMPILE
  model on 2 CPU / 4 GiB; no walk truth (`record_walk_parameters` is off: a report, not a
  prior); a realism rejection draws a fresh structure, where the library redraws
  parameters on the same structure; rejected/surplus counts and the dashboard's speed and
  cost cards are not comparable (compare worker-seconds per world and billing); the
  dashboard header shows no code revision.
- **Partial runs.** Partitions start in member order (the calm members of every split
  first), so a run that is still going is not balanced over regimes: rate finished runs.
- **Budget.** The envelope counts every partition at its full timeouts: 50 partitions
  reserve about $2.65 with 1 CPU / 2 GiB each.

## Execution, memory and storage

- One process per partition worker; **one CPU core and 2 GiB**, numerical-library threads
  limited to one. `STREAM_WORKERS` bounds concurrent containers (default 2, maximum 16).
- One compiled model per worker, not per batch or realized causal graph. Active counts,
  causal structures and parameters vary per world. Each new container/resume compiles
  once again; the in-process compiled function is not persisted across containers.
- Default publication batch: 10 worlds. A bounded per-worker queue holds at most two
  batches, plus the writer's batch and the batch being generated. Audit histories are
  cleared between chunks. Slow storage backpressures the producer instead of accumulating
  the corpus. Writer-side save/readback temporarily adds a bounded number of batch copies.
- A single writer owns all payload validation, serialization, Volume commit/reload and
  readback while generation runs. Main-thread discovery occurs before the thread starts;
  the writer is joined before returning. Different workers write disjoint task paths.
- Payloads are persisted before their completion markers. Enqueued is **not** published.
  Each batch is verified by readback. Publication errors propagate to the worker; there
  is no automatic retry or silent fallback to ordinary generation.
- Scoped per-task discovery avoids rescanning every completion on every write. This
  relies on deterministic task-prefixed output/world IDs and exclusive task ownership;
  it is not distributed locking. Before it starts, a worker validates only its own tasks;
  the coordinator validates the whole run, conflicting records included, when it reports.
- The queue currently publishes each generated batch directly (no shard aggregation).
  Larger publication batches can reduce file/commit overhead, but change the run identity.

## Scientific configuration — important

Without `--design` this is `stream-composable-v1`, **not** the five-regime
`structural-splits` prior:

- 208 reported weeks, 8-week burn-in, 5–15 active channels, 3–10 controls;
- original compositional input prior, including possible connections between inputs;
- one specified saturation × adstock pair per worker; independent component inclusion
  flags (seasonality, trends, pulses, on/off flighting, etc.) per input;
- **signed controls and negative control contributions are allowed**. Negative-part
  crosses in the dashboard are not hidden or clipped. Do not compare its speed or
  scientific distribution directly with the positive-controls 1,000-world demo;
- one world per fresh structure, with deferred train/validation membership (`is_val=2`);
- its prior enables no mechanism priors, so saturation-family arrays are absent and the
  structural-split chart can be empty; each completion records its family. This backend
  never records walk truth.

The self-contained backend and its exact research provenance are documented in
[`_specialized/README.md`](_specialized/README.md). No external research worktree is
required at runtime.

## Stop, failure and resume

Workers have a 30-minute timeout, stop accepting batches with a three-minute margin,
and drain queued batches on graceful stop or generation exceptions. An in-progress
compiler or stuck storage call can still hit the hard platform timeout. Abrupt death
can lose queued/current batches but cannot turn them into valid publications. A
numeric draw failure inside the model (the library's retryable kind) leaves its
candidate without a world and counts as a draw failure (a task left without worlds
records why); a task none of whose candidates draws, and any other error, stop the worker.

The candidate allowance is at most twice the target per logical batch. Rejections are
counted; exhausted batches publish a visible shortfall, not an unbounded replenishment
loop. A completed shortfall is authoritative and is **not** silently topped up on resume.

To resume an interrupted run:

1. Inspect its Modal app and confirm **all earlier workers have stopped**. Stop that
   specific app if needed. Never overlap dispatchers/attempts for a run.
2. Use the **same checkout, numerical environment, label, seed, family order (or
   `--design`), compile mode, batch size and quota**. Source/configuration digests are
   checked; mismatches require a new label.
3. Repeat the launch with `--resume --previous-stopped`. The second flag is an operator
   assertion, not an automatic lease check. Re-approve the estimated budget for each wave.
4. Confirm valid publications are reused and only missing tasks run. Payload-only
   attempts are unfinished; their replacement uses a new attempt directory.

Deterministic candidate IDs and per-family root seeds make results independent of
worker scheduling and queue timing. Queue size/concurrency may change on resume;
scientific settings and logical batch size may not. Do not change the old request's
quota to extend a run: create a new separately identified run.

## Costs and scale

The launch checks a conservative requested-resource/time envelope including startup,
helper calls and a build allowance against `--max-cost`. It is **an estimate, not a
hard billing cap**; prices, storage, egress and actual usage are separate. No retries
are dispatched automatically. Per-worker reports include duration, compilation count,
peak RSS and requested-resource compute estimates. The local coordinator is not billed
as a generation worker; the dashboard's legacy attempt-cost card may show no estimate.

For a separately approved 200k-world plan with two splits, use
`--worlds-per-split 100000` and keep batches small. Multiple bounded resume waves may
be needed. Do **not** infer a total-run bill from one wave's estimate.

Bounded raw-data memory and compilation reuse are tested. **200k-world production
throughput/storage/dashboard scalability is not yet qualified**: resume validation
reads completed payloads one at a time; metadata and dashboard extracted metrics grow
with the run; 10-world batches create 20,000 publications. Benchmark storage/file-count
limits and dashboard refresh time before choosing full-scale shard sizes.

## What to verify

- per-worker `compilations == 1` when it generated anything; `0` for all-reused workers;
- incremental ten-world publications, no repeated world/task identities;
- dashboard-readable payloads and zero unresolved/integrity errors;
- stop/resume reuses completed tasks without changing their bytes;
- bounded RSS over many chunks and queue backpressure under a slow writer;
- compile vs warm draw vs storage time on Modal before claiming cost improvements.

Reports are saved under `local-runs/<run-id>/<launch-time>/` and in the run's Volume
`stream-<attempt-id>.json`. A `finished` summary means the target was published;
`stopped` requires inspection (deadline or shortfall), and `failed` requires reconciliation.
The attempt record the dashboard reads is `finished` for a stopped attempt too (its
reader knows running, finished and failed attempts); the run's pending tasks show it.
