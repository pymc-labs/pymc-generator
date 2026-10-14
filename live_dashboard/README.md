# PyMC Generator live dashboard

Follow a generation run while it happens. A generation worker publishes each completed
task (a corpus payload plus a versioned JSON completion record) into a store; a separate
web app polls the store, verifies and extracts every newly published output once, and
redraws its cards and figures in the browser. Generation never reruns for the UI.

On Modal, both live in the **`pfn` environment**: `generate_run` runs on an A100
container and the dashboard on a separate CPU-only container with a stable URL behind
HTTP Basic authentication. The same code runs locally against a folder.

> **The A100 does not accelerate generation.** PyTensor 3.2 has no CUDA backend and
> generation compiles to Python/NumPy thunks, so worlds are generated on the
> container's CPU while the GPU sits idle. The worker records the GPU it ran beside
> (`worker.gpu_name`) so this stays visible.

## What the page shows

| Element | Definition |
|---|---|
| Card: worlds generated | Consumable worlds (published, verified, extracted) of the run's target. |
| Card: generation speed | Consumable worlds per minute of active generation time (the union of task start–finish intervals, so gaps between attempts do not count). |
| Speed over time | Per-task rate (worlds ÷ generation time) and the cumulative average. |
| Worlds over time | Cumulative consumable worlds at each publication, with the target. |
| Channel impact | Two distributions: each active channel's direct contribution share of its world's outcome (pooled), and each world's total channel share (direct + indirect). |
| Composition triangle | Each world's channel / control / baseline shares of its total outcome, at its exact barycentric position (corner = 100 %). Semi-transparent dots, so overlap shows density; worlds with a negative part sit outside and are drawn as ×. |
| Status line | Run state, recorded tasks, zero-world tasks and shortfalls, rejected candidates, draw failures, unavailable/conflicting/invalid outputs, duplicates, the newest attempt and its GPU. |

Shares use the trajectory-showcase notebook's definitions (`realized_shares`):
channel = Σ treatment contributions + Σ indirect effects, control = Σ covariate
contributions (net, can be negative), baseline = Σ intrinsic baseline + Σ latent
contributions + Σ observation noise, each divided by Σ outcome. Channel impact is the
library's `outcome_distributions(...)["treatment_contribution"].unit_share`. Channel
slots are activated as a prefix (slot k is active only in worlds with at least k
channels), so impact is shown pooled, never per slot.

Only published worlds enter the distributions. Pending, zero-world, unavailable,
conflicting and invalid tasks appear in the status line instead.

## Local use

`live_dashboard` is a repository tool (it is not part of the installed `pymc-generator`
package): run its commands from the repository root.

```bash
uv sync --locked --extra dev --extra dashboard
# Credentials: a username and a PBKDF2 hash, written to a 0600 JSON file.
uv run --no-sync python -m live_dashboard hash-password --username demo \
    --output ~/.config/pymc-generator-dashboard/auth.json
# Terminal 1: the dashboard on http://127.0.0.1:8000
uv run --no-sync python -m live_dashboard serve --data-dir /tmp/live-runs \
    --credentials ~/.config/pymc-generator-dashboard/auth.json
# Terminal 2: generate 20 worlds as 10 published tasks of 2 worlds
uv run --no-sync python -m live_dashboard generate --data-dir /tmp/live-runs \
    --label local-demo --worlds 20 --worlds-per-task 2
```

Presets: `showcase-variety` (the trajectory-showcase notebook's parameter ranges with
independent active-count draws per world instead of the notebook's stratified integer
design, which needs one call for a multiple of 10 worlds) and `small` (a tiny
configuration for tests and smoke runs). `generate --max-seconds` (default 3600) is the
attempt's time limit; rerun the same command to resume a stopped run.

## Modal deployment (environment `pfn`)

Run from the repository root with the Modal CLI (`modal` ≥ 1.5). Both entry points
refuse to run unless the CLI environment is `pfn`.

```bash
# 1. Credentials into a Modal Secret (JSON keeps the hash free of shell expansion).
uv run --no-sync python -m live_dashboard hash-password --username demo --output auth.json
modal secret create -e pfn pymc-generator-dashboard-auth --from-json auth.json
rm auth.json   # keep the password elsewhere; the secret holds only its hash
# 2. Deploy the worker and the dashboard (prints the stable dashboard URL).
modal deploy -e pfn live_dashboard/modal_app.py
# 3. Start a run (prints the run id and the dashboard link for it).
modal run -e pfn live_dashboard/launch.py --label demo --worlds 20 --worlds-per-task 2
```

Resources: app `pymc-generator-live-dashboard`, Volume `pymc-generator-live-dashboard`
(v1), Secret `pymc-generator-dashboard-auth`, web label `pymc-generator-dashboard`, all
in `pfn`. The dashboard mounts the Volume read-only and calls `reload()` on it (verified
on the first deployment); if a future Modal release refused that, mounting it read-write
keeps the same behaviour because the dashboard never writes. Rotate credentials with
`modal secret create --force ...` and redeploy.

Limits: `generate_run` uses `gpu="A100"`, 2 CPU cores and 4 GiB, one container at a
time, a 1200 s timeout and one retry, so one launch costs at most about two attempts of
20 minutes (≈ $1.9 at Modal's A100-80GB list price; most runs take a few minutes). A
run is capped at 400 worlds and 20 worlds per task. Requests the worker can never
satisfy (unknown preset, too large, a different generator than the stored run) are
refused without a retry; the refusal is recorded as a failed attempt when the run
exists. Each attempt records its kill time as `deadline_at` and stops starting tasks at
least max(90 s, twice its longest task) before it; a resumed run continues with the
unpublished tasks. The dashboard uses one CPU core and 2 GiB, one container, and scales
to zero after 10 idle minutes. Polling is every 3 s while a run is generating (or while
the store cannot be read) and every 10 s otherwise (`DASHBOARD_POLL_SECONDS`), plus up
to one browser poll; this is a refresh cadence, not a latency guarantee.

## Publication protocol (`pymc-generator-live-publication` 1.0)

This is the demo's **single-writer** protocol: correctness relies on one writer per run
(`generate_run` runs one container at a time and processes tasks in order; launch runs
only through the deployed function). It is not the multi-writer publication and
arbitration mechanism still being decided in pymc-labs/pymc-generator#54, and the
reader rules below are written so such a writer can replace this one.

Identities derive from the request only: `run_id = <label>-<sha256(request)[:12]>`
over (label, preset, worlds, worlds per task, seed); task `i` generates with seed
`seed + i` (exactly batch `i` of `DataGenerator.iter_batches`, which is reproducible but
not the same worlds as one larger call); world ids are `<run_id>/<task_id>/w<row>`.

```
runs/<run_id>/run.json                              run manifest (create-only)
runs/<run_id>/tasks/<task_id>/<attempt_id>.json     task completion record (create-only)
runs/<run_id>/payloads/<task_id>/<attempt_id>.npz   corpus payload (create-only)
runs/<run_id>/attempts/<attempt_id>.json            provisional progress of one attempt
```

The manifest records the effective configuration, a digest of the generating source,
the numerical package versions and the git revision; resuming a run with a different
configuration, code or environment is refused, so a changed generator needs a new run.

**Writer.** Each task's payload is written under its attempt's own path, verified
(reloaded and validated), committed, and only then is the task record written and
committed. Every record is decoded with the reader's decoder before it is written. A
task that the library cannot fill within its bounded rounds publishes a zero-world
record with the full shortfall and no payload; any other failure fails the attempt
(recorded in its attempt record) and leaves the task for a retry. Resuming reuses a
task when a record exists whose payload verifies (or that has no payload).

**Reader.** Every poll lists every task directory (no high-water cursor), so
out-of-order publications are never skipped. Records with the same scientific content
(accepted worlds and payload digest) are duplicates: the smallest attempt id is
authoritative. Different content for one task is a **conflict**: the task is excluded
and shown. A payload must match its recorded size and SHA-256; otherwise the task is
**unavailable** and never silently replaced (it becomes available again only if bytes
matching the original digest reappear). Unreadable JSON is treated as not yet
published; unsupported contract major versions are rejected, not guessed; unknown
fields are ignored. Files whose names start with `.` are staging files and are never
read. Extraction runs once per payload and extraction version.

Crash windows: before a payload commit nothing is visible; between the payload commit
and the record commit the payload is an orphan that readers never consume (the retry
writes its own); after the record commit a retry reuses the published task.

Known limits: an in-place modification of an already verified payload that keeps its
size is not detected until the dashboard restarts; Volume v1 recommends at most five
concurrent writers and about 50,000 files, far above this demo's scale.
