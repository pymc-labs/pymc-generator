# Modal-only publication: local first slice

Status: **prototype with local and bounded Modal smoke validation**, not a deployed
production pipeline. See [local evidence](local-structural-smoke.md) and
[18-way Modal evidence](modal-structural-smoke.md). Planning source: [lifecycle/publication decision](https://github.com/pymc-labs/pymc-generator/issues/54).
Carlos approved the consumer boundary, not the later possibility of irrecoverable
metadata loss. That clarification remains open. No paid execution is authorized.

## Scope and preconditions

`pymc_generator.publication.FilePublicationStore` wraps existing `save_corpus`,
`load_corpus` and `DataGenerator.validate_corpus`; it does not change generation,
normalization, arrays, scientific selection or seeds. It accepts already-selected
corpora and explicit accounting from its caller. It is not a task scheduler or a
coverage/top-up implementation, and does not select the optimized generation method.

Exactly one driver owns a run. Different tasks may execute concurrently, but
attempts of the same task must never overlap. Confirm the old driver/affected
workers have stopped before takeover or retry. Unknown execution status pauses
recovery. These are **operational preconditions**, not distributed locks supplied
by this module. IDs and world allocations must be fixed before dispatch.

The local prototype does not implement driver ownership, cancellation, automatic
recovery, retry budgets, quarantine moves, append-instruction scheduling or access
control. Errors are surfaced to the caller; it must pause the affected instruction
on a reproducibility conflict. Unrelated task failures do not prevent publication.

## Layout and identity

```
runs/<run_id>/tasks/<task_id>/attempts/<attempt_id>/
    corpus.npz          # absent for zero-world completion
    completion.json
snapshots/<snapshot_id>.json
```

Identifiers match `[A-Za-z0-9][A-Za-z0-9_-]{0,127}`. Paths are storage references,
not scientific identities. `task_id` is unique within a run; `output_id` and
`world_ids` must not overlap across tasks in that run. References are relative to
the configured private store. Do not reuse a run ID for a different fixed spec.
The prototype assumes trusted, immutable payloads, not adversarial writers.

## Completion schema, version 1

All metadata is UTF-8 JSON, with finite numbers and no duplicate keys. Required
fields:

| Field | Meaning |
| --- | --- |
| `version` | Integer `1`; independent of corpus/generation versions |
| `kind` | `completion` |
| `run_id`, `instruction_id`, `task_id` | Stable logical identities |
| `output_id` | Stable published grouping identity, fixed independently of attempt |
| `attempt_id` | Execution identity, not part of scientific equivalence |
| `world_ids` | Unique selected candidate identities, ordered exactly like corpus rows |
| `corpus_schema_version` | Current library `CORPUS_SCHEMA_VERSION` (5 today) |
| `counts` | Nonnegative integer `target`, `candidates`, `rejected`, `surplus`, `published`, `shortfall` |
| `rejection_reasons` | Reason-to-count mapping partitioning rejected candidates |
| `provenance` | Described below |
| `payload` | `null`, or `{name: "corpus.npz", bytes: <positive integer>, sha256: <hex>}` |
| `scientific_sha256` | `null`, or digest of persisted corpus values |

Accounting invariants:

- `published == len(world_ids)`;
- `target == published + shortfall`;
- `candidates == rejected + surplus + published`.

Counts describe scientific candidates, not infrastructure retries. Failures are
not fabricated as rejection counts. Infrastructure progress/failure reporting
belongs to the future driver, not a fake completion record. A zero-world result
has no payload or scientific digest but retains all accounting/provenance.

`provenance` requires `run_spec_sha256` (64 lowercase hex characters), nonempty
`generation_mode` and `generation_version` strings, and nonempty JSON objects
`environment`, `seed_context`, `effective_configuration`. The producer must supply
actual effective generation/acceptance settings, fixed task grouping and candidate
seed context, source revision, dependency lock/backend/hardware provenance. The
prototype validates their presence/types, not their truth or completeness; it
**does not implement canonical run-spec hashing or deterministic seed allocation**.
Those are upstream orchestration work, not inferred from paths or arrival order.

The scientific digest hashes canonical JSON describing persisted arrays by dtype,
shape and SHA-256 of C-order bytes, recursively including persisted diagnostics.
`save_corpus` already omits nondeterministic timing. This distinguishes scientific
content from NPZ ZIP-container bytes. Payload checksums bind the exact stored file
for reads/restoration. No cross-environment byte-equality promise is implied.

Unsupported metadata versions fail validation. Additive optional fields are
accepted within version 1. No dashboard analytical summaries are produced here.

## Write, visibility and recovery sequence

### Implemented locally

1. Discover earlier task completions. Unresolved records for the same task block
   retries until reconciliation. Read/validate existing same-task results.
2. Exclusively create a new attempt directory. Never overwrite old attempts.
3. Validate the selected corpus, save NPZ, close/fsync it, reload its persisted
   representation and calculate byte/scientific digests. Validate count alignment.
4. Compare with any prior result: matching retries reuse its reference; conflicts
   raise an error and leave the new attempt unpublished for diagnosis.
5. Exclusively write the JSON completion record, flush/fsync and close it.
6. Consumers validate the record **and** payload before treating it as usable.

A partially visible JSON file is an explicit unresolved discovery result, not a
completion. A payload without a record is unfinished work and is not discovered.
A crash after a complete publication but before acknowledgment is handled by
validating/reusing it. Abandoned attempt directories are left for later cleanup.
Local `fsync` is not a claim of cross-host visibility or power-loss durability of
directory entries. Neither file names nor write order establish distributed atomicity.

### Mounted-Volume adapter (offline tests plus successful-publication cloud smoke)

`MountedVolumePublicationStore(root, volume)` accepts an already-mounted Volume's
synchronous handle. It does not import the Modal SDK, create a Volume, deploy an
app or invoke remote work. The owning Modal app supplies the handle and mount path:

```python
from pymc_generator.publication import MountedVolumePublicationStore

# Inside a configured Modal function; `volume` is its mounted Volume handle.
store = MountedVolumePublicationStore("/data/generation", volume)
```

The adapter closes payload handles → explicitly commits payload persistence →
writes/closes completion JSON → commits again. Zero-world records need only the
completion commit. Snapshots commit after writing. Discovery and reads explicitly
reload first. All container-local store operations must be serialized; no code
in the container may retain handles on this mount across a reload. These are
application preconditions, not a distributed-lock implementation.

`PublicationUnconfirmedError` means a commit acknowledgment failed. The error identifies
payload, completion or snapshot stage. Stop the affected operation and reconcile
before retry: an exception does not prove that persistence failed. Never overwrite
a possibly persisted completion/snapshot. Refresh errors propagate instead of
returning a stale success. Background commits may expose partial files; validation
remains mandatory regardless of explicit write ordering.

Documentation checked 2026-10-14: [Modal Volume guide](https://modal.com/docs/guide/volumes),
sections “Volume commits and reloads”, “Background commits”, “Filesystem consistency”
and “Busy Volume errors”. These document explicit commits/reloads, background
commits, no distributed file locking, last-write-wins conflicts and the restriction
against reloading with open files. A reloading container's mount appears empty
while reload is in progress. They do not establish atomic publication.

Offline recording-fake tests verify barrier order and failure propagation. The
[bounded Volume v2 smoke](modal-structural-smoke.md) additionally verified 18-way
publication, a separate reader container and authenticated external download/readback.
It did not inject commit failures, unknown acknowledgments, metadata loss or stopping
during writes. Those recovery claims still require platform evidence before a
production handoff; the successful smoke is not a general atomicity guarantee.

## Consumer interface

- `discover(run_id)` rescans small completion JSON files and returns candidate
  records plus per-reference problems. These are **not yet verified availability**.
  No monotonically increasing ID cursor: late/out-of-order arrivals remain visible.
- `read_output(reference)` verifies sizes, SHA-256, corpus validation, world count
  and scientific digest before returning the record and loaded corpus (or `None`).
- `validated_outputs(run_id)` validates and deduplicates matching retries, refusing
  conflicting identities. This conservative convenience operation fails if any
  scanned completion is unresolved; a dashboard may inspect individual candidates
  and show failures separately instead.
- `freeze_snapshot(run_id, snapshot_id)` writes an exclusive immutable JSON file
  with `version`, `kind: snapshot`, `run_id`, `snapshot_id` and ordered `members`.
  Each member contains its `reference` and full validated completion `record`.
  Zero-world members retain accounting but contribute no training rows.
- `read_snapshot(reference)` requires every referenced completion to match the
  embedded record and revalidates every payload. It never silently drops a member.

The dashboard owns extraction/caches keyed by output identity plus extraction
version. Consumers must retain prior verified identities if they wish to label
later missing records as unavailable: a stateless directory scan cannot remember
what was lost. A frozen snapshot supplies such retained membership for training.
Restoration must match the original integrity data; otherwise use a new identity.
Metadata/snapshot loss with no intact verifiable copy stops recovery. Stronger
backup guarantees are outside this first slice, pending Carlos's clarification.

## Verification and remaining boundaries

Run `uv run --no-sync pytest tests/test_publication.py tests/test_publication_volume.py`.
Tests include actual
ordinary generation → NPZ → discovery → read → snapshot; zero-world completion;
interrupted writes; retries/conflicts; late arrival; missing/corrupt payloads;
unsupported metadata; path confinement; and immutable snapshot creation.

### Deferred train/validation assignment

`sample_prior_predictive(..., defer_cell_split=True)` and the `DataGenerator`
`generate`, `iter_batches` and `generate_and_save` methods now support deferred
cell membership, including one retained world. Legacy defaults and validation
remain unchanged. Deferred mode skips validation-specific temporal-mask repair;
per-world temporal masks are sampled normally and determine normalization without
reference to future train/validation membership. It does not change the realism
acceptance gates or promise same-mode parity with legacy split-repaired arrays.

The opt-in representation retains the existing arrays/dtypes but uses `is_val=2`
for **unassigned**, accompanied by `diagnostics.cell_split = {mode: deferred,
version: 1}`. This is an explicit extension of the corpus contract, **not binary
membership and not an all-training batch**. Older consumers must not cast this
field to boolean. The existing strict `validate_corpus` rejects it by default;
publication opts in with `allow_deferred_split=True`. That opt-in still rejects
mixed or falsely labelled membership. The persisted marker is required. Downstream
dashboard/training readers must support this representation before consuming it.

Generation retains the configured draws-per-cell grouping in deferred mode. The
existing prior still requires at least two configured cells, so requesting a single
world currently generates a larger internal grid and retains the requested prefix.
This is correctness support, not a single-draw performance optimization. The
fixed effective grid/request size must be part of the upstream run specification.

For a frozen training dataset, pass `validation_fraction` and `split_seed` to
`freeze_snapshot`. Every nonempty member must provide an explicit
`provenance.cell_namespace`: a stable identity for its generation cell grid,
independent of output filename or retry. Preserve that namespace when packaging
one grid across files/tasks; never reuse it for independent grids. Local numeric
`cell_id` alone is not globally unique. The producer is responsible for truthful
namespace provenance; this module cannot infer scientific grouping from files.

The snapshot stores `cell_split` with version 1, algorithm `sha256-cell-rank-v1`,
seed, validation fraction and a world-ID-to-`train`/`validation` membership map.
The algorithm ranks distinct `(namespace, cell_id)` groups using SHA-256 of the
JSON tuple `["cell-split-v1", seed, namespace, cell_id]`; it selects the rounded
fraction of groups, clamped to leave at least one group on each side. It is
independent of discovery order. This targets a **fraction of cells**, not an exact
fraction of worlds. Membership can differ between different snapshots; within one
snapshot it is frozen. No payload, temporal mask or normalization is rewritten.
`read_snapshot` rechecks the assignment; `read_training_split` returns verified
membership. `assign_cell_split` in `pymc_generator.splitting` exposes the pure
assignment function for other consumers.

One-world publications and single-cell snapshots without a split are valid.
Assigning a nonempty train/validation split still requires at least two distinct
cells across the selected snapshot. Related worlds never get separated just to
satisfy that requirement. Missing cells/worlds are never fabricated or duplicated.

Local test success is not evidence of Modal atomicity, dashboard deployment,
production-ready throughput or fixed-spec seed reproducibility of a new scheduler.
The first slice is opt-in and leaves all existing generation entry points intact.
