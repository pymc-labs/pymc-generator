# Modal structural-partition smoke test

Date: 2026-10-15. Status: **PASS**.

[Modal run](https://modal.com/apps/pymc-labs/main/ap-JHB3tO3Yzbi0wTCZODinS4)
(workspace authentication required). The app finished and is stopped with zero
active containers. Storage is retained for dashboard review.

## Results

| Measure | Result |
| --- | --- |
| Family partitions | 18 |
| Peak overlapping worker calls | **18**, in 18 distinct single-use containers |
| Target / published | **1,800 / 1,800 worlds** (100 per partition) |
| Candidates | 2,800 |
| Scientific rejections | 20 |
| Valid surplus discarded | 980 |
| Shortfalls | 0 |
| Builds / compilations | One of each per partition |
| Persisted NPZ bytes | 44,190,035 |
| Frozen split | 1,440 training / 360 validation worlds |
| Summed worker-call time | 3,546.51 seconds |
| Independent reader time | 93.27 seconds |

Workers ran through one concurrent `.map()` fan-out, not a sequential `.remote()`
loop. Each held one family-compatible model and generated fresh structures
sequentially. Ten partitions used their single bounded 100-candidate top-up;
that coarse allocation explains the surplus. This is a correctness smoke test,
not a claim that those top-up sizes are efficient or selected for production.

The workload matches the [local 18 × 10 smoke](local-structural-smoke.md), except
for the target/allocation size and parallel execution. It preserves the original
Carlos composable prior and fixed-capacity runtime masks. Balancing all 18 family
pairs is intentional test coverage, not the production pre-rejection family prior.

## Verification evidence

1. Workers validated before publication and verified persisted readback.
2. A **separate reader container** reloaded the Volume and validated all 18 outputs,
   froze a snapshot and verified its exact membership and payload integrity.
3. After the app stopped, authenticated `modal volume get` downloaded the store.
   A local reader independently validated all **1,800 worlds in 18 snapshot members**.

Committed machine-readable results:

- [Local result](evidence/structural-smoke/local-result.json)
- [Modal result](evidence/structural-smoke/modal-result.json)
- [Per-worker results and overlap timestamps](evidence/structural-smoke/modal-workers.json)
- [Outside-Modal readback](evidence/structural-smoke/external-readback.json)

The records include actual rejection/surplus accounting, rather than counting all
candidate draws as usable worlds. Every payload is bound by completion-record
SHA-256; snapshots embed those completion records. Deferred membership remains
`is_val=2` in raw corpora; the frozen snapshot supplies train/validation labels.

## Where Carlos can read the data

Workspace: `pymc-labs`; environment: `main`.
Volume v2: **`pymc-generator-smoke-20261015`**.
Run directory: **`modal_smoke_20261015a`**.

```
modal_smoke_20261015a/
    modal-result.json
    evidence/partition-00/ ... partition-17/
        run-spec.json, environment.json, result.json, worker-result.json
        events.jsonl, deferred_research_packing.py
    store/
        runs/modal_smoke_20261015a/tasks/<family_pair>/attempts/first/
            corpus.npz
            completion.json
        snapshots/training.json
```

Inside Modal, mount the Volume and construct `MountedVolumePublicationStore` at
`<mount>/modal_smoke_20261015a/store`. Outside Modal:

```sh
modal volume get --env main pymc-generator-smoke-20261015 \
  /modal_smoke_20261015a/store ./downloaded-store
```

The CLI creates `./downloaded-store/store/`; pass that to `FilePublicationStore`.
`read_snapshot("snapshots/training.json")` verifies the fixed dataset;
`read_training_split(...)` returns its verified membership. Dashboard extraction
and caches remain dashboard-owned, separate from these immutable raw outputs.
No public URLs or shared credentials are needed.

## Budget and lifecycle

The operator approved proceeding after the proposed **$5 stop-new-work threshold**.
Preflight reserved **$1.48473544**, pricing all 18 full worker timeouts (900 seconds),
startup/scaledown allowances, the reader and a $1 build/startup reserve. Workers
requested and limited CPU to one physical core and memory to 2 GiB, with zero
automatic retries and single-use containers. No second wave was launched.

Estimated requested-resource compute for measured worker/reader time was
**$0.06385834**, using official pricing checked on the execution date:
$0.0000131/core/second and $0.00000222/GiB/second. This is **not the invoice or total
billed cost**: image building, startup, storage and billing lag are separate. The
threshold and conservative reservation do not promise a platform-enforced hard
cap. The approximately 44-MB output store remains provisioned as requested.

## Boundaries

The smoke harness still imports pinned research implementation
`d61f98f596c5de885947eb5144897ed0909e7c5d`; its source manifest is verified inside
workers. It is not a merged production scheduler or a finalized generation-method
selection. `scripts/modal_structural_smoke.py` prepares the locked dependency image,
mounts the already-created Volume and uses the shared bounded partition driver.

This verifies successful concurrent publication, cross-container visibility and
inside/outside-Modal reads. It does **not** prove cloud crash recovery, arbitrary
stop/resume, metadata-loss durability, performance scaling, or integration with an
already-running Carlos dashboard. Failure cases have offline tests, not cloud
fault-injection evidence. The PR asks Carlos to review the consumer contract,
especially deferred membership and the explicitly bounded Modal-only durability.
