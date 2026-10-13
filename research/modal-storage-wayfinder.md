# Modal shared storage and incremental publication semantics

**Ticket:** [Research Modal shared storage and incremental publication semantics](https://github.com/pymc-labs/pymc-generator/issues/49)
**Retrieved:** 2026-10-13
**Repository source revision for all code claims:** `b941d3a222fb91475cc2b59db14281e2b6498c49` ([immutable tree](https://github.com/pymc-labs/pymc-generator/tree/b941d3a222fb91475cc2b59db14281e2b6498c49)).
**Scope:** documentation and source inspection only; no Modal execution, provisioning, credential changes, or deployment.

## Bottom line (conditional)

*If the immediate requirement is Modal-internal, write-once shard publication plus a live Modal dashboard/trainer*, a **Modal Volume** is a candidate, with immutable per-shard paths and explicit publication records as one proposed protocol, not a proven or uniquely possible design. Writers must `commit()` after fully writing their shard; readers must `reload()` before discovering the next publication. Do not use a shared mutable manifest or rely on a writer's filesystem `rename()` as an undocumented cross-container atomic-publication primitive.

*If direct object-store integration/lifecycle controls are required*, evaluate a separately selected external object store. Outside-Modal readers alone do not require this: Volumes also offer authenticated SDK/CLI access. `modal.CloudBucketMount` is a mount adapter for an external S3/R2/GCS bucket, **not** a Modal-owned object-store API. Its mount semantics are explicitly limited (sequential-read optimized; no append or seek+write; rename is not supported), so publish completed immutable files using the selected provider's direct object API/semantics or stage locally and copy completed files to the mount. Provider, region, access model, cost, and object-publication semantics remain decisions.

## Evidence: Modal Volumes

[Modal's Volumes guide](https://modal.com/docs/guide/volumes) describes a Volume as a distributed filesystem optimized for write-once/read-many workloads and says it is backed by multiple cloud providers for high availability. It also documents:

| Concern | Documented behavior / consequence |
| --- | --- |
| Visibility | A mounted container starts with the then-latest state. A writer's changes become visible outside that container after `Volume.commit()`; an already-running reader needs `Volume.reload()` to see a later commit. Background commits occur every few seconds and on shutdown, but are unsuitable as the sole completion signal. |
| Durability | A successful `commit()` is documented to persist changes in durable storage and make them available to other containers ([`Volume.commit`](https://modal.com/docs/reference/modal.Volume#commit)). This does **not** document a transaction spanning a shard and a manifest. |
| Concurrent writes | Concurrent modification is supported, but same-file writes are last-write-wins; lost changes are possible and distributed locks are not supported. V1 guidance is no more than five concurrent small commits; distinct-file writers still contend. V2 says distinct-file concurrent writers can scale to hundreds, but is Beta and Modal says it cannot yet guarantee no data loss for mission-critical data. |
| Listing/discovery | SDK `listdir`/`iterdir` and CLI `modal volume ls` are documented ([reference](https://modal.com/docs/reference/modal.Volume#listdir)); mounted readers use normal filesystem listing after `reload()`. V1 performs best below 50,000 files/dirs and has a 500,000-inode hard limit. V2 has no total-file limit but 262,144 files per directory and potentially slower traversal. Therefore a full-corpus scan per dashboard refresh is a poor design. |
| Read safety | `reload()` fails with open Volume files; during reload the initiating container sees the Volume as empty and cannot read/write it. |
| External access | Modal documents Python SDK and CLI access outside Modal, including `batch_upload`; frontend download is limited to 16 MB but CLI has no stated per-file limit. This is authenticated operational access, not an unauthenticated dashboard/object URL. |
| Price (live pricing, not a commitment) | The [pricing page](https://modal.com/pricing) lists Volumes at **$0.09/GiB/month**, including 1 TiB/month, and $0.04/GiB network egress; the Volume guide says usage is snapshotted daily and deletion can remain billable for up to four days. Measure actual account/plan pricing before acceptance. |

The Volume guide also says v1 bandwidth is *up to* 2.5 GB/s, not guaranteed; it must not become a throughput acceptance criterion.

## Evidence: object-storage route

[Modal Cloud Bucket Mounts](https://modal.com/docs/guide/cloud-bucket-mounts) supports AWS S3, Cloudflare R2, and Google Cloud Storage through `modal.CloudBucketMount`; the [API reference](https://modal.com/docs/reference/modal.CloudBucketMount) says it mounts a cloud bucket and uses AWS S3 Mountpoint. It requires an existing bucket and, for a private bucket, credentials/secret or OIDC role configuration. No bucket has been selected by this ticket.

| Concern | Documented behavior / consequence |
| --- | --- |
| External readers | The selected external provider exposes its own native APIs/access controls, which can serve non-Modal consumers. Modal's mount is not that provider API and does not itself establish dashboard authorization. |
| I/O/atomicity | Mountpoint is optimized for large sequential reads, does not support every file operation, cannot append, cannot seek-and-write, and the Modal guide says rename is not supported. Modal makes no claim here that `rename`, a multi-object update, or a completion marker is atomic/transactional. |
| Listing | Requires bucket list permission. Prefix layout permits scoped discovery, but listing consistency, pagination, and cost are properties of the selected provider/API; no provider has been chosen, so they cannot be promised here. |
| Durability/cost | Durability, storage/request/list/egress pricing, retention, and failure semantics are external-provider decisions, not Modal Volume pricing. For requestor-pays, the API specifically notes the caller is billed for requests. |

## Current corpus and descriptor constraints

The project currently persists **one compressed `.npz` archive per `save_corpus` call**, not an appendable corpus store: [`data_generator.py` lines 1872–1956](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/data_generator.py#L1872-L1956) serializes arrays, JSON-encodes `diagnostics`, removes nondeterministic `timing`, then calls `np.savez_compressed`. Loading reads the entire archive with `np.load(..., allow_pickle=False)` ([lines 1959–2005](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/data_generator.py#L1959-L2005)). A corrupt/partial visible NPZ therefore has no partial-read contract.

`DataGenerator.iter_batches` is a generator of independent validated batches and changes the seed by batch index ([lines 805–855](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/data_generator.py#L805-L855)). It is useful for shard-shaped work but is **not** an orchestrator-level stable task-ID/retry protocol.

Schema v5 is required; old archives are rejected rather than migrated ([`slots.py` lines 69–81](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/slots.py#L69-L81)). Required tensor inventory includes observed arrays, truth, `cell_id`, masks, and metadata ([lines 117–161](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/slots.py#L117-L161)); a shard descriptor alone is not a replacement for corpus training data.

Descriptors offer a useful *per-shard summary sidecar*, not persistence: `WorldDescriptors` requires unique `(source_id, world_id)` identities and a group/cell identity ([`descriptors.py` lines 311–383](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/descriptors.py#L311-L383)); `concat` requires equal definitions and disjoint identities ([lines 505–540](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/descriptors.py#L505-L540)). `world_descriptors` permits caller-provided original world IDs and source IDs ([lines 1229–1276](https://github.com/pymc-labs/pymc-generator/blob/b941d3a222fb91475cc2b59db14281e2b6498c49/pymc_generator/descriptors.py#L1229-L1276)). Thus descriptor publication needs stable run/shard identity and globally non-colliding world IDs, plus a descriptor-definition/version field.

## Candidate incremental-publication pattern (unvalidated, not a consumer contract)

1. Preassign immutable `run_id`, `shard_id`, logical world-ID range, seed/attempt policy, schema version, and descriptor definition/version. Retries write the same logical shard identity, never a second contribution.
2. A worker writes `runs/<run>/shards/<shard>/corpus.npz` and optional `descriptors.<format>` to a **unique staging location**, closes files, validates/checksums locally, then places them at immutable final keys/paths. Do not expose a manifest entry before all required artifacts exist.
3. Write one small immutable `published/<shard>.json` *last*, containing logical IDs, artifact paths, byte lengths/checksums, schema/config and descriptor versions, accepted/rejected counts, and producer attempt. For a Volume, explicitly `commit()` artifacts before this marker and `commit()` the marker; a reader `reload()`s before reading it. This reduces, but does not eliminate, the inter-commit window.
4. Dashboard/trainer discovers new marker names via an append-only/paginated prefix or an independently serialized index; it verifies referenced artifacts/checksums before counting them. It stores a deduplication set keyed by `run_id/shard_id`, not a byte offset in a mutable manifest. A high-water mark alone is unsafe with out-of-order completion unless an independently ordered index also guarantees gap handling.
5. Reconciliation periodically lists final artifacts/markers and repairs orphaned artifacts or marker-with-missing-artifact cases. It must not silently count a retry twice.

**What “completed” can guarantee:** only that the chosen publisher completed its ordered protocol and the reader observed the corresponding committed/object-visible marker. It cannot prove the worker will never retry, that all dashboard caches reloaded, or atomicity across artifacts and marker. On a Volume, a crash after artifact commit but before marker creation leaves orphaned data. A marker written before an explicit commit may already be visible because background commits also run; explicit commit calls are not the sole visibility boundary. After any commit but before a reader reload, readers can be stale. A partially written marker may also be exposed by background persistence: robust readers must reject incomplete/invalid records, and a backend-supported publication/arbitration mechanism must be chosen and tested before treating this protocol as safe. Across two commits, a marker may become visible after its referenced artifact commit but before consumer validation—hence checksum/size verification and reconciliation.

## Uncertainty and unresolved decisions

* Modal documentation reviewed is live documentation on the retrieval date; no Modal release/API version or filesystem transaction/rename guarantee was supplied. No runtime experiment was authorized, so latency, background-commit timing, retry behavior, list performance, and cross-region visibility are unmeasured.
* The report does not select Volume v1/v2, an object provider, provider region, bucket policy, dashboard access model, retention policy, artifact/descriptor encoding, shard size, index authority, or reconciliation owner. Deterministic IDs alone do not arbitrate simultaneous retry writers; attempts need isolated writes and a specified publication winner/conflict policy before outputs can be called immutable.
* If scientific reproducibility requires invariance to worker count/scheduling/retries, the orchestrator must specify the logical shard/world seed derivation; current `iter_batches`' `base_seed + i` is insufficient evidence of that broader contract.
* Decide whether dashboard needs raw corpus access, descriptor-only summaries, or both. That determines whether it can remain on a small publication-index path or must retrieve NPZ shards.

## Sources

* Modal, [Volumes guide](https://modal.com/docs/guide/volumes), retrieved 2026-10-13.
* Modal, [`modal.Volume` API reference](https://modal.com/docs/reference/modal.Volume), retrieved 2026-10-13.
* Modal, [Cloud bucket mounts guide](https://modal.com/docs/guide/cloud-bucket-mounts), retrieved 2026-10-13.
* Modal, [`modal.CloudBucketMount` API reference](https://modal.com/docs/reference/modal.CloudBucketMount), retrieved 2026-10-13.
* Modal, [Pricing](https://modal.com/pricing), retrieved 2026-10-13.
* pymc-generator source at revision `b941d3a222fb91475cc2b59db14281e2b6498c49`, links embedded above.
