# Local structural-partition smoke test

Date: 2026-10-15. Status: **PASS, local only**.

## Scope and results

The operator requested the planned training configuration with ten accepted
fresh-structure worlds per worker-compatible saturation × carryover partition,
executed sequentially on this machine, writing to a local folder.

| Measure | Result |
| --- | --- |
| Partitions | 6 saturation × 3 carryover = 18 |
| Target / accepted / persisted | 180 / 180 / 180 worlds |
| Candidates / scientific rejections | 180 / 0 |
| Surplus / shortfalls | 0 / 0 |
| Builds / compilations | Exactly one of each per partition |
| Successful run wall time | 608.31 seconds |
| NPZ payload bytes | 4,823,639 |
| Verified snapshot | 18 members; 144 training / 36 validation worlds |

The balanced ten-per-family design is an integration test, **not** a representative
sample from the production family prior or a performance benchmark. A 180-candidate
sample with no rejection is not evidence of a zero rejection rate. Compiler caches
were not cleared; preceding failed harness attempts warmed part of the cache.

## Artifacts

On the executing machine, relative to the repository:

```
local-runs/structural-smoke-20261015-final/
    run-spec.json
    environment.json
    deferred_research_packing.py
    events.jsonl
    progress.json
    result.json
    store/runs/structural_smoke/tasks/<family_pair>/attempts/first/
        corpus.npz
        completion.json
    store/snapshots/training.json
local-runs/structural-smoke-20261015-final.log
```

These are local artifacts, **not uploaded or public download links**. The generated
folder is gitignored. Completion records contain SHA-256 integrity data, source
hashes, the dependency-lock digest, effective settings, candidate identities, counts
and cell namespaces. `result.json` lists each partition's completion reference.

After the runner completed, a separate Python invocation called
`validated_outputs("structural_smoke")` and `read_snapshot("snapshots/training.json")`:
all 18 outputs and all snapshot members passed again. The runner also compares
persisted top-level arrays against their in-memory dtypes and raw bytes.

## Exact workload and execution

- Capacities: 15 media, 10 controls, one latent; active counts independently uniform
  over media 5–15 and controls 3–10.
- 104 reported weeks, carryover length 8, burn-in 8; fresh structure per world.
- Six saturation families: linear, Hill, logistic, Michaelis–Menten, tanh, root.
- Three world-shared carryover families: none, geometric, Weibull PDF.
- Carlos's composable component probabilities and ranges retained; runtime activity
  masks act inside causal computation. Parameters may differ across channels.
- Family-specialized reusable templates, in-graph component flags, lazy composition,
  explicit CVM with fast-compile rewrites; logical-ID research reseeding.
- Deferred train/validation assignment; split only on the final immutable snapshot.
- One CPU and 2-GiB systemd scope limit; single-thread numerical libraries.
- Envelope: ten initial candidates and at most one ten-candidate top-up per
  partition, 360 total candidate ceiling; aggregate local work capped at 30 minutes.
  The successful invocation had a remaining 1,500-second external timeout.

The harness is `scripts/local_structural_smoke.py`. It imports the isolated research
implementation at commit `d61f98f596c5de885947eb5144897ed0909e7c5d` from a supplied
research checkout, verifies its revision and tracked benchmark cleanliness, and
records source hashes. It does **not** merge that research branch into production.
It saves an explicit adapted packing source as a run artifact: legacy split/mask
repair is removed before normalization, and deferred membership is marked before
final diagnostics. Selected cells are numbered contiguously inside each partition;
world IDs and partition-specific cell namespaces retain their independent identity.

Example invocation (requires that pinned research checkout and locked environment):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout --kill-after=15s 1500 .venv/bin/python scripts/local_structural_smoke.py \
  --research-dir /tmp/pymc-generator-200k-study \
  --output local-runs/<new-run-directory>
```

The recorded run additionally used `systemd-run --user --scope` with
`MemoryMax=2G` and `CPUQuota=100%`. The script refuses to reuse an existing output
directory. The supplied target is deliberately fixed at ten worlds per partition.

## Harness failures preserved

Two earlier attempts are retained under `local-runs/` with logs and failure records:

1. `structural-smoke-20261015T065941Z`: generation passed for the first partition,
   but the harness incorrectly assumed optional `sat_family` reporting was enabled.
   Fixed by checking the structural family/compiled partition, plus persisted
   carryover labels. No completion was published by that failed attempt.
2. `structural-smoke-20261015-retry`: the first partition published ten worlds;
   the next failed validation because the harness used global candidate IDs as
   corpus-local cell IDs. Fixed by using contiguous local IDs and partition-specific
   cell namespaces, retaining logical candidate IDs in publication metadata.

The successful-run totals above exclude these debugging attempts and their outputs.
They must not be pooled with the final snapshot.

## Follow-up

The subsequent **18 parallel Modal worker partitions × 100 accepted worlds** test
also passed; see the [Modal report](modal-structural-smoke.md) for its distinct
cross-container evidence, storage access, budget estimate and limitations. The local
result alone did not establish that behavior. Neither harness is a production driver.

Both reports accompany the implementation PR for Carlos's review. The local JSON
summary is also [committed as evidence](evidence/structural-smoke/local-result.json);
large payloads remain in the local artifact folder rather than Git.
