# Four-case cold MAP recovery smoke report

**Implementation/preflight is not a fit release.** Independent IMPLEMENT and SAFE-RUN
acceptance of these exact four files, then a separate coordinator release, are required
before `--execute-released`. No generation or optimization is part of engineering tests.
This is an illustrative population, not MAP20 prevalence, a paired baseline experiment,
or a scientific/publication release. No helper or rejected Phase-A source is used.

## Frozen public recipe

The configuration JSON freezes the complete schema, four cases, resources, regime
ranges and both **106-field resolved `dataclasses.asdict(SCMPrior)`** configurations.
Its canonical SHA-256 is
`bd31e76743a8c51e0906dab5f5ecd8e2d9f196219039629c7cbce2d92d9ebbd0`.
The script rejects any semantic change, including unknown keys, before generation.
Resolved constant/TV hashes are respectively
`336c5ae9fa792dc799926654114ea508fb77fca88caffc207142032f7750b18b` and
`ec2affaa248bdb634882d772871f9f4b46efdcdf5a8aa5d1e7b4c75485b4fd0b`.

Exactly one `sample_scm` per case: world seeds 201/202 (constant), 203/204
(time-varying); support seeds 301–304. There are no separate structure seeds.
`make_scm_prior` fixes 2 treatments/1 covariate/1 latent/104 observations,
B=0/discard=0, diverse nonlinearities, texture trajectories, relative outcome noise
(.010,.028), `l_max=8`, smoothness alpha/beta=2/2 and maximum=26 weeks, no floor,
conditioning or treatment shocks. Only baseline-walk range changes, (0,0) versus
(.01,.03). All native family laws/defaults and sampler acceptance remain unchanged.
No screening, restart, replacement, forced graph/family, truth start or retries.

Use only `world.oracle_model(latent="marginal")` and
`model.initial_point(random_seed=support_seed)`. The native likelihood consumes
all 104 rows; the known **Issue42 startup/warmup limitation remains unfixed**.
One native `find_MAP(start=..., vars=list(model.free_RVs), method="L-BFGS-B",
return_raw=True, include_transformed=True, progressbar=False, maxeval=998,
options={maxiter:1000,maxfun:1000,ftol:1e-12,gtol:1e-6,maxls:20})` follows.
A public symbolic gradient check and a scoped observer of PyMC's fallback warning
abort rather than dispatch a non-gradient solver. No optimizer/compiler monkeypatch
or extra solve is used. The objective is native **no-Jacobian free-primitive density
in transformed search coordinates**, not a derived-coordinate mode.
Raw None, missing exits/counts, cap exits, nonfinite objective/gradient/point and
unsuccessful exits are failures. Convergence is not identifiability or calibration.

## Runtime and process contract

The launcher guards a separate supervisor (aggregate TERM5400/KILL30); the supervisor
launches at most four fresh case sessions, each once, serially (TERM900/KILL30).
Linux `/proc`, process groups and subreapers track/reap descendants, including orphaned
new sessions. Real exit codes, stdout/stderr files and hashes, exact child argv/environment,
resource samples and TERM/KILL events are recorded. No SIGALRM is used. An additional
independent stdlib-only watchdog protects the launcher and final report I/O, with
TERM5400/KILL30, resource/process polling and a required startup/completion handshake.
It excludes itself from cleanup and signals the launcher by PID, never its inherited
shell group. The documented GNU timeout is an additional guard, not an assumed
substitute for this internal enforcement.

Cache, artifact allocated/logical bytes, free disk and process RSS/address space are
sampled every **one second** while a child runs, including generation/compilation/solve.
Both soft/hard AS limits must be 16 GiB. Cache plus artifacts stop at 10 GiB less a
128 MiB finalization reserve; filesystem free space must also retain that reserve.
The private 0700 cache must be owned, canonical and exclusively locked, disjoint from
artifacts; symlink/unowned entries are rejected. Each child inherits the same run-root
lock. Every case receives a **new mode0700 cache directory named by case ID beneath
that root**, created exclusively before its process starts; an existing case directory
blocks launch. No two fits share a PyTensor compiledir. Supervisor/preflight imports
use the root, never a case directory. All one-second monitors continue to account for
the **whole root**, including every completed case cache, plus artifacts. OS/library
caches may still warm across fresh processes; no fully cold-machine claim is made.

No new case starts with less than **930+180=1110 seconds** left. Work stops by5220,
finalization is scheduled before5370, aggregate TERM is5400 and hard KILL5430.
SIGTERM/SIGINT stop work and preserve the remaining unattempted ledger. The outer
launcher requests work-stop at5220 and terminates a supervisor still present at5370.
There is no promise to serialize new evidence after an uncatchable SIGKILL; already
fsynced immutable checkpoints survive it. Resource observations are one-second samples,
not continuous/cgroup measurements; per-child `getrusage` also records peak RSS.

Before generation in **each** fresh case, preflight verifies actual imported
PyMC6.2.0/PyTensor3.2.4/SciPy1.18.0 and their C3 package paths; cold-lane paths of
`pymc_generator`, `worlds`, `world_model`, `presets`, `sampler`; actual compiledir,
interpreter/prefix, AS limits, threads, exact PYTHONPATH/cache lock and resolved factory.
It hashes all122 tracked files against accepted git blob OIDs/modes, checks exact HEAD
`0e43c6b708b1584c94fd2271124142a312409a65`, branch and empty index, and records the four
report-file freezes. Porcelain status must contain exactly those four authorized
untracked paths; ignored runtime metadata stays ignored. Metadata versions or
PYTHONPATH text alone never establish approval. The sanitized child environment
contains no inherited credentials.

### Prospective launch identity, not a cross-run metadata digest

The exact launcher is the absolute script path, with explicit `--config`, `--output`,
then the mode flag, as quoted below. Before scientific imports, the runner checks
`sys.orig_argv == [C3_python, *sys.argv]` and the exact `sys.path` prefix
`[cold_lane/scripts, cold_lane]`. An imported preflight, `python -c`, relative script,
extra interpreter flag or different argument order is not the accepted launch context.
The full observed `sys.path` is recorded, never rewritten after execution.

Stable expectations are interpreter/script/cwd/import-path prefix, sourceOID, dependency
versions, four thread variables equal to1 and the case-cache policy. Exact argv,
sanitized environment, output, lock descriptor, process-role arguments, run-cache root
and its four case-directory names are bound **per launch**, not compared blindly across
process roles or runs. Each supervisor/case `.launch.json` is durably written **before
Popen**; the child's preflight checks its actual argv/environment/cache against that
record before imports. A case's `PYTENSOR_FLAGS` change from the root to its declared
case directory is planned before launch, not a post-hoc normalization. The launcher
and watchdog also retain prospective launch records. The preflight receipt declares
all four future case-cache paths without creating them or launching cases.

PID/time/resource observations, platform-specific compiledir suffixes, the remaining
observed sys.path and any threadpool metadata are not cross-run identity fields.
No full runtime-metadata digest or measured threadpool equivalence is claimed. Actual
source origins, versions, AS limits and the active compiledir are independently checked;
this does not replace the exact launch-context check.

## Accepted execution environment / CLI (not executed during implementation)

Run from the accepted cold lane. Substitute **new** absolute private cache/output paths;
the output directory must not exist. Never reuse a previous attempt directory. Compare
`sha256sum` of the four files with the independent review receipt before release. Creating
a private temporary cache is local run setup, not an installation or shared-env change.
Use only the already installed C3 interpreter; do not install or change dependencies.

```bash
cd /home/teemu/pymc-labs/prior-generator/.worktrees/issue-30-map-cold-smoke-report
PY=/home/teemu/pymc-labs/prior-generator/.worktrees/issue-30-identifiable-linear-recovery/.venv/bin/python3
SCRIPT="$PWD/scripts/map_recovery_pilot.py"
CONFIG="$PWD/docs/examples/data/map-recovery-pilot-config.json"
CACHE=$(mktemp -d /tmp/issue30-cold-cache-XXXXXX)  # owned mode0700, private to this run
OUT=/tmp/issue30-cold-report-$(date -u +%Y%m%dT%H%M%SZ)  # must not exist
# Engineering gate only: actual imports/config/provenance; NO generation/oracle/MAP.
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTENSOR_FLAGS="base_compiledir=$CACHE" \
  timeout --signal=TERM --kill-after=30s 600s \
  prlimit --as=17179869184:17179869184 -- "$PY" "$SCRIPT" \
  --config "$CONFIG" --output "$OUT" --preflight

# ONLY after independent exact-file acceptance AND separate coordinator release:
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTENSOR_FLAGS="base_compiledir=$CACHE" \
  timeout --signal=TERM --kill-after=30s 5400s \
  prlimit --as=17179869184:17179869184 -- "$PY" "$SCRIPT" \
  --config "$CONFIG" --output "$OUT" --execute-released
```

A nonzero exit is not permission to rerun/replace worlds. Retain and review the four-case
ledger, including generation failures and unattempted fits. A release is a coordinator
decision, not inferred from the CLI flag or passing tests.

## Durable reports, recovery and counterfactual costs

The output directory is exclusively claimed. Each case has no-clobber attempt, phase,
preflight, native result, process and final checkpoint JSON files, plus stdout/stderr.
Payload and directory fsync plus atomic hard-link publication protect immutable records;
aggregate JSON and Markdown use fsynced atomic replacement. JSON is authoritative if an
interruption falls between its replacement and Markdown replacement. The launcher repairs
an interrupted supervisor's aggregate from completed checkpoints or the last case phase.
No resume/retry is performed. An abrupt machine-wide kill leaves the last durable phase,
not a fabricated optimizer exit.

Independent watchdog startup, actual argv/environment, one-second JSONL samples,
completion and true exit are also retained. Case evidence includes actual graph/structural families; observed and post-fit complete-data
shape/dtype/digests; full configs; free-RV/value order, shapes/dtypes/transforms,
coordinates and flattened slices; raw result and exception tracebacks. Truth is inspected
only after MAP returns/raises, through public `params` and `primitive_parameters`.
Name/shape-verified mappings produce descriptive bias/MAE/RMSE/max errors and retain both
truth/estimate arrays; unavailable marginal parameters are explicitly listed, never guessed.

Monotonic timings separate config/generation, model build, support point, contract
diagnostics, native MAP (inclusive of internal compilation/optimization/reporting), post-fit
diagnostics, checkpoint serialization/I/O and finalization/total. Serialization is also
inside inclusive phase/total timings where a checkpoint occurs; these are not an additive
cost decomposition. The parent process duration includes final-result publication.
`eval_rv_shapes` may compile a small shape
function; that cost is in contract diagnostics, not silently assigned to MAP or reuse savings.
Internal compilation/solve costs remain unknown. No timing from a test double is real-fit evidence.

K is computed from actual graph, structural families, full config, prior conditioning,
saturation-scale constants, likelihood rows and observed/free-value layout/transforms.
K is **not** two regime labels. Missing contracts make total K unavailable and identify a
known-contract subset. Counterfactual avoided work is `(N-K)*C - N*bind`, where C is
build+init+unknown internal compile. Explicit compile scenarios range from zero to the
maximum observed inclusive API duration; assumed bind ranges use fractions0/.01/.1/1
of the maximum setup+inclusive duration. Missing/interrupted API timers instead produce
explicit symbolic unknown compile/bind ranges, never zero-cost measurements. Negative
ranges are retained. These are
sensitivity assumptions, not measured binding/compilation, seconds saved or speedup.
Solve work is unchanged. N20 is a symbolic projection with unknown K20 in1..20, never a
measured twenty-world estimate. Memory savings are not claimed.

## Bounded engineering verification

Only the new-file suite, using C3, TERM600/KILL30:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
timeout --signal=TERM --kill-after=30s 600s "$PY" -m pytest \
  -q -p no:cacheprovider tests/test_map_recovery_pilot.py
```

All optimizer/world results are labeled fixture doubles. Real OS tests run only disposable
sleepers to verify TERM/KILL, orphan/session cleanup, stdout/stderr/exits, resource stopping,
remaining/aggregate deadlines, no-clobber and atomic-checkpoint durability. Config, source,
module/version/path, memory/cache guards, full report/recovery/raw-failure behavior and
graph-derived K/scenario arithmetic are exercised without historical suites or actual fits.
Additional engineering fixtures reject imported/relative/wrong-argv contexts before imports,
check per-launch plans and root-lock inheritance, and stop at a labeled Popen boundary to
prove four distinct empty case caches plus cumulative resource accounting. A real absolute-script
`--preflight` subprocess verifies the declared argv/import context using actual imports/config
only; the `--execute-released` parser fixture replaces execution with an explicit double.
