#!/usr/bin/env bash
# Internal runner: call through comparison_budget.py, never directly.
set -euo pipefail
: "${ROOT:?}" "${DEST:?}" "${SCRATCH_BYTES:?}" "${INVOCATION_SECONDS:?}" "${WORKLOAD:?}" "${REPEAT:?}"
unshare --user --map-root-user --mount bash <<'INNER'
set -euo pipefail
scratch=$(mktemp -d)
mount -t tmpfs -o "size=$SCRATCH_BYTES" tmpfs "$scratch"
cleanup() { umount "$scratch"; rmdir "$scratch"; }
trap cleanup EXIT
mkdir "$scratch/output" "$scratch/cache" "$scratch/home" "$scratch/tmp"
export COMPARISON_OUTPUT="$scratch/output"
export PYTENSOR_FLAGS="base_compiledir=$scratch/cache,floatX=float64"
export MPLCONFIGDIR="$scratch/home/matplotlib" XDG_CACHE_HOME="$scratch/home/cache" TMPDIR="$scratch/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
unit="wayfinder-comparison-$(date +%s)-$$"
set +e
systemd-run --user --scope --unit="$unit" -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    timeout --signal=TERM --kill-after=5s "$((INVOCATION_SECONDS-5))s" \
    "$ROOT/.venv/bin/python" benchmarks/comparison_run.py --workload "$WORKLOAD" --repeat "$REPEAT" \
    > "$scratch/output/process.log" 2>&1
status=$?
set -e
cp -a "$scratch/output/." "$DEST/"
printf '{"exit_code":%d,"unit":"%s","memory_max_bytes":8589934592,"scratch_max_bytes":%d,"invocation_max_seconds":%d}\n' \
    "$status" "$unit" "$SCRATCH_BYTES" "$INVOCATION_SECONDS" > "$DEST/envelope.json"
exit "$status"
INNER
