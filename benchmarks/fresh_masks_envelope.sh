#!/usr/bin/env bash
# Small local test requested by Teemu; self-imposed ten-minute total stop.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ROOT START_EPOCH=1791930009 RUN_DEADLINE=1791930429 FULL_DEADLINE=1791930609
# Second attempt corrects a diagnostic assumption about relative noise. Preserve
# first-attempt evidence and the same absolute budget; never reset the ledger.
test -f "$ROOT/benchmarks/fresh-masks-evidence/budget.json"
export DEST="$ROOT/benchmarks/fresh-masks-evidence/attempt2"
mkdir "$DEST"
unshare --user --map-root-user --mount bash <<'INNER'
set -euo pipefail
scratch=$(mktemp -d)
mount -t tmpfs -o size=128M tmpfs "$scratch"
cleanup() { cp -a "$scratch/output/." "$DEST/"; umount "$scratch"; rmdir "$scratch"; }
trap cleanup EXIT
mkdir "$scratch/output" "$scratch/cache" "$scratch/home" "$scratch/tmp"
export COMPARISON_OUTPUT="$scratch/output" PYTENSOR_FLAGS="base_compiledir=$scratch/cache,floatX=float64"
export MPLCONFIGDIR="$scratch/home/matplotlib" XDG_CACHE_HOME="$scratch/home/cache" TMPDIR="$scratch/tmp"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
remaining=$((RUN_DEADLINE-$(date +%s)-3))
[[ "$remaining" -gt 0 ]]
set +e
systemd-run --user --scope --unit=wayfinder61-fresh-masks -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
 timeout --signal=TERM --kill-after=3s "${remaining}s" taskset -c 0 \
 "$ROOT/.venv/bin/python" benchmarks/fresh_masks_test.py > "$scratch/output/process.log" 2>&1
status=$?
set -e
printf '{"exit_code":%d,"finished_epoch":%s}\n' "$status" "$(date +%s)" > "$scratch/output/envelope.json"
exit "$status"
INNER
