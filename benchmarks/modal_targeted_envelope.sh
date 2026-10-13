#!/usr/bin/env bash
# One-shot absolute budget: preflight began 2026-10-13T20:34:12Z.
# Execution stops two minutes before the full 15-minute deadline for reporting.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ROOT
export START_EPOCH=1791923652 RUN_DEADLINE=1791924432 FULL_DEADLINE=1791924552
# Explicit second attempt after metadata serialization failure, within the SAME
# absolute reservation. Original evidence and budget are never reset.
test -f "$ROOT/benchmarks/modal-targeted-evidence/budget.json"
export DEST="$ROOT/benchmarks/modal-targeted-evidence/attempt2"
mkdir "$DEST" # refuse reuse
unshare --user --map-root-user --mount bash <<'INNER'
set -euo pipefail
scratch=$(mktemp -d)
mount -t tmpfs -o size=256M tmpfs "$scratch"
cleanup() { cp -a "$scratch/output/." "$DEST/"; umount "$scratch"; rmdir "$scratch"; }
trap cleanup EXIT
mkdir "$scratch/output" "$scratch/cache" "$scratch/home" "$scratch/tmp"
export SCRATCH="$scratch" MPLCONFIGDIR="$scratch/home/matplotlib" XDG_CACHE_HOME="$scratch/home/cache" TMPDIR="$scratch/tmp"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
remaining=$((RUN_DEADLINE-$(date +%s)))
[[ "$remaining" -gt 0 ]]
set +e
systemd-run --user --scope --unit=wayfinder61-comparison -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
 timeout --signal=TERM --kill-after=3s "${remaining}s" taskset -c 0 \
 python benchmarks/modal_targeted_supervise.py > "$scratch/output/supervisor.log" 2>&1
status=$?
set -e
printf '{"exit_code":%d,"finished_epoch":%s}\n' "$status" "$(date +%s)" > "$scratch/output/envelope.json"
exit "$status"
INNER
