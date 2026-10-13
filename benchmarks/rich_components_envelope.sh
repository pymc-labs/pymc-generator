#!/usr/bin/env bash
# New self-imposed 15-minute envelope for the requested component comparison.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ROOT START_EPOCH=1791931578 RUN_DEADLINE=1791932298 FULL_DEADLINE=1791932478
export DEST="$ROOT/benchmarks/rich-components-evidence"
mkdir "$DEST"
printf '{"start_epoch":%s,"run_deadline":%s,"full_deadline":%s,"reserved_seconds":900,"status":"reserved","memory_max_bytes":2147483648,"scratch_max_bytes":268435456}\n' "$START_EPOCH" "$RUN_DEADLINE" "$FULL_DEADLINE" > "$DEST/budget.json"
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
remaining=$((RUN_DEADLINE-$(date +%s)-3))
[[ "$remaining" -gt 0 ]]
set +e
timeout --signal=TERM --kill-after=3s "${remaining}s" python benchmarks/rich_components_supervise.py > "$scratch/output/supervisor.log" 2>&1
status=$?
set -e
printf '{"exit_code":%d,"finished_epoch":%s}\n' "$status" "$(date +%s)" > "$scratch/output/envelope.json"
exit "$status"
INNER
