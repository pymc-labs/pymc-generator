#!/usr/bin/env bash
# One bounded invocation, not a retry loop. Keep the namespace owner outside the
# memory-limited scope so a killed benchmark leaves its tmpfs evidence readable.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:?provide a new evidence directory outside the worktree}
mkdir "$DEST"
DEST=$(realpath "$DEST")
export ROOT DEST
unshare --user --map-root-user --mount bash <<'INNER'
set -euo pipefail
scratch=$(mktemp -d)
mount -t tmpfs -o size=2G tmpfs "$scratch"
cleanup() { umount "$scratch"; rmdir "$scratch"; }
trap cleanup EXIT
mkdir "$scratch/output" "$scratch/cache" "$scratch/home" "$scratch/tmp"
export COMPARISON_OUTPUT="$scratch/output"
export PYTENSOR_FLAGS="base_compiledir=$scratch/cache,floatX=float64"
export MPLCONFIGDIR="$scratch/home/matplotlib"
export XDG_CACHE_HOME="$scratch/home/cache"
export TMPDIR="$scratch/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
unit="wayfinder-comparison-$(date +%s)-$$"
start=$(date +%s)
set +e
systemd-run --user --scope --unit="$unit" -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    timeout --signal=TERM --kill-after=5s 895s \
    "$ROOT/.venv/bin/python" benchmarks/comparison_preflight.py > "$scratch/output/process.log" 2>&1
status=$?
set -e
end=$(date +%s)
printf '{"exit_code":%d,"wall_seconds":%d,"memory_max_bytes":8589934592,"swap_max_bytes":0,"scratch_max_bytes":2147483648,"timeout_term_seconds":895,"timeout_kill_seconds":900,"unit":"%s"}\n' \
    "$status" "$((end-start))" "$unit" > "$DEST/envelope.json"
# Only the output directory is retained. Cache and temporary writes share its
# stricter 2-GiB filesystem cap but are discarded, never shared with another run.
cp -a "$scratch/output/." "$DEST/"
exit "$status"
INNER
