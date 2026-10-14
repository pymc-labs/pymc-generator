#!/usr/bin/env bash
# One attempt, under immutable shared budget. Arguments: name seconds runner args...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ROOT
export DEST="$ROOT/benchmarks/study-200k-evidence"
mkdir -p "$DEST"
export ATTEMPT_NAME="$1" ATTEMPT_SECONDS="$2"
shift 2
[[ "$ATTEMPT_NAME" =~ ^[a-z0-9-]+$ ]]
export STUDY_ARGS_JSON
STUDY_ARGS_JSON=$(python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")
unshare --user --map-root-user --mount bash <<'INNER'
set -euo pipefail
scratch=$(mktemp -d)
mount -t tmpfs -o size=512M tmpfs "$scratch"
cleanup() { umount "$scratch"; rmdir "$scratch"; }
trap cleanup EXIT
mkdir "$scratch/cache" "$scratch/home" "$scratch/tmp" "$scratch/output"
export SCRATCH="$scratch" MPLCONFIGDIR="$scratch/home/matplotlib" XDG_CACHE_HOME="$scratch/home/cache" TMPDIR="$scratch/tmp"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
exec_python="$ROOT/.venv/bin/python"
"$exec_python" benchmarks/study_200k_supervise.py
INNER
