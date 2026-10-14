#!/usr/bin/env bash
# Remaining adaptive plan. Existing oracle/reference queue is allowed to finish.
# No deadline reset; every execution uses study_200k_envelope.sh.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [[ -n "${WAIT_PID:-}" ]]; then
  while ps -p "$WAIT_PID" -o stat= | grep -qv '^Z'; do sleep 5; done
fi
# Resume safely even if an owned wrapper outlives a replaced queue shell.
flock benchmarks/study-200k-evidence/.lock true
passed_oracle() {
  python - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path('benchmarks/study-200k-evidence')/sys.argv[1]/'result.json'
try:
 r=json.loads(p.read_text()); assert r['status']=='pass' and len(r['families'])==18
except (OSError,ValueError,AssertionError,KeyError): raise SystemExit(1)
PY
}
if [[ ! -d benchmarks/study-200k-evidence/seed-audit-v2 ]]; then
  bash benchmarks/study_200k_envelope.sh seed-audit-v2 120 universal_external_eager --phase seed_audit --candidates 2
fi
if [[ ! -d benchmarks/study-200k-evidence/o-logical-reference ]]; then
  bash benchmarks/study_200k_envelope.sh o-logical-reference 900 ordinary --mode cvm_compile --phase primary --candidates 8 --seed-contract logical_id
fi
for item in 'ui-logical-primary universal_internal_eager ui-oracle-cached 512 256' 'uil-primary universal_internal_lazy uil-oracle-cached 1024 512'; do
  read -r name arm gate count cohort <<< "$item"
  if [[ -d "benchmarks/study-200k-evidence/$name" ]]; then echo "SKIP existing $name"; continue; fi
  if passed_oracle "$gate"; then
    bash benchmarks/study_200k_envelope.sh "$name" 750 "$arm" --mode cvm_compile --phase primary --candidates "$count" --cohort-size "$cohort" --seed-contract logical_id --rebuild-check
  else
    echo "SKIP $name: oracle incomplete/failed"
  fi
done
for item in 'se specialized_external_eager' 'sl specialized_external_lazy' 'si specialized_internal_eager' 'sil specialized_internal_lazy'; do
  read -r name arm <<< "$item"
  if ! passed_oracle ui-oracle-cached; then echo "SKIP $arm: no passed primitive source"; continue; fi
  if [[ ! -d "benchmarks/study-200k-evidence/$name-oracle" ]]; then
    bash benchmarks/study_200k_envelope.sh "$name-oracle" 1000 "$arm" --mode cvm_compile --phase oracle --all-pairs --primitives-from benchmarks/study-200k-evidence/ui-oracle-cached
  fi
  if [[ -d "benchmarks/study-200k-evidence/$name-primary" ]]; then echo "SKIP existing $name-primary"; continue; fi
  if passed_oracle "$name-oracle"; then
    bash benchmarks/study_200k_envelope.sh "$name-primary" 1900 "$arm" --mode cvm_compile --phase primary --candidates 1024 --seed-contract logical_id --rebuild-check
  else
    echo "SKIP $name-primary: oracle incomplete/failed"
  fi
done
