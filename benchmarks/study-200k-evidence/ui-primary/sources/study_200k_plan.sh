#!/usr/bin/env bash
# Remaining adaptive plan. Existing oracle/reference queue is allowed to finish.
# No deadline reset; every execution uses study_200k_envelope.sh.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [[ -n "${WAIT_PID:-}" ]]; then
  while ps -p "$WAIT_PID" -o stat= | grep -qv '^Z'; do sleep 5; done
fi
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
for item in 'ui-primary universal_internal_eager ui-oracle-cached' 'uil-primary universal_internal_lazy uil-oracle-cached'; do
  read -r name arm gate <<< "$item"
  if passed_oracle "$gate"; then
    bash benchmarks/study_200k_envelope.sh "$name" 750 "$arm" --mode cvm_compile --phase primary --candidates 1024 --rebuild-check
  else
    echo "SKIP $name: oracle incomplete/failed"
  fi
done
for item in 'se specialized_external_eager' 'sl specialized_external_lazy' 'si specialized_internal_eager' 'sil specialized_internal_lazy'; do
  read -r name arm <<< "$item"
  if ! passed_oracle ui-oracle-cached; then echo "SKIP $arm: no passed primitive source"; continue; fi
  bash benchmarks/study_200k_envelope.sh "$name-oracle" 1000 "$arm" --mode cvm_compile --phase oracle --all-pairs --primitives-from benchmarks/study-200k-evidence/ui-oracle-cached
  if passed_oracle "$name-oracle"; then
    bash benchmarks/study_200k_envelope.sh "$name-primary" 1900 "$arm" --mode cvm_compile --phase primary --candidates 1024 --rebuild-check
  else
    echo "SKIP $name-primary: oracle incomplete/failed"
  fi
done
