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
p=Path('benchmarks/study-200k-evidence')/sys.argv[1]
records=[]
try:
 if (p/'result.json').exists():
  r=json.loads((p/'result.json').read_text()); assert r['status']=='pass'; records+=r['families']
 elif (p/'oracle-progress.json').exists(): records+=json.loads((p/'oracle-progress.json').read_text())
 for gap in p.parent.glob(p.name+'-gap-*'):
  if (gap/'result.json').exists():
   r=json.loads((gap/'result.json').read_text()); assert r['status']=='pass'; records+=r['families']
 assert {tuple(r['family']) for r in records}=={(s,a) for s in range(6) for a in range(3)}
 assert all(r['flag_cases']==['off','on','mixed'] and r['dormant_edge_poison'] for r in records)
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
if [[ ! -d benchmarks/study-200k-evidence/o-rng-probe ]]; then
  bash benchmarks/study_200k_envelope.sh o-rng-probe 360 ordinary --mode cvm_compile --phase rng_probe --candidates 2 --seed-contract logical_id
fi
if [[ ! -d benchmarks/study-200k-evidence/o-logical-reference-fixed ]]; then
  bash benchmarks/study_200k_envelope.sh o-logical-reference-fixed 750 ordinary --mode cvm_compile --phase primary --candidates 8 --seed-contract logical_id
fi
for item in 'se specialized_external_eager' 'sl specialized_external_lazy' 'si specialized_internal_eager' 'sil specialized_internal_lazy'; do
  read -r name arm <<< "$item"
  if ! passed_oracle ui-oracle-cached; then echo "SKIP $arm: no passed primitive source"; continue; fi
  if [[ ! -d "benchmarks/study-200k-evidence/$name-oracle" ]]; then
    bash benchmarks/study_200k_envelope.sh "$name-oracle" 1200 "$arm" --mode cvm_compile --phase oracle --all-pairs --primitives-from benchmarks/study-200k-evidence/ui-oracle-cached
  fi
  if ! passed_oracle "$name-oracle"; then
    missing=$(python - "$name-oracle" <<'PY'
import json,sys
from pathlib import Path
p=Path('benchmarks/study-200k-evidence')/sys.argv[1]/'oracle-progress.json'
r=json.loads(p.read_text()) if p.exists() else []
seen={tuple(x['family']) for x in r}
print(' '.join(f'{s},{a}' for s in range(6) for a in range(3) if (s,a) not in seen))
PY
)
    for pair in $missing; do
      gap="$name-oracle-gap-${pair/,/-}"
      if [[ ! -d "benchmarks/study-200k-evidence/$gap" ]]; then
        bash benchmarks/study_200k_envelope.sh "$gap" 150 "$arm" --mode cvm_compile --phase oracle --candidates 18 --pair "$pair" --primitives-from benchmarks/study-200k-evidence/ui-oracle-cached
      fi
    done
  fi
  if [[ -d "benchmarks/study-200k-evidence/$name-primary" ]]; then echo "SKIP existing $name-primary"; continue; fi
  if passed_oracle "$name-oracle"; then
    bash benchmarks/study_200k_envelope.sh "$name-primary" 1900 "$arm" --mode cvm_compile --phase primary --candidates 1024 --seed-contract logical_id --rebuild-check
  else
    echo "SKIP $name-primary: oracle incomplete/failed"
  fi
done
# Independent root/process repetition of a fully gated measured contender.
# Larger persistence groups amortize cleanup; physical raw-world invariance,
# not identical normalized shard labels, is the contract.
if passed_oracle uil-oracle-cached && [[ ! -d benchmarks/study-200k-evidence/uil-independent-repeat ]]; then
  bash benchmarks/study_200k_envelope.sh uil-independent-repeat 900 universal_internal_lazy --mode cvm_compile --phase primary --candidates 4096 --chunk 128 --cohort-size 2048 --seed 631228 --seed-contract logical_id --rebuild-check
fi
