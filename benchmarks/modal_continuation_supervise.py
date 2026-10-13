"""Approved 20-minute continuation, sequential arms and immutable ledger.

Repetition means the same fixed inputs in a fresh process/cache, not new worlds.
"""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ['ROOT'])
scratch = Path(os.environ['SCRATCH'])
out = scratch / 'output'
ledger = []
# Missing evidence first, then reversed-order dense repeats to expose variation.
plan = [(0,'dynamic',0), (3,'ordinary',0), (3,'dynamic',0),
        (3,'specialized',0), (3,'specialized',1), (3,'dynamic',1),
        (3,'ordinary',1)]
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for workload, method, repeat in plan:
    remaining = int(os.environ['RUN_DEADLINE']) - time.time() - 5
    if remaining < 30:
        break
    name = f'{workload}-{method}-r{repeat}'
    dest = out/name
    dest.mkdir()
    cache = scratch/'cache'/name
    cache.mkdir()
    env = {**os.environ, 'COMPARISON_OUTPUT':str(dest),
           'PYTENSOR_FLAGS':f'base_compiledir={cache},floatX=float64'}
    start = time.time()
    entry = dict(arm=name,start_epoch=start,reserved_s=remaining,status='reserved')
    ledger.append(entry)
    (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
    with (dest/'process.log').open('w') as log:
        result = subprocess.run(['timeout','--signal=TERM','--kill-after=3s',str(remaining-3),
            str(root/'.venv/bin/python'),'benchmarks/modal_targeted_run.py',method,str(workload)],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode,elapsed_s=time.time()-start,
                 status='pass' if result.returncode==0 else 'partial_or_failed')
    (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(entry),flush=True)
    if result.returncode:
        # Preserve the first blocker; do not blindly repeat a harness failure.
        break
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 64*1024**2:
        raise RuntimeError('64-MiB output stop threshold exceeded')
(out/'completion.json').write_text(json.dumps(dict(planned=len(plan),executed=len(ledger),
    complete=len(ledger)==len(plan) and all(e['exit_code']==0 for e in ledger),
    finished_epoch=time.time()),indent=2)+'\n')
