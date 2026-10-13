"""Sequential isolated arms; absolute deadline cannot be reset by retries."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(os.environ['ROOT'])
scratch=Path(os.environ['SCRATCH'])
out=scratch/'output'
ledger=[]
# Complete the initial sparse/dense screen before expanding any workloads.
for workload in (0,3):
    for method in ('ordinary','dynamic','specialized'):
        remaining=int(os.environ['RUN_DEADLINE'])-time.time()-4
        if remaining < 15:
            break
        name=f'{workload}-{method}'
        dest=out/name
        dest.mkdir()
        cache=scratch/'cache'/name
        cache.mkdir()
        env={**os.environ,'COMPARISON_OUTPUT':str(dest),'PYTENSOR_FLAGS':f'base_compiledir={cache},floatX=float64'}
        start=time.time()
        entry=dict(arm=name,start_epoch=start,reserved_s=min(150,remaining),status='reserved')
        ledger.append(entry)
        (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
        with (dest/'process.log').open('w') as log:
            result=subprocess.run(['timeout','--signal=TERM','--kill-after=3s',str(min(147,remaining-3)),str(root/'.venv/bin/python'),'benchmarks/modal_targeted_run.py',method,str(workload)],env=env,stdout=log,stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode,elapsed_s=time.time()-start,status='pass' if result.returncode==0 else 'partial_or_failed')
        (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 64*1024**2:
            raise RuntimeError('64-MiB output stop threshold exceeded; scratch filesystem remains hard-bounded')
