"""Four sequential isolated memory/CPU scopes with independent compiler caches."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(os.environ['ROOT']); scratch=Path(os.environ['SCRATCH']); out=scratch/'output'
ledger=[]
for arm in ['baseline','rich_legacy','rich_random','rich_all']:
    remaining=int(os.environ['RUN_DEADLINE'])-time.time()-5
    if remaining<30:
        break
    dest=out/arm; dest.mkdir()
    cache=scratch/'cache'/arm; cache.mkdir()
    env={**os.environ,'COMPARISON_OUTPUT':str(dest),'PYTENSOR_FLAGS':f'base_compiledir={cache},floatX=float64'}
    start=time.time(); entry=dict(arm=arm,start_epoch=start,reserved_s=remaining,status='reserved')
    ledger.append(entry); (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
    cmd=['systemd-run','--user','--scope',f'--unit=wayfinder-rich-{arm.replace("_","-")}',
         '-p','MemoryMax=2G','-p','MemorySwapMax=0','-p','CPUQuota=100%','-p','TasksMax=64',
         'timeout','--signal=TERM','--kill-after=3s',str(remaining-3),'taskset','-c','0',
         str(root/'.venv/bin/python'),'benchmarks/rich_components_run.py',arm]
    with (dest/'process.log').open('w') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode,elapsed_s=time.time()-start,
                 status='pass' if result.returncode==0 else 'partial_or_failed')
    (out/'arms.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(entry),flush=True)
    if result.returncode:
        break
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>64*1024**2:
        raise RuntimeError('64-MiB output stop threshold')
(out/'completion.json').write_text(json.dumps(dict(complete=len(ledger)==4 and all(e['exit_code']==0 for e in ledger),finished_epoch=time.time()),indent=2)+'\n')
