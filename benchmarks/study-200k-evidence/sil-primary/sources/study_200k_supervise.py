"""Hard absolute/cumulative budgets, bounded tmpfs, process-tree kill, durable ledger."""
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

START = 1791933087
DEADLINE = 1791954687
CUTOFF = DEADLINE - 900
root = Path(os.environ['ROOT']); dest = Path(os.environ['DEST']); scratch = Path(os.environ['SCRATCH'])
name = os.environ['ATTEMPT_NAME']; args = json.loads(os.environ['STUDY_ARGS_JSON'])
lock = (dest / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
ledger_path = dest / 'ledger.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(start_epoch=START, deadline_epoch=DEADLINE,
    scientific_cutoff_epoch=CUTOFF, memory_max_bytes=2*1024**3, scratch_max_bytes=512*1024**2,
    retained_max_bytes=2*1024**3, cpu=0, cpu_quota='100%', threads=1, attempts=[])
def save():
    tmp=ledger_path.with_suffix('.tmp'); tmp.write_text(json.dumps(ledger,indent=2)+'\n'); tmp.replace(ledger_path)
def size(path): return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
approach=args[0]
spent=sum(e.get('elapsed_s',e.get('reserved_s',2700)) for e in ledger['attempts'] if e['approach']==approach)
# Reserve outer scope teardown/evidence-copy time as well as worker time.
# An earlier timeout accumulated 2700.310s of wrapper wall time; do not hide
# that sub-second overrun or reset its ledger, and prevent further ones.
seconds=min(int(os.environ['ATTEMPT_SECONDS']), int(CUTOFF-time.time()-10), int(2700-spent-5))
assert seconds > 0, 'absolute or approach budget exhausted'
assert size(dest)<2*1024**3-600*1024**2, 'reserve 512MiB output plus supervisor logs; retained evidence cap'
output=dest/name; output.mkdir()
unit=f'wayfinder-200k-{name}'
env={**os.environ, 'COMPARISON_OUTPUT':str(scratch/'output'),
     'PYTENSOR_FLAGS':f'base_compiledir={scratch}/cache,floatX=float64'}
entry=dict(name=name,approach=approach,args=args,start_epoch=time.time(),reserved_s=seconds,status='reserved',
           cache='cold isolated tmpfs',unit=unit)
ledger['attempts'].append(entry); save()
cmd=['systemd-run','--user','--scope',f'--unit={unit}', '-p','MemoryMax=2G','-p','MemorySwapMax=0',
     '-p','CPUQuota=100%','-p','TasksMax=64', 'timeout','--signal=TERM','--kill-after=3s',str(seconds),
     'taskset','-c','0',str(root/'.venv/bin/python'),'benchmarks/study_200k_run.py',*args]
(output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
sources=output/'sources'; sources.mkdir()
for source in ['study_200k_run.py','study_200k_methods.py','study_200k_packing.py','study_200k_envelope.sh','study_200k_supervise.py','study_200k_plan.sh']:
    shutil.copy2(root/'benchmarks'/source,sources/source)
# Per-file protection; tmpfs globally bounds compiler scratch plus worker output.
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
process=None
stopped=False
def terminate(signum=None,frame=None):
    global stopped
    stopped=True
    subprocess.run(['systemctl','--user','kill','--kill-whom=all',f'{unit}.scope'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if process and process.poll() is None: process.terminate()
signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
try:
    with (output/'process.log').open('w') as log, (output/'accounting.jsonl').open('w') as account:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        end=time.monotonic()+seconds+5; next_copy=0
        while process.poll() is None:
            if time.time()>=CUTOFF or time.monotonic()>=end or stopped: terminate(); break
            group=subprocess.run(['systemctl','--user','show',f'{unit}.scope','--property=ControlGroup','--value'],capture_output=True,text=True).stdout.strip()
            if group:
                cg=Path('/sys/fs/cgroup')/group.lstrip('/')
                record=dict(epoch=time.time(),scratch_bytes=size(scratch))
                for f in ['memory.current','memory.peak','memory.events','cpu.stat','memory.max','memory.swap.max','cpu.max']:
                    try: record[f]=(cg/f).read_text().strip()
                    except FileNotFoundError: pass
                account.write(json.dumps(record)+'\n'); account.flush()
            if time.monotonic()>=next_copy:
                shutil.copytree(scratch/'output',output,dirs_exist_ok=True); next_copy=time.monotonic()+15
                if size(dest)>2*1024**3: terminate(); raise RuntimeError('retained evidence cap')
            time.sleep(1)
        try: rc=process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            terminate(); os.killpg(process.pid,signal.SIGKILL); rc=process.wait()
finally:
    terminate()
    subprocess.run(['systemctl','--user','kill','--signal=SIGKILL','--kill-whom=all',f'{unit}.scope'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    shutil.copytree(scratch/'output',output,dirs_exist_ok=True)
    entry.update(exit_code=process.returncode if process else None,elapsed_s=time.time()-entry['start_epoch'],
                 finished_epoch=time.time(),artifact_bytes=size(output),scratch_final_bytes=size(scratch))
    entry['status']='pass' if entry['exit_code']==0 else 'partial_or_failed'
    save()
print(json.dumps(entry),flush=True)
raise SystemExit(entry['exit_code'] or 0)
