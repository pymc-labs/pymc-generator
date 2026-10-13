"""One-of-seven trajectory kernel: eager switch versus lazy ifelse.

Deterministic kernel experiment, NOT a replacement for the compositional DGP.
Inputs carry the base series, per-slot selector, activity and parameters.
"""
import json
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.ifelse import ifelse
from pytensor.link.vm import VMLinker

OUT=Path(os.environ['COMPARISON_OUTPUT'])
START=time.monotonic()
CALLS=[]
SHAPES=['plain','seasonal','trend','flighting','onset','offset','level_jump']

class CountBranch(Op):
    __props__=('slot','shape')
    def __init__(self,slot,shape): self.slot,self.shape=slot,shape
    def make_node(self,x):
        x=pt.as_tensor_variable(x); return Apply(self,[x],[x.type()])
    def perform(self,node,inputs,outputs):
        CALLS.append((self.slot,self.shape)); outputs[0][0]=inputs[0]
    def infer_shape(self,fgraph,node,shapes): return shapes

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def build(method,slots=25,weeks=108,instrument=False):
    base=pt.dmatrix('base'); selector=pt.lvector('shape_id'); active=pt.lvector('active'); pars=pt.dmatrix('parameters')
    t=np.arange(-4,weeks-4,dtype='float64')
    ramp=np.maximum(t,0)/max(1,weeks-5)
    cols=[]
    for k in range(slots):
        b=base[:,k]; p=pars[k]
        seasonal=p[0]*pt.sin(2*np.pi*t/p[1]+p[2])
        trend=p[3]*ramp
        is_media=k<min(15,slots)
        values=[b,
            b*pt.exp(seasonal) if is_media else b+seasonal,
            b*pt.exp(trend) if is_media else b+trend,
            b*pt.cast(pt.lt(pt.mod(t+p[4],p[5]),p[6]),'float64'),
            b*pt.cast(pt.ge(t,p[7]),'float64'),
            b*pt.cast(pt.lt(t,p[8]),'float64'),
            b*pt.switch(pt.ge(t,p[9]),p[10],1.0) if is_media else b+pt.switch(pt.ge(t,p[9]),p[11],0.0)]
        if instrument:
            values=[CountBranch(k,i)(v) for i,v in enumerate(values)]
        result=values[0]
        for i in range(1,len(values)):
            cond=pt.eq(selector[k],i)
            result=ifelse(cond,values[i],result) if method=='lazy' else pt.switch(cond,values[i],result)
        zero=pt.zeros_like(b)
        result=ifelse(pt.neq(active[k],0),result,zero) if method=='lazy' else pt.switch(pt.neq(active[k],0),result,zero)
        cols.append(result)
    output=pt.stack(cols,axis=1)
    mode=Mode(linker=VMLinker(use_cloop=False,lazy=True),optimizer=None if instrument else 'fast_compile')
    start=time.monotonic(); fn=pytensor.function([base,selector,active,pars],output,mode=mode)
    return fn,time.monotonic()-start

def oracle(base,selectors,active,p):
    t=np.arange(-4,base.shape[0]-4,dtype=float); ramp=np.maximum(t,0)/max(1,base.shape[0]-5)
    result=np.zeros_like(base)
    for k,s in enumerate(selectors):
        if not active[k]: continue
        b=base[:,k]; q=p[k]; media=k<15
        if s==0: v=b
        elif s==1:
            a=q[0]*np.sin(2*np.pi*t/q[1]+q[2]); v=b*np.exp(a) if media else b+a
        elif s==2: v=b*np.exp(q[3]*ramp) if media else b+q[3]*ramp
        elif s==3: v=b*((t+q[4])%q[5]<q[6])
        elif s==4: v=b*(t>=q[7])
        elif s==5: v=b*(t<q[8])
        else: v=b*np.where(t>=q[9],q[10],1) if media else b+np.where(t>=q[9],q[11],0)
        result[:,k]=v
    return result

def inputs(seed,case):
    rng=np.random.default_rng(seed)
    base=np.cumsum(rng.normal(0,.1,(108,25)),axis=0)
    base[:,:15]=np.logaddexp(0,base[:,:15])+1
    selector=rng.integers(0,7,25,dtype='int64')
    if case=='all_seasonal': selector[:]=1
    if case=='all_trend': selector[:]=2
    active=np.ones(25,dtype='int64')
    if case=='mixed_counts':
        active[:15]=(np.arange(15)<rng.integers(5,16))
        active[15:]=(np.arange(10)<rng.integers(3,11))
    p=np.zeros((25,12))
    p[:,0]=rng.uniform(.1,.5,25);p[:,1]=52;p[:,2]=rng.uniform(0,2*np.pi,25)
    p[:,3]=rng.uniform(-1,1,25);p[:,4]=rng.uniform(0,4,25);p[:,5]=rng.integers(4,14,25)
    p[:,6]=np.maximum(1,np.floor(p[:,5]*.5));p[:,7]=rng.integers(5,27,25);p[:,8]=rng.integers(63,94,25)
    p[:,9]=rng.integers(1,104,25);p[:,10]=rng.uniform(.5,2,25);p[:,11]=rng.uniform(-1,1,25)
    return base,selector,active,p

def main():
    group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[-1].lstrip('/')
    assert (group/'memory.max').read_text().strip()==str(2*1024**3)
    assert (group/'memory.swap.max').read_text().strip()=='0'
    assert os.sched_getaffinity(0)=={0}
    # Observable branch execution, with instrumentation excluded from timings.
    probes={}
    data=list(inputs(18,'mixed_counts'))
    data=[data[0][:8,:2],np.array([2,5],dtype='int64'),np.array([1,0],dtype='int64'),data[3][:2]]
    for method in ['eager','lazy']:
        probe,_=build(method,slots=2,weeks=8,instrument=True)
        CALLS.clear(); value=probe(*data)
        probes[method]=list(CALLS)
        assert len(CALLS)==(14 if method=='eager' else 1),CALLS
        if method=='lazy': assert CALLS==[(0,2)]
    save('branch-probe.json',dict(shapes=SHAPES,executed=probes,
        explanation='two slots, seven branches, second slot inactive; lazy executes only selected active branch'))
    fns={}; compiles={}
    for method in ['eager','lazy']:
        fn,seconds=build(method);fns[method]=fn;compiles[method]=seconds
    rows=[]
    for case in ['all_seasonal','all_trend','mixed_full','mixed_counts']:
        batch=[inputs(6000+i,case) for i in range(128)]
        for data in batch:
            a=fns['eager'](*data);b=fns['lazy'](*data);reference=oracle(*data)
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(a,reference,rtol=1e-12,atol=1e-12)
        for repeat in range(3):
            for method in (['eager','lazy'] if repeat%2==0 else ['lazy','eager']):
                start=time.perf_counter()
                for data in batch: fns[method](*data)
                elapsed=time.perf_counter()-start
                rows.append(dict(case=case,repeat=repeat,method=method,calls=128,seconds=elapsed,
                    milliseconds_per_call=elapsed/128*1000))
        save('timings.json',rows)
    save('result.json',dict(status='pass',shapes=SHAPES,compile_s=compiles,
        compile_count_per_method=1,slots=25,weeks=108,correctness_cases=512,
        branch_probe=probes,timings=rows,pytensor_version=pytensor.__version__,
        mode='Python VMLinker(lazy=True), fast_compile; instrumented probe optimizer=None',
        elapsed_s=time.monotonic()-START,self_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope_peak_bytes=int((group/'memory.peak').read_text()),memory_events=(group/'memory.events').read_text(),
        scope='deterministic shape application only; excludes random walks, parameter generation, adstock/saturation, causal graph, acceptance and persistence'))

if __name__=='__main__':
    try: main()
    except Exception as e:
        save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
