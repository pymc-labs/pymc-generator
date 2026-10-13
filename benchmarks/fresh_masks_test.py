"""Local correctness/timing smoke test: new uniform counts per world, one compile.

No production edits. Invoked only through fresh_masks_envelope.sh.
"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from comparison_methods import Adapter
from comparison_preflight import plain
from comparison_run import arrays, memory
from fixed_allocation import generate_fixed
from template_vs_per_world import CellWorkload, CORPUS_NAMES, dea_prior
from pymc_generator.active_counts import draw_active_counts, plan_active_counts
from pymc_generator.sampler import sample_g_additive, _slice_g_active
from pymc_generator.world_model import sample_structure
from pymc_generator.world_model_template import build_cell_inputs
from pymc_generator.data_generator import DataGenerator, save_corpus, load_corpus

OUT = Path(os.environ['COMPARISON_OUTPUT'])
START = time.monotonic()
EVENTS = []

def event(kind, **data):
    row = plain(dict(event=kind, elapsed_s=time.monotonic()-START, **data))
    EVENTS.append(row)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())

def save(name, data):
    (OUT/name).write_text(json.dumps(plain(data),indent=2)+'\n')

def make_cell(cfg, candidate_id):
    # Task identity, not traversal order, determines count/graph/draw streams.
    count_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed,candidate_id,100]))
    graph_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed,candidate_id,200]))
    assert plan_active_counts(cfg,count_rng) is None
    counts = draw_active_counts(cfg,count_rng,None,candidate_id)
    g = sample_g_additive(graph_rng,cfg,cfg.layout,*counts)
    active_g = _slice_g_active(g,*counts)
    structure = sample_structure(active_g,cfg,graph_rng)
    seed = int(np.random.SeedSequence([cfg.seed,candidate_id,300]).generate_state(1)[0])
    return CellWorkload(active_g,structure,build_cell_inputs(cfg,g,g,structure),seed)

def mask_edges(inputs):
    x = {k:v.copy() for k,v in inputs.items()}
    c,m,j = [x[k] for k in ['active_treatment','active_covariate','active_latent']]
    for key,a in [('g_cy',c),('g_zy',m),('g_dy',j)]:
        x[key] *= a
    for key,a,b in [('g_dc',j,c),('g_dz',j,m),('g_zc',m,c),('g_cc',c,c),('g_zz',m,m)]:
        x[key] *= a[:,None]*b[None,:]
    return x

def main():
    cfg = dataclasses.replace(dea_prior(n_cells=24,draws_per_cell=1,seed=472020),
        n_treatments=15,n_covariates=10,
        n_treatments_active_range=(5,15),n_covariates_active_range=(3,10),
        edge_budget={'cy':(1,10),'zy':(1,10),'dc':0,'dz':0,'dy':0,
                     'zc':(1,4),'cc':(1,4),'zz':(1,4)})
    cfg.validate()
    group = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[-1].lstrip('/')
    controls={k:(group/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}
    assert controls=={'memory.max':str(2*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
    assert os.sched_getaffinity(0)=={0}
    save('environment.json',dict(config=dataclasses.asdict(cfg),controls=controls,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions={k:importlib.metadata.version(k) for k in ['pymc','pytensor','numpy']},
        pytensor_flags=os.environ['PYTENSOR_FLAGS'],affinity=sorted(os.sched_getaffinity(0)),
        thread_settings={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']}))

    # An exact API-contract test, not a stochastic goodness-of-fit assertion.
    class BoundaryRng:
        def __init__(self,upper): self.upper,self.calls=upper,[]
        def integers(self,lo,hi):
            self.calls.append((lo,hi)); return hi-1 if self.upper else lo
    for upper, expected in [(False,(5,3,1)),(True,(15,10,1))]:
        rng=BoundaryRng(upper)
        assert draw_active_counts(cfg,rng,None,0)==expected
        assert rng.calls==[(5,16),(3,11),(1,2)]
    # Descriptive counts only: finite samples are not required to have equal bins.
    histogram=np.zeros((11,8),dtype=int)
    for candidate in range(10000):
        rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,candidate,100]))
        k,m,j=draw_active_counts(cfg,rng,None,candidate)
        histogram[k-5,m-3]+=1
    save('count-contract.json',dict(bounds_inclusive=[[5,15],[3,10]],
        exact_integer_rng_bounds_pass=True,samples=10000,joint_histogram=histogram,
        media_histogram=histogram.sum(axis=1),control_histogram=histogram.sum(axis=0),
        rule='independent discrete uniform counts; prefix masks, not Bernoulli masks or random subsets'))

    t=time.monotonic()
    cells=[make_cell(cfg,i) for i in range(cfg.n_cells)]
    preparation=time.monotonic()-t
    save('world-inputs.json',[dict(candidate_id=i,seed=c.seed,g=c.g,structure=c.structure,
        template_inputs=c.template_inputs) for i,c in enumerate(cells)])
    counts=[(len(c.g['g_cy']),len(c.g['g_zy'])) for c in cells]
    assert len(set(counts))>1
    for i,c in enumerate(cells):
        k,m=counts[i]
        np.testing.assert_array_equal(c.template_inputs['active_treatment'],np.arange(15)<k)
        np.testing.assert_array_equal(c.template_inputs['active_covariate'],np.arange(10)<m)
        assert len(set(c.structure['sat_family']))==1
        assert len(set(c.structure['carryover_family']))==1
        again=make_cell(cfg,i)
        for name,value in c.template_inputs.items():
            np.testing.assert_array_equal(value,again.template_inputs[name])

    adapter=Adapter('specialized',cfg,cells,event)
    t=time.monotonic(); corpus=generate_fixed(cfg,adapter); generation=time.monotonic()-t
    assert corpus is not None
    primary_stages=dict(adapter.times)
    assert sum(e['event']=='build' for e in EVENTS)==1
    assert sum(e['event']=='compile' for e in EVENTS)==1
    compiled=adapter.draws[0]
    t=time.monotonic()
    errors=DataGenerator.validate_corpus(corpus)
    assert not errors, errors
    validation=time.monotonic()-t
    t=time.monotonic(); path=OUT/'fresh-worlds.npz'; save_corpus(corpus,path); persistence=time.monotonic()-t
    t=time.monotonic(); back=load_corpus(path)
    left,right=dict(arrays(corpus)),dict(arrays(back))
    assert left.keys()==right.keys()
    for name,a in left.items():
        b=right[name]; assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name
    assert not DataGenerator.validate_corpus(back)
    readback=time.monotonic()-t
    # Reversed execution after all other worlds: every candidate must replay.
    t=time.monotonic()
    for i in reversed(range(len(cells))):
        adapter.current_cell=i
        adapter.draw(CORPUS_NAMES,1,diagnostic=True)
        assert adapter.draws[0] is compiled
    reverse_replay=time.monotonic()-t

    # Diagnostic counterexample with known edges, not a prior candidate.
    # Same sampler, same seed and identical parameters/innovations across variants.
    diagnostic_cfg=dataclasses.replace(cfg,n_treatments_active_range=(15,15),n_covariates_active_range=(10,10))
    full={k:v.copy() for k,v in make_cell(diagnostic_cfg,90000).template_inputs.items()}
    for key in ['g_dc','g_dz','g_dy','g_zc','g_cc','g_zz']: full[key][:]=0
    full['g_cy'][:]=1; full['g_zy'][:]=1
    checks=[]
    for label, causal in [('independent_additive',False),('causal_interactions',True)]:
        payload={k:v.copy() for k,v in full.items()}
        if causal:
            payload['g_zc'][0,1]=1
            payload['g_cc'][0,1]=1
            payload['g_zz'][0,1]=1
        masked={k:v.copy() for k,v in payload.items()}
        masked['active_treatment'][0]=0
        masked['active_covariate'][0]=0
        masked=mask_edges(masked)
        # Independent reference representation: remove every incident edge but
        # keep disconnected nodes alive. They cannot influence retained nodes/Y.
        isolated={k:v.copy() for k,v in masked.items()}
        isolated['active_treatment'][:]=1; isolated['active_covariate'][:]=1
        f=compiled(payload,seed=77111,draws=1)
        m=compiled(masked,seed=77111,draws=1)
        ref=compiled(isolated,seed=77111,draws=1)
        for key in ['outcome','baseline','contributions','indirect_effects','indirect_effects_by_source']:
            np.testing.assert_allclose(m[key],ref[key],rtol=1e-10,atol=1e-10,err_msg=key)
        np.testing.assert_allclose(m['treatments'][...,1:],ref['treatments'][...,1:],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(m['covariates'][...,1:],ref['covariates'][...,1:],rtol=1e-10,atol=1e-10)
        assert np.count_nonzero(m['treatments'][...,0])==0
        assert np.count_nonzero(m['covariates'][...,0])==0
        naive=f['outcome']-f['contributions'][...,0]-f['covariate_contribution'][...,0]
        delta=float(np.max(np.abs(naive-m['outcome'])))
        if causal:
            assert delta>1e-6, 'negative control did not expose downstream contamination'
        else:
            np.testing.assert_allclose(naive,m['outcome'],rtol=1e-10,atol=1e-10)
        checks.append(dict(case=label,max_abs_posthoc_vs_causal_sales_difference=delta,
            masked_vs_isolated_graph_pass=True))
        np.savez_compressed(OUT/f'{label}.npz',full_sales=f['outcome'],masked_sales=m['outcome'],
            reference_sales=ref['outcome'],posthoc_sales=naive)
    assert len(adapter.models)==len(adapter.draws)==1
    save('result.json',dict(status='pass',primary_candidates=adapter.evaluated,
        accepted=len(adapter.accepted_ids),rejected=adapter.rejected,counts=counts,
        unique_count_pairs=len(set(counts)),build_count=1,compile_count=1,
        same_compiled_function_for_all_primary_replay_and_diagnostic_calls=True,
        preparation_s=preparation,generation_s=generation,stages=primary_stages,
        validation_s=validation,persistence_s=persistence,readback_s=readback,
        primary_pipeline_s=preparation+generation+validation+persistence+readback,
        reversed_replay_s=reverse_replay,masking_checks=checks,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        memory=memory(),memory_events=(group/'memory.events').read_text(),elapsed_s=time.monotonic()-START))
    event('test_pass')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        event('failure',error=repr(e),traceback=traceback.format_exc(),memory=memory())
        raise
