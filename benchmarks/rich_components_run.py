"""Matched local candidate-drawing comparison for runtime trajectory flags.

No production edits. The rich specialization extends the earlier benchmark's
scope guard, retaining its exact builder delegation and scientific helpers.
"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
from comparison_methods import clone_function, controlled_evaluate, active_result, digest
from comparison_preflight import plain
from fresh_masks_test import make_cell
from template_vs_per_world import CORPUS_NAMES, dea_prior
from pymc_generator import symbolic_graph as sg, world_model as wm, world_model_template as wt
from pymc_generator.sampler import SATURATION_FAMILY_KEYS, CARRYOVER_FAMILY_KEYS
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from pymc_generator.trajectories import structural_key

OUT=Path(os.environ['COMPARISON_OUTPUT'])
START=time.monotonic()
GROUP=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[-1].lstrip('/')

def memory():
    return dict(self_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope_peak_bytes=int((GROUP/'memory.peak').read_text()),
        scope_current_bytes=int((GROUP/'memory.current').read_text()))

def save(name,data):
    (OUT/name).write_text(json.dumps(plain(data),indent=2)+'\n')

def event(kind,**data):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(plain(dict(event=kind,elapsed_s=time.monotonic()-START,**data)))+'\n')
        f.flush(); os.fsync(f.fileno())

def build_specialized_rich(cfg,inputs,weeks):
    wt.check_template_supported(cfg)
    if cfg.mechanism_priors_enabled or cfg.record_walk_parameters:
        raise ValueError('this experiment does not extend mechanism/walk-report support')
    sat=[i for i,k in enumerate(SATURATION_FAMILY_KEYS) if cfg.saturation_family_probs[k]>0]
    ad=[i for i,k in enumerate(CARRYOVER_FAMILY_KEYS) if cfg.carryover_family_probs[k]>0]
    assert len(sat)==len(ad)==1
    def parameters(*args,**kwargs):
        kwargs['sat_family']=np.full(cfg.n_treatments,sat[0],dtype='int64')
        kwargs['carryover_family']=np.full(cfg.n_treatments,ad[0],dtype='int64')
        return wt._scm_params(*args,**kwargs)
    def carry(col,params,k,*,dynamic_family=False):
        return sg._carryover_col(col,params,k,dynamic_family=not isinstance(params['carryover_family'],np.ndarray))
    def saturate(col,scale,params,k,*,dynamic_family=False):
        return sg._saturate_col(col,scale,params,k,dynamic_family=not isinstance(params['sat_family'],np.ndarray))
    graph=clone_function(sg.build_symbolic_graph,_carryover_col=carry,_saturate_col=saturate)
    builder=clone_function(wt.build_world_model_template,_scm_params=parameters,build_symbolic_graph=graph)
    return builder(cfg,inputs,weeks)

def check(value,cell):
    assert all(np.isfinite(v).all() for v in value.values())
    np.testing.assert_allclose(value['baseline']+value['contributions'].sum(-1)+value['indirect_effects'],value['outcome'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(value['indirect_effects_by_source'].sum(-1),value['indirect_effects'],rtol=1e-10,atol=1e-10)
    for role,fields in [('treatment',['treatments','contributions']),('covariate',['covariates','covariate_contribution'])]:
        inactive=cell.template_inputs[f'active_{role}']==0
        for field in fields:
            assert np.count_nonzero(value[field][...,inactive])==0,field

def main():
    arm=sys.argv[1]
    assert arm in ['baseline','rich_legacy','rich_random','rich_all']
    base=dataclasses.replace(dea_prior(n_cells=24,draws_per_cell=1,seed=472020),
        n_treatments=15,n_covariates=10,n_treatments_active_range=(5,15),n_covariates_active_range=(3,10),
        edge_budget={'cy':(1,10),'zy':(1,10),'dc':0,'dz':0,'dy':0,'zc':(1,4),'cc':(1,4),'zz':(1,4)},
        covariate_hf_sigma_range=(0.05,0.15),covariate_pulse_prob_range=(0.1,0.3),covariate_pulse_amp_range=(0.1,0.3))
    cfg=base if arm=='baseline' else dataclasses.replace(base,**{
        f'{role}_{component}_inclusion_prob':0.5 for role in TRAJECTORY_INPUTS for component in TRAJECTORY_COMPONENTS})
    cfg.validate()
    controls={k:(GROUP/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}
    assert controls=={'memory.max':str(2*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
    assert os.sched_getaffinity(0)=={0}
    save('environment.json',dict(arm=arm,config=dataclasses.asdict(cfg),controls=controls,
        versions={k:importlib.metadata.version(k) for k in ['pymc','pytensor','numpy']},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        pytensor_flags=os.environ['PYTENSOR_FLAGS'],mode='FAST_COMPILE',common_outputs=CORPUS_NAMES,
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']}))
    event('start',memory=memory())
    t=time.monotonic()
    cells=[make_cell(cfg,i) for i in range(cfg.n_cells)]
    base_cells=[make_cell(base,i) for i in range(cfg.n_cells)]
    for cell,reference in zip(cells,base_cells):
        assert cell.seed==reference.seed
        for name,value in cell.g.items(): np.testing.assert_array_equal(value,reference.g[name])
        for name,value in cell.structure.items():
            if not name.startswith('use_'): np.testing.assert_array_equal(value,reference.structure[name])
        if arm in ['rich_legacy','rich_all']:
            for role in TRAJECTORY_INPUTS:
                width=len(cell.g['g_cy' if role=='treatment' else 'g_zy'])
                for comp in TRAJECTORY_COMPONENTS:
                    key=structural_key(role,comp)
                    enabled=arm=='rich_all' or comp in ('hf','pulse')
                    cell.structure[key][:]=enabled
                    payload=cell.template_inputs[key]
                    payload[:]=0
                    payload[:width]=enabled
    preparation=time.monotonic()-t
    counts=[(len(c.g['g_cy']),len(c.g['g_zy'])) for c in cells]
    flags={role:{component:sum(int(c.structure.get(structural_key(role,component),np.array([])).sum()) for c in cells)
                 for component in TRAJECTORY_COMPONENTS} for role in TRAJECTORY_INPUTS}
    save('inputs.json',dict(counts=counts,enabled_counts=flags,
        cells=[dict(seed=c.seed,g=c.g,structure=c.structure,template_inputs=c.template_inputs) for c in cells]))
    event('build_start')
    t=time.monotonic(); model,_,_=build_specialized_rich(cfg,cells[0].template_inputs,cfg.n_time_steps); build=time.monotonic()-t
    event('build_complete',seconds=build,memory=memory())
    t=time.monotonic(); draw=wt.compile_template_draw_fn(model,CORPUS_NAMES); compile_s=time.monotonic()-t
    graph_nodes=len(draw.fn.maker.fgraph.apply_nodes) if hasattr(draw,'fn') else None
    event('compile_complete',seconds=compile_s,memory=memory())
    hashes={}; saved={}; timings=[]
    orders=[list(range(24)),list(reversed(range(24))),list(np.random.default_rng(171).permutation(24))]
    for repeat,order in enumerate(orders):
        drawing=checking=0.0; durations=[]
        for i in order:
            t=time.monotonic(); result=draw(cells[i].template_inputs,seed=cells[i].seed,draws=1); dt=time.monotonic()-t
            drawing+=dt; durations.append(dt)
            t=time.monotonic(); check(result,cells[i]); h=digest(result)
            if repeat:
                assert h==hashes[i],f'{arm}/{repeat}/{i}'
            else:
                hashes[i]=h
                for name,value in result.items(): saved[f'w{i:02d}_{name}']=value
            checking+=time.monotonic()-t
        timings.append(dict(repeat=repeat,draw_s=drawing,check_s=checking,per_world_draw_s=durations,memory=memory()))
        event('pass_complete',**timings[-1])
    t=time.monotonic(); np.savez_compressed(OUT/'raw-candidates.npz',**saved)
    with np.load(OUT/'raw-candidates.npz',allow_pickle=False) as back:
        assert set(back.files)==set(saved)
        for name,a in saved.items():
            b=back[name]; assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name
    persistence=time.monotonic()-t
    primary=dict(status='pass',arm=arm,primary_candidates=24,diagnostic_replays=48,
        counts=counts,component_enabled_counts=flags,prepare_and_pair_check_s=preparation,
        build_s=build,compile_s=compile_s,graph_apply_nodes=graph_nodes,
        passes=timings,median_draw_s=float(np.median([t['draw_s'] for t in timings])),
        persistence_and_readback_s=persistence,primary_memory=memory(),primary_elapsed_s=time.monotonic()-START,
        output_bytes=(OUT/'raw-candidates.npz').stat().st_size,
        acceptance_and_corpus_packing='not exercised: common-output candidate comparison')
    save('primary-result.json',primary)
    # Independent builder comparison on identical primitive inputs, separate
    # from primary timing/memory. Only one representative rich-random world.
    oracle=None
    if arm=='rich_random':
        event('oracle_start')
        names=tuple(dict.fromkeys(CORPUS_NAMES+tuple(rv.name for rv in model.free_RVs)))
        t=time.monotonic(); audit=wt.compile_template_draw_fn(model,names)
        joint=audit(cells[0].template_inputs,seed=771224,draws=1)
        ordinary,_,_=wm.build_world_model(cells[0].g,cfg,cells[0].structure,cfg.n_time_steps)
        expected=controlled_evaluate(ordinary,CORPUS_NAMES,joint)
        k,m=counts[0]
        actual=active_result(joint,(k,m,1))
        for name in CORPUS_NAMES:
            np.testing.assert_allclose(actual[name],expected[name],rtol=1e-10,atol=1e-10,err_msg=name)
        oracle=dict(status='pass',fields=len(CORPUS_NAMES),seconds=time.monotonic()-t,memory=memory())
        event('oracle_pass',**oracle)
    save('result.json',dict(**primary,oracle=oracle,total_elapsed_s=time.monotonic()-START,
        final_memory_events=(GROUP/'memory.events').read_text()))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        event('failure',error=repr(e),traceback=traceback.format_exc(),memory=memory())
        raise
