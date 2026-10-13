"""Bounded full-corpus comparison for independently composable worlds.

Run only through study_200k_envelope.sh. No top-up of rejected structures:
accepted-world rate is measured explicitly rather than disguised as draw rate.
"""
from __future__ import annotations
import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
from collections import Counter

import numpy as np
from comparison_methods import clone_function, controlled_evaluate, digest
from comparison_preflight import plain
from study_200k_methods import StudyAdapter, crop, flag_probabilities
from study_200k_packing import generate_fixed
from template_vs_per_world import CellWorkload, CORPUS_NAMES
from pymc_generator import world_model as wm, world_model_template as wt
from pymc_generator.active_counts import draw_active_counts
from pymc_generator.data_generator import DataGenerator, save_corpus, load_corpus
from pymc_generator.presets import make_scm_prior
from pymc_generator.sampler import sample_g_additive, _slice_g_active, SATURATION_FAMILY_KEYS, CARRYOVER_FAMILY_KEYS
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from pymc_generator.trajectories import structural_key

OUT = Path(os.environ['COMPARISON_OUTPUT'])
START = time.monotonic()
GROUP = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':', 2)[-1].lstrip('/')


def memory():
    return dict(peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope_peak_bytes=int((GROUP/'memory.peak').read_text()),
                scope_current_bytes=int((GROUP/'memory.current').read_text()),
                memory_events=(GROUP/'memory.events').read_text())


def save(name, value):
    (OUT/name).write_text(json.dumps(plain(value), indent=2)+'\n')


def event(kind, **kw):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(plain(dict(event=kind, elapsed_s=time.monotonic()-START, **kw)))+'\n')
        f.flush(); os.fsync(f.fileno())


def arrays(value, prefix=''):
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from arrays(v, f'{prefix}/{k}')


def config(n, seed):
    # Keep the original composable prior intact. Only dimensions, structure
    # budgets and the explicitly requested world-shared uniform family policy
    # differ. Default l_max=8 and its 8-week burn-in are retained.
    cfg = make_scm_prior(trajectories='composable', n_treatments=15, n_covariates=10,
        n_latent=1, n_time_steps=104, n_cells=n, draws_per_cell=1, seed=seed,
        n_treatments_active_range=(5,15), n_covariates_active_range=(3,10),
        edge_budget={'cy':(1,10), 'zy':(1,10), 'dc':0, 'dz':0, 'dy':0,
                     'zc':(1,4), 'cc':(1,4), 'zz':(1,4)},
        saturation_family_probs={k:1/len(SATURATION_FAMILY_KEYS) for k in SATURATION_FAMILY_KEYS},
        carryover_family_probs={k:1/len(CARRYOVER_FAMILY_KEYS) for k in CARRYOVER_FAMILY_KEYS})
    expected = {'treatment':(.9,.7,.15,.1,.2,.2,.3,.3), 'covariate':(.9,.6,.1,.1,.15,.3,.35,.3)}
    for role, values in expected.items():
        assert tuple(cfg.trajectory_inclusion_probs()[role][c] for c in TRAJECTORY_COMPONENTS) == values
    cfg.validate()
    return cfg


def make_cell(cfg, ident, internal):
    count_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, ident, 100]))
    graph_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, ident, 200]))
    counts = draw_active_counts(cfg, count_rng, None, ident)
    g = sample_g_additive(graph_rng, cfg, cfg.layout, *counts)
    active = _slice_g_active(g, *counts)
    sampler = wm.sample_structure
    if internal:
        # Placeholders only: do not waste a second flag sample outside the graph.
        def no_flags(cfg, n_treatments, n_covariates, rng):
            return {structural_key(role,c):np.zeros(n, dtype=bool)
                    for role,n in zip(TRAJECTORY_INPUTS,(n_treatments,n_covariates))
                    for c in TRAJECTORY_COMPONENTS}
        sampler = clone_function(wm.sample_structure, sample_component_flags=no_flags)
    structural = sampler(active, cfg, graph_rng)
    family_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, ident, 400]))
    for name, keys in [('sat_family',SATURATION_FAMILY_KEYS),('carryover_family',CARRYOVER_FAMILY_KEYS)]:
        structural[name][:] = family_rng.integers(0, len(keys))
    seed = int(np.random.SeedSequence([cfg.seed, ident, 300]).generate_state(1)[0])
    return CellWorkload(active, structural, wt.build_cell_inputs(cfg, g, g, structural), seed)


def pair(cell):
    return tuple(int(cell.structure[n][0]) for n in ('sat_family','carryover_family'))


def primary(args, cfg, cells, ids, adapter):
    # Family sorting is scheduling only: IDs were sampled first, without quotas
    # or changes to their independent uniform family assignments.
    groups = {}
    for ident, cell in zip(ids, cells): groups.setdefault(pair(cell), []).append((ident,cell))
    accepted = []; chunks = []; all_pairs = {}; checks = 0.0; writing = 0.0; reading = 0.0
    total_generation = 0.0
    for family, members in sorted(groups.items()):
        for offset in range(0, len(members), args.chunk):
            batch = members[offset:offset+args.chunk]
            batch_ids, batch_cells = map(list, zip(*batch))
            adapter.set_chunk(batch_cells, batch_ids)
            chunk_cfg = dataclasses.replace(cfg, n_cells=len(batch))
            before = time.monotonic()
            corpus = generate_fixed(chunk_cfg, adapter)
            generation = time.monotonic()-before
            total_generation += generation
            chosen = [batch_ids[i] for i,b in adapter.accepted_ids]
            accepted.extend(chosen)
            all_pairs.update({i:family for i in batch_ids})
            detail = dict(index=len(chunks), family=family, candidates=batch_ids, accepted=chosen,
                          generation_s=generation, rejected=adapter.rejected, memory=memory())
            if corpus is not None:
                start = time.monotonic(); errors = DataGenerator.validate_corpus(corpus)
                checks += time.monotonic()-start
                if errors: save('validation-errors.json',errors)
                assert not errors, errors
                for row, ident in enumerate(chosen):
                    flags = adapter.realised_flags[ident]
                    for role in TRAJECTORY_INPUTS:
                        n = flags[role].shape[0]
                        np.testing.assert_array_equal(corpus[f'{role}_components'][row,:n],flags[role])
                        assert not corpus[f'{role}_components'][row,n:].any()
                path = OUT/f'corpus-{len(chunks):03d}.npz'
                start=time.monotonic(); save_corpus(corpus,path); writing+=time.monotonic()-start
                start=time.monotonic(); back=load_corpus(path)
                left,right=dict(arrays(corpus)),dict(arrays(back))
                assert left.keys()==right.keys()
                for name,a in left.items():
                    b=right[name]
                    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name
                assert not DataGenerator.validate_corpus(back)
                reading+=time.monotonic()-start
                detail.update(bytes=path.stat().st_size, arrays=len(left), file=path.name,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                del corpus,back,left,right
            # Primary scope is captured before replay; replay has the same model
            # and payload (no separate oracle model to contaminate peak RSS).
            if not adapter.ordinary:
                start=time.monotonic()
                for index in dict.fromkeys((0,len(batch_cells)-1)):
                    adapter.current_cell=index
                    adapter.draw(adapter.last_names,1,diagnostic=True)
                detail['replay_seconds']=time.monotonic()-start
            chunks.append(detail); event('chunk',**detail)
            save('progress.json',dict(candidates=sum(len(c['candidates']) for c in chunks), accepted=len(accepted), chunks=chunks,
                                     stage_times=dict(adapter.times), memory=memory()))
            gc.collect()
        event('family_complete',family=family,memory=memory())
    totals={role:np.zeros(len(TRAJECTORY_COMPONENTS),dtype=int) for role in TRAJECTORY_INPUTS}
    totals_accepted={role:v.copy() for role,v in totals.items()}
    ninputs=Counter(); accepted_inputs=Counter()
    accepted_set=set(accepted)
    for ident, flags in adapter.realised_flags.items():
        for role,a in flags.items():
            totals[role]+=a.sum(axis=0,dtype=int); ninputs[role]+=len(a)
            if ident in accepted_set:
                totals_accepted[role]+=a.sum(axis=0,dtype=int); accepted_inputs[role]+=len(a)
    service = total_generation+checks+writing
    candidate_draws=[e['seconds'] for e in adapter.executions if not e['diagnostic']]
    report=dict(status='pass',approach=args.approach,mode=args.mode,phase=args.phase,
        candidates=len(ids),accepted=len(accepted),accepted_ids=accepted,
        candidate_family_counts={str(k):len(v) for k,v in groups.items()},
        accepted_family_counts=dict(Counter(str(all_pairs[i]) for i in accepted)),
        input_counts=dict(ninputs),component_counts=totals,
        accepted_input_counts=dict(accepted_inputs),accepted_component_counts=totals_accepted,
        generation_s=total_generation,validate_before_write_s=checks,write_s=writing,readback_and_check_s=reading,
        accepted_persisted_service_s=service,
        accepted_per_service_s=len(accepted)/service,
        steady_service_s=service-adapter.times['build_s']-adapter.times['compile_s'],
        stage_times=dict(adapter.times),draw_seconds=candidate_draws,
        compile_records=adapter.compile_records,build_records=adapter.build_records,
        chunks=chunks,primary_memory=memory(),elapsed_s=time.monotonic()-START,
        raw_candidate_hashes=adapter.all_primary_hashes,
        artifact_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        projection_scope='local candidate/accepted/persisted shards; no upload, Modal scheduling or exact accepted-family quotas')
    save('result.json',report)
    np.savez_compressed(OUT/'realised-flags.npz',**{f'{ident}_{role}':a for ident,flags in adapter.realised_flags.items() for role,a in flags.items()})


def oracle(args, cfg, cells, ids, adapter):
    tested=[]
    by_family={}
    for ident,cell in zip(ids,cells): by_family.setdefault(pair(cell),(ident,cell))
    for family,(ident,cell) in sorted(by_family.items()):
        adapter.set_chunk([cell],[ident]); adapter.current_cell=0
        adapter.build(cell.g,cfg,cell.structure,cfg.n_time_steps)
        model=adapter.models[0][0]
        requested=tuple(dict.fromkeys(CORPUS_NAMES+tuple(rv.name for rv in model.free_RVs)))
        fn=adapter.compile(requested)
        joint=fn(cell.template_inputs,seed=cell.seed,draws=1)
        if adapter.internal:
            for role,n in zip(TRAJECTORY_INPUTS,adapter.counts(0)[:2]):
                for c in TRAJECTORY_COMPONENTS:
                    name=structural_key(role,c)
                    cell.structure[name]=np.asarray(joint[name][0,:n],dtype=bool)
        # Actual selected flags and common primitive draws drive the independent
        # active-size scientific builder. This tests values, not stream equality
        # between two differently traversed compiled graphs.
        ordinary,_,_=wm.build_world_model(cell.g,cfg,cell.structure,cfg.n_time_steps)
        expected=controlled_evaluate(ordinary,CORPUS_NAMES,joint)
        actual=crop(joint,adapter.counts(0))
        errors={}
        for name in CORPUS_NAMES:
            np.testing.assert_allclose(actual[name],expected[name],rtol=1e-10,atol=1e-10,err_msg=f'{family}/{name}')
            errors[name]=float(np.max(np.abs(actual[name]-expected[name]),initial=0))
        # State must progress when called without reseeding, then replay exactly.
        args_in=tuple(np.asarray(cell.template_inputs[n]) for n in fn.input_names)
        one=fn.fn(*args_in); two=fn.fn(*args_in)
        assert any(not np.array_equal(a,b) for a,b in zip(one,two))
        if adapter.internal:
            positions=[fn.out_names.index(n) for n in flag_probabilities(cfg)]
            assert any(not np.array_equal(one[j],two[j]) for j in positions)
        again=fn(cell.template_inputs,seed=cell.seed,draws=1)
        assert digest(joint)==digest(again)
        record=dict(family=family, candidate_id=ident, fields=len(CORPUS_NAMES),
                    max_abs_errors=errors,rng_progression=True,reseed_replay=True,memory=memory())
        tested.append(record); event('oracle_pass',**record); save('oracle-progress.json',tested)
        del ordinary,expected,actual,one,two,again,joint,fn,model
        gc.collect()
    save('result.json',dict(status='pass',phase='oracle',approach=args.approach,mode=args.mode,
                           families=tested,elapsed_s=time.monotonic()-START,memory=memory()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('approach')
    parser.add_argument('--mode',choices=['fast_compile','fast_run'],default='fast_compile')
    parser.add_argument('--phase',choices=['pilot','primary','oracle'],default='primary')
    parser.add_argument('--candidates',type=int,default=1024)
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--seed',type=int,default=631227)
    parser.add_argument('--pair',default=None)
    args=parser.parse_args()
    assert args.approach in ['ordinary','specialized_external_eager','specialized_external_lazy',
        'specialized_internal_eager','specialized_internal_lazy','universal_external_eager','universal_internal_eager','universal_internal_lazy']
    cfg=config(args.candidates,args.seed)
    controls={k:(GROUP/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}
    assert controls=={'memory.max':str(2*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
    assert os.sched_getaffinity(0)=={0}
    save('environment.json',dict(args=vars(args),config=dataclasses.asdict(cfg),controls=controls,
        versions={k:importlib.metadata.version(k) for k in ['pymc','pytensor','numpy']},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        pytensor_flags=os.environ['PYTENSOR_FLAGS'],affinity=sorted(os.sched_getaffinity(0)),
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']}))
    before=time.monotonic();cells=[];ids=[];ident=0
    required=tuple(int(x) for x in args.pair.split(',')) if args.pair else None
    while len(cells)<args.candidates:
        cell=make_cell(cfg,ident,'internal' in args.approach)
        if required is None or pair(cell)==required:
            cells.append(cell);ids.append(ident)
        ident+=1
    preparation=time.monotonic()-before
    # Check exterior inputs coincide even when flag sampling moves inside.
    reference=make_cell(cfg,ids[0],not ('internal' in args.approach))
    for k,v in cells[0].template_inputs.items():
        if not k.startswith('use_'): np.testing.assert_array_equal(v,reference.template_inputs[k])
    save('instructions.json',dict(preparation_s=preparation,ids=ids,seeds=[c.seed for c in cells],
        counts=[(len(c.g['g_cy']),len(c.g['g_zy'])) for c in cells],families=[pair(c) for c in cells],
        nonflag_hashes=[digest({k:v for k,v in c.template_inputs.items() if not k.startswith('use_')}) for c in cells]))
    np.savez_compressed(OUT/'inputs.npz',**{key:np.stack([c.template_inputs[key] for c in cells]) for key in cells[0].template_inputs})
    adapter=StudyAdapter(args.approach,cfg,event,args.mode)
    event('start',preparation_s=preparation,memory=memory())
    if args.phase=='oracle': oracle(args,cfg,cells,ids,adapter)
    else: primary(args,cfg,cells,ids,adapter)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        event('failure',error=repr(exc),traceback=traceback.format_exc(),memory=memory())
        raise
