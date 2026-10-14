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
from comparison_methods import clone_function, digest
import pytensor
from pytensor.graph.replace import clone_replace
from pytensor.tensor.random.op import RandomVariable
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


def controlled_evaluate(model, names, primitives):
    # Same common-primitive oracle as comparison_methods, but cache the
    # deterministic template evaluator. Recompiling its ~10k-node graph for
    # every flag boundary/family obscures correctness and wastes the budget.
    cache=getattr(model,'_study_controlled_cache',None)
    if cache is None:
        cache={}; object.__setattr__(model,'_study_controlled_cache',cache)
    names=tuple(names)
    if names not in cache:
        before=time.monotonic()
        replacements={rv:rv.type(name=f'given_{rv.name}') for rv in model.free_RVs}
        expressions=clone_replace([model[n] for n in names],replace=replacements,rebuild_strict=False)
        fn=pytensor.function(list(replacements.values()),expressions,on_unused_input='ignore',mode='FAST_COMPILE')
        specs=[(rv.name,rv.dtype,tuple(int(n) for n in rv.shape.eval())) for rv in replacements]
        cache[names]=(fn,specs)
        event('deterministic_oracle_compile',seconds=time.monotonic()-before,nodes=len(fn.maker.fgraph.apply_nodes))
    fn,specs=cache[names]
    values=[]
    for name,dtype,shape in specs:
        raw=np.asarray(primitives[name][0],dtype=dtype)
        values.append(raw[tuple(slice(n) for n in shape)] if shape else raw)
    return {name:np.asarray(v)[None] for name,v in zip(names,fn(*values))}


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
    rejection_reasons=Counter()
    # Ordinary builds each cell independently; do not create singleton
    # family shards that cannot satisfy the corpus train/validation contract.
    workgroups={('mixed',):list(zip(ids,cells))} if adapter.ordinary else groups
    for family, members in sorted(workgroups.items()):
        batches=[]
        for cohort in sorted({i//args.cohort_size for i,c in members}):
            subset=[(i,c) for i,c in members if i//args.cohort_size==cohort]
            pieces=[subset[o:o+args.chunk] for o in range(0,len(subset),args.chunk)]
            if len(pieces)>1 and len(pieces[-1])==1:
                pieces[-2].extend(pieces.pop())
            batches.extend((cohort,piece) for piece in pieces)
        for cohort,batch in batches:
            batch_ids, batch_cells = map(list, zip(*batch))
            adapter.set_chunk(batch_cells, batch_ids)
            chunk_cfg = dataclasses.replace(cfg, n_cells=len(batch))
            before = time.monotonic(); before_stages=dict(adapter.times)
            corpus = generate_fixed(chunk_cfg, adapter)
            stage_delta={k:v-before_stages.get(k,0) for k,v in adapter.times.items()}
            generation = time.monotonic()-before
            total_generation += generation
            chosen = [batch_ids[i] for i,b in adapter.accepted_ids]
            accepted.extend(chosen)
            rejection_reasons.update(adapter.rejection_reasons)
            assert sum(adapter.rejection_reasons.values())==adapter.rejected
            all_pairs.update({i:pair(c) for i,c in batch})
            detail = dict(index=len(chunks), family=family, cohort=cohort, candidates=batch_ids, accepted=chosen,
                          stage_times=stage_delta,validate_s=0.0,write_s=0.0,readback_s=0.0,
                          generation_s=generation, rejected=adapter.rejected,
                          rejection_reasons=dict(adapter.rejection_reasons),rejection_records=adapter.rejection_records,
                          memory=memory())
            if corpus is not None:
                start = time.monotonic(); errors = DataGenerator.validate_corpus(corpus)
                detail['validate_s']=time.monotonic()-start; checks+=detail['validate_s']
                if errors: save('validation-errors.json',errors)
                assert not errors, errors
                for row, ident in enumerate(chosen):
                    flags = adapter.realised_flags[ident]
                    for role in TRAJECTORY_INPUTS:
                        n = flags[role].shape[0]
                        np.testing.assert_array_equal(corpus[f'{role}_components'][row,:n],flags[role])
                        assert not corpus[f'{role}_components'][row,n:].any()
                path = OUT/f'corpus-{len(chunks):03d}.npz'
                start=time.monotonic(); save_corpus(corpus,path)
                detail['write_s']=time.monotonic()-start; writing+=detail['write_s']
                start=time.monotonic(); back=load_corpus(path)
                left,right=dict(arrays(corpus)),dict(arrays(back))
                assert left.keys()==right.keys()
                for name,a in left.items():
                    b=right[name]
                    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name
                assert not DataGenerator.validate_corpus(back)
                detail['readback_s']=time.monotonic()-start; reading+=detail['readback_s']
                detail.update(bytes=path.stat().st_size, arrays=len(left), file=path.name,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                del corpus,back,left,right
            # Primary scope is captured before replay; replay has the same model
            # and payload (no separate oracle model to contaminate peak RSS).
            start=time.monotonic()
            indices=(len(batch_cells)-1,) if adapter.ordinary else dict.fromkeys((0,len(batch_cells)-1))
            for index in indices:
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
    cohort_reports=[]
    for cohort in sorted({c['cohort'] for c in chunks}):
        subset=[c for c in chunks if c['cohort']==cohort]
        count=sum(len(c['candidates']) for c in subset); kept=sum(len(c['accepted']) for c in subset)
        steady=sum(c['generation_s']-c['stage_times'].get('build_s',0)-c['stage_times'].get('compile_s',0)+c['validate_s']+c['write_s'] for c in subset)
        verified=steady+sum(c['readback_s'] for c in subset)
        cohort_reports.append(dict(cohort=cohort,candidates=count,accepted=kept,
            seed_contract=args.seed_contract,root_seed=cfg.seed,
            steady_accepted_persisted_s=steady,steady_accepted_per_s=kept/steady,
            steady_verified_s=verified,steady_verified_accepted_per_s=kept/verified,
            scope='independent fresh logical-ID cohort, shared resident model/compiler cache'))
    report=dict(status='pass',approach=args.approach,mode=args.mode,phase=args.phase,
        seed_contract=args.seed_contract,
        cohorts=cohort_reports,
        candidates=len(ids),accepted=len(accepted),accepted_ids=accepted,
        rejection_reasons=dict(rejection_reasons),
        candidate_family_counts={str(k):len(v) for k,v in groups.items()},
        accepted_family_counts=dict(Counter(str(all_pairs[i]) for i in accepted)),
        input_counts=dict(ninputs),component_counts=totals,
        accepted_input_counts=dict(accepted_inputs),accepted_component_counts=totals_accepted,
        generation_s=total_generation,validate_before_write_s=checks,write_s=writing,readback_and_check_s=reading,
        accepted_persisted_service_s=service,
        accepted_per_service_s=len(accepted)/service,
        verified_service_s=service+reading,accepted_per_verified_service_s=len(accepted)/(service+reading),
        steady_verified_s=service+reading-adapter.times['build_s']-adapter.times['compile_s'],
        planning_s=json.loads((OUT/'instructions.json').read_text())['preparation_s'],
        steady_service_s=service-adapter.times['build_s']-adapter.times['compile_s'],
        stage_times=dict(adapter.times),draw_seconds=candidate_draws,
        compile_records=adapter.compile_records,build_records=adapter.build_records,
        rng_work=adapter.rng_work_records,
        chunks=chunks,primary_memory=memory(),elapsed_s=time.monotonic()-START,
        raw_candidate_hashes=adapter.all_primary_hashes,
        artifact_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        projection_scope='local candidate/accepted/persisted shards; no upload, Modal scheduling or exact accepted-family quotas')
    save('result.json',report)
    np.savez_compressed(OUT/'realised-flags.npz',**{f'{ident}_{role}':a for ident,flags in adapter.realised_flags.items() for role,a in flags.items()})


def oracle(args, cfg, cells, ids, adapter):
    tested=[]
    archive=Path(args.primitives_from) if args.primitives_from else None
    if archive:
        source=json.loads((archive/'result.json').read_text())
        assert source['status']=='pass' and len(source['families'])==18
        source_cfg=json.loads((archive/'environment.json').read_text())['config']
        assert plain(dataclasses.asdict(cfg))==source_cfg
        source_inst=json.loads((archive/'instructions.json').read_text())
        hashes=dict(zip(source_inst['ids'],source_inst['nonflag_hashes']))
    by_family={}
    for ident,cell in zip(ids,cells): by_family.setdefault(pair(cell),(ident,cell))
    for family,(ident,cell) in sorted(by_family.items()):
        adapter.set_chunk([cell],[ident]); adapter.current_cell=0
        adapter.build(cell.g,cfg,cell.structure,cfg.n_time_steps)
        model=adapter.models[0][0]
        fn=None
        if archive:
            assert hashes[ident]==digest({k:v for k,v in cell.template_inputs.items() if not k.startswith('use_')})
            path=archive/f'oracle-primitives-{family[0]}-{family[1]}.npz'
            with np.load(path) as data: joint={k:data[k].copy() for k in data.files}
            input_names=wt.TEMPLATE_STRUCTURE_INPUT_NAMES + (() if adapter.internal else tuple(flag_probabilities(cfg)))
            event('oracle_archived_primitives',family=family,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            requested=tuple(dict.fromkeys(CORPUS_NAMES+tuple(rv.name for rv in model.free_RVs)))
            fn=adapter.compile(requested)
            input_names=fn.input_names
            joint=adapter.sample(fn,cell)
            np.savez_compressed(OUT/f'oracle-primitives-{family[0]}-{family[1]}.npz',**joint)
            event('oracle_primitive_drawn',family=family,memory=memory())
        if adapter.internal or archive:
            for role,n in zip(TRAJECTORY_INPUTS,adapter.counts(0)[:2]):
                for c in TRAJECTORY_COMPONENTS:
                    name=structural_key(role,c)
                    cell.structure[name]=np.asarray(joint[name][0,:n],dtype=bool)
                    cell.template_inputs[name][:]=0
                    cell.template_inputs[name][:n]=cell.structure[name]
        with model: __import__('pymc').set_data({n:cell.template_inputs[n] for n in input_names})
        # Actual selected flags and common primitive draws drive the independent
        # active-size scientific builder. This tests values, not stream equality
        # between two differently traversed compiled graphs.
        ordinary,_,_=wm.build_world_model(cell.g,cfg,cell.structure,cfg.n_time_steps)
        expected=controlled_evaluate(ordinary,CORPUS_NAMES,joint)
        actual=crop(controlled_evaluate(model,CORPUS_NAMES,joint) if archive else joint,adapter.counts(0))
        errors={}
        for name in CORPUS_NAMES:
            np.testing.assert_allclose(actual[name],expected[name],rtol=1e-10,atol=1e-10,err_msg=f'{family}/{name}')
            errors[name]=float(np.max(np.abs(actual[name]-expected[name]),initial=0))
        event('oracle_sampled_flags_pass',family=family)
        # State must progress when called without reseeding, then replay exactly.
        changed=[]; one=two=again=None
        if fn is not None:
            args_in=tuple(np.asarray(cell.template_inputs[n]) for n in fn.input_names)
            def rng_states():
                return [json.dumps(plain(rng.get_value().bit_generator.state), sort_keys=True) for rng in fn.ordered_rngs]
            state0=rng_states(); one=fn.fn(*args_in); state1=rng_states()
            two=fn.fn(*args_in); state2=rng_states()
            changed=[a != b and b != c for a,b,c in zip(state0,state1,state2)]
            assert all(changed), 'a demanded RNG update did not advance'
            assert any(not np.array_equal(a,b) for a,b in zip(one,two))
            if adapter.internal:
                positions=[fn.out_names.index(n) for n in flag_probabilities(cfg)]
                assert any(not np.array_equal(one[j],two[j]) for j in positions)
            again=adapter.sample(fn,cell)
            assert digest(joint)==digest(again)
        # Force boundary/mixed flags as common primitive inputs; this is a
        # deterministic scientific oracle, NOT a change to the sampling prior.
        # Setting pm.Data is restricted to this diagnostic process.
        cases=[]
        shared_inputs={n:np.asarray(cell.template_inputs[n]) for n in input_names}
        with model: __import__('pymc').set_data(shared_inputs)
        for case in ('off','on','mixed'):
            common={k:v.copy() for k,v in joint.items()}
            for role,n in zip(TRAJECTORY_INPUTS,adapter.counts(0)[:2]):
                for ci,c in enumerate(TRAJECTORY_COMPONENTS):
                    key=structural_key(role,c)
                    width=cfg.n_treatments if role=='treatment' else cfg.n_covariates
                    flags=np.zeros(width,dtype='int64')
                    flags[:n]=0 if case=='off' else (1 if case=='on' else (np.arange(n)+ci)%2)
                    cell.structure[key]=flags[:n].astype(bool)
                    if adapter.internal: common[key]=flags[None]
                    else:
                        with model: __import__('pymc').set_data({key:flags})
            ordinary_case,_,_=wm.build_world_model(cell.g,cfg,cell.structure,cfg.n_time_steps)
            expected_case=controlled_evaluate(ordinary_case,CORPUS_NAMES,common)
            actual_case=crop(controlled_evaluate(model,CORPUS_NAMES,common),adapter.counts(0))
            for name in CORPUS_NAMES:
                np.testing.assert_allclose(actual_case[name],expected_case[name],rtol=1e-10,atol=1e-10,err_msg=f'{family}/{case}/{name}')
            cases.append(case); event('oracle_flag_case_pass',family=family,case=case)
            del ordinary_case,expected_case,actual_case,common
            gc.collect()
        # Poison only dormant adjacency slots; preserve upper-triangular DAGs.
        # Identical outputs prove masks act before reference/noise scaling, not
        # only on final arrays. This is diagnostic input, not candidate prior.
        clean=controlled_evaluate(model,CORPUS_NAMES,joint)
        dirty={k:np.asarray(cell.template_inputs[k]).copy() for k in
            ('g_cy','g_zy','g_dy','g_dc','g_dz','g_zc','g_cc','g_zz')}
        c,z,d=(np.asarray(cell.template_inputs[f'active_{r}']) for r in ('treatment','covariate','latent'))
        masks={'g_cy':c,'g_zy':z,'g_dy':d,'g_dc':d[:,None]*c[None,:],
            'g_dz':d[:,None]*z[None,:],'g_zc':z[:,None]*c[None,:],
            'g_cc':c[:,None]*c[None,:],'g_zz':z[:,None]*z[None,:]}
        for key,a in dirty.items():
            dormant=masks[key]==0
            if key in ('g_cc','g_zz'): dormant &= np.triu(np.ones_like(a,dtype=bool),1)
            a[dormant]=1
        with model: __import__('pymc').set_data(dirty)
        poisoned=controlled_evaluate(model,CORPUS_NAMES,joint)
        for name in CORPUS_NAMES:
            np.testing.assert_allclose(poisoned[name],clean[name],rtol=1e-10,atol=1e-10,err_msg=f'{family}/dormant-edge-poison/{name}')
        with model: __import__('pymc').set_data({k:cell.template_inputs[k] for k in dirty})
        del clean,poisoned,dirty
        record=dict(family=family, candidate_id=ident, fields=len(CORPUS_NAMES),
                    max_abs_errors=errors,flag_cases=cases,dormant_edge_poison=True,rng_progression=bool(fn),
                    primitive_source=str(archive) if archive else 'fresh joint draw',
                    rng_updates_advanced=sum(changed),rng_updates_total=len(changed),
                    reseed_replay=bool(fn),memory=memory())
        tested.append(record); event('oracle_pass',**record); save('oracle-progress.json',tested)
        del ordinary,expected,actual,one,two,again,joint,fn,model
        gc.collect()
    save('result.json',dict(status='pass',phase='oracle',approach=args.approach,mode=args.mode,
                           families=tested,elapsed_s=time.monotonic()-START,memory=memory()))


def rng_probe(args,cfg):
    ident=7
    records=[json.loads(s) for s in Path('benchmarks/study-200k-evidence/o-logical-reference/events.jsonl').read_text().splitlines()]
    names=next(e['output_names'] for e in records if e['event']=='compile' and e['key']==ident)
    cell=make_cell(cfg,ident,False)
    adapter=StudyAdapter('ordinary',cfg,event,args.mode,args.seed_contract)
    adapter.set_chunk([cell],[ident]); adapter.build(cell.g,cfg,cell.structure,cfg.n_time_steps)
    model=adapter.models[0][0]; fn=adapter.compile(tuple(names)); adapter.sample(fn,cell)
    states=lambda:[json.dumps(plain(r.get_value().bit_generator.state),sort_keys=True) for r in fn.ordered_rngs]
    before=states(); one=fn.fn(); middle=states(); two=fn.fn(); after=states()
    changed=[a!=b and b!=c for a,b,c in zip(before,middle,after)]
    details=[]
    for i,rng in enumerate(fn.ordered_rngs):
        if changed[i]: continue
        original=[rv for rv in model.free_RVs if rv.owner and rv.owner.inputs[0] is rng]
        compiled=[n for n in fn.fn.maker.fgraph.toposort() if isinstance(n.op,RandomVariable) and n.inputs[0] is rng]
        details.append(dict(index=i,original=[dict(name=rv.name,shape=rv.type.shape,
            size=getattr(rv.owner.inputs[1],'data',None)) for rv in original],
            compiled_nodes=[dict(op=type(n.op).__name__,name=n.outputs[-1].name,size=getattr(n.inputs[1],'data',None)) for n in compiled]))
    save('result.json',dict(status='observed',phase='rng_probe',candidate_id=ident,
        rngs=len(changed),advanced=sum(changed),unchanged=details,
        any_outputs_changed=any(not np.array_equal(a,b) for a,b in zip(one,two)),
        elapsed_s=time.monotonic()-START,memory=memory()))


def seed_audit(args):
    # Audit IDs only: no structures, models, candidate worlds or corpus draws.
    from pymc.util import _get_seeds_per_chain
    n=200000
    worlds={}; compiled={}; logical={}; examples=[]
    def add(table,key,ident):
        if key not in table: table[key]=[ident,1]
        else:
            if len(examples)<12: examples.append(dict(stage='world_seed' if table is worlds else ('compile_seed' if table is compiled else 'logical_state'),ids=[table[key][0],ident]))
            table[key][1]+=1
    def state(ident):
        s=np.random.PCG64(np.random.SeedSequence([args.seed,ident,300]).spawn(1)[0]).state['state']
        return s['state'],s['inc']
    for ident in range(n):
        seed=int(np.random.SeedSequence([args.seed,ident,300]).generate_state(1)[0])
        compile_seed=_get_seeds_per_chain(np.random.default_rng(seed),1)[0]
        add(worlds,seed,ident); add(compiled,compile_seed,ident); add(logical,state(ident),ident)
    for ident in (n-1,1000,500,1,0):
        assert logical[state(ident)][0]==ident
        root=[args.seed,ident,300]
        from pymc.pytensorf import reseed_rngs
        rng=pytensor.shared(np.random.default_rng(1))
        reseed_rngs([rng],root)
        actual=rng.get_value().bit_generator.state['state']
        assert (actual['state'],actual['inc'])==state(ident)
    summary=lambda table:dict(unique=len(table),duplicate_ids=n-len(table),colliding_pairs=sum(c*(c-1)//2 for i,c in table.values()))
    save('result.json',dict(status='pass',phase='seed_audit',logical_ids=n,generated_worlds=0,root_seed=args.seed,
        legacy_uint32_world_seeds=summary(worlds),legacy_downstream_30bit_compile_seeds=summary(compiled),
        logical_id_first_rng_states=summary(logical),collision_examples=examples,
        root_policy='reseed_rngs(ordered_rngs,[root_seed,logical_candidate_id,300]); no worker/shard index',
        deterministic_rebuild_and_reverse_mapping=True,
        caveat='Seed/shared-stream collisions are not claims of duplicate complete worlds; structures and families can differ.',
        elapsed_s=time.monotonic()-START,memory=memory()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('approach')
    parser.add_argument('--mode',choices=['fast_compile','cvm_compile','fast_run'],default='fast_compile')
    parser.add_argument('--seed-contract',choices=['legacy','logical_id'],default='legacy')
    parser.add_argument('--phase',choices=['pilot','primary','oracle','seed_audit','rng_probe'],default='primary')
    parser.add_argument('--candidates',type=int,default=1024)
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--cohort-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=631227)
    parser.add_argument('--pair',default=None)
    parser.add_argument('--all-pairs',action='store_true',help='diagnostic one naturally drawn ID per family pair, oracle only')
    parser.add_argument('--rebuild-check',action='store_true',help='separate post-primary rebuild/reordered replay; memory/timing excluded from primary report')
    parser.add_argument('--primitives-from',default=None,help='oracle only: use a passed universal oracle archive; RNG progression checked separately in primary')
    args=parser.parse_args()
    assert args.candidates>0 and args.chunk>0 and args.cohort_size>0
    assert args.primitives_from is None or args.phase=='oracle'
    if args.all_pairs:
        assert args.phase=='oracle' and args.pair is None
        args.candidates=len(SATURATION_FAMILY_KEYS)*len(CARRYOVER_FAMILY_KEYS)
    assert args.approach in ['ordinary','specialized_external_eager','specialized_external_lazy',
        'specialized_internal_eager','specialized_internal_lazy','universal_external_eager','universal_internal_eager','universal_internal_lazy']
    cfg=config(args.candidates,args.seed)
    controls={k:(GROUP/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}
    assert controls=={'memory.max':str(2*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
    assert os.sched_getaffinity(0)=={0}
    from threadpoolctl import threadpool_info
    pools=threadpool_info()
    assert all(p['num_threads']==1 for p in pools), pools
    assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'])
    save('environment.json',dict(args=vars(args),config=dataclasses.asdict(cfg),controls=controls,
        versions={k:importlib.metadata.version(k) for k in ['pymc','pytensor','numpy']},
        threadpools=pools,python_version=__import__('sys').version,kernel=os.uname().release,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['benchmarks/study_200k_run.py','benchmarks/study_200k_methods.py',
             'benchmarks/study_200k_packing.py','benchmarks/study_200k_envelope.sh',
             'benchmarks/study_200k_supervise.py']},
        pytensor_flags=os.environ['PYTENSOR_FLAGS'],affinity=sorted(os.sched_getaffinity(0)),
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']}))
    if args.phase=='seed_audit':
        seed_audit(args); return
    if args.phase=='rng_probe':
        rng_probe(args,cfg); return
    before=time.monotonic();cells=[];ids=[];ident=0
    required=tuple(int(x) for x in args.pair.split(',')) if args.pair else None
    seen_pairs=set()
    while len(cells)<args.candidates:
        cell=make_cell(cfg,ident,'internal' in args.approach)
        if (required is None or pair(cell)==required) and (not args.all_pairs or pair(cell) not in seen_pairs):
            cells.append(cell);ids.append(ident);seen_pairs.add(pair(cell))
        ident+=1
    preparation=time.monotonic()-before
    # Check exterior inputs coincide even when flag sampling moves inside.
    for ident,cell in zip(ids,cells):
        reference=make_cell(cfg,ident,not ('internal' in args.approach))
        for k,v in cell.template_inputs.items():
            if not k.startswith('use_'): np.testing.assert_array_equal(v,reference.template_inputs[k])
    save('instructions.json',dict(preparation_s=preparation,ids=ids,seeds=[c.seed for c in cells],
        counts=[(len(c.g['g_cy']),len(c.g['g_zy'])) for c in cells],families=[pair(c) for c in cells],
        saturation_families=SATURATION_FAMILY_KEYS,carryover_families=CARRYOVER_FAMILY_KEYS,
        l_max=cfg.l_max,burn_in=cfg.carryover_burn_in,seed_contract=args.seed_contract,
        draw_seed_entropy=[[cfg.seed,i,300] for i in ids] if args.seed_contract=='logical_id' else None,
        nonflag_hashes=[digest({k:v for k,v in c.template_inputs.items() if not k.startswith('use_')}) for c in cells]))
    np.savez_compressed(OUT/'inputs.npz',**{key:np.stack([c.template_inputs[key] for c in cells]) for key in cells[0].template_inputs})
    adapter=StudyAdapter(args.approach,cfg,event,args.mode,args.seed_contract)
    event('start',preparation_s=preparation,memory=memory())
    if args.phase=='oracle': oracle(args,cfg,cells,ids,adapter)
    else:
        primary(args,cfg,cells,ids,adapter)
        if args.rebuild_check:
            assert not adapter.ordinary
            before=dict(adapter.times); names=adapter.last_names
            # Primary report was saved before any diagnostic rebuild allocation.
            adapter.set_chunk(cells,ids); adapter.current_key=None
            indices=list(dict.fromkeys((len(cells)-1,len(cells)//2,0)))
            if adapter.specialized:
                family=pair(cells[indices[0]])
                indices=[i for i,c in reversed(list(enumerate(cells))) if pair(c)==family][:3]
            for i in indices:
                adapter.current_cell=i
                adapter.build(cells[i].g,cfg,cells[i].structure,cfg.n_time_steps)
                adapter.draw(names,1,diagnostic=True)
            save('rebuild.json',dict(status='pass',ids=[ids[i] for i in indices],
                reversed_scheduling=True,new_model=True,exact_raw_bytes=True,
                stage_times={k:v-before.get(k,0) for k,v in adapter.times.items()},memory=memory()))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        event('failure',error=repr(exc),traceback=traceback.format_exc(),memory=memory())
        raise
