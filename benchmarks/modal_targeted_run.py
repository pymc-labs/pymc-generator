"""Bounded follow-up: one compatible live model, immutable prior inputs.

Run only via modal_targeted_envelope.sh. No production modifications.
"""
from __future__ import annotations
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from comparison_methods import Adapter
from comparison_run import arrays, decode, memory
from comparison_preflight import plain
from fixed_allocation import generate_fixed
from template_vs_per_world import CellWorkload, dea_prior, CORPUS_NAMES
from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus
import pymc_generator
import pytensor

OUT = Path(os.environ['COMPARISON_OUTPUT'])
START = time.monotonic()

def event(kind, **data):
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(plain(dict(event=kind, elapsed_s=time.monotonic()-START, **data)))+'\n')
        f.flush()
        os.fsync(f.fileno())

def save(name, data):
    (OUT / name).write_text(json.dumps(plain(data), indent=2)+'\n')

class SingleModel(Adapter):
    def build(self, g, cfg, structural, weeks, *, prior_cond=None):
        key = self.current_cell if self.method == 'ordinary' else 0
        if self.models and key not in self.models:
            self.models.clear()
            self.draws.clear()
            gc.collect()
            event('discard_incompatible_model', cell=self.current_cell)
        return super().build(g, cfg, structural, weeks, prior_cond=prior_cond)

    def replay_cells(self, count, draws):
        for cell in range(count):
            self.current_cell = cell
            self.build(self.cells[cell].g, self.cfg, self.cells[cell].structure,
                       self.cfg.n_time_steps)
            self.draw(CORPUS_NAMES, draws, diagnostic=True)


def main():
    method, workload = sys.argv[1], int(sys.argv[2])
    manifest_path = Path(__file__).parent / 'comparison-preflight-evidence/workloads.json'
    entry = next(e for e in json.loads(manifest_path.read_text()) if e['workload']==workload and e['repeat']==0)
    source = entry['config']
    cfg = dataclasses.replace(dea_prior(n_cells=5, draws_per_cell=1, seed=source['seed']),
        n_treatments=15,n_covariates=10,
        n_treatments_active_range=tuple(source['n_treatments_active_range']),
        n_covariates_active_range=tuple(source['n_covariates_active_range']),
        saturation_family_probs=source['saturation_family_probs'],
        carryover_family_probs=source['carryover_family_probs'])
    assert plain(dataclasses.asdict(cfg)) == source
    cfg.validate()
    cells = [CellWorkload(**decode(e)) for e in entry['cells'][:2]]
    adapter = SingleModel(method, cfg, cells, event)
    save('environment.json',dict(method=method, workload=workload,
        input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        selected_cells=[0,1], config=source, seed_set=0,
        generator_file=pymc_generator.__file__,
        harness_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=sys.version, versions={n:importlib.metadata.version(n) for n in ['pymc','pytensor','numpy','scipy']},
        mode='FAST_COMPILE', floatX=pytensor.config.floatX,
        cxx=str(pytensor.config.cxx), cache=str(pytensor.config.base_compiledir),
        affinity=sorted(os.sched_getaffinity(0)),
        thread_settings={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        memory=memory()))
    # First-cell grouping screen, then a two-cell turnover and replay probe.
    for label, count, draws in [('first',1,1),('warm10',1,10),('warm100',1,100),('turnover',2,10)]:
        adapter.accepted_ids=[]
        before=dict(adapter.times)
        event('group_start',label=label,cells=count,draws=draws)
        t=time.monotonic()
        corpus=generate_fixed(dataclasses.replace(cfg,n_cells=count,draws_per_cell=draws),adapter)
        generation=time.monotonic()-t
        # normalization timing is per-call, other fields are cumulative.
        stages={k:(v if k=='normalization_and_diagnostics_s' else v-before.get(k,0)) for k,v in adapter.times.items()}
        row=dict(label=label, method=method,workload=workload,cells=count,draws=draws,
            candidates=adapter.evaluated,accepted=len(adapter.accepted_ids),rejected=adapter.rejected,
            logical_ids=adapter.accepted_ids,stages=stages,generation_s=generation)
        if corpus is not None:
            t=time.monotonic()
            assert not DataGenerator.validate_corpus(corpus)
            row['validation_s']=time.monotonic()-t
            path=OUT/f'{label}.npz'
            t=time.monotonic(); save_corpus(corpus,path); row['persistence_s']=time.monotonic()-t
            t=time.monotonic(); back=load_corpus(path)
            left,right=dict(arrays(corpus)),dict(arrays(back))
            assert left.keys()==right.keys()
            for name,a in left.items():
                b=right[name]
                assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(), name
            assert not DataGenerator.validate_corpus(back)
            row['readback_s']=time.monotonic()-t
            row['persisted_bytes']=path.stat().st_size
            del corpus,back,left,right
        row['pipeline_s']=generation+sum(row.get(k,0) for k in ['validation_s','persistence_s','readback_s'])
        row['memory']=memory()
        save(f'{label}-primary.json',row)
        t=time.monotonic(); adapter.replay_cells(count,draws)
        row['replay_s']=time.monotonic()-t
        row['replay_pass']=True
        row['memory']=memory()
        save(f'{label}.json',row)
        event('group_complete',**row)
    save('result.json',dict(status='pass',memory=memory(),elapsed_s=time.monotonic()-START,executions=adapter.executions))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        event('failure',error=repr(e),traceback=traceback.format_exc(),memory=memory())
        raise
