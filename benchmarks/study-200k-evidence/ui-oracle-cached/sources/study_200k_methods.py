"""Isolated 200k-world strategy prototypes; production modules stay unchanged.

The scientific constructors and packing are delegated to existing code. These
prototypes change execution, not Carlos's composable prior. Component flags are
optionally RVs; response-family instructions remain external, shared per world.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
import gc
import time
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.ifelse import ifelse
from pytensor.link.vm import VMLinker

from comparison_methods import Adapter, active_result, clone_function, digest, widths
from pymc_generator import symbolic_graph as sg, trajectories as tr
from pymc_generator import world_model as wm, world_model_template as wt
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS, TRAJECTORY_PARAM_REPORTS


def flag_probabilities(cfg):
    probs = cfg.trajectory_inclusion_probs()
    return {tr.structural_key(role, c): probs[role][c]
            for role in TRAJECTORY_INPUTS for c in TRAJECTORY_COMPONENTS}


def family_config(cfg, pair):
    s, a = pair
    return dataclasses.replace(cfg,
        saturation_family_probs={k: float(i == s) for i, k in enumerate(SATURATION_FAMILY_KEYS)},
        carryover_family_probs={k: float(i == a) for i, k in enumerate(CARRYOVER_FAMILY_KEYS)})


def build_template(cfg, inputs, weeks, *, specialized, internal, lazy):
    wt.check_template_supported(cfg)
    assert not cfg.mechanism_priors_enabled and not cfg.record_walk_parameters
    bindings = {}
    if specialized:
        sat = [i for i, k in enumerate(SATURATION_FAMILY_KEYS) if cfg.saturation_family_probs[k] > 0]
        ad = [i for i, k in enumerate(CARRYOVER_FAMILY_KEYS) if cfg.carryover_family_probs[k] > 0]
        assert len(sat) == len(ad) == 1
        def parameters(*args, **kwargs):
            kwargs['sat_family'] = np.full(cfg.n_treatments, sat[0], dtype='int64')
            kwargs['carryover_family'] = np.full(cfg.n_treatments, ad[0], dtype='int64')
            return wt._scm_params(*args, **kwargs)
        def carry(col, params, k, *, dynamic_family=False):
            return sg._carryover_col(col, params, k, dynamic_family=False)
        def saturate(col, scale, params, k, *, dynamic_family=False):
            return sg._saturate_col(col, scale, params, k, dynamic_family=False)
        bindings.update(_carryover_col=carry, _saturate_col=saturate)
    else:
        parameters = wt._scm_params
    if lazy:
        full_time = weeks + cfg.carryover_burn_in
        def choose(flag, value, neutral):
            value = pt.as_tensor_variable(value)
            # Constant neutrals, not zeros_like(value): don't introduce a
            # shape dependency that accidentally demands the disabled branch.
            assert value.ndim in (0, 1)
            neutral_value = np.full((), neutral, dtype=value.dtype) if value.ndim == 0 else np.full(full_time, neutral, dtype=value.dtype)
            return ifelse(pt.neq(flag, 0), value, pt.as_tensor_variable(neutral_value))
        def selected(spec, component, i, value, neutral):
            flags = spec['use'][component]
            return choose(flags[i], value, neutral) if isinstance(flags, pt.TensorVariable) else value
        def flagged(flags, i, expr):
            return choose(flags[i], expr, 0.0) if isinstance(flags, pt.TensorVariable) else expr
        smooth = clone_function(tr._smooth_terms, _selected_component=selected)
        replacements = {'_selected_component': selected, '_smooth_terms': smooth}
        trajectory_proxy = SimpleNamespace(**{name: getattr(tr, name) for name in dir(tr)})
        for name in ('activity_column', 'level_shift_column', 'treatment_multiplier_column'):
            setattr(trajectory_proxy, name, clone_function(getattr(tr, name), **replacements))
        bindings.update(trajectories=trajectory_proxy, _flagged=flagged)
    scientific_graph = clone_function(sg.build_symbolic_graph, **bindings)
    # Do not rely solely on the planner zero-padding adjacency. Active masks
    # must remove dormant edges before parameters, reference scales and noise
    # scaling, as well as before trajectory propagation.
    def masked_edges(g, active):
        c,z,d = (active[f'active_{r}'] for r in ('treatment','covariate','latent'))
        return {**g, 'g_cy':g['g_cy']*c, 'g_zy':g['g_zy']*z, 'g_dy':g['g_dy']*d,
            'g_dc':g['g_dc']*d[:,None]*c[None,:],
            'g_dz':g['g_dz']*d[:,None]*z[None,:],
            'g_zc':g['g_zc']*z[:,None]*c[None,:],
            'g_cc':g['g_cc']*c[:,None]*c[None,:],
            'g_zz':g['g_zz']*z[:,None]*z[None,:]}
    def active_data():
        return {f'active_{r}':pm.modelcontext(None)[f'active_{r}'] for r in ('treatment','covariate','latent')}
    original_parameters=parameters
    def parameters(*args, **kwargs):
        kwargs['g']=masked_edges(kwargs['g'],active_data())
        return original_parameters(*args,**kwargs)
    def graph(g,*args,**kwargs):
        return scientific_graph(masked_edges(g,kwargs['active']),*args,**kwargs)
    def outcome_scale(cfg,rw,g_cy,beta):
        return wt._apply_outcome_std_scale(cfg,rw,g_cy*active_data()['active_treatment'],beta)
    pm_proxy = pm
    if internal:
        probs = flag_probabilities(cfg)
        def data(name, value, **kwargs):
            if name not in probs:
                return pm.Data(name, value, **kwargs)
            role = 'covariate' if name.startswith('use_covariate_') else 'treatment'
            raw = pm.Bernoulli(name, p=probs[name], shape=np.shape(value))
            return raw * pm.modelcontext(None)[f'active_{role}']
        pm_proxy = SimpleNamespace(**{name: getattr(pm, name) for name in dir(pm)})
        pm_proxy.Data = data
    builder = clone_function(wt.build_world_model_template, _scm_params=parameters,
                             build_symbolic_graph=graph, pm=pm_proxy,
                             _apply_outcome_std_scale=outcome_scale)
    return builder(cfg, inputs, weeks)


def crop(value, counts):
    result = active_result(value, counts)
    k, m, _ = counts
    extra = {
        'treatments_natural': k, 'treatments_unshocked': k,
        'treatment_activity': k, 'treatment_log_level_shift': k,
        'covariate_activity': m, 'covariate_level_shift': m,
    }
    for key, report in TRAJECTORY_PARAM_REPORTS.items():
        extra[report] = k if key.startswith('trajectory_treatment_') else m
    for name, width in extra.items():
        if name in result:
            result[name] = result[name][..., :width]
    return result


class StudyAdapter(Adapter):
    def __init__(self, approach, cfg, event, mode_name='fast_compile'):
        super().__init__(approach, cfg, [], event)
        self.internal = 'internal' in approach
        self.lazy = 'lazy' in approach
        self.specialized = approach.startswith('specialized')
        self.ordinary = approach == 'ordinary'
        self.mode_name = mode_name
        self.logical_ids = []
        self.current_key = None
        self.all_primary_hashes = {}
        self.realised_flags = {}
        self.last_names = ()
        self.compile_records = []
        self.build_records = []
        self.pending_flags = {}
        self.last_raw = None

    def set_chunk(self, cells, ids):
        self.cells, self.logical_ids = cells, ids
        self.accepted_ids = []
        self.retained_cell_ids = []
        self.rejection_reasons = Counter()
        self.rejection_records = []

    def reject(self, reason):
        self.rejection_reasons[reason] += 1
        self.rejection_records.append(dict(candidate_id=self.logical_ids[self.current_cell],reason=reason))

    def build(self, g, cfg, structural, weeks, *, prior_cond=None):
        assert prior_cond is None
        pair = int(structural['sat_family'][0]), int(structural['carryover_family'][0])
        assert np.unique(structural['sat_family']).size == np.unique(structural['carryover_family']).size == 1
        key = self.logical_ids[self.current_cell] if self.ordinary else (pair if self.specialized else 'universal')
        if key != self.current_key:
            # Worker lifecycle: one model, not an unbounded family/count cache.
            self.models.clear(); self.draws.clear(); self.last_raw = None
            gc.collect()
            self.current_key = key
        if not self.models:
            started = time.perf_counter()
            if self.ordinary:
                value = wm.build_world_model(g, cfg, structural, weeks)
            else:
                model_cfg = family_config(cfg, pair) if self.specialized else cfg
                value = build_template(model_cfg, self.cells[self.current_cell].template_inputs, weeks,
                    specialized=self.specialized, internal=self.internal, lazy=self.lazy)
            seconds = time.perf_counter() - started
            self.models[0] = value
            self.times['build_s'] += seconds
            self.build_records.append(dict(key=key, seconds=seconds))
            self.event('build', key=key, seconds=seconds)
        return self.models[0]

    def compile(self, requested):
        flag_names = tuple(flag_probabilities(self.cfg)) if self.internal else ()
        names = tuple(dict.fromkeys(tuple(requested) + flag_names))
        if names not in self.draws:
            start = time.perf_counter()
            if self.mode_name == 'fast_run':
                mode = 'FAST_RUN'
            else:
                # Same rewrite selection; CVM is an independently measured
                # execution option, not a scientific model change.
                mode = Mode(linker=VMLinker(use_cloop=self.mode_name=='cvm_compile', lazy=True), optimizer='fast_compile')
            inputs = () if self.ordinary else (wt.TEMPLATE_STRUCTURE_INPUT_NAMES if self.internal else None)
            fn = wt.compile_template_draw_fn(self.models[0][0], names, input_names=inputs, mode=mode)
            self.draws[names] = fn
            elapsed = time.perf_counter()-start
            self.times['compile_s'] += elapsed
            record = dict(key=self.current_key, seconds=elapsed, nodes=len(fn.fn.maker.fgraph.apply_nodes),
                          rngs=len(fn.ordered_rngs), input_names=fn.input_names, output_names=fn.out_names,
                          linker=type(fn.fn.maker.mode.linker).__name__,vm=type(fn.fn.vm).__name__,mode=self.mode_name)
            self.compile_records.append(record)
            self.event('compile', **record)
        self.last_names = tuple(requested)
        return self.draws[names]

    def draw(self, names, draws, *, diagnostic=False):
        assert draws == 1
        cell = self.cells[self.current_cell]
        ident = self.logical_ids[self.current_cell]
        fn = self.compile(names)
        start = time.perf_counter()
        raw = fn(cell.template_inputs, seed=cell.seed, draws=draws)
        elapsed = time.perf_counter()-start
        self.times['replay_s' if diagnostic else 'draw_s'] += elapsed
        self.executions.append(dict(candidate_id=ident, seconds=elapsed, diagnostic=diagnostic))
        self.last_raw = raw
        counts = self.counts(self.current_cell)
        if self.internal:
            for role, n in zip(TRAJECTORY_INPUTS, counts[:2]):
                for c in TRAJECTORY_COMPONENTS:
                    key = tr.structural_key(role, c)
                    cell.structure[key] = np.asarray(raw[key][0, :n], dtype=bool)
                    cell.template_inputs[key][:] = 0
                    cell.template_inputs[key][:n] = cell.structure[key]
        flags = {role: np.stack([cell.structure[tr.structural_key(role,c)] for c in TRAJECTORY_COMPONENTS], axis=1)
                 for role in TRAJECTORY_INPUTS}
        self.realised_flags[ident] = flags
        check_start = time.perf_counter()
        if all(np.isfinite(v).all() for v in raw.values()):
            np.testing.assert_allclose(raw['baseline']+raw['contributions'].sum(-1)+raw['indirect_effects'], raw['outcome'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(raw['indirect_effects_by_source'].sum(-1),raw['indirect_effects'],rtol=1e-10,atol=1e-10)
            if not self.ordinary:
                for name in ('treatments','contributions','covariates','covariate_contribution'):
                    assert np.count_nonzero(raw[name][..., widths(counts)[name]:]) == 0, name
        value = crop({name: raw[name] for name in names}, counts)
        h = digest(value)
        if diagnostic:
            assert self.all_primary_hashes[ident] == h, f'replay {ident}'
        else:
            self.all_primary_hashes[ident] = h
        self.times['invariants_and_hash_s'] += time.perf_counter()-check_start
        return value
