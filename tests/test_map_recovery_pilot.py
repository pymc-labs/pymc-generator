"""Safe unit tests for the cold MAP report; no generator or optimizer is run."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/map_recovery_pilot.py"
spec = importlib.util.spec_from_file_location("map_recovery_pilot", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_fixed_config_cases_and_regime_ranges():
    cfg = module.load_config(Path(__file__).parents[1] / "docs/examples/data/map-recovery-pilot-config.json")
    assert [(c["world_seed"], c["support_seed"]) for c in cfg["cases"]] == [(201,301),(202,302),(203,303),(204,304)]
    assert cfg["generator"]["n_time_steps"] == 104
    assert cfg["resources"]["case_timeout_seconds"] == 900


def test_rejects_changed_seed_protocol(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version":"map-recovery-smoke/v1", "cases":[]}))
    try:
        module.load_config(path)
    except ValueError as exc:
        assert "exactly four" in str(exc)
    else:
        raise AssertionError("invalid case budget accepted")


def test_counterfactual_savings_does_not_invent_costs_or_groups():
    value = module.reuse_estimate([{"case_id":"a", "status":"success"}])
    assert value["N"] == 1
    assert value["K"].startswith("unavailable")
    assert value["measured_compile_cost"] == "unavailable"
    assert value["counterfactual_only"] is True
    assert value["solve_work_avoided"] == 0


def test_exception_case_is_preserved(monkeypatch):
    class BrokenWorld:
        pass
    class FakePymc:
        pass
    import sys
    monkeypatch.setitem(sys.modules, "pymc", FakePymc())
    fake_generator = type("Generator", (), {"sample_scm": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fixture failure"))})
    monkeypatch.setitem(sys.modules, "pymc_generator", fake_generator)
    monkeypatch.setattr(module, "build_config", lambda regime: object())
    case = {"case_id":"constant-201", "world_seed":201, "support_seed":301}
    record = module._case(case, "constant", Path("."))
    assert record["status"] == "failure"
    assert record["exception_type"] == "RuntimeError"
    assert "fixture failure" in record["exception"]


def test_markdown_keeps_unattempted_and_failure_cases():
    text = module.render_markdown({"cases":[{"case_id":"x","status":"failure"},{"case_id":"y","status":"unattempted"}]})
    assert "| x | failure |" in text
    assert "| y | unattempted |" in text
    assert "not MAP20 prevalence" in text


# Everything below is engineering evidence: explicit doubles and disposable OS
# sleepers only. No public world generation, oracle construction or real MAP.
import copy
import dataclasses
import logging
import os
import resource
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

CONFIG = SCRIPT.parents[1] / module.CONFIG_DEFAULT


@pytest.mark.parametrize('mutate', [
    lambda c: c['cases'][0].update(world_seed=205),
    lambda c: c['cases'][0].update(support_seed=305),
    lambda c: c['cases'][0].update(regime='anything'),
    lambda c: c['cases'].reverse(),
    lambda c: c['resources'].update(case_timeout_seconds=901),
    lambda c: c['resources'].update(check_seconds=2),
    lambda c: c['regimes'].update(constant=[0, 1]),
    lambda c: c['generator'].update(carryover_burn_in=1),
    lambda c: c['resolved_factory']['constant'].update(mm_scale_prior='normal'),
    lambda c: c.update(extra='unapproved'),
])
def test_full_config_freeze_rejects_every_protocol_mutation(tmp_path, mutate):
    cfg = json.loads(CONFIG.read_text())
    mutate(cfg)
    path = tmp_path / 'mutated.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='freeze mismatch'):
        module.load_config(path)


def test_resolved_defaults_are_complete_and_only_regime_changes():
    cfg = module.load_config(CONFIG)
    a, b = cfg['resolved_factory']['constant'], cfg['resolved_factory']['time_varying']
    assert len(a) == len(b) == 106
    assert [k for k in a if a[k] != b[k]] == ['rw_baseline_std_range']
    assert a['n_treatments_active_range'] == [2, 2]
    assert a['n_covariates_active_range'] == [1, 1]
    assert a['n_latent_active_range'] == [1, 1]
    assert a['carryover_burn_in'] == 0
    assert a['mm_scale_prior'] == 'uniform'
    assert a['carryover_family_probs'] == {'none': .15, 'geometric': .425, 'weibull': .425}
    assert a['saturation_family_probs']['linear'] == .15
    assert sum(a['saturation_family_probs'].values()) == pytest.approx(1)
    assert module.build_config.__code__.co_names.count('make_scm_prior') >= 1


@pytest.fixture
def simulated_world(monkeypatch):
    """A fully labeled fake public SCM/model/optimizer; no scientific APIs run."""
    calls = []
    @dataclasses.dataclass
    class FakeConfig:
        n_time_steps: int = 104
        carryover_burn_in: int = 0
    class Variable:
        def __init__(self, name):
            self.name, self.dtype = name, 'float64'
    rv, value = Variable('beta'), Variable('beta_interval__')
    class ModelDouble:
        free_RVs = [rv]
        value_vars = [value]
        rvs_to_values = {rv: value}
        rvs_to_transforms = {rv: SimpleNamespace(name='interval')}
        coords = {'treatment': (0, 1)}
        named_vars_to_dims = {'beta': ('treatment',)}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def initial_point(self, **kwargs):
            calls.append(('initial_point', kwargs))
            return {'beta_interval__': np.array([0., .1])}
        def eval_rv_shapes(self):
            calls.append(('shapes', {}))
            return {'beta': (2,), 'beta_interval__': (2,)}
        def dlogp(self, **kwargs):
            calls.append(('symbolic_gradient', kwargs))
    class WorldDouble:
        g = {'g_cy': np.array([1, 0]), 'g_zy': np.array([1])}
        extras = {'structural': {'carryover_family': np.array(['none', 'geometric']),
                                'sat_family': np.array(['linear', 'hill'])}}
        data = {'outcome': np.arange(104.), 'treatments': np.ones((104, 2)),
                'covariates': np.ones((104, 1)), 'saturation_scale': np.ones(2),
                'latent_unobserved': np.ones((104, 1))}
        @property
        def params(self):
            assert any(name == 'find_MAP' for name, _ in calls), 'truth read before fit'
            return {'beta': np.array([1., 2.]), 'rw_b': {'std': .1}}
        @property
        def primitive_parameters(self):
            assert any(name == 'find_MAP' for name, _ in calls), 'truth read before fit'
            return {'beta': np.array([1., 2.]), 'unused_primitive': np.array([3.])}
        def oracle_model(self, **kwargs):
            calls.append(('oracle_model', kwargs))
            return ModelDouble()
    world = WorldDouble()
    def sample_scm(cfg, **kwargs):
        calls.append(('sample_scm', kwargs))
        return world
    raw = SimpleNamespace(success=True, status=0, message='SIMULATED convergence', nfev=4,
                          njev=4, nit=2, fun=1.5, x=np.array([0., .1]), jac=np.array([0., 0.]))
    def find_MAP(**kwargs):
        calls.append(('find_MAP', kwargs))
        return {'beta': np.array([1.5, 1.5]), 'beta_interval__': np.array([0., .1])}, raw
    monkeypatch.setattr(module, 'build_config', lambda regime: FakeConfig())
    monkeypatch.setitem(sys.modules, 'pymc_generator', SimpleNamespace(sample_scm=sample_scm))
    monkeypatch.setitem(sys.modules, 'pymc', SimpleNamespace(find_MAP=find_MAP))
    return world, calls, raw


def test_simulated_case_has_full_contract_recovery_and_exact_native_call(simulated_world):
    world, calls, _ = simulated_world
    case = module.load_config(CONFIG)['cases'][0]
    checkpoints = []
    result = module._case(case, case['regime'], module.ROOT,
                          lambda record: checkpoints.append(copy.deepcopy(record)))
    assert result['status'] == 'success'
    assert result['generation_attempted'] is result['fit_attempted'] is True
    assert [name for name, _ in calls].count('sample_scm') == 1
    assert [name for name, _ in calls].count('find_MAP') == 1
    assert dict(calls)['sample_scm'] == dict(seed=201, connect_all=False, name='constant-201', purpose='issue-30-cold-smoke-report')
    assert dict(calls)['oracle_model'] == {'latent': 'marginal'}
    assert dict(calls)['initial_point'] == {'random_seed': 301}
    kwargs = dict(calls)['find_MAP']
    assert set(kwargs) == {'start', 'vars', 'method', 'return_raw', 'include_transformed', 'progressbar', 'maxeval', 'options'}
    assert kwargs['method'] == 'L-BFGS-B'
    assert kwargs['options'] == {'maxiter':1000,'maxfun':1000,'ftol':1e-12,'gtol':1e-6,'maxls':20}
    assert kwargs['maxeval'] == 998 and kwargs['return_raw'] and kwargs['include_transformed']
    assert kwargs['progressbar'] is False
    assert result['data']['likelihood_rows'] == list(range(104))
    assert set(result['full_data']['arrays']) == set(world.data)
    assert result['data']['arrays']['outcome']['shape'] == [104]
    assert result['full_data']['sha256'] == module.digest(module.canonical(world.data))
    assert result['graph'] == module.plain(world.g)
    mapping = result['free_value_layout']['mapping'][0]
    assert mapping['rv'] == 'beta' and mapping['value'] == 'beta_interval__'
    assert mapping['rv_shape'] == mapping['value_shape'] == [2]
    assert mapping['rv_dtype'] == mapping['value_dtype'] == 'float64'
    assert mapping['coordinates'] == {'treatment': [0, 1]}
    assert mapping['transform']['name'] == 'interval' and mapping['flat_slice'] == [0, 2]
    recovery = result['recovery']
    assert recovery['mapped']['beta']['mae'] == recovery['mapped']['beta']['rmse'] == .5
    assert recovery['mapped']['beta']['bias'] == 0
    assert recovery['unmapped']
    assert len(recovery['truth_digest']) == 64
    assert set(result['timers']) == {'config_generation','model_build','support_initial_point',
        'pre_fit_contract_diagnostics','find_MAP_solve_inclusive_compilation','diagnostics','serialization'}
    assert all(value >= 0 for value in result['timers'].values())
    assert result['started_at'] <= result['ended_at']
    assert result['peak_memory_bytes'] > 0
    assert all('recovery' not in p for p in checkpoints if p['phase'] != 'diagnostics')
    assert checkpoints[-1]['status'] == 'success'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['none','missing_status','missing_message','missing_njev','missing_fun',
    'missing_success','missing_x','missing_jac','nan_fun','inf_x','nan_jac','false_success','bad_status',
    'cap','bad_count','nonfinite_point','empty_point'])
def test_simulated_raw_failures_are_failures(simulated_world, change):
    _, _, raw = simulated_world
    point = {'beta': np.array([1., 2.])}
    if change == 'none':
        raw = None
    elif change.startswith('missing_'):
        delattr(raw, change.removeprefix('missing_'))
    elif change == 'nan_fun': raw.fun = float('nan')
    elif change == 'inf_x': raw.x = np.array([float('inf')])
    elif change == 'nan_jac': raw.jac = np.array([float('nan')])
    elif change == 'false_success': raw.success = False
    elif change == 'bad_status': raw.status = 1
    elif change == 'cap': raw.nfev = 1000
    elif change == 'bad_count': raw.nfev = None
    elif change == 'nonfinite_point': point['beta'][0] = float('nan')
    elif change == 'empty_point': point = {}
    evidence, success = module.validate_optimizer(point, raw)
    assert not success and evidence['failure_reasons']
    json.dumps(evidence, allow_nan=False)


def test_simulated_native_exception_retains_timers_truth_and_no_retry(simulated_world, monkeypatch):
    _, calls, _ = simulated_world
    def fail(**kwargs):
        calls.append(('find_MAP', kwargs))
        raise RuntimeError('SIMULATED native cap/compile failure')
    monkeypatch.setattr(sys.modules['pymc'], 'find_MAP', fail)
    case = module.load_config(CONFIG)['cases'][0]
    result = module._case(case, case['regime'], module.ROOT)
    assert result['status'] == 'failure' and result['exception_type'] == 'RuntimeError'
    assert 'SIMULATED native cap/compile failure' in result['traceback']
    assert result['recovery']['truth_digest']
    assert result['timers']['find_MAP_solve_inclusive_compilation'] >= 0
    assert result['timers']['diagnostics'] >= 0
    assert [name for name, _ in calls].count('find_MAP') == 1


def test_native_fallback_warning_aborts_before_other_solver():
    logger = logging.getLogger('pymc')
    before = list(logger.handlers)
    with pytest.raises(RuntimeError, match='fallback forbidden'):
        with module.forbid_native_fallback():
            logger.warning("Defaulting to non-gradient minimization 'Powell'.")
    assert logger.handlers == before


def test_k_depends_on_graph_families_config_prior_layout_not_labels(simulated_world):
    world, _, _ = simulated_world
    base = module._case(module.load_config(CONFIG)['cases'][0], 'constant', module.ROOT)
    same = copy.deepcopy(base)
    same.update(case_id='different-label', regime='unrelated-label')
    result = module.reuse_estimate([base, same])
    assert result['N'] == 2 and result['K'] == 1
    for field in ('graph','families_and_structure','full_config','prior_conditioning',
                  'free_value_layout','data_layout','saturation_scale','rows'):
        altered = copy.deepcopy(same)
        altered['compatibility']['contract'][field] = ['SIMULATED distinct ' + field]
        assert module.reuse_estimate([base, altered])['K'] == 2
    result = module.reuse_estimate([base, same, {'case_id':'missing', 'status':'failure'}])
    assert result['K'].startswith('unavailable') and result['known_K'] == 1
    assert result['unknown_cases'] == ['missing']
    result = module.reuse_estimate([base, same])
    assert len(result['scenarios']) == 4
    for scenario in result['scenarios']:
        lo, hi = scenario['setup_plus_compile_seconds_range']
        _, bind_hi = scenario['bind_seconds_range']
        assert scenario['avoided_seconds_range'] == [lo - 2 * bind_hi, hi]
    assert result['scenarios'][-1]['avoided_seconds_range'][0] < 0
    assert result['projection20']['K_range'] == [1,20]
    assert result['solve_work_avoided'] == 0


def test_atomic_case_checkpoint_no_clobber_and_aggregate_survival(tmp_path, monkeypatch):
    casepath = tmp_path / 'case.checkpoint.json'
    module.write_json(casepath, {'status': 'failure'})
    with pytest.raises(FileExistsError):
        module.write_json(casepath, {'status': 'success'})
    assert json.loads(casepath.read_text()) == {'status':'failure'}
    report = {'cases':[{'case_id':'simulated','status':'failure'}]}
    module._checkpoint(tmp_path, report)
    previous = (tmp_path / 'map-recovery-smoke.json').read_bytes()
    def interrupted(*args):
        raise InterruptedError('SIMULATED interruption before atomic rename')
    monkeypatch.setattr(module.os, 'replace', interrupted)
    with pytest.raises(InterruptedError):
        module._checkpoint(tmp_path, {'cases':[]})
    assert (tmp_path / 'map-recovery-smoke.json').read_bytes() == previous
    assert not list(tmp_path.glob('.pending-*'))


def test_atomic_fsyncs_file_and_directory(tmp_path, monkeypatch):
    original, calls = os.fsync, []
    def fsync(fd):
        calls.append(os.fstat(fd).st_mode)
        original(fd)
    monkeypatch.setattr(module.os, 'fsync', fsync)
    module.write_json(tmp_path / 'durable.json', {'simulated': True})
    import stat
    assert any(stat.S_ISREG(mode) for mode in calls)
    assert any(stat.S_ISDIR(mode) for mode in calls)


def test_recover_interrupted_case_from_latest_immutable_phase(tmp_path):
    case = module.load_config(CONFIG)['cases'][0]
    phase = {**case, 'phase':'find_MAP_solve_inclusive_compilation','fit_attempted':True,'status':'started'}
    module.write_json(tmp_path / (case['case_id'] + '.phase-01.json'), phase)
    process = {'exit_code':-signal.SIGKILL,'reason':'timeout'}
    result = module.recover_entry(tmp_path, case, process, 'timeout')
    assert result['status'] == 'timeout' and result['fit_attempted']
    assert result['phase'] == 'find_MAP_solve_inclusive_compilation'
    assert result['process']['exit_code'] == -9


def test_run_no_release_and_no_clobber_before_children(tmp_path):
    with pytest.raises(RuntimeError, match='separate owner release'):
        module.run(CONFIG, tmp_path / 'new')
    with pytest.raises(FileExistsError, match='clobber'):
        module.run(CONFIG, tmp_path, released=True)


@pytest.mark.parametrize('elapsed, expected', [(0,True),(4289.9,True),(4290,True),(4290.01,False),(5219,False),(5220,False),(5400,False)])
def test_exact_remaining_930_plus_180_and_workstop(elapsed, expected):
    assert module.remaining_allows_case(elapsed) is expected


def test_resource_reserve_cache_plus_artifact_and_free_space(tmp_path, monkeypatch):
    cache, output = tmp_path/'cache', tmp_path/'artifacts'
    cache.mkdir(); output.mkdir()
    (cache/'tiny').write_bytes(b'123')
    measured = module._resource_snapshot(cache, output)
    assert measured['cache_bytes'] >= 3 and measured['artifact_bytes'] == 0
    def full(path):
        return 10737418240 - 134217728 if path == cache else 0
    monkeypatch.setattr(module, '_tree_bytes', full)
    with pytest.raises(RuntimeError, match='finalization reserve'):
        module._resource_snapshot(cache, output)
    monkeypatch.setattr(module, '_tree_bytes', lambda p:0)
    monkeypatch.setattr(module.os, 'statvfs', lambda p:SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(RuntimeError, match='free space'):
        module._resource_snapshot(cache, output)


def test_private_cache_path_and_lock_are_enforced(tmp_path, monkeypatch):
    cache = tmp_path/'private'
    cache.mkdir(mode=0o700)
    monkeypatch.setenv('PYTENSOR_FLAGS', f'base_compiledir={cache}')
    assert module.cache_path() == cache
    with module.cache_lock(cache) as fd:
        module._verify_lock(cache, fd)
        with pytest.raises(BlockingIOError):
            with module.cache_lock(cache):
                pass
    cache.chmod(0o755)
    with pytest.raises(RuntimeError, match='0700'):
        module.cache_path()
    monkeypatch.setenv('PYTENSOR_FLAGS', 'base_compiledir=/tmp')
    with pytest.raises(RuntimeError, match='0700'):
        module.cache_path()
    monkeypatch.setenv('PYTENSOR_FLAGS', f'base_compiledir={cache},floatX=float32')
    with pytest.raises(RuntimeError, match='exactly'):
        module.cache_path()


@pytest.fixture
def simulated_preflight(monkeypatch, tmp_path):
    """Provenance fixtures only: no dependency imports or factory calls."""
    cfg = module.load_config(CONFIG)
    cache = tmp_path/'cache'; cache.mkdir(mode=0o700)
    monkeypatch.setattr(module, 'source_proof', lambda:{'head':module.BASE,'index_empty':True})
    monkeypatch.setattr(module.sys, 'executable', module.PYTHON)
    monkeypatch.setattr(module.sys, 'prefix', str(Path(module.PYTHON).parents[1]))
    monkeypatch.setattr(module.resource, 'getrlimit', lambda kind:(17179869184,)*2)
    monkeypatch.setattr(module, 'runtime_bindings', lambda:{'versions':module.EXPECTED_VERSIONS,'module_paths':{}})
    monkeypatch.setattr(module, 'build_config', lambda regime: SimpleNamespace(regime=regime))
    monkeypatch.setattr(module.dataclasses, 'asdict', lambda obj:cfg['resolved_factory'][obj.regime])
    monkeypatch.setattr(module.os, 'environ', {'PYTHONPATH':str(module.ROOT),
        'PYTENSOR_FLAGS':f'base_compiledir={cache}', **dict.fromkeys(module.THREAD_ENV,'1')})
    argv = module.launcher_command(CONFIG, tmp_path/'output', '--preflight')
    monkeypatch.setattr(module.sys, 'argv', argv[1:])
    monkeypatch.setattr(module.sys, 'orig_argv', argv)
    monkeypatch.setattr(module.sys, 'path', [str(module.ROOT/'scripts'), str(module.ROOT), *sys.path[2:]])
    return cache


def test_simulated_preflight_checks_full_runtime_not_metadata(simulated_preflight, monkeypatch):
    # Pytest inserts its phase marker after fixture setup; the documented launch
    # uses env -i, so remove the harness-only marker in this simulated environment.
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    gate = module.preflight(CONFIG)
    assert gate['ok'], gate['errors']
    assert gate['versions'] == module.EXPECTED_VERSIONS
    assert set(gate['resolved_factory']) == {'constant','time_varying'}
    assert gate['actual_address_space_limits'] == [17179869184]*2
    assert gate['cache']['exclusive_lock']


@pytest.mark.parametrize('kind', ['pythonpath','shared_cache','threads','unbounded_as','wrong_python','full_factory','source','version'])
def test_simulated_preflight_rejects_before_generation(simulated_preflight, monkeypatch, kind):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    def fail(): raise RuntimeError('SIMULATED provenance failure')
    if kind == 'pythonpath': monkeypatch.delenv('PYTHONPATH')
    elif kind == 'shared_cache': monkeypatch.setenv('PYTENSOR_FLAGS','base_compiledir=/tmp')
    elif kind == 'threads': monkeypatch.setenv('OMP_NUM_THREADS','2')
    elif kind == 'unbounded_as': monkeypatch.setattr(module.resource,'getrlimit',lambda kind:(-1,-1))
    elif kind == 'wrong_python': monkeypatch.setattr(module.sys,'executable','/usr/bin/python3')
    elif kind == 'full_factory': monkeypatch.setattr(module.dataclasses,'asdict',lambda cfg:{})
    elif kind == 'source': monkeypatch.setattr(module,'source_proof',fail)
    elif kind == 'version': monkeypatch.setattr(module,'runtime_bindings',fail)
    gate = module.preflight(CONFIG)
    assert not gate['ok'] and gate['errors']
    expected = {'pythonpath':'PYTHONPATH', 'shared_cache':'0700', 'threads':'thread',
                'unbounded_as':'address-space', 'wrong_python':'interpreter',
                'full_factory':'full resolved', 'source':'SIMULATED provenance', 'version':'SIMULATED provenance'}
    assert expected[kind] in gate['errors'][0]
    # No public sample_scm or MAP is reachable from preflight.
    assert 'sample_scm' not in module.preflight.__code__.co_names
    assert 'find_MAP' not in module.preflight.__code__.co_names


@pytest.mark.parametrize('defect', ['version','dependency_path','source_path'])
def test_actual_import_binding_rejection_with_labeled_modules(monkeypatch, tmp_path, defect):
    def simulated_import(name):
        if name in module.EXPECTED_VERSIONS:
            return SimpleNamespace(__version__='0.0' if defect == 'version' else module.EXPECTED_VERSIONS[name],
                __file__=('/wrong/site-packages/module.py' if defect == 'dependency_path' else
                    str(Path(module.PYTHON).parents[1] / 'lib/python3.13/site-packages' / name / '__init__.py')))
        return SimpleNamespace(__file__='/rejected-phase-A/pymc_generator/worlds.py')
    monkeypatch.setattr(module.importlib,'import_module',simulated_import)
    with pytest.raises(RuntimeError, match='version/path mismatch|escaped cold lane'):
        module.runtime_bindings()


def test_source_proof_rejects_head_index_and_modified_blob(monkeypatch):
    original = module._git
    monkeypatch.setattr(module, '_git', lambda *a:'wrong' if a == ('rev-parse','HEAD') else original(*a))
    with pytest.raises(RuntimeError, match='HEAD/branch'):
        module.source_proof()
    monkeypatch.setattr(module, '_git', lambda *a:'staged.py' if a == ('diff','--cached','--name-only') else original(*a))
    with pytest.raises(RuntimeError, match='source/index'):
        module.source_proof()
    monkeypatch.setattr(module,'_git',original)
    read = Path.read_bytes
    def changed_bytes(path):
        data = read(path)
        return data + b'SIMULATED MODIFICATION' if path.name == 'worlds.py' else data
    monkeypatch.setattr(Path,'read_bytes',changed_bytes)
    with pytest.raises(RuntimeError, match='byte/mode mismatch'):
        module.source_proof()


def test_source_proof_accepts_exact_four_untracked_paths():
    proof = module.source_proof()
    assert proof['head'] == module.BASE
    assert proof['index_empty'] is True
    assert len(proof['tracked_files']) == 122


@pytest.mark.parametrize('defect', [
    'fifth_path', 'newline_path', 'missing_path', 'unexpected_path',
    'modified_status', 'staged_status', 'duplicate_path', 'rename_status',
    'empty_status', 'missing_terminator',
])
def test_exact_source_status_rejects_before_generation(monkeypatch, simulated_world, defect):
    # Raw porcelain doubles avoid creating unauthorized files in the protected lane.
    records = [b'?? docs/examples/data/map-recovery-pilot-config.json',
               b'?? docs/examples/map-recovery-pilot.md',
               b'?? scripts/map_recovery_pilot.py',
               b'?? tests/test_map_recovery_pilot.py']
    if defect == 'fifth_path': records.append(b'?? pymc_generator/unauthorized.py')
    elif defect == 'newline_path': records.append(b'?? unexpected\npath.py')
    elif defect == 'missing_path': records.pop()
    elif defect == 'unexpected_path': records[-1] = b'?? tests/unexpected.py'
    elif defect == 'modified_status': records[-1] = b' M tests/test_map_recovery_pilot.py'
    elif defect == 'staged_status': records[-1] = b'A  tests/test_map_recovery_pilot.py'
    elif defect == 'duplicate_path': records.append(records[-1])
    elif defect == 'rename_status': records.extend([b'R  unexpected.py', b'original.py'])
    elif defect == 'empty_status': records.clear()
    status = b'\0'.join(records) + (b'\0' if records and defect != 'missing_terminator' else b'')
    original = module.subprocess.check_output
    status_argv = ['git', '-C', str(module.ROOT), 'status', '--porcelain=v1', '--untracked-files=all', '-z']
    monkeypatch.setattr(module.subprocess, 'check_output',
                        lambda argv, **kwargs: status if argv == status_argv else original(argv, **kwargs))
    monkeypatch.setattr(module.os, 'environ', module.clean_environment())
    def forbidden_runtime():
        raise AssertionError('source rejection must precede dependency imports')
    monkeypatch.setattr(module, 'runtime_bindings', forbidden_runtime)
    gate = module.preflight(CONFIG)
    assert gate['ok'] is False
    assert 'exactly the four authorized untracked report paths' in gate['errors'][0]
    assert simulated_world[1] == []  # No generation, model construction, or MAP calls.


@pytest.fixture
def process_dirs(tmp_path):
    """Real OS sleepers, no scientific code; tight test-only wall clocks."""
    cache, output = tmp_path/'cache', tmp_path/'output'
    cache.mkdir(mode=0o700); output.mkdir(mode=0o700)
    module.subreaper()
    return cache, output


def test_real_once_only_process_exit_stdout_stderr_env_and_one_second_monitor(process_dirs):
    cache, output = process_dirs
    argv = [sys.executable,'-c',"import sys,time; print('SIMULATED stdout',flush=True); print('SIMULATED stderr',file=sys.stderr,flush=True); time.sleep(1.1); sys.exit(7)"]
    record = module.supervise(argv,output,'fixture',cache,5)
    assert record['exit_code'] == 7
    assert record['argv'] == argv and record['environment']['PYTHONDONTWRITEBYTECODE'] == '1'
    assert record['poll_seconds'] == 1
    assert len(record['samples']) >= 2
    assert 'SIMULATED stdout' in (output / record['stdout']).read_text()
    assert 'SIMULATED stderr' in (output / record['stderr']).read_text()
    assert record['stdout_bytes'] and record['stderr_bytes']
    assert record['survivors'] == []


def test_real_timeout_term_kill_and_descendant_cleanup(process_dirs):
    cache, output = process_dirs
    script = """import signal,subprocess,sys,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'],start_new_session=True)
print('SIMULATED descendant',child.pid,flush=True)
time.sleep(60)
"""
    record = module.supervise([sys.executable,'-c',script],output,'timeout',cache,.4,
                              poll_seconds=.03,kill_grace=.15)
    assert record['reason'] == 'timeout' and record['exit_code'] == -signal.SIGKILL
    assert [s['signal'] for s in record['signals']] == ['TERM','KILL']
    assert len(record['observed_processes']) >= 2
    assert record['survivors'] == []
    table = module.process_table()
    assert all(p['pid'] not in table or table[p['pid']]['state'] == 'Z' for p in record['observed_processes'])


def test_real_early_parent_exit_cleans_orphan_session(process_dirs):
    cache, output = process_dirs
    script = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)"
    record = module.supervise([sys.executable,'-c',script],output,'orphan',cache,2,poll_seconds=.03,kill_grace=.15)
    assert record['exit_code'] == 0
    assert record['reason'] == 'descendant_cleanup_after_exit'
    assert record['signals'][0]['signal'] == 'TERM'
    assert not record['survivors']


def test_live_resource_monitor_stops_running_process(process_dirs, monkeypatch):
    cache, output = process_dirs
    original = module._resource_snapshot
    count = [0]
    def exhaust(*args):
        count[0] += 1
        if count[0] > 2:
            raise RuntimeError('SIMULATED cache/artifact reserve exhaustion')
        return original(*args)
    monkeypatch.setattr(module,'_resource_snapshot',exhaust)
    record = module.supervise([sys.executable,'-c','import time; time.sleep(60)'], output,'resource',cache,2,
                              poll_seconds=.03,kill_grace=.15)
    assert count[0] >= 3
    assert record['reason'].startswith('resource_monitor:')
    assert record['exit_code'] == -signal.SIGTERM and record['survivors'] == []


def test_live_external_stop_cleanup(process_dirs):
    cache, output = process_dirs
    begin = time.monotonic()
    record = module.supervise([sys.executable,'-c','import time; time.sleep(60)'],output,'interrupt',cache,2,
        poll_seconds=.03,kill_grace=.15, stop=lambda:'SIMULATED external TERM' if time.monotonic()-begin > .1 else None)
    assert record['reason'] == 'SIMULATED external TERM'
    assert record['exit_code'] == -signal.SIGTERM and not record['survivors']


def test_aggregate_work_and_finalization_boundaries_with_real_signal_receiver(process_dirs):
    cache, output = process_dirs
    # Artificial monotonic start is labeled simulated; receiving process is real.
    argv = [sys.executable,'-c','import time; time.sleep(60)']
    record = module.supervise(argv,output,'aggregate',cache,5400,
        poll_seconds=.03,kill_grace=.15,aggregate_start=time.monotonic()-5370)
    assert record['reason'] == 'finalization_boundary_5370'
    assert [s['signal'] for s in record['signals']][:2] == ['USR1','TERM']
    assert not record['survivors']


def test_simulated_four_fresh_cases_once_and_durable_ledger(tmp_path, monkeypatch):
    cfg = module.load_config(CONFIG)
    output = tmp_path/'output'; output.mkdir()
    calls = []
    monkeypatch.setattr(module,'preflight',lambda *a:{'ok':True,'simulated':True})
    monkeypatch.setattr(module,'cache_path',lambda:tmp_path)
    def simulate(argv, directory, label, cache, timeout, **kwargs):
        calls.append((argv, timeout, kwargs))
        index = int(argv[argv.index('--case-index')+1])
        case = cfg['cases'][index]
        module.write_json(directory/f"{case['case_id']}.result.json", {**case,'status':'optimizer_failure','simulated':True})
        return {'exit_code':1,'reason':None,'argv':argv,'simulated':True}
    monkeypatch.setattr(module,'supervise',simulate)
    args = SimpleNamespace(config=CONFIG,output=output,lock_fd=99,started=str(time.monotonic()))
    saved = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    try:
        assert module._supervisor(args) == 1
    finally:
        for sig,handler in saved.items(): signal.signal(sig,handler)
    assert len(calls) == 4
    assert [int(argv[argv.index('--case-index')+1]) for argv, _, _ in calls] == list(range(4))
    assert all(timeout == 900 for _,timeout,_ in calls)
    report = json.loads((output/'map-recovery-smoke.json').read_text())
    assert len(report['cases']) == 4
    assert all(entry['status'] == 'failure' and entry['process']['exit_code'] == 1 for entry in report['cases'])
    assert len(list(output.glob('*.checkpoint.json'))) == 4
    assert len(list(output.glob('*.attempt.json'))) == 4


def test_simulated_insufficient_remaining_keeps_all_unattempted(tmp_path, monkeypatch):
    monkeypatch.setattr(module,'preflight',lambda *a:{'ok':True})
    monkeypatch.setattr(module,'supervise',lambda *a,**k:pytest.fail('case must not launch'))
    args=SimpleNamespace(config=CONFIG,output=tmp_path,lock_fd=99,started=str(time.monotonic()-4291))
    saved={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    try:
        assert module._supervisor(args) == 1
    finally:
        for sig,handler in saved.items(): signal.signal(sig,handler)
    report=json.loads((tmp_path/'map-recovery-smoke.json').read_text())
    assert len(report['cases']) == 4
    assert all(c['status'] == 'unattempted' and '930+180' in c['reason'] for c in report['cases'])
    assert not list(tmp_path.glob('*.attempt.json'))


def test_paths_and_actual_binding_evidence_are_json_serializable(monkeypatch, tmp_path):
    cache = tmp_path / 'cache'; cache.mkdir(mode=0o700)
    fake_tensor = SimpleNamespace(__version__='3.2.4',
        __file__=str(Path(module.PYTHON).parents[1]/'lib/python3.13/site-packages/pytensor/__init__.py'),
        config=SimpleNamespace(compiledir=cache/'compiledir_fixture',floatX='float64'))
    def simulated_import(name):
        if name == 'pytensor': return fake_tensor
        if name in module.EXPECTED_VERSIONS:
            return SimpleNamespace(__version__=module.EXPECTED_VERSIONS[name],
                __file__=str(Path(module.PYTHON).parents[1]/'lib/python3.13/site-packages'/name/'__init__.py'))
        return SimpleNamespace(__file__=str(module.ROOT/'pymc_generator'/(name.split('.')[-1]+'.py')))
    monkeypatch.setattr(module.importlib,'import_module',simulated_import)
    monkeypatch.setitem(sys.modules,'pytensor',fake_tensor)
    monkeypatch.setattr(module,'cache_path',lambda:cache)
    evidence=module.runtime_bindings()
    assert evidence['versions'] == module.EXPECTED_VERSIONS
    assert isinstance(evidence['compiledir'],str)
    assert set(evidence['module_paths']) == set(module.MODULES) | set(module.EXPECTED_VERSIONS)
    assert json.loads(module.canonical({'path':cache})) == {'path':str(cache)}
    json.dumps(module.plain(evidence),allow_nan=False)


def test_child_preflight_blocks_before_case_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'preflight',lambda *a:{'ok':False,'errors':['SIMULATED wrong version']})
    monkeypatch.setattr(module,'_case',lambda *a,**k:pytest.fail('generation must not be reachable'))
    args=SimpleNamespace(config=CONFIG,output=tmp_path,lock_fd=99,case_index=0)
    assert module._case_child(args) == 2
    result=json.loads((tmp_path/'constant-201.result.json').read_text())
    assert result['status'] == 'failure' and result['reason'] == 'preflight blocked'


def test_resource_missing_and_unreadable_are_not_silently_zero(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError,match='measurement unavailable'):
        module._resource_snapshot(tmp_path/'missing',tmp_path)
    def unreadable(path, **kwargs):
        kwargs['onerror'](PermissionError('SIMULATED unreadable cache'))
        return []
    monkeypatch.setattr(module.os,'walk',unreadable)
    with pytest.raises(PermissionError,match='unreadable'):
        module._tree_bytes(tmp_path)


def test_live_process_address_space_sample_violation_stops_child(process_dirs, monkeypatch):
    cache,output=process_dirs
    original=module.process_table
    def oversized():
        table=original()
        for pid,info in table.items():
            if info['ppid'] == os.getpid():
                info['vsize_bytes']=17179869185  # explicit simulated resource evidence
        return table
    monkeypatch.setattr(module,'process_table',oversized)
    record=module.supervise([sys.executable,'-c','import time; time.sleep(60)'],output,'aslimit',cache,2,
                            poll_seconds=.03,kill_grace=.15)
    assert 'address space' in record['reason']
    assert not record['survivors'] and record['exit_code'] < 0


def test_launcher_repairs_interrupted_aggregate_and_retains_all_four(tmp_path,monkeypatch):
    cache=tmp_path/'cache'; cache.mkdir(mode=0o700)
    output=tmp_path/'new-output'
    monkeypatch.setattr(module,'cache_path',lambda:cache)
    @module.contextlib.contextmanager
    def simulated_watchdog(*args):
        yield -123  # explicitly simulated guard identity, no live watchdog here
    monkeypatch.setattr(module,'hard_watchdog',simulated_watchdog)
    calls=[]
    def interrupted(argv,directory,label,cache,timeout,**kwargs):
        calls.append((argv,timeout,kwargs))
        case=module.load_config(CONFIG)['cases'][0]
        module.write_json(directory/'constant-201.attempt.json',case)
        module.write_json(directory/'constant-201.phase-03.json',{**case,'status':'started',
            'phase':'find_MAP_solve_inclusive_compilation','fit_attempted':True,'simulated':True})
        return {'exit_code':-9,'reason':'finalization_boundary_5370','simulated':True}
    monkeypatch.setattr(module,'supervise',interrupted)
    saved={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        assert module.run(CONFIG,output,released=True) == 1
    finally:
        for sig,handler in saved.items(): signal.signal(sig,handler)
    assert len(calls) == 1 and calls[0][1] == 5400
    assert '--supervisor' in calls[0][0] and calls[0][2]['pass_fds']
    report=json.loads((output/'map-recovery-smoke.json').read_text())
    assert len(report['cases']) == 4
    assert report['cases'][0]['status'] == 'failure' and report['cases'][0]['fit_attempted']
    assert all(c['status'] == 'unattempted' for c in report['cases'][1:])
    assert len(list(output.glob('*.checkpoint.json'))) == 4
    assert report['supervisor_process']['exit_code'] == -9


def test_resource_exhaustion_stops_subsequent_case_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'preflight',lambda *a:{'ok':True,'simulated':True})
    monkeypatch.setattr(module,'cache_path',lambda:tmp_path)
    calls=[]
    def exhausted(*args,**kwargs):
        calls.append(args)
        return {'exit_code':-15,'reason':'resource_monitor: SIMULATED reserve exhausted'}
    monkeypatch.setattr(module,'supervise',exhausted)
    args=SimpleNamespace(config=CONFIG,output=tmp_path,lock_fd=99,started=str(time.monotonic()))
    saved={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    try:
        assert module._supervisor(args) == 1
    finally:
        for sig,handler in saved.items(): signal.signal(sig,handler)
    assert len(calls) == 1
    report=json.loads((tmp_path/'map-recovery-smoke.json').read_text())
    assert report['cases'][0]['status'] == 'failure'
    assert all(c['status'] == 'unattempted' and 'reserve exhausted' in c['reason'] for c in report['cases'][1:])


def test_json_stays_authoritative_if_markdown_publication_is_interrupted(tmp_path,monkeypatch):
    module._checkpoint(tmp_path,{'cases':[{'case_id':'first','status':'failure'}]})
    before=(tmp_path/'map-recovery-smoke.md').read_bytes()
    original=module._atomic
    def interrupted(path,content,**kwargs):
        if path.suffix == '.md': raise InterruptedError('SIMULATED Markdown publication interrupted')
        return original(path,content,**kwargs)
    monkeypatch.setattr(module,'_atomic',interrupted)
    with pytest.raises(InterruptedError):
        module._checkpoint(tmp_path,{'cases':[{'case_id':'second','status':'unattempted'}]})
    assert json.loads((tmp_path/'map-recovery-smoke.json').read_text())['cases'][0]['case_id'] == 'second'
    assert (tmp_path/'map-recovery-smoke.md').read_bytes() == before


def test_independent_watchdog_real_start_handshake_and_true_exit(tmp_path):
    """Dedicated engineering controller process; no generation/model/MAP."""
    import subprocess
    cache, output = tmp_path/'cache',tmp_path/'watchdog'
    cache.mkdir(mode=0o700); output.mkdir(mode=0o700)
    script = f"""
import importlib.util,os,resource,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('cold_guard_fixture',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.environ['PYTENSOR_FLAGS']='base_compiledir='+{str(cache)!r}
resource.setrlimit(resource.RLIMIT_AS,(17179869184,17179869184))
with m.hard_watchdog(Path({str(CONFIG)!r}),Path({str(output)!r}),time.monotonic()):
    time.sleep(.1)
print('SIMULATED controller completed')
"""
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15,
                          env=module.clean_environment())
    assert result.returncode == 0, result.stderr
    assert 'SIMULATED controller completed' in result.stdout
    ready=json.loads((output/'watchdog.ready.json').read_text())
    assert ready['aggregate_term_seconds'] == 5400 and ready['kill_grace_seconds'] == 30
    assert ready['poll_seconds'] == 1 and ready['actual_address_space_limits'] == [17179869184]*2
    assert json.loads((output/'watchdog.process.json').read_text())['exit_code'] == 0
    assert '"status": "completed"' in (output/'watchdog.stdout').read_text()


def test_independent_watchdog_hard_5400_term_5430_kill_labeled_clock_double(tmp_path,monkeypatch):
    parent=987001
    fake={parent:{'pid':parent,'state':'S','ppid':1,'pgid':9000,'start_ticks':1,'vsize_bytes':1,'rss_bytes':1}}
    monkeypatch.setattr(module,'process_table',lambda:copy.deepcopy(fake))
    monkeypatch.setattr(module.os,'getppid',lambda:parent)
    monkeypatch.setattr(module,'cache_path',lambda:tmp_path)
    monkeypatch.setattr(module,'_resource_snapshot',lambda *a:{'simulated':True})
    monkeypatch.setattr(module.select,'select',lambda *a:([],[],[]))
    clock=iter([5400.,5430.])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    monkeypatch.setattr(module.os,'fsync',lambda _:None)  # pytest capture pipe, not a report file
    signals=[]
    monkeypatch.setattr(module,'signal_tree',lambda p,known,sig:signals.append((known,sig)))
    args=SimpleNamespace(watchdog_parent=parent,started='0',output=tmp_path,control_fd=-1)
    assert module._watchdog(args) == 1
    assert [s for _,s in signals] == [signal.SIGTERM,signal.SIGKILL]
    assert all(known[parent]['pgid'] == os.getpgrp() for known,_ in signals), 'never kill inherited shell group'


def test_completed_child_cannot_be_misclassified_as_living_descendant(process_dirs,monkeypatch):
    # An already exited short process must never turn a successful result into
    # a fabricated descendant-cleanup failure due to stale /proc sampling.
    cache,output=process_dirs
    for index in range(4):
        record=module.supervise([sys.executable,'-c','pass'],output,f'quick-{index}',cache,2,
                                poll_seconds=.01,kill_grace=.1)
        assert record['exit_code'] == 0 and record['reason'] is None
        assert record['signals'] == [] and record['survivors'] == []


def test_interrupted_native_timing_never_becomes_zero_compile_measurement(simulated_world):
    record=module._case(module.load_config(CONFIG)['cases'][0],'constant',module.ROOT)
    del record['timers']['find_MAP_solve_inclusive_compilation']  # explicit interrupted-phase fixture
    record['status']='timeout'
    report=module.reuse_estimate([record])
    assert report['K'] == 1 and report['N_complete_timing_evidence'] == 0
    assert len(report['scenarios']) == 1
    assert report['scenarios'][0]['compile_seconds_range'][1] == 'C_compile_hi (unknown)'
    assert 'symbolic only' in report['scenarios'][0]['label']
    assert report['measured_compile_cost'] == 'unavailable'


@pytest.mark.parametrize('defect', ['imported_path', 'python_c', 'relative_script', 'extra_python_flag', 'reordered_args'])
def test_prospective_script_context_rejects_before_imports(simulated_preflight, monkeypatch, defect):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    if defect == 'imported_path':
        monkeypatch.setattr(module.sys, 'path', ['', str(module.ROOT), *sys.path[2:]])
    elif defect == 'python_c':
        monkeypatch.setattr(module.sys, 'orig_argv', [module.PYTHON, '-c', 'SIMULATED imported preflight'])
    elif defect == 'relative_script':
        monkeypatch.setattr(module.sys, 'argv', ['scripts/map_recovery_pilot.py', *sys.argv[1:]])
        monkeypatch.setattr(module.sys, 'orig_argv', [module.PYTHON, *sys.argv])
    elif defect == 'extra_python_flag':
        monkeypatch.setattr(module.sys, 'orig_argv', [module.PYTHON, '-I', *sys.argv])
    elif defect == 'reordered_args':
        monkeypatch.setattr(module.sys, 'argv', [sys.argv[0], *sys.argv[3:5], *sys.argv[1:3], sys.argv[5]])
        monkeypatch.setattr(module.sys, 'orig_argv', [module.PYTHON, *sys.argv])
    monkeypatch.setattr(module, 'runtime_bindings', lambda: pytest.fail('scientific imports must not occur'))
    gate = module.preflight(CONFIG)
    assert gate['ok'] is False
    assert any(word in gate['errors'][0] for word in ('context', 'argv'))


def test_prospective_contract_declares_stable_and_dynamic_fields(simulated_preflight, monkeypatch):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    gate = module.preflight(CONFIG)
    assert gate['ok'], gate['errors']
    binding = gate['launch_contract']
    stable = binding['expected']['stable']
    dynamic = binding['expected']['per_launch']
    assert stable['source_oid'] == module.BASE
    assert stable['versions'] == module.EXPECTED_VERSIONS
    assert stable['threads'] == dict.fromkeys(module.THREAD_ENV, '1')
    assert stable['sys_path_prefix'] == [str(module.ROOT/'scripts'), str(module.ROOT)]
    assert binding['actual_import_context']['orig_argv'] == dynamic['argv'] == sys.orig_argv
    assert binding['actual_import_context']['sys_path'] == sys.path
    assert dynamic['environment'] == module.clean_environment()
    assert binding['expected']['full_runtime_metadata_digest_claimed'] is False
    assert set(dynamic['case_cache_directories'].values()) == {
        str(simulated_preflight/c['case_id']) for c in module.load_config(CONFIG)['cases']}
    assert len(set(dynamic['case_cache_directories'].values())) == 4
    assert not any((simulated_preflight/c['case_id']).exists() for c in module.load_config(CONFIG)['cases'])


@pytest.mark.parametrize('defect', [None, 'plan_argv', 'plan_environment', 'plan_cache_root', 'wrong_case_cache'])
def test_case_launch_contract_and_run_root_lock_before_imports(simulated_preflight, monkeypatch, tmp_path, defect):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    root = simulated_preflight
    output = tmp_path/'output'; output.mkdir(mode=0o700)
    case = module.load_config(CONFIG)['cases'][0]
    cache = root/case['case_id']; cache.mkdir(mode=0o700)
    with module.cache_lock(root) as fd:
        argv = module.child_command(CONFIG, output, fd, '--case-index', '0')
        env = {**module.clean_environment(), 'PYTENSOR_FLAGS':f'base_compiledir={cache}'}
        plan = module.expected_launch(argv, env, root)
        if defect == 'plan_argv': plan['per_launch']['argv'][-1] = '1'
        elif defect == 'plan_environment': plan['per_launch']['environment']['LANG'] = 'unexpected'
        elif defect == 'plan_cache_root': plan['per_launch']['cache_root'] = '/unexpected'
        elif defect == 'wrong_case_cache':
            wrong = root/'constant-202'; wrong.mkdir(mode=0o700)
            env['PYTENSOR_FLAGS'] = f'base_compiledir={wrong}'
        module.write_json(output/f"{case['case_id']}.launch.json", plan)
        monkeypatch.setattr(module.os, 'environ', env)
        monkeypatch.setattr(module.sys, 'argv', argv[1:])
        monkeypatch.setattr(module.sys, 'orig_argv', argv)
        if defect:
            monkeypatch.setattr(module, 'runtime_bindings', lambda: pytest.fail('imports must follow contract verification'))
        gate = module.preflight(CONFIG, output, fd)
    if defect:
        assert gate['ok'] is False
        assert 'contract' in gate['errors'][0] or 'case cache' in gate['errors'][0]
    else:
        assert gate['ok'], gate['errors']
        assert gate['cache']['path'] == str(cache)
        assert gate['cache']['run_root'] == str(root)
        assert gate['cache']['exclusive_lock'] is True
        assert not (cache/'.map-smoke-lock').exists(), 'lock stays at cumulative run root'


def test_four_case_caches_and_contracts_exist_before_popen_and_cannot_be_reused(process_dirs, monkeypatch):
    """Popen boundary double: never dispatch the scientific command."""
    root, output = process_dirs
    cfg = module.load_config(CONFIG)
    launches = []
    class LaunchBoundary(RuntimeError):
        pass
    def boundary(argv, **kwargs):
        index = int(argv[argv.index('--case-index')+1])
        case = cfg['cases'][index]
        cache = root/case['case_id']
        plan = json.loads((output/f"{case['case_id']}.launch.json").read_text())
        assert plan == module.expected_launch(argv, kwargs['env'], root)
        assert plan['per_launch']['environment']['PYTENSOR_FLAGS'] == f'base_compiledir={cache}'
        assert kwargs['cwd'] == module.ROOT
        assert cache.is_dir() and cache.stat().st_mode & 0o777 == 0o700
        assert list(cache.iterdir()) == []
        launches.append(cache)
        raise LaunchBoundary('SIMULATED stop before actual Popen')
    monkeypatch.setattr(module.subprocess, 'Popen', boundary)
    with module.cache_lock(root) as fd:
        for index, case in enumerate(cfg['cases']):
            argv = module.child_command(CONFIG, output, fd, '--case-index', str(index))
            with pytest.raises(LaunchBoundary, match='SIMULATED'):
                module.supervise(argv, output, case['case_id'], root, 900, pass_fds=(fd,))
        with pytest.raises(FileExistsError):
            module.supervise(argv, output, case['case_id'], root, 900, pass_fds=(fd,))
    assert len(launches) == len(set(launches)) == 4
    for index, cache in enumerate(launches):
        (cache/'fixture-bytes').write_bytes(b'x'*(index+1))
    measured = module._resource_snapshot(root, output)
    assert measured['cache_bytes'] == module._tree_bytes(root)
    assert measured['cache_bytes'] >= sum(module._tree_bytes(cache) for cache in launches) > 0
    assert measured['artifact_bytes'] == module._tree_bytes(output)
    with pytest.raises(RuntimeError, match='reserve'):
        module._resource_snapshot(root, output, limit=measured['cache_bytes']+measured['artifact_bytes']+134217728)


def test_exact_released_cli_dispatch_with_engineering_double(simulated_preflight, monkeypatch, tmp_path):
    """Exact actual parser argv; run is a labeled double, never a science release."""
    output = tmp_path/'not-created'
    argv = module.launcher_command(CONFIG, output, '--execute-released')
    monkeypatch.setattr(module.sys, 'argv', argv[1:])
    monkeypatch.setattr(module.sys, 'orig_argv', argv)
    calls = []
    def simulated_run(*args):
        calls.append(args)
        return 17
    monkeypatch.setattr(module, 'run', simulated_run)
    assert module.main() == 17
    assert calls == [(CONFIG, output, False, True)]
    assert not output.exists()


def test_actual_absolute_script_preflight_matches_declared_import_context(tmp_path):
    """Real CLI/import/config gate only; no generation, oracle or optimizer."""
    cache = tmp_path/'cli-cache'; cache.mkdir(mode=0o700)
    output = tmp_path/'no-science-output'
    argv = module.launcher_command(CONFIG, output, '--preflight')
    env = {'HOME':os.environ['HOME'], 'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8',
        **dict.fromkeys(module.THREAD_ENV, '1'), 'PYTHONPATH':str(module.ROOT),
        'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1',
        'PYTENSOR_FLAGS':f'base_compiledir={cache}'}
    process = module.subprocess.run(['prlimit', '--as=17179869184:17179869184', '--', *argv],
        cwd=module.ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    gate = json.loads(process.stdout)
    assert gate['ok'] and not gate['errors']
    binding = gate['launch_contract']
    assert binding['expected'] == module.expected_launch(argv, env, cache)
    assert binding['actual_import_context']['orig_argv'] == argv
    assert binding['actual_import_context']['sys_path'][:2] == [str(module.ROOT/'scripts'), str(module.ROOT)]
    assert gate['argv'] == argv[1:]
    assert gate['environment'] == env
    assert gate['versions'] == module.EXPECTED_VERSIONS
    assert gate['actual_address_space_limits'] == [17179869184]*2
    assert gate['source']['head'] == module.BASE
    assert not output.exists()
    assert not any((cache/c['case_id']).exists() for c in module.load_config(CONFIG)['cases'])
