"""Isolated benchmark adapters; no production module is monkeypatched.

Function cloning keeps the existing builders' code and private namespace, with
only explicitly named collaborators rebound in the cloned namespace. This lets
specialization reuse the exact prior constructors and causal masking, without
copying or editing their scientific definitions. Unsupported broader features
are deliberately rejected by this benchmark prototype.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from types import FunctionType

import numpy as np
import pytensor
from pytensor.graph.replace import clone_replace
from template_vs_per_world import CORPUS_NAMES

from pymc_generator import symbolic_graph as sg
from pymc_generator import world_model as wm
from pymc_generator import world_model_template as wt
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS


def clone_function(fn, **bindings):
    result = FunctionType(
        fn.__code__, {**fn.__globals__, **bindings}, fn.__name__, fn.__defaults__, fn.__closure__
    )
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def build_specialized(cfg, inputs, weeks):
    wt.check_template_supported(cfg)
    if (
        cfg.trajectory_components_enabled
        or cfg.mechanism_priors_enabled
        or cfg.record_walk_parameters
    ):
        raise ValueError(
            "benchmark specialization only supports the approved baseline mechanisms/texture"
        )
    saturation = [
        i for i, key in enumerate(SATURATION_FAMILY_KEYS) if cfg.saturation_family_probs[key] > 0
    ]
    if len(saturation) != 1:
        raise ValueError("specialized benchmark requires exactly one saturation family")
    carryover = [
        i for i, key in enumerate(CARRYOVER_FAMILY_KEYS) if cfg.carryover_family_probs[key] > 0
    ]

    def parameters(*args, **kwargs):
        # Same helper, same priors, same creation order. Mixed carryover remains
        # the original runtime family vector; never turn its mixture homogeneous.
        kwargs["sat_family"] = np.full(cfg.n_treatments, saturation[0], dtype="int64")
        if len(carryover) == 1:
            kwargs["carryover_family"] = np.full(cfg.n_treatments, carryover[0], dtype="int64")
        return wt._scm_params(*args, **kwargs)

    def carry(col, params, k, *, dynamic_family=False):
        return sg._carryover_col(
            col, params, k, dynamic_family=not isinstance(params["carryover_family"], np.ndarray)
        )

    def saturate(col, scale, params, k, *, dynamic_family=False):
        return sg._saturate_col(
            col, scale, params, k, dynamic_family=not isinstance(params["sat_family"], np.ndarray)
        )

    graph = clone_function(sg.build_symbolic_graph, _carryover_col=carry, _saturate_col=saturate)
    builder = clone_function(
        wt.build_world_model_template, _scm_params=parameters, build_symbolic_graph=graph
    )
    return builder(cfg, inputs, weeks)


def widths(counts):
    k, m, j = counts
    return {
        **dict.fromkeys(
            (
                "treatments",
                "contributions",
                "saturation_scale",
                "param_treatment_level",
                "param_carryover_alpha",
                "param_weibull_lam",
                "param_weibull_k",
                "treatment_shock_mask",
            ),
            k,
        ),
        "covariates": m,
        "covariate_contribution": m,
        "latent_unobserved": j,
        "latent_unobserved_contribution": j,
    }


def active_result(result, counts):
    sizes = widths(counts)
    return {
        name: (value[..., : sizes[name]] if name in sizes else value)
        for name, value in result.items()
    }


def digest(result):
    h = hashlib.sha256()
    for name, value in sorted(result.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class Adapter:
    def __init__(self, method, cfg, cells, event):
        self.method, self.cfg, self.cells, self.event = method, cfg, cells, event
        self.models = {}
        self.draws = {}
        self.current_cell = 0
        self.times = defaultdict(float)
        self.accepted_ids = []
        self.retained_cell_ids = []
        self.evaluated = 0
        self.rejected = 0
        self.primary_hashes = {}
        self.executions = []

    def counts(self, cell):
        g = self.cells[cell].g
        return len(g["g_cy"]), len(g["g_zy"]), len(g["g_dy"])

    def padded_g(self, cell):
        data = self.cells[cell].template_inputs
        k, m, j = self.counts(cell)
        return {
            **{
                key: data[key].copy()
                for key in (
                    "g_cy",
                    "g_dc",
                    "g_dz",
                    "g_dy",
                    "g_zy",
                    "g_zc",
                    "g_cc",
                    "g_zz",
                    "active_treatment",
                    "active_covariate",
                    "active_latent",
                )
            },
            "treatment_active": ((data["g_cy"] != 0) | (data["g_cc"] != 0).any(axis=1)).astype(
                "uint8"
            ),
            "n_treatments_active": k,
            "n_covariates_active": m,
            "n_latent_active": j,
        }

    def build(self, g, cfg, structural, weeks, *, prior_cond=None):
        assert prior_cond is None
        key = self.current_cell if self.method == "ordinary" else 0
        if key not in self.models:
            t = time.perf_counter()
            if self.method == "ordinary":
                value = wm.build_world_model(g, cfg, structural, weeks)
            else:
                builder = (
                    wt.build_world_model_template if self.method == "dynamic" else build_specialized
                )
                value = builder(cfg, self.cells[self.current_cell].template_inputs, weeks)
            self.models[key] = value
            self.times["build_s"] += time.perf_counter() - t
            self.event("build", method=self.method, cell=self.current_cell)
        return self.models[key]

    def _validate_specialized_inputs(self, cell):
        if self.method != "specialized":
            return
        structural = self.cells[cell].structure
        for name, keys, probs in (
            ("sat_family", SATURATION_FAMILY_KEYS, self.cfg.saturation_family_probs),
            ("carryover_family", CARRYOVER_FAMILY_KEYS, self.cfg.carryover_family_probs),
        ):
            allowed = [i for i, k in enumerate(keys) if probs[k] > 0]
            if not np.isin(structural[name], allowed).all():
                raise ValueError(f"instruction violates specialized {name} contract")

    def draw(self, names, draws, *, diagnostic=False):
        assert tuple(names) == CORPUS_NAMES, "unapproved output contract"
        cell = self.current_cell
        key = cell if self.method == "ordinary" else 0
        self._validate_specialized_inputs(cell)
        model = self.models[key][0]
        if key not in self.draws:
            t = time.perf_counter()
            self.draws[key] = wt.compile_template_draw_fn(
                model, names, input_names=() if self.method == "ordinary" else None
            )
            self.times["compile_s"] += time.perf_counter() - t
            self.event("compile", method=self.method, cell=cell)
        t = time.perf_counter()
        value = self.draws[key](
            self.cells[cell].template_inputs, seed=self.cells[cell].seed, draws=draws
        )
        elapsed = time.perf_counter() - t
        self.times["replay_s" if diagnostic else "draw_s"] += elapsed
        self.executions.append(
            {
                "cell": cell,
                "draws": draws,
                "diagnostic": diagnostic,
                "seed": self.cells[cell].seed,
                "seconds": elapsed,
            }
        )
        self.event("draw", method=self.method, **self.executions[-1])
        # Check causal outputs before cropping, so post-hoc truncation cannot
        # hide leakage into inactive outputs. Nonfinite candidates are left to
        # the shared scientific rejection gate, not replaced or resampled.
        t = time.perf_counter()
        if all(np.isfinite(v).all() for v in value.values()):
            np.testing.assert_allclose(
                value["baseline"] + value["contributions"].sum(axis=-1) + value["indirect_effects"],
                value["outcome"],
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                value["indirect_effects_by_source"].sum(axis=-1),
                value["indirect_effects"],
                rtol=1e-10,
                atol=1e-10,
            )
            if self.method != "ordinary":
                sizes = widths(self.counts(cell))
                for name in (
                    "treatments",
                    "contributions",
                    "covariates",
                    "covariate_contribution",
                    "latent_unobserved",
                    "latent_unobserved_contribution",
                ):
                    assert np.count_nonzero(value[name][..., sizes[name] :]) == 0, name
        active = active_result(value, self.counts(cell))
        fingerprint = digest(active)
        if diagnostic:
            assert fingerprint == self.primary_hashes[(cell, draws)], (
                f"replay mismatch {self.method}/{cell}/{draws}"
            )
        else:
            self.primary_hashes[(cell, draws)] = fingerprint
        self.times["invariants_and_hash_s"] += time.perf_counter() - t
        return active

    def replay(self, draws):
        # The original order ended at cell 4. Replay 0..4 after intervening tasks.
        for cell in range(len(self.cells)):
            self.current_cell = cell
            self.draw(CORPUS_NAMES, draws, diagnostic=True)


def controlled_evaluate(model, names, primitives):
    replacements = {rv: rv.type(name=f"given_{rv.name}") for rv in model.free_RVs}
    expressions = clone_replace(
        [model[name] for name in names], replace=replacements, rebuild_strict=False
    )
    fn = pytensor.function(
        list(replacements.values()), expressions, on_unused_input="ignore", mode="FAST_COMPILE"
    )
    values = []
    for rv in replacements:
        shape = tuple(int(n) for n in rv.shape.eval())
        raw = np.asarray(primitives[rv.name][0], dtype=rv.dtype)
        values.append(raw[tuple(slice(n) for n in shape)] if shape else raw)
    return {name: np.asarray(value)[None] for name, value in zip(names, fn(*values))}


def equivalence_gate(adapters, event):
    dynamic = adapters["dynamic"].models[0][0]
    specialized = adapters["specialized"].models[0][0]
    ordinary = adapters["ordinary"].models[0][0]

    def signature(model):
        return [(rv.name, str(rv.owner.op), str(rv.dtype), rv.ndim) for rv in model.free_RVs]

    assert signature(dynamic) == signature(specialized), (
        "specialization changed primitive constructors"
    )
    # One additional diagnostic joint draw; no attempt to align independent seeds.
    primitive_names = tuple(rv.name for rv in dynamic.free_RVs)
    t = time.perf_counter()
    draw = wt.compile_template_draw_fn(dynamic, primitive_names)
    primitives = draw(adapters["dynamic"].cells[0].template_inputs, seed=770001, draws=1)
    event("controlled_primitive_draw", draws=1, seed=770001, seconds=time.perf_counter() - t)
    results = {}
    for method, model in (
        ("ordinary", ordinary),
        ("dynamic", dynamic),
        ("specialized", specialized),
    ):
        t = time.perf_counter()
        result = controlled_evaluate(model, CORPUS_NAMES, primitives)
        results[method] = active_result(result, adapters[method].counts(0))
        event("controlled_evaluation", method=method, seconds=time.perf_counter() - t)
    for method in ("dynamic", "specialized"):
        for name in CORPUS_NAMES:
            np.testing.assert_allclose(
                results[method][name],
                results["ordinary"][name],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"{method}/{name}",
            )
    event(
        "controlled_equivalence_pass",
        methods=list(results),
        fields=len(CORPUS_NAMES),
        rtol=1e-10,
        atol=1e-10,
    )
