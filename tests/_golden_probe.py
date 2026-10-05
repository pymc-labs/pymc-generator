"""Record seeded features-off outputs for the main-reproduction golden test.

Usage: ``python tests/_golden_probe.py OUT.npz [--sections ordinary,scm,...]``

``tests/test_main_reproduction.py`` runs this script once with main's package
tree (ce068d5) first on ``PYTHONPATH`` and once with the current tree, and
compares the two recordings byte for byte. The script therefore uses only APIs
present at both commits and enables no opt-in feature. ``OUT.npz`` holds numeric
arrays only. ``OUT.json``, next to it, holds the resolved file of every loaded
``pymc_generator`` module, library versions, section timings, the logged
``draw_worlds`` calls, planned skips, the configs checked to be features-off, and
the ``None``/``str``/``bool`` values found among the recorded mappings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytensor

import pymc_generator as pg
from pymc_generator import world_model
from pymc_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _slice_g_active,
    sample_g_additive,
)
from pymc_generator.world_model_template import (
    build_cell_inputs,
    build_world_model_template,
    compile_template_draw_fn,
    sample_cell_structures,
)

SECTIONS = ("ordinary", "scm", "oracle", "datagen", "template")
#: Opt-in switches that must stay off; neither property exists at main.
FEATURE_FLAGS = ("mechanism_priors_enabled", "trajectory_metadata_enabled")
CORPUS_NAMES = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES

CARRYOVER = ("none", "geometric", "weibull")
SATURATION = ("linear", "hill", "logistic", "michaelis_menten", "tanh", "root")
UNIFORM_SATURATION = {family: 1.0 / len(SATURATION) for family in SATURATION}
TEXTURE_OFF = dict.fromkeys(
    (
        "treatment_hf_sigma_range",
        "treatment_pulse_prob_range",
        "covariate_hf_sigma_range",
        "covariate_pulse_prob_range",
        "covariate_pulse_amp_range",
    ),
    (0.0, 0.0),
)


def _one_hot(keys: tuple[str, ...], hot: str) -> dict[str, float]:
    return {key: float(key == hot) for key in keys}


SMALL = {"n_treatments": 3, "n_covariates": 2, "n_latent": 1, "n_time_steps": 24}
BASE = {**SMALL, "n_cells": 3, "draws_per_cell": 2}
LARGE = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,
    "n_time_steps": 104,
    "n_cells": 4,
    "draws_per_cell": 2,
}
BIG = {
    "n_treatments": 5,
    "n_covariates": 3,
    "n_latent": 1,
    "n_time_steps": 104,
    "n_cells": 30,
    "draws_per_cell": 4,
}

#: Corpus overrides of ``BASE``; the ``scm`` section reuses some of them.
ORDINARY: dict[str, dict[str, Any]] = {
    "default": {},
    "seed7": {"seed": 7},
    "seed123": {"seed": 123},
    "shocks": {"n_treatment_shocks": 1},
    "shocks_long": {"n_treatment_shocks": 1, "treatment_shock_length_range": (3, 5)},
    "shock_level": {"n_treatment_shocks": 2, "treatment_shock_level_range": (0.5, 1.5)},
    "prior_cond": {"prior_conditioning": True},
    "pc_shocks": {"prior_conditioning": True, "n_treatment_shocks": 1},
    "absolute_noise": {"outcome_std_mode": "absolute"},
    "confounding": {"confounding_strength_range": (0.2, 0.6)},
    **{
        f"carry_{family}": {"carryover_family_probs": _one_hot(CARRYOVER, family)}
        for family in CARRYOVER
    },
    **{
        f"sat_{family}": {"saturation_family_probs": _one_hot(SATURATION, family)}
        for family in SATURATION
    },
    "all_sat": {"saturation_family_probs": UNIFORM_SATURATION},
    "floor_intercept": {"baseline_floor": 0.0},
    "floor_non_treatment": {"baseline_floor": 0.0, "baseline_floor_scope": "non_treatment"},
    "padded": {"n_treatments_active_range": (1, 3), "n_covariates_active_range": (1, 2)},
    "linear": {"nonlinearity": "linear"},
    # make_scm_prior warns (FutureWarning) that every texture term is off.
    "texture_off": TEXTURE_OFF,
    "smooth_covariates": {
        name: bounds for name, bounds in TEXTURE_OFF.items() if name.startswith("covariate_")
    },
    "dense_budget": {
        "edge_budget": {"cc": (2, 3), "zz": (1, 1), "zc": (2, 4), "dz": (1, 2), "dc": (1, 1)},
        "min_no_direct_effect_treatments": 1,
    },
    "beta_degen": {"beta_additive_range": (1.0, 1.0)},
    "rw_b_zero": {"rw_baseline_std_range": (0.0, 0.0)},
    "alpha_degen": {"carryover_alpha_range": (0.5, 0.5)},
    "lmax1": {"l_max": 1},
    "burnin0": {"carryover_burn_in": 0},
    "treat1": {"n_treatments": 1},
    "large_seed5": {**LARGE, "seed": 5},
    "big_tanh": {**BIG, "saturation_family_probs": _one_hot(SATURATION, "tanh")},
}

SCM_WORLDS = (
    "default",
    "sat_michaelis_menten",
    "sat_tanh",
    "absolute_noise",
    "floor_non_treatment",
    "shocks",
    "prior_cond",
    # Unused texture innovations take leftover streams.
    "texture_off",
    "smooth_covariates",
)
SCM_SEEDS = (0, 1)
#: None is floored, so the marginal oracle builds for every one of them. Root's
#: library graph matters only in oracle evaluation (sampled worlds agree either way).
ORACLE_WORLDS = ("default", "sat_michaelis_menten", "sat_tanh", "sat_root", "shocks")
ORACLE_LATENTS = ("marginal", "sampled")
#: The single oracle also evaluated in FAST_RUN (the Numba backend).
NUMBA_ORACLE = ("default", "marginal")

#: ``benchmarks/template_vs_per_world.py:dea_prior`` at 3 cells x 2 draws.
DEA_PRIOR: dict[str, Any] = {
    "n_treatments": 10,
    "n_covariates": 10,
    "n_latent": 1,
    "n_time_steps": 104,
    "l_max": 4,
    "carryover_burn_in": 4,
    "n_cells": 3,
    "draws_per_cell": 2,
    "seed": 999000,
    "n_treatments_active_range": (3, 10),
    "n_covariates_active_range": (1, 10),
    "n_latent_active_range": (1, 1),
    "carryover_alpha_range": (0.2, 0.8),
    "weibull_lam_range": (2.0, 8.0),
    "weibull_k_range": (1.5, 4.0),
    "beta_additive_range": (0.5, 2.0),
    "dc_coeff_range": (0.1, 0.5),
    "dz_coeff_range": (0.1, 0.5),
    "zc_coeff_range": (0.05, 0.3),
    "cc_coeff_range": (0.05, 0.3),
    "zz_coeff_range": (-0.2, 0.2),
    "dy_coeff_range": (0.15, 0.45),
    "zy_coeff_range": (0.1, 0.4),
    "rw_covariate_mean_range": (-1.0, 1.0),
    "rw_positive_mean_range": (0.3, 4.0),
    "rw_baseline_mean_range": (3.0, 8.0),
    "rw_baseline_std_range": (0.05, 0.2),
    "rw_outcome_std_range": (0.01, 0.028),
    "rw_treatment_std_range": (0.15, 0.8),
    "treatment_hf_sigma_range": (0.08, 0.6),
    "treatment_pulse_prob_range": (0.0, 0.25),
    "treatment_pulse_amp_range": (0.4, 2.5),
    "treatment_shock_length_range": (2, 2),
    "treatment_shock_level_range": (0.0, 0.0),
    "edge_budget": {
        "cy": (1, 10),
        "zy": (1, 10),
        "dc": 0,
        "dz": 0,
        "dy": 0,
        "zc": 0,
        "cc": 0,
        "zz": 0,
    },
    "confounding_strength_range": (0.0, 0.0),
    "min_no_direct_effect_treatments": 1,
    "carryover_family_probs": _one_hot(CARRYOVER, "geometric"),
    "saturation_family_probs": _one_hot(SATURATION, "michaelis_menten"),
}

#: Template configs. Every layout keeps at least two treatments and two covariates:
#: main cannot compile a template with one of either (its 1x1 ``g_cc`` / ``g_zz``
#: input goes unused).
TEMPLATE: dict[str, dict[str, Any]] = {
    "probe": {**SMALL, "n_cells": 3, "confounding_strength_range": (0.3, 0.3), "seed": 11},
    "active_all_families": {
        "n_treatments": 5,
        "n_covariates": 4,
        "n_latent": 2,
        "n_time_steps": 30,
        "n_cells": 6,
        "seed": 5,
        "n_treatments_active_range": (2, 5),
        "n_covariates_active_range": (1, 4),
        "n_latent_active_range": (1, 2),
        "saturation_family_probs": UNIFORM_SATURATION,
    },
    "texture_off_absolute": {
        "n_treatments": 4,
        "n_covariates": 3,
        "n_latent": 1,
        "n_time_steps": 20,
        "n_cells": 4,
        "seed": 21,
        **TEXTURE_OFF,
        "outcome_std_mode": "absolute",
    },
    "linear": {
        "n_treatments": 3,
        "n_covariates": 3,
        "n_latent": 2,
        "n_time_steps": 16,
        "n_cells": 4,
        "seed": 99,
        "nonlinearity": "linear",
    },
    "minimal_2_2_1": {
        "n_treatments": 2,
        "n_covariates": 2,
        "n_latent": 1,
        "n_time_steps": 24,
        "n_cells": 3,
        "seed": 4,
    },
    "dense_all_families": {
        "n_treatments": 4,
        "n_covariates": 3,
        "n_latent": 2,
        "n_time_steps": 20,
        "n_cells": 5,
        "seed": 8,
        "edge_budget": {"cc": (3, 3), "zz": (2, 2), "zc": (5, 5)},
        "saturation_family_probs": UNIFORM_SATURATION,
    },
    "floor_intercept": {**SMALL, "n_cells": 3, "seed": 3, "baseline_floor": 0.0},
    "floor_non_treatment": {
        **SMALL,
        "n_cells": 3,
        "seed": 3,
        "baseline_floor": 0.0,
        "baseline_floor_scope": "non_treatment",
    },
    "benchmark": DEA_PRIOR,
}
TEMPLATE_DRAWS = 2
#: The template config whose ``sample_cell_structures`` cells are also drawn.
STRUCTURES_TEMPLATE = "active_all_families"
STRUCTURES_SEED = 10_000


class Recording:
    """Arrays and sidecar records, keyed by ``/``-joined paths."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.scalars: dict[str, None | bool | str] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        self.skipped: list[list[str]] = []
        self.features_off: list[str] = []
        self._worlds: dict[tuple[str, int], Any] = {}

    def put(self, key: str, value: Any) -> None:
        array = np.array(value, copy=True)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"{key}: {type(value).__name__} of dtype {array.dtype} is not numeric")
        if key in self.arrays:
            raise KeyError(f"{key} recorded twice")
        self.arrays[key] = array

    def record(self, prefix: str, value: Any) -> None:
        """Record nested mappings; ``None``, ``str`` and ``bool`` leaves go to ``scalars``."""
        if isinstance(value, Mapping):
            for key, item in value.items():
                self.record(f"{prefix}/{key}", item)
        elif value is None or isinstance(value, (bool, str)):
            # Checked before the numeric case: a Python bool is also an int.
            if prefix in self.scalars:
                raise KeyError(f"{prefix} recorded twice")
            self.scalars[prefix] = value
        else:
            self.put(prefix, value)

    def corpus(self, prefix: str, corpus: Mapping[str, Any]) -> None:
        """Every array plus ``identifiability``; diagnostics hold wall-clock timings."""
        for key, value in corpus.items():
            if key != "diagnostics":
                self.record(f"{prefix}/{key}", value)

    def prior(self, label: str, **fields: Any) -> Any:
        cfg = pg.make_scm_prior(**fields)
        enabled = [flag for flag in FEATURE_FLAGS if getattr(cfg, flag, False) is not False]
        if enabled:
            raise RuntimeError(f"{label}: opt-in features enabled: {enabled}")
        self.features_off.append(label)
        return cfg

    def world(self, name: str, seed: int) -> Any:
        """``sample_scm`` world of an ``ORDINARY`` config, drawn once per process."""
        if (name, seed) not in self._worlds:
            cfg = self.prior(f"scm/{name}/seed{seed}", **{**BASE, **ORDINARY[name]})
            self._worlds[name, seed] = pg.sample_scm(cfg, seed=seed)
        return self._worlds[name, seed]

    @contextmanager
    def drawn_worlds(self, prefix: str) -> Iterator[None]:
        """Record every ``draw_worlds`` call and its raw candidate batch, arguments untouched."""
        original = world_model.draw_worlds
        signature = inspect.signature(original)
        calls = self.calls.setdefault(prefix, [])

        def draw_worlds(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            index = len(calls)
            call = {
                "names": list(bound.arguments["out_names"]),
                "seed": int(bound.arguments["seed"]),
                "draws": int(bound.arguments["draws"]),
                "mode": str(bound.arguments["mode"]),
                "rng_reference_names": list(bound.arguments["rng_reference_names"] or ()),
            }
            calls.append(call)
            try:
                drawn = original(*args, **kwargs)
            except Exception as exc:
                call["raised"] = type(exc).__name__
                raise
            for name, value in drawn.items():
                self.put(f"{prefix}/draw{index:04d}/{name}", value)
            return drawn

        world_model.draw_worlds = draw_worlds
        try:
            yield
        finally:
            world_model.draw_worlds = original


def ordinary(rec: Recording) -> None:
    for name, overrides in ORDINARY.items():
        cfg = rec.prior(f"ordinary/{name}", **{**BASE, **overrides})
        with rec.drawn_worlds(f"ordinary/{name}"):
            corpus = pg.sample_prior_predictive(cfg)
        rec.corpus(f"ordinary/{name}/corpus", corpus)


def scm(rec: Recording) -> None:
    for name in SCM_WORLDS:
        for seed in SCM_SEEDS:
            world = rec.world(name, seed)
            prefix = f"scm/{name}/seed{seed}"
            rec.record(f"{prefix}/data", world.data)
            rec.record(f"{prefix}/g", world.g)
            rec.record(f"{prefix}/exogenous", world.exogenous)
            # structural, plus prior_cond {quantity: (low, width)} when conditioned.
            rec.record(f"{prefix}/extras", world.extras)
            rec.record(f"{prefix}/params", world.params)


def oracle(rec: Recording) -> None:
    for name in ORACLE_WORLDS:
        world = rec.world(name, 0)
        for latent in ORACLE_LATENTS:
            prefix = f"oracle/{name}/seed0/{latent}"
            try:
                model = world.oracle_model(latent=latent)
            except ValueError as exc:
                # The planned skip: a marginal oracle cannot represent a floor.
                if latent != "marginal" or world.cfg.baseline_floor is None:
                    raise
                rec.skipped.append([prefix, str(exc)])
                continue
            point = {
                key: value + 0.125 * np.sin(1.0 + np.arange(value.size)).reshape(value.shape)
                for key, value in sorted(model.initial_point().items())
            }
            rec.record(f"{prefix}/point", point)
            numba = (name, latent) == NUMBA_ORACLE
            for mode in ("FAST_COMPILE", "FAST_RUN") if numba else ("FAST_COMPILE",):
                rec.put(f"{prefix}/{mode}/logp", model.compile_logp(mode=mode)(point))
                rec.put(f"{prefix}/{mode}/dlogp", model.compile_dlogp(mode=mode)(point))
            evaluate = pytensor.function(
                model.value_vars,
                model.replace_rvs_by_values(model.deterministics),
                mode="FAST_COMPILE",
                on_unused_input="ignore",
            )
            values = evaluate(*(point[var.name] for var in model.value_vars))
            for var, value in zip(model.deterministics, values, strict=True):
                rec.put(f"{prefix}/FAST_COMPILE/det/{var.name}", value)


def datagen(rec: Recording) -> None:
    generator = pg.DataGenerator(rec.prior("datagen", **BASE))
    for i, batch in enumerate(generator.iter_batches(12, 6, seed=3)):
        rec.corpus(f"datagen/batch{i}", batch)


def _benchmark_cells(cfg: Any) -> list[tuple[dict[str, np.ndarray], int]]:
    """Cell inputs and draw seeds of ``benchmarks/template_vs_per_world.py:make_workload``."""
    rng = np.random.default_rng(cfg.seed)
    cells = []
    for _ in range(cfg.n_cells):
        counts = [
            int(rng.integers(lo, hi + 1))
            for lo, hi in (
                cfg.n_treatments_active_range_effective,
                cfg.n_covariates_active_range_effective,
                cfg.n_latent_active_range_effective,
            )
        ]
        g = sample_g_additive(rng, cfg, cfg.layout, *counts)
        structure = world_model.sample_structure(_slice_g_active(g, *counts), cfg, rng)
        cells.append((build_cell_inputs(cfg, g, g, structure), int(rng.integers(2**31 - 1))))
    return cells


def template(rec: Recording) -> None:
    for name, fields in TEMPLATE.items():
        cfg = rec.prior(f"template/{name}", **fields)
        cell_sets = {f"template/{name}": _benchmark_cells(cfg)}
        if name == STRUCTURES_TEMPLATE:
            structures = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))
            cell_sets[f"template/{name}/sample_cell_structures"] = [
                (inputs, STRUCTURES_SEED + i) for i, inputs in enumerate(structures)
            ]
        first_inputs = cell_sets[f"template/{name}"][0][0]
        model, _outputs, _params = build_world_model_template(cfg, first_inputs, cfg.n_time_steps)
        draw = compile_template_draw_fn(model, CORPUS_NAMES)
        for set_prefix, cells in cell_sets.items():
            for i, (inputs, seed) in enumerate(cells):
                prefix = f"{set_prefix}/cell{i}"
                rec.put(f"{prefix}/seed", seed)
                rec.record(f"{prefix}/inputs", inputs)
                rec.record(f"{prefix}/draws", draw(inputs, seed=seed, draws=TEMPLATE_DRAWS))


RUNNERS: dict[str, Callable[[Recording], None]] = {
    "ordinary": ordinary,
    "scm": scm,
    "oracle": oracle,
    "datagen": datagen,
    "template": template,
}


def _versions() -> dict[str, str]:
    versions = {"python": sys.version, "pymc_generator": pg.__version__}
    for distribution in ("numpy", "scipy", "pytensor", "pymc", "pymc-marketing", "numba"):
        versions[distribution] = metadata.version(distribution)
    return versions


def _loaded_package_files() -> dict[str, str | None]:
    return {
        name: str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None
        for name, module in sorted(sys.modules.items())
        if name == "pymc_generator" or name.startswith("pymc_generator.")
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="output .npz; the sidecar is written beside it")
    parser.add_argument("--sections", default=",".join(SECTIONS), help="comma-separated subset")
    args = parser.parse_args(argv)
    requested = args.sections.split(",")
    unknown = sorted(set(requested) - set(SECTIONS))
    if unknown:
        parser.error(f"unknown sections {unknown}; choose from {list(SECTIONS)}")

    rec = Recording()
    timings = {}
    # Canonical order, whatever the request order: oracle reuses the scm worlds.
    for section in SECTIONS:
        if section in requested:
            start = time.perf_counter()
            RUNNERS[section](rec)
            timings[section] = round(time.perf_counter() - start, 1)

    np.savez(args.out, **rec.arrays)
    sidecar = {
        "modules": _loaded_package_files(),
        "versions": _versions(),
        "sections": timings,
        "calls": rec.calls,
        "skipped": rec.skipped,
        "features_off": rec.features_off,
        "scalars": rec.scalars,
    }
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
