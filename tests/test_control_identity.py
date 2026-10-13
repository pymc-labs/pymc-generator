"""A control's outcome contribution is the literal ``(g_zy * rho_zy) * Z`` on the reported Z (#28).

Under the default ``"intercept"`` floor scope every covariate column is that
product, multiplied in that order, in float64 — with or without the composable
trajectory components, signed or reference-derived ``rho_zy``, and on both the
ordinary and the template path. Where the reported covariate is exactly zero
(a gated-off week), the column is zero too but its sign is unspecified: the
compiled graph moves ``g_zy * rho_zy`` inside the gate's switch, whose off branch
is the constant ``+0.0``, while the NumPy product keeps ``sign(rho)``.

A zeroing probe then binds the world's own primitive draws, removes every other
outcome term and sets ``rho_zy`` to a unit vector: the outcome must be the
reported covariate itself, byte for byte.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytest
from pytensor.graph.replace import clone_replace

import pymc_generator as pg
import pymc_generator.world_model as world_model
from pymc_generator.sampler import _ADDITIVE_OUT_NAMES, _CORPUS_PARAM_NAMES, _CORPUS_SHOCK_NAMES
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from pymc_generator.world_model_template import (
    build_world_model_template,
    compile_template_draw_fn,
    sample_cell_structures,
)

CORPUS_NAMES = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES

_BASE: dict[str, Any] = {
    "n_treatments": 3,
    "n_covariates": 2,
    "n_latent": 1,
    "n_time_steps": 40,
    "n_cells": 3,
    "draws_per_cell": 2,
}
_SHORT_GATES = {
    "onset_frac_range": (0.1, 0.2),
    "flighting_period_weeks_range": (2, 3),
    "flighting_duty_range": (0.5, 0.8),
}


def _components(*roles: str) -> dict[str, Any]:
    """Every composable component on every input of ``roles``, with gates that fit 40 weeks."""
    settings: dict[str, Any] = {}
    for role in roles:
        settings |= {f"{role}_{c}_inclusion_prob": 1.0 for c in TRAJECTORY_COMPONENTS}
        settings |= {f"{role}_{key}": value for key, value in _SHORT_GATES.items()}
    return settings


_SIGNED_RHO = {"zy_coeff_range": (-0.4, 0.4)}
_REFERENCE = {
    "covariate_reference_contribution_range": (-0.8, 0.5),
    "covariate_reference_scale": 1.75,
}
CONFIGS: dict[str, dict[str, Any]] = {
    "no_components": {},
    "all_covariate_components": _components("covariate"),
    "all_components": _components(*TRAJECTORY_INPUTS),
    "signed_rho": _SIGNED_RHO,
    "covariate_reference": _REFERENCE,
    "intercept_floor": {"baseline_floor": 0.0},
    "signed_rho_covariate_components": _SIGNED_RHO | _components("covariate"),
    "covariate_reference_covariate_components": _REFERENCE | _components("covariate"),
}
#: Configurations with a gate on every covariate: some reported weeks are exactly 0.
GATED = {
    name for name, overrides in CONFIGS.items() if "covariate_onset_inclusion_prob" in overrides
}


def _cfg(name: str, **extra: Any):
    return pg.make_scm_prior(**_BASE, **CONFIGS[name], **extra)


def _assert_same_bits(actual, expected, where: str) -> None:
    """Exact float64 bits, signed zeros included, reporting which elements differ."""
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual, dtype=np.float64).view(np.uint64),
        np.ascontiguousarray(expected, dtype=np.float64).view(np.uint64),
        err_msg=where,
    )


def _check_product(cc, g_zy, rho, z, counts, where: str) -> None:
    """``cc == (g_zy * rho) * z`` bit for bit where ``z != 0``; ``cc == 0`` where ``z == 0``."""
    nonzero = z != 0.0
    # A zero week's contribution has an unspecified sign: compare it as +0.0, so a
    # failure still reports the week itself.
    actual = np.where(nonzero, cc, np.abs(cc))
    _assert_same_bits(actual, np.where(nonzero, (g_zy * rho) * z, 0.0), where)
    if g_zy:
        counts["live"] += int(nonzero.sum())
        counts["gated"] += int((~nonzero).sum())
    else:
        counts["absent"] += int(nonzero.sum())


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def corpus_draws(request):
    """Every raw float64 candidate batch a corpus draws, with ``rho_zy`` and its cell's ``g_zy``.

    ``param_rho_zy`` is requested ahead of the corpus's own names, with those names
    as the stream reference: ``rho_zy`` already feeds ``covariate_contribution``, so
    no stream moves, and the sampler sees exactly the arrays it asked for.
    """
    original_build, original_draw = world_model.build_world_model, world_model.draw_worlds
    g_zy: dict[int, np.ndarray] = {}
    batches: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []

    def build(g, *args, **kwargs):
        built = original_build(g, *args, **kwargs)
        g_zy[id(built[0])] = np.asarray(g["g_zy"], dtype=np.float64)
        return built

    def draw(model, names, seed, draws=1, mode="FAST_COMPILE", *, rng_reference_names=None):
        names = tuple(names)
        drawn = original_draw(
            model,
            ("param_rho_zy", *names),
            seed,
            draws,
            mode,
            rng_reference_names=rng_reference_names or names,
        )
        batches.append((g_zy[id(model)], drawn))
        return {name: drawn[name] for name in names}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(world_model, "build_world_model", build)
        patch.setattr(world_model, "draw_worlds", draw)
        pg.sample_prior_predictive(_cfg(request.param, edge_budget={"zy": (1, 2)}))
    return request.param, batches


def test_corpus_candidates_keep_the_literal_control_product(corpus_draws):
    """An absent edge is folded out of the ordinary graph: its column is zero, of either sign."""
    name, draws = corpus_draws
    counts = {"live": 0, "gated": 0, "absent": 0}
    for call, (g_zy, drawn) in enumerate(draws):
        for b in range(drawn["covariates"].shape[0]):
            for m, edge in enumerate(g_zy):
                where = f"draw call {call}, candidate {b}, covariate {m}"
                cc = drawn["covariate_contribution"][b, :, m]
                if not edge:
                    assert np.all(cc == 0.0), where
                    counts["absent"] += cc.size
                    continue
                z = drawn["covariates"][b, :, m]
                _check_product(cc, edge, drawn["param_rho_zy"][b, m], z, counts, where)
    assert counts["live"] > 0
    assert counts["absent"] > 0
    if name in GATED:
        assert counts["gated"] > 0


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def template_draws(request):
    """Template draws of the config's cells, with ``rho_zy``; one case also pads covariates."""
    extra: dict[str, Any] = {
        "confounding_strength_range": (0.3, 0.3),
        "edge_budget": {"zy": (1, 2)},
    }
    if request.param == "no_components":
        extra["n_covariates_active_range"] = (1, 2)
    cfg = _cfg(request.param, **extra)
    cells = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))
    model, _, _ = build_world_model_template(cfg, cells[0], cfg.n_time_steps)
    draw = compile_template_draw_fn(model, (*CORPUS_NAMES, "param_rho_zy"))
    return request.param, [
        (cell, draw(cell, seed=10_000 + i, draws=2)) for i, cell in enumerate(cells)
    ]


def test_template_draws_keep_the_literal_control_product(template_draws):
    """Absent edges follow the same rule: ``(0 * rho) * z`` is a zero signed like ``rho * z``."""
    name, results = template_draws
    counts = {"live": 0, "gated": 0, "absent": 0, "padded": 0}
    for i, (cell, drawn) in enumerate(results):
        active = np.asarray(cell["active_covariate"]) != 0
        for b in range(drawn["covariates"].shape[0]):
            for m, edge in enumerate(np.asarray(cell["g_zy"], dtype=np.float64)):
                where = f"cell {i}, draw {b}, covariate {m}"
                z, cc = drawn["covariates"][b, :, m], drawn["covariate_contribution"][b, :, m]
                if not active[m]:
                    assert np.all(cc == 0.0), where
                    counts["padded"] += cc.size
                    continue
                _check_product(cc, edge, drawn["param_rho_zy"][b, m], z, counts, where)
    assert counts["live"] > 0
    assert counts["absent"] > 0
    if name in GATED:
        assert counts["gated"] > 0
    if name == "no_components":
        assert counts["padded"] > 0


#: Primitives of every non-covariate outcome term: intercept walk, D -> Y,
#: observation noise and the treatment responses.
_ZEROED = ("rw_b_mean", "rw_b_std_rel", "delta_dy", "rw_y_std_rel", "beta")
_PROBE_IDS = (
    "no_components",
    "all_covariate_components",
    "all_components",
    "signed_rho",
    "intercept_floor",
    "signed_rho_covariate_components",
    "covariate_reference_covariate_components",
)


def _unit_control(primitives: dict[str, np.ndarray], m: int, cfg) -> dict[str, np.ndarray]:
    """The same primitives with every other outcome term off and ``rho_zy = e_m``."""
    zeroed = {name: np.array(value, copy=True) for name, value in primitives.items()}
    for name in _ZEROED:
        zeroed[name] = np.zeros_like(zeroed[name])
    unit = np.zeros_like(zeroed.get("rho_zy", zeroed.get("covariate_reference_contribution")))
    unit[m] = 1.0
    if "covariate_reference_contribution" in zeroed:
        # rho_zy = target / scale; 1.75 * fl(1 / 1.75) == 1 exactly.
        zeroed["covariate_reference_contribution"] = unit * cfg.covariate_reference_scale
    else:
        zeroed["rho_zy"] = unit
    return zeroed


def _assert_outcome_is_the_reported_control(outputs, m: int) -> None:
    z = outputs["covariates"][:, m]
    _assert_same_bits(outputs["baseline"], z, f"baseline vs covariate {m}")
    _assert_same_bits(outputs["outcome"], z, f"outcome vs covariate {m}")


#: The second seed repeats every check on another world; it runs in the slow job.
_SEEDS = (0, pytest.param(1, marks=pytest.mark.slow))


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("name", _PROBE_IDS)
def test_ordinary_outcome_reads_the_reported_covariates(name, seed):
    cfg = _cfg(name, edge_budget={"zy": (2, 2)})
    world = pg.sample_scm(cfg, seed=seed)
    g_zy = np.asarray(world.g["g_zy"], dtype=np.float64)
    counts = {"live": 0, "gated": 0, "absent": 0}
    for m, edge in enumerate(g_zy):
        _check_product(
            world.data["covariate_contribution"][:, m],
            edge,
            world.params["rho_zy"][m],
            world.data["covariates"][:, m],
            counts,
            f"covariate {m}",
        )
    assert counts["live"] > 0
    if name in GATED:
        assert counts["gated"] > 0
    for m in range(len(g_zy)):
        primitives = _unit_control(world.primitive_parameters, m, cfg)
        replayed = dataclasses.replace(world, _primitive_parameters=primitives).replay()
        # Zeroing outcome-side primitives leaves the reported covariates untouched.
        _assert_same_bits(replayed["covariates"], world.data["covariates"], f"covariates, m={m}")
        _assert_outcome_is_the_reported_control(replayed, m)


@pytest.fixture(scope="module", params=_PROBE_IDS)
def bound_template(request):
    """A compiled template plus its forward graph with every free RV bound as an input."""
    cfg = _cfg(
        request.param,
        edge_budget={"zy": (2, 2)},
        confounding_strength_range=(0.3, 0.3),
        n_covariates_active_range=(2, 2),
    )
    cell = sample_cell_structures(cfg, np.random.default_rng(cfg.seed))[0]
    model, _, _ = build_world_model_template(cfg, cell, cfg.n_time_steps)
    free = tuple(rv.name for rv in model.free_RVs)
    draw = compile_template_draw_fn(model, tuple(dict.fromkeys((*CORPUS_NAMES, *free))))
    given = {rv: rv.type(name=f"given_{rv.name}") for rv in model.free_RVs}
    with model:
        pm.set_data(cell)
    outputs = clone_replace(
        [model[name] for name in ("baseline", "outcome", "covariates")],
        replace=given,
        rebuild_strict=False,
    )
    forward = pytensor.function(
        list(given.values()), outputs, on_unused_input="ignore", mode="FAST_COMPILE"
    )
    return cfg, model, cell, draw, forward


@pytest.mark.parametrize("seed", _SEEDS)
def test_template_outcome_reads_the_reported_covariates(bound_template, seed):
    cfg, model, cell, draw, forward = bound_template
    drawn = draw(cell, seed=seed)
    primitives = {rv.name: np.asarray(drawn[rv.name][0], dtype=rv.dtype) for rv in model.free_RVs}
    assert np.asarray(cell["g_zy"]).all()
    for m in range(cfg.n_covariates):
        values = _unit_control(primitives, m, cfg)
        baseline, outcome, covariates = forward(*(values[rv.name] for rv in model.free_RVs))
        _assert_same_bits(covariates, drawn["covariates"][0], f"covariates, m={m}")
        _assert_outcome_is_the_reported_control(
            {"baseline": baseline, "outcome": outcome, "covariates": covariates}, m
        )
