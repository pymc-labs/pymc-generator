"""Main's random-stream owner orders, pinned with every feature off (#28).

A seeded draw hands stream ``i`` of one spawned ``SeedSequence`` to the ``i``-th
RNG of its draw function's reseed list, so that list's random-variable order
decides every seeded value. ``tests/_rng_owner_baseline.py`` records the order on
main (ce068d5) for every carryover x saturation pair of the probe cell, in a
corpus cell and in a ``sample_scm`` world, for a shock corpus and for three
template layouts. These tests measure the same orders on this tree from the
draw names its real calls request (``tests/_rng_owners.py``) and require them
unchanged, the appended tail included.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pymc_generator import make_scm_prior, world_model
from pymc_generator.sampler import _CORPUS_NATURAL_NAMES, _CORPUS_TRAJECTORY_NAMES
from pymc_generator.slots import TRAJECTORY_INPUTS
from pymc_generator.trajectories import SCHEDULE_COMPONENTS
from pymc_generator.world_model_template import compile_template_draw_fn

_TESTS = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    """A helper module of this directory, loaded by path: it is not a test module."""
    spec = importlib.util.spec_from_file_location(name, _TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_owners = _load("_rng_owners")
_main = _load("_rng_owner_baseline")


def _schedule_probs(p: float) -> dict[str, float]:
    """Every gate and level component of both input types at inclusion probability ``p``."""
    return {f"{x}_{c}_inclusion_prob": p for x in TRAJECTORY_INPUTS for c in SCHEDULE_COMPONENTS}


#: Gate priors that keep every gate valid at a 16-week horizon (support prefix 8).
_SHORT_GATES: dict[str, Any] = {
    **{f"{x}_onset_frac_range": (0.1, 0.2) for x in TRAJECTORY_INPUTS},
    **{f"{x}_flighting_period_weeks_range": (2, 3) for x in TRAJECTORY_INPUTS},
    **{f"{x}_flighting_duty_range": (0.5, 0.8) for x in TRAJECTORY_INPUTS},
}


@pytest.fixture(scope="module")
def corpus_request():
    return _owners.corpus_request()


@pytest.fixture(scope="module")
def world_request():
    return _owners.world_request()


@pytest.mark.parametrize("pair", _owners.PAIRS)
def test_ordinary_corpus_rng_owners_match_main(corpus_request, pair):
    probe = make_scm_prior(**_owners.PROBE)
    assert _owners.probe_order(probe, pair, corpus_request) == _main.ORDINARY[pair]


@pytest.mark.parametrize("pair", _owners.PAIRS)
def test_single_world_rng_owners_match_main(world_request, pair):
    """``sample_scm`` draws more names than main; their RNGs may only append."""
    probe = make_scm_prior(**_owners.PROBE)
    assert _owners.probe_order(probe, pair, world_request) == _main.SINGLE_WORLD[pair]


def test_shock_corpus_rng_owners_match_main():
    request = _owners.corpus_request(n_treatment_shocks=1)
    shocked = make_scm_prior(**_owners.PROBE, n_treatment_shocks=1)
    assert _owners.probe_order(shocked, "weibull/hill", request) == _main.SHOCKS["weibull/hill"]


@pytest.mark.parametrize("name", tuple(_owners.TEMPLATES))
def test_template_rng_owners_match_main(name):
    """Every family and parent is wired behind data switches, so each sum's order counts."""
    assert _owners.template_order(name) == _main.TEMPLATE[name]


def test_inert_schedules_keep_the_single_world_rng_owners():
    """Admitted schedule components that no input carries keep the default world's streams.

    The world also draws the schedule audit outputs. They stay out of the stream
    reference, so the random variables only they reach would be appended rather
    than move the default ones.
    """
    inert = {**_schedule_probs(1e-9), **_SHORT_GATES}
    request = _owners.world_request(**inert)
    assert set(_CORPUS_TRAJECTORY_NAMES + _CORPUS_NATURAL_NAMES) <= set(request[0])
    order = _owners.probe_order(make_scm_prior(**_owners.PROBE, **inert), "weibull/hill", request)
    assert order == _main.SINGLE_WORLD["weibull/hill"]


def test_orders_are_measured_on_this_checkout(corpus_request, world_request):
    """Every loaded ``pymc_generator`` module is this checkout's, so no other tree is measured."""
    package = _TESTS.parent / "pymc_generator"
    files = _owners.package_files()
    assert "pymc_generator.world_model" in files
    assert {name: path for name, path in files.items() if not path.is_relative_to(package)} == {}


def _rv_names(model, rngs) -> tuple[str, ...]:
    owner = {rv.owner.inputs[0]: rv.name for rv in model.basic_RVs}
    return tuple(owner[rng] for rng in rngs)


@pytest.mark.parametrize("which", ("corpus", "world"))
def test_measured_order_is_the_list_draw_worlds_reseeds(
    which, corpus_request, world_request, monkeypatch
):
    """The orders above are measured, not assumed: they are what a real draw reseeds."""
    names, reference = corpus_request if which == "corpus" else world_request
    model = _owners.probe_model(make_scm_prior(**_owners.PROBE), "weibull", "hill")
    reseeded, execute = [], world_model._execute_draws

    def recording(draw_fn, ordered_rngs, **kwargs):
        reseeded.append(list(ordered_rngs))
        return execute(draw_fn, ordered_rngs, **kwargs)

    monkeypatch.setattr(world_model, "_execute_draws", recording)
    world_model.draw_worlds(model, names, 0, rng_reference_names=reference or None)
    (rngs,) = reseeded
    assert _rv_names(model, rngs) == _owners.reseeded_order(model, names, reference)


@pytest.mark.parametrize("name", tuple(_owners.TEMPLATES))
def test_measured_template_order_is_the_compiled_draw_functions(name):
    model = _owners.template_model(make_scm_prior(**_owners.TEMPLATES[name]))
    draw = compile_template_draw_fn(model, _owners.CORPUS_NAMES)
    assert _rv_names(model, draw.ordered_rngs) == _owners.template_order(name)
