"""Original compositional study prior, with 208 reported weeks."""

from dataclasses import dataclass

import numpy as np

from pymc_generator import world_model as wm
from pymc_generator import world_model_template as wt
from pymc_generator.active_counts import draw_active_counts
from pymc_generator.presets import make_scm_prior
from pymc_generator.sampler import (
    CARRYOVER_FAMILY_KEYS,
    SATURATION_FAMILY_KEYS,
    _slice_g_active,
    sample_g_additive,
)
from pymc_generator.slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from pymc_generator.trajectories import structural_key

from .helpers import clone_function


@dataclass
class CellWorkload:
    g: dict
    structure: dict
    template_inputs: dict
    seed: int


def config(n, seed):
    # Keep the original composable prior intact. Only dimensions, structure
    # budgets and the explicitly requested world-shared uniform family policy
    # differ. Default l_max=8 and its 8-week burn-in are retained.
    cfg = make_scm_prior(
        trajectories="composable",
        n_treatments=15,
        n_covariates=10,
        n_latent=1,
        n_time_steps=208,
        n_cells=n,
        draws_per_cell=1,
        seed=seed,
        n_treatments_active_range=(5, 15),
        n_covariates_active_range=(3, 10),
        edge_budget={
            "cy": (1, 10),
            "zy": (1, 10),
            "dc": 0,
            "dz": 0,
            "dy": 0,
            "zc": (1, 4),
            "cc": (1, 4),
            "zz": (1, 4),
        },
        saturation_family_probs={
            k: 1 / len(SATURATION_FAMILY_KEYS) for k in SATURATION_FAMILY_KEYS
        },
        carryover_family_probs={k: 1 / len(CARRYOVER_FAMILY_KEYS) for k in CARRYOVER_FAMILY_KEYS},
    )
    expected = {
        "treatment": (0.9, 0.7, 0.15, 0.1, 0.2, 0.2, 0.3, 0.3),
        "covariate": (0.9, 0.6, 0.1, 0.1, 0.15, 0.3, 0.35, 0.3),
    }
    for role, values in expected.items():
        assert (
            tuple(cfg.trajectory_inclusion_probs()[role][c] for c in TRAJECTORY_COMPONENTS)
            == values
        )
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
            return {
                structural_key(role, c): np.zeros(n, dtype=bool)
                for role, n in zip(TRAJECTORY_INPUTS, (n_treatments, n_covariates))
                for c in TRAJECTORY_COMPONENTS
            }

        sampler = clone_function(wm.sample_structure, sample_component_flags=no_flags)
    structural = sampler(active, cfg, graph_rng)
    family_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, ident, 400]))
    for name, keys in [
        ("sat_family", SATURATION_FAMILY_KEYS),
        ("carryover_family", CARRYOVER_FAMILY_KEYS),
    ]:
        structural[name][:] = family_rng.integers(0, len(keys))
    seed = int(np.random.SeedSequence([cfg.seed, ident, 300]).generate_state(1)[0])
    return CellWorkload(active, structural, wt.build_cell_inputs(cfg, g, g, structural), seed)
