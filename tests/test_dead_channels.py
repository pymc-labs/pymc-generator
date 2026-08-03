"""The dead-channel floor: a per-cell negative class for the C->Y signal.

A *dead* channel is an ACTIVE channel with no direct C->Y arrow — its spend is
observed and its true contribution is exactly zero. It is the negative class for
any consumer learning "which channels affect sales", so a corpus whose cells all
happen to be fully live teaches only the positive case.

``edge_budget["cy"]`` cannot express that: it is an absolute arrow count clamped
to the eligible slots, so a cell drawing few active channels can have every one
of them live (``cy=(2, 10)`` with ``n_treatments_active=2`` => 2 live, 0 dead).
``min_dead_channels`` caps the live count at ``n_treatments_active - min_dead_channels``
instead, which decouples the live-channel range from the active-count range.

Mechanics are exercised through ``sample_g_additive`` (no graph compile) for
speed; one end-to-end corpus goes through the factory.
"""

from __future__ import annotations

import numpy as np
import pytest

from prior_generator import make_scm_prior, sample_prior_predictive
from prior_generator.sampler import SCMPrior, _sample_g, sample_g_additive

# The recipe that motivated the knob: wide active range, wide cy range,
# slots. Without a floor, every cell whose cy draw reaches its active count is
# fully live.
WIDE_ACTIVE = (2, 10)
WIDE_CY = {"cy": (1, 10), "zc": (0, 6), "cc": (0, 4), "dc": (0, 6)}


def _cfg(*, min_dead: int, active: tuple[int, int] = WIDE_ACTIVE, **kw) -> SCMPrior:
    return make_scm_prior(
        n_treatments=10,
        n_covariates=6,
        n_latent=3,
        n_treatments_active_range=active,
        edge_budget=kw.pop("edge_budget", WIDE_CY),
        min_dead_channels=min_dead,
        n_time_steps=52,
        n_cells=2,
        draws_per_cell=1,
        seed=0,
        **kw,
    )


def _cells(cfg: SCMPrior, n: int, seed: int = 0) -> list[np.ndarray]:
    """Draw ``n`` cells, each with its own active count, and return the live cy blocks."""
    rng = np.random.default_rng(seed)
    lo, hi = cfg.n_treatments_active_range_effective
    out = []
    for _ in range(n):
        k = int(rng.integers(lo, hi + 1))
        g = sample_g_additive(
            rng, cfg, cfg.layout, n_treatments_active=k, n_covariates_active=3, n_latent_active=2
        )
        out.append(g["g_cy"][:k])
        assert not g["g_cy"][k:].any(), "padding slots must never carry a C->Y edge"
    return out


def test_unfloored_wide_ranges_collapse_the_negative_class():
    """The failure the floor exists to prevent — kept as the regression baseline."""
    cells = _cells(_cfg(min_dead=0), 200)
    assert sum(1 for cy in cells if cy.all()) > 0


def test_floor_leaves_a_dead_channel_in_every_cell():
    for cy in _cells(_cfg(min_dead=1), 400):
        n_live = int(cy.sum())
        assert 1 <= n_live <= cy.size - 1, f"{n_live} live of {cy.size} active"


def test_floor_two_leaves_two_dead_channels():
    for cy in _cells(_cfg(min_dead=2, active=(3, 10)), 400):
        assert cy.size - int(cy.sum()) >= 2


def test_floor_keeps_the_live_range_wide():
    """The point of the knob: live counts still span 1..n_treatments_active-1 in ONE config.

    Chunking the corpus by active count (per-chunk ``cy_high <= active_low - 1``)
    buys the same guarantee only across chunks; here a single config does it.
    """
    live = {int(cy.sum()) for cy in _cells(_cfg(min_dead=1), 400)}
    assert live == set(range(1, 10))


def test_dead_channels_are_not_a_slot_position():
    """Slot index must not predict the label, or the negative class is trivial."""
    dead_slots: set[int] = set()
    live_slots: set[int] = set()
    for cy in _cells(_cfg(min_dead=1), 400):
        dead_slots |= set(np.flatnonzero(cy == 0).tolist())
        live_slots |= set(np.flatnonzero(cy == 1).tolist())
    assert dead_slots == set(range(10))
    assert live_slots == set(range(10))


def test_floor_binds_on_the_bernoulli_path_too():
    """No cy budget: surplus live channels are demoted after the per-slot draw."""
    cfg = _cfg(
        min_dead=4,
        active=(10, 10),
        edge_budget=None,
        edge_rate_overrides={"cy": 1.0},  # every slot draws live
    )
    for cy in _cells(cfg, 20):
        assert int(cy.sum()) == 6


def test_floor_never_starves_the_last_live_channel():
    """``n_treatments_active`` below the floor still yields one live channel, not zero."""
    cfg = _cfg(min_dead=1, active=(2, 10))
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = sample_g_additive(
            rng, cfg, cfg.layout, n_treatments_active=1, n_covariates_active=3, n_latent_active=2
        )
        assert int(g["g_cy"].sum()) == 1


def test_default_floor_is_inert():
    """Zero floor must not perturb the draw or the RNG stream."""
    cfg = _cfg(min_dead=0)
    assert cfg.min_dead_channels == 0
    floored = _sample_g(
        np.random.default_rng(3), cfg.layout, 7, 3, 2, budget=cfg.edge_budget, min_dead=0
    )
    legacy = _sample_g(np.random.default_rng(3), cfg.layout, 7, 3, 2, budget=cfg.edge_budget)
    for key, value in legacy.items():
        assert np.array_equal(floored[key], value), key


@pytest.mark.parametrize("min_dead", [-1, 1.0, True, np.float64(2.0)])
def test_validate_rejects_non_integer_floors(min_dead):
    with pytest.raises(ValueError, match="min_dead_channels"):
        SCMPrior(min_dead_channels=min_dead).validate()


@pytest.mark.parametrize("active", [(2, 10), (3, 3)])
def test_validate_rejects_an_unsatisfiable_floor(active):
    """A floor that the smallest drawable cell cannot honour is a config error."""
    with pytest.raises(ValueError, match="min_dead_channels"):
        SCMPrior(
            n_treatments=10,
            n_treatments_active_range=active,
            min_dead_channels=active[0],
        ).validate()


def test_validate_accepts_the_largest_satisfiable_floor():
    SCMPrior(
        n_treatments=10,
        n_treatments_active_range=(4, 10),
        min_dead_channels=3,
    ).validate()


def test_corpus_carries_a_dead_channel_per_task_end_to_end():
    cfg = make_scm_prior(
        n_treatments=6,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(2, 6),
        edge_budget={"cy": (1, 6)},
        min_dead_channels=1,
        n_time_steps=52,
        n_cells=4,
        draws_per_cell=2,
        seed=0,
    )
    corpus = sample_prior_predictive(cfg)
    cy = corpus["g"][:, cfg.layout.slices["cy"]]
    n_live = cy.sum(axis=1)
    n_active = corpus["n_treatments_active"]
    assert np.all(n_live >= 1)
    assert np.all(n_live <= n_active - 1)
    # The dead channels are inside the active range, not padding.
    assert np.array_equal(cy, cy * corpus["treatment_active_mask"])
    assert corpus["diagnostics"]["min_dead_channels"] == 1


def test_dead_channels_have_exactly_zero_contribution():
    """The negative class must be a true zero target, not a small one."""
    cfg = make_scm_prior(
        n_treatments=5,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(2, 5),
        edge_budget={"cy": (1, 5)},
        min_dead_channels=1,
        n_time_steps=52,
        n_cells=3,
        draws_per_cell=1,
        seed=0,
    )
    corpus = sample_prior_predictive(cfg)
    cy = corpus["g"][:, cfg.layout.slices["cy"]]
    dead = (cy == 0) & (corpus["treatment_active_mask"] == 1)
    assert dead.any()
    for task, channel in zip(*np.nonzero(dead)):
        assert np.all(corpus["contributions_raw"][task, :, channel] == 0.0)
