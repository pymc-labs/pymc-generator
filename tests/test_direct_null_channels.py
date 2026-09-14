"""Direct-null channels have zero direct contribution but may remain feeders."""

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


def _cfg(*, min_no_direct: int, active: tuple[int, int] = WIDE_ACTIVE, **kw) -> SCMPrior:
    return make_scm_prior(
        n_treatments=10,
        n_covariates=6,
        n_latent=3,
        n_treatments_active_range=active,
        edge_budget=kw.pop("edge_budget", WIDE_CY),
        min_no_direct_effect_channels=min_no_direct,
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
    cells = _cells(_cfg(min_no_direct=0), 200)
    assert sum(1 for cy in cells if cy.all()) > 0


def test_floor_leaves_a_direct_null_channel_in_every_cell():
    for cy in _cells(_cfg(min_no_direct=1), 400):
        n_live = int(cy.sum())
        assert 1 <= n_live <= cy.size - 1, f"{n_live} live of {cy.size} active"


def test_floor_two_leaves_two_direct_null_channels():
    for cy in _cells(_cfg(min_no_direct=2, active=(3, 10)), 400):
        assert cy.size - int(cy.sum()) >= 2


def test_floor_keeps_the_live_range_wide():
    """The point of the knob: live counts still span 1..n_treatments_active-1 in ONE config.

    Chunking the corpus by active count (per-chunk ``cy_high <= active_low - 1``)
    buys the same guarantee only across chunks; here a single config does it.
    """
    live = {int(cy.sum()) for cy in _cells(_cfg(min_no_direct=1), 400)}
    assert live == set(range(1, 10))


def test_direct_null_channels_are_not_a_slot_position():
    """Slot index must not predict the label, or the negative class is trivial."""
    no_direct_slots: set[int] = set()
    live_slots: set[int] = set()
    for cy in _cells(_cfg(min_no_direct=1), 400):
        no_direct_slots |= set(np.flatnonzero(cy == 0).tolist())
        live_slots |= set(np.flatnonzero(cy == 1).tolist())
    assert no_direct_slots == set(range(10))
    assert live_slots == set(range(10))


def test_floor_binds_on_the_bernoulli_path_too():
    """No cy budget: surplus live channels are demoted after the per-slot draw."""
    cfg = _cfg(
        min_no_direct=4,
        active=(10, 10),
        edge_budget=None,
        edge_rate_overrides={"cy": 1.0},  # every slot draws live
    )
    for cy in _cells(cfg, 20):
        assert int(cy.sum()) == 6


def test_floor_never_starves_the_last_live_channel():
    """``n_treatments_active`` below the floor still yields one live channel, not zero."""
    cfg = _cfg(min_no_direct=1, active=(2, 10))
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = sample_g_additive(
            rng, cfg, cfg.layout, n_treatments_active=1, n_covariates_active=3, n_latent_active=2
        )
        assert int(g["g_cy"].sum()) == 1


def test_default_floor_is_inert():
    """Zero floor must not perturb the draw or the RNG stream."""
    cfg = _cfg(min_no_direct=0)
    assert cfg.min_no_direct_effect_channels == 0
    floored = _sample_g(
        np.random.default_rng(3), cfg.layout, 7, 3, 2, budget=cfg.edge_budget, min_no_direct=0
    )
    legacy = _sample_g(np.random.default_rng(3), cfg.layout, 7, 3, 2, budget=cfg.edge_budget)
    for key, value in legacy.items():
        assert np.array_equal(floored[key], value), key


@pytest.mark.parametrize("min_no_direct", [-1, 1.0, True, np.float64(2.0)])
def test_validate_rejects_non_integer_floors(min_no_direct):
    with pytest.raises(ValueError, match="min_no_direct_effect_channels"):
        SCMPrior(min_no_direct_effect_channels=min_no_direct).validate()


@pytest.mark.parametrize("active", [(2, 10), (3, 3)])
def test_validate_rejects_an_unsatisfiable_floor(active):
    """A floor that the smallest drawable cell cannot honour is a config error."""
    with pytest.raises(ValueError, match="min_no_direct_effect_channels"):
        SCMPrior(
            n_treatments=10,
            n_treatments_active_range=active,
            min_no_direct_effect_channels=active[0],
        ).validate()


def test_validate_accepts_the_largest_satisfiable_floor():
    SCMPrior(
        n_treatments=10,
        n_treatments_active_range=(4, 10),
        min_no_direct_effect_channels=3,
    ).validate()


def test_corpus_carries_a_direct_null_channel_per_task():
    cfg = make_scm_prior(
        n_treatments=6,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(2, 6),
        edge_budget={"cy": (1, 6)},
        min_no_direct_effect_channels=1,
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
    # Direct-null channels are active nodes, not padding.
    assert np.array_equal(cy, cy * corpus["treatment_active_mask"])
    assert corpus["diagnostics"]["min_no_direct_effect_channels"] == 1


def test_direct_null_channels_have_exactly_zero_direct_contribution():
    """The negative class must be a true zero target, not a small one."""
    cfg = make_scm_prior(
        n_treatments=5,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(2, 5),
        edge_budget={"cy": (1, 5)},
        min_no_direct_effect_channels=1,
        n_time_steps=52,
        n_cells=3,
        draws_per_cell=1,
        seed=0,
    )
    corpus = sample_prior_predictive(cfg)
    cy = corpus["g"][:, cfg.layout.slices["cy"]]
    no_direct = (cy == 0) & (corpus["treatment_active_mask"] == 1)
    assert no_direct.any()
    for task, channel in zip(*np.nonzero(no_direct)):
        assert np.all(corpus["contributions_raw"][task, :, channel] == 0.0)


def test_direct_null_floor_does_not_exclude_indirect_influence():
    from prior_generator import sample_scm
    from prior_generator.worlds import channel_role

    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=24,
        nonlinearity="linear",
        min_no_direct_effect_channels=1,
        edge_budget={
            "cy": (1, 1),
            "cc": (1, 1),
            "dy": (1, 1),
            "zy": (1, 1),
            "dc": 0,
            "zc": 0,
            "dz": 0,
            "zz": 0,
        },
    )
    world = sample_scm(cfg, seed=3, connect_all=True)
    assert channel_role(world.g, 0) == "feeder"
    assert np.all(world.data["contributions"][:, 0] == 0.0)
    assert np.any(world.data["indirect_effects_by_source"][:, 0] > 0.0)
    assert world.identity_error() < 1e-10
