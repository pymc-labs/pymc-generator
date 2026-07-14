"""CorpusConfig validation, the preset factory, and the deprecation policy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.sampler import CorpusConfig
from prior_generator.slots import EDGE_TYPES_EXTENDED


def test_factory_pins_additive_schema():
    cfg = pg.make_l1_additive_cfg(K_max=8, M_max=4, J_max=3)
    assert cfg.rung == "L1_additive"
    assert (cfg.K_max_effective, cfg.M_max_effective, cfg.J_max_effective) == (8, 4, 3)
    assert cfg.layout.edge_types == EDGE_TYPES_EXTENDED
    # default active ranges pin every node active
    assert cfg.K_active_range == (8, 8)
    assert cfg.M_active_range == (4, 4)
    assert cfg.J_active_range == (3, 3)


def test_factory_diverse_texture_defaults():
    cfg = pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, l_max=8)
    assert cfg.channel_hf_sigma_range[1] > 0
    assert cfg.channel_pulse_prob_range[1] > 0
    assert cfg.rw_channel_std_range is not None
    assert cfg.adstock_burn_in == cfg.l_max


def test_factory_linear_nonlinearity():
    cfg = pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, nonlinearity="linear")
    assert cfg.adstock_family_probs == (1.0, 0.0, 0.0)
    assert cfg.saturation_family_probs == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_factory_overrides_win():
    cfg = pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, T=200, spend_cv_floor=0.2)
    assert cfg.T == 200
    assert cfg.spend_cv_floor == 0.2


def test_legacy_texture_rejected():
    with pytest.raises(ValueError, match="texture"):
        pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, texture="legacy")


def test_bad_nonlinearity_rejected():
    with pytest.raises(ValueError, match="nonlinearity"):
        pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, nonlinearity="quadratic")


@pytest.mark.parametrize("rung", ["L0", "L1", "nonsense"])
def test_non_additive_rung_rejected(rung):
    with pytest.raises(ValueError, match="L1_additive"):
        CorpusConfig(rung=rung).validate()


def test_edge_budget_unknown_key_rejected():
    with pytest.raises(ValueError, match="edge_budget"):
        pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, edge_budget={"xy": 3})


@pytest.mark.parametrize("spec", [-1, (2, 1), (1, 2, 3), 1.5, True])
def test_edge_budget_bad_spec_rejected(spec):
    with pytest.raises(ValueError):
        pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, edge_budget={"cc": spec})


# --- deprecation / steering policy -----------------------------------------


def test_diverse_texture_does_not_warn():
    cfg = pg.make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, T=32, n_cells=2, draws_per_cell=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pg.generate_corpus(cfg)  # must not raise


def test_flat_texture_warns():
    # A hand-built config with the texture disabled (hf & pulse ranges zero) is
    # the deprecated smooth-walk-only prior and must warn.
    cfg = CorpusConfig(
        T=32,
        K=4,
        M=2,
        J=1,
        K_active_range=(4, 4),
        M_active_range=(2, 2),
        J_active_range=(1, 1),
        n_cells=2,
        draws_per_cell=1,
    )
    with pytest.warns(FutureWarning, match="texture"):
        pg.generate_corpus(cfg)
