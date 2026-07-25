"""Symbolic per-draw channel-shock scheduling (not trajectory application)."""

from __future__ import annotations

import numpy as np
import pytest

from prior_generator import make_scm_prior
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure


def _built(*, T=12, K=3, S=0, length=(1, 1), level=(0.0, 0.0), burn_in=0, direct=None):
    cfg = make_scm_prior(
        n_treatments=K,
        n_covariates=1,
        n_latent=1,
        T=T,
        adstock_burn_in=burn_in,
        n_channel_shocks=S,
        channel_shock_length_range=length,
        channel_shock_level_range=level,
        edge_budget={"cy": (K, K)},
    )
    if direct is None:
        direct = np.ones(K, dtype=int)
    g = {
        "g_cy": np.asarray(direct, dtype=int),
        "g_dc": np.zeros((1, K), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, K), dtype=int),
        "g_cc": np.zeros((K, K), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    structural = sample_structure(g, cfg, np.random.default_rng(4))
    return (*build_world_model(g, cfg, structural, T), cfg, g)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_channel_shocks": True},
        {"n_channel_shocks": -1},
        {"channel_shock_length_range": (True, 1)},
        {"channel_shock_length_range": (2, 1)},
        {"channel_shock_length_range": (1, 13)},
        {"channel_shock_level_range": (-1.0, 0.0)},
        {"channel_shock_level_range": (1.0, 0.0)},
        {"channel_shock_level_range": (0.0, np.inf)},
        {"n_channel_shocks": 13},
        {"n_channel_shocks": 4, "channel_shock_length_range": (4, 4)},
    ],
)
def test_channel_shock_validation(kwargs):
    with pytest.raises(ValueError):
        make_scm_prior(n_treatments=3, n_covariates=1, n_latent=1, T=12, **kwargs)


def test_disabled_schedule_has_empty_tensors_and_no_shock_rvs():
    model, names, _, cfg, _ = _built()
    assert not any(rv.name.startswith("channel_shock") for rv in model.free_RVs)
    d = draw_worlds(model, names, seed=1)
    assert d["channel_shock_mask"].shape == (1, cfg.T, 3)
    assert d["channel_shock_mask"].sum() == 0
    for name in ("channel_shock_channel", "channel_shock_start", "channel_shock_length"):
        assert d[name].shape == (1, 0)


def test_schedule_slots_containment_levels_and_burn_in_offset():
    model, names, param_names, cfg, g = _built(
        T=10, S=3, length=(1, 3), level=(0.5, 1.5), burn_in=8
    )
    d = draw_worlds(model, names + param_names, seed=3, draws=8)
    selected_level = np.take_along_axis(
        d["param_channel_level"], d["channel_shock_channel"], axis=1
    )
    assert np.array_equal(
        d["channel_shock_level"], d["channel_shock_level_multiplier"] * selected_level
    )
    assert np.isin(d["channel_shock_channel"], np.flatnonzero(g["g_cy"])).all()
    for b in range(8):
        starts, lengths = d["channel_shock_start"][b], d["channel_shock_length"][b]
        for s, (start, length) in enumerate(zip(starts, lengths)):
            lo, hi = s * cfg.T // 3, (s + 1) * cfg.T // 3
            assert lo <= start and start + length <= hi
        assert d["channel_shock_mask"][b].sum() == lengths.sum()
        assert np.array_equal(d["channel_shock_mask_full"][b, 8:], d["channel_shock_mask"][b])


def test_exact_fill_uneven_slots_and_same_seed_reproducibility():
    model, names, _, _, _ = _built(T=10, S=5, length=(2, 2), level=(1.0, 1.0))
    a, b = draw_worlds(model, names, seed=9, draws=4), draw_worlds(model, names, seed=9, draws=4)
    assert np.array_equal(a["channel_shock_mask"], b["channel_shock_mask"])
    assert (a["channel_shock_mask"].sum(axis=(1, 2)) == 10).all()


def test_single_direct_channel_can_be_selected_repeatedly_and_batched_values_vary():
    model, names, _, _, _ = _built(T=12, S=4, length=(1, 2), level=(0.1, 0.9), direct=[0, 1, 0])
    d = draw_worlds(model, names, seed=12, draws=12)
    assert (d["channel_shock_channel"] == 1).all()
    assert np.unique(d["channel_shock_length"]).size > 1
    assert np.unique(d["channel_shock_level_multiplier"]).size > 1
