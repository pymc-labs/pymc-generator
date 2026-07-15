"""The PyMC-model draw path: priors are distributions, worlds are pm.draws.

Locks the correctness of the RV engine at the low level (build_world_model +
draw_worlds) — the exact interventional decomposition, positivity, seed
determinism, and batched draws — independently of the high-level wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from prior_generator import make_scm_prior
from prior_generator.sampler import _slice_g_active, sample_g_additive
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure


@pytest.fixture(scope="module")
def built():
    cfg = make_scm_prior(
        K_max=4,
        M_max=2,
        J_max=1,
        T=48,
        edge_budget={"cy": (4, 4), "cc": (1, 2), "zc": (1, 2), "dc": (1, 2)},
    )
    rng = np.random.default_rng(0)
    g = sample_g_additive(rng, cfg, cfg.layout)
    g_act = _slice_g_active(g, 4, 2, 1)
    structural = sample_structure(g_act, cfg, rng)
    model, out_names, _param_names = build_world_model(g_act, cfg, structural, cfg.T)
    return model, out_names


def test_model_is_pm_model_with_priors_and_outputs(built):
    model, out_names = built
    # continuous priors + noise are real RVs; every graph output is registered
    assert len(model.free_RVs) > 10
    assert len(out_names) == 13
    assert "sales" in out_names and "indirect_effects_by_source" in out_names


def test_pm_draw_preserves_additive_identity(built):
    model, out_names = built
    d = {k: v[0] for k, v in draw_worlds(model, out_names, seed=123, draws=1).items()}
    s = d["sales"]
    identity = np.abs(s - (d["baseline"] + d["contributions"].sum(1) + d["indirect_effects"])).max()
    telescoping = np.abs(d["indirect_effects_by_source"].sum(1) - d["indirect_effects"]).max()
    full = np.abs(
        d["baseline_intrinsic"]
        + d["confounder_contribution"].sum(1)
        + d["control_contribution"].sum(1)
        + d["contributions"].sum(1)
        + d["indirect_effects_by_source"].sum(1)
        - s
    ).max()
    assert identity < 1e-9
    assert telescoping < 1e-9
    assert full < 1e-9


def test_channels_positive(built):
    model, out_names = built
    d = draw_worlds(model, out_names, seed=1, draws=1)
    assert (d["channels"] >= 0).all()


def test_seed_determinism(built):
    model, out_names = built
    a = draw_worlds(model, out_names, seed=42, draws=1)
    b = draw_worlds(model, out_names, seed=42, draws=1)
    c = draw_worlds(model, out_names, seed=43, draws=1)
    assert np.array_equal(a["sales"], b["sales"])
    assert not np.array_equal(a["sales"], c["sales"])


def test_batched_draws_have_leading_axis(built):
    model, out_names = built
    d = draw_worlds(model, out_names, seed=7, draws=5)
    assert d["sales"].shape == (5, 48)
    assert d["contributions"].shape == (5, 48, 4)


def test_diverse_texture_gives_nonflat_targets(built):
    model, out_names = built
    contrib = draw_worlds(model, out_names, seed=3, draws=1)["contributions"][0]  # (T, K)
    cv = contrib.std(0) / (np.abs(contrib.mean(0)) + 1e-9)
    assert cv.max() > 0.05


def test_single_draw_has_leading_axis(built):
    # regression: draw_worlds always keeps a leading draws axis, so
    # sample_scm(max_eps_draws=1) can index candidate 0 without hitting time.
    model, out_names = built
    d = draw_worlds(model, out_names, seed=9, draws=1)
    assert d["sales"].shape == (1, 48)
    assert d["contributions"].shape == (1, 48, 4)
