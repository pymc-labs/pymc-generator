"""Per-edge-type arrow budgets ("pots") for the additive SCM.

A budget caps the arrow count for an edge type ("up to N"): the per-task count
is drawn uniformly in {0..N} (or {lo..hi} for a tuple; (N, N) for exactly N)
and scattered uniformly over the eligible node pairs. How they clump (one
source fanning out vs. one arrow each) is emergent. Each type's pot is
independent, and the count is capped at the number of eligible pairs.

The pot mechanics are exercised through ``sample_g_additive`` (no graph
compile) for speed; one end-to-end corpus smoke goes through the factory.
"""

from __future__ import annotations

import numpy as np
import pytest

from prior_generator import make_scm_prior, sample_prior_predictive
from prior_generator.sampler import SCMPrior, sample_g_additive


def _cfg(edge_budget=None, *, n_treatments=5, n_covariates=5, n_latent=2, **kw) -> SCMPrior:
    return SCMPrior(
        n_treatments=n_treatments,
        n_covariates=n_covariates,
        n_latent=n_latent,
        edge_budget=edge_budget,
        **kw,
    )


def _draw(cfg: SCMPrior, seed: int) -> dict[str, np.ndarray]:
    """One all-active extended DAG cell (K_active=M_active=J_active = max)."""
    return sample_g_additive(np.random.default_rng(seed), cfg, cfg.layout)


def test_int_budget_is_up_to():
    cfg = _cfg({"zc": 5})  # 5 controls x 5 channels = 25 eligible pairs
    counts = {int(_draw(cfg, s)["g_zc"].sum()) for s in range(120)}
    assert counts <= {0, 1, 2, 3, 4, 5}
    assert max(counts) == 5 and min(counts) == 0
    assert len(counts) > 1


def test_exact_count_via_equal_bounds():
    cfg = _cfg({"zc": (5, 5)})
    for seed in range(30):
        assert _draw(cfg, seed)["g_zc"].sum() == 5


def test_pot_capped_at_eligible_pairs():
    cfg = _cfg({"cc": 99}, n_treatments=3)  # strict-upper C->C eligible = 3
    counts = {int(_draw(cfg, s)["g_cc"].sum()) for s in range(30)}
    assert counts <= {0, 1, 2, 3}
    assert max(counts) == 3


def test_range_pot_varies_within_bounds():
    cfg = _cfg({"cc": (0, 3)}, n_treatments=6)
    counts = {int(_draw(cfg, s)["g_cc"].sum()) for s in range(60)}
    assert counts <= {0, 1, 2, 3}
    assert len(counts) > 1


def test_pots_are_independent():
    cfg = _cfg({"zc": (5, 5), "zb": (3, 3)})
    for seed in range(30):
        g = _draw(cfg, seed)
        assert g["g_zc"].sum() == 5
        assert g["g_zb"].sum() == 3


def test_budgeting_one_type_leaves_others_bernoulli():
    cfg = _cfg({"zc": 5})
    zb_counts = {int(_draw(cfg, s)["g_zb"].sum()) for s in range(40)}
    assert len(zb_counts) > 1
    assert all(_draw(cfg, s)["g_zc"].sum() <= 5 for s in range(40))


def test_fan_out_variety():
    cfg = _cfg({"zc": (5, 5)})
    zc = [_draw(cfg, s)["g_zc"] for s in range(100)]
    assert all(m.sum() == 5 for m in zc)
    max_row = max(int(m.sum(axis=1).max()) for m in zc)
    assert max_row >= 2
    assert not all(np.array_equal(m, zc[0]) for m in zc)


def test_cc_pot_is_strict_upper_triangular():
    cfg = _cfg({"cc": (4, 4)}, n_treatments=5)
    for seed in range(10):
        g_cc = _draw(cfg, seed)["g_cc"]
        assert g_cc.sum() == 4
        assert np.array_equal(g_cc, np.triu(g_cc, k=1))


def test_none_and_empty_budget_are_identical():
    g_none = _draw(_cfg(None), seed=0)
    g_empty = _draw(_cfg({}), seed=0)
    for key in g_none:
        assert np.array_equal(g_none[key], g_empty[key]), key


def test_validate_rejects_unknown_edge_type():
    with pytest.raises(ValueError, match="edge_budget keys"):
        _cfg({"nope": 1}).validate()


@pytest.mark.parametrize("spec", [-1, (3, 1), (1, 2, 3), 1.5, True, (True, True)])
def test_validate_rejects_bad_specs(spec):
    with pytest.raises(ValueError, match="edge_budget"):
        _cfg({"zc": spec}).validate()


def test_validate_accepts_numpy_ints():
    _cfg({"zc": np.int64(3)}).validate()
    _cfg({"cc": (np.int64(1), np.int64(2))}).validate()
    g = _draw(_cfg({"zc": (np.int64(4), np.int64(4))}), seed=0)
    assert g["g_zc"].sum() == 4


def test_generate_corpus_honors_pot_end_to_end():
    cfg = make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        edge_budget={"zc": 3},
        T=52,
        n_cells=3,
        draws_per_cell=2,
        seed=0,
    )
    corpus = sample_prior_predictive(cfg)
    per_task_zc = corpus["g"][:, cfg.layout.slices["zc"]].sum(axis=1)
    assert np.all(per_task_zc <= 3)
    assert corpus["diagnostics"]["edge_budget"] == {"zc": 3}
