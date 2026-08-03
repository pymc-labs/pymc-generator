"""Prior-conditioning hyperprior (ACE): per-cell prior intervals in the corpus.

Locks the to-do/01 invariants: containment, nesting, determinism, RNG
isolation of the disabled path, and exact .npz round-trip of the
``prior_cond`` key and its diagnostics echo.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.sampler import _slice_g_active, sample_g_additive
from prior_generator.slots import PRIOR_COND_LAYOUT, PRIOR_COND_QUANTITIES
from prior_generator.world_model import (
    build_world_model,
    draw_worlds,
    sample_prior_cond,
    sample_structure,
)


def _cfg(**overrides):
    return pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=40,
        n_cells=2,
        draws_per_cell=3,
        seed=7,
        **overrides,
    )


@pytest.fixture(scope="module")
def conditioned_corpus():
    return pg.sample_prior_predictive(_cfg(prior_conditioning=True))


def _cols(corpus):
    """{layout name: (n_tasks,) column} of the packed prior_cond array."""
    return {name: corpus["prior_cond"][:, i] for i, name in enumerate(PRIOR_COND_LAYOUT)}


# -- corpus contract ---------------------------------------------------------


def test_prior_cond_key_present_iff_enabled(conditioned_corpus):
    n = conditioned_corpus["spend_raw"].shape[0]
    assert conditioned_corpus["prior_cond"].shape == (n, len(PRIOR_COND_LAYOUT))
    assert conditioned_corpus["prior_cond"].dtype == np.float32
    off = pg.sample_prior_predictive(_cfg())
    assert "prior_cond" not in off
    assert "prior_cond" not in off["diagnostics"]


def test_rows_broadcast_from_cell_draw(conditioned_corpus):
    pc, cell_id = conditioned_corpus["prior_cond"], conditioned_corpus["cell_id"]
    for cell in np.unique(cell_id):
        rows = pc[cell_id == cell]
        assert (rows == rows[0]).all(), "worlds within a cell must share their intervals"
    # cells differ (4 continuous draws colliding across cells is measure-zero)
    cells = np.unique(cell_id)
    assert not np.array_equal(pc[cell_id == cells[0]][0], pc[cell_id == cells[1]][0])


def test_diagnostics_echo(conditioned_corpus):
    echo = conditioned_corpus["diagnostics"]["prior_cond"]
    assert echo["layout"] == list(PRIOR_COND_LAYOUT)
    assert set(echo["supports"]) == set(PRIOR_COND_QUANTITIES)
    assert set(echo["width_ranges"]) == set(PRIOR_COND_QUANTITIES)
    assert echo["supports"]["adstock_alpha"] == [0.2, 0.8]
    assert echo["width_ranges"]["hill_shape"] == [0.2, 1.6]
    assert pg.DataGenerator.validate_corpus(conditioned_corpus) == []


def test_validator_rejects_malformed_prior_conditioning_metadata(conditioned_corpus):
    malformed_array = dict(conditioned_corpus)
    malformed_array["prior_cond"] = conditioned_corpus["prior_cond"].tolist()
    assert pg.DataGenerator.validate_corpus(malformed_array) == ["prior_cond must be an ndarray"]

    missing_echo = dict(conditioned_corpus)
    missing_echo["diagnostics"] = dict(conditioned_corpus["diagnostics"])
    del missing_echo["diagnostics"]["prior_cond"]
    assert any(
        "must be present together" in error
        for error in pg.DataGenerator.validate_corpus(missing_echo)
    )

    missing_array = dict(conditioned_corpus)
    del missing_array["prior_cond"]
    assert any(
        "must be present together" in error
        for error in pg.DataGenerator.validate_corpus(missing_array)
    )


def test_validator_rejects_invalid_or_non_cellwise_prior_intervals(conditioned_corpus):
    bad_layout = dict(conditioned_corpus)
    bad_layout["diagnostics"] = dict(conditioned_corpus["diagnostics"])
    bad_layout["diagnostics"]["prior_cond"] = dict(conditioned_corpus["diagnostics"]["prior_cond"])
    bad_layout["diagnostics"]["prior_cond"]["layout"] = ["unknown"]
    assert any("layout" in error for error in pg.DataGenerator.validate_corpus(bad_layout))

    bad_bounds = dict(conditioned_corpus)
    bad_bounds["diagnostics"] = dict(conditioned_corpus["diagnostics"])
    bad_bounds["diagnostics"]["prior_cond"] = dict(conditioned_corpus["diagnostics"]["prior_cond"])
    bad_bounds["diagnostics"]["prior_cond"]["supports"] = dict(
        conditioned_corpus["diagnostics"]["prior_cond"]["supports"]
    )
    bad_bounds["diagnostics"]["prior_cond"]["supports"]["adstock_alpha"] = [0.8, 0.2]
    assert "diagnostics prior_cond adstock_alpha bounds are invalid" in (
        pg.DataGenerator.validate_corpus(bad_bounds)
    )

    bad_width = dict(conditioned_corpus)
    bad_width["prior_cond"] = conditioned_corpus["prior_cond"].copy()
    bad_width["prior_cond"][0, 1] = 0.0
    assert any(
        "intervals are outside" in error for error in pg.DataGenerator.validate_corpus(bad_width)
    )

    inconsistent = dict(conditioned_corpus)
    inconsistent["prior_cond"] = conditioned_corpus["prior_cond"].copy()
    same_cell_rows = np.flatnonzero(
        conditioned_corpus["cell_id"] == conditioned_corpus["cell_id"][0]
    )
    assert len(same_cell_rows) > 1
    inconsistent["prior_cond"][same_cell_rows[1], 0] += 1e-3
    assert any(
        "rows differ within a cell" in error
        for error in pg.DataGenerator.validate_corpus(inconsistent)
    )

    forged_cell = dict(inconsistent)
    forged_cell["cell_id"] = conditioned_corpus["cell_id"].copy()
    forged_cell["cell_id"][same_cell_rows[1]] = 99
    errors = pg.DataGenerator.validate_corpus(forged_cell)
    assert "cell_id must contain contiguous nonnegative ids" in errors
    assert "diagnostics n_cells does not match cell_id" in errors


# -- invariant 2: nesting ----------------------------------------------------


def test_nesting(conditioned_corpus):
    cols = _cols(conditioned_corpus)
    spec = _cfg(prior_conditioning=True).prior_cond_spec()
    for q in PRIOR_COND_QUANTITIES:
        lo, width = cols[f"{q}_low"], cols[f"{q}_width"]
        (s_lo, s_hi) = spec[q]["support"]
        (w_lo, w_hi) = spec[q]["width_range"]
        # float32 storage of a float64 draw: allow one ulp of slack
        tol = 1e-6
        assert (width >= w_lo - tol).all() and (width <= w_hi + tol).all()
        assert (lo >= s_lo - tol).all() and (lo + width <= s_hi + tol).all()


# -- invariant 1: containment ------------------------------------------------


def test_containment_single_cell():
    """Drawn conditioned params lie in the recorded interval (all channels)."""
    cfg = _cfg(prior_conditioning=True)
    rng = np.random.default_rng(3)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=4, n_covariates_active=2, n_latent_active=1
    )
    g_act = _slice_g_active(g, 4, 2, 1)
    structural = sample_structure(g_act, cfg, rng)
    prior_cond = sample_prior_cond(cfg, rng)
    model, _out_names, _param_names = build_world_model(
        g_act, cfg, structural, cfg.n_time_steps, prior_cond=prior_cond
    )
    drawn = draw_worlds(model, ("adstock_alpha", "hill_slope"), seed=5, draws=8)
    for q, var in (("adstock_alpha", "adstock_alpha"), ("hill_shape", "hill_slope")):
        lo, width = prior_cond[q]
        vals = drawn[var]
        assert (vals >= lo).all() and (vals <= lo + width).all(), q


def test_containment_sample_scm():
    cfg = _cfg(prior_conditioning=True)
    world = pg.sample_scm(cfg, seed=11)
    lo, width = world.extras["prior_cond"]["adstock_alpha"]
    alpha = np.asarray(world.params["adstock_alpha"])
    geometric = np.asarray(world.params["adstock_family"]) == 1
    assert (alpha >= lo).all() and (alpha <= lo + width).all()
    assert geometric.shape == alpha.shape  # family mask well-formed


# -- invariant 3: determinism ------------------------------------------------


def test_determinism(conditioned_corpus):
    again = pg.sample_prior_predictive(_cfg(prior_conditioning=True))
    assert np.array_equal(again["prior_cond"], conditioned_corpus["prior_cond"])
    assert np.array_equal(again["sales_raw"], conditioned_corpus["sales_raw"])


# -- invariant 4: RNG isolation of the disabled path -------------------------


def test_disabled_draw_consumes_no_rng():
    cfg = _cfg()  # prior_conditioning=False
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state
    assert sample_prior_cond(cfg, rng) is None
    assert rng.bit_generator.state == state_before


def test_disabled_path_ignores_width_ranges():
    """Width ranges are inert while the flag is off — corpora are identical."""
    base = pg.sample_prior_predictive(_cfg())
    with_ranges = pg.sample_prior_predictive(
        _cfg(prior_cond_width_ranges={"adstock_alpha": (0.1, 0.3)})
    )
    assert "prior_cond" not in with_ranges
    for key, val in base.items():
        if isinstance(val, np.ndarray):
            assert np.array_equal(with_ranges[key], val), key


# -- invariant 5: .npz round-trip --------------------------------------------


def test_npz_roundtrip(tmp_path, conditioned_corpus):
    path = tmp_path / "conditioned.npz"
    pg.save_corpus(conditioned_corpus, path)
    loaded = pg.load_corpus(path)
    assert np.array_equal(loaded["prior_cond"], conditioned_corpus["prior_cond"])
    assert loaded["diagnostics"]["prior_cond"]["layout"] == list(PRIOR_COND_LAYOUT)


# -- config validation & description -----------------------------------------


def test_validate_rejects_bad_configs():
    with pytest.raises(ValueError, match="conditioned set"):
        _cfg(prior_conditioning=True, prior_cond_width_ranges={"weibull_lam": (0.1, 0.2)})
    with pytest.raises(ValueError, match="support width"):
        _cfg(prior_conditioning=True, prior_cond_width_ranges={"adstock_alpha": (0.1, 0.9)})
    with pytest.raises(ValueError, match="support width"):
        _cfg(prior_conditioning=True, prior_cond_width_ranges={"hill_shape": (0.0, 1.0)})
    # degenerate support cannot be conditioned
    with pytest.raises(ValueError, match="support width"):
        _cfg(prior_conditioning=True, adstock_alpha_range=(0.5, 0.5))


def test_replace_preserves_conditioning():
    cfg = _cfg(prior_conditioning=True)
    assert dataclasses.replace(cfg, seed=99).prior_conditioning is True


def test_describe_prints_intervals():
    world = pg.sample_scm(_cfg(prior_conditioning=True), seed=11)
    text = pg.describe_scm(world)
    assert "Prior conditioning (ACE)" in text
    assert "adstock_alpha" in text and "hill_shape" in text
    lo, width = world.extras["prior_cond"]["adstock_alpha"]
    assert f"U({lo:.3f}, {lo + width:.3f})" in text
