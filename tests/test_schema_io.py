"""Corpus schema, validation, and .npz persistence (the PFN-consumable format)."""

from __future__ import annotations

import numpy as np
import pytest

import prior_generator as pg
from prior_generator import DataGenerator
from prior_generator.slots import EDGE_TYPES_EXTENDED, SlotLayout


@pytest.fixture(scope="module")
def corpus():
    cfg = pg.make_world_config(K_max=4, M_max=2, J_max=1, T=40, n_cells=2, draws_per_cell=3, seed=7)
    return pg.generate_corpus(cfg)


def test_required_keys_and_shapes(corpus):
    N, T, K = corpus["spend_raw"].shape
    M = corpus["controls"].shape[2]
    J = corpus["demand"].shape[2]
    expected = {
        "spend_raw": (N, T, K),
        "spend_norm": (N, T, K),
        "spend_share": (N, T, K),
        "controls": (N, T, M),
        "sales_raw": (N, T),
        "sales_norm": (N, T),
        "support_mask": (N, T),
        "contributions_raw": (N, T, K),
        "baseline_raw": (N, T),
        "indirect_effects": (N, T),
        "indirect_effects_by_source": (N, T, 3),
        "control_contribution": (N, T, M),
        "confounder_contribution": (N, T, J),
        "baseline_intrinsic": (N, T),
        "channel_active": (N, K),
        "active_c_mask": (N, K),
    }
    for key, shape in expected.items():
        assert corpus[key].shape == shape, f"{key}: {corpus[key].shape} != {shape}"


def test_g_layout_is_extended_8_block(corpus):
    N, T, K = corpus["spend_raw"].shape
    M, J = corpus["controls"].shape[2], corpus["demand"].shape[2]
    layout = SlotLayout(K=K, M=M, J=J, edge_types=EDGE_TYPES_EXTENDED)
    assert corpus["g"].shape == (N, layout.n_slots)
    assert set(np.unique(corpus["g"])).issubset({0, 1})


def test_validate_corpus_accepts_generated(corpus):
    assert DataGenerator.validate_corpus(corpus) == []


def test_validate_corpus_flags_missing_key(corpus):
    broken = {k: v for k, v in corpus.items() if k != "sales_scale"}
    errors = DataGenerator.validate_corpus(broken)
    assert any("sales_scale" in e for e in errors)


def test_validate_corpus_flags_nan(corpus):
    broken = dict(corpus)
    bad = corpus["spend_raw"].copy()
    bad[0, 0, 0] = np.nan
    broken["spend_raw"] = bad
    errors = DataGenerator.validate_corpus(broken)
    assert any("NaN" in e or "Inf" in e for e in errors)


def test_save_load_roundtrip(tmp_path, corpus):
    path = tmp_path / "corpus.npz"
    pg.save_corpus(corpus, path)
    loaded = pg.load_corpus(path)
    for key, val in corpus.items():
        if key == "diagnostics":
            assert loaded[key]["rung"] == "L1_additive"
        else:
            assert np.array_equal(loaded[key], val), f"{key} changed across roundtrip"


def test_datagenerator_generate_n_tasks():
    cfg = pg.make_world_config(K_max=4, M_max=2, J_max=1, T=32, draws_per_cell=5, seed=1)
    gen = DataGenerator(cfg)
    corpus = gen.generate(n_tasks=7, seed=1)
    assert corpus["spend_raw"].shape[0] == 7
    assert corpus["is_val"].sum() >= 1


def test_sales_norm_matches_scale(corpus):
    expected = (
        corpus["sales_raw"].astype(np.float64) / corpus["sales_scale"].astype(np.float64)[:, None]
    )
    assert np.allclose(corpus["sales_norm"].astype(np.float64), expected, rtol=1e-5)
