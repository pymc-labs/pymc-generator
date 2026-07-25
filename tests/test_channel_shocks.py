"""Symbolic per-draw channel-shock scheduling and trajectory application."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from prior_generator import load_corpus, make_scm_prior, sample_prior_predictive, save_corpus
from prior_generator.sampler import _additive_task_ok
from prior_generator.symbolic_graph import _adstock_col, _adstock_col_with_resets
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure


def _built(
    *,
    T=12,
    K=3,
    S=0,
    length=(1, 1),
    level=(0.0, 0.0),
    burn_in=0,
    direct=None,
    adstock_family=None,
):
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
    if adstock_family is not None:
        structural["adstock_family"][:] = adstock_family
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
        for s, (channel, start, length, level) in enumerate(
            zip(
                d["channel_shock_channel"][b],
                starts,
                lengths,
                d["channel_shock_level"][b],
            )
        ):
            assert np.array_equal(
                d["channels"][b, start : start + length, channel], np.full(length, level)
            )


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


@pytest.mark.parametrize("adstock_family", (1, 2), ids=("geometric", "weibull"))
def test_shocks_hold_observed_spend_zero_direct_response_and_natural_spend_resumes(
    adstock_family,
):
    """A held window is absolute, resets response history, and changes no natural path."""
    model, names, _, _, _ = _built(
        T=16,
        K=1,
        S=1,
        length=(3, 3),
        level=(0.0, 0.0),
        adstock_family=adstock_family,
    )
    d = draw_worlds(model, names, seed=19, draws=1)
    mask = d["channel_shock_mask"][0, :, 0].astype(bool)
    assert np.array_equal(d["channels"][0, mask, 0], np.zeros(mask.sum()))
    # Reset-aware adstock makes a zero window have no direct response despite
    # positive spend immediately before it (the no-input graph has softplus > 0).
    assert (d["channels_unshocked"][0, : np.flatnonzero(mask)[0], 0] > 0).all()
    assert np.array_equal(d["contributions"][0, mask, 0], np.zeros(mask.sum()))
    assert np.array_equal(d["contributions_observed"][0, mask, 0], np.zeros(mask.sum()))
    # This one-channel graph has no C parents, so outside intervention windows
    # the observed and natural spend paths are identical (including resumption).
    assert np.array_equal(d["channels"][0, ~mask, 0], d["channels_unshocked"][0, ~mask, 0])


def test_shocked_world_preserves_every_decomposition_identity():
    model, names, _, _, _ = _built(
        T=20,
        K=2,
        S=2,
        length=(2, 3),
        level=(0.0, 1.2),
        adstock_family=2,
    )
    d = {name: value[0] for name, value in draw_worlds(model, names, seed=31).items()}
    inside = d["channel_shock_mask"].any(axis=1)
    assert inside.any() and (~inside).any()

    additive_error = np.abs(
        d["baseline"] + d["contributions"].sum(1) + d["indirect_effects"] - d["sales"]
    )
    telescoping_error = np.abs(d["indirect_effects_by_source"].sum(1) - d["indirect_effects"])
    baseline_error = np.abs(
        d["baseline_intrinsic"]
        + d["confounder_contribution"].sum(1)
        + d["control_contribution"].sum(1)
        - d["baseline"]
    )
    full_error = np.abs(
        d["baseline_intrinsic"]
        + d["confounder_contribution"].sum(1)
        + d["control_contribution"].sum(1)
        + d["contributions"].sum(1)
        + d["indirect_effects_by_source"].sum(1)
        - d["sales"]
    )
    for error in (additive_error, telescoping_error, baseline_error, full_error):
        assert error.max() < 1e-9
        assert error[inside].max() < 1e-9
        assert error[~inside].max() < 1e-9


def test_persisted_shocked_corpus_preserves_float32_decomposition_inside_and_outside_masks(
    tmp_path,
):
    """Public persisted arrays retain exact held-zero responses and decomposition labels."""
    corpus = sample_prior_predictive(
        make_scm_prior(
            n_treatments=2,
            n_covariates=2,
            n_latent=1,
            T=20,
            n_cells=2,
            draws_per_cell=2,
            seed=71,
            nonlinearity="linear",
            edge_budget={"cy": (2, 2)},
            n_channel_shocks=1,
            channel_shock_length_range=(3, 3),
            channel_shock_level_range=(0.0, 0.0),
        )
    )
    path = tmp_path / "shocked.npz"
    save_corpus(corpus, path)
    persisted = load_corpus(path)
    inside = persisted["channel_shock_mask"].any(axis=2).astype(bool)
    assert inside.any() and (~inside).any()
    assert np.array_equal(persisted["channel_shock_level"], np.zeros((4, 1), dtype=np.float32))

    f = lambda key: persisted[key].astype(np.float64)  # noqa: E731
    errors = (
        np.abs(
            f("baseline_raw")
            + f("contributions_raw").sum(2)
            + f("indirect_effects")
            - f("sales_raw")
        ),
        np.abs(
            f("baseline_intrinsic")
            + f("confounder_contribution").sum(2)
            + f("control_contribution").sum(2)
            + f("contributions_raw").sum(2)
            + f("indirect_effects_by_source").sum(2)
            - f("sales_raw")
        ),
        np.abs(f("indirect_effects_by_source").sum(2) - f("indirect_effects")),
    )
    tolerance = (
        32
        * np.finfo(persisted["sales_raw"].dtype).eps
        * max(float(np.abs(persisted["sales_raw"]).max()), 1.0)
    )
    for error in errors:
        assert error.max() <= tolerance
        assert error[inside].max() <= tolerance
        assert error[~inside].max() <= tolerance
    assert np.array_equal(
        persisted["contributions_raw"][persisted["channel_shock_mask"].astype(bool)],
        np.zeros(persisted["channel_shock_mask"].sum(), dtype=np.float32),
    )


def test_realism_cv_and_spike_guards_use_the_unshocked_reference():
    actual_spend = np.array([[0.0], [100.0], [0.0]])
    actual_sales = np.array([1.0, 100.0, 1.0])
    natural_spend = np.array([[1.0], [2.0], [1.0]])
    natural_sales = np.array([2.0, 2.0, 2.0])
    assert _additive_task_ok(
        actual_spend,
        actual_sales,
        {"actual": actual_spend},
        np.array([1]),
        cv_floor=0.1,
        sales_spike_ratio=8.0,
        spend_spike_ratio=50.0,
        realism_spend=natural_spend,
        realism_sales=natural_sales,
    )


def test_downstream_channel_recursion_sees_the_clamped_parent():
    # The only C->C arrow retains channel zero as the sole shockable direct channel.
    g = {
        "g_cy": np.array([1, 0]),
        "g_dc": np.zeros((1, 2)),
        "g_dz": np.zeros((1, 1)),
        "g_db": np.zeros(1),
        "g_zb": np.zeros(1),
        "g_zc": np.zeros((1, 2)),
        "g_cc": np.array([[0, 1], [0, 0]]),
        "g_zz": np.zeros((1, 1)),
    }
    cfg = make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        T=12,
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.0, 0.0),
    )
    structural = sample_structure(g, cfg, np.random.default_rng(7))
    model, names, _ = build_world_model(g, cfg, structural, cfg.T)
    d = draw_worlds(model, names, seed=22)
    mask = d["channel_shock_mask"][0, :, 0].astype(bool)
    assert (d["channels"][0, mask, 1] < d["channels_unshocked"][0, mask, 1]).all()


@pytest.mark.parametrize("family", (1, 2), ids=("geometric", "weibull"))
def test_adstock_resets_discard_pre_window_history_and_latest_reset_wins(family):
    c = pt.as_tensor_variable(np.array([4.0, 3.0, 0.0, 0.0, 7.0, 8.0]))
    params = {
        "l_max": 4,
        "adstock_family": np.array([family]),
        "adstock_alpha": np.array([0.5]),
        "weibull_lam": np.array([2.0]),
        "weibull_k": np.array([2.0]),
        "channel_shock": {
            "n_shocks": 2,
            "channel": pt.as_tensor_variable(np.array([0, 0], dtype="int64")),
            "start_full": pt.as_tensor_variable(np.array([2, 4], dtype="int64")),
        },
    }
    reset = _adstock_col_with_resets(c, params, 0)
    # The second reset must supersede the first: after t=4 this is precisely
    # the ordinary adstock of the suffix starting at t=4, not a carryover.
    suffix = _adstock_col(c * pt.cast(pt.arange(6) >= 4, "float64"), params, 0)
    got, expected = pytensor.function([], [reset, suffix])()
    assert np.array_equal(got[2:4], np.zeros(2))
    assert np.array_equal(got[4:], expected[4:])
